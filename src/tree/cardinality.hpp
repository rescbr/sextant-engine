#pragma once

/// @file cardinality.hpp
/// Global cardinality table for selectivity estimation (Phase D).
///
/// At build time, counts the frequency of each distinct value in each
/// filter column (numeric, string, set). Serialized to a page extent
/// pointed to by the superblock's cardinality_page.
///
/// At search time, used to estimate predicate selectivity:
///   Eq:    selectivity = count(value) / N
///   Range: selectivity = Σ count(v) for v in matching range / N
/// This drives adaptive W (§3.9) and the brute-force fallback trigger (<1%).
///
/// Memory bounds (per column, when cap > 0, default 1<<20 entries):
///   - String/set freq maps are capped at `cap` entries. Once full, inserts
///     become insert-if-present: heavy hitters already present keep exact
///     counts; unseen values only feed the distinct-value estimator (HLL).
///   - Numeric exact histograms (std::map) are capped at `cap` entries. On
///     overflow the column switches to a fixed 4096-bin linear histogram
///     over [min, max] with saturating u32 counters; min/max stay exact.
/// Per-thread build shards get cap / n_threads so the aggregate stays
/// bounded across the merge.

#include "sextant/schema.hpp"
#include "filter_hash.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sextant/error.hpp"

namespace sextant::tree {

/// Default per-column cap on exact cardinality entries (freq map keys or
/// numeric histogram nodes). 0 = unlimited (legacy behavior).
inline constexpr uint32_t kDefaultCardinalityCap = 1u << 20;

/// `auto` columns are tracked exactly like `on` until this many values have
/// been merged into the global table, then a keep/reject decision is made.
/// 100k is enough for the HLL distinct estimate to converge well within its
/// ~1.6% standard error while bounding the wasted work on identity columns.
inline constexpr uint64_t kCardinalitySample = 100'000;

/// `auto` rejection threshold: if distinct_estimate / adds exceeds this,
/// the column is treated as identifier-like (url/uuid: ratio ≈ 1.0 at any
/// sample size; categorical columns: ratio ≈ distinct/n, tiny) and flipped
/// to untracked with a frozen distinct estimate. 0.5 splits the two regimes
/// with a wide margin on both sides.
inline constexpr double kCardinalityIdentityRatio = 0.5;

/// Per-column cardinality tracking policy.
///   Off       — not tracked at all (DEFAULT): add_* are no-ops; search-time
///               Eq uses the uniform prior 1/n_vectors (never 0).
///   On        — tracked exactly (capped exact maps + HLL / binned numerics;
///               today's behavior).
///   Auto      — tracked like `on` during the sampling phase; decided into
///               `on` or `rejected` at a merge point once kCardinalitySample
///               values have been merged (see CardinalityTable::evaluate_auto).
///   Rejected  — internal terminal state of a rejected `auto` column: no
///               exact data, only the distinct estimate frozen at rejection
///               time (Eq → min(1, 1/est_distinct)).
enum class CardinalityMode : uint8_t {
    Off = 0,
    On = 1,
    Auto = 2,
    Rejected = 3,
};

/// User-facing cardinality tracking configuration (`--cardinality-col`).
/// `default_mode` applies to every filter column not named in `per_column`:
/// the bare keyword "auto" sets it to Auto; the CLI default leaves it Off.
/// `per_column` entries override per name (name, name=on, name=off, name=auto).
struct CardinalitySpec {
    CardinalityMode default_mode = CardinalityMode::Off;
    std::map<std::string, CardinalityMode> per_column;

    /// Legacy behavior: every column tracked (`on`). Used by tests.
    static CardinalitySpec all_on() {
        CardinalitySpec s;
        s.default_mode = CardinalityMode::On;
        return s;
    }
};

/// Parse a comma-separated cardinality spec ("name", "name=on|off|auto",
/// bare "auto" for the default mode). Throws std::invalid_argument on a
/// bad token. Shared by the CLI --cardinality-col flag and the C API
/// sextant_build_opts.cardinality field.
CardinalitySpec parse_cardinality_spec(const std::string& s);

/// Number of linear bins used when a numeric histogram overflows its cap.
inline constexpr uint32_t kCardinalityNumBins = 4096;

/// Per-column cardinality data.
struct ColumnCardinality {
    // String/set: hash(value) → count. Capped: once size() reaches the cap,
    // only keys already present keep counting (see CardinalityTable::add_string).
    std::unordered_map<uint32_t, uint32_t> freq;

    // Numeric: sorted (value_as_double, count) for range selectivity, exact
    // until the cap. When `binned` is true this map is empty and the fixed
    // linear histogram below is used instead.
    std::map<double, uint32_t> numeric_hist;

    // Binned fallback (numeric overflow). Linear bins over the binning
    // frame [frame_lo, frame_hi] (which grows geometrically to amortize
    // rescaling on monotone streams), saturating u32 counters. bin_min/
    // bin_max are the EXACT observed min/max (used by queries); the frame
    // is only for bin indexing and may extend beyond them.
    std::vector<uint32_t> bins;
    double bin_min = 0.0;
    double bin_max = 0.0;
    double frame_lo = 0.0;
    double frame_hi = 0.0;
    uint64_t bin_total = 0;
    bool binned = false;

    // True once the exact map hit its cap (string/set) or switched to bins
    // (numeric). When overflowed, an Eq probe for a value NOT in the exact
    // map must fall back to 1/est_distinct instead of reporting 0 (absent),
    // otherwise rare values would spuriously trigger the brute-force path.
    bool overflowed = false;

    // Distinct-value estimate after deserialize (search side never adds, so
    // the HLL registers are not serialized — only the estimate is). Also
    // carries the frozen estimate of a Rejected `auto` column (see mode).
    uint64_t est_distinct_stored = 0;

    bool is_numeric = false;

    // Tracking policy for this column (see CardinalityMode). Off by default.
    CardinalityMode mode = CardinalityMode::Off;

    // Values added to this column since tracking started (merge-cumulative
    // on the global table; per-chunk on shard tables). Drives the `auto`
    // sample-phase decision.
    uint64_t adds = 0;

    // --- HyperLogLog distinct estimator (2^12 registers, 6 bits each) ---
    // Fed on every string/set add (before the cap check) so the estimate
    // covers ALL observed values, not just the ones admitted to `freq`.
    static constexpr uint32_t kHllBits = 12;
    static constexpr uint32_t kHllRegs = 1u << kHllBits;  // 4096
    static constexpr uint32_t kHllPadding = 32 - kHllBits;  // bits left after the index
    std::vector<uint8_t> hll;  // one byte per register; empty until first add

    /// Current distinct estimate: HLL if it has data, else the value stored
    /// at deserialize time, else the exact freq-map size.
    double distinct_estimate() const;
};

/// Global cardinality table — one entry per filter column.
class CardinalityTable {
public:
    /// Initialize for the given schema. Allocates per-column maps for
    /// all filter column types (numeric, string, set). `cap` bounds the
    /// exact per-column entries (0 = unlimited). `spec` (optional) sets
    /// per-column tracking modes; null = every column Off (the default
    /// policy). Throws Error(InvalidParam) if `spec` names a column that
    /// is not in the schema (typo protection).
    void init(const Schema& schema, uint64_t n_vectors,
              uint32_t cap = kDefaultCardinalityCap,
              const CardinalitySpec* spec = nullptr);

    /// Record a string value for a column during the build emission pass.
    void add_string(uint32_t col_idx, std::string_view val);

    /// Record a set's elements for a column.
    void add_set(uint32_t col_idx,
                  const uint8_t* counts, const uint32_t* offsets,
                  const uint16_t* elem_lengths, const char* elem_data,
                  uint32_t row_idx);

    /// Record a numeric value for a column.
    void add_numeric(uint32_t col_idx, double val);

    /// Merge another table's per-column statistics into this one. Combines
    /// frequency counters (string/set) and numeric histograms by summing
    /// counts, so the result is equivalent to having added all rows serially
    /// (modulo capping). Used to fold per-thread sharded cardinality tables
    /// back into the global table after the parallel emission append phase.
    /// `other` must share this table's schema (same col_to_idx_ layout).
    void merge_from(const CardinalityTable& other);

    /// Reset all per-column statistics to empty, keeping the schema layout
    /// (col_to_idx_ / is_numeric flags / tracking modes). Used to reuse a
    /// sharded table across chunks without reallocating its column structure.
    void clear_stats();

    /// Run the `auto` keep/reject decision for every Auto column that has
    /// reached the sample size (kCardinalitySample merged adds). Called at
    /// shard merge points and after a vacuum rebuild. Rejected columns are
    /// freed (freq/HLL/numeric data) and keep only their frozen distinct
    /// estimate; accepted columns are promoted to On and keep what they
    /// accumulated. Logs each rejection.
    void evaluate_auto();

    /// Copy this table's per-column tracking modes onto `shard` (matched by
    /// schema column index) and free any accumulated shard data for
    /// Off/Rejected columns, so shards stop hashing values the global table
    /// no longer wants. Called by the build loop after each merge.
    void propagate_modes_to(CardinalityTable& shard) const;

    /// Copy per-column modes from `src` (matched by schema column index).
    /// Used by the vacuum rebuild to preserve the existing blob's policy.
    /// Columns absent from `src` keep their current mode.
    void copy_modes_from(const CardinalityTable& src);

    /// Estimate selectivity for a string equality predicate.
    float selectivity_string(uint32_t col_idx, std::string_view val) const;

    /// Estimate selectivity for a set CONTAINS predicate.
    float selectivity_set(uint32_t col_idx, std::string_view val) const;

    /// Estimate selectivity for a numeric predicate (eq, gt, ge, lt, le, between).
    float selectivity_numeric(uint32_t col_idx, const Predicate& pred) const;

    /// Estimate selectivity for a geo predicate (GeoBox, GeoRadius).
    /// `lat_col` is the latitude column index; `lng_col` is the longitude
    /// column index. Returns the area-overlap fraction of the query bounding
    /// box within the columns' observed [min, max] ranges.
    float selectivity_geo(uint32_t lat_col, uint32_t lng_col,
                           const Predicate& pred) const;

    /// Combined selectivity for all predicates (product of individual
    /// selectivities, assuming independence). Clamped to [0, 1].
    float selectivity_combined(
        const Schema& schema,
        const std::vector<struct Predicate>& preds,
        const std::vector<uint32_t>& pred_col_indices,
        const std::vector<uint32_t>& geo_lng_col_indices = {}) const;

    bool empty() const { return columns_.empty(); }
    uint64_t n_vectors() const { return n_vectors_; }
    uint32_t cap() const { return cap_; }

    /// Tracking mode of a schema column (Off if the column is unknown or
    /// absent from the table).
    CardinalityMode column_mode(uint32_t schema_col) const;

    /// Diagnostics: per-column tracking modes as "name=mode" pairs plus a
    /// per-mode count summary for the build log.
    std::string mode_report() const;

    /// True if any column overflowed its exact-entry cap (diagnostics).
    bool any_overflowed() const;

    /// Total entries currently held across all exact maps (diagnostics).
    size_t total_entries() const;

    /// Serialize to a binary blob. Versioned: starts with a magic u32.
    /// No pre-release compat: the on-disk format is not finalized and no
    /// released trees exist — deserialize accepts ONLY this version.
    /// Format (v3):
    ///   [magic: u32 = 0xC4DA0003][version: u8 = 3]
    ///   [n_vectors: u64][n_columns: u32]
    ///   for each column:
    ///     [col_id: u32][flags: u8]   bit0 is_numeric, bit1 binned,
    ///                                bit2 overflowed, bits3-4 mode
    ///     (Off: nothing further)
    ///     (Rejected: [est_distinct: u64])
    ///     if string/set: [n_entries: u32][est_distinct: u64]
    ///                     [n_entries × (hash: u32, count: u32)]
    ///     elif numeric exact: [n_entries: u32][n_entries × (value: f64, count: u32)]
    ///     else (binned): [bin_min: f64][bin_max: f64][frame_lo: f64]
    ///                     [frame_hi: f64][bin_total: u64]
    ///                     [4096 × (count: u32)]
    std::vector<uint8_t> serialize() const;

    /// Deserialize from a binary blob. Accepts only the versioned format
    /// above (magic 0xC4DA0003); any other blob is rejected (logs a
    /// warning and leaves the table empty — the search path treats an
    /// empty table as "no estimates" and uses its default selectivity).
    void deserialize(const uint8_t* data, size_t len);

    // --- HLL helpers (exposed for testing) ---
    static void hll_add(std::vector<uint8_t>& hll, uint32_t finalized_hash);
    static double hll_estimate(const std::vector<uint8_t>& hll);
    /// splitmix-style finalizer: filter_hash's 32-bit output needs extra
    /// avalanche before HLL bit-slicing.
    static uint32_t hll_finalize(uint32_t h);

private:
    std::vector<ColumnCardinality> columns_;
    std::vector<uint32_t> col_to_idx_;  // schema col index → columns_ index
    std::vector<std::string> col_names_;  // schema col name per columns_ index
    uint64_t n_vectors_ = 0;
    uint32_t cap_ = kDefaultCardinalityCap;  // 0 = unlimited

    /// Uniform Eq prior for an untracked column: 1/est_distinct when a
    /// frozen sample-phase estimate exists, else 1/n_vectors. NEVER 0 —
    /// selectivity 0 would spuriously trigger the brute-force fallback.
    float untracked_prior_(const ColumnCardinality& col) const;

    /// Free all per-column data (exact maps, HLL, bins). Keeps mode.
    static void free_data_(ColumnCardinality& col);

    /// Exact-map insert for a hashed string/set value, honoring the cap.
    void freq_add_(ColumnCardinality& col, uint32_t h);

    /// Bin a (value, count) pair into a binned column, updating exact
    /// min/max and saturating counters.
    static void bin_add_(ColumnCardinality& col, double val, uint32_t cnt);

    /// Convert an exact numeric column at/over its cap into binned mode.
    void numeric_to_binned_(ColumnCardinality& col);

    /// Rebin the existing bins onto a new [lo, hi] frame (geometric growth).
    static void rescale_bins_(ColumnCardinality& col, double lo, double hi);

    /// Selectivity for range ops on a binned column.
    float selectivity_binned_(const ColumnCardinality& col,
                              const Predicate& pred) const;
};

}  // namespace sextant::tree
