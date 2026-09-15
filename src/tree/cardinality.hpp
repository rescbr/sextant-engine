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

namespace sextant::tree {

/// Default per-column cap on exact cardinality entries (freq map keys or
/// numeric histogram nodes). 0 = unlimited (legacy behavior).
inline constexpr uint32_t kDefaultCardinalityCap = 1u << 20;

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
    // the HLL registers are not serialized — only the estimate is).
    uint64_t est_distinct_stored = 0;

    bool is_numeric = false;

    // --- HyperLogLog distinct estimator (2^12 registers, 6 bits each) ---
    // Fed on every string/set add (before the cap check) so the estimate
    // covers ALL observed values, not just the ones admitted to `freq`.
    static constexpr uint32_t kHllBits = 12;
    static constexpr uint32_t kHllRegs = 1u << kHllBits;  // 4096
    static constexpr uint32_t kHllPadding = 32 - kHllBits;  // bits left after the index
    std::vector<uint8_t> hll;  // packed 6-bit registers; empty until first add

    /// Current distinct estimate: HLL if it has data, else the value stored
    /// at deserialize time, else the exact freq-map size.
    double distinct_estimate() const;
};

/// Global cardinality table — one entry per filter column.
class CardinalityTable {
public:
    /// Initialize for the given schema. Allocates per-column maps for
    /// all filter column types (numeric, string, set). `cap` bounds the
    /// exact per-column entries (0 = unlimited).
    void init(const Schema& schema, uint64_t n_vectors,
              uint32_t cap = kDefaultCardinalityCap);

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
    /// (col_to_idx_ / is_numeric flags). Used to reuse a sharded table across
    /// chunks without reallocating its column structure.
    void clear_stats();

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

    /// True if any column overflowed its exact-entry cap (diagnostics).
    bool any_overflowed() const;

    /// Total entries currently held across all exact maps (diagnostics).
    size_t total_entries() const;

    /// Serialize to a binary blob. Versioned: starts with a magic u32 with
    /// the high bit set (legacy blobs start with n_vectors:u64, which cannot
    /// have bit 31 of its low word set for realistic row counts).
    /// Format:
    ///   [magic: u32 = 0xC4DA0002][version: u8 = 2]
    ///   [n_vectors: u64][n_columns: u32]
    ///   for each column with data:
    ///     [col_id: u32][flags: u8]   bit0 is_numeric, bit1 binned, bit2 overflowed
    ///     if string/set: [n_entries: u32][est_distinct: u64]
    ///                     [n_entries × (hash: u32, count: u32)]
    ///     elif numeric exact: [n_entries: u32][n_entries × (value: f64, count: u32)]
    ///     else (binned): [bin_min: f64][bin_max: f64][frame_lo: f64]
    ///                     [frame_hi: f64][bin_total: u64]
    ///                     [4096 × (count: u32)]
    std::vector<uint8_t> serialize() const;

    /// Deserialize from a binary blob. Accepts both the versioned format
    /// above and the legacy layout ([n_vectors: u64][n_columns: u32] ...).
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
    uint64_t n_vectors_ = 0;
    uint32_t cap_ = kDefaultCardinalityCap;  // 0 = unlimited

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
