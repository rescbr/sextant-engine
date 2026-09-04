#pragma once

/// @file filter_scan.hpp
/// Search-time filter evaluation: summary pruning + per-candidate predicate
/// checking (Phase D).
///
/// This module reads the on-disk filter column data (written by Phase C's
/// filter_column_write.hpp) from the mmap'd leaf extent and evaluates
/// predicates. It has two modes:
///
/// 1. **Summary check** (leaf-level pre-filter): reads the leaf's summary
///    region (numeric min/max + string/set blooms) and checks whether the
///    leaf CAN contain a matching vector. Used to skip entire leaves before
///    scanning.
///
/// 2. **Per-candidate check** (inline filter): reads a single candidate's
///    filter column values from the leaf and evaluates all predicates. Used
///    during the FastScan loop to reject non-matching candidates.
///
/// All filter column data is read from the mmap'd leaf — zero copies, zero
/// allocations on the hot path.

#include "sextant/schema.hpp"
#include "sextant/types.hpp"
#include "tree/tree_nodes.hpp"
#include "filter_hash.hpp"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <string_view>

namespace sextant::tree {

// ===========================================================================
// On-disk filter column layout reader
// ===========================================================================

/// Locates the filter column regions within a mmap'd leaf extent.
/// Computed once per leaf, then used for per-candidate lookups.
struct LeafGeometry;  // tree/leaf_coder.hpp

struct LeafFilterLayout {
    uint32_t count = 0;               // live vector count
    uint32_t n_blocks = 0;
    uint32_t block_bytes = 0;
    uint32_t codes_per_block = 0;
    uint32_t summary_size = 0;

    // Pointers into the mmap'd leaf (all relative to leaf_ptr):
    const uint8_t* codes = nullptr;
    const RowId*   row_ids = nullptr;
    // Filter column data starts after row_ids.
    const uint8_t* filter_base = nullptr;

    /// Compute the layout for a mmap'd leaf. leaf_ptr = start of leaf extent.
    /// NOTE: this overload assumes the global-PQ FastScan layout; prefer
    /// from_geometry() with the owning LeafCoder's geometry() — the offsets
    /// differ for the scalar / local families.
    static LeafFilterLayout compute(const uint8_t* leaf_ptr, uint16_t m4,
                                     uint8_t pq_bits,
                                     uint32_t summary_size);

    /// Family-agnostic layout from a LeafGeometry (the LeafCoder-computed
    /// row_ids / filter offsets). Fixes the hardcoded-global-PQ offset bug
    /// for scalar / local families.
    static LeafFilterLayout from_geometry(const uint8_t* leaf_ptr,
                                          const struct LeafGeometry& geo);
};

/// View into a single column's on-disk data within a leaf.
/// For fixed-width: data[i] is at base + i * width.
/// For string: offsets[i], lengths[i], hashes[i], data region.
/// For set: counts[i], offsets[i], hashes region, data region.
struct ColumnView {
    ColumnType type = ColumnType::Int32;

    // Fixed-width.
    const uint8_t* fixed_base = nullptr;
    uint8_t fixed_width = 0;

    // String.
    const uint32_t* str_offsets = nullptr;  // byte offset into str_data
    const uint16_t* str_lengths = nullptr;
    const uint32_t* str_hashes = nullptr;
    const char*     str_data = nullptr;

    // Set.
    const uint8_t*  set_counts = nullptr;
    const uint32_t* set_offsets = nullptr;  // element index into set_hashes
    const uint32_t* set_hashes = nullptr;   // total_elements entries
    const uint8_t*  set_data = nullptr;     // [u16 len][bytes] per element
};

/// Parse the filter column regions from a mmap'd leaf. Returns one ColumnView
/// per schema column. The schema MUST match the schema the leaf was built with.
/// `filter_base` is the start of the filter column data region (from LeafFilterLayout).
/// After the fixed-width columns (in schema order), string columns, then set columns.
///
/// Returns views in schema column order. The vector is sized to schema.columns.size().
std::vector<ColumnView> parse_filter_columns(const uint8_t* filter_base,
                                              uint32_t count,
                                              const Schema& schema);

// ===========================================================================
// Per-candidate predicate evaluation
// ===========================================================================

/// Evaluate a single predicate against a candidate's filter column value.
/// `col` is the ColumnView for the predicate's column. `idx` is the local
/// vector index within the leaf.
/// Returns true if the candidate passes the predicate.
///
/// NOTE: This overload CANNOT evaluate geo predicates (GeoBox, GeoRadius),
/// which span two columns (latitude + longitude). For geo ops use
/// eval_predicate_geo(), which has access to all column views. When called
/// with a geo op this returns true (no-op) — callers that support geo must
/// route through eval_predicate_geo instead.
bool eval_predicate(const ColumnView& col, uint32_t idx, const Predicate& pred);

/// Evaluate a geo predicate (GeoBox, GeoRadius) against a candidate.
/// `lat_col` / `lng_col` are the ColumnViews for the latitude and longitude
/// columns respectively. `idx` is the local vector index within the leaf.
/// Returns true if the candidate passes.
bool eval_predicate_geo(const ColumnView& lat_col, const ColumnView& lng_col,
                         uint32_t idx, const Predicate& pred);

/// Great-circle distance between two lat/lng points (degrees) in kilometers.
/// Uses the haversine formula. Scalar (NEON has no native trig).
inline double haversine_km(double lat1, double lng1,
                            double lat2, double lng2) {
    constexpr double kEarthRadiusKm = 6371.0;
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double dlat = (lat2 - lat1) * kDegToRad;
    const double dlng = (lng2 - lng1) * kDegToRad;
    const double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
        std::cos(lat1 * kDegToRad) * std::cos(lat2 * kDegToRad) *
        std::sin(dlng / 2.0) * std::sin(dlng / 2.0);
    return kEarthRadiusKm * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
}

/// Evaluate ALL predicates against a candidate. Returns true only if ALL pass.
/// `cols` has one ColumnView per schema column; predicates reference columns
/// by name (resolved to index via schema). Geo predicates (GeoBox, GeoRadius)
/// reference two columns: the latitude column (pred_col_indices[p]) and the
/// longitude column (geo_lng_col_indices[p]).
inline bool eval_all_predicates(const std::vector<ColumnView>& cols,
                                 const Schema& schema, uint32_t idx,
                                 const std::vector<Predicate>& preds,
                                 const std::vector<uint32_t>& pred_col_indices,
                                 const std::vector<uint32_t>& geo_lng_col_indices = {}) {
    for (uint32_t p = 0; p < preds.size(); ++p) {
        const auto& pred = preds[p];
        if (pred.op == PredicateOp::GeoBox ||
            pred.op == PredicateOp::GeoRadius) {
            const uint32_t lng_col = (p < geo_lng_col_indices.size())
                ? geo_lng_col_indices[p] : UINT32_MAX;
            if (lng_col >= cols.size()) return false;  // misconfigured
            if (!eval_predicate_geo(cols[pred_col_indices[p]],
                                    cols[lng_col], idx, pred))
                return false;
        } else {
            if (!eval_predicate(cols[pred_col_indices[p]], idx, pred))
                return false;
        }
    }
    return true;
}

// ===========================================================================
// SIMD batch predicate evaluation (4 candidates at once for numeric columns)
// ===========================================================================

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define FILTER_HAS_NEON 1
#else
#define FILTER_HAS_NEON 0
#endif

/// Evaluate a single int32 equality predicate for 4 candidates using SIMD.
/// Returns a 4-bit mask: bit j set = candidate (start_idx + j) passes.
/// For non-int32-eq predicates, falls back to scalar (returns appropriate bits).
inline uint32_t eval_predicate_batch4(const ColumnView& col, uint32_t start_idx,
                                       const Predicate& pred) {
    // Only optimize the common case: int32 Eq/Lt/Le/Gt/Ge/Between.
    // Other types fall back to scalar per-element.
    if (col.type != ColumnType::Int32) {
        uint32_t mask = 0;
        for (uint32_t j = 0; j < 4; ++j) {
            if (eval_predicate(col, start_idx + j, pred))
                mask |= (1u << j);
        }
        return mask;
    }

    const int32_t* vals = reinterpret_cast<const int32_t*>(col.fixed_base) + start_idx;
    const int32_t pv = static_cast<int32_t>(pred.value);
    const int32_t pv2 = static_cast<int32_t>(pred.value2);

#if FILTER_HAS_NEON
    const int32x4_t v = vld1q_s32(vals);
    uint32x4_t cmp;
    switch (pred.op) {
        case PredicateOp::Eq:
            cmp = vreinterpretq_u32_s32(vceqq_s32(v, vdupq_n_s32(pv))); break;
        case PredicateOp::Lt:
            cmp = vreinterpretq_u32_s32(vcltq_s32(v, vdupq_n_s32(pv))); break;
        case PredicateOp::Le:
            cmp = vreinterpretq_u32_s32(vcleq_s32(v, vdupq_n_s32(pv))); break;
        case PredicateOp::Gt:
            cmp = vreinterpretq_u32_s32(vcgtq_s32(v, vdupq_n_s32(pv))); break;
        case PredicateOp::Ge:
            cmp = vreinterpretq_u32_s32(vcgeq_s32(v, vdupq_n_s32(pv))); break;
        case PredicateOp::Between: {
            const int32x4_t ge_lo = vcgeq_s32(v, vdupq_n_s32(pv));
            const int32x4_t le_hi = vcleq_s32(v, vdupq_n_s32(pv2));
            cmp = vandq_u32(vreinterpretq_u32_s32(ge_lo), vreinterpretq_u32_s32(le_hi));
            break;
        }
        default:
            // Fall through to scalar for NotEq, In, NotIn.
            cmp = vdupq_n_u32(0); break;
    }
    // Extract 4-bit mask: each lane is 0xFFFFFFFF (pass) or 0 (fail).
    // Take the MSB of each lane → narrow to 4 bits.
    uint32_t words[4];
    vst1q_u32(words, cmp);
    uint32_t mask = 0;
    for (uint32_t j = 0; j < 4; ++j)
        if (words[j]) mask |= (1u << j);
    // If op was NotEq/In/NotIn, cmp is all-zero → scalar fallback.
    if (pred.op == PredicateOp::NotEq || pred.op == PredicateOp::In ||
        pred.op == PredicateOp::NotIn) {
        mask = 0;
        for (uint32_t j = 0; j < 4; ++j)
            if (eval_predicate(col, start_idx + j, pred)) mask |= (1u << j);
    }
    return mask;
#endif
    // Scalar fallback.
    uint32_t smask = 0;
    for (uint32_t j = 0; j < 4; ++j) {
        const int32_t val = vals[j];
        bool pass = false;
        switch (pred.op) {
            case PredicateOp::Eq:      pass = (val == pv); break;
            case PredicateOp::Lt:      pass = (val < pv); break;
            case PredicateOp::Le:      pass = (val <= pv); break;
            case PredicateOp::Gt:      pass = (val > pv); break;
            case PredicateOp::Ge:      pass = (val >= pv); break;
            case PredicateOp::Between: pass = (val >= pv && val <= pv2); break;
            default: pass = eval_predicate(col, start_idx + j, pred); break;
        }
        if (pass) smask |= (1u << j);
    }
    return smask;
}

/// Evaluate ALL predicates for 4 candidates. Returns a 4-bit mask:
/// bit j set = candidate (start_idx + j) passes ALL predicates.
inline uint32_t eval_all_predicates_batch4(
    const std::vector<ColumnView>& cols, const Schema& schema,
    uint32_t start_idx,
    const std::vector<Predicate>& preds,
    const std::vector<uint32_t>& pred_col_indices,
    const std::vector<uint32_t>& geo_lng_col_indices = {}) {
    // Start with all candidates passing; AND out failures.
    uint32_t mask = 0xF;
    for (uint32_t p = 0; p < preds.size() && mask; ++p) {
        const auto& pred = preds[p];
        if (pred.op == PredicateOp::GeoBox ||
            pred.op == PredicateOp::GeoRadius) {
            // Geo predicates span two columns and use scalar trig; evaluate
            // per-candidate (haversine is the documented scalar exception).
            const uint32_t lng_col = (p < geo_lng_col_indices.size())
                ? geo_lng_col_indices[p] : UINT32_MAX;
            if (lng_col >= cols.size()) { mask = 0; break; }
            uint32_t gmask = 0;
            for (uint32_t j = 0; j < 4; ++j) {
                if (eval_predicate_geo(cols[pred_col_indices[p]],
                                       cols[lng_col],
                                       start_idx + j, pred))
                    gmask |= (1u << j);
            }
            mask &= gmask;
        } else {
            mask &= eval_predicate_batch4(cols[pred_col_indices[p]],
                                          start_idx, pred);
        }
    }
    return mask;
}

/// Check whether a leaf's summary CAN contain a match for the given predicates.
/// Returns false if the summary definitively rules out ALL matches (bloom
/// negative or numeric range miss for any predicate). Returns true if any
/// predicate MIGHT match (conservative — never false-negatives).
///
/// `summary` points to the summary region (summary_size bytes).
bool summary_may_match(const uint8_t* summary, uint32_t summary_size,
                        const Schema& schema,
                        const std::vector<Predicate>& preds,
                        const std::vector<uint32_t>& pred_col_indices,
                        const std::vector<uint32_t>& geo_lng_col_indices = {});

/// Check a bloom filter for a hash. Returns true if the hash MAY be present
/// (bloom positive or empty bloom). False = definitely absent.
inline bool bloom_check(const uint8_t* bloom, uint16_t n_bits, uint32_t hash) {
    if (n_bits == 0) return true;  // no bloom → can't prune
    const BloomProbe bp = bloom_probe(hash);
    for (uint32_t k = 0; k < kBloomK; ++k) {
        const uint32_t bit = bloom_bit(bp, k, n_bits);
        if (!((bloom[bit / 8u] >> (bit % 8u)) & 1u))
            return false;  // bit not set → definitely absent
    }
    return true;
}

// ===========================================================================
// Numeric selectivity estimation from root-level summaries (§3.9)
// ===========================================================================

/// Estimate the selectivity of a single numeric predicate by scanning
/// root-level summary min/max ranges. Returns the fraction of root children
/// whose [min, max] overlaps the predicate's matching range.
///
/// For range predicates (gt, ge, lt, le, between, eq), the "matching range"
/// is the set of values that satisfy the predicate. For Eq, the matching range
/// is [value, value] (a point). We check if that point falls within [min, max].
///
/// This is an OVERESTIMATE of true selectivity (uniform distribution assumed
/// within each subtree), which is conservative for W sizing — better to
/// over-scan than under-scan.
///
/// `root_summaries` = array of pointers to each root child's summary region.
/// `n_root_children` = number of root children.
/// `summary_size` = byte size of each summary.
inline float estimate_numeric_selectivity(
    const uint8_t* const* root_summaries, uint32_t n_root_children,
    uint32_t summary_size, const Schema& schema,
    const Predicate& pred, uint32_t col_idx) {

    if (n_root_children == 0 || summary_size == 0) return 1.0f;

    // Determine the matching range [lo, hi] for this predicate.
    double lo = -std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::max();
    switch (pred.op) {
        case PredicateOp::Eq:      lo = hi = pred.value; break;
        case PredicateOp::Gt:      lo = pred.value; break;  // open, but [min,max] check is the same
        case PredicateOp::Ge:      lo = pred.value; break;
        case PredicateOp::Lt:      hi = pred.value; break;
        case PredicateOp::Le:      hi = pred.value; break;
        case PredicateOp::Between: lo = pred.value; hi = pred.value2; break;
        default: return 1.0f;  // NotEq, In, NotIn: can't estimate range.
    }

    // Count root children whose [min, max] overlaps [lo, hi].
    uint32_t matching = 0;
    for (uint32_t c = 0; c < n_root_children; ++c) {
        if (!root_summaries[c]) continue;
        // Parse this root child's summary to find the numeric range for col_idx.
        const uint8_t* p = root_summaries[c];
        const uint8_t n_numeric = *p++;
        for (uint8_t i = 0; i < n_numeric; ++i) {
            const uint8_t cid = *p++;
            double mn, mx;
            std::memcpy(&mn, p, 8); p += 8;
            std::memcpy(&mx, p, 8); p += 8;
            if (cid == col_idx) {
                // Check overlap: [mn, mx] overlaps [lo, hi] iff mn <= hi && mx >= lo.
                if (mn <= hi && mx >= lo) ++matching;
                break;
            }
        }
    }

    return static_cast<float>(matching) / static_cast<float>(n_root_children);
}

}  // namespace sextant::tree
