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

#include "sextant/schema.hpp"
#include "filter_hash.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sextant::tree {

/// Per-column cardinality data.
struct ColumnCardinality {
    // String/set: hash(value) → count.
    std::unordered_map<uint32_t, uint32_t> freq;

    // Numeric: sorted (value_as_double, count) for range selectivity.
    // Built at finalize time from the raw value counts.
    std::map<double, uint32_t> numeric_hist;
    bool is_numeric = false;
};

/// Global cardinality table — one entry per filter column.
class CardinalityTable {
public:
    /// Initialize for the given schema. Allocates per-column maps for
    /// all filter column types (numeric, string, set).
    void init(const Schema& schema, uint64_t n_vectors);

    /// Record a string value for a column during the build emission pass.
    void add_string(uint32_t col_idx, std::string_view val);

    /// Record a set's elements for a column.
    void add_set(uint32_t col_idx,
                 const uint8_t* counts, const uint32_t* offsets,
                 const uint16_t* elem_lengths, const char* elem_data,
                 uint32_t row_idx);

    /// Record a numeric value for a column.
    void add_numeric(uint32_t col_idx, double val);

    /// Estimate selectivity for a string equality predicate.
    float selectivity_string(uint32_t col_idx, std::string_view val) const;

    /// Estimate selectivity for a set CONTAINS predicate.
    float selectivity_set(uint32_t col_idx, std::string_view val) const;

    /// Estimate selectivity for a numeric predicate (eq, gt, ge, lt, le, between).
    float selectivity_numeric(uint32_t col_idx, const Predicate& pred) const;

    /// Combined selectivity for all predicates (product of individual
    /// selectivities, assuming independence). Clamped to [0, 1].
    float selectivity_combined(
        const Schema& schema,
        const std::vector<struct Predicate>& preds,
        const std::vector<uint32_t>& pred_col_indices) const;

    bool empty() const { return columns_.empty(); }
    uint64_t n_vectors() const { return n_vectors_; }

    /// Serialize to a binary blob.
    /// Format:
    ///   [n_vectors: u64]
    ///   [n_columns: u32]
    ///   for each column with data:
    ///     [col_id: u32][is_numeric: u8][n_entries: u32]
    ///     if numeric: [n_entries × (value: f64, count: u32)]
    ///     else:       [n_entries × (hash: u32, count: u32)]
    std::vector<uint8_t> serialize() const;

    /// Deserialize from a binary blob.
    void deserialize(const uint8_t* data, size_t len);

private:
    std::vector<ColumnCardinality> columns_;
    std::vector<uint32_t> col_to_idx_;  // schema col index → columns_ index
    uint64_t n_vectors_ = 0;
};

}  // namespace sextant::tree
