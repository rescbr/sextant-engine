#pragma once

/// @file schema.hpp
/// Type system for filter columns, schemas, and predicates.
///
/// A schema declares the typed filter columns present in an index. Predicates
/// reference columns by name and are evaluated against per-leaf columnar data
/// (Phase C) and per-leaf summaries (Phase D). Phase A locks these interfaces;
/// no behavioral change.

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace sextant {

/// Filter column value types. No fp16/fp64. int32/int64/float/bool/string/set.
enum class ColumnType : uint8_t {
    Int32,
    Int64,
    Float,   // f32
    Bool,
    String,
    Set,     // set of strings
};

/// Byte width of a fixed-type column value. Returns 0 for String/Set (variable).
inline uint8_t column_type_width(ColumnType t) {
    switch (t) {
        case ColumnType::Int32:  return 4;
        case ColumnType::Int64:  return 8;
        case ColumnType::Float:  return 4;
        case ColumnType::Bool:   return 1;
        case ColumnType::String: return 0;
        case ColumnType::Set:    return 0;
    }
    return 0;
}

/// Whether a column type is fixed-width (int32/int64/float/bool).
inline bool is_fixed_width(ColumnType t) {
    return column_type_width(t) != 0;
}

/// Lowercase name of a column type (error messages / diagnostics).
inline std::string_view column_type_name(ColumnType t) {
    switch (t) {
        case ColumnType::Int32:  return "int32";
        case ColumnType::Int64:  return "int64";
        case ColumnType::Float:  return "float";
        case ColumnType::Bool:   return "bool";
        case ColumnType::String: return "string";
        case ColumnType::Set:    return "set";
    }
    return "?";
}

/// A declared filter column.
struct FilterColumn {
    std::string name;
    ColumnType type = ColumnType::Int32;
    /// Nullable columns store a per-row validity byte in leaf extents
    /// (1 = NULL) and evaluate predicates with SQL three-valued logic:
    /// NULL fails every comparison. Non-nullable columns (default, and
    /// all trees built before the flag existed) store no validity data.
    bool nullable = false;
};

/// A declared schema (set of filter columns + payload flag).
struct Schema {
    std::vector<FilterColumn> columns;
    bool has_payload = false;

    bool empty() const { return columns.empty() && !has_payload; }

    uint32_t n_filter_columns() const {
        return static_cast<uint32_t>(columns.size());
    }

    /// Find a column by name. Returns nullptr if not found.
    const FilterColumn* find(std::string_view name) const {
        for (const auto& col : columns) {
            if (col.name == name) return &col;
        }
        return nullptr;
    }

    /// Maximum bloom byte allocation for a string/set column at the given
    /// distinct-value threshold. bloom_max_bytes(threshold) =
    /// ceil((-threshold × ln(0.01) / ln(2)²) / 8). At threshold=150 → ~180.
    static uint32_t bloom_max_bytes(uint32_t threshold) {
        // Number of bits for 1% false-positive rate:
        //   n_bits = -n × ln(p) / (ln 2)²,  with p = 0.01.
        constexpr double kLn2 = 0.6931471805599453;
        constexpr double kNegLn001 = 4.605170185988091;  // -ln(0.01)
        const double denom = kLn2 * kLn2;
        // n_bits = -n × ln(p) / (ln 2)², p = 0.01 → n_bits = n × kNegLn001 / denom.
        const double n_bits =
            static_cast<double>(threshold) * kNegLn001 / denom;
        // Ceiling division to whole bytes.
        const uint64_t n_bytes = (static_cast<uint64_t>(n_bits) + 7) / 8;
        return static_cast<uint32_t>(n_bytes);
    }

    /// Schema-determined summary size (same for ALL leaves in the index).
    /// Computed at build time from the column types. 0 when columns empty.
    ///
    /// Summary layout (plan §3.4):
    ///   [n_numeric:u8]
    ///   [n_numeric × (col_id:u8 + min:8B + max:8B)]   — 17B per numeric column
    ///   [n_string_bloom:u8]
    ///   [n_string_bloom × (col_id:u8 + n_bits:u16 + bloom_bytes)]
    ///   [n_set_bloom:u8]
    ///   [n_set_bloom × (col_id:u8 + n_bits:u16 + bloom_bytes)]
    uint32_t summary_size(uint32_t bloom_threshold = 150) const {
        if (columns.empty()) return 0;

        uint32_t n_numeric = 0;
        uint32_t n_string_bloom = 0;  // string + set columns
        for (const auto& col : columns) {
            switch (col.type) {
                case ColumnType::Int32:
                case ColumnType::Int64:
                case ColumnType::Float:
                    ++n_numeric;
                    break;
                case ColumnType::Bool:
                    // Bool is fixed-width but contributes no numeric min/max
                    // and no bloom (only 2 distinct values).
                    break;
                case ColumnType::String:
                case ColumnType::Set:
                    ++n_string_bloom;
                    break;
            }
        }

        const uint32_t bloom_bytes = bloom_max_bytes(bloom_threshold);

        // 3 count bytes (n_numeric, n_string_bloom, n_set_bloom).
        uint32_t size = 3;
        size += n_numeric * (1u + 8u + 8u);  // col_id + min + max
        // Each string/set column: col_id (1) + n_bits (2) + bloom_bytes.
        size += n_string_bloom * (1u + 2u + bloom_bytes);
        return size;
    }
};

// ===========================================================================
// Predicates
// ===========================================================================

/// Predicate operators.
enum class PredicateOp : uint8_t {
    Eq, NotEq, Lt, Le, Gt, Ge, Between, Prefix, In, NotIn,
    // Set ops
    Contains, ContainsAny, ContainsAll,
    // Geo
    GeoRadius, GeoBox,
    // Null tests (SQL IS NULL / IS NOT NULL). On non-nullable columns
    // IsNull is statically false / IsNotNull statically true.
    IsNull, IsNotNull,
};

/// A single predicate on one filter column.
struct Predicate {
    std::string column;         // column name
    PredicateOp op = PredicateOp::Eq;

    // Value variants. Which is active depends on the column type + op.
    // For Between: value = low, value2 = high.
    // For In/NotIn: values is the set.
    // For GeoRadius: value = center_lat, value2 = center_lng, radius_km = radius.
    //                `column` names the latitude column, geo_lng_column names
    //                the longitude column.
    // For GeoBox: value = min_lat, value2 = min_lng,
    //             value3 = max_lat, value4 = max_lng.
    //             `column` names the latitude column, geo_lng_column names
    //             the longitude column.
    double value = 0.0;
    double value2 = 0.0;
    double value3 = 0.0;
    double value4 = 0.0;
    double radius_km = 0.0;
    // For geo predicates (GeoBox, GeoRadius): name of the longitude column.
    // `column` names the latitude column. Empty = unset.
    // Resolved to a numeric index at search time alongside `column`.
    std::string geo_lng_column;
    std::string str_value;             // for Eq/NotEq/Prefix on string columns
    std::vector<std::string> values;   // for In/NotIn (string or numeric-as-string)
};

/// Is `op` evaluable against a column of type `type`? Search-time predicate
/// resolution rejects combinations this returns false for (loud InvalidParam
/// instead of the historical silent pass-all / zero-result behaviors).
/// Geo ops (GeoRadius/GeoBox) apply to the lat/lng column pair and require
/// numeric columns.
inline bool predicate_applies_to(PredicateOp op, ColumnType type) {
    const bool numeric = type == ColumnType::Int32 ||
                         type == ColumnType::Int64 ||
                         type == ColumnType::Float;
    switch (op) {
        case PredicateOp::Eq:
        case PredicateOp::NotEq:
            return numeric || type == ColumnType::String ||
                   type == ColumnType::Bool;
        case PredicateOp::In:
        case PredicateOp::NotIn:
            return numeric || type == ColumnType::String;
        case PredicateOp::Lt:
        case PredicateOp::Le:
        case PredicateOp::Gt:
        case PredicateOp::Ge:
        case PredicateOp::Between:
            return numeric;
        case PredicateOp::Prefix:
            return type == ColumnType::String;
        case PredicateOp::Contains:
        case PredicateOp::ContainsAny:
        case PredicateOp::ContainsAll:
            return type == ColumnType::Set;
        case PredicateOp::GeoRadius:
        case PredicateOp::GeoBox:
            return numeric;  // lat + lng columns
        case PredicateOp::IsNull:
        case PredicateOp::IsNotNull:
            return true;  // applies to every column type
    }
    return false;
}

}  // namespace sextant
