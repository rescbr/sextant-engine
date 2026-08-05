#pragma once

#include <sextant/schema.hpp>
#include <cstdint>
#include <vector>

namespace sextant {

/// Per-column in-memory data storage for filter columns. Used by the build
/// path (BuildConfig), filter column I/O, and insert/delete internals.
struct ColumnData {
    ColumnType type = ColumnType::Int32;

    // Fixed-width: raw bytes, width per row. width = column_type_width(type).
    std::vector<uint8_t> fixed_data;

    // String: per-row (offset, length) into str_data.
    std::vector<uint32_t> str_offsets;
    std::vector<uint16_t> str_lengths;
    std::vector<char>     str_data;

    // Set: per-row (count, offset into elem_lengths), plus element data.
    std::vector<uint8_t>  set_counts;
    std::vector<uint32_t> set_offsets;
    std::vector<uint16_t> set_elem_lengths;
    std::vector<char>     set_elem_data;
};

}  // namespace sextant