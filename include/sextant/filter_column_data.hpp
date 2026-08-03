#pragma once

/// @file filter_column_data.hpp
/// In-memory representations of filter column data for one chunk.
///
/// These structures are produced by VectorSource implementations and consumed
/// by the tree build path (leaf emission). They are NOT the on-disk layout —
/// the on-disk layout is defined in tree_nodes.hpp / the leaf writer.
///
/// For fixed-width types (int32/int64/float/bool), the chunk's filter_columns
/// entry points directly to a contiguous array (count × width bytes). No
/// wrapper struct needed.
///
/// For string and set types, the entry points to one of these structs.

#include <cstdint>

namespace sextant {

/// String column data for one chunk (non-owning view).
/// Layout matches the on-disk format (§3.3 of the architecture plan):
///   [offsets: count × uint32]   — byte offset into data for each row's string
///   [lengths: count × uint16]   — string length per row
///   [data:     packed bytes]    — actual string content (not null-terminated)
///
/// Hashes are computed at build time during leaf emission (not provided by
/// the source — the source provides raw strings, the builder hashes them).
struct FilterStringColumn {
    const uint32_t* offsets;  ///< count entries; byte offset into data
    const uint16_t* lengths;  ///< count entries; string length per row
    const char*     data;     ///< packed string bytes
    uint32_t        total_data_bytes;  ///< size of the data region
};

/// Set column data for one chunk (non-owning view).
/// Each row has a variable number of string elements.
/// Layout (§3.3.1):
///   [counts:  count × uint8]    — number of elements per row (0-255)
///   [offsets: count × uint32]   — element index offset into element_lengths/element_data
///   [element_lengths: total_elements × uint16] — length of each element string
///   [element_data:     packed bytes]           — all element strings concatenated
struct FilterSetColumn {
    const uint8_t*  counts;           ///< count entries; elements per row
    const uint32_t* offsets;          ///< count entries; index offset into element arrays
    const uint16_t* element_lengths;  ///< total_elements entries
    const char*     element_data;     ///< packed element string bytes
    uint32_t        total_elements;   ///< total number of elements across all rows
    uint32_t        total_data_bytes; ///< size of the element_data region
};

}  // namespace sextant
