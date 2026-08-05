#pragma once

/// @file filter_column_read.hpp
/// Read on-disk filter column data back into ColumnData.
///
/// This is the inverse of write_filter_columns (filter_column_write.hpp).
/// Used by the mutable path (insert/delete/split) to read existing filter
/// data from a leaf extent, modify it, and write it back.
///
/// On-disk layout per column (in schema order):
///   fixed-width: [count × width bytes]
///   string:      [offsets: count × u32][lengths: count × u16]
///                [hashes: count × u32][data: packed bytes]
///   set:         [counts: count × u8][offsets: count × u32]
///                [hashes: packed u32][data: [u16 len][bytes] per element]

#include "sextant/schema.hpp"
#include <sextant/column_data.hpp>

#include <cstdint>
#include <cstring>

namespace sextant::tree {

/// Read filter column data from `buf` into `out_cols`.
/// `buf` points to the start of the filter column region in the leaf extent.
/// `count` = number of vectors (rows). `schema` determines column types/order.
/// `out_cols` is resized to schema.columns.size() and populated.
/// Returns bytes consumed (should match filter_columns_bytes for the same data).
inline uint64_t read_filter_columns(const uint8_t* buf, uint32_t count,
                                     const Schema& schema,
                                     std::vector<ColumnData>& out_cols) {
    if (schema.columns.empty()) {
        out_cols.clear();
        return 0;
    }

    out_cols.assign(schema.columns.size(), ColumnData{});
    const uint8_t* p = buf;

    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        auto& fc = out_cols[c];
        fc.type = col.type;

        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool: {
                const uint8_t w = column_type_width(col.type);
                fc.fixed_data.assign(p, p + static_cast<size_t>(count) * w);
                p += static_cast<size_t>(count) * w;
                break;
            }
            case ColumnType::String: {
                // [offsets: count × u32][lengths: count × u16]
                // [hashes: count × u32][data: packed bytes]
                p += static_cast<size_t>(count) * 4;  // skip offsets (recomputed)
                const auto* lengths = reinterpret_cast<const uint16_t*>(p);
                p += static_cast<size_t>(count) * 2;
                p += static_cast<size_t>(count) * 4;  // skip hashes (recomputed on write)

                fc.str_offsets.resize(count);
                fc.str_lengths.resize(count);
                uint32_t total_data = 0;
                for (uint32_t i = 0; i < count; ++i) {
                    fc.str_offsets[i] = total_data;
                    fc.str_lengths[i] = lengths[i];
                    total_data += lengths[i];
                }
                fc.str_data.assign(
                    reinterpret_cast<const char*>(p),
                    reinterpret_cast<const char*>(p) + total_data);
                p += total_data;
                break;
            }
            case ColumnType::Set: {
                // [counts: count × u8][offsets: count × u32]
                // [hashes: packed u32][data: [u16 len][bytes] per element]
                const uint8_t* counts = p;
                p += static_cast<size_t>(count) * 1;
                p += static_cast<size_t>(count) * 4;  // skip offsets (recomputed)

                // Total elements.
                uint32_t total_elem = 0;
                for (uint32_t i = 0; i < count; ++i)
                    total_elem += counts[i];
                p += static_cast<size_t>(total_elem) * 4;  // skip hashes

                fc.set_counts.assign(counts, counts + count);
                fc.set_offsets.resize(count);

                // Read element data: [u16 len][bytes] per element.
                // The offsets array gives the element index offset per row.
                // We flatten into set_elem_lengths + set_elem_data.
                uint32_t elem_acc = 0;
                for (uint32_t i = 0; i < count; ++i) {
                    fc.set_offsets[i] = elem_acc;
                    const uint8_t ec = counts[i];
                    for (uint32_t e = 0; e < ec; ++e) {
                        uint16_t elen;
                        std::memcpy(&elen, p, 2);
                        p += 2;
                        fc.set_elem_lengths.push_back(elen);
                        const size_t pos = fc.set_elem_data.size();
                        fc.set_elem_data.resize(pos + elen);
                        std::memcpy(fc.set_elem_data.data() + pos, p, elen);
                        p += elen;
                        ++elem_acc;
                    }
                }
                break;
            }
        }
    }

    return static_cast<uint64_t>(p - buf);
}

/// Select a subset of rows from filter column data (used by delete compaction
/// and split partitioning). `indices` are the row indices to keep, in order.
/// Returns a new vector of ColumnData with only the selected rows.
inline std::vector<ColumnData> select_filter_rows(
    const std::vector<ColumnData>& src,
    const Schema& schema,
    const std::vector<uint32_t>& indices) {

    std::vector<ColumnData> out(schema.columns.size());
    const uint32_t n = static_cast<uint32_t>(indices.size());

    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        const auto& s = src[c];
        auto& d = out[c];
        d.type = col.type;

        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool: {
                const uint8_t w = column_type_width(col.type);
                d.fixed_data.resize(static_cast<size_t>(n) * w);
                for (uint32_t i = 0; i < n; ++i) {
                    std::memcpy(d.fixed_data.data() + static_cast<size_t>(i) * w,
                                s.fixed_data.data() + static_cast<size_t>(indices[i]) * w,
                                w);
                }
                break;
            }
            case ColumnType::String: {
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t src_idx = indices[i];
                    const uint16_t len = s.str_lengths[src_idx];
                    d.str_offsets.push_back(static_cast<uint32_t>(d.str_data.size()));
                    d.str_lengths.push_back(len);
                    const char* str = s.str_data.data() + s.str_offsets[src_idx];
                    d.str_data.insert(d.str_data.end(), str, str + len);
                }
                break;
            }
            case ColumnType::Set: {
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t src_idx = indices[i];
                    const uint8_t ec = s.set_counts[src_idx];
                    const uint32_t src_off = s.set_offsets[src_idx];
                    d.set_counts.push_back(ec);
                    d.set_offsets.push_back(
                        static_cast<uint32_t>(d.set_elem_lengths.size()));
                    for (uint8_t e = 0; e < ec; ++e) {
                        const uint16_t elen = s.set_elem_lengths[src_off + e];
                        d.set_elem_lengths.push_back(elen);
                        // Compute byte offset in source data.
                        uint32_t src_byte_off = 0;
                        for (uint32_t k = 0; k < src_off + e; ++k)
                            src_byte_off += s.set_elem_lengths[k];
                        const char* edata = s.set_elem_data.data() + src_byte_off;
                        d.set_elem_data.insert(d.set_elem_data.end(),
                                               edata, edata + elen);
                    }
                }
                break;
            }
        }
    }

    return out;
}

/// Append filter rows from `src` (at the given indices) to `dst`.
/// Used by insert to add new rows to existing leaf filter data.
inline void append_filter_rows(
    std::vector<ColumnData>& dst,
    const std::vector<ColumnData>& src,
    const Schema& schema,
    const std::vector<uint32_t>& indices) {

    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        const auto& s = src[c];
        auto& d = dst[c];
        d.type = col.type;

        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool: {
                const uint8_t w = column_type_width(col.type);
                for (uint32_t idx : indices) {
                    const size_t pos = d.fixed_data.size();
                    d.fixed_data.resize(pos + w);
                    std::memcpy(d.fixed_data.data() + pos,
                                s.fixed_data.data() + static_cast<size_t>(idx) * w, w);
                }
                break;
            }
            case ColumnType::String: {
                for (uint32_t idx : indices) {
                    const uint16_t len = s.str_lengths[idx];
                    d.str_offsets.push_back(static_cast<uint32_t>(d.str_data.size()));
                    d.str_lengths.push_back(len);
                    const char* str = s.str_data.data() + s.str_offsets[idx];
                    d.str_data.insert(d.str_data.end(), str, str + len);
                }
                break;
            }
            case ColumnType::Set: {
                for (uint32_t idx : indices) {
                    const uint8_t ec = s.set_counts[idx];
                    const uint32_t src_off = s.set_offsets[idx];
                    d.set_counts.push_back(ec);
                    d.set_offsets.push_back(
                        static_cast<uint32_t>(d.set_elem_lengths.size()));
                    for (uint8_t e = 0; e < ec; ++e) {
                        const uint16_t elen = s.set_elem_lengths[src_off + e];
                        d.set_elem_lengths.push_back(elen);
                        uint32_t src_byte_off = 0;
                        for (uint32_t k = 0; k < src_off + e; ++k)
                            src_byte_off += s.set_elem_lengths[k];
                        const char* edata = s.set_elem_data.data() + src_byte_off;
                        d.set_elem_data.insert(d.set_elem_data.end(),
                                               edata, edata + elen);
                    }
                }
                break;
            }
        }
    }
}

}  // namespace sextant::tree
