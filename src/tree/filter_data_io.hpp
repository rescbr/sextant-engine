#pragma once

/// @file filter_data_io.hpp
/// Sidecar filter data file format (.fdat) for CLI integration.
///
/// Format:
///   [magic: u32 = 0x46444154 "FDAT"]
///   [n_rows: u64]
///   [n_cols: u32]
///   [col_types: n_cols × u8]  (ColumnType enum values)
///   [has_payload: u8]
///   [payload_offsets: (n_rows+1) × u32]  (if has_payload)
///   [payload_data: packed bytes]          (if has_payload)
///   [per column, in order:]
///     fixed-width: n_rows × width bytes
///     string: [n_rows × u32 offsets][n_rows × u16 lengths][packed data bytes]
///     set: [n_rows × u8 counts][n_rows × u32 offsets]
///          [total_elems × u16 elem_lengths][packed elem data bytes]
///
/// The file is read at build time and passed to BuildConfig::filter_column_data.

#include "sextant/schema.hpp"
#include "engine/mem_source.hpp"  // MemColumnData

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace sextant::tree {

inline constexpr uint32_t kFilterDataMagic = 0x46444154u;  // "FDAT" LE

/// Write filter column data to a .fdat file.
/// `cols` = per-column data (n_rows entries each).
/// `schema` = schema describing the columns.
/// `payload_data` / `payload_offsets` = optional payload (empty = no payload).
void write_filter_data(const std::string& path,
                        const Schema& schema,
                        const std::vector<MemColumnData>& cols,
                        const std::vector<uint8_t>& payload_data = {},
                        const std::vector<uint32_t>& payload_offsets = {});

/// Read filter column data from a .fdat file.
/// Returns (schema, cols, payload_data, payload_offsets).
struct FilterDataFile {
    Schema schema;
    std::vector<MemColumnData> cols;
    std::vector<uint8_t> payload_data;
    std::vector<uint32_t> payload_offsets;
    uint64_t n_rows = 0;
    bool has_payload = false;
};

FilterDataFile read_filter_data(const std::string& path);

}  // namespace sextant::tree
