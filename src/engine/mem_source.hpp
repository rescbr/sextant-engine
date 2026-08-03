#pragma once

/// @file mem_source.hpp
/// In-memory VectorSource with filter columns and payload.
///
/// Used for testing the filter column storage pipeline (Phase C). Owns all
/// data internally — vectors, filter column values, and payload blobs.
///
/// Usage:
///   MemSourceBuilder b(dim);
///   b.set_schema({{"category", ColumnType::String}, {"year", ColumnType::Int32}});
///   for each vector:
///     uint32_t row = b.add_vector(vec, row_id);
///     b.set_string(0, "cs.AI");    // column 0 = "category"
///     b.set_int32(1, 2024);        // column 1 = "year"
///     b.set_payload(blob_bytes);
///   auto source = b.build();

#include <sextant/error.hpp>
#include <sextant/filter_column_data.hpp>
#include <sextant/schema.hpp>
#include <sextant/types.hpp>
#include <sextant/vector_source.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sextant {

/// Per-column in-memory data storage (used by MemSourceBuilder + MemSource).
struct MemColumnData {
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

/// Builder for MemSource. Collects vectors + filter column values + payload,
/// then produces a MemSource that owns all data.
class MemSourceBuilder {
public:
    explicit MemSourceBuilder(Dim dim) : dim_(dim) {}

    void set_schema(const Schema& schema) {
        schema_ = schema;
        col_data_.resize(schema.columns.size());
        for (uint32_t i = 0; i < schema.columns.size(); ++i) {
            col_data_[i].type = schema.columns[i].type;
        }
    }

    /// Add a vector (dim floats). Returns the row index (0-based).
    uint32_t add_vector(const float* vec, RowId row_id) {
        vectors_.insert(vectors_.end(), vec, vec + dim_);
        row_ids_.push_back(row_id);
        // Grow per-column storage with a default (empty/zero) row.
        for (auto& col : col_data_) {
            switch (col.type) {
                case ColumnType::Int32:
                case ColumnType::Int64:
                case ColumnType::Float:
                case ColumnType::Bool: {
                    uint8_t w = column_type_width(col.type);
                    col.fixed_data.resize(col.fixed_data.size() + w, 0);
                    break;
                }
                case ColumnType::String:
                    col.str_offsets.push_back(static_cast<uint32_t>(col.str_data.size()));
                    col.str_lengths.push_back(0);
                    break;
                case ColumnType::Set:
                    col.set_counts.push_back(0);
                    col.set_offsets.push_back(static_cast<uint32_t>(col.set_elem_lengths.size()));
                    break;
            }
        }
        payload_offsets_.push_back(static_cast<uint32_t>(payload_data_.size()));
        return static_cast<uint32_t>(row_ids_.size() - 1);
    }

    // --- Fixed-width column setters (call after add_vector for the current row) ---

    void set_int32(uint32_t col_idx, int32_t val) {
        auto& col = col_data_[col_idx];
        std::memcpy(&col.fixed_data[col.fixed_data.size() - 4], &val, 4);
    }
    void set_int64(uint32_t col_idx, int64_t val) {
        auto& col = col_data_[col_idx];
        std::memcpy(&col.fixed_data[col.fixed_data.size() - 8], &val, 8);
    }
    void set_float(uint32_t col_idx, float val) {
        auto& col = col_data_[col_idx];
        std::memcpy(&col.fixed_data[col.fixed_data.size() - 4], &val, 4);
    }
    void set_bool(uint32_t col_idx, bool val) {
        auto& col = col_data_[col_idx];
        col.fixed_data[col.fixed_data.size() - 1] = val ? 1 : 0;
    }

    // --- Variable-length column setters ---

    void set_string(uint32_t col_idx, std::string_view val) {
        if (val.size() > 65535) {
            throw Error(ErrorCode::InvalidParam,
                "MemSourceBuilder::set_string: length " +
                std::to_string(val.size()) + " exceeds uint16 max (65535)");
        }
        auto& col = col_data_[col_idx];
        col.str_offsets.back() = static_cast<uint32_t>(col.str_data.size());
        col.str_lengths.back() = static_cast<uint16_t>(val.size());
        col.str_data.insert(col.str_data.end(), val.data(), val.data() + val.size());
    }

    void set_set(uint32_t col_idx, const std::vector<std::string_view>& elements) {
        if (elements.size() > 255) {
            throw Error(ErrorCode::InvalidParam,
                "MemSourceBuilder::set_set: element count " +
                std::to_string(elements.size()) + " exceeds uint8 max (255)");
        }
        for (const auto& e : elements) {
            if (e.size() > 65535) {
                throw Error(ErrorCode::InvalidParam,
                    "MemSourceBuilder::set_set: element length " +
                    std::to_string(e.size()) + " exceeds uint16 max (65535)");
            }
        }
        auto& col = col_data_[col_idx];
        col.set_counts.back() = static_cast<uint8_t>(elements.size());
        col.set_offsets.back() = static_cast<uint32_t>(col.set_elem_lengths.size());
        for (auto& e : elements) {
            col.set_elem_lengths.push_back(static_cast<uint16_t>(e.size()));
            col.set_elem_data.insert(col.set_elem_data.end(), e.data(), e.data() + e.size());
        }
    }

    // --- Payload ---

    void set_payload(std::string_view blob) {
        payload_data_.insert(payload_data_.end(),
                             reinterpret_cast<const uint8_t*>(blob.data()),
                             reinterpret_cast<const uint8_t*>(blob.data()) + blob.size());
    }

    // --- Build ---

    /// Produce a MemSource that owns all the collected data.
    std::unique_ptr<class MemSource> build();

    Schema& schema_mut() { return schema_; }

private:
    friend class MemSource;

    Dim dim_;
    Schema schema_;
    std::vector<float> vectors_;
    std::vector<RowId> row_ids_;
    std::vector<MemColumnData> col_data_;
    std::vector<uint8_t> payload_data_;
    std::vector<uint32_t> payload_offsets_;
};

/// In-memory VectorSource with filter columns + payload. Built by MemSourceBuilder.
/// Owns all data. Chunked reading for the build path.
class MemSource : public VectorSource {
public:
    Dim dim() const override { return dim_; }
    uint64_t count() const override { return count_; }
    Schema schema() const override { return schema_; }
    void reset() override { cursor_ = 0; }

    bool next(Chunk& out) override;

private:
    friend class MemSourceBuilder;

    Dim dim_ = 0;
    uint64_t count_ = 0;
    Schema schema_;
    uint32_t chunk_size_ = 2048;
    uint64_t cursor_ = 0;

    // Owned data.
    std::vector<float> vectors_;
    std::vector<RowId> row_ids_;
    std::vector<MemColumnData> col_data_;
    std::vector<uint8_t> payload_data_;
    std::vector<uint32_t> payload_offsets_;

    // Per-chunk scratch (valid until next next() call).
    std::vector<const void*> chunk_col_ptrs_;
    std::vector<FilterStringColumn> chunk_str_cols_;
    std::vector<FilterSetColumn> chunk_set_cols_;
};

}  // namespace sextant
