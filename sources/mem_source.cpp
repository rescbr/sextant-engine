#include "mem_source.hpp"

#include <cassert>

namespace sextant {

std::unique_ptr<MemSource> MemSourceBuilder::build() {
    // Finalize payload offsets (one extra entry for the last range end).
    payload_offsets_.push_back(static_cast<uint32_t>(payload_data_.size()));

    auto src = std::make_unique<MemSource>();
    src->dim_ = dim_;
    src->count_ = row_ids_.size();
    src->schema_ = std::move(schema_);
    src->vectors_ = std::move(vectors_);
    src->row_ids_ = std::move(row_ids_);
    src->col_data_ = std::move(col_data_);
    src->payload_data_ = std::move(payload_data_);
    src->payload_offsets_ = std::move(payload_offsets_);

    // Pre-size per-chunk scratch.
    src->chunk_col_ptrs_.resize(src->col_data_.size(), nullptr);
    src->chunk_str_cols_.resize(src->col_data_.size());
    src->chunk_set_cols_.resize(src->col_data_.size());

    return src;
}

bool MemSource::next(Chunk& out) {
    if (cursor_ >= count_) {
        out.count = 0;
        return false;
    }

    const uint64_t remaining = count_ - cursor_;
    const uint32_t this_chunk = static_cast<uint32_t>(
        std::min<uint64_t>(chunk_size_, remaining));

    out.vectors = vectors_.data() + cursor_ * dim_;
    out.row_ids = row_ids_.data() + cursor_;
    out.count = this_chunk;

    // Fill filter column pointers for this chunk.
    if (!col_data_.empty()) {
        for (uint32_t c = 0; c < col_data_.size(); ++c) {
            const auto& col = col_data_[c];
            switch (col.type) {
                case ColumnType::Int32:
                case ColumnType::Int64:
                case ColumnType::Float:
                case ColumnType::Bool: {
                    uint8_t w = column_type_width(col.type);
                    chunk_col_ptrs_[c] = col.fixed_data.data() + cursor_ * w;
                    break;
                }
                case ColumnType::String: {
                    auto& sc = chunk_str_cols_[c];
                    sc.offsets = col.str_offsets.data() + cursor_;
                    sc.lengths = col.str_lengths.data() + cursor_;
                    sc.data = col.str_data.data();
                    sc.total_data_bytes = static_cast<uint32_t>(col.str_data.size());
                    chunk_col_ptrs_[c] = &sc;
                    break;
                }
                case ColumnType::Set: {
                    auto& ssc = chunk_set_cols_[c];
                    ssc.counts = col.set_counts.data() + cursor_;
                    ssc.offsets = col.set_offsets.data() + cursor_;
                    ssc.element_lengths = col.set_elem_lengths.data();
                    ssc.element_data = col.set_elem_data.data();
                    ssc.total_elements = static_cast<uint32_t>(col.set_elem_lengths.size());
                    ssc.total_data_bytes = static_cast<uint32_t>(col.set_elem_data.size());
                    chunk_col_ptrs_[c] = &ssc;
                    break;
                }
            }
        }
        out.filter_columns = chunk_col_ptrs_.data();
    } else {
        out.filter_columns = nullptr;
    }

    // Fill payload pointers for this chunk.
    if (!payload_data_.empty()) {
        out.payload_data = payload_data_.data();
        out.payload_offsets = payload_offsets_.data() + cursor_;
    } else {
        out.payload_data = nullptr;
        out.payload_offsets = nullptr;
    }

    cursor_ += this_chunk;
    return true;
}

}  // namespace sextant
