#pragma once
/// @file memory_source.hpp
/// A VectorSource backed by an in-memory float buffer. Used by estimate_config
/// to build mini-indices on a sample without file I/O for the input data.

#include <sextant/vector_source.hpp>
#include <sextant/types.hpp>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace sextant {

class MemorySource : public VectorSource {
public:
    /// `vectors` must outlive this source. count × dim floats, row-major.
    /// Row IDs are identity [0..count-1].
    MemorySource(const float* vectors, uint64_t count, Dim dim,
                 uint32_t chunk_size = 2048)
        : vectors_(vectors), count_(count), dim_(dim),
          chunk_size_(chunk_size == 0 ? 2048 : chunk_size),
          row_ids_(count) {
        for (uint64_t i = 0; i < count; ++i) row_ids_[i] = static_cast<RowId>(i);
    }

    Dim dim() const override { return dim_; }
    uint64_t count() const override { return count_; }
    void reset() override { cursor_ = 0; }

    bool next(Chunk& out) override {
        if (cursor_ >= count_) { out.count = 0; return false; }
        const uint64_t remaining = count_ - cursor_;
        const uint32_t this_chunk = static_cast<uint32_t>(
            std::min<uint64_t>(chunk_size_, remaining));
        out.vectors = vectors_ + cursor_ * dim_;
        out.row_ids = row_ids_.data() + cursor_;
        out.count = this_chunk;
        cursor_ += this_chunk;
        return true;
    }

private:
    const float* vectors_;
    uint64_t count_;
    Dim dim_;
    uint32_t chunk_size_;
    uint64_t cursor_ = 0;
    std::vector<RowId> row_ids_;
};

}  // namespace sextant
