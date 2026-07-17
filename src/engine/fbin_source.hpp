#pragma once

/// @file fbin_source.hpp
/// FbinSource: reads .fbin / .ibin / .bbin / .bvecs vector files.
///
/// .fbin layout (little-endian):
///   [uint32 n] [uint32 dim] [n × dim × float32 row-major]
/// .ibin: int8 values (cast to float on read, Issue 39 — NO normalization).
/// .bbin/.bvecs: uint8 values (cast to float on read).
///
/// Regular buffered open()/read() — input data is a sequential scan, not
/// direct-IO. The Engine drives the pull loop via next().

#include "sextant/vector_source.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sextant {

class FbinSource : public VectorSource {
public:
    /// Chunk size in vectors (default 2048).
    static constexpr uint32_t kDefaultChunkSize = 2048;

    /// Open `path` and read the header. Infers element type from the extension.
    explicit FbinSource(const std::string& path,
                        uint32_t chunk_size = kDefaultChunkSize);
    ~FbinSource() override;

    FbinSource(const FbinSource&) = delete;
    FbinSource& operator=(const FbinSource&) = delete;

    Dim dim() const override { return dim_; }
    uint64_t count() const override { return count_; }
    void reset() override;
    bool next(Chunk& out) override;
    std::string path() const override { return path_; }

private:
    enum class ElemType { Float32, Int8, Uint8 };
    static ElemType infer_type(const std::string& path);

    std::string path_;
    int fd_ = -1;
    ElemType type_ = ElemType::Float32;
    uint32_t dim_ = 0;
    uint64_t count_ = 0;
    uint32_t chunk_size_ = 0;

    /// File offset where the vector data begins (after the 8-byte header).
    uint64_t data_offset_ = 8;

    /// Next vector index to read (advances across next() calls).
    uint64_t cursor_ = 0;

    /// Internal float buffer for decoded vectors (chunk_size × dim).
    std::vector<float> vec_buf_;
    /// Sequential 0-indexed row_ids.
    std::vector<RowId> rowid_buf_;
    /// Byte buffer for int8/uint8 raw reads (cast into vec_buf_).
    std::vector<uint8_t> raw_buf_;
};

}  // namespace sextant
