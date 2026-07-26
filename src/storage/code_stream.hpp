#pragma once

/// @file code_stream.hpp
/// Zero-copy sequential reader for 4-bit FastScan codes (Option A IVF scan).
///
/// Each `.codes4` file is mmap'd (MAP_PRIVATE) on open. The OS page cache
/// pages become the process's address space — `scan()` iterates the mapped
/// region directly with zero kernel→user copies. This eliminates the 30%
/// `__arch_copy_to_user` overhead measured with `pread`-based buffered I/O
/// on c4a V2. Hot shards stay resident in the page cache automatically; cold
/// shards page-fault from NVMe on first access (sequential, readahead-friendly).
///
/// The reader is stateless across queries: each `scan()` call streams from the
/// beginning. The per-worker staging buffer is owned by the caller and loaned
/// to `scan()` as a fallback when mmap fails at open time.

#include "storage/buffered_io.hpp"
#include "storage/sidecar_header.hpp"

#include <sextant/error.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace sextant {

class CodeStream {
public:
    CodeStream(const std::string& path, uint32_t m);
    ~CodeStream();

    CodeStream(const CodeStream&) = delete;
    CodeStream& operator=(const CodeStream&) = delete;
    CodeStream(CodeStream&&) noexcept;
    CodeStream& operator=(CodeStream&&) noexcept;

    uint32_t n_blocks() const { return n_blocks_; }
    uint32_t block_bytes() const { return block_bytes_; }
    uint32_t n_vectors() const { return n_vectors_; }
    uint32_t chunk_blocks() const { return chunk_blocks_; }

    bool scan(const std::function<void(uint32_t /*block_idx*/,
                                       const uint8_t* /*block_ptr*/,
                                       uint32_t /*valid_mask*/)>& on_block,
              uint8_t* staging);

private:
    BufferedFile file_;
    uint32_t m_ = 0;
    uint32_t block_bytes_ = 0;
    uint32_t n_blocks_ = 0;
    uint32_t n_vectors_ = 0;
    uint32_t chunk_blocks_ = 1;
    const uint8_t* mapped_ = nullptr;  // mmap'd payload; nullptr if mmap failed
    uint64_t file_size_ = 0;
};

inline size_t code_stream_staging_bytes(const CodeStream& cs) {
    return static_cast<size_t>(cs.chunk_blocks()) * cs.block_bytes();
}

}  // namespace sextant
