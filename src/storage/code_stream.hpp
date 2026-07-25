#pragma once

/// @file code_stream.hpp
/// Sequential pread streamer for 4-bit FastScan codes (Option A IVF scan).
///
/// The scan path reads every code in a probed shard exactly once — sequential
/// access, no reuse. Routing these reads through the W-TinyLFU `BlockCache`
/// (tuned for random graph access) would be pure pollution: the cache window
/// is ~1% of capacity, so a single shard stream evicts it ~20× per query, and
/// the blocks are never re-read. `CodeStream` bypasses the cache entirely and
/// reads sequentially via O_DIRECT pread, in 1 MB aligned chunks (4 ×
/// kBlockSize, matching the graph path's pre-read width).
///
/// The reader is stateless across queries: each `scan()` call streams from the
/// beginning. The per-worker staging buffer is owned by the caller (typically
/// the IVF scan worker state) and loaned to `scan()` so multiple shards in one
/// query can reuse one allocation.

#include "storage/direct_io.hpp"
#include "storage/sidecar_header.hpp"

#include <sextant/error.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace sextant {

/// Read-only sequential stream over a `.codes4` sidecar.
///
/// The file layout is `[SidecarHeader 64B][n_blocks × block_bytes]` where
/// `block_bytes = m × 16` (one FastScan block). `n_blocks = ceil(n_vectors /
/// 32)`. `CodeStream` validates the magic on open and exposes the shard's
/// block count + block size; the caller drives the scan via `scan()`.
class CodeStream {
public:
    /// Open `path` (a `.codes4` file) for direct sequential reads. Throws on
    /// open failure or magic mismatch. The file stays open for the lifetime of
    /// this object.
    CodeStream(const std::string& path, uint32_t m);
    ~CodeStream();

    CodeStream(const CodeStream&) = delete;
    CodeStream& operator=(const CodeStream&) = delete;
    CodeStream(CodeStream&&) noexcept;
    CodeStream& operator=(CodeStream&&) noexcept;

    /// Total number of FastScan blocks in this shard (= ceil(n_vectors / 32)).
    uint32_t n_blocks() const { return n_blocks_; }

    /// Bytes per block (= m × 16).
    uint32_t block_bytes() const { return block_bytes_; }

    /// Total number of vectors (decoded from the header; tail block may pad).
    uint32_t n_vectors() const { return n_vectors_; }

    /// Stream the whole shard sequentially, invoking `on_block(block_idx,
    /// block_ptr, valid_mask)` for each block. `block_ptr` points into the
    /// staging buffer; the caller MUST consume it before returning (the next
    /// iteration overwrites the buffer). `valid_mask` is the 32-bit lane mask:
    /// bit j set ⇒ lane j is a real vector; cleared ⇒ padding. The final
    /// partial block has only the real-vector bits set; full blocks get
    /// 0xFFFFFFFF.
    ///
    /// `staging` is a caller-owned aligned buffer of at least `chunk_blocks()
    /// × block_bytes_` bytes. Loaning it (rather than allocating per call)
    /// lets one worker reuse the same buffer across all shards it probes.
    ///
    /// Returns true on success, false on read error (logged via spdlog).
    ///
    /// Not const: `read_exact` advances the underlying file descriptor (the
    /// `DirectFile` API is non-const for reads, matching O_DIRECT pread).
    bool scan(const std::function<void(uint32_t /*block_idx*/,
                                       const uint8_t* /*block_ptr*/,
                                       uint32_t /*valid_mask*/)>& on_block,
              uint8_t* staging);

    /// Number of blocks per sequential pread chunk (4 × 256 KB / block_bytes,
    /// clamped to n_blocks and to at least 1). Matches the graph path's
    /// `kBlocksPerRead = 4` 1 MB pre-read.
    uint32_t chunk_blocks() const { return chunk_blocks_; }

private:
    DirectFile file_;
    uint32_t m_ = 0;
    uint32_t block_bytes_ = 0;     // m × 16
    uint32_t n_blocks_ = 0;
    uint32_t n_vectors_ = 0;
    uint32_t chunk_blocks_ = 1;
};

/// Required staging-buffer byte size for `CodeStream::scan` on `cs`.
/// Allocates enough for one chunk; loan an aligned buffer of at least this
/// size from the worker scratch.
inline size_t code_stream_staging_bytes(const CodeStream& cs) {
    return static_cast<size_t>(cs.chunk_blocks()) * cs.block_bytes();
}

}  // namespace sextant
