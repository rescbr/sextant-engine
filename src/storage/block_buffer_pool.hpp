#pragma once

/// @file block_buffer_pool.hpp
/// Pool of pre-allocated, aligned block buffers for the block cache.
///
/// Recycling buffers avoids the `aligned_alloc`/`aligned_free` syscall churn
/// (on macOS every free triggers `madvise(MADV_FREE)`, which dominates under
/// high eviction rates). The pool grows organically to the cache's steady-state
/// working set and then serves all requests from the free list.
///
/// Thread safety: a mutex guards the free list. It is contended only on insert
/// (cache miss) and free (eviction) — never on the hot lookup path. Each
/// CacheShard owns its own pool, so the mutex is never contended across
/// shards (acquire/release happen under the shard's existing write lock).

#include <sextant/sync.hpp>
#include <sextant/types.hpp>
#include <cstdint>
#include <vector>

namespace sextant {

class BlockBufferPool {
public:
    /// `block_size` is the size of each buffer (typically kBlockSize = 256KB).
    explicit BlockBufferPool(uint32_t block_size);
    ~BlockBufferPool();

    BlockBufferPool(const BlockBufferPool&) = delete;
    BlockBufferPool& operator=(const BlockBufferPool&) = delete;

    /// Acquire a buffer. Returns a kDiskAlign-aligned buffer of `block_size_`
    /// bytes. If the pool has a recycled buffer, returns it; otherwise
    /// allocates a new one.
    uint8_t* acquire();

    /// Release a buffer back to the pool for reuse.
    void release(uint8_t* buf);

    /// Number of buffers currently sitting in the free list (diagnostic).
    uint32_t free_count() const;

private:
    uint32_t block_size_;
    std::vector<uint8_t*> free_list_;
    mutable Mutex mu_;
};

}  // namespace sextant
