// NodeStore backends.
//
// FlatNodeStore is trivial pointer arithmetic. PagedNodeStore drives the LRU
// cache + DirectFile path: each pin is a shard-locked cache lookup with a
// disk read on miss. The block size is kBlockSize (256 KB); node and code
// records are packed contiguously within their respective files.

#include "node_store.hpp"

#include "sextant/error.hpp"

#include <algorithm>
#include <cstring>

namespace sextant {

namespace {

/// Thread-local aligned staging buffer for cache-miss reads.
/// Avoids per-miss heap allocation and ensures 4096-byte alignment for O_DIRECT.
/// kBlockSize (256KB) is the maximum block size we'll ever read.
struct alignas(kDiskAlign) AlignedStaging {
    uint8_t data[kBlockSize];
};

thread_local AlignedStaging g_staging;

}  // namespace

// ===========================================================================
// FlatNodeStore
// ===========================================================================

FlatNodeStore::FlatNodeStore(uint8_t* nodes, const uint8_t* codes,
                             uint32_t node_size, uint8_t code_size)
    : nodes_(nodes),
      codes_(codes),
      node_size_(node_size),
      code_size_(code_size) {}

const uint8_t* FlatNodeStore::pin_node(uint32_t id) {
    return nodes_ + static_cast<size_t>(id) * node_size_;
}

const uint8_t* FlatNodeStore::pin_code(uint32_t id) {
    return codes_ + static_cast<size_t>(id) * code_size_;
}

uint8_t* FlatNodeStore::mutable_node(uint32_t id) {
    return nodes_ + static_cast<size_t>(id) * node_size_;
}

uint8_t* FlatNodeStore::mutable_code(uint32_t id) {
    // codes_ is const-correct for read; build needs write access. The engine
    // owns a writable buffer and passes it through as non-const via the
    // constructor taking `const uint8_t*` for read safety — we cast away
    // const here because the buffer is in fact writable (allocated by the
    // engine with aligned_alloc).
    return const_cast<uint8_t*>(codes_) +
           static_cast<size_t>(id) * code_size_;
}

// ===========================================================================
// PagedNodeStore
// ===========================================================================

PagedNodeStore::PagedNodeStore(const std::string& graph_path,
                               const std::string& codes_path,
                               uint32_t node_size, uint8_t code_size,
                               uint32_t num_shards,
                               uint64_t cache_size_bytes)
    : graph_file_(graph_path, /*create=*/false),
      codes_file_(codes_path, /*create=*/false),
      node_size_(node_size),
      code_size_(code_size),
      nodes_per_block_(std::max(1u, kBlockSize / std::max(1u, node_size))),
      codes_per_block_(std::max(1u, kBlockSize / std::max(1u, static_cast<uint32_t>(code_size)))),
      graph_block_size_(nodes_per_block_ * node_size),
      codes_block_size_(codes_per_block_ * static_cast<uint32_t>(code_size)),
      cache_(num_shards,
             std::max(1u, static_cast<uint32_t>(
                               cache_size_bytes / kBlockSize / num_shards)),
             kBlockSize) {}

const uint8_t* PagedNodeStore::get_node_block(uint64_t block_idx) {
    LRUShard& shard = cache_.shard(block_idx);
    ScopedWriteLock lock(shard.mutex());
    uint8_t* hit = shard.lookup(block_idx);
    if (hit) {
        return hit;
    }
    // Miss: read the block from disk into the thread-local staging buffer,
    // then insert (which copies into the shard's own aligned buffer).
    const uint64_t off = sizeof(SidecarHeader) + block_idx * graph_block_size_;
    graph_file_.pread_aligned(g_staging.data, graph_block_size_, off);
    {
        std::lock_guard<std::mutex> g(reads_mu_);
        ++graph_reads_;
    }
    return shard.insert(block_idx, g_staging.data, graph_block_size_);
}

const uint8_t* PagedNodeStore::get_code_block(uint64_t block_idx) {
    const uint64_t key = block_idx | kCodeKeyBit;
    LRUShard& shard = cache_.shard(key);
    ScopedWriteLock lock(shard.mutex());
    uint8_t* hit = shard.lookup(key);
    if (hit) {
        return hit;
    }
    const uint64_t off = sizeof(SidecarHeader) + block_idx * codes_block_size_;
    codes_file_.pread_aligned(g_staging.data, codes_block_size_, off);
    {
        std::lock_guard<std::mutex> g(reads_mu_);
        ++code_reads_;
    }
    return shard.insert(key, g_staging.data, codes_block_size_);
}

const uint8_t* PagedNodeStore::pin_node(uint32_t id) {
    const uint64_t block_idx = id / nodes_per_block_;
    const uint32_t off_in_block = (id % nodes_per_block_) * node_size_;
    return get_node_block(block_idx) + off_in_block;
}

void PagedNodeStore::unpin_node(uint32_t) {
    // Phase 1: the LRU manages eviction; no per-pin refcount. The returned
    // pointer stays valid until the block is evicted, which only happens when
    // enough other blocks are pinned to push it out of the LRU window.
}

const uint8_t* PagedNodeStore::pin_code(uint32_t id) {
    const uint64_t block_idx = id / codes_per_block_;
    const uint32_t off_in_block = (id % codes_per_block_) * code_size_;
    return get_code_block(block_idx) + off_in_block;
}

void PagedNodeStore::unpin_code(uint32_t) {}

uint8_t* PagedNodeStore::mutable_node(uint32_t) {
    throw Error(ErrorCode::NotImplemented,
                "PagedNodeStore does not support mutable_node (build-only)");
}

uint8_t* PagedNodeStore::mutable_code(uint32_t) {
    throw Error(ErrorCode::NotImplemented,
                "PagedNodeStore does not support mutable_code (build-only)");
}

}  // namespace sextant
