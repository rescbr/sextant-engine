// NodeStore backends.
//
// FlatNodeStore is trivial pointer arithmetic. PagedNodeStore drives the LRU
// cache + DirectFile path: each pin is a shard-locked cache lookup with a
// disk read on miss. The block size is kBlockSize (256 KB); node and code
// records are packed contiguously within their respective files.
//
// Layer 3 (2026-07-21): PagedNodeStore is single-threaded per worker — each
// Searcher pool worker owns its own instance. All former thread_local
// machinery (g_tl_caches, tl_l1_counters_, tl_fast_, g_staging) is now plain
// per-instance state.

#include "node_store.hpp"

#include "sextant/error.hpp"

#include <algorithm>
#include <cstring>

namespace sextant {

// ===========================================================================
// FlatNodeStore
// ===========================================================================

FlatNodeStore::FlatNodeStore(uint8_t* nodes, const uint8_t* codes,
                              uint32_t node_size, uint32_t code_size)
    : nodes_(nodes),
      codes_(codes),
      node_size_(node_size),
      code_size_(code_size) {}

PinResult FlatNodeStore::pin_node(uint32_t id) {
    return {nodes_ + static_cast<size_t>(id) * node_size_, false};
}

PinResult FlatNodeStore::pin_code(uint32_t id) {
    return {codes_ + static_cast<size_t>(id) * code_size_, false};
}

// ===========================================================================
// PagedNodeStore
// ===========================================================================

PagedNodeStore::PagedNodeStore(const std::string& graph_path,
                                const std::string& codes_path,
                                uint32_t node_size, uint32_t code_size,
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
      graph_cache_(num_shards,
                   [&] {
                       const uint64_t graph_bytes = graph_file_.size();
                       const uint64_t codes_bytes = codes_file_.size();
                       const uint64_t total = graph_bytes + codes_bytes;
                       const uint64_t graph_budget = total > 0
                           ? cache_size_bytes * graph_bytes / total
                           : cache_size_bytes / 2;
                       return std::max(1u, static_cast<uint32_t>(
                                                graph_budget / kBlockSize /
                                                num_shards));
                   }(),
                   kBlockSize),
      code_cache_(num_shards,
                  [&] {
                      const uint64_t graph_bytes = graph_file_.size();
                      const uint64_t codes_bytes = codes_file_.size();
                      const uint64_t total = graph_bytes + codes_bytes;
                      const uint64_t graph_budget = total > 0
                          ? cache_size_bytes * graph_bytes / total
                          : cache_size_bytes / 2;
                      const uint64_t code_budget =
                          cache_size_bytes - graph_budget;
                      return std::max(1u, static_cast<uint32_t>(
                                               code_budget / kBlockSize /
                                               num_shards));
                  }(),
                  kBlockSize) {
    const uint32_t total_blocks_per_shard = std::max(
        1u, static_cast<uint32_t>(cache_size_bytes / kBlockSize / num_shards));
    cache_controller_ = std::make_unique<CacheController>(
        graph_cache_, code_cache_, num_shards, cache_size_bytes, kBlockSize);
    cache_controller_->seed_fraction_from_caches();
    (void)total_blocks_per_shard;
}

PagedNodeStore::~PagedNodeStore() = default;

PinResult PagedNodeStore::batched_read(DirectFile& file, uint32_t block_size,
                                         uint64_t block_idx, uint64_t key_base,
                                         uint64_t& counter, BlockCache& cache) {
    // Per-worker L1 + staging — no thread_local machinery (Layer 3).
    TLBlockCache& l1 = l1_;
    L1Counters& c = l1_counters_;

    const uint64_t epoch = cache.shard_epoch(block_idx);
    if (const uint8_t* hit = l1.lookup(key_base, epoch)) {
        ++l1.hits;
        c.local_hits++;
        if (!((c.local_hits + c.local_misses) & kL1FlushMask)) {
            tl_hits_ += c.local_hits;
            tl_misses_ += c.local_misses;
            c.local_hits = c.local_misses = 0;
        }
        return {hit, false};
    }
    ++l1.misses;
    c.local_misses++;
    if (!((c.local_hits + c.local_misses) & kL1FlushMask)) {
        tl_hits_ += c.local_hits;
        tl_misses_ += c.local_misses;
        c.local_hits = c.local_misses = 0;
    }

    // Step 1–2: check the requested block's shard under a READ lock.
    {
        CacheShard& shard = cache.shard(block_idx);
        ScopedReadLock lock(shard.mutex());
        uint8_t* hit = shard.lookup_unlocked(block_idx);
        if (hit) {
            const uint64_t post_epoch = cache.shard_epoch(block_idx);
            l1.insert(key_base, hit, post_epoch);
            return {hit, false};
        }
    }

    // Step 3: clamp the read so we don't run past EOF.
    const uint64_t header = sizeof(SidecarHeader);
    const uint64_t payload = file.size() > header ? file.size() - header : 0;
    const uint64_t total_blocks = payload / block_size;
    const uint32_t n_to_read = static_cast<uint32_t>(
        std::min<uint64_t>(kBlocksPerRead,
                           total_blocks > block_idx ? total_blocks - block_idx
                                                    : 1));
    const uint32_t read_count = std::max(1u, n_to_read);

    // Step 4: one pread of the whole batch into the per-worker staging area.
    const uint64_t off = header + block_idx * block_size;
    file.pread_aligned(staging_.data,
                       static_cast<size_t>(read_count) * block_size, off);

    counter++;

    // Step 5: insert each fetched block into its own shard.
    for (uint32_t i = 0; i < read_count; ++i) {
        const uint64_t key_i = block_idx + i;
        CacheShard& shard = cache.shard(key_i);
        ScopedWriteLock lock(shard.mutex());
        if (shard.lookup_unlocked(key_i) != nullptr) {
            continue;
        }
        shard.insert(key_i, staging_.data + i * block_size, block_size);
    }

    // Step 6: re-acquire the requested block's shard and return its pointer.
    {
        CacheShard& shard = cache.shard(block_idx);
        ScopedWriteLock lock(shard.mutex());
        uint8_t* stored = shard.lookup_unlocked(block_idx);
        if (stored != nullptr) {
            const uint64_t post_epoch = cache.shard_epoch(block_idx);
            l1.insert(key_base, stored, post_epoch);
            return {stored, true};
        }
        file.pread_aligned(staging_.data, block_size, off);
        counter++;
        stored = shard.insert(block_idx, staging_.data, block_size);
        if (stored != nullptr) {
            const uint64_t post_epoch = cache.shard_epoch(block_idx);
            l1.insert(key_base, stored, post_epoch);
        }
        return {stored, true};
    }
}

PinResult PagedNodeStore::get_node_block(uint64_t block_idx) {
    return batched_read(graph_file_, graph_block_size_, block_idx,
                        /*key_base=*/block_idx, graph_reads_, graph_cache_);
}

PinResult PagedNodeStore::get_code_block(uint64_t block_idx) {
    const uint64_t l1_key = block_idx | kCodeL1KeyBit;
    return batched_read(codes_file_, codes_block_size_, block_idx,
                        /*key_base=*/l1_key, code_reads_, code_cache_);
}

PinResult PagedNodeStore::pin_node(uint32_t id) {
    const uint64_t block_idx = id / nodes_per_block_;
    const uint32_t off_in_block = (id % nodes_per_block_) * node_size_;
    PinResult pr = get_node_block(block_idx);
    return {pr.data + off_in_block, pr.from_ssd};
}

PinResult PagedNodeStore::pin_code(uint32_t id) {
    const uint64_t block_idx = id / codes_per_block_;
    const uint32_t off_in_block = (id % codes_per_block_) * code_size_;
    PinResult pr = get_code_block(block_idx);
    return {pr.data + off_in_block, pr.from_ssd};
}

void PagedNodeStore::maybe_rebalance_caches() {
    if (cache_controller_) {
        cache_controller_->maybe_rebalance();
    }
}

double PagedNodeStore::graph_cache_fraction() const {
    const uint32_t g = graph_cache_.shard(0).capacity();
    const uint32_t c = code_cache_.shard(0).capacity();
    const uint32_t total = g + c;
    return total > 0 ? static_cast<double>(g) / static_cast<double>(total) : 0.5;
}

}  // namespace sextant
