#pragma once

/// @file sharded_lru.hpp
/// Sharded LRU cache for graph node / PQ code blocks.
///
/// N shards (N = hardware_concurrency), each protected by an nsync_mu
/// reader-writer lock. Route: block_idx % N_SHARDS.

#include <sextant/sync.hpp>
#include <sextant/types.hpp>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cstdint>

namespace sextant {

struct DirectFile;

/// A single LRU shard. ShardedLRUCache owns N of these.
///
/// Locking contract (Phase 1): the caller MUST hold the shard's write lock
/// (`ScopedWriteLock lock(shard.mutex())`) around every call to lookup(),
/// insert(), and mark_dirty(). Phase 1 deliberately uses a single write lock
/// for both reads and writes rather than a read-lock lookup + lock-upgrade on
/// miss. The rationale:
///   - Correctness is trivial (no upgrade race, no torn linked-list updates).
///   - Sharding by `block_idx % N_SHARDS` (N = hardware_concurrency) already
///     spreads contention thinly, so the coarser lock is not a bottleneck at
///     Phase 1 scale.
/// A future perf pass can introduce read-lock lookup with upgrade-to-write on
/// miss if profiling shows contention. The external API (mutex() accessor +
/// Scoped locks) won't change.
struct LRUEntry {
    uint64_t block_idx;
    uint8_t* data;
    bool dirty;
    LRUEntry* prev;
    LRUEntry* next;
};

class LRUShard {
public:
    explicit LRUShard(uint32_t capacity_blocks);
    ~LRUShard();

    LRUShard(const LRUShard&) = delete;
    LRUShard& operator=(const LRUShard&) = delete;

    /// Look up a block. Returns pointer to block data if hit, nullptr if miss.
    /// On hit, moves the entry to MRU position.
    uint8_t* lookup(uint64_t block_idx);

    /// Insert a block (data is copied into the shard's own buffer).
    /// Evicts the LRU entry if at capacity. Returns pointer to the stored data.
    /// The caller's `data` buffer is consumed (copied).
    uint8_t* insert(uint64_t block_idx, const uint8_t* data, uint32_t size);

    /// Mark a block as dirty (needs writeback on eviction).
    void mark_dirty(uint64_t block_idx);

    Mutex& mutex() { return mu_; }

private:
    void move_to_front(LRUEntry* e);
    void evict_lru();

    Mutex mu_;
    uint32_t capacity_;
    std::unordered_map<uint64_t, LRUEntry> map_;
    LRUEntry* head_ = nullptr;  // MRU
    LRUEntry* tail_ = nullptr;  // LRU
};

class ShardedLRUCache {
public:
    /// Create a cache with `num_shards` shards, each holding `blocks_per_shard`
    /// blocks of `block_size` bytes.
    ShardedLRUCache(uint32_t num_shards, uint32_t blocks_per_shard,
                    uint32_t block_size);
    ~ShardedLRUCache();

    ShardedLRUCache(const ShardedLRUCache&) = delete;
    ShardedLRUCache& operator=(const ShardedLRUCache&) = delete;

    uint32_t num_shards() const { return shards_.size(); }
    uint32_t block_size() const { return block_size_; }

    LRUShard& shard(uint64_t block_idx) {
        return *shards_[block_idx % shards_.size()];
    }

private:
    std::vector<std::unique_ptr<LRUShard>> shards_;
    uint32_t block_size_;
};

}  // namespace sextant
