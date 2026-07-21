#pragma once

/// @file block_cache.hpp
/// Sharded block-level cache with W-TinyLFU eviction.
///
/// N shards (N = hardware_concurrency), each protected by an nsync_mu
/// write lock. Route: block_idx % N_SHARDS.
///
/// Eviction policy: W-TinyLFU (Window TinyLFU) with adaptive window sizing.
/// - Each shard maintains three LRU lists: window, probation, protected.
/// - New blocks enter the window (plain LRU, no admission filter).
/// - On window overflow, the TinyLFU admission policy compares the
///   candidate's historic frequency (via CountMinSketch) against the
///   probation victim's frequency, admitting the more frequent one.
/// - Second access promotes from probation to protected.
/// - The window/main split is adapted by a hill-climbing optimizer on
///   BlockCache that samples the aggregate hit rate.
///
/// This makes the cache scan-resistant: speculative blocks from batched
/// pre-read (which inserts 4 blocks per miss) pass through the window
/// without polluting the protected working set.

#include <sextant/sync.hpp>
#include <sextant/types.hpp>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cstdint>

#include "frequency_sketch.hpp"
#include "block_buffer_pool.hpp"

namespace sextant {

struct DirectFile;
class BlockCache;

/// Which of the three W-TinyLFU lists an entry currently belongs to.
enum class Status : uint8_t { WINDOW, PROBATION, PROTECTED };

/// A single cache entry. Stored by value in the shard's map; the linked-list
/// pointers stitch it into one of the three sentinel-headed lists.
struct LRUEntry {
    uint64_t block_idx = 0;
    uint8_t* data = nullptr;
    bool dirty = false;
    Status status = Status::WINDOW;
    LRUEntry* prev = nullptr;
    LRUEntry* next = nullptr;
};

/// Per-list profiling counters (aggregated by BlockCache::stats()).
struct CacheStats {
    uint64_t hits_window = 0;
    uint64_t hits_probation = 0;
    uint64_t hits_protected = 0;
    uint64_t misses = 0;
    uint64_t evictions_window = 0;
    uint64_t evictions_admitted = 0;
    uint64_t evictions_rejected = 0;
};

/// A sentinel-headed doubly-linked list for one W-TinyLFU segment.
/// `head_` is the MRU sentinel; `head_->prev` is the LRU sentinel. An empty
/// list has both sentinels pointing at each other.
struct LRUList {
    LRUEntry head;  // sentinel; head.next = MRU entry, head.prev = LRU entry

    LRUList() {
        head.next = &head;
        head.prev = &head;
    }

    bool empty() const { return head.next == &head; }

    /// MRU entry (nullptr if empty).
    LRUEntry* mru() const { return empty() ? nullptr : head.next; }

    /// LRU entry (nullptr if empty).
    LRUEntry* lru() const { return empty() ? nullptr : head.prev; }

    /// Unlink `e` from this list (does not touch `e`'s own prev/next).
    static void unlink(LRUEntry* e) {
        e->prev->next = e->next;
        e->next->prev = e->prev;
    }

    /// Insert `e` at the MRU end (right after the head sentinel).
    void push_mru(LRUEntry* e) {
        e->prev = &head;
        e->next = head.next;
        head.next->prev = e;
        head.next = e;
    }

    /// Move `e` to MRU of this list (unlink + push_mru).
    void move_to_mru(LRUEntry* e) {
        unlink(e);
        push_mru(e);
    }
};

/// A single W-TinyLFU shard. BlockCache owns N of these.
///
/// Locking contract:
///   - lookup() takes a READ lock (via ScopedReadLock). It no longer mutates
///     LRU lists (the W-TinyLFU recency bump on hit was dropped — see
///     CacheShard::lookup comment). The frequency sketch and profiling
///     counters are atomic, so concurrent readers proceed in parallel.
///   - insert() and mark_dirty() take a WRITE lock (via ScopedWriteLock).
///     They mutate the LRU lists / entry fields.
///   - maybe_adapt() (which reconciles the shard's window/protected sizes
///     with the cache's atomic limits) runs only under the write lock, called
///     from insert().
///
/// The window / main split is **adaptive**, but the hill-climbing optimizer
/// lives on the owning BlockCache (it aggregates samples across all
/// shards for a less noisy signal). The cache periodically computes a new
/// window ratio and writes it to each shard's atomic `max_window_` /
/// `max_protected_`. Each shard lazily adapts to the new limits on its next
/// lookup()/insert() via `maybe_adapt()` (called under its own write lock).
class CacheShard {
public:
    /// `capacity_blocks` is the per-shard block capacity. The initial window
    /// is 1% of capacity (matching Caffeine); the owning cache's hill-climber
    /// adjusts it at runtime by writing the atomic limits.
    explicit CacheShard(uint32_t capacity_blocks, uint32_t block_size,
                        BlockCache* cache = nullptr, uint32_t shard_index = 0);
    ~CacheShard();

    CacheShard(const CacheShard&) = delete;
    CacheShard& operator=(const CacheShard&) = delete;

    /// Per-shard buffer pool (diagnostic / test hook).
    BlockBufferPool& pool() { return pool_; }

    /// Look up a block. Returns pointer to block data if hit, nullptr if miss.
    /// Updates the frequency sketch on every call. Read-locked internally —
    /// safe to call concurrently from any thread.
    uint8_t* lookup(uint64_t block_idx);

    /// Same as lookup() but assumes the caller already holds the shard's lock
    /// (either read or write). Used by tests that batch insert+lookup under a
    /// single write-lock scope to model atomic operations. Production code
    /// should call lookup() instead.
    uint8_t* lookup_unlocked(uint64_t block_idx);

    /// Insert a block (data is copied into the shard's own buffer). New blocks
    /// enter the window; overflow is subject to TinyLFU admission into main.
    /// If the block already exists, its data is updated and it is treated as a
    /// hit. Returns pointer to the stored data (nullptr if capacity_ == 0).
    uint8_t* insert(uint64_t block_idx, const uint8_t* data, uint32_t size);

    /// Mark a block as dirty (needs writeback on eviction).
    void mark_dirty(uint64_t block_idx);

    Mutex& mutex() { return mu_; }

    // --- adaptive-window introspection (atomic reads; lock optional) -------
    uint32_t max_window() const { return max_window_.load(std::memory_order_relaxed); }
    uint32_t max_protected() const { return max_protected_.load(std::memory_order_relaxed); }
    uint32_t capacity() const { return capacity_.load(std::memory_order_relaxed); }

    // --- profiling counters (read under the caller's lock) -------------------
    uint64_t hits_window() const { return hits_window_.load(std::memory_order_relaxed); }
    uint64_t hits_probation() const { return hits_probation_.load(std::memory_order_relaxed); }
    uint64_t hits_protected() const { return hits_protected_.load(std::memory_order_relaxed); }
    uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
    uint64_t evictions_window() const { return evictions_window_.load(std::memory_order_relaxed); }
    uint64_t evictions_admitted() const { return evictions_admitted_.load(std::memory_order_relaxed); }
    uint64_t evictions_rejected() const { return evictions_rejected_.load(std::memory_order_relaxed); }

private:
    friend class BlockCache;

    void on_window_hit(LRUEntry* e);
    void on_probation_hit(LRUEntry* e);
    void on_protected_hit(LRUEntry* e);
    void admit_one_from_window();
    void free_entry(LRUEntry* e);

    /// Lazily reconcile this shard's window/protected sizes with the current
    /// atomic limits (which the cache's hill-climber may have changed since
    /// the last operation). Called under mu_ from lookup()/insert().
    void maybe_adapt();

    /// Resize this shard to `new_capacity` blocks. The caller MUST hold mu_.
    /// Evicts excess entries (window LRU first, then probation, then protected)
    /// until map_.size() <= capacity_, recomputes the window/protected limits,
    /// and calls maybe_adapt() to reconcile. If new_capacity == 0, evicts ALL
    /// entries.
    void resize(uint32_t new_capacity);

    // Fast xorshift RNG for anti-starvation admission on frequency ties.
    uint32_t hc_rng_state_ = 0x12345678u;
    uint32_t hc_rng() {
        uint32_t x = hc_rng_state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return hc_rng_state_ = x;
    }

    /// Back-pointer to the owning cache (set at construction). Used to feed
    /// hit/miss samples into the cache-level hill-climber, and to bump this
    /// shard's epoch on eviction. Nullptr in tests that construct a bare shard.
    BlockCache* cache_ = nullptr;

    /// Index of this shard within the owning BlockCache (0..N-1). Used to
    /// bump the correct per-shard epoch counter on eviction.
    uint32_t shard_index_ = 0;

    /// Per-shard buffer pool. Declared BEFORE map_ (and every other member
    /// that holds block buffers) because ~CacheShard() iterates map_ and calls
    /// pool_.release() on each entry's data — members are destroyed in reverse
    /// declaration order, so pool_ must outlive map_.
    BlockBufferPool pool_;

    Mutex mu_;
    // Atomic so PagedNodeStore::graph_cache_fraction() can read it without
    // holding the shard lock (diagnostic read from the rebalance cadence
    // logic). All writes happen under mu_ (resize/constructor), so relaxed
    // ordering is sufficient.
    std::atomic<uint32_t> capacity_;
    // Atomic so the cache's hill-climber can update them without holding this
    // shard's lock. maybe_adapt() (under mu_) moves entries to respect them.
    std::atomic<uint32_t> max_window_;
    std::atomic<uint32_t> max_protected_;

    FrequencySketch sketch_;
    std::unordered_map<uint64_t, LRUEntry> map_;

    LRUList window_;
    LRUList probation_;
    LRUList protected_;

    // Live entry counts per segment (kept in sync with the lists).
    uint32_t size_window_ = 0;
    uint32_t size_protected_ = 0;
    // size_probation_ = map_.size() - size_window_ - size_protected_

    // --- profiling counters --------------------------------------------------
    std::atomic<uint64_t> hits_window_{0};
    std::atomic<uint64_t> hits_probation_{0};
    std::atomic<uint64_t> hits_protected_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_window_{0};
    std::atomic<uint64_t> evictions_admitted_{0};
    std::atomic<uint64_t> evictions_rejected_{0};
};

class BlockCache {
public:
    /// Create a cache with `num_shards` shards, each holding `blocks_per_shard`
    /// blocks of `block_size` bytes.
    BlockCache(uint32_t num_shards, uint32_t blocks_per_shard,
               uint32_t block_size);
    ~BlockCache();

    BlockCache(const BlockCache&) = delete;
    BlockCache& operator=(const BlockCache&) = delete;

    uint32_t num_shards() const { return shards_.size(); }
    uint32_t block_size() const { return block_size_; }

    /// Read the epoch for the shard owning `block_key`. Used by the L1 to
    /// validate pointers: an eviction in shard K only bumps `shard_epochs_[K]`,
    /// so L1 entries from other shards remain valid.
    uint64_t shard_epoch(uint64_t block_key) const {
        return shard_epochs_[block_key % shards_.size()].load(
            std::memory_order_relaxed);
    }


    CacheShard& shard(uint64_t block_idx) {
        return *shards_[block_idx % shards_.size()];
    }
    const CacheShard& shard(uint64_t block_idx) const {
        return *shards_[block_idx % shards_.size()];
    }

    /// Aggregated profiling counters across all shards.
    CacheStats stats() const;

    /// Resize the cache: sets a new per-shard block capacity and evicts excess
    /// entries from each shard (under each shard's write lock). Also resets the
    /// cache-level hill-climber fields to match the new capacity.
    void resize(uint32_t new_blocks_per_shard);

    // --- adaptive-window introspection (cache-level hill-climber) ------------
    /// Current window capacity applied uniformly across all shards
    /// (read from shard 0; all shards share the same ratio).
    uint32_t max_window() const { return shards_[0]->max_window(); }
    uint32_t max_protected() const { return shards_[0]->max_protected(); }

    /// Feed a hit/miss sample into the cache-level hill-climber. Called by each
    /// shard's lookup()/insert() via its back-pointer.
    ///
    /// Performance: this is on the hottest path of the search (every L2 cache
    /// lookup). Doing a shared-atomic `fetch_add` per call caused severe cache-
    /// line bouncing across cores — at 8 threads, the atomic instruction alone
    /// consumed ~30% of CPU cycles. We instead accumulate hits/misses in a
    /// thread-local counter and flush to the shared atomics every `kFlushMask+1`
    /// calls (currently 64), reducing cross-core traffic by ~64×. The
    /// hill-climber's sample window is in the millions, so this coarsening is
    /// well within its tolerance.
    void record_access(bool hit);

    /// Flush this cache's pending samples to the shared atomics. With
    /// per-worker cache ownership (Layer 3) the flush is trivial — the
    /// counters live on the cache itself — but kept for explicit teardown
    /// points (tests).
    void flush_thread_local();

    /// Mask used to determine when to flush: flush when
    /// `(local_hits + local_misses) & kFlushMask == 0`. Power of two minus 1.
    static constexpr uint64_t kFlushMask = 63;

private:
    friend class CacheShard;

    /// Compute a new window ratio from the accumulated sample and write it to
    /// every shard's atomic limits (no shard locks acquired).
    void climb();

    /// Threshold check + single-writer guard around climb().
    void maybe_climb();

    uint32_t block_size_;

    /// Per-shard epoch counters. Incremented when a block is evicted from
    /// that specific shard. The L1 checks only the shard that owns the block
    /// being looked up — an eviction in shard 3 doesn't invalidate L1 entries
    /// from shard 7. Declared BEFORE shards_ so epochs outlive the shards
    /// during destruction. (unique_ptr<atomic[]> because std::vector<atomic>
    /// is not MoveInsertable.)
    std::unique_ptr<std::atomic<uint64_t>[]> shard_epochs_;

    std::vector<std::unique_ptr<CacheShard>> shards_;

    // --- hill-climbing state (shared across all shards) ---------------------
    // Per-shard capacity (blocks_per_shard). Used for step magnitude and
    // clamping the per-shard window limits.
    uint32_t hc_per_shard_capacity_ = 0;
    // Total capacity across all shards (per_shard × num_shards). Used for the
    // sample window size so the signal sees the full access stream.
    uint64_t hc_capacity_ = 0;
    // Sample window = 10 × hc_capacity_ (NOT per-shard), so the signal sees
    // the full access stream across all shards.
    uint64_t hc_sample_size_ = 0;
    // Relaxed-atomic sample counters (statistical; contention-tolerant).
    std::atomic<uint64_t> hc_hits_in_sample_{0};
    std::atomic<uint64_t> hc_misses_in_sample_{0};
    // CAS guard: ensures only one thread executes climb() at a time.
    std::atomic<bool> hc_climbing_{false};
    double hc_prev_hit_rate_ = 0.0;
    double hc_step_size_ = 0.0;

    // --- per-worker sample accumulation (Layer 3: was thread_local map) -----
    // Per-worker accumulation buffer. `record_access` bumps these (no atomics
    // — single-threaded per worker) and flushes to the shared
    // `hc_hits_in_sample_` / `hc_misses_in_sample_` every 64 calls.
    //
    // alignas(64): padding prevents false-sharing between distinct worker
    // caches when several BlockCaches happen to sit in adjacent heap slots.
    // (Each is single-threaded, but cheap insurance against aliasing.)
    struct alignas(64) HotCounters {
        uint64_t local_hits = 0;
        uint64_t local_misses = 0;
    };
    HotCounters hot_counters_;
};

}  // namespace sextant
