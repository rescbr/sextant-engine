#pragma once

/// @file tl_cache.hpp
/// Per-thread L1 block cache (TLBlockCache). One instance per search thread;
/// accessed by exactly one thread, so no locking is required.
///
/// ## Design (Layer 4 perf, 2026-07-21)
///
/// Hash-indexed direct-mapped cache. Capacity is a power of two (default 256).
/// Each block_key hashes to exactly one slot; lookups are O(1) with a single
/// cache-line access (vs the previous linear scan over 32-64 entries).
///
/// Replacement: direct-mapped overwrite (the new entry evicts whatever was in
/// its slot). With ~50-block working sets and 256 slots, collision rate is
/// low (~20% load factor). The L2 still handles misses; L1 is purely a
/// latency optimization to avoid the L2 shard lock on hot blocks.
///
/// Epoch validation: each entry stores the owning shard's epoch at insertion
/// time. On lookup, the caller passes the current shard epoch; if it differs,
/// an eviction occurred in that shard → the cached pointer MAY be stale →
/// treat as a miss. This is fine-grained: an eviction in shard K only
/// invalidates L1 entries whose blocks live in shard K.
///
/// ## Thread-safety
///
/// Single-threaded only. The caller (PagedNodeStore via thread_local or
/// SearchWorkerState) guarantees exclusive access. The shard-epoch reads use
/// relaxed atomics (the epoch is a shared per-shard counter updated by L2
/// eviction on other threads).

#include <sextant/types.hpp>

#include <array>
#include <cstdint>

namespace sextant {

/// Per-thread L1 block cache. One instance per search thread.
struct TLBlockCache {
    /// Number of slots (power of two for cheap modulo). 512 slots × 24B =
    /// 12KB per thread. Sized to hold the full arxiv-nomic 1.34M working set
    /// (~280 blocks touched across 1000 queries) with ~2× headroom — direct-
    /// mapped caches need headroom because hot blocks that hash to the same
    /// slot evict each other. Below the working-set size, collision-induced
    /// evictions dominate and hit rate suffers.
    /// Number of slots (power of two for cheap modulo). 512 slots × 24B =
    /// 12KB per thread — negligible. Empirically validated on arxiv-nomic
    /// 1.34M (2026-07-21): hit rate plateaus at ~86% from 512→1024, so 512
    /// captures the working set. Below 512, collision-induced evictions
    /// dominate (256 → 68%, 64 → 51%).
    static constexpr uint32_t kCapacity = 512;
    static constexpr uint32_t kMask = kCapacity - 1;

    /// Sentinel key marking an unused slot. Real keys are always < this
    /// (block indices, possibly OR'd with the code L1 namespace bit, are well
    /// below 2^63-1).
    static constexpr uint64_t kInvalidKey = UINT64_MAX;

    struct Entry {
        uint64_t block_key = kInvalidKey;
        /// Pointer into L2's memory. Valid only while `epoch` matches the
        /// owning shard's current epoch.
        const uint8_t* data_ptr = nullptr;
        /// Value of the owning shard's epoch at insertion time. If it no
        /// longer matches the current shard epoch, an eviction occurred in
        /// that shard → the pointer may be stale.
        uint64_t epoch = 0;
    };

    std::array<Entry, kCapacity> slots_{};

    /// Multiplicative hash. Distributes sequential block indices across slots
    /// (the 2^64/phi constant is the standard Knuth multiplicative hash).
    static constexpr uint32_t hash(uint64_t key) {
        return static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ULL) >> 48);
    }

    /// Returns pointer to block data if cached AND the epoch is still valid
    /// (no eviction has occurred since insertion). Returns nullptr on a miss
    /// or when the entry is stale (epoch mismatch → pointer may dangle).
    ///
    /// `shard_epoch` is the per-shard epoch for the block's owning shard,
    /// read by the caller at the top of batched_read (passed in to avoid a
    /// redundant load). An eviction in a different shard does NOT change this
    /// value, so L1 entries from other shards stay valid.
    const uint8_t* lookup(uint64_t block_key, uint64_t shard_epoch) const {
        const Entry& e = slots_[hash(block_key) & kMask];
        if (e.block_key == block_key && e.epoch == shard_epoch) {
            return e.data_ptr;
        }
        return nullptr;
    }

    /// Inserts a POINTER to block data (NO COPY). `epoch` should be the
    /// owning shard's epoch captured AFTER the L2 operation that produced
    /// `ptr`, so the entry is valid for at least as long as no further
    /// eviction occurs in that shard. Direct-mapped: overwrites whatever was
    /// in the slot (no eviction list, no replacement policy).
    void insert(uint64_t block_key, const uint8_t* ptr, uint64_t epoch) {
        Entry& e = slots_[hash(block_key) & kMask];
        e.block_key = block_key;
        e.data_ptr = ptr;
        e.epoch = epoch;
    }

    /// Profiling counters (per-thread; aggregated by PagedNodeStore).
    /// Incremented by the caller (PagedNodeStore::batched_read).
    uint64_t hits = 0;
    uint64_t misses = 0;
};

}  // namespace sextant
