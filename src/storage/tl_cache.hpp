#pragma once

/// @file tl_cache.hpp
/// Thread-local L1 block cache (pointer-based with epoch validation).
///
/// Each search thread gets its own private L1 that sits in front of the shared
/// L2 (BlockCache / W-TinyLFU). This eliminates cross-thread eviction
/// contention: a thread's working set (~8 blocks) is never evicted by another
/// thread accessing a different region of the graph.
///
/// ## Design: pointer-based with epoch validation
///
/// Each entry holds a POINTER into L2's memory plus the global epoch value
/// captured at insertion time. No data is copied. This avoids the
/// memcpy-induced cache thrash that crippled the copy-based design (256 KB ×
/// ~40 misses/query = 10 MB of copies thrashing L2/L3 per query).
///
/// The risk of storing raw pointers into L2 is use-after-free: L2 can evict a
/// block (freeing its memory) while L1 still references it. Mitigated by
/// per-shard epoch counters (`BlockCache::shard_epochs_`): every eviction in
/// shard K increments `shard_epochs_[K]`. On L1 lookup, the caller passes the
/// epoch for the block's owning shard; if the stored epoch differs from the
/// current shard epoch, an eviction in that shard happened since insertion →
/// the pointer MAY be stale → treat as a miss (fall through to L2).
///
/// This is fine-grained: an eviction in shard K only invalidates L1 entries
/// whose blocks live in shard K — entries from other shards stay valid. The
/// cost is a single relaxed atomic load on the hot path (~1 ns).
///
/// Memory cost: ~200 bytes per thread (8 entries × ~24 bytes) vs 2 MB for the
/// copy-based design.
///
/// ## Replacement: FIFO
///
/// A round-robin slot cursor overwrites the oldest entry. For 8 slots a linear
/// scan on lookup is faster than a hash table (the entries fit in a handful of
/// cache lines of metadata).
///
/// ## Thread-safety
///
/// TLBlockCache is intended for thread-local storage only. A single instance
/// is accessed by exactly one thread, so no locking is required on lookup or
/// insert. The shard-epoch reads use relaxed atomics (the epoch is a shared
/// per-shard counter updated by L2 eviction on other threads). The
/// hits/misses counters use non-atomic increments; they are aggregated
/// per-store via relaxed atomics by the owning PagedNodeStore.

#include <sextant/types.hpp>

#include <array>
#include <cstdint>

namespace sextant {

/// Thread-local L1 block cache. One instance per search thread.
struct TLBlockCache {
    /// Number of blocks cached per thread.
    static constexpr uint32_t kCapacity = 32;

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
    uint32_t next_slot_ = 0;

    /// Returns pointer to block data if cached AND the epoch is still valid
    /// (no eviction has occurred since insertion). Returns nullptr on a miss
    /// or when the entry is stale (epoch mismatch → pointer may dangle).
    ///
    /// `shard_epoch` is the per-shard epoch for the block's owning shard,
    /// read by the caller at the top of batched_read (passed in to avoid a
    /// redundant load). An eviction in a different shard does NOT change this
    /// value, so L1 entries from other shards stay valid.
    const uint8_t* lookup(uint64_t block_key, uint64_t shard_epoch) const {
        for (uint32_t i = 0; i < kCapacity; ++i) {
            if (slots_[i].block_key == block_key) {
                if (slots_[i].epoch == shard_epoch) {
                    return slots_[i].data_ptr;
                }
                return nullptr;  // stale — an eviction happened since insertion
            }
        }
        return nullptr;  // not in L1
    }

    /// Inserts a POINTER to block data (NO COPY). `epoch` should be the
    /// owning shard's epoch captured AFTER the L2 operation that produced
    /// `ptr`, so the entry is valid for at least as long as no further
    /// eviction occurs in that shard. FIFO replacement: round-robin over
    /// slots 0..kCapacity-1.
    void insert(uint64_t block_key, const uint8_t* ptr, uint64_t epoch) {
        slots_[next_slot_].block_key = block_key;
        slots_[next_slot_].data_ptr = ptr;
        slots_[next_slot_].epoch = epoch;
        next_slot_ = (next_slot_ + 1) % kCapacity;
    }

    /// Profiling counters (per-thread; aggregated by PagedNodeStore).
    /// Incremented by the caller (PagedNodeStore::batched_read).
    uint64_t hits = 0;
    uint64_t misses = 0;
};

}  // namespace sextant
