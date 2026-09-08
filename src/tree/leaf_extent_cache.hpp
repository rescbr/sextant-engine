#pragma once

/// @file leaf_extent_cache.hpp
/// Engine-owned leaf-extent cache for the IVF tree search path.
///
/// Why: tree search reads leaf extents through a read-only mmap, so warm
/// behavior free-rides the OS page cache — not a reportable warm layer under
/// docs/BENCHMARK_RULES.md (the engine's own explicitly-budgeted cache is the
/// only legitimate data cache). This cache gives the tree path that layer:
/// keyed by leaf start page, value = the whole contiguous extent (variable
/// size — the graph path's fixed-block BlockCache does not fit), filled by
/// pread on miss.
///
/// Policy: W-TinyLFU (window + SLRU probation/protected with CountMinSketch
/// admission), reusing src/storage/frequency_sketch.hpp. The window is a
/// fixed 1% of capacity (Caffeine's default; the BlockCache hill-climber was
/// deliberately NOT ported — v1 keeps the machinery small, and the adaptive
/// window can be layered on later if hit-rate data justifies it). Admission
/// matters here because a cold near-scan query stream (probe_fraction 0.5)
/// would otherwise flush hot leaves out of a plain LRU.
///
/// Budgeting is in BYTES, not entry counts — leaf extents vary in size
/// (hundreds of 4KB pages each).
///
/// Concurrency: per-entry refcount + deferred free.
///   - pin() returns a buffer pointer with the entry's refcount incremented
///     (under the shard lock, so the entry cannot be evicted between lookup
///     and increment). unpin() decrements; the LAST unpin of an evicted entry
///     frees it. Eviction never touches a pinned entry's data; a scan holds
///     its pin for the whole scan_leaf call with zero locks held.
///   - Misses pread OUTSIDE the shard lock (misses are serialized per shard
///     otherwise); insertion re-takes the lock and resolves the (rare)
///     duplicate-insert race in favor of the incumbent.
///   - An entry rejected by TinyLFU admission is still returned to the caller
///     as a transient (uncached) buffer — pin() must return data regardless;
///     it is freed on unpin.
///
/// Invalidation: invalidate_all() drops every entry (used by
/// IVFTreeIndex::remap_() — mutations may relocate leaves). Entries pinned
/// at invalidation time keep their readers' view consistent and are freed on
/// last unpin.

#include <sextant/sync.hpp>
#include "tree/page_file.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sextant {
class FrequencySketch;
}

namespace sextant::tree {

class LeafExtentCache {
public:
    /// Aggregate counters (delta between reads for window semantics).
    struct Stats {
        uint64_t hits = 0;          ///< pin() served from cache
        uint64_t misses = 0;        ///< pin() required a fill (pread)
        uint64_t bytes_filled = 0;  ///< disk bytes read into the cache
        uint64_t evictions = 0;     ///< entries evicted to stay in budget
        uint64_t rejections = 0;    ///< fills refused entry (TinyLFU / size)
    };

    /// Opaque pin handle. Value-initialized = "no pin" (cache off or
    /// transient). Must be released with unpin() exactly once per pin.
    struct Handle {
        void* entry = nullptr;
    };

    /// `capacity_bytes` is the total DRAM budget for extent copies across all
    /// shards. `fd` is the (open) index file to pread misses from.
    LeafExtentCache(uint64_t capacity_bytes, uint32_t num_shards, int fd);
    ~LeafExtentCache();

    LeafExtentCache(const LeafExtentCache&) = delete;
    LeafExtentCache& operator=(const LeafExtentCache&) = delete;

    /// Hint the sketch / admission machinery about the working-set size
    /// (number of distinct leaves). Call once after open, when n_leaves is
    /// known. Optional — the sketch self-configures to a small default.
    void set_expected_entries(uint32_t n);

    /// True if `page` is currently cached (no refcount taken). Used by the
    /// search path to decide whether fadvise(WILLNEED) is worth issuing.
    bool contains(PageId page);

    /// Pin the extent at `page` (`pages` consecutive 4KB pages). Returns a
    /// pointer to the extent copy and fills `h`. On miss the extent is pread
    /// from fd_ (bytes = pages * kPageSize, plus the TreeLeafHeader etc. —
    /// the copy is byte-identical to the on-disk extent). Optional
    /// out-params report the outcome for per-window search accounting
    /// (SearchStats::on_cache_op).
    const uint8_t* pin(PageId page, uint32_t pages, Handle& h,
                       bool* was_hit = nullptr,
                       uint64_t* filled_bytes = nullptr);

    /// Release a pin. Last unpin of an evicted/transient entry frees it.
    void unpin(Handle& h);

    /// Drop every cached entry (post-mutation). Pinned entries stay alive
    /// until their last unpin; new pins of the same page miss + refill.
    void invalidate_all();

    Stats stats() const;
    uint64_t capacity_bytes() const { return capacity_bytes_; }
    uint32_t num_shards() const { return static_cast<uint32_t>(shards_.size()); }

private:
    struct Entry;   // defined in the .cpp (references Shard*)
    struct List;    // defined in the .cpp
    struct Shard;   // defined in the .cpp (contains List + Entry*)

    static uint64_t entry_size(uint32_t pages) {
        return static_cast<uint64_t>(pages) * kPageSize;
    }
    Shard& shard_for(PageId page);
    Entry* new_entry(Shard& s, PageId page, uint32_t pages);  // buf + pread
    static void maybe_delete(Entry* e);
    static void maybe_delete_locked(Entry* e);  // caller holds owner mu
    static void drop_entry(Entry* e);           // retire + maybe deferred free

    uint64_t capacity_bytes_;
    int fd_;
    std::unique_ptr<FrequencySketch[]> sketches_;  // one per shard
    std::vector<std::unique_ptr<Shard>> shards_;

    // Cache-wide counters (relaxed; aggregated from pin/unpin/invalidate).
    mutable std::atomic<uint64_t> hits_{0};
    mutable std::atomic<uint64_t> misses_{0};
    mutable std::atomic<uint64_t> bytes_filled_{0};
    mutable std::atomic<uint64_t> evictions_{0};
    mutable std::atomic<uint64_t> rejections_{0};
};

}  // namespace sextant::tree
