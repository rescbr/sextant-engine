#pragma once

/// @file node_store.hpp
/// NodeStore abstraction: decouples VamanaCore from the flat-RAM buffer layout.
///
/// Backends:
///   - FlatNodeStore: direct pointers into flat RAM buffers (build + small
///     indices at search). Zero-copy, O(1).
///   - PagedNodeStore: on-demand block reads from sidecar files via DirectFile,
///     cached in a BlockCache. This is the SSD-resident search path.
///   - MemGraph: topology-based RAM cache for the entry-point neighborhood.
///     Delegates cold-node misses to a backing PagedNodeStore.
///
/// VamanaCore goes through this interface for ALL node/code access (both build
/// and search). During build, Engine installs a FlatNodeStore over the flat
/// buffers; during search, Engine installs a MemGraph over a PagedNodeStore.

#include "storage/block_cache.hpp"
#include "storage/cache_controller.hpp"
#include "storage/direct_io.hpp"
#include "storage/sidecar_header.hpp"
#include "storage/tl_cache.hpp"
#include "sextant/types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace sextant {

/// Result of a pin operation. Tells the caller where the data came from,
/// which determines whether SSD-amortization techniques (PageSearch) are
/// worthwhile for this node.
struct PinResult {
    const uint8_t* data = nullptr;
    /// True if this pin required I/O (cache miss → SSD read or LRU fetch).
    /// False if served from RAM (flat buffer, MemGraph hit, or LRU hit).
    /// beam_search uses this to decide whether PageSearch is worth the
    /// CPU cost: only amortize when we actually paid for I/O.
    bool from_ssd = false;
};

/// Abstract interface for accessing graph nodes and PQ codes.
class NodeStore {
public:
    virtual ~NodeStore() = default;

    /// Read a graph node. Returns pointer + whether it caused I/O.
    virtual PinResult pin_node(uint32_t internal_id) = 0;

    /// Read a PQ code. Returns pointer + whether it caused I/O.
    virtual PinResult pin_code(uint32_t internal_id) = 0;

    /// Batch-pin PQ codes for multiple nodes. Fills `out_ptrs[i]` with the
    /// code pointer for `ids[i]`. Default implementation calls pin_code in
    /// a loop; PagedNodeStore overrides to batch by shard (one lock
    /// acquisition per shard instead of per-id).
    ///
    /// Used by beam_search_into's batched distance evaluation (Layer 4 perf):
    /// collecting all unvisited neighbor codes under minimal locking,
    /// enabling lut_distance_batch4 (SVE2 4-way SIMD gather) over the
    /// returned pointers. Callers must size `out_ptrs` to at least `n`.
    virtual void pin_codes(const uint32_t* ids, size_t n,
                           const uint8_t** out_ptrs);

    /// True if this store can serve SSD reads (PagedNodeStore or MemGraph
    /// with a paged backing store). Used to gate DynamicWidth.
    virtual bool is_paged() const = 0;

    /// Optional: return a higher-precision vector for `id` if the store has
    /// one (e.g. MemGraph caches FP16 "ball" vectors for the entry-point
    /// neighborhood). Default: no precise vector (PQ-only path).
    ///
    /// Contract: this is a *policy hook* — the store advertises "I can give
    /// you a more precise vector for this node" and the algorithm decides
    /// whether to use it. The actual storage format (FP16, FP32, ...) is
    /// private to the store; the algorithm only needs the typed pointer and
    /// the dim (from VamanaParams).
    virtual const float16_t* precise_vec(uint32_t id) const { (void)id; return nullptr; }
};

/// Flat buffer backing. Used during build and for small indices at search.
class FlatNodeStore : public NodeStore {
public:
    FlatNodeStore(uint8_t* nodes, const uint8_t* codes,
                  uint32_t node_size, uint32_t code_size);

    PinResult pin_node(uint32_t id) override;
    PinResult pin_code(uint32_t id) override;
    bool is_paged() const override { return false; }

private:
    uint8_t* nodes_;
    const uint8_t* codes_;
    uint32_t node_size_;
    uint32_t code_size_;
};

/// LRU-paged backing — the SSD-resident search path.
class PagedNodeStore : public NodeStore {
public:
    PagedNodeStore(const std::string& graph_path,
                   const std::string& codes_path,
                   uint32_t node_size, uint32_t code_size,
                   uint32_t num_shards,
                   uint64_t cache_size_bytes);
    ~PagedNodeStore();

    PinResult pin_node(uint32_t internal_id) override;
    PinResult pin_code(uint32_t internal_id) override;
    bool is_paged() const override { return true; }

    /// Number of blocks to read per cache miss (requested + neighbors).
    /// Exploits graph locality after PageShuffle — adjacent disk blocks
    /// contain nearby graph nodes. 4 × 256KB = 1MB per miss.
    static constexpr uint32_t kBlocksPerRead = 4;

    uint64_t graph_reads() const {
        return graph_reads_.load(std::memory_order_relaxed);
    }
    uint64_t code_reads() const {
        return code_reads_.load(std::memory_order_relaxed);
    }

    /// Drain this thread's pending L1 hit/miss samples into the shared
    /// atomics, then return the current totals. Only needed by callers that
    /// read stats immediately after a small number of operations (e.g. tests);
    /// in normal operation the per-call flush every 64 samples is sufficient.
    /// Note: only drains the CALLING thread's pending samples — other threads
    /// may still have unflushed counts. Single-threaded tests are exact;
    /// multi-threaded stats are already approximate (see comment on
    /// tl_l1_counters_).
    uint64_t tl_hits() const {
        flush_l1_counters();
        return tl_hits_.load(std::memory_order_relaxed);
    }
    uint64_t tl_misses() const {
        flush_l1_counters();
        return tl_misses_.load(std::memory_order_relaxed);
    }

private:
    void flush_l1_counters() const {
        auto it = tl_l1_counters_.find(this);
        if (it == tl_l1_counters_.end()) return;
        L1Counters& c = it->second;
        if (c.local_hits || c.local_misses) {
            tl_hits_.fetch_add(c.local_hits, std::memory_order_relaxed);
            tl_misses_.fetch_add(c.local_misses, std::memory_order_relaxed);
            c.local_hits = c.local_misses = 0;
        }
    }

public:

    /// W-TinyLFU cache profiling counters (window/probation/protected hits,
    /// misses, admission decisions). Combined across the graph and code caches.
    struct CombinedCacheStats {
        CacheStats graph;
        CacheStats code;
    };
    CombinedCacheStats cache_stats() const {
        return {graph_cache_.stats(), code_cache_.stats()};
    }

    /// Periodically rebalance the graph/code cache split based on hit/miss
    /// ratios. Safe to call from any thread.
    void maybe_rebalance_caches();

    /// Current fraction [0,1] of the total cache budget assigned to graph
    /// blocks. Diagnostic read for the rebalance controller's cadence logic.
    /// Reads shard(0) capacities (uniform across shards); atomic loads, no lock.
    double graph_cache_fraction() const;

private:
    DirectFile graph_file_;
    DirectFile codes_file_;
    uint32_t node_size_;
    uint32_t code_size_;
    uint32_t nodes_per_block_;
    uint32_t codes_per_block_;
    uint32_t graph_block_size_;
    uint32_t codes_block_size_;
    // Separate L2 caches for graph (.graph) and code (.codes) blocks so they
    // don't evict each other. An adaptive CacheController may rebalance their
    // relative sizes based on hit/miss ratios.
    BlockCache graph_cache_;
    BlockCache code_cache_;

    // L1 key namespace separator: graph uses plain block_idx as the L1 key;
    // code uses block_idx | kCodeL1KeyBit. This ONLY disambiguates the L1
    // (thread-local) entries — the L2 caches are fully separate and use plain
    // block_idx as keys.
    static constexpr uint64_t kCodeL1KeyBit = 1ULL << 63;

    // Relaxed atomics: these are statistical counters read only after a
    // benchmark run completes. No ordering needed, just lock-free increments
    // on the cache-miss hot path.
    mutable std::atomic<uint64_t> graph_reads_{0};
    mutable std::atomic<uint64_t> code_reads_{0};

    // Adaptive controller that rebalances graph_cache_ / code_cache_ sizes.
    // Constructed after the caches (lazily initialized in the constructor body
    // since it needs the resolved budget, which depends on file sizes read
    // after the files are open).
    std::unique_ptr<CacheController> cache_controller_;

    // --- Thread-local L1 (per-search-thread block cache) -------------------
    // Each thread that calls batched_read() gets its own TLBlockCache for THIS
    // store instance. Per-instance isolation is essential: two different
    // PagedNodeStore objects backing different index files must never share L1
    // entries (a block key like 0 means different data in each file).
    //
    // Implementation: a thread_local unordered_map keyed by a per-instance
    // unique ID (instance_id_), NOT by `this`. Using a monotonic ID avoids the
    // address-reuse hazard: if a destroyed store's memory is recycled for a
    // new store, the new store has a different ID and won't match stale
    // entries left on threads that didn't run the destructor.
    static std::atomic<uint64_t> next_instance_id_;
    uint64_t instance_id_;
    // Per-thread L1 hit/miss accumulators. The hot path of the search does an
    // L1 lookup on every cache access (~4.5M per 1000-query bench); updating
    // shared atomics per call caused severe cache-line bouncing across cores
    // (~30% of cycles at 8 threads, second only to BlockCache::record_access).
    // We instead accumulate per-thread and flush to the shared atomics every
    // `kL1FlushMask + 1` calls. Stats are read only at end-of-bench, so the
    // ≤63-sample loss on thread exit is invisible.
    struct alignas(64) L1Counters {
        uint64_t local_hits = 0;
        uint64_t local_misses = 0;
    };
    static thread_local std::unordered_map<const PagedNodeStore*, L1Counters>
        tl_l1_counters_;
    static constexpr uint64_t kL1FlushMask = 63;
    mutable std::atomic<uint64_t> tl_hits_{0};
    mutable std::atomic<uint64_t> tl_misses_{0};

    // --- per-thread fast-path cache (avoids 2 hash lookups per L1 access) ----
    // The hot path of batched_read() used to call tl_cache() (one hash lookup
    // in g_tl_caches) AND access tl_l1_counters_[this] (another hash lookup)
    // on every L1 access. Together these added ~9% of 8t cycles and caused
    // the 1t regression seen after step 1 (-2% to -4% depending on config).
    // We cache both pointers in a single thread_local struct, invalidated
    // only when the calling thread switches stores. The slow path runs once
    // per thread per store.
    //
    // Lifetime coupling: the cached pointers are valid only while the owning
    // PagedNodeStore lives. In practice stores outlive all search threads
    // (engine-lifetime objects); if a store is destroyed while a thread still
    // holds a stale pointer, the next identity check (`fp.owner != this`)
    // catches the switch and re-hashes. A subtle aliasing risk remains if a
    // new store reuses the address of a destroyed one AND the same thread
    // accessed the old one — then owner==this matches but the pointers are
    // stale. Mitigated by instance_id_ being monotonic, but the fast path
    // doesn't check instance_id_. Acceptable for engine-lifetime coupling.
    struct TLFastPath {
        const PagedNodeStore* owner = nullptr;
        TLBlockCache* l1 = nullptr;
        L1Counters* counters = nullptr;
    };
    static thread_local TLFastPath tl_fast_;

    /// Returns this thread's TLBlockCache for this store (lazily allocated).
    TLBlockCache& tl_cache() const;

    /// Batch-pin PQ codes. Overrides the default (per-id pin_code loop) with
    /// a shard-batched implementation: L1 hits are lock-free, L1 misses are
    /// grouped by their owning shard so each shard's read lock is acquired
    /// at most once per batch. Used by beam_search_into's batched distance
    /// evaluation (Layer 4 perf).
    void pin_codes(const uint32_t* ids, size_t n,
                   const uint8_t** out_ptrs) override;

    PinResult get_node_block(uint64_t block_idx);
    PinResult get_code_block(uint64_t block_idx);

    /// Shared batched-read helper. On a cache miss for `block_idx`, reads up
    /// to kBlocksPerRead contiguous blocks from `file` in a single pread and
    /// inserts each into its own shard. Returns the pointer to the requested
    /// block (always a miss → from_ssd=true). `key_base` is the L1 key
    /// (block_idx for graph, block_idx | kCodeL1KeyBit for code — disambiguates
    /// the shared L1). `cache` is the L2 cache to use (graph_cache_ or
    /// code_cache_).
    PinResult batched_read(DirectFile& file, uint32_t block_size,
                           uint64_t block_idx, uint64_t key_base,
                           std::atomic<uint64_t>& counter,
                           BlockCache& cache);
};

}  // namespace sextant
