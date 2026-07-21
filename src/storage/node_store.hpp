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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

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
    ///
    /// VamanaCore's dist_to lambda uses this to switch between PQ LUT
    /// distance (default) and direct L2sq on the precise vector when the
    /// query also has a precise (FP16) form — the "hybrid FP16+PQ" path.
    virtual const float16_t* precise_vec(uint32_t id) const {
        (void)id; return nullptr;
    }
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

    uint64_t graph_reads() const { return graph_reads_; }
    uint64_t code_reads() const { return code_reads_; }

    /// L1 (per-worker block cache) hit/miss counters. Single-threaded per
    /// PagedNodeStore instance (Layer 3: per-worker ownership). Returns the
    /// flushed total PLUS pending per-call accumulations (≤ kL1FlushMask
    /// samples) so tests get exact counts even after a handful of pins.
    uint64_t tl_hits() const {
        return tl_hits_ + l1_counters_.local_hits;
    }
    uint64_t tl_misses() const {
        return tl_misses_ + l1_counters_.local_misses;
    }

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
    /// ratios. Single-threaded per worker (Layer 3).
    void maybe_rebalance_caches();

    /// Current fraction [0,1] of the total cache budget assigned to graph
    /// blocks. Reads shard(0) capacities (uniform across shards).
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
    // (per-worker) entries — the L2 caches are fully separate and use plain
    // block_idx as keys.
    static constexpr uint64_t kCodeL1KeyBit = 1ULL << 63;

    // Statistical counters (single-threaded per worker; uint64_t is fine).
    uint64_t graph_reads_ = 0;
    uint64_t code_reads_ = 0;
    uint64_t tl_hits_ = 0;
    uint64_t tl_misses_ = 0;

    // Adaptive controller that rebalances graph_cache_ / code_cache_ sizes.
    // Constructed after the caches (lazily initialized in the constructor body
    // since it needs the resolved budget, which depends on file sizes read
    // after the files are open).
    std::unique_ptr<CacheController> cache_controller_;

    // --- per-worker L1 (Layer 3: was thread_local machinery) ----------------
    // Each PagedNodeStore is owned by ONE pool worker, so the L1 (TLBlockCache),
    // the L1 hit/miss counters, and the pread staging buffer are plain
    // per-instance state. No cross-thread coordination, no instance-id keying.
    TLBlockCache l1_;
    struct alignas(64) L1Counters {
        uint64_t local_hits = 0;
        uint64_t local_misses = 0;
    };
    L1Counters l1_counters_;
    static constexpr uint64_t kL1FlushMask = 63;

    // Per-worker aligned staging buffer for batched pread (was thread_local
    // g_staging). 1MB aligned to kDiskAlign for O_DIRECT.
    struct alignas(kDiskAlign) AlignedStaging {
        static constexpr size_t kCapacity =
            kBlockSize * PagedNodeStore::kBlocksPerRead;
        uint8_t data[kCapacity];
    };
    AlignedStaging staging_;

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
                           uint64_t& counter, BlockCache& cache);
};

}  // namespace sextant
