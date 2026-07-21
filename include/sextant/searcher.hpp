#pragma once

/// @file searcher.hpp
/// Searcher — read-only query serving over an Index.
///
/// Layer 3 design: Searcher owns a `ctpl::thread_pool_tls<SearchWorkerState>`
/// with N workers. Each worker holds the per-query scratch (VamanaTLS —
/// 4MB visited_flags + heaps). All search work runs on pool workers; there
/// is no separate single-thread path. `search()` is a convenience wrapper
/// that pushes one query and waits on its future; `search_batch()` fans a
/// batch. `num_threads=1` is a pool of one worker — same code path.
///
/// Today (Phase AB) the Index's VamanaCore and PagedNodeStore are still
/// shared across workers; later phases (CD) move them per-worker so each
/// search thread has its own cache + fds (the scaling win — kills
/// cross-core atomic contention on BlockCache).
///
/// Lifecycle: constructed over a populated Index (built via Builder, or
/// loaded via Index::read(path)). The Index must outlive the Searcher.

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"
#include "sextant/config.hpp"
#include "sextant/index.hpp"
#include "sextant/types.hpp"

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

namespace sextant {

class NodeStore;
class PagedNodeStore;
class MemGraph;
class VamanaCore;

/// Per-pool-worker scratch. In paged mode each worker owns its own
/// PagedNodeStore (own caches, own fds, own L1, own staging) + VamanaTLS +
/// a search-only VamanaCore + MemGraph wired over the store. In flat-store
/// mode (small RAM-only indices, post-insert search, build tests), workers
/// share the Index's flat_store + core directly — the per-worker owns[]
/// pointers are null and shared_store/shared_core point at the Index.
///
/// Paged-mode construction is DEFERRED to the first search call (so the
/// Searcher ctor doesn't throw on an unopened Index). Until then the
/// paged_* fields hold the params needed to lazily build the store/core.
struct SearchWorkerState {
    VamanaTLS tls;
    // Paged mode (owned per-worker, lazily constructed on first search):
    std::unique_ptr<PagedNodeStore> store;
    std::unique_ptr<MemGraph> memgraph;
    std::unique_ptr<VamanaCore> core;
    // Flat mode (shared with Index):
    NodeStore* shared_store = nullptr;
    VamanaCore* shared_core = nullptr;

    // Deferred-construction params (paged mode only).
    std::string paged_graph_path, paged_codes_path, paged_ball_path;
    uint32_t paged_node_size = 0, paged_code_size = 0;
    uint32_t paged_total_count = 0, paged_dim = 0;
    std::vector<uint32_t> paged_entry_points;
    ResolvedParams paged_params;
    bool paged_have_memgraph = false;
    PqQuantizer* paged_quantizer = nullptr;
    uint32_t paged_num_shards = 1;
    uint64_t paged_cache_bytes = 0;

    /// True when this worker contributes paged-store diagnostics.
    bool is_paged() const { return static_cast<bool>(store); }

    /// Resolve the active VamanaCore for this worker (per-worker in paged
    /// mode, shared with the Index in flat mode).
    VamanaCore* active_core() {
        return shared_core ? shared_core : core.get();
    }
};

class Searcher {
public:
    /// Construct over a populated Index. `num_threads` controls the pool
    /// size (and thus the parallelism of search_batch). 1 = single-threaded.
    explicit Searcher(Index& index, uint32_t num_threads = 1);
    ~Searcher();

    Searcher(const Searcher&) = delete;
    Searcher& operator=(const Searcher&) = delete;
    Searcher(Searcher&&) = delete;
    Searcher& operator=(Searcher&&) = delete;

    /// Search for k nearest neighbors of `query`. Returns candidates sorted
    /// ascending by distance. Synchronous (pushes one task, waits on future).
    std::vector<Candidate> search(const float* query, uint32_t k,
                                   const SearchConfig& config);

    /// Push a single search to the pool and return its future immediately
    /// (no wait). Callers pushing a batch of queries can collect futures up
    /// front, then `.get()` them in order — this lets workers pull the next
    /// query as soon as they finish (no per-batch barrier).
    std::future<std::vector<Candidate>> search_one_async(
        const float* query, uint32_t k, const SearchConfig& config);

    /// Search a batch of `n` queries (each `index.dim` floats, packed
    /// row-major). Fans out across the pool — at most `num_threads` queries
    /// run at once. Returns results in input order.
    std::vector<std::vector<Candidate>> search_batch(
        const float* queries, uint32_t n,
        uint32_t k, const SearchConfig& config);

    // --- Cache management (paged mode only; no-op otherwise) ---

    void set_cache_rebalance_enabled(bool enabled) {
        cache_rebalance_enabled_ = enabled;
    }

    void rebalance_caches();

    // --- Diagnostics ---

    uint64_t cache_graph_reads() const;
    uint64_t cache_code_reads() const;

    struct AdmissionStats {
        uint64_t hits_window = 0;
        uint64_t hits_probation = 0;
        uint64_t hits_protected = 0;
        uint64_t misses = 0;
        uint64_t evictions_admitted = 0;
        uint64_t evictions_rejected = 0;
    };
    AdmissionStats cache_admission_stats() const;

    uint64_t tl_hits() const;
    uint64_t tl_misses() const;

    uint32_t memgraph_cached_count() const;

    uint32_t num_threads() const;

private:
    Index& index_;
    uint32_t num_threads_;
    std::atomic<uint64_t> search_count_{0};
    bool cache_rebalance_enabled_ = true;
    std::atomic<bool> rebalancing_{false};
    static constexpr uint64_t kRebalanceCadenceInitial = 1000;
    static constexpr uint64_t kRebalanceCadenceMax = 16000;
    uint64_t rebalance_cadence_ = kRebalanceCadenceInitial;

    struct PoolImpl;
    std::unique_ptr<PoolImpl> pool_;

    /// Core search body (LUT build + VamanaCore::search). Runs on a pool
    /// worker thread with its own VamanaCore + VamanaTLS. Lazily constructs
    /// the per-worker PagedNodeStore + VamanaCore + MemGraph on first call.
    std::vector<Candidate> search_body_(const float* query, uint32_t k,
                                         const SearchConfig& config,
                                         VamanaCore& core,
                                         VamanaTLS& tls);

    /// Lazily construct this worker's PagedNodeStore + MemGraph + VamanaCore.
    /// No-op in flat mode (shared with Index). Called by search_body_ before
    /// each search; the construction runs once per worker.
    static void ensure_paged_state_(SearchWorkerState& w);

    /// Run a callable on each populated worker slot. Skips slots that
    /// haven't been initialized yet (lazy init on first task).
    template <typename F>
    void each_worker_(F&& fn) const;

    void maybe_rebalance_();
};

}  // namespace sextant
