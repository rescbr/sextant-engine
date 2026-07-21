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
#include "sextant/config.hpp"
#include "sextant/index.hpp"
#include "sextant/types.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace sextant {

/// Per-pool-worker scratch. Phase AB: only VamanaTLS (heaps + visit marks).
/// Phase CD will grow this to own a PagedNodeStore + VamanaCore per worker.
struct SearchWorkerState {
    VamanaTLS tls;
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
    /// worker thread with its own VamanaTLS.
    std::vector<Candidate> search_body_(const float* query, uint32_t k,
                                         const SearchConfig& config,
                                         VamanaTLS& tls);

    void maybe_rebalance_();
};

}  // namespace sextant
