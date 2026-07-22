#pragma once

/// @file searcher.hpp
/// Searcher — read-only query serving over an Index.
///
/// Architecture (Layer 3+4 revised 2026-07-21 after the per-worker cache
/// regression): the Searcher owns a `ctpl::thread_pool_tls<SearchWorkerState>`
/// with N workers. Each worker holds ONLY scratch (VamanaTLS — visited flags,
/// heaps). The cache-bearing state (PagedNodeStore, BlockCache, MemGraph,
/// VamanaCore) is SHARED — owned by the Index, referenced by workers via
/// non-owning pointers.
///
/// Why shared caches: on arxiv-nomic 1.34M the per-query working set is
/// ~70MB, small enough that ONE shared cache (with relaxed-atomic sample
/// accumulation, flushed every 64 calls) outperforms N independent caches
/// by 10× at 8 threads. The cross-core atomic traffic from record_access
/// is negligible compared to the cache-hit benefit of cross-worker reuse.
///
/// The L1 (TLBlockCache) and per-thread staging remain thread_local inside
/// PagedNodeStore — keyed by instance pointer, lifetime-coupled to the
/// store. This is sound because the store outlives all search threads.
///
/// `search()` is a convenience wrapper that pushes one query and waits on
/// its future; `search_batch()` fans a batch. `num_threads=1` is a pool of
/// one worker — same code path.

#include "algo/vamana_core.hpp"
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
class VamanaCore;

/// Per-pool-worker scratch. Owns ONLY the per-query mutable state (visit
/// marks, heaps). All cache-bearing state is shared via non-owning pointers
/// to objects owned by the Index.
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

    /// Precomputed per-query PQ state, shared across searches that use the
    /// same trained quantizer (e.g. IVF shard searches). Building it once and
    /// passing it to `search_with_lut` avoids redundant `preprocess_query`
    /// (the PQ LUT) and query→FP16 conversion per shard — ~8% of IVF query
    /// time at n_probe=9 on arxiv100k. Owned by the caller; must outlive the
    /// search call.
    struct QueryLUT {
        std::vector<float> lut;          // lut_size() floats
        std::vector<float16_t> query_fp16;  // dim float16_t
    };

    /// Build the QueryLUT for a query (PQ LUT + FP16 cast). The LUT is
    /// quantizer-specific; reusing it across Searchers that share the same
    /// trained quantizer (e.g. all shards of an IVF index) is safe.
    QueryLUT build_query_lut(const float* query) const;

    /// Search with a precomputed QueryLUT (skips preprocess_query + FP16
    /// cast). Same semantics as `search`. The LUT must have been built by
    /// `build_query_lut` on a Searcher over the same quantizer.
    std::vector<Candidate> search_with_lut(const float* query,
                                            const QueryLUT& lut,
                                            uint32_t k,
                                            const SearchConfig& config);

    /// Push a single search to the pool and return its future immediately
    /// (no wait). Callers pushing a batch of queries can collect futures up
    /// front, then `.get()` them in order — this lets workers pull the next
    /// query as soon as they finish (no per-bank barrier).
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
    /// worker thread. Uses the Index's shared VamanaCore + store.
    std::vector<Candidate> search_body_(const float* query, uint32_t k,
                                         const SearchConfig& config,
                                         VamanaTLS& tls);

    /// Core search body with a precomputed LUT (skips preprocess_query +
    /// FP16 cast). The LUT pointers must remain valid for the call duration.
    std::vector<Candidate> search_body_with_lut_(const float* query,
                                                  const float* lut,
                                                  const float16_t* query_fp16,
                                                  uint32_t k,
                                                  const SearchConfig& config,
                                                  VamanaTLS& tls);

    void maybe_rebalance_();
};

}  // namespace sextant
