#pragma once

/// @file ivf_searcher.hpp
/// IVFSearcher — query serving over an IVFIndex.
///
/// Routes each query to `n_probe` nearest shards by FP16 centroid distance,
/// searches each probed shard, then merges/dedups the candidates by row_id
/// and truncates to k. See `docs/ivf_probe_design.md` and the optimization
/// plan at `~/.local/state/maki/plans/allowing-enhanced-kangaroo.md`.
///
/// Architecture (Phase 1: query-parallel):
///   - ONE IVF pool of N workers (N = min(num_threads, hardware_concurrency)).
///     Each worker runs a FULL query end-to-end: route → search its n_probe
///     shards serially → merge. This mirrors how the merged-graph Searcher
///     scales (queries are the unit of parallelism, not shards).
///   - The worker calls `shard_index->core->search(...)` DIRECTLY with its
///     own `VamanaTLS`. This bypasses the per-shard Searcher's size-1 pool
///     entirely (no K nested pools, no K×Q future-waits per batch). One
///     `VamanaTLS` per worker is reused across the n_probe shards it probes
///     in one query (VamanaCore::search lazily resizes on count mismatch).
///   - Query LUT preprocessing is done ONCE per query (inside the worker) and
///     reused across that worker's n_probe shard searches. All shards share
///     the same trained quantizer.
///   - NO shard Searcher objects. The IVFSearcher references the K shard
///     Indexes directly via the IVFIndex. Cache-rebalance / diagnostics hooks
///     from the Searcher class are intentionally dropped from the hot path.
///
/// Routing uses FP16 L2sq (`l2sq_f16`) — the same primitive the MemGraph
/// ball uses for tier-1 distance — keeping the whole query path in FP16.

#include "algo/vamana_core.hpp"
#include "sextant/config.hpp"
#include "sextant/ivf_index.hpp"
#include "sextant/types.hpp"

#include <cstdint>
#include <future>
#include <memory>
#include <vector>

namespace sextant {

struct IVFIndex;

/// Per-IVF-worker scratch. Owns the per-query mutable state (routing buffers,
/// merge accumulator, LUT) AND one reusable VamanaTLS. One worker searches
/// its n_probe shards serially within a single query, reusing the same tls
/// (VamanaCore::search resizes lazily on shard-count mismatch).
struct IVFWorkerState {
    std::vector<float16_t> query_fp16;                  // dim (routing + ball distance)
    std::vector<std::pair<float, uint32_t>> cent_dists;  // K (dist, shard_idx)
    std::vector<std::pair<float, RowId>> scored;          // n_probe × k_local (dist, global_id)
    std::vector<float> query_lut;                        // lut_size() floats (shared quantizer)
    VamanaTLS tls;                                        // reused across n_probe shards
    /// Hash dedup of merged candidates (Phase 2a): row_id → min distance.
    /// Avoids the old double-sort (O(N log N) × 2) — hash insert is O(N),
    /// then we collect pairs and partial_sort only the top-k. Reused across
    /// queries (clear() preserves capacity).
    std::unordered_map<RowId, float> dedup;
};

class IVFSearcher {
public:
    /// Construct over an IVFIndex. `num_threads` controls the IVF-level pool
    /// (parallelism across queries). 0 → hardware_concurrency; 1 = single
    /// worker, same code path (project rule: single-threaded = pool of 1).
    explicit IVFSearcher(IVFIndex& index, uint32_t num_threads = 1);
    ~IVFSearcher();

    IVFSearcher(const IVFSearcher&) = delete;
    IVFSearcher& operator=(const IVFSearcher&) = delete;
    IVFSearcher(IVFSearcher&&) = delete;
    IVFSearcher& operator=(IVFSearcher&&) = delete;

    /// Search for k nearest neighbors of `query` (dim floats).
    ///
    /// Routes to `config.n_probe` (0 → index default) nearest centroids,
    /// searches each probed shard at `k_local = k × config.merge_oversample`,
    /// merges candidates (dedup by row_id, keep min distance), truncates to k.
    /// Returns candidates sorted ascending by distance. Synchronous (pushes
    /// one task to the pool, waits on its future).
    std::vector<Candidate> search(const float* query, uint32_t k,
                                   const SearchConfig& config);

    /// Push a single search and return its future immediately. Lets callers
    /// pushing a batch collect futures up front, then `.get()` them in order
    /// (workers pull the next query as soon as they finish).
    std::future<std::vector<Candidate>> search_one_async(
        const float* query, uint32_t k, const SearchConfig& config);

    /// Search a batch of `n` queries (each `index.dim` floats, packed
    /// row-major). Fans out across the pool. Returns results in input order.
    std::vector<std::vector<Candidate>> search_batch(
        const float* queries, uint32_t n, uint32_t k,
        const SearchConfig& config);

    uint32_t num_threads() const { return num_threads_; }

private:
    IVFIndex& index_;
    uint32_t num_threads_;

    struct PoolImpl;
    std::unique_ptr<PoolImpl> pool_;

    /// Core query body (route + n_probe shard searches + merge). Runs on a
    /// pool worker thread. Uses `w` for all scratch — no instance-level
    /// mutable state is touched (enables query-level parallelism).
    std::vector<Candidate> search_body_(const float* query, uint32_t k,
                                          const SearchConfig& config,
                                          IVFWorkerState& w);
};

}  // namespace sextant
