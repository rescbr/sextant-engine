#pragma once

/// @file ivf_scan_searcher.hpp
/// IVFScanSearcher — query serving over an IVFScanIndex (Option A production
/// path).
///
/// Per query:
///   1. Build the m×16 uint8 4-bit FastScan LUT ONCE (all shards share the
///      codebook; the LUT stays hot in L1 across all probed shards/blocks).
///   2. Cast query → FP16, route to n_probe nearest centroids (+multi-probe).
///   3. For each probed shard: stream `.codes4` sequentially (bypass cache),
///      run `simd::pq4_scan_many` per chunk, accumulate into a per-shard
///      top-W min-heap of (4-bit distance, shard-local idx).
///   4. Map shard-local idx → RowId via the shard's .rowids map.
///   5. Merge per-shard top-W into global top-W (dedup by RowId, keep min
///      distance). Return as Candidates (4-bit distance — exact rerank is the
///      DB layer's job).
///
/// Architecture (mirrors IVFSearcher): one IVF-scan pool of N workers (N =
/// min(num_threads, K)). Each worker runs a full query end-to-end. All mutable
/// state lives in `IVFScanWorkerState`; the searcher instance is read-only
/// across queries.

#include "algo/vamana_core.hpp"  // VamanaTLS-free; kept for BeamQuery-style consistency (none used)
#include "sextant/config.hpp"
#include "sextant/ivf_scan_index.hpp"
#include "sextant/types.hpp"

#include <cstdint>
#include <future>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sextant {

struct IVFScanIndex;

/// Per-worker scratch for the IVF-scan search path. All per-query mutable
/// state lives here; the IVFScanSearcher instance is touched read-only. One
/// worker searches its n_probe shards serially, reusing the same buffers.
struct IVFScanWorkerState {
    std::vector<float16_t> query_fp16;                  // dim (routing)
    std::vector<std::pair<float, uint32_t>> cent_dists;  // K (dist, shard_idx)
    std::vector<uint8_t> lut4;                           // m × 16 bytes
    /// Per-query top-W accumulator: (4-bit distance, RowId). Reused across
    /// queries (clear() preserves capacity).
    std::vector<std::pair<uint32_t, RowId>> scored;
    /// Hash dedup of merged candidates: RowId → min 4-bit distance.
    std::unordered_map<RowId, uint32_t> dedup;
    /// Per-shard top-W heap during scan (dist max-heap of size ≤ W; smaller
    /// distance = nearer). Reused across the n_probe shards of one query.
    /// Stored as (dist, local_idx) pairs with the max at front.
    std::vector<std::pair<uint32_t, uint32_t>> shard_heap;
    /// Staging buffer for CodeStream sequential pread. Owned per-worker so
    /// every probed shard reuses one allocation. Lazily sized on first use.
    std::vector<uint8_t> code_staging;
};

class IVFScanSearcher {
public:
    explicit IVFScanSearcher(IVFScanIndex& index, uint32_t num_threads = 1);
    ~IVFScanSearcher();

    IVFScanSearcher(const IVFScanSearcher&) = delete;
    IVFScanSearcher& operator=(const IVFScanSearcher&) = delete;
    IVFScanSearcher(IVFScanSearcher&&) = delete;
    IVFScanSearcher& operator=(IVFScanSearcher&&) = delete;

    /// Search for the top-W (config.fastscan_W, default 300) candidates by
    /// 4-bit PQ distance. Routes to `config.n_probe` (0 → index default)
    /// nearest centroids, scans each probed shard, merges, returns W (or
    /// fewer) Candidates sorted ascending by 4-bit distance. Exact rerank +
    /// final top-k is the caller's responsibility (DB layer).
    std::vector<Candidate> search(const float* query, uint32_t k,
                                   const SearchConfig& config);

    std::future<std::vector<Candidate>> search_one_async(
        const float* query, uint32_t k, const SearchConfig& config);

    std::vector<std::vector<Candidate>> search_batch(
        const float* queries, uint32_t n, uint32_t k,
        const SearchConfig& config);

    uint32_t num_threads() const { return num_threads_; }

private:
    IVFScanIndex& index_;
    uint32_t num_threads_;

    struct PoolImpl;
    std::unique_ptr<PoolImpl> pool_;

    std::vector<Candidate> search_body_(const float* query, uint32_t k,
                                          const SearchConfig& config,
                                          IVFScanWorkerState& w);
};

}  // namespace sextant
