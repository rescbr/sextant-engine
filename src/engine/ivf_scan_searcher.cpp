#include "sextant/ivf_scan_searcher.hpp"

#include "algo/vamana_core.hpp"  // simd::dist_f16
#include "quant/pq_quantizer.hpp"
#include "sextant/error.hpp"
#include "storage/code_stream.hpp"
#include "util/fp16.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace sextant {

// ===========================================================================
// PoolImpl — persistent worker pool with per-worker IVFScanWorkerState.
// ===========================================================================
struct IVFScanSearcher::PoolImpl {
    ctpl::thread_pool_tls<IVFScanWorkerState> pool;

    PoolImpl(uint32_t n_threads, Dim dim, uint32_t K, uint32_t lut_bytes)
        : pool(n_threads,
               [dim, K, lut_bytes](size_t /*id*/,
                                    std::shared_ptr<IVFScanWorkerState>& w) {
                   w = std::make_shared<IVFScanWorkerState>();
                   w->query_fp16.resize(dim);
                   w->cent_dists.reserve(K);
                   w->scored.reserve(512);
                   w->lut4.resize(lut_bytes > 0 ? lut_bytes : 1);
                   w->dedup.reserve(512);
               }) {}
};

IVFScanSearcher::IVFScanSearcher(IVFScanIndex& index, uint32_t num_threads)
    : index_(index),
      num_threads_(num_threads == 0 ? std::thread::hardware_concurrency()
                                      : num_threads) {
    if (num_threads_ == 0) num_threads_ = 1;
    num_threads_ = std::min(num_threads_, std::max(1u, index_.K));

    if (!index_.quantizer) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFScanSearcher: index has no 4-bit quantizer");
    }
    const uint32_t lut_bytes =
        static_cast<uint32_t>(index_.m4) * 16;  // m4 × 16 (K=16 at bits=4)
    pool_ = std::make_unique<PoolImpl>(num_threads_, index_.dim, index_.K,
                                        lut_bytes);
}

IVFScanSearcher::~IVFScanSearcher() = default;

// ===========================================================================
// search_body_ — full per-query path: build LUT → route → scan each probed
// shard → merge top-W → return Candidates by 4-bit distance.
// ===========================================================================
std::vector<Candidate> IVFScanSearcher::search_body_(
    const float* query, uint32_t k, const SearchConfig& config,
    IVFScanWorkerState& w) {
    const uint32_t K = index_.K;
    // W is the rerank shortlist handed to the DB layer. config.fastscan_W
    // (default 300 per the spike's recall-0.99 point) replaces the graph
    // path's k × merge_oversample shortlist.
    uint32_t W = config.fastscan_W > 0 ? config.fastscan_W : 300u;
    if (W == 0) W = 1;
    // `k` is informational for the DB layer; we still size W to be ≥ k so the
    // caller has at least k candidates to rerank.
    W = std::max(W, k);
    (void)k;

    uint32_t n_probe = config.n_probe > 0 ? config.n_probe : index_.n_probe_default;
    n_probe = std::max(1u, std::min(n_probe, K));

    // --- 1. Build 4-bit LUT once + cast query → FP16 ---
    // The LUT (m × 16 = ~3 KB at m=192) stays hot in L1 across all probed
    // shards and all of their blocks — the per-query constant working set.
    index_.quantizer->build_fastscan_lut4(query, w.lut4.data(),
                                          /*scale_out=*/nullptr);
    cast_fp32_to_fp16(query, w.query_fp16.data(), index_.dim);
    const MetricKind metric = index_.quantizer->metric();
    const uint32_t m = index_.m4;

    // --- 2. Route: FP16 L2sq/IP to each centroid, pick n_probe nearest ---
    // (Plus the multi-probe extension from IVFSearcher — variable per query.)
    w.cent_dists.clear();
    for (uint32_t c = 0; c < K; c++) {
        if (!index_.shards[c]) {
            w.cent_dists.push_back({std::numeric_limits<float>::max(), c});
            continue;
        }
        const float16_t* centroid =
            index_.centroids.data() + static_cast<size_t>(c) * index_.dim;
        const float d = simd::dist_f16(metric, w.query_fp16.data(), centroid,
                                         index_.dim);
        w.cent_dists.push_back({d, c});
    }
    std::sort(w.cent_dists.begin(), w.cent_dists.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    uint32_t n_probe_eff = n_probe;
    const float ratio = config.multiprobe_ratio;
    if (ratio > 1.0f && n_probe < K) {
        const float d_nth = w.cent_dists[n_probe - 1].first;
        const float thresh = ratio * d_nth;
        for (uint32_t p = n_probe; p < K; p++) {
            if (w.cent_dists[p].first <= thresh) n_probe_eff++;
            else break;
        }
    }
    n_probe_eff = std::min(n_probe_eff, K);

    // --- 3. Scan each probed shard SERIALLY; maintain per-shard top-W heap ---
    // Max-heap of (dist, local_idx), size ≤ W. Smaller dist = nearer, so the
    // heap's front is the W-th nearest (the eviction candidate). When the heap
    // is full and a new dist is smaller than the front, pop the front and push.
    //
    // Profiling note: tried replacing this with flat-collect-all + post-sort
    // (the spike's pattern) — it was ~15% SLOWER despite avoiding log-W heap
    // ops, because flat-collect grows a per-shard buffer to ~22k pairs (vs the
    // heap's fixed W=300), and the reallocation + 22k pair writes outweigh the
    // heap-op savings. The heap's bounded memory is the right trade here.
    w.scored.clear();
    const uint32_t invalid = 0xFFFFFFFFu;

    for (uint32_t p = 0; p < n_probe_eff; p++) {
        const uint32_t c = w.cent_dists[p].second;
        auto& shard = index_.shards[c];
        if (!shard || !shard->codes) continue;

        // Size the staging buffer to one chunk (loaned to CodeStream::scan).
        const size_t need = code_stream_staging_bytes(*shard->codes);
        if (w.code_staging.size() < need) w.code_staging.resize(need);

        // Reset the per-shard heap. Reserve to avoid realloc during warmup.
        // Hand-written fixed-size max-heap (SoA layout: separate dist[] and
        // idx[] arrays) — avoids std::pair overhead, lambda comparator
        // indirection, and the double-traversal of pop_heap+push_heap. A
        // single "replace" operation (sift-down the root, then sift-up) does
        // one log-W pass instead of two.
        auto& heap = w.shard_heap;
        heap.clear();
        heap.reserve(W + 32);
        const auto heap_less = [](const auto& a, const auto& b) {
            return a.first < b.first;
        };

        // Inline heap replace-root: replace the max element (root) with a
        // new value and restore the max-heap property via a single sift-down.
        // Equivalent to pop_heap+assign+push_heap but with one traversal.
        auto heap_replace = [&heap, W](uint32_t new_d, uint32_t new_idx) {
            // Place new value at root, sift down.
            heap[0] = {new_d, new_idx};
            uint32_t pos = 0;
            const uint32_t n = W;
            while (true) {
                const uint32_t left = 2 * pos + 1;
                const uint32_t right = 2 * pos + 2;
                uint32_t largest = pos;
                if (left < n && heap[left].first > heap[largest].first)
                    largest = left;
                if (right < n && heap[right].first > heap[largest].first)
                    largest = right;
                if (largest == pos) break;
                std::swap(heap[pos], heap[largest]);
                pos = largest;
            }
        };

        auto on_block = [&](uint32_t block_idx, const uint8_t* blk,
                            uint32_t valid_mask) {
            uint32_t out[32];
            simd::pq4_block32(blk, w.lut4.data(), m, out);
            const uint32_t base = block_idx * 32;

            // Warmup phase: heap not yet full. Emplace all valid lanes.
            if (heap.size() < W) {
                for (uint32_t j = 0; j < 32; j++) {
                    if (!((valid_mask >> j) & 1u)) continue;
                    heap.emplace_back(out[j], base + j);
                    if (heap.size() == W) {
                        std::make_heap(heap.begin(), heap.end(), heap_less);
                        break;  // full now; remaining lanes go to steady-state
                    }
                }
                if (heap.size() < W) return;  // still warming up
            }

            // Steady-state fast path: check if ANY lane beats the front.
            const uint32_t front_d = heap[0].first;
#if defined(SEXTANT_HAS_NEON)
            {
                const uint32x4_t front4 = vdupq_n_u32(front_d);
                const uint32x4_t* o = reinterpret_cast<const uint32x4_t*>(out);
                uint32x4_t any = vcltq_u32(o[0], front4);
                any = vorrq_u32(any, vcltq_u32(o[1], front4));
                any = vorrq_u32(any, vcltq_u32(o[2], front4));
                any = vorrq_u32(any, vcltq_u32(o[3], front4));
                any = vorrq_u32(any, vcltq_u32(o[4], front4));
                any = vorrq_u32(any, vcltq_u32(o[5], front4));
                any = vorrq_u32(any, vcltq_u32(o[6], front4));
                any = vorrq_u32(any, vcltq_u32(o[7], front4));
                const uint64x2_t a64 = vreinterpretq_u64_u32(any);
                if (vgetq_lane_u64(a64, 0) == 0 && vgetq_lane_u64(a64, 1) == 0)
                    return;  // no winner in this block → skip
            }
#else
            {
                uint32_t block_min = 0xFFFFFFFFu;
                for (uint32_t j = 0; j < 32; j++)
                    if ((valid_mask >> j) & 1u)
                        block_min = std::min(block_min, out[j]);
                if (block_min >= front_d) return;
            }
#endif

            // At least one lane beats the front. Walk and do heap replace
            // only for winners (typically 0-3 per block after warmup).
            // heap_replace does a single sift-down + sift-up instead of the
            // std pop_heap + push_heap double traversal.
            for (uint32_t j = 0; j < 32; j++) {
                if (!((valid_mask >> j) & 1u)) continue;
                const uint32_t d = out[j];
                if (d >= heap[0].first) continue;  // re-check front (may have changed)
                heap_replace(d, base + j);
            }
        };
        shard->codes->scan(on_block, w.code_staging.data());

        // Heap now holds the shard's top-W (or all of it if shard_n < W).
        // Map local_idx → RowId and append to the global accumulator.
        for (const auto& [d, local_idx] : heap) {
            if (local_idx < shard->row_ids.size()) {
                w.scored.emplace_back(d, shard->row_ids[local_idx]);
            }
        }
    }

    // --- 4. Merge per-shard top-W via hash dedup (keep min distance), ---
    // --- then partial_sort the global top-W by 4-bit distance.         ---
    w.dedup.clear();
    for (const auto& [d, rid] : w.scored) {
        auto [it, inserted] = w.dedup.try_emplace(rid, d);
        if (!inserted && d < it->second) it->second = d;
    }
    w.scored.clear();
    w.scored.reserve(w.dedup.size());
    for (const auto& [rid, d] : w.dedup) {
        w.scored.emplace_back(d, rid);
    }
    const uint32_t out_n =
        std::min<uint32_t>(W, static_cast<uint32_t>(w.scored.size()));
    if (out_n > 0) {
        std::nth_element(w.scored.begin(),
                         w.scored.begin() + static_cast<long>(out_n),
                         w.scored.end(),
                         [](const auto& a, const auto& b) {
                             return a.first < b.first;
                         });
        std::sort(w.scored.begin(),
                  w.scored.begin() + static_cast<long>(out_n),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
    }

    std::vector<Candidate> out;
    out.reserve(out_n);
    for (uint32_t i = 0; i < out_n; i++) {
        // The 4-bit distance is an ordering token, not a real L2; hand it
        // back as `dist` so callers can re-sort if they wish. The DB layer
        // is expected to ignore it for rerank.
        out.push_back({w.scored[i].second,
                       static_cast<float>(w.scored[i].first)});
    }
    return out;
}

std::future<std::vector<Candidate>> IVFScanSearcher::search_one_async(
    const float* query, uint32_t k, const SearchConfig& config) {
    return pool_->pool.push(
        [this, query, k, &config](size_t /*id*/, IVFScanWorkerState& w) {
            return search_body_(query, k, config, w);
        });
}

std::vector<Candidate> IVFScanSearcher::search(const float* query, uint32_t k,
                                                 const SearchConfig& config) {
    return search_one_async(query, k, config).get();
}

std::vector<std::vector<Candidate>> IVFScanSearcher::search_batch(
    const float* queries, uint32_t n, uint32_t k, const SearchConfig& config) {
    std::vector<std::vector<Candidate>> results(n);
    if (n == 0) return results;

    std::vector<std::future<std::vector<Candidate>>> futs;
    futs.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        const float* q = &queries[static_cast<size_t>(i) * index_.dim];
        futs.push_back(search_one_async(q, k, config));
    }
    for (uint32_t i = 0; i < n; i++) {
        results[i] = futs[i].get();
    }
    return results;
}

}  // namespace sextant
