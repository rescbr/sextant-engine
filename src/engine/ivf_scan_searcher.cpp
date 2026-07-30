#include "sextant/ivf_scan_searcher.hpp"

#include "algo/vamana_core.hpp"  // simd::dist_f16
#include "quant/pq_quantizer.hpp"
#include "quant/rabitq_quantizer.hpp"
#include "sextant/error.hpp"
#include "storage/code_stream.hpp"
#include "util/fp16.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
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

    PoolImpl(uint32_t n_threads, Dim dim, uint32_t K,
             uint32_t lut4_bytes, uint32_t lut8_bytes)
        : pool(n_threads,
               [dim, K, lut4_bytes, lut8_bytes](size_t /*id*/,
                                     std::shared_ptr<IVFScanWorkerState>& w) {
                   w = std::make_shared<IVFScanWorkerState>();
                   w->query_fp16.resize(dim);
                   w->cent_dists.reserve(K);
                   w->scored.reserve(512);
                   w->lut4.resize(lut4_bytes > 0 ? lut4_bytes : 1);
                   w->lut8.resize(lut8_bytes > 0 ? lut8_bytes : 1);
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
    const uint32_t lut4_bytes =
        static_cast<uint32_t>(index_.m4) * 16;      // m4 × 16 (4-bit: K=16)
    const uint32_t lut8_bytes =
        static_cast<uint32_t>(index_.m4) * 256;     // m4 × 256 (8-bit: K=256)
    pool_ = std::make_unique<PoolImpl>(num_threads_, index_.dim, index_.K,
                                        lut4_bytes, lut8_bytes);
}

IVFScanSearcher::~IVFScanSearcher() = default;

// ===========================================================================
// search_body_ — full per-query path: build LUT → route → scan each probed
// shard → merge top-W → return Candidates by 4-bit distance.
// ===========================================================================
std::vector<Candidate> IVFScanSearcher::search_body_(
    const float* query, uint32_t k, const SearchConfig& config,
    IVFScanWorkerState& w) {
    // RaBitQ needs a fundamentally different scan path (per-shard LUT rebuild
    // + per-vector distance finalization + selective rerank). Dispatch to the
    // dedicated method so the PQ/PRQ/anisotropic path below stays untouched.
    if (index_.quantizer_type == "rabitq") {
        return search_body_rabitq_(query, k, config, w);
    }
    const uint32_t K = index_.K;
    // W (rerank shortlist). Default 300 (validated on arxiv100k for recall
    // 0.99 at k=10). The --fastscan-w flag overrides for manual tuning.
    // TODO: auto-derive from shard size + distortion (W ∝ S^0.15, not √S
    // as first modeled — the sqrt scaling over-shrinks at small shards).
    // Needs a 3rd calibration point before shipping the formula.
    uint32_t W = config.fastscan_W > 0 ? config.fastscan_W : 300u;
    if (W == 0) W = 1;
    W = std::max(W, k);
    (void)k;

    uint32_t n_probe = config.n_probe > 0 ? config.n_probe : index_.n_probe_default;
    n_probe = std::max(1u, std::min(n_probe, K));

    // --- 1. Build FastScan LUT once + cast query → FP16 ---
    // The LUT stays hot in L1 across all probed shards and all of their
    // blocks — the per-query constant working set. Pick builder + buffer
    // by scan_pq_bits.
    const bool scan_8bit = (index_.scan_pq_bits == 8);
    if (scan_8bit) {
        float scale, offset;  // unused for argmin within one LUT
        index_.quantizer->build_fastscan_lut(query, w.lut8.data(),
                                             &scale, &offset);
    } else {
        index_.quantizer->build_fastscan_lut4(query, w.lut4.data(),
                                              /*scale_out=*/nullptr);
    }
    cast_fp32_to_fp16(query, w.query_fp16.data(), index_.dim);
    const MetricKind metric = index_.quantizer->metric();
    const uint32_t m = index_.m4;
    const uint32_t codes_per_block = scan_8bit ? 16 : 32;

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
            if (scan_8bit) {
                // 8-bit path: 16 codes/block. fastscan_block16 writes
                // 0xFFFFFFFF to invalid lanes itself (sentinel auto-loses
                // any comparison), so no external mask walk is needed.
                uint32_t out[16];
                simd::fastscan_block16(blk, w.lut8.data(), m,
                                       static_cast<uint16_t>(valid_mask), out);
                const uint32_t base = block_idx * 16;

                if (heap.size() < W) {
                    for (uint32_t j = 0; j < 16; j++) {
                        if (out[j] == 0xFFFFFFFFu) continue;
                        heap.emplace_back(out[j], base + j);
                        if (heap.size() == W) {
                            std::make_heap(heap.begin(), heap.end(), heap_less);
                            break;
                        }
                    }
                    if (heap.size() < W) return;
                }
                const uint32_t front_d = heap[0].first;
                // Front-compare: any of the 16 lanes beat the front?
    #if defined(SEXTANT_HAS_NEON)
                {
                    const uint32x4_t front4 = vdupq_n_u32(front_d);
                    const uint32x4_t* o = reinterpret_cast<const uint32x4_t*>(out);
                    uint32x4_t any = vcltq_u32(o[0], front4);
                    any = vorrq_u32(any, vcltq_u32(o[1], front4));
                    any = vorrq_u32(any, vcltq_u32(o[2], front4));
                    any = vorrq_u32(any, vcltq_u32(o[3], front4));
                    const uint64x2_t a64 = vreinterpretq_u64_u32(any);
                    if (vgetq_lane_u64(a64, 0) == 0 && vgetq_lane_u64(a64, 1) == 0)
                        return;
                }
    #else
                {
                    uint32_t block_min = 0xFFFFFFFFu;
                    for (uint32_t j = 0; j < 16; j++)
                        if (out[j] < block_min) block_min = out[j];
                    if (block_min >= front_d) return;
                }
    #endif
                for (uint32_t j = 0; j < 16; j++) {
                    const uint32_t d = out[j];
                    if (d >= heap[0].first) continue;
                    heap_replace(d, base + j);
                }
                return;
            }

            // 4-bit path: 32 codes/block, mask applied by caller.
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

    // --- 4. Dedup by RowId (keep first occurrence, not min distance). ---
    // Under a shared codebook, a replicated vector gets IDENTICAL PQ distance
    // in every shard it appears in (same code, same LUT). So keep-first and
    // keep-min are equivalent. Under per-shard codebooks, PQ distances live
    // in different scales per shard — comparing them (keep-min) is wrong.
    // Keep-first avoids the cross-shard comparison; FP32 rerank handles
    // final ranking. Zero regression for shared codebooks; strictly correct
    // for per-shard.
    w.dedup.clear();
    for (const auto& [d, rid] : w.scored) {
        w.dedup.try_emplace(rid, d);  // keeps first; ignores subsequent
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

// ===========================================================================
// search_body_rabitq_ — RaBitQ-specific scan path (Phase 2 + 3).
//
// Two structural differences from the PQ path (search_body_):
//   1. The FastScan LUT is rebuilt PER SHARD. PQ builds one LUT per query
//      (shared codebook); RaBitQ's LUT encodes `query - centroid`, and each
//      shard has a different centroid. The rebuild is cheap (M×16 bytes) but
//      must happen inside the per-shard loop.
//   2. The raw FastScan output is an INTERMEDIATE value, not the final
//      distance. Each vector's final L2sq distance is
//        est = or_minus_c_l2sqr + qr_to_c_l2sqr - 2*dp_multiplier*(raw - c34)
//      with per-vector factors (dp_multiplier, or_minus_c_l2sqr) read from the
//      `.factors` sidecar. c34 / qr_to_c_l2sqr are set per-shard by the LUT
//      build. Finalization happens per-lane during the scan.
//
// Selective rerank (Phase 3): the per-vector error bound lets borderline
// candidates survive the top-W cut. A candidate whose estimated distance is
// slightly above the W-th element may still be a true neighbor if its estimate
// is uncertain; we admit it when `est - error_bound < heap_max`. The engine
// does NOT hold the original FP32 vectors (that's the DB/benchmark layer's
// rerank job, identical to the PQ path), so the error bound is used for
// lower-bound admission rather than in-engine FP32 refinement.
// ===========================================================================
std::vector<Candidate> IVFScanSearcher::search_body_rabitq_(
    const float* query, uint32_t k, const SearchConfig& config,
    IVFScanWorkerState& w) {
    const uint32_t K = index_.K;
    uint32_t W = config.fastscan_W > 0 ? config.fastscan_W : 300u;
    if (W == 0) W = 1;
    W = std::max(W, k);
    (void)k;

    uint32_t n_probe = config.n_probe > 0 ? config.n_probe : index_.n_probe_default;
    n_probe = std::max(1u, std::min(n_probe, K));

    auto& rabitq = static_cast<RaBitQQuantizer&>(*index_.quantizer);
    const uint32_t m = index_.m4;

    // --- 1. Route: FP16 L2sq/IP to each centroid, pick n_probe nearest ---
    cast_fp32_to_fp16(query, w.query_fp16.data(), index_.dim);
    const MetricKind metric = rabitq.metric();
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

    // --- 2. Per-shard: rebuild LUT → scan → finalize → float heap ---
    // The heap carries REAL L2sq distances (finalized per-lane), not raw scan
    // tokens. Lower-bound admission: a candidate survives if its estimated
    // distance minus its error bound is below the current W-th distance.
    w.scored_f.clear();

    for (uint32_t p = 0; p < n_probe_eff; p++) {
        const uint32_t c = w.cent_dists[p].second;
        auto& shard = index_.shards[c];
        if (!shard || !shard->codes) continue;

        // FP16 → FP32 centroid (RaBitQ rotation needs FP32).
        const float16_t* centroid_f16 =
            index_.centroids.data() + static_cast<size_t>(c) * index_.dim;
        if (w.centroid_f32.size() < index_.dim) w.centroid_f32.resize(index_.dim);
        for (uint32_t i = 0; i < index_.dim; i++) {
            w.centroid_f32[i] = static_cast<float>(centroid_f16[i]);
        }

        // Rebuild the FastScan LUT for THIS shard (encodes query-centroid).
        // Sets rabitq's per-sharch c34_ / qr_to_c_l2sqr_ used by finalize.
        // RaBitQ LUT build populates w.rabitq_qs (thread-local query factors).
        rabitq.build_lut4_with_state(query, w.centroid_f32.data(),
                                     w.lut4.data(), w.rabitq_qs);

        const size_t need = code_stream_staging_bytes(*shard->codes);
        if (w.code_staging.size() < need) w.code_staging.resize(need);

        const float* factors_base = shard->factors.data();
        const uint32_t shard_n = shard->count;

        // Float max-heap of (dist, local_idx), size ≤ W. Max at front.
        auto& heap = w.shard_heap_f;
        heap.clear();
        heap.reserve(W + 32);
        const auto heap_less = [](const auto& a, const auto& b) {
            return a.first < b.first;
        };
        // g_error = ||query-centroid|| (precomputed by the LUT build as
        // qr_to_c_l2sqr_; sqrt once per shard). The per-vector error bound is
        // f_error(dim, factors) * g_error — get_error_bound folds both in.
        auto heap_replace = [&heap, W](float new_d, uint32_t new_idx) {
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

            for (uint32_t j = 0; j < 32; j++) {
                if (!((valid_mask >> j) & 1u)) continue;
                const uint32_t local_idx = base + j;
                if (local_idx >= shard_n) continue;
                const float* fac = factors_base +
                                   static_cast<size_t>(local_idx) * 2;
                // Dequantize the uint4 FastScan result to the true float
                // sign-dot, then finalize into a real L2sq estimate.
                const float est = rabitq.dequant_and_finalize(out[j], fac,
                                                                w.rabitq_qs);

                if (heap.size() < W) {
                    heap.emplace_back(est, local_idx);
                    if (heap.size() == W) {
                        std::make_heap(heap.begin(), heap.end(), heap_less);
                    }
                    continue;
                }
                // Lower-bound admission: admit if the candidate COULD be nearer
                // than the current W-th element (est minus its error bound).
                // This is the selective-rerank criterion — borderline
                // candidates with large uncertainty survive the cut.
                if (est >= heap[0].first) {
                    const float err = rabitq.error_bound(fac, w.rabitq_qs);
                    if (est - err >= heap[0].first) continue;
                }
                heap_replace(est, local_idx);
            }
        };
        shard->codes->scan(on_block, w.code_staging.data());

        // If the shard had fewer than W vectors, the heap was never heapified.
        if (heap.size() > 1 && heap.size() < W) {
            std::make_heap(heap.begin(), heap.end(), heap_less);
        }

        // Map local_idx → RowId; carry the finalized distance.
        for (const auto& [d, local_idx] : heap) {
            if (local_idx < shard->row_ids.size()) {
                w.scored_f.emplace_back(d, shard->row_ids[local_idx]);
            }
        }
    }

    // --- 3. Dedup by RowId, keep MIN finalized distance (RaBitQ distances ---
    // are on a common scale — the per-shard finalization produces comparable
    // L2sq estimates, unlike per-shard-codebook PQ). So keep-min is correct.
    std::unordered_map<RowId, float> best;
    best.reserve(w.scored_f.size());
    for (const auto& [d, rid] : w.scored_f) {
        auto it = best.find(rid);
        if (it == best.end() || d < it->second) best[rid] = d;
    }
    w.scored_f.clear();
    w.scored_f.reserve(best.size());
    for (const auto& [rid, d] : best) {
        w.scored_f.emplace_back(d, rid);
    }
    const uint32_t out_n =
        std::min<uint32_t>(W, static_cast<uint32_t>(w.scored_f.size()));
    if (out_n > 0) {
        std::nth_element(w.scored_f.begin(),
                         w.scored_f.begin() + static_cast<long>(out_n),
                         w.scored_f.end(),
                         [](const auto& a, const auto& b) {
                             return a.first < b.first;
                         });
        std::sort(w.scored_f.begin(),
                  w.scored_f.begin() + static_cast<long>(out_n),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
    }

    std::vector<Candidate> out;
    out.reserve(out_n);
    for (uint32_t i = 0; i < out_n; i++) {
        // RaBitQ distances are real L2sq estimates (much tighter than PQ's
        // 4-bit token). Callers may still FP32-rerank for final top-k.
        out.push_back({w.scored_f[i].second, w.scored_f[i].first});
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
