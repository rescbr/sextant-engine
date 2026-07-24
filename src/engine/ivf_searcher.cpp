#include "sextant/ivf_searcher.hpp"

#include "algo/vamana_core.hpp"  // simd::dist_f16
#include "quant/pq_quantizer.hpp"
#include "sextant/error.hpp"
#include "util/fp16.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sextant {

// ===========================================================================
// PoolImpl — persistent worker pool with per-worker IVFWorkerState scratch.
// Hides ctpl from the header. Workers are created once and reused across
// queries (no per-query thread spawn). Each worker's IVFWorkerState is lazily
// sized on first use.
// ===========================================================================
struct IVFSearcher::PoolImpl {
    ctpl::thread_pool_tls<IVFWorkerState> pool;

    PoolImpl(uint32_t n_threads, Dim dim, uint32_t K, uint32_t lut_sz)
        : pool(n_threads,
               [dim, K, lut_sz](size_t /*id*/,
                                 std::shared_ptr<IVFWorkerState>& w) {
                   w = std::make_shared<IVFWorkerState>();
                   w->query_fp16.resize(dim);
                   w->cent_dists.reserve(K);
                   w->scored.reserve(256);
                   w->query_lut.resize(lut_sz > 0 ? lut_sz : 1);
                   // tls resized lazily by VamanaCore::search on first use.
               }) {}
};

IVFSearcher::IVFSearcher(IVFIndex& index, uint32_t num_threads)
    : index_(index),
      num_threads_(num_threads == 0 ? std::thread::hardware_concurrency()
                                      : num_threads) {
    if (num_threads_ == 0) num_threads_ = 1;
    // No point having more IVF workers than shards.
    num_threads_ = std::min(num_threads_, std::max(1u, index_.K));

    // LUT size comes from the (shared) quantizer on the first present shard.
    uint32_t lut_sz = 0;
    for (uint32_t k = 0; k < index_.K; k++) {
        if (index_.shards[k] && index_.shards[k]->quantizer) {
            lut_sz = index_.shards[k]->quantizer->lut_size();
            break;
        }
    }
    pool_ = std::make_unique<PoolImpl>(num_threads_, index_.dim, index_.K, lut_sz);
}

IVFSearcher::~IVFSearcher() = default;

// ===========================================================================
// search_body_ — the full per-query path. Runs on a pool worker thread.
// All mutable state lives in `w` (the worker's IVFWorkerState); the
// IVFSearcher instance itself is touched read-only. This is what enables
// query-level parallelism — N workers can run search_body_ concurrently.
// ===========================================================================
std::vector<Candidate> IVFSearcher::search_body_(const float* query, uint32_t k,
                                                   const SearchConfig& config,
                                                   IVFWorkerState& w) {
    if (k == 0) return {};

    const uint32_t K = index_.K;
    uint32_t n_probe = config.n_probe > 0 ? config.n_probe : index_.n_probe_default;
    n_probe = std::max(1u, std::min(n_probe, K));

    const uint32_t oversample = std::max(1u, config.merge_oversample);
    const uint32_t k_local = k * oversample;

    // Build the PQ LUT + FP16 query ONCE (all shards share the same trained
    // quantizer). query_lut / query_fp16 live in the worker scratch and are
    // reused across this worker's n_probe shard searches below.
    PqQuantizer* quantizer = nullptr;
    for (uint32_t c = 0; c < K && !quantizer; c++) {
        if (index_.shards[c]) quantizer = index_.shards[c]->quantizer.get();
    }
    if (quantizer && !w.query_lut.empty()) {
        quantizer->preprocess_query(query, w.query_lut.data());
    }
    const MetricKind metric =
        quantizer ? quantizer->metric() : MetricKind::L2Sq;
    cast_fp32_to_fp16(query, w.query_fp16.data(), index_.dim);

    // --- 1. Route: FP16 L2sq to each centroid, pick n_probe nearest (+multi-probe) ---
    // Multi-probe (config.multiprobe_ratio > 1.0): after picking the n_probe
    // nearest centroids, extend the probe set to include any centroid whose
    // distance ≤ ratio × d[n_probe-1]. This recovers true NNs in boundary
    // shards that strict nearest-centroid routing misses (Voronoi-boundary
    // ambiguity). See docs/ivf_routing_analysis.md. The probe set is variable
    // per query — unambiguous queries (large gap to the (n_probe+1)-th
    // centroid) probe exactly n_probe; boundary queries probe more.
    w.cent_dists.clear();
    for (uint32_t c = 0; c < K; c++) {
        if (!index_.shards[c]) {
            w.cent_dists.push_back({std::numeric_limits<float>::max(), c});
            continue;
        }
        const float16_t* centroid =
            index_.centroids.data() + static_cast<size_t>(c) * index_.dim;
        const float d = simd::dist_f16(metric, w.query_fp16.data(), centroid, index_.dim);
        w.cent_dists.push_back({d, c});
    }
    // Sort ascending by distance. K is small (≤64), so a full sort is cheap
    // and lets us apply the multi-probe threshold cleanly.
    std::sort(w.cent_dists.begin(), w.cent_dists.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    // Determine the probe set size: at least n_probe, extended by the ratio.
    uint32_t n_probe_eff = n_probe;
    const float ratio = config.multiprobe_ratio;
    if (ratio > 1.0f && n_probe < K) {
        const float d_nth = w.cent_dists[n_probe - 1].first;
        const float thresh = ratio * d_nth;
        for (uint32_t p = n_probe; p < K; p++) {
            if (w.cent_dists[p].first <= thresh) n_probe_eff++;
            else break;  // sorted ascending
        }
    }
    n_probe_eff = std::min(n_probe_eff, K);

    // --- 2. Search the probed shards SERIALY on this worker.
    // n_probe_eff = n_probe (strict) or n_probe + multi-probe extensions
    // (boundary shards within the ratio threshold). Each shard's VamanaCore::
    // search is const + takes the worker's tls, so this is thread-safe. The
    // tls is resized lazily on shard-count mismatch (handled inside
    // VamanaCore::search). This bypasses the shard's own Searcher/pool
    // entirely — no nested pool, no per-probe future-wait.
    //
    // IVF Workstream A2/A3: when the shard carries sub-cluster entry-point data
    // (shard_sub_centroids, k'×dim FP16; shard_sub_medoids, k'×M disk-pos IDs),
    // pick the query's closest sub-cluster by FP16 L2sq (k' distances, ~free)
    // and seed beam_search from THAT sub-cluster's M medoids via
    // forced_entry_points. This is the novel IVF-only win at A3 scale: the
    // default multi-start would scan all k'×M=32 entry points (32 distances) to
    // pick the top-4; A2 scans k'=8 sub-centroids then fetches the chosen
    // sub-cluster's M=4 medoids by index (8 distances + 0). So E=k'×M can grow
    // without search-time penalty — the cost is fixed at k'. Merged structurally
    // cannot do this (one global entry set, no per-region tag).
    const float* lut_ptr = (!w.query_lut.empty()) ? w.query_lut.data() : nullptr;
    BeamQuery bq;
    bq.query_lut = lut_ptr;
    bq.query_fp16 = w.query_fp16.data();
    w.scored.clear();

    // L division: treat config.L_search as a GLOBAL beam-width budget across
    // all n_probe shards, not per-shard. The prior path passed the full L to
    // every shard (total work = n_probe × L), which made the user-facing L
    // knob decoupled from IVF behavior — L=25 and L=400 gave identical recall
    // AND QPS at n_probe=5 (each shard's search converges naturally in <25
    // pops, so the extra budget was simply unused). Dividing restores the
    // knob's meaning.
    //
    // Floor at the shard's actual R (R_shard, already persisted per-shard in
    // .meta and reopened via `shard->params.R`). Beam_search needs L ≥ R for
    // graph traversal to make sense (at least one full neighbor expansion);
    // below R, the search can't reach beyond the entry point's immediate
    // neighborhood. Flooring at R_shard (typically ~23 for R=35, since
    // R_shard = 2R/3) finally exposes the L knob: L=400 np=21 → L_shard=23
    // (the floor); L=400 np=5 → L_shard=80; L=100 np=2 → L_shard=50.
    //
    // Earlier floors (k_local=200, then k=100) were far above natural
    // convergence, making the L knob a no-op for any reasonable L.
    //
    // All shards in an index share the same R_shard (set at build time from
    // the global R), so reading it from the first shard is safe.
    const uint32_t R_shard = index_.shards.empty() ? 1u
        : static_cast<uint32_t>(index_.shards[0]->params.R);
    const uint32_t L_shard = std::max(R_shard,
        config.L_search / std::max(1u, n_probe_eff));

    // One-time hint when the user's L is being clamped hard by the floor.
    // Helps users who pass --search-beam-width 800 expecting more recall —
    // for IVF the recall knob is n_probe, not L (shards converge in ~R_shard
    // pops regardless of L_shard).
    static std::once_flag hint_flag;
    std::call_once(hint_flag, [&]() {
        const uint32_t L_effective = L_shard * std::max(1u, n_probe_eff);
        if (L_effective < config.L_search / 2 && config.L_search > L_shard * 4) {
            spdlog::info("[sextant] IVF: L_search={} divided by n_probe={} → "
                         "L_shard={} (floored at R_shard={}). The effective "
                         "recall knob for IVF is --n-probe, not L.",
                         config.L_search, n_probe_eff, L_shard, R_shard);
        }
    });

    for (uint32_t p = 0; p < n_probe_eff; p++) {
        const uint32_t c = w.cent_dists[p].second;
        auto& shard = index_.shards[c];
        if (!shard || !shard->core) continue;

        // A2/A3: query-adaptive entry-point selection (OFF by default).
        // Validated 2026-07-22: A2's sub-cluster indexing saves ~nothing vs
        // the default multi-start (entry-point selection was never the
        // bottleneck; beam_search's neighbor expansion is), and at A3 scale
        // (M=4/32 entry points) recall+0.4pp costs −6% QPS. The .epc infra
        // stays for experimentation; enable with SEXTANT_ENABLE_A2=1.
        const auto& sub_centroids = index_.shard_sub_centroids[c];
        const auto& sub_medoids = index_.shard_sub_medoids[c];
        std::vector<uint32_t> forced_ep;  // empty unless A2 selects ≥1
        static const char* kA2Enable = std::getenv("SEXTANT_ENABLE_A2");
        static const bool a2_enabled =
            kA2Enable && kA2Enable[0] == '1';
        if (a2_enabled && !sub_centroids.empty() && !sub_medoids.empty()) {
            const uint32_t k_sub =
                static_cast<uint32_t>(sub_centroids.size() / index_.dim);
            const uint32_t M = k_sub > 0
                ? static_cast<uint32_t>(sub_medoids.size() / k_sub) : 0;
            // Pick the closest sub-cluster (argmin FP16 L2sq to query).
            float best_d = std::numeric_limits<float>::max();
            uint32_t best_sc = 0;
            for (uint32_t sc = 0; sc < k_sub; sc++) {
                const float16_t* cen =
                    sub_centroids.data() + static_cast<size_t>(sc) * index_.dim;
                const float d = simd::dist_f16(metric, w.query_fp16.data(), cen, index_.dim);
                if (d < best_d) { best_d = d; best_sc = sc; }
            }
            // Seed from that sub-cluster's M medoids (disk positions already).
            forced_ep.reserve(M);
            const uint32_t* slice =
                sub_medoids.data() + static_cast<size_t>(best_sc) * M;
            for (uint32_t m = 0; m < M; m++) {
                forced_ep.push_back(slice[m]);
            }
            bq.forced_entry_points = &forced_ep;
        } else {
            bq.forced_entry_points = nullptr;
        }

        auto cands = shard->core->search(bq, k_local, L_shard,
                                          config.io_limit, w.tls,
                                          config.early_exit_patience);
        for (const auto& cand : cands) {
            w.scored.emplace_back(cand.dist, cand.row_id);
        }
    }
    bq.forced_entry_points = nullptr;  // dangling-pointer hygiene

    // --- 3. Merge + dedup by row_id (keep min distance), take top-k ---
    // Phase 2a: hash dedup (O(N)) + partial_sort top-k (O(N + k log k)),
    // replacing the old double-sort (O(N log N) × 2). At n_probe=4/k_local=200
    // the old merge was the #1 hot spot (2.7× the graph search cost).
    w.dedup.clear();
    for (const auto& [dist, rid] : w.scored) {
        auto [it, inserted] = w.dedup.try_emplace(rid, dist);
        if (!inserted && dist < it->second) {
            it->second = dist;  // keep min distance for duplicates
        }
    }
    // Collect unique (dist, row_id) pairs and partial_sort the top-k by dist.
    // We reuse `scored` as the scratch buffer (its contents are already consumed).
    w.scored.clear();
    w.scored.reserve(w.dedup.size());
    for (const auto& [rid, dist] : w.dedup) {
        w.scored.emplace_back(dist, rid);
    }
    const uint32_t out_n = std::min<uint32_t>(k, static_cast<uint32_t>(w.scored.size()));
    if (out_n > 0) {
        std::nth_element(w.scored.begin(),
                         w.scored.begin() + static_cast<long>(out_n),
                         w.scored.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        std::sort(w.scored.begin(),
                  w.scored.begin() + static_cast<long>(out_n),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    std::vector<Candidate> out;
    out.reserve(out_n);
    for (uint32_t i = 0; i < out_n; i++) {
        out.push_back({w.scored[i].second, w.scored[i].first});
    }
    return out;
}

// ===========================================================================
// search_one_async — push one query, return future immediately.
// ===========================================================================
std::future<std::vector<Candidate>> IVFSearcher::search_one_async(
    const float* query, uint32_t k, const SearchConfig& config) {
    return pool_->pool.push(
        [this, query, k, &config](size_t /*id*/, IVFWorkerState& w) {
            return search_body_(query, k, config, w);
        });
}

// ===========================================================================
// search — synchronous: push one task, wait on its future.
// ===========================================================================
std::vector<Candidate> IVFSearcher::search(const float* query, uint32_t k,
                                             const SearchConfig& config) {
    return search_one_async(query, k, config).get();
}

// ===========================================================================
// search_batch — fan a batch across the pool; collect futures in input order.
// Mirrors Searcher::search_batch. This is the hot path for the benchmark:
// all n queries are pushed up front so workers steal the next query as soon
// as they finish (no per-query barrier).
// ===========================================================================
std::vector<std::vector<Candidate>> IVFSearcher::search_batch(
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
