// Estimator — sample-driven BuildConfig resolution.
//
// Extracted from engine.cpp during Layer 2 Phase D. Builds temporary in-RAM
// Indices via Builder (no temp files — architecture decision), measures graph
// topology + search quality, and returns the optimal ResolvedParams.
//
// estimate_config is the most expensive operation in the analyze/autobuild
// pipeline (3-min sweep at 20K-sample × 7 mini-builds on arxiv-nomic), so the
// mini-build path skips the sidecar flush — it goes through Builder's private
// pass1/pass2/parallel_construct directly via friend access.

#include "sextant/estimator.hpp"

#include "fbin_source.hpp"
#include "memory_source.hpp"
#include "probe.hpp"
#include "sextant/builder.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"
#include "sextant/system.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/direct_io.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sextant {

// ===========================================================================
// estimate_config — sample-driven parameter estimation.
//
// Builds one or more mini-indices on a reservoir sample, measures graph
// topology (OPT-SNG R prediction) and search quality (alpha sweep), and
// returns the optimal build config. Does NOT write sidecar files — all
// measurement is in-RAM on fresh Engine instances.
// ===========================================================================
namespace {
/// Sample target for estimate_config mini-builds. Smaller than the production
/// kSampleTarget (256K) because each candidate (R, alpha) builds a full
/// mini-index — at 256K, 7 mini-builds would take >10 min. 20K keeps each
/// build under ~20s while preserving graph topology (degrees, clustering,
// dead-ends) and search quality (recall is identical at 20K for FlatNodeStore
// since the cache is 100% hits). Matches kProbePool.
inline constexpr uint64_t kEstimateSampleTarget = 20000;
}  // namespace

std::unique_ptr<Index> Estimator::build_mini_(const float* sample,
                                             uint64_t sample_n, Dim dim,
                                             const ResolvedParams& params) {
    auto mini = std::make_unique<Index>();
    mini->count = sample_n;
    mini->dim = dim;
    mini->path = "/dev/null";  // never written; just non-empty

    // Threads: respect any user override (--threads); otherwise use all
    // available cores. The construct loop is work-stealing and embarrassingly
    // parallel — there's no reason to cap below hardware concurrency, and
    // doing so wastes cores during the ~3-minute analyze sweep.
    ResolvedParams mp = params;
    if (mp.num_threads == 0) {
        mp.num_threads = std::thread::hardware_concurrency();
    }

    MemorySource src(sample, sample_n, dim);
    Builder b(*mini);

    // Mirror build_partitioned K==1 fast path: pass1 → buffers → pass2 →
    // FP16 → core/flat_store → parallel_construct → entry points.
    b.pass1_sample_and_train(src, mp);
    mini->node_size =
        VamanaCore::static_node_size(mp.R, mini->code_size);

    // Allocate codes_buffer + nodes_buffer (aligned, zeroed).
    {
        const size_t codes_bytes =
            static_cast<size_t>(sample_n) * mini->code_size;
        const size_t nodes_bytes =
            static_cast<size_t>(sample_n) * mini->node_size;
        mini->codes_buffer = static_cast<uint8_t*>(
            aligned_alloc(kDiskAlign, codes_bytes));
        mini->nodes_buffer = static_cast<uint8_t*>(
            aligned_alloc(kDiskAlign, nodes_bytes));
        if (!mini->codes_buffer || !mini->nodes_buffer) {
            throw Error(ErrorCode::OutOfMemory,
                        "build_mini_: buffer alloc failed");
        }
        std::memset(mini->codes_buffer, 0, codes_bytes);
        std::memset(mini->nodes_buffer, 0, nodes_bytes);
    }

    b.pass2_encode(src, mp);

    // Load raw vectors as FP16 for the FP16 prune.
    {
        const size_t vecs_bytes =
            static_cast<size_t>(sample_n) * dim * sizeof(float16_t);
        mini->raw_vecs_buffer = static_cast<float16_t*>(
            aligned_alloc(kDiskAlign, vecs_bytes));
        if (!mini->raw_vecs_buffer) {
            throw Error(ErrorCode::OutOfMemory,
                        "build_mini_: raw_vecs_buffer alloc failed");
        }
        for (uint64_t i = 0; i < sample_n; i++) {
            float16_t* dst =
                mini->raw_vecs_buffer + static_cast<size_t>(i) * dim;
            const float* svec = sample + static_cast<size_t>(i) * dim;
            for (uint32_t d = 0; d < dim; d++) {
                dst[d] = static_cast<float16_t>(svec[d]);
            }
        }
    }

    const uint32_t n = static_cast<uint32_t>(sample_n);

    // Set up core + flat_store (mirror build_partitioned K==1).
    VamanaParams vp = VamanaParams::from_resolved(mp, dim);
    mini->core = std::make_unique<VamanaCore>(vp, *mini->quantizer);
    mini->core->set_build_codes(mini->codes_buffer, n);
    mini->core->set_build_nodes(mini->nodes_buffer);
    mini->core->prepare_for_build(n);
    mini->flat_store = std::make_unique<FlatNodeStore>(
        mini->nodes_buffer, mini->codes_buffer,
        mini->node_size, mini->code_size);
    mini->core->set_store(mini->flat_store.get());
    mini->core->set_build_vecs(mini->raw_vecs_buffer);

    b.parallel_construct(mp);

    mini->core->set_build_vecs(nullptr);
    mini->core->compute_entry_points();

    return mini;
}

GraphStats Estimator::measure_graph_stats_(const Index& mini) {
    GraphStats stats;
    const uint32_t n = static_cast<uint32_t>(mini.count);
    if (n == 0 || !mini.nodes_buffer) return stats;

    // avg_degree + dead_end_frac: single pass over all nodes.
    uint64_t total_degree = 0;
    uint32_t dead_ends = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t* node =
            mini.nodes_buffer + static_cast<size_t>(i) * mini.node_size;
        const uint16_t deg = VamanaCore::get_neighbor_count(node);
        total_degree += deg;
        if (deg <= 1) ++dead_ends;
    }
    stats.avg_degree = static_cast<double>(total_degree) / n;
    stats.dead_end_frac = static_cast<double>(dead_ends) / n;

    // clustering_coeff: sample min(1000, N) random nodes. For each sampled
    // node, get its neighbors, count pairs of neighbors that are also mutual
    // neighbors (triangle fraction). Bounded by R² per node.
    std::mt19937 rng(0xABCD1234u);
    const uint32_t sample_cnt = std::min<uint32_t>(1000, n);
    double coeff_sum = 0.0;
    uint32_t coeff_count = 0;
    for (uint32_t s = 0; s < sample_cnt; s++) {
        const uint32_t i = (sample_cnt == n) ? s : rng() % n;
        const uint8_t* node =
            mini.nodes_buffer + static_cast<size_t>(i) * mini.node_size;
        const uint16_t deg = VamanaCore::get_neighbor_count(node);
        if (deg < 2) continue;

        // Collect neighbor IDs into a hash set.
        std::unordered_set<uint32_t> nbrs;
        nbrs.reserve(deg);
        for (uint16_t j = 0; j < deg; j++) {
            nbrs.insert(VamanaCore::get_neighbor(node, j));
        }

        // Count connected pairs among neighbors (triangles through node i).
        uint32_t pairs = 0;
        uint32_t connected = 0;
        for (auto it1 = nbrs.begin(); it1 != nbrs.end(); ++it1) {
            const uint32_t nb1 = *it1;
            if (nb1 >= n) continue;
            const uint8_t* nb1_node =
                mini.nodes_buffer + static_cast<size_t>(nb1) * mini.node_size;
            const uint16_t nb1_deg =
                VamanaCore::get_neighbor_count(nb1_node);
            // Build nb1's neighbor set.
            std::unordered_set<uint32_t> nb1_nbrs;
            for (uint16_t j = 0; j < nb1_deg; j++) {
                nb1_nbrs.insert(
                    VamanaCore::get_neighbor(nb1_node, j));
            }
            auto it2 = it1;
            ++it2;
            for (; it2 != nbrs.end(); ++it2) {
                ++pairs;
                if (nb1_nbrs.count(*it2)) ++connected;
            }
        }
        if (pairs > 0) {
            coeff_sum += static_cast<double>(connected) / pairs;
            ++coeff_count;
        }
    }
    stats.clustering_coeff =
        coeff_count > 0 ? coeff_sum / coeff_count : 0.0;
    // median_lid is filled by estimate_config from truth distances.
    return stats;
}

Estimator::SearchQuality Estimator::measure_search_(
    const Index& mini, const float* sample, uint64_t sample_n, Dim dim,
    const std::vector<std::vector<uint32_t>>& truth_ids,
    const std::vector<std::vector<float>>& truth_dists,
    const std::vector<uint32_t>& qidx,
    uint32_t L, uint32_t k, uint32_t rerank) {
    SearchQuality sq{0.0, 0.0};
    if (!mini.core || !mini.quantizer || qidx.empty()) return sq;

    PqQuantizer& q = *mini.quantizer;

    // Per-call scratch (single-threaded measurement). Sized once; reused
    // across all sampled queries in this measure_search_ call.
    VamanaTLS tls;
    if (mini.count > 0) tls.resize(static_cast<uint32_t>(mini.count));
    std::vector<float> lut(q.lut_size());
    // Over-fetch for rerank: k * rerank (matches the production benchmark tool
    // at benchmark.cpp:225). Previously `k + rerank`, which under-fetched and
    // made the mini-build recall measurement pessimistic.
    const uint32_t fetch_k = k * rerank;

    double recall_sum = 0.0;
    uint64_t prox_in = 0, prox_total = 0;
    constexpr double kEps = 1e-12;

    for (size_t qi = 0; qi < qidx.size(); qi++) {
        const float* qv = sample + static_cast<size_t>(qidx[qi]) * dim;
        q.preprocess_query(qv, lut.data());

        // Production-style search: index_->core->search returns candidates ranked by
        // PQ LUT distance. We over-fetch k+rerank candidates.
        auto results = mini.core->search(lut.data(), fetch_k, L, /*io_limit=*/0,
                                          tls);
        if (results.empty()) continue;

        // Rerank by true L2sq distance computed from the FP32 sample buffer.
        std::vector<std::pair<float, int32_t>> scored;
        scored.reserve(results.size());
        for (const auto& c : results) {
            if (c.row_id < 0 ||
                static_cast<uint64_t>(c.row_id) >= sample_n) continue;
            const float* bv =
                sample + static_cast<size_t>(c.row_id) * dim;
            double d = 0.0;
            for (uint32_t d2 = 0; d2 < dim; d2++) {
                const double diff = double(qv[d2]) - double(bv[d2]);
                d += diff * diff;
            }
            scored.emplace_back(float(d), c.row_id);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        const uint32_t topk = std::min<uint32_t>(k, scored.size());
        if (topk == 0) continue;

        // Recall@k: fraction of truth_ids[qi][:k] found in top-k.
        std::unordered_set<uint32_t> result_set;
        for (uint32_t i = 0; i < topk; i++) {
            result_set.insert(static_cast<uint32_t>(scored[i].second));
        }
        const uint32_t tk =
            std::min<uint32_t>(k, truth_ids[qi].size());
        uint32_t hits = 0;
        for (uint32_t t = 0; t < tk; t++) {
            if (result_set.count(truth_ids[qi][t])) ++hits;
        }
        recall_sum += double(hits) / double(tk > 0 ? tk : 1);

        // Proximity: ratio = d_target / d_result (>=1 = at/inside band).
        // d_target = the k-th true NN distance (truth_dists[qi][k-1]).
        const float d_target =
            (truth_dists[qi].size() >= k) ? truth_dists[qi][k - 1] :
            (!truth_dists[qi].empty()) ? truth_dists[qi].back() :
            std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < topk; i++) {
            const double d = scored[i].first;
            double ratio;
            if (d_target <= kEps && d <= kEps) {
                ratio = 1.0;
            } else if (d <= kEps) {
                ratio = 2.0;  // found a closer point
            } else if (d_target <= kEps) {
                ratio = 0.0;
            } else {
                ratio = double(d_target) / d;
            }
            if (ratio >= 1.0) ++prox_in;
            ++prox_total;
        }
    }

    sq.recall = recall_sum / qidx.size();
    sq.proximity = prox_total > 0 ? double(prox_in) / prox_total : 0.0;
    return sq;
}

EstimateResult Estimator::estimate_config(VectorSource& source,
                                        const BuildConfig& overrides) {
    const auto t_ec_start = std::chrono::steady_clock::now();
    const uint64_t total_n = source.count();
    const Dim dim = source.dim();
    if (total_n == 0 || dim == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "estimate_config: empty source (n=" +
                        std::to_string(total_n) +
                        ", dim=" + std::to_string(dim) + ")");
    }

    spdlog::info("[sextant] estimate_config: n={} dim={}", total_n, dim);

    // --- 1. Sample (seek-based when N >> sample, reservoir otherwise) ---
    // sample_n = min(kEstimateSampleTarget, total_n). Use the SAME RNG seed
    // (0xC0DE1234) as pass1_sample_and_train for reproducibility. When the
    // source is a seekable file and N > 30 × sample_cap, draw the sample via
    // seek-based random sampling (O(k) I/O) instead of the full O(N) scan.
    // Threshold is measurement-driven: see pass1_sample_and_train for the
    // seek-vs-scan cost crossover analysis.
    const uint64_t sample_cap =
        std::min<uint64_t>(kEstimateSampleTarget, total_n);
    std::vector<float> sample(static_cast<size_t>(sample_cap) * dim);
    uint64_t sample_n = 0;
    uint64_t seen = total_n;

    const std::string src_path = source.path();
    constexpr uint64_t kSeekRatioThreshold = 30;
    const bool seek_eligible = !src_path.empty() &&
                               total_n > kSeekRatioThreshold * sample_cap;
    if (seek_eligible) {
        spdlog::info("[sextant] estimate_config: seek-based sampling ({} of {} "
                     "vectors; ratio {:.1f}:1)", sample_cap, total_n,
                     static_cast<double>(total_n) /
                         static_cast<double>(sample_cap));
        sample = draw_random_sample(src_path, total_n, dim, sample_cap);
        sample_n = sample_cap;
    } else {
        source.reset();
        std::mt19937_64 rng(0xC0DE1234ULL);
        Chunk chunk{};
        seen = 0;
        uint64_t filled = 0;
        while (source.next(chunk)) {
            for (uint32_t r = 0; r < chunk.count; r++) {
                const float* vec = chunk.vectors + static_cast<size_t>(r) * dim;
                if (filled < sample_cap) {
                    std::memcpy(sample.data() + filled * dim, vec, dim * sizeof(float));
                    ++filled;
                } else {
                    // rng() % (seen+1): faster than uniform_int_distribution
                    // (see pass1_sample_and_train for the bias rationale).
                    const uint64_t j = rng() % (seen + 1);
                    if (j < sample_cap) {
                        std::memcpy(sample.data() + j * dim, vec, dim * sizeof(float));
                    }
                }
                ++seen;
            }
        }
        sample_n = filled;
    }
    spdlog::info("[sextant] estimate_config: sampled {} / {} vectors",
                 sample_n, seen);

    // --- Resolve the measurement k and quality targets ---
    // target_k: the k at which recall/proximity are measured. Clamp to
    // [1, min(500, sample_n)] — can't measure recall@k > qidx size (500), and
    // at k > sample_n the truth set is too shallow for a meaningful measurement.
    uint32_t target_k = overrides.target_topk > 0 ? overrides.target_topk : 100;
    {
        const uint32_t kmax = std::min<uint32_t>(500u, static_cast<uint32_t>(sample_n));
        if (target_k > kmax) target_k = kmax;
        if (target_k < 1) target_k = 1;
    }
    spdlog::info("[sextant] estimate_config: target_k={} (measurement k for all "
                 "mini-builds)", target_k);

    // Resolve quality targets. Default: no gate (pick highest-recall config).
    // Both must be met when both are set (stricter binds). No buffer — the
    // mini-build's scale gap (20K sample overestimates recall vs full-scale)
    // is compensated by the search early-exit at runtime, not by over-building.
    double recall_thresh = (overrides.recall_target > 0.0f)
        ? static_cast<double>(overrides.recall_target) : 0.0;
    double prox_thresh = (overrides.proximity_target > 0.0f)
        ? static_cast<double>(overrides.proximity_target) : 0.0;
    // If neither set, no gate — pick the config with the highest recall/proximity.
    spdlog::info("[sextant] estimate_config: quality gate — recall{} "
                 "proximity{} (stricter binds)",
                 recall_thresh > 0.0
                     ? "≥" + std::to_string(recall_thresh) : ": off",
                 prox_thresh > 0.0
                     ? "≥" + std::to_string(prox_thresh) : ": off");

    // Lambda: does a SearchQuality measurement meet the dual gate?
    auto meets_target = [&](const SearchQuality& sq) {
        return (recall_thresh == 0.0 || sq.recall >= recall_thresh) &&
               (prox_thresh == 0.0 || sq.proximity >= prox_thresh);
    };

    // --- 2. PQ config resolution (probe if pq_m or pq_bits auto) ---
    ResolvedParams base = resolve_params(total_n, dim, overrides);
    uint16_t pq_m = overrides.pq_m;
    uint8_t pq_bits = overrides.pq_bits;
    Builder::PqSelection pq_sel{0, 0, {}, ""};
    // PQ verify candidates: filled by the probe (section 2), consumed by the
    // verify (section 3, after truth is computed). Ordered ascending m, then
    // 4-bit before 8-bit — stepping forward yields higher fidelity.
    std::vector<Builder::ProbedRow> pq_verify_candidates;
    bool pq_verify_pending = false;
    if (pq_m == 0 || pq_bits == 0) {
         // Distortion recalibration: when recall_target is set, scale the
         // distortion bound inversely with the target. The default bound (0.05)
         // achieves recall@100 ≈ 0.99 with rerank=10 at full scale. Lower
         // targets (0.90) can tolerate higher distortion (cheaper PQ); higher
         // targets (0.99+) need tighter distortion. The mapping is linear
         // and conservative — it errs toward over-provisioning PQ quality.
         // The mini-build verify is skipped (mini-build recall@100 is unreliable
         // at small sample sizes; the distortion proxy is more trustworthy).
         ResolvedParams probe_params = base;
         const bool calibrate_distortion = (overrides.recall_target > 0.0f);
         if (calibrate_distortion) {
             // At recall_target=0.95: bound = 0.05 (default — works at full scale).
             // At recall_target=0.90: bound = 0.10 (cheaper PQ acceptable).
             // At recall_target=0.99: bound = 0.02 (tighter PQ needed).
             // Linear interpolation: bound = 0.05 × (1.0 - (target - 0.95) × 3).
             // Clamp to [0.01, 0.20].
             const float t = overrides.recall_target;
             float d_bound = 0.05f * (1.0f - (t - 0.95f) * 3.0f);
             d_bound = std::clamp(d_bound, 0.01f, 0.20f);
             probe_params.pq_max_distortion = d_bound;
             spdlog::info("[sextant] estimate_config: probing PQ config "
                          "(distortion bound {:.4f} calibrated from recall_target "
                          "{:.2f})...", d_bound, t);
         } else {
             spdlog::info("[sextant] estimate_config: probing PQ config...");
         }
        pq_sel = Builder::probe_pq_config(sample.data(), sample_n, dim, probe_params);
        pq_m = pq_sel.m;
        pq_bits = pq_sel.bits;
        spdlog::info("[sextant] estimate_config: PQ probe → m={}, bits={}",
                     pq_m, static_cast<int>(pq_bits));

        // Build the verify candidate list: start from the selected config, then
        // all configs that come *after* it in the probe's `all` list (higher m,
        // or 8-bit if 4-bit was selected at the same m). The verify (section 3)
        // steps through these until one meets the recall/proximity target.
        if (!pq_sel.all.empty()) {
            const auto sel_it = std::find_if(
                pq_sel.all.begin(), pq_sel.all.end(),
                [&](const Builder::ProbedRow& r) {
                    return r.m == pq_m && r.bits == pq_bits;
                });
            if (sel_it != pq_sel.all.end()) {
                pq_verify_candidates.assign(sel_it, pq_sel.all.end());
            } else {
                pq_verify_candidates = pq_sel.all;
            }
            pq_verify_pending = true;
        }
        spdlog::info("[sextant] estimate_config: PQ → m={}, bits={} "
                     "(pending verify at k={})", pq_m,
                     static_cast<int>(pq_bits), target_k);
    }

    // --- 3. Compute truth + LID ---
    // Query indices: first min(500, sample_n) of the sample.
    const uint32_t nq = std::min<uint32_t>(500, static_cast<uint32_t>(sample_n));
    std::vector<uint32_t> qidx(nq);
    for (uint32_t i = 0; i < nq; i++) qidx[i] = i;

    spdlog::info("[sextant] estimate_config: computing truth ({} queries, "
                 "depth {})...", nq, target_k);
    const auto t_truth_start = std::chrono::steady_clock::now();
    const ProbeTruth truth =
        compute_truth(sample.data(), static_cast<uint32_t>(sample_n), dim,
                      qidx, /*truth_k=*/target_k);
    const double truth_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_truth_start).count();
    spdlog::debug("[sextant] estimate_config: truth compute took {:.3f}s "
                  "({} queries x {} pool x {} dim = {:.1f}G FLOPs)",
                  truth_sec, nq, sample_n, dim,
                  static_cast<double>(nq) * sample_n * dim / 1e9);

    // MLE LID per query: LID = [(1/(k-1)) Σ log(r_k / r_i)]^{-1}.
    // Use the first k=10 true distances.
    const auto t_lid_start = std::chrono::steady_clock::now();
    std::vector<double> lids;
    lids.reserve(nq);
    const uint32_t lid_k = std::min<uint32_t>(10, kProbeTopk);
    for (size_t qi = 0; qi < nq; qi++) {
        const auto& dists = truth.dists[qi];
        if (dists.size() < lid_k) continue;
        const float r_k = dists[lid_k - 1];
        if (r_k <= 1e-12f) continue;
        double sum = 0.0;
        for (uint32_t i = 0; i < lid_k - 1; i++) {
            if (dists[i] <= 1e-12f) { sum = 0; break; }
            sum += std::log(double(r_k) / double(dists[i]));
        }
        if (sum > 1e-9) {
            lids.push_back(double(lid_k - 1) / sum);
        }
    }
    double median_lid = 0.0;
    if (!lids.empty()) {
        std::nth_element(lids.begin(), lids.begin() + lids.size() / 2,
                         lids.end());
        median_lid = lids[lids.size() / 2];
    }
    spdlog::info("[sextant] estimate_config: median LID = {:.2f}", median_lid);
    spdlog::debug("[sextant] estimate_config: LID MLE over {} queries took {:.3f}s",
                  nq, std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t_lid_start).count());

    // d_eff: from overrides if set, else max(5.0, median_lid / 3.0).
    const double d_eff = overrides.closure_d_eff > 0.0f
                             ? overrides.closure_d_eff
                             : std::max(5.0, median_lid / 3.0);

    // --- 4. Closure factor from measured d_eff ---
    const float f_target = overrides.closure_f_target > 0.0f
                               ? overrides.closure_f_target
                               : 0.15f;
    float closure_c = static_cast<float>(
        std::pow(1.0 - f_target, -1.0 / d_eff));
    if (closure_c < 1.0f) closure_c = 1.0f;
    if (closure_c > 1.2f) closure_c = 1.2f;
    spdlog::info("[sextant] estimate_config: closure_factor={:.4f} "
                 "(f_target={:.2f}, d_eff={:.2f})",
                 closure_c, f_target, d_eff);

    // --- 4b. PQ verify (mini-build confirm of the probe selection) ---
    // SKIPPED when recall_target is set: the calibrated distortion bound
    // (section 2 above) already selected the right config, and mini-build
    // recall@100 is unreliable at small sample sizes (the graph is too sparse
    // for k=100 — the ceiling is graph-limited, not PQ-limited). When
    // recall_target is NOT set (proximity-only mode), the verify runs as a
    // diagnostic check on the probe's selection.
    if (pq_verify_pending && overrides.recall_target == 0.0f) {
        spdlog::info("[sextant] estimate_config: PQ verify (mini-build at "
                     "k={}, {} candidates)...", target_k,
                     pq_verify_candidates.size());
        ResolvedParams verify_params = base;
        verify_params.R = 64;
        verify_params.alpha = 1.2f;
        // L must exceed fetch_k = target_k × rerank for the rerank to have
        // enough candidates to pick from.
        const uint32_t verify_L =
            std::max<uint32_t>(target_k * 10u * 2u, 200u);
        const uint32_t verify_rerank = 10;
        bool met = false;
        for (const auto& cand : pq_verify_candidates) {
            verify_params.pq_m = cand.m;
            verify_params.pq_bits = cand.bits;
            auto mini = build_mini_(sample.data(), sample_n, dim, verify_params);
            const auto sq = measure_search_(
                *mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, verify_L, target_k, verify_rerank);
            const bool meets = meets_target(sq);
            const bool is_4bit = (cand.bits == 4);
            spdlog::info("[sextant]   PQ verify: m={} bits={} "
                         "recall@{}={:.4f} proximity={:.4f}{}{}",
                         cand.m, static_cast<int>(cand.bits),
                         target_k, sq.recall, sq.proximity,
                         meets ? " ✓ meets" : " ✗ below",
                         is_4bit ? " [4-bit]" : "");
            if (meets) {
                if (pq_m != cand.m || pq_bits != cand.bits) {
                    pq_m = cand.m;
                    pq_bits = cand.bits;
                    spdlog::info("[sextant] estimate_config: PQ verify stepped "
                                 "up → m={}, bits={} (probe selection didn't "
                                 "meet target)", pq_m,
                                 static_cast<int>(pq_bits));
                }
                if (is_4bit) {
                    spdlog::info("[sextant] estimate_config: 4-bit PQ selected "
                                 "(8× smaller codes → faster search) — meets "
                                 "recall@{}={:.4f}", target_k, sq.recall);
                }
                met = true;
                break;
            }
        }
        if (!met) {
            const auto& last = pq_verify_candidates.back();
            pq_m = last.m;
            pq_bits = last.bits;
            spdlog::warn("[sextant] estimate_config: no PQ config met the "
                         "target in verify; using highest-fidelity candidate "
                         "(m={}, bits={}) as best-effort",
                         pq_m, static_cast<int>(pq_bits));
        }
        spdlog::info("[sextant] estimate_config: PQ → m={}, bits={} "
                     "[verify complete]", pq_m, static_cast<int>(pq_bits));
    }

    // --- 5. Alpha resolution (sweep if alpha auto) ---
    float alpha_rec = overrides.alpha;
    if (alpha_rec == 0.0f) {
        spdlog::info("[sextant] estimate_config: sweeping alpha (k={})...",
                     target_k);
        static constexpr std::array<float, 4> kAlphas = {{1.0f, 1.1f, 1.2f, 1.5f}};
        // Production-realistic rerank and beam width. The previous values
        // (rerank=10, L=target_k*10*2=2000) probed the mini-build with a beam
        // 5× deeper than production search (L=400 at k=100/rr=2), which masked
        // alpha differences — every alpha looked equally good and the sweep
        // defaulted to the cheapest (alpha=1.0), collapsing recall in half.
        // Match the production default (rr=2 → fetch_k=200 → L=400) so the
        // sweep sees what the user will actually see.
        constexpr uint32_t kAlphaRerank = 2;
        const uint32_t kAlphaL =
            std::max<uint32_t>(target_k * kAlphaRerank * 2u, 200u);
        // Note: even at production L, the 20K mini-build can't reliably
        // discriminate alpha — small-scale navigation doesn't expose the
        // alpha<1.2 failure mode that surfaces at production N. So when no
        // explicit quality gate is set, skip the sweep and use the documented
        // default (alpha=1.2). The sweep is still useful when the user pins
        // a quality target with --recall-target or --proximity-target — in
        // that case we honor the gate even if the mini-build can't really
        // see the difference (best-effort).
        const bool gate_active = (recall_thresh > 0.0 || prox_thresh > 0.0);

        ResolvedParams alpha_params = base;
        alpha_params.R = 64;
        alpha_params.pq_m = pq_m;
        alpha_params.pq_bits = pq_bits;

        // Dual-gate alpha selection: among alphas that meet the target gate
        // (recall AND proximity, stricter binds), pick the LOWEST alpha (faster
        // build). If none meet target, pick the one with the best recall (or
        // proximity, if recall gate is off) as best-effort and warn.
        float best_meeting_alpha = 0.0f;  // lowest alpha meeting the gate
        float best_effort_alpha = kAlphas.front();
        double best_effort_score = -1.0;  // recall if recall gate, else proximity
        const bool gate_on_recall = (recall_thresh > 0.0);
        if (!gate_active) {
            // No user-supplied quality gate: the alpha sweep is uninformative
            // because the 20K mini-build can't expose production-scale alpha
            // effects. Use the documented default (1.2). Matches the
            // resolve_params auto-default and the --prune-threshold docstring.
            spdlog::info("[sextant] estimate_config: alpha → 1.2 "
                         "(no quality gate set; mini-build can't discriminate)");
            alpha_rec = 1.2f;
        } else {
        for (float a : kAlphas) {
            alpha_params.alpha = a;
            const auto t0 = std::chrono::steady_clock::now();
            auto mini = build_mini_(sample.data(), sample_n, dim, alpha_params);
            const auto sq = measure_search_(
                *mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, kAlphaL, target_k, kAlphaRerank);
            const auto t1 = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(t1 - t0).count();
            const bool meets = meets_target(sq);
            spdlog::info("[sextant]   alpha={:.1f}  recall@{}={:.4f}  "
                         "proximity={:.4f}  L={}{}  ({:.1f}s)",
                         a, target_k, sq.recall, sq.proximity, kAlphaL,
                         meets ? " ✓" : "", dt);
            if (meets && best_meeting_alpha == 0.0f) {
                best_meeting_alpha = a;  // first (lowest) meeting alpha
            }
            const double score = gate_on_recall ? sq.recall : sq.proximity;
            if (score > best_effort_score) {
                best_effort_score = score;
                best_effort_alpha = a;
            }
        }
        if (best_meeting_alpha != 0.0f) {
            alpha_rec = best_meeting_alpha;
            spdlog::info("[sextant] estimate_config: alpha → {:.1f} (lowest "
                         "alpha meeting target gate)", alpha_rec);
        } else {
            alpha_rec = best_effort_alpha;
            spdlog::warn("[sextant] estimate_config: no alpha met target gate; "
                         "selecting best-effort alpha={:.1f} ({}={:.4f})",
                         alpha_rec,
                         gate_on_recall ? "recall" : "proximity",
                         best_effort_score);
        }
        }  // end if (gate_active)
    }

    // --- 6. R resolution (OPT-SNG) ---
    // Build reference mini-index at R=64, measure graph stats.
    spdlog::info("[sextant] estimate_config: building reference mini-index "
                 "(R=64) for R prediction...");
    ResolvedParams ref_params = base;
    ref_params.R = 64;
    ref_params.alpha = alpha_rec;
    ref_params.pq_m = pq_m;
    ref_params.pq_bits = pq_bits;

    const auto t_ref0 = std::chrono::steady_clock::now();
    auto ref_mini = build_mini_(sample.data(), sample_n, dim, ref_params);
    GraphStats gstats = measure_graph_stats_(*ref_mini);
    gstats.median_lid = median_lid;
    const auto t_ref1 = std::chrono::steady_clock::now();
    spdlog::info("[sextant] estimate_config: reference graph: R̄={:.2f}, "
                 "clustering={:.4f}, dead_ends={:.4f} ({:.1f}s)",
                 gstats.avg_degree, gstats.clustering_coeff,
                 gstats.dead_end_frac,
                 std::chrono::duration<double>(t_ref1 - t_ref0).count());

    // OPT-SNG: R_predicted = R̄_mini * log(N_full)/log(N_sample) * (alpha_ref/alpha_rec)^2.
    // The reference build is at alpha_rec (whether locked or swept), so
    // alpha_ref == alpha_rec and the coupling term is 1. We write it
    // explicitly for clarity / future extension.
    const double alpha_coupling =
        (alpha_rec != 0.0f)
            ? std::pow(double(alpha_rec) / double(alpha_rec), 2.0)
            : 1.0;
    double R_predicted = gstats.avg_degree *
                         std::log(double(total_n)) / std::log(double(sample_n)) *
                         alpha_coupling;
    // Guardrails (exact thresholds from spec).
    if (gstats.clustering_coeff > 0.12) {
        R_predicted = std::min(R_predicted, 48.0);
        spdlog::info("[sextant]   guardrail: clustering {:.4f} > 0.12 → cap "
                     "R ≤ 48", gstats.clustering_coeff);
    }
    if (gstats.dead_end_frac > 0.05) {
        R_predicted = std::max(R_predicted, 64.0);
        spdlog::info("[sextant]   guardrail: dead_ends {:.4f} > 0.05 → floor "
                     "R ≥ 64", gstats.dead_end_frac);
    }

    uint16_t R_full = static_cast<uint16_t>(
        std::round(std::clamp(R_predicted, 32.0, 128.0)));
    spdlog::info("[sextant] estimate_config: R predicted={:.1f} → R_full={}",
                 R_predicted, R_full);

    if (overrides.R != 0) {
        R_full = overrides.R;
        spdlog::info("[sextant] estimate_config: R locked by override → {} "
                     "(skipping validation sweep)", R_full);
    } else {
        // --- 7. R sweep validation (only when R is estimated) ---
        // Build 2 more mini-indices at R_full-16 and R_full+16 (clamp ≥32).
        // Dual-gate: the sweep-best is the smallest R meeting the target gate
        // (cheapest). If none meet, keep the OPT-SNG prediction.
         spdlog::info("[sextant] estimate_config: R validation sweep (k={})...",
                      target_k);
         const auto t_valid_start = std::chrono::steady_clock::now();
        constexpr uint32_t kValidRerank = 10;
        // L must exceed fetch_k = k × rerank (same rationale as the alpha sweep).
        const uint32_t kValidL =
            std::max<uint32_t>(target_k * kValidRerank * 2u, 200u);

        struct RPoint { uint16_t R; double proximity; double recall; };
        std::vector<RPoint> rpoints;

        // Measure the prediction point R_full (reuse ref_mini if R_full==64).
        if (R_full == 64) {
            const auto sq = measure_search_(
                *ref_mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, kValidL, target_k, kValidRerank);
            rpoints.push_back({R_full, sq.proximity, sq.recall});
            spdlog::info("[sextant]   R={}  recall@{}={:.4f}  proximity={:.4f}{}",
                         R_full, target_k, sq.recall, sq.proximity,
                         meets_target(sq) ? " ✓" : "");
        } else {
            ResolvedParams rp = base;
            rp.R = R_full; rp.alpha = alpha_rec;
            rp.pq_m = pq_m; rp.pq_bits = pq_bits;
            auto mini = build_mini_(sample.data(), sample_n, dim, rp);
            const auto sq = measure_search_(
                *mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, kValidL, target_k, kValidRerank);
            rpoints.push_back({R_full, sq.proximity, sq.recall});
            spdlog::info("[sextant]   R={}  recall@{}={:.4f}  proximity={:.4f}{}",
                         R_full, target_k, sq.recall, sq.proximity,
                         meets_target(sq) ? " ✓" : "");
        }

        for (int delta : {-16, +16}) {
            uint16_t r = static_cast<uint16_t>(std::max(32, int(R_full) + delta));
            ResolvedParams rp = base;
            rp.R = r; rp.alpha = alpha_rec;
            rp.pq_m = pq_m; rp.pq_bits = pq_bits;
            const auto t0 = std::chrono::steady_clock::now();
            auto mini = build_mini_(sample.data(), sample_n, dim, rp);
            const auto sq = measure_search_(
                *mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, kValidL, target_k, kValidRerank);
            const auto t1 = std::chrono::steady_clock::now();
            rpoints.push_back({r, sq.proximity, sq.recall});
            spdlog::info("[sextant]   R={}  recall@{}={:.4f}  proximity={:.4f}"
                         "{}  ({:.1f}s)", r, target_k, sq.recall, sq.proximity,
                         meets_target(sq) ? " ✓" : "",
                         std::chrono::duration<double>(t1 - t0).count());
        }

        // Dual-gate selection: among R values meeting the target gate, pick the
        // smallest (cheapest). If none meet, keep the OPT-SNG prediction.
        uint16_t smallest_meeting = 0;
        for (const auto& rp : rpoints) {
            const bool meets =
                (recall_thresh == 0.0 || rp.recall >= recall_thresh) &&
                (prox_thresh == 0.0 || rp.proximity >= prox_thresh);
            if (meets && (smallest_meeting == 0 || rp.R < smallest_meeting)) {
                smallest_meeting = rp.R;
            }
        }
        if (smallest_meeting != 0 && smallest_meeting != R_full) {
            spdlog::info("[sextant]   validation: smallest R meeting target "
                         "gate = {} (prediction was {}) → using sweep-best",
                         smallest_meeting, R_full);
            R_full = smallest_meeting;
        } else if (smallest_meeting != 0) {
            spdlog::info("[sextant]   validation: R={} meets target gate "
                         "(matches prediction)", R_full);
        } else {
            spdlog::warn("[sextant]   validation: no R in sweep met target "
                         "gate; keeping OPT-SNG prediction R={}", R_full);
        }
         spdlog::debug("[sextant] estimate_config: R validation sweep took {:.2f}s",
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t_valid_start).count());
     }

    // --- 8. Final param resolution ---
    ResolvedParams p;
    p.R = R_full;
    p.alpha = alpha_rec;
    p.pq_m = pq_m;
    p.pq_bits = pq_bits;
    // Honor an explicit L override (the T5 dynamic-ramp ceiling). Mirror
    // resolve_params's behavior: overrides.L != 0 locks L_build directly;
    // otherwise L_build = max(2*R, 100). Previously this was silently ignored
    // — an inconsistency between the heuristic and the estimation paths.
    if (overrides.L != 0) {
        p.L_build = overrides.L;
    } else {
        p.L_build = static_cast<uint16_t>(std::max<uint32_t>(2u * p.R, 100u));
    }
    p.L = p.L_build;
    p.max_occlusion = (overrides.max_occlusion != 0)
                           ? overrides.max_occlusion
                           : std::max<uint32_t>(p.L_build,
                                                static_cast<uint32_t>(p.R) + 1u);
    p.closure_factor = closure_c;
    p.metric = overrides.metric;
    p.num_threads = base.num_threads;
    p.build_ram_budget = base.build_ram_budget;
    p.pq_max_distortion = base.pq_max_distortion;

    // Recompute K from the final R (K depends on per-vec size).
    {
        const uint32_t code_sz = static_cast<uint32_t>(p.pq_m);
        const uint32_t node_sz =
            ((16u + static_cast<uint32_t>(p.R) * 4u + 7u) & ~7u);
        const uint64_t per_vec = code_sz + node_sz;
        uint32_t k = 1;
        uint64_t max_per_partition = 0;
        if (per_vec > 0 && p.build_ram_budget > 0) {
            max_per_partition = p.build_ram_budget / per_vec;
            if (max_per_partition > 0) {
                k = static_cast<uint32_t>(
                    (total_n + max_per_partition - 1) / max_per_partition);
                if (k < 1) k = 1;
            }
        }
        p.partition_count = k;
    }

    spdlog::info("[sextant] estimate_config: final → R={} alpha={:.1f} "
                 "pq_m={} pq_bits={} L_build={} K={}",
                 p.R, p.alpha, p.pq_m, static_cast<int>(p.pq_bits), p.L_build,
                  p.partition_count);

    spdlog::info("[sextant] estimate_config: total time {:.1f}s",
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_ec_start).count());
    return EstimateResult{
        std::move(p),
        EstimationDiagnostics{
            /*median_lid=*/median_lid,
            /*avg_degree=*/gstats.avg_degree,
            /*clustering_coeff=*/gstats.clustering_coeff,
            /*dead_end_frac=*/gstats.dead_end_frac,
        },
    };
}

}  // namespace sextant
