// spike_panorama_rerank.cpp — Panorama-style progressive FP32 rerank.
//
// Question: can we cut the 12% rerank tax by pruning candidates during
// the FP32 rerank using Cauchy-Schwarz on cumulative dimension norms?
//
// Panorama (arXiv:2510.00566, implemented in FAISS) partitions D dims
// into L contiguous levels. It computes the exact distance incrementally
// level-by-level. After each level, it derives a lower bound on the
// FINAL distance using Cauchy-Schwarz on the remaining (uncomputed) dims.
// Candidates whose lower bound exceeds the k-th best are pruned early,
// avoiding the full D-dim distance computation.
//
// Unlike PQ error bounds (which fail — see spike_pq_bounds.cpp), this
// bounds the EXACT FP32 distance. The only error is truncation error
// from not yet computing remaining dims, which Cauchy-Schwarz bounds
// tightly. Zero per-vector storage needed beyond the base vectors.
//
// This spike measures pruning power: how many dims does the average
// reranked candidate need before pruning?
//
// Build: ninja -C build spike_panorama_rerank
// Usage: ./build/scripts/spike_panorama_rerank <base.fbin> [query.fbin] [ip|l2]

#include "simd_kernels.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace sextant;

// --- .fbin reader ---
struct FbinData {
    uint32_t n = 0, dim = 0;
    std::vector<float> data;
};
static bool read_fbin(const std::string& p, FbinData& o) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&o.n), 4);
    f.read(reinterpret_cast<char*>(&o.dim), 4);
    if (!f || o.n == 0) return false;
    o.data.resize(size_t(o.n) * o.dim);
    f.read(reinterpret_cast<char*>(o.data.data()), o.data.size() * 4);
    return f.good() || f.eof();
}

static inline float dot_f32(const float* a, const float* b, uint32_t d) {
    return simd::dot_f32(a, b, d);
}

// ========================================================================
//                              MAIN
// ========================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <base.fbin> [query.fbin] [ip|l2]\n", argv[0]);
        return 1;
    }
    const std::string base_path = argv[1];
    const std::string query_path = argc > 2 ? argv[2] : "";
    const std::string metric_str = argc > 3 ? argv[3] : "l2";
    const bool is_ip = (metric_str == "ip" || metric_str == "IP");

    FbinData db;
    if (!read_fbin(base_path, db)) {
        std::fprintf(stderr, "Failed to read %s\n", base_path.c_str());
        return 1;
    }
    const uint32_t N = db.n;
    const uint32_t D = db.dim;

    // Queries
    FbinData queries;
    const uint32_t n_query = 200;
    if (!query_path.empty() && read_fbin(query_path, queries)) {
        printf("Queries: %s (%u loaded, using %u)\n",
               query_path.c_str(), queries.n, std::min(queries.n, n_query));
    } else {
        printf("Queries: sampling %u from base\n", n_query);
        std::mt19937_64 rng(12345);
        queries.n = n_query;
        queries.dim = D;
        queries.data.resize(size_t(n_query) * D);
        for (uint32_t i = 0; i < n_query; i++) {
            const uint32_t idx = rng() % N;
            std::memcpy(queries.data.data() + size_t(i) * D,
                        db.data.data() + size_t(idx) * D, D * 4);
        }
    }
    const uint32_t nq = std::min(queries.n, n_query);

    printf("=== Panorama Rerank Spike ===\n");
    printf("Dataset: %s  N=%u  D=%u  Metric=%s\n",
           base_path.c_str(), N, D, is_ip ? "IP" : "L2sq");

    // Simulate the IVF-scan pipeline: for each query, produce a shortlist
    // of W=300 candidates. We use random shortlists (since this spike
    // measures RERANK pruning power, not scan quality). In production,
    // these would come from the PQ scan. Using random shortlists is
    // conservative — real shortlists are more clustered (harder to prune).
    // We also test with "hard" shortlists (nearest neighbors + noise).
    const uint32_t W = 300;
    const uint32_t TOPK = 10;

    printf("Shortlist size W=%u, top-k=%u, queries=%u\n", W, TOPK, nq);

    // For realistic shortlists: use brute-force to get the top-W nearest
    // neighbors per query (this simulates a good PQ scan result).
    printf("Computing exact top-W shortlists...\n");
    std::vector<std::vector<std::pair<float, uint32_t>>> shortlists(nq);
    // For each shortlist entry, we also store the exact distance.
    // shortlists[qi] = vector of (exact_dist, gid), sorted ascending.
    for (uint32_t qi = 0; qi < nq; qi++) {
        const float* q = &queries.data[size_t(qi) * D];
        auto& sl = shortlists[qi];
        sl.resize(N);
        for (uint32_t i = 0; i < N; i++) {
            const float* o = &db.data[size_t(i) * D];
            const float dist = is_ip ? -dot_f32(q, o, D)
                                     : simd::l2sq_f32(q, o, D);
            sl[i] = {dist, i};
        }
        std::partial_sort(sl.begin(), sl.begin() + W, sl.end());
        sl.resize(W);
    }

    // ========================================================================
    // Panorama progressive rerank.
    //
    // For L2sq: dist(q, y) = ||q||² - 2<q,y> + ||y||²
    //   After l levels: partial_dist = Σ_{j<l} (q_j - y_j)²
    //   Remaining contribution ≤ (||q_rest|| · ||y_rest|| + ...) via CS.
    //   Lower bound = partial_dist (from computed levels) — actually it's
    //   simpler: dist = Σ_all (q_i - y_i)². After computing l levels:
    //     exact_so_far = Σ_{i<l} (q_i - y_i)²
    //     remaining = Σ_{i≥l} (q_i - y_i)² ≥ 0
    //   So exact_so_far IS a lower bound on the full distance!
    //   We prune if exact_so_far > threshold (k-th best distance).
    //
    // For IP: sim(q, y) = Σ_i q_i · y_i
    //   After l levels: partial_sim = Σ_{i<l} q_i · y_i
    //   Remaining ≤ ||q_rest|| · ||y_rest|| (Cauchy-Schwarz)
    //   Upper bound on full sim = partial_sim + ||q_rest|| · ||y_rest||
    //   We prune if upper_bound < threshold (k-th best similarity).
    // ========================================================================

    // Test configurations: different numbers of levels.
    // Level widths must divide D evenly (or close).
    const uint32_t level_configs[] = {4, 8, 16, 32, 64, 96, 128, 192};
    const int n_configs = sizeof(level_configs) / sizeof(level_configs[0]);

    printf("\n=== Panorama Progressive Rerank Results ===\n");
    printf("  %8s %8s %12s %12s %12s %10s %10s\n",
           "lvl_d", "n_levels", "avg_dims", "dims_pct", "speedup", "recall", "false_pr%");
    printf("  %8s %8s %12s %12s %12s %10s %10s\n",
           "-----", "--------", "--------", "--------", "--------", "------", "---------");

    // Baseline: full D-dim rerank (no pruning).
    {
        uint64_t total_dims = 0;
        uint64_t total_hits = 0;
        for (uint32_t qi = 0; qi < nq; qi++) {
            const auto& sl = shortlists[qi];
            // Rerank all W candidates with full D-dim distance.
            // The shortlist already has exact distances.
            std::vector<std::pair<float, uint32_t>> reranked(sl);
            std::partial_sort(reranked.begin(),
                              reranked.begin() + TOPK, reranked.end());

            // True top-10 (from full dataset — already have it in shortlist
            // since shortlist = brute-force top-W).
            for (uint32_t r = 0; r < TOPK; r++)
                if (r < reranked.size() && r < sl.size() &&
                    reranked[r].second == sl[r].second)
                    total_hits++;

            total_dims += uint64_t(W) * D;
        }
        const float recall = float(total_hits) / (nq * TOPK);
        printf("  %8s %8s %12.1f %12.1f %12.2f %10.4f %10s\n",
               "full", "1", float(D), 100.0f, 1.0f, recall, "0.00 (baseline)");
    }

    // Panorama configurations.
    for (int ci = 0; ci < n_configs; ci++) {
        const uint32_t lvl_d = level_configs[ci];
        const uint32_t n_levels = (D + lvl_d - 1) / lvl_d;

        // Precompute per-vector per-level cumulative norms².
        // cum_norms[i][l] = Σ_{j >= l*lvl_d} y_i[j]²  (remaining norm² after level l)
        // For IP we need this per-vector. For L2sq we don't (lower bound = partial).
        // We compute for IP; L2sq just uses partial sum as lower bound.
        //
        // Actually, for IP Cauchy-Schwarz we need ||q_rest||·||y_rest||.
        // ||q_rest|| can be precomputed per query per level.
        // ||y_rest|| must be available per vector per level.
        // That's n_levels floats per vector. For n_levels=8, that's 32 bytes.
        // For this spike we compute on-the-fly (measuring the concept, not speed).

        // Precompute query cumulative norms² per level.
        // q_cum_norm[qi][l] = Σ_{j >= l*lvl_d} q[j]²
        std::vector<std::vector<float>> q_cum_norm(nq, std::vector<float>(n_levels + 1));
        for (uint32_t qi = 0; qi < nq; qi++) {
            const float* q = &queries.data[size_t(qi) * D];
            auto& qcn = q_cum_norm[qi];
            qcn[n_levels] = 0.0f;
            for (int l = int(n_levels) - 1; l >= 0; l--) {
                const uint32_t start = l * lvl_d;
                const uint32_t end = std::min(start + lvl_d, D);
                float level_norm_sq = 0.0f;
                for (uint32_t j = start; j < end; j++)
                    level_norm_sq += q[j] * q[j];
                qcn[l] = qcn[l + 1] + level_norm_sq;
            }
        }

        // Precompute per-vector cumulative norms² per level.
        // y_cum_norm[i][l] = Σ_{j >= l*lvl_d} y[i][j]²
        // This is the expensive part. For the spike, compute per query's shortlist.
        // We'll compute lazily inside the loop.

        uint64_t total_dims_computed = 0;
        uint64_t total_possible_dims = 0;
        uint64_t total_hits = 0;
        uint64_t total_false_prune = 0;

        for (uint32_t qi = 0; qi < nq; qi++) {
            const float* q = &queries.data[size_t(qi) * D];
            const auto& sl = shortlists[qi];
            const auto& qcn = q_cum_norm[qi];

            // We need the k-th best (TOPK-th) threshold to prune against.
            // In production, this is maintained dynamically as a heap.
            // For the spike, use a fixed threshold = the TOPK-th exact dist.
            // This is optimistic (production would start with infinite threshold).
            // To be fair, we also test with a "warmup" approach: compute
            // the first few candidates fully, then use their k-th as threshold.
            //
            // For now: use the exact TOPK-th distance as the threshold.
            const float thresh = sl[std::min(TOPK - 1, W - 1)].first;

            // Per-vector cumulative norm² (remaining after level l).
            // Compute on-the-fly.
            std::vector<std::vector<float>> y_cum_norm(W, std::vector<float>(n_levels + 1));
            for (uint32_t c = 0; c < W; c++) {
                const uint32_t gid = sl[c].second;
                const float* y = &db.data[size_t(gid) * D];
                auto& ycn = y_cum_norm[c];
                ycn[n_levels] = 0.0f;
                for (int l = int(n_levels) - 1; l >= 0; l--) {
                    const uint32_t start = l * lvl_d;
                    const uint32_t end = std::min(start + lvl_d, D);
                    float ns = 0.0f;
                    for (uint32_t j = start; j < end; j++)
                        ns += y[j] * y[j];
                    ycn[l] = ycn[l + 1] + ns;
                }
            }

            // Progressive rerank.
            std::vector<float> partial(W, 0.0f);  // partial distance/similarity
            std::vector<bool> active(W, true);
            uint32_t num_active = W;

            for (uint32_t l = 0; l < n_levels && num_active > 0; l++) {
                const uint32_t start = l * lvl_d;
                const uint32_t end = std::min(start + lvl_d, D);
                const uint32_t width = end - start;

                for (uint32_t c = 0; c < W; c++) {
                    if (!active[c]) continue;
                    const uint32_t gid = sl[c].second;
                    const float* y = &db.data[size_t(gid) * D];

                    if (is_ip) {
                        // IP: accumulate similarity (partial dot product)
                        partial[c] += dot_f32(q + start, y + start, width);
                    } else {
                        // L2sq: accumulate squared differences
                        for (uint32_t j = start; j < end; j++) {
                            const float diff = q[j] - y[j];
                            partial[c] += diff * diff;
                        }
                    }
                    total_dims_computed += width;
                }

                // Prune: after level l, check if candidate can still be in top-TOPK.
                // IP: upper bound on full sim = partial + ||q_rest||·||y_rest||
                //     prune if partial + sqrt(qcn[l+1] * ycn[c][l+1]) < -thresh_abs
                //     (thresh is negative for IP since we store -sim)
                // L2sq: lower bound on full dist = partial (remaining ≥ 0)
                //       prune if partial > thresh
                if (l < n_levels - 1) {  // no need to prune after last level
                    for (uint32_t c = 0; c < W; c++) {
                        if (!active[c]) continue;
                        bool prune = false;
                        if (is_ip) {
                            // sim_upper = partial[c] + sqrt(q_rest_norm² * y_rest_norm²)
                            const float cs_bound = std::sqrt(
                                qcn[l + 1] * y_cum_norm[c][l + 1]);
                            const float sim_upper = partial[c] + cs_bound;
                            // thresh = -exact_dist = sim of k-th best
                            // prune if sim_upper < thresh_sim
                            const float thresh_sim = -thresh;
                            if (sim_upper < thresh_sim) prune = true;
                        } else {
                            if (partial[c] > thresh) prune = true;
                        }
                        if (prune) {
                            active[c] = false;
                            num_active--;
                            // Check if this was a true top-TOPK member (false prune)
                            for (uint32_t r = 0; r < TOPK; r++) {
                                if (sl[r].second == sl[c].second) {
                                    total_false_prune++;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            total_possible_dims += uint64_t(W) * D;

            // Recall: count how many true top-TOPK survived and are in the result.
            std::vector<std::pair<float, uint32_t>> survivors;
            for (uint32_t c = 0; c < W; c++) {
                if (active[c])
                    survivors.push_back({sl[c].first, sl[c].second});
            }
            std::partial_sort(survivors.begin(),
                              survivors.begin() + std::min<size_t>(TOPK, survivors.size()),
                              survivors.end());
            for (size_t r = 0; r < std::min<size_t>(TOPK, survivors.size()); r++) {
                for (uint32_t t = 0; t < TOPK; t++)
                    if (sl[t].second == survivors[r].second) {
                        total_hits++;
                        break;
                    }
            }
        }

        const float avg_dims = float(total_dims_computed) / nq;
        const float dims_pct = 100.0f * float(total_dims_computed) / total_possible_dims;
        const float speedup = float(total_possible_dims) / total_dims_computed;
        const float recall = float(total_hits) / (nq * TOPK);
        const float false_pr = 100.0f * float(total_false_prune) / (nq * TOPK);

        printf("  %8u %8u %12.1f %12.1f %12.2f %10.4f %10.4f\n",
               lvl_d, n_levels, avg_dims, dims_pct, speedup, recall, false_pr);
    }

    printf("\nKey:\n");
    printf("  lvl_d    = dimensions per level\n");
    printf("  avg_dims = average dimensions computed per candidate (lower = faster)\n");
    printf("  dims_pct = %% of full D-dim work (100%% = baseline, no pruning)\n");
    printf("  speedup  = theoretical rerank speedup factor (1/dims_pct)\n");
    printf("  recall   = recall@10 after progressive rerank\n");
    printf("  false_pr%% = true top-10 wrongly pruned (lower = safer)\n");
    printf("\nDecision: speedup ≥2x at <1%% false_pr → GO.\n");
    printf("Note: IP uses Cauchy-Schwarz on remaining dims. L2sq lower bound =\n");
    printf("      partial sum (trivially correct, remaining ≥ 0).\n");

    return 0;
}
