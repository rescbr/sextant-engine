// spike_pq_bounds.cpp — PQ error bounds for selective rerank.
//
// Question: can a statistical error bound on PQ distance estimates enable
// selective rerank (shrink W=300 shortlist) where E1's fixed-margin
// heuristic failed?
//
// Bound derivation:
//   c = PQ_decode(code(o))     (reconstruction)
//   r = o - c                  (residual, unknown per-vector)
//
//   IP:  true_IP(q,o) = <q,c> + <q,r>
//        <q,r> concentrates as N(0, σ_r²·||q||²) if residuals near-isotropic
//        upper_bound = <q,c> + z·σ_r·||q||       [statistical, 0 bytes/vec]
//        upper_bound = <q,c> + ||q||·||r||        [Cauchy-Schwarz, +4 B/vec]
//
//   L2sq: true_L2(q,o) = est_L2 - 2<q-c,r> + ||r||²
//         lower_bound = est_L2 + ||r||² - 2||q-c||·||r||  [needs ||r|| per vec]
//
// Experiments:
//   1. Isotropy: does <q,r> ~ N(0, σ²||q||²)?
//   2. Shortlist pruning: can W shrink at ≤1% recall loss?
//   3. Sidecar (||r|| per vec) vs free statistical bound.
//   4. E1 margin comparison (reproduce + explain).
//
// Build: ninja -C build spike_pq_bounds
// Usage: ./build/scripts/spike_pq_bounds <base.fbin> [query.fbin] [metric] [m]
//   metric: ip or l2 (default: l2)
//   m: number of PQ subspaces (default: 96 for d=768, 32 for d=128)

#include "quant/pq_quantizer.hpp"
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

// --- .fbin reader (same as spike_margin_rerank / spike_pq4_recall) ---
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

// --- SIMD helpers ---
static inline float dot_f32(const float* a, const float* b, uint32_t d) {
    return simd::dot_f32(a, b, d);
}
static inline float l2sq_f32(const float* a, const float* b, uint32_t d) {
    return simd::l2sq_f32(a, b, d);
}

// --- Normal CDF tail: P(|Z| > z) = 2*(1 - Φ(z)) ---
static double normal_two_sided(double z) {
    // Abramowitz-Stegun 7.1.26 erf approximation → Φ
    // Φ(z) = 0.5 * (1 + erf(z/√2))
    // erf(x) ≈ 1 - (a1 t + a2 t² + ... + a5 t⁵) e^{-x²}, t = 1/(1+p x)
    auto erf_approx = [](double x) -> double {
        const double p = 0.3275911;
        const double a1 = 0.254829592, a2 = -0.284496736,
                     a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
        const int sign = x < 0 ? -1 : 1;
        x = std::abs(x);
        const double t = 1.0 / (1.0 + p * x);
        const double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                                std::exp(-x * x);
        return sign * y;
    };
    const double phi = 0.5 * (1.0 + erf_approx(z / std::sqrt(2.0)));
    return 2.0 * (1.0 - phi);
}

// ========================================================================
//                              MAIN
// ========================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <base.fbin> [query.fbin] [ip|l2] [m]\n", argv[0]);
        return 1;
    }
    const std::string base_path = argv[1];
    const std::string query_path = argc > 2 ? argv[2] : "";
    const std::string metric_str = argc > 3 ? argv[3] : "l2";
    const MetricKind metric = (metric_str == "ip" || metric_str == "IP")
                                  ? MetricKind::InnerProduct
                                  : MetricKind::L2Sq;

    FbinData db;
    if (!read_fbin(base_path, db)) {
        std::fprintf(stderr, "Failed to read %s\n", base_path.c_str());
        return 1;
    }
    const uint32_t N = db.n;
    const uint32_t D = db.dim;

    // Default m: 96 for 768-d, 32 for 128-d
    uint32_t m = argc > 4 ? std::atoi(argv[4]) : (D == 768 ? 96 : 32);
    while (D % m != 0) m--;
    const uint32_t bits = 4;
    const uint32_t K = 1u << bits;

    printf("=== PQ Bounds Spike ===\n");
    printf("Dataset: %s  N=%u  D=%u\n", base_path.c_str(), N, D);
    printf("Metric: %s  M=%u  K=%u  bits=%u  sub_dim=%u\n",
           metric == MetricKind::InnerProduct ? "IP" : "L2sq",
           m, K, bits, D / m);

    // --- Load queries (or sample from base) ---
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

    // --- Train PQ on a sample ---
    const uint32_t train_n = std::min<uint32_t>(20000, N);
    printf("Training PQ on %u vectors...\n", train_n);
    // PqQuantizer train samples from the provided array directly.
    PqQuantizer pq(metric, D, static_cast<uint16_t>(m),
                   static_cast<uint8_t>(bits));
    auto t0 = std::chrono::steady_clock::now();
    pq.train(db.data.data(), train_n);
    auto t1 = std::chrono::steady_clock::now();
    printf("  PQ trained in %.2fs\n",
           std::chrono::duration<double>(t1 - t0).count());

    const uint32_t code_sz = pq.code_size();
    const uint32_t sub_dim = pq.sub_dim();

    // --- Encode all vectors ---
    printf("Encoding all %u vectors...\n", N);
    std::vector<uint8_t> codes(size_t(N) * code_sz);
    t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < N; i++)
        pq.encode(&db.data[size_t(i) * D], &codes[size_t(i) * code_sz]);
    t1 = std::chrono::steady_clock::now();
    printf("  Encoded in %.2fs\n", std::chrono::duration<double>(t1 - t0).count());

    // --- Decode all vectors → reconstructed c[i] ---
    printf("Decoding all vectors (for residual computation)...\n");
    std::vector<float> reconstructions(size_t(N) * D);
    t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < N; i++)
        pq.decode_code(&codes[size_t(i) * code_sz],
                       &reconstructions[size_t(i) * D]);
    t1 = std::chrono::steady_clock::now();
    printf("  Decoded in %.2fs\n", std::chrono::duration<double>(t1 - t0).count());

    // --- Compute per-vector residual norm² ||r[i]||² ---
    //   Also compute per-subspace σ_r: the RMS of ||r_s|| over all vectors
    //   (σ_r² = E[||r_s||²] per subspace s).
    printf("Computing residuals...\n");
    std::vector<float> r_norm_sq(N);  // total residual norm² per vector
    std::vector<double> sigma_r_sq(m, 0.0);  // per-subspace E[||r_s||²]
    t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < N; i++) {
        const float* o = &db.data[size_t(i) * D];
        const float* c = &reconstructions[size_t(i) * D];
        float total = 0.0f;
        for (uint32_t s = 0; s < m; s++) {
            float ss = 0.0f;
            for (uint32_t j = 0; j < sub_dim; j++) {
                const float diff = o[s * sub_dim + j] - c[s * sub_dim + j];
                ss += diff * diff;
            }
            sigma_r_sq[s] += ss;
            total += ss;
        }
        r_norm_sq[i] = total;
    }
    for (uint32_t s = 0; s < m; s++)
        sigma_r_sq[s] /= double(N);
    t1 = std::chrono::steady_clock::now();
    printf("  Residuals computed in %.2fs\n",
           std::chrono::duration<double>(t1 - t0).count());

    // Global aggregate σ_r (total over all subspaces)
    double sigma_r_total_sq = 0.0;
    for (uint32_t s = 0; s < m; s++) sigma_r_total_sq += sigma_r_sq[s];
    const double sigma_r_total = std::sqrt(sigma_r_total_sq);
    printf("  Mean ||r||² = %.4f  σ_r(total) = %.4f\n",
           std::accumulate(r_norm_sq.begin(), r_norm_sq.end(), 0.0) / N,
           sigma_r_total);

    // Per-subspace σ_r stats
    {
        double mn = sigma_r_sq[0], mx = sigma_r_sq[0], mean = 0;
        for (uint32_t s = 0; s < m; s++) {
            mn = std::min(mn, sigma_r_sq[s]);
            mx = std::max(mx, sigma_r_sq[s]);
            mean += sigma_r_sq[s];
        }
        mean /= m;
        printf("  Per-subspace σ_r² range: [%.4f, %.4f]  mean=%.4f\n", mn, mx, mean);
    }

    // ========================================================================
    // EXPERIMENT 1: Residual isotropy
    // Does <q, r> concentrate as N(0, σ²_r · ||q||²)?
    // ========================================================================
    printf("\n=== Experiment 1: Residual Isotropy ===\n");
    printf("Testing if <q,r> ~ N(0, σ²_r·||q||²) for %u random queries\n", nq);
    printf("  %-6s  %12s  %12s  %8s\n", "z", "empirical_α", "theory_α", "ratio");
    {
        std::mt19937_64 rng(999);
        const double z_vals[] = {1.0, 2.0, 3.0, 4.0};
        const int nz = 4;
        uint64_t exceed_count[nz] = {};
        uint64_t total_count = 0;

        for (uint32_t qi = 0; qi < nq; qi++) {
            const float* q = &queries.data[size_t(qi) * D];
            const float q_norm = std::sqrt(dot_f32(q, q, D));
            if (q_norm < 1e-10f) continue;
            // Threshold for z: |<q,r>| > z · σ_r · ||q||
            // = |Σ_s <q_s, r_s>| > z · σ_r_total · ||q||
            // We need σ_r_total = sqrt(Σ_s E[||r_s||²]) — but for the dot
            // product <q,r>, the variance is Σ_s σ²_{r,s} · ||q_s||² (weighted).
            // For simplicity, compute the per-query σ:
            //   σ_q = sqrt(Σ_s σ²_{r,s} · ||q_s||² / D_sub)... but that's per-dim.
            // Actually: Var[<q,r>] = Σ_s ||q_s||² · σ²_{r,s}
            // because <q_s, r_s> ~ (0, ||q_s||² · σ²_{r,s}) if r_s ~ isotropic.
            // Wait — σ²_{r,s} = E[||r_s||²] is the variance of the NORM, not per-dim.
            // Per-dim variance: σ²_{r,s} / sub_dim (isotropy assumption).
            // Var[<q_s, r_s>] = Σ_j q_{s,j}² · σ²_{r,s}/sub_dim
            //                  = ||q_s||² · σ²_{r,s} / sub_dim
            // Total: Var[<q,r>] = Σ_s ||q_s||² · σ²_{r,s} / sub_dim
            double var_qr = 0.0;
            for (uint32_t s = 0; s < m; s++) {
                const float* qs = q + s * sub_dim;
                const double qs_norm_sq = dot_f32(qs, qs, sub_dim);
                var_qr += qs_norm_sq * sigma_r_sq[s] / sub_dim;
            }
            const double sigma_qr = std::sqrt(var_qr);
            if (sigma_qr < 1e-12) continue;

            for (uint32_t i = 0; i < N; i += (N > 50000 ? 3 : 1)) {
                const float* o = &db.data[size_t(i) * D];
                const float* c = &reconstructions[size_t(i) * D];
                // <q, r> = <q, o-c> = <q,o> - <q,c>
                const float qr = dot_f32(q, o, D) - dot_f32(q, c, D);
                total_count++;
                for (int zi = 0; zi < nz; zi++) {
                    if (std::abs(qr) > z_vals[zi] * sigma_qr)
                        exceed_count[zi]++;
                }
            }
        }
        for (int zi = 0; zi < nz; zi++) {
            const double emp = double(exceed_count[zi]) / total_count;
            const double th = normal_two_sided(z_vals[zi]);
            printf("  %-6.1f  %12.6f  %12.6f  %8.2f\n",
                   z_vals[zi], emp, th, emp / (th + 1e-15));
        }
        printf("  (ratio ~1.0 means isotropy holds → statistical bound valid)\n");
    }

    // ========================================================================
    // EXPERIMENT 2 + 3 + 4: Shortlist pruning power
    // ========================================================================
    printf("\n=== Experiment 2-4: Shortlist Pruning ===\n");
    printf("Computing exact top-10 for %u queries...\n", nq);

    // For each query: exact distances, PQ estimates, residual norms.
    const uint32_t TOPK = 10;
    const uint32_t W = 300;

    // Precompute PQ LUT for each query once.
    std::vector<float> lut(pq.lut_size());

    printf("  %-8s %-8s %8s %8s %8s %10s\n",
           "method", "param", "prune%", "recall", "eff_W", "false_pr%");
    printf("  %-8s %-8s %8s %8s %8s %10s\n",
           "------", "------", "------", "------", "------", "---------");

    // Collect per-query results for analysis.
    // For each query we store: all candidates sorted by PQ distance.
    // Then we apply different pruning rules and measure recall.

    struct QueryResult {
        std::vector<std::pair<float, uint32_t>> pq_ranked;  // (pq_dist, gid)
        std::vector<float> exact_dists;                      // exact dist per gid (indexed by gid)
    };
    std::vector<QueryResult> qrs(nq);

    for (uint32_t qi = 0; qi < nq; qi++) {
        const float* q = &queries.data[size_t(qi) * D];
        pq.preprocess_query(q, lut.data());

        auto& qr = qrs[qi];
        qr.pq_ranked.resize(N);
        qr.exact_dists.resize(N);

        for (uint32_t i = 0; i < N; i++) {
            const float pq_dist = pq.lut_distance(
                &codes[size_t(i) * code_sz], lut.data());
            qr.pq_ranked[i] = {pq_dist, i};

            const float* o = &db.data[size_t(i) * D];
            if (metric == MetricKind::InnerProduct) {
                // For IP: similarity = dot product (higher = better).
                // Store as negative so sorting ascending gives best first.
                qr.exact_dists[i] = -dot_f32(q, o, D);
            } else {
                qr.exact_dists[i] = l2sq_f32(q, o, D);
            }
        }
        std::sort(qr.pq_ranked.begin(), qr.pq_ranked.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    printf("  PQ scan + exact distances computed.\n\n");

    // Helper: evaluate a pruning rule that returns a set of gids to rerank.
    // Measures recall@TOPK and effective W.
    auto evaluate = [&](const char* method, const char* param,
                        auto get_gids) {
        uint64_t total_rerank = 0;
        uint64_t total_hits = 0;
        uint64_t total_false_prune = 0;
        uint64_t total_shortlist = 0;

        for (uint32_t qi = 0; qi < nq; qi++) {
            const auto& qr = qrs[qi];
            const uint32_t Weff = std::min<uint32_t>(W, (uint32_t)qr.pq_ranked.size());

            // Determine the true top-TOPK set
            std::vector<std::pair<float, uint32_t>> exact_all;
            exact_all.reserve(Weff);
            for (uint32_t r = 0; r < Weff; r++) {
                const uint32_t gid = qr.pq_ranked[r].second;
                exact_all.push_back({qr.exact_dists[gid], gid});
            }
            std::partial_sort(exact_all.begin(),
                              exact_all.begin() + std::min<size_t>(TOPK, exact_all.size()),
                              exact_all.end());

            // True top-TOPK gids (from the ENTIRE dataset, not just shortlist)
            std::vector<std::pair<float, uint32_t>> global_exact(N);
            for (uint32_t i = 0; i < N; i++)
                global_exact[i] = {qr.exact_dists[i], i};
            std::partial_sort(global_exact.begin(),
                              global_exact.begin() + TOPK,
                              global_exact.end());
            std::vector<uint32_t> true_topk(TOPK);
            for (uint32_t r = 0; r < TOPK; r++)
                true_topk[r] = global_exact[r].second;

            // Apply pruning rule → get gids to rerank
            auto gids = get_gids(qi, qr, Weff);
            total_rerank += gids.size();
            total_shortlist += Weff;

            // Count how many of true_topk were pruned (false prunes)
            std::sort(gids.begin(), gids.end());
            for (uint32_t r = 0; r < TOPK; r++) {
                if (!std::binary_search(gids.begin(), gids.end(),
                                        true_topk[r])) {
                    // Was this gid in the shortlist at all?
                    bool in_shortlist = false;
                    for (uint32_t j = 0; j < Weff; j++) {
                        if (qr.pq_ranked[j].second == true_topk[r]) {
                            in_shortlist = true;
                            break;
                        }
                    }
                    if (in_shortlist) total_false_prune++;
                }
            }

            // Rerank: take the top-TOPK by exact distance from gids
            std::vector<std::pair<float, uint32_t>> reranked;
            reranked.reserve(gids.size());
            for (uint32_t gid : gids)
                reranked.push_back({qr.exact_dists[gid], gid});
            std::partial_sort(reranked.begin(),
                              reranked.begin() + std::min<size_t>(TOPK, reranked.size()),
                              reranked.end());

            // Count hits
            for (size_t r = 0; r < std::min<size_t>(TOPK, reranked.size()); r++) {
                for (uint32_t t = 0; t < TOPK; t++)
                    if (true_topk[t] == reranked[r].second) {
                        total_hits++;
                        break;
                    }
            }
        }

        const float recall = float(total_hits) / (nq * TOPK);
        const float avg_rerank = float(total_rerank) / nq;
        const float avg_shortlist = float(total_shortlist) / nq;
        const float prune_pct = 100.0f * (1.0f - avg_rerank / avg_shortlist);
        const float false_prune_pct = 100.0f * float(total_false_prune) /
                                      (nq * TOPK);
        printf("  %-8s %-8s %8.1f %8.4f %8.1f %10.4f\n",
               method, param, prune_pct, recall, avg_rerank, false_prune_pct);
        return recall;
    };

    // --- Debug: trace values for first query ---
    if (nq > 0) {
        const auto& qr0 = qrs[0];
        printf("\n  [DEBUG] Query 0: est[0]=%.4f est[9]=%.4f est[299]=%.4f\n",
               qr0.pq_ranked[0].first, qr0.pq_ranked[9].first,
               qr0.pq_ranked[std::min<uint32_t>(299u, N-1)].first);
        // Print exact dists for top-10 by PQ
        printf("  [DEBUG]   PQ-best gid=%u: pq_est=%.4f exact=%.4f r_norm²=%.4f\n",
               qr0.pq_ranked[0].second, qr0.pq_ranked[0].first,
               qr0.exact_dists[qr0.pq_ranked[0].second],
               r_norm_sq[qr0.pq_ranked[0].second]);
        // Exact top-10 threshold
        std::vector<float> se;
        for (uint32_t r = 0; r < std::min<uint32_t>(300u, N); r++)
            se.push_back(qr0.exact_dists[qr0.pq_ranked[r].second]);
        std::partial_sort(se.begin(), se.begin() + 10, se.end());
        printf("  [DEBUG]   exact_thresh(10th in shortlist)=%.4f\n", se[9]);
        // sigma_qr for query 0
        const float* q0 = &queries.data[0];
        double var0 = 0.0;
        for (uint32_t s = 0; s < m; s++) {
            const float* qs = q0 + s * sub_dim;
            var0 += dot_f32(qs, qs, sub_dim) * sigma_r_sq[s] / sub_dim;
        }
        printf("  [DEBUG]   sigma_qr=%.4f  ||q||=%.4f\n",
               std::sqrt(var0), std::sqrt(dot_f32(q0, q0, D)));
        // Check: for best candidate, est - sigma_qr vs thresh
        printf("  [DEBUG]   best est - sigma_qr = %.4f vs thresh = %.4f → keep=%d\n",
               qr0.pq_ranked[0].first - std::sqrt(var0), se[9],
               (qr0.pq_ranked[0].first - std::sqrt(var0)) <= se[9]);
    }

    // --- Baseline: fixed top-W (no pruning) ---
    printf("  --- Baseline: fixed top-W ---\n");
    for (uint32_t Wtest : {100u, 200u, 300u}) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%u", Wtest);
        evaluate("TOP-W", buf, [Wtest](uint32_t, const auto& qr, uint32_t) {
            const uint32_t Weff = std::min<uint32_t>(Wtest, (uint32_t)qr.pq_ranked.size());
            std::vector<uint32_t> gids;
            gids.reserve(Weff);
            for (uint32_t i = 0; i < Weff; i++)
                gids.push_back(qr.pq_ranked[i].second);
            return gids;
        });
    }

    // --- Strategy: statistical bound (IP: upper bound, L2sq: lower bound) ---
    // For IP: est_IP = -lut_distance (negate because LUT stores -<q,c>)
    //   true_IP ≤ est_IP + z·σ_qr  where σ_qr = sqrt(Σ_s ||q_s||²·σ²_{r,s}/sub_dim)
    //   Prune candidate if its upper bound < the TOPK-th best among the shortlist.
    // For L2sq: est_L2 = lut_distance
    //   true_L2 ≥ est_L2 - z·σ_qr (statistical, ignores ||r||² shift — approximation)
    //   Prune candidate if its lower bound > the TOPK-th worst among the shortlist.
    printf("\n  --- Statistical bound (free, 0 bytes/vec) ---\n");
    {
        // Precompute σ_qr per query
        std::vector<double> sigma_qr(nq);
        for (uint32_t qi = 0; qi < nq; qi++) {
            const float* q = &queries.data[size_t(qi) * D];
            double var = 0.0;
            for (uint32_t s = 0; s < m; s++) {
                const float* qs = q + s * sub_dim;
                var += dot_f32(qs, qs, sub_dim) * sigma_r_sq[s] / sub_dim;
            }
            sigma_qr[qi] = std::sqrt(var);
        }

        for (double z : {1.0, 2.0, 3.0, 4.0}) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "z=%.1f", z);
            const double zz = z;
            evaluate("STAT", buf,
                     [&](uint32_t qi, const auto& qr, uint32_t Weff) {
                // Compute the TOPK-th threshold from the full shortlist's exact dists
                // (In production we wouldn't have exact dists — we'd use the W-th
                //  PQ distance as threshold. Here we use exact for honest recall.)
                std::vector<float> shortlist_exact;
                shortlist_exact.reserve(Weff);
                for (uint32_t r = 0; r < Weff; r++) {
                    const uint32_t gid = qr.pq_ranked[r].second;
                    shortlist_exact.push_back(qr.exact_dists[gid]);
                }
                std::partial_sort(shortlist_exact.begin(),
                                  shortlist_exact.begin() + TOPK,
                                  shortlist_exact.end());
                const float thresh = shortlist_exact[std::min<uint32_t>(TOPK - 1, Weff - 1)];

                const double sq = sigma_qr[qi];
                std::vector<uint32_t> gids;
                gids.reserve(Weff);
                for (uint32_t r = 0; r < Weff; r++) {
                    const uint32_t gid = qr.pq_ranked[r].second;
                    const float est = qr.pq_ranked[r].first;  // pq estimate
                    // For BOTH metrics, est and exact_dist are in "ascending = best" order.
                    //   IP:   est = -<q,c>, exact = -<q,o>. True exact = est - <q,r>.
                    //   L2sq: est = Σ||q-c||², exact = Σ||q-o||². True = est - 2<q-c,r> + ||r||².
                    // In both cases, the error term (±<q,·,r>) has mean ~0 and std σ_qr.
                    // To decide if a candidate COULD be in top-TOPK:
                    //   Keep if (est - z·σ) <= thresh  [lower bound on true dist ≤ thresh]
                    //   Prune if (est - z·σ) > thresh  [even best case can't beat thresh]
                    // This is the SAME direction for both metrics.
                    const float bound = float(est - zz * sq);
                    if (bound <= thresh) gids.push_back(gid);
                }
                return gids;
            });
        }
    }

    // --- Strategy: Cauchy-Schwarz with ||r|| sidecar (+4 B/vec) ---
    printf("\n  --- Cauchy-Schwarz bound (||r|| sidecar, +4 B/vec) ---\n");
    {
        // Precompute ||q|| per query
        std::vector<float> q_norms(nq);
        for (uint32_t qi = 0; qi < nq; qi++) {
            const float* q = &queries.data[size_t(qi) * D];
            q_norms[qi] = std::sqrt(dot_f32(q, q, D));
        }

        for (double z : {1.0, 2.0, 3.0}) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "z=%.1f", z);
            const double zz = z;
            evaluate("CS||r||", buf,
                     [&](uint32_t qi, const auto& qr, uint32_t Weff) {
                std::vector<float> shortlist_exact;
                shortlist_exact.reserve(Weff);
                for (uint32_t r = 0; r < Weff; r++) {
                    const uint32_t gid = qr.pq_ranked[r].second;
                    shortlist_exact.push_back(qr.exact_dists[gid]);
                }
                std::partial_sort(shortlist_exact.begin(),
                                  shortlist_exact.begin() + TOPK,
                                  shortlist_exact.end());
                const float thresh = shortlist_exact[std::min<uint32_t>(TOPK - 1, Weff - 1)];

                const float qn = q_norms[qi];
                std::vector<uint32_t> gids;
                gids.reserve(Weff);
                for (uint32_t r = 0; r < Weff; r++) {
                    const uint32_t gid = qr.pq_ranked[r].second;
                    const float est = qr.pq_ranked[r].first;
                    // Cauchy-Schwarz: |<q,r>| ≤ ||q||·||r||
                    // True dist = est ± <q,·,r>. Lower bound = est - ||q||·||r||.
                    // Keep if est - z·||q||·||r|| ≤ thresh.
                    // (z scales the CS bound: z=1 is full CS, z<1 tighter, z>1 conservative)
                    const float r_norm = std::sqrt(r_norm_sq[gid]);
                    const float bound = est - float(zz * qn * r_norm);
                    if (bound <= thresh) gids.push_back(gid);
                }
                return gids;
            });
        }
    }

    // --- Strategy: E1 margin (fixed margin on PQ distance) ---
    printf("\n  --- E1: Fixed margin (reproduction) ---\n");
    {
        // pq_dist is in LUT units. The "best" pq_dist is pq_ranked[0].first.
        // We rerank any candidate with pq_dist <= best + margin.
        // Sweep margin as a fraction of the shortlist spread.
        for (float frac : {0.1f, 0.3f, 0.5f, 1.0f, 2.0f}) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "m=%.0f%%", frac * 100);
            const float ffrac = frac;
            evaluate("MARGIN", buf,
                     [ffrac](uint32_t, const auto& qr, uint32_t Weff) {
                const float best = qr.pq_ranked[0].first;
                const float worst = qr.pq_ranked[Weff - 1].first;
                const float span = worst - best;
                const float margin = ffrac * span;
                const float thresh = best + margin;
                std::vector<uint32_t> gids;
                gids.reserve(Weff);
                for (uint32_t r = 0; r < Weff; r++) {
                    if (qr.pq_ranked[r].first <= thresh)
                        gids.push_back(qr.pq_ranked[r].second);
                }
                return gids;
            });
        }
    }

    printf("\n=== Done ===\n");
    printf("Key: prune%% = fraction of W=300 safely excluded from rerank.\n");
    printf("     recall = recall@10 after reranking the pruned shortlist.\n");
    printf("     eff_W  = average candidates reranked (lower = faster rerank).\n");
    printf("     false_pr%% = true top-10 vectors wrongly pruned (lower = safer).\n");
    printf("\nDecision: STAT with ≥40%% prune at <1%% false_pr → GO (free bound).\n");

    return 0;
}
