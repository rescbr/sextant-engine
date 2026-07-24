// route_quality.cpp — measure IVF routing recall (how often are true NNs' shards probed?).
//
// Loads: IVF index (FP16 centroids + K), base .fbin (for NN→shard membership),
// query .fbin + ground-truth .gt. Computes, per query:
//   - routed top-n_probe shards (FP16 L2sq to centroids — exactly what IVFSearcher does)
//   - for each true top-k NN: its primary shard (argmin FP16 L2sq to centroids)
//   - coverage = fraction of true top-k NNs whose primary shard ∈ routed shards
//
// Reports mean coverage (routing recall) for n_probe ∈ {1,2,4,8,16} and k ∈ {10,100}.
// Also reports the ORACLE coverage: if we picked the n_probe shards that contain
// the MOST true NNs (perfect routing), what's the coverage? The gap between
// FP16-routed and oracle is the routing headroom.
//
// Membership is computed in FP16 L2sq space (consistent with routing). closure
// overlap is ignored for this measurement (primary-shard assignment only) —
// overlap can only INCREASE coverage, so this is a lower bound.

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "sextant/ivf_index.hpp"
#include "util/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using sextant::float16_t;
using sextant::Dim;
using namespace sextant;

namespace {

struct FbinHeader { uint32_t n = 0, dim = 0; };
bool read_fbin_header(const std::string& p, FbinHeader& h) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&h.n), 4);
    f.read(reinterpret_cast<char*>(&h.dim), 4);
    return f.good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <ivf_index.shards_dir> <base.fbin> <query.fbin> <gt.gt> [max_queries] [--fp32-route] [--multiprobe RATIO]\n", argv[0]);
        std::fprintf(stderr, "  --fp32-route: route in FP32 (tests FP16 quantization loss)\n");
        std::fprintf(stderr, "  --multiprobe R: multi-probe — include all centroids within R×d[np-1]\n");
        return 1;
    }
    const std::string shards_dir = argv[1];
    const std::string base_path = argv[2];
    const std::string query_path = argv[3];
    const std::string gt_path = argv[4];
    uint32_t max_q = 0;
    bool fp32_route = false;
    float multiprobe_ratio = 0.0f;  // 0 = off
    for (int a = 5; a < argc; a++) {
        std::string arg = argv[a];
        if (arg == "--fp32-route") fp32_route = true;
        else if (arg == "--multiprobe" && a + 1 < argc) { multiprobe_ratio = std::stof(argv[++a]); }
        else if (max_q == 0) max_q = static_cast<uint32_t>(std::atoi(arg.c_str()));
    }

    // Load IVF index (centroids + K + dim).
    auto ivf = sextant::IVFIndex::read(shards_dir);
    const uint32_t K = ivf->K;
    const Dim dim = ivf->dim;
    const float16_t* centroids = ivf->centroids.data();  // K × dim FP16
    std::fprintf(stderr, "[route] IVF: K=%u dim=%u\n", K, (unsigned)dim);

    // Load base .fbin → FP16 (for NN→shard membership).
    FbinHeader bh;
    if (!read_fbin_header(base_path, bh) || bh.dim != dim) {
        std::fprintf(stderr, "[route] bad base header\n"); return 1;
    }
    const uint32_t N = bh.n;
    std::vector<float> base_f32(static_cast<size_t>(N) * dim);
    {
        std::ifstream f(base_path, std::ios::binary);
        f.seekg(8);
        f.read(reinterpret_cast<char*>(base_f32.data()),
               static_cast<std::streamsize>(base_f32.size() * sizeof(float)));
        if (!f) { std::fprintf(stderr, "[route] base short read\n"); return 1; }
    }
    std::vector<float16_t> base_f16(static_cast<size_t>(N) * dim);
    cast_fp32_to_fp16(base_f32.data(), base_f16.data(), static_cast<size_t>(N) * dim);
    std::fprintf(stderr, "[route] loaded %u base vectors\n", N);

    // Load query .fbin → FP16.
    FbinHeader qh;
    if (!read_fbin_header(query_path, qh) || qh.dim != dim) {
        std::fprintf(stderr, "[route] bad query header\n"); return 1;
    }
    uint32_t n_queries = qh.n;
    if (max_q > 0 && max_q < n_queries) n_queries = max_q;
    std::vector<float> qf32(static_cast<size_t>(n_queries) * dim);
    {
        std::ifstream f(query_path, std::ios::binary);
        f.seekg(8);
        f.read(reinterpret_cast<char*>(qf32.data()),
               static_cast<std::streamsize>(qf32.size() * sizeof(float)));
    }
    std::vector<float16_t> qf16(static_cast<size_t>(n_queries) * dim);
    cast_fp32_to_fp16(qf32.data(), qf16.data(), static_cast<size_t>(n_queries) * dim);

    // Load ground truth.
    std::ifstream gf(gt_path, std::ios::binary);
    uint32_t gt_n = 0, gt_k = 0;
    gf.read(reinterpret_cast<char*>(&gt_n), 4);
    gf.read(reinterpret_cast<char*>(&gt_k), 4);
    std::vector<uint32_t> gt_ids(static_cast<size_t>(gt_n) * gt_k);
    gf.read(reinterpret_cast<char*>(gt_ids.data()),
            static_cast<std::streamsize>(gt_ids.size() * 4));
    std::fprintf(stderr, "[route] GT: n=%u k=%u (routing: %s)\n", gt_n, gt_k,
                 fp32_route ? "FP32" : "FP16");
    if (gt_n < n_queries) n_queries = gt_n;

    // FP32 copies of centroids + base + queries for the --fp32-route mode.
    // (Upcast from FP16 — tests whether FP16 quantization noise in the stored
    // centroids is the routing-quality limiter. Note: the centroids themselves
    // are decoded from PQ codes, so this tests FP16-vs-FP32 on PQ-resolution
    // data, NOT true FP32 k-means centroids.)
    std::vector<float> cent_f32, base_f32_route;
    if (fp32_route) {
        cent_f32.resize(static_cast<size_t>(K) * dim);
        for (uint32_t i = 0; i < static_cast<size_t>(K) * dim; i++) {
            cent_f32[i] = static_cast<float>(centroids[i]);
        }
        base_f32_route = base_f32;  // already have FP32 base
    }

    // Distance helpers (precision-agnostic).
    auto base_dist = [&](const float* a16f32, const float* b16f32) {
        // a16f32/b16f32 are FP32 values (upcast from FP16); route in FP32.
        float acc = 0.0f;
        for (uint32_t i = 0; i < dim; i++) { float d = a16f32[i] - b16f32[i]; acc += d * d; }
        return acc;
    };

    // --- Compute each base vector's shard membership (closure-expanded).
    // A vector belongs to shard c if d(vec, centroid_c) ≤ closure_factor × d_best
    // — EXACTLY matching partition.cpp's assignment. This is the authoritative
    // membership; "primary shard" (argmin) would UNDERCOUNT coverage because it
    // ignores the build-time overlap that already puts boundary vectors in 2+
    // shards. We store membership as a flat CSR: shard_members_offsets[v],
    // shard_members[v][].
    // O(N × K × dim) — the dominant cost. Parallelize across vectors.
    const float closure = ivf->closure_factor;
    std::vector<uint32_t> mem_offsets(N + 1, 0);  // CSR offsets
    {
        std::vector<std::thread> pool;
        uint32_t nthreads = std::max(1u, std::thread::hardware_concurrency());
        uint32_t per = (N + nthreads - 1) / nthreads;
        // First pass: count members per vector (accumulate CSR offsets).
        auto worker = [&](uint32_t /*tid*/, uint32_t lo, uint32_t hi) {
            std::vector<float> dists(K);
            for (uint32_t v = lo; v < hi; v++) {
                float best = std::numeric_limits<float>::max();
                if (fp32_route) {
                    const float* vec = base_f32_route.data() + static_cast<size_t>(v) * dim;
                    for (uint32_t c = 0; c < K; c++) {
                        dists[c] = base_dist(vec, cent_f32.data() + static_cast<size_t>(c) * dim);
                        if (dists[c] < best) best = dists[c];
                    }
                } else {
                    const float16_t* vec = base_f16.data() + static_cast<size_t>(v) * dim;
                    for (uint32_t c = 0; c < K; c++) {
                        dists[c] = simd::l2sq_f16(vec, centroids + static_cast<size_t>(c) * dim, dim);
                        if (dists[c] < best) best = dists[c];
                    }
                }
                const float thresh = closure * best;
                uint32_t cnt = 0;
                for (uint32_t c = 0; c < K; c++) if (dists[c] <= thresh) cnt++;
                mem_offsets[v + 1] = cnt;
            }
        };
        for (uint32_t t = 0; t < nthreads; t++) {
            uint32_t lo = t * per, hi = std::min(N, lo + per);
            if (lo < hi) pool.emplace_back(worker, t, lo, hi);
        }
        for (auto& th : pool) th.join();
        // CSR prefix sum.
        for (uint32_t v = 0; v < N; v++) mem_offsets[v + 1] += mem_offsets[v];
        // Report avg replication for sanity (should match the build log).
        double avg_repl = static_cast<double>(mem_offsets[N]) / N;
        std::fprintf(stderr, "[route] closure=%.4f, avg replication %.3f×\n",
                     closure, avg_repl);
    }
    // Second pass: fill members.
    std::vector<uint8_t> members(mem_offsets[N]);
    {
        std::vector<std::thread> pool;
        uint32_t nthreads = std::max(1u, std::thread::hardware_concurrency());
        uint32_t per = (N + nthreads - 1) / nthreads;
        auto worker = [&](uint32_t lo, uint32_t hi) {
            std::vector<float> dists(K);
            for (uint32_t v = lo; v < hi; v++) {
                float best = std::numeric_limits<float>::max();
                if (fp32_route) {
                    const float* vec = base_f32_route.data() + static_cast<size_t>(v) * dim;
                    for (uint32_t c = 0; c < K; c++) {
                        dists[c] = base_dist(vec, cent_f32.data() + static_cast<size_t>(c) * dim);
                        if (dists[c] < best) best = dists[c];
                    }
                } else {
                    const float16_t* vec = base_f16.data() + static_cast<size_t>(v) * dim;
                    for (uint32_t c = 0; c < K; c++) {
                        dists[c] = simd::l2sq_f16(vec, centroids + static_cast<size_t>(c) * dim, dim);
                        if (dists[c] < best) best = dists[c];
                    }
                }
                const float thresh = closure * best;
                uint32_t pos = mem_offsets[v];
                for (uint32_t c = 0; c < K; c++) {
                    if (dists[c] <= thresh) members[pos++] = static_cast<uint8_t>(c);
                }
            }
        };
        for (uint32_t t = 0; t < nthreads; t++) {
            uint32_t lo = t * per, hi = std::min(N, lo + per);
            if (lo < hi) pool.emplace_back(worker, lo, hi);
        }
        for (auto& th : pool) th.join();
    }
    // Helper: is shard c in vector v's membership set?
    auto in_membership = [&](uint32_t v, uint8_t c) {
        const uint32_t lo = mem_offsets[v], hi = mem_offsets[v + 1];
        for (uint32_t p = lo; p < hi; p++) if (members[p] == c) return true;
        return false;
    };

    // --- For each query: routed shards (top-n_probe by FP16) + coverage analysis.
    const uint32_t nps[] = {1, 2, 4, 8, 16, 32};
    const uint32_t ks[] = {10, 100};
    // accumulators[routed_np][eval_k] = sum of coverage over queries
    double routed_cov[6][2] = {{0}};
    double oracle_cov[6][2] = {{0}};
    double multiprobe_cov[6][2] = {{0}};   // coverage with ratio-extended probe set
    double multiprobe_avg_np[6] = {0};     // avg shards probed under multi-probe

    std::vector<std::pair<float, uint32_t>> cdists(K);
    for (uint32_t qi = 0; qi < n_queries; qi++) {
        // Query→centroid distances (FP16 or FP32 per mode).
        if (fp32_route) {
            const float* q = qf32.data() + static_cast<size_t>(qi) * dim;
            for (uint32_t c = 0; c < K; c++) {
                cdists[c] = {base_dist(q, cent_f32.data() + static_cast<size_t>(c) * dim), c};
            }
        } else {
            const float16_t* q = qf16.data() + static_cast<size_t>(qi) * dim;
            for (uint32_t c = 0; c < K; c++) {
                cdists[c] = {simd::l2sq_f16(q, centroids + static_cast<size_t>(c) * dim, dim), c};
            }
        }
        // Routed order: sort ascending by distance (full sort; K is small).
        std::vector<std::pair<float, uint32_t>> routed = cdists;
        std::sort(routed.begin(), routed.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // For each true NN (up to gt_k), tally hits per shard. With closure
        // overlap, an NN is a member of multiple shards — count it in EACH of
        // its member shards (for oracle). hits_k[ki][c] = # NNs (among first
        // eval_ks[ki]) that are members of shard c.
        std::vector<std::vector<uint32_t>> hits(2, std::vector<uint32_t>(K, 0));
        uint32_t eval_ks[2] = {std::min(10u, gt_k), std::min(100u, gt_k)};
        uint32_t max_eval = std::max(eval_ks[0], eval_ks[1]);
        for (uint32_t j = 0; j < max_eval; j++) {
            const uint32_t nn = gt_ids[static_cast<size_t>(qi) * gt_k + j];
            if (nn >= N) continue;
            for (uint32_t p = mem_offsets[nn]; p < mem_offsets[nn + 1]; p++) {
                const uint8_t sh = members[p];
                for (uint32_t ki = 0; ki < 2; ki++) {
                    if (j < eval_ks[ki]) hits[ki][sh]++;
                }
            }
        }

        // For each (routed_np, eval_k): coverage = fraction of NNs covered by
        // the routed probe set. An NN is covered if ANY of its member shards is
        // in the routed top-np (closure-aware).
        for (uint32_t ni = 0; ni < 6; ni++) {
            const uint32_t np = std::min(nps[ni], K);
            for (uint32_t ki = 0; ki < 2; ki++) {
                const uint32_t ek = eval_ks[ki];
                if (ek == 0) continue;
                // Routed coverage (closure-aware): NN covered if any member
                // shard ∈ routed[0..np).
                uint32_t r_sum = 0, total = 0;
                for (uint32_t j = 0; j < ek; j++) {
                    const uint32_t nn = gt_ids[static_cast<size_t>(qi) * gt_k + j];
                    if (nn < N) {
                        total++;
                        for (uint32_t p = 0; p < np; p++) {
                            if (in_membership(nn, static_cast<uint8_t>(routed[p].second))) {
                                r_sum++; break;
                            }
                        }
                    }
                }
                routed_cov[ni][ki] += (total > 0) ? static_cast<double>(r_sum) / total : 0.0;

                // Oracle (upper bound via greedy set-cover): pick the np shards
                // covering the most NNs (closure-aware: an NN covered if any
                // member shard picked). hits[ki][c] already tallies per-shard
                // membership; greedy picks the np highest-hit shards. This is
                // an upper bound on perfect routing with closure overlap.
                std::vector<std::pair<uint32_t, uint32_t>> h(K);  // (hits, shard)
                for (uint32_t c = 0; c < K; c++) h[c] = {hits[ki][c], c};
                std::partial_sort(h.begin(), h.begin() + np, h.end(),
                                  [](const auto& a, const auto& b) {
                                      if (a.first != b.first) return a.first > b.first;
                                      return a.second < b.second;
                                  });
                // Greedy coverage: count NNs covered by any of the top-np shards.
                uint32_t o_sum = 0;
                for (uint32_t j = 0; j < ek; j++) {
                    const uint32_t nn = gt_ids[static_cast<size_t>(qi) * gt_k + j];
                    if (nn >= N) continue;
                    for (uint32_t p = 0; p < np; p++) {
                        if (in_membership(nn, static_cast<uint8_t>(h[p].second))) {
                            o_sum++; break;
                        }
                    }
                }
                oracle_cov[ni][ki] += (total > 0) ? static_cast<double>(o_sum) / total : 0.0;

                // Multi-probe coverage: extend the probe set to all centroids
                // within multiprobe_ratio × d_np. Computed once per (query, np)
                // — guard on ki==0 so it's not redone for ki==1.
                if (multiprobe_ratio > 0.0f && ki == 0) {
                    const float d_np = routed[np - 1].first;
                    const float thresh = multiprobe_ratio * d_np;
                    uint32_t mp_count = np;
                    for (uint32_t p = np; p < K; p++) {
                        if (routed[p].first <= thresh) mp_count++;
                        else break;  // sorted ascending
                    }
                    multiprobe_avg_np[ni] += mp_count;
                    // coverage at k=100 (ki loop will also hit ki==1 below, but
                    // we only compute here on ki==0; fill both ki slots).
                }
                if (multiprobe_ratio > 0.0f) {
                    const float d_np = routed[np - 1].first;
                    const float thresh = multiprobe_ratio * d_np;
                    // Build the multi-probe shard set (top-np + within-ratio).
                    uint32_t m_sum = 0, total2 = 0;
                    for (uint32_t j = 0; j < ek; j++) {
                        const uint32_t nn = gt_ids[static_cast<size_t>(qi) * gt_k + j];
                        if (nn < N) {
                            total2++;
                            // NN covered if any of its member shards is in the
                            // multi-probe set (top-np OR within ratio).
                            for (uint32_t p = 0; p < K; p++) {
                                if (p >= np && routed[p].first > thresh) break;
                                if (in_membership(nn, static_cast<uint8_t>(routed[p].second))) {
                                    m_sum++; break;
                                }
                            }
                        }
                    }
                    multiprobe_cov[ni][ki] += (total2 > 0) ? static_cast<double>(m_sum) / total2 : 0.0;
                }
            }
        }
    }

    // --- Report.
    std::printf("\n=== Routing recall (FP16) vs Oracle (perfect) — %u queries ===\n", n_queries);
    std::printf("%-8s | %-26s | %-26s\n", "n_probe", "FP16 routing recall", "Oracle routing recall");
    std::printf("%-8s | %-12s %-12s | %-12s %-12s\n", "", "k=10", "k=100", "k=10", "k=100");
    std::printf("--------|----------------------------|----------------------------\n");
    for (uint32_t ni = 0; ni < 6; ni++) {
        const uint32_t np = std::min(nps[ni], K);
        if (nps[ni] > K) break;  // K=8 → stop after np=8
        std::printf("np=%-4u | %-12.4f %-12.4f | %-12.4f %-12.4f\n",
                    np,
                    routed_cov[ni][0] / n_queries, routed_cov[ni][1] / n_queries,
                    oracle_cov[ni][0] / n_queries, oracle_cov[ni][1] / n_queries);
    }
    std::printf("\nThe gap between FP16 and Oracle = routing headroom (recall recoverable\n");
    std::printf("by better routing without increasing n_probe).\n");

    if (multiprobe_ratio > 0.0f) {
        std::printf("\n=== Multi-probe (ratio=%.2f) — coverage vs avg shards probed ===\n",
                    multiprobe_ratio);
        std::printf("%-8s | %-12s | %-20s | %-12s\n", "base_np", "FP16 recall", "multi-probe recall", "avg shards");
        std::printf("        |              | k=10       k=100     | probed\n");
        std::printf("--------|--------------|----------------------|------------\n");
        for (uint32_t ni = 0; ni < 6; ni++) {
            const uint32_t np = std::min(nps[ni], K);
            if (nps[ni] > K) break;
            std::printf("np=%-4u | %-12.4f | %-9.4f    %-8.4f | %-12.2f\n",
                        np,
                        routed_cov[ni][1] / n_queries,
                        multiprobe_cov[ni][0] / n_queries,
                        multiprobe_cov[ni][1] / n_queries,
                        multiprobe_avg_np[ni] / n_queries);
        }
        std::printf("\nIf avg-shards-probed is close to base_np, multi-probe is cheap.\n");
    }
    return 0;
}
