// Robust prefetch sweep: longer runs, stable baseline, single-threaded.
// The key question: does prefetching candidate codes help when they're
// scattered across a 32MB buffer (the real robust_prune pattern)?

#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace sextant;
static const uint32_t M = 32, K = 256;

double bench(const PqQuantizer& q, const uint8_t* codes, uint32_t cs,
             const uint8_t* anchor, const std::vector<uint32_t>& cand_rowids,
             int pf_dist) {
    const uint32_t n_cands = cand_rowids.size();
    const uint32_t REPEAT = 50000;
    volatile float sink = 0;
    float acc = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t r = 0; r < REPEAT; r++) {
        for (uint32_t i = 0; i < n_cands; i++) {
            if (pf_dist > 0) {
                uint32_t ahead = i + pf_dist;
                if (ahead < n_cands) {
                    __builtin_prefetch(codes + (size_t)cand_rowids[ahead] * cs, 0, 0);
                }
            }
            acc += q.code_distance(anchor, codes + (size_t)cand_rowids[i] * cs);
        }
        if ((r & 0xFFF) == 0) sink += acc;
    }
    auto t1 = std::chrono::steady_clock::now();
    sink += acc;
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    PqQuantizer q(MetricKind::L2Sq, 128, 32, 8);
    std::vector<float> train(50000 * 128);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    for (auto& v : train) v = u(rng);
    q.train(train.data(), 50000);

    const uint32_t N = 1000000;  // 32MB codes buffer
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes((size_t)N * cs);
    std::vector<float> tmp(128);
    for (uint32_t i = 0; i < N; i++) {
        for (auto& v : tmp) v = u(rng);
        q.encode(tmp.data(), codes.data() + (size_t)i * cs);
    }

    const uint32_t L_build = 128;
    std::vector<uint32_t> cand_rowids(L_build);
    for (auto& c : cand_rowids) c = rng() % N;
    const uint8_t* anchor = codes.data() + (rng() % N) * cs;

    // Warm up.
    bench(q, codes.data(), cs, anchor, cand_rowids, 0);

    // Stable baseline: run 3 times, take min.
    double best0 = 1e9;
    for (int t = 0; t < 3; t++) best0 = std::min(best0, bench(q, codes.data(), cs, anchor, cand_rowids, 0));
    printf("baseline (no prefetch): %.4fs (%.1f ns/call)\n", best0, best0 / (50000.0 * L_build) * 1e9);

    for (int dist : {1, 2, 4, 8, 12}) {
        double best = 1e9;
        for (int t = 0; t < 3; t++) best = std::min(best, bench(q, codes.data(), cs, anchor, cand_rowids, dist));
        printf("pf_dist=%2d: %.4fs (%.1f ns/call)  [%.0f%% of baseline]\n",
               dist, best, best / (50000.0 * L_build) * 1e9, best / best0 * 100);
    }
    return 0;
}
