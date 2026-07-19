// Realistic cold-miss benchmark: each "insert" gets a fresh set of candidates
// (not repeated), matching the real build's access pattern.
#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace sextant;
static const uint32_t M = 32, K = 256, KK = K*K;

int main() {
    PqQuantizer q(MetricKind::L2Sq, 128, 32, 8);
    std::vector<float> train(50000 * 128);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    for (auto& v : train) v = u(rng);
    q.train(train.data(), 50000);

    const uint32_t N = 1000000;  // 32MB codes
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes((size_t)N * cs);
    std::vector<float> tmp(128);
    for (uint32_t i = 0; i < N; i++) {
        for (auto& v : tmp) v = u(rng);
        q.encode(tmp.data(), codes.data() + (size_t)i * cs);
    }

    // Simulate the real robust_prune pattern: many "inserts", each with a
    // fresh anchor + fresh L_build candidates (cold misses every insert).
    const uint32_t N_INSERTS = 100000;
    const uint32_t L_build = 128;
    // Pre-generate all candidate row_id sets (so generation doesn't bias timing).
    std::vector<uint32_t> all_cands((size_t)N_INSERTS * L_build);
    std::vector<uint32_t> all_anchors(N_INSERTS);
    for (uint32_t i = 0; i < N_INSERTS; i++) {
        all_anchors[i] = rng() % N;
        for (uint32_t c = 0; c < L_build; c++)
            all_cands[i * L_build + c] = rng() % N;
    }

    const uint32_t pf_dist = 12;
    double calls = (double)N_INSERTS * L_build;

    // V1: no prefetch (q.code_distance — the real call path).
    {
        volatile float sink = 0; float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)all_anchors[ins] * cs;
            const uint32_t* cands = &all_cands[ins * L_build];
            for (uint32_t c = 0; c < L_build; c++) {
                acc += q.code_distance(anchor, codes.data() + (size_t)cands[c] * cs);
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("V1 no-pf (code_distance): %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    // V2: prefetch candidate codes (dist=12), still using q.code_distance.
    {
        volatile float sink = 0; float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)all_anchors[ins] * cs;
            const uint32_t* cands = &all_cands[ins * L_build];
            for (uint32_t c = 0; c < L_build; c++) {
                if (c + pf_dist < L_build)
                    __builtin_prefetch(codes.data() + (size_t)cands[c+pf_dist]*cs, 0, 0);
                acc += q.code_distance(anchor, codes.data() + (size_t)cands[c] * cs);
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("V2 pf-cand (code_dist):   %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    return 0;
}
