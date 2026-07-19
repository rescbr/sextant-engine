// Microbenchmark for PqQuantizer::code_distance — the #1 construct hotspot.
//
// We train a real PQ on SIFT-1M, then time three things:
//   1. The current scalar code_distance (baseline).
//   2. A batched variant: fixed anchor, loop over many candidates.
//   3. Alternative table layouts (transposed) — TODO if baseline shows cache misses.
//
// Build: see scripts/bench_code_distance.sh

#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

using namespace sextant;

int main() {
    // SIFT-1M params: dim=128, m=32, bits=8, K=256.
    const Dim dim = 128;
    const uint8_t m = 32;
    const uint8_t bits = 8;
    const uint32_t K = 1u << bits;

    // Generate synthetic training data (random is fine — we only care about
    // the table-lookup cost, not codebook quality).
    const uint64_t n_train = 50000;
    std::vector<float> train(n_train * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    for (auto& v : train) v = u(rng);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits);
    q.train(train.data(), n_train);

    // Generate N random codes.
    const uint32_t N = 100000;
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes(N * cs);
    // Realistic-ish: encode random vectors (random codes would be fine too).
    std::vector<float> tmp(dim);
    for (uint32_t i = 0; i < N; i++) {
        for (auto& v : tmp) v = u(rng);
        q.encode(tmp.data(), codes.data() + i * cs);
    }

    // Simulate robust_prune's occlusion pattern: for each of N anchors,
    // compute code_distance(anchor, candidate) for ~128 candidates.
    // This matches the actual call pattern (L_build ~128 candidates per node).
    const uint32_t L_build = 128;
    std::vector<uint32_t> cand_idx(L_build);
    std::mt19937 rng2(7);
    for (uint32_t i = 0; i < L_build; i++) cand_idx[i] = rng2() % N;

    // --- Baseline: current code_distance ---
    volatile float sink = 0;
    float acc = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t a = 0; a < N; a++) {
        const uint8_t* anchor = codes.data() + a * cs;
        for (uint32_t c = 0; c < L_build; c++) {
            const uint8_t* cand = codes.data() + cand_idx[c] * cs;
            acc += q.code_distance(anchor, cand);
        }
        if ((a & 0x3FFF) == 0) sink += acc;  // prevent dead-code elim
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs_baseline = std::chrono::duration<double>(t1 - t0).count();
    sink += acc;
    double total_calls = (double)N * L_build;
    printf("=== code_distance (current scalar) ===\n");
    printf("%.0f calls in %.4fs → %.1f M calls/s\n",
           total_calls, secs_baseline, total_calls / secs_baseline / 1e6);
    printf("(per call: %.1f ns)\n", secs_baseline / total_calls * 1e9);
    printf("(simulates robust_prune: N=%u anchors × L_build=%u cands)\n\n",
           N, L_build);

    // --- Batched: fixed anchor, many candidates — does the anchor's table
    //     region stay hot if we reuse it? (This is what robust_prune already
    //     does, but it confirms the throughput ceiling.) ---
    acc = 0;
    t0 = std::chrono::steady_clock::now();
    for (uint32_t a = 0; a < N; a++) {
        const uint8_t* anchor = codes.data() + a * cs;
        for (uint32_t c = 0; c < L_build; c++) {
            const uint8_t* cand = codes.data() + (a + c + 1) % N * cs;
            acc += q.code_distance(anchor, cand);
        }
        if ((a & 0x3FFF) == 0) sink += acc;
    }
    t1 = std::chrono::steady_clock::now();
    double secs_seq = std::chrono::duration<double>(t1 - t0).count();
    sink += acc;
    printf("=== code_distance (sequential cand) ===\n");
    printf("%.0f calls in %.4fs → %.1f M calls/s\n\n",
           total_calls, secs_seq, total_calls / secs_seq / 1e6);

    // Report table size for context.
    printf("cross_distance_table: %.1f MB (m=%u K=%u, %zu floats)\n",
           (double)m * K * K * 4 / 1e6, m, K, (size_t)m * K * K);
    printf("per-segment sub-table: %.1f KB (K×K floats)\n",
           (double)K * K * 4 / 1e3);

    (void)sink;
    return 0;
}
