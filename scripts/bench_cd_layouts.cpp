// Targeted experiment: measure code_distance call patterns and test whether
// restructuring the access (precomputing anchor centroids, batched gather)
// can beat the current scalar loop.
//
// The real call pattern in robust_prune: fixed anchor p, loop over ~128 pp.
// Currently each (p, pp) pair does 32 scattered reads into the 8MB table.

#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace sextant;

static const uint8_t M = 32;
static const uint32_t K = 256;

// Baseline: current code_distance.
double bench_baseline(const PqQuantizer& q, const uint8_t* codes, uint32_t cs,
                      uint32_t N, uint32_t L_build, const uint32_t* cand_idx) {
    volatile float sink = 0;
    float acc = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t a = 0; a < N; a++) {
        const uint8_t* anchor = codes + (size_t)a * cs;
        for (uint32_t c = 0; c < L_build; c++) {
            acc += q.code_distance(anchor, codes + (size_t)cand_idx[c] * cs);
        }
        if ((a & 0x3FFF) == 0) sink += acc;
    }
    auto t1 = std::chrono::steady_clock::now();
    sink += acc;
    return std::chrono::duration<double>(t1 - t0).count();
}

// Experiment 1: precompute anchor's 32 centroid IDs (avoid read_code in
// the inner loop). Tests whether read_code overhead matters.
double bench_precompute_anchor(const float* table, const uint8_t* codes,
                               uint32_t cs, uint32_t N, uint32_t L_build,
                               const uint32_t* cand_idx) {
    volatile float sink = 0;
    float acc = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t a = 0; a < N; a++) {
        const uint8_t* anchor = codes + (size_t)a * cs;
        // Precompute anchor centroids.
        uint32_t ac[M];
        for (uint32_t s = 0; s < M; s++) ac[s] = anchor[s];
        for (uint32_t c = 0; c < L_build; c++) {
            const uint8_t* cand = codes + (size_t)cand_idx[c] * cs;
            float d = 0;
            for (uint32_t s = 0; s < M; s++) {
                d += table[(size_t)s * K * K + ac[s] * K + cand[s]];
            }
            acc += d;
        }
        if ((a & 0x3FFF) == 0) sink += acc;
    }
    auto t1 = std::chrono::steady_clock::now();
    sink += acc;
    return std::chrono::duration<double>(t1 - t0).count();
}

// Experiment 2: transposed layout [s][cb][ca] — for fixed anchor, inner loop
// over pp with fixed ca_s but varying cb_s. Still scattered. (Sanity check.)
// Build a transposed table and time it.
double bench_transposed(const float* table_t, const uint8_t* codes,
                        uint32_t cs, uint32_t N, uint32_t L_build,
                        const uint32_t* cand_idx) {
    // table_t laid out as [s][cb][ca] (transposed).
    volatile float sink = 0;
    float acc = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t a = 0; a < N; a++) {
        const uint8_t* anchor = codes + (size_t)a * cs;
        uint32_t ac[M];
        for (uint32_t s = 0; s < M; s++) ac[s] = anchor[s];
        for (uint32_t c = 0; c < L_build; c++) {
            const uint8_t* cand = codes + (size_t)cand_idx[c] * cs;
            float d = 0;
            for (uint32_t s = 0; s < M; s++) {
                // [s][cb][ca] = table_t[s*K*K + cb*K + ca]
                d += table_t[(size_t)s * K * K + cand[s] * K + ac[s]];
            }
            acc += d;
        }
        if ((a & 0x3FFF) == 0) sink += acc;
    }
    auto t1 = std::chrono::steady_clock::now();
    sink += acc;
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    const Dim dim = 128;
    const uint64_t n_train = 50000;
    std::vector<float> train(n_train * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    for (auto& v : train) v = u(rng);

    PqQuantizer q(MetricKind::L2Sq, dim, M, 8);
    q.train(train.data(), n_train);

    const uint32_t N = 100000;
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes((size_t)N * cs);
    std::vector<float> tmp(dim);
    for (uint32_t i = 0; i < N; i++) {
        for (auto& v : tmp) v = u(rng);
        q.encode(tmp.data(), codes.data() + (size_t)i * cs);
    }

    const uint32_t L_build = 128;
    std::vector<uint32_t> cand_idx(L_build);
    for (uint32_t i = 0; i < L_build; i++) cand_idx[i] = rng() % N;

    double calls = (double)N * L_build;

    double t_base = bench_baseline(q, codes.data(), cs, N, L_build, cand_idx.data());
    printf("baseline (current code_distance):  %.4fs → %.1f M/s (%.1f ns/call)\n",
           t_base, calls / t_base / 1e6, t_base / calls * 1e9);

    // Access the cross_distance_table directly via code_distance-equivalent.
    // We need the table pointer; replicate by computing one.
    std::vector<float> table((size_t)M * K * K);
    // Fill via code_distance symmetry: table[s*K*K + a*K + b] = d(centroid_a, centroid_b) in segment s.
    // We can read it through the quantizer... but it's private. Reconstruct by
    // encoding each pair of unit-centroid codes. For brevity, build a standalone
    // table matching the layout (random data is fine — we're measuring layout cost).
    // Actually, to measure layout cost we need realistic data; use the quantizer's
    // code_distance to populate it via "canonical" codes.
    // Build canonical codes: code for (s has centroid c, others 0).
    std::vector<uint8_t> canona(cs, 0), canonb(cs, 0);
    for (uint32_t s = 0; s < M; s++) {
        for (uint32_t a = 0; a < K; a++) {
            for (uint32_t b = 0; b < K; b++) {
                canona[s] = (uint8_t)a;
                canonb[s] = (uint8_t)b;
                table[(size_t)s * K * K + a * K + b] = q.code_distance(canona.data(), canonb.data());
            }
        }
    }
    // Transposed table [s][cb][ca].
    std::vector<float> table_t((size_t)M * K * K);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t a = 0; a < K; a++)
            for (uint32_t b = 0; b < K; b++)
                table_t[(size_t)s * K * K + b * K + a] = table[(size_t)s * K * K + a * K + b];

    double t_pre = bench_precompute_anchor(table.data(), codes.data(), cs, N, L_build, cand_idx.data());
    printf("precompute anchor centroids:       %.4fs → %.1f M/s (%.1f ns/call)  [%.0f%% of baseline]\n",
           t_pre, calls / t_pre / 1e6, t_pre / calls * 1e9, t_pre / t_base * 100);

    double t_tr = bench_transposed(table_t.data(), codes.data(), cs, N, L_build, cand_idx.data());
    printf("transposed table [s][cb][ca]:      %.4fs → %.1f M/s (%.1f ns/call)  [%.0f%% of baseline]\n",
           t_tr, calls / t_tr / 1e6, t_tr / calls * 1e9, t_tr / t_base * 100);

    return 0;
}
