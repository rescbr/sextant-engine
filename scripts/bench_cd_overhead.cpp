// Pinpoint exactly WHERE the code_distance overhead comes from.
// Variants tested at -O3 -march=native, matching the real build.

#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace sextant;
static const uint32_t M = 32, K = 256, KK = K*K;

// V1: exact replica of current code_distance (function call, .empty() check,
//     read_code, mul per iter). Inlined here to isolate from quantizer state.
double v1_current(const float* table, const uint8_t* a, const uint8_t* b) {
    float acc = 0;
    for (uint32_t s = 0; s < M; s++) {
        const uint32_t ca = a[s];
        const uint32_t cb = b[s];
        acc += table[s * KK + ca * K + cb];
    }
    return acc;
}

// V2: hoist anchor centroids (simulate "fixed anchor, many b's").
void v2_hoisted(const float* table, const uint8_t* a, const uint8_t* b_arr,
                uint32_t n, float* out) {
    uint32_t ac[M];
    for (uint32_t s = 0; s < M; s++) ac[s] = a[s];
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t* b = b_arr + i * M;
        float acc = 0;
        for (uint32_t s = 0; s < M; s++) {
            acc += table[s * KK + ac[s] * K + b[s]];
        }
        out[i] = acc;
    }
}

// V3: like v1 but per-pair (no hoist) — the "fair" baseline matching v1's
//     call granularity. This is what robust_prune effectively does.
int main() {
    PqQuantizer q(MetricKind::L2Sq, 128, 32, 8);
    std::vector<float> train(50000 * 128);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    for (auto& v : train) v = u(rng);
    q.train(train.data(), 50000);

    // Get the table. Reconstruct via canonical codes.
    std::vector<float> table((size_t)M * K * K);
    std::vector<uint8_t> ca(32, 0), cb(32, 0);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t a = 0; a < K; a++)
            for (uint32_t b = 0; b < K; b++) {
                ca[s] = a; cb[s] = b;
                table[s*KK + a*K + b] = q.code_distance(ca.data(), cb.data());
            }

    const uint32_t N = 100000, L = 128;
    std::vector<uint8_t> codes((size_t)N * 32);
    std::vector<float> tmp(128);
    for (uint32_t i = 0; i < N; i++) {
        for (auto& v : tmp) v = u(rng);
        q.encode(tmp.data(), codes.data() + (size_t)i * 32);
    }
    std::vector<uint32_t> cidx(L);
    for (auto& c : cidx) c = rng() % N;

    volatile float sink = 0;
    double calls = (double)N * L;

    // v1: per-pair, inline table access (no quantizer call overhead).
    {
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t a = 0; a < N; a++) {
            const uint8_t* anc = codes.data() + (size_t)a * 32;
            for (uint32_t c = 0; c < L; c++) {
                acc += v1_current(table.data(), anc, codes.data() + (size_t)cidx[c] * 32);
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("v1 inline per-pair:     %.4fs → %.1f M/s (%.1f ns)\n", s, calls/s/1e6, s/calls*1e9);
    }
    // v2: batched, hoisted anchor centroids.
    {
        std::vector<float> out(L);
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t a = 0; a < N; a++) {
            const uint8_t* anc = codes.data() + (size_t)a * 32;
            // Gather candidate pointers... v2 takes contiguous b_arr.
            // Build a contiguous candidate block (realistic: candidates aren't
            // contiguous in memory, so copy). Time the copy too.
            uint8_t cblock[128 * 32];
            for (uint32_t c = 0; c < L; c++)
                __builtin_memcpy(cblock + c*32, codes.data() + (size_t)cidx[c]*32, 32);
            v2_hoisted(table.data(), anc, cblock, L, out.data());
            sink += out[0];
        }
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("v2 batched+hoist+copy:  %.4fs → %.1f M/s (%.1f ns) [incl gather copy]\n", s, calls/s/1e6, s/calls*1e9);
    }
    // v2b: batched, hoisted, NO copy (candidates already contiguous — upper bound).
    {
        std::vector<float> out(L);
        // Use the first N-L candidates as a contiguous block.
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t a = 0; a < N; a++) {
            const uint8_t* anc = codes.data() + (size_t)a * 32;
            const uint8_t* blk = codes.data() + ((a+1) % (N-L)) * 32;
            v2_hoisted(table.data(), anc, blk, L, out.data());
            sink += out[0];
        }
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("v2b batched+hoist nc:   %.4fs → %.1f M/s (%.1f ns) [no copy, upper bound]\n", s, calls/s/1e6, s/calls*1e9);
    }
    (void)sink;
    return 0;
}
