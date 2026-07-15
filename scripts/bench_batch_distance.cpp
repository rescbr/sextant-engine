// Benchmark: batch code_distance — process multiple candidates per anchor
// to increase load-level parallelism and amortize LUT building.
//
// The core insight: code_distance is gather-limited (1-2 load ports/cycle,
// data-dependent addresses). Processing 4 candidates at once gives the CPU
// 4× more independent loads to schedule, keeping both load ports saturated.
//
// Layouts tested:
//   V1: scalar lut_distance (current hot path) — 1 candidate at a time
//   V2: batch4 — 4 candidates, interleaved segment loads, NEON vadd
//   V3: batch4 — 4 candidates, per-candidate accumulation, NEON horizontal reduce
//   V4: scalar code_distance (SDC cross-table, no LUT) — the fallback path
#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <arm_neon.h>

using namespace sextant;

// m=96 (our arxiv-768 config), K=256 (8-bit codes)
static constexpr uint32_t M = 96, K = 256, KK = K * K;

int main() {
    // We don't need a trained quantizer — just the table structure.
    // Build a random cross_distance_table and random codes.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    // Cross-distance table [s][ca][cb] — 96 * 256 * 256 * 4 = 24 MB
    std::vector<float> table((size_t)M * KK);
    for (auto& v : table) v = u(rng);

    // Codes — 100K nodes, m=96 bytes each
    const uint32_t N = 100000;
    const uint32_t cs = M;  // 8-bit codes, 1 byte per segment
    std::vector<uint8_t> codes((size_t)N * cs);
    std::uniform_int_distribution<int> bu(0, 255);
    for (auto& c : codes) c = bu(rng);

    // Benchmark: 10K inserts, 200 candidates each (realistic L_build)
    const uint32_t N_INSERTS = 10000, L_build = 200;
    std::vector<uint32_t> anchors(N_INSERTS);
    std::vector<uint32_t> cands((size_t)N_INSERTS * L_build);
    for (uint32_t i = 0; i < N_INSERTS; i++) {
        anchors[i] = rng() % N;
        for (uint32_t c = 0; c < L_build; c++)
            cands[i * L_build + c] = rng() % N;
    }
    const double calls = (double)N_INSERTS * L_build;
    const float* tbl = table.data();

    // V1: per-anchor LUT + scalar lut_distance (current production path)
    {
        std::vector<float> lut((size_t)M * K);
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * cs;
            // Build LUT: lut[s*K + cb] = table[s*KK + anchor[s]*K + cb]
            for (uint32_t s = 0; s < M; s++)
                std::memcpy(lut.data() + s * K, tbl + s * KK + anchor[s] * K, K * sizeof(float));
            const uint32_t* csp = &cands[ins * L_build];
            for (uint32_t c = 0; c < L_build; c++) {
                const uint8_t* b = codes.data() + (size_t)csp[c] * cs;
                float d = 0;
                for (uint32_t s = 0; s < M; s++) d += lut[s * K + b[s]];
                acc += d;
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("V1 LUT scalar (m=%u):     %7.3fs (%5.1f ns/call)\n", M, s, s / calls * 1e9);
    }

    // V2: batch4 — process 4 candidates simultaneously, NEON accumulate.
    // For each segment s, load lut[s*K + b0[s]], lut[s*K + b1[s]],
    // lut[s*K + b2[s]], lut[s*K + b3[s]] — 4 independent loads that
    // the CPU can pipeline across its 2 load ports.
    {
        std::vector<float> lut((size_t)M * K);
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * cs;
            for (uint32_t s = 0; s < M; s++)
                std::memcpy(lut.data() + s * K, tbl + s * KK + anchor[s] * K, K * sizeof(float));
            const uint32_t* csp = &cands[ins * L_build];
            uint32_t c = 0;
            for (; c + 3 < L_build; c += 4) {
                const uint8_t* b0 = codes.data() + (size_t)csp[c+0] * cs;
                const uint8_t* b1 = codes.data() + (size_t)csp[c+1] * cs;
                const uint8_t* b2 = codes.data() + (size_t)csp[c+2] * cs;
                const uint8_t* b3 = codes.data() + (size_t)csp[c+3] * cs;
                float32x4_t vacc = vdupq_n_f32(0.0f);
                for (uint32_t s = 0; s < M; s++) {
                    const float* row = lut.data() + s * K;
                    float vals[4] = { row[b0[s]], row[b1[s]], row[b2[s]], row[b3[s]] };
                    vacc = vaddq_f32(vacc, vld1q_f32(vals));
                }
                // vacc = [d0, d1, d2, d3] — accumulate all 4
                acc += vaddvq_f32(vacc);
            }
            // Tail
            for (; c < L_build; c++) {
                const uint8_t* b = codes.data() + (size_t)csp[c] * cs;
                float d = 0;
                for (uint32_t s = 0; s < M; s++) d += lut[s * K + b[s]];
                acc += d;
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("V2 batch4 NEON (m=%u):    %7.3fs (%5.1f ns/call)\n", M, s, s / calls * 1e9);
    }

    // V3: scalar SDC code_distance (cross_distance_table, no LUT build)
    {
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * cs;
            const uint32_t* csp = &cands[ins * L_build];
            for (uint32_t c = 0; c < L_build; c++) {
                const uint8_t* b = codes.data() + (size_t)csp[c] * cs;
                float d = 0;
                for (uint32_t s = 0; s < M; s++)
                    d += tbl[s * KK + anchor[s] * K + b[s]];
                acc += d;
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("V3 SDC scalar (m=%u):     %7.3fs (%5.1f ns/call)\n", M, s, s / calls * 1e9);
    }

    // V4: batch4 SDC — 4 candidates, cross_distance_table directly (no LUT)
    {
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * cs;
            const uint32_t* csp = &cands[ins * L_build];
            uint32_t c = 0;
            for (; c + 3 < L_build; c += 4) {
                const uint8_t* b0 = codes.data() + (size_t)csp[c+0] * cs;
                const uint8_t* b1 = codes.data() + (size_t)csp[c+1] * cs;
                const uint8_t* b2 = codes.data() + (size_t)csp[c+2] * cs;
                const uint8_t* b3 = codes.data() + (size_t)csp[c+3] * cs;
                float32x4_t vacc = vdupq_n_f32(0.0f);
                for (uint32_t s = 0; s < M; s++) {
                    const float* row = tbl + s * KK + anchor[s] * K;
                    float vals[4] = { row[b0[s]], row[b1[s]], row[b2[s]], row[b3[s]] };
                    vacc = vaddq_f32(vacc, vld1q_f32(vals));
                }
                acc += vaddvq_f32(vacc);
            }
            for (; c < L_build; c++) {
                const uint8_t* b = codes.data() + (size_t)csp[c] * cs;
                float d = 0;
                for (uint32_t s = 0; s < M; s++)
                    d += tbl[s * KK + anchor[s] * K + b[s]];
                acc += d;
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("V4 batch4 SDC (m=%u):     %7.3fs (%5.1f ns/call)\n", M, s, s / calls * 1e9);
    }

    return 0;
}
