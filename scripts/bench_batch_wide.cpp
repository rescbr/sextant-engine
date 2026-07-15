// Benchmark: wider batch sizes + prefetch for code_distance
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <arm_neon.h>

static constexpr uint32_t M = 96, K = 256, KK = K * K;

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    std::vector<float> table((size_t)M * KK);
    for (auto& v : table) v = u(rng);

    const uint32_t N = 100000, code_size = M;
    std::vector<uint8_t> codes((size_t)N * code_size);
    std::uniform_int_distribution<int> bu(0, 255);
    for (auto& c : codes) c = bu(rng);

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

    // Helper: batchN SDC with NEON, variable batch width
    auto bench_batch = [&](int batch_w, const char* label) {
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * code_size;
            const uint32_t* csp = &cands[ins * L_build];
            uint32_t c = 0;
            for (; c + batch_w <= L_build; c += batch_w) {
                const uint8_t* bs[8];
                for (int b = 0; b < batch_w; b++)
                    bs[b] = codes.data() + (size_t)csp[c + b] * code_size;
                // Process segments in groups of 4 for NEON
                float32x4_t vacc = vdupq_n_f32(0.0f);  // only works for batch_w<=4
                if (batch_w == 4) {
                    for (uint32_t s = 0; s < M; s++) {
                        const float* row = tbl + s * KK + anchor[s] * K;
                        float vals[4] = { row[bs[0][s]], row[bs[1][s]], row[bs[2][s]], row[bs[3][s]] };
                        vacc = vaddq_f32(vacc, vld1q_f32(vals));
                    }
                    acc += vaddvq_f32(vacc);
                } else {
                    // Scalar fallback for batch_w != 4
                    for (int b = 0; b < batch_w; b++) {
                        float d = 0;
                        for (uint32_t s = 0; s < M; s++)
                            d += tbl[s * KK + anchor[s] * K + bs[b][s]];
                        acc += d;
                    }
                }
            }
            for (; c < L_build; c++) {
                const uint8_t* b = codes.data() + (size_t)csp[c] * code_size;
                float d = 0;
                for (uint32_t s = 0; s < M; s++) d += tbl[s * KK + anchor[s] * K + b[s]];
                acc += d;
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("%-30s %7.3fs (%5.1f ns/call)\n", label, s, s / calls * 1e9);
    };

    // V4 batch4 SDC — baseline
    bench_batch(4, "V4 batch4 SDC:");

    // V5 batch2 SDC — less ILP
    bench_batch(2, "V5 batch2 SDC (scalar):");

    // V6 batch4 SDC + prefetch next batch's codes
    {
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * code_size;
            const uint32_t* csp = &cands[ins * L_build];
            // Prefetch first batch
            for (int b = 0; b < 4; b++)
                __builtin_prefetch(codes.data() + (size_t)csp[b] * code_size, 0, 0);
            uint32_t c = 0;
            for (; c + 7 < L_build; c += 4) {
                // Prefetch next batch
                for (int b = 0; b < 4; b++)
                    __builtin_prefetch(codes.data() + (size_t)csp[c + 4 + b] * code_size, 0, 0);
                const uint8_t* b0 = codes.data() + (size_t)csp[c+0] * code_size;
                const uint8_t* b1 = codes.data() + (size_t)csp[c+1] * code_size;
                const uint8_t* b2 = codes.data() + (size_t)csp[c+2] * code_size;
                const uint8_t* b3 = codes.data() + (size_t)csp[c+3] * code_size;
                float32x4_t vacc = vdupq_n_f32(0.0f);
                for (uint32_t s = 0; s < M; s++) {
                    const float* row = tbl + s * KK + anchor[s] * K;
                    float vals[4] = { row[b0[s]], row[b1[s]], row[b2[s]], row[b3[s]] };
                    vacc = vaddq_f32(vacc, vld1q_f32(vals));
                }
                acc += vaddvq_f32(vacc);
            }
            for (; c < L_build; c++) {
                const uint8_t* b = codes.data() + (size_t)csp[c] * code_size;
                float d = 0;
                for (uint32_t s = 0; s < M; s++) d += tbl[s * KK + anchor[s] * K + b[s]];
                acc += d;
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("%-30s %7.3fs (%5.1f ns/call)\n", "V6 batch4 SDC + pf:", s, s / calls * 1e9);
    }

    // V7: batch4 SDC + 8-wide loop unroll (process 2 batches of 4 per iteration)
    // This gives 8 independent load streams
    {
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * code_size;
            const uint32_t* csp = &cands[ins * L_build];
            uint32_t c = 0;
            for (; c + 7 < L_build; c += 8) {
                const uint8_t* bs[8];
                for (int b = 0; b < 8; b++)
                    bs[b] = codes.data() + (size_t)csp[c + b] * code_size;
                float32x4_t vacc0 = vdupq_n_f32(0.0f);
                float32x4_t vacc1 = vdupq_n_f32(0.0f);
                for (uint32_t s = 0; s < M; s++) {
                    const float* row = tbl + s * KK + anchor[s] * K;
                    float vals0[4] = { row[bs[0][s]], row[bs[1][s]], row[bs[2][s]], row[bs[3][s]] };
                    float vals1[4] = { row[bs[4][s]], row[bs[5][s]], row[bs[6][s]], row[bs[7][s]] };
                    vacc0 = vaddq_f32(vacc0, vld1q_f32(vals0));
                    vacc1 = vaddq_f32(vacc1, vld1q_f32(vals1));
                }
                acc += vaddvq_f32(vacc0) + vaddvq_f32(vacc1);
            }
            for (; c < L_build; c++) {
                const uint8_t* b = codes.data() + (size_t)csp[c] * code_size;
                float d = 0;
                for (uint32_t s = 0; s < M; s++) d += tbl[s * KK + anchor[s] * K + b[s]];
                acc += d;
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("%-30s %7.3fs (%5.1f ns/call)\n", "V7 batch8 SDC (2xNEON):", s, s / calls * 1e9);
    }

    return 0;
}
