// Test NEON SIMD gather for code_distance.
#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <arm_neon.h>

using namespace sextant;
static const uint32_t M = 32, K = 256, KK = K*K;

int main() {
    PqQuantizer q(MetricKind::L2Sq, 128, 32, 8);
    std::vector<float> train(50000 * 128);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    for (auto& v : train) v = u(rng);
    q.train(train.data(), 50000);

    const uint32_t N = 1000000;
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes((size_t)N * cs);
    std::vector<float> tmp(128);
    for (uint32_t i = 0; i < N; i++) {
        for (auto& v : tmp) v = u(rng);
        q.encode(tmp.data(), codes.data() + (size_t)i * cs);
    }
    std::vector<float> table((size_t)M * K * K);
    std::vector<uint8_t> ca8(cs,0), cb8(cs,0);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t a = 0; a < K; a++)
            for (uint32_t b = 0; b < K; b++) {
                ca8[s]=a; cb8[s]=b;
                table[s*KK+a*K+b] = q.code_distance(ca8.data(), cb8.data());
            }

    const uint32_t N_INSERTS = 100000, L_build = 128, pf = 12;
    std::vector<uint32_t> all_cands((size_t)N_INSERTS * L_build);
    std::vector<uint32_t> all_anchors(N_INSERTS);
    for (uint32_t i = 0; i < N_INSERTS; i++) {
        all_anchors[i] = rng() % N;
        for (uint32_t c = 0; c < L_build; c++)
            all_cands[i*L_build+c] = rng() % N;
    }
    double calls = (double)N_INSERTS * L_build;
    const float* tbl = table.data();

    // V1: scalar + dual pf.
    {
        volatile float sink=0; float acc=0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data()+(size_t)all_anchors[ins]*cs;
            const uint32_t* cands = &all_cands[ins*L_build];
            uint32_t ac[M];
            for (uint32_t s=0;s<M;s++) ac[s]=anchor[s];
            for (uint32_t c=0;c<L_build;c++) {
                if (c+24<L_build) __builtin_prefetch(codes.data()+(size_t)cands[c+24]*cs,0,0);
                if (c+12<L_build) __builtin_prefetch(codes.data()+(size_t)cands[c+12]*cs,0,0);
                const uint8_t* b = codes.data()+(size_t)cands[c]*cs;
                float d=0;
                for (uint32_t s=0;s<M;s++) d += tbl[s*KK+ac[s]*K+b[s]];
                acc += d;
            }
            if ((ins&0x3FF)==0) sink+=acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink+=acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("V1 scalar + dual-pf:  %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    // V2: NEON 4-wide accumulation (scalar loads, vector adds).
    {
        volatile float sink=0; float acc=0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data()+(size_t)all_anchors[ins]*cs;
            const uint32_t* cands = &all_cands[ins*L_build];
            uint32_t ac[M];
            for (uint32_t s=0;s<M;s++) ac[s]=anchor[s];
            for (uint32_t c=0;c<L_build;c++) {
                if (c+24<L_build) __builtin_prefetch(codes.data()+(size_t)cands[c+24]*cs,0,0);
                if (c+12<L_build) __builtin_prefetch(codes.data()+(size_t)cands[c+12]*cs,0,0);
                const uint8_t* b = codes.data()+(size_t)cands[c]*cs;
                float32x4_t vacc = vdupq_n_f32(0.0f);
                for (uint32_t g=0; g<M; g+=4) {
                    float vals[4] = {
                        tbl[g*KK+ac[g]*K+b[g]],
                        tbl[(g+1)*KK+ac[g+1]*K+b[g+1]],
                        tbl[(g+2)*KK+ac[g+2]*K+b[g+2]],
                        tbl[(g+3)*KK+ac[g+3]*K+b[g+3]]
                    };
                    vacc = vaddq_f32(vacc, vld1q_f32(vals));
                }
                acc += vaddvq_f32(vacc);
            }
            if ((ins&0x3FF)==0) sink+=acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink+=acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("V2 NEON4 + dual-pf:   %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    return 0;
}
