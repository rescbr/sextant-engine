// Test a duplicated [ca][cb][s] transposed table for SIMD-friendly gather.
// For a fixed anchor ca and candidate cb, the 32 segment distances are
// CONTIGUOUS → 2 NEON ld1 q loads (16 floats each) instead of 32 scattered.
//
// Original table: [s][ca][cb] = 8MB (used by build_code_lut/lut_distance)
// Transposed copy: [ca][cb][s] = 8MB (used by robust_prune occlusion)
// Total: 16MB (doubles memory, both stay in L2 on Apple M-series 12-16MB L2)
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

    // Original table [s][ca][cb].
    std::vector<float> table((size_t)M * K * K);
    std::vector<uint8_t> ca8(cs,0), cb8(cs,0);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t a = 0; a < K; a++)
            for (uint32_t b = 0; b < K; b++) {
                ca8[s]=a; cb8[s]=b;
                table[s*KK+a*K+b] = q.code_distance(ca8.data(), cb8.data());
            }

    // Transposed table [ca][cb][s] — for SIMD contiguous gather.
    // table_t[ca*K*K + cb*M + s] = table[s*KK + ca*K + cb]
    std::vector<float> table_t((size_t)K * K * M);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t a = 0; a < K; a++)
            for (uint32_t b = 0; b < K; b++)
                table_t[(size_t)a*K*M + b*M + s] = table[s*KK+a*K+b];

    const uint32_t N_INSERTS = 100000, L_build = 128;
    std::vector<uint32_t> all_cands((size_t)N_INSERTS * L_build);
    std::vector<uint32_t> all_anchors(N_INSERTS);
    for (uint32_t i = 0; i < N_INSERTS; i++) {
        all_anchors[i] = rng() % N;
        for (uint32_t c = 0; c < L_build; c++)
            all_cands[i*L_build+c] = rng() % N;
    }
    double calls = (double)N_INSERTS * L_build;
    const float* tbl = table.data();
    const float* tbl_t = table_t.data();

    // V1: original scalar [s][ca][cb] (current code_distance pattern).
    {
        volatile float sink=0; float acc=0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data()+(size_t)all_anchors[ins]*cs;
            const uint32_t* cands = &all_cands[ins*L_build];
            uint32_t ac[M];
            for (uint32_t s=0;s<M;s++) ac[s]=anchor[s];
            for (uint32_t c=0;c<L_build;c++) {
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
        printf("V1 scalar [s][ca][cb]:      %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    // V2: transposed [ca][cb][s] with NEON — 2 × ld1q (16 floats) + fadd.
    //     For fixed anchor: precompute the 32 (ca_s) → base offsets into tbl_t.
    //     addr = ac[s]*K*M + cb_s*M + s. But cb_s varies per candidate.
    //     So per candidate: load cb[0..31] (32 bytes), then for each s compute
    //     ac[s]*K*M + cb[s]*M → that's the base, then the 32 floats are at
    //     base+0..31. BUT the base depends on BOTH ac[s] AND cb[s] per segment,
    //     so they're NOT contiguous in the [ca][cb][s] layout!
    //
    //     Wait — [ca][cb][s] means for FIXED ca and cb, the 32 s-values are
    //     contiguous. But ca varies per segment (ac[0..31] are different).
    //     So the 32 values are at DIFFERENT ca bases → NOT contiguous. Oops.
    //
    //     The right layout for "fixed anchor, varying candidate" is:
    //     [s][ca][cb] (original) — for fixed ca_s per segment, cb_s varies
    //     within a 1KB row. That's already as good as it gets for this access.
    //
    //     The [ca][cb][s] layout helps when BOTH ca and cb are fixed — i.e.
    //     computing ONE pair's distance. But robust_prune loops over many cb
    //     for a fixed anchor (ca fixed per segment). So [s][ca][cb] is correct.
    //
    //     Let me test [ca][cb][s] anyway to confirm the theory (it should be
    //     SLOWER because it breaks the per-segment row locality).
    {
        volatile float sink=0; float acc=0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data()+(size_t)all_anchors[ins]*cs;
            const uint32_t* cands = &all_cands[ins*L_build];
            uint32_t ac[M];
            for (uint32_t s=0;s<M;s++) ac[s]=anchor[s];
            for (uint32_t c=0;c<L_build;c++) {
                const uint8_t* b = codes.data()+(size_t)cands[c]*cs;
                float d=0;
                for (uint32_t s=0;s<M;s++)
                    d += tbl_t[(size_t)ac[s]*K*M + b[s]*M + s];
                acc += d;
            }
            if ((ins&0x3FF)==0) sink+=acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink+=acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("V2 scalar [ca][cb][s]:      %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    // V3: per-anchor LUT (build_code_lut) + lut_distance.
    //     This IS the right structure: for fixed anchor, lut[s*K+cb] is
    //     contiguous per segment (1KB rows). 32 reads from 8KB, L1-resident.
    //     The cost is building the LUT (32KB memcpy) once per anchor.
    {
        std::vector<float> lut((size_t)M * K);
        volatile float sink=0; float acc=0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data()+(size_t)all_anchors[ins]*cs;
            const uint32_t* cands = &all_cands[ins*L_build];
            // Build LUT: lut[s*K+cb] = table[s*KK + anchor[s]*K + cb]
            for (uint32_t s=0;s<M;s++) {
                const float* src = tbl + s*KK + anchor[s]*K;
                float* dst = lut.data() + s*K;
                __builtin_memcpy(dst, src, K*sizeof(float));
            }
            for (uint32_t c=0;c<L_build;c++) {
                const uint8_t* b = codes.data()+(size_t)cands[c]*cs;
                float d=0;
                for (uint32_t s=0;s<M;s++) d += lut[s*K+b[s]];
                acc += d;
            }
            if ((ins&0x3FF)==0) sink+=acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink+=acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("V3 per-anchor LUT + gather: %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    return 0;
}
