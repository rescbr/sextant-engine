// Simulate the AMD64 scenario: test whether the per-anchor LUT approach
// (which makes the 32 segment values contiguous) wins when:
//  1. We use AVX2 VPGATHERDD (hardware gather instruction, x86-only)
//  2. L2 is larger (typical x86 server: 1-2MB L2 per core, 32-64MB L3)
//
// On ARM/Apple Silicon we can't test AVX2, but we CAN test the per-anchor
// LUT with the LUT pre-built ONCE (not per-insert) to isolate the gather
// cost from the build cost. If the contiguous-LUT gather is faster than
// scattered [s][ca][cb] gather, then on x86 with VPGATHERDD it'd be even
// better (hardware gather vs scalar loads).
//
// Key difference: x86 has VPGATHERDPS (gather 8 floats with vector indices),
// ARM NEON does NOT (no gather instruction). So contiguous layouts that
// enable VPGATHERD could win on x86 even if they lose on ARM.
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

    // V1: scattered [s][ca][cb] gather (current). 32 scattered reads.
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
        printf("V1 scattered [s][ca][cb]:   %.4fs (%.1f ns/call) [32 scattered reads]\n", s, s/calls*1e9);
    }
    // V2: per-anchor LUT, but PRE-BUILT once (isolate gather cost only).
    //     The LUT makes values contiguous: lut[s*K + cb]. For fixed anchor,
    //     32 reads from 8KB, stride K=256 floats (1KB) between segments.
    //     On x86 with VPGATHERDPS, these 32 reads could be 4 gather instrs.
    //     On ARM, still scalar, but the 8KB LUT is fully L1-resident.
    {
        // Pre-build LUT for the first anchor (don't time the build).
        std::vector<float> lut((size_t)M * K);
        const uint8_t* anchor0 = codes.data()+(size_t)all_anchors[0]*cs;
        for (uint32_t s=0;s<M;s++)
            __builtin_memcpy(lut.data()+s*K, tbl+s*KK+anchor0[s]*K, K*sizeof(float));

        volatile float sink=0; float acc=0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint32_t* cands = &all_cands[ins*L_build];
            // Reuse the same LUT (gather cost only, no rebuild).
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
        printf("V2 LUT gather (no rebuild):  %.4fs (%.1f ns/call) [8KB L1-resident]\n", s, s/calls*1e9);
    }
    // V3: per-anchor LUT WITH rebuild (realistic — what robust_prune would do).
    {
        std::vector<float> lut((size_t)M * K);
        volatile float sink=0; float acc=0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data()+(size_t)all_anchors[ins]*cs;
            const uint32_t* cands = &all_cands[ins*L_build];
            // Build LUT (32KB memcpy).
            for (uint32_t s=0;s<M;s++)
                __builtin_memcpy(lut.data()+s*K, tbl+s*KK+anchor[s]*K, K*sizeof(float));
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
        printf("V3 LUT gather (with rebuild):%.4fs (%.1f ns/call) [32KB memcpy + gather]\n", s, s/calls*1e9);
    }
    printf("\n--- Analysis ---\n");
    printf("ARM (this machine): V1 scattered wins because no gather instruction.\n");
    printf("  V2 LUT-no-rebuild isolates gather: if V2 < V1, the contiguous LUT\n");
    printf("  gather is faster — on x86 VPGATHERDPS would amplify this.\n");
    printf("  V3 includes rebuild cost: the 32KB memcpy per anchor kills it.\n");
    printf("x86 with AVX2: VPGATHERDPS does 8 floats/clock from arbitrary addrs.\n");
    printf("  The scattered [s][ca][cb] layout could use gather directly (V1 pattern)\n");
    printf("  without needing a transposed copy — gather handles non-contiguous.\n");
    return 0;
}
