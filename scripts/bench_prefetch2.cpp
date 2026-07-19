// Test prefetching the cross_distance_table rows in addition to candidate codes.
// code_distance does 32 scattered reads into the 8MB table (one per segment,
// each segment is 256KB). If those miss, prefetching the row for the next
// candidate could help.

#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace sextant;
static const uint32_t M = 32, K = 256, KK = K*K;

// Access the table directly (exposed via friend or replicate).
// We reconstruct the table via canonical codes.
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

    // Reconstruct the cross-distance table.
    std::vector<float> table((size_t)M * K * K);
    std::vector<uint8_t> ca(cs, 0), cb(cs, 0);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t a = 0; a < K; a++)
            for (uint32_t b = 0; b < K; b++) {
                ca[s] = a; cb[s] = b;
                table[s*KK + a*K + b] = q.code_distance(ca.data(), cb.data());
            }

    const uint32_t L_build = 128;
    std::vector<uint32_t> cand_rowids(L_build);
    for (auto& c : cand_rowids) c = rng() % N;
    const uint8_t* anchor = codes.data() + (rng() % N) * cs;

    // Hoisted anchor centroids (for inline gather).
    uint32_t ac[M];
    for (uint32_t s = 0; s < M; s++) ac[s] = anchor[s];

    const uint32_t REPEAT = 50000;
    double calls = (double)REPEAT * L_build;

    // V1: no prefetch (inline gather, hoisted anchor).
    {
        volatile float sink = 0; float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < REPEAT; r++) {
            for (uint32_t i = 0; i < L_build; i++) {
                const uint8_t* b = codes.data() + (size_t)cand_rowids[i] * cs;
                float d = 0;
                for (uint32_t s = 0; s < M; s++)
                    d += table[s*KK + ac[s]*K + b[s]];
                acc += d;
            }
            if ((r & 0xFFF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("v1 inline no-pf:       %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    // V2: prefetch candidate code only (dist=12).
    {
        volatile float sink = 0; float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < REPEAT; r++) {
            for (uint32_t i = 0; i < L_build; i++) {
                if (i + 12 < L_build)
                    __builtin_prefetch(codes.data() + (size_t)cand_rowids[i+12]*cs, 0, 0);
                const uint8_t* b = codes.data() + (size_t)cand_rowids[i] * cs;
                float d = 0;
                for (uint32_t s = 0; s < M; s++)
                    d += table[s*KK + ac[s]*K + b[s]];
                acc += d;
            }
            if ((r & 0xFFF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("v2 pf cand(dist=12):   %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    // V3: prefetch candidate code + prefetch the next candidate's first few
    //     table rows (the table addresses depend on the next cand's bytes,
    //     which we can't know without loading the next code — so prefetch the
    //     code first, then we can't prefetch table rows until the code lands).
    //     Instead, prefetch the CURRENT candidate's table rows after loading
    //     its code but before the gather — no, that's what the gather does.
    //     Skip: can't prefetch table rows ahead without the code bytes.
    //
    // V3b: prefetch the anchor's table rows into L1 once (they're reused
    //      across all candidates). The anchor's 32 rows (one per segment)
    //      are 32 × 1KB = 32KB total. If these stay in L1, the gather is fast.
    {
        // Touch the anchor's 32 rows to warm them.
        for (uint32_t s = 0; s < M; s++) {
            volatile float x = table[s*KK + ac[s]*K];
            (void)x;
        }
        volatile float sink = 0; float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < REPEAT; r++) {
            for (uint32_t i = 0; i < L_build; i++) {
                if (i + 12 < L_build)
                    __builtin_prefetch(codes.data() + (size_t)cand_rowids[i+12]*cs, 0, 0);
                const uint8_t* b = codes.data() + (size_t)cand_rowids[i] * cs;
                float d = 0;
                for (uint32_t s = 0; s < M; s++)
                    d += table[s*KK + ac[s]*K + b[s]];
                acc += d;
            }
            if ((r & 0xFFF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1-t0).count();
        printf("v3 pf cand + warm anc: %.4fs (%.1f ns/call)\n", s, s/calls*1e9);
    }
    return 0;
}
