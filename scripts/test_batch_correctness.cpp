// Correctness + perf: verify code_distance_batch4 matches scalar code_distance.
#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace sextant;

int main() {
    const uint16_t M = 96;
    PqQuantizer q(MetricKind::L2Sq, 768, M, 8);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);

    // Train
    std::vector<float> train(50000 * 768);
    for (auto& v : train) v = u(rng);
    q.train(train.data(), 50000);
    q.build_cross_distance_table();

    // Encode random vectors
    const uint32_t N = 20000;
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes((size_t)N * cs);
    std::vector<float> tmp(768);
    for (uint32_t i = 0; i < N; i++) {
        for (auto& v : tmp) v = u(rng);
        q.encode(tmp.data(), codes.data() + (size_t)i * cs);
    }

    // Correctness: compare batch4 vs scalar
    uint32_t checks = 0;
    float max_err = 0;
    for (uint32_t trial = 0; trial < 1000; trial++) {
        uint32_t ia = rng() % N;
        uint32_t ib0 = rng() % N, ib1 = rng() % N, ib2 = rng() % N, ib3 = rng() % N;
        const uint8_t* anchor = codes.data() + (size_t)ia * cs;
        float batch[4];
        q.code_distance_batch4(
            anchor,
            codes.data() + (size_t)ib0 * cs,
            codes.data() + (size_t)ib1 * cs,
            codes.data() + (size_t)ib2 * cs,
            codes.data() + (size_t)ib3 * cs,
            batch);
        float scalar[4] = {
            q.code_distance(anchor, codes.data() + (size_t)ib0 * cs),
            q.code_distance(anchor, codes.data() + (size_t)ib1 * cs),
            q.code_distance(anchor, codes.data() + (size_t)ib2 * cs),
            q.code_distance(anchor, codes.data() + (size_t)ib3 * cs),
        };
        for (int b = 0; b < 4; b++) {
            float err = std::abs(batch[b] - scalar[b]);
            max_err = std::max(max_err, err);
            checks++;
        }
    }
    printf("Correctness: %u checks, max_err = %.2e\n", checks, max_err);
    if (max_err > 1e-5f) {
        printf("FAIL: batch4 results don't match scalar!\n");
        return 1;
    }
    printf("PASS\n\n");

    // Performance: simulate robust_prune occlusion loop
    const uint32_t N_INSERTS = 10000, L_build = 200;
    std::vector<uint32_t> anchors(N_INSERTS);
    std::vector<uint32_t> cands((size_t)N_INSERTS * L_build);
    for (uint32_t i = 0; i < N_INSERTS; i++) {
        anchors[i] = rng() % N;
        for (uint32_t c = 0; c < L_build; c++)
            cands[i * L_build + c] = rng() % N;
    }
    const double calls = (double)N_INSERTS * L_build;

    // Scalar: one code_distance at a time
    {
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * cs;
            const uint32_t* csp = &cands[ins * L_build];
            for (uint32_t c = 0; c < L_build; c++) {
                acc += q.code_distance(anchor, codes.data() + (size_t)csp[c] * cs);
            }
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("Scalar code_distance:       %7.3fs (%5.1f ns/call)\n", s, s / calls * 1e9);
    }

    // Batch4
    {
        volatile float sink = 0;
        float acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t ins = 0; ins < N_INSERTS; ins++) {
            const uint8_t* anchor = codes.data() + (size_t)anchors[ins] * cs;
            const uint32_t* csp = &cands[ins * L_build];
            uint32_t c = 0;
            for (; c + 3 < L_build; c += 4) {
                float out[4];
                q.code_distance_batch4(
                    anchor,
                    codes.data() + (size_t)csp[c+0] * cs,
                    codes.data() + (size_t)csp[c+1] * cs,
                    codes.data() + (size_t)csp[c+2] * cs,
                    codes.data() + (size_t)csp[c+3] * cs,
                    out);
                acc += out[0] + out[1] + out[2] + out[3];
            }
            for (; c < L_build; c++)
                acc += q.code_distance(anchor, codes.data() + (size_t)csp[c] * cs);
            if ((ins & 0x3FF) == 0) sink += acc;
        }
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("Batch4 code_distance_batch4:%7.3fs (%5.1f ns/call)\n", s, s / calls * 1e9);
    }

    return 0;
}
