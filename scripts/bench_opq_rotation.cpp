// Microbenchmark: per-query rotation matmul cost via PqQuantizer.
// Measures preprocess_query() with vs without OPQ rotation to isolate the
// rotation overhead on the search critical path.
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <random>
#include "quant/pq_quantizer.hpp"

using namespace sextant;
using clk = std::chrono::steady_clock;

int main() {
    const uint32_t dim = 768;
    const uint16_t m = 96;
    const uint8_t bits = 8;
    const uint64_t n_train = 5000;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> train(n_train * dim);
    for (auto& v : train) v = nd(rng);

    // Train baseline (no rotation).
    PqQuantizer q_base(MetricKind::L2Sq, dim, m, bits);
    q_base.train(train.data(), n_train);

    // Train OPQ.
    PqQuantizer q_opq(MetricKind::L2Sq, dim, m, bits);
    q_opq.enable_opq();
    q_opq.train(train.data(), n_train);
    if (!q_opq.has_rotation()) {
        fprintf(stderr, "OPQ did not compute rotation\n");
        return 1;
    }

    const uint32_t lut_size = q_base.lut_size();
    std::vector<float> lut(lut_size);

    // Query vectors.
    const uint32_t N = 5000;
    std::vector<float> queries(N * dim);
    for (auto& v : queries) v = nd(rng);

    // Warm up.
    for (uint32_t i = 0; i < 200; i++)
        q_base.preprocess_query(queries.data() + (i % N) * dim, lut.data());
    for (uint32_t i = 0; i < 200; i++)
        q_opq.preprocess_query(queries.data() + (i % N) * dim, lut.data());

    // Time baseline preprocess_query.
    auto t0 = clk::now();
    for (uint32_t i = 0; i < N; i++)
        q_base.preprocess_query(queries.data() + i * dim, lut.data());
    auto t1 = clk::now();
    // Time OPQ preprocess_query (includes rotation).
    for (uint32_t i = 0; i < N; i++)
        q_opq.preprocess_query(queries.data() + i * dim, lut.data());
    auto t2 = clk::now();

    double base_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    double opq_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / N;
    double rot_us = opq_us - base_us;
    printf("dim=%u m=%u\n", dim, m);
    printf("preprocess_query baseline:  %7.1f us/query\n", base_us);
    printf("preprocess_query OPQ:       %7.1f us/query\n", opq_us);
    printf("rotation overhead (delta):  %7.1f us/query  (%.3f ms)\n",
           rot_us, rot_us / 1000.0);
    printf("target: < 500 us.  %s\n", rot_us < 500.0 ? "PASS" : "FAIL");

    // Also time encode (build-path rotation).
    std::vector<uint8_t> code(q_base.code_size());
    t0 = clk::now();
    for (uint32_t i = 0; i < N; i++)
        q_opq.encode(queries.data() + i * dim, code.data());
    t1 = clk::now();
    double enc_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    printf("OPQ encode (rot + lookup):  %7.1f us/vector\n", enc_us);
    return 0;
}
