// Microbenchmark: preprocess_query cost under L2sq vs IP vs DecomposedL2sq,
// plus the raw per-call cost of simd::l2sq_f32 / simd::dot_f32 at small sub_dim.
//
// P1.4 of docs/plans/metric_per_tier_plan.md: verify preprocess_query has no
// scalar fallback and that the m*K = 24,576 tiny NumKong calls at sub_dim=8
// don't bottleneck on dispatch overhead. If they do, hoist to one nk_dot_f32
// per subspace × all K centroids (matvec).
//
// Build: ninja -C build-prof bench_preprocess_query
// Run:   ./build-prof/scripts/bench_preprocess_query
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <random>
#include "quant/pq_quantizer.hpp"
#include <numkong/numkong.h>

using namespace sextant;
using clk = std::chrono::steady_clock;

template <typename F>
static double time_us(F&& f, uint64_t iters) {
    auto t0 = clk::now();
    for (uint64_t i = 0; i < iters; i++) f();
    auto t1 = clk::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

int main() {
    const uint32_t dim = 768;
    const uint16_t m = 96;
    const uint8_t bits = 8;
    const uint64_t n_train = 5000;
    const uint64_t iters = 20000;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> train(n_train * dim);
    for (auto& v : train) v = nd(rng);

    // A query vector.
    std::vector<float> q(dim);
    for (auto& v : q) v = nd(rng);

    PqQuantizer q_l2(MetricKind::L2Sq, dim, m, bits);
    q_l2.train(train.data(), n_train);
    PqQuantizer q_ip(MetricKind::InnerProduct, dim, m, bits);
    q_ip.train(train.data(), n_train);

    const uint32_t lut_sz = q_l2.lut_size();
    std::vector<float> lut(lut_sz);

    printf("=== preprocess_query (dim=%u, m=%u, sub_dim=%u, K=%u) ===\n",
           dim, m, dim / m, q_l2.K());
    printf("LUT size: %u floats (%zu bytes)\n", lut_sz, lut_sz * sizeof(float));
    printf("Iterations: %llu\n\n", (unsigned long long)iters);

    const double us_l2 = time_us([&] { q_l2.preprocess_query(q.data(), lut.data()); }, iters);
    const double us_ip = time_us([&] { q_ip.preprocess_query(q.data(), lut.data()); }, iters);

    // Also measure preprocess_query_as explicitly to compare IP-via-L2sq-quantizer.
    const double us_l2_via_as = time_us(
        [&] { q_l2.preprocess_query_as(MetricKind::L2Sq, q.data(), lut.data()); }, iters);
    const double us_ip_via_as = time_us(
        [&] { q_l2.preprocess_query_as(MetricKind::InnerProduct, q.data(), lut.data()); }, iters);

    printf("preprocess_query L2sq:            %7.2f us/query  (%5.1f ns/LUT-entry)\n",
           us_l2, us_l2 * 1000.0 / (m * q_l2.K()));
    printf("preprocess_query IP:              %7.2f us/query  (%5.1f ns/LUT-entry)\n",
           us_ip, us_ip * 1000.0 / (m * q_ip.K()));
    printf("preprocess_query_as L2sq:         %7.2f us/query\n", us_l2_via_as);
    printf("preprocess_query_as IP:           %7.2f us/query\n", us_ip_via_as);
    printf("Ratio IP/L2sq:                    %.3f\n", us_ip / us_l2);
    printf("\n");

    // Per-call dispatch overhead probe: at sub_dim=8 each NumKong call is tiny.
    // Measure the raw nk_sqeuclidean_f32 vs nk_dot_f32 at sub_dim=8 to see if
    // dispatch overhead dominates the actual SIMD work.
    const uint32_t sub_dim = dim / m;
    std::vector<float> a(sub_dim), b(sub_dim);
    for (uint32_t i = 0; i < sub_dim; i++) { a[i] = nd(rng); b[i] = nd(rng); }
    const uint64_t tiny_iters = 1000000;
    double acc = 0;
    const double ns_l2sq_raw = [&]{
        auto t0 = clk::now();
        for (uint64_t i = 0; i < tiny_iters; i++) {
            nk_f64_t r; nk_sqeuclidean_f32(a.data(), b.data(), sub_dim, &r); acc += static_cast<double>(r);
        }
        auto t1 = clk::now();
        return std::chrono::duration<double, std::nano>(t1 - t0).count() / tiny_iters;
    }();
    const double ns_dot_raw = [&]{
        auto t0 = clk::now();
        for (uint64_t i = 0; i < tiny_iters; i++) {
            nk_f64_t r; nk_dot_f32(a.data(), b.data(), sub_dim, &r); acc += static_cast<double>(r);
        }
        auto t1 = clk::now();
        return std::chrono::duration<double, std::nano>(t1 - t0).count() / tiny_iters;
    }();
    // Prevent dead-code elimination of acc.
    if (acc == 0.0) printf("(acc=0 — surprising)\n");

    printf("=== raw NumKong per-call at sub_dim=%u ===\n", sub_dim);
    printf("nk_sqeuclidean_f32: %6.1f ns/call\n", ns_l2sq_raw);
    printf("nk_dot_f32:         %6.1f ns/call\n", ns_dot_raw);
    printf("\n");
    printf("=== interpretation ===\n");
    const double us_expected = m * q_l2.K() * ns_l2sq_raw / 1000.0;
    printf("If preprocess_query is dispatch-bound: %.2f us (= %u entries × %.1f ns)\n",
           us_expected, m * q_l2.K(), ns_l2sq_raw);
    printf("Measured L2sq:                        %.2f us\n", us_l2);
    printf("Ratio (measured/expected):            %.2f  (<1.0 = better than naive; >1.0 = overhead beyond dispatch)\n",
           us_l2 / us_expected);
}
