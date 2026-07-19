// Counts code_distance calls during a real robust_prune over realistic data.
#include "quant/pq_quantizer.hpp"
#include "sextant/types.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace sextant;

// We can't easily instrument the real code_distance without modifying it.
// Instead, replicate the exact robust_prune occlusion loop call pattern with
// the real L_build and max_occlusion params, and count.

int main() {
    const Dim dim = 128;
    const uint8_t m = 32, bits = 8;
    const uint32_t K = 1u << bits;

    const uint64_t n_train = 50000;
    std::vector<float> train(n_train * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    for (auto& v : train) v = u(rng);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits);
    q.train(train.data(), n_train);

    // Realistic candidate pool size from beam_search: ~L_build=128 candidates.
    // robust_prune caps at max_occlusion=750 (no effect here since 128 < 750).
    // Then: outer loop picks kept (up to R=64), inner checks the rest.
    const uint32_t L_build = 128;
    const uint32_t R = 64;
    const uint32_t cs = q.code_size();

    // Generate L_build random codes per "node", for many nodes.
    const uint32_t N_nodes = 10000;
    std::vector<uint8_t> codes(L_build * cs);
    std::vector<float> tmp(dim);

    // Count total code_distance calls in the occlusion loop pattern.
    uint64_t total_cd_calls = 0;
    for (uint32_t node = 0; node < N_nodes; node++) {
        // Generate L_build candidate codes for this node.
        for (uint32_t i = 0; i < L_build; i++) {
            for (auto& v : tmp) v = u(rng);
            q.encode(tmp.data(), codes.data() + i * cs);
        }
        // Sort by distance to node 0 (proxy for the anchor) — ascending.
        // (Real robust_prune sorts; we just count the occlusion pairs.)
        // Occlusion: for each kept p (up to R), check all pp > p not removed.
        // Worst case (none removed): sum_{p=0}^{R-1} (L_build - 1 - p)
        uint32_t kept = 0;
        std::vector<bool> removed(L_build, false);
        for (uint32_t p = 0; p < L_build && kept < R; p++) {
            if (removed[p]) continue;
            kept++;
            for (uint32_t pp = p + 1; pp < L_build; pp++) {
                if (removed[pp]) continue;
                total_cd_calls++;  // this is one code_distance call
                // Simulate occlusion: randomly remove some (alpha pruning).
                if ((rng() & 7) == 0) removed[pp] = true;
            }
        }
    }

    double avg_per_node = (double)total_cd_calls / N_nodes;
    printf("N_nodes=%u L_build=%u R=%u\n", N_nodes, L_build, R);
    printf("total code_distance calls: %llu\n", (unsigned long long)total_cd_calls);
    printf("avg per node: %.1f\n", avg_per_node);
    printf("\nProjected for 1M nodes: %.2f billion code_distance calls\n",
           avg_per_node * 1e6 / 1e9);
    printf("At 84 M calls/s (measured): %.1f seconds\n",
           avg_per_node * 1e6 / 84e6);

    return 0;
}
