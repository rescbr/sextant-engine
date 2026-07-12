// PartitionInfo: k-means partitioning on PQ codes (Step 11).
//
// K-means runs entirely on PQ codes using SDC distance (code-to-code via the
// quantizer's cross-distance table). Centroids are represented as PQ codes.
// Assignment uses closure_factor overlap so boundary vectors land in multiple
// shards, keeping the merged graph connected.

#include "partition.hpp"
#include "quant/pq_quantizer.hpp"
#include "sextant/error.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace sextant {

namespace {

/// Compute SDC distance from a code to all K centroids.
/// Returns the nearest distance in `best_d` (if non-null) and fills `dists`
/// (length K) with all centroid distances.
inline void centroid_distances(const PqQuantizer& q,
                               const uint8_t* code,
                               const std::vector<std::vector<uint8_t>>& centroids,
                               uint32_t code_size, float* dists,
                               float* best_d = nullptr) {
    const uint32_t K = static_cast<uint32_t>(centroids.size());
    float best = std::numeric_limits<float>::max();
    for (uint32_t k = 0; k < K; k++) {
        const float d = q.code_distance(code, centroids[k].data());
        dists[k] = d;
        if (d < best) best = d;
    }
    if (best_d) *best_d = best;
}

}  // namespace

PartitionAssignment partition_codes(const PqQuantizer& quantizer,
                                     const uint8_t* codes, uint32_t n,
                                     uint32_t code_size, uint32_t K,
                                     float closure_factor,
                                     uint32_t iterations, uint64_t seed) {
    if (K == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "partition_codes: K must be > 0");
    }
    if (n == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "partition_codes: n must be > 0");
    }
    // If K >= n, each vector is its own shard (degenerate but valid).
    K = std::min<uint32_t>(K, n);

    std::mt19937_64 rng(seed);

    // --- Initialize centroids: distinct random codes (k-means++-ish via
    //     random distinct picks; k-means refinement handles quality). ---
    std::vector<std::vector<uint8_t>> centroids(K,
        std::vector<uint8_t>(code_size, 0));
    {
        std::vector<uint32_t> picks;
        picks.reserve(K);
        std::uniform_int_distribution<uint32_t> dist(0, n - 1);
        while (picks.size() < K) {
            const uint32_t idx = dist(rng);
            if (std::find(picks.begin(), picks.end(), idx) == picks.end()) {
                picks.push_back(idx);
            }
        }
        for (uint32_t k = 0; k < K; k++) {
            std::memcpy(centroids[k].data(),
                        codes + static_cast<size_t>(picks[k]) * code_size,
                        code_size);
        }
    }

    spdlog::info("[sextant] partition: k-means on {} codes, K={}, {} iters",
                 n, K, iterations);

    std::vector<float> dists(K);
    std::vector<std::vector<uint32_t>> clusters(K);

    for (uint32_t iter = 0; iter < iterations; iter++) {
        for (auto& c : clusters) c.clear();

        // Assignment pass: nearest centroid (hard, no overlap during k-means).
        for (uint32_t i = 0; i < n; i++) {
            const uint8_t* code =
                codes + static_cast<size_t>(i) * code_size;
            centroid_distances(quantizer, code, centroids, code_size,
                               dists.data());
            uint32_t best_k = 0;
            float best_d = dists[0];
            for (uint32_t k = 1; k < K; k++) {
                if (dists[k] < best_d) {
                    best_d = dists[k];
                    best_k = k;
                }
            }
            clusters[best_k].push_back(i);
        }

        // Update pass: new centroid = element-wise medoid (the member whose
        // sum of SDC distances to all other members is smallest). For PQ codes
        // a true mean isn't defined, so medoid is the natural choice.
        for (uint32_t k = 0; k < K; k++) {
            if (clusters[k].empty()) {
                // Reseed an empty cluster from a random point.
                std::uniform_int_distribution<uint32_t> d(0, n - 1);
                std::memcpy(centroids[k].data(),
                            codes + static_cast<size_t>(d(rng)) * code_size,
                            code_size);
                continue;
            }
            // Medoid: minimize sum of distances. For small clusters this is
            // O(|cluster|^2) code-distance evals; clusters are ~n/K each.
            // To bound cost for large clusters, subsample candidates to 256.
            auto& cl = clusters[k];
            const uint32_t cand_count =
                static_cast<uint32_t>(std::min<size_t>(256, cl.size()));
            std::vector<uint32_t> cand_idx(cand_count);
            for (uint32_t c = 0; c < cand_count; c++) {
                cand_idx[c] = cl[static_cast<size_t>(c) * cl.size() /
                                  cand_count];
            }
            uint32_t best_medoid = cand_idx[0];
            float best_cost = std::numeric_limits<float>::max();
            for (uint32_t ci = 0; ci < cand_count; ci++) {
                const uint8_t* cand =
                    codes + static_cast<size_t>(cand_idx[ci]) * code_size;
                float cost = 0.0f;
                for (uint32_t mi : cl) {
                    const uint8_t* mem =
                        codes + static_cast<size_t>(mi) * code_size;
                    cost += quantizer.code_distance(cand, mem);
                }
                if (cost < best_cost) {
                    best_cost = cost;
                    best_medoid = cand_idx[ci];
                }
            }
            std::memcpy(centroids[k].data(),
                        codes + static_cast<size_t>(best_medoid) * code_size,
                        code_size);
        }
    }

    // --- Final assignment WITH closure_factor overlap ---
    // Each vector joins its nearest centroid AND any centroid within
    // closure_factor × d_best. This produces ~15% replication (Issue 24).
    PartitionAssignment pa;
    pa.shards.assign(K, {});
    pa.centroids = centroids;
    pa.closure_factor = closure_factor;

    uint64_t total_assignments = 0;
    uint32_t max_shard = 0, min_shard = UINT32_MAX;

    for (uint32_t i = 0; i < n; i++) {
        const uint8_t* code =
            codes + static_cast<size_t>(i) * code_size;
        float best_d;
        centroid_distances(quantizer, code, centroids, code_size,
                           dists.data(), &best_d);
        const float threshold = closure_factor * best_d;
        for (uint32_t k = 0; k < K; k++) {
            if (dists[k] <= threshold) {
                pa.shards[k].push_back(i);
                total_assignments++;
            }
        }
    }

    for (uint32_t k = 0; k < K; k++) {
        max_shard = std::max(max_shard,
                             static_cast<uint32_t>(pa.shards[k].size()));
        min_shard = std::min(min_shard,
                             static_cast<uint32_t>(pa.shards[k].size()));
    }
    const double replication =
        n > 0 ? static_cast<double>(total_assignments) / n : 0.0;
    spdlog::info("[sextant] partition: K={}, total assignments={} "
                 "(replication {:.3f}), shard sizes min={} max={}",
                 K, total_assignments, replication, min_shard, max_shard);

    return pa;
}

}  // namespace sextant
