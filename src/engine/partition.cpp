// PartitionInfo: k-means partitioning on PQ codes (Step 11).
//
// K-means runs entirely on PQ codes using PQ code distance (code-to-code via the
// quantizer's cross-distance table). Centroids are represented as PQ codes.
// Assignment uses closure_factor overlap so boundary vectors land in multiple
// shards, keeping the merged graph connected.
//
// Parallelism: the assignment passes (nearest-centroid for all N codes) and
// the medoid update (over the K clusters) are both embarrassingly parallel.
// At 1.34M codes the serial version left 7/8 cores idle for the whole
// partition step; this version uses a flat assignment array (no per-thread
// vector merging) + ctpl work-stealing, matching the builder's pattern.

#include "partition.hpp"
#include "quant/pq_quantizer.hpp"
#include "sextant/error.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace sextant {

namespace {

/// Trivial TLS marker for the partition pool (workers carry no per-thread
/// state; each task is self-contained).
struct PartWorkerState {};

/// Compute PQ code distance from a code to all K centroids.
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
                                     uint32_t iterations, uint32_t num_threads,
                                     uint64_t seed, float balance_factor) {
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
    num_threads = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
    num_threads = std::max(1u, std::min(num_threads, n));

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

    spdlog::info("[sextant] partition: k-means on {} codes, K={}, {} iters, "
                 "{} threads", n, K, iterations, num_threads);

    // Flat assignment array (parallel-written, no contention). Reused across
    // iterations; the medoid pass bucketizes it into per-cluster lists.
    std::vector<uint32_t> assign(n, 0);
    // Scratch for the assignment-pass distance computation (per-thread, via
    // the pool's TLS slot would be cleaner but the K-sized buffer is tiny —
    // ~256 floats — so per-thread heap is fine).
    const uint32_t lut_stride = K;  // dists is K floats per code

    ctpl::thread_pool_tls<PartWorkerState> pool(num_threads);

    // Chunked work-stealing over the N codes (matches the builder's pattern).
    constexpr uint32_t kChunk = 512;
    auto parallel_assign = [&](void) {
        std::atomic<uint32_t> next_id{0};
        std::vector<std::future<void>> futs;
        futs.reserve(num_threads);
        for (uint32_t t = 0; t < num_threads; t++) {
            futs.push_back(pool.push(
                [&quantizer, &centroids, codes, n, code_size, K, lut_stride,
                 &assign, &next_id]
                (size_t /*id*/, PartWorkerState& /*w*/) {
                    std::vector<float> dists(K);
                    while (true) {
                        const uint32_t lo = next_id.fetch_add(kChunk,
                                                              std::memory_order_relaxed);
                        if (lo >= n) break;
                        const uint32_t hi = std::min(lo + kChunk, n);
                        for (uint32_t i = lo; i < hi; i++) {
                            const uint8_t* code =
                                codes + static_cast<size_t>(i) * code_size;
                            centroid_distances(quantizer, code, centroids,
                                               code_size, dists.data());
                            uint32_t best_k = 0;
                            float best_d = dists[0];
                            for (uint32_t k = 1; k < K; k++) {
                                if (dists[k] < best_d) {
                                    best_d = dists[k];
                                    best_k = k;
                                }
                            }
                            assign[i] = best_k;
                        }
                    }
                }));
        }
        for (auto& f : futs) f.get();
        (void)lut_stride;
    };

    for (uint32_t iter = 0; iter < iterations; iter++) {
        // --- Assignment pass (parallel): assign[i] = nearest centroid ---
        parallel_assign();

        // --- Bucket assign[] into per-cluster lists (serial, cheap: O(n)) ---
        std::vector<std::vector<uint32_t>> clusters(K);
        for (uint32_t i = 0; i < n; i++) {
            clusters[assign[i]].push_back(i);
        }

        // --- Update pass: medoid per cluster (parallel over K clusters) ---
        // Each cluster's medoid search is independent. K is small (16-512),
        // so we fan out across clusters.
        std::atomic<uint32_t> next_cluster{0};
        std::vector<std::future<void>> futs;
        futs.reserve(num_threads);
        for (uint32_t t = 0; t < num_threads; t++) {
            futs.push_back(pool.push(
                [&quantizer, &centroids, &clusters, codes, code_size, n, K,
                 &next_cluster, &rng]
                (size_t id, PartWorkerState& /*w*/) {
                    // Each worker pulls cluster IDs from the shared counter.
                    // Reseed draws need a per-worker RNG to stay deterministic-
                    // ish (the empty-cluster case is rare and the exact reseed
                    // point doesn't affect partition quality).
                    std::mt19937_64 local_rng(rng() + id);
                    while (true) {
                        const uint32_t k = next_cluster.fetch_add(
                            1, std::memory_order_relaxed);
                        if (k >= K) break;
                        auto& cl = clusters[k];
                        if (cl.empty()) {
                            // Reseed an empty cluster from a random point.
                            std::uniform_int_distribution<uint32_t> d(0, n - 1);
                            std::memcpy(centroids[k].data(),
                                        codes + static_cast<size_t>(d(local_rng)) * code_size,
                                        code_size);
                            continue;
                        }
                        // Medoid: minimize sum of distances. Subsample
                        // candidates to 256 to bound cost for large clusters.
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
                }));
        }
        for (auto& f : futs) f.get();
    }

    // --- Final assignment WITH closure_factor overlap (parallel) ---
    // Each vector joins its nearest centroid AND any centroid within
    // closure_factor × d_best. This produces ~15% replication (Issue 24).
    // Each code may land in multiple shards → per-thread local shard buffers,
    // concatenated at the end (avoids lock contention on shared vectors).
    //
    // When balance_factor > 0 (SPANN-style), per-shard size is capped at
    // target_size × (1 + tolerance), where target = n/K and tolerance =
    // 1/balance_factor. Vectors that would push a shard over its cap are
    // dropped from THAT shard only (they keep their other closure
    // assignments). This bounds max_shard, preventing the long-tail disk-read
    // pathology at billion-scale paged search. See SPANN arXiv:2111.08566.
    PartitionAssignment pa;
    pa.shards.assign(K, {});
    pa.centroids = centroids;
    pa.closure_factor = closure_factor;

    // Per-shard atomic size counters (only used when balance_factor > 0).
    // The cap is soft: concurrent threads may push a few vectors past it
    // before the counter updates propagate, but the overshoot is bounded
    // (~num_threads vectors per shard) and irrelevant at scale.
    const float cap_per_shard = (balance_factor > 0.0f && K > 0)
        ? std::ceil(static_cast<float>(n) / K * (1.0f + 1.0f / balance_factor))
        : std::numeric_limits<float>::max();
    std::vector<std::atomic<uint32_t>> shard_sizes(K);
    for (uint32_t k = 0; k < K; k++) shard_sizes[k].store(0, std::memory_order_relaxed);

    std::atomic<uint32_t> next_id{0};
    // Per-thread shard buffers. Each thread owns T ShardBufs (K each).
    const uint32_t T = num_threads;
    std::vector<std::vector<std::vector<uint32_t>>> local_shards(
        T, std::vector<std::vector<uint32_t>>(K));
    std::vector<std::future<void>> futs;
    futs.reserve(T);
    for (uint32_t t = 0; t < T; t++) {
        futs.push_back(pool.push(
            [&quantizer, &centroids, codes, n, code_size, K, closure_factor,
             &local_shards, t, &next_id, &shard_sizes, cap_per_shard]
            (size_t /*id*/, PartWorkerState& /*w*/) {
                std::vector<float> dists(K);
                auto& my_shards = local_shards[t];
                while (true) {
                    const uint32_t lo = next_id.fetch_add(kChunk,
                                                          std::memory_order_relaxed);
                    if (lo >= n) break;
                    const uint32_t hi = std::min(lo + kChunk, n);
                    for (uint32_t i = lo; i < hi; i++) {
                        const uint8_t* code =
                            codes + static_cast<size_t>(i) * code_size;
                        float best_d;
                        centroid_distances(quantizer, code, centroids,
                                           code_size, dists.data(), &best_d);
                        const float threshold = closure_factor * best_d;
                        for (uint32_t k = 0; k < K; k++) {
                            if (dists[k] > threshold) continue;
                            // Size cap (balance_factor). The PRIMARY cluster
                            // (argmin dist) is always admitted even if over
                            // cap — losing a vector from its nearest shard
                            // would defeat the partition. Secondary closure
                            // clusters respect the cap. This means the cap
                            // bounds the closure-replication overhead but
                            // never drops a vector's primary membership.
                            if (cap_per_shard < std::numeric_limits<float>::max()) {
                                const uint32_t cur = shard_sizes[k].load(
                                    std::memory_order_relaxed);
                                if (cur >= cap_per_shard && dists[k] > best_d) {
                                    continue;  // over cap and not primary → skip
                                }
                            }
                            my_shards[k].push_back(i);
                            shard_sizes[k].fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }));
    }
    for (auto& f : futs) f.get();

    // Concatenate per-thread shard buffers into pa.shards (serial, cheap).
    uint64_t total_assignments = 0;
    for (uint32_t k = 0; k < K; k++) {
        size_t total = 0;
        for (uint32_t t = 0; t < T; t++) total += local_shards[t][k].size();
        pa.shards[k].reserve(total);
        for (uint32_t t = 0; t < T; t++) {
            pa.shards[k].insert(pa.shards[k].end(),
                                local_shards[t][k].begin(),
                                local_shards[t][k].end());
        }
        total_assignments += total;
    }

    uint32_t max_shard = 0, min_shard = UINT32_MAX;
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
