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
#include "simd_kernels.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
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

/// Optimized nearest-centroid assignment using SIMD batch4 distance +
/// progressive pruning for L2sq.
///
/// Uses code_distance_batch4 (SVE2/NEON gather-load, 4 independent load
/// streams) to evaluate 4 centroids simultaneously per vector. The cross-
/// distance table is symmetric (||a-b||² for L2sq, -<a,b> for IP), so
/// passing the vector as anchor and centroids as candidates is correct.
///
/// For L2sq: progressive pruning — partial distance is monotonically
/// increasing (each subspace contributes ≥ 0), so prune if partial ≥ best_d.
/// This is done at subspace-chunk granularity within code_distance, NOT
/// here (would require a new code_distance_pruned function).
///
/// For IP: batch4 only (no pruning — distances can decrease).
/// CS-bound pruning for IP is implemented but adds overhead that exceeds
/// the pruning savings at K ≤ 4096; left for future tuning at K ≥ 16K.
inline uint32_t assign_nearest_centroid(
    const PqQuantizer& q,
    const uint8_t* code,
    const std::vector<std::vector<uint8_t>>& centroids,
    uint32_t K, uint32_t /*m*/, uint32_t /*bits*/, uint32_t /*K_sub*/,
    const float* cross_dist_table,
    uint32_t prev_assign,
    const float* /*code_sub_norms*/,
    const float* /*cent_sub_norms*/,
    bool /*is_ip*/) {

    // Fallback: no cross-distance table (untrained or PRQ with empty table).
    if (cross_dist_table == nullptr) {
        float best_d = std::numeric_limits<float>::max();
        uint32_t best_k = 0;
        for (uint32_t k = 0; k < K; k++) {
            const float d = q.code_distance(code, centroids[k].data());
            if (d < best_d) { best_d = d; best_k = k; }
        }
        return best_k;
    }

    // Warm start: evaluate the previous centroid first to get a tight best_d.
    float best_d = q.code_distance(code, centroids[prev_assign].data());
    uint32_t best_k = prev_assign;

    // Process all centroids in groups of 4 using SIMD batch4.
    // Centroids are padded to a multiple of 4, so no scalar tail.
    // The vector code is the anchor; 4 centroid codes are the candidates.
    // code_distance_batch4 uses SVE2 gather-load (4 independent load streams).
    float out[4];
    for (uint32_t kg = 0; kg < K; kg += 4) {
        q.code_distance_batch4(code,
            centroids[kg].data(),
            centroids[kg + 1].data(),
            centroids[kg + 2].data(),
            centroids[kg + 3].data(),
            out);
        for (uint32_t j = 0; j < 4; j++) {
            if (kg + j < K && out[j] < best_d) {
                best_d = out[j];
                best_k = kg + j;
            }
        }
    }

    return best_k;
}

}  // namespace

// ===========================================================================
// kmeans_pq: core k-means loop on PQ codes.
//
// Centroid init → parallel assignment → medoid update → convergence check.
// Returns medoid centroids (PQ codes) + assignment array.
// ===========================================================================

KMeansResult kmeans_pq(const PqQuantizer& quantizer,
                       const uint8_t* codes, uint32_t n,
                       uint32_t code_size, uint32_t K,
                       uint32_t iterations, uint32_t num_threads,
                       uint64_t seed) {
    if (K == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "kmeans_pq: K must be > 0");
    }
    if (n == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "kmeans_pq: n must be > 0");
    }
    K = std::min<uint32_t>(K, n);
    num_threads = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
    num_threads = std::max(1u, std::min(num_threads, n));

    std::mt19937_64 rng(seed);

    // --- Initialize centroids: distinct random codes. ---
    // Pad to a multiple of 4 so the batch4 loop has no scalar tail.
    const uint32_t K_padded = (K + 3) & ~3u;
    std::vector<std::vector<uint8_t>> centroids(K_padded,
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
        for (uint32_t k = K; k < K_padded; k++) {
            std::memcpy(centroids[k].data(), centroids[K - 1].data(), code_size);
        }
    }

    spdlog::info("[sextant] kmeans_pq: {} codes, K={}, {} iters, {} threads",
                 n, K, iterations, num_threads);

    std::vector<uint32_t> assign(n, 0);
    ctpl::thread_pool_tls<PartWorkerState> pool(num_threads);
    constexpr uint32_t kChunk = 512;
    const uint32_t m = quantizer.m();
    const uint8_t bits = quantizer.bits();
    const uint32_t K_sub = quantizer.K();
    const float* tbl = quantizer.cross_distance_table();
    const bool is_ip = (quantizer.metric() == MetricKind::InnerProduct);

    if (is_ip) {
        quantizer.build_ip_cross_distance_table();
        tbl = quantizer.ip_cross_distance_table().data();
        spdlog::info("[sextant] kmeans_pq: using IP-aware code distances");
    }

    auto parallel_assign = [&]() {
        std::atomic<uint32_t> next_id{0};
        std::vector<std::future<void>> futs;
        futs.reserve(num_threads);
        for (uint32_t t = 0; t < num_threads; t++) {
            futs.push_back(pool.push(
                [&quantizer, &centroids, codes, n, code_size, K, m, bits, K_sub,
                 tbl, is_ip, &assign, &next_id]
                (size_t /*id*/, PartWorkerState& /*w*/) {
                    while (true) {
                        const uint32_t lo = next_id.fetch_add(kChunk,
                                                              std::memory_order_relaxed);
                        if (lo >= n) break;
                        const uint32_t hi = std::min(lo + kChunk, n);
                        for (uint32_t i = lo; i < hi; i++) {
                            const uint8_t* code =
                                codes + static_cast<size_t>(i) * code_size;
                            assign[i] = assign_nearest_centroid(
                                quantizer, code, centroids, K, m, bits, K_sub,
                                tbl, assign[i],
                                nullptr, nullptr, is_ip);
                        }
                    }
                }));
        }
        for (auto& f : futs) f.get();
    };

    std::vector<uint32_t> prev_assign;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        const auto iter_t0 = std::chrono::steady_clock::now();

        uint64_t n_changed = 0;
        if (iter > 0) {
            prev_assign.assign(n, UINT32_MAX);
            std::memcpy(prev_assign.data(), assign.data(),
                        static_cast<size_t>(n) * sizeof(uint32_t));
        }

        // --- Assignment pass ---
        parallel_assign();
        const auto assign_t1 = std::chrono::steady_clock::now();

        if (iter > 0) {
            for (uint32_t i = 0; i < n; i++) {
                if (assign[i] != prev_assign[i]) ++n_changed;
            }
        }

        // --- Bucket into per-cluster lists ---
        std::vector<std::vector<uint32_t>> clusters(K);
        for (uint32_t i = 0; i < n; i++) {
            clusters[assign[i]].push_back(i);
        }

        spdlog::info("[sextant] kmeans_pq: iter {}/{} assign={:.1f}s, "
                     "cluster sizes min={} max={}",
                     iter + 1, iterations,
                     std::chrono::duration<double>(assign_t1 - iter_t0).count(),
                     std::accumulate(clusters.begin(), clusters.end(), UINT32_MAX,
                                     [](uint32_t a, const auto& b) {
                                         return std::min(a, (uint32_t)b.size());
                                     }),
                     std::accumulate(clusters.begin(), clusters.end(), 0u,
                                     [](uint32_t a, const auto& b) {
                                         return std::max(a, (uint32_t)b.size());
                                     }));

        // --- Medoid update (parallel over K clusters) ---
        std::atomic<uint32_t> next_cluster{0};
        std::vector<std::future<void>> futs;
        futs.reserve(num_threads);
        for (uint32_t t = 0; t < num_threads; t++) {
            futs.push_back(pool.push(
                [&quantizer, &centroids, &clusters, codes, code_size, n, K,
                 &next_cluster, &rng]
                (size_t id, PartWorkerState& /*w*/) {
                    std::mt19937_64 local_rng(rng() + id);
                    while (true) {
                        const uint32_t k = next_cluster.fetch_add(
                            1, std::memory_order_relaxed);
                        if (k >= K) break;
                        auto& cl = clusters[k];
                        if (cl.empty()) {
                            std::uniform_int_distribution<uint32_t> d(0, n - 1);
                            std::memcpy(centroids[k].data(),
                                        codes + static_cast<size_t>(d(local_rng)) * code_size,
                                        code_size);
                            continue;
                        }
                        constexpr uint32_t kMedoidSample = 1024;
                        const uint32_t cand_count =
                            static_cast<uint32_t>(std::min<size_t>(256, cl.size()));
                        std::vector<uint32_t> cand_idx(cand_count);
                        for (uint32_t c = 0; c < cand_count; c++) {
                            cand_idx[c] = cl[static_cast<size_t>(c) * cl.size() /
                                              cand_count];
                        }
                        const uint32_t mem_count =
                            static_cast<uint32_t>(std::min<size_t>(kMedoidSample,
                                                                    cl.size()));
                        std::vector<uint32_t> mem_idx(mem_count);
                        for (uint32_t m = 0; m < mem_count; m++) {
                            mem_idx[m] = cl[static_cast<size_t>(m) * cl.size() /
                                             mem_count];
                        }
                        uint32_t best_medoid = cand_idx[0];
                        float best_cost = std::numeric_limits<float>::max();
                        for (uint32_t ci = 0; ci < cand_count; ci++) {
                            const uint8_t* cand =
                                codes + static_cast<size_t>(cand_idx[ci]) * code_size;
                            float cost = 0.0f;
                            for (uint32_t mi : mem_idx) {
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
        const auto update_t1 = std::chrono::steady_clock::now();
        const std::string change_str = (iter > 0)
            ? std::string(", changed=") + std::to_string(n_changed) +
              " (" + std::to_string(100.0 * n_changed / n).substr(0, 4) + "%)"
            : "";
        spdlog::info("[sextant] kmeans_pq: iter {}/{} medoid_update={:.1f}s{}",
                     iter + 1, iterations,
                     std::chrono::duration<double>(update_t1 - assign_t1).count(),
                     change_str);

        if (iter >= 2 && iter > 0 && n_changed > 0 &&
            static_cast<double>(n_changed) / n < 0.01) {
            spdlog::info("[sextant] kmeans_pq: converged at iter {} "
                         "({:.2f}% changed)", iter + 1,
                         100.0 * n_changed / n);
            break;
        }
    }

    KMeansResult result;
    result.centroids = std::move(centroids);
    result.assign = std::move(assign);
    result.K = K;
    return result;
}

// ===========================================================================
// assign_with_closure: build final shards with closure overlap, balance cap,
// and RNG redundancy pruning.
// ===========================================================================

PartitionAssignment assign_with_closure(
    const PqQuantizer& quantizer,
    const uint8_t* codes, uint32_t n, uint32_t code_size,
    const KMeansResult& km,
    float closure_factor,
    float closure_epsilon,
    float balance_factor,
    uint32_t num_threads) {

    const uint32_t K = km.K;
    num_threads = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
    num_threads = std::max(1u, std::min(num_threads, n));
    const auto& centroids = km.centroids;
    const auto& assign = km.assign;

    // --- Auto-compute closure_epsilon from data if requested ---
    float effective_epsilon = closure_epsilon;
    if (closure_epsilon < 0.0f) {
        constexpr uint32_t kSampleSize = 4096;
        const uint32_t step = std::max(1u, n / kSampleSize);
        double sum_nn = 0.0;
        uint32_t count = 0;
        for (uint32_t i = 0; i < n; i += step) {
            const uint8_t* code =
                codes + static_cast<size_t>(i) * code_size;
            const uint8_t* cent = centroids[assign[i]].data();
            sum_nn += quantizer.code_distance(code, cent);
            count++;
        }
        const float mean_nn = count > 0 ? float(sum_nn / count) : 0.0f;
        effective_epsilon = 0.2f * mean_nn;
        spdlog::info("[sextant] assign_with_closure: closure_epsilon = {:.4f} "
                     "(auto, mean_nn={:.4f}, sample={})",
                     effective_epsilon, mean_nn, count);
    }

    PartitionAssignment pa;
    pa.shards.assign(K, {});
    pa.centroids = centroids;
    pa.closure_factor = closure_factor;
    pa.closure_epsilon = closure_epsilon;

    const bool use_absolute = effective_epsilon > 0.0f;
    const float cap_per_shard = (balance_factor > 0.0f && K > 0)
        ? std::ceil(static_cast<float>(n) / K * (1.0f + 1.0f / balance_factor))
        : std::numeric_limits<float>::max();
    std::vector<std::atomic<uint32_t>> shard_sizes(K);
    for (uint32_t k = 0; k < K; k++) shard_sizes[k].store(0, std::memory_order_relaxed);

    ctpl::thread_pool_tls<PartWorkerState> pool(num_threads);
    constexpr uint32_t kChunk = 512;

    std::atomic<uint32_t> next_id{0};
    const uint32_t T = num_threads;
    std::vector<std::vector<std::vector<uint32_t>>> local_shards(
        T, std::vector<std::vector<uint32_t>>(K));
    std::vector<uint32_t> primary(n, 0);
    std::vector<std::future<void>> futs;
    futs.reserve(T);

    for (uint32_t t = 0; t < T; t++) {
        futs.push_back(pool.push(
            [&quantizer, &centroids, codes, n, code_size, K, closure_factor,
             &local_shards, t, &next_id, &shard_sizes, cap_per_shard, &primary,
             use_absolute, effective_epsilon]
            (size_t /*id*/, PartWorkerState& /*w*/) {
                std::vector<float> dists(centroids.size());
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
                        uint32_t prim_k = 0;
                        for (uint32_t k = 1; k < K; k++) {
                            if (dists[k] < dists[prim_k]) prim_k = k;
                        }
                        primary[i] = prim_k;
                        const float threshold = use_absolute
                            ? (best_d + effective_epsilon)
                            : (closure_factor * best_d);
                        for (uint32_t k = 0; k < K; k++) {
                            if (dists[k] > threshold) continue;
                            if (cap_per_shard < std::numeric_limits<float>::max()) {
                                const uint32_t cur = shard_sizes[k].load(
                                    std::memory_order_relaxed);
                                if (cur >= cap_per_shard && dists[k] > best_d) {
                                    continue;
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

    // Concatenate per-thread shard buffers.
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

    // --- RNG redundancy pruning ---
    const uint64_t before_prune = total_assignments;
    if (closure_factor > 1.0f && K > 1) {
        std::vector<std::vector<float>> D(K, std::vector<float>(K, 0.0f));
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t j = k + 1; j < K; j++) {
                const float d = quantizer.code_distance(centroids[k].data(),
                                                        centroids[j].data());
                D[k][j] = d;
                D[j][k] = d;
            }
        }
        std::atomic<uint32_t> next_shard{0};
        std::vector<std::future<void>> pfuts;
        pfuts.reserve(num_threads);
        for (uint32_t t = 0; t < num_threads; t++) {
            pfuts.push_back(pool.push(
                [&pa, &D, &primary, K, &next_shard]
                (size_t /*id*/, PartWorkerState& /*w*/) {
                    while (true) {
                        const uint32_t k = next_shard.fetch_add(
                            1, std::memory_order_relaxed);
                        if (k >= K) break;
                        auto& members = pa.shards[k];
                        size_t write = 0;
                        for (size_t i = 0; i < members.size(); i++) {
                            const uint32_t v = members[i];
                            const uint32_t p = primary[v];
                            if (p == k) {
                                members[write++] = members[i];
                                continue;
                            }
                            const float d_kp = D[k][p];
                            bool occluded = false;
                            for (uint32_t j = 0; j < K; j++) {
                                if (j == k || j == p) continue;
                                if (D[k][j] < d_kp) { occluded = true; break; }
                            }
                            if (!occluded) {
                                members[write++] = members[i];
                            }
                        }
                        members.resize(write);
                    }
                }));
        }
        for (auto& f : pfuts) f.get();

        total_assignments = 0;
        for (uint32_t k = 0; k < K; k++) total_assignments += pa.shards[k].size();
        spdlog::info("[sextant] assign_with_closure: RNG pruning dropped {} "
                     "replicas ({:.1f}% of closure set)",
                     before_prune - total_assignments,
                     before_prune > 0
                         ? 100.0 * double(before_prune - total_assignments) /
                               double(before_prune)
                         : 0.0);
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
    spdlog::info("[sextant] assign_with_closure: K={}, total assignments={} "
                 "(replication {:.3f}), shard sizes min={} max={}",
                 K, total_assignments, replication, min_shard, max_shard);

    return pa;
}

// ===========================================================================
// partition_codes: convenience wrapper (kmeans_pq → assign_with_closure).
// ===========================================================================

PartitionAssignment partition_codes(const PqQuantizer& quantizer,
                                     const uint8_t* codes, uint32_t n,
                                     uint32_t code_size, uint32_t K,
                                     float closure_factor,
                                     uint32_t iterations, uint32_t num_threads,
                                     uint64_t seed, float balance_factor,
                                     float closure_epsilon) {
    auto km = kmeans_pq(quantizer, codes, n, code_size, K,
                        iterations, num_threads, seed);
    return assign_with_closure(quantizer, codes, n, code_size, km,
                               closure_factor, closure_epsilon,
                               balance_factor, num_threads);
}

}  // namespace sextant
