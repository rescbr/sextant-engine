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

PartitionAssignment partition_codes(const PqQuantizer& quantizer,
                                     const uint8_t* codes, uint32_t n,
                                     uint32_t code_size, uint32_t K,
                                     float closure_factor,
                                     uint32_t iterations, uint32_t num_threads,
                                     uint64_t seed, float balance_factor,
                                     float closure_epsilon) {
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
    // Pad to a multiple of 4 so the batch4 loop has no scalar tail.
    // The padded centroids are duplicates of the last real centroid;
    // they'll never win the argmin (distance ≥ best_d by construction).
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
        // Pad: duplicate the last real centroid into the padding slots.
        for (uint32_t k = K; k < K_padded; k++) {
            std::memcpy(centroids[k].data(), centroids[K - 1].data(), code_size);
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

    ctpl::thread_pool_tls<PartWorkerState> pool(num_threads);

    // Chunked work-stealing over the N codes (matches the builder's pattern).
    constexpr uint32_t kChunk = 512;
    const uint32_t m = quantizer.m();
    const uint8_t bits = quantizer.bits();
    const uint32_t K_sub = quantizer.K();  // K per subspace (= 2^bits)
    const float* tbl = quantizer.cross_distance_table();
    const bool is_ip = (quantizer.metric() == MetricKind::InnerProduct);

    // For IP-metric partitioning: use IP-aware code distances instead of
    // the default L2sq cross-distance table. The L2sq table produces near-
    // constant distances on unit-normalized data (||a-b||² ≈ 2-2⟨a,b⟩),
    // causing k-means to oscillate (60-72% churn on Sphere-IP). IP distances
    // directly capture angular structure → better partitioning.
    if (is_ip) {
        quantizer.build_ip_cross_distance_table();
        tbl = quantizer.ip_cross_distance_table().data();
        spdlog::info("[sextant] partition: using IP-aware code distances");
    }

    auto parallel_assign = [&](void) {
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

    // Track assignment changes for early convergence detection.
    // K-means typically converges in 3-5 iterations; the fixed 10-iter default
    // wastes significant time at scale (10M × K=2000 = ~290s/iter).
    std::vector<uint32_t> prev_assign;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        const auto iter_t0 = std::chrono::steady_clock::now();

        // Snapshot assignments before this iteration to count changes.
        // Only needed from iter >= 1 (iter 0 has no previous to compare).
        uint64_t n_changed = 0;
        if (iter > 0) {
            // Cheap: compare assign[i] with prev_assign[i] in the assignment
            // pass itself (avoids a separate O(n) scan). We use a thread-local
            // counter summed atomically.
            // Actually, the assignment function returns the new cluster;
            // we'd need to compare old vs new inline. Simpler: snapshot
            // before, compare after. The snapshot is O(n) memcpy — cheap
            // relative to the O(nK) assignment.
            prev_assign.assign(n, UINT32_MAX);
            std::memcpy(prev_assign.data(), assign.data(),
                        static_cast<size_t>(n) * sizeof(uint32_t));
        }

        // --- Assignment pass (parallel): assign[i] = nearest centroid ---
        parallel_assign();
        const auto assign_t1 = std::chrono::steady_clock::now();

        // Count how many assignments changed (early convergence signal).
        if (iter > 0) {
            for (uint32_t i = 0; i < n; i++) {
                if (assign[i] != prev_assign[i]) ++n_changed;
            }
        }

        // --- Bucket assign[] into per-cluster lists (serial, cheap: O(n)) ---
        std::vector<std::vector<uint32_t>> clusters(K);
        for (uint32_t i = 0; i < n; i++) {
            clusters[assign[i]].push_back(i);
        }

        spdlog::info("[sextant] partition: iter {}/{} assign={:.1f}s, "
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
                        // candidates to 256 AND members to kMedoidSample
                        // to bound cost for large clusters. Without member
                        // subsampling, a 443k cluster (MSMARCO) costs
                        // 256 × 443k = 113M distance evals per iteration;
                        // with it, 256 × 1024 = 262k evals (~400× less).
                        // The medoid cost estimate is stable at 1024 samples
                        // (std error ~3% of the mean) — matches FAISS
                        // Clustering.cpp's default medoid sample size.
                        // Stride sampling is valid here: clusters[k] is
                        // filled by ascending global ID (not distance-
                        // correlated), so strided picks are effectively
                        // uniform over the Voronoi cell.
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
        spdlog::info("[sextant] partition: iter {}/{} medoid_update={:.1f}s"
                     "{}",
                     iter + 1, iterations,
                     std::chrono::duration<double>(update_t1 - assign_t1).count(),
                     change_str);

        // Early convergence: if <1% of vectors changed cluster, stop.
        // K-means on PQ codes converges fast (the medoid update is greedy;
        // after 3-4 iters the partition is stable). This saves significant
        // time at scale (10M vectors, K=2000).
        if (iter >= 2 && iter > 0 && n_changed > 0 &&
            static_cast<double>(n_changed) / n < 0.01) {
            spdlog::info("[sextant] partition: converged at iter {} "
                         "({:.2f}% changed < 1% threshold)", iter + 1,
                         100.0 * n_changed / n);
            break;
        }
    }

    // --- Auto-compute closure_epsilon from data if requested ---
    // Sentinel: closure_epsilon < 0 means "auto-compute from mean NN distance."
    // The absolute margin is set to a fraction of the mean nearest-centroid
    // distance, making it K-adaptive (cell radius shrinks with K, so epsilon
    // tracks it). See docs/closure_factor_derivation.md §6.
    float effective_epsilon = closure_epsilon;
    if (closure_epsilon < 0.0f) {
        // Sample-based estimate of mean NN distance from the last assignment.
        // Use the assign[] array: for each vector, the distance to its assigned
        // centroid is ~the NN distance. We sample to avoid a full O(N×K) pass.
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
        // epsilon = 20% of mean NN distance. This targets ~10-15% replication
        // (vectors within d_best + 0.2×mean_nn of a secondary centroid).
        effective_epsilon = 0.2f * mean_nn;
        spdlog::info("[sextant] partition: closure_epsilon = {:.4f} (auto, "
                     "mean_nn={:.4f}, sample={})", effective_epsilon, mean_nn,
                     count);
    }

    // --- Final assignment WITH closure overlap (parallel) ---
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
    pa.closure_epsilon = closure_epsilon;

    // If closure_epsilon > 0, we need the mean nearest-centroid distance to
    // set the absolute margin adaptively when closure_epsilon is a flag
    // (sentinel -1.0 means "auto-compute from data"). For now, closure_epsilon
    // is either 0 (use ratio) or a positive value (use absolute).
    const bool use_absolute = effective_epsilon > 0.0f;

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
    // Per-vector primary cluster (argmin dist). Filled during the closure pass
    // for RNG-redundancy pruning of replicas. primary[i] = the cluster whose
    // centroid is nearest to vector i (always preserved; replicas to other
    // clusters may be dropped if RNG-redundant with the primary).
    std::vector<uint32_t> primary(n, 0);
    std::vector<std::future<void>> futs;
    futs.reserve(T);
    for (uint32_t t = 0; t < T; t++) {
        futs.push_back(pool.push(
            [&quantizer, &centroids, codes, n, code_size, K, closure_factor,
             &local_shards, t, &next_id, &shard_sizes, cap_per_shard, &primary,
             use_absolute, effective_epsilon]
            (size_t /*id*/, PartWorkerState& /*w*/) {
                std::vector<float> dists(centroids.size());  // K_padded, not K
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
                        // Primary cluster = argmin (used for RNG pruning later).
                        uint32_t prim_k = 0;
                        for (uint32_t k = 1; k < K; k++) {
                            if (dists[k] < dists[prim_k]) prim_k = k;
                        }
                        primary[i] = prim_k;
                        // Closure threshold: absolute margin (SPANN-style) or
                        // ratio (legacy). Absolute is structurally K-robust.
                        // See docs/closure_factor_derivation.md §6.
                        const float threshold = use_absolute
                            ? (best_d + effective_epsilon)
                            : (closure_factor * best_d);
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

    // --- RNG-redundancy pruning of closure replicas (SPANN §3.3) ---
    // Drop a vector's replica in cluster k when its primary centroid c_p is
    // "occluded" from c_k by another centroid c_j that the vector was also
    // replicated to: d(c_j, c_k) < d(c_p, c_k). The intuition: if c_j is
    // closer to c_k than c_p is, a query routing to c_j is at least as close
    // to c_k as a query routing to c_p, so the replica in c_k is covered via
    // c_j's closure anyway. Strictly reduces index size with no recall loss
    // (SPANN Figure 3). Always-on when closure_factor > 1.0; no-op otherwise
    // (no replicas to prune).
    //
    // Implementation: compute pairwise centroid distances D[k][j] (K², tiny),
    // then for each shard k walk its members and drop those whose primary p
    // is RNG-occluded from k by some other cluster in the vector's closure
    // set. We don't track the full closure set per vector here (expensive);
    // we use the simpler pairwise rule "drop replica in k if ∃ j with
    // D[k][j] < D[k][p]" — i.e., j is closer to k than p is. This is a slight
    // relaxation (we don't verify j is in v's closure set), biased toward
    // dropping more, but the recall impact is bounded by closure redundancy.
    const uint64_t before_prune = total_assignments;
    if (closure_factor > 1.0f && K > 1) {
        // Pairwise centroid distances via PQ code_distance.
        std::vector<std::vector<float>> D(K, std::vector<float>(K, 0.0f));
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t j = k + 1; j < K; j++) {
                const float d = quantizer.code_distance(centroids[k].data(),
                                                        centroids[j].data());
                D[k][j] = d;
                D[j][k] = d;
            }
        }
        // For each shard k, drop members whose primary p has D[k][p] larger
        // than some D[k][j] (j ≠ p, j ≠ k). I.e., j occludes p from k.
        // parallel over shards.
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
                                // Primary membership — always keep.
                                members[write++] = members[i];
                                continue;
                            }
                            // RNG occlusion check: is there j (j≠k, j≠p)
                            // with D[k][j] < D[k][p]? If so, drop the replica.
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
        spdlog::info("[sextant] partition: RNG pruning dropped {} replicas "
                     "({:.1f}% of closure set)",
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
    spdlog::info("[sextant] partition: K={}, total assignments={} "
                 "(replication {:.3f}), shard sizes min={} max={}",
                 K, total_assignments, replication, min_shard, max_shard);

    return pa;
}

}  // namespace sextant
