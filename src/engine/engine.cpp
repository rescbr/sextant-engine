// Engine facade — build/open/search/insert/flush orchestration.
//
// Owns the PqQuantizer + VamanaCore + flat-in-RAM build buffers. The build
// pipeline is:
//   1. resolve_params (auto-defaults from N, dim)
//   2. allocate flat codes + nodes buffers
//   3. Pass 1: reservoir sample (256K) + PQ train
//   4. Pass 2: encode all vectors → codes_buffer_
//   5. Parallel HDC construct via CTPL (disjoint node ranges)
//   6. Finalize: compute_entry_points + finalize_inline_codes
//   7. Flush sidecars (.graph/.codes/.meta/.manifest)
//
// The search/load path is in search.cpp.

#include "build.hpp"
#include "fbin_source.hpp"
#include "memory_source.hpp"
#include "partition.hpp"
#include "sidecar_io.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/direct_io.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"
#include "storage/sidecar_header.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#ifdef __linux__
#include <sys/mman.h>  // madvise(MADV_HUGEPAGE) for TLB-friendly build buffers
#endif
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace sextant {

using engine_detail::fill_header;
using engine_detail::read_exact;
using engine_detail::write_padded;

// ---------------------------------------------------------------------------
// VamanaParams::from_resolved — factory used by all build/search cores.
// Defined here (not in vamana_core.cpp) because it needs the full
// ResolvedParams definition from engine.hpp.
// ---------------------------------------------------------------------------
VamanaParams VamanaParams::from_resolved(const ResolvedParams& p, Dim dim,
                                           uint16_t R_override,
                                           uint16_t inline_pq) {
    VamanaParams v;
    v.dim = dim;
    v.R = R_override ? R_override : p.R;
    v.L = p.L;
    v.L_build = p.L_build;
    v.alpha = p.alpha;
    v.inline_pq_count = inline_pq;
    v.n_entry_points = p.n_entry_points;
    v.n_search_entry_points = p.n_search_entry_points;
    v.early_exit_patience = p.early_exit_patience;
    v.max_occlusion = p.max_occlusion;
    return v;
}

// ---------------------------------------------------------------------------
// Node layout offsets (mirror vamana_core.cpp — kept private to the engine).
// ---------------------------------------------------------------------------
namespace {
inline constexpr uint32_t kNeighborArrayOffset = 16;

/// Reservoir sample target (Issue 19).
inline constexpr uint64_t kSampleTarget = 256'000;

/// Generate a shared index UUID (low + high) for a build, from a random device.
std::pair<uint64_t, uint64_t> make_uuid() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> dist;
    return {dist(rd), dist(rd)};
}

/// PageShuffle (Issue: graph node reordering at flush time).
///
/// Computes a BFS ordering of the build graph starting from the entry points.
/// Returns `bfs_order` where `bfs_order[new_pos] = old_id` (i.e. the inverse
/// remap). Placing frequently-traversed nodes near the front and clustering
/// neighbors together improves paged-search cache hit rates.
///
/// `build_node_size` is the stride between consecutive nodes in `nodes_buffer`
/// (the inline_pq=0 build layout). `n` is the node count.
std::vector<uint32_t> compute_bfs_order(
    const uint8_t* nodes_buffer, uint32_t n, uint32_t build_node_size,
    const std::vector<uint32_t>& entry_points) {

    std::vector<uint32_t> bfs_order;
    bfs_order.reserve(n);
    std::vector<bool> visited(n, false);
    std::queue<uint32_t> queue;

    // Seed BFS with all entry points.
    for (uint32_t ep : entry_points) {
        if (ep < n && !visited[ep]) {
            visited[ep] = true;
            queue.push(ep);
        }
    }

    while (!queue.empty()) {
        uint32_t node = queue.front();
        queue.pop();
        bfs_order.push_back(node);

        const uint8_t* np = nodes_buffer + static_cast<size_t>(node) * build_node_size;
        uint16_t ncount = VamanaCore::get_neighbor_count(np);
        for (uint16_t i = 0; i < ncount; i++) {
            uint32_t nb = VamanaCore::get_neighbor(np, i);
            if (nb < n && !visited[nb]) {
                visited[nb] = true;
                queue.push(nb);
            }
        }
    }

    // Append any unreachable nodes (shouldn't happen in a connected graph).
    for (uint32_t i = 0; i < n; i++) {
        if (!visited[i]) bfs_order.push_back(i);
    }

    return bfs_order;  // bfs_order[new_pos] = old_id
}

}  // namespace

// ===========================================================================
// Construction / destruction
// ===========================================================================

Engine::Engine() {
    // Default logger is spdlog's global; callers may init_logging() first.
}

Engine::~Engine() {
    if (codes_buffer_) aligned_free(codes_buffer_);
    if (nodes_buffer_) aligned_free(nodes_buffer_);
    if (raw_vecs_buffer_) aligned_free(raw_vecs_buffer_);
}

// ===========================================================================
// build()
// ===========================================================================

BuildResult Engine::build(VectorSource& source, const std::string& index_path,
                          const BuildConfig& config) {
    // Resolve params, then forward to the ResolvedParams overload which does
    // the count/dim/index_path setup, build_partitioned, and timing.
    count_ = source.count();
    dim_ = source.dim();
    return build(source, index_path, resolve_params(count_, dim_, config));
}

BuildResult Engine::build(VectorSource& source, const std::string& index_path,
                          const ResolvedParams& params) {
    const auto t0 = std::chrono::steady_clock::now();

    count_ = source.count();
    dim_ = source.dim();
    if (count_ == 0) {
        throw Error(ErrorCode::InvalidParam, "Engine::build: source is empty");
    }
    if (dim_ == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::build: source has dim=0");
    }
    index_path_ = index_path;

    spdlog::info("[sextant] build: n={} dim={} → '{}'", count_, dim_,
                 index_path);

    auto result = build_partitioned(source, index_path, params);
    const auto t1 = std::chrono::steady_clock::now();
    result.build_time_sec =
        std::chrono::duration<double>(t1 - t0).count() + result.build_time_sec;
    spdlog::info("[sextant] build complete (K={}) in {:.2f}s", params.K,
                 result.build_time_sec);
    return result;
}

// ===========================================================================
// Pass 1: reservoir sample + (optional) global bits probe + PQ train
// ===========================================================================

namespace {

/// Probe pool size: we measure recall on a subset of the reservoir to keep
/// the brute-force KNN cheap. 20K is what pq_explore uses; stable enough.
constexpr uint32_t kProbePool = 20000;
constexpr uint32_t kProbeQueries = 500;
constexpr uint32_t kProbeTopk = 10;
constexpr uint32_t kProbeRecallK = 30;

/// Brute-force truth for the probe: per query, the top-kProbeTopk neighbor ids,
/// their true L2sq distances (sorted ascending), and counts of pool vectors
/// within the k-th and kProbeRecallK-th NN distances (tie bands — how many
/// vectors are as near as the k-th / shortlist-boundary neighbor).
struct ProbeTruth {
    std::vector<std::vector<uint32_t>> ids;      // [qi] → top-k ids
    std::vector<std::vector<float>> dists;       // [qi] → top-k true L2sq
    std::vector<uint32_t> band_counts;           // [qi] → # vectors within k-th NN dist
    std::vector<uint32_t> band30_counts;         // [qi] → # vectors within kProbeRecallK-th NN dist
};

/// Compute brute-force true top-`truth_k` for a set of query indices within
/// the pool. Returns ids + true distances (ascending) + per-query band counts.
/// `truth_k` defaults to kProbeTopk (10) for the PQ probe path; estimate_config
/// passes its target_topk (e.g. 100) so recall@k and proximity@k are measured
/// against truth of the right depth. Must have truth_k ≤ pool_n-1.
ProbeTruth compute_truth(const float* pool, uint32_t pool_n, Dim dim,
                         const std::vector<uint32_t>& qidx,
                         uint32_t truth_k = kProbeTopk) {
    const auto t_norms_start = std::chrono::steady_clock::now();
    std::vector<float> norms(pool_n);
    for (uint32_t i = 0; i < pool_n; i++) {
        const float* v = pool + static_cast<size_t>(i) * dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; d++) acc += double(v[d]) * v[d];
        norms[i] = float(acc);
    }
    const double norms_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_norms_start).count();

    ProbeTruth truth;
    truth.ids.resize(qidx.size());
    truth.dists.resize(qidx.size());
    truth.band_counts.resize(qidx.size(), 0);
    truth.band30_counts.resize(qidx.size(), 0);

    // Parallelize across queries — each query's truth computation is fully
    // independent: it reads pool/norms/qidx (read-only) and writes only its
    // own disjoint truth.*[qi] slots. The per-query `ranked` vector is
    // stack-local. No shared mutable state → no locking needed. Mirrors the
    // pass2_encode atomic-counter work-stealing pattern (engine.cpp:1030+).
    const uint32_t nthreads = std::thread::hardware_concurrency();
    const auto t_qloop_start = std::chrono::steady_clock::now();
    std::atomic<size_t> next_qi{0};
    auto worker = [&]() {
        while (true) {
            const size_t qi = next_qi.fetch_add(1, std::memory_order_relaxed);
            if (qi >= qidx.size()) break;
            const float* q = pool + static_cast<size_t>(qidx[qi]) * dim;
            double qn = 0.0;
            for (uint32_t d = 0; d < dim; d++) qn += double(q[d]) * q[d];
            std::vector<std::pair<float, uint32_t>> ranked;
            ranked.reserve(pool_n - 1);
            for (uint32_t i = 0; i < pool_n; i++) {
                if (i == qidx[qi]) continue;
                double dot = 0.0;
                const float* v = pool + static_cast<size_t>(i) * dim;
                for (uint32_t d = 0; d < dim; d++)
                    dot += double(q[d]) * v[d];
                ranked.emplace_back(float(norms[i] - 2.0 * dot + qn), i);
            }
            // Partial-sort deep enough to cover both the requested truth_k and
            // the band-count radius (kProbeRecallK). Whichever is larger wins.
            const size_t ksort = std::min<size_t>(
                std::max<uint32_t>(truth_k, kProbeRecallK), ranked.size());
            std::partial_sort(ranked.begin(), ranked.begin() + ksort,
                              ranked.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first < b.first;
                              });
            const size_t kk = std::min<size_t>(truth_k, ranked.size());
            for (size_t k = 0; k < kk; k++) {
                truth.ids[qi].push_back(ranked[k].second);
                truth.dists[qi].push_back(ranked[k].first);
            }
            // Band counts: how many pool vectors are within each radius?
            // Computed once here (not per-config) since they're dataset
            // properties. kProbeRecallK band is for the PQ probe's tie metric.
            const float band_radius = ranked[kk - 1].first;
            const float band30_radius = ranked[ksort - 1].first;
            uint32_t bc = 0, bc30 = 0;
            for (uint32_t i = 0; i < pool_n; i++) {
                if (i == qidx[qi]) continue;
                const float* v = pool + static_cast<size_t>(i) * dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < dim; d++)
                    dot += double(q[d]) * v[d];
                const float td = float(norms[i] - 2.0 * dot + qn);
                if (td <= band30_radius + 1e-6f) {
                    ++bc30;
                    if (td <= band_radius + 1e-6f) ++bc;
                }
            }
            truth.band_counts[qi] = bc;
            truth.band30_counts[qi] = bc30;
        }
    };
    std::vector<std::thread> thrpool;
    for (uint32_t t = 0; t < std::min(nthreads, uint32_t(qidx.size())); t++) {
        thrpool.emplace_back(worker);
    }
    for (auto& th : thrpool) th.join();

    const double qloop_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_qloop_start).count();
    spdlog::debug("[sextant] compute_truth: norms precompute {:.3f}s, "
                  "query loop {:.3f}s ({} threads, {} queries)",
                  norms_sec, qloop_sec, nthreads, qidx.size());
    return truth;
}

/// Distortion + diagnostics measured for one (m, bits) config against truth.
struct ProbeScore {
    double distortion = 0.0;     // median |1 - pq_dist/true_dist| (≥0; 0 = perfect)
    double band_recall = 0.0;    // frac truth neighbors within band found in PQ top-K
    double tie_fraction = 0.0;   // frac queries whose band(@k) has >kProbeTopk tied vectors
    double tie30_fraction = 0.0; // frac queries whose band(@kProbeRecallK) has >kProbeRecallK tied
};

/// Measure PQ quality for a given (m, bits) config.
///
/// distortion: for each query's true top-k neighbors, the ratio of the
/// PQ-estimated distance to the true distance. We report the median across all
/// truth neighbors (robust to PQ outliers). 1.0 = perfect; 1.3 = PQ overestimates
/// true-neighbor distances by 30% on the median. This is the selection signal:
/// it's geometrically grounded and transfers across dataset classes.
///
/// band_recall (diagnostic): fraction of truth neighbors for which the PQ
/// top-kProbeRecallK shortlist contains SOME id within the band radius (the
/// k-th true NN distance). Unlike id-recall, this is insensitive to tie-breaking
/// among co-located vectors. On non-clustered data it ≈ id-recall.
///
/// tie_fraction (diagnostic): fraction of queries where the band (vectors at or
/// inside the k-th NN distance) contains more than kProbeTopk vectors. When high,
/// id-based recall is structurally capped below 1.0 regardless of PQ quality.
ProbeScore probe_recall(const float* pool, uint32_t pool_n, Dim dim,
                        MetricKind metric, uint16_t pq_m, uint8_t bits,
                        const ProbeTruth& truth, const std::vector<uint32_t>& qidx) {
    PqQuantizer q(metric, dim, pq_m, bits);
    q.train(pool, pool_n);
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes(static_cast<size_t>(pool_n) * cs);
    for (uint32_t i = 0; i < pool_n; i++)
        q.encode(pool + static_cast<size_t>(i) * dim,
                 codes.data() + static_cast<size_t>(i) * cs);
    std::vector<float> lut(q.lut_size());

    // For distortion: collect pq_dist/true_dist ratios over all truth neighbors.
    std::vector<double> ratios;
    ratios.reserve(qidx.size() * kProbeTopk);

    // For band_recall: per truth neighbor, did the PQ top-K shortlist contain
    // some id within the band? We need the true distance from the query to every
    // id in the PQ shortlist. Compute norms once.
    std::vector<float> norms(pool_n);
    for (uint32_t i = 0; i < pool_n; i++) {
        const float* v = pool + static_cast<size_t>(i) * dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; d++) acc += double(v[d]) * v[d];
        norms[i] = float(acc);
    }

    uint64_t band_hits = 0, band_total = 0;

    for (size_t qi = 0; qi < qidx.size(); qi++) {
        const float* qv = pool + static_cast<size_t>(qidx[qi]) * dim;
        double qn = 0.0;
        for (uint32_t d = 0; d < dim; d++) qn += double(qv[d]) * qv[d];

        q.preprocess_query(qv, lut.data());

        // PQ-rank the whole pool, keep top-kProbeRecallK.
        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(pool_n - 1);
        for (uint32_t i = 0; i < pool_n; i++) {
            if (i == qidx[qi]) continue;
            ranked.emplace_back(
                q.lut_distance(codes.data() + static_cast<size_t>(i) * cs,
                               lut.data()), i);
        }
        const uint32_t kp = std::min<uint32_t>(kProbeRecallK, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + kp, ranked.end(),
                          [](const auto& a, const auto& b){ return a.first < b.first; });

        // Distortion: |1 - pq_dist/true_dist| for each truth neighbor.
        // Median absolute deviation from perfect (0.0 = PQ exactly preserves
        // neighbor distances). Always ≥0, lower = better, direction-agnostic.
        for (size_t t = 0; t < truth.ids[qi].size(); t++) {
            const uint32_t tid = truth.ids[qi][t];
            const float true_d = truth.dists[qi][t];
            const float pq_d = q.lut_distance(codes.data() + static_cast<size_t>(tid) * cs,
                                              lut.data());
            if (true_d > 1e-9f) {
                ratios.push_back(std::fabs(1.0 -
                    static_cast<double>(pq_d) / static_cast<double>(true_d)));
            }
        }

        // band_recall: did the PQ shortlist surface any vector within the
        // band radius (the k-th true NN distance)? This is cluster-aware:
        // insensitive to which specific co-located id PQ picked.
        const float band_radius = truth.dists[qi].empty()
            ? std::numeric_limits<float>::infinity()
            : truth.dists[qi].back();
        bool any_in_band = false;
        for (uint32_t k = 0; k < kp; k++) {
            const uint32_t id = ranked[k].second;
            const float* v = pool + static_cast<size_t>(id) * dim;
            double dot = 0.0;
            for (uint32_t d = 0; d < dim; d++) dot += double(qv[d]) * v[d];
            const float td = float(norms[id] - 2.0 * dot + qn);
            if (td <= band_radius + 1e-6f) { any_in_band = true; break; }
        }
        for (size_t t = 0; t < truth.ids[qi].size(); t++) {
            band_total++;
            if (any_in_band) band_hits++;
        }
    }

    ProbeScore s;
    if (!ratios.empty()) {
        std::nth_element(ratios.begin(),
                         ratios.begin() + ratios.size() / 2,
                         ratios.end());
        s.distortion = ratios[ratios.size() / 2];
    }
    s.band_recall = band_total ? double(band_hits) / double(band_total) : 0.0;
    // tie_fraction is a dataset property (from compute_truth), same for all
    // configs. Report it from the truth, not recomputed per-config.
    if (!truth.band_counts.empty()) {
        uint64_t tied = 0;
        for (uint32_t bc : truth.band_counts) {
            if (bc > kProbeTopk) ++tied;
        }
        s.tie_fraction = double(tied) / double(truth.band_counts.size());
    }
    if (!truth.band30_counts.empty()) {
        uint64_t tied30 = 0;
        for (uint32_t bc : truth.band30_counts) {
            if (bc > kProbeRecallK) ++tied30;
        }
        s.tie30_fraction = double(tied30) / double(truth.band30_counts.size());
    }
    return s;
}

/// Generate candidate m values: divisors of dim spanning sub_dim ~1 to ~12,
/// capped to [4, 256]. The low sub_dim bound (1) matters for low-dimensional
/// datasets (e.g. SIFT-128), where high m / sub_dim=1-2 with 4-bit gives
/// excellent recall (16 centroids densely tile a 1-2D sub-space). Sorted ascending.
std::vector<uint16_t> candidate_ms(Dim dim) {
    std::vector<uint16_t> ms;
    for (uint32_t m = 4; m <= 256; m++) {
        if (dim % m == 0) {
            const uint32_t sd = dim / m;
            if (sd >= 1 && sd <= 12) ms.push_back(static_cast<uint16_t>(m));
        }
    }
    return ms;
}

    /// A probed (m, bits) config with its measured quality and cost.
    struct ProbedConfig {
        uint16_t m;
        uint8_t bits;
        uint32_t code_bytes;   // m * bits / 8
        uint32_t table_bytes;  // m * K^2 * 4 (cross-distance table)
        double distortion;       // median |1 - pq_dist/true_dist| (≥0; 0 = perfect)
        double band_recall;      // cluster-aware recall (diagnostic)
        double tie_fraction;     // frac queries with >topk tied at @k (diagnostic)
        double tie30_fraction;   // frac queries with >30 tied at @30 (diagnostic)
        double cost;           // m * residency_factor (lower = faster)
    };

/// Cross-distance table size in bytes for a given (m, bits).
inline uint32_t pq_table_bytes(uint16_t m, uint8_t bits) {
    const uint32_t K = 1u << bits;
    return static_cast<uint32_t>(m) * K * K * sizeof(float);
}

/// Sweep (m, bits) candidate pairs, measure distortion, and select the minimum-
/// cost config whose distortion is within `max_distortion`.
///
/// Selection policy (distortion-bounded min cost):
///   1. Compute distortion (median |1 - pq_dist/true_dist|) and cost (m ×
///      residency) for every candidate.
///   2. Filter to configs with distortion ≤ max_distortion.
///   3. Among eligible configs, pick minimum cost. Tie-break by lower
///      distortion, then smaller code_bytes.
///
/// The max_distortion bound is the selector — it expresses "what's good enough"
/// in geometric terms (how much PQ may distort true-neighbor distances). This
/// replaces the prior recall-count floor, which was fragile under near-duplicate
/// clustering. Distortion transfers across dataset classes because it measures
/// PQ's fundamental property (distance fidelity) rather than a tie-breaking-
/// dependent count.
///
/// If no config meets max_distortion, the lowest-distortion config is returned
/// (with a warning). When `fixed_m != 0` or `fixed_bits != 0`, the probe is
/// restricted to that m/bits but the quality gate still applies.
ProbedConfig probe_best_config(const float* pool, uint32_t pool_n, Dim dim,
                               MetricKind metric,
                               const ProbeTruth& truth,
                               const std::vector<uint32_t>& qidx,
                               uint16_t fixed_m, uint8_t fixed_bits,
                               double max_distortion,
                               std::vector<ProbedConfig>& all_out,
                               std::string& reason_out) {
    // Build candidate list: m values, constrained if fixed.
    std::vector<uint16_t> ms;
    if (fixed_m != 0) {
        ms.push_back(fixed_m);
    } else {
        ms = candidate_ms(dim);
    }
    // Bit values to probe: both, or just the fixed one.
    std::vector<uint8_t> bit_vals;
    if (fixed_bits != 0) bit_vals.push_back(fixed_bits);
    else { bit_vals.push_back(4); bit_vals.push_back(8); }

    std::vector<ProbedConfig> all;
    for (const uint16_t m : ms) {
        if (dim % m != 0) continue;
        for (const uint8_t bits : bit_vals) {
            const ProbeScore s = probe_recall(pool, pool_n, dim, metric, m, bits, truth, qidx);
            const uint32_t cb = (static_cast<uint32_t>(m) * bits + 7) / 8;
            const uint32_t tb = pq_table_bytes(m, bits);
            all.push_back({m, bits, cb, tb, s.distortion, s.band_recall,
                           s.tie_fraction, s.tie30_fraction, 0.0});
            spdlog::info("[sextant] pass 1:   probe m={} bits={} code={}B table={}KB "
                         "distortion={:.4f} band_recall@{}={:.4f} ties@{}={:.2f} ties@{}={:.2f}",
                         m, bits, cb, tb / 1024, s.distortion, kProbeRecallK,
                         s.band_recall, kProbeTopk, s.tie_fraction,
                         kProbeRecallK, s.tie30_fraction);        }
    }
    if (all.empty()) {
        all_out = all;
        return {ms.empty() ? uint16_t{0} : ms.front(), uint8_t{8}, 0, 0,
                0.0, 0.0, 0.0, 0.0};
    }

    // Cost model: cost = m (number of gathers per distance computation).
    //
    // m is the dominant cost factor in both build and search — each
    // code_distance/lut_distance call performs m table gathers. Among configs
    // meeting the distortion bound, the fewest-gather config is cheapest.
    //
    // When two configs have the same m (e.g. 4-bit vs 8-bit at the same m),
    // tie-break by table size: the smaller table is always faster (fits a
    // closer cache level). Since 4-bit K²=256 and 8-bit K²=65536, the 4-bit
    // table is 256× smaller at the same m — this naturally selects 4-bit when
    // distortion is equal, without magic residency factors or cache detection.
    for (auto& c : all) {
        c.cost = static_cast<double>(c.m);
    }

    // Filter to configs meeting the distortion bound.
    std::vector<const ProbedConfig*> eligible;
    for (const auto& c : all) {
        if (c.distortion <= max_distortion) eligible.push_back(&c);
    }

    ProbedConfig best;
    std::string reason;
    if (eligible.empty()) {
        // No config meets the bound — pick lowest distortion, warn.
        best = *std::min_element(all.begin(), all.end(),
            [](const auto& a, const auto& b){ return a.distortion < b.distortion; });
        reason = "above max_distortion; lowest distortion";
        spdlog::warn("[sextant] pass 1: no config meets max_distortion {:.3f}; "
                     "selected lowest-distortion (m={} bits={} distortion={:.4f})",
                     max_distortion, best.m, best.bits, best.distortion);
    } else if (eligible.size() == 1) {
        best = *eligible[0];
        reason = "only config within max_distortion";
    } else {
        // Min cost (m), tie-break by smaller table_bytes, then code_bytes.
        const ProbedConfig* pick = eligible[0];
        for (const auto* c : eligible) {
            if (c->cost < pick->cost ||
                (c->cost == pick->cost && c->table_bytes < pick->table_bytes) ||
                (c->cost == pick->cost && c->table_bytes == pick->table_bytes &&
                 c->code_bytes < pick->code_bytes)) {
                pick = c;
            }
        }
        best = *pick;
        reason = "min cost (distortion ≤ bound)";
    }
    spdlog::info("[sextant] pass 1: selected m={} bits={} (code={}B table={}KB "
                 "distortion={:.4f} cost={:.0f}) [{}; max_distortion {:.3f}]",
                 best.m, best.bits, best.code_bytes, best.table_bytes / 1024,
                 best.distortion, best.cost, reason, max_distortion);
    all_out = std::move(all);
    reason_out = reason;
    return best;
}

// ---------------------------------------------------------------------------
// Seek-based random sampling.
//
// When N >> k, seeks directly to k random offsets (O(k) I/O) instead of
// reading the full N-vector file (O(N) I/O). Statistically identical to
// Algorithm R reservoir sampling: every vector has probability k/N of being
// in the sample, because this is a uniform random sample without replacement
// (each subset of size k is equally likely → each element has probability
// exactly k/N).
//
// Index generation: partial Fisher-Yates with a sparse swap map — O(k)
// expected time, O(k) memory (no [0,N) array). Each of the C(n,k) subsets is
// equally likely. This is the spec-sanctioned correct alternative to Vitter's
// geometric-skip method (which has tricky overshoot edge cases near the end
// of the range); see the inline comment in draw_random_sample for details.
//
// Reference: Vitter, "Faster Methods for Random Sampling" (1987); Knuth,
// TAOCP vol. 2 (Fisher-Yates / Algorithm P).
// ---------------------------------------------------------------------------

/// Element type tag for draw_random_sample's cast logic.
enum class SampleElemType { Float32, Int8, Uint8 };

SampleElemType infer_sample_elem_type(const std::string& path) {
    auto ends_with_ci = [&](const char* suf) {
        const size_t nn = path.size();
        const size_t mm = std::strlen(suf);
        if (nn < mm) return false;
        for (size_t i = 0; i < mm; i++) {
            char c = path[nn - mm + i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != suf[i]) return false;
        }
        return true;
    };
    if (ends_with_ci(".ibin")) return SampleElemType::Int8;
    if (ends_with_ci(".bbin") || ends_with_ci(".bvecs"))
        return SampleElemType::Uint8;
    return SampleElemType::Float32;
}

/// Draw `k` vectors as a uniform random sample without replacement from a
/// .fbin/.ibin/.bbin file, returning them as a flat float buffer (k × dim).
///
/// @param path   file path (.fbin/.ibin/.bbin/.bvecs).
/// @param n      vector count (from the file header).
/// @param dim    vector dimensionality.
/// @param k      sample size (must be <= n).
/// @return std::vector<float> of size k*dim, row-major. Vectors are in
///         ascending file-index order (callers don't care about order — PQ
///         training and LID are order-invariant).
/// @throws Error if the file can't be opened or k > n.
std::vector<float> draw_random_sample(const std::string& path,
                                       uint64_t n, Dim dim, uint64_t k) {
    if (k == 0) return {};
    if (k > n) {
        throw Error(ErrorCode::InvalidParam,
                    "draw_random_sample: k (" + std::to_string(k) +
                        ") > n (" + std::to_string(n) + ")");
    }

    const SampleElemType etype = infer_sample_elem_type(path);
    const uint32_t elem_size = (etype == SampleElemType::Float32) ? 4 : 1;
    const size_t vec_bytes = static_cast<size_t>(dim) * elem_size;

    // --- Generate k sorted distinct indices in [0, n). ---
    //
    // We use partial Fisher-Yates with a sparse swap map: iterate i = 0..k-1,
    // pick j uniformly in [i, n), and swap the values at logical positions i
    // and j. Because only positions that have been swapped are recorded in the
    // map, memory is O(k) — NOT the O(n) a full [0,n) index array would cost
    // (infeasible at N=1B). Each of the C(n,k) subsets of size k is equally
    // likely, so every vector has selection probability exactly k/N —
    // statistically identical to Algorithm R reservoir sampling.
    //
    // (This is the spec-sanctioned correct alternative to Vitter's
    // geometric-skip method: it avoids the overshoot edge cases that make
    // skip-value generation tricky near the end of the range, while still
    // being O(k) expected time and O(k) memory.)
    std::vector<uint64_t> indices;
    indices.reserve(static_cast<size_t>(k));
    std::unordered_map<uint64_t, uint64_t> swap_map;
    swap_map.reserve(static_cast<size_t>(k) * 2);

    const auto val_at = [&](uint64_t i) -> uint64_t {
        const auto it = swap_map.find(i);
        return it == swap_map.end() ? i : it->second;
    };
    const auto set_at = [&](uint64_t i, uint64_t v) { swap_map[i] = v; };

    std::mt19937_64 rng(0xC0DE1234ULL);  // same seed as reservoir for continuity
    for (uint64_t i = 0; i < k; ++i) {
        const uint64_t span = n - i;
        const uint64_t j = i + (rng() % span);
        const uint64_t vi = val_at(i);
        const uint64_t vj = val_at(j);
        set_at(i, vj);
        if (j != i) set_at(j, vi);
        indices.push_back(vj);  // the value now at position i
    }
    // Sort into ascending file-index order (an artifact callers don't care
    // about — PQ training and LID are order-invariant).
    std::sort(indices.begin(), indices.end());

    // --- Debug-only correctness invariants. ---
    // The generated sequence must be exactly k indices, strictly increasing,
    // all within [0, n).
    if (indices.size() != k) {
        throw Error(ErrorCode::InvalidParam,
                    "draw_random_sample: produced " +
                        std::to_string(indices.size()) + " indices, expected " +
                        std::to_string(k));
    }
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= n) {
            throw Error(ErrorCode::InvalidParam,
                        "draw_random_sample: index " +
                            std::to_string(indices[i]) + " >= n=" +
                            std::to_string(n));
        }
        if (i > 0 && indices[i] <= indices[i - 1]) {
            throw Error(ErrorCode::InvalidParam,
                        "draw_random_sample: indices not strictly increasing "
                        "at position " +
                            std::to_string(i));
        }
    }
    spdlog::debug("[sextant] seek-based sampling: k={} of N={}, k seeks",
                  k, n);

    // --- Read the selected vectors via seek + read. ---
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw Error(ErrorCode::IoError,
                    "draw_random_sample: failed to open '" + path + "': " +
                        std::strerror(errno));
    }

    std::vector<float> out(static_cast<size_t>(k) * dim);
    // Reusable raw buffer for i8/u8 casts.
    std::vector<uint8_t> raw;
    if (etype != SampleElemType::Float32) {
        raw.resize(vec_bytes);
    }

    double io_sec = 0.0;
    size_t seek_count = 0;

    for (size_t i = 0; i < indices.size(); i++) {
        const uint64_t idx = indices[i];
        const off_t off = static_cast<off_t>(
            8 + idx * static_cast<uint64_t>(dim) * elem_size);
        float* dst = out.data() + static_cast<size_t>(i) * dim;

        const auto t0 = std::chrono::steady_clock::now();

        // Seek to the vector.
        if (::lseek(fd, off, SEEK_SET) < 0) {
            ::close(fd);
            throw Error(ErrorCode::IoError,
                        "draw_random_sample: lseek failed on '" + path +
                            "': " + std::strerror(errno));
        }
        ++seek_count;

        // Read the vector bytes.
        if (etype == SampleElemType::Float32) {
            size_t got = 0;
            while (got < vec_bytes) {
                ssize_t r = ::read(fd,
                                   reinterpret_cast<uint8_t*>(dst) + got,
                                   vec_bytes - got);
                if (r <= 0) {
                    ::close(fd);
                    throw Error(ErrorCode::CorruptIndex,
                                "draw_random_sample: unexpected EOF on '" +
                                    path + "'");
                }
                got += static_cast<size_t>(r);
            }
        } else {
            size_t got = 0;
            while (got < vec_bytes) {
                ssize_t r = ::read(fd, raw.data() + got, vec_bytes - got);
                if (r <= 0) {
                    ::close(fd);
                    throw Error(ErrorCode::CorruptIndex,
                                "draw_random_sample: unexpected EOF on '" +
                                    path + "'");
                }
                got += static_cast<size_t>(r);
            }
            if (etype == SampleElemType::Int8) {
                for (uint32_t d = 0; d < dim; d++) {
                    dst[d] = static_cast<float>(
                        static_cast<int8_t>(raw[d]));
                }
            } else {  // Uint8
                for (uint32_t d = 0; d < dim; d++) {
                    dst[d] = static_cast<float>(raw[d]);
                }
            }
        }

        io_sec += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }

    ::close(fd);

    spdlog::debug("[sextant] seek-based sampling: read {} vectors in "
                  "{:.3f}s ({} seeks)",
                  k, io_sec, seek_count);

    return out;
}

}  // namespace

Engine::PqSelection Engine::probe_pq_config(const float* sample, uint64_t n,
                                            Dim dim, const ResolvedParams& params) {
    const uint16_t pq_m = params.pq_m;
    const uint8_t pq_bits = params.pq_bits;
    if (pq_m != 0 && pq_bits != 0) {
        // Both explicit — no probe needed.
        return {pq_m, pq_bits, {}, ""};
    }
    const uint32_t pool_n =
        static_cast<uint32_t>(std::min<uint64_t>(kProbePool, n));
    // Pick query indices (deterministic).
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_int_distribution<uint32_t> u(0, pool_n - 1);
    std::vector<uint32_t> qidx;
    std::unordered_set<uint32_t> seen;
    while (qidx.size() < std::min<uint32_t>(kProbeQueries, pool_n - 1)) {
        const uint32_t q = u(rng);
        if (seen.insert(q).second) qidx.push_back(q);
    }
    const auto truth = compute_truth(sample, pool_n, dim, qidx);
    std::vector<ProbedConfig> all_cfg;
    std::string reason;
    const auto best = probe_best_config(sample, pool_n, dim, params.metric,
                                        truth, qidx,
                                        /*fixed_m=*/pq_m,
                                        /*fixed_bits=*/pq_bits,
                                        /*max_distortion=*/params.pq_max_distortion,
                                        all_cfg, reason);
    uint16_t m = best.m;
    uint8_t bits = best.bits;
    if (m == 0) {
        uint32_t mm = dim / 4; if (mm < 4) mm = 4;
        while (mm > 1 && dim % mm != 0) mm--;
        m = static_cast<uint16_t>(mm);
    }
    if (bits == 0) bits = 8;
    // Copy probed rows for display.
    std::vector<Engine::ProbedRow> rows;
    rows.reserve(all_cfg.size());
    for (const auto& c : all_cfg) {
        rows.push_back({c.m, c.bits, c.code_bytes, c.table_bytes,
                        c.distortion, c.band_recall, c.tie_fraction,
                        c.tie30_fraction, c.cost});
    }
    return {m, bits, std::move(rows), std::move(reason)};
}

void Engine::pass1_sample_and_train(VectorSource& source,
                                    const ResolvedParams& params) {
    spdlog::info("[sextant] pass 1: reservoir sample (target {} vectors)",
                 kSampleTarget);

    // Reservoir sampling (Algorithm R). We sample floats directly.
    const uint64_t sample_cap =
        std::min<uint64_t>(kSampleTarget, count_);
    if (sample_cap == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::pass1: cannot sample from empty source");
    }
    std::vector<float> reservoir(static_cast<size_t>(sample_cap) * dim_);
    uint64_t actual_sample = 0;
    uint64_t seen = 0;

    // --- Sampling strategy dispatch ---
    // When N >> sample (and the source is a seekable file), use seek-based
    // random sampling: O(sample) seeks instead of O(N) sequential I/O.
    //
    // Threshold choice (measurement-driven, not heuristic): on this machine
    // a sequential read sustains ~3.5 GB/s (~0.88us per 3KB vector) while a
    // seek+read costs ~26us. Seek-based total cost is ~k*26us regardless of N;
    // full-scan cost is ~N*0.88us. They cross over near N/k ~ 30. Below that,
    // the sequential scan is faster (cache/RAID-stripe prefetching dominates
    // scattered-seek overhead). Above it, seek-based wins, dramatically so as
    // N grows (full-scan reads 3TB at N=1B; seek-based still reads ~60MB).
    // Use N > 30*sample_cap as the cutoff.
    //
    // Only applies when source.path() is non-empty (a real on-disk file);
    // MemorySource and future non-file sources fall back to reservoir.
    const std::string src_path = source.path();
    constexpr uint64_t kSeekRatioThreshold = 30;
    const bool seek_eligible = !src_path.empty() &&
                               count_ > kSeekRatioThreshold * sample_cap;

    if (seek_eligible) {
        spdlog::info("[sextant] pass 1: seek-based sampling ({} of {} vectors; "
                     "ratio {:.1f}:1, threshold {}:1)", sample_cap, count_,
                     static_cast<double>(count_) /
                         static_cast<double>(sample_cap),
                     kSeekRatioThreshold);
        reservoir = draw_random_sample(src_path, count_, dim_, sample_cap);
        actual_sample = sample_cap;
        seen = count_;
    } else {
        spdlog::info("[sextant] pass 1: reservoir sample (target {} vectors)",
                     kSampleTarget);
        source.reset();

    // Algorithm R: keep the first `sample_cap`, then replace index j (j<k)
    // with probability k/i for the i-th seen item.
    //
    // rng() % (seen+1) instead of std::uniform_int_distribution: the latter
    // uses general-purpose rejection sampling (~107M/s); raw modulo is ~5x
    // faster (~530M/s). At billion scale Phase B (the post-fill steady state)
    // is ~9s with the distribution vs ~2s with modulo. The modulo bias at
    // seen ~ 1e9 against the 2^64 range is ~1e-10 — far below the statistical
    // precision a 20K sample carries, and irrelevant for parameter estimation.
    // (Phase B does no memcpy in the steady state: replacement probability
    // drops to sample_cap/seen ~ 2e-5 at 1B, so nearly every iteration is
    // just RNG + compare.)
    //
    // Why not parallelize the sampling? Not because of reproducibility — a
    // fixed seed gives determinism regardless of iteration order, and nothing
    // here pins to a specific sample (the result feeds statistical aggregations).
    // The actual reasons:
    //   1. The workload is one sequential pass over the base file (large
    //      6MB chunked reads). On the GCP bench VM's 2-SSD RAID0, the kernel's
    //      md layer already parallelizes these reads across both devices under
    //      a single thread (read size >> stripe size), so app-level parallel
    //      reads wouldn't add bandwidth.
    //   2. Total I/O dwarfs compute at billion scale (1B × 3KB = 3TB; nothing
    //      in the sampling loop speeds that up). The only way to go faster is
    //      to NOT read the whole file — which is what seek-based sampling
    //      (above) does when N >> sample.
    //   3. The modulo swap above already captured the cheap serial CPU win.
    // Distributed reservoir sampling (per-shard Algorithm R + weighted merge)
    // is a solved algorithm if we ever shard the input across hosts.
    std::mt19937_64 rng(0xC0DE1234ULL);
    Chunk chunk{};
    uint64_t filled = 0;
    // Time I/O (source.next) and compute (the per-vector reservoir update)
    // separately: I/O scales with dataset size, compute is what we'd consider
    // parallelizing. Algorithm R is inherently serial (single ordered RNG
    // stream), so compute parallelism isn't applicable here — but measuring
    // confirms whether it would matter if it were.
    double io_sec = 0.0;
    double compute_sec = 0.0;
    while (true) {
        const auto t_io0 = std::chrono::steady_clock::now();
        const bool got = source.next(chunk);
        io_sec += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_io0).count();
        if (!got) break;
        const auto t_c0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < chunk.count; r++) {
            const float* vec = chunk.vectors + static_cast<size_t>(r) * dim_;
            if (filled < sample_cap) {
                std::memcpy(reservoir.data() + filled * dim_, vec,
                            dim_ * sizeof(float));
                filled++;
            } else {
                const uint64_t j = rng() % (seen + 1);
                if (j < sample_cap) {
                    std::memcpy(reservoir.data() + j * dim_, vec,
                                dim_ * sizeof(float));
                }
            }
            seen++;
        }
        compute_sec += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_c0).count();
    }
    actual_sample = filled;
    {
        const double total_sec = io_sec + compute_sec;
        const double vecs_per_s = (total_sec > 1e-9)
            ? static_cast<double>(seen) / total_sec : 0.0;
        spdlog::debug("[sextant] pass 1: reservoir sampling took {:.3f}s "
                      "(I/O {:.3f}s + compute {:.3f}s); {} vectors scanned "
                      "({:.0f} vecs/s)",
                      total_sec, io_sec, compute_sec, seen, vecs_per_s);
    }
    }  // end reservoir branch

    spdlog::info("[sextant] pass 1: sampled {} / {} vectors", actual_sample,
                 seen);

    // Build requires explicit pq_m and pq_bits. The probe-based auto-selection
    // belongs in `sextant analyze` (the advisory tool); build is a committed
    // path and should never spend 30-130s probing on the reservoir.
    const uint16_t pq_m = params.pq_m;
    const uint8_t pq_bits = params.pq_bits;
    if (pq_m == 0 || pq_bits == 0) {
        throw Error(
            ErrorCode::InvalidParam,
            "build requires explicit --pq-m and --pq-bits (got pq_m=" +
                std::to_string(pq_m) + ", pq_bits=" +
                std::to_string(pq_bits) +
                "). Pass them on the command line, e.g. `sextant build ... "
                "--pq-m 96 --pq-bits 8`, or run `sextant analyze` first to "
                "get a dataset-specific recommendation.");
    }

    // Construct + train the quantizer at the resolved params.
    quantizer_ = std::make_unique<PqQuantizer>(
        params.metric, dim_, pq_m, pq_bits);
    if (params.pq_anisotropy_lambda > 0.0f) {
        quantizer_->set_anisotropy(params.pq_anisotropy_lambda);
    }
    if (params.pq_opq > 0.0f) {
        quantizer_->enable_opq();
    }
    code_size_ = quantizer_->code_size();
    std::string aniso_note;
    if (params.pq_anisotropy_lambda > 0.0f) {
        aniso_note = std::string(", anisotropy=on (covariance-based)");
    }
    if (params.pq_opq > 0.0f) {
        aniso_note += ", opq=on (PCA rotation)";
    }
    spdlog::info("[sextant] pass 1: training PQ (m={}, bits={}{}) on {} samples",
                 pq_m, pq_bits, aniso_note, actual_sample);
    quantizer_->train(reservoir.data(), actual_sample);
    spdlog::info("[sextant] pass 1: PQ trained (code_size={})", code_size_);

    // Compute global k-means centroids for entry-point selection (EP-2).
    // Run on the FP32 reservoir (precise; the reservoir is alive here and
    // destroyed when this function returns). Centroids are stored in FP32
    // and snapped to nearest data vectors at flush time (snap_entry_points_).
    if (actual_sample >= params.n_entry_points) {
        const uint32_t k = std::min<uint32_t>(params.n_entry_points,
                                              static_cast<uint32_t>(actual_sample));
        entry_centroids_.resize(static_cast<size_t>(k) * dim_);
        // Adapt k-means++ + Lloyd from pq_quantizer.cpp (generic algorithm).
        // Inline implementation to avoid cross-module linkage issues.
        std::mt19937_64 rng(0xC0DE1234ULL);
        // k-means++ seeding.
        {
            entry_centroids_.assign(static_cast<size_t>(k) * dim_, 0.0f);
            std::memcpy(entry_centroids_.data(), reservoir.data(),
                        static_cast<size_t>(dim_) * sizeof(float));
            std::vector<float> min_d2(actual_sample,
                                      std::numeric_limits<float>::max());
            for (uint32_t c = 1; c < k; c++) {
                const float* prev = entry_centroids_.data() + static_cast<size_t>(c - 1) * dim_;
                for (uint64_t i = 0; i < actual_sample; i++) {
                    double d = 0.0;
                    const float* v = reservoir.data() + static_cast<size_t>(i) * dim_;
                    for (uint32_t dd = 0; dd < dim_; dd++) {
                        const double diff = double(v[dd]) - double(prev[dd]);
                        d += diff * diff;
                    }
                    if (float(d) < min_d2[i]) min_d2[i] = float(d);
                }
                std::discrete_distribution<uint64_t> dist(min_d2.begin(), min_d2.end());
                const uint64_t picked = dist(rng);
                std::memcpy(entry_centroids_.data() + static_cast<size_t>(c) * dim_,
                            reservoir.data() + static_cast<size_t>(picked) * dim_,
                            static_cast<size_t>(dim_) * sizeof(float));
            }
        }
        // Lloyd iterations (10).
        std::vector<uint32_t> assign(actual_sample, 0);
        for (uint32_t iter = 0; iter < 10; iter++) {
            // Assign.
            for (uint64_t i = 0; i < actual_sample; i++) {
                const float* v = reservoir.data() + static_cast<size_t>(i) * dim_;
                float best = std::numeric_limits<float>::infinity();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < k; c++) {
                    const float* cen = entry_centroids_.data() + static_cast<size_t>(c) * dim_;
                    double d = 0.0;
                    for (uint32_t dd = 0; dd < dim_; dd++) {
                        const double diff = double(v[dd]) - double(cen[dd]);
                        d += diff * diff;
                    }
                    if (float(d) < best) { best = float(d); best_c = c; }
                }
                assign[i] = best_c;
            }
            // Update.
            std::vector<double> sum(static_cast<size_t>(k) * dim_, 0.0);
            std::vector<uint64_t> cnt(k, 0);
            for (uint64_t i = 0; i < actual_sample; i++) {
                const float* v = reservoir.data() + static_cast<size_t>(i) * dim_;
                double* s = sum.data() + static_cast<size_t>(assign[i]) * dim_;
                for (uint32_t dd = 0; dd < dim_; dd++) s[dd] += double(v[dd]);
                cnt[assign[i]]++;
            }
            for (uint32_t c = 0; c < k; c++) {
                float* cen = entry_centroids_.data() + static_cast<size_t>(c) * dim_;
                if (cnt[c] > 0) {
                    const double* s = sum.data() + static_cast<size_t>(c) * dim_;
                    for (uint32_t dd = 0; dd < dim_; dd++)
                        cen[dd] = float(s[dd] / double(cnt[c]));
                }
            }
        }
        spdlog::info("[sextant] pass 1: computed {} FP32 k-means centroids "
                     "for entry-point selection", k);
    }
}

// ===========================================================================
// Pass 2: encode all vectors → codes_buffer_
// ===========================================================================

void Engine::pass2_encode(VectorSource& source,
                          const ResolvedParams& params) {
    const uint32_t nthreads = params.num_threads > 0
                                  ? params.num_threads
                                  : std::thread::hardware_concurrency();

    spdlog::info("[sextant] pass 2: encoding {} vectors ({} threads)", count_,
                 nthreads);

    // Streaming encode: pull vectors one chunk at a time and parallel-encode
    // each chunk directly into codes_buffer_ at the vector's row_id slot.
    // This avoids materializing the full dataset into a transient flat buffer
    // (peak RAM is now chunk_size × dim × 4, not N × dim × 4).
    source.reset();
    Chunk chunk{};
    uint64_t encoded = 0;

    while (source.next(chunk)) {
        const uint32_t chunk_n = chunk.count;
        if (chunk_n == 0) continue;

        // Parallel encode with atomic-counter work-stealing (same pattern as
        // parallel_construct). Each thread encodes disjoint row_ids into the
        // appropriate slot of codes_buffer_. encode() is const (reads only the
        // codebook, writes only its disjoint output slot) → thread-safe.
        std::atomic<uint32_t> next_r{0};
        auto worker = [this, &chunk, chunk_n, &next_r]() {
            const PqQuantizer& q = *quantizer_;
            uint32_t r;
            while ((r = next_r.fetch_add(1, std::memory_order_relaxed))
                   < chunk_n) {
                const RowId rid = chunk.row_ids[r];
                if (rid < 0 || static_cast<uint64_t>(rid) >= count_) {
                    throw Error(ErrorCode::InvalidParam,
                                "Engine::pass2: row_id out of range");
                }
                const float* vec =
                    chunk.vectors + static_cast<size_t>(r) * dim_;
                q.encode(vec,
                         codes_buffer_ + static_cast<size_t>(rid) * code_size_);
            }
        };

        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < std::min(nthreads, chunk_n); t++) {
            pool.emplace_back(worker);
        }
        for (auto& th : pool) th.join();

        encoded += chunk_n;
    }

    spdlog::info("[sextant] pass 2: encoded {} vectors", encoded);
}

// ===========================================================================
// Parallel construct (HDC, Issue 29 Mode C)
// ===========================================================================

void Engine::parallel_construct(const ResolvedParams& params) {
    const uint32_t n = static_cast<uint32_t>(count_);
    const uint32_t nthreads = params.num_threads > 0
                                  ? params.num_threads
                                  : std::thread::hardware_concurrency();
    const uint32_t lut_sz = quantizer_ ? quantizer_->lut_size() : 0;
    construct_into(*core_, n,
                   [](uint32_t id) { return static_cast<RowId>(id); },
                   lut_sz, nthreads, "construct");
}

// ===========================================================================
// construct_into — the canonical parallel construct loop.
// Chunked work-stealing + T5 dynamic L_build + progress logger.
// Shared by K==1 (parallel_construct → identity mapper) and K>1 (shard build
// → membership mapper). No special cases.
// ===========================================================================

void Engine::construct_into(VamanaCore& core, uint32_t count,
                            const std::function<RowId(uint32_t)>& row_id_at,
                            uint32_t lut_sz, uint32_t nthreads,
                            const char* label) {
    if (count == 0) return;
    nthreads = std::max(1u, nthreads);
    spdlog::info("[sextant] {}: {} nodes across {} threads (HDC)", label, count,
                 nthreads);

    // The very first insert must be serialized before spawning tasks: it
    // claims the entry point (see VamanaCore::insert_build_from_code).
    {
        VamanaTLS tls;
        tls.resize(count);
        tls.resize_lut(lut_sz);
        core.insert_build_from_code(0, /*row_id=*/row_id_at(0), tls);
    }

    const uint32_t lo = 1;
    const uint32_t hi = count;
    if (hi <= lo) {
        spdlog::info("[sextant] {}: only entry-point node (n=1)", label);
        return;
    }

    // Thread pool with per-thread VamanaTLS scratch.
    ctpl::thread_pool_tls<VamanaTLS> pool(
        nthreads,
        [count, lut_sz](size_t /*tid*/, std::shared_ptr<VamanaTLS>& tls) {
            tls = std::make_shared<VamanaTLS>();
            tls->resize(count);
            tls->resize_lut(lut_sz);
        });

    // Dynamic work-stealing: threads pull node IDs from a shared atomic
    // counter. Chunked stealing (kChunk=64): each fetch_add grabs a chunk of
    // IDs, reducing atomic-counter contention by ~chunk_size×.
    constexpr uint32_t kChunk = 64;
    std::atomic<uint32_t> next_id{lo};
    core.set_build_progress(&next_id);
    auto worker = [&core, &row_id_at, hi,
                   &next_id](size_t /*tid*/, VamanaTLS& tls) {
        uint32_t chunk_lo;
        while ((chunk_lo = next_id.fetch_add(kChunk, std::memory_order_relaxed)) < hi) {
            const uint32_t chunk_hi = std::min(chunk_lo + kChunk, hi);
            for (uint32_t id = chunk_lo; id < chunk_hi; id++) {
                core.insert_build_from_code(id, row_id_at(id), tls);
            }
        }
    };

    std::vector<std::future<void>> futs;
    for (uint32_t t = 0; t < nthreads; t++) {
        futs.push_back(pool.push(worker));
    }

    // Progress logger: reads the atomic counter every 5s. Zero contention.
    std::thread logger([&]() {
        const auto t_start = std::chrono::steady_clock::now();
        auto t_last = t_start;
        uint32_t last_done = lo;
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const auto now = std::chrono::steady_clock::now();
            const uint32_t done = std::min(
                next_id.load(std::memory_order_relaxed), hi);
            const double elapsed = std::chrono::duration<double>(
                now - t_start).count();
            const double interval = std::chrono::duration<double>(
                now - t_last).count();
            const uint32_t processed = done - lo;
            const uint32_t interval_processed = done - last_done;
            const double rate = processed / elapsed;
            const uint32_t remaining = (hi - lo) - processed;
            const double eta = rate > 0 ? remaining / rate : 0;
            spdlog::info("[sextant] {}: {}/{} nodes ({:.0f}/s, ETA {:.0f}s)",
                         label, processed + 1, hi - lo,
                         interval_processed / interval, eta);
            if (done >= hi) break;
            last_done = done;
            t_last = now;
        }
    });

    for (auto& f : futs) {
        f.get();
    }
    logger.join();
    core.set_build_progress(nullptr);

    spdlog::info("[sextant] {}: all {} nodes inserted", label, count);
}

// ===========================================================================
// Partitioned build (Step 11)
//
// Phases:
//   1. Global PQ train + encode all vectors → codes_buffer_ (global codebook).
//   2. K-means on PQ codes → K shards with closure_factor overlap.
//   3. Per-shard build: each shard builds at R_shard = 2R/3, referencing a
//      contiguous copy of its members' PQ codes. Nodes store GLOBAL row_ids.
//   4. Merge: union neighbor lists per global node, remap shard-local IDs →
//      global IDs, truncate to R by PQ code distance.
//   5. Flush: standard write_sidecars_ (merged graph + global codes).
// ===========================================================================

BuildResult Engine::build_partitioned(VectorSource& source,
                                       const std::string& index_path,
                                       const ResolvedParams& params) {
    using engine_detail::write_padded;
    using engine_detail::fill_header;
    using engine_detail::read_exact;

    // Partitioned build uses HDC shard construct exclusively — RAM savings
    // matter more than marginal quality at large scale.

    spdlog::info("[sextant] build: N={} K={} closure_factor={:.4f}",
                 count_, params.K, params.closure_factor);

    // --- 1. Quantizer (global): train via pass1 (resolves pq_bits if auto) ---
    // pass1 fills the reservoir, runs the global probe if pq_bits==0, then
    // constructs + trains the quantizer. After it, code_size_ is valid.
    pass1_sample_and_train(source, params);

    // node_size for the FINAL build buffer (inline_pq=0). Shards build at
    // R_shard, but the merged result lands in nodes_buffer_ at full R.
    node_size_ = VamanaCore::static_node_size(params.R, 0,
                                              code_size_);

    // Allocate global flat buffers.
    {
        const size_t codes_bytes = static_cast<size_t>(count_) * code_size_;
        const size_t nodes_bytes = static_cast<size_t>(count_) * node_size_;
        codes_buffer_ = static_cast<uint8_t*>(
            aligned_alloc(kDiskAlign, codes_bytes));
        nodes_buffer_ = static_cast<uint8_t*>(
            aligned_alloc(kDiskAlign, nodes_bytes));
        if (!codes_buffer_ || !nodes_buffer_) {
            throw Error(ErrorCode::OutOfMemory,
                        "build_partitioned: buffer alloc failed");
        }
        std::memset(codes_buffer_, 0, codes_bytes);
        std::memset(nodes_buffer_, 0, nodes_bytes);
#ifdef __linux__
        // Same THP hint as the single-partition build path (see Engine::build).
        if (codes_bytes > 0 && madvise(codes_buffer_, codes_bytes,
                                       MADV_HUGEPAGE) != 0) {
            spdlog::debug("[sextant] huge pages unavailable for codes buffer "
                          "(partitioned), using standard pages");
        }
        if (nodes_bytes > 0 && madvise(nodes_buffer_, nodes_bytes,
                                       MADV_HUGEPAGE) != 0) {
            spdlog::debug("[sextant] huge pages unavailable for nodes buffer "
                          "(partitioned), using standard pages");
        }
#endif
    }

    // Encode (pass2). The quantizer needs its cross-distance table for HDC;
    // PqQuantizer builds it during train() in pass1.
    pass2_encode(source, params);

    const uint32_t n = static_cast<uint32_t>(count_);

    // Load raw vectors as FP16 for the FP16 prune. Used by BOTH K=1 (full
    // graph) and K>1 (per-shard copy). They enable the FP16 prune (exact FP16
    // L2sq occlusion check instead of PQ code_distance). Stored as FP16 (half
    // the RAM of FP32) and consumed directly by l2sq_f16 — no per-call
    // conversion.
    {
        const size_t vecs_bytes =
            static_cast<size_t>(count_) * dim_ * sizeof(float16_t);
        raw_vecs_buffer_ = static_cast<float16_t*>(
            aligned_alloc(kDiskAlign, vecs_bytes));
        if (!raw_vecs_buffer_) {
            throw Error(ErrorCode::OutOfMemory,
                        "build_partitioned: raw_vecs_buffer_ alloc failed");
        }
        spdlog::info("[sextant] loading raw vectors as FP16 ({:.1f}MB) for "
                      "HDC (FP16 prune)",
                     vecs_bytes / 1e6);
        source.reset();
        Chunk chunk{};
        uint64_t loaded = 0;
        while (source.next(chunk)) {
            for (uint32_t r = 0; r < chunk.count; r++) {
                const RowId rid = chunk.row_ids[r];
                if (rid >= 0 && static_cast<uint64_t>(rid) < count_) {
                    float16_t* dst = raw_vecs_buffer_ +
                                  static_cast<size_t>(rid) * dim_;
                    const float* src = chunk.vectors +
                                       static_cast<size_t>(r) * dim_;
                    for (uint32_t d = 0; d < dim_; d++) {
                        dst[d] = static_cast<float16_t>(src[d]);
                    }
                    loaded++;
                }
            }
        }
        spdlog::info("[sextant] loaded {} raw vectors as FP16", loaded);
    }

    // =====================================================================
    // K==1 fast path: build the full graph directly into the global buffers.
    // This is the unified monolithic path — K=1 is a special case of the
    // partitioned path with a single identity shard. It reuses parallel_construct
    // (chunked work-stealing + T5 dynamic L_build progress logger) so the graph
    // is bit-identical to the former standalone monolithic build.
    // =====================================================================
    if (params.K == 1) {
        // K==1 full-graph build core. inline_pq=0 (build layout is flat).
        VamanaParams vp_full =
            VamanaParams::from_resolved(params, dim_, 0, 0);

        core_ = std::make_unique<VamanaCore>(vp_full, *quantizer_);
        core_->set_build_codes(codes_buffer_, n);
        core_->set_build_nodes(nodes_buffer_);
        core_->prepare_for_build(n);

        // FlatNodeStore over the flat buffers so beam_search (used in
        // insert_build_from_code) goes through the store interface.
        flat_store_ = std::make_unique<FlatNodeStore>(
            nodes_buffer_, codes_buffer_, node_size_, code_size_);
        core_->set_store(flat_store_.get());

        core_->set_build_vecs(raw_vecs_buffer_);

        parallel_construct(params);

        // Raw vectors are no longer needed after construct (for K=1).
        core_->set_build_vecs(nullptr);
    } else {
    // =====================================================================
    // K>1 partitioned path: partition → per-shard build (R_shard=2R/3) → merge.
    // =====================================================================

    // --- 2. Partition via k-means on PQ codes ---
    auto assignment = partition_codes(*quantizer_, codes_buffer_, n,
                                      code_size_, params.K,
                                      params.closure_factor);
    const uint32_t K = static_cast<uint32_t>(assignment.shards.size());

    // --- 3. Per-shard build ---
    // Each shard: contiguous codes buffer (copy of members' global codes),
    // VamanaCore at R_shard, parallel construct. Nodes store row_id = global ID.
    const uint16_t R_shard = static_cast<uint16_t>(
        std::max<uint32_t>(8u, (2u * params.R) / 3u));
    spdlog::info("[sextant] partitioned: R_shard={} (2R/3, R={})", R_shard,
                 params.R);

    const uint32_t shard_node_size = VamanaCore::static_node_size(
        R_shard, 0, code_size_);

    // Store each shard's node buffer + local→global map for the merge.
    std::vector<std::vector<uint8_t>> shard_node_bufs(K);
    std::vector<std::vector<uint32_t>> shard_local_to_global(K);

    // Build a single VamanaParams template; R/L per shard.
    // Build a single VamanaParams template; R/L per shard. R_shard is the
    // per-shard degree (2R/3); inline_pq=0 (build layout is flat).
    VamanaParams vp_shard =
        VamanaParams::from_resolved(params, dim_, R_shard, 0);

    for (uint32_t k = 0; k < K; k++) {
        auto& members = assignment.shards[k];
        const uint32_t shard_n = static_cast<uint32_t>(members.size());
        if (shard_n == 0) {
            spdlog::warn("[sextant] shard {} empty, skipping", k);
            continue;
        }
        spdlog::info("[sextant] building shard {}/{} ({} vectors, RAM≈{:.1f}MB, "
                     "{} threads)",
                     k, K, shard_n,
                     (static_cast<double>(shard_n) *
                          (code_size_ + shard_node_size +
                           static_cast<size_t>(dim_) * sizeof(float16_t))) /
                         1e6,
                     params.num_threads);

        // Contiguous shard codes: local index i → members[i]'s global code.
        std::vector<uint8_t> shard_codes(
            static_cast<size_t>(shard_n) * code_size_, 0);
        // Contiguous shard FP16 vectors: same mapping, for the FP16 prune.
        std::vector<float16_t> shard_vecs(
            static_cast<size_t>(shard_n) * dim_, 0);
        shard_local_to_global[k].resize(shard_n);
        for (uint32_t i = 0; i < shard_n; i++) {
            const uint32_t gid = members[i];
            shard_local_to_global[k][i] = gid;
            std::memcpy(shard_codes.data() +
                            static_cast<size_t>(i) * code_size_,
                        codes_buffer_ + static_cast<size_t>(gid) * code_size_,
                        code_size_);
            std::memcpy(shard_vecs.data() +
                            static_cast<size_t>(i) * dim_,
                        raw_vecs_buffer_ + static_cast<size_t>(gid) * dim_,
                        dim_ * sizeof(float16_t));
        }

        // Shard node buffer (aligned for direct-IO reuse).
        const size_t shard_nodes_bytes =
            static_cast<size_t>(shard_n) * shard_node_size;
        shard_node_bufs[k].resize(shard_nodes_bytes, 0);
        uint8_t* shard_nodes = shard_node_bufs[k].data();
        uint8_t* shard_codes_ptr = shard_codes.data();

        // Build a fresh VamanaCore for this shard.
        VamanaCore core(vp_shard, *quantizer_);
        core.set_build_codes(shard_codes_ptr, shard_n);
        core.set_build_nodes(shard_nodes);
        core.set_build_vecs(shard_vecs.data());
        core.prepare_for_build(shard_n);

        // Parallel construct via the canonical loop (chunked work-stealing,
        // T5 dynamic L_build, progress logger). Row IDs remapped to global.
        const uint32_t shard_lut_sz = quantizer_ ? quantizer_->lut_size() : 0;
        const uint32_t nthreads = params.num_threads > 0
                                      ? params.num_threads
                                      : std::thread::hardware_concurrency();
        const std::string shard_label = "shard " + std::to_string(k) + "/" + std::to_string(K);
        construct_into(
            core, shard_n,
            [&members](uint32_t local_id) { return static_cast<RowId>(members[local_id]); },
            shard_lut_sz, nthreads, shard_label.c_str());
        // Shard built; node buffer retained in shard_node_bufs[k].
    }

    // --- 4. Merge ---
    // For each global node, collect neighbor lists from all shards that
    // contain it (remapping shard-local IDs → global IDs), union, dedup,
    // truncate to R by PQ distance.
    // Merge is a three-phase streaming pass:
    //   (a) Gather each global node's candidates from its shard neighbor lists
    //       (shard-local IDs remapped to global IDs).
    //   (b) Add reciprocal edges: if A→B is a candidate, B→A becomes one too.
    //       This makes the merged graph undirected and guarantees connectivity
    //       (each shard's Vamana build is internally connected via the shared
    //       entry-point seed, and closure_factor overlap bridges shards).
    //   (c) For each node: dedup + truncate to R by PQ distance, write out.
    spdlog::info("[sextant] merging {} shards into global graph (R={})", K,
                 params.R);

    // (a) Gather candidates into per-node adjacency sets. We store one sorted,
    // deduped neighbor list per node. To bound memory we use vector-of-vectors
    // and reserve R_shard×K max.
    std::vector<std::vector<uint32_t>> adj(n);
    for (uint32_t k = 0; k < K; k++) {
        const auto& l2g = shard_local_to_global[k];
        const uint32_t shard_n = static_cast<uint32_t>(l2g.size());
        for (uint32_t lid = 0; lid < shard_n; lid++) {
            const uint32_t gid = l2g[lid];
            const uint8_t* snode =
                shard_node_bufs[k].data() +
                static_cast<size_t>(lid) * shard_node_size;
            const uint16_t ndeg = VamanaCore::get_neighbor_count(snode);
            for (uint16_t i = 0; i < ndeg; i++) {
                const uint32_t local_nb = VamanaCore::get_neighbor(snode, i);
                if (local_nb >= shard_n) continue;
                const uint32_t gnb = l2g[local_nb];
                if (gnb == gid) continue;  // no self-loops
                adj[gid].push_back(gnb);
            }
        }
    }

    // (b) Add reciprocal edges. For each A→B already gathered, ensure B also
    // lists A. We append to adj[B]; dedup happens in the truncate pass.
    for (uint32_t a = 0; a < n; a++) {
        for (uint32_t b : adj[a]) {
            adj[b].push_back(a);
        }
    }

    // (c) Dedup + truncate to R by PQ distance, write to nodes_buffer_.
    std::vector<uint32_t> seen(n, 0);
    uint32_t visit_token = 0;
    for (uint32_t gid = 0; gid < n; gid++) {
        uint8_t* out_node =
            nodes_buffer_ + static_cast<size_t>(gid) * node_size_;
        std::memset(out_node, 0,
                    kNeighborArrayOffset +
                        static_cast<size_t>(params.R) * sizeof(uint32_t));
        VamanaCore::set_row_id(out_node, static_cast<RowId>(gid));
        VamanaCore::set_internal_id(out_node, gid);
        VamanaCore::set_neighbor_count(out_node, 0);
        VamanaCore::set_inline_pq_count(out_node, 0);

        ++visit_token;
         std::vector<std::pair<float, uint32_t>> cands;
         cands.reserve(adj[gid].size());
         const float16_t* my_vec =
             raw_vecs_buffer_ + static_cast<size_t>(gid) * dim_;
         for (uint32_t gnb : adj[gid]) {
             if (gnb == gid) continue;
             if (seen[gnb] == visit_token) continue;  // dedup
             seen[gnb] = visit_token;
             const float16_t* nb_vec =
                 raw_vecs_buffer_ + static_cast<size_t>(gnb) * dim_;
             const float d = raw_vecs_buffer_
                 ? l2sq_f16(my_vec, nb_vec, dim_)
                 : quantizer_->code_distance(
                       codes_buffer_ + static_cast<size_t>(gid) * code_size_,
                       codes_buffer_ + static_cast<size_t>(gnb) * code_size_);
             cands.emplace_back(d, gnb);
         }
        // Free the adjacency now that we've consumed it.
        std::vector<uint32_t>().swap(adj[gid]);

        // Truncate to R: keep the R closest by PQ distance.
        if (cands.size() > params.R) {
            std::nth_element(
                cands.begin(), cands.begin() + params.R, cands.end(),
                [](const std::pair<float, uint32_t>& a,
                   const std::pair<float, uint32_t>& b) {
                    return a.first < b.first;
                });
            cands.resize(params.R);
        }
        std::sort(cands.begin(), cands.end(),
                  [](const std::pair<float, uint32_t>& a,
                     const std::pair<float, uint32_t>& b) {
                      return a.first < b.first;
                  });

        const uint16_t deg = static_cast<uint16_t>(cands.size());
        VamanaCore::set_neighbor_count(out_node, deg);
        for (uint16_t i = 0; i < deg; i++) {
            VamanaCore::set_neighbor(out_node, i, cands[i].second);
        }
    }

    // (d) Connectivity repair. The reciprocal edges make each node's local
    // neighborhood undirected, but the K shards may still form separate
    // connected components when closure_factor overlap is low on tightly-
    // clustered data. Union-Find detects components; we bridge each minor
    // component to the largest one with a single bidirectional edge between
    // the PQ-closest pair. This guarantees a single connected component.
    {
        std::vector<uint32_t> parent(n);
        for (uint32_t i = 0; i < n; i++) parent[i] = i;
        std::function<uint32_t(uint32_t)> find = [&](uint32_t x) -> uint32_t {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };
        auto uni = [&](uint32_t a, uint32_t b) {
            parent[find(a)] = find(b);
        };
        for (uint32_t a = 0; a < n; a++) {
            const uint8_t* node =
                nodes_buffer_ + static_cast<size_t>(a) * node_size_;
            const uint16_t deg = VamanaCore::get_neighbor_count(node);
            for (uint16_t i = 0; i < deg; i++) {
                uni(a, VamanaCore::get_neighbor(node, i));
            }
        }
        // Count components and their representative (smallest member).
        std::unordered_map<uint32_t, std::vector<uint32_t>> comp_map;
        for (uint32_t i = 0; i < n; i++) comp_map[find(i)].push_back(i);

        if (comp_map.size() > 1) {
            // Identify the largest component as the root.
            uint32_t root_rep = 0;
            size_t root_size = 0;
            for (auto& [rep, members] : comp_map) {
                if (members.size() > root_size) {
                    root_size = members.size();
                    root_rep = rep;
                }
            }
            spdlog::warn("[sextant] merge: {} components (largest={}); "
                         "bridging", comp_map.size(), root_size);

            // For each minor component, bridge to the root via the PQ-nearest
            // pair (greedy O(|comp| × |root|) is too costly for large roots;
            // we sample the root side to 1024 candidates).
            std::vector<uint32_t> root_sample;
            const auto& root_members = comp_map[root_rep];
            if (root_members.size() > 1024) {
                std::mt19937 rng(0xBAD5eed);
                std::uniform_int_distribution<size_t> d(
                    0, root_members.size() - 1);
                root_sample.reserve(1024);
                for (size_t s = 0; s < 1024; s++) {
                    root_sample.push_back(root_members[d(rng)]);
                }
            } else {
                root_sample = root_members;
            }
            for (auto& [rep, members] : comp_map) {
                if (rep == root_rep) continue;
                 // Find nearest (comp_member, root_sample) pair by FP16 L2sq.
                 float best_d = std::numeric_limits<float>::max();
                 uint32_t best_c = members[0];
                 uint32_t best_r = root_sample[0];
                 for (uint32_t c : members) {
                     const float16_t* cv =
                         raw_vecs_buffer_ + static_cast<size_t>(c) * dim_;
                     for (uint32_t r : root_sample) {
                         const float16_t* rv =
                             raw_vecs_buffer_ + static_cast<size_t>(r) * dim_;
                         const float d = raw_vecs_buffer_
                             ? l2sq_f16(cv, rv, dim_)
                             : quantizer_->code_distance(
                                 codes_buffer_ + static_cast<size_t>(c) * code_size_,
                                 codes_buffer_ + static_cast<size_t>(r) * code_size_);
                         if (d < best_d) {
                             best_d = d;
                             best_c = c;
                             best_r = r;
                         }
                     }
                 }
                // Add bidirectional edge best_c ↔ best_r. Append to each
                // node's neighbor list (both have room since R_shard*... but
                // final R may be full). We overwrite the last neighbor slot if
                // full to guarantee the bridge edge exists.
                auto append_edge = [&](uint32_t from, uint32_t to) {
                    uint8_t* node = nodes_buffer_ +
                                    static_cast<size_t>(from) * node_size_;
                    uint16_t deg = VamanaCore::get_neighbor_count(node);
                    uint16_t slot = deg;
                    // Check if 'to' already a neighbor.
                    for (uint16_t i = 0; i < deg; i++) {
                        if (VamanaCore::get_neighbor(node, i) == to) {
                            slot = 0xFFFF;  // already present
                            break;
                        }
                    }
                    if (slot == 0xFFFF) return;
                    if (deg < params.R) {
                        VamanaCore::set_neighbor(node, deg, to);
                        VamanaCore::set_neighbor_count(node, deg + 1);
                    } else {
                        // Replace the farthest neighbor (last slot, since list
                        // is sorted by distance ascending).
                        VamanaCore::set_neighbor(node, params.R - 1, to);
                    }
                };
                append_edge(best_c, best_r);
                append_edge(best_r, best_c);
            }
        }
    }
    // Debug: merged-graph degree histogram (post-repair).
    {
        uint64_t zero_deg = 0;
        uint64_t total_edges = 0;
        uint64_t max_deg = 0;
        for (uint32_t gid = 0; gid < n; gid++) {
            const uint8_t* node =
                nodes_buffer_ + static_cast<size_t>(gid) * node_size_;
            const uint16_t d = VamanaCore::get_neighbor_count(node);
            if (d == 0) zero_deg++;
            total_edges += d;
            if (d > max_deg) max_deg = d;
        }
        spdlog::info("[sextant] merge degrees: zero={}, avg={:.1f}, max={}",
                     zero_deg, static_cast<double>(total_edges) / n, max_deg);
    }

    spdlog::info("[sextant] merge complete; flushing sidecars");

    // Set up the master VamanaCore (full R) over the merged nodes_buffer_ for
    // entry-point computation and final-layout inlining during flush. (K==1
    // already has core_ set up over the global buffers in the fast path above.)
    // K>1 merged-graph build core. inline_pq=0 (build layout is flat).
    VamanaParams vp_full =
        VamanaParams::from_resolved(params, dim_, 0, 0);
    core_ = std::make_unique<VamanaCore>(vp_full, *quantizer_);
    core_->set_build_codes(codes_buffer_, n);
    core_->set_build_nodes(nodes_buffer_);
    core_->prepare_for_build(n);
    }  // end K>1 partitioned branch

    // --- Flush (entry points + final-layout inlining + sidecars) ---
    // Shared by both K==1 (graph built directly) and K>1 (merged graph). For
    // K==1, core_ is already wired to nodes_buffer_/codes_buffer_.
    snap_entry_points_(params);
    if (params.inline_pq_count > 0) {
        spdlog::warn("[sextant] inline_pq_count={} is deprecated and provides no "
                     "benefit (two-cache search handles code locality). Set to 0.",
                     params.inline_pq_count);
    }
    core_->finalize_inline_codes();
    auto bfs = compute_bfs_reorder_(params);
    write_sidecars_(index_path_, bfs, params);

    // Post-flush: switch from flat build buffers to a PagedNodeStore so
    // post-build search is SSD-resident. The sidecars now hold the FINAL-layout
    // graph (node_size includes inline_pq_count). This runs for ALL K — the
    // partitioned path previously omitted it (left flat buffers resident and
    // no paged store), which was an asymmetry vs the monolithic path.
    const uint32_t final_node_size = VamanaCore::static_node_size(
        params.R, params.inline_pq_count, code_size_);
    node_size_ = final_node_size;
    flat_store_.reset();  // disconnect store before freeing buffers
    core_->set_store(nullptr);
    if (codes_buffer_) { aligned_free(codes_buffer_); codes_buffer_ = nullptr; }
    if (nodes_buffer_) { aligned_free(nodes_buffer_); nodes_buffer_ = nullptr; }
    if (raw_vecs_buffer_) { aligned_free(raw_vecs_buffer_); raw_vecs_buffer_ = nullptr; }

    paged_store_ = std::make_unique<PagedNodeStore>(
        index_path_ + ".graph", index_path_ + ".codes",
        final_node_size, code_size_,
        std::max(1u, std::thread::hardware_concurrency()),
        /*cache_size_bytes=*/64ull * 1024 * 1024);  // 64MB default cache
    core_->set_store(paged_store_.get());

    opened_ = true;

    BuildResult result;
    result.index_path = index_path;
    result.n_vectors = count_;
    result.dim = dim_;
    result.R = params.R;
    result.L_build = params.L_build;
    result.pq_m = quantizer_ ? quantizer_->m() : params.pq_m;
    result.pq_bits = quantizer_ ? quantizer_->bits() : params.pq_bits;
    return result;
}

// ===========================================================================
// compute_bfs_reorder_ — pure BFS reorder of build IDs → disk positions.
// write_sidecars_  — streams all four sidecars using the precomputed reorder.
// ===========================================================================

Engine::BfsReorder Engine::compute_bfs_reorder_(const ResolvedParams& params) const {
    const uint32_t n = static_cast<uint32_t>(count_);
    const uint32_t build_node_size = node_size_;  // inline_pq=0 layout
    const auto& raw_entry_points = core_->entry_points();
    BfsReorder bfs;
    bfs.order = compute_bfs_order(nodes_buffer_, n, build_node_size, raw_entry_points);
    bfs.remap.resize(n);
    for (uint32_t new_pos = 0; new_pos < n; new_pos++) {
        bfs.remap[bfs.order[new_pos]] = new_pos;
    }
     return bfs;
 }

void Engine::snap_entry_points_(const ResolvedParams& params) {
    // Snap stored FP32 centroids to nearest data vectors (medoids) and set
    // them as entry points. Falls back to stride sampling if no centroids.
    if (entry_centroids_.empty() || !raw_vecs_buffer_ || count_ == 0) {
        core_->compute_entry_points();  // stride-sampled fallback
        return;
    }
    const uint32_t n = static_cast<uint32_t>(count_);
    const uint32_t k = static_cast<uint32_t>(entry_centroids_.size() / dim_);
    // Parallel medoid search: each centroid scans the full FP16 dataset
    // for its nearest data vector. O(k × N × dim) but embarrassingly
    // parallel across centroids.
    std::vector<uint32_t> medoid_ids(k);
    std::vector<std::thread> pool;
    auto worker = [&](uint32_t c) {
        const float* centroid = entry_centroids_.data() + static_cast<size_t>(c) * dim_;
        // Convert centroid to FP16 for l2sq_f16 comparison.
        std::vector<float16_t> centroid_f16(dim_);
        for (uint32_t d = 0; d < dim_; d++)
            centroid_f16[d] = static_cast<float16_t>(centroid[d]);
        float best_d = std::numeric_limits<float>::infinity();
        uint32_t best_id = 0;
        for (uint32_t i = 0; i < n; i++) {
            const float d = l2sq_f16(centroid_f16.data(),
                                     raw_vecs_buffer_ + static_cast<size_t>(i) * dim_,
                                     dim_);
            if (d < best_d) { best_d = d; best_id = i; }
        }
        medoid_ids[c] = best_id;
    };
    for (uint32_t c = 0; c < k; c++) pool.emplace_back(worker, c);
    for (auto& t : pool) t.join();
    // Dedup (two centroids might snap to the same medoid).
    std::sort(medoid_ids.begin(), medoid_ids.end());
    medoid_ids.erase(std::unique(medoid_ids.begin(), medoid_ids.end()),
                     medoid_ids.end());
    // Top up with stride-sampled fallbacks if dedup reduced the count.
    while (medoid_ids.size() < k) {
        uint32_t id = static_cast<uint32_t>(medoid_ids.size() * n) / k;
        if (std::find(medoid_ids.begin(), medoid_ids.end(), id) == medoid_ids.end())
            medoid_ids.push_back(id);
        else break;
    }
    core_->set_entry_points(std::move(medoid_ids));
    spdlog::info("[sextant] entry points: {} k-means medoids (FP32 centroids, "
                 "FP16 snap)", core_->entry_points().size());
}


void Engine::write_sidecars_(const std::string& index_path,
                             const BfsReorder& bfs,
                             const ResolvedParams& params) {
    const auto uuid = make_uuid();
    const uint32_t n = static_cast<uint32_t>(count_);

    // Final node layout: node_size with the resolved inline_pq_count.
    const uint32_t final_node_size = VamanaCore::static_node_size(
        params.R, params.inline_pq_count, code_size_);
    const uint32_t build_node_size = node_size_;  // inline_pq=0 layout

    // ----- .codes (reordered to BFS order, STREAMED) -----
    {
        const std::string path = index_path + ".codes";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicCodes, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);

        const size_t codes_bytes = static_cast<size_t>(n) * code_size_;
        // Stream the reorder through a block-aligned ring buffer (256KB) instead
        // of materializing the full N×code_size in RAM. At 1B×96B the old path
        // allocated 96GB transiently here.
        //
        // Chunk sizing: each FULL chunk must hold a kDiskAlign-multiple of bytes
        // so write_padded emits it verbatim (no interior zero-padding). Only the
        // final partial chunk is padded — matching the former single-write tail,
        // which is what keeps the on-disk layout byte-identical to the old path.
        const size_t block_cap = kBlockSize;  // 256KB
        const uint32_t align_step =
            kDiskAlign / std::gcd(kDiskAlign, code_size_);
        uint32_t codes_per_block =
            static_cast<uint32_t>(block_cap / code_size_);
        codes_per_block -= codes_per_block % align_step;  // round down
        codes_per_block = std::max<uint32_t>(codes_per_block, align_step);
        const size_t buf_cap = static_cast<size_t>(codes_per_block) * code_size_;
        uint8_t* ring = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, buf_cap));
        if (!ring) {
            throw Error(ErrorCode::OutOfMemory,
                        "write_sidecars_: .codes ring alloc failed");
        }

        uint64_t write_off = sizeof(h);
        uint32_t in_block = 0;
        for (uint32_t new_pos = 0; new_pos < n; new_pos++) {
            const uint32_t old_id = bfs.order[new_pos];
            std::memcpy(ring + static_cast<size_t>(in_block) * code_size_,
                        codes_buffer_ + static_cast<size_t>(old_id) * code_size_,
                        code_size_);
            if (++in_block >= codes_per_block) {
                write_padded(f, ring,
                             static_cast<size_t>(in_block) * code_size_, write_off);
                write_off += static_cast<size_t>(in_block) * code_size_;
                in_block = 0;
            }
        }
        if (in_block > 0) {
            write_padded(f, ring,
                         static_cast<size_t>(in_block) * code_size_, write_off);
        }
        aligned_free(ring);
        f.sync();
        spdlog::info("[sextant] wrote {} ({} bytes, BFS-reordered, streamed)",
                     path, codes_bytes);
    }

    // ----- .graph (reformat to final layout, inline neighbor PQ codes) -----
    {
        const std::string path = index_path + ".graph";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicGraph, count_, dim_, uuid);
        // Stash the final node_size in the checksum_algo field? No — keep
        // header clean; the reader recomputes final_node_size from params.
        write_padded(f, &h, sizeof(h), 0);

        // For each build node, produce a final-layout node in a staging
        // buffer and write it. We batch into a ring of aligned blocks to
        // amortize syscalls (Issue 32).
        const size_t block_cap = kBlockSize;  // 256KB
        const uint32_t per_block =
            std::max<uint32_t>(1, static_cast<uint32_t>(block_cap / final_node_size));
        const size_t buf_cap = static_cast<size_t>(per_block) * final_node_size;
        uint8_t* ring = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, buf_cap));

        const uint32_t neighbor_region_end =
            kNeighborArrayOffset + params.R * sizeof(uint32_t);
        const uint32_t inline_region_off =
            (neighbor_region_end + 7u) & ~7u;

        uint64_t write_off = sizeof(h);
        uint32_t in_block = 0;
        for (uint32_t new_pos = 0; new_pos < n; new_pos++) {
            // PageShuffle: emit nodes in BFS order. The node at disk position
            // new_pos is the build node whose old_id = bfs.order[new_pos].
            const uint32_t old_id = bfs.order[new_pos];
            const uint8_t* src = nodes_buffer_ +
                                 static_cast<size_t>(old_id) * build_node_size;
            uint8_t* dst = ring + static_cast<size_t>(in_block) * final_node_size;

            // Copy the fixed header + neighbor array (identical in both layouts
            // since the build and final layouts share the same R).
            const uint32_t copy_len = (neighbor_region_end + 7u) & ~7u;
            std::memcpy(dst, src, copy_len);

            // PageShuffle: remap this node's internal_id and every neighbor ID
            // from build (old) IDs to BFS (new) IDs.
            VamanaCore::set_internal_id(dst, new_pos);
            const uint16_t ndeg = VamanaCore::get_neighbor_count(dst);
            for (uint16_t i = 0; i < ndeg; i++) {
                const uint32_t old_nb = VamanaCore::get_neighbor(dst, i);
                if (old_nb < n) {
                    VamanaCore::set_neighbor(dst, i, bfs.remap[old_nb]);
                }
            }

            // Inline the first inline_pq_count neighbors' PQ codes. The code
            // bytes belong to the (old) vector; neighbor labels are remapped
            // above, but code contents are order-independent.
            if (params.inline_pq_count > 0) {
                const uint16_t nin = std::min<uint16_t>(
                    ndeg, params.inline_pq_count);
                for (uint16_t i = 0; i < nin; i++) {
                    const uint32_t nb = VamanaCore::get_neighbor(src, i);
                    if (nb < n) {
                        std::memcpy(
                            dst + inline_region_off +
                                static_cast<size_t>(i) * code_size_,
                            codes_buffer_ + static_cast<size_t>(nb) * code_size_,
                            code_size_);
                    }
                }
            }

            in_block++;
            if (in_block >= per_block) {
                write_padded(f, ring,
                             static_cast<size_t>(in_block) * final_node_size,
                             write_off);
                write_off += static_cast<size_t>(in_block) * final_node_size;
                in_block = 0;
            }
        }
        if (in_block > 0) {
            write_padded(f, ring,
                         static_cast<size_t>(in_block) * final_node_size,
                         write_off);
        }
        aligned_free(ring);
        f.sync();
        spdlog::info("[sextant] wrote {} ({} nodes, {} bytes/node)", path, n,
                     final_node_size);
    }

    // ----- .meta (serialized quantizer + entry points + params) -----
    // PageShuffle: remap entry points from build (old) IDs to BFS (new) IDs
    // so the search path starts at the correct disk positions.
    {
        const auto& eps = core_->entry_points();
        std::vector<uint32_t> remapped_eps;
        remapped_eps.reserve(eps.size());
        for (uint32_t ep : eps) {
            remapped_eps.push_back(ep < n ? bfs.remap[ep] : ep);
        }
        write_meta_file(params, remapped_eps, uuid);
    }

    // ----- .vecs (FP16 for the MemGraph entry-point ball) -----
    // Ball-only: stores FP16 vectors for the nodes within 3 hops of the entry
    // points. At search time, beam_search uses l2sq_f16 for these nodes (high
    // precision in the approach phase) and PQ for the rest. Scales: the file
    // size is bounded by ball size (entry_points × R^hops), not N.
    //
    // The entries are in the SAME BFS-visit order as MemGraph's `collected_`
    // vector at open time. The build-time BFS here runs on nodes_buffer_
    // (build-order, neighbors in build-order); the open-time BFS runs on the
    // .graph file (disk-order, neighbors remapped via bfs.remap). The visit
    // ORDER is identical because the remap preserves neighbor-list ordering
    // (only IDs change, not positions). So .vecs position i == MemGraph local
    // index i.
    {
        const std::string path = index_path + ".vecs";
        constexpr uint32_t kVecsNumHops = 3;
        // Run the same BFS MemGraph runs at open time: from the (build-order)
        // entry points, 3 hops, on nodes_buffer_ (build-order). Collect
        // build-order IDs in BFS-visit order.
        std::vector<uint8_t> visited(n, 0);
        struct BfsItem { uint32_t id; uint32_t hop; };
        std::deque<BfsItem> bfs_queue;
        std::vector<uint32_t> ball_ids;
        const auto& eps = core_->entry_points();  // build-order IDs
        for (uint32_t ep : eps) {
            if (ep < n && !visited[ep]) {
                visited[ep] = 1;
                bfs_queue.push_back({ep, 0});
            }
        }
        while (!bfs_queue.empty()) {
            const BfsItem it = bfs_queue.front();
            bfs_queue.pop_front();
            ball_ids.push_back(it.id);
            if (it.hop >= kVecsNumHops) continue;
            const uint8_t* node =
                nodes_buffer_ + static_cast<size_t>(it.id) * build_node_size;
            const uint16_t ncount = VamanaCore::get_neighbor_count(node);
            for (uint16_t i = 0; i < ncount; i++) {
                const uint32_t nb = VamanaCore::get_neighbor(node, i);
                if (nb < n && !visited[nb]) {
                    visited[nb] = 1;
                    bfs_queue.push_back({nb, it.hop + 1});
                }
            }
        }

        // Write .vecs: header + ball_ids.size() × dim × float16_t, streamed
        // through a block-aligned ring buffer (like .codes above) to amortize
        // syscalls.
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicVecs, ball_ids.size(), dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);

        const size_t vec_bytes = static_cast<size_t>(dim_) * sizeof(float16_t);
        const size_t block_cap = kBlockSize;  // 256KB
        const uint32_t vecs_per_block =
            std::max<uint32_t>(1u, static_cast<uint32_t>(block_cap / vec_bytes));
        const size_t buf_cap = static_cast<size_t>(vecs_per_block) * vec_bytes;
        uint8_t* ring = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, buf_cap));
        if (!ring) {
            throw Error(ErrorCode::OutOfMemory,
                        "write_sidecars_: .vecs ring alloc failed");
        }
        uint64_t write_off = sizeof(h);
        uint32_t in_block = 0;
        for (uint32_t bid : ball_ids) {
            const float16_t* vec =
                raw_vecs_buffer_ + static_cast<size_t>(bid) * dim_;
            std::memcpy(ring + static_cast<size_t>(in_block) * vec_bytes,
                        vec, vec_bytes);
            if (++in_block >= vecs_per_block) {
                write_padded(f, ring,
                             static_cast<size_t>(in_block) * vec_bytes, write_off);
                write_off += static_cast<size_t>(in_block) * vec_bytes;
                in_block = 0;
            }
        }
        if (in_block > 0) {
            write_padded(f, ring,
                         static_cast<size_t>(in_block) * vec_bytes, write_off);
        }
        aligned_free(ring);
        f.sync();
        spdlog::info("[sextant] wrote {} ({} FP16 vectors, {} bytes each)",
                     path, ball_ids.size(), dim_ * sizeof(float16_t));
    }

    // ----- .manifest (atomic commit — written LAST via temp + rename) -----
    write_manifest_file(params, uuid);
}

// ===========================================================================
// write_meta_file / write_manifest_file — shared .meta and .manifest writers.
//
// Both write_sidecars_ (build path) and flush (post-insert path) emit the same
// .meta payload (serialized quantizer + entry points + ResolvedParams) and the
// same .manifest commit point. The only caller-specific detail is the entry-
// point vector: write_sidecars_ BFS-remaps build IDs to disk positions, while
// flush passes core_->entry_points() verbatim (buffers are already final).
// Callers prepare the entry-point vector and pass it in.
// ===========================================================================

void Engine::write_meta_file(const ResolvedParams& params,
                              const std::vector<uint32_t>& entry_points,
                              const std::pair<uint64_t, uint64_t>& uuid) {
    const std::string path = index_path_ + ".meta";
    DirectFile f(path, true);
    SidecarHeader h;
    fill_header(h, kMagicMeta, count_, dim_, uuid);
    write_padded(f, &h, sizeof(h), 0);

    // Serialize the quantizer.
    std::vector<uint8_t> qblob;
    quantizer_->serialize(qblob);
    uint64_t qsize = qblob.size();
    // Payload layout: [u64 quantizer_size][quantizer_bytes]
    //                 [u16 entry_point_count][entry_point_count × u32]
    //                 [ResolvedParams POD block]
    std::vector<uint8_t> payload;
    payload.insert(payload.end(),
                   reinterpret_cast<uint8_t*>(&qsize),
                   reinterpret_cast<uint8_t*>(&qsize) + sizeof(qsize));
    payload.insert(payload.end(), qblob.begin(), qblob.end());

    uint16_t ep_count = static_cast<uint16_t>(entry_points.size());
    payload.insert(payload.end(),
                   reinterpret_cast<uint8_t*>(&ep_count),
                   reinterpret_cast<uint8_t*>(&ep_count) + sizeof(ep_count));
    for (uint32_t ep : entry_points) {
        payload.insert(payload.end(),
                       reinterpret_cast<uint8_t*>(&ep),
                       reinterpret_cast<uint8_t*>(&ep) + sizeof(ep));
    }

    // Append the resolved params as a POD block so open() can rebuild the
    // VamanaCore with the same R/L/alpha/inline_pq/max_occlusion.
    ResolvedParams p = params;  // copy
    payload.insert(payload.end(),
                   reinterpret_cast<uint8_t*>(&p),
                   reinterpret_cast<uint8_t*>(&p) + sizeof(p));

    write_padded(f, payload.data(), payload.size(), sizeof(h));
    f.sync();
    spdlog::info("[sextant] wrote {} ({} bytes payload, {} entry points)",
                 path, payload.size(), entry_points.size());
}

void Engine::write_manifest_file(const ResolvedParams& params,
                                  const std::pair<uint64_t, uint64_t>& uuid) {
    const std::string path = index_path_ + ".manifest";
    const std::string tmp = path + ".tmp";
    {
        DirectFile f(tmp, true);
        SidecarHeader h;
        fill_header(h, kMagicManifest, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);
        // The manifest is the commit point. We record the four sidecar
        // basenames + a "ready" marker.
        std::string commit =
            std::string("ready\n") +
            std::to_string(count_) + "\n" +
            std::to_string(dim_) + "\n" +
            std::to_string(params.R) + "\n" +
            std::to_string(params.pq_m) + "\n";
        write_padded(f, commit.data(), commit.size(), sizeof(h));
        f.sync();
    }
    // Atomic rename: the manifest appears all at once, signaling a
    // complete, consistent build.
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        throw Error(ErrorCode::IoError,
                    "Engine::flush: manifest rename failed: " + ec.message());
    }
    spdlog::info("[sextant] wrote {} (commit point)", path);
}

// ===========================================================================
// insert / flush
//
// Live insert = "build one node". We grow the flat codes/nodes buffers by one
// slot, encode the vector, then drive the Vamana insert flow (beam_search →
// robust_prune → connect_and_prune) via VamanaCore::insert_build_from_code.
//
// PHASE 1: insert requires mutable flat buffers. After open() the engine is
// in SSD-resident (paged) mode; the first insert materializes the flat
// buffers from the sidecar files (O(N) I/O), then switches the core to
// FlatNodeStore. This is acceptable because insert is NOT the hot path.
// The realloc strategy is O(N) per insert (copy the whole buffers).
// ===========================================================================

void Engine::insert(const float* vec, Dim dim, RowId row_id) {
    if (!opened_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: index not opened");
    }
    if (!quantizer_ || !core_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: quantizer/core not initialized");
    }
    if (vec == nullptr) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: null vector");
    }
    if (dim != dim_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: dim mismatch");
    }
    if (code_size_ == 0 || node_size_ == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: code/node size not initialized");
    }

    // Phase 1: insert requires mutable flat buffers. If we're in paged
    // (SSD-resident) mode after open(), materialize the flat buffers from the
    // sidecar files on first insert. The FlatNodeStore is (re)created after the
    // reallocs below so it always points at the live buffers.
    // This is O(N) I/O but acceptable because insert is NOT the hot path.
    if (paged_store_ && !codes_buffer_) {
        spdlog::info("[sextant] insert: materializing flat buffers from sidecars "
                     "for mutable insert");
        // Codes.
        {
            const std::string path = index_path_ + ".codes";
            DirectFile f(path, false);
            const size_t codes_bytes =
                static_cast<size_t>(count_) * code_size_;
            codes_buffer_ = static_cast<uint8_t*>(
                aligned_alloc(kDiskAlign, codes_bytes));
            std::memset(codes_buffer_, 0, codes_bytes);
            read_exact(f, codes_buffer_, codes_bytes, sizeof(SidecarHeader));
        }
        // Nodes.
        {
            const std::string path = index_path_ + ".graph";
            DirectFile f(path, false);
            const size_t nodes_bytes =
                static_cast<size_t>(count_) * node_size_;
            nodes_buffer_ = static_cast<uint8_t*>(
                aligned_alloc(kDiskAlign, nodes_bytes));
            std::memset(nodes_buffer_, 0, nodes_bytes);
            read_exact(f, nodes_buffer_, nodes_bytes, sizeof(SidecarHeader));
        }
        paged_store_.reset();
        core_->set_store(nullptr);  // cleared; recreated below
    }

    const uint32_t new_internal = static_cast<uint32_t>(count_);
    const uint64_t new_count = count_ + 1;

    // --- 1. Grow the codes buffer by one code (O(N) copy). ---
    {
        const size_t old_bytes = static_cast<size_t>(count_) * code_size_;
        const size_t new_bytes = static_cast<size_t>(new_count) * code_size_;
        const size_t alloc_bytes = (new_bytes + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
        uint8_t* nb =
            static_cast<uint8_t*>(aligned_alloc(kDiskAlign, alloc_bytes));
        if (!nb) {
            throw Error(ErrorCode::OutOfMemory,
                        "Engine::insert: codes realloc failed");
        }
        std::memcpy(nb, codes_buffer_, old_bytes);
        // Encode the new vector into the appended slot.
        quantizer_->encode(vec, nb + old_bytes);
        aligned_free(codes_buffer_);
        codes_buffer_ = nb;
    }

    // --- 2. Grow the nodes buffer by one node (O(N) copy). ---
    {
        const size_t old_bytes = static_cast<size_t>(count_) * node_size_;
        const size_t new_bytes = static_cast<size_t>(new_count) * node_size_;
        const size_t alloc_bytes = (new_bytes + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
        uint8_t* nb =
            static_cast<uint8_t*>(aligned_alloc(kDiskAlign, alloc_bytes));
        if (!nb) {
            throw Error(ErrorCode::OutOfMemory,
                        "Engine::insert: nodes realloc failed");
        }
        std::memcpy(nb, nodes_buffer_, old_bytes);
        std::memset(nb + old_bytes, 0, node_size_);
        aligned_free(nodes_buffer_);
        nodes_buffer_ = nb;
    }

    // --- 3. Publish the grown buffers + new count to the core. ---
    count_ = new_count;
    core_->set_build_codes(codes_buffer_, static_cast<uint32_t>(count_));
    core_->set_build_nodes(nodes_buffer_);

    // (Re)create the FlatNodeStore over the (possibly realloc'd) live buffers
    // so beam_search inside insert_build_from_code reads current data.
    flat_store_ = std::make_unique<FlatNodeStore>(
        nodes_buffer_, codes_buffer_,
        node_size_, code_size_);
    core_->set_store(flat_store_.get());

    // --- 3b. Grow raw_vecs_buffer_ by one FP16 vector (O(N) copy). ---
    // Needed so build_vec_ptr(new_internal) works for the FP16 prune.
    {
        const size_t old_bytes = static_cast<size_t>(count_ - 1) * dim_ * sizeof(float16_t);
        const size_t new_bytes = static_cast<size_t>(count_) * dim_ * sizeof(float16_t);
        const size_t alloc_bytes = (new_bytes + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
        float16_t* nb =
            static_cast<float16_t*>(aligned_alloc(kDiskAlign, alloc_bytes));
        if (!nb) {
            throw Error(ErrorCode::OutOfMemory,
                        "Engine::insert: raw_vecs realloc failed");
        }
        if (raw_vecs_buffer_) {
            std::memcpy(nb, raw_vecs_buffer_, old_bytes);
            aligned_free(raw_vecs_buffer_);
        }
        float16_t* dst = nb + static_cast<size_t>(new_internal) * dim_;
        for (uint32_t d = 0; d < dim_; d++) {
            dst[d] = static_cast<float16_t>(vec[d]);
        }
        raw_vecs_buffer_ = nb;
    }
    core_->set_build_vecs(raw_vecs_buffer_);

    // --- 4. Drive the Vamana insert flow (single-thread). ---
    //    insert_build_from_code handles the first-node case (becomes an entry
    //    point) and the general case (beam_search → robust_prune →
    //    connect_and_prune). It also bumps core_->count_ to internal_id + 1.
    //    build_vecs_ is set above so the prune uses FP16 L2sq.
    VamanaTLS tls;
    tls.resize(static_cast<uint32_t>(count_));
    tls.resize_lut(quantizer_ ? quantizer_->lut_size() : 0);
    core_->insert_build_from_code(new_internal, row_id, tls);

    spdlog::info("[sextant] insert: row_id={} internal_id={} (count now {})",
                 row_id, new_internal, count_);
}

void Engine::flush() {
    // After build(), all sidecars are already written synchronously.
    // flush() is meaningful after open() + insert(): it rewrites the sidecars
    // with the grown buffers (already in final layout, so no reformatting).
    if (!opened_ || !params_loaded_) {
        return;  // nothing pending
    }
    if (count_ == 0) {
        return;
    }
    if (!quantizer_ || !core_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::flush: quantizer/core not initialized");
    }

    const auto uuid = make_uuid();
    const ResolvedParams& params = loaded_params_;
    const uint32_t n = static_cast<uint32_t>(count_);

    // After open(), nodes_buffer_ is already in final layout (node_size_
    // includes inline_pq_count), so we write it verbatim — no reformat pass.
    spdlog::info("[sextant] flush: persisting {} vectors to '{}'", n,
                 index_path_);

    // ----- .codes -----
    {
        const std::string path = index_path_ + ".codes";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicCodes, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);
        const size_t codes_bytes = static_cast<size_t>(n) * code_size_;
        write_padded(f, codes_buffer_, codes_bytes, sizeof(h));
        f.sync();
    }

    // ----- .graph (verbatim — buffers already in final layout) -----
    {
        const std::string path = index_path_ + ".graph";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicGraph, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);
        const size_t nodes_bytes = static_cast<size_t>(n) * node_size_;
        write_padded(f, nodes_buffer_, nodes_bytes, sizeof(h));
        f.sync();
    }

    // ----- .meta (quantizer + entry points + params) -----
    // Post-insert: entry points are already in final disk layout, so pass
    // them verbatim.
    write_meta_file(params, core_->entry_points(), uuid);

    // ----- .manifest (atomic commit) -----
    write_manifest_file(params, uuid);

    spdlog::info("[sextant] flush: sidecars rewritten (count={})", count_);
}

// ===========================================================================
// estimate_config — sample-driven parameter estimation.
//
// Builds one or more mini-indices on a reservoir sample, measures graph
// topology (OPT-SNG R prediction) and search quality (alpha sweep), and
// returns the optimal build config. Does NOT write sidecar files — all
// measurement is in-RAM on fresh Engine instances.
// ===========================================================================
namespace {
/// Sample target for estimate_config mini-builds. Smaller than the production
/// kSampleTarget (256K) because each candidate (R, alpha) builds a full
/// mini-index — at 256K, 7 mini-builds would take >10 min. 20K keeps each
/// build under ~20s while preserving graph topology (degrees, clustering,
// dead-ends) and search quality (recall is identical at 20K for FlatNodeStore
// since the cache is 100% hits). Matches kProbePool.
inline constexpr uint64_t kEstimateSampleTarget = 20000;
}  // namespace

std::unique_ptr<Engine> Engine::build_mini_(const float* sample,
                                             uint64_t sample_n, Dim dim,
                                             const ResolvedParams& params) {
    auto mini = std::make_unique<Engine>();
    mini->count_ = sample_n;
    mini->dim_ = dim;
    mini->index_path_ = "/dev/null";  // never written; just non-empty

    // Threads: respect any user override (--threads); otherwise use all
    // available cores. The construct loop is work-stealing and embarrassingly
    // parallel — there's no reason to cap below hardware concurrency, and
    // doing so wastes cores during the ~3-minute analyze sweep.
    ResolvedParams mp = params;
    if (mp.num_threads == 0) {
        mp.num_threads = std::thread::hardware_concurrency();
    }

    MemorySource src(sample, sample_n, dim);

    // Mirror build_partitioned K==1 fast path (engine.cpp ~897-1005):
    mini->pass1_sample_and_train(src, mp);
    mini->node_size_ = VamanaCore::static_node_size(mp.R, 0, mini->code_size_);

    // Allocate codes_buffer_ + nodes_buffer_ (aligned, zeroed).
    {
        const size_t codes_bytes =
            static_cast<size_t>(sample_n) * mini->code_size_;
        const size_t nodes_bytes =
            static_cast<size_t>(sample_n) * mini->node_size_;
        mini->codes_buffer_ = static_cast<uint8_t*>(
            aligned_alloc(kDiskAlign, codes_bytes));
        mini->nodes_buffer_ = static_cast<uint8_t*>(
            aligned_alloc(kDiskAlign, nodes_bytes));
        if (!mini->codes_buffer_ || !mini->nodes_buffer_) {
            throw Error(ErrorCode::OutOfMemory,
                        "build_mini_: buffer alloc failed");
        }
        std::memset(mini->codes_buffer_, 0, codes_bytes);
        std::memset(mini->nodes_buffer_, 0, nodes_bytes);
    }

    mini->pass2_encode(src, mp);

    // Load raw vectors as FP16 for the FP16 prune.
    {
        const size_t vecs_bytes =
            static_cast<size_t>(sample_n) * dim * sizeof(float16_t);
        mini->raw_vecs_buffer_ = static_cast<float16_t*>(
            aligned_alloc(kDiskAlign, vecs_bytes));
        if (!mini->raw_vecs_buffer_) {
            throw Error(ErrorCode::OutOfMemory,
                        "build_mini_: raw_vecs_buffer_ alloc failed");
        }
        for (uint64_t i = 0; i < sample_n; i++) {
            float16_t* dst =
                mini->raw_vecs_buffer_ + static_cast<size_t>(i) * dim;
            const float* svec = sample + static_cast<size_t>(i) * dim;
            for (uint32_t d = 0; d < dim; d++) {
                dst[d] = static_cast<float16_t>(svec[d]);
            }
        }
    }

    const uint32_t n = static_cast<uint32_t>(sample_n);

    // Set up core_ + flat_store_ (mirror build_partitioned K==1 lines 984-1000).
    VamanaParams vp = VamanaParams::from_resolved(mp, dim, 0, 0);
    mini->core_ = std::make_unique<VamanaCore>(vp, *mini->quantizer_);
    mini->core_->set_build_codes(mini->codes_buffer_, n);
    mini->core_->set_build_nodes(mini->nodes_buffer_);
    mini->core_->prepare_for_build(n);
    mini->flat_store_ = std::make_unique<FlatNodeStore>(
        mini->nodes_buffer_, mini->codes_buffer_, mini->node_size_,
        mini->code_size_);
    mini->core_->set_store(mini->flat_store_.get());
    mini->core_->set_build_vecs(mini->raw_vecs_buffer_);

    mini->parallel_construct(mp);

    mini->core_->set_build_vecs(nullptr);
    mini->core_->compute_entry_points();

    return mini;
}

GraphStats Engine::measure_graph_stats_(const Engine& mini) {
    GraphStats stats;
    const uint32_t n = static_cast<uint32_t>(mini.count_);
    if (n == 0 || !mini.nodes_buffer_) return stats;

    // avg_degree + dead_end_frac: single pass over all nodes.
    uint64_t total_degree = 0;
    uint32_t dead_ends = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t* node =
            mini.nodes_buffer_ + static_cast<size_t>(i) * mini.node_size_;
        const uint16_t deg = VamanaCore::get_neighbor_count(node);
        total_degree += deg;
        if (deg <= 1) ++dead_ends;
    }
    stats.avg_degree = static_cast<double>(total_degree) / n;
    stats.dead_end_frac = static_cast<double>(dead_ends) / n;

    // clustering_coeff: sample min(1000, N) random nodes. For each sampled
    // node, get its neighbors, count pairs of neighbors that are also mutual
    // neighbors (triangle fraction). Bounded by R² per node.
    std::mt19937 rng(0xABCD1234u);
    const uint32_t sample_cnt = std::min<uint32_t>(1000, n);
    double coeff_sum = 0.0;
    uint32_t coeff_count = 0;
    for (uint32_t s = 0; s < sample_cnt; s++) {
        const uint32_t i = (sample_cnt == n) ? s : rng() % n;
        const uint8_t* node =
            mini.nodes_buffer_ + static_cast<size_t>(i) * mini.node_size_;
        const uint16_t deg = VamanaCore::get_neighbor_count(node);
        if (deg < 2) continue;

        // Collect neighbor IDs into a hash set.
        std::unordered_set<uint32_t> nbrs;
        nbrs.reserve(deg);
        for (uint16_t j = 0; j < deg; j++) {
            nbrs.insert(VamanaCore::get_neighbor(node, j));
        }

        // Count connected pairs among neighbors (triangles through node i).
        uint32_t pairs = 0;
        uint32_t connected = 0;
        for (auto it1 = nbrs.begin(); it1 != nbrs.end(); ++it1) {
            const uint32_t nb1 = *it1;
            if (nb1 >= n) continue;
            const uint8_t* nb1_node =
                mini.nodes_buffer_ + static_cast<size_t>(nb1) * mini.node_size_;
            const uint16_t nb1_deg =
                VamanaCore::get_neighbor_count(nb1_node);
            // Build nb1's neighbor set.
            std::unordered_set<uint32_t> nb1_nbrs;
            for (uint16_t j = 0; j < nb1_deg; j++) {
                nb1_nbrs.insert(
                    VamanaCore::get_neighbor(nb1_node, j));
            }
            auto it2 = it1;
            ++it2;
            for (; it2 != nbrs.end(); ++it2) {
                ++pairs;
                if (nb1_nbrs.count(*it2)) ++connected;
            }
        }
        if (pairs > 0) {
            coeff_sum += static_cast<double>(connected) / pairs;
            ++coeff_count;
        }
    }
    stats.clustering_coeff =
        coeff_count > 0 ? coeff_sum / coeff_count : 0.0;
    // median_lid is filled by estimate_config from truth distances.
    return stats;
}

Engine::SearchQuality Engine::measure_search_(
    const Engine& mini, const float* sample, uint64_t sample_n, Dim dim,
    const std::vector<std::vector<uint32_t>>& truth_ids,
    const std::vector<std::vector<float>>& truth_dists,
    const std::vector<uint32_t>& qidx,
    uint32_t L, uint32_t k, uint32_t rerank) {
    SearchQuality sq{0.0, 0.0};
    if (!mini.core_ || !mini.quantizer_ || qidx.empty()) return sq;

    PqQuantizer& q = *mini.quantizer_;
    std::vector<float> lut(q.lut_size());
    // Over-fetch for rerank: k * rerank (matches the production benchmark tool
    // at benchmark.cpp:225). Previously `k + rerank`, which under-fetched and
    // made the mini-build recall measurement pessimistic.
    const uint32_t fetch_k = k * rerank;

    double recall_sum = 0.0;
    uint64_t prox_in = 0, prox_total = 0;
    constexpr double kEps = 1e-12;

    for (size_t qi = 0; qi < qidx.size(); qi++) {
        const float* qv = sample + static_cast<size_t>(qidx[qi]) * dim;
        q.preprocess_query(qv, lut.data());

        // Production-style search: core_->search returns candidates ranked by
        // PQ LUT distance. We over-fetch k+rerank candidates.
        auto results = mini.core_->search(lut.data(), fetch_k, L, /*io_limit=*/0);
        if (results.empty()) continue;

        // Rerank by true L2sq distance computed from the FP32 sample buffer.
        std::vector<std::pair<float, int32_t>> scored;
        scored.reserve(results.size());
        for (const auto& c : results) {
            if (c.row_id < 0 ||
                static_cast<uint64_t>(c.row_id) >= sample_n) continue;
            const float* bv =
                sample + static_cast<size_t>(c.row_id) * dim;
            double d = 0.0;
            for (uint32_t d2 = 0; d2 < dim; d2++) {
                const double diff = double(qv[d2]) - double(bv[d2]);
                d += diff * diff;
            }
            scored.emplace_back(float(d), c.row_id);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        const uint32_t topk = std::min<uint32_t>(k, scored.size());
        if (topk == 0) continue;

        // Recall@k: fraction of truth_ids[qi][:k] found in top-k.
        std::unordered_set<uint32_t> result_set;
        for (uint32_t i = 0; i < topk; i++) {
            result_set.insert(static_cast<uint32_t>(scored[i].second));
        }
        const uint32_t tk =
            std::min<uint32_t>(k, truth_ids[qi].size());
        uint32_t hits = 0;
        for (uint32_t t = 0; t < tk; t++) {
            if (result_set.count(truth_ids[qi][t])) ++hits;
        }
        recall_sum += double(hits) / double(tk > 0 ? tk : 1);

        // Proximity: ratio = d_target / d_result (>=1 = at/inside band).
        // d_target = the k-th true NN distance (truth_dists[qi][k-1]).
        const float d_target =
            (truth_dists[qi].size() >= k) ? truth_dists[qi][k - 1] :
            (!truth_dists[qi].empty()) ? truth_dists[qi].back() :
            std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < topk; i++) {
            const double d = scored[i].first;
            double ratio;
            if (d_target <= kEps && d <= kEps) {
                ratio = 1.0;
            } else if (d <= kEps) {
                ratio = 2.0;  // found a closer point
            } else if (d_target <= kEps) {
                ratio = 0.0;
            } else {
                ratio = double(d_target) / d;
            }
            if (ratio >= 1.0) ++prox_in;
            ++prox_total;
        }
    }

    sq.recall = recall_sum / qidx.size();
    sq.proximity = prox_total > 0 ? double(prox_in) / prox_total : 0.0;
    return sq;
}

ResolvedParams Engine::estimate_config(VectorSource& source,
                                        const BuildConfig& overrides) {
    const auto t_ec_start = std::chrono::steady_clock::now();
    const uint64_t total_n = source.count();
    const Dim dim = source.dim();
    if (total_n == 0 || dim == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "estimate_config: empty source (n=" +
                        std::to_string(total_n) +
                        ", dim=" + std::to_string(dim) + ")");
    }

    spdlog::info("[sextant] estimate_config: n={} dim={}", total_n, dim);

    // --- 1. Sample (seek-based when N >> sample, reservoir otherwise) ---
    // sample_n = min(kEstimateSampleTarget, total_n). Use the SAME RNG seed
    // (0xC0DE1234) as pass1_sample_and_train for reproducibility. When the
    // source is a seekable file and N > 30 × sample_cap, draw the sample via
    // seek-based random sampling (O(k) I/O) instead of the full O(N) scan.
    // Threshold is measurement-driven: see pass1_sample_and_train for the
    // seek-vs-scan cost crossover analysis.
    const uint64_t sample_cap =
        std::min<uint64_t>(kEstimateSampleTarget, total_n);
    std::vector<float> sample(static_cast<size_t>(sample_cap) * dim);
    uint64_t sample_n = 0;
    uint64_t seen = total_n;

    const std::string src_path = source.path();
    constexpr uint64_t kSeekRatioThreshold = 30;
    const bool seek_eligible = !src_path.empty() &&
                               total_n > kSeekRatioThreshold * sample_cap;
    if (seek_eligible) {
        spdlog::info("[sextant] estimate_config: seek-based sampling ({} of {} "
                     "vectors; ratio {:.1f}:1)", sample_cap, total_n,
                     static_cast<double>(total_n) /
                         static_cast<double>(sample_cap));
        sample = draw_random_sample(src_path, total_n, dim, sample_cap);
        sample_n = sample_cap;
    } else {
        source.reset();
        std::mt19937_64 rng(0xC0DE1234ULL);
        Chunk chunk{};
        seen = 0;
        uint64_t filled = 0;
        while (source.next(chunk)) {
            for (uint32_t r = 0; r < chunk.count; r++) {
                const float* vec = chunk.vectors + static_cast<size_t>(r) * dim;
                if (filled < sample_cap) {
                    std::memcpy(sample.data() + filled * dim, vec, dim * sizeof(float));
                    ++filled;
                } else {
                    // rng() % (seen+1): faster than uniform_int_distribution
                    // (see pass1_sample_and_train for the bias rationale).
                    const uint64_t j = rng() % (seen + 1);
                    if (j < sample_cap) {
                        std::memcpy(sample.data() + j * dim, vec, dim * sizeof(float));
                    }
                }
                ++seen;
            }
        }
        sample_n = filled;
    }
    spdlog::info("[sextant] estimate_config: sampled {} / {} vectors",
                 sample_n, seen);

    // --- Resolve the measurement k and quality targets ---
    // target_k: the k at which recall/proximity are measured. Clamp to
    // [1, min(500, sample_n)] — can't measure recall@k > qidx size (500), and
    // at k > sample_n the truth set is too shallow for a meaningful measurement.
    uint32_t target_k = overrides.target_topk > 0 ? overrides.target_topk : 100;
    {
        const uint32_t kmax = std::min<uint32_t>(500u, static_cast<uint32_t>(sample_n));
        if (target_k > kmax) target_k = kmax;
        if (target_k < 1) target_k = 1;
    }
    spdlog::info("[sextant] estimate_config: target_k={} (measurement k for all "
                 "mini-builds)", target_k);

    // Resolve quality targets. Default: no gate (pick highest-recall config).
    // Both must be met when both are set (stricter binds). No buffer — the
    // mini-build's scale gap (20K sample overestimates recall vs full-scale)
    // is compensated by the search early-exit at runtime, not by over-building.
    double recall_thresh = (overrides.recall_target > 0.0f)
        ? static_cast<double>(overrides.recall_target) : 0.0;
    double prox_thresh = (overrides.proximity_target > 0.0f)
        ? static_cast<double>(overrides.proximity_target) : 0.0;
    // If neither set, no gate — pick the config with the highest recall/proximity.
    spdlog::info("[sextant] estimate_config: quality gate — recall{} "
                 "proximity{} (stricter binds)",
                 recall_thresh > 0.0
                     ? "≥" + std::to_string(recall_thresh) : ": off",
                 prox_thresh > 0.0
                     ? "≥" + std::to_string(prox_thresh) : ": off");

    // Lambda: does a SearchQuality measurement meet the dual gate?
    auto meets_target = [&](const SearchQuality& sq) {
        return (recall_thresh == 0.0 || sq.recall >= recall_thresh) &&
               (prox_thresh == 0.0 || sq.proximity >= prox_thresh);
    };

    // --- 2. PQ config resolution (probe if pq_m or pq_bits auto) ---
    ResolvedParams base = resolve_params(total_n, dim, overrides);
    uint16_t pq_m = overrides.pq_m;
    uint8_t pq_bits = overrides.pq_bits;
    PqSelection pq_sel{0, 0, {}, ""};
    // PQ verify candidates: filled by the probe (section 2), consumed by the
    // verify (section 3, after truth is computed). Ordered ascending m, then
    // 4-bit before 8-bit — stepping forward yields higher fidelity.
    std::vector<ProbedRow> pq_verify_candidates;
    bool pq_verify_pending = false;
    if (pq_m == 0 || pq_bits == 0) {
         // Distortion recalibration: when recall_target is set, scale the
         // distortion bound inversely with the target. The default bound (0.05)
         // achieves recall@100 ≈ 0.99 with rerank=10 at full scale. Lower
         // targets (0.90) can tolerate higher distortion (cheaper PQ); higher
         // targets (0.99+) need tighter distortion. The mapping is linear
         // and conservative — it errs toward over-provisioning PQ quality.
         // The mini-build verify is skipped (mini-build recall@100 is unreliable
         // at small sample sizes; the distortion proxy is more trustworthy).
         ResolvedParams probe_params = base;
         const bool calibrate_distortion = (overrides.recall_target > 0.0f);
         if (calibrate_distortion) {
             // At recall_target=0.95: bound = 0.05 (default — works at full scale).
             // At recall_target=0.90: bound = 0.10 (cheaper PQ acceptable).
             // At recall_target=0.99: bound = 0.02 (tighter PQ needed).
             // Linear interpolation: bound = 0.05 × (1.0 - (target - 0.95) × 3).
             // Clamp to [0.01, 0.20].
             const float t = overrides.recall_target;
             float d_bound = 0.05f * (1.0f - (t - 0.95f) * 3.0f);
             d_bound = std::clamp(d_bound, 0.01f, 0.20f);
             probe_params.pq_max_distortion = d_bound;
             spdlog::info("[sextant] estimate_config: probing PQ config "
                          "(distortion bound {:.4f} calibrated from recall_target "
                          "{:.2f})...", d_bound, t);
         } else {
             spdlog::info("[sextant] estimate_config: probing PQ config...");
         }
        pq_sel = probe_pq_config(sample.data(), sample_n, dim, probe_params);
        pq_m = pq_sel.m;
        pq_bits = pq_sel.bits;
        spdlog::info("[sextant] estimate_config: PQ probe → m={}, bits={}",
                     pq_m, static_cast<int>(pq_bits));

        // Build the verify candidate list: start from the selected config, then
        // all configs that come *after* it in the probe's `all` list (higher m,
        // or 8-bit if 4-bit was selected at the same m). The verify (section 3)
        // steps through these until one meets the recall/proximity target.
        if (!pq_sel.all.empty()) {
            const auto sel_it = std::find_if(
                pq_sel.all.begin(), pq_sel.all.end(),
                [&](const ProbedRow& r) {
                    return r.m == pq_m && r.bits == pq_bits;
                });
            if (sel_it != pq_sel.all.end()) {
                pq_verify_candidates.assign(sel_it, pq_sel.all.end());
            } else {
                pq_verify_candidates = pq_sel.all;
            }
            pq_verify_pending = true;
        }
        spdlog::info("[sextant] estimate_config: PQ → m={}, bits={} "
                     "(pending verify at k={})", pq_m,
                     static_cast<int>(pq_bits), target_k);
    }

    // --- 3. Compute truth + LID ---
    // Query indices: first min(500, sample_n) of the sample.
    const uint32_t nq = std::min<uint32_t>(500, static_cast<uint32_t>(sample_n));
    std::vector<uint32_t> qidx(nq);
    for (uint32_t i = 0; i < nq; i++) qidx[i] = i;

    spdlog::info("[sextant] estimate_config: computing truth ({} queries, "
                 "depth {})...", nq, target_k);
    const auto t_truth_start = std::chrono::steady_clock::now();
    const ProbeTruth truth =
        compute_truth(sample.data(), static_cast<uint32_t>(sample_n), dim,
                      qidx, /*truth_k=*/target_k);
    const double truth_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_truth_start).count();
    spdlog::debug("[sextant] estimate_config: truth compute took {:.3f}s "
                  "({} queries x {} pool x {} dim = {:.1f}G FLOPs)",
                  truth_sec, nq, sample_n, dim,
                  static_cast<double>(nq) * sample_n * dim / 1e9);

    // MLE LID per query: LID = [(1/(k-1)) Σ log(r_k / r_i)]^{-1}.
    // Use the first k=10 true distances.
    const auto t_lid_start = std::chrono::steady_clock::now();
    std::vector<double> lids;
    lids.reserve(nq);
    const uint32_t lid_k = std::min<uint32_t>(10, kProbeTopk);
    for (size_t qi = 0; qi < nq; qi++) {
        const auto& dists = truth.dists[qi];
        if (dists.size() < lid_k) continue;
        const float r_k = dists[lid_k - 1];
        if (r_k <= 1e-12f) continue;
        double sum = 0.0;
        for (uint32_t i = 0; i < lid_k - 1; i++) {
            if (dists[i] <= 1e-12f) { sum = 0; break; }
            sum += std::log(double(r_k) / double(dists[i]));
        }
        if (sum > 1e-9) {
            lids.push_back(double(lid_k - 1) / sum);
        }
    }
    double median_lid = 0.0;
    if (!lids.empty()) {
        std::nth_element(lids.begin(), lids.begin() + lids.size() / 2,
                         lids.end());
        median_lid = lids[lids.size() / 2];
    }
    spdlog::info("[sextant] estimate_config: median LID = {:.2f}", median_lid);
    spdlog::debug("[sextant] estimate_config: LID MLE over {} queries took {:.3f}s",
                  nq, std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t_lid_start).count());

    // d_eff: from overrides if set, else max(5.0, median_lid / 3.0).
    const double d_eff = overrides.closure_d_eff > 0.0f
                             ? overrides.closure_d_eff
                             : std::max(5.0, median_lid / 3.0);

    // --- 4. Closure factor from measured d_eff ---
    const float f_target = overrides.closure_f_target > 0.0f
                               ? overrides.closure_f_target
                               : 0.15f;
    float closure_c = static_cast<float>(
        std::pow(1.0 - f_target, -1.0 / d_eff));
    if (closure_c < 1.0f) closure_c = 1.0f;
    if (closure_c > 1.2f) closure_c = 1.2f;
    spdlog::info("[sextant] estimate_config: closure_factor={:.4f} "
                 "(f_target={:.2f}, d_eff={:.2f})",
                 closure_c, f_target, d_eff);

    // --- 4b. PQ verify (mini-build confirm of the probe selection) ---
    // SKIPPED when recall_target is set: the calibrated distortion bound
    // (section 2 above) already selected the right config, and mini-build
    // recall@100 is unreliable at small sample sizes (the graph is too sparse
    // for k=100 — the ceiling is graph-limited, not PQ-limited). When
    // recall_target is NOT set (proximity-only mode), the verify runs as a
    // diagnostic check on the probe's selection.
    if (pq_verify_pending && overrides.recall_target == 0.0f) {
        spdlog::info("[sextant] estimate_config: PQ verify (mini-build at "
                     "k={}, {} candidates)...", target_k,
                     pq_verify_candidates.size());
        ResolvedParams verify_params = base;
        verify_params.R = 64;
        verify_params.alpha = 1.2f;
        // L must exceed fetch_k = target_k × rerank for the rerank to have
        // enough candidates to pick from.
        const uint32_t verify_L =
            std::max<uint32_t>(target_k * 10u * 2u, 200u);
        const uint32_t verify_rerank = 10;
        bool met = false;
        for (const auto& cand : pq_verify_candidates) {
            verify_params.pq_m = cand.m;
            verify_params.pq_bits = cand.bits;
            auto mini = build_mini_(sample.data(), sample_n, dim, verify_params);
            const auto sq = measure_search_(
                *mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, verify_L, target_k, verify_rerank);
            const bool meets = meets_target(sq);
            const bool is_4bit = (cand.bits == 4);
            spdlog::info("[sextant]   PQ verify: m={} bits={} "
                         "recall@{}={:.4f} proximity={:.4f}{}{}",
                         cand.m, static_cast<int>(cand.bits),
                         target_k, sq.recall, sq.proximity,
                         meets ? " ✓ meets" : " ✗ below",
                         is_4bit ? " [4-bit]" : "");
            if (meets) {
                if (pq_m != cand.m || pq_bits != cand.bits) {
                    pq_m = cand.m;
                    pq_bits = cand.bits;
                    spdlog::info("[sextant] estimate_config: PQ verify stepped "
                                 "up → m={}, bits={} (probe selection didn't "
                                 "meet target)", pq_m,
                                 static_cast<int>(pq_bits));
                }
                if (is_4bit) {
                    spdlog::info("[sextant] estimate_config: 4-bit PQ selected "
                                 "(8× smaller codes → faster search) — meets "
                                 "recall@{}={:.4f}", target_k, sq.recall);
                }
                met = true;
                break;
            }
        }
        if (!met) {
            const auto& last = pq_verify_candidates.back();
            pq_m = last.m;
            pq_bits = last.bits;
            spdlog::warn("[sextant] estimate_config: no PQ config met the "
                         "target in verify; using highest-fidelity candidate "
                         "(m={}, bits={}) as best-effort",
                         pq_m, static_cast<int>(pq_bits));
        }
        spdlog::info("[sextant] estimate_config: PQ → m={}, bits={} "
                     "[verify complete]", pq_m, static_cast<int>(pq_bits));
    }

    // --- 5. Alpha resolution (sweep if alpha auto) ---
    float alpha_rec = overrides.alpha;
    if (alpha_rec == 0.0f) {
        spdlog::info("[sextant] estimate_config: sweeping alpha (k={})...",
                     target_k);
        static constexpr std::array<float, 4> kAlphas = {{1.0f, 1.1f, 1.2f, 1.5f}};
        constexpr uint32_t kAlphaRerank = 10;
        // L (beam width) must exceed fetch_k = k × rerank so the rerank has a
        // deep enough candidate pool. At target_k=100/rerank=10, fetch_k=1000,
        // so L = max(target_k × rerank × 2, 200) = 2000.
        const uint32_t kAlphaL =
            std::max<uint32_t>(target_k * kAlphaRerank * 2u, 200u);

        ResolvedParams alpha_params = base;
        alpha_params.R = 64;
        alpha_params.pq_m = pq_m;
        alpha_params.pq_bits = pq_bits;

        // Dual-gate alpha selection: among alphas that meet the target gate
        // (recall AND proximity, stricter binds), pick the LOWEST alpha (faster
        // build). If none meet target, pick the one with the best recall (or
        // proximity, if recall gate is off) as best-effort and warn.
        float best_meeting_alpha = 0.0f;  // lowest alpha meeting the gate
        float best_effort_alpha = kAlphas.front();
        double best_effort_score = -1.0;  // recall if recall gate, else proximity
        const bool gate_on_recall = (recall_thresh > 0.0);
        for (float a : kAlphas) {
            alpha_params.alpha = a;
            const auto t0 = std::chrono::steady_clock::now();
            auto mini = build_mini_(sample.data(), sample_n, dim, alpha_params);
            const auto sq = measure_search_(
                *mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, kAlphaL, target_k, kAlphaRerank);
            const auto t1 = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(t1 - t0).count();
            const bool meets = meets_target(sq);
            spdlog::info("[sextant]   alpha={:.1f}  recall@{}={:.4f}  "
                         "proximity={:.4f}  L={}{}  ({:.1f}s)",
                         a, target_k, sq.recall, sq.proximity, kAlphaL,
                         meets ? " ✓" : "", dt);
            if (meets && best_meeting_alpha == 0.0f) {
                best_meeting_alpha = a;  // first (lowest) meeting alpha
            }
            const double score = gate_on_recall ? sq.recall : sq.proximity;
            if (score > best_effort_score) {
                best_effort_score = score;
                best_effort_alpha = a;
            }
        }
        if (best_meeting_alpha != 0.0f) {
            alpha_rec = best_meeting_alpha;
            spdlog::info("[sextant] estimate_config: alpha → {:.1f} (lowest "
                         "alpha meeting target gate)", alpha_rec);
        } else {
            alpha_rec = best_effort_alpha;
            spdlog::warn("[sextant] estimate_config: no alpha met target gate; "
                         "selecting best-effort alpha={:.1f} ({}={:.4f})",
                         alpha_rec,
                         gate_on_recall ? "recall" : "proximity",
                         best_effort_score);
        }
    }

    // --- 6. R resolution (OPT-SNG) ---
    // Build reference mini-index at R=64, measure graph stats.
    spdlog::info("[sextant] estimate_config: building reference mini-index "
                 "(R=64) for R prediction...");
    ResolvedParams ref_params = base;
    ref_params.R = 64;
    ref_params.alpha = alpha_rec;
    ref_params.pq_m = pq_m;
    ref_params.pq_bits = pq_bits;

    const auto t_ref0 = std::chrono::steady_clock::now();
    auto ref_mini = build_mini_(sample.data(), sample_n, dim, ref_params);
    GraphStats gstats = measure_graph_stats_(*ref_mini);
    gstats.median_lid = median_lid;
    const auto t_ref1 = std::chrono::steady_clock::now();
    spdlog::info("[sextant] estimate_config: reference graph: R̄={:.2f}, "
                 "clustering={:.4f}, dead_ends={:.4f} ({:.1f}s)",
                 gstats.avg_degree, gstats.clustering_coeff,
                 gstats.dead_end_frac,
                 std::chrono::duration<double>(t_ref1 - t_ref0).count());

    // OPT-SNG: R_predicted = R̄_mini * log(N_full)/log(N_sample) * (alpha_ref/alpha_rec)^2.
    // The reference build is at alpha_rec (whether locked or swept), so
    // alpha_ref == alpha_rec and the coupling term is 1. We write it
    // explicitly for clarity / future extension.
    const double alpha_coupling =
        (alpha_rec != 0.0f)
            ? std::pow(double(alpha_rec) / double(alpha_rec), 2.0)
            : 1.0;
    double R_predicted = gstats.avg_degree *
                         std::log(double(total_n)) / std::log(double(sample_n)) *
                         alpha_coupling;
    // Guardrails (exact thresholds from spec).
    if (gstats.clustering_coeff > 0.12) {
        R_predicted = std::min(R_predicted, 48.0);
        spdlog::info("[sextant]   guardrail: clustering {:.4f} > 0.12 → cap "
                     "R ≤ 48", gstats.clustering_coeff);
    }
    if (gstats.dead_end_frac > 0.05) {
        R_predicted = std::max(R_predicted, 64.0);
        spdlog::info("[sextant]   guardrail: dead_ends {:.4f} > 0.05 → floor "
                     "R ≥ 64", gstats.dead_end_frac);
    }

    uint16_t R_full = static_cast<uint16_t>(
        std::round(std::clamp(R_predicted, 32.0, 128.0)));
    spdlog::info("[sextant] estimate_config: R predicted={:.1f} → R_full={}",
                 R_predicted, R_full);

    if (overrides.R != 0) {
        R_full = overrides.R;
        spdlog::info("[sextant] estimate_config: R locked by override → {} "
                     "(skipping validation sweep)", R_full);
    } else {
        // --- 7. R sweep validation (only when R is estimated) ---
        // Build 2 more mini-indices at R_full-16 and R_full+16 (clamp ≥32).
        // Dual-gate: the sweep-best is the smallest R meeting the target gate
        // (cheapest). If none meet, keep the OPT-SNG prediction.
         spdlog::info("[sextant] estimate_config: R validation sweep (k={})...",
                      target_k);
         const auto t_valid_start = std::chrono::steady_clock::now();
        constexpr uint32_t kValidRerank = 10;
        // L must exceed fetch_k = k × rerank (same rationale as the alpha sweep).
        const uint32_t kValidL =
            std::max<uint32_t>(target_k * kValidRerank * 2u, 200u);

        struct RPoint { uint16_t R; double proximity; double recall; };
        std::vector<RPoint> rpoints;

        // Measure the prediction point R_full (reuse ref_mini if R_full==64).
        if (R_full == 64) {
            const auto sq = measure_search_(
                *ref_mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, kValidL, target_k, kValidRerank);
            rpoints.push_back({R_full, sq.proximity, sq.recall});
            spdlog::info("[sextant]   R={}  recall@{}={:.4f}  proximity={:.4f}{}",
                         R_full, target_k, sq.recall, sq.proximity,
                         meets_target(sq) ? " ✓" : "");
        } else {
            ResolvedParams rp = base;
            rp.R = R_full; rp.alpha = alpha_rec;
            rp.pq_m = pq_m; rp.pq_bits = pq_bits;
            auto mini = build_mini_(sample.data(), sample_n, dim, rp);
            const auto sq = measure_search_(
                *mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, kValidL, target_k, kValidRerank);
            rpoints.push_back({R_full, sq.proximity, sq.recall});
            spdlog::info("[sextant]   R={}  recall@{}={:.4f}  proximity={:.4f}{}",
                         R_full, target_k, sq.recall, sq.proximity,
                         meets_target(sq) ? " ✓" : "");
        }

        for (int delta : {-16, +16}) {
            uint16_t r = static_cast<uint16_t>(std::max(32, int(R_full) + delta));
            ResolvedParams rp = base;
            rp.R = r; rp.alpha = alpha_rec;
            rp.pq_m = pq_m; rp.pq_bits = pq_bits;
            const auto t0 = std::chrono::steady_clock::now();
            auto mini = build_mini_(sample.data(), sample_n, dim, rp);
            const auto sq = measure_search_(
                *mini, sample.data(), sample_n, dim, truth.ids, truth.dists,
                qidx, kValidL, target_k, kValidRerank);
            const auto t1 = std::chrono::steady_clock::now();
            rpoints.push_back({r, sq.proximity, sq.recall});
            spdlog::info("[sextant]   R={}  recall@{}={:.4f}  proximity={:.4f}"
                         "{}  ({:.1f}s)", r, target_k, sq.recall, sq.proximity,
                         meets_target(sq) ? " ✓" : "",
                         std::chrono::duration<double>(t1 - t0).count());
        }

        // Dual-gate selection: among R values meeting the target gate, pick the
        // smallest (cheapest). If none meet, keep the OPT-SNG prediction.
        uint16_t smallest_meeting = 0;
        for (const auto& rp : rpoints) {
            const bool meets =
                (recall_thresh == 0.0 || rp.recall >= recall_thresh) &&
                (prox_thresh == 0.0 || rp.proximity >= prox_thresh);
            if (meets && (smallest_meeting == 0 || rp.R < smallest_meeting)) {
                smallest_meeting = rp.R;
            }
        }
        if (smallest_meeting != 0 && smallest_meeting != R_full) {
            spdlog::info("[sextant]   validation: smallest R meeting target "
                         "gate = {} (prediction was {}) → using sweep-best",
                         smallest_meeting, R_full);
            R_full = smallest_meeting;
        } else if (smallest_meeting != 0) {
            spdlog::info("[sextant]   validation: R={} meets target gate "
                         "(matches prediction)", R_full);
        } else {
            spdlog::warn("[sextant]   validation: no R in sweep met target "
                         "gate; keeping OPT-SNG prediction R={}", R_full);
        }
         spdlog::debug("[sextant] estimate_config: R validation sweep took {:.2f}s",
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t_valid_start).count());
     }

    // --- 8. Final param resolution ---
    ResolvedParams p;
    p.R = R_full;
    p.alpha = alpha_rec;
    p.pq_m = pq_m;
    p.pq_bits = pq_bits;
    // Honor an explicit L override (the T5 dynamic-ramp ceiling). Mirror
    // resolve_params's behavior: overrides.L != 0 locks L_build directly;
    // otherwise L_build = max(2*R, 100). Previously this was silently ignored
    // — an inconsistency between the heuristic and the estimation paths.
    if (overrides.L != 0) {
        p.L_build = overrides.L;
    } else {
        p.L_build = static_cast<uint16_t>(std::max<uint32_t>(2u * p.R, 100u));
    }
    p.L = p.L_build;
    p.max_occlusion = (overrides.max_occlusion != 0)
                           ? overrides.max_occlusion
                           : std::max<uint32_t>(p.L_build,
                                                static_cast<uint32_t>(p.R) + 1u);
    p.closure_factor = closure_c;
    p.inline_pq_count = 0;  // deprecated
    p.measured_median_lid = median_lid;
    p.measured_avg_degree = gstats.avg_degree;
    p.measured_clustering = gstats.clustering_coeff;
    p.measured_dead_end_frac = gstats.dead_end_frac;
    p.build_mode = overrides.build_mode;
    p.metric = overrides.metric;
    p.num_threads = base.num_threads;
    p.build_ram_budget = base.build_ram_budget;
    p.pq_max_distortion = base.pq_max_distortion;

    // Recompute K from the final R (K depends on per-vec size).
    {
        const uint32_t code_sz = static_cast<uint32_t>(p.pq_m);
        const uint32_t node_sz =
            ((16u + static_cast<uint32_t>(p.R) * 4u + 7u) & ~7u);
        const uint64_t per_vec = code_sz + node_sz;
        uint32_t k = 1;
        uint64_t max_per_partition = 0;
        if (per_vec > 0 && p.build_ram_budget > 0) {
            max_per_partition = p.build_ram_budget / per_vec;
            if (max_per_partition > 0) {
                k = static_cast<uint32_t>(
                    (total_n + max_per_partition - 1) / max_per_partition);
                if (k < 1) k = 1;
            }
        }
        p.K = k;
    }

    spdlog::info("[sextant] estimate_config: final → R={} alpha={:.1f} "
                 "pq_m={} pq_bits={} L_build={} K={}",
                 p.R, p.alpha, p.pq_m, static_cast<int>(p.pq_bits), p.L_build,
                  p.K);

    spdlog::info("[sextant] estimate_config: total time {:.1f}s",
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_ec_start).count());
    return p;
}

}  // namespace sextant
