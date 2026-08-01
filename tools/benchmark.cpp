// benchmark — recall + latency benchmark for a Sextant index.
//
// Workflow:
//   1. Open the index.
//   2. Read query vectors from a .fbin file.
//   3. For each query: search + rerank against base data, take top-k row_ids.
//   4. Read ground-truth neighbor IDs from a .gt file.
//   5. Compute Recall@k = |results ∩ gt[:k]| / k, averaged over all queries.
//   6. Report mean recall@k, p50/p99 latency, total search time, QPS.
//
// Usage:
//   benchmark --index myindex --queries query.fbin --base-data base.fbin \
//             --ground-truth gt.gt --k 10 --L 200 --rerank 10

#include "fbin_io.hpp"
#include "sextant/config.hpp"
#include "sextant/crash_handler.hpp"
#include "sextant/ivf_scan_searcher.hpp"
#include "sextant/ivf_searcher.hpp"
#include "sextant/searcher.hpp"
#include "sextant/index.hpp"
#include "sextant/ivf_index.hpp"
#include "sextant/ivf_scan_index.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"
#include "quant/pq_quantizer.hpp"  // PqQuantizer::metric() for rerank dispatch
#include "simd_kernels.hpp"        // simd::dist_f32 — f32-accumulated rerank kernels
// NOTE: NumKong (<numkong/numkong.h>) was used here for the rerank distance
// kernels; replaced by simd::dist_f32 (f32-accumulated, ~2x faster than
// NumKong's f64 nk_sqeuclidean_f32/nk_dot_f32). Don't re-add unless a specific
// nk_ call is needed. NumKong remains linked for breadth — see pq_quantizer.cpp.

#include <cmdline/cmdline.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using sextant::Error;
using sextant::ErrorCode;
using Clock = std::chrono::steady_clock;
using US = std::chrono::duration<double, std::micro>;
using namespace sextant::fbin_io;  // FbinHeader, read_fbin_header, read_fbin_vector, l2sq_distance

// ---------------------------------------------------------------------------
// Ground-truth .gt format:
//   Legacy:  [uint32 n][uint32 k][n×k uint32 ids][n×k float dists]
//   Current: [magic "GTMM" 4B][uint32 n][uint32 k][uint8 metric][n×k uint32 ids]
//            [n×k float dists]
// The magic is detected by peeking the first 4 bytes — old files start with
// the n field (small), so "GTMM" (= 0x4D4D5447 LE = 1.3 billion) never collides
// with a real query count. Current files carry the metric the GT was computed
// under; the benchmark verifies it matches the build's --metric to prevent
// silent recall artifacts (the Sphere-IP-vs-L2sq bug).
inline constexpr uint32_t kGtMagic = 0x4D4D5447u;  // "GTMM" LE

struct GroundTruth {
    uint32_t n = 0;
    uint32_t k = 0;
    sextant::MetricKind metric = sextant::MetricKind::L2Sq;  // default for legacy files
    std::vector<uint32_t> ids;    // n × k, row-major
    std::vector<float> dists;     // n × k, row-major, ascending per row
};

GroundTruth read_ground_truth(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw Error(ErrorCode::IoError,
                    "cannot open ground-truth '" + path + "'");
    }
    GroundTruth gt;

    // Peek the first 4 bytes to detect the magic.
    uint32_t maybe_magic = 0;
    f.read(reinterpret_cast<char*>(&maybe_magic), sizeof(uint32_t));
    if (!f) {
        throw Error(ErrorCode::CorruptIndex,
                    "invalid ground-truth header in '" + path + "'");
    }

    if (maybe_magic == kGtMagic) {
        // Current format: [magic][n][k][metric][ids][dists]
        f.read(reinterpret_cast<char*>(&gt.n), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&gt.k), sizeof(uint32_t));
        uint8_t metric_byte = 0;
        f.read(reinterpret_cast<char*>(&metric_byte), sizeof(uint8_t));
        gt.metric = static_cast<sextant::MetricKind>(metric_byte);
    } else {
        // Legacy format: the 4 bytes we just read were the n field.
        gt.n = maybe_magic;
        f.read(reinterpret_cast<char*>(&gt.k), sizeof(uint32_t));
        gt.metric = sextant::MetricKind::L2Sq;  // legacy files assumed L2sq
    }
    if (!f || gt.n == 0 || gt.k == 0) {
        throw Error(ErrorCode::CorruptIndex,
                    "invalid ground-truth header in '" + path + "'");
    }
    gt.ids.resize(static_cast<size_t>(gt.n) * gt.k);
    f.read(reinterpret_cast<char*>(gt.ids.data()),
           static_cast<std::streamsize>(gt.ids.size() * sizeof(uint32_t)));
    if (!f) {
        throw Error(ErrorCode::CorruptIndex,
                    "short read on ground-truth ids in '" + path + "'");
    }
    // Distances are needed for proximity evaluation (k-th NN radius per query).
    gt.dists.resize(static_cast<size_t>(gt.n) * gt.k);
    f.read(reinterpret_cast<char*>(gt.dists.data()),
           static_cast<std::streamsize>(gt.dists.size() * sizeof(float)));
    if (!f) {
        // Older GT files without distances: tolerate, disable proximity metrics.
        gt.dists.clear();
    }
    return gt;
}

/// Verify the GT's recorded metric matches what the index was built under.
/// Hard-error on mismatch — this is the cheap check that would have caught
/// the Sphere-IP-vs-L2sq ceiling immediately (we wasted ~5h of 8-bit
/// experiments on that artifact before brute-forcing the GT).
/// Returns 1 on mismatch (for run_*_benchmark's return-code convention),
/// 0 on match. Legacy GT files (no metric tag) always match — we can't
/// know what they were computed under, so we trust the user.
int check_gt_metric(const GroundTruth& gt, sextant::MetricKind expected,
                    const std::string& gt_path) {
    if (gt.metric != expected) {
        std::cerr << "benchmark: FATAL — ground-truth metric mismatch.\n"
                  << "  GT '" << gt_path << "' was computed under "
                  << (gt.metric == sextant::MetricKind::InnerProduct
                          ? "inner product"
                          : "squared Euclidean")
                  << ", but the index/search metric is "
                  << (expected == sextant::MetricKind::InnerProduct
                          ? "inner product"
                          : "squared Euclidean")
                  << ".\n  Recall numbers will be WRONG — rebuild with "
                     "--metric "
                  << (gt.metric == sextant::MetricKind::InnerProduct
                          ? "ip"
                          : "l2sq")
                  << " to match the GT.\n  (Override with "
                     "--i-know-the-metric-is-right if you've verified.)\n";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Statistics helpers.
// ---------------------------------------------------------------------------

/// Percentile from a sorted vector of latencies (microseconds).
double percentile(std::vector<double>& sorted_us, double pct) {
    if (sorted_us.empty()) return 0.0;
    if (sorted_us.size() == 1) return sorted_us[0];
    // Linear interpolation between closest ranks.
    const double rank = (pct / 100.0) * (sorted_us.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
     if (lo == hi) return sorted_us[lo];
     const double frac = rank - static_cast<double>(lo);
     return sorted_us[lo] + frac * (sorted_us[hi] - sorted_us[lo]);
 }

/// Rerank distance under a metric. Returns a value ordered consistently with
/// a min-heap / ascending sort (L2sq: ascending; IP: negated dot, so smaller
/// = higher dot product = nearer). Delegates to `simd::dist_f32` in simd_kernels.hpp
/// — f32-accumulated, ~2× the throughput of NumKong's f64-accumulating
/// `nk_sqeuclidean_f32` / `nk_dot_f32` at dim=768. Rerank is 12% of profile.
inline float rerank_dist(sextant::MetricKind metric,
                         const float* a, const float* b, uint32_t dim) {
    return sextant::simd::dist_f32(metric, a, b, dim);
}

/// Per-query metrics produced by a worker thread. The main thread aggregates
/// these — no shared mutable state during the benchmark loop.
struct QueryMetrics {
    double latency_us = 0.0;
    double recall_sum = 0.0;       // hits/gt_k for this query
    uint32_t recall_hits = 0;
    uint32_t recall_total = 0;
    uint64_t prox_in = 0;
    uint64_t prox_total = 0;
    // Proximity ratios for this query's top-k. Stored as a packed vector
    // (rare >0 only when have_gt_dists && base_in_ram). The main thread
    // concatenates these across queries for the distribution report.
    std::vector<double> prox_ratios;
};

/// Shared context for the per-query rerank + recall + proximity logic.
/// Bundles the read-only inputs that the original `process_query` lambda
/// captured by reference, so both the single-index and IVF paths can call
/// the same free `process_results` without duplicating the rerank/recall/
/// proximity code.
struct RerankCtx {
    uint32_t dim;
    uint32_t k;
    uint64_t n_base;            // base-data row count (bh.n)
    bool do_rerank;
    bool base_in_ram;
    const float* base_all;           // base_in_ram payload (mmap'd or nullptr)
    const std::string* base_data;         // path for read_fbin_vector fallback
    const GroundTruth* gt;
    bool have_gt_dists;
    /// Metric for rerank distance. Matches the index's quantizer metric so the
    /// rerank reordering is consistent with the engine's search-time metric.
    /// For L2-normalized data both metrics give the same final top-k; the
    /// rerank cost (dot vs l2sq) is what differs and affects the measured QPS.
    sextant::MetricKind metric = sextant::MetricKind::L2Sq;
    /// Panorama progressive rerank (0 = disabled). When >0, `lvl_d` is the
    /// number of dimensions per level. The rerank loop computes distances
    /// incrementally over level-chunks and prunes candidates whose lower
    /// bound exceeds the k-th best after each level.
    uint32_t panorama_lvl_d = 0;
    /// Per-vector per-level cumulative norms²: `base_cum_norms[row_id * (n_levels+1) + l]`
    /// = Σ_{j >= l*lvl_d} base[row_id][j]². nullptr when panorama_lvl_d == 0.
    /// Allocated/computed by the benchmark before the timed loop.
    const float* base_cum_norms = nullptr;
    uint32_t panorama_n_levels = 0;
};

/// Rerank + recall + proximity post-processing shared by the single-index
/// and IVF benchmark paths. Takes the raw search `results` (candidates from
/// either Searcher or IVFSearcher) and the measured `latency_us`, returns
/// the per-query metrics. The `queries` pointer is used for the rerank
/// distance computation.
QueryMetrics process_results(const RerankCtx& ctx,
                             const std::vector<float>& queries,
                             uint32_t qi,
                             std::vector<sextant::Candidate>&& results,
                             double latency_us) {
    QueryMetrics m;
    m.latency_us = latency_us;
    const float* q = &queries[static_cast<size_t>(qi) * ctx.dim];

    std::vector<std::pair<float, sextant::RowId>> topk_scored;
    topk_scored.reserve(ctx.k);
    std::vector<float> base_vec(ctx.dim);  // only used if !base_in_ram
    if (ctx.do_rerank && !results.empty()) {
        std::vector<std::pair<float, sextant::RowId>> scored;
        scored.reserve(results.size());
        if (ctx.panorama_lvl_d > 0 && ctx.base_in_ram && ctx.base_cum_norms) {
            // --- Panorama progressive rerank ---
            // Compute distances incrementally over level-chunks. After each
            // level, prune candidates whose lower bound exceeds the k-th best.
            // IP: Cauchy-Schwarz bound on remaining dims.
            // L2sq: partial sum is a trivial lower bound (remaining ≥ 0).
            const uint32_t lvl_d = ctx.panorama_lvl_d;
            const uint32_t n_levels = ctx.panorama_n_levels;
            const float* bcn = ctx.base_cum_norms;
            const bool is_ip = (ctx.metric == sextant::MetricKind::InnerProduct);

            // Per-query cumulative norms²: q_cum_norm[l] = Σ_{j>=l*lvl_d} q[j]²
            std::vector<float> q_cum_norm(n_levels + 1);
            q_cum_norm[n_levels] = 0.0f;
            for (int l = int(n_levels) - 1; l >= 0; l--) {
                const uint32_t start = l * lvl_d;
                const uint32_t end = std::min(start + lvl_d, ctx.dim);
                float ns = 0.0f;
                for (uint32_t j = start; j < end; j++)
                    ns += q[j] * q[j];
                q_cum_norm[l] = q_cum_norm[l + 1] + ns;
            }

            const uint32_t ncand = results.size();
            std::vector<float> partial(ncand, 0.0f);  // partial dist/sim
            std::vector<bool> active(ncand, true);

            // First pass: compute level 0 for all candidates, build initial
            // heap of top-k to establish the pruning threshold.
            // We maintain a running k-th best as a simple approach: after
            // each level, sort actives and use the k-th as threshold.
            for (uint32_t l = 0; l < n_levels; l++) {
                const uint32_t start = l * lvl_d;
                const uint32_t end = std::min(start + lvl_d, ctx.dim);
                const uint32_t width = end - start;

                for (uint32_t c = 0; c < ncand; c++) {
                    if (!active[c]) continue;
                    const auto rid = results[c].row_id;
                    if (rid < 0 || static_cast<uint64_t>(rid) >= ctx.n_base) {
                        active[c] = false;
                        continue;
                    }
                    const float* bv =
                        ctx.base_all + static_cast<size_t>(rid) * ctx.dim;
                    if (is_ip) {
                        partial[c] += sextant::simd::dot_f32(q + start, bv + start, width);
                    } else {
                        for (uint32_t j = start; j < end; j++) {
                            const float diff = q[j] - bv[j];
                            partial[c] += diff * diff;
                        }
                    }
                }

                if (l == n_levels - 1) break;  // last level: no pruning needed

                // Compute pruning threshold = k-th best among active candidates.
                std::vector<float> active_partials;
                active_partials.reserve(ncand);
                for (uint32_t c = 0; c < ncand; c++) {
                    if (active[c]) active_partials.push_back(partial[c]);
                }
                if (active_partials.size() <= ctx.k) continue;  // can't prune yet
                std::nth_element(active_partials.begin(),
                                 active_partials.begin() + ctx.k,
                                 active_partials.end());
                const float thresh = active_partials[ctx.k];

                // Prune: for IP, sim_upper = partial + sqrt(q_rest * y_rest).
                //   prune if sim_upper < thresh_sim (thresh is the k-th best sim).
                // For L2sq: lower bound = partial (remaining ≥ 0).
                //   prune if partial > thresh.
                for (uint32_t c = 0; c < ncand; c++) {
                    if (!active[c]) continue;
                    bool prune = false;
                    if (is_ip) {
                        const auto rid = results[c].row_id;
                        const float y_rest_sq =
                            bcn[static_cast<size_t>(rid) * (n_levels + 1) + l + 1];
                        const float cs_bound =
                            std::sqrt(q_cum_norm[l + 1] * y_rest_sq);
                        // partial[c] is similarity so far. thresh is similarity
                        // of k-th best (higher = better). prune if can't reach thresh.
                        if (partial[c] + cs_bound < thresh) prune = true;
                    } else {
                        if (partial[c] > thresh) prune = true;
                    }
                    if (prune) active[c] = false;
                }
            }

            // Collect surviving candidates, compute final distances.
            // For survivors that were processed through all levels, partial IS
            // the final distance. For pruned candidates, we skip them.
            // However, for IP we need to negate (dist_f32 returns -dot).
            for (uint32_t c = 0; c < ncand; c++) {
                if (!active[c]) continue;
                const auto rid = results[c].row_id;
                if (rid < 0 || static_cast<uint64_t>(rid) >= ctx.n_base) continue;
                const float d = is_ip ? -partial[c] : partial[c];
                scored.emplace_back(d, rid);
            }
            // If pruning eliminated too many (shouldn't happen at safe settings),
            // fall back: we should have at least k survivors for a valid result.
            // If not, the threshold was too aggressive — but the nth_element
            // approach guarantees ≥k survivors by construction.
        } else {
            // --- Standard full-rerank ---
            for (const auto& c : results) {
                if (c.row_id < 0 ||
                    static_cast<uint64_t>(c.row_id) >= ctx.n_base) {
                    continue;
                }
                float d;
                if (ctx.base_in_ram) {
                    const float* bv =
                        ctx.base_all + static_cast<size_t>(c.row_id) * ctx.dim;
                    d = rerank_dist(ctx.metric, q, bv, ctx.dim);
                } else {
                    read_fbin_vector(*ctx.base_data, ctx.dim,
                                     static_cast<uint64_t>(c.row_id),
                                     base_vec);
                    d = rerank_dist(ctx.metric, q, base_vec.data(), ctx.dim);
                }
                scored.emplace_back(d, c.row_id);
            }
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        const uint32_t topk_n = std::min<uint32_t>(ctx.k, scored.size());
        for (uint32_t i = 0; i < topk_n; i++) {
            topk_scored.push_back(scored[i]);
        }
    } else {
        const uint32_t topk_n = std::min<uint32_t>(ctx.k, results.size());
        for (uint32_t i = 0; i < topk_n; i++) {
            const auto rid = results[i].row_id;
            float d = std::numeric_limits<float>::infinity();
            if (rid >= 0 &&
                static_cast<uint64_t>(rid) < ctx.n_base) {
                if (ctx.base_in_ram) {
                    const float* bv =
                        ctx.base_all + static_cast<size_t>(rid) * ctx.dim;
                    d = rerank_dist(ctx.metric, q, bv, ctx.dim);
                } else {
                    read_fbin_vector(*ctx.base_data, ctx.dim,
                                     static_cast<uint64_t>(rid),
                                     base_vec);
                    d = rerank_dist(ctx.metric, q, base_vec.data(), ctx.dim);
                }
            }
            topk_scored.emplace_back(d, rid);
        }
    }

    const GroundTruth& gt = *ctx.gt;
    if (qi < gt.n) {
        const uint32_t* gt_row =
            &gt.ids[static_cast<size_t>(qi) * gt.k];
        const uint32_t gt_k = std::min(gt.k, ctx.k);
        uint32_t hits = 0;
        for (uint32_t i = 0; i < gt_k; i++) {
            const uint32_t g = gt_row[i];
            for (const auto& [d, r] : topk_scored) {
                (void)d;
                if (r >= 0 && static_cast<uint32_t>(r) == g) {
                    ++hits;
                    break;
                }
            }
        }
        m.recall_sum = static_cast<double>(hits) /
                        static_cast<double>(gt_k);
        m.recall_hits = hits;
        m.recall_total = gt_k;

        if (ctx.have_gt_dists && ctx.base_in_ram) {
            const uint32_t gt_kth_id =
                gt.ids[static_cast<size_t>(qi) * gt.k + (gt_k - 1)];
            float d_target;
            if (gt_kth_id < ctx.n_base) {
                const float* bv_kth =
                    ctx.base_all + static_cast<size_t>(gt_kth_id) * ctx.dim;
                d_target = rerank_dist(ctx.metric, q, bv_kth, ctx.dim);
            } else {
                d_target = gt.dists[static_cast<size_t>(qi) * gt.k +
                                     (gt_k - 1)];
            }
            // ratio = d_target / d_result. Capped to keep the mean sane when
            // the result is far closer than the target.
            constexpr double kRatioCap = 10.0;
            constexpr double kEps = 1e-6f;
            m.prox_ratios.reserve(topk_scored.size());
            for (const auto& [d, r] : topk_scored) {
                if (r < 0) continue;
                double ratio;
                if (d_target <= kEps && d <= kEps) {
                    ratio = 1.0;
                } else if (d <= kEps) {
                    ratio = kRatioCap;
                } else if (d_target <= kEps) {
                    ratio = 0.0;
                } else {
                    ratio = static_cast<double>(d_target) /
                            static_cast<double>(d);
                }
                m.prox_ratios.push_back(ratio);
                if (ratio >= 1.0) ++m.prox_in;
                ++m.prox_total;
            }
        }
    }
    return m;
}

}  // namespace

/// IVF benchmark path. Opened when `<index>.shards/` exists. Runs queries
/// synchronously through IVFSearcher (no search_one_async yet — parallelism
/// across shards is Milestone 3). Shares rerank/recall/proximity logic with
/// the single-index path via `process_results`. Returns 0 on success, 1 on
/// error (mirroring main's exit codes).
 int run_ivf_benchmark(const std::string& index,
                      const std::string& query_path,
                      const std::string& base_data,
                      const std::string& gt_path,
                      uint32_t k, uint32_t L, uint32_t rerank,
                      uint32_t io_limit, uint32_t limit,
                      uint32_t n_threads_hint,
                      uint64_t cache_size_req,
                      uint32_t n_probe_req,
                       uint32_t early_exit_req,
                       float multiprobe_ratio_req);

/// IVF-list-scan benchmark path (Option A default). Opened when
/// `<index>.shards/codebook4.bin` exists. Runs queries through
/// IVFScanSearcher, which returns top-W candidates by 4-bit PQ distance; the
/// benchmark then applies exact FP32 rerank (the DB layer's job in
/// production, done inline here for measurement) and computes recall vs GT.
/// Returns 0 on success, 1 on error.
int run_ivf_scan_benchmark(const std::string& index,
                           const std::string& query_path,
                           const std::string& base_data,
                           const std::string& gt_path,
                           uint32_t k, uint32_t L, uint32_t rerank,
                           uint32_t io_limit, uint32_t limit,
                           uint32_t n_threads_hint,
                           uint64_t cache_size_req,
                           uint32_t n_probe_req,
                           uint32_t early_exit_req,
                           float multiprobe_ratio_req,
                            uint32_t fastscan_w_req,
                            uint32_t panorama_lvl_d,
                            uint32_t sub_shard_np_req,
                            float adaptive_gap_req,
                            uint32_t scan_budget_req);

int main(int argc, char* argv[]) {
    sextant::install_crash_handler();
    sextant::init_logging();

    cmdline::parser p;
    p.add<std::string>("index", 0, "Index name/path prefix", true);
    p.add<std::string>("queries", 0, "Query .fbin file", true);
    p.add<std::string>("base-data", 0, "Original base .fbin for rerank", true);
    p.add<std::string>("ground-truth", 0, "Ground-truth .gt file", true);
    p.add<uint32_t>("topk", 0, "Number of nearest neighbors to return per query (k in ANN literature). Default 100 (VIBE / modern-retrieval convention).", false, 100);
    p.add<uint32_t>("search-beam-width", 0,
        "Search-time beam width (L in Vamana literature). Higher = more accurate, slower.",
        false, 200);
    p.add<uint32_t>("rerank", 0, "Rerank factor (0/1 = no rerank)", false, 10);
    p.add<uint32_t>("fastscan-w", 0,
        "FastScan candidate window W (per-shard top-W shortlist before "
        "merge). Default 300. Increase to bypass PQ-distance merge issues "
        "(e.g. testing per-shard codebooks where cross-shard distances "
        "aren't comparable). At W >> shard_n the merge becomes irrelevant "
        "and FP32 rerank dominates.",
        false, 0);
    p.add<uint32_t>("panorama-levels", 0,
        "Panorama progressive rerank: dimensions per level (0=off). "
        "When >0, rerank computes distances incrementally over level-chunks "
        "and prunes candidates via Cauchy-Schwarz lower bounds. ~2x rerank "
        "speedup on IP at 0 recall loss. Recommended: 8 for IP, 0 for L2sq.",
        false, 0);
    p.add<uint32_t>("sub-shard-n-probe", 0,
        "Override sub-shard probe count at search time. 0 = use the index's "
        "built-in value from the manifest. >0 = scan only the N nearest "
        "sub-shards per coarse shard.",
        false, 0);
    p.add<std::string>("adaptive-probe-gap", 0,
        "Adaptive probe early-exit: stop scanning when the next shard's centroid "
        "is significantly farther than the current (ratio > gap). "
        "'auto' (default) = use the index's baked value from the manifest. "
        "'off' = disabled (scan all n_probe). "
        "A float >1.0 (e.g. 1.3) = explicit gap value.",
        false, "auto");
    p.add<uint32_t>("scan-code-budget", 0,
        "Maximum total codes (vectors) to scan across all probed shards. "
        "0 = unlimited. Shards scanned in centroid-distance order until budget "
        "exhausted. Adapts to skew: small shards cheap, large ones may be skipped.",
        false, 0);
    p.add<uint32_t>("io-limit", 0, "Search I/O budget (0 = unlimited)", false, 0);
    p.add<uint32_t>("limit", 0, "Max queries to run (0 = all)", false, 0);
    p.add<uint32_t>("threads", 0, "Search threads (0 = hardware_concurrency)",
                    false, 0);
    p.add<uint64_t>("cache-size", 0,
                     "Search LRU cache size in bytes (0 = auto)", false, 0);
    p.add("no-cache-rebalance", 0,
          "Disable adaptive graph/code cache rebalancing (default: enabled in paged mode)");
    p.add<std::string>("log-level", 0,
                       "Log level: debug, info, warn, error",
                       false, "info");
    p.add<uint32_t>("n-probe", 0,
                    "IVF: number of shards to probe per query "
                    "(0 = use the index's built-in n_probe_default)",
                    false, 0);
    p.add<uint32_t>("early-exit-patience", 0,
                    "Search-time early-exit patience (post-convergence stall "
                    "count before terminating). 0 = disabled (full L_search). "
                    "Default: defer to the index's baked-in value. Higher = "
                    "deeper shard search (more recall, less QPS).",
                    false, 0xFFFFFFFFu);
    p.add<float>("multiprobe-ratio", 0,
                 "IVF multi-probe ratio: extend the probe set to all centroids "
                 "within ratio × d[n_probe-1]. 1.0 = strict n_probe (off). "
                 ">1.0 recovers boundary-shard recall at variable per-query cost. "
                 "Default 1.05 — measured +11.8pp recall at np=1 for +0.51 avg "
                 "shards/query (c4a K=21 arxiv-nomic 1.34M). Set to 1.0 to disable. "
                 "See docs/ivf_routing_analysis.md.",
                 false, 1.05f);
    p.parse_check(argc, argv);

    // Set log level (must come after init_logging() above and before any
    // spdlog calls). debug exposes the cache-rebalance controller's per-call
    // sample log, BFS reorder details, and other diagnostic output.
    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") sextant::set_log_level(sextant::LogLevel::Debug);
        else if (lvl == "warn") sextant::set_log_level(sextant::LogLevel::Warn);
        else if (lvl == "error") sextant::set_log_level(sextant::LogLevel::Error);
    }

    const std::string index = p.get<std::string>("index");
    const std::string query_path = p.get<std::string>("queries");
    const std::string base_data = p.get<std::string>("base-data");
    const std::string gt_path = p.get<std::string>("ground-truth");
    const uint32_t k = p.get<uint32_t>("topk");
    const uint32_t L = p.get<uint32_t>("search-beam-width");
    const uint32_t rerank = p.get<uint32_t>("rerank");
    const uint32_t io_limit = p.get<uint32_t>("io-limit");
    const uint32_t limit = p.get<uint32_t>("limit");
    const uint32_t threads_req = p.get<uint32_t>("threads");
    const uint64_t cache_size_req = p.get<uint64_t>("cache-size");

    // Compute thread count early so we can warn about cache thrashing.
    const uint32_t n_threads_hint = threads_req > 0
        ? threads_req
        : static_cast<uint32_t>(std::thread::hardware_concurrency());

    // Warn if an explicit cache size is too small for the thread count.
    // Each concurrent search needs ~50-100 blocks in its working set;
    // a shared LRU that's too small causes cross-thread eviction thrashing
    // (4 threads on 32MB can be slower than 1 thread).
    //
    // When --cache-size is small enough to force misses, the adaptive
    // rebalance controller will shift capacity between graph/code caches;
    // use --no-cache-rebalance to disable for A/B comparison.
    if (cache_size_req > 0 && n_threads_hint > 1) {
        constexpr uint64_t kMinCachePerThread = 32ull * 1024 * 1024;
        const uint64_t recommended = kMinCachePerThread * n_threads_hint;
        if (cache_size_req < recommended) {
            spdlog::warn("benchmark: --cache-size {:.0f}MB with {} threads may "
                         "cause LRU thrashing (recommended ≥ {:.0f}MB = {}MB/thread)",
                         cache_size_req / 1e6, n_threads_hint,
                         recommended / 1e6, kMinCachePerThread / 1e6);
        }
    }

    const uint32_t n_probe_req = p.get<uint32_t>("n-probe");
    const uint32_t early_exit_req = p.get<uint32_t>("early-exit-patience");
    const float multiprobe_ratio_req = p.get<float>("multiprobe-ratio");
    const uint32_t fastscan_w_req = p.get<uint32_t>("fastscan-w");
    const uint32_t panorama_levels_req = p.get<uint32_t>("panorama-levels");
    const uint32_t sub_shard_np_req = p.get<uint32_t>("sub-shard-n-probe");
    // Parse adaptive-probe-gap: "auto" → 0 (use manifest), "off" → -1,
    // otherwise parse as float.
    const std::string adaptive_gap_str = p.get<std::string>("adaptive-probe-gap");
    float adaptive_gap_req = 0.0f;  // auto
    if (adaptive_gap_str == "off" || adaptive_gap_str == "0") {
        adaptive_gap_req = -1.0f;   // disabled
    } else if (adaptive_gap_str != "auto") {
        adaptive_gap_req = std::stof(adaptive_gap_str);
    }
    const uint32_t scan_budget_req = p.get<uint32_t>("scan-code-budget");

    // IVF dispatch: if `<index>.shards/` is a directory, this is an IVF
    // index. Two IVF flavors share the `.shards/` layout:
    //   - IVF-list-scan (default build path): `<prefix>.shards/codebook4.bin`
    //     present → open via IVFScanIndex::read, search via IVFScanSearcher.
    //   - graph-inside-shard (--ivf): no codebook4.bin → open via
    //     IVFIndex::read, search via IVFSearcher.
    // The single-index (merged-graph) path below runs when `.shards/` is
    // absent.
    if (std::filesystem::is_directory(index + ".shards")) {
        const bool is_scan = std::filesystem::exists(
                                 index + ".shards/codebook4.bin") ||
                             std::filesystem::exists(
                                 index + ".shards/codebook8.bin");
        if (is_scan) {
            return run_ivf_scan_benchmark(
                index, query_path, base_data, gt_path,
                k, L, rerank, io_limit, limit,
                n_threads_hint, cache_size_req,
                n_probe_req, early_exit_req, multiprobe_ratio_req,
                fastscan_w_req, panorama_levels_req, sub_shard_np_req,
                adaptive_gap_req, scan_budget_req);
        }
        return run_ivf_benchmark(index, query_path, base_data, gt_path,
                                 k, L, rerank, io_limit, limit,
                                 n_threads_hint, cache_size_req,
                                 n_probe_req, early_exit_req, multiprobe_ratio_req);
    }

    try {
        std::unique_ptr<sextant::Index> idx =
            sextant::Index::read(index, cache_size_req);
        sextant::Searcher searcher(*idx, n_threads_hint);
        if (p.exist("no-cache-rebalance")) {
            searcher.set_cache_rebalance_enabled(false);
        }
        const uint32_t dim = idx->dim;
        const uint64_t n_base = idx->count;

        FbinHeader qh;
        if (!read_fbin_header(query_path, qh) || qh.dim != dim) {
            std::cerr << "benchmark: invalid query file '" << query_path
                      << "' (dim=" << qh.dim << ", expected " << dim << ")\n";
            return 1;
        }
        if (qh.n == 0) {
            std::cerr << "benchmark: query file is empty\n";
            return 1;
        }

        FbinHeader bh{};
        if (!read_fbin_header(base_data, bh) || bh.dim != dim) {
            std::cerr << "benchmark: invalid base-data '" << base_data
                      << "' (dim=" << bh.dim << ", expected " << dim << ")\n";
            return 1;
        }

        GroundTruth gt = read_ground_truth(gt_path);
        if (gt.k < k) {
            std::cerr << "benchmark: ground-truth k=" << gt.k
                      << " < requested k=" << k << "\n";
            return 1;
        }

        const uint32_t n_queries = std::min<uint32_t>(
            qh.n, limit ? limit : qh.n);
        if (gt.n < n_queries) {
            spdlog::warn("benchmark: ground-truth has {} queries but running {}",
                         gt.n, n_queries);
        }

        const bool do_rerank = rerank > 1;
        const uint32_t fetch_k = do_rerank
                                     ? std::min<uint32_t>(k * rerank, n_base)
                                     : k;

        const uint32_t n_threads = std::max(1u, std::min(n_threads_hint,
                                                          n_queries));

        std::cout << "[benchmark] queries: " << n_queries << ", k: " << k
                  << ", L: " << L << ", rerank: " << rerank
                  << ", threads: " << n_threads << "\n";

        // Read all query vectors into RAM (10k × 128 × 4B = ~5MB, fine).
        std::vector<float> queries(static_cast<size_t>(n_queries) * dim);
        {
            std::ifstream qf(query_path, std::ios::binary);
            qf.seekg(8);
            qf.read(reinterpret_cast<char*>(queries.data()),
                    static_cast<std::streamsize>(queries.size() *
                                                 sizeof(float)));
            if (!qf) {
                std::cerr << "benchmark: short read on query file\n";
                return 1;
            }
        }

        // mmap the base file for zero-copy rerank. No size cap — the OS page
        // cache manages residency. The 8-byte fbin header is skipped by
        // offsetting the pointer.
        const float* base_all = nullptr;
        bool base_in_ram = false;
        void* base_mmap = nullptr;
        uint64_t base_mmap_size = 0;
        {
            int fd = ::open(base_data.c_str(), O_RDONLY);
            if (fd >= 0) {
                struct stat st;
                if (::fstat(fd, &st) == 0 && st.st_size > 8) {
                    base_mmap_size = st.st_size;
                    base_mmap = ::mmap(nullptr, base_mmap_size, PROT_READ,
                                       MAP_SHARED, fd, 0);
                    if (base_mmap != MAP_FAILED) {
                        ::madvise(base_mmap, base_mmap_size, MADV_RANDOM);
                        // Skip the 8-byte fbin header (n + dim).
                        base_all = reinterpret_cast<const float*>(
                            static_cast<char*>(base_mmap) + 8);
                        base_in_ram = true;
                        spdlog::info("benchmark: mmap'd {} base vectors ({}MB)",
                                     bh.n, base_mmap_size / (1024 * 1024));
                    } else { base_mmap = nullptr; }
                }
                ::close(fd);
            }
        }

        // Aggregates (collected on the main thread from per-query metrics).
        std::vector<double> latencies_us;
        latencies_us.reserve(n_queries);

        const bool have_gt_dists = !gt.dists.empty();

        // Shared rerank/recall/proximity context. Both the single-index and
        // IVF paths route results through process_results via this struct.
        const sextant::MetricKind idx_metric =
            idx->quantizer ? idx->quantizer->metric() : sextant::MetricKind::L2Sq;
        if (int c = check_gt_metric(gt, idx_metric, gt_path)) return c;
        const RerankCtx rctx{
            dim, k, bh.n, do_rerank, base_in_ram,
            base_in_ram ? base_all : nullptr,
            &base_data, &gt, have_gt_dists, idx_metric
        };

        // Push all queries to the Searcher pool up front (work-stealing).
        // Workers pull the next query as soon as they finish. The main
        // thread collects futures in order and runs process_results (rerank
        // + recall + proximity) inline. This is serial post-processing but
        // uses simd::dist_f32 for the rerank.
        //
        // NOTE: serial post-processing is the 8t bottleneck. A future
        // workstream could parallelize it by extending Searcher to accept
        // a per-query post-processing callback that runs on the worker.
        // For now the SIMD rerank recovers most of the gap.
        sextant::SearchConfig scfg;
         scfg.k = fetch_k;
         scfg.L_search = L;
         scfg.io_limit = io_limit;
         scfg.early_exit_patience = early_exit_req;

         const auto t_start = Clock::now();
         double recall_sum = 0.0;
         uint64_t recall_hits = 0, recall_total = 0;
         uint64_t prox_in = 0, prox_total = 0;
         std::vector<double> prox_ratios;
         std::vector<std::future<std::vector<sextant::Candidate>>> futs;

         futs.reserve(n_queries);

         for (uint32_t qi = 0; qi < n_queries; qi++) {

             const float* q = &queries[static_cast<size_t>(qi) * dim];

             futs.push_back(searcher.search_one_async(q, scfg.k, scfg));

         }

         auto prev_q_end = Clock::now();

        for (uint32_t qi = 0; qi < n_queries; qi++) {

            auto results = futs[qi].get();

            // NOTE: process_results (rerank + recall) runs on the main thread.
            // The timestamp must bracket BOTH the search (future get) AND the
            // rerank to capture true per-query latency. Previously q_end was
            // captured before process_results, hiding the rerank cost from
            // p50/p99 (it only showed up in the aggregate QPS).
            QueryMetrics m = process_results(rctx, queries, qi,
                                             std::move(results), 0.0);
            const auto q_end = Clock::now();
            const double per_q_us = US(q_end - prev_q_end).count();
            prev_q_end = q_end;
            m.latency_us = per_q_us;
            latencies_us.push_back(m.latency_us);
             recall_sum += m.recall_sum;
             recall_hits += m.recall_hits;
             recall_total += m.recall_total;
             if (!m.prox_ratios.empty()) {
                 prox_ratios.insert(prox_ratios.end(),
                                    m.prox_ratios.begin(),
                                    m.prox_ratios.end());
             }
             prox_in += m.prox_in;
             prox_total += m.prox_total;
         }
        const auto t_end = Clock::now();
        const double total_sec =
            std::chrono::duration<double>(t_end - t_start).count();

        const double mean_recall =
            (n_queries > 0) ? recall_sum / static_cast<double>(n_queries)
                            : 0.0;
        const double micro_recall =
            (recall_total > 0)
                ? static_cast<double>(recall_hits) /
                      static_cast<double>(recall_total)
                : 0.0;

        std::sort(latencies_us.begin(), latencies_us.end());
        const double p50_us = percentile(latencies_us, 50.0);
        const double p99_us = percentile(latencies_us, 99.0);
        const double qps =
            (total_sec > 0.0)
                ? static_cast<double>(n_queries) / total_sec
                : 0.0;

        std::cout << "[benchmark] recall@" << k << ": "
                  << std::fixed << std::setprecision(4) << mean_recall << "\n";
        std::cout << "[benchmark] recall@" << k << " (micro-avg): "
                  << std::fixed << std::setprecision(4) << micro_recall << "\n";

        // Proximity: how close results get to the k-th true NN.
        // ratio = d_target / d_result  (>=1.0 = at or inside target radius).
        // Reports: in-band fraction (results at/inside target), mean ratio over
        // all results, and the ratio distribution of OUT-of-band results so you
        // can see how far past the boundary misses land (e.g. p50=0.8 means the
        // median miss is ~25% outside the target distance).
        if (have_gt_dists && prox_total > 0) {
            const double in_band =
                static_cast<double>(prox_in) /
                static_cast<double>(prox_total);
            double prox_sum = 0.0;
            std::vector<double> miss_ratios;
            miss_ratios.reserve(prox_total - prox_in);
            for (double r : prox_ratios) {
                prox_sum += r;
                if (r < 1.0) miss_ratios.push_back(r);
            }
            const double mean_ratio = prox_sum /
                static_cast<double>(prox_total);
            std::cout << "[benchmark] proximity: in-band="
                      << std::fixed << std::setprecision(4) << in_band
                      << " (" << prox_in << "/" << prox_total
                      << " results at/inside d" << k << ")"
                      << ", mean_ratio=" << std::setprecision(4)
                      << mean_ratio << "\n";
            std::cout << "[benchmark] proximity: miss_ratio";
            if (miss_ratios.empty()) {
                std::cout << "=n/a (no out-of-band results)\n";
            } else {
                std::sort(miss_ratios.begin(), miss_ratios.end());
                const double mp25 = percentile(miss_ratios, 25.0);
                const double mp50 = percentile(miss_ratios, 50.0);
                const double mp75 = percentile(miss_ratios, 75.0);
                const double mp90 = percentile(miss_ratios, 90.0);
                std::cout << " p25=" << std::fixed
                          << std::setprecision(4) << mp25
                          << " p50=" << mp50
                          << " p75=" << mp75
                          << " p90=" << mp90
                          << " (over " << miss_ratios.size()
                          << " out-of-band; 1.0=target edge)\n";
            }
        }
        std::cout << "[benchmark] latency p50: " << std::fixed
                  << std::setprecision(3) << p50_us / 1000.0 << "ms, p99: "
                  << p99_us / 1000.0 << "ms\n";
        std::cout << "[benchmark] total search time: " << std::fixed
                  << std::setprecision(3) << total_sec << "s\n";
         std::cout << "[benchmark] QPS: " << std::fixed
                   << std::setprecision(1) << qps << "\n";
         if (idx->is_paged()) {
             std::cout << "[benchmark] cache: graph_reads="
                       << searcher.cache_graph_reads()
                       << " code_reads=" << searcher.cache_code_reads() << "\n";
             const auto as = searcher.cache_admission_stats();
             const uint64_t total_hits = as.hits_window + as.hits_probation + as.hits_protected;
             const uint64_t total_accesses = total_hits + as.misses;
             const double hit_rate = total_accesses > 0
                 ? 100.0 * static_cast<double>(total_hits) / static_cast<double>(total_accesses)
                 : 0.0;
             const uint64_t total_evicts = as.evictions_admitted + as.evictions_rejected;
             const double admit_rate = total_evicts > 0
                 ? 100.0 * static_cast<double>(as.evictions_admitted) / static_cast<double>(total_evicts)
                 : 0.0;
             std::cout << "[benchmark] w-tinylfu: hit_rate=" << std::fixed
                       << std::setprecision(1) << hit_rate << "%"
                       << " (w=" << as.hits_window
                       << " p=" << as.hits_probation
                       << " pr=" << as.hits_protected
                       << " miss=" << as.misses << ")"
                       << " admit=" << std::setprecision(1) << admit_rate << "%"
                        << " (" << as.evictions_admitted << "/" << total_evicts << ")\n";
             const uint64_t tl_h = searcher.tl_hits();
             const uint64_t tl_m = searcher.tl_misses();
             const uint64_t tl_total = tl_h + tl_m;
             const double tl_rate = tl_total > 0
                 ? 100.0 * static_cast<double>(tl_h) / static_cast<double>(tl_total)
                 : 0.0;
             std::cout << "[benchmark] tl-l1: hit_rate=" << std::fixed
                       << std::setprecision(1) << tl_rate << "%"
                       << " (hits=" << tl_h << " misses=" << tl_m << ")\n";
          }
    } catch (const Error& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// IVF benchmark path.
//
// Mirrors the single-index flow but uses IVFIndex + IVFSearcher. Queries run
// SYNCHRONOUSLY (IVFSearcher has no search_one_async yet — per-shard
// parallelism is Milestone 3). Rerank/recall/proximity are shared with the
// single-index path via process_results.
//
// `index` is the prefix; the shards live in `<index>.shards/`. cache_size_req
// is passed through to each shard's Index::read. n_probe_req (0 = index
// default) is forwarded as scfg.n_probe.
// ---------------------------------------------------------------------------
 int run_ivf_benchmark(const std::string& index,
                      const std::string& query_path,
                      const std::string& base_data,
                      const std::string& gt_path,
                      uint32_t k, uint32_t L, uint32_t rerank,
                      uint32_t io_limit, uint32_t limit,
                      uint32_t n_threads_hint,
                      uint64_t cache_size_req,
                      uint32_t n_probe_req,
                      uint32_t early_exit_req,
                      float multiprobe_ratio_req) {
    try {
        std::unique_ptr<sextant::IVFIndex> ivf_idx =
            sextant::IVFIndex::read(index + ".shards", cache_size_req);
        sextant::IVFSearcher ivf_searcher(*ivf_idx, n_threads_hint);

        const uint32_t dim = ivf_idx->dim;
        const uint32_t K = ivf_idx->K;

        FbinHeader qh;
        if (!read_fbin_header(query_path, qh) || qh.dim != dim) {
            std::cerr << "benchmark: invalid query file '" << query_path
                      << "' (dim=" << qh.dim << ", expected " << dim << ")\n";
            return 1;
        }
        if (qh.n == 0) {
            std::cerr << "benchmark: query file is empty\n";
            return 1;
        }

        FbinHeader bh{};
        if (!read_fbin_header(base_data, bh) || bh.dim != dim) {
            std::cerr << "benchmark: invalid base-data '" << base_data
                      << "' (dim=" << bh.dim << ", expected " << dim << ")\n";
            return 1;
        }

        GroundTruth gt = read_ground_truth(gt_path);
        if (gt.k < k) {
            std::cerr << "benchmark: ground-truth k=" << gt.k
                      << " < requested k=" << k << "\n";
            return 1;
        }

        const uint32_t n_queries = std::min<uint32_t>(
            qh.n, limit ? limit : qh.n);
        if (gt.n < n_queries) {
            spdlog::warn("benchmark: ground-truth has {} queries but running {}",
                         gt.n, n_queries);
        }

        const bool do_rerank = rerank > 1;
        const uint32_t fetch_k = do_rerank
                                     ? std::min<uint32_t>(k * rerank, bh.n)
                                     : k;

        const uint32_t effective_n_probe = n_probe_req > 0
            ? n_probe_req
            : ivf_idx->n_probe_default;

        std::cout << "[benchmark] (IVF mode, sequential shard search)\n";
        std::cout << "[benchmark] queries: " << n_queries << ", k: " << k
                  << ", L: " << L << ", rerank: " << rerank
                  << ", K (shards): " << K
                  << ", n_probe: " << effective_n_probe
                  << ", multiprobe_ratio: " << multiprobe_ratio_req
                  << ", threads: " << ivf_searcher.num_threads() << "\n";

        // Read all query vectors into RAM.
        std::vector<float> queries(static_cast<size_t>(n_queries) * dim);
        {
            std::ifstream qf(query_path, std::ios::binary);
            qf.seekg(8);
            qf.read(reinterpret_cast<char*>(queries.data()),
                    static_cast<std::streamsize>(queries.size() *
                                                 sizeof(float)));
            if (!qf) {
                std::cerr << "benchmark: short read on query file\n";
                return 1;
            }
        }

        // mmap the base file for zero-copy rerank.
        const float* base_all = nullptr;
        bool base_in_ram = false;
        void* base_mmap = nullptr;
        uint64_t base_mmap_size = 0;
        {
            int fd = ::open(base_data.c_str(), O_RDONLY);
            if (fd >= 0) {
                struct stat st;
                if (::fstat(fd, &st) == 0 && st.st_size > 8) {
                    base_mmap_size = st.st_size;
                    base_mmap = ::mmap(nullptr, base_mmap_size, PROT_READ,
                                       MAP_SHARED, fd, 0);
                    if (base_mmap != MAP_FAILED) {
                        ::madvise(base_mmap, base_mmap_size, MADV_RANDOM);
                        base_all = reinterpret_cast<const float*>(
                            static_cast<char*>(base_mmap) + 8);
                        base_in_ram = true;
                        spdlog::info("benchmark: mmap'd {} base vectors ({}MB)",
                                     bh.n, base_mmap_size / (1024 * 1024));
                    } else { base_mmap = nullptr; }
                }
                ::close(fd);
            }
        }

        const bool have_gt_dists = !gt.dists.empty();
        // All shards share the same trained quantizer (built together); pull
        // the metric from the first present shard's quantizer.
        sextant::MetricKind ivf_metric = sextant::MetricKind::L2Sq;
        for (const auto& shard : ivf_idx->shards) {
            if (shard && shard->quantizer) {
                ivf_metric = shard->quantizer->metric();
                break;
            }
        }
        if (int c = check_gt_metric(gt, ivf_metric, gt_path)) return c;
        const RerankCtx rctx{
            dim, k, bh.n, do_rerank, base_in_ram,
            base_in_ram ? base_all : nullptr,
            &base_data, &gt, have_gt_dists, ivf_metric
        };

        // IVFSearcher.search is synchronous (no search_one_async). Loop over
        // queries, timing each call. n_probe is forwarded via SearchConfig.
        sextant::SearchConfig scfg;
        scfg.k = fetch_k;
        scfg.L_search = L;
        scfg.io_limit = io_limit;
        scfg.n_probe = n_probe_req;  // 0 → IVFSearcher uses index default.
        scfg.early_exit_patience = early_exit_req;
        scfg.multiprobe_ratio = multiprobe_ratio_req;

        std::vector<double> latencies_us;
        latencies_us.reserve(n_queries);

        double recall_sum = 0.0;
        uint64_t recall_hits = 0, recall_total = 0;
        uint64_t prox_in = 0, prox_total = 0;
        std::vector<double> prox_ratios;

        // Push all queries to the IVF pool up front (work-stealing across
        // queries — mirrors the merged-graph Searcher path). Workers pull the
        // next query as soon as they finish. The main thread collects futures
        // in order and runs process_results (rerank + recall + proximity)
        // inline.
        //
        // Marker for `perf record -p <pid>` attach: printed right before the
        // timed region so profiling tools can trigger on actual program state
        // instead of an unreliable wall-clock delay (see optimization plan,
        // "Profiling discipline").
        std::cout << "[benchmark] starting timed search\n";
        const auto t_start = Clock::now();
        std::vector<std::future<std::vector<sextant::Candidate>>> futs;

        futs.reserve(n_queries);

        for (uint32_t qi = 0; qi < n_queries; qi++) {

            const float* q = &queries[static_cast<size_t>(qi) * dim];

            futs.push_back(ivf_searcher.search_one_async(q, scfg.k, scfg));

        }

        auto prev_q_end = Clock::now();

        for (uint32_t qi = 0; qi < n_queries; qi++) {

            auto results = futs[qi].get();

            const auto q_end = Clock::now();

            const double per_q_us = US(q_end - prev_q_end).count();

            prev_q_end = q_end;
            QueryMetrics m = process_results(rctx, queries, qi,
                                             std::move(results), per_q_us);
            latencies_us.push_back(m.latency_us);
            recall_sum += m.recall_sum;
            recall_hits += m.recall_hits;
            recall_total += m.recall_total;
            if (!m.prox_ratios.empty()) {
                prox_ratios.insert(prox_ratios.end(),
                                   m.prox_ratios.begin(),
                                   m.prox_ratios.end());
            }
            prox_in += m.prox_in;
            prox_total += m.prox_total;
        }
        const auto t_end = Clock::now();
        const double total_sec =
            std::chrono::duration<double>(t_end - t_start).count();

        const double mean_recall =
            (n_queries > 0) ? recall_sum / static_cast<double>(n_queries)
                            : 0.0;
        const double micro_recall =
            (recall_total > 0)
                ? static_cast<double>(recall_hits) /
                      static_cast<double>(recall_total)
                : 0.0;

        std::sort(latencies_us.begin(), latencies_us.end());
        const double p50_us = percentile(latencies_us, 50.0);
        const double p99_us = percentile(latencies_us, 99.0);
        const double qps =
            (total_sec > 0.0)
                ? static_cast<double>(n_queries) / total_sec
                : 0.0;

        std::cout << "[benchmark] recall@" << k << ": "
                  << std::fixed << std::setprecision(4) << mean_recall << "\n";
        std::cout << "[benchmark] recall@" << k << " (micro-avg): "
                  << std::fixed << std::setprecision(4) << micro_recall << "\n";

        // Proximity report (same format as the single-index path).
        if (have_gt_dists && prox_total > 0) {
            const double in_band =
                static_cast<double>(prox_in) /
                static_cast<double>(prox_total);
            double prox_sum = 0.0;
            std::vector<double> miss_ratios;
            miss_ratios.reserve(prox_total - prox_in);
            for (double r : prox_ratios) {
                prox_sum += r;
                if (r < 1.0) miss_ratios.push_back(r);
            }
            const double mean_ratio = prox_sum /
                static_cast<double>(prox_total);
            std::cout << "[benchmark] proximity: in-band="
                      << std::fixed << std::setprecision(4) << in_band
                      << " (" << prox_in << "/" << prox_total
                      << " results at/inside d" << k << ")"
                      << ", mean_ratio=" << std::setprecision(4)
                      << mean_ratio << "\n";
            std::cout << "[benchmark] proximity: miss_ratio";
            if (miss_ratios.empty()) {
                std::cout << "=n/a (no out-of-band results)\n";
            } else {
                std::sort(miss_ratios.begin(), miss_ratios.end());
                const double mp25 = percentile(miss_ratios, 25.0);
                const double mp50 = percentile(miss_ratios, 50.0);
                const double mp75 = percentile(miss_ratios, 75.0);
                const double mp90 = percentile(miss_ratios, 90.0);
                std::cout << " p25=" << std::fixed
                          << std::setprecision(4) << mp25
                          << " p50=" << mp50
                          << " p75=" << mp75
                          << " p90=" << mp90
                          << " (over " << miss_ratios.size()
                          << " out-of-band; 1.0=target edge)\n";
            }
        }
        std::cout << "[benchmark] latency p50: " << std::fixed
                  << std::setprecision(3) << p50_us / 1000.0 << "ms, p99: "
                  << p99_us / 1000.0 << "ms\n";
        std::cout << "[benchmark] total search time: " << std::fixed
                  << std::setprecision(3) << total_sec << "s\n";
        std::cout << "[benchmark] QPS: " << std::fixed
                  << std::setprecision(1) << qps << "\n";
        // NOTE: cache diagnostics (graph_reads/code_reads/w-tinylfu/tl-l1) are
        // omitted here — those are Searcher-scoped; aggregating per-shard
        // caches is Milestone 3+ work.
    } catch (const Error& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// IVF-list-scan benchmark path (Option A default).
//
// Mirrors run_ivf_benchmark but uses IVFScanIndex + IVFScanSearcher. The
// searcher returns the top-W candidates by 4-bit PQ distance (W =
// config.fastscan_W, default 300). For recall measurement we apply exact
// FP32 rerank inline (in production this is the DB layer's job). `rerank`
// and `L` are unused — the scan path has no beam width and rerank is
// mandatory for meaningful recall.
// ---------------------------------------------------------------------------
int run_ivf_scan_benchmark(const std::string& index,
                           const std::string& query_path,
                           const std::string& base_data,
                           const std::string& gt_path,
                           uint32_t k, uint32_t L, uint32_t rerank,
                           uint32_t io_limit, uint32_t limit,
                           uint32_t n_threads_hint,
                           uint64_t cache_size_req,
                           uint32_t n_probe_req,
                           uint32_t early_exit_req,
                           float multiprobe_ratio_req,
                            uint32_t fastscan_w_req,
                            uint32_t panorama_lvl_d,
                            uint32_t sub_shard_np_req,
                            float adaptive_gap_req,
                            uint32_t scan_budget_req) {
    (void)L;             // scan path has no beam width
    (void)io_limit;      // scan reads the whole shard — no node visit cap
    (void)cache_size_req;// scan bypasses the BlockCache (sequential stream)
    (void)early_exit_req;// no graph convergence check
    (void)rerank;        // rerank is mandatory in the scan benchmark path

    try {
        std::unique_ptr<sextant::IVFScanIndex> ivf_idx =
            sextant::IVFScanIndex::read(index + ".shards");
        sextant::IVFScanSearcher ivf_searcher(*ivf_idx, n_threads_hint);

        const uint32_t dim = ivf_idx->dim;
        const uint32_t K = ivf_idx->K;

        FbinHeader qh;
        if (!read_fbin_header(query_path, qh) || qh.dim != dim) {
            std::cerr << "benchmark: invalid query file '" << query_path
                      << "' (dim=" << qh.dim << ", expected " << dim << ")\n";
            return 1;
        }
        if (qh.n == 0) {
            std::cerr << "benchmark: query file is empty\n";
            return 1;
        }

        FbinHeader bh{};
        if (!read_fbin_header(base_data, bh) || bh.dim != dim) {
            std::cerr << "benchmark: invalid base-data '" << base_data
                      << "' (dim=" << bh.dim << ", expected " << dim << ")\n";
            return 1;
        }

        GroundTruth gt = read_ground_truth(gt_path);
        if (gt.k < k) {
            std::cerr << "benchmark: ground-truth k=" << gt.k
                      << " < requested k=" << k << "\n";
            return 1;
        }

        const uint32_t n_queries = std::min<uint32_t>(
            qh.n, limit ? limit : qh.n);
        if (gt.n < n_queries) {
            spdlog::warn("benchmark: ground-truth has {} queries but running {}",
                         gt.n, n_queries);
        }

        const uint32_t effective_n_probe = n_probe_req > 0
            ? n_probe_req
            : ivf_idx->n_probe_default;

        std::cout << "[benchmark] (IVF-list-scan mode)\n";
        std::cout << "[benchmark] queries: " << n_queries << ", k: " << k
                  << ", K (shards): " << K
                  << ", m4: " << ivf_idx->m4
                  << ", n_probe: " << effective_n_probe
                  << ", multiprobe_ratio: " << multiprobe_ratio_req
                  << ", threads: " << ivf_searcher.num_threads() << "\n";

        // Read all query vectors into RAM.
        std::vector<float> queries(static_cast<size_t>(n_queries) * dim);
        {
            std::ifstream qf(query_path, std::ios::binary);
            qf.seekg(8);
            qf.read(reinterpret_cast<char*>(queries.data()),
                    static_cast<std::streamsize>(queries.size() *
                                                 sizeof(float)));
            if (!qf) {
                std::cerr << "benchmark: short read on query file\n";
                return 1;
            }
        }

        // mmap the base file for zero-copy rerank.
        const float* base_all = nullptr;
        bool base_in_ram = false;
        void* base_mmap = nullptr;
        uint64_t base_mmap_size = 0;
        {
            int fd = ::open(base_data.c_str(), O_RDONLY);
            if (fd >= 0) {
                struct stat st;
                if (::fstat(fd, &st) == 0 && st.st_size > 8) {
                    base_mmap_size = st.st_size;
                    base_mmap = ::mmap(nullptr, base_mmap_size, PROT_READ,
                                       MAP_SHARED, fd, 0);
                    if (base_mmap != MAP_FAILED) {
                        ::madvise(base_mmap, base_mmap_size, MADV_RANDOM);
                        base_all = reinterpret_cast<const float*>(
                            static_cast<char*>(base_mmap) + 8);
                        base_in_ram = true;
                        spdlog::info("benchmark: mmap'd {} base vectors ({}MB)",
                                     bh.n, base_mmap_size / (1024 * 1024));
                    } else { base_mmap = nullptr; }
                }
                ::close(fd);
            }
        }

        const bool have_gt_dists = !gt.dists.empty();
        const sextant::MetricKind scan_metric =
            ivf_idx->quantizer ? ivf_idx->quantizer->metric()
                               : sextant::MetricKind::L2Sq;
        if (int c = check_gt_metric(gt, scan_metric, gt_path)) return c;
        // Panorama progressive rerank: precompute per-vector per-level
        // cumulative norms² before the timed loop. Layout:
        //   base_cum_norms[row_id * (n_levels+1) + l] = Σ_{j>=l*lvl_d} base[row_id][j]²
        // Cost: n_base * dim muls (one full pass). Excluded from timed search.
        std::vector<float> base_cum_norms_storage;
        uint32_t panorama_n_levels = 0;
        const float* base_cum_norms_ptr = nullptr;
        if (panorama_lvl_d > 0 && base_in_ram && panorama_lvl_d <= dim) {
            panorama_n_levels = (dim + panorama_lvl_d - 1) / panorama_lvl_d;
            const uint32_t stride = panorama_n_levels + 1;
            base_cum_norms_storage.resize(size_t(bh.n) * stride);
            spdlog::info("benchmark: precomputing Panorama norms ({} levels, "
                         "lvl_d={}, {} vectors)...",
                         panorama_n_levels, panorama_lvl_d, bh.n);
            for (uint64_t i = 0; i < bh.n; i++) {
                const float* v = base_all + i * dim;
                float* row = &base_cum_norms_storage[size_t(i) * stride];
                row[panorama_n_levels] = 0.0f;
                for (int l = int(panorama_n_levels) - 1; l >= 0; l--) {
                    const uint32_t start = l * panorama_lvl_d;
                    const uint32_t end = std::min(start + panorama_lvl_d, dim);
                    float ns = 0.0f;
                    for (uint32_t j = start; j < end; j++)
                        ns += v[j] * v[j];
                    row[l] = row[l + 1] + ns;
                }
            }
            base_cum_norms_ptr = base_cum_norms_storage.data();
            spdlog::info("benchmark: Panorama norms computed ({}MB)",
                         base_cum_norms_storage.size() * 4 / (1024 * 1024));
        }

        const RerankCtx rctx{
            dim, k, bh.n, /*do_rerank=*/true, base_in_ram,
            base_in_ram ? base_all : nullptr,
            &base_data, &gt, have_gt_dists, scan_metric,
            panorama_lvl_d, base_cum_norms_ptr, panorama_n_levels
        };

        // The scan path's W (rerank shortlist) is config.fastscan_W. Default
        // 300 per the spike's recall-0.99 point. fetch_k is informational
        // here — the searcher returns min(W, available).
        sextant::SearchConfig scfg;
        scfg.k = k;
        scfg.n_probe = n_probe_req;  // 0 → index default
        scfg.multiprobe_ratio = multiprobe_ratio_req;
        // fastscan_W: 0 (default) → auto-derived from shard size in the
        // searcher. Override with --fastscan-w for manual tuning.
        scfg.fastscan_W = fastscan_w_req;
        scfg.sub_shard_n_probe_override = sub_shard_np_req;
        scfg.adaptive_probe_gap = adaptive_gap_req;
        scfg.scan_code_budget = scan_budget_req;

        std::vector<double> latencies_us;
        latencies_us.reserve(n_queries);

        double recall_sum = 0.0;
        uint64_t recall_hits = 0, recall_total = 0;
        uint64_t prox_in = 0, prox_total = 0;
        std::vector<double> prox_ratios;

        std::cout << "[benchmark] starting timed search\n";
        const auto t_start = Clock::now();
        std::vector<std::future<std::vector<sextant::Candidate>>> futs;

        futs.reserve(n_queries);

        for (uint32_t qi = 0; qi < n_queries; qi++) {

            const float* q = &queries[static_cast<size_t>(qi) * dim];

            futs.push_back(ivf_searcher.search_one_async(q, scfg.k, scfg));

        }

        auto prev_q_end = Clock::now();

        for (uint32_t qi = 0; qi < n_queries; qi++) {

            auto results = futs[qi].get();

            const auto q_end = Clock::now();

            const double per_q_us = US(q_end - prev_q_end).count();

            prev_q_end = q_end;
            QueryMetrics m = process_results(rctx, queries, qi,
                                             std::move(results), per_q_us);
            latencies_us.push_back(m.latency_us);
            recall_sum += m.recall_sum;
            recall_hits += m.recall_hits;
            recall_total += m.recall_total;
            if (!m.prox_ratios.empty()) {
                prox_ratios.insert(prox_ratios.end(),
                                   m.prox_ratios.begin(),
                                   m.prox_ratios.end());
            }
            prox_in += m.prox_in;
            prox_total += m.prox_total;
        }
        const auto t_end = Clock::now();
        const double total_sec =
            std::chrono::duration<double>(t_end - t_start).count();

        const double mean_recall =
            (n_queries > 0) ? recall_sum / static_cast<double>(n_queries)
                            : 0.0;
        const double micro_recall =
            (recall_total > 0)
                ? static_cast<double>(recall_hits) /
                      static_cast<double>(recall_total)
                : 0.0;

        std::sort(latencies_us.begin(), latencies_us.end());
        const double p50_us = percentile(latencies_us, 50.0);
        const double p99_us = percentile(latencies_us, 99.0);
        const double qps =
            (total_sec > 0.0)
                ? static_cast<double>(n_queries) / total_sec
                : 0.0;

        std::cout << "[benchmark] recall@" << k << ": "
                  << std::fixed << std::setprecision(4) << mean_recall << "\n";
        std::cout << "[benchmark] recall@" << k << " (micro-avg): "
                  << std::fixed << std::setprecision(4) << micro_recall << "\n";

        if (have_gt_dists && prox_total > 0) {
            const double in_band =
                static_cast<double>(prox_in) /
                static_cast<double>(prox_total);
            double prox_sum = 0.0;
            std::vector<double> miss_ratios;
            miss_ratios.reserve(prox_total - prox_in);
            for (double r : prox_ratios) {
                prox_sum += r;
                if (r < 1.0) miss_ratios.push_back(r);
            }
            const double mean_ratio = prox_sum /
                static_cast<double>(prox_total);
            std::cout << "[benchmark] proximity: in-band="
                      << std::fixed << std::setprecision(4) << in_band
                      << " (" << prox_in << "/" << prox_total
                      << " results at/inside d" << k << ")"
                      << ", mean_ratio=" << std::setprecision(4)
                      << mean_ratio << "\n";
            std::cout << "[benchmark] proximity: miss_ratio";
            if (miss_ratios.empty()) {
                std::cout << "=n/a (no out-of-band results)\n";
            } else {
                std::sort(miss_ratios.begin(), miss_ratios.end());
                const double mp25 = percentile(miss_ratios, 25.0);
                const double mp50 = percentile(miss_ratios, 50.0);
                const double mp75 = percentile(miss_ratios, 75.0);
                const double mp90 = percentile(miss_ratios, 90.0);
                std::cout << " p25=" << std::fixed
                          << std::setprecision(4) << mp25
                          << " p50=" << mp50
                          << " p75=" << mp75
                          << " p90=" << mp90
                          << " (over " << miss_ratios.size()
                          << " out-of-band; 1.0=target edge)\n";
            }
        }
        std::cout << "[benchmark] latency p50: " << std::fixed
                  << std::setprecision(3) << p50_us / 1000.0 << "ms, p99: "
                  << p99_us / 1000.0 << "ms\n";
        std::cout << "[benchmark] total search time: " << std::fixed
                  << std::setprecision(3) << total_sec << "s\n";
        std::cout << "[benchmark] QPS: " << std::fixed
                  << std::setprecision(1) << qps << "\n";
    } catch (const Error& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
