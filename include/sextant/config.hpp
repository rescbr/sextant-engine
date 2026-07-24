#pragma once

/// @file config.hpp
/// Build and search configuration structs.

#include <sextant/types.hpp>
#include <cstdint>

namespace sextant {

/// Build configuration. Fields set to 0/default are auto-resolved.
struct BuildConfig {
    uint16_t R = 0;             ///< 0 = auto from N
    uint16_t L = 0;             ///< 0 = auto from R
    float alpha = 0.0f;         ///< Vamana prune threshold: alpha * d(p,pp) <= d(q,pp).
                                ///< 0 = auto (1.2).
    uint16_t pq_m = 0;          ///< 0 = auto (reservoir probe)
    uint8_t pq_bits = 8;        ///< 4, 8, or 0 (auto via reservoir probe)
    /// PQ max distortion for auto (m, bits) selection. Used only when pq_m or
    /// pq_bits is auto. 0 = use the built-in default (0.05).
    ///
    /// Distortion = median |1 - pq_estimated_distance / true_distance| over the
    /// true top-10 neighbors of 200 probe queries (measured on a 20K reservoir
    /// subset). 0.0 = PQ perfectly preserves neighbor distances; 0.15 = the
    /// median true-neighbor distance estimate is off by 15%. Always ≥0, lower is
    /// better; direction-agnostic (doesn't matter if PQ over- or under-estimates).
    /// This is a geometric, dataset-class-robust signal — unlike recall counts,
    /// which are fragile under near-duplicate clustering.
    ///
    /// Selection policy (distortion-bounded min cost):
    ///   1. Filter to configs with distortion ≤ max_distortion.
    ///   2. Among eligible configs, pick minimum cost, where
    ///      cost = m × residency_factor(table_bytes).
    ///        - `m` is the gather count per code_distance/lut_distance call —
    ///          the dominant cost factor in both build and search. Linear in m.
    ///        - residency_factor captures the effective per-gather latency from
    ///          table residency: L2 ≈ 1.0, L3 ≈ 1.5, RAM ≈ 1.9. These are
    ///          empirically calibrated (not raw cycle counts) because hardware
    ///          prefetching and temporal locality dilute the cache cliff. The
    ///          search path uses the PQ LUT (m×K×4 bytes, always L2-resident),
    ///          so residency affects only the build path (code_distance from
    ///          the m×K²×4 cross-distance table).
    ///
    /// Tuning:
    ///   0.05 (default): tight — mirrors the prior 0.96 recall floor's effective
    ///     strictness. SIFT-128 → m=32/8-bit (recall 0.997). Most 8-bit configs
    ///     at adequate m qualify; 4-bit only when sub_dim is small enough to
    ///     keep distortion low. Tighten further for quality-critical builds;
    ///     loosen (0.10–0.15) for high-dimensional data where coarser PQ is
    ///     acceptable.
    ///   LOWER (e.g. 0.05): forces higher fidelity PQ (higher m). Use when recall
    ///     is critical and you're willing to pay for more distance computations.
    ///   HIGHER (e.g. 0.30): admits aggressive low-m configs. Good for very
    ///     large N where throughput dominates.
    ///
    /// Note: the bound applies to the *probe* distortion (measured on a 20K
    /// reservoir subset), not end-to-end search recall. The elbow + bound pick
    /// a PQ config with good distance fidelity; the graph then navigates using
    /// those estimates. End-to-end recall depends on graph params (R, L) and
    /// dataset structure (clustering, dimensionality) in addition to PQ quality.
    float pq_max_distortion = 0.0f;
    uint64_t build_ram_budget = 0;  ///< 0 = auto (50% of physical RAM)
    MetricKind metric = MetricKind::L2Sq;
    uint32_t num_threads = 0;   ///< 0 = hardware_concurrency
    uint32_t max_occlusion = 0; ///< 0 = auto (max(L_build, R+1))

    /// Target proximity (in-band fraction) for auto-parameter estimation.
    /// 0 = use default (0.95). Measured at target_topk. Drives R selection.
    float proximity_target = 0.0f;

    /// Target recall@target_topk for auto-parameter estimation.
    /// 0 = use default (0.95). Workload-dependent: set BOTH proximity_target and
    /// recall_target to gate on both (the stricter one binds); set just one to
    /// gate on that alone. Neither → default recall@0.95.
    float recall_target = 0.0f;

    /// The k at which recall/proximity targets are measured and at which mini-build
    /// search quality is evaluated. Default 100 (VIBE / modern-retrieval convention).
    /// Mini-build measurements (alpha sweep, R validation, PQ verify) all run at
    /// this k. Note: proximity semantics depend on k — at k=100 the k-th NN
    /// distance is larger than at k=10, so proximity is looser at higher k.
    uint32_t target_topk = 100;

    /// Entry-point count (k-means centroids). 0 = default (16). Higher = better
    /// coverage for clustered datasets, more startup distance computations.
    uint16_t n_entry_points = 0;
    /// Multi-start: number of entry points to seed each search from (top-M
    /// closest to the query). 0 = default (4).
    uint16_t n_search_entry_points = 0;
    // NOTE: recall_target (defined above) also serves as the search early-exit
    // target. When set (e.g. 0.95), estimate_config picks the cheapest config
    // meeting it AND beam_search terminates early when convergence is detected.
    // No separate target_recall field needed.

    /// Closure-factor inputs for auto estimation. 0 = auto-estimated by
    /// estimate_config (closure_f_target defaults to 0.15; closure_d_eff
    /// estimated from LID). When non-zero, these lock the closure inputs.
    float closure_f_target = 0.0f;
    float closure_d_eff = 0.0f;

    /// PQ anisotropic codebook training (scale-transform k-means). When true,
    /// each subspace's dims are scaled by √(eigval/mean_eigval) of the subspace
    /// covariance before k-means; centroids are unscaled back afterward. Only
    /// affects codebook training — encoding/search/serialized format unchanged.
    /// One-time build-time cost.
    bool pq_anisotropy = false;

    /// OPQ (PCA rotation). When true, a d×d PCA rotation is learned from the
    /// training-sample covariance and applied to vectors before PQ encoding
    /// and to queries before LUT construction, so PQ splits align with the
    /// principal components. The rotation matrix is serialized with the
    /// quantizer. Adds a one-time ~20s eigendecomposition at d=768.
    bool pq_opq = false;

    /// Partition count override. 0 = auto (resolved from build_ram_budget,
    /// plus a recall-driven floor when `ivf_mode` is true). >0 forces exactly
    /// this many partitions. Only meaningful for partitioned / IVF builds.
    uint32_t partition_count = 0;

    /// IVF-probe mode. When true, resolve_params applies a recall-driven
    /// partition-count floor of `clamp(sqrt(N)/8, 2, 256)` so the IVF path
    /// has enough shards for good routing even when RAM would allow K=1.
    /// See docs/ivf_probe_design.md "K selection". The recall floor is a
    /// lower bound — the RAM-driven K still wins when it's larger.
    bool ivf_mode = false;

    /// Metric-recommendation strategy for `Estimator::estimate_config`.
    /// "both" (default): build mini-indexes under L2sq AND IP, measure
    /// end-to-end search recall@k for each — the authoritative signal.
    /// "l2sq"/"ip": probe only the named metric (skips the second mini-build).
    /// Stored as a string (not an enum) to keep BuildConfig CLI-friendly.
    std::string metric_reco = "both";
};

/// Result of a build operation.
struct BuildResult {
    std::string index_path;
    uint64_t n_vectors = 0;
    Dim dim = 0;
    double build_time_sec = 0;
    uint16_t R = 0;
    uint16_t L_build = 0;
    uint16_t pq_m = 0;
    uint8_t pq_bits = 8;
};

/// Search configuration.
struct SearchConfig {
    uint32_t k = 10;
    uint32_t L_search = 200;
    uint32_t io_limit = 0;  ///< 0 = unlimited (visit as many nodes as L_search allows)

    /// IVF-probe: number of shards to probe per query. 0 = auto (max(1, K/4)).
    /// Ignored for single-shard (non-IVF) indexes.
    uint32_t n_probe = 0;

    /// IVF-probe merge oversampling: each probed shard is searched at
    /// k_local = k × merge_oversample, then results are merged/deduped and
    /// truncated to k. 1 = no oversampling. Ignored for single-shard indexes.
    uint32_t merge_oversample = 2;

    /// IVF-probe multi-probe ratio (Voronoi-boundary recall recovery). After
    /// picking the n_probe nearest centroids, extend the probe set to include
    /// ANY centroid whose distance ≤ ratio × d[n_probe-1] (the n_probe-th
    /// nearest centroid's distance). 1.0 = strict n_probe (off, back-compat).
    /// Values >1.0 catch true NNs that live in boundary shards routing missed.
    /// Increases recall at variable per-query cost (boundary queries probe more
    /// shards). See docs/ivf_routing_analysis.md. Measured (c4a K=32, np=4):
    /// ratio=1.05 recovers ~3pp routing recall for +1.9 avg shards/query.
    /// Ignored for single-shard indexes.
    float multiprobe_ratio = 1.0f;

    /// Search-time early-exit patience (post-convergence stall count before
    /// terminating beam_search). 0 = disabled (full L_search). UINT32_MAX =
    /// defer to the index's baked-in value (back-compat default). Any other
    /// value overrides at query time. Build-time beam_search is NEVER affected
    /// (it always runs to completion regardless of this or the index value).
    /// IVF uses this to pass a higher patience than the merged default.
    uint32_t early_exit_patience = 0xFFFFFFFFu;  // kDeferToParams
};

/// Adaptive parameters resolved from dataset/machine properties (Issue 37).
struct ResolvedParams {
    uint16_t R = 64;
    uint16_t L = 100;
    uint16_t L_build = 100;
    float alpha = 1.2f;
    uint16_t pq_m = 32;
    uint8_t pq_bits = 8;          ///< 0 = auto (resolved by reservoir probe in pass1)
    float pq_max_distortion = 0.0f;   ///< 0 = default (1.20); max acceptable PQ distortion
    uint32_t max_occlusion = 750;
    MetricKind metric = MetricKind::L2Sq;
    uint64_t build_ram_budget = 0;
    uint32_t num_threads = 0;
    uint32_t partition_count = 1; ///< Partition count (>1 → partitioned build)
    float closure_factor = 1.033f;  ///< Shard overlap radius ratio
    uint16_t n_entry_points = 16;   ///< K-means centroid count for entry-point selection
    uint16_t n_search_entry_points = 4;  ///< Multi-start: top-M entry points per query
    float target_recall = 0.0f;     ///< Recall target the index was built for (0 = unspecified). Drives search early-exit.
    uint32_t early_exit_patience = 0;  ///< Search early-exit: terminate after N stalled pops post-convergence (0 = disabled; set by resolve_params when recall_target is set)
    bool pq_anisotropy = false;  ///< PQ covariance-based anisotropic codebook training. Threaded into PqQuantizer::train.
    bool pq_opq = false;          ///< OPQ PCA rotation. Threaded into PqQuantizer::train.
};

/// Estimation diagnostics produced by `Engine::estimate_config` (and the
/// Estimator in the post-refactor world). Display-only — they do NOT influence
/// the build (the inputs that drove the resolved params are already in
/// `ResolvedParams`) and are NOT serialized to `.meta`. Kept in a separate
/// struct so the build-input `ResolvedParams` stays pure.
struct EstimationDiagnostics {
    double median_lid = 0.0;       ///< MLE LID from probe truth (0 = unmeasured)
    double avg_degree = 0.0;       ///< R̄ from mini-build (0 = unmeasured)
    double clustering_coeff = 0.0; ///< clustering coefficient (0 = unmeasured)
    double dead_end_frac = 0.0;    ///< fraction of nodes with degree ≤ 1

    /// L2 norm statistics over the sampled vectors. Detects L2-normalization
    /// (tight clustering at ||x|| ≈ 1). `norm_cv` = stddev/mean; 0 = unmeasured.
    /// NOTE: tight original norms are necessary but NOT sufficient for IP-safety;
    /// `ip_l2sq_overlap` is the authoritative metric-recommendation signal.
    double norm_mean = 0.0;
    double norm_stddev = 0.0;
    double norm_min = 0.0;
    double norm_max = 0.0;
    double norm_cv = 0.0;          ///< 0 = unmeasured; <0.01 ≈ unit-normalized

    /// Mean top-k recall of IP-ADC and L2sq-ADC against brute-force ground truth,
    /// measured by `Estimator::estimate_config` after the reference mini-index
    /// is built. The authoritative metric-recommendation signal: if `ip_recall`
    /// ≥ `l2sq_recall`, IP gives equal or better recall at lower per-eval cost;
    /// otherwise L2sq is required (IP misranks, recall collapses — SIFT-1M:
    /// 0.99 → 0.66). -1.0 = unmeasured. See `docs/plans/metric_per_tier_plan.md`.
    double ip_recall_adc = -1.0;
    double l2sq_recall_adc = -1.0;
};

/// Result of `Engine::estimate_config`: resolved build inputs + the measured
/// signals that informed them (for display / `--explain`).
struct EstimateResult {
    ResolvedParams params;
    EstimationDiagnostics diag;
};

/// Structural properties of a built graph, measured for parameter estimation.
struct GraphStats {
    double avg_degree = 0.0;       ///< mean post-prune degree (R̄)
    double clustering_coeff = 0.0; ///< sampled triangle fraction
    double dead_end_frac = 0.0;    ///< fraction of nodes with degree ≤ 1
    double median_lid = 0.0;       ///< MLE LID from k-NN distances
};

/// Auto-resolve parameters from dataset properties + machine properties.
/// If any field in `overrides` is non-zero/non-default, it takes precedence.
ResolvedParams resolve_params(uint64_t n_vectors, Dim dim,
                               const BuildConfig& overrides);

}  // namespace sextant
