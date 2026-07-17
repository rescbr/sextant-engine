#pragma once

/// @file config.hpp
/// Build and search configuration structs.

#include <sextant/types.hpp>
#include <cstdint>

namespace sextant {

/// Build mode. ADC (raw-vector construct) has been removed — the FP16 prune
/// hybrid made it redundant (only +0.0003 recall at 1.4× build cost). The enum
/// is retained with SDC as the sole value for API compatibility.
enum class BuildMode : uint8_t { SDC = 0 };

/// Build configuration. Fields set to 0/default are auto-resolved.
struct BuildConfig {
    uint16_t R = 0;             ///< 0 = auto from N
    uint16_t L = 0;             ///< 0 = auto from R
    float alpha = 0.0f;         ///< Vamana prune threshold: alpha * d(p,pp) <= d(q,pp).
                                ///< 0 = auto (1.2). Purely the prune threshold;
                                ///< build mode is set via `build_mode`.
    BuildMode build_mode = BuildMode::SDC;  ///< SDC (code-distance construct); ADC removed
    /// Inline PQ codes per node. 0 = compact (no inline). DEPRECATED: the
    /// two-cache search architecture handles code locality without inflating
    /// node size, so inline PQ provides no benefit. Retained for backward
    /// compatibility with old indexes; new builds should use 0.
    uint16_t inline_pq_count = 0;
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
    ///          search path uses the ADC LUT (m×K×4 bytes, always L2-resident),
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
    /// 0 = use default (0.95). Drives R selection via OPT-SNG formula in
    /// estimate_config. Only consulted when R is auto.
    float proximity_target = 0.0f;

    /// Target recall@k for auto-parameter estimation (alternative to
    /// proximity_target). 0 = not used. If both set, proximity_target wins.
    float recall_target = 0.0f;

    /// Closure-factor inputs for auto estimation. 0 = auto-estimated by
    /// estimate_config (closure_f_target defaults to 0.15; closure_d_eff
    /// estimated from LID). When non-zero, these lock the closure inputs.
    float closure_f_target = 0.0f;
    float closure_d_eff = 0.0f;
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
    uint32_t rerank_factor = 10;
    uint32_t io_limit = 0;  ///< 0 = unlimited (visit as many nodes as L_search allows)
};

}  // namespace sextant
