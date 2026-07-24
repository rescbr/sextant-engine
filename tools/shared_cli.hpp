#pragma once

/// Shared flag definitions and pretty-printers for the build/analyze/autobuild
/// commands. All three accept a common base of override knobs so new params can
/// never be added to one command and missed in another. Mode-specific extras
/// are added per-command after calling add_common_flags.

#include "sextant/config.hpp"
#include "sextant/estimator.hpp"
#include "sextant/logging.hpp"
#include "sextant/vector_source.hpp"

#include <cmdline/cmdline.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace sextant_cli {

/// Mode bitmask for which extras to add.
enum class Mode {
    Analyze    = 0,                          // read-only estimate
    Build      = 1 << 0,                     // explicit knobs + build
    Autobuild  = 1 << 1,                     // estimate + build
};

/// Add the common-base flags to `p`. Called by all three commands.
/// Flags use self-documenting long names; the single-letter paper notation
/// appears in each flag's help text for researchers familiar with the
/// Vamana/DiskANN/PQ literature.
inline void add_common_flags(cmdline::parser& p) {
    p.add<std::string>("input", 0, "Input .fbin/.ibin/.bbin file", true);
    p.add<std::string>("index", 0,
        "Index name/path prefix (required for build/autobuild; ignored by analyze)",
        false, "");
    // --- Knobs (0 / auto = resolved later) ---
    p.add<uint16_t>("max-node-neighbors", 0,
        "Max graph degree — neighbors per node — after robust-prune "
        "(R in Vamana literature). 0 = auto.",
        false, 0);
    p.add<uint16_t>("beam-width-ceiling", 0,
        "Construct-time beam-width ceiling (L in Vamana literature). The T5 "
        "dynamic ramp uses ½L → L as construction progresses. 0 = auto "
        "(= max(2*R, 100)).",
        false, 0);
    p.add<float>("prune-threshold", 0,
        "Robust-prune threshold (alpha in Vamana literature): "
        "alpha * d(p,pp) <= d(q,pp). 1.0 = no geometric stretching, "
        "1.2 = default, 1.5 = aggressive. 0 = auto (1.2).",
        false, 0.0f);
    p.add<uint16_t>("pq-segments", 0,
        "PQ segment count (m in PQ literature) — the vector is split into "
        "this many sub-vectors, each quantized independently. Higher = more "
        "faithful, larger codes. 0 = auto (probe-selected).",
        false, 0);
    p.add<std::string>("pq-bits", 0,
        "PQ bits per segment: 4, 8, or auto.", false, "auto");
    p.add<float>("pq-max-distortion", 0,
        "Max PQ distortion (median |1 - pq_dist/true_dist|) for auto (m,bits) "
        "selection. 0 = default (0.05).",
        false, 0.0f);
    p.add<float>("pq-anisotropy", 0,
        "Enable covariance-based anisotropic PQ codebook training (0 = off, "
        ">0 = on). Each subspace's dims are scaled by √(eigval/mean_eigval) "
        "of the subspace covariance before k-means (scale-transform trick), "
        "so high-variance dims get more weight. Build-time only; "
        "encoding/search/format unchanged. The value is a boolean "
        "(0/off vs 1/on); the covariance determines the per-dim weights.",
        false, 0.0f);
    p.add<float>("pq-opq", 0,
        "Enable OPQ (Optimized Product Quantization) via PCA rotation (0 = off, "
        ">0 = on). Decorrelates dimensions before PQ splitting via a learned "
        "d x d PCA rotation, improving PQ quality (~7.9% lower reconstruction "
        "MSE on arxiv-nomic). Adds ~20s to build time (768x768 Jacobi "
        "eigendecomposition, one-time) and ~0.3ms/query (rotation matmul, "
        "SIMD). The rotation is serialized with the quantizer. Boolean flag "
        "(the value is ignored); covariance determines the rotation.",
        false, 0.0f);
    p.add<std::string>("quantizer", 0,
        "Quantizer type: 'pq' (default, standard k-means) or 'anisotropic-pq' "
        "(ScaNN-style anisotropic Lloyd's training; search path identical). "
        "See docs/plans/metric_per_tier_plan.md Phase 2.",
        false, "pq");
    p.add<float>("anisotropy-threshold", 0,
        "ScaNN anisotropic threshold T for anisotropic-pq training (default "
        "0.2 → η ≈ 4.125). Higher T weights parallel quantization error more. "
        "Only meaningful with --quantizer anisotropic-pq. See ScaNN paper §3.",
        false, 0.2f);
    p.add<uint32_t>("threads", 0,
        "Threads for build/mini-builds (0 = hardware_concurrency).",
        false, 0);
    p.add<std::string>("metric", 0, "l2sq or ip", false, "l2sq");
    p.add<std::string>("log-level", 0,
        "Log level: debug, info, warn, error", false, "info");
}

/// Add mode-specific extras. Call after add_common_flags.
inline void add_mode_extras(cmdline::parser& p, Mode mode) {
    if (mode == Mode::Analyze || mode == Mode::Autobuild) {
        p.add<float>("proximity-target", 0,
            "Target proximity (in-band fraction, 0-1) for auto R selection "
            "via OPT-SNG. 0 = default (0.95).",
            false, 0.0f);
        p.add<float>("recall-target", 0,
            "Target recall@k (alternative to proximity-target). 0 = not used. "
            "Set both proximity-target and recall-target to gate on both "
            "(stricter binds); set just one to gate on that alone.",
            false, 0.0f);
        p.add<uint32_t>("target-topk", 0,
            "The k at which recall/proximity targets are measured and "
            "mini-builds are evaluated. Default 100 (VIBE / modern-retrieval "
            "convention). Proximity semantics depend on k — at k=100 the k-th "
            "NN distance is larger than at k=10, so proximity is looser at "
            "higher k.",
            false, 100);
        // build-ram drives K (partition count) resolution, so analyze must
        // accept it too — K is one of the resolved params analyze reports.
        p.add<uint64_t>("build-ram", 0,
            "Build RAM budget in bytes (forces partitioning if small). "
            "0 = auto (50% of physical RAM). Affects K (partition count) "
            "resolution in analyze/autobuild.",
            false, 0);
        // IVF + partition-count affect K resolution — analyze previews them.
        p.add("ivf", 0,
            "Resolve params for IVF-probe mode: applies a recall-driven "
            "partition-count floor of sqrt(N)/8 (clamped [2, 256]) so the "
            "IVF path has enough shards for good routing. Analyze shows the "
            "resulting K; autobuild then builds the shard set. See "
            "docs/ivf_probe_design.md.");
        p.add<uint32_t>("partition-count", 0,
            "Partition count (K) override. 0 = auto (RAM-driven, plus the "
            "sqrt(N)/8 recall floor when --ivf is set).",
            false, 0);
        // Metric-recommendation strategy: build mini-indexes under both metrics
        // (default) and measure end-to-end search recall@k for each, OR probe
        // only the requested metric. 'both' is the authoritative signal but
        // costs an extra mini-build (~10-30s). See docs/plans/metric_per_tier_plan.md.
        p.add<std::string>("metric-reco", 0,
            "Metric recommendation strategy for analyze: 'both' (build "
            "mini-indexes under L2sq AND IP, measure search recall@k for each), "
            "'l2sq', or 'ip' (probe only the named metric). Default 'both'.",
            false, "both");
    }
    if (mode == Mode::Build || mode == Mode::Autobuild) {
        p.add<uint32_t>("prune-candidate-cap", 0,
            "Cap on the robust-prune candidate pool (max-occlusion in "
            "DiskANN). 0 = auto (= max(L_build, R+1)).",
            false, 0);
        p.add<uint32_t>("n-search-entry-points", 0,
            "Multi-start: number of entry points to seed each search from "
            "(top-M closest to the query). 0 = default (4).",
            false, 0);
        // build-ram also accepted by build (forces partitioning / sets K).
        if (mode == Mode::Build) {
            p.add<uint64_t>("build-ram", 0,
                "Build RAM budget in bytes (forces partitioning if small). "
                "0 = auto (50% of physical RAM).",
                false, 0);
        }
    }
    if (mode == Mode::Autobuild) {
        p.add<uint32_t>("ivf-n-probe", 0,
            "IVF default n_probe (shards probed per query). 0 = auto "
            "(max(1, K/4)). Only meaningful with --ivf.",
            false, 0);
    }
}

/// Parse the common flags into a BuildConfig. Called by all three commands
/// AFTER p.parse_check(). Pulls values by the new flag names.
inline sextant::BuildConfig build_config_from_parser(const cmdline::parser& p) {
    sextant::BuildConfig cfg;
    cfg.R              = p.get<uint16_t>("max-node-neighbors");
    cfg.L              = p.get<uint16_t>("beam-width-ceiling");
    cfg.alpha          = p.get<float>("prune-threshold");
    cfg.pq_m           = p.get<uint16_t>("pq-segments");
    {
        const std::string b = p.get<std::string>("pq-bits");
        if (b == "auto" || b == "0")      cfg.pq_bits = 0;
        else if (b == "4")                cfg.pq_bits = 4;
        else if (b == "8")                cfg.pq_bits = 8;
        else throw std::runtime_error("--pq-bits must be 4, 8, or auto");
    }
    cfg.pq_max_distortion = p.get<float>("pq-max-distortion");
    cfg.pq_anisotropy = (p.get<float>("pq-anisotropy") > 0.0f);
    cfg.pq_opq = (p.get<float>("pq-opq") > 0.0f);
    cfg.anisotropic_pq = (p.get<std::string>("quantizer") == "anisotropic-pq");
    cfg.num_threads       = p.get<uint32_t>("threads");
    {
        const std::string m = p.get<std::string>("metric");
        cfg.metric = (m == "ip") ? sextant::MetricKind::InnerProduct
                                  : sextant::MetricKind::L2Sq;
    }
    // target-topk is only defined for Analyze/Autobuild modes; guard with
    // a try-catch because cmdline::parser::exist() throws if the flag wasn't
    // registered (it doesn't return false for undefined flags).
    try {
        if (p.exist("target-topk")) {
            cfg.target_topk = p.get<uint32_t>("target-topk");
        }
    } catch (...) {
        // Flag not registered for this mode — leave default.
    }
    // n-search-entry-points: same guard (only on Build/Autobuild).
    try {
        if (p.exist("n-search-entry-points")) {
            cfg.n_search_entry_points = p.get<uint32_t>("n-search-entry-points");
        }
    } catch (...) {}
    // partition-count: same guard (only on Build/Autobuild).
    try {
        if (p.exist("partition-count")) {
            cfg.partition_count = p.get<uint32_t>("partition-count");
        }
    } catch (...) {}
    return cfg;
}

/// Apply log level from the parsed flags.
inline void apply_log_level(const cmdline::parser& p) {
    const auto lvl = p.get<std::string>("log-level");
    if (lvl == "debug")      sextant::set_log_level(sextant::LogLevel::Debug);
    else if (lvl == "warn")  sextant::set_log_level(sextant::LogLevel::Warn);
    else if (lvl == "error") sextant::set_log_level(sextant::LogLevel::Error);
    // info is default
}

/// Pretty-print the estimate_config analysis (measured signals + resolved
/// params + recommended build command). Shared by `sextant analyze` and
/// `sextant autobuild` so the output is identical. `input` is the original
/// input path (for display + the recommended build command).
inline void print_analysis_(sextant::VectorSource& source,
                            const std::string& input,
                            const sextant::BuildConfig& cfg,
                            const sextant::ResolvedParams& params,
                            const sextant::EstimationDiagnostics& diag) {
    const uint64_t n = source.count();
    const sextant::Dim dim = source.dim();
    const std::string metric_str =
        (cfg.metric == sextant::MetricKind::InnerProduct) ? "ip" : "l2sq";

    const bool R_locked = (cfg.R != 0);
    const bool alpha_locked = (cfg.alpha != 0.0f);
    const bool pq_m_locked = (cfg.pq_m != 0);
    const bool pq_bits_locked = (cfg.pq_bits != 0);

    std::cout << "\n═══ Sextant Analyze ═══\n";
    std::cout << "  input:      " << input << "\n";
    std::cout << "  n_vectors:  " << n << "\n";
    std::cout << "  dim:        " << dim << "\n";
    std::cout << "  metric:     " << metric_str << "\n\n";

    // Measured signals.
    std::cout << "─── Measured Signals ───\n";
    std::cout << "  median LID:           " << std::fixed
              << std::setprecision(2) << diag.median_lid << "\n";
    std::cout << "  avg degree (R̄):       " << std::setprecision(2)
              << diag.avg_degree << "\n";
    std::cout << "  clustering coeff:     " << std::setprecision(4)
              << diag.clustering_coeff << "\n";
    std::cout << "  dead-end fraction:    " << std::setprecision(4)
              << diag.dead_end_frac << "\n";

    // Metric recommendation. Authoritative signal: measured recall of IP-ADC vs
    // L2sq-ADC against brute-force ground truth (computed in estimate_config
    // after the reference mini-index is built). Recommend the higher-recall
    // metric; on a tie, prefer IP (cheaper per eval). For normalized data the
    // two are close; for non-normalized data L2sq wins by a wide margin
    // (SIFT-1M: L2sq 0.99 vs IP 0.66). Original-norm stats shown as context.
    if (diag.norm_mean > 0.0) {
        std::cout << "  L2 norm:              mean=" << std::setprecision(4)
                  << diag.norm_mean << " cv=" << diag.norm_cv
                  << " range=[" << diag.norm_min << ", " << diag.norm_max
                  << "]\n";
    }
    if (diag.ip_recall_adc >= 0.0 && diag.l2sq_recall_adc >= 0.0) {
        const bool ip_wins = diag.ip_recall_adc >= diag.l2sq_recall_adc;
        const char* reco_metric = ip_wins ? "ip" : "l2sq";
        std::cout << "  PQ-ADC recall:        IP="
                  << std::setprecision(4) << diag.ip_recall_adc
                  << "  L2sq=" << diag.l2sq_recall_adc << "\n";
        std::cout << "  recommended metric:   " << reco_metric;
        if (std::string(reco_metric) != metric_str) {
            std::cout << "  [you passed --metric " << metric_str
                      << " — see note below]";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Resolved params.
    std::cout << "─── Resolved Parameters ───\n";
    std::cout << "  R (degree):     " << params.R << "  ["
              << (R_locked ? "locked" : "estimated") << "]\n";
    std::cout << "  alpha:          " << std::setprecision(1) << params.alpha
              << "  [" << (alpha_locked ? "locked" : "estimated") << "]\n";
    std::cout << "  pq_m:           " << params.pq_m << "  ["
              << (pq_m_locked ? "locked" : "estimated") << "]\n";
    std::cout << "  pq_bits:        " << static_cast<int>(params.pq_bits)
              << "  [" << (pq_bits_locked ? "locked" : "estimated") << "]\n";
    std::cout << "  L_build:        " << params.L_build
              << (cfg.L != 0 ? "  [locked]\n" : "\n");
    std::cout << "  max_occlusion:  " << params.max_occlusion << "\n";
    std::cout << "  closure_factor: " << std::setprecision(4)
              << params.closure_factor << "\n";
    std::cout << "  K (partitions): " << params.partition_count << "\n\n";

    // Recommended build command.
    std::cout << "─── Build Command ───\n";
    std::cout << "  sextant build --input " << input
              << " --index <INDEX> --metric " << metric_str
              << " --max-node-neighbors " << params.R
              << " --prune-threshold " << std::setprecision(1) << params.alpha
              << " --pq-segments " << params.pq_m
              << " --pq-bits " << static_cast<int>(params.pq_bits);
    if (cfg.num_threads > 0)
        std::cout << " --threads " << cfg.num_threads;
    std::cout << "\n";

    // Prominent metric mismatch warning. The user passed --metric X but the
    // measured PQ-ADC recall recommends Y. This is the one place we second-guess
    // the user, because picking the wrong metric silently destroys recall on
    // non-normalized data (SIFT-1M: 0.99 → 0.66).
    if (diag.ip_recall_adc >= 0.0 && diag.l2sq_recall_adc >= 0.0) {
        const bool ip_wins = diag.ip_recall_adc >= diag.l2sq_recall_adc;
        const char* reco = ip_wins ? "ip" : "l2sq";
        if (std::string(reco) != metric_str) {
            std::cout << "\n  ⚠ metric mismatch: measured PQ-ADC recall IP="
                      << std::setprecision(4) << diag.ip_recall_adc
                      << " vs L2sq=" << diag.l2sq_recall_adc
                      << ". Recommended --metric " << reco << ", you passed "
                      << "--metric " << metric_str << ".\n";
            if (ip_wins) {
                std::cout << "    IP matches or beats L2sq on recall and is "
                          << "cheaper per eval; use IP.\n";
            } else {
                std::cout << "    L2sq has higher recall on this data (IP-ADC "
                          << "misranks when ||x̃||² varies across codes).\n";
            }
        }
    }
}

}  // namespace sextant_cli
