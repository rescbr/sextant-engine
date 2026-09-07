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

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
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
    p.add<uint16_t>("scan-pq-bits", 0,
        "IVF-scan codebook bits per segment: 4 (default, FastScan nibble) "
        "or 8 (byte-per-code, ~½ throughput but breaks the recall ceiling "
        "on high-LID datasets). The routing PQ is unaffected (always 8-bit).",
        false, 4);
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
        "Quantizer type: 'local_scalar' (DEFAULT — per-leaf 4-bit uniform "
        "ruler, ~0.77 KB/vec; no global training sample, append/rebalance "
        "safe; best recall-per-byte on embedding corpora), 'local_pq' "
        "(per-leaf residual codebooks, smallest footprint), 'pq' (global "
        "k-means), 'anisotropic-pq', 'scalar_uniform'/'scalar_lloydmax' "
        "(global per-dim rulers), 'prq'. GLOBAL quantizers (pq, slm, sun, "
        "shape, prq) train on a 20K random reservoir — their rulers/"
        "codebooks are FROZEN AT BUILD: appended vectors are encoded against "
        "the build-time distribution, so pick them when the corpus is "
        "homogeneous and stable; use a local quantizer when data drifts or "
        "appends are expected. See docs/quantizer-selection.md.",
        false, "local_scalar");
    p.add<float>("anisotropy-threshold", 0,
        "ScaNN anisotropic threshold T for anisotropic-pq training (default "
        "0.2 → η ≈ 4.125). Higher T weights parallel quantization error more. "
        "Only meaningful with --quantizer anisotropic-pq. See ScaNN paper §3.",
        false, 0.2f);
    p.add<uint32_t>("prq-nsplits", 0,
        "PRQ nsplits (sub-space count) for --quantizer prq. 0 = auto (dim/8, "
        "i.e. sub_dim=8 per sub-space). Must divide both dim and m4. Only "
        "meaningful with --quantizer prq.",
        false, 0);
    p.add<uint32_t>("prq-beam-size", 0,
        "PRQ beam size for encoding (1 = greedy, >1 = beam search). "
        "Higher = better recall, slower build. Only meaningful with "
        "--quantizer prq. Default: 1.",
        false, 1);
    p.add<std::string>("prq-encode-mode", 0,
        "PRQ encoding strategy: 'greedy' (default, sequential residual), "
        "'beam' (beam search), or 'icm' (ICM+ILS coordinate descent — best "
        "quality, on-the-fly compute). Only meaningful with --quantizer prq.",
        false, "greedy");
    p.add<uint32_t>("prq-icm-iters", 0,
        "PRQ ICM sweeps per ILS cycle (default 4). Only meaningful with "
        "--prq-encode-mode icm.",
        false, 4);
    p.add<uint32_t>("prq-ils-iters", 0,
        "PRQ ILS cycles: perturb + ICM + accept (default 4). Only meaningful "
        "with --prq-encode-mode icm.",
        false, 4);
    p.add<uint32_t>("prq-ils-perturb", 0,
        "PRQ ILS perturbation count: codes to randomize per cycle (default 4). "
        "Only meaningful with --prq-encode-mode icm.",
        false, 4);
    p.add<uint32_t>("prq-lsq-train-iters", 0,
        "PRQ LSQ training iterations: alternating codebook update + ICM "
        "re-encode (default 0 = k-means only). ~10-25 for full LSQ training. "
        "Only meaningful with --quantizer prq.",
        false, 0);
    p.add<uint32_t>("threads", 0,
        "Threads for build/mini-builds (0 = hardware_concurrency).",
        false, 0);
    p.add<std::string>("metric", 0, "l2sq or ip", false, "l2sq");
    p.add<std::string>("log-level", 0,
        "Log level: debug, info, warn, error", false, "info");

    // K-driving params (build-ram, partition-count) are common to all modes —
    // build needs them to actually partition, analyze previews them, autobuild
    // threads them through. The K heuristic lives in resolve_params
    // (target ~200k vectors/shard, clamped [16, 8192]); explicit
    // --partition-count wins.
    p.add<uint64_t>("build-ram", 0,
        "Build RAM budget in bytes (forces partitioning if small). "
        "0 = auto (50% of physical RAM). Affects K (partition count) "
        "resolution.",
        false, 0);
    p.add<uint32_t>("partition-count", 0,
        "Partition count (K) override. 0 = auto (RAM-driven, plus the "
        "shard-size heuristic for the chosen build path).",
        false, 0);
    p.add<float>("partition-balance-factor", 0,
        "Partition size-balance strength (SPANN-style λ, arXiv:2111.08566). "
        "0 = off (plain k-means + closure). >0 caps each shard's size at "
        "(N/K)·(1 + 1/λ), bounding the worst-case disk read per shard at "
        "paged-billion-scale. Default 4 (zero search-time cost, reduces skew). "
        "Only the closure-replication overhead is capped; primary membership "
        "is never dropped.",
        false, 4.0f);
    p.add<float>("closure-epsilon", 0,
        "Absolute margin for SPANN-style boundary posting (arXiv:2111.08566). "
        "When >0, vectors within d_best + epsilon of a centroid are replicated "
        "to that shard. Structurally K-robust: as K grows, epsilon covers an "
        "increasing fraction of the cell radius. -1 = auto (0.2 × mean NN "
        "distance, default). 0 = use ratio-based closure (closure_factor × d_best, "
        "K-fragile, deprecated). See docs/closure_factor_derivation.md.",
        false, -1.0f);
    p.add<float>("closure-f-target", 0,
        "Target fraction of non-replicated vectors for closure_factor auto-"
        "computation. Default 0.15 (85% replicated). Higher = more replication "
        "= better boundary recall, more storage. Only affects ratio-based "
        "closure (when closure-epsilon=0).",
        false, 0.0f);
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
    }
    if (mode == Mode::Build || mode == Mode::Autobuild) {
        p.add<std::string>("graph-build-metric", 0,
            "Graph construction distance source: pq (default), fp16, or "
            "fp32 (K=1 path only; fp32 loads +N*dim*4B of raw vectors)",
            false, "pq");
        p.add<uint32_t>("prune-candidate-cap", 0,
            "Cap on the robust-prune candidate pool (max-occlusion in "
            "DiskANN). 0 = auto (= max(L_build, R+1)).",
            false, 0);
        p.add<uint32_t>("n-search-entry-points", 0,
            "Multi-start: number of entry points to seed each search from "
            "(top-M closest to the query). 0 = default (4).",
            false, 0);
        // (build-ram, partition-count are in add_common_flags now.)
    }
}

/// Parse one scan-feedback probe spec (SearchConfig::feedback):
///   fixed:F | stall:M | kth:M   optionally followed by ,min:N
/// Throws std::runtime_error with the flag name on malformed input.
inline sextant::FeedbackProbe parse_feedback_spec(const std::string& spec) {
    sextant::FeedbackProbe fb;
    std::string main = spec;
    if (const auto comma = spec.find(','); comma != std::string::npos) {
        main = spec.substr(0, comma);
        const std::string mins = spec.substr(comma + 1);
        if (mins.rfind("min:", 0) != 0)
            throw std::runtime_error("--feedback: expected ,min:N suffix");
        uint32_t mv = 0;
        auto [ptr, ec] = std::from_chars(mins.data() + 4, mins.data() + mins.size(), mv);
        if (ec != std::errc() || ptr != mins.data() + mins.size() || mv == 0)
            throw std::runtime_error("--feedback: bad min value");
        fb.min_blocks = mv;
    }
    const auto colon = main.find(':');
    if (colon == std::string::npos)
        throw std::runtime_error("--feedback: expected fixed:F | stall:M | kth:M");
    const std::string name = main.substr(0, colon);
    const std::string val = main.substr(colon + 1);
    if (name == "fixed") {
        float f = 0;
        auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), f);
        if (ec != std::errc() || ptr != val.data() + val.size()
            || f <= 0.0f || f > 1.0f)
            throw std::runtime_error("--feedback: fixed fraction must be in (0,1]");
        fb.mode = sextant::FeedbackProbe::Mode::Fixed;
        fb.fixed_fraction = f;
    } else {
        uint32_t m = 0;
        auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), m);
        if (ec != std::errc() || ptr != val.data() + val.size() || m == 0)
            throw std::runtime_error("--feedback: threshold M must be a positive integer");
        fb.mode = name == "stall" ? sextant::FeedbackProbe::Mode::Stall
                                  : sextant::FeedbackProbe::Mode::Kth;
        if (name != "stall" && name != "kth")
            throw std::runtime_error("--feedback: unknown mode '" + name + "'");
        fb.m = m;
    }
    return fb;
}

/// Parse a comma-separated list of feedback specs into (spec, parsed) rows.
inline std::vector<std::pair<std::string, sextant::FeedbackProbe>>
parse_feedback_list(const std::string& list) {
    std::vector<std::pair<std::string, sextant::FeedbackProbe>> rows;
    std::string item;
    std::istringstream iss(list);
    while (std::getline(iss, item, ',')) {
        if (item.empty()) continue;
        rows.emplace_back(item, parse_feedback_spec(item));
    }
    return rows;
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
    {
        const uint16_t sb = p.get<uint16_t>("scan-pq-bits");
        if (sb == 4 || sb == 8) cfg.scan_pq_bits = static_cast<uint8_t>(sb);
        else throw std::runtime_error("--scan-pq-bits must be 4 or 8");
    }
    cfg.pq_max_distortion = p.get<float>("pq-max-distortion");
    cfg.pq_anisotropy = (p.get<float>("pq-anisotropy") > 0.0f);
    cfg.pq_opq = (p.get<float>("pq-opq") > 0.0f);
    cfg.quantizer_type = p.get<std::string>("quantizer");
    cfg.anisotropic_pq = (cfg.quantizer_type == "anisotropic-pq");
    {
        const std::string gbm = p.get<std::string>("graph-build-metric");
        if (gbm == "pq")      cfg.graph_build_metric = sextant::GraphBuildMetric::PqConstruct;
        else if (gbm == "fp16") cfg.graph_build_metric = sextant::GraphBuildMetric::Fp16;
        else if (gbm == "fp32") cfg.graph_build_metric = sextant::GraphBuildMetric::Fp32;
        else throw std::runtime_error("--graph-build-metric must be pq, fp16, or fp32");
    }
    if (p.exist("prq-nsplits")) cfg.prq_nsplits = p.get<uint32_t>("prq-nsplits");
    if (p.exist("prq-beam-size")) cfg.prq_beam_size = p.get<uint32_t>("prq-beam-size");
    if (p.exist("prq-encode-mode")) cfg.prq_encode_mode = p.get<std::string>("prq-encode-mode");
    if (p.exist("prq-icm-iters")) cfg.prq_icm_iters = p.get<uint32_t>("prq-icm-iters");
    if (p.exist("prq-ils-iters")) cfg.prq_ils_iters = p.get<uint32_t>("prq-ils-iters");
    if (p.exist("prq-ils-perturb")) cfg.prq_ils_perturb = p.get<uint32_t>("prq-ils-perturb");
    if (p.exist("prq-lsq-train-iters")) cfg.prq_lsq_train_iters = p.get<uint32_t>("prq-lsq-train-iters");
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
    try {
        if (p.exist("partition-balance-factor")) {
            cfg.partition_balance_factor = p.get<float>("partition-balance-factor");
        }
    } catch (...) {}
    try {
        if (p.exist("closure-epsilon")) {
            cfg.closure_epsilon = p.get<float>("closure-epsilon");
        }
    } catch (...) {}
    try {
        if (p.exist("closure-f-target")) {
            cfg.closure_f_target = p.get<float>("closure-f-target");
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

    // Metric recommendation. Heuristic signal: L2 norm distribution over
    // the sampled vectors. Two conditions indicate L2-normalized data where
    // IP and L2sq are rank-equivalent (||q-x||² = 2 - 2<q,x>) and IP is
    // cheaper per eval:
    //   1. mean norm ≈ 1   (within ±5%)
    //   2. tight norms      (cv < 0.05)
    // Both conditions are required: SIFT-1M has cv≈0.001 (very tight) but
    // mean=508 — definitely NOT normalized; IP collapses recall there.
    // Non-normalized data with wide norms (cv > 0.10) requires L2sq.
    if (diag.norm_mean > 0.0) {
        std::cout << "  L2 norm:              mean=" << std::setprecision(4)
                  << diag.norm_mean << " cv=" << diag.norm_cv
                  << " range=[" << diag.norm_min << ", " << diag.norm_max
                  << "]\n";
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

    // Metric note based on the norm distribution heuristic. Picking the
    // wrong metric silently destroys recall on non-normalized data
    // (SIFT-1M: 0.99 → 0.66). We only emit a note when there's something
    // actionable: mismatch between the user's --metric choice and what the
    // heuristic suggests, OR when the signal is borderline so the user
    // knows it's weak. "Normalized" requires BOTH mean≈1 AND cv<0.05.
    if (diag.norm_mean > 0.0) {
        const bool near_unit = (std::abs(diag.norm_mean - 1.0) <= 0.05);
        const bool tight = (diag.norm_cv < 0.05);
        const bool normalized = near_unit && tight;
        // Borderline = mean near 1 but cv loose, OR cv tight but mean far from 1.
        // If both signals point the same direction (mean near 1 AND cv tight,
        // or mean far from 1 AND cv loose), the classification is confident.
        const bool borderline = !normalized &&
            ((near_unit && diag.norm_cv >= 0.05 && diag.norm_cv <= 0.20) ||
             (tight && !near_unit && std::abs(diag.norm_mean - 1.0) <= 0.30));
        if (normalized && metric_str == "l2sq") {
            std::cout << "\n  note: data appears L2-normalized (norm_mean="
                      << std::setprecision(4) << diag.norm_mean
                      << ", norm_cv=" << diag.norm_cv
                      << "). IP and L2sq are rank-equivalent here; IP is "
                      << "cheaper per eval. --metric ip may be faster.\n";
        } else if (!normalized && !borderline && metric_str == "ip") {
            std::cout << "\n  [warning] data is NOT normalized (norm_mean="
                      << std::setprecision(4) << diag.norm_mean
                      << ", norm_cv=" << diag.norm_cv
                      << "). IP misranks on non-normalized data (recall "
                      << "collapse). Recommend --metric l2sq.\n";
        } else if (borderline) {
            std::cout << "\n  note: norm_mean=" << std::setprecision(4)
                      << diag.norm_mean
                      << ", norm_cv=" << diag.norm_cv
                      << " - borderline (not clearly normalized). The metric "
                      << "choice may affect recall; benchmark both if recall "
                      << "is critical.\n";
        }
    }
}

}  // namespace sextant_cli
