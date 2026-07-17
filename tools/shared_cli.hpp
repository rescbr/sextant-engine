#pragma once

/// Shared flag definitions and pretty-printers for the build/analyze/autobuild
/// commands. All three accept a common base of override knobs so new params can
/// never be added to one command and missed in another. Mode-specific extras
/// are added per-command after calling add_common_flags.

#include "sextant/config.hpp"
#include "sextant/engine.hpp"
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
            "If both set, proximity-target wins.",
            false, 0.0f);
    }
    if (mode == Mode::Build || mode == Mode::Autobuild) {
        p.add<uint64_t>("build-ram", 0,
            "Build RAM budget in bytes (forces partitioning if small). "
            "0 = auto (50% of physical RAM).",
            false, 0);
        p.add<uint32_t>("prune-candidate-cap", 0,
            "Cap on the robust-prune candidate pool (max-occlusion in "
            "DiskANN). 0 = auto (= max(L_build, R+1)).",
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
    cfg.num_threads       = p.get<uint32_t>("threads");
    {
        const std::string m = p.get<std::string>("metric");
        cfg.metric = (m == "ip") ? sextant::MetricKind::InnerProduct
                                  : sextant::MetricKind::L2Sq;
    }
    cfg.build_mode = sextant::BuildMode::SDC;  // SDC only; ADC removed
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
                            const sextant::ResolvedParams& params) {
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
              << std::setprecision(2) << params.measured_median_lid << "\n";
    std::cout << "  avg degree (R̄):       " << std::setprecision(2)
              << params.measured_avg_degree << "\n";
    std::cout << "  clustering coeff:     " << std::setprecision(4)
              << params.measured_clustering << "\n";
    std::cout << "  dead-end fraction:    " << std::setprecision(4)
              << params.measured_dead_end_frac << "\n\n";

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
    std::cout << "  K (partitions): " << params.K << "\n\n";

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
}

}  // namespace sextant_cli
