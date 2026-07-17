// analyze — dataset-adaptive parameter advisor.
//
// Runs Engine::estimate_config on the input dataset: reservoir-samples the
// data, trains PQ, measures LID, builds mini-indices at candidate (R, alpha)
// values, and reports the optimal build configuration (R, alpha, pq_m,
// pq_bits) with the measured signals that drove each choice.
//
// This is a read-only advisory tool — it does NOT write any index files.
// Use `sextant build` with the recommended parameters to create an index.
//
// Usage:
//   sextant analyze --input data.fbin \
//       [--metric l2sq] [--proximity-target 0.95] [--recall-target 0.99] \
//       [--pq-m 0] [--pq-bits 0] [--R 0] [--alpha 0] [--threads 0] \
//       [--log-level info]

#include "engine/fbin_source.hpp"
#include "sextant/config.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <cmdline/cmdline.h>
#include <spdlog/spdlog.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

namespace {

int cmd_analyze(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("input", 0, "Input .fbin file", true);
    p.add<std::string>("metric", 0, "l2sq or ip", false, "l2sq");
    p.add<float>("proximity-target", 0,
                 "Target proximity in-band (0-1): fraction of results at/inside "
                 "the true k-th NN distance. 0 = default (0.95). Drives R "
                 "selection via OPT-SNG.",
                 false, 0.0f);
    p.add<float>("recall-target", 0,
                 "Target recall@k (alternative to proximity-target). "
                 "0 = not used. If both set, proximity-target wins.",
                 false, 0.0f);
    p.add<uint16_t>("R", 0, "Graph degree (auto if 0)", false, 0);
    p.add<float>("alpha", 0, "Vamana prune threshold (auto if 0)", false, 0.0f);
    p.add<uint16_t>("pq-m", 0, "PQ segments (auto if 0)", false, 0);
    p.add<std::string>("pq-bits", 0,
                        "PQ bits: 4, 8, or auto (0)", false, "auto");
    p.add<float>("pq-max-distortion", 0,
                 "Max PQ distortion for auto (m,bits) selection "
                 "(0 = default 0.05)", false, 0.0f);
    p.add<uint32_t>("threads", 0,
                     "Threads for mini-builds (0 = hardware_concurrency)",
                     false, 0);
    p.add<std::string>("log-level", 0,
                        "Log level: debug, info, warn, error",
                        false, "info");
    p.parse_check(argc, argv);

    const std::string input = p.get<std::string>("input");
    const std::string log_level = p.get<std::string>("log-level");
    if (log_level == "debug") sextant::set_log_level(sextant::LogLevel::Debug);
    else if (log_level == "info") sextant::set_log_level(sextant::LogLevel::Info);
    else if (log_level == "warn") sextant::set_log_level(sextant::LogLevel::Warn);
    else sextant::set_log_level(sextant::LogLevel::Error);

    const std::string metric_str = p.get<std::string>("metric");
    sextant::MetricKind metric =
        (metric_str == "ip") ? sextant::MetricKind::InnerProduct
                              : sextant::MetricKind::L2Sq;

    // Open the source to get N and dim.
    sextant::FbinSource source(input);
    const uint64_t n = source.count();
    const sextant::Dim dim = source.dim();
    if (n == 0 || dim == 0) {
        std::cerr << "sextant analyze: empty or invalid source '" << input
                  << "'\n";
        return 1;
    }

    // Build the override config from CLI args.
    sextant::BuildConfig cfg;
    cfg.metric = metric;
    cfg.R = p.get<uint16_t>("R");
    cfg.alpha = p.get<float>("alpha");
    cfg.pq_m = p.get<uint16_t>("pq-m");
    cfg.proximity_target = p.get<float>("proximity-target");
    cfg.recall_target = p.get<float>("recall-target");
    cfg.pq_max_distortion = p.get<float>("pq-max-distortion");
    cfg.num_threads = p.get<uint32_t>("threads");
    {
        const std::string b = p.get<std::string>("pq-bits");
        if (b == "auto" || b == "0") cfg.pq_bits = 0;
        else if (b == "4") cfg.pq_bits = 4;
        else if (b == "8") cfg.pq_bits = 8;
        else {
            std::cerr << "--pq-bits must be 4, 8, or auto\n";
            return 1;
        }
    }

    // Run estimate_config.
    sextant::Engine engine;
    const auto params = engine.estimate_config(source, cfg);

    // --- Pretty-print results ---
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
    std::cout << "  L_build:        " << params.L_build << "\n";
    std::cout << "  max_occlusion:  " << params.max_occlusion << "\n";
    std::cout << "  closure_factor: " << std::setprecision(4)
              << params.closure_factor << "\n";
    std::cout << "  K (partitions): " << params.K << "\n\n";

    // Recommended build command.
    std::cout << "─── Build Command ───\n";
    std::cout << "  sextant build --input " << input
              << " --index <INDEX> --metric " << metric_str
              << " --R " << params.R
              << " --alpha " << std::setprecision(1) << params.alpha
              << " --pq-m " << params.pq_m
              << " --pq-bits " << static_cast<int>(params.pq_bits);
    if (cfg.num_threads > 0)
        std::cout << " --threads " << cfg.num_threads;
    std::cout << "\n";

    return 0;
}

}  // namespace

int run_analyze(int argc, char* argv[]) {
    return cmd_analyze(argc, argv);
}
