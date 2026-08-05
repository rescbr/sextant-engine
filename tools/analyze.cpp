// analyze — dataset-adaptive parameter advisor.
//
// Runs Engine::estimate_config on the input dataset: reservoir-samples the
// data, trains PQ, measures LID, builds mini-indices at candidate (R, alpha)
// values, and reports the optimal build configuration (R, alpha, pq_m,
// pq_bits) with the measured signals that drove each choice.
//
// This is a read-only advisory tool — it does NOT write any index files.
// Use `sextant build` with the recommended parameters to create an index,
// or `sextant autobuild` to estimate + build in one step.
//
// Shares the common flag parser with build/autobuild (tools/shared_cli.hpp).

#include "fbin_source.hpp"
#include "shared_cli.hpp"
#include "sextant/config.hpp"
#include "sextant/estimator.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <cmdline/cmdline.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string>

namespace {

int cmd_analyze(int argc, char* argv[]) {
    using namespace sextant_cli;
    cmdline::parser p;
    add_common_flags(p);
    add_mode_extras(p, Mode::Analyze);
    p.parse_check(argc, argv);
    apply_log_level(p);

    const std::string input = p.get<std::string>("input");

    sextant::FbinSource source(input);
    const uint64_t n = source.count();
    const sextant::Dim dim = source.dim();
    if (n == 0 || dim == 0) {
        std::cerr << "sextant analyze: empty or invalid source '" << input
                  << "'\n";
        return 1;
    }

    sextant::BuildConfig cfg = build_config_from_parser(p);
    cfg.proximity_target = p.get<float>("proximity-target");
    cfg.recall_target    = p.get<float>("recall-target");
    cfg.build_ram_budget = p.get<uint64_t>("build-ram");
    // (cfg.sharded_graph and cfg.merged_graph are set in build_config_from_parser
    // from the --ivf / --graph flags, with mutual-exclusion validation.)

    sextant::Estimator estimator;
    const auto est = estimator.estimate_config(source, cfg);
    print_analysis_(source, input, cfg, est.params, est.diag);
    return 0;
}

}  // namespace

int run_analyze(int argc, char* argv[]) {
    return cmd_analyze(argc, argv);
}
