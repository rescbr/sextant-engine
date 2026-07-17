// sextant CLI — build / search / insert.
//
// Uses the vendored cmdline.h (header-only) for argument parsing.
// Commands:
//   sextant build   — build an index from a .fbin file
//   sextant search  — search an index with query vectors
//   sextant insert  — insert a single vector into an index
//
// The CLI catches all exceptions, prints to stderr, returns non-zero.

#include "engine/fbin_source.hpp"
#include "fbin_io.hpp"
#include "sextant/config.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <cmdline/cmdline.h>

#include <spdlog/spdlog.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace sextant::fbin_io;  // FbinHeader, read_fbin_header, read_fbin_vector, l2sq_distance

// ---------------------------------------------------------------------------
// Command handlers. Each returns a process exit code.
// ---------------------------------------------------------------------------

int cmd_build(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("input", 0, "Input .fbin/.ibin/.bbin file", true);
    p.add<std::string>("index", 0, "Index name/path prefix", true);
    p.add<uint16_t>("R", 0, "Graph degree (auto if 0)", false, 0);
    p.add<uint16_t>("L", 0, "Beam width (auto if 0)", false, 0);
    p.add<float>("alpha", 0, "Vamana prune threshold (default 1.2).", false, 1.2f);
    p.add<std::string>("build-mode", 0, "Build mode (deprecated, always sdc). "
                                      "Accepted for backward compat: sdc only.", false, "sdc");
    p.add<uint16_t>("pq-m", 0, "PQ segments — REQUIRED for build (e.g. 96). "
                         "Run `sextant analyze` first if unsure what to pick.", false, 0);
    p.add<std::string>("pq-bits", 0, "PQ bits — REQUIRED for build: 4 or 8 (e.g. 8). "
                                  "Run `sextant analyze` first if unsure what to pick.", false, "auto");
    p.add<float>("pq-max-distortion", 0,
                 "Max PQ distortion (median |1 - pq_dist/true_dist|) for auto (m,bits) selection "
                 "(0 = default 0.05). Filters configs; the min-cost eligible config is selected. "
                 "Lower (e.g. 0.03) forces higher-fidelity PQ; higher (e.g. 0.10-0.15) admits "
                 "aggressive low-m configs for high-dimensional data.",
                 false, 0.0f);
    p.add<std::string>("metric", 0, "l2sq or ip", false, "l2sq");
    p.add<uint32_t>("threads", 0, "Build threads (auto if 0)", false, 0);
    p.add<uint64_t>("build-ram", 0,
                     "Build RAM budget in bytes (forces partitioning if small)",
                     false, 0);
    p.add<uint32_t>("inline-pq", 0,
                     "Neighbor PQ codes inlined per node (0=compact, R=all; deprecated)",
                     false, 0);
    p.add<uint32_t>("max-occlusion", 0,
                     "RobustPrune candidate cap (auto if 0)",
                     false, 0);
    p.add<std::string>("log-level", 0,
                       "Log level: debug, info, warn, error",
                       false, "info");
    p.add("explain", 0, "Print resolved params and exit (dry-run). When pq_m or "
                        "pq_bits is auto, reads a random sample and runs the "
                        "full PQ probe to show the actual selection.");
    p.add<uint32_t>("probe-sample", 0,
                    "Sample size for --explain PQ probe (default 20000). "
                    "Reads this many random vectors via seek; no full scan.",
                    false, 20000);
    p.parse_check(argc, argv);

    // Set log level.
    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") sextant::set_log_level(sextant::LogLevel::Debug);
        else if (lvl == "warn") sextant::set_log_level(sextant::LogLevel::Warn);
        else if (lvl == "error") sextant::set_log_level(sextant::LogLevel::Error);
    }

    const std::string input = p.get<std::string>("input");
    const std::string index = p.get<std::string>("index");

    sextant::FbinSource source(input);
    const uint64_t n = source.count();
    const sextant::Dim dim = source.dim();
    if (n == 0 || dim == 0) {
        std::cerr << "sextant build: empty or invalid source '" << input
                  << "'\n";
        return 1;
    }

    sextant::BuildConfig cfg;
    cfg.R = p.get<uint16_t>("R");
    cfg.L = p.get<uint16_t>("L");
    cfg.alpha = p.get<float>("alpha");
    // build-mode: deprecated, always SDC. Accept "sdc" for backward compat;
    // error on "adc" (removed — FP16 prune hybrid made it redundant).
    {
        const std::string bm = p.get<std::string>("build-mode");
        if (bm == "sdc" || bm == "0") {
            cfg.build_mode = sextant::BuildMode::SDC;
        } else if (bm == "adc" || bm == "1") {
            std::fprintf(stderr, "--build-mode adc is no longer supported (FP16 prune "
                                 "hybrid made it redundant). Use sdc.\n");
            return 1;
        } else {
            std::fprintf(stderr, "--build-mode must be sdc (got '%s')\n", bm.c_str());
            return 1;
        }
    }
    cfg.pq_m = p.get<uint16_t>("pq-m");
    // pq-bits: "auto" (default) → 0 (resolved by global probe in pass1), else 4|8.
    {
        const std::string b = p.get<std::string>("pq-bits");
        if (b == "auto" || b == "0") {
            cfg.pq_bits = 0;
        } else if (b == "4") {
            cfg.pq_bits = 4;
        } else if (b == "8") {
            cfg.pq_bits = 8;
        } else {
            std::fprintf(stderr, "--pq-bits must be 4, 8, or auto (got '%s')\n", b.c_str());
            return 1;
        }
    }
    cfg.pq_max_distortion = p.get<float>("pq-max-distortion");
    cfg.num_threads = p.get<uint32_t>("threads");
    cfg.build_ram_budget = p.get<uint64_t>("build-ram");
    cfg.inline_pq_count = p.get<uint32_t>("inline-pq");
    cfg.max_occlusion = p.get<uint32_t>("max-occlusion");
    const std::string metric = p.get<std::string>("metric");
    cfg.metric = (metric == "ip") ? sextant::MetricKind::InnerProduct
                                  : sextant::MetricKind::L2Sq;

    if (p.exist("explain")) {
        // If ANY of (R, alpha, pq_m, pq_bits) is auto, run the full
        // sample-driven estimate_config (it does PQ probe + alpha sweep +
        // R prediction via mini-builds). If ALL are locked, use the fast
        // resolve_params path (no sampling needed).
        const bool any_auto = (cfg.R == 0) || (cfg.alpha == 0.0f) ||
                              (cfg.pq_m == 0) || (cfg.pq_bits == 0);

        sextant::ResolvedParams resolved;
        bool used_estimate_config = false;
        if (any_auto) {
            sextant::Engine engine;
            resolved = engine.estimate_config(source, cfg);
            used_estimate_config = true;
        } else {
            resolved = sextant::resolve_params(n, dim, cfg);
        }

        std::cout << "input:      " << input << "\n"
                  << "index:      " << index << "\n"
                  << "n_vectors:  " << n << "\n"
                  << "dim:        " << dim << "\n"
                  << "R:          " << resolved.R
                  << (cfg.R != 0 ? "  [locked]\n" : "  [estimated]\n")
                  << "L:          " << resolved.L << "\n"
                  << "L_build:    " << resolved.L_build << "\n"
                  << "alpha:      " << resolved.alpha
                  << (cfg.alpha != 0.0f ? "  [locked]\n" : "  [estimated]\n");

        if (used_estimate_config) {
            std::cout << "pq_m:       " << resolved.pq_m
                      << (cfg.pq_m != 0 ? "  [locked]\n" : "  [estimated]\n");
            std::cout << "pq_bits:    " << static_cast<int>(resolved.pq_bits)
                      << (cfg.pq_bits != 0 ? "  [locked]\n" : "  [estimated]\n");
            // Show the measured signals that drove the estimation.
            std::cout << "\n─── Measured signals (estimate_config) ───\n";
            std::cout << "  median LID:         " << std::fixed
                      << std::setprecision(2) << resolved.measured_median_lid
                      << "\n";
            std::cout << "  avg degree (R̄):     " << std::setprecision(2)
                      << resolved.measured_avg_degree << "\n";
            std::cout << "  clustering coeff:   " << std::setprecision(4)
                      << resolved.measured_clustering << "\n";
            std::cout << "  dead-end fraction:  " << std::setprecision(4)
                      << resolved.measured_dead_end_frac << "\n";
            std::cout << "  closure_factor:     " << std::setprecision(4)
                      << resolved.closure_factor << "\n";
        } else {
            std::cout << "pq_m:       " << resolved.pq_m << "\n"
                      << "pq_bits:    " << static_cast<int>(resolved.pq_bits)
                      << "\n";
            std::cout << "pq_max_distortion:  " << resolved.pq_max_distortion
                      << " (bound; not used — m and bits are explicit)\n";
        }
        std::cout << "max_occlusion:  " << resolved.max_occlusion << "\n"
                  << "inline_pq:  " << resolved.inline_pq_count << "\n"
                  << "threads:    " << resolved.num_threads << "\n"
                  << "build_ram:  " << resolved.build_ram_budget
                  << " bytes\n"
                  << "K:          " << resolved.K << "\n"
                  << "metric:     " << metric << "\n";
        return 0;
    }

    sextant::Engine engine;
    sextant::BuildResult result = engine.build(source, index, cfg);
    std::cout << "built index '" << index << "': n=" << result.n_vectors
              << " dim=" << result.dim
              << " R=" << result.R
              << " L_build=" << result.L_build
              << " pq_m=" << static_cast<int>(result.pq_m)
              << " pq_bits=" << static_cast<int>(result.pq_bits)
              << " in " << result.build_time_sec << "s\n";
    return 0;
}

int cmd_search(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("index", 0, "Index name/path prefix", true);
    p.add<std::string>("query", 0, "Query .fbin file", true);
    p.add<uint32_t>("k", 0, "Number of results", false, 10);
    p.add<uint32_t>("L", 0, "Search beam width", false, 200);
    p.add<uint32_t>("rerank", 0, "Rerank factor", false, 10);
    p.add<uint32_t>("threads", 0, "Search threads (0 = 1, serial)", false, 1);
    p.add<std::string>("output", 0, "Output file (default: stdout)", false, "");
    p.add<std::string>(
        "base-data", 0,
        "Original base .fbin for rerank (defaults to none)", false, "");
    p.add<uint64_t>("cache-size", 0,
                     "Search LRU cache size in bytes (0 = auto)", false, 0);
    p.add("no-cache-rebalance", 0,
          "Disable adaptive graph/code cache rebalancing (default: enabled in paged mode)");
    p.add<std::string>("log-level", 0,
                       "Log level: debug, info, warn, error", false, "info");
    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") sextant::set_log_level(sextant::LogLevel::Debug);
        else if (lvl == "warn") sextant::set_log_level(sextant::LogLevel::Warn);
        else if (lvl == "error") sextant::set_log_level(sextant::LogLevel::Error);
    }

    const std::string index = p.get<std::string>("index");
    const std::string query_path = p.get<std::string>("query");
    const uint32_t k = p.get<uint32_t>("k");
    const uint32_t L = p.get<uint32_t>("L");
    const uint32_t rerank = p.get<uint32_t>("rerank");
    const std::string output = p.get<std::string>("output");
    const std::string base_data = p.get<std::string>("base-data");
    const uint32_t num_threads = p.get<uint32_t>("threads");

    sextant::Engine engine;
    engine.set_cache_size(p.get<uint64_t>("cache-size"));
    if (p.exist("no-cache-rebalance")) {
        engine.set_cache_rebalance_enabled(false);
    }
    engine.open(index);

    // Read the query file header.
    FbinHeader qh;
    if (!read_fbin_header(query_path, qh) || qh.dim != engine.dim()) {
        std::cerr << "sextant search: invalid query file '" << query_path
                  << "' (dim=" << qh.dim << ", expected " << engine.dim()
                  << ")\n";
        return 1;
    }
    if (qh.n == 0) {
        std::cerr << "sextant search: query file is empty\n";
        return 1;
    }

    // For rerank: open the base .fbin and verify dim.
    const bool do_rerank = (rerank > 1) && !base_data.empty();
    FbinHeader bh{};
    if (do_rerank) {
        if (!read_fbin_header(base_data, bh) || bh.dim != engine.dim()) {
            std::cerr << "sextant search: invalid base-data file '" << base_data
                      << "' for rerank; skipping rerank\n";
        }
    }
    const bool rerank_ok = do_rerank && bh.dim == engine.dim();

    std::ofstream out_file;
    std::ostream* out = &std::cout;
    if (!output.empty()) {
        out_file.open(output);
        if (!out_file) {
            std::cerr << "sextant search: cannot open output '" << output
                      << "'\n";
            return 1;
        }
        out = &out_file;
    }

    const uint32_t dim = engine.dim();
    const uint32_t fetch_k =
        rerank_ok ? std::min<uint32_t>(k * rerank, engine.count()) : k;

    if (num_threads <= 1) {
        // Serial path (unchanged): stream queries one at a time.
        std::ifstream qf(query_path, std::ios::binary);
        qf.seekg(8);  // skip header
        std::vector<float> qvec(dim);

        for (uint32_t qi = 0; qi < qh.n; qi++) {
            qf.read(reinterpret_cast<char*>(qvec.data()),
                    static_cast<std::streamsize>(dim * sizeof(float)));
            if (!qf.good()) {
                std::cerr << "sextant search: short read on query " << qi << "\n";
                break;
            }

            sextant::SearchConfig scfg;
            scfg.k = fetch_k;
            scfg.L_search = L;
            scfg.rerank_factor = rerank;
            auto results = engine.search(qvec.data(), scfg.k, scfg);

            if (rerank_ok && !results.empty()) {
                // Fetch actual vectors and re-sort by exact L2-sq distance.
                std::vector<std::pair<float, sextant::RowId>> scored;
                scored.reserve(results.size());
                std::vector<float> base_vec(dim);
                for (const auto& c : results) {
                    if (c.row_id < 0 ||
                        static_cast<uint64_t>(c.row_id) >= bh.n) {
                        continue;
                    }
                    if (!read_fbin_vector(base_data, dim,
                                          static_cast<uint64_t>(c.row_id),
                                          base_vec)) {
                        scored.emplace_back(c.dist, c.row_id);
                        continue;
                    }
                    scored.emplace_back(
                        l2sq_distance(qvec.data(), base_vec.data(), dim),
                        c.row_id);
                }
                std::sort(scored.begin(), scored.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
                const uint32_t topk = std::min<uint32_t>(k, scored.size());
                for (uint32_t i = 0; i < topk; i++) {
                    (*out) << qi << "\t" << scored[i].second << "\t"
                           << scored[i].first << "\n";
                }
            } else {
                const uint32_t topk = std::min<uint32_t>(k, results.size());
                for (uint32_t i = 0; i < topk; i++) {
                    (*out) << qi << "\t" << results[i].row_id << "\t"
                           << results[i].dist << "\n";
                }
            }
        }
    } else {
        // Parallel path: read all queries into RAM, dispatch across threads.
        std::vector<float> queries(static_cast<size_t>(qh.n) * dim);
        {
            std::ifstream qf(query_path, std::ios::binary);
            qf.seekg(8);
            qf.read(reinterpret_cast<char*>(queries.data()),
                    static_cast<std::streamsize>(queries.size() *
                                                 sizeof(float)));
            if (!qf) {
                std::cerr << "sextant search: short read on query file\n";
                return 1;
            }
        }

        const uint32_t n_threads =
            std::max(1u, std::min(num_threads, qh.n));
        std::vector<std::string> formatted_results(qh.n);
        std::atomic<uint32_t> next{0};
        std::vector<std::thread> workers;
        workers.reserve(n_threads);

        for (uint32_t t = 0; t < n_threads; t++) {
            workers.emplace_back([&]() {
                // Thread-local scratch.
                std::vector<float> base_vec(dim);
                uint32_t qi;
                while ((qi = next.fetch_add(1, std::memory_order_relaxed)) < qh.n) {
                    const float* q = &queries[static_cast<size_t>(qi) * dim];

                    sextant::SearchConfig scfg;
                    scfg.k = fetch_k;
                    scfg.L_search = L;
                    scfg.rerank_factor = rerank;
                    auto results = engine.search(q, scfg.k, scfg);

                    std::string buf;
                    if (rerank_ok && !results.empty()) {
                        std::vector<std::pair<float, sextant::RowId>> scored;
                        scored.reserve(results.size());
                        for (const auto& c : results) {
                            if (c.row_id < 0 ||
                                static_cast<uint64_t>(c.row_id) >= bh.n) {
                                continue;
                            }
                            if (!read_fbin_vector(base_data, dim,
                                                  static_cast<uint64_t>(c.row_id),
                                                  base_vec)) {
                                scored.emplace_back(c.dist, c.row_id);
                                continue;
                            }
                            scored.emplace_back(
                                l2sq_distance(q, base_vec.data(), dim),
                                c.row_id);
                        }
                        std::sort(scored.begin(), scored.end(),
                                  [](const auto& a, const auto& b) {
                                      return a.first < b.first;
                                  });
                        const uint32_t topk = std::min<uint32_t>(k, scored.size());
                        for (uint32_t i = 0; i < topk; i++) {
                            buf += std::to_string(qi);
                            buf += "\t";
                            buf += std::to_string(scored[i].second);
                            buf += "\t";
                            buf += std::to_string(scored[i].first);
                            buf += "\n";
                        }
                    } else {
                        const uint32_t topk = std::min<uint32_t>(k, results.size());
                        for (uint32_t i = 0; i < topk; i++) {
                            buf += std::to_string(qi);
                            buf += "\t";
                            buf += std::to_string(results[i].row_id);
                            buf += "\t";
                            buf += std::to_string(results[i].dist);
                            buf += "\n";
                        }
                    }
                    formatted_results[qi] = std::move(buf);
                }
            });
        }
        for (auto& w : workers) w.join();

        // Emit in query order.
        for (uint32_t qi = 0; qi < qh.n; qi++) {
            (*out) << formatted_results[qi];
        }
    }

    return 0;
}

int cmd_insert(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("index", 0, "Index name/path prefix", true);
    p.add<std::string>("vector", 0, "Single-vector .fbin file", true);
    p.add<int64_t>("row-id", 0, "Row ID for the inserted vector", true);
    p.parse_check(argc, argv);

    const std::string index = p.get<std::string>("index");
    const std::string vector_path = p.get<std::string>("vector");
    const int64_t row_id = p.get<int64_t>("row-id");

    sextant::Engine engine;
    engine.open(index);

    FbinHeader vh;
    if (!read_fbin_header(vector_path, vh) || vh.dim != engine.dim() ||
        vh.n != 1) {
        std::cerr << "sextant insert: need a single-vector .fbin with dim="
                  << engine.dim() << " (got n=" << vh.n << " dim=" << vh.dim
                  << ")\n";
        return 1;
    }

    std::vector<float> vec(vh.dim);
    std::ifstream vf(vector_path, std::ios::binary);
    vf.seekg(8);
    vf.read(reinterpret_cast<char*>(vec.data()),
            static_cast<std::streamsize>(vh.dim * sizeof(float)));
    if (!vf.good()) {
        std::cerr << "sextant insert: failed to read vector\n";
        return 1;
    }

    engine.insert(vec.data(), vh.dim, static_cast<sextant::RowId>(row_id));
    engine.flush();
    std::cout << "inserted row_id=" << row_id
              << " (count now " << engine.count() << ")\n";
    return 0;
}

void print_usage() {
    std::cerr << "Usage: sextant <command> [options]\n"
              << "Commands:\n"
              << "  build    Build an index from a .fbin file\n"
              << "  search   Search an index with query vectors\n"
              << "  insert   Insert a single vector into an index\n"
              << "  analyze  PQ sensitivity advisory (pre-build mini-graph sweep)\n";
}

}  // namespace

/// Implemented in tools/analyze.cpp.
int run_analyze(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    sextant::init_logging();

    if (argc < 2) {
        print_usage();
        return 1;
    }

    const std::string cmd = argv[1];

    // Rebuild argv for the sub-command parser (drop the sub-command token).
    std::vector<char*> sub_argv;
    sub_argv.push_back(argv[0]);
    for (int i = 2; i < argc; i++) sub_argv.push_back(argv[i]);
    int sub_argc = static_cast<int>(sub_argv.size());

    try {
        if (cmd == "build") {
            return cmd_build(sub_argc, sub_argv.data());
        } else if (cmd == "search") {
            return cmd_search(sub_argc, sub_argv.data());
        } else if (cmd == "insert") {
            return cmd_insert(sub_argc, sub_argv.data());
        } else if (cmd == "analyze") {
            return run_analyze(sub_argc, sub_argv.data());
        } else if (cmd == "--help" || cmd == "-h" || cmd == "help") {
            print_usage();
            return 0;
        } else {
            std::cerr << "sextant: unknown command '" << cmd << "'\n";
            print_usage();
            return 1;
        }
    } catch (const sextant::Error& e) {
        std::cerr << "sextant " << cmd << ": " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "sextant " << cmd << ": " << e.what() << "\n";
        return 1;
    }
}
