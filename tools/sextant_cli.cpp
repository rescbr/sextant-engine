// sextant CLI — build / autobuild / search / insert / analyze.
//
// Uses the vendored cmdline.h (header-only) for argument parsing.
// Commands:
//   sextant build      — build an index from a .fbin file (explicit params)
//   sextant autobuild  — estimate_config then build, in one process
//   sextant search     — search an index with query vectors
//   sextant insert     — insert a single vector into an index
//   sextant analyze    — read-only dataset-adaptive parameter advisory
//
// build/autobuild/analyze share a common flag parser (tools/shared_cli.hpp).
//
// The CLI catches all exceptions, prints to stderr, returns non-zero.

#include "engine/fbin_source.hpp"
#include "fbin_io.hpp"
#include "shared_cli.hpp"
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
    using namespace sextant_cli;
    cmdline::parser p;
    add_common_flags(p);
    add_mode_extras(p, Mode::Build);
    p.parse_check(argc, argv);
    apply_log_level(p);

    const std::string input = p.get<std::string>("input");
    const std::string index = p.get<std::string>("index");
    if (index.empty()) {
        std::cerr << "sextant build: --index is required\n";
        return 1;
    }

    sextant::FbinSource source(input);
    const uint64_t n = source.count();
    const sextant::Dim dim = source.dim();
    if (n == 0 || dim == 0) {
        std::cerr << "sextant build: empty or invalid source '" << input
                  << "'\n";
        return 1;
    }

    sextant::BuildConfig cfg = build_config_from_parser(p);
    cfg.build_ram_budget = p.get<uint64_t>("build-ram");
    cfg.max_occlusion    = p.get<uint32_t>("prune-candidate-cap");

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

int cmd_autobuild(int argc, char* argv[]) {
    using namespace sextant_cli;
    cmdline::parser p;
    add_common_flags(p);
    add_mode_extras(p, Mode::Autobuild);
    p.parse_check(argc, argv);
    apply_log_level(p);

    const std::string input = p.get<std::string>("input");
    const std::string index = p.get<std::string>("index");
    if (index.empty()) {
        std::cerr << "sextant autobuild: --index is required\n";
        return 1;
    }

    sextant::FbinSource source(input);
    const uint64_t n = source.count();
    const sextant::Dim dim = source.dim();
    if (n == 0 || dim == 0) {
        std::cerr << "sextant autobuild: empty or invalid source '" << input
                  << "'\n";
        return 1;
    }

    sextant::BuildConfig cfg = build_config_from_parser(p);
    cfg.proximity_target = p.get<float>("proximity-target");
    cfg.recall_target    = p.get<float>("recall-target");
    cfg.build_ram_budget = p.get<uint64_t>("build-ram");
    cfg.max_occlusion    = p.get<uint32_t>("prune-candidate-cap");

    sextant::Engine engine;
    // estimate_config handles all auto knobs; locked ones override.
    const sextant::ResolvedParams params = engine.estimate_config(source, cfg);

    // Print the analysis (shared pretty-print with analyze).
    print_analysis_(source, input, cfg, params);

    // Build with the resolved params (in-process, no string round-trip).
    const sextant::BuildResult result = engine.build(source, index, params);
    std::cout << "\n═══ Build Result ═══\n";
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
    p.add<uint32_t>("topk", 0,
        "Number of nearest neighbors to return per query (k in ANN literature)",
        false, 10);
    p.add<uint32_t>("search-beam-width", 0,
        "Search-time beam width (L in Vamana literature). Higher = more accurate, "
        "slower. Must be >= topk.",
        false, 200);
    p.add<uint32_t>("rerank", 0, "Rerank factor (0/1 = no rerank)", false, 10);
    p.add<uint32_t>("threads", 0,
        "Search threads (0 = hardware_concurrency; 1 = force serial path)",
        false, 0);
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
    const uint32_t k = p.get<uint32_t>("topk");
    const uint32_t L = p.get<uint32_t>("search-beam-width");
    const uint32_t rerank = p.get<uint32_t>("rerank");
    const std::string output = p.get<std::string>("output");
    const std::string base_data = p.get<std::string>("base-data");
    uint32_t num_threads = p.get<uint32_t>("threads");
    // 0 = hardware_concurrency (parallel by default — ANN search is
    // embarrassingly parallel across queries). 1 explicitly selects the
    // serial code path (used for deterministic-output / single-query debug).
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

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
              << "  build      Build an index from a .fbin file (explicit params).\n"
              << "             Common flags: --input --index --max-node-neighbors (R)\n"
              << "             --beam-width-ceiling (L) --prune-threshold (alpha)\n"
              << "             --pq-segments (m) --pq-bits --threads --metric\n"
              << "             --build-ram --prune-candidate-cap --log-level\n"
              << "  autobuild  Estimate config (analyze) then build, in-process.\n"
              << "             Accepts all build flags plus estimate knobs:\n"
              << "             --proximity-target --recall-target\n"
              << "  analyze    Read-only dataset-adaptive parameter advisory.\n"
              << "             Flags: --input --metric --proximity-target\n"
              << "             --recall-target --max-node-neighbors --prune-threshold\n"
              << "             --pq-segments --pq-bits --pq-max-distortion --threads\n"
              << "             --log-level\n"
              << "  search     Search an index with query vectors\n"
              << "  insert     Insert a single vector into an index\n";
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
        } else if (cmd == "autobuild") {
            return cmd_autobuild(sub_argc, sub_argv.data());
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
