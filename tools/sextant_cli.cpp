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
#include "sextant/builder.hpp"
#include "sextant/config.hpp"
#include "sextant/crash_handler.hpp"
#include "sextant/error.hpp"
#include "sextant/estimator.hpp"
#include "sextant/index.hpp"
#include "sextant/logging.hpp"
#include "sextant/searcher.hpp"
#include "tree/ivf_tree_index.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"

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
#include <unordered_set>
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
    const std::string index_path = p.get<std::string>("index");
    if (index_path.empty()) {
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
    // (cfg.sharded_graph and cfg.merged_graph are set in build_config_from_parser
    // from the --sharded-graph / --merged-graph flags, with mutual-exclusion
    // validation.)

    sextant::Index idx;
    sextant::Builder builder(idx);
    sextant::BuildResult result;
    const char* path_label;
    if (cfg.sharded_graph) {
        result = builder.build_ivf(source, index_path, cfg);
        path_label = "IVF (graph-inside-shard)";
    } else if (cfg.merged_graph) {
        result = builder.build(source, index_path, cfg);
        path_label = "merged-graph";
    } else {
        result = builder.build_ivf_scan(source, index_path, cfg);
        path_label = "IVF-list-scan";
    }
    std::cout << "built " << path_label << " index '" << index_path
              << "': n=" << result.n_vectors
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
    const std::string index_path = p.get<std::string>("index");
    if (index_path.empty()) {
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
    // (cfg.sharded_graph and cfg.merged_graph are set in build_config_from_parser.)

    sextant::Estimator estimator;
    // estimate_config handles all auto knobs; locked ones override.
    sextant::EstimateResult est = estimator.estimate_config(source, cfg);

    // Print the analysis (shared pretty-print with analyze).
    print_analysis_(source, input, cfg, est.params, est.diag);

    if (cfg.sharded_graph) {
        const uint32_t n_probe_default = p.get<uint32_t>("sharded-graph-n-probe");
        sextant::Index idx;
        const sextant::BuildResult result =
            sextant::Builder(idx).build_ivf(source, index_path, est.params,
                                             n_probe_default);
        std::cout << "\n═══ Build Result (IVF graph-inside-shard) ═══\n";
        std::cout << "built IVF index '" << index_path << ".shards': n="
                  << result.n_vectors << " dim=" << result.dim
                  << " K=" << est.params.partition_count
                  << " R=" << result.R
                  << " L_build=" << result.L_build
                  << " pq_m=" << static_cast<int>(result.pq_m)
                  << " pq_bits=" << static_cast<int>(result.pq_bits)
                  << " in " << result.build_time_sec << "s\n";
        return 0;
    }

    if (!cfg.merged_graph) {
        // Default: IVF-list-scan + 4-bit PQ FastScan.
        const uint32_t n_probe_default = p.get<uint32_t>("scan-n-probe");
        sextant::Index idx;
        const sextant::BuildResult result =
            sextant::Builder(idx).build_ivf_scan(source, index_path,
                                                  est.params, n_probe_default);
        std::cout << "\n═══ Build Result (IVF-list-scan) ═══\n";
        std::cout << "built IVF-scan index '" << index_path << ".shards': n="
                  << result.n_vectors << " dim=" << result.dim
                  << " K=" << est.params.partition_count
                  << " m4=" << static_cast<int>(est.params.pq4_m)
                  << " pq_bits=4"
                  << " in " << result.build_time_sec << "s\n";
        return 0;
    }

    // --merged-graph: merged-graph (K=1 or partition+merge).
    sextant::Index idx;
    const sextant::BuildResult result =
        sextant::Builder(idx).build(source, index_path, est.params);
    std::cout << "\n═══ Build Result (merged-graph) ═══\n";
    std::cout << "built index '" << index_path << "': n=" << result.n_vectors
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
        "Number of nearest neighbors to return per query (k in ANN literature). "
        "Default 100 (VIBE / modern-retrieval convention).",
        false, 100);
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

    const std::string index_path = p.get<std::string>("index");
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

    std::unique_ptr<sextant::Index> idx =
        sextant::Index::read(index_path, p.get<uint64_t>("cache-size"));
    sextant::Searcher searcher(*idx, num_threads);
    if (p.exist("no-cache-rebalance")) {
        searcher.set_cache_rebalance_enabled(false);
    }

    // Read the query file header.
    FbinHeader qh;
    if (!read_fbin_header(query_path, qh) || qh.dim != idx->dim) {
        std::cerr << "sextant search: invalid query file '" << query_path
                  << "' (dim=" << qh.dim << ", expected " << idx->dim
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
        if (!read_fbin_header(base_data, bh) || bh.dim != idx->dim) {
            std::cerr << "sextant search: invalid base-data file '" << base_data
                      << "' for rerank; skipping rerank\n";
        }
    }
    const bool rerank_ok = do_rerank && bh.dim == idx->dim;

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

    const uint32_t dim = idx->dim;
    const uint32_t fetch_k =
        rerank_ok ? std::min<uint32_t>(k * rerank, idx->count) : k;

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
            auto results = searcher.search(qvec.data(), scfg.k, scfg);

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
        // Chunked batch through the Searcher's pool (size = num_threads).
        // Per-query formatting (rerank, base lookup) is cheap relative to
        // search and runs inline on the caller thread as batches return.
        constexpr uint32_t kChunkMin = 1;
        const uint32_t chunk_size = std::max(kChunkMin, n_threads);
        std::vector<std::string> formatted_results(qh.n);
        std::vector<float> base_vec(dim);

        for (uint32_t base = 0; base < qh.n; base += chunk_size) {
            const uint32_t n_this =
                std::min<uint32_t>(chunk_size, qh.n - base);
            sextant::SearchConfig scfg;
            scfg.k = fetch_k;
            scfg.L_search = L;
            const float* q0 = &queries[static_cast<size_t>(base) * dim];
            auto batch = searcher.search_batch(q0, n_this, scfg.k, scfg);
            for (uint32_t j = 0; j < n_this; j++) {
                const uint32_t qi = base + j;
                const float* q = &queries[static_cast<size_t>(qi) * dim];
                auto& results = batch[j];
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
        }

        // Emit in query order.
        for (uint32_t qi = 0; qi < qh.n; qi++) {
            (*out) << formatted_results[qi];
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// build-tree: build a hierarchical IVF tree index
// ---------------------------------------------------------------------------

int cmd_build_tree(int argc, char* argv[]) {
    using namespace sextant_cli;
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("input", 0, "Base vectors (.fbin)", true);
    p.add<std::string>("index", 0, "Output tree file path", true);
    p.add<uint32_t>("k-root", 0, "Root branching factor (0=auto)", false, 0);
    p.add<uint32_t>("leaf-capacity", 0, "Max vectors per leaf", false, 5000);
    p.add<uint16_t>("pq4-m", 0, "PQ subquantizers for 4-bit scan (0=dim/4)", false, 0);
    p.add<uint32_t>("pq-bits", 0, "PQ bits (4 or 8)", false, 4);
    p.add<std::string>("quantizer", 0, "pq / prq / rabitq", false, "pq");
    p.add<std::string>("metric", 0, "l2sq / ip", false, "l2sq");
    p.add<float>("closure-epsilon", 0, "Absolute margin for closure (-1=auto)", false, -1.0f);
    p.add<float>("balance-factor", 0, "SPANN balance factor (0=off)", false, 4.0f);
    p.add<uint32_t>("threads", 0, "Build threads (0=auto)", false, 0);
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");
    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") set_log_level(LogLevel::Debug);
        else if (lvl == "warn") set_log_level(LogLevel::Warn);
        else if (lvl == "error") set_log_level(LogLevel::Error);
    }

    tree::IVFTreeIndex::BuildConfig cfg;
    cfg.k_root = p.get<uint32_t>("k-root");
    cfg.leaf_capacity = p.get<uint32_t>("leaf-capacity");
    cfg.num_threads = p.get<uint32_t>("threads");
    cfg.params.pq4_m = p.get<uint16_t>("pq4-m");
    cfg.params.scan_pq_bits = static_cast<uint8_t>(p.get<uint32_t>("pq-bits"));
    cfg.params.quantizer_type = p.get<std::string>("quantizer");
    const std::string metric = p.get<std::string>("metric");
    cfg.params.metric = (metric == "ip") ? MetricKind::InnerProduct
                                          : MetricKind::L2Sq;
    cfg.params.closure_epsilon = p.get<float>("closure-epsilon");
    cfg.params.partition_balance_factor = p.get<float>("balance-factor");

    const std::string input = p.get<std::string>("input");
    const std::string index_path = p.get<std::string>("index");

    auto result = tree::IVFTreeIndex::build(input, index_path, cfg);
    std::cout << "built tree index '" << index_path
              << "': n=" << result.n_vectors
              << " dim=" << result.dim
              << " m4=" << static_cast<int>(result.pq_m)
              << " in " << result.build_time_sec << "s\n";
    return 0;
}

// ---------------------------------------------------------------------------
// tree-search: search a hierarchical IVF tree index
// ---------------------------------------------------------------------------

int cmd_tree_search(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("index", 0, "Tree index file", true);
    p.add<std::string>("query", 0, "Query vectors (.fbin)", true);
    p.add<std::string>("ground-truth", 0, "Ground-truth .gt file", false, "");
    p.add<uint32_t>("topk", 0, "K nearest neighbors", false, 10);
    p.add<uint32_t>("n-probe", 0, "Root probe count (0=manifest default)", false, 0);
    p.add<uint32_t>("fastscan-w", 0, "Rerank shortlist per shard (0=300)", false, 0);
    p.add<float>("adaptive-probe-gap", 0, "Geometric gap pruning (0=manifest)", false, 0.0f);
    p.add<uint32_t>("threads", 0, "Search threads (0=auto)", false, 0);
    p.add<std::string>("output", 0, "Output file (default stdout)", false, "");
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");
    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") set_log_level(LogLevel::Debug);
        else if (lvl == "warn") set_log_level(LogLevel::Warn);
        else if (lvl == "error") set_log_level(LogLevel::Error);
    }

    auto idx = tree::IVFTreeIndex::open(p.get<std::string>("index"));

    // Read query file.
    FbinHeader qh;
    if (!read_fbin_header(p.get<std::string>("query"), qh) ||
        qh.dim != idx->dim()) {
        std::cerr << "tree-search: invalid query file (dim=" << qh.dim
                  << ", expected " << idx->dim() << ")\n";
        return 1;
    }

    const uint32_t k = p.get<uint32_t>("topk");
    uint32_t num_threads = p.get<uint32_t>("threads");
    if (num_threads == 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());

    SearchConfig scfg;
    scfg.k = k;
    scfg.n_probe = p.get<uint32_t>("n-probe");
    scfg.fastscan_W = p.get<uint32_t>("fastscan-w");
    scfg.adaptive_probe_gap = p.get<float>("adaptive-probe-gap");

    std::ifstream qf(p.get<std::string>("query"), std::ios::binary);
    qf.seekg(8);
    // (queries are read in bulk below for parallel processing)

    // Load ground truth if provided.
    std::vector<std::vector<RowId>> gt;
    if (!p.get<std::string>("ground-truth").empty()) {
        std::ifstream gtf(p.get<std::string>("ground-truth"), std::ios::binary);
        if (gtf) {
            // GT format: [magic "GTMM":4B][n:u32][k:u32][metric:u8][ids][dists]
            // Legacy: [n:u32][k:u32][ids][dists] (no magic)
            constexpr uint32_t kGtMagic = 0x4D4D5447u;  // "GTMM" LE
            uint32_t maybe_magic = 0;
            gtf.read(reinterpret_cast<char*>(&maybe_magic), 4);
            uint32_t gt_n = 0, gt_k = 0;
            if (maybe_magic == kGtMagic) {
                gtf.read(reinterpret_cast<char*>(&gt_n), 4);
                gtf.read(reinterpret_cast<char*>(&gt_k), 4);
                uint8_t metric_byte = 0;
                gtf.read(reinterpret_cast<char*>(&metric_byte), 1);
            } else {
                // Legacy: first 4 bytes are n, not magic.
                gt_n = maybe_magic;
                gtf.read(reinterpret_cast<char*>(&gt_k), 4);
            }
            gt.resize(gt_n);
            std::vector<uint32_t> row(gt_k);
            for (uint32_t i = 0; i < gt_n; ++i) {
                gtf.read(reinterpret_cast<char*>(row.data()),
                         gt_k * sizeof(uint32_t));
                gt[i].assign(row.begin(), row.end());  // uint32 → int64
            }
            std::cerr << "loaded ground truth: " << gt_n << " queries, k="
                      << gt_k << "\n";
        }
    }

    std::ofstream out_file;
    std::ostream* out = &std::cout;
    if (!p.get<std::string>("output").empty()) {
        out_file.open(p.get<std::string>("output"));
        out = &out_file;
    }

    uint64_t total_hits = 0;
    uint64_t total_queries = 0;

    // Read all queries into memory for parallel processing.
    std::vector<float> queries(static_cast<size_t>(qh.n) * qh.dim);
    qf.read(reinterpret_cast<char*>(queries.data()),
            static_cast<std::streamsize>(qh.n * qh.dim * sizeof(float)));
    qf.close();

    std::vector<std::vector<Candidate>> all_results(qh.n);

    const auto t0 = std::chrono::steady_clock::now();

    if (num_threads <= 1) {
        for (uint32_t qi = 0; qi < qh.n; ++qi) {
            all_results[qi] = idx->search(
                &queries[static_cast<size_t>(qi) * qh.dim], k, scfg);
        }
    } else {
        // Query-level parallelism: each query is independent.
        std::vector<std::future<void>> futs;
        std::atomic<uint32_t> next_qi{0};
        for (uint32_t t = 0; t < num_threads; ++t) {
            futs.push_back(std::async(std::launch::async, [&]() {
                while (true) {
                    const uint32_t qi = next_qi.fetch_add(1);
                    if (qi >= qh.n) break;
                    all_results[qi] = idx->search(
                        &queries[static_cast<size_t>(qi) * qh.dim], k, scfg);
                }
            }));
        }
        for (auto& f : futs) f.get();
    }

    // Tally recall + emit results (serial — I/O bound).
    for (uint32_t qi = 0; qi < qh.n; ++qi) {
        const auto& results = all_results[qi];
        if (!gt.empty() && qi < gt.size()) {
            std::unordered_set<RowId> gt_set(gt[qi].begin(), gt[qi].end());
            for (const auto& c : results) {
                if (gt_set.count(c.row_id)) ++total_hits;
            }
            ++total_queries;
        }
        for (const auto& c : results) {
            *out << qi << '\t' << c.row_id << '\t' << c.dist << '\n';
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double qps = (secs > 0) ? qh.n / secs : 0;

    std::cerr << "\n═══ Tree Search Results ═══\n";
    std::cerr << "queries: " << qh.n << "\n";
    std::cerr << "k: " << k << "\n";
    std::cerr << "time: " << secs << "s (" << qps << " QPS)\n";
    if (total_queries > 0) {
        const float recall = float(total_hits) / (total_queries * k);
        std::cerr << "recall@" << k << ": " << recall << "\n";
    }
    return 0;
}

int cmd_insert(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("index", 0, "Index name/path prefix", true);
    p.add<std::string>("vector", 0, "Single-vector .fbin file", true);
    p.add<int64_t>("row-id", 0, "Row ID for the inserted vector", true);
    p.parse_check(argc, argv);

    const std::string index_path = p.get<std::string>("index");
    const std::string vector_path = p.get<std::string>("vector");
    const int64_t row_id = p.get<int64_t>("row-id");

    auto idx = sextant::Index::read(index_path);

    FbinHeader vh;
    if (!read_fbin_header(vector_path, vh) || vh.dim != idx->dim ||
        vh.n != 1) {
        std::cerr << "sextant insert: need a single-vector .fbin with dim="
                  << idx->dim << " (got n=" << vh.n << " dim=" << vh.dim
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

    sextant::Builder builder(*idx);
    builder.insert(vec.data(), vh.dim, static_cast<sextant::RowId>(row_id));
    builder.flush();
    std::cout << "inserted row_id=" << row_id
              << " (count now " << idx->count << ")\n";
    return 0;
}

void print_usage() {
    std::cerr << "Usage: sextant <command> [options]\n"
              << "Commands:\n"
              << "  build-tree  Build a hierarchical IVF tree index (single file).\n"
              << "              --input --index --k-root --leaf-capacity --pq4-m\n"
              << "              --pq-bits --quantizer (pq/prq/rabitq) --metric\n"
              << "              --closure-epsilon --balance-factor --threads\n"
              << "  tree-search Search a tree index. Computes recall if --ground-truth.\n"
              << "              --index --query --topk --n-probe --fastscan-w\n"
              << "              --adaptive-probe-gap --ground-truth --threads\n"
              << "  build      Build an index from a .fbin file (explicit params).\n"
              << "             Build path selection (mutually exclusive):\n"
              << "               (default)  IVF-list-scan + 4-bit PQ FastScan\n"
              << "               --merged-graph  merged-graph (Vamana; K=1 or partition+merge)\n"
              << "               --sharded-graph graph-inside-shard IVF (prior default)\n"
              << "             Common flags: --input --index --max-node-neighbors (R)\n"
              << "             --beam-width-ceiling (L) --prune-threshold (alpha)\n"
              << "             --pq-segments (m) --pq-bits --threads --metric\n"
              << "             --build-ram --prune-candidate-cap --partition-count\n"
              << "             --log-level\n"
              << "  autobuild  Estimate config (analyze) then build, in-process.\n"
              << "             Accepts all build flags plus estimate knobs:\n"
              << "             --proximity-target --recall-target\n"
              << "             --scan-n-probe (default path) --sharded-graph-n-probe (--sharded-graph)\n"
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
    sextant::install_crash_handler();
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
        } else if (cmd == "build-tree") {
            return cmd_build_tree(sub_argc, sub_argv.data());
        } else if (cmd == "autobuild") {
            return cmd_autobuild(sub_argc, sub_argv.data());
        } else if (cmd == "search") {
            return cmd_search(sub_argc, sub_argv.data());
        } else if (cmd == "tree-search") {
            return cmd_tree_search(sub_argc, sub_argv.data());
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
