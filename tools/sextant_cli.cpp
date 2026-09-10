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

#include "fbin_source.hpp"
#include "fbin_io.hpp"
#include "parquet_source.hpp"
#include "parquet_glob_source.hpp"
#include "shared_cli.hpp"
#include "sextant_version.hpp"
#include "sextant/builder.hpp"
#include "sextant/config.hpp"
#include "sextant/crash_handler.hpp"
#include "sextant/engine_trace.hpp"
#include "sextant/ground_truth.hpp"
#include "sextant/error.hpp"
#include "sextant/estimator.hpp"
#include "sextant/index.hpp"
#include "sextant/logging.hpp"
#include "sextant/searcher.hpp"
#include "sextant/metrics.hpp"
#include "tree/fsck.hpp"
#include "tree/filter_data_io.hpp"
#include "tree/ivf_tree_index.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "simd_kernels.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"

#include <cmdline/cmdline.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glob.h>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
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
    p.add<std::string>("metrics-file", 0,
        "Append build phase metrics as JSON lines (build.* field names)",
        false, "");
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

    const std::string metrics_path = p.get<std::string>("metrics-file");
    std::unique_ptr<sextant::metrics::JsonlMetricsSink> metrics_json;
    sextant::metrics::LogMetricsSink metrics_log;
    sextant::metrics::MultiMetricsSink metrics_sink;
    metrics_sink.add(&metrics_log);
    if (!metrics_path.empty()) {
        metrics_json =
            std::make_unique<sextant::metrics::JsonlMetricsSink>(metrics_path);
        metrics_sink.add(metrics_json.get());
    }
    cfg.metrics_sink = &metrics_sink;

    sextant::Index idx;
    sextant::Builder builder(idx);
    const sextant::BuildResult result =
        builder.build(source, index_path, cfg);
    std::cout << "built index '" << index_path
              << "': n=" << result.n_vectors
              << " dim=" << result.dim
              << " R=" << result.R
              << " L_build=" << result.L_build
              << " pq_m=" << static_cast<int>(result.pq_m)
              << " pq_bits=" << static_cast<int>(result.pq_bits)
              << " in " << result.build_time_sec << "s"
              << " (cpu " << result.cpu_time_sec << "s, util "
              << (result.build_time_sec > 0
                      ? 100.0 * result.cpu_time_sec / result.build_time_sec : 0.0)
              << "%, src_wait " << result.source_wait_sec << "s, read "
              << result.bytes_read << "B, peak rss " << result.peak_rss_bytes
              << "B)\n";
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

    sextant::Estimator estimator;
    // estimate_config handles all auto knobs; locked ones override.
    sextant::EstimateResult est = estimator.estimate_config(source, cfg);

    // Print the analysis (shared pretty-print with analyze).
    print_analysis_(source, input, cfg, est.params, est.diag);

    sextant::Index idx;
    const sextant::BuildResult result =
        sextant::Builder(idx).build(source, index_path, est.params);
    std::cout << "\n═══ Build Result ═══\n";
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
// build-tree-pca: PCA-preconditioned streaming build
// ---------------------------------------------------------------------------
int cmd_build_tree_pca(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("input", 0, "Base vectors (.fbin/.parquet)", true);
    p.add<std::string>("vector-col", 0, "Vector column name in parquet (default: embedding)", false, "embedding");
    p.add<std::string>("index", 0, "Output tree file path", true);
    p.add<uint32_t>("k-root", 0, "Root branching factor (0=auto)", false, 0);
    p.add<uint32_t>("leaf-capacity", 0, "Max vectors per leaf", false, 5000);
    p.add<uint32_t>("chunk-vectors", 0,
        "Source chunk size in vectors (fbin builds; larger = fewer "
        "thread-spawn boundaries in the streaming passes)", false, 2048);
    p.add<uint16_t>("pq4-m", 0, "PQ subquantizers (0=dim/4)", false, 0);
    p.add<uint32_t>("pq-bits", 0, "PQ bits (4 or 8)", false, 4);
    p.add<std::string>("quantizer", 0, "pq / prq / local_pq / scalar_lloydmax / scalar_uniform / scalar_shape / anisotropic_pq", false, "pq");
    p.add<uint32_t>("prq-nsplits", 0,
        "PRQ nsplits (sub-space count) for --quantizer prq. 0 = auto. "
        "Must divide both dim and m4.", false, 0);
    p.add<uint32_t>("prq-beam-size", 0,
        "PRQ beam size for encoding (1=greedy, >1=beam search).", false, 1);
    p.add<std::string>("prq-encode-mode", 0,
        "PRQ encoding strategy: greedy (default), beam, or icm "
        "(ICM+ILS coordinate descent, best quality).", false, "greedy");
    p.add<uint32_t>("prq-icm-iters", 0,
        "PRQ ICM sweeps per ILS cycle (icm mode only).", false, 4);
    p.add<uint32_t>("prq-ils-iters", 0,
        "PRQ ILS cycles: perturb + ICM + accept (icm mode only).", false, 4);
    p.add<uint32_t>("prq-ils-perturb", 0,
        "PRQ codes randomized per ILS cycle (icm mode only).", false, 4);
    p.add<uint32_t>("prq-lsq-train-iters", 0,
        "LSQ training iterations (0 = progressive k-means only).", false, 0);
    p.add<std::string>("metric", 0, "l2sq / ip", false, "l2sq");
    p.add<uint32_t>("threads", 0, "Build threads (0=auto)", false, 0);
    p.add<uint32_t>("pca-dims", 0, "PCA dimensions (default 32)", false, 32);
    p.add<uint32_t>("max-lloyd-passes", 0, "Max streaming Lloyd passes (default 10)", false, 10);
    p.add<float>("closure-mult", 0, "Closure epsilon multiplier (default 0.15)", false, 0.15f);
    p.add<std::string>("filter-data", 0, "Filter column data sidecar (.fdat)", false, "");
    p.add<std::string>("label-file", 0, "Parquet file with filter columns (joined by row position)", false, "");
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");
    p.add<std::string>("metrics-file", 0,
        "Append build phase metrics as JSON lines (build.* field names)",
        false, "");
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
    cfg.pca_dims = p.get<uint32_t>("pca-dims");
    cfg.max_lloyd_passes = p.get<uint32_t>("max-lloyd-passes");
    cfg.closure_multiplier = p.get<float>("closure-mult") > 0 ? p.get<float>("closure-mult") : 0.15f;
    cfg.params.pq4_m = p.get<uint16_t>("pq4-m");
    cfg.params.scan_pq_bits = static_cast<uint8_t>(p.get<uint32_t>("pq-bits"));
    cfg.params.quantizer_type = p.get<std::string>("quantizer");
    if (p.exist("prq-nsplits"))
        cfg.params.prq_nsplits = p.get<uint32_t>("prq-nsplits");
    if (p.exist("prq-beam-size"))
        cfg.params.prq_beam_size = p.get<uint32_t>("prq-beam-size");
    if (p.exist("prq-encode-mode"))
        cfg.params.prq_encode_mode = p.get<std::string>("prq-encode-mode");
    if (p.exist("prq-icm-iters"))
        cfg.params.prq_icm_iters = p.get<uint32_t>("prq-icm-iters");
    if (p.exist("prq-ils-iters"))
        cfg.params.prq_ils_iters = p.get<uint32_t>("prq-ils-iters");
    if (p.exist("prq-ils-perturb"))
        cfg.params.prq_ils_perturb = p.get<uint32_t>("prq-ils-perturb");
    if (p.exist("prq-lsq-train-iters"))
        cfg.params.prq_lsq_train_iters =
            p.get<uint32_t>("prq-lsq-train-iters");
    const std::string metric = p.get<std::string>("metric");
    cfg.params.metric = (metric == "ip") ? MetricKind::InnerProduct
                                          : MetricKind::L2Sq;

    // Load filter column data sidecar (.fdat) if provided.
    const std::string fdat_path = p.get<std::string>("filter-data");
    if (!fdat_path.empty()) {
        auto fdat = tree::read_filter_data(fdat_path);
        cfg.filter_schema = fdat.schema;
        cfg.filter_column_data = std::move(fdat.cols);
        if (fdat.has_payload) {
            cfg.filter_schema.has_payload = true;
            cfg.payload_data = fdat.payload_data.data();
            cfg.payload_offsets = fdat.payload_offsets.data();
        }
        std::cerr << "loaded filter data: " << fdat.n_rows << " rows, "
                  << fdat.schema.columns.size() << " columns"
                  << (fdat.has_payload ? ", with payload" : "") << "\n";
    }

    const std::string input_path = p.get<std::string>("input");
    const std::string vector_col = p.exist("vector-col")
        ? p.get<std::string>("vector-col") : "embedding";
    const std::string label_file = p.get<std::string>("label-file");
    const bool need_normalize = (metric == "ip");

    // Expand glob patterns (e.g., "train-*.parquet") into a list of paths.
    auto expand_glob = [](const std::string& pattern) -> std::vector<std::string> {
        std::vector<std::string> result;
        // Simple glob using POSIX glob()
        glob_t g;
        memset(&g, 0, sizeof(g));
        int rc = ::glob(pattern.c_str(), GLOB_TILDE | GLOB_BRACE, nullptr, &g);
        if (rc == 0) {
            for (size_t i = 0; i < g.gl_pathc; ++i) {
                result.emplace_back(g.gl_pathv[i]);
            }
        }
        globfree(&g);
        if (result.empty()) {
            // Not a glob or no matches — treat as literal path
            result.push_back(pattern);
        }
        std::sort(result.begin(), result.end());
        return result;
    };

    bool is_parquet = input_path.size() >= 8 &&
        input_path.compare(input_path.size() - 8, 8, ".parquet") == 0;
    // Also check if it's a glob that matches parquet files
    if (!is_parquet && input_path.find('*') != std::string::npos) {
        auto matches = expand_glob(input_path);
        if (!matches.empty() && matches[0].size() >= 8 &&
            matches[0].compare(matches[0].size() - 8, 8, ".parquet") == 0) {
            is_parquet = true;
        }
    }

    const std::string metrics_path = p.get<std::string>("metrics-file");
    std::unique_ptr<metrics::JsonlMetricsSink> metrics_json;
    if (!metrics_path.empty())
        metrics_json = std::make_unique<metrics::JsonlMetricsSink>(metrics_path);
    cfg.metrics_sink = metrics_json.get();  // null = default log line only

    BuildResult result;
    if (is_parquet) {
        auto shard_paths = expand_glob(input_path);
        bool use_glob = shard_paths.size() > 1 || !label_file.empty();

        if (use_glob) {
            ParquetGlobSource::Config gcfg;
            gcfg.vector_col = vector_col;
            gcfg.normalize = need_normalize;
            gcfg.label_file = label_file;
            gcfg.batch_size = 8192;
            // When --filter-data is provided, suppress shard-local filter columns
            // (the builder uses the global cfg.filter_column_data from the fdat).
            gcfg.vectors_only = !fdat_path.empty();
            ParquetGlobSource source(shard_paths, gcfg);
            if (fdat_path.empty()) {
                cfg.filter_schema = source.schema();
            }
            result = tree::IVFTreeIndex::build_streaming_pca(
                source, p.get<std::string>("index"), cfg);
        } else {
            ParquetSourceConfig pcfg;
            pcfg.vector_col = vector_col;
            pcfg.normalize = need_normalize;
            ParquetSource source(shard_paths[0], pcfg);
            if (fdat_path.empty()) {
                cfg.filter_schema = source.schema();
            }
            result = tree::IVFTreeIndex::build_streaming_pca(
                source, p.get<std::string>("index"), cfg);
        }
    } else {
        FbinSource source(input_path,
            std::max<uint32_t>(1u, p.get<uint32_t>("chunk-vectors")));
        result = tree::IVFTreeIndex::build_streaming_pca(
            source, p.get<std::string>("index"), cfg);
    }
    std::cout << "built tree index (pca) '" << result.index_path
              << "': n=" << result.n_vectors
              << " dim=" << result.dim
              << " m4=" << static_cast<int>(result.pq_m)
              << " in " << result.build_time_sec << "s"
              << " (cpu " << result.cpu_time_sec << "s, util "
              << (result.build_time_sec > 0
                      ? 100.0 * result.cpu_time_sec / result.build_time_sec : 0.0)
              << "%, src_wait " << result.source_wait_sec << "s, read "
              << result.bytes_read << "B, peak rss " << result.peak_rss_bytes
              << "B)\n";
    return 0;
}

// ---------------------------------------------------------------------------
// tree-search: search a hierarchical IVF tree index
// ---------------------------------------------------------------------------

namespace {

// Split a delimiter-separated string into a vector of non-empty tokens.
std::vector<std::string> split_on(char delim, const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == delim) { out.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Parse one "--filter" predicate string ("column:op:value[:value2...]") into a
// Predicate. Returns false and prints to stderr on a malformed spec.
bool parse_filter_predicate(const std::string& filter_str, sextant::Predicate& pred) {
    using sextant::PredicateOp;
    const std::vector<std::string> parts = split_on(':', filter_str);
    if (parts.size() < 3) {
        std::cerr << "malformed --filter (need column:op:value): " << filter_str << "\n";
        return false;
    }

    pred = sextant::Predicate{};
    pred.column = parts[0];
    const std::string& op_str = parts[1];
    const std::string& val_str = parts[2];

    if      (op_str == "eq")            pred.op = PredicateOp::Eq;
    else if (op_str == "ne")            pred.op = PredicateOp::NotEq;
    else if (op_str == "lt")            pred.op = PredicateOp::Lt;
    else if (op_str == "le")            pred.op = PredicateOp::Le;
    else if (op_str == "gt")            pred.op = PredicateOp::Gt;
    else if (op_str == "ge")            pred.op = PredicateOp::Ge;
    else if (op_str == "between")       pred.op = PredicateOp::Between;
    else if (op_str == "prefix")        pred.op = PredicateOp::Prefix;
    else if (op_str == "in")            pred.op = PredicateOp::In;
    else if (op_str == "not_in")        pred.op = PredicateOp::NotIn;
    else if (op_str == "contains")      pred.op = PredicateOp::Contains;
    else if (op_str == "contains_any")  pred.op = PredicateOp::ContainsAny;
    else if (op_str == "contains_all")  pred.op = PredicateOp::ContainsAll;
    else if (op_str == "geo_radius")    pred.op = PredicateOp::GeoRadius;
    else if (op_str == "geo_box")       pred.op = PredicateOp::GeoBox;
    else {
        std::cerr << "unknown filter op: " << op_str << "\n";
        return false;
    }

    // Populate value fields based on the op semantics.
    //   value/value2/value3/value4 (double): Between(low,high),
    //     GeoRadius(lat,lng) + radius_km, GeoBox(min_lat,min_lng,max_lat,max_lng).
    //   str_value (string): Eq/NotEq/Prefix on string columns, Contains element.
    //   values (vector<string>): In/NotIn membership, ContainsAny/ContainsAll sets.
    switch (pred.op) {
        case PredicateOp::Between:
            if (parts.size() < 4) {
                std::cerr << "between needs low:high: " << filter_str << "\n";
                return false;
            }
            pred.value  = std::stod(val_str);
            pred.value2 = std::stod(parts[3]);
            break;
        case PredicateOp::GeoRadius:
            // lat_col:lng_col:geo_radius:lat:lng:radius_km
            if (parts.size() < 6) {
                std::cerr << "geo_radius needs lat_col:lng_col:geo_radius:lat:lng:radius_km: "
                          << filter_str << "\n";
                return false;
            }
            pred.geo_lng_column = parts[1];               // lng column name
            pred.value     = std::stod(parts[3]);          // center lat
            pred.value2    = std::stod(parts[4]);          // center lng
            pred.radius_km = std::stod(parts[5]);          // radius km
            break;
        case PredicateOp::GeoBox:
            // lat_col:lng_col:geo_box:lat_min:lat_max:lng_min:lng_max
            // schema.hpp: value=min_lat, value2=min_lng, value3=max_lat, value4=max_lng
            if (parts.size() < 7) {
                std::cerr << "geo_box needs lat_col:lng_col:geo_box:lat_min:lat_max:lng_min:lng_max: "
                          << filter_str << "\n";
                return false;
            }
            pred.geo_lng_column = parts[1];               // lng column name
            pred.value  = std::stod(parts[3]);            // lat_min
            pred.value3 = std::stod(parts[4]);            // lat_max
            pred.value2 = std::stod(parts[5]);            // lng_min
            pred.value4 = std::stod(parts[6]);            // lng_max
            break;
        case PredicateOp::In:
        case PredicateOp::NotIn:
            pred.values = split_on(',', val_str);
            break;
        case PredicateOp::Contains:
            pred.str_value = val_str;
            break;
        case PredicateOp::ContainsAny:
        case PredicateOp::ContainsAll:
            // Comma-separated set elements. The engine folds str_value + values
            // together, so stash the first element in str_value and the rest in
            // values to mirror how the other set ops are populated.
            {
                auto elems = split_on(',', val_str);
                if (!elems.empty()) {
                    pred.str_value = std::move(elems.front());
                    pred.values.assign(std::make_move_iterator(elems.begin() + 1),
                                       std::make_move_iterator(elems.end()));
                }
            }
            break;
        case PredicateOp::Prefix:
        case PredicateOp::Eq:
        case PredicateOp::NotEq:
            // Could be string or numeric. Try numeric first; fall back to string.
            try { pred.value = std::stod(val_str); }
            catch (...) { pred.str_value = val_str; }
            break;
        default:
            // Lt/Le/Gt/Ge: numeric comparison.
            try { pred.value = std::stod(val_str); }
            catch (...) {
                std::cerr << "non-numeric value for numeric op '" << op_str
                          << "': " << val_str << "\n";
                return false;
            }
            break;
    }
    return true;
}

}  // namespace

int cmd_tree_search(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("index", 0, "Tree index file", true);
    p.add<std::string>("query", 0, "Query vectors (.fbin or .parquet)", true);
    p.add<std::string>("ground-truth", 0, "Ground-truth .gtmm file (GTMM format)", false, "");
    p.add<uint32_t>("topk", 0, "K nearest neighbors", false, 10);
    p.add<uint32_t>("n-probe", 0, "Root probe count (0=manifest default)", false, 0);
    p.add<uint32_t>("n-probe-ln", 0, "Leaf probe count per root child (0=manifest)", false, 0);
    p.add<float>("probe-fraction", 0,
        "Probe budget as a corpus fraction (0=manifest default, new trees 0.5): "
        "select root children nearest-first until their cumulative subtree "
        "extent reaches this fraction, then probe all their leaves. "
        "Scale-stable where counts are not (measured dbpedia 100K→933K: "
        "f=0.25 ≈0.96, f=0.5 ≈0.99 recall@10). Ignored when --n-probe > 0.",
        false, 0.0f);
    p.add<uint32_t>("fastscan-w", 0, "Rerank shortlist per shard (0=300)", false, 0);
    p.add("no-rerank", 0, "Disable FP32 rerank (use raw PQ distances)");
    p.add<std::string>("exact-rerank-base", 0,
        "Original base .fbin (mmap'd read-only): rerank against the TRUE "
        "vectors instead of decoded quantized codes — removes the "
        "quantization ranking error entirely (harness-measured: perfect "
        "recall AND faster than decode-rerank). Implies rerank. Must be "
        "the corpus this tree was built from, in build row order.",
        false, "");
    p.add<float>("adaptive-probe-gap", 0, "Geometric gap pruning (0=manifest)", false, 0.0f);
    p.add<float>("adaptive-w-gap", 0,
        "Adaptive shortlist cut: truncate results at the first reranked "
        "distance gap past k (0=off)", false, 0.0f);
    p.add<uint32_t>("threads", 0, "Search threads (0=auto)", false, 0);
    p.add<int>("scan-i8", 0,
        "Scalar scan kernel i8 SDOT/VNNI mode (-1=auto: int8 on AVX512/VNNI, "
        "float otherwise; 0=float FMA, 1=int8, 2=int8+dual residual); "
        "uniform and shared-shape families",
        false, -1);
    p.add<uint32_t>("search-threads", 0,
        "Within-query leaf-parallel scan threads (0=serial; orthorgonal to --threads)", false, 0);
    p.add<uint32_t>("batch-window", 0,
        "Subtree-major batch mode: queries per search_batch window. Routes "
        "the window, inverts probe sets, and sweeps each UNIQUE probed leaf "
        "once in page order (coalesced I/O at zero engine DRAM; results "
        "identical to per-query search). 0 = per-query search. Incompatible "
        "with payload output", false, 0);
    p.add<uint32_t>("cache-mb", 0,
        "Engine-owned leaf cache in MiB (0=off: mmap path). Warm results "
        "with a cache must be labeled with this size (BENCHMARK_RULES)", false, 0);
    p.add<uint32_t>("cache-window-pct", 0,
        "LeafCache W-TinyLFU window as % of capacity (default 1, Caffeine)", false, 1);
    p.add<uint32_t>("passes", 0,
        "Run the query set N times. N>=2 discards the first pass as warmup "
        "and times the rest (BENCHMARK_RULES warm measurement; combine with "
        "--cache-mb and report the cache size with the result)", false, 1);
    p.add<std::string>("metrics-file", 0,
        "Append search window metrics as one JSON line (search.* field names)", false, "");
    p.add<std::string>("output", 0, "Output file (default stdout)", false, "");
    p.add<std::string>("filter", 0,
        "Filter predicate (repeatable: pass --filter multiple times for AND). "
        "Format: column:op:value[:value2...]. "
        "Ops: eq,ne,lt,le,gt,ge,between,prefix,in,not_in,contains,"
        "contains_any,contains_all,geo_radius,geo_box. "
        "Numeric: year:eq:2020 year:between:2010:2020 "
        "String: category:eq:cs.AI category:prefix:cs. "
        "Set: tags:contains:ml tags:contains_any:ml,ai tags:contains_all:ml,ai. "
        "Multi-value: cat:in:a,b cat:not_in:x,y. "
        "Geo: lat_col:lng_col:geo_radius:lat:lng:radius_km lat_col:lng_col:geo_box:lat_min:lat_max:lng_min:lng_max.",
        false, "");
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");
    p.add<std::string>("vector-col", 0, "Vector column name for parquet queries (default: emb)", false, "emb");
    p.add("with-payload", 0,
          "Fetch + print opaque payload blobs with results");
    // The cmdline library only keeps the LAST value of a repeated option, so
    // collect every --filter occurrence from argv ourselves (supports AND of
    // multiple predicates). Both "--filter X" and "--filter=X" forms are
    // handled; cmdline still sees the flags (harmlessly overwriting its single
    // "filter" slot) — we ignore its parsed value and use this vector instead.
    std::vector<std::string> filter_strings;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--filter") {
            if (i + 1 < argc) filter_strings.emplace_back(argv[++i]);
        } else if (arg.rfind("--filter=", 0) == 0) {
            filter_strings.emplace_back(arg.substr(9));
        }
    }

    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") set_log_level(LogLevel::Debug);
        else if (lvl == "warn") set_log_level(LogLevel::Warn);
        else if (lvl == "error") set_log_level(LogLevel::Error);
    }

    const bool with_payload = p.exist("with-payload");

    // Scalar scan kernel selection: resolved into the coder's params at
    // open (must precede open — the factory reads the override once).
    // Default -1 = auto (AVX512/VNNI when available), matching the C-API.
    sextant::tree::set_scan_i8_override(p.exist("scan-i8")
        ? p.get<int>("scan-i8") : -1);
    auto idx = tree::IVFTreeIndex::open(p.get<std::string>("index"),
        static_cast<uint64_t>(p.get<uint32_t>("cache-mb")) * 1024 * 1024,
        p.get<uint32_t>("cache-window-pct"));

    // Read query file (.fbin or .parquet).
    const std::string query_path = p.get<std::string>("query");
    const bool query_is_parquet = query_path.size() >= 8 &&
        query_path.compare(query_path.size() - 8, 8, ".parquet") == 0;

    // Determine if queries need normalization (IP metric = cosine on unit vecs).
    const bool need_normalize =
        idx->metric() == sextant::MetricKind::InnerProduct;

    uint32_t qdim = 0, qcount = 0;
    std::vector<float> queries;

    if (query_is_parquet) {
        // Read queries from parquet (list<float> or FLBA).
        ParquetSourceConfig pcfg;
        pcfg.vector_col = p.exist("vector-col")
            ? p.get<std::string>("vector-col") : "emb";
        pcfg.normalize = need_normalize;
        ParquetSource qsrc(query_path, pcfg);
        qdim = qsrc.dim();
        qcount = static_cast<uint32_t>(qsrc.count());
        queries.resize(static_cast<size_t>(qcount) * qdim);
        Chunk chunk;
        size_t offset = 0;
        while (qsrc.next(chunk)) {
            std::memcpy(&queries[offset], chunk.vectors,
                        static_cast<size_t>(chunk.count) * qdim * sizeof(float));
            offset += static_cast<size_t>(chunk.count) * qdim;
        }
    } else {
        FbinHeader qh;
        if (!read_fbin_header(query_path, qh) ||
            qh.dim != idx->dim()) {
            std::cerr << "tree-search: invalid query file (dim=" << qh.dim
                      << ", expected " << idx->dim() << ")\n";
            return 1;
        }
        qdim = qh.dim;
        qcount = qh.n;
        queries.resize(static_cast<size_t>(qcount) * qdim);
        std::ifstream qf(query_path, std::ios::binary);
        qf.seekg(8);
        qf.read(reinterpret_cast<char*>(queries.data()),
                static_cast<std::streamsize>(qcount) * qdim * sizeof(float));
        qf.close();
        if (need_normalize) {
            for (uint32_t i = 0; i < qcount; ++i) {
                simd::normalize_row_f32(&queries[i * qdim], qdim);
            }
        }
    }

    if (qdim != idx->dim()) {
        std::cerr << "tree-search: query dim " << qdim
                  << " != index dim " << idx->dim() << "\n";
        return 1;
    }

    const uint32_t k = p.get<uint32_t>("topk");
    uint32_t num_threads = p.get<uint32_t>("threads");
    if (num_threads == 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());

    SearchConfig scfg;
    scfg.k = k;
    scfg.n_probe = p.get<uint32_t>("n-probe");
    scfg.n_probe_ln = p.get<uint32_t>("n-probe-ln");
    scfg.probe_fraction = p.get<float>("probe-fraction");
    scfg.fastscan_W = p.get<uint32_t>("fastscan-w");
    scfg.rerank = !p.exist("no-rerank");
    scfg.adaptive_w_gap = p.get<float>("adaptive-w-gap");
    scfg.adaptive_probe_gap = p.get<float>("adaptive-probe-gap");
    scfg.search_threads = p.get<uint32_t>("search-threads");
    const uint32_t batch_window = p.get<uint32_t>("batch-window");
    if (batch_window > 0) {
        if (with_payload) {
            std::cerr << "tree-search: --batch-window is incompatible with "
                         "payload output (per-query mmap locations)\n";
            return 1;
        }
        // Batch mode has no cross-query thread pool: search_batch
        // parallelizes internally (route + sweep chunks).
        if (scfg.search_threads == 0) scfg.search_threads = num_threads;
    }

    // Exact-rerank base: mmap the original corpus read-only and hand the
    // engine a pointer (caller-owned, per the SearchConfig contract). The
    // mapping outlives all searches below.
    int erb_fd = -1;
    void* erb_map = nullptr;
    size_t erb_map_len = 0;
    {
        const std::string erb = p.get<std::string>("exact-rerank-base");
        if (!erb.empty()) {
            FbinHeader bh{};
            if (!read_fbin_header(erb, bh)) {
                std::cerr << "tree-search: cannot read '" << erb << "'\n";
                return 1;
            }
            if (bh.dim != idx->dim()) {
                std::cerr << "tree-search: exact-rerank-base dim " << bh.dim
                          << " != index dim " << idx->dim() << "\n";
                return 1;
            }
            erb_fd = ::open(erb.c_str(), O_RDONLY);
            if (erb_fd < 0) {
                std::cerr << "tree-search: open '" << erb << "' failed\n";
                return 1;
            }
            struct stat st{};
            if (::fstat(erb_fd, &st) != 0 ||
                static_cast<uint64_t>(st.st_size) <
                    8 + static_cast<uint64_t>(bh.n) * bh.dim * 4) {
                std::cerr << "tree-search: exact-rerank-base truncated\n";
                return 1;
            }
            erb_map_len = static_cast<size_t>(st.st_size);
            erb_map = ::mmap(nullptr, erb_map_len, PROT_READ, MAP_PRIVATE,
                             erb_fd, 0);
            if (erb_map == MAP_FAILED) {
                std::cerr << "tree-search: mmap of '" << erb << "' failed\n";
                return 1;
            }
            scfg.exact_rerank_base = reinterpret_cast<const float*>(
                static_cast<const uint8_t*>(erb_map) + 8);
            scfg.rerank = true;
        }
    }

    // Parse every collected --filter predicate (AND-composed).
    for (const std::string& fs : filter_strings) {
        Predicate pred;
        if (!parse_filter_predicate(fs, pred))
            return 1;
        std::cerr << "filter: " << pred.column << " " << fs << "\n";
        scfg.predicates.push_back(std::move(pred));
    }

    std::ifstream qf(p.get<std::string>("query"), std::ios::binary);
    qf.seekg(8);
    // (queries are read in bulk below for parallel processing)

    // Load ground truth if provided.
    //   .gtmm — canonical GTMM (see include/sextant/ground_truth.hpp):
    //          [magic][n][k][metric] + per-query [ids k×u32][dists k×f32].
    //          (The legacy headerless format is removed; the reader
    //          hard-errors on it.)
    //   .parquet — carquet-read: requires "neighbors_id" column (int64 list).
    //          Variable-length neighbor lists supported.
    std::vector<std::vector<RowId>> gt;
    const std::string gt_path = p.get<std::string>("ground-truth");
    if (!gt_path.empty()) {
        const bool is_parquet = gt_path.size() >= 8 &&
            gt_path.compare(gt_path.size() - 8, 8, ".parquet") == 0;
        if (is_parquet) {
            // --- Parquet GT via carquet ---
            carquet_error_t cerr = {};
            carquet_reader_t* raw_reader =
                carquet_reader_open(gt_path.c_str(), nullptr, &cerr);
            if (!raw_reader) {
                std::cerr << "tree-search: cannot open GT parquet '" << gt_path
                          << "'\n";
                return 1;
            }
            struct CarquetReaderCloser {
                void operator()(carquet_reader_t* r) const {
                    carquet_reader_close(r);
                }
            };
            std::unique_ptr<carquet_reader_t, CarquetReaderCloser> reader(
                raw_reader);

            const carquet_schema_t* fs = carquet_reader_schema(reader.get());
            int32_t gt_col = carquet_schema_find_column(fs, "neighbors_id");
            if (gt_col < 0) {
                // For list columns, find_column matches the leaf name (e.g. "item"),
                // not the parent field name. Fall back to path-prefix matching.
                int32_t n_gt_cols = carquet_schema_num_columns(fs);
                for (int32_t i = 0; i < n_gt_cols; ++i) {
                    const char* path[8];
                    int32_t depth = carquet_schema_column_path(fs, i, path, 8);
                    if (depth > 0 && path[0] &&
                        std::string("neighbors_id") == path[0]) {
                        gt_col = i;
                        break;
                    }
                }
            }
            if (gt_col < 0) {
                std::cerr << "tree-search: GT parquet has no 'neighbors_id' "
                             "column\n";
                return 1;
            }
            const int64_t total_rows = carquet_reader_num_rows(reader.get());

            carquet_batch_reader_config_t bcfg;
            carquet_batch_reader_config_init(&bcfg);
            bcfg.batch_size = 10000;
            carquet_batch_reader_t* raw_br = carquet_batch_reader_create(
                reader.get(), &bcfg, &cerr);
            if (!raw_br) {
                std::cerr << "tree-search: GT batch reader failed\n";
                return 1;
            }
            struct CarquetBRFree {
                void operator()(carquet_batch_reader_t* r) const {
                    carquet_batch_reader_free(r);
                }
            };
            std::unique_ptr<carquet_batch_reader_t, CarquetBRFree> br(raw_br);

            int64_t loaded = 0;
            while (true) {
                carquet_row_batch_t* batch = nullptr;
                carquet_status_t s = carquet_batch_reader_next(br.get(), &batch);
                if (s == CARQUET_ERROR_END_OF_DATA || !batch) break;
                if (s != CARQUET_OK) {
                    std::cerr << "tree-search: GT batch read error\n";
                    return 1;
                }
                const int32_t* offsets = nullptr;
                int64_t n_lists = 0;
                const void* values = nullptr;
                const uint8_t* val_valid = nullptr;
                int64_t n_vals = 0;
                const uint8_t* list_valid = nullptr;
                s = carquet_row_batch_column_list(batch, gt_col, &offsets,
                                                   &n_lists, &values,
                                                   &val_valid, &n_vals,
                                                   &list_valid);
                if (s != CARQUET_OK) {
                    std::cerr << "tree-search: GT list column read error\n";
                    carquet_row_batch_free(batch);
                    return 1;
                }
                const int64_t* vals64 = static_cast<const int64_t*>(values);
                for (int64_t i = 0; i < n_lists; ++i) {
                    const int32_t begin = offsets[i];
                    const int32_t end = offsets[i + 1];
                    std::vector<RowId> neighbors;
                    neighbors.reserve(end - begin);
                    for (int32_t j = begin; j < end; ++j)
                        neighbors.push_back(
                            static_cast<RowId>(vals64[j]));
                    gt.push_back(std::move(neighbors));
                }
                loaded += n_lists;
                carquet_row_batch_free(batch);
            }
            (void)total_rows;
            std::cerr << "loaded parquet ground truth: " << loaded
                      << " queries\n";
        } else {
            std::ifstream gtf(gt_path, std::ios::binary);
            if (gtf) {
                // GTMM format: [magic "GTMM":4B][n:u32][k:u32][metric:u8]
                //               [ids_q0 (k×u32)][dists_q0 (k×f32)]
                //               [ids_q1]... per query (interleaved).
                constexpr uint32_t kGtMagic = 0x4D4D5447u;  // "GTMM" LE
                uint32_t magic = 0;
                gtf.read(reinterpret_cast<char*>(&magic), 4);
                if (magic != kGtMagic) {
                    throw std::runtime_error(
                        "Ground truth file missing GTMM magic. "
                        "Legacy format removed — regenerate with GTMM header.");
                }
                uint32_t gt_n = 0, gt_k = 0;
                gtf.read(reinterpret_cast<char*>(&gt_n), 4);
                gtf.read(reinterpret_cast<char*>(&gt_k), 4);
                uint8_t metric_byte = 0;
                gtf.read(reinterpret_cast<char*>(&metric_byte), 1);
                gt.resize(gt_n);
                std::vector<uint32_t> row(gt_k);
                for (uint32_t i = 0; i < gt_n; ++i) {
                    gtf.read(reinterpret_cast<char*>(row.data()),
                             gt_k * sizeof(uint32_t));
                    gt[i].assign(row.begin(), row.end());
                    gtf.seekg(gt_k * sizeof(float), std::ios::cur);
                }
                std::cerr << "loaded ground truth: " << gt_n << " queries, k="
                          << gt_k << "\n";
            }
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

    // Queries are already loaded above (from .fbin or .parquet).

    std::vector<std::vector<Candidate>> all_results(qcount);
    using PayloadLoc = std::pair<const uint8_t*, uint32_t>;
    std::vector<std::vector<PayloadLoc>> all_payload_locs;
    if (with_payload) all_payload_locs.resize(qcount);

    const uint32_t passes = std::max(1u, p.get<uint32_t>("passes"));
    auto run_pass = [&]() {
        if (batch_window > 0) {
            // Subtree-major: one search_batch call per window; unique
            // leaves swept once, page-ordered. Results land in the same
            // all_results slots as the per-query path (recall + output
            // code shared).
            std::vector<std::vector<Candidate>> win_results;
            for (uint32_t w = 0; w < qcount; w += batch_window) {
                const uint32_t n =
                    std::min(batch_window, qcount - w);
                idx->search_batch(
                    &queries[static_cast<size_t>(w) * qdim], n, k, scfg,
                    win_results);
                for (uint32_t i = 0; i < n; ++i)
                    all_results[w + i] = std::move(win_results[i]);
            }
            return;
        }
        if (num_threads <= 1) {
            for (uint32_t qi = 0; qi < qcount; ++qi) {
                if (with_payload) {
                    all_results[qi] = idx->search(
                        &queries[static_cast<size_t>(qi) * qdim], k, scfg,
                        &all_payload_locs[qi]);
                } else {
                    all_results[qi] = idx->search(
                        &queries[static_cast<size_t>(qi) * qdim], k, scfg);
                }
            }
        } else {
            // Query-level parallelism: each query is independent.
            std::vector<std::future<void>> futs;
            std::atomic<uint32_t> next_qi{0};
            for (uint32_t t = 0; t < num_threads; ++t) {
                futs.push_back(std::async(std::launch::async, [&]() {
                    while (true) {
                        const uint32_t qi = next_qi.fetch_add(1);
                        if (qi >= qcount) break;
                        if (with_payload) {
                            all_results[qi] = idx->search(
                                &queries[static_cast<size_t>(qi) * qdim], k,
                                scfg, &all_payload_locs[qi]);
                        } else {
                            all_results[qi] = idx->search(
                                &queries[static_cast<size_t>(qi) * qdim], k,
                                scfg);
                        }
                    }
                }));
            }
            for (auto& f : futs) f.get();
        }
    };

    // Discarded warmup pass (BENCHMARK_RULES): with --passes>=2 the first
    // pass fills the engine caches and is not timed.
    if (passes > 1) {
        run_pass();
        (void)idx->search_stats().snapshot_and_reset();
    }

    const auto t0 = std::chrono::steady_clock::now();
    struct rusage ru0{};
    getrusage(RUSAGE_SELF, &ru0);
    const auto cpu0 = static_cast<double>(ru0.ru_utime.tv_sec) +
                      1e-6 * static_cast<double>(ru0.ru_utime.tv_usec) +
                      static_cast<double>(ru0.ru_stime.tv_sec) +
                      1e-6 * static_cast<double>(ru0.ru_stime.tv_usec);

    for (uint32_t pass = 0; pass < (passes > 1 ? passes - 1 : passes); ++pass)
        run_pass();

    // Tally recall + emit results (serial — I/O bound).
    // Recall metric: fraction of search top-k results that appear in the
    // GT top-k (not the full GT list). This is the standard recall@k:
    // "of the k results returned, how many are true top-k neighbors."
    // (k, not results.size(): the adaptive-W path returns variable-length
    // shortlists — hits anywhere in the returned list count, denominator k.)
    const uint32_t recall_k = std::min(k,
        gt.empty() ? 0 : static_cast<uint32_t>(gt[0].size()));
    for (uint32_t qi = 0; qi < qcount; ++qi) {
        const auto& results = all_results[qi];
        if (!gt.empty() && qi < gt.size()) {
            // Only check against the first recall_k GT entries (true top-k).
            std::unordered_set<RowId> gt_set(
                gt[qi].begin(),
                gt[qi].begin() + std::min(recall_k, static_cast<uint32_t>(gt[qi].size())));
            for (const auto& c : results) {
                if (gt_set.count(c.row_id)) ++total_hits;
            }
            ++total_queries;
        }
        for (uint32_t ri = 0; ri < results.size(); ++ri) {
            const auto& c = results[ri];
            *out << qi << '\t' << c.row_id << '\t' << c.dist;
            if (with_payload) {
                auto blob = idx->fetch_payload(all_payload_locs[qi][ri].first,
                                                all_payload_locs[qi][ri].second);
                *out << '\t' << blob.size();
                // Hex-encode the payload (handles arbitrary binary content).
                for (char bch : blob) {
                    const uint8_t b = static_cast<uint8_t>(bch);
                    *out << "0123456789abcdef"[b >> 4]
                         << "0123456789abcdef"[b & 0x0f];
                }
            }
            *out << '\n';
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double qps = (secs > 0) ? qcount / secs : 0;

    // Search window metrics: engine counters (queries/leaves/bytes_touched/
    // rerank) + window CPU via rusage (accurate at window granularity;
    // per-query rusage under concurrency would misattribute).
    struct rusage ru1{};
    getrusage(RUSAGE_SELF, &ru1);
    const double cpu1 = static_cast<double>(ru1.ru_utime.tv_sec) +
                        1e-6 * static_cast<double>(ru1.ru_utime.tv_usec) +
                        static_cast<double>(ru1.ru_stime.tv_sec) +
                        1e-6 * static_cast<double>(ru1.ru_stime.tv_usec);
    SearchStats::Snapshot st = idx->search_stats().snapshot_and_reset();
    BatchStats::Snapshot bst = idx->batch_stats().snapshot_and_reset();
    const double cpu_win = cpu1 - cpu0;
    const double util = secs > 0 ? cpu_win / secs : 0.0;
    std::cerr << "metrics: queries=" << st.queries
              << " leaves/query=" << (st.queries ? double(st.leaves_probed) / st.queries : 0.0)
              << " bytes/query=" << (st.queries ? double(st.bytes_touched) / st.queries : 0.0)
              << " shortlist/query=" << (st.queries ? double(st.rerank_count) / st.queries : 0.0);
    {
        const uint64_t cache_ops = st.cache_hits + st.cache_misses;
        if (cache_ops > 0) {
            const uint64_t cache_mb = p.get<uint32_t>("cache-mb");
            const double hit_rate =
                static_cast<double>(st.cache_hits) /
                static_cast<double>(cache_ops);
            std::cerr << " LeafCache=" << cache_mb << "MB"
                      << " hit=" << (100.0 * hit_rate) << "%"
                      << " disk_bytes/query="
                      << (st.queries ? double(st.cache_bytes_filled) / st.queries : 0.0);
        }
    }
    std::cerr << " cpu=" << cpu_win << "s util=" << (100.0 * util) << "%";
    if (st.wall_seconds > 0) {
        const double rshare = static_cast<double>(st.routing_ns) / 1e9 /
                              st.wall_seconds;
        std::cerr << " routing=" << (100.0 * rshare) << "%"
                  << " node_bytes/query="
                  << (st.queries ? double(st.node_bytes_read) / st.queries
                                 : 0.0);
    }
    std::cerr << "\n";
    double read_stream_gbps = 0, uncoalesced_capacity_qps = 0;
    if (bst.batches > 0) {
        const double fanout = bst.leaves_unique > 0
            ? static_cast<double>(bst.leaf_scans) / bst.leaves_unique
            : 0.0;
        // read_stream_gbps = bytes / summed per-leaf pread wall = mean
        // PER-STREAM sequential bandwidth. uncoalesced_capacity_qps (the
        // immediate-class request-rate ceiling) is only derivable from
        // singleton windows: 1/mean-lonely-sweep-wall; multi-query
        // windows report 0 (measure via scripts/sched_transition rate-1).
        read_stream_gbps =
            bst.read_ns > 0
                ? static_cast<double>(bst.bytes_unique) /
                      static_cast<double>(bst.read_ns)
                : 0.0;
        uncoalesced_capacity_qps =
            (bst.batches > 0 && bst.queries == bst.batches && secs > 0
                 ? static_cast<double>(st.queries) / secs
                 : 0.0);
        std::cerr << "batch: windows=" << bst.batches
                  << " leaves_unique=" << bst.leaves_unique
                  << " fanout=" << fanout
                  << " unique_bytes/query="
                  << (st.queries
                          ? double(bst.bytes_unique) / st.queries : 0.0)
                  << " coalescing="
                  << (bst.bytes_unique > 0
                          ? double(st.bytes_touched) / bst.bytes_unique
                          : 0.0)
                  << "x"
                  << " read_stream=" << read_stream_gbps << "GB/s"
                  << " uncoalesced_capacity=" << uncoalesced_capacity_qps
                  << "QPS\n";
    }
    {
        const std::string mf = p.get<std::string>("metrics-file");
        if (!mf.empty()) {
            metrics::JsonlMetricsSink sink(mf);
            metrics::PhaseMetrics w;
            w.phase = "window";
            w.source = batch_window > 0 ? "tree-search-batch" : "tree-search";
            w.wall_seconds = st.wall_seconds;
            w.cpu_seconds = cpu_win;
            // bytes_touched is the window's leaf traffic; wait counters n/a.
            w.bytes_read = st.bytes_touched;
            w.wait_count = st.queries;
            w.cache_hits = st.cache_hits;
            w.cache_misses = st.cache_misses;
            w.cache_bytes_filled = st.cache_bytes_filled;
            w.routing_seconds =
                static_cast<double>(st.routing_ns) / 1e9;
            w.node_bytes_read = st.node_bytes_read;
            w.batches = bst.batches;
            w.leaves_unique = bst.leaves_unique;
            w.leaf_scans = bst.leaf_scans;
            w.bytes_unique = bst.bytes_unique;
            w.read_ns = bst.read_ns;
            w.scan_ns = bst.scan_ns;
            w.sweep_wall_seconds =
                static_cast<double>(bst.sweep_wall_ns) / 1e9;
            w.read_stream_gbps = read_stream_gbps;
            w.uncoalesced_capacity_qps = uncoalesced_capacity_qps;
            w.timestamp = std::chrono::duration<double>(
                t0.time_since_epoch()).count();
            sink.emit(w);
        }
    }

    std::cerr << "\n═══ Tree Search Results ═══\n";
    std::cerr << "queries: " << qcount << "\n";
    std::cerr << "k: " << k << "\n";
    std::cerr << "time: " << secs << "s (" << qps << " QPS)\n";
    {
        uint64_t len_sum = 0;
        for (const auto& r : all_results) len_sum += r.size();
        if (!all_results.empty()) {
            const double mean_len =
                static_cast<double>(len_sum) / all_results.size();
            std::cerr << "mean results/query: " << mean_len << "\n";
            // With --adaptive-w-gap the engine deliberately returns
            // variable-length shortlists of up to W ids (gap-truncated
            // past k) — mean > k is expected, not a bug.
            if (mean_len > k + 0.5 && scfg.adaptive_w_gap > 0.0f)
                std::cerr << "  (adaptive-W shortlists: up to "
                          << scfg.fastscan_W << " ids, gap-cut past k="
                          << k << ")\n";
        }
    }
    if (total_queries > 0) {
        const float recall = float(total_hits) / (total_queries * k);
        std::cerr << "recall@" << k << ": " << recall << "\n";
    }
    if (erb_map && erb_map != MAP_FAILED) ::munmap(erb_map, erb_map_len);
    if (erb_fd >= 0) ::close(erb_fd);
    return 0;
}

// ---------------------------------------------------------------------------
// sweep: n-probe × rerank-shortlist (W) recall/QPS grid.
//
// Exploits W-sharing: one search() per query with sweep_Ws = all W values
// scans once at W_max and evaluates every W as a prefix cut (bit-exact
// equivalent to a standalone search per W). The grid costs one scan per
// n-probe instead of one scan per (n-probe, W) pair — ~L× cheaper for L W
// values. shared_qps is the throughput of that shared scan+rerank pass (the
// per-W prefix selection is O(W log W), negligible by comparison).
// ---------------------------------------------------------------------------

/// Parse a comma-separated list of unsigned integers ("4,8,16").
static std::vector<uint32_t> parse_u32_list(const std::string& s) {
    std::vector<uint32_t> vals;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        vals.push_back(static_cast<uint32_t>(std::stoul(tok)));
    }
    return vals;
}

int cmd_sweep(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("index", 0, "Tree index file", true);
    p.add<std::string>("query", 0, "Query vectors (.fbin)", true);
    p.add<std::string>("ground-truth", 0, "Ground-truth .gtmm file", true);
    p.add<uint32_t>("topk", 0, "K nearest neighbors", false, 10);
    p.add<std::string>("n-probe", 0,
        "Comma-separated root probe counts, e.g. \"4,8,16,32\" (required "
        "unless --feedback rows are used)", false, "");
    p.add<std::string>("fastscan-w", 0,
        "Comma-separated rerank shortlist widths, e.g. \"100,300,1000,3000\"",
        true);
    p.add<uint32_t>("n-probe-ln", 0,
        "Leaf probe count per root child (0=manifest default)", false, 0);
    p.add<std::string>("probe-fraction", 0,
        "Comma-separated corpus fractions swept as an extra row set, e.g. "
        "\"0.25,0.5\" (empty = none; fraction rows probe root children "
        "nearest-first to this cumulative extent, all their leaves)",
        false, "");
    p.add("no-rerank", 0, "Disable FP32 rerank (use raw PQ distances)");
    p.add<float>("adaptive-probe-gap", 0, "Geometric gap pruning (0=manifest)",
                 false, 0.0f);
    p.add<uint32_t>("threads", 0, "Query-parallel threads (0=auto)", false, 0);
    p.add<uint32_t>("search-threads", 0,
        "Within-query leaf-parallel scan threads (0=serial)", false, 0);
    p.add<uint32_t>("dump-results", 0,
        "Debug: dump per-query result row_ids for this W "
        "(first n-probe, forces serial)", false, 0);
    p.add<std::string>("filter", 0,
        "(unsupported — sweep runs unfiltered)", false, "");
    p.add<std::string>("feedback", 0,
        "Scan-feedback probing rows: comma-separated specs "
        "fixed:F | stall:M | kth:M (optional ,min:N suffix), e.g. "
        "\"stall:2,kth:3\". Each spec is one grid row (labelled fb:<spec>). "
        "Ignored unless the tree is depth<=2 and --n-probe row is 0-only; "
        "feedback requires n_probe=0",
        false, "");
    p.add<std::string>("trace-file", 0,
        "EngineTrace output path for feedback rows (one feedback row only "
        "when tracing). Records are analyzed by `sextant trace`",
        false, "");
    p.add<std::string>("output", 0, "Output file (default stdout)", false, "");
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");

    // No predicates in sweep mode. The engine API supports sweeping with
    // predicates (filtering commutes with the prefix cut), but sweeping under
    // a fixed selectivity changes the W semantics — keep the tool unfiltered.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--filter" || arg.rfind("--filter=", 0) == 0) {
            std::cerr << "sweep: --filter is not supported (run unfiltered)\n";
            return 1;
        }
    }

    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") set_log_level(LogLevel::Debug);
        else if (lvl == "warn") set_log_level(LogLevel::Warn);
        else if (lvl == "error") set_log_level(LogLevel::Error);
    }

    const uint32_t k = p.get<uint32_t>("topk");
    std::vector<uint32_t> n_probes = parse_u32_list(p.get<std::string>("n-probe"));
    std::vector<uint32_t> w_vals = parse_u32_list(p.get<std::string>("fastscan-w"));
    if (w_vals.empty()) {
        std::cerr << "sweep: --fastscan-w must be a non-empty "
                     "comma-separated list\n";
        return 1;
    }
    if (n_probes.empty() && p.get<std::string>("feedback").empty()
        && p.get<std::string>("probe-fraction").empty()) {
        std::cerr << "sweep: --n-probe is required unless --probe-fraction "
                     "or --feedback is used\n";
        return 1;
    }
    // Ascending, deduped W list (the engine evaluates prefix cuts in scan
    // order; the output rows stay sorted by W).
    std::sort(w_vals.begin(), w_vals.end());
    w_vals.erase(std::unique(w_vals.begin(), w_vals.end()), w_vals.end());
    const uint32_t dump_w = p.get<uint32_t>("dump-results");
    const size_t dump_idx = dump_w
        ? std::find(w_vals.begin(), w_vals.end(), dump_w) - w_vals.begin()
        : static_cast<size_t>(-1);
    if (dump_w && dump_idx >= w_vals.size()) {
        std::cerr << "sweep: --dump-results W " << dump_w
                  << " is not in the --fastscan-w list\n";
        return 1;
    }
    const bool dump = dump_w != 0;

    auto idx = tree::IVFTreeIndex::open(p.get<std::string>("index"));

    // Load queries (.fbin only — sweep is a benchmarking tool).
    const std::string query_path = p.get<std::string>("query");
    FbinHeader qh;
    if (!read_fbin_header(query_path, qh) || qh.dim != idx->dim()) {
        std::cerr << "sweep: invalid query file (dim=" << qh.dim
                  << ", expected " << idx->dim() << ")\n";
        return 1;
    }
    const uint32_t qdim = qh.dim;
    const uint32_t qcount = qh.n;
    std::vector<float> queries(static_cast<size_t>(qcount) * qdim);
    {
        std::ifstream qf(query_path, std::ios::binary);
        qf.seekg(8);
        qf.read(reinterpret_cast<char*>(queries.data()),
                static_cast<std::streamsize>(qcount) * qdim * sizeof(float));
        if (!qf.good()) {
            std::cerr << "sweep: failed to read query vectors\n";
            return 1;
        }
    }
    if (idx->metric() == MetricKind::InnerProduct) {
        for (uint32_t i = 0; i < qcount; ++i)
            simd::normalize_row_f32(&queries[static_cast<size_t>(i) * qdim],
                                    qdim);
    }

    // Load ground truth (GTMM binary — same layout as tree-search):
    // [magic "GTMM":4B][n:u32][k:u32][metric:u8][ids (k×u32)][dists (k×f32)]...
    std::vector<std::vector<RowId>> gt;
    {
        std::ifstream gtf(p.get<std::string>("ground-truth"), std::ios::binary);
        if (!gtf) {
            std::cerr << "sweep: cannot open ground truth\n";
            return 1;
        }
        constexpr uint32_t kGtMagic = 0x4D4D5447u;  // "GTMM" LE
        uint32_t magic = 0;
        gtf.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != kGtMagic) {
            std::cerr << "sweep: ground truth missing GTMM magic\n";
            return 1;
        }
        uint32_t gt_n = 0, gt_k = 0;
        gtf.read(reinterpret_cast<char*>(&gt_n), 4);
        gtf.read(reinterpret_cast<char*>(&gt_k), 4);
        uint8_t metric_byte = 0;
        gtf.read(reinterpret_cast<char*>(&metric_byte), 1);
        (void)metric_byte;
        gt.resize(gt_n);
        std::vector<uint32_t> row(gt_k);
        for (uint32_t i = 0; i < gt_n; ++i) {
            gtf.read(reinterpret_cast<char*>(row.data()),
                     static_cast<std::streamsize>(gt_k) * sizeof(uint32_t));
            gt[i].assign(row.begin(), row.end());
            gtf.seekg(static_cast<std::streamoff>(gt_k) * sizeof(float),
                      std::ios::cur);
        }
        std::cerr << "loaded ground truth: " << gt_n << " queries, k="
                  << gt_k << "\n";
    }
    if (gt.empty()) {
        std::cerr << "sweep: ground truth is empty\n";
        return 1;
    }
    // Recall metric (identical to tree-search): fraction of result top-k
    // row_ids present in the GT top-k.
    const uint32_t recall_k = std::min(k, static_cast<uint32_t>(gt[0].size()));
    const uint32_t scored = std::min(qcount,
                                     static_cast<uint32_t>(gt.size()));

    uint32_t num_threads = p.get<uint32_t>("threads");
    if (num_threads == 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    if (dump) num_threads = 1;  // deterministic, inspectable dump pass

    SearchConfig scfg;
    scfg.k = k;
    scfg.n_probe_ln = p.get<uint32_t>("n-probe-ln");
    scfg.probe_fraction = 0.0f;  // np sweep rows; fraction rows set it per-row
    scfg.rerank = !p.exist("no-rerank");
    scfg.adaptive_probe_gap = p.get<float>("adaptive-probe-gap");
    scfg.search_threads = p.get<uint32_t>("search-threads");

    // Feedback rows (see parse_feedback_spec). When tracing, exactly one
    // feedback row is allowed (the trace's query sequence maps 1:1 to rows).
    std::vector<std::pair<std::string, FeedbackProbe>> fb_rows;
    try {
        fb_rows = sextant_cli::parse_feedback_list(
            p.get<std::string>("feedback"));
    } catch (const std::runtime_error& e) {
        std::cerr << "sweep: " << e.what() << '\n';
        return 1;
    }
    std::unique_ptr<EngineTrace> trace;
    if (!p.get<std::string>("trace-file").empty()) {
        if (fb_rows.size() != 1) {
            std::cerr << "sweep: --trace-file requires exactly one "
                         "--feedback spec\n";
            return 1;
        }
        trace = EngineTrace::create(p.get<std::string>("trace-file"),
                                    "sweep " + p.get<std::string>("index"));
        if (!trace) {
            std::cerr << "sweep: cannot open trace file '"
                      << p.get<std::string>("trace-file") << "'\n";
            return 1;
        }
        scfg.trace = trace.get();
    }

    std::ofstream out_file;
    std::ostream* out = &std::cout;
    if (!p.get<std::string>("output").empty()) {
        out_file.open(p.get<std::string>("output"));
        out = &out_file;
    }
    *out << "# sweep grid — W evaluation is shared: one scan+rerank per "
            "(n-probe) serves all W; shared_qps is that shared pass\n";
    *out << "n_probe\tW\trecall@" << k << "\tshared_qps\n";
    std::cerr << "sweep: " << n_probes.size() << " n-probe x " << w_vals.size()
              << " W values, " << scored << " queries\n";

    // Optional fraction rows extend the grid: the same sweep body runs
    // with scfg.probe_fraction set and n_probe = 0 so the fraction path
    // is active. Labelled f<f> in the n_probe column.
    std::vector<std::pair<float, std::string>> frac_rows;
    {
        const std::string fs = p.get<std::string>("probe-fraction");
        if (!fs.empty()) {
            std::string item;
            std::istringstream iss(fs);
            while (std::getline(iss, item, ',')) {
                if (item.empty()) continue;
                const float f = std::stof(item);
                if (f <= 0.0f || f > 1.0f) {
                    std::cerr << "sweep: --probe-fraction values must be in "
                                 "(0, 1] (got " << item << ")\n";
                    return 1;
                }
                frac_rows.emplace_back(f, item);
            }
        }
    }

    for (uint32_t np : n_probes) {
        scfg.n_probe = np;
        scfg.probe_fraction = 0.0f;
        std::vector<std::atomic<uint64_t>> hits(w_vals.size());
        for (auto& h : hits) h.store(0, std::memory_order_relaxed);
        std::vector<std::vector<Candidate>> dump_rows;
        if (dump) dump_rows.resize(qcount);

        // Score one query's sweep output against the GT top-k set.
        auto score = [&](uint32_t qi,
                         const std::vector<std::vector<Candidate>>& so) {
            std::unordered_set<RowId> gt_set(
                gt[qi].begin(),
                gt[qi].begin() + std::min(recall_k,
                                          static_cast<uint32_t>(gt[qi].size())));
            for (size_t wi = 0; wi < w_vals.size(); ++wi) {
                if (wi >= so.size()) break;
                for (const auto& c : so[wi])
                    if (gt_set.count(c.row_id))
                        hits[wi].fetch_add(1, std::memory_order_relaxed);
            }
        };

        const auto t0 = std::chrono::steady_clock::now();
        if (num_threads <= 1) {
            for (uint32_t qi = 0; qi < qcount; ++qi) {
                std::vector<std::vector<Candidate>> sweep_out;
                idx->search(&queries[static_cast<size_t>(qi) * qdim], k, scfg,
                            nullptr, &w_vals, &sweep_out);
                if (qi < scored) score(qi, sweep_out);
                if (dump) dump_rows[qi] = sweep_out[dump_idx];
            }
        } else {
            // Query-level parallelism (same pattern as tree-search): each
            // worker owns its sweep_out; the thread-local scratch arena
            // inside search() is per-worker automatically.
            std::vector<std::future<void>> futs;
            std::atomic<uint32_t> next_qi{0};
            for (uint32_t t = 0; t < num_threads; ++t) {
                futs.push_back(std::async(std::launch::async, [&]() {
                    while (true) {
                        const uint32_t qi = next_qi.fetch_add(1);
                        if (qi >= qcount) break;
                        std::vector<std::vector<Candidate>> sweep_out;
                        idx->search(&queries[static_cast<size_t>(qi) * qdim],
                                    k, scfg, nullptr, &w_vals, &sweep_out);
                        if (qi < scored) score(qi, sweep_out);
                    }
                }));
            }
            for (auto& f : futs) f.get();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double qps = (secs > 0) ? qcount / secs : 0;

        for (size_t wi = 0; wi < w_vals.size(); ++wi) {
            const float recall = static_cast<float>(hits[wi].load())
                                 / (static_cast<double>(scored) * k);
            *out << np << '\t' << w_vals[wi] << '\t' << recall << '\t'
                 << qps << '\n';
        }
        std::cerr << "n_probe=" << np << ": " << secs << "s (" << qps
                  << " shared QPS)\n";

        if (dump && np == n_probes[0]) {
            for (uint32_t qi = 0; qi < qcount; ++qi)
                for (const auto& c : dump_rows[qi])
                    std::cerr << "dump\t" << qi << '\t' << c.row_id << '\t'
                              << c.dist << '\n';
        }
    }

    for (const auto& [frac, label] : frac_rows) {
        scfg.n_probe = 0;             // fraction path requires no absolute np
        scfg.probe_fraction = frac;
        std::vector<std::atomic<uint64_t>> hits(w_vals.size());
        for (auto& h : hits) h.store(0, std::memory_order_relaxed);

        auto score = [&](uint32_t qi,
                         const std::vector<std::vector<Candidate>>& so) {
            std::unordered_set<RowId> gt_set(
                gt[qi].begin(),
                gt[qi].begin() + std::min(recall_k,
                                          static_cast<uint32_t>(gt[qi].size())));
            for (size_t wi = 0; wi < w_vals.size(); ++wi) {
                if (wi >= so.size()) break;
                for (const auto& c : so[wi])
                    if (gt_set.count(c.row_id))
                        hits[wi].fetch_add(1, std::memory_order_relaxed);
            }
        };

        const auto t0 = std::chrono::steady_clock::now();
        if (num_threads <= 1) {
            for (uint32_t qi = 0; qi < qcount; ++qi) {
                std::vector<std::vector<Candidate>> sweep_out;
                idx->search(&queries[static_cast<size_t>(qi) * qdim], k, scfg,
                            nullptr, &w_vals, &sweep_out);
                if (qi < scored) score(qi, sweep_out);
            }
        } else {
            std::vector<std::future<void>> futs;
            std::atomic<uint32_t> next_qi{0};
            for (uint32_t t = 0; t < num_threads; ++t) {
                futs.push_back(std::async(std::launch::async, [&]() {
                    while (true) {
                        const uint32_t qi = next_qi.fetch_add(1);
                        if (qi >= qcount) break;
                        std::vector<std::vector<Candidate>> sweep_out;
                        idx->search(&queries[static_cast<size_t>(qi) * qdim],
                                    k, scfg, nullptr, &w_vals, &sweep_out);
                        if (qi < scored) score(qi, sweep_out);
                    }
                }));
            }
            for (auto& f : futs) f.get();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double qps = (secs > 0) ? qcount / secs : 0;

        for (size_t wi = 0; wi < w_vals.size(); ++wi) {
            const float recall = static_cast<float>(hits[wi].load())
                                 / (static_cast<double>(scored) * k);
            *out << "f" << label << '\t' << w_vals[wi] << '\t' << recall
                 << '\t' << qps << '\n';
        }
        std::cerr << "probe_fraction=" << frac << ": " << secs << "s ("
                  << qps << " shared QPS)\n";
    }

    // Feedback rows: scan-feedback probing replaces fraction routing
    // (n_probe=0, probe_fraction=0). One row per spec, labelled fb:<spec>.
    // Within-query scan is inherently serial in feedback mode (sequential
    // block decisions); query-level parallelism still applies.
    for (auto& [spec, fb] : fb_rows) {
        scfg.n_probe = 0;
        scfg.probe_fraction = 0.0f;
        scfg.feedback = fb;
        std::vector<std::atomic<uint64_t>> hits(w_vals.size());
        for (auto& h : hits) h.store(0, std::memory_order_relaxed);

        auto score = [&](uint32_t qi,
                         const std::vector<std::vector<Candidate>>& so) {
            std::unordered_set<RowId> gt_set(
                gt[qi].begin(),
                gt[qi].begin() + std::min(recall_k,
                                          static_cast<uint32_t>(gt[qi].size())));
            for (size_t wi = 0; wi < w_vals.size(); ++wi) {
                if (wi >= so.size()) break;
                for (const auto& c : so[wi])
                    if (gt_set.count(c.row_id))
                        hits[wi].fetch_add(1, std::memory_order_relaxed);
            }
        };

        const auto t0 = std::chrono::steady_clock::now();
        if (num_threads <= 1) {
            for (uint32_t qi = 0; qi < qcount; ++qi) {
                std::vector<std::vector<Candidate>> sweep_out;
                idx->search(&queries[static_cast<size_t>(qi) * qdim], k, scfg,
                            nullptr, &w_vals, &sweep_out);
                if (qi < scored) score(qi, sweep_out);
            }
        } else {
            std::vector<std::future<void>> futs;
            std::atomic<uint32_t> next_qi{0};
            for (uint32_t t = 0; t < num_threads; ++t) {
                futs.push_back(std::async(std::launch::async, [&]() {
                    while (true) {
                        const uint32_t qi = next_qi.fetch_add(1);
                        if (qi >= qcount) break;
                        std::vector<std::vector<Candidate>> sweep_out;
                        idx->search(&queries[static_cast<size_t>(qi) * qdim],
                                    k, scfg, nullptr, &w_vals, &sweep_out);
                        if (qi < scored) score(qi, sweep_out);
                    }
                }));
            }
            for (auto& f : futs) f.get();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double qps = (secs > 0) ? qcount / secs : 0;

        for (size_t wi = 0; wi < w_vals.size(); ++wi) {
            const float recall = static_cast<float>(hits[wi].load())
                                 / (static_cast<double>(scored) * k);
            *out << "fb:" << spec << '\t' << w_vals[wi] << '\t' << recall
                 << '\t' << qps << '\n';
        }
        std::cerr << "feedback=" << spec << ": " << secs << "s ("
                  << qps << " shared QPS)\n";
        scfg.feedback = FeedbackProbe{};
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

// ---------------------------------------------------------------------------
// tree-insert: batch-insert vectors from a .fbin into an existing tree index.
// Row IDs are assigned sequentially starting from --start-row-id (default:
// current live_count).
// ---------------------------------------------------------------------------

int cmd_tree_insert(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("index", 0, "Tree index file", true);
    p.add<std::string>("vectors", 0, "Vectors to insert (.fbin)", true);
    p.add<int64_t>("start-row-id", 0,
        "Starting row ID for inserted vectors (default: current live count)",
        false, -1);
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");
    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") set_log_level(LogLevel::Debug);
        else if (lvl == "warn") set_log_level(LogLevel::Warn);
        else if (lvl == "error") set_log_level(LogLevel::Error);
    }

    const std::string index_path = p.get<std::string>("index");
    const std::string vec_path = p.get<std::string>("vectors");

    // Read the vector file.
    FbinHeader vh;
    if (!read_fbin_header(vec_path, vh)) {
        std::cerr << "tree-insert: cannot read " << vec_path << "\n";
        return 1;
    }

    auto idx = tree::IVFTreeIndex::open(index_path);
    if (vh.dim != idx->dim()) {
        std::cerr << "tree-insert: dim mismatch (file=" << vh.dim
                  << ", index=" << idx->dim() << ")\n";
        return 1;
    }

    // Determine starting row_id.
    int64_t start_rid = p.get<int64_t>("start-row-id");
    if (start_rid < 0)
        start_rid = static_cast<int64_t>(idx->live_count());

    // Read all vectors and build InsertPoints.
    const uint32_t dim = vh.dim;
    std::vector<float> storage(static_cast<size_t>(vh.n) * dim);
    {
        std::ifstream f(vec_path, std::ios::binary);
        f.seekg(8);  // skip header
        f.read(reinterpret_cast<char*>(storage.data()),
               static_cast<std::streamsize>(storage.size() * sizeof(float)));
        if (!f.good()) {
            std::cerr << "tree-insert: failed to read vectors\n";
            return 1;
        }
    }

    std::vector<tree::IVFTreeIndex::InsertPoint> points;
    points.reserve(vh.n);
    for (uint32_t i = 0; i < vh.n; ++i) {
        points.push_back({&storage[static_cast<size_t>(i) * dim],
                          static_cast<RowId>(start_rid + i), {}, {}});
    }

    const uint64_t before = idx->live_count();
    idx->insert_batch(points);
    const uint64_t after = idx->live_count();

    std::cout << "inserted " << (after - before) << " vectors into " << index_path
              << " (count: " << before << " → " << after
              << ", leaves: " << idx->n_leaves() << ")\n";
    return 0;
}

// ---------------------------------------------------------------------------
// tree-delete: delete vectors by row ID from an existing tree index.
// Row IDs are read from a text file (one per line) or a binary .uids file
// (uint64, little-endian).
// ---------------------------------------------------------------------------

int cmd_tree_delete(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("index", 0, "Tree index file", true);
    p.add<std::string>("row-ids", 0,
        "Row IDs to delete. Text file (one per line) or binary .uids "
        "(uint64 LE)", true);
    p.add<std::string>("format", 0, "text or binary", false, "text");
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");
    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") set_log_level(LogLevel::Debug);
        else if (lvl == "warn") set_log_level(LogLevel::Warn);
        else if (lvl == "error") set_log_level(LogLevel::Error);
    }

    const std::string index_path = p.get<std::string>("index");
    const std::string rids_path = p.get<std::string>("row-ids");
    const std::string fmt = p.get<std::string>("format");

    // Read row IDs.
    std::vector<RowId> row_ids;
    if (fmt == "binary") {
        std::ifstream f(rids_path, std::ios::binary);
        if (!f) {
            std::cerr << "tree-delete: cannot open " << rids_path << "\n";
            return 1;
        }
        uint64_t rid;
        while (f.read(reinterpret_cast<char*>(&rid), sizeof(rid)))
            row_ids.push_back(static_cast<RowId>(rid));
    } else {
        std::ifstream f(rids_path);
        if (!f) {
            std::cerr << "tree-delete: cannot open " << rids_path << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            // Trim whitespace.
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            if (line.empty()) continue;
            try {
                row_ids.push_back(static_cast<RowId>(std::stoll(line)));
            } catch (...) {
                std::cerr << "tree-delete: skipping invalid row ID: " << line << "\n";
            }
        }
    }

    auto idx = tree::IVFTreeIndex::open(index_path);
    const uint64_t before = idx->live_count();
    idx->delete_batch(row_ids);
    const uint64_t after = idx->live_count();

    std::cout << "deleted " << (before - after) << " of " << row_ids.size()
              << " row IDs from " << index_path
              << " (count: " << before << " → " << after << ")\n";
    return 0;
}

// ---------------------------------------------------------------------------
// tree-vacuum: repair stale filter summaries after deletes.
// ---------------------------------------------------------------------------

int cmd_tree_vacuum(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("index", 0, "Tree index file", true);
    p.add("rebuild-cardinality", 0,
        "Rebuild cardinality table (re-scans all filter columns)");
    p.add<uint32_t>("batch-size", 0, "Leaves per commit batch", false, 256);
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");
    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") set_log_level(LogLevel::Debug);
        else if (lvl == "warn") set_log_level(LogLevel::Warn);
        else if (lvl == "error") set_log_level(LogLevel::Error);
    }

    const std::string index_path = p.get<std::string>("index");

    auto idx = tree::IVFTreeIndex::open(index_path);
    tree::IVFTreeIndex::VacuumConfig cfg;
    cfg.rebuild_cardinality = p.exist("rebuild-cardinality");
    cfg.batch_size = p.get<uint32_t>("batch-size");
    auto result = idx->vacuum(cfg);

    std::cout << "vacuum: " << result.summaries_repaired
              << " summaries repaired (" << result.leaves_scanned
              << " leaves scanned)";
    if (cfg.rebuild_cardinality)
        std::cout << ", " << result.cardinality_entries_rebuilt
                  << " cardinality entries rebuilt";
    std::cout << " in " << std::fixed << std::setprecision(3)
              << result.elapsed_sec << "s\n";
    return 0;
}

// ---------------------------------------------------------------------------
// tree-defrag: compact fragmented leaf extents + shrink file.
// ---------------------------------------------------------------------------

int cmd_tree_defrag(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("index", 0, "Tree index file", true);
    p.add("no-shrink", 0, "Skip file truncation");
    p.add<uint32_t>("batch-size", 0, "Leaves per commit batch", false, 256);
    p.add<std::string>("log-level", 0, "debug/info/warn/error", false, "info");
    p.parse_check(argc, argv);

    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") set_log_level(LogLevel::Debug);
        else if (lvl == "warn") set_log_level(LogLevel::Warn);
        else if (lvl == "error") set_log_level(LogLevel::Error);
    }

    const std::string index_path = p.get<std::string>("index");

    auto idx = tree::IVFTreeIndex::open(index_path);
    tree::IVFTreeIndex::DefragConfig cfg;
    cfg.shrink_file = !p.exist("no-shrink");
    cfg.batch_size = p.get<uint32_t>("batch-size");
    auto result = idx->defrag(cfg);

    std::cout << "defrag: " << result.leaves_relocated
              << " leaves relocated, " << result.pages_reclaimed
              << " pages reclaimed in " << std::fixed << std::setprecision(3)
              << result.elapsed_sec << "s\n";
    return 0;
}
// trace: read an EngineTrace file and analyze scan-feedback probe traces.
// Replays stop rules (fixed fractions, stall/kth thresholds) over the
// per-query block sequences recorded by a full-probe run and reports
// mean/p50/p99 probed fraction plus pq-level recall@k (with --ground-truth).
// The replay semantics mirror search()'s feedback loop exactly.
int cmd_trace(int argc, char* argv[]) {
    using namespace sextant;

    cmdline::parser p;
    p.add<std::string>("file", 0, "EngineTrace file (FB records)", true);
    p.add<std::string>("ground-truth", 0,
        "Ground-truth .gtmm file (enables the pq_recall column)", false, "");
    p.add<uint32_t>("topk", 0, "K for recall (default 10)", false, 10);
    p.add<std::string>("fixed", 0,
        "Comma-separated fixed fractions to replay (default 0.05,0.1,0.2,0.3,0.5)",
        false, "0.05,0.1,0.2,0.3,0.5");
    p.add<std::string>("stall", 0,
        "Comma-separated stall thresholds M to replay (default 1,2,3,4)",
        false, "1,2,3,4");
    p.add<std::string>("kth", 0,
        "Comma-separated kth thresholds M to replay (default 1,2,3,4)",
        false, "1,2,3,4");
    p.add<uint32_t>("min-blocks", 0,
        "Blocks probed before stall/kth rules may fire", false, 1);
    p.add("debug-join", 0,
        "Dump the gt/topk join for the first 5 queries (stderr)");
    p.parse_check(argc, argv);

    // --- Parse the trace: FB records grouped by query, in file order ---
    struct Block {
        uint64_t cum, tot;
        uint32_t kth, gained;
        std::vector<RowId> topk;
    };
    std::vector<std::vector<Block>> queries;  // index = q-1
    std::ifstream in(p.get<std::string>("file"));
    if (!in) {
        std::cerr << "trace: cannot open '" << p.get<std::string>("file")
                  << "'\n";
        return 1;
    }
    std::string line;
    bool header_ok = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.rfind("# sextant-trace v1", 0) != 0) {
                std::cerr << "trace: not a sextant-trace v1 file\n";
                return 1;
            }
            header_ok = true;
            continue;
        }
        if (!header_ok || line.rfind("FB ", 0) != 0) continue;
        // FB q= b= child= pages= cum= tot= kth= g= topk=id,id,...
        std::map<std::string, std::string> kv;
        std::istringstream ls(line.substr(3));
        std::string tok;
        while (ls >> tok) {
            const auto eq = tok.find('=');
            if (eq == std::string::npos) continue;
            kv[tok.substr(0, eq)] = tok.substr(eq + 1);
        }
        const uint32_t q = std::stoul(kv.at("q"));
        Block b;
        b.cum = std::stoull(kv.at("cum"));
        b.tot = std::stoull(kv.at("tot"));
        b.kth = std::stoul(kv.at("kth"));
        b.gained = std::stoul(kv.at("g"));
        std::istringstream ts(kv.at("topk"));
        std::string id;
        while (std::getline(ts, id, ',')) {
            if (!id.empty()) b.topk.push_back(std::stoul(id));
        }
        if (q > queries.size()) queries.resize(q);
        queries[q - 1].push_back(std::move(b));
    }
    if (queries.empty()) {
        std::cerr << "trace: no FB records found\n";
        return 1;
    }

    // --- Ground truth (optional) ---
    const uint32_t k = p.get<uint32_t>("topk");
    // Scored positions per query: the trace's top-k list length (what the
    // engine recorded), clamped by --topk — never the raw --topk, which
    // may exceed the traced list and silently deflate the denominator.
    uint32_t k_scored = k;
    for (const auto& qv : queries)
        if (!qv.empty()) {
            k_scored = std::min(k, static_cast<uint32_t>(qv[0].topk.size()));
            break;
        }
    std::vector<std::unordered_set<RowId>> gt;
    if (!p.get<std::string>("ground-truth").empty()) {
        GroundTruth g;
        try {
            g = GroundTruth::load(p.get<std::string>("ground-truth"));
        } catch (const std::runtime_error& e) {
            std::cerr << "trace: " << e.what() << '\n';
            return 1;
        }
        const uint32_t gk_eff = std::min(k, g.k());
        gt.resize(g.n());
        for (uint32_t i = 0; i < g.n(); ++i) {
            const auto ids = g.top(i, gk_eff);
            gt[i].insert(ids.begin(), ids.end());
        }
    }

    // --- Replay: run one rule over one query's blocks.
    // Returns the index of the last-scanned block. Rule kinds mirror
    // search()'s feedback stop rules exactly ('f' uses val as a fraction;
    // 's'/'k' as a consecutive-block threshold).
    const uint32_t min_blocks = p.get<uint32_t>("min-blocks");
    auto replay = [&](const std::vector<Block>& blocks, char kind,
                      double val) -> size_t {
        uint32_t stall_run = 0, kth_run = 0, prev_kth = UINT32_MAX;
        size_t bi = 0;
        for (; bi < blocks.size(); ++bi) {
            const Block& b = blocks[bi];
            bool stop = false;
            switch (kind) {
            case 'f':
                stop = static_cast<double>(b.cum) >=
                       val * static_cast<double>(b.tot);
                break;
            case 's':
                if (bi + 1 >= min_blocks) {
                    if (b.gained == 0) {
                        if (++stall_run >= (uint32_t)val) stop = true;
                    } else {
                        stall_run = 0;
                    }
                }
                break;
            case 'k':
                if (bi + 1 >= min_blocks) {
                    if (b.kth >= prev_kth) {
                        if (++kth_run >= (uint32_t)val) stop = true;
                    } else {
                        kth_run = 0;
                    }
                }
                break;
            }
            prev_kth = b.kth;
            if (stop) break;
        }
        return std::min(bi, blocks.size() - 1);
    };

    // --- Rule grid ---
    struct RowSpec { std::string label; char kind; double val; };
    std::vector<RowSpec> rows;
    auto add_list = [&](const std::string& list, char kind, const char* pfx) {
        std::string item;
        std::istringstream iss(list);
        while (std::getline(iss, item, ',')) {
            if (item.empty()) continue;
            rows.push_back({std::string(pfx) + item, kind, std::stod(item)});
        }
    };
    add_list(p.get<std::string>("fixed"), 'f', "fixed:");
    add_list(p.get<std::string>("stall"), 's', "stall:");
    add_list(p.get<std::string>("kth"), 'k', "kth:");

    double blocks_mean = 0;
    for (const auto& qv : queries) blocks_mean += qv.size();
    blocks_mean /= queries.size();
    if (p.exist("debug-join")) {
        // Per-query join debug: replicate the fixed:0.05 join for the
        // first 5 queries and dump the gt set + hits.
        for (size_t qi = 0; qi < std::min<size_t>(5, queries.size()); ++qi) {
            const size_t bi = replay(queries[qi], 'f', 0.05);
            const Block& b = queries[qi][bi];
            uint32_t hits = 0;
            std::cerr << "dbg q=" << (qi + 1) << " bi=" << bi << " gt=";
            if (!gt.empty() && qi < gt.size())
                for (RowId id : gt[qi]) std::cerr << id << ',';
            for (RowId id : b.topk)
                if (!gt.empty() && qi < gt.size() && gt[qi].count(id)) ++hits;
            std::cerr << " hits=" << hits << " topk=";
            for (RowId id : b.topk) std::cerr << id << ',';
            std::cerr << '\n';
        }
    }
    std::cout << "queries=" << queries.size()
              << " blocks/query(mean)=" << std::fixed << std::setprecision(1)
              << blocks_mean << '\n';
    std::cout << "rule\tmean_frac\tp50_frac\tp99_frac\tblocks(mean)\t"
              << (gt.empty() ? std::string("-") : std::string("pq_recall"))
              << '\n';
    for (const auto& r : rows) {
        std::vector<double> fracs;
        uint64_t blocks_sum = 0;
        double hits = 0;
        for (size_t qi = 0; qi < queries.size(); ++qi) {
            const size_t bi = replay(queries[qi], r.kind, r.val);
            const Block& b = queries[qi][bi];
            fracs.push_back(static_cast<double>(b.cum) /
                            static_cast<double>(b.tot));
            blocks_sum += bi + 1;
            if (!gt.empty() && qi < gt.size()) {
                // Dedupe the top-k ids: closure replication stores a
                // vector in multiple leaves, so the same row_id can occupy
                // several heap slots (delivered recall dedupes too).
                std::unordered_set<RowId> uniq(b.topk.begin(),
                                               b.topk.end());
                for (RowId id : uniq)
                    if (gt[qi].count(id)) ++hits;
            }
        }
        std::sort(fracs.begin(), fracs.end());
        const size_t n = fracs.size();
        const auto pct = [&](double q) {
            return fracs[std::min(n - 1, (size_t)(q * (n - 1)))];
        };
        double mean = 0;
        for (double f : fracs) mean += f;
        mean /= n;
        std::cout << r.label << '\t' << std::setprecision(4) << mean << '\t'
                  << pct(0.5) << '\t' << pct(0.99) << '\t'
                  << (double)blocks_sum / n;
        if (!gt.empty())
            std::cout << '\t' << hits / (queries.size() * k_scored);
        std::cout << '\n';
    }
    return 0;
}

int cmd_fsck(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("file", 0, "Tree index file", true);
    p.add<bool>("repair", 0, "Rebuild bitmap/free-list from tree walk", false,
                false);
    p.parse_check(argc, argv);

    const std::string file_path = p.get<std::string>("file");
    const bool repair = p.get<bool>("repair");

    auto result = sextant::tree::fsck(file_path, repair);
    std::cout << result.summary();
    if (!result.all_ok()) {
        std::cout << "\nfsck: ERRORS FOUND\n";
        return 1;
    }
    std::cout << "\nfsck: OK\n";
    return 0;
}

void print_usage() {
    std::cerr << sextant::version_string("sextant") << "\n\n"
              << "Usage: sextant <command> [options]\n"
              << "Commands:\n"
              << "  build-tree  Build a hierarchical IVF tree index (single file).\n"
              << "              (alias for build-tree-pca, the PCA streaming build)\n"
              << "  tree-search Search a tree index. Computes recall if --ground-truth.\n"
              << "              --index --query --topk --n-probe --fastscan-w\n"
              << "              --adaptive-probe-gap --ground-truth --threads\n"
              << "              --search-threads (within-query leaf-parallel scan)\n"
              << "  build      Build an index from a .fbin file (explicit params).\n"
              << "             Common flags: --input --index --max-node-neighbors (R)\n"
              << "             --beam-width-ceiling (L) --prune-threshold (alpha)\n"
              << "             --pq-segments (m) --pq-bits --threads --metric\n"
              << "             --build-ram --prune-candidate-cap --partition-count\n"
              << "             --log-level\n"
              << "  autobuild  Estimate config (analyze) then build, in-process.\n"
              << "             Accepts all build flags plus estimate knobs:\n"
              << "             --proximity-target --recall-target\n"
              << "  analyze    Read-only dataset-adaptive parameter advisory.\n"
              << "             Flags: --input --metric --proximity-target\n"
              << "             --recall-target --max-node-neighbors --prune-threshold\n"
              << "  sweep       n-probe × W recall/QPS grid with shared scans\n"
              << "             (one scan per n-probe serves all W)\n"
              << "             --pq-segments --pq-bits --pq-max-distortion --threads\n"
              << "             --log-level. Fraction rows via --probe-fraction;\n"
              << "             scan-feedback rows via --feedback (fixed:F|stall:M|kth:M)\n"
              << "             with optional --trace-file for probe traces\n"
              << "  trace      Analyze an EngineTrace file: replay scan-feedback\n"
              << "             stop rules (fixed/stall/kth grids), report fraction\n"
              << "             distribution + pq-recall (--ground-truth)\n"
              << "  search     Search an index with query vectors\n"
              << "  insert     Insert a single vector into an index\n"
              << "  tree-insert  Batch-insert vectors (.fbin) into a tree index.\n"
              << "              --index --vectors --start-row-id\n"
              << "  tree-delete  Delete vectors by row ID from a tree index.\n"
              << "              --index --row-ids --format (text|binary)\n"
              << "  tree-vacuum Repair stale filter summaries after deletes.\n"
              << "              --index --rebuild-cardinality --batch-size\n"
              << "  tree-defrag Compact fragmented leaf extents + shrink file.\n"
              << "              --index --no-shrink --batch-size\n"
              << "  fsck       Check a tree index file. --repair rebuilds the bitmap/free-list.\n";
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
        } else if (cmd == "build-tree" || cmd == "build-tree-pca") {
            return cmd_build_tree_pca(sub_argc, sub_argv.data());
        } else if (cmd == "autobuild") {
            return cmd_autobuild(sub_argc, sub_argv.data());
        } else if (cmd == "search") {
            return cmd_search(sub_argc, sub_argv.data());
        } else if (cmd == "tree-search") {
            return cmd_tree_search(sub_argc, sub_argv.data());
        } else if (cmd == "sweep") {
            return cmd_sweep(sub_argc, sub_argv.data());
        } else if (cmd == "insert") {
            return cmd_insert(sub_argc, sub_argv.data());
        } else if (cmd == "tree-insert") {
            return cmd_tree_insert(sub_argc, sub_argv.data());
        } else if (cmd == "tree-delete") {
            return cmd_tree_delete(sub_argc, sub_argv.data());
        } else if (cmd == "tree-vacuum") {
            return cmd_tree_vacuum(sub_argc, sub_argv.data());
        } else if (cmd == "tree-defrag") {
            return cmd_tree_defrag(sub_argc, sub_argv.data());
        } else if (cmd == "trace") {
            return cmd_trace(sub_argc, sub_argv.data());
        } else if (cmd == "fsck") {
            return cmd_fsck(sub_argc, sub_argv.data());
        } else if (cmd == "analyze") {
            return run_analyze(sub_argc, sub_argv.data());
        } else if (cmd == "--help" || cmd == "-h" || cmd == "help") {
            print_usage();
            return 0;
        } else if (cmd == "--version" || cmd == "-V") {
            std::cerr << sextant::version_string("sextant") << '\n';
            return 0;
        } else {
            std::cerr << "sextant: unknown command '" << cmd << "'\n";
            print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "sextant " << cmd << ": " << e.what() << "\n";
        return 1;
    }
}
