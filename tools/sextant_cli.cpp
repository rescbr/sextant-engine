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
#include "sextant_version.hpp"
#include "sextant/builder.hpp"
#include "sextant/config.hpp"
#include "sextant/crash_handler.hpp"
#include "sextant/error.hpp"
#include "sextant/estimator.hpp"
#include "sextant/index.hpp"
#include "sextant/logging.hpp"
#include "sextant/searcher.hpp"
#include "tree/fsck.hpp"
#include "tree/filter_data_io.hpp"
#include "tree/ivf_tree_index.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"

#include <cmdline/cmdline.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
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
    p.add<std::string>("input", 0, "Base vectors (.fbin)", true);
    p.add<std::string>("index", 0, "Output tree file path", true);
    p.add<uint32_t>("k-root", 0, "Root branching factor (0=auto)", false, 0);
    p.add<uint32_t>("leaf-capacity", 0, "Max vectors per leaf", false, 5000);
    p.add<uint16_t>("pq4-m", 0, "PQ subquantizers (0=dim/4)", false, 0);
    p.add<uint32_t>("pq-bits", 0, "PQ bits (4 or 8)", false, 4);
    p.add<std::string>("quantizer", 0, "pq / prq / rabitq", false, "pq");
    p.add<std::string>("metric", 0, "l2sq / ip", false, "l2sq");
    p.add<uint32_t>("threads", 0, "Build threads (0=auto)", false, 0);
    p.add<uint32_t>("pca-dims", 0, "PCA dimensions (default 32)", false, 32);
    p.add<uint32_t>("max-lloyd-passes", 0, "Max streaming Lloyd passes (default 10)", false, 10);
    p.add<float>("closure-mult", 0, "Closure epsilon multiplier (default 0.15)", false, 0.15f);
    p.add<std::string>("filter-data", 0, "Filter column data sidecar (.fdat)", false, "");
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
    cfg.pca_dims = p.get<uint32_t>("pca-dims");
    cfg.max_lloyd_passes = p.get<uint32_t>("max-lloyd-passes");
    cfg.closure_multiplier = p.get<float>("closure-mult") > 0 ? p.get<float>("closure-mult") : 0.15f;
    cfg.params.pq4_m = p.get<uint16_t>("pq4-m");
    cfg.params.scan_pq_bits = static_cast<uint8_t>(p.get<uint32_t>("pq-bits"));
    cfg.params.quantizer_type = p.get<std::string>("quantizer");
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

    auto result = tree::IVFTreeIndex::build_streaming_pca(
        p.get<std::string>("input"), p.get<std::string>("index"), cfg);
    std::cout << "built tree index (pca) '" << result.index_path
              << "': n=" << result.n_vectors
              << " dim=" << result.dim
              << " m4=" << static_cast<int>(result.pq_m)
              << " in " << result.build_time_sec << "s\n";
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
    p.add<std::string>("query", 0, "Query vectors (.fbin)", true);
    p.add<std::string>("ground-truth", 0, "Ground-truth .gt file", false, "");
    p.add<uint32_t>("topk", 0, "K nearest neighbors", false, 10);
    p.add<uint32_t>("n-probe", 0, "Root probe count (0=manifest default)", false, 0);
    p.add<uint32_t>("n-probe-ln", 0, "Leaf probe count per root child (0=manifest)", false, 0);
    p.add<uint32_t>("fastscan-w", 0, "Rerank shortlist per shard (0=300)", false, 0);
    p.add<float>("adaptive-probe-gap", 0, "Geometric gap pruning (0=manifest)", false, 0.0f);
    p.add<uint32_t>("threads", 0, "Search threads (0=auto)", false, 0);
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
    scfg.n_probe_ln = p.get<uint32_t>("n-probe-ln");
    scfg.fastscan_W = p.get<uint32_t>("fastscan-w");
    scfg.adaptive_probe_gap = p.get<float>("adaptive-probe-gap");

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
    using PayloadLoc = std::pair<const uint8_t*, uint32_t>;
    std::vector<std::vector<PayloadLoc>> all_payload_locs;
    if (with_payload) all_payload_locs.resize(qh.n);

    const auto t0 = std::chrono::steady_clock::now();

    if (num_threads <= 1) {
        for (uint32_t qi = 0; qi < qh.n; ++qi) {
            if (with_payload) {
                all_results[qi] = idx->search(
                    &queries[static_cast<size_t>(qi) * qh.dim], k, scfg,
                    &all_payload_locs[qi]);
            } else {
                all_results[qi] = idx->search(
                    &queries[static_cast<size_t>(qi) * qh.dim], k, scfg);
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
                    if (qi >= qh.n) break;
                    if (with_payload) {
                        all_results[qi] = idx->search(
                            &queries[static_cast<size_t>(qi) * qh.dim], k, scfg,
                            &all_payload_locs[qi]);
                    } else {
                        all_results[qi] = idx->search(
                            &queries[static_cast<size_t>(qi) * qh.dim], k, scfg);
                    }
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
              << "             --pq-segments --pq-bits --pq-max-distortion --threads\n"
              << "             --log-level\n"
              << "  search     Search an index with query vectors\n"
              << "  insert     Insert a single vector into an index\n"
              << "  tree-insert  Batch-insert vectors (.fbin) into a tree index.\n"
              << "              --index --vectors --start-row-id\n"
              << "  tree-delete  Delete vectors by row ID from a tree index.\n"
              << "              --index --row-ids --format (text|binary)\n"
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
        } else if (cmd == "insert") {
            return cmd_insert(sub_argc, sub_argv.data());
        } else if (cmd == "tree-insert") {
            return cmd_tree_insert(sub_argc, sub_argv.data());
        } else if (cmd == "tree-delete") {
            return cmd_tree_delete(sub_argc, sub_argv.data());
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
        if (sextant::last_throw_trace()) {
            std::cerr << "Throw-site trace:\n"
                      << sextant::last_throw_trace()->to_string() << "\n";
        }
        std::cerr << "(trapping for debugger/core dump)\n";
        __builtin_trap();
    }
}
