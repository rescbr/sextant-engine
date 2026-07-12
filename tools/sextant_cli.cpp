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
#include "sextant/config.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <cmdline/cmdline.h>

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// .fbin helpers (header: [u32 n][u32 dim][n × dim × float32]).
// ---------------------------------------------------------------------------

struct FbinHeader {
    uint32_t n = 0;
    uint32_t dim = 0;
};

bool read_fbin_header(const std::string& path, FbinHeader& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&out.n), sizeof(out.n));
    f.read(reinterpret_cast<char*>(&out.dim), sizeof(out.dim));
    return f.good();
}

/// Read a single vector (row `idx`) from a .fbin file into `out` (dim floats).
bool read_fbin_vector(const std::string& path, uint32_t dim, uint64_t idx,
                      std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    const uint64_t off = 8 + idx * static_cast<uint64_t>(dim) * sizeof(float);
    f.seekg(off);
    if (!f.good()) return false;
    out.resize(dim);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(dim * sizeof(float)));
    return f.good();
}

/// Exact L2-squared distance between two float vectors.
float l2sq_distance(const float* a, const float* b, uint32_t dim) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Command handlers. Each returns a process exit code.
// ---------------------------------------------------------------------------

int cmd_build(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("input", 0, "Input .fbin/.ibin/.bbin file", true);
    p.add<std::string>("index", 0, "Index name/path prefix", true);
    p.add<uint16_t>("R", 0, "Graph degree (auto if 0)", false, 0);
    p.add<uint16_t>("L", 0, "Beam width (auto if 0)", false, 0);
    p.add<float>("alpha", 0, "Prune threshold", false, 1.2f);
    p.add<uint8_t>("pq-m", 0, "PQ segments (auto from dim if 0)", false, 0);
    p.add<uint8_t>("pq-bits", 0, "PQ bits", false, 8);
    p.add<std::string>("metric", 0, "l2sq or ip", false, "l2sq");
    p.add<uint32_t>("threads", 0, "Build threads (auto if 0)", false, 0);
    p.add("explain", 0, "Print resolved params and exit (dry-run)");
    p.parse_check(argc, argv);

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
    cfg.pq_m = p.get<uint8_t>("pq-m");
    cfg.pq_bits = p.get<uint8_t>("pq-bits");
    cfg.num_threads = p.get<uint32_t>("threads");
    const std::string metric = p.get<std::string>("metric");
    cfg.metric = (metric == "ip") ? sextant::MetricKind::InnerProduct
                                  : sextant::MetricKind::L2Sq;

    if (p.exist("explain")) {
        auto resolved = sextant::resolve_params(n, dim, cfg);
        std::cout << "input:      " << input << "\n"
                  << "index:      " << index << "\n"
                  << "n_vectors:  " << n << "\n"
                  << "dim:        " << dim << "\n"
                  << "R:          " << resolved.R << "\n"
                  << "L:          " << resolved.L << "\n"
                  << "L_build:    " << resolved.L_build << "\n"
                  << "alpha:      " << resolved.alpha << "\n"
                  << "pq_m:       " << static_cast<int>(resolved.pq_m) << "\n"
                  << "pq_bits:    " << static_cast<int>(resolved.pq_bits) << "\n"
                  << "inline_pq:  " << resolved.inline_pq_count << "\n"
                  << "threads:    " << resolved.num_threads << "\n"
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
    p.add<std::string>("output", 0, "Output file (default: stdout)", false, "");
    p.add<std::string>(
        "base-data", 0,
        "Original base .fbin for rerank (defaults to none)", false, "");
    p.parse_check(argc, argv);

    const std::string index = p.get<std::string>("index");
    const std::string query_path = p.get<std::string>("query");
    const uint32_t k = p.get<uint32_t>("k");
    const uint32_t L = p.get<uint32_t>("L");
    const uint32_t rerank = p.get<uint32_t>("rerank");
    const std::string output = p.get<std::string>("output");
    const std::string base_data = p.get<std::string>("base-data");

    sextant::Engine engine;
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
              << "  insert   Insert a single vector into an index\n";
}

}  // namespace

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
