// analyze — PQ sensitivity analysis tool.
//
// For each candidate (m, bits) PQ config, this tool:
//   1. Measures PQ distortion (via Engine::probe_pq_config).
//   2. Builds a mini Vamana graph on the 20K probe pool.
//   3. Searches 500 probe queries with rerank against brute-force truth.
//   4. Reports recall@10 on the mini-graph.
//
// This reveals whether improving PQ quality (lower distortion) actually
// improves end-to-end recall on the dataset, or whether the data is
// PQ-insensitive (e.g. tie-capped clustered data where recall is bounded
// by tie structure, not PQ quality).
//
// Usage:
//   sextant analyze --input data.fbin [--pq-max-distortion 0.05]
//
// See docs/sensitivity_probe.md for the design rationale.

#include "engine/fbin_source.hpp"
#include "sextant/config.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/node_store.hpp"

#include <cmdline/cmdline.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr uint32_t kProbePool = 20000;
constexpr uint32_t kProbeQueries = 500;
constexpr uint32_t kProbeTopk = 10;

using sextant::Dim;
using sextant::MetricKind;
using sextant::RowId;
using sextant::Candidate;
using sextant::Engine;
using sextant::FlatNodeStore;
using sextant::PqQuantizer;
using sextant::VamanaCore;
using sextant::VamanaParams;
using sextant::VamanaTLS;

/// Read the header + first `n` vectors from a .fbin file into a flat buffer.
std::vector<float> read_sample(const std::string& path, uint32_t n,
                               uint32_t& out_n, uint32_t& out_dim) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw sextant::Error(sextant::ErrorCode::IoError,
                                 "cannot open '" + path + "'");
    uint32_t total_n = 0, dim = 0;
    f.read(reinterpret_cast<char*>(&total_n), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    if (!f || dim == 0) throw sextant::Error(sextant::ErrorCode::CorruptIndex,
                                             "invalid .fbin header");
    const uint32_t read_n = std::min(n, total_n);
    // Read from random offsets for a representative sample.
    std::vector<float> buf(static_cast<size_t>(read_n) * dim);
    if (read_n == total_n) {
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(buf.size() * sizeof(float)));
    } else {
        std::mt19937_64 rng(0xC0DE1234ULL);
        for (uint32_t i = 0; i < read_n; i++) {
            const uint64_t idx = rng() % total_n;
            f.seekg(8 + idx * static_cast<uint64_t>(dim) * sizeof(float));
            f.read(reinterpret_cast<char*>(buf.data() + static_cast<size_t>(i) * dim),
                   static_cast<std::streamsize>(dim * sizeof(float)));
        }
    }
    out_n = read_n;
    out_dim = dim;
    return buf;
}

/// Brute-force truth: for each query, the top-kProbeTopk neighbor ids + distances.
struct Truth {
    std::vector<std::vector<uint32_t>> ids;
    std::vector<std::vector<float>> dists;
};

Truth compute_truth(const float* pool, uint32_t pool_n, Dim dim,
                    const std::vector<uint32_t>& qidx) {
    std::vector<float> norms(pool_n);
    for (uint32_t i = 0; i < pool_n; i++) {
        const float* v = pool + static_cast<size_t>(i) * dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; d++) acc += double(v[d]) * v[d];
        norms[i] = float(acc);
    }
    Truth t;
    t.ids.resize(qidx.size());
    t.dists.resize(qidx.size());
    for (size_t qi = 0; qi < qidx.size(); qi++) {
        const float* q = pool + static_cast<size_t>(qidx[qi]) * dim;
        double qn = 0.0;
        for (uint32_t d = 0; d < dim; d++) qn += double(q[d]) * q[d];
        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(pool_n - 1);
        for (uint32_t i = 0; i < pool_n; i++) {
            if (i == qidx[qi]) continue;
            double dot = 0.0;
            const float* v = pool + static_cast<size_t>(i) * dim;
            for (uint32_t d = 0; d < dim; d++) dot += double(q[d]) * v[d];
            ranked.emplace_back(float(norms[i] - 2.0 * dot + qn), i);
        }
        const size_t kk = std::min<size_t>(kProbeTopk, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + kk, ranked.end(),
                          [](const auto& a, const auto& b){ return a.first < b.first; });
        for (size_t k = 0; k < kk; k++) {
            t.ids[qi].push_back(ranked[k].second);
            t.dists[qi].push_back(ranked[k].first);
        }
    }
    return t;
}

/// Build a mini Vamana graph at (m, bits), search with rerank, return recall@10.
double mini_graph_recall(const float* pool, uint32_t pool_n, Dim dim,
                         MetricKind metric, uint16_t pq_m, uint8_t pq_bits,
                         const Truth& truth,
                         const std::vector<uint32_t>& qidx) {
    PqQuantizer q(metric, dim, pq_m, pq_bits);
    q.train(pool, pool_n);
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes(static_cast<size_t>(pool_n) * cs);
    for (uint32_t i = 0; i < pool_n; i++)
        q.encode(pool + static_cast<size_t>(i) * dim,
                 codes.data() + static_cast<size_t>(i) * cs);

    VamanaParams vp;
    vp.dim = dim;
    vp.R = 32;
    vp.L = 100;
    vp.L_build = 100;
    vp.alpha = 1.2f;
    vp.inline_pq_count = 0;
    vp.n_entry_points = 16;
    vp.max_occlusion = 750;

    VamanaCore core(vp, q);
    core.prepare_for_build(pool_n);
    core.set_build_codes(codes.data(), pool_n);
    const uint32_t ns = VamanaCore::static_node_size(vp.R, 0, cs);
    std::vector<uint8_t> nodes(static_cast<size_t>(pool_n) * ns, 0);
    core.set_build_nodes(nodes.data());
    FlatNodeStore store(nodes.data(), codes.data(), ns, cs);
    core.set_store(&store);

    VamanaTLS tls;
    tls.resize(pool_n);
    tls.resize_lut(q.lut_size());
    for (uint32_t i = 0; i < pool_n; i++)
        core.insert_build_from_code(i, static_cast<RowId>(i), tls);
    core.compute_entry_points();
    core.finalize_inline_codes();

    std::vector<float> lut(q.lut_size());
    const uint32_t fetch_k = kProbeTopk * 10;
    const uint32_t probe_L = 50;
    uint64_t hits = 0, total = 0;
    for (size_t qi = 0; qi < qidx.size(); qi++) {
        const float* qv = pool + static_cast<size_t>(qidx[qi]) * dim;
        q.preprocess_query(qv, lut.data());
        auto results = core.search(lut.data(), fetch_k, probe_L, 0);
        if (results.empty()) continue;
        std::vector<std::pair<float, uint32_t>> scored;
        scored.reserve(results.size());
        for (const auto& c : results) {
            if (c.row_id < 0) continue;
            const uint32_t id = static_cast<uint32_t>(c.row_id);
            const float* bv = pool + static_cast<size_t>(id) * dim;
            double d = 0.0;
            for (uint32_t d2 = 0; d2 < dim; d2++)
                d += (double(qv[d2]) - double(bv[d2])) *
                     (double(qv[d2]) - double(bv[d2]));
            scored.emplace_back(static_cast<float>(d), id);
        }
        std::sort(scored.begin(), scored.end());
        const uint32_t topk = std::min<uint32_t>(kProbeTopk, scored.size());
        std::unordered_set<uint32_t> result_set;
        for (uint32_t i = 0; i < topk; i++) result_set.insert(scored[i].second);
        for (uint32_t tid : truth.ids[qi]) {
            total++;
            if (result_set.count(tid)) hits++;
        }
    }
    return total ? double(hits) / double(total) : 0.0;
}

int cmd_analyze(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("input", 0, "Input .fbin file", true);
    p.add<float>("pq-max-distortion", 0,
                 "Max PQ distortion for eligibility (0 = default 0.05)", false, 0.0f);
    p.add<uint32_t>("probe-sample", 0, "Sample size for the probe pool (default 20000)",
                    false, 20000);
    p.add<std::string>("metric", 0, "l2sq or ip", false, "l2sq");
    p.add<std::string>("log-level", 0, "Log level: debug, info, warn, error",
                       false, "warn");
    p.parse_check(argc, argv);

    const std::string input = p.get<std::string>("input");
    const float max_distortion = p.get<float>("pq-max-distortion");
    const uint32_t sample_sz = p.get<uint32_t>("probe-sample");
    const std::string metric_str = p.get<std::string>("metric");
    const std::string log_level = p.get<std::string>("log-level");

    if (log_level == "debug") sextant::set_log_level(sextant::LogLevel::Debug);
    else if (log_level == "info") sextant::set_log_level(sextant::LogLevel::Info);
    else if (log_level == "warn") sextant::set_log_level(sextant::LogLevel::Warn);
    else sextant::set_log_level(sextant::LogLevel::Error);

    MetricKind metric = MetricKind::L2Sq;
    if (metric_str == "ip") metric = MetricKind::InnerProduct;

    // Read sample.
    uint32_t pool_n = 0, dim = 0;
    std::cerr << "Reading " << sample_sz << " vectors from " << input << "...\n";
    auto pool = read_sample(input, sample_sz, pool_n, dim);
    std::cerr << "Pool: " << pool_n << " vectors, dim=" << dim << "\n";

    // Run the distortion probe via Engine::probe_pq_config.
    sextant::ResolvedParams params;
    params.metric = metric;
    params.pq_m = 0;       // auto: probe all candidates
    params.pq_bits = 0;    // auto: probe all candidates
    params.pq_max_distortion = max_distortion > 0 ? max_distortion : 0.05f;
    const auto sel = Engine::probe_pq_config(pool.data(), pool_n, dim, params);

    // Compute brute-force truth for the probe queries.
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_int_distribution<uint32_t> u(0, pool_n - 1);
    std::vector<uint32_t> qidx;
    std::unordered_set<uint32_t> seen;
    while (qidx.size() < std::min(kProbeQueries, pool_n - 1)) {
        const uint32_t q = u(rng);
        if (seen.insert(q).second) qidx.push_back(q);
    }
    std::cerr << "Computing brute-force truth (" << qidx.size() << " queries)...\n";
    const auto truth = compute_truth(pool.data(), pool_n, dim, qidx);

    // Check constraints for the sensitivity probe.
    const bool dim_ok = (dim >= 32);
    const bool pool_ok = (pool_n >= 4000);

    // Print header.
    std::cout << "\nPQ sensitivity analysis (pool=" << pool_n << ", dim=" << dim
              << ", queries=" << qidx.size() << ", max_distortion="
              << params.pq_max_distortion << "):\n\n";

    if (!dim_ok) {
        std::cout << "  NOTE: dim < 32 — sensitivity probe skipped (density ratio\n"
                  << "  too high, see docs/sensitivity_probe.md). Showing distortion only.\n\n";
    } else if (!pool_ok) {
        std::cout << "  NOTE: pool < 4000 — sensitivity probe skipped (graph too\n"
                  << "  small to navigate, see docs/sensitivity_probe.md). Showing distortion only.\n\n";
    }

    std::cout << "  m    bits  distort  eligible  mini_recall  note\n";
    std::cout << "  ----------------------------------------------------------\n";

    // For each probed config, run the mini-graph if constraints are met.
    for (const auto& r : sel.all) {
        const bool eligible = r.distortion <= params.pq_max_distortion;
        const bool selected = (r.m == sel.m && r.bits == sel.bits);

        std::cout << "  " << std::left << std::setw(5) << r.m
                  << std::setw(5) << static_cast<int>(r.bits)
                  << std::fixed << std::setprecision(4) << r.distortion
                  << "   " << (eligible ? "yes" : "no ");

        if (dim_ok && pool_ok && eligible) {
            std::cerr << "  Building mini-graph (m=" << r.m << " bits="
                      << static_cast<int>(r.bits) << ")...\r";
            const double recall = mini_graph_recall(pool.data(), pool_n, dim,
                                                   metric, r.m, r.bits,
                                                   truth, qidx);
            std::cout << "    " << std::fixed << std::setprecision(4) << recall;
        } else {
            std::cout << "       n/a ";
        }

        std::string note;
        if (selected) note = "← SELECTED by distortion probe";
        std::cout << "  " << note << "\n";
    }

    // Recommendation.
    std::cout << "\n";
    if (dim_ok && pool_ok) {
        std::cout << "The distortion probe selected m=" << sel.m << "/"
                  << static_cast<int>(sel.bits) << " (" << sel.reason << ").\n";
        std::cout << "Compare the mini_recall column: if it barely changes across\n"
                  << "eligible configs, the dataset is PQ-insensitive and you can\n"
                  << "safely use the cheapest eligible config (--pq-max-distortion\n"
                  << "to loosen the bound, or --pq-m/--pq-bits to force it).\n";
    } else {
        std::cout << "Sensitivity probe was skipped (see note above).\n";
        std::cout << "The distortion probe selected m=" << sel.m << "/"
                  << static_cast<int>(sel.bits) << ".\n";
    }

    return 0;
}

}  // namespace

int run_analyze(int argc, char* argv[]) {
    return cmd_analyze(argc, argv);
}
