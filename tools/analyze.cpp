// analyze — PQ sensitivity advisory tool.
//
// Helps the user decide whether their dataset is PQ-sensitive (spend more on
// higher m) or PQ-insensitive (use the cheapest eligible config). Not on the
// build path — the user runs it manually before building.
//
// For each eligible (m, bits) PQ config, this tool:
//   1. Measures PQ distortion + ties@10 (via Engine::probe_pq_config).
//   2. Builds a mini Vamana graph on the probe pool (once per config).
//   3. Sweeps no-rerank recall@10 at L=50, 100, 200. Without rerank, PQ
//      ranking errors compound through navigation, revealing how much PQ
//      quality affects graph navigation.
//   4. Reports the table + an honest recommendation.
//
// IMPORTANT LIMITATION: the mini-graph no-rerank sweep correctly shows the
// *relative* ranking of configs (which is better) but CANNOT predict the
// absolute full-scale rerank gap. On a 20K pool, navigation is easy — the
// shortlist almost always contains true neighbors, so a with-rerank
// measurement converges to the same value for all configs (false negative).
// The distortion bound (from the build path) remains the primary selection
// signal; the no-rerank sweep is supporting evidence for relative comparison.
//
// See docs/sensitivity_probe.md for validation data and rationale.
//
// Usage:
//   sextant analyze --input data.fbin \
//       [--pq-max-distortion 0.05] [--probe-sample 20000] [--threads 0]

#include "engine/fbin_source.hpp"
#include "sextant/config.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/node_store.hpp"

#include <cmdline/cmdline.h>
#include <ctpl/ctpl_stl_tls.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

constexpr uint32_t kProbeQueries = 500;
constexpr uint32_t kProbeTopk = 10;
constexpr uint32_t kDefaultProbePool = 20000;

// The three no-rerank L values for the sweep.
const std::array<uint32_t, 3> kNoRerankLs = {{50, 100, 200}};

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

/// Read just the header from a .fbin file (total_n, dim).
void read_fbin_header(const std::string& path, uint64_t& out_n, uint32_t& out_dim) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw sextant::Error(sextant::ErrorCode::IoError,
                                 "cannot open '" + path + "'");
    uint32_t n = 0, dim = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    if (!f || dim == 0) throw sextant::Error(sextant::ErrorCode::CorruptIndex,
                                             "invalid .fbin header");
    out_n = n;
    out_dim = dim;
}

/// Compute the probe pool size needed to achieve a target 10th-NN density
/// ratio. The k-th NN distance scales as (k/N)^(1/d); the ratio between pool
/// and full dataset is (N_full/N_pool)^(1/d). We solve for N_pool given a
/// target ratio, clamped to [10000, 100000].
///   N_pool = N_full / ratio^dim = N_full * ratio^(-dim)
uint32_t auto_pool_size(uint64_t total_n, uint32_t dim) {
    constexpr double kTargetRatio = 1.05;  // 5% density distortion max
    constexpr uint32_t kMinPool = 10000;   // navigability + statistical power
    constexpr uint32_t kMaxPool = 100000;  // time budget ceiling
    if (dim == 0) return kDefaultProbePool;
    double needed = double(total_n) * std::pow(kTargetRatio, -double(dim));
    uint32_t pool = static_cast<uint32_t>(needed);
    pool = std::max(kMinPool, std::min(kMaxPool, pool));
    pool = std::min(pool, static_cast<uint32_t>(total_n));
    return pool;
}

/// Read the header + a random sample of `n` vectors from a .fbin file.
/// Uses seek-based random access for a representative sample when the file
/// is larger than n; reads contiguously when n >= total_n.
std::vector<float> read_sample(const std::string& path, uint32_t n,
                               uint32_t& out_n, uint32_t& out_dim,
                               uint64_t& out_total_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw sextant::Error(sextant::ErrorCode::IoError,
                                 "cannot open '" + path + "'");
    uint32_t total_n = 0, dim = 0;
    f.read(reinterpret_cast<char*>(&total_n), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    if (!f || dim == 0) throw sextant::Error(sextant::ErrorCode::CorruptIndex,
                                             "invalid .fbin header");
    const uint32_t read_n = std::min(n, total_n);
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
    out_total_n = total_n;
    return buf;
}

/// Brute-force truth: for each query, the top-kProbeTopk neighbor ids + distances.
struct Truth {
    std::vector<std::vector<uint32_t>> ids;
    std::vector<float> d10;  // 10th-NN distance per query (for proximity)
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
    t.d10.resize(qidx.size());
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
        }
        t.d10[qi] = kk > 0 ? ranked[kk - 1].first : 0.0f;
    }
    return t;
}

/// A built mini Vamana graph at a fixed (m, bits) config. Owns all buffers
/// and the core, so it can be searched repeatedly at different L values
/// without rebuilding the graph.
struct MiniGraph {
    PqQuantizer quantizer;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> nodes;
    uint32_t node_size = 0;
    uint32_t code_sz = 0;
    uint32_t pool_n = 0;
    std::unique_ptr<VamanaCore> core_;
    std::unique_ptr<FlatNodeStore> store_;

    MiniGraph(MetricKind metric, Dim dim, uint16_t pq_m, uint8_t pq_bits,
              const float* pool, uint32_t n, float alpha = 1.2f)
        : quantizer(metric, dim, pq_m, pq_bits), pool_n(n) {
        quantizer.train(pool, n);
        code_sz = quantizer.code_size();
        codes.resize(static_cast<size_t>(n) * code_sz);
        for (uint32_t i = 0; i < n; i++)
            quantizer.encode(pool + static_cast<size_t>(i) * dim,
                             codes.data() + static_cast<size_t>(i) * code_sz);

        // Defaults (R=32, L=100, alpha=1.2, max_occlusion=750) match a default
        // ResolvedParams with R overridden to 32. inline_pq=0 (build layout).
        VamanaParams vp =
            VamanaParams::from_resolved(sextant::ResolvedParams{}, dim,
                                        /*R_override=*/32);
        vp.alpha = alpha;  // override prune occlusion aggressiveness

        core_ = std::make_unique<VamanaCore>(vp, quantizer);
        core_->prepare_for_build(n);
        core_->set_build_codes(codes.data(), n);
        node_size = VamanaCore::static_node_size(vp.R, 0, code_sz);
        nodes.resize(static_cast<size_t>(n) * node_size, 0);
        core_->set_build_nodes(nodes.data());
        store_ = std::make_unique<FlatNodeStore>(nodes.data(), codes.data(),
                                                  node_size, code_sz);
        core_->set_store(store_.get());

        VamanaTLS tls;
        tls.resize(n);
        tls.resize_lut(quantizer.lut_size());
        for (uint32_t i = 0; i < n; i++)
            core_->insert_build_from_code(i, static_cast<RowId>(i), tls);
        core_->compute_entry_points();
        core_->finalize_inline_codes();
    }

    VamanaCore& core() { return *core_; }
    PqQuantizer& pq() { return quantizer; }
};

/// Search result: both id-recall@10 and proximity in-band.
struct SweepResult {
    double recall;       // id-recall@10 (fraction of true neighbor ids found)
    double proximity;    // in-band fraction (results at/inside d10 radius)
};

/// Search the built mini-graph WITHOUT rerank (rank purely by PQ distance).
/// PQ ranking errors compound through navigation, revealing how much PQ
/// quality affects graph navigation. Returns both id-recall@10 and proximity
/// in-band (cluster-aware: measures whether results land at the right
/// distance, regardless of which co-located id was returned).
SweepResult search_no_rerank(MiniGraph& mg, const float* pool, Dim dim,
                             uint32_t L, const Truth& truth,
                             const std::vector<uint32_t>& qidx) {
    PqQuantizer& q = mg.pq();
    std::vector<float> lut(q.lut_size());
    uint64_t hits = 0, total = 0;
    uint64_t prox_in = 0, prox_total = 0;
    constexpr float kEps = 1e-12f;
    for (size_t qi = 0; qi < qidx.size(); qi++) {
        const float* qv = pool + static_cast<size_t>(qidx[qi]) * dim;
        q.preprocess_query(qv, lut.data());
        auto results = mg.core().search(lut.data(), kProbeTopk, L, 0);
        if (results.empty()) continue;
        // Id-recall.
        std::unordered_set<uint32_t> result_set;
        const uint32_t topk = std::min<uint32_t>(kProbeTopk, results.size());
        for (uint32_t i = 0; i < topk; i++) {
            if (results[i].row_id < 0) continue;
            result_set.insert(static_cast<uint32_t>(results[i].row_id));
        }
        for (uint32_t tid : truth.ids[qi]) {
            total++;
            if (result_set.count(tid)) hits++;
        }
        // Proximity: ratio = d_target / d_result (>=1 = at/inside target).
        const float d_target = truth.d10[qi];
        for (uint32_t i = 0; i < topk; i++) {
            if (results[i].row_id < 0) continue;
            const uint32_t id = static_cast<uint32_t>(results[i].row_id);
            const float* bv = pool + static_cast<size_t>(id) * dim;
            double d = 0.0;
            for (uint32_t d2 = 0; d2 < dim; d2++)
                d += (double(qv[d2]) - double(bv[d2])) *
                     (double(qv[d2]) - double(bv[d2]));
            double ratio;
            if (d_target <= kEps && d <= kEps) {
                ratio = 1.0;  // both co-located: perfect
            } else if (d <= kEps) {
                ratio = 2.0;  // found a closer point
            } else if (d_target <= kEps) {
                ratio = 0.0;
            } else {
                ratio = double(d_target) / d;
            }
            if (ratio >= 1.0) prox_in++;
            prox_total++;
        }
    }
    return {
        total ? double(hits) / double(total) : 0.0,
        prox_total ? double(prox_in) / double(prox_total) : 0.0
    };
}

/// Per-config results row for the output table.
struct ConfigRow {
    uint16_t m;
    uint8_t bits;
    double distortion;
    double ties10;
    bool eligible;
    // no-rerank {recall, proximity} at each L in kNoRerankLs.
    std::array<double, kNoRerankLs.size()> no_rerank_recall;
    std::array<double, kNoRerankLs.size()> no_rerank_proximity;
    bool cheapest_eligible;
};

/// Run an alpha sweep on the probe pool: build mini-graphs at several alpha
/// values using the recommended PQ config and measure proximity at L=200.
/// Alpha controls prune occlusion aggressiveness; the optimal value is
/// dataset-dependent (some datasets prefer dense graphs, others sparse).
/// Returns the alpha with the highest proximity.
float sweep_alpha(MetricKind metric, Dim dim, uint16_t pq_m, uint8_t pq_bits,
                  const float* pool, uint32_t pool_n, const Truth& truth,
                  const std::vector<uint32_t>& qidx) {
    static constexpr std::array<float, 4> kAlphas = {{1.0f, 1.1f, 1.2f, 1.5f}};
    constexpr uint32_t kAlphaL = 200;

    std::cout << "\n  Alpha sweep (m=" << pq_m << "/"
              << static_cast<int>(pq_bits) << ", L=" << kAlphaL << "):\n";

    float best_alpha = 1.2f;
    double best_proximity = -1.0;
    for (float a : kAlphas) {
        MiniGraph mg(metric, dim, pq_m, pq_bits, pool, pool_n, a);
        auto sr = search_no_rerank(mg, pool, dim, kAlphaL, truth, qidx);
        std::cout << "    α=" << std::fixed << std::setprecision(1) << a
                  << "  proximity=" << std::setprecision(3) << sr.proximity;
        if (a == 1.2f) std::cout << "  ← current default";
        std::cout << "\n";
        if (sr.proximity > best_proximity) {
            best_proximity = sr.proximity;
            best_alpha = a;
        }
    }
    return best_alpha;
}

int cmd_analyze(int argc, char* argv[]) {
    cmdline::parser p;
    p.add<std::string>("input", 0, "Input .fbin file", true);
    p.add<float>("pq-max-distortion", 0,
                 "Max PQ distortion for eligibility when no target is set "
                 "(0 = default 0.05). Ignored when --proximity-target or "
                 "--id-recall-target is provided",
                 false, 0.0f);
    p.add<float>("proximity-target", 0,
                 "Target proximity in-band (0-1): fraction of results at/inside "
                 "the true k-th NN distance. The primary quality metric — measures "
                 "whether the index finds the right neighborhood, robust to "
                 "clustering. 0 = knee-based (default). "
                 "See docs/proximity_vs_recall.md",
                 false, 0.0f);
    p.add<float>("id-recall-target", 0,
                 "Override: target id-recall@k instead of proximity. For "
                 "entity matching/dedup where exact ids matter. "
                 "0 = not used (default)",
                 false, 0.0f);
    p.add<uint32_t>("id-recall-k", 0,
                 "k for --id-recall-target (default 10)",
                 false, 10);
    p.add<uint32_t>("probe-sample", 0,
                    "Sample size for the probe pool (0 = auto-size from density ratio)",
                    false, 0);
    p.add<uint32_t>("threads", 0,
                    "Threads for the mini-graph sweep (0 = hardware_concurrency)",
                    false, 0);
    p.add<std::string>("metric", 0, "l2sq or ip", false, "l2sq");
    p.add<std::string>("log-level", 0, "Log level: debug, info, warn, error",
                       false, "warn");
    p.parse_check(argc, argv);

    const std::string input = p.get<std::string>("input");
    const float max_distortion = p.get<float>("pq-max-distortion");
    const float proximity_target = p.get<float>("proximity-target");
    const float id_recall_target = p.get<float>("id-recall-target");
    const uint32_t id_recall_k = p.get<uint32_t>("id-recall-k");
    const uint32_t threads_req = p.get<uint32_t>("threads");
    const uint32_t nthreads = threads_req > 0
        ? threads_req
        : std::max(1u, static_cast<uint32_t>(std::thread::hardware_concurrency()));
    const std::string metric_str = p.get<std::string>("metric");
    const std::string log_level = p.get<std::string>("log-level");

    if (log_level == "debug") sextant::set_log_level(sextant::LogLevel::Debug);
    else if (log_level == "info") sextant::set_log_level(sextant::LogLevel::Info);
    else if (log_level == "warn") sextant::set_log_level(sextant::LogLevel::Warn);
    else sextant::set_log_level(sextant::LogLevel::Error);

    MetricKind metric = MetricKind::L2Sq;
    if (metric_str == "ip") metric = MetricKind::InnerProduct;

    // Peek at the header to auto-size the pool if requested.
    uint64_t total_n = 0;
    uint32_t header_dim = 0;
    read_fbin_header(input, total_n, header_dim);

    uint32_t sample_sz = p.get<uint32_t>("probe-sample");
    bool auto_sized = false;
    if (sample_sz == 0) {
        sample_sz = auto_pool_size(total_n, header_dim);
        auto_sized = true;
    }

    // Read sample.
    uint32_t pool_n = 0, dim = 0;
    std::cerr << "Reading " << sample_sz << " vectors from " << input;
    if (auto_sized)
        std::cerr << " (auto-sized for dim=" << header_dim << ", N=" << total_n
                  << ")";
    std::cerr << "...\n";
    auto pool = read_sample(input, sample_sz, pool_n, dim, total_n);
    std::cerr << "Pool: " << pool_n << "/" << total_n << " vectors, dim=" << dim << "\n";

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
    while (qidx.size() < std::min<uint32_t>(kProbeQueries, pool_n - 1)) {
        const uint32_t q = u(rng);
        if (seen.insert(q).second) qidx.push_back(q);
    }
    std::cerr << "Computing brute-force truth (" << qidx.size() << " queries)...\n";
    const auto truth = compute_truth(pool.data(), pool_n, dim, qidx);

    // Check constraints for the sensitivity probe.
    const bool dim_ok = (dim >= 32);
    const bool pool_ok = (pool_n >= 4000);

    // Identify eligible configs. When a target is set (proximity or id-recall),
    // all configs are eligible (distortion becomes diagnostic-only). Otherwise,
    // the distortion bound filters eligibility.
    const bool target_mode = (proximity_target > 0 || id_recall_target > 0);
    std::vector<const Engine::ProbedRow*> eligible;
    uint16_t cheapest_m = UINT16_MAX;
    uint8_t cheapest_bits = 0;
    for (const auto& r : sel.all) {
        const bool is_eligible =
            target_mode || r.distortion <= params.pq_max_distortion;
        if (is_eligible) {
            eligible.push_back(&r);
            if (r.m < cheapest_m ||
                (r.m == cheapest_m && r.bits < cheapest_bits)) {
                cheapest_m = r.m;
                cheapest_bits = r.bits;
            }
        }
    }

    // Build the output rows.
    std::vector<ConfigRow> rows;
    rows.reserve(sel.all.size());
    for (const auto& r : sel.all) {
        ConfigRow row;
        row.m = r.m;
        row.bits = r.bits;
        row.distortion = r.distortion;
        row.ties10 = r.tie_fraction;
        row.eligible =
            target_mode || r.distortion <= params.pq_max_distortion;
        row.no_rerank_recall.fill(-1.0);
        row.no_rerank_proximity.fill(-1.0);
        row.cheapest_eligible =
            row.eligible && r.m == cheapest_m && r.bits == cheapest_bits;
        rows.push_back(row);
    }

    // Print header.
    std::cout << "\nPQ sensitivity analysis (pool=" << pool_n << "/" << total_n
              << ", dim=" << dim
              << ", queries=" << qidx.size() << ", max_distortion="
              << params.pq_max_distortion;
    if (proximity_target > 0)
        std::cout << ", proximity_target=" << proximity_target;
    if (id_recall_target > 0)
        std::cout << ", id_recall_target=" << id_recall_target << "@"
                  << id_recall_k;
    std::cout << ", threads=" << nthreads << "):\n";

    // Explain pool representativeness.
    // The k-th NN distance scales as (k/N)^(1/d). The ratio between the pool's
    // and full dataset's 10th-NN distance is (N_full/N_pool)^(1/d). When this
    // ratio is close to 1.0, the pool's neighborhood structure matches the full
    // dataset. Above ~1.15, PQ quality appears better on the pool than it is
    // in reality (the pool is artificially dense).
    if (pool_n < total_n) {
        const double density_ratio = std::pow(
            double(total_n) / double(pool_n), 1.0 / double(dim));
        std::cout << "  Pool: " << pool_n << " random vectors sampled from "
                  << total_n << " (" << std::fixed << std::setprecision(1)
                  << 100.0 * pool_n / total_n << "%)";
        if (auto_sized)
            std::cout << ", auto-sized";
        std::cout << ". 10th-NN density ratio: "
                  << std::setprecision(2) << density_ratio
                  << " (1.0 = perfect; <1.15 = representative).\n";
        if (density_ratio > 1.15) {
            std::cout << "  WARNING: density ratio > 1.15 — even the max pool size\n"
                      << "  (100K) cannot reach the target for dim=" << dim
                      << ", N=" << total_n << ". The pool is denser than\n"
                      << "  the full dataset; PQ quality will appear better than\n"
                      << "  at scale. Treat results as optimistic.\n";
        }
    } else {
        std::cout << "  Pool: full dataset (" << pool_n << " vectors).\n";
    }
    std::cout << "\n";

    const bool run_sweep = dim_ok && pool_ok && !eligible.empty();

    if (!dim_ok) {
        std::cout << "  NOTE: dim < 32 — sensitivity probe skipped (density ratio\n"
                  << "  too high, see docs/sensitivity_probe.md). Showing distortion only.\n\n";
    } else if (!pool_ok) {
        std::cout << "  NOTE: pool < 4000 — sensitivity probe skipped (graph too\n"
                  << "  small to navigate, see docs/sensitivity_probe.md). Showing distortion only.\n\n";
    } else if (eligible.empty()) {
        std::cout << "  NOTE: no eligible configs at max_distortion="
                  << params.pq_max_distortion << ". Showing distortion only.\n\n";
    }

    // Run the no-rerank L-sweep on each eligible config in parallel.
    // Each config is fully independent: MiniGraph owns its own buffers;
    // pool/truth/qidx are read-only; each task writes to its own rows[ri].
    if (run_sweep) {
        std::cerr << "Sweeping " << eligible.size() << " eligible configs on "
                  << nthreads << " threads...\n";
        ctpl::thread_pool_tls<VamanaTLS> pool_threads(nthreads);

        std::mutex cerr_mu;
        std::vector<std::future<void>> futs;
        for (size_t ri = 0; ri < rows.size(); ri++) {
            if (!rows[ri].eligible) continue;
            futs.push_back(pool_threads.push(
                [&, ri](size_t /*tid*/, VamanaTLS& /*tls*/) {
                    {
                        std::lock_guard<std::mutex> lk(cerr_mu);
                        std::cerr << "  Building mini-graph (m=" << rows[ri].m
                                  << " bits=" << static_cast<int>(rows[ri].bits)
                                  << ")...\n";
                    }
                    MiniGraph mg(metric, dim, rows[ri].m, rows[ri].bits,
                                 pool.data(), pool_n);
                    for (size_t li = 0; li < kNoRerankLs.size(); li++) {
                        auto sr = search_no_rerank(
                            mg, pool.data(), dim, kNoRerankLs[li], truth, qidx);
                        rows[ri].no_rerank_recall[li] = sr.recall;
                        rows[ri].no_rerank_proximity[li] = sr.proximity;
                    }
                }));
        }
        for (auto& f : futs) f.get();
    }

    // Print the table. Columns: m, bits, distort, ties@10, id-recall@10
    // (no-rerank) at L=50/100/200, proximity in-band at L=200.
    std::cout << "  m    bits  distort  ties@10";
    for (uint32_t L : kNoRerankLs)
        std::cout << "  r@" << std::setw(3) << L;
    std::cout << "  prox@200  note\n";
    std::cout << "  " << std::string(72, '-') << "\n";

    for (const auto& row : rows) {
        std::cout << "  " << std::left << std::setw(5) << row.m
                  << std::setw(5) << static_cast<int>(row.bits)
                  << std::fixed << std::setprecision(4) << row.distortion
                  << "   " << std::setprecision(2) << row.ties10;
        if (row.eligible && run_sweep) {
            for (size_t li = 0; li < kNoRerankLs.size(); li++) {
                std::cout << "  " << std::fixed << std::setprecision(3)
                          << (row.no_rerank_recall[li] >= 0
                                  ? row.no_rerank_recall[li] : 0.0);
            }
            std::cout << "  " << std::setprecision(3)
                      << (row.no_rerank_proximity[kNoRerankLs.size() - 1] >= 0
                              ? row.no_rerank_proximity[kNoRerankLs.size() - 1]
                              : 0.0);
        } else {
            for (size_t li = 0; li < kNoRerankLs.size(); li++)
                std::cout << "    n/a";
            std::cout << "    n/a";
        }
        std::string note;
        if (row.cheapest_eligible) note = "← cheapest eligible";
        if (row.m == sel.m && row.bits == sel.bits) {
            if (!note.empty()) note += "; ";
            note += "distortion-probe pick";
        }
        std::cout << "  " << note << "\n";
    }

    // ── Recommendation ──
    //
    // Three modes:
    //   proximity-target: probe ALL configs, recommend the cheapest whose
    //     mini-graph proximity ≥ target. Primary metric for most workloads.
    //   id-recall-target: same, but using id-recall@k. For entity matching.
    //   knee_mode (default): use --pq-max-distortion to filter eligible
    //     configs, then find the diminishing-returns knee.
    std::cout << "\n";
    if (!run_sweep) {
        std::cout << "Sensitivity probe was skipped (see note above).\n";
        std::cout << "The distortion probe selected m=" << sel.m << "/"
                  << static_cast<int>(sel.bits) << ".\n";
        return 0;
    }

    if (eligible.size() == 1) {
        std::cout << "Only one eligible config: m=" << cheapest_m << "/"
                  << static_cast<int>(cheapest_bits)
                  << ". No sensitivity comparison needed — use it.\n";
        return 0;
    }

    // ties@10 from the cheapest eligible config (data property, not config).
    double ties10 = 0.0;
    for (const auto& row : rows)
        if (row.cheapest_eligible) { ties10 = row.ties10; break; }

    // Best config per m (highest proximity at L=200). At the same m, the user
    // would always pick the better option. Gives a clean curve vs cost (m).
    constexpr size_t kSweepIdx = 2;  // L=200 (kNoRerankLs[2])
    std::map<uint16_t, const ConfigRow*> best_per_m;
    for (const auto& row : rows) {
        if (!row.eligible) continue;
        auto it = best_per_m.find(row.m);
        if (it == best_per_m.end() ||
            row.no_rerank_proximity[kSweepIdx] >
                it->second->no_rerank_proximity[kSweepIdx]) {
            best_per_m[row.m] = &row;
        }
    }
    std::vector<const ConfigRow*> sorted_eligible;
    for (const auto& [m, ptr] : best_per_m)
        sorted_eligible.push_back(ptr);

    uint16_t max_m = 0;
    for (const auto* r : eligible) max_m = std::max(max_m, r->m);

    if (target_mode) {
        // ── Target-driven mode ──
        // Determine which metric and target value to use.
        const bool use_id_recall = (id_recall_target > 0);
        const double target_val = use_id_recall ? id_recall_target
                                                 : proximity_target;
        const std::string metric_name = use_id_recall
            ? ("id-recall@" + std::to_string(id_recall_k))
            : "proximity";

        // Walk from cheapest; recommend the first config whose metric
        // meets the target. If none meet it, recommend the highest m.
        uint16_t rec_m = sorted_eligible.back()->m;
        uint8_t rec_bits = sorted_eligible.back()->bits;
        double rec_metric = use_id_recall
            ? sorted_eligible.back()->no_rerank_recall[kSweepIdx]
            : sorted_eligible.back()->no_rerank_proximity[kSweepIdx];
        bool met = false;
        for (const auto* r : sorted_eligible) {
            const double val = use_id_recall
                ? r->no_rerank_recall[kSweepIdx]
                : r->no_rerank_proximity[kSweepIdx];
            if (val >= target_val) {
                rec_m = r->m;
                rec_bits = r->bits;
                rec_metric = val;
                met = true;
                break;
            }
        }

        std::cout << "  Target: " << metric_name << " >= "
                  << std::setprecision(2) << target_val
                  << " (no-rerank mini-graph, conservative)\n\n";

        std::cout << "  → Recommended: m=" << rec_m << "/"
                  << static_cast<int>(rec_bits);
        if (rec_m == cheapest_m && rec_bits == cheapest_bits) {
            std::cout << " (cheapest; meets target)";
        } else if (rec_m == max_m) {
            std::cout << " (highest quality eligible)";
        } else {
            const int pct = static_cast<int>(
                100.0 * (1.0 - double(rec_m) / double(max_m)));
            std::cout << " (saves ~" << pct << "% build cost vs m=" << max_m
                      << ")";
        }
        std::cout << "\n";

        if (met) {
            std::cout << "    " << metric_name << " " << std::setprecision(3)
                      << rec_metric << " >= target " << std::setprecision(2)
                      << target_val << ". This is a conservative lower bound;\n"
                      << "    full-scale quality with rerank will be higher.\n";
        } else {
            std::cout << "    No config meets the target on the mini-graph (best: "
                      << std::setprecision(3) << rec_metric << " < "
                      << std::setprecision(2) << target_val << ").\n"
                      << "    Full-scale quality with rerank will be higher.\n"
                      << "    Validate with sextant_bench.\n";
        }
        if (ties10 > 0.20 && !use_id_recall) {
            std::cout << "    Note: ties@10=" << std::setprecision(2) << ties10
                      << " — id-recall is structurally capped by clustering,\n"
                      << "    but proximity measures neighborhood quality.\n";
        }

        // ── Alpha sweep ──
        // Build mini-graphs at several alpha values using the recommended PQ
        // config. Alpha controls prune occlusion aggressiveness; the optimal
        // value is dataset-dependent (some datasets prefer dense graphs,
        // others sparse).
        const float rec_alpha = sweep_alpha(
            metric, dim, rec_m, rec_bits, pool.data(), pool_n, truth, qidx);

        std::cout << "\n  Build with: --pq-m " << rec_m << " --pq-bits "
                  << static_cast<int>(rec_bits);
        if (rec_alpha != 1.2f) {
            std::cout << " --alpha " << std::setprecision(1) << rec_alpha;
        }
        std::cout << "\n";

        std::cout << "\n  NOTE: " << (use_id_recall ? "Id-recall" : "Proximity")
                  << " is measured without rerank on a " << pool_n
                  << "-vector mini-graph — a\n"
                  << "  conservative lower bound on full-scale quality"
                  << " (rerank + larger graph both help).\n";
        return 0;
    }

    // ── Knee-driven mode (no recall target) ──
    // Compute consecutive proximity deltas and find the knee.
    std::vector<double> deltas(sorted_eligible.size() - 1);
    double max_delta = 0;
    for (size_t i = 0; i + 1 < sorted_eligible.size(); i++) {
        deltas[i] = sorted_eligible[i + 1]->no_rerank_proximity[kSweepIdx] -
                    sorted_eligible[i]->no_rerank_proximity[kSweepIdx];
        max_delta = std::max(max_delta, deltas[i]);
    }
    constexpr double kKneeFraction = 0.30;
    const double knee_threshold = max_delta * kKneeFraction;

    std::cout << "  Per-step proximity gains (L=" << kNoRerankLs[kSweepIdx]
              << ", best config per m):\n";
    for (size_t i = 0; i < deltas.size(); i++) {
        const auto* a = sorted_eligible[i];
        const auto* b = sorted_eligible[i + 1];
        const int pct = max_delta > 0
            ? static_cast<int>(100.0 * deltas[i] / max_delta) : 0;
        std::cout << "    m=" << std::left << std::setw(3) << a->m << "/"
                  << static_cast<int>(a->bits) << " → m=" << std::setw(3) << b->m
                  << "/" << static_cast<int>(b->bits) << ": +"
                  << std::fixed << std::setprecision(3) << deltas[i]
                  << " (" << pct << "%)";
        if (deltas[i] < knee_threshold) std::cout << "  ← diminishing";
        std::cout << "\n";
    }
    std::cout << "  ties@10: " << std::setprecision(2) << ties10 << "\n\n";

    bool all_above = true;
    for (double d : deltas)
        if (d < knee_threshold) { all_above = false; break; }
    const bool no_knee = (ties10 <= 0.20) && all_above &&
                         sorted_eligible.size() > 1;

    uint16_t rec_m = cheapest_m;
    uint8_t rec_bits = cheapest_bits;
    if (ties10 > 0.20) {
        rec_m = cheapest_m;
        rec_bits = cheapest_bits;
    } else {
        for (size_t i = 0; i < deltas.size(); i++) {
            if (deltas[i] >= knee_threshold) {
                rec_m = sorted_eligible[i + 1]->m;
                rec_bits = sorted_eligible[i + 1]->bits;
            } else {
                break;
            }
        }
    }

    std::cout << "  → Recommended: m=" << rec_m << "/" << static_cast<int>(rec_bits);
    if (rec_m == cheapest_m && rec_bits == cheapest_bits) {
        std::cout << " (cheapest eligible)";
    } else if (rec_m == max_m) {
        std::cout << " (highest quality eligible)";
    } else {
        const int pct = static_cast<int>(
            100.0 * (1.0 - double(rec_m) / double(max_m)));
        std::cout << " (saves ~" << pct << "% build cost vs m=" << max_m << ")";
    }
    std::cout << "\n";

    if (ties10 > 0.20) {
        std::cout << "    ties@10=" << std::setprecision(2) << ties10
                  << " > 0.20 — recall is structurally capped by near-duplicate\n"
                  << "    clustering. PQ quality is secondary. The cheapest eligible\n"
                  << "    config is safe. Loosen --pq-max-distortion for cheaper configs.\n";
    } else if (no_knee) {
        std::cout << "    No diminishing-returns knee — every m increase produces\n"
                  << "    significant proximity gains (>30% of max step). Use the\n"
                  << "    highest eligible m.\n";
    } else if (rec_m == cheapest_m && rec_bits == cheapest_bits) {
        std::cout << "    First step already shows diminishing returns (<30% of max\n"
                  << "    gain). PQ quality barely affects navigation. Cheapest is safe.\n";
    } else {
        std::cout << "    Proximity gains flatten after this config (next step <30% of\n"
                  << "    max gain). Higher m brings diminishing returns.";
        if (rec_m != sel.m || rec_bits != sel.bits) {
            std::cout << " This differs from the\n"
                      << "    distortion-probe pick (m=" << sel.m << "/"
                      << static_cast<int>(sel.bits) << ").\n";
        } else {
            std::cout << " Matches the distortion-probe pick.\n";
        }
    }

    // ── Alpha sweep ──
    // Build mini-graphs at several alpha values using the recommended PQ
    // config. Alpha controls prune occlusion aggressiveness; the optimal
    // value is dataset-dependent (some datasets prefer dense graphs,
    // others sparse).
    const float rec_alpha = sweep_alpha(
        metric, dim, rec_m, rec_bits, pool.data(), pool_n, truth, qidx);

    std::cout << "\n  Build with: --pq-m " << rec_m << " --pq-bits "
              << static_cast<int>(rec_bits);
    if (rec_alpha != 1.2f) {
        std::cout << " --alpha " << std::fixed << std::setprecision(1)
                  << rec_alpha;
    }
    std::cout << "\n";

    std::cout << "\n  NOTE: Proximity is measured without rerank on a "
              << pool_n << "-vector mini-graph — a conservative lower bound\n"
              << "  on full-scale proximity (rerank + larger graph both help).";
    if (no_knee) {
        std::cout << " Validate with `sextant_bench`.\n";
    } else {
        std::cout << " The real knee may be 1-2 configs earlier.\n";
    }

    return 0;
}

}  // namespace

int run_analyze(int argc, char* argv[]) {
    return cmd_analyze(argc, argv);
}
