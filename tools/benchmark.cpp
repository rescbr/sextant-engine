// benchmark — recall + latency benchmark for a Sextant index.
//
// Workflow:
//   1. Open the index.
//   2. Read query vectors from a .fbin file.
//   3. For each query: search + rerank against base data, take top-k row_ids.
//   4. Read ground-truth neighbor IDs from a .gt file.
//   5. Compute Recall@k = |results ∩ gt[:k]| / k, averaged over all queries.
//   6. Report mean recall@k, p50/p99 latency, total search time, QPS.
//
// Usage:
//   benchmark --index myindex --queries query.fbin --base-data base.fbin \
//             --ground-truth gt.gt --k 10 --L 200 --rerank 10

#include "sextant/config.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <cmdline/cmdline.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using sextant::Error;
using sextant::ErrorCode;
using Clock = std::chrono::steady_clock;
using US = std::chrono::duration<double, std::micro>;

// ---------------------------------------------------------------------------
// .fbin / .gt readers.
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

/// Read a single vector (row `idx`) from a .fbin file into `out`.
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

/// Exact L2-squared distance.
float l2sq_distance(const float* a, const float* b, uint32_t dim) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Ground-truth .gt reader: [uint32 n][uint32 k][n×k uint32 ids][n×k float dists].
// We only need the IDs.
// ---------------------------------------------------------------------------

struct GroundTruth {
    uint32_t n = 0;
    uint32_t k = 0;
    std::vector<uint32_t> ids;  // n × k, row-major
};

GroundTruth read_ground_truth(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw Error(ErrorCode::IoError,
                    "cannot open ground-truth '" + path + "'");
    }
    GroundTruth gt;
    f.read(reinterpret_cast<char*>(&gt.n), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&gt.k), sizeof(uint32_t));
    if (!f || gt.n == 0 || gt.k == 0) {
        throw Error(ErrorCode::CorruptIndex,
                    "invalid ground-truth header in '" + path + "'");
    }
    gt.ids.resize(static_cast<size_t>(gt.n) * gt.k);
    f.read(reinterpret_cast<char*>(gt.ids.data()),
           static_cast<std::streamsize>(gt.ids.size() * sizeof(uint32_t)));
    if (!f) {
        throw Error(ErrorCode::CorruptIndex,
                    "short read on ground-truth ids in '" + path + "'");
    }
    return gt;
}

// ---------------------------------------------------------------------------
// Statistics helpers.
// ---------------------------------------------------------------------------

/// Percentile from a sorted vector of latencies (microseconds).
double percentile(std::vector<double>& sorted_us, double pct) {
    if (sorted_us.empty()) return 0.0;
    if (sorted_us.size() == 1) return sorted_us[0];
    // Linear interpolation between closest ranks.
    const double rank = (pct / 100.0) * (sorted_us.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    if (lo == hi) return sorted_us[lo];
    const double frac = rank - static_cast<double>(lo);
    return sorted_us[lo] + frac * (sorted_us[hi] - sorted_us[lo]);
}

}  // namespace

int main(int argc, char* argv[]) {
    sextant::init_logging();

    cmdline::parser p;
    p.add<std::string>("index", 0, "Index name/path prefix", true);
    p.add<std::string>("queries", 0, "Query .fbin file", true);
    p.add<std::string>("base-data", 0, "Original base .fbin for rerank", true);
    p.add<std::string>("ground-truth", 0, "Ground-truth .gt file", true);
    p.add<uint32_t>("k", 0, "Number of results to evaluate", false, 10);
    p.add<uint32_t>("L", 0, "Search beam width", false, 200);
    p.add<uint32_t>("rerank", 0, "Rerank factor (0/1 = no rerank)", false, 10);
    p.add<uint32_t>("io-limit", 0, "Search I/O budget (0 = unlimited)", false, 0);
    p.add<uint32_t>("limit", 0, "Max queries to run (0 = all)", false, 0);
    p.parse_check(argc, argv);

    const std::string index = p.get<std::string>("index");
    const std::string query_path = p.get<std::string>("queries");
    const std::string base_data = p.get<std::string>("base-data");
    const std::string gt_path = p.get<std::string>("ground-truth");
    const uint32_t k = p.get<uint32_t>("k");
    const uint32_t L = p.get<uint32_t>("L");
    const uint32_t rerank = p.get<uint32_t>("rerank");
    const uint32_t io_limit = p.get<uint32_t>("io-limit");
    const uint32_t limit = p.get<uint32_t>("limit");

    try {
        sextant::Engine engine;
        engine.open(index);
        const uint32_t dim = engine.dim();
        const uint64_t n_base = engine.count();

        FbinHeader qh;
        if (!read_fbin_header(query_path, qh) || qh.dim != dim) {
            std::cerr << "benchmark: invalid query file '" << query_path
                      << "' (dim=" << qh.dim << ", expected " << dim << ")\n";
            return 1;
        }
        if (qh.n == 0) {
            std::cerr << "benchmark: query file is empty\n";
            return 1;
        }

        FbinHeader bh{};
        if (!read_fbin_header(base_data, bh) || bh.dim != dim) {
            std::cerr << "benchmark: invalid base-data '" << base_data
                      << "' (dim=" << bh.dim << ", expected " << dim << ")\n";
            return 1;
        }

        GroundTruth gt = read_ground_truth(gt_path);
        if (gt.k < k) {
            std::cerr << "benchmark: ground-truth k=" << gt.k
                      << " < requested k=" << k << "\n";
            return 1;
        }

        const uint32_t n_queries = std::min<uint32_t>(
            qh.n, limit ? limit : qh.n);
        if (gt.n < n_queries) {
            spdlog::warn("benchmark: ground-truth has {} queries but running {}",
                         gt.n, n_queries);
        }

        const bool do_rerank = rerank > 1;
        const uint32_t fetch_k = do_rerank
                                     ? std::min<uint32_t>(k * rerank, n_base)
                                     : k;

        std::cout << "[benchmark] queries: " << n_queries << ", k: " << k
                  << ", L: " << L << ", rerank: " << rerank << "\n";

        // Read all query vectors into RAM (10k × 128 × 4B = ~5MB, fine).
        std::vector<float> queries(static_cast<size_t>(n_queries) * dim);
        {
            std::ifstream qf(query_path, std::ios::binary);
            qf.seekg(8);
            qf.read(reinterpret_cast<char*>(queries.data()),
                    static_cast<std::streamsize>(queries.size() *
                                                 sizeof(float)));
            if (!qf) {
                std::cerr << "benchmark: short read on query file\n";
                return 1;
            }
        }

        // Optionally mmap-style: read the whole base file for fast rerank.
        std::vector<float> base_all;
        bool base_in_ram = false;
        {
            std::ifstream bf(base_data, std::ios::binary | std::ios::ate);
            if (bf) {
                const std::streamoff sz = bf.tellg();
                // Only load if it fits comfortably (cap at ~4GB).
                if (sz > 8 &&
                    static_cast<uint64_t>(sz) <= uint64_t{4} * 1024 * 1024 *
                                                      1024) {
                    base_all.resize(static_cast<size_t>(sz - 8) / sizeof(float));
                    bf.seekg(8);
                    bf.read(reinterpret_cast<char*>(base_all.data()),
                            static_cast<std::streamsize>(base_all.size() *
                                                         sizeof(float)));
                    base_in_ram = bf.good();
                    if (base_in_ram) {
                        spdlog::info("benchmark: loaded {} base vectors into RAM",
                                     bh.n);
                    }
                }
            }
        }

        std::vector<double> latencies_us;
        latencies_us.reserve(n_queries);
        double recall_sum = 0.0;
        uint64_t recall_hits = 0;
        uint64_t recall_total = 0;

        std::vector<float> qvec(dim);
        std::vector<float> base_vec(dim);

        const auto t_start = Clock::now();
        for (uint32_t qi = 0; qi < n_queries; qi++) {
            const float* q = &queries[static_cast<size_t>(qi) * dim];

            sextant::SearchConfig scfg;
            scfg.k = fetch_k;
            scfg.L_search = L;
            scfg.rerank_factor = rerank;
            scfg.io_limit = io_limit;

            const auto q_start = Clock::now();
            auto results = engine.search(q, scfg.k, scfg);

            // Resolve final top-k row_ids (with optional exact rerank).
            std::vector<sextant::RowId> topk;
            topk.reserve(k);
            if (do_rerank && !results.empty()) {
                std::vector<std::pair<float, sextant::RowId>> scored;
                scored.reserve(results.size());
                for (const auto& c : results) {
                    if (c.row_id < 0 ||
                        static_cast<uint64_t>(c.row_id) >= bh.n) {
                        continue;
                    }
                    float d;
                    if (base_in_ram) {
                        const float* bv =
                            &base_all[static_cast<size_t>(c.row_id) * dim];
                        d = l2sq_distance(q, bv, dim);
                    } else {
                        read_fbin_vector(base_data, dim,
                                         static_cast<uint64_t>(c.row_id),
                                         base_vec);
                        d = l2sq_distance(q, base_vec.data(), dim);
                    }
                    scored.emplace_back(d, c.row_id);
                }
                std::sort(scored.begin(), scored.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
                const uint32_t topk_n = std::min<uint32_t>(k, scored.size());
                for (uint32_t i = 0; i < topk_n; i++) {
                    topk.push_back(scored[i].second);
                }
            } else {
                const uint32_t topk_n = std::min<uint32_t>(k, results.size());
                for (uint32_t i = 0; i < topk_n; i++) {
                    topk.push_back(results[i].row_id);
                }
            }
            const auto q_end = Clock::now();
            latencies_us.push_back(US(q_end - q_start).count());

            // Recall@k against ground truth.
            if (qi < gt.n) {
                const uint32_t* gt_row =
                    &gt.ids[static_cast<size_t>(qi) * gt.k];
                const uint32_t gt_k = std::min(gt.k, k);
                uint32_t hits = 0;
                for (uint32_t i = 0; i < gt_k; i++) {
                    const uint32_t g = gt_row[i];
                    for (sextant::RowId r : topk) {
                        if (r >= 0 && static_cast<uint32_t>(r) == g) {
                            ++hits;
                            break;
                        }
                    }
                }
                recall_sum += static_cast<double>(hits) / static_cast<double>(gt_k);
                recall_hits += hits;
                recall_total += gt_k;
            }
        }
        const auto t_end = Clock::now();
        const double total_sec =
            std::chrono::duration<double>(t_end - t_start).count();

        const double mean_recall =
            (n_queries > 0) ? recall_sum / static_cast<double>(n_queries)
                            : 0.0;
        const double micro_recall =
            (recall_total > 0)
                ? static_cast<double>(recall_hits) /
                      static_cast<double>(recall_total)
                : 0.0;

        std::sort(latencies_us.begin(), latencies_us.end());
        const double p50_us = percentile(latencies_us, 50.0);
        const double p99_us = percentile(latencies_us, 99.0);
        const double qps =
            (total_sec > 0.0)
                ? static_cast<double>(n_queries) / total_sec
                : 0.0;

        std::cout << "[benchmark] recall@" << k << ": "
                  << std::fixed << std::setprecision(4) << mean_recall << "\n";
        std::cout << "[benchmark] recall@" << k << " (micro-avg): "
                  << std::fixed << std::setprecision(4) << micro_recall << "\n";
        std::cout << "[benchmark] latency p50: " << std::fixed
                  << std::setprecision(3) << p50_us / 1000.0 << "ms, p99: "
                  << p99_us / 1000.0 << "ms\n";
        std::cout << "[benchmark] total search time: " << std::fixed
                  << std::setprecision(3) << total_sec << "s\n";
        std::cout << "[benchmark] QPS: " << std::fixed
                  << std::setprecision(1) << qps << "\n";
    } catch (const Error& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
