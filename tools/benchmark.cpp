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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
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
    p.add<uint32_t>("threads", 0, "Search threads (0 = hardware_concurrency)",
                    false, 0);
    p.add<uint64_t>("cache-size", 0,
                     "Search LRU cache size in bytes (0 = auto)", false, 0);
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
    const uint32_t threads_req = p.get<uint32_t>("threads");
    const uint64_t cache_size_req = p.get<uint64_t>("cache-size");

    // Compute thread count early so we can warn about cache thrashing.
    const uint32_t n_threads_hint = threads_req > 0
        ? threads_req
        : static_cast<uint32_t>(std::thread::hardware_concurrency());

    // Warn if an explicit cache size is too small for the thread count.
    // Each concurrent search needs ~50-100 blocks in its working set;
    // a shared LRU that's too small causes cross-thread eviction thrashing
    // (4 threads on 32MB can be slower than 1 thread).
    if (cache_size_req > 0 && n_threads_hint > 1) {
        constexpr uint64_t kMinCachePerThread = 32ull * 1024 * 1024;
        const uint64_t recommended = kMinCachePerThread * n_threads_hint;
        if (cache_size_req < recommended) {
            spdlog::warn("benchmark: --cache-size {:.0f}MB with {} threads may "
                         "cause LRU thrashing (recommended ≥ {:.0f}MB = {}MB/thread)",
                         cache_size_req / 1e6, n_threads_hint,
                         recommended / 1e6, kMinCachePerThread / 1e6);
        }
    }

    try {
        sextant::Engine engine;
        engine.set_cache_size(cache_size_req);
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

        const uint32_t n_threads = std::max(1u, std::min(
            threads_req > 0 ? threads_req
                            : static_cast<uint32_t>(std::thread::hardware_concurrency()),
            n_queries));

        std::cout << "[benchmark] queries: " << n_queries << ", k: " << k
                  << ", L: " << L << ", rerank: " << rerank
                  << ", threads: " << n_threads << "\n";

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

        std::atomic<uint32_t> next{0};
        std::vector<std::thread> workers;
        workers.reserve(n_threads);

        // Per-thread accumulators (merged after join).
        std::vector<std::vector<double>> local_latencies(n_threads);
        std::vector<double> local_recall_sum(n_threads, 0.0);
        std::vector<uint64_t> local_recall_hits(n_threads, 0);
        std::vector<uint64_t> local_recall_total(n_threads, 0);

        const auto t_start = Clock::now();
        for (uint32_t t = 0; t < n_threads; t++) {
            workers.emplace_back([&, t]() {
                // Thread-local scratch.
                std::vector<float> qvec;  // unused (queries in RAM), kept for clarity
                std::vector<float> base_vec(dim);
                auto& latencies_local = local_latencies[t];
                double& recall_sum_local = local_recall_sum[t];
                uint64_t& recall_hits_local = local_recall_hits[t];
                uint64_t& recall_total_local = local_recall_total[t];
                latencies_local.reserve(n_queries / n_threads + 16);

                uint32_t qi;
                while ((qi = next.fetch_add(1, std::memory_order_relaxed)) < n_queries) {
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
                    latencies_local.push_back(US(q_end - q_start).count());

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
                        recall_sum_local += static_cast<double>(hits) / static_cast<double>(gt_k);
                        recall_hits_local += hits;
                        recall_total_local += gt_k;
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
        const auto t_end = Clock::now();
        const double total_sec =
            std::chrono::duration<double>(t_end - t_start).count();

        // Merge thread-local accumulators into aggregates.
        double recall_sum = 0.0;
        uint64_t recall_hits = 0;
        uint64_t recall_total = 0;
        for (uint32_t t = 0; t < n_threads; t++) {
            latencies_us.insert(latencies_us.end(),
                                local_latencies[t].begin(),
                                local_latencies[t].end());
            recall_sum += local_recall_sum[t];
            recall_hits += local_recall_hits[t];
            recall_total += local_recall_total[t];
        }

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
         if (engine.is_paged()) {
             std::cout << "[benchmark] cache: graph_reads="
                       << engine.cache_graph_reads()
                       << " code_reads=" << engine.cache_code_reads() << "\n";
             const auto as = engine.cache_admission_stats();
             const uint64_t total_hits = as.hits_window + as.hits_probation + as.hits_protected;
             const uint64_t total_accesses = total_hits + as.misses;
             const double hit_rate = total_accesses > 0
                 ? 100.0 * static_cast<double>(total_hits) / static_cast<double>(total_accesses)
                 : 0.0;
             const uint64_t total_evicts = as.evictions_admitted + as.evictions_rejected;
             const double admit_rate = total_evicts > 0
                 ? 100.0 * static_cast<double>(as.evictions_admitted) / static_cast<double>(total_evicts)
                 : 0.0;
             std::cout << "[benchmark] w-tinylfu: hit_rate=" << std::fixed
                       << std::setprecision(1) << hit_rate << "%"
                       << " (w=" << as.hits_window
                       << " p=" << as.hits_probation
                       << " pr=" << as.hits_protected
                       << " miss=" << as.misses << ")"
                       << " admit=" << std::setprecision(1) << admit_rate << "%"
                        << " (" << as.evictions_admitted << "/" << total_evicts << ")\n";
             const uint64_t tl_h = engine.tl_hits();
             const uint64_t tl_m = engine.tl_misses();
             const uint64_t tl_total = tl_h + tl_m;
             const double tl_rate = tl_total > 0
                 ? 100.0 * static_cast<double>(tl_h) / static_cast<double>(tl_total)
                 : 0.0;
             std::cout << "[benchmark] tl-l1: hit_rate=" << std::fixed
                       << std::setprecision(1) << tl_rate << "%"
                       << " (hits=" << tl_h << " misses=" << tl_m << ")\n";
          }
    } catch (const Error& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "benchmark: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
