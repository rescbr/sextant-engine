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

#include "fbin_io.hpp"
#include "sextant/config.hpp"
#include "sextant/searcher.hpp"
#include "sextant/index.hpp"
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
using namespace sextant::fbin_io;  // FbinHeader, read_fbin_header, read_fbin_vector, l2sq_distance

// ---------------------------------------------------------------------------
// Ground-truth .gt reader: [uint32 n][uint32 k][n×k uint32 ids][n×k float dists].
// We only need the IDs.
// ---------------------------------------------------------------------------

struct GroundTruth {
    uint32_t n = 0;
    uint32_t k = 0;
    std::vector<uint32_t> ids;    // n × k, row-major
    std::vector<float> dists;     // n × k, row-major (L2sq, ascending per row)
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
    // Distances are needed for proximity evaluation (k-th NN radius per query).
    gt.dists.resize(static_cast<size_t>(gt.n) * gt.k);
    f.read(reinterpret_cast<char*>(gt.dists.data()),
           static_cast<std::streamsize>(gt.dists.size() * sizeof(float)));
    if (!f) {
        // Older GT files without distances: tolerate, disable proximity metrics.
        gt.dists.clear();
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
    p.add<uint32_t>("topk", 0, "Number of nearest neighbors to return per query (k in ANN literature). Default 100 (VIBE / modern-retrieval convention).", false, 100);
    p.add<uint32_t>("search-beam-width", 0,
        "Search-time beam width (L in Vamana literature). Higher = more accurate, slower.",
        false, 200);
    p.add<uint32_t>("rerank", 0, "Rerank factor (0/1 = no rerank)", false, 10);
    p.add<uint32_t>("io-limit", 0, "Search I/O budget (0 = unlimited)", false, 0);
    p.add<uint32_t>("limit", 0, "Max queries to run (0 = all)", false, 0);
    p.add<uint32_t>("threads", 0, "Search threads (0 = hardware_concurrency)",
                    false, 0);
    p.add<uint64_t>("cache-size", 0,
                     "Search LRU cache size in bytes (0 = auto)", false, 0);
    p.add("no-cache-rebalance", 0,
          "Disable adaptive graph/code cache rebalancing (default: enabled in paged mode)");
    p.add<std::string>("log-level", 0,
                       "Log level: debug, info, warn, error",
                       false, "info");
    p.parse_check(argc, argv);

    // Set log level (must come after init_logging() above and before any
    // spdlog calls). debug exposes the cache-rebalance controller's per-call
    // sample log, BFS reorder details, and other diagnostic output.
    {
        const auto lvl = p.get<std::string>("log-level");
        if (lvl == "debug") sextant::set_log_level(sextant::LogLevel::Debug);
        else if (lvl == "warn") sextant::set_log_level(sextant::LogLevel::Warn);
        else if (lvl == "error") sextant::set_log_level(sextant::LogLevel::Error);
    }

    const std::string index = p.get<std::string>("index");
    const std::string query_path = p.get<std::string>("queries");
    const std::string base_data = p.get<std::string>("base-data");
    const std::string gt_path = p.get<std::string>("ground-truth");
    const uint32_t k = p.get<uint32_t>("topk");
    const uint32_t L = p.get<uint32_t>("search-beam-width");
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
    //
    // When --cache-size is small enough to force misses, the adaptive
    // rebalance controller will shift capacity between graph/code caches;
    // use --no-cache-rebalance to disable for A/B comparison.
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
        std::unique_ptr<sextant::Index> idx =
            sextant::Index::read(index, cache_size_req);
        sextant::Searcher searcher(*idx, n_threads_hint);
        if (p.exist("no-cache-rebalance")) {
            searcher.set_cache_rebalance_enabled(false);
        }
        const uint32_t dim = idx->dim;
        const uint64_t n_base = idx->count;

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

        const uint32_t n_threads = std::max(1u, std::min(n_threads_hint,
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

        const bool have_gt_dists = !gt.dists.empty();
        // ratio = d_target / d_result. Capped to keep the mean sane when the
        // result is far closer than the target (common with co-located points).
        constexpr double kRatioCap = 10.0;
        constexpr double kEps = 1e-6f;

        // Aggregates (collected on the main thread as batches return).
        double recall_sum = 0.0;
        uint64_t recall_hits = 0;
        uint64_t recall_total = 0;
        std::vector<double> prox_ratios;
        uint64_t prox_in = 0;
        uint64_t prox_total = 0;
        prox_ratios.reserve(n_queries * std::max(1u, k));

        // Per-query rerank + metrics processing — runs on the caller thread
        // as each batch returns. The pool handles the parallel search; all
        // post-processing is single-threaded (cheap compared to search).
        auto process_result = [&](uint32_t qi,
                                  std::vector<sextant::Candidate>&& results,
                                  double latency_us) {
            const float* q = &queries[static_cast<size_t>(qi) * dim];
            latencies_us.push_back(latency_us);

            std::vector<std::pair<float, sextant::RowId>> topk_scored;
            topk_scored.reserve(k);
            std::vector<float> base_vec(dim);
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
                    topk_scored.push_back(scored[i]);
                }
            } else {
                const uint32_t topk_n = std::min<uint32_t>(k, results.size());
                for (uint32_t i = 0; i < topk_n; i++) {
                    const auto rid = results[i].row_id;
                    float d = std::numeric_limits<float>::infinity();
                    if (rid >= 0 &&
                        static_cast<uint64_t>(rid) < bh.n) {
                        if (base_in_ram) {
                            const float* bv =
                                &base_all[static_cast<size_t>(rid) * dim];
                            d = l2sq_distance(q, bv, dim);
                        } else {
                            read_fbin_vector(base_data, dim,
                                             static_cast<uint64_t>(rid),
                                             base_vec);
                            d = l2sq_distance(q, base_vec.data(), dim);
                        }
                    }
                    topk_scored.emplace_back(d, rid);
                }
            }

            if (qi < gt.n) {
                const uint32_t* gt_row =
                    &gt.ids[static_cast<size_t>(qi) * gt.k];
                const uint32_t gt_k = std::min(gt.k, k);
                uint32_t hits = 0;
                for (uint32_t i = 0; i < gt_k; i++) {
                    const uint32_t g = gt_row[i];
                    for (const auto& [d, r] : topk_scored) {
                        (void)d;
                        if (r >= 0 && static_cast<uint32_t>(r) == g) {
                            ++hits;
                            break;
                        }
                    }
                }
                recall_sum += static_cast<double>(hits) / static_cast<double>(gt_k);
                recall_hits += hits;
                recall_total += gt_k;

                if (have_gt_dists && base_in_ram) {
                    const uint32_t gt_kth_id =
                        gt.ids[static_cast<size_t>(qi) * gt.k + (gt_k - 1)];
                    float d_target;
                    if (gt_kth_id < bh.n) {
                        const float* bv_kth =
                            &base_all[static_cast<size_t>(gt_kth_id) * dim];
                        d_target = l2sq_distance(q, bv_kth, dim);
                    } else {
                        d_target = gt.dists[static_cast<size_t>(qi) * gt.k +
                                             (gt_k - 1)];
                    }
                    for (const auto& [d, r] : topk_scored) {
                        if (r < 0) continue;
                        double ratio;
                        if (d_target <= kEps && d <= kEps) {
                            ratio = 1.0;
                        } else if (d <= kEps) {
                            ratio = kRatioCap;
                        } else if (d_target <= kEps) {
                            ratio = 0.0;
                        } else {
                            ratio = static_cast<double>(d_target) /
                                    static_cast<double>(d);
                        }
                        prox_ratios.push_back(ratio);
                        if (ratio >= 1.0) ++prox_in;
                        ++prox_total;
                    }
                }
            }
        };

        // Chunked batch driver: submit chunks of queries to the pool, then
        // process results as each chunk returns. Chunk size = pool size so
        // the pool stays saturated without over-allocating futures.
        constexpr uint32_t kChunkMin = 1;
        const uint32_t chunk_size = std::max(kChunkMin, n_threads);
        const auto t_start = Clock::now();
        for (uint32_t base = 0; base < n_queries; base += chunk_size) {
            const uint32_t n_this = std::min<uint32_t>(chunk_size,
                                                       n_queries - base);
            sextant::SearchConfig scfg;
            scfg.k = fetch_k;
            scfg.L_search = L;
            scfg.rerank_factor = rerank;
            scfg.io_limit = io_limit;

            const float* q0 = &queries[static_cast<size_t>(base) * dim];
            const auto batch_start = Clock::now();
            auto batch_results = searcher.search_batch(q0, n_this, scfg.k, scfg);
            const auto batch_end = Clock::now();
            const double batch_us = US(batch_end - batch_start).count();
            // Per-query latency: average across the batch (the batch ran
            // concurrently, so wall-time / n_this is the throughput view).
            const double per_q_us = batch_us / static_cast<double>(n_this);
            for (uint32_t i = 0; i < n_this; i++) {
                process_result(base + i, std::move(batch_results[i]), per_q_us);
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

        // Proximity: how close results get to the k-th true NN.
        // ratio = d_target / d_result  (>=1.0 = at or inside target radius).
        // Reports: in-band fraction (results at/inside target), mean ratio over
        // all results, and the ratio distribution of OUT-of-band results so you
        // can see how far past the boundary misses land (e.g. p50=0.8 means the
        // median miss is ~25% outside the target distance).
        if (have_gt_dists && prox_total > 0) {
            const double in_band =
                static_cast<double>(prox_in) /
                static_cast<double>(prox_total);
            double prox_sum = 0.0;
            std::vector<double> miss_ratios;
            miss_ratios.reserve(prox_total - prox_in);
            for (double r : prox_ratios) {
                prox_sum += r;
                if (r < 1.0) miss_ratios.push_back(r);
            }
            const double mean_ratio = prox_sum /
                static_cast<double>(prox_total);
            std::cout << "[benchmark] proximity: in-band="
                      << std::fixed << std::setprecision(4) << in_band
                      << " (" << prox_in << "/" << prox_total
                      << " results at/inside d" << k << ")"
                      << ", mean_ratio=" << std::setprecision(4)
                      << mean_ratio << "\n";
            std::cout << "[benchmark] proximity: miss_ratio";
            if (miss_ratios.empty()) {
                std::cout << "=n/a (no out-of-band results)\n";
            } else {
                std::sort(miss_ratios.begin(), miss_ratios.end());
                const double mp25 = percentile(miss_ratios, 25.0);
                const double mp50 = percentile(miss_ratios, 50.0);
                const double mp75 = percentile(miss_ratios, 75.0);
                const double mp90 = percentile(miss_ratios, 90.0);
                std::cout << " p25=" << std::fixed
                          << std::setprecision(4) << mp25
                          << " p50=" << mp50
                          << " p75=" << mp75
                          << " p90=" << mp90
                          << " (over " << miss_ratios.size()
                          << " out-of-band; 1.0=target edge)\n";
            }
        }
        std::cout << "[benchmark] latency p50: " << std::fixed
                  << std::setprecision(3) << p50_us / 1000.0 << "ms, p99: "
                  << p99_us / 1000.0 << "ms\n";
        std::cout << "[benchmark] total search time: " << std::fixed
                  << std::setprecision(3) << total_sec << "s\n";
         std::cout << "[benchmark] QPS: " << std::fixed
                   << std::setprecision(1) << qps << "\n";
         if (idx->is_paged()) {
             std::cout << "[benchmark] cache: graph_reads="
                       << searcher.cache_graph_reads()
                       << " code_reads=" << searcher.cache_code_reads() << "\n";
             const auto as = searcher.cache_admission_stats();
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
             const uint64_t tl_h = searcher.tl_hits();
             const uint64_t tl_m = searcher.tl_misses();
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
