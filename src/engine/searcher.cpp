// Searcher — query serving + cache management.
//
// Layer 3 (2026-07-21): Searcher now owns a ctpl::thread_pool_tls of
// SearchWorkerState, each holding a VamanaTLS. This kills the thread_local
// VamanaTLS + g_search_scratch in vamana_core.cpp. Phase AB only — later
// phases move the PagedNodeStore/VamanaCore per-worker.

#include "sextant/searcher.hpp"

#include "sidecar_io.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <memory>
#include <vector>

namespace sextant {

using engine_detail::fill_header;
using engine_detail::read_exact;
using engine_detail::write_padded;

// ===========================================================================
// PoolImpl — hides the ctpl type from the header.
// ===========================================================================

struct Searcher::PoolImpl {
    // The pool init lambda captures Index metadata by value (paths, sizes,
    // entry points). Each worker constructs its own PagedNodeStore +
    // MemGraph + VamanaCore from these. The cache budget is split evenly
    // across workers; per-worker cache guards against LRU thrashing within
    // one worker.
    PoolImpl(uint32_t n_threads, const Index& idx, uint32_t num_shards)
        : pool(n_threads, make_init_(idx, n_threads, num_shards)) {}

    ctpl::thread_pool_tls<SearchWorkerState> pool;

    private:
    static ctpl::thread_pool_tls<SearchWorkerState>::init_func_type
    make_init_(const Index& idx, uint32_t n_threads, uint32_t num_shards) {
        // Each worker gets the FULL cache budget (not 1/N). Total RSS is
        // N × cache_bytes, but each worker's cache is sized to the working
        // set — critical for small-to-medium datasets where 1/N would
        // thrash. At billion scale the cache is already undersized relative
        // to the index, so per-worker sizing doesn't change the miss rate
        // but eliminates cross-core BlockCache contention (the Layer 3 goal).
        const uint64_t per_worker_cache = std::max<uint64_t>(
            16ull * 1024 * 1024, idx.cache_size_bytes);
        const std::string graph_path = idx.path + ".graph";
        const std::string codes_path = idx.path + ".codes";
        const std::string ball_path  = idx.path + ".ball";
        const uint32_t node_size = idx.node_size;
        const uint32_t code_size = idx.code_size;
        const uint32_t total_count = static_cast<uint32_t>(idx.count);
        const uint32_t dim = static_cast<uint32_t>(idx.dim);
        const auto entry_points = idx.entry_points;
        const auto params = idx.params;
        const bool have_memgraph = (idx.memgraph != nullptr);
        const bool flat_mode = (idx.flat_store != nullptr);
        PqQuantizer* quantizer = idx.quantizer.get();
        NodeStore* shared_store = idx.flat_store.get();
        VamanaCore* shared_core = idx.core.get();

        // The init lambda runs eagerly on each worker thread when the pool
        // is constructed. We MUST NOT throw here (an uncaught exception in
        // a thread calls std::terminate). For a flat-store Index we wire up
        // shared pointers; for a paged Index we defer PagedNodeStore/MemGraph
        // construction to the first search (search_body_ calls ensure_store_*).
        return [=](size_t /*id*/, std::shared_ptr<SearchWorkerState>& w) {
            auto ws = std::make_shared<SearchWorkerState>();
            if (flat_mode) {
                ws->shared_store = shared_store;
                ws->shared_core = shared_core;
            }
            // Paged-mode construction is deferred; capture the params needed.
            ws->paged_graph_path = graph_path;
            ws->paged_codes_path = codes_path;
            ws->paged_ball_path = ball_path;
            ws->paged_node_size = node_size;
            ws->paged_code_size = code_size;
            ws->paged_total_count = total_count;
            ws->paged_dim = dim;
            ws->paged_entry_points = entry_points;
            ws->paged_params = params;
            ws->paged_have_memgraph = have_memgraph;
            ws->paged_quantizer = quantizer;
            ws->paged_num_shards = num_shards;
            ws->paged_cache_bytes = per_worker_cache;
            w = ws;
        };
    }
};

// each_worker_ is defined out-of-line here (after PoolImpl is complete) so
// the template can dereference pool_->pool.tls_slots(). Inline definition in
// the header would require PoolImpl to be complete there.
template <typename F>
void Searcher::each_worker_(F&& fn) const {
    for (const auto& slot : pool_->pool.tls_slots()) {
        if (slot && slot->store) fn(*slot);
    }
}

// ===========================================================================
// Searcher ctor / dtor
// ===========================================================================

Searcher::Searcher(Index& index, uint32_t num_threads)
    : index_(index),
      num_threads_(std::max(1u, num_threads)),
      pool_(std::make_unique<PoolImpl>(
          std::max(1u, num_threads), index,
          std::max(1u, std::thread::hardware_concurrency()))) {}

Searcher::~Searcher() = default;

uint32_t Searcher::num_threads() const { return num_threads_; }

// ===========================================================================
// ensure_paged_state_ — lazily construct per-worker PagedNodeStore + core.
// ===========================================================================

void Searcher::ensure_paged_state_(SearchWorkerState& w) {
    if (w.shared_core) return;        // flat mode: shared with Index, done.
    if (w.core) return;               // paged mode: already constructed.
    if (w.paged_graph_path.empty()) return;  // empty Index: nothing to build.

    // Paged mode first-use construction.
    w.store = std::make_unique<PagedNodeStore>(
        w.paged_graph_path, w.paged_codes_path,
        w.paged_node_size, w.paged_code_size,
        w.paged_num_shards, w.paged_cache_bytes);
    w.core = std::make_unique<VamanaCore>(
        VamanaParams::from_resolved(w.paged_params,
                                     static_cast<Dim>(w.paged_dim)),
        *w.paged_quantizer);
    if (w.paged_have_memgraph) {
        w.memgraph = std::make_unique<MemGraph>(
            w.paged_graph_path, w.paged_codes_path, w.paged_ball_path,
            w.paged_node_size, w.paged_code_size,
            w.paged_total_count, w.paged_dim,
            w.paged_entry_points, /*num_hops=*/3);
        w.memgraph->set_backing(w.store.get());
        w.core->set_store(w.memgraph.get());
    } else {
        w.core->set_store(w.store.get());
    }
    w.core->set_search_state(w.paged_total_count, w.paged_entry_points);
}

// ===========================================================================
// search_body_ — the single core path used by search() and search_batch()
// ===========================================================================

std::vector<Candidate> Searcher::search_body_(const float* query, uint32_t k,
                                                const SearchConfig& config,
                                                VamanaCore& core,
                                                VamanaTLS& tls) {
    if (!index_.quantizer || core.size() == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Searcher::search: index not built/opened "
                    "(quantizer/core missing)");
    }
    if (k == 0) return {};

    // Preprocess the query into a PQ LUT.
    const uint32_t lut_sz = index_.quantizer->lut_size();
    std::vector<float> lut(lut_sz > 0 ? lut_sz : 1, 0.0f);
    if (lut_sz > 0) {
        index_.quantizer->preprocess_query(query, lut.data());
    }

    // Convert the query to FP16 for the hybrid FP16+PQ distance path
    // (MemGraph ball nodes use l2sq_f16; the rest use PQ lut_distance).
    std::vector<float16_t> query_fp16(index_.dim);
    for (uint32_t d = 0; d < index_.dim; d++) {
        query_fp16[d] = static_cast<float16_t>(query[d]);
    }

    return core.search(lut.data(), k, config.L_search,
                       config.io_limit, tls, query_fp16.data());
}

// ===========================================================================
// search_one_async() — push one task, return future immediately.
// ===========================================================================

std::future<std::vector<Candidate>> Searcher::search_one_async(
    const float* query, uint32_t k, const SearchConfig& config) {
    return pool_->pool.push(
        [this, query, k, &config](size_t /*id*/, SearchWorkerState& w) {
            VamanaCore* core;
            if (index_.flat_store && index_.core) {
                core = index_.core.get();
            } else {
                ensure_paged_state_(w);
                core = w.active_core();
            }
            return search_body_(query, k, config, *core, w.tls);
        });
}

// ===========================================================================
// search() — synchronous: push one task, wait on its future.
// ===========================================================================

std::vector<Candidate> Searcher::search(const float* query, uint32_t k,
                                         const SearchConfig& config) {
    // Validate upfront so an unopened Index throws the expected Error rather
    // than a DirectFile error from lazy pool init.
    if (!index_.quantizer || index_.count == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Searcher::search: index not built/opened "
                    "(quantizer/count missing)");
    }
    auto fut = pool_->pool.push(
        [this, query, k, &config](size_t /*id*/, SearchWorkerState& w) {
            // If the Index is in flat-store mode (live insert path), bypass
            // the per-worker paged state and use the Index's core + flat
            // store directly. Otherwise lazily construct per-worker state.
            VamanaCore* core;
            if (index_.flat_store && index_.core) {
                core = index_.core.get();
            } else {
                ensure_paged_state_(w);
                core = w.active_core();
            }
            return search_body_(query, k, config, *core, w.tls);
        });
    auto result = fut.get();

    if (cache_rebalance_enabled_ && index_.is_paged()) {
        maybe_rebalance_();
    }
    return result;
}

// ===========================================================================
// search_batch() — fan a batch across the pool.
// ===========================================================================

std::vector<std::vector<Candidate>> Searcher::search_batch(
    const float* queries, uint32_t n,
    uint32_t k, const SearchConfig& config) {
    std::vector<std::vector<Candidate>> results(n);
    if (n == 0) return results;

    std::vector<std::future<std::vector<Candidate>>> futs;
    futs.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        const float* q = &queries[static_cast<size_t>(i) * index_.dim];
        futs.push_back(pool_->pool.push(
            [this, q, k, &config](size_t /*id*/, SearchWorkerState& w) {
                VamanaCore* core;
                if (index_.flat_store && index_.core) {
                    core = index_.core.get();
                } else {
                    ensure_paged_state_(w);
                    core = w.active_core();
                }
                return search_body_(q, k, config, *core, w.tls);
            }));
    }
    for (uint32_t i = 0; i < n; i++) {
        results[i] = futs[i].get();
    }

    if (cache_rebalance_enabled_ && index_.is_paged()) {
        for (uint32_t i = 0; i < n; i++) maybe_rebalance_();
    }
    return results;
}

// ===========================================================================
// Cache management + diagnostics (aggregated across all workers)
// ===========================================================================

uint64_t Searcher::cache_graph_reads() const {
    uint64_t sum = 0;
    each_worker_([&](const SearchWorkerState& w) {
        if (w.store) sum += w.store->graph_reads();
    });
    return sum;
}

uint64_t Searcher::cache_code_reads() const {
    uint64_t sum = 0;
    each_worker_([&](const SearchWorkerState& w) {
        if (w.store) sum += w.store->code_reads();
    });
    return sum;
}

Searcher::AdmissionStats Searcher::cache_admission_stats() const {
    AdmissionStats total{};
    each_worker_([&](const SearchWorkerState& w) {
        if (!w.store) return;
        const auto cs = w.store->cache_stats();
        total.hits_window     += cs.graph.hits_window     + cs.code.hits_window;
        total.hits_probation  += cs.graph.hits_probation  + cs.code.hits_probation;
        total.hits_protected  += cs.graph.hits_protected  + cs.code.hits_protected;
        total.misses          += cs.graph.misses          + cs.code.misses;
        total.evictions_admitted  += cs.graph.evictions_admitted  + cs.code.evictions_admitted;
        total.evictions_rejected  += cs.graph.evictions_rejected  + cs.code.evictions_rejected;
    });
    return total;
}

void Searcher::rebalance_caches() {
    each_worker_([](SearchWorkerState& w) {
        if (w.store) w.store->maybe_rebalance_caches();
    });
}

void Searcher::maybe_rebalance_() {
    const uint64_t n = search_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n < rebalance_cadence_) return;
    bool expected = false;
    if (!rebalancing_.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel)) {
        return;
    }
    double prev_frac_sum = 0.0, new_frac_sum = 0.0;
    uint32_t worker_count = 0;
    each_worker_([&](SearchWorkerState& w) {
        if (!w.store) return;
        prev_frac_sum += w.store->graph_cache_fraction();
        w.store->maybe_rebalance_caches();
        new_frac_sum += w.store->graph_cache_fraction();
        ++worker_count;
    });
    rebalancing_.store(false, std::memory_order_release);
    if (worker_count == 0) return;
    const double prev_frac = prev_frac_sum / worker_count;
    const double new_frac  = new_frac_sum  / worker_count;
    if (std::abs(new_frac - prev_frac) < 1e-6) {
        rebalance_cadence_ = std::min(kRebalanceCadenceMax,
                                      rebalance_cadence_ * 2);
    } else {
        rebalance_cadence_ = kRebalanceCadenceInitial;
    }
}

uint32_t Searcher::memgraph_cached_count() const {
    uint32_t count = 0;
    bool found = false;
    each_worker_([&](const SearchWorkerState& w) {
        if (!found && w.memgraph) {
            count = w.memgraph->cached_count();
            found = true;
        }
    });
    if (found) return count;
    return index_.memgraph ? index_.memgraph->cached_count() : 0;
}

uint64_t Searcher::tl_hits() const {
    uint64_t sum = 0;
    each_worker_([&](const SearchWorkerState& w) {
        if (w.store) sum += w.store->tl_hits();
    });
    return sum;
}

uint64_t Searcher::tl_misses() const {
    uint64_t sum = 0;
    each_worker_([&](const SearchWorkerState& w) {
        if (w.store) sum += w.store->tl_misses();
    });
    return sum;
}

}  // namespace sextant
