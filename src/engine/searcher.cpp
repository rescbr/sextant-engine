// Searcher — query serving + cache management.
//
// Architecture (Layer 3+4 revised 2026-07-21 after the per-worker cache
// regression): the Searcher owns a pool of workers, each holding only
// VamanaTLS (scratch). The cache-bearing state (PagedNodeStore, BlockCache,
// MemGraph, VamanaCore) is shared — owned by the Index, referenced by
// workers. This restores the Layer 2 cache-sharing pattern that hit
// 1704 QPS at 8t, while keeping the Searcher-owned pool API and the
// Layer 4 cleanups (BuildContext, BeamQuery, etc.).

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
    ctpl::thread_pool_tls<SearchWorkerState> pool;

    explicit PoolImpl(uint32_t n_threads)
        : pool(n_threads,
               [](size_t /*id*/, std::shared_ptr<SearchWorkerState>& w) {
                   w = std::make_shared<SearchWorkerState>();
               }) {}
};

// ===========================================================================
// Searcher ctor / dtor
// ===========================================================================

Searcher::Searcher(Index& index, uint32_t num_threads)
    : index_(index),
      num_threads_(std::max(1u, num_threads)),
      pool_(std::make_unique<PoolImpl>(std::max(1u, num_threads))) {}

Searcher::~Searcher() = default;

uint32_t Searcher::num_threads() const { return num_threads_; }

// ===========================================================================
// search_body_ — the single core path used by search() and search_batch()
// ===========================================================================

std::vector<Candidate> Searcher::search_body_(const float* query, uint32_t k,
                                                const SearchConfig& config,
                                                VamanaTLS& tls) {
    if (!index_.quantizer || !index_.core) {
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

    BeamQuery sq;
    sq.query_lut = lut.data();
    sq.query_fp16 = query_fp16.data();
    return index_.core->search(sq, k, config.L_search, config.io_limit, tls);
}

// ===========================================================================
// search_one_async() — push one task, return future immediately.
// ===========================================================================

std::future<std::vector<Candidate>> Searcher::search_one_async(
    const float* query, uint32_t k, const SearchConfig& config) {
    if (!index_.quantizer || index_.count == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Searcher::search: index not built/opened "
                    "(quantizer/count missing)");
    }
    return pool_->pool.push(
        [this, query, k, &config](size_t /*id*/, SearchWorkerState& w) {
            return search_body_(query, k, config, w.tls);
        });
}

// ===========================================================================
// search() — synchronous: push one task, wait on its future.
// ===========================================================================

std::vector<Candidate> Searcher::search(const float* query, uint32_t k,
                                         const SearchConfig& config) {
    auto result = search_one_async(query, k, config).get();

    if (index_.paged_store && cache_rebalance_enabled_) {
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
        futs.push_back(search_one_async(q, k, config));
    }
    for (uint32_t i = 0; i < n; i++) {
        results[i] = futs[i].get();
    }

    if (index_.paged_store && cache_rebalance_enabled_) {
        for (uint32_t i = 0; i < n; i++) maybe_rebalance_();
    }
    return results;
}

// ===========================================================================
// Cache management + diagnostics
// ===========================================================================

uint64_t Searcher::cache_graph_reads() const {
    return index_.paged_store ? index_.paged_store->graph_reads() : 0;
}

uint64_t Searcher::cache_code_reads() const {
    return index_.paged_store ? index_.paged_store->code_reads() : 0;
}

Searcher::AdmissionStats Searcher::cache_admission_stats() const {
    if (!index_.paged_store) return {};
    const auto cs = index_.paged_store->cache_stats();
    return {
        cs.graph.hits_window + cs.code.hits_window,
        cs.graph.hits_probation + cs.code.hits_probation,
        cs.graph.hits_protected + cs.code.hits_protected,
        cs.graph.misses + cs.code.misses,
        cs.graph.evictions_admitted + cs.code.evictions_admitted,
        cs.graph.evictions_rejected + cs.code.evictions_rejected,
    };
}

void Searcher::rebalance_caches() {
    if (index_.paged_store) {
        index_.paged_store->maybe_rebalance_caches();
    }
}

void Searcher::maybe_rebalance_() {
    const uint64_t n = search_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n < rebalance_cadence_) return;
    bool expected = false;
    if (!rebalancing_.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel)) {
        return;
    }
    const double prev_frac = index_.paged_store->graph_cache_fraction();
    index_.paged_store->maybe_rebalance_caches();
    const double new_frac = index_.paged_store->graph_cache_fraction();
    rebalancing_.store(false, std::memory_order_release);
    if (std::abs(new_frac - prev_frac) < 1e-6) {
        rebalance_cadence_ = std::min(kRebalanceCadenceMax,
                                      rebalance_cadence_ * 2);
    } else {
        rebalance_cadence_ = kRebalanceCadenceInitial;
    }
}

uint32_t Searcher::memgraph_cached_count() const {
    return index_.memgraph ? index_.memgraph->cached_count() : 0;
}

uint64_t Searcher::tl_hits() const {
    return index_.paged_store ? index_.paged_store->tl_hits() : 0;
}

uint64_t Searcher::tl_misses() const {
    return index_.paged_store ? index_.paged_store->tl_misses() : 0;
}

}  // namespace sextant
