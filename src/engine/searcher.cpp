// Searcher — query serving + cache management.
//
// Extracted from search.cpp during Layer 2 Phase C. The open()/load_sidecars
// path stays on Engine for now (it's the natural Index::read work); Searcher
// is constructed over an already-populated Index.

#include "sextant/searcher.hpp"

#include "sidecar_io.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace sextant {

using engine_detail::fill_header;
using engine_detail::read_exact;
using engine_detail::write_padded;

// ===========================================================================
// Searcher ctor
// ===========================================================================

Searcher::Searcher(Index& index) : index_(index) {}

// ===========================================================================
// search()
// ===========================================================================

std::vector<Candidate> Searcher::search(const float* query, uint32_t k,
                                        const SearchConfig& config) {
    if (!index_.quantizer || !index_.core) {
        throw Error(ErrorCode::InvalidParam,
                    "Searcher::search: index not built/opened "
                    "(quantizer/core missing)");
    }
    if (k == 0) {
        return {};
    }

    // Preprocess the query into a PQ LUT.
    const uint32_t lut_sz = index_.quantizer->lut_size();
    std::vector<float> lut(lut_sz > 0 ? lut_sz : 1, 0.0f);
    if (lut_sz > 0) {
        index_.quantizer->preprocess_query(query, lut.data());
    }

    // VamanaCore::search resolves internal_ids → row_ids for us.
    // Convert the query to FP16 for the hybrid FP16+PQ distance path
    // (MemGraph ball nodes use l2sq_f16; the rest use PQ lut_distance).
    std::vector<float16_t> query_fp16(index_.dim);
    for (uint32_t d = 0; d < index_.dim; d++) {
        query_fp16[d] = static_cast<float16_t>(query[d]);
    }
    auto results =
        index_.core->search(lut.data(), k, config.L_search, config.io_limit,
                            query_fp16.data());

    // Adaptive cache rebalance (paged mode only). Cheap relaxed-atomic add per
    // search; the rare resize is CAS-guarded so only one thread runs it.
    if (index_.paged_store && cache_rebalance_enabled_) {
        maybe_rebalance_();
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
    // CAS: only one thread rebalances at a time. Losers bail (they'll retry
    // next cadence window; the counter is monotonic so no double-fire).
    bool expected = false;
    if (!rebalancing_.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel)) {
        return;
    }
    const double prev_frac = index_.paged_store->graph_cache_fraction();
    index_.paged_store->maybe_rebalance_caches();   // may call BlockCache::resize
    const double new_frac = index_.paged_store->graph_cache_fraction();
    rebalancing_.store(false, std::memory_order_release);
    // Adaptive cadence: if the split didn't move, back off (steady state).
    // If it moved, reset to initial (workload shifting — stay responsive).
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
