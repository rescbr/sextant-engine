#pragma once

/// @file searcher.hpp
/// Searcher — read-only query serving over an Index.
///
/// Phase C: search + cache management (rebalance + stats) moved out of the
/// Engine god-object. Searcher holds an Index& and serves queries through
/// the Index's VamanaCore + NodeStore stack. It owns the adaptive cache
/// rebalance state (cadence + CAS guard); the cache itself lives on the
/// Index's PagedNodeStore.
///
/// Lifecycle: Engine constructs a Searcher over an Index after build() or
/// open() completes. Phase E will delete Engine and callers will construct
/// a Searcher directly.
///
/// Index loading (open/load_sidecars) STAYS on Engine for now — it's the
/// natural `Index::read(path)` work and lands when Index grows that static.
/// Searcher takes an already-populated Index.

#include "sextant/config.hpp"
#include "sextant/index.hpp"
#include "sextant/types.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace sextant {

class Searcher {
public:
    /// Construct over a populated Index (built or opened). The Index must
    /// outlive the Searcher.
    explicit Searcher(Index& index);

    /// Search for k nearest neighbors of `query`. Returns candidates sorted
    /// ascending by distance. `query` is dim floats. The LUT is built from
    /// `query` via the Index's quantizer; FP16 conversion happens internally
    /// for the hybrid FP16+PQ distance path.
    std::vector<Candidate> search(const float* query, uint32_t k,
                                   const SearchConfig& config);

    // --- Cache management (paged mode only; no-op otherwise) ---

    /// Enable/disable adaptive cache rebalance (default: enabled in paged mode).
    void set_cache_rebalance_enabled(bool enabled) {
        cache_rebalance_enabled_ = enabled;
    }

    /// Force a rebalance now (e.g. for benchmarks). No-op if not paged.
    void rebalance_caches();

    // --- Diagnostics ---

    /// L2 (BlockCache) cache stats. Zero when not paged.
    uint64_t cache_graph_reads() const;
    uint64_t cache_code_reads() const;

    struct AdmissionStats {
        uint64_t hits_window = 0;
        uint64_t hits_probation = 0;
        uint64_t hits_protected = 0;
        uint64_t misses = 0;
        uint64_t evictions_admitted = 0;
        uint64_t evictions_rejected = 0;
    };
    AdmissionStats cache_admission_stats() const;

    /// L1 (TLBlockCache) hit/miss counters. Zero when not paged.
    uint64_t tl_hits() const;
    uint64_t tl_misses() const;

    /// Number of nodes held in the MemGraph neighborhood cache.
    uint32_t memgraph_cached_count() const;

private:
    Index& index_;
    bool cache_rebalance_enabled_ = true;
    std::atomic<uint64_t> search_count_{0};
    std::atomic<bool> rebalancing_{false};
    static constexpr uint64_t kRebalanceCadenceInitial = 1000;
    static constexpr uint64_t kRebalanceCadenceMax = 16000;
    uint64_t rebalance_cadence_ = kRebalanceCadenceInitial;

    void maybe_rebalance_();
};

}  // namespace sextant
