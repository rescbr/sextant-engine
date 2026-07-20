#pragma once

/// @file engine.hpp
/// Top-level engine facade. Orchestrates build, search, and insert.
///
/// The Engine owns the PqQuantizer, VamanaCore, and storage layer. It exposes
/// a simple API: build from a VectorSource, search with a query vector.

#include <sextant/types.hpp>
#include <sextant/vector_source.hpp>
#include <sextant/config.hpp>
#include <sextant/index.hpp>
#include <sextant/searcher.hpp>
#include <sextant/estimator.hpp>
#include <atomic>
#include <memory>
#include <string>
#include <functional>
#include <utility>
#include <vector>

namespace sextant {

class PqQuantizer;
class VamanaCore;
class NodeStore;
class FlatNodeStore;
class PagedNodeStore;
class MemGraph;

/// The Sextant engine. Owns the index state.
class Engine {
public:
    Engine() = default;
    ~Engine() = default;

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// Build an index from a vector source. Writes sidecar files.
    /// Auto-resolves params from `config` via resolve_params() (any field
    /// set to 0/auto in `config` is resolved heuristically — but pq_m and
    /// pq_bits MUST be explicit; pass1 throws otherwise).
    /// Phase B: thin delegator; the real work is in Builder.
    BuildResult build(VectorSource& source, const std::string& index_path,
                      const BuildConfig& config);

    /// Build overload taking a fully-resolved params struct directly (e.g.
    /// the result of estimate_config). Skips resolve_params entirely — the
    /// caller is responsible for producing a complete ResolvedParams. This
    /// is the path used by `sextant autobuild` so the analyze→build chain
    /// stays in-process (no string serialization round-trip).
    BuildResult build(VectorSource& source, const std::string& index_path,
                      const ResolvedParams& params);

    /// Estimate the optimal build configuration from a data sample. Samples the
    /// source (reservoir), trains PQ, measures LID, builds one or more
    /// mini-indices (using the canonical Engine::build() path) on the sample at
    /// candidate (R, alpha) values, measures search quality and graph structure,
    /// and returns the optimal config for the user's recall/proximity target.
    ///
    /// Does NOT modify this Engine's state. Each mini-build uses a fresh local
    /// Engine instance writing to a temp directory.
    ///
    /// Fields in `overrides` that control estimation:
    ///   - target_topk: the k at which recall/proximity are measured and all
    ///     mini-builds are evaluated (default 100, VIBE convention). Proximity
    ///     semantics depend on k — at k=100 the k-th NN distance is larger than
    ///     at k=10, so proximity is looser at higher k.
    ///   - proximity_target / recall_target: quality goals (drive R and alpha
    ///     selection). Dual-gate: when BOTH are set, both must be met (the
    ///     stricter binds); when one is set, only that gates; when neither is
    ///     set, the default recall@0.95 applies. A +0.03 buffer is added to the
    ///     recall threshold because the 20K mini-build sample overestimates
    ///     recall vs the full-scale index (fewer distractors, shorter paths).
    ///   - pq_max_distortion: PQ quality bound (drives m/bits selection). When
    ///     recall_target is set, distortion is relaxed to 0.20 and a mini-build
    ///     verify at target_topk is the real gate.
    ///   - R, alpha, L, pq_m, pq_bits: if non-zero, locked (skip estimation)
    ///   - closure_f_target, closure_d_eff: closure factor inputs (0 = estimated)
    ///   - max_occlusion: if 0, auto (= max(L_build, R+1))
    ///
    /// Phase D: thin delegator; the real work is in Estimator
    /// (src/engine/estimator.cpp).
    EstimateResult estimate_config(VectorSource& source, const BuildConfig& overrides);

    /// Load an index from sidecar files for searching.
    void open(const std::string& index_path);

    /// Search for k nearest neighbors. Returns candidate row_ids.
    /// Phase C: thin delegator; the real work is in Searcher.
    std::vector<Candidate> search(const float* query, uint32_t k,
                                   const SearchConfig& config);

    /// Insert a single vector (live insert after build).
    void insert(const float* vec, Dim dim, RowId row_id);

    /// Flush any pending state to disk.
    void flush();

    bool is_open() const { return opened_; }

    /// Set the search cache size override (0 = auto from graph_size/RAM).
    /// Must be called before open().
    void set_cache_size(uint64_t bytes) { cache_size_override_ = bytes; }

    /// Enable/disable adaptive cache rebalance (default: enabled in paged mode).
    /// Stored on Engine until the Searcher is constructed (post-build/open),
    /// then propagated. Must be called before open()/build().
    void set_cache_rebalance_enabled(bool enabled) {
        cache_rebalance_enabled_ = enabled;
        if (searcher_) searcher_->set_cache_rebalance_enabled(enabled);
    }
    uint64_t count() const { return index_ ? index_->count : 0; }
    Dim dim() const { return index_ ? index_->dim : 0; }

    /// True when the engine is in SSD-resident (paged) mode — flat RAM buffers
    /// are NOT loaded. Used by tests to verify the low-idle-RAM invariant.
    bool is_paged() const { return index_ && index_->is_paged(); }

    /// Cache diagnostics (only valid when is_paged()). Phase C: delegate to
    /// Searcher.
    uint64_t cache_graph_reads() const {
        return searcher_ ? searcher_->cache_graph_reads() : 0;
    }
    uint64_t cache_code_reads() const {
        return searcher_ ? searcher_->cache_code_reads() : 0;
    }
    /// W-TinyLFU admission stats. Mirrors Searcher::AdmissionStats.
    using AdmissionStats = Searcher::AdmissionStats;
    AdmissionStats cache_admission_stats() const {
        return searcher_ ? searcher_->cache_admission_stats() : AdmissionStats{};
    }

    /// Force a rebalance now. No-op if not in paged mode.
    void rebalance_caches() {
        if (searcher_) searcher_->rebalance_caches();
    }
    /// Thread-local L1 cache hit/miss counters.
    uint64_t tl_hits() const {
        return searcher_ ? searcher_->tl_hits() : 0;
    }
    uint64_t tl_misses() const {
        return searcher_ ? searcher_->tl_misses() : 0;
    }
    /// Number of nodes held in the MemGraph neighborhood cache (0 if none).
    uint32_t memgraph_cached_count() const {
        return searcher_ ? searcher_->memgraph_cached_count() : 0;
    }
    /// True when flat buffers are resident in RAM (build or post-insert).
    bool has_flat_buffers() const { return index_ && index_->has_flat_buffers(); }

private:
    bool opened_ = false;
    std::string index_path_;  ///< Set by open()/build() before index_ exists.
    std::unique_ptr<Index> index_;
    std::unique_ptr<Searcher> searcher_;  ///< Lazily constructed over index_.

    uint64_t cache_size_override_ = 0;  ///< 0 = auto
    bool cache_rebalance_enabled_ = true;  ///< Propagated to searcher_ on construct.

    /// Load sidecar files for search.
    void load_sidecars();
};

}  // namespace sextant
