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
    BuildResult build(VectorSource& source, const std::string& index_path,
                      const BuildConfig& config);

    /// Build overload taking a fully-resolved params struct directly (e.g.
    /// the result of estimate_config). Skips resolve_params entirely — the
    /// caller is responsible for producing a complete ResolvedParams. This
    /// is the path used by `sextant autobuild` so the analyze→build chain
    /// stays in-process (no string serialization round-trip).
    BuildResult build(VectorSource& source, const std::string& index_path,
                      const ResolvedParams& params);

    /// Probe PQ (m, bits) selection on a sample. Runs the full auto-selection
    /// policy (recall floor + cache-band preference) on `sample` (n × dim
    /// floats) and returns the resolved (m, bits). Used by the build path
    /// (pass1) and by --explain (dry-run preview without building).
    /// `pq_m`/`pq_bits` in `params`: 0 = auto, else fixed. Does not modify
    /// engine state.
    struct ProbedRow {
        uint16_t m;
        uint8_t bits;
        uint32_t code_bytes;
        uint32_t table_bytes;
        double distortion;       ///< median |1 - pq_dist/true_dist| (≥0; 0 = perfect)
        double band_recall;      ///< cluster-aware recall (diagnostic)
        double tie_fraction;     ///< frac queries with >topk tied at @k (diagnostic)
        double tie30_fraction;   ///< frac queries with >30 tied at @30 (diagnostic)
        double cost;
    };
    struct PqSelection {
        uint16_t m;
        uint8_t bits;
        std::vector<ProbedRow> all;     ///< every probed config (for display)
        std::string reason;             ///< selection rationale
    };
    static PqSelection probe_pq_config(const float* sample, uint64_t n, Dim dim,
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
    /// Implementation lives in src/engine/estimate_config.cpp (Part 2).
    EstimateResult estimate_config(VectorSource& source, const BuildConfig& overrides);

    /// Load an index from sidecar files for searching.
    void open(const std::string& index_path);

    /// Search for k nearest neighbors. Returns candidate row_ids.
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
    /// When enabled, the two caches (graph/code) are periodically resized based
    /// on observed hit/miss ratios. Must be called before open(); zero overhead
    /// when the index is not paged or when disabled.
    void set_cache_rebalance_enabled(bool enabled) { cache_rebalance_enabled_ = enabled; }
    uint64_t count() const { return index_ ? index_->count : 0; }
    Dim dim() const { return index_ ? index_->dim : 0; }

    /// True when the engine is in SSD-resident (paged) mode — flat RAM buffers
    /// are NOT loaded. Used by tests to verify the low-idle-RAM invariant.
    bool is_paged() const { return index_ && index_->is_paged(); }

    /// Cache diagnostics (only valid when is_paged()).
    uint64_t cache_graph_reads() const;
    uint64_t cache_code_reads() const;
    /// W-TinyLFU admission stats: {window, probation, protected} hit counts,
    /// misses, and admission decisions. All zero when not paged.
    struct AdmissionStats {
        uint64_t hits_window = 0;
        uint64_t hits_probation = 0;
        uint64_t hits_protected = 0;
        uint64_t misses = 0;
        uint64_t evictions_admitted = 0;
        uint64_t evictions_rejected = 0;
    };
    AdmissionStats cache_admission_stats() const;

    /// Periodically rebalance the graph/code cache split based on hit/miss
    /// ratios. No-op if not in paged mode. Wired up but NOT called
    /// automatically from the hot search loop yet (follow-up).
    void rebalance_caches();
    /// Thread-local L1 cache hit/miss counters.
    uint64_t tl_hits() const;
    uint64_t tl_misses() const;
    /// Number of nodes held in the MemGraph neighborhood cache (0 if none).
    uint32_t memgraph_cached_count() const;
    /// True when flat buffers are resident in RAM (build or post-insert).
    bool has_flat_buffers() const { return index_ && index_->has_flat_buffers(); }

private:
    bool opened_ = false;
    std::string index_path_;  ///< Set by open()/build() before index_ exists.
    std::unique_ptr<Index> index_;

    // Build-only scratch (not part of the post-build Index handoff).
    std::vector<float> entry_centroids_;  // k × dim, FP32 (k-means centroids for entry points)

    uint64_t cache_size_override_ = 0;  ///< 0 = auto

    // Adaptive cache rebalance (paged mode only).
    bool cache_rebalance_enabled_ = true;
    std::atomic<uint64_t> search_count_{0};
    std::atomic<bool> rebalancing_{false};
    static constexpr uint64_t kRebalanceCadenceInitial = 1000;
    static constexpr uint64_t kRebalanceCadenceMax = 16000;
    uint64_t rebalance_cadence_ = kRebalanceCadenceInitial;
    /// Private: per-search adaptive rebalance hook.
    void maybe_rebalance_();

    /// Build helpers.
    /// pass1 fills the reservoir, resolves pq_bits (if auto, via global probe),
    /// constructs + trains the quantizer. After return, code_size_ is valid.
    void pass1_sample_and_train(VectorSource& source,
                                 const ResolvedParams& params);
    void pass2_encode(VectorSource& source, const ResolvedParams& params);
    void parallel_construct(const ResolvedParams& params);

    /// Core construct loop: chunked work-stealing + T5 dynamic L_build + progress
    /// logger. Shared by K==1 (full graph) and K>1 (per-shard). The row-id mapper
    /// translates local construct IDs to global RowIds (identity for K==1,
    /// shard membership table for K>1).
    void construct_into(VamanaCore& core, uint32_t count,
                        const std::function<RowId(uint32_t)>& row_id_at,
                        uint32_t lut_sz, uint32_t nthreads,
                        const char* label);

    /// Unified build (K==1 fast path + K>1 partitioned): partition → per-shard
    /// build → merge → flush. K==1 builds the full graph directly (no
    /// partition/merge) and is bit-identical to the former monolithic path.
    BuildResult build_partitioned(VectorSource& source,
                                   const std::string& index_path,
                                   const ResolvedParams& params);

    /// BFS reorder of build IDs → disk positions (PageShuffle). Pure: does not
    /// mutate nodes_buffer_. Computed once per build, consumed by write_sidecars_.
    struct BfsReorder {
        std::vector<uint32_t> order;  // order[new_pos] = old_id
        std::vector<uint32_t> remap;  // remap[old_id]  = new_pos
    };

    /// Compute the BFS reorder from the build entry points. Pure.
    BfsReorder compute_bfs_reorder_(const ResolvedParams& params) const;

    /// Snap stored FP32 centroids to nearest data vectors (medoids) in
    /// raw_vecs_buffer_ and set them as core_->entry_points_. Called at
    /// flush time when raw_vecs_buffer_ is available. Falls back to stride
    /// sampling if no centroids were stored.
    void snap_entry_points_(const ResolvedParams& params);

    /// Stream all four sidecars (.codes, .graph, .meta, .manifest) to
    /// `index_path` using the precomputed BFS reorder. Does not mutate buffers.
    /// Replaces the former monolithic flush; the .codes write is now
    /// streamed (was a full N×code_size transient allocation — 96GB at 1B).
    void write_sidecars_(const std::string& index_path,
                         const BfsReorder& bfs,
                         const ResolvedParams& params);

    /// Write the .meta sidecar (serialized quantizer + entry points + params).
    /// `entry_points` is already in final disk layout (BFS-remapped by the
    /// build path, verbatim by the post-insert path). Shared by write_sidecars_
    /// and flush.
    void write_meta_file(const ResolvedParams& params,
                         const std::vector<uint32_t>& entry_points,
                         const std::pair<uint64_t, uint64_t>& uuid);

    /// Write the .manifest sidecar (atomic commit point). Shared by
    /// write_sidecars_ and flush.
    void write_manifest_file(const ResolvedParams& params,
                             const std::pair<uint64_t, uint64_t>& uuid);

    /// --- estimate_config helpers (in-RAM mini-build + measurement) ---

    /// Build a mini-index in-RAM on `sample` (sample_n × dim floats) at the
    /// given params. Returns a fresh Engine whose flat buffers + core_ are
    /// populated and entry points computed (ready to measure). Does NOT flush
    /// sidecars or write any files. `params` must have pq_m/pq_bits resolved
    /// (non-zero) — pass1 requires explicit PQ config. num_threads is clamped
    /// to min(params.num_threads, 4) to avoid spawning huge pools for ~20K
    /// vectors. `this` is not modified (static — constructs its own Engine).
    static std::unique_ptr<Engine> build_mini_(const float* sample,
                                                uint64_t sample_n, Dim dim,
                                                const ResolvedParams& params);

    /// Measure graph topology (avg degree, dead-end fraction, clustering
    /// coefficient) from the mini-index's node buffer. LID is NOT computed
    /// here (it comes from truth distances in estimate_config).
    static GraphStats measure_graph_stats_(const Engine& mini);

    /// Run production-style search on the mini-index at L, return mean
    /// recall@k and mean proximity (in-band fraction) against `truth_ids`/
    /// `truth_dists` (per-query top-k true neighbor ids and true L2sq
    /// distances, ascending). Reranks search candidates by true L2sq distance
    /// computed from the FP32 `sample` buffer. `qidx` selects query indices.
    struct SearchQuality { double recall; double proximity; };
    static SearchQuality measure_search_(
        const Engine& mini, const float* sample, uint64_t sample_n, Dim dim,
        const std::vector<std::vector<uint32_t>>& truth_ids,
        const std::vector<std::vector<float>>& truth_dists,
        const std::vector<uint32_t>& qidx,
        uint32_t L, uint32_t k, uint32_t rerank);

    /// Load sidecar files for search.
    void load_sidecars();
};

}  // namespace sextant
