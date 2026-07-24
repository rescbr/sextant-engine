#pragma once

/// @file vamana_core.hpp
/// Vamana graph algorithm core: BeamSearch, RobustPrune, ConnectAndPrune.
///
/// Evolved from Sextant's exploratory phase:
/// - Storage backend replaced (flat-in-RAM buffers + BlockCache)
/// - Per-node spinlocks replaced with nsync sharded lock pool
/// - DuckDB dependencies stripped (stdlib + CTPL only)
/// - LabelFilter machinery stripped (Phase 1 is label-less)

#include <sextant/types.hpp>
#include <sextant/sync.hpp>
#include "storage/node_store.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <memory>
#include <random>
#include <thread>

/// Hand-written SIMD distance/dot kernels (simd::l2sq_f16, simd::dot_f16, simd::l2sq_f32,
/// simd::dot_f32, simd::dist_f16, simd::dist_f32, simd::pq_matvec_f32). f32-accumulated — ~2-3× the
/// throughput of NumKong's f64-accumulating `_f32` kernels at our shapes.
/// Included OUTSIDE namespace sextant because simd_kernels.hpp opens its own
/// `namespace sextant { ... }` block.
#include "simd_kernels.hpp"

namespace sextant {

struct PqQuantizer;
struct ResolvedParams;  // defined in engine.hpp; only the factory needs the full type.

/// Parameters for the Vamana graph.
struct VamanaParams {
    Dim dim = 0;
    uint16_t R = 64;              ///< Max graph degree
    uint16_t L = 100;             ///< Search beam width
    uint16_t L_build = 100;       ///< Build beam width
    float alpha = 1.2f;           ///< Prune distance threshold
    uint16_t n_entry_points = 16;
    uint16_t n_search_entry_points = 4;  ///< Multi-start: seed top-M entry points per query
    uint32_t early_exit_patience = 0;    ///< Search early-exit: terminate after N stalled pops post-convergence (0 = disabled)
    uint32_t max_occlusion = 750; ///< RobustPrune occlusion set size

    /// Factory: build VamanaParams from resolved engine params.
    /// `R_override` (0 = use p.R) lets callers pick a per-shard R (e.g. 2R/3).
    /// n_entry_points is set from ResolvedParams.
    static VamanaParams from_resolved(const ResolvedParams& p, Dim dim,
                                       uint16_t R_override = 0);
};

/// Search-internal heap item types. Exposed so per-worker scratch
/// (VamanaTLS) can own the heap buffers — no more thread_local globals.
struct FrontierItem {
    float dist;
    uint32_t internal_id;
};
struct FrontierCmp {
    bool operator()(const FrontierItem& a, const FrontierItem& b) const {
        return a.dist > b.dist;
    }
};
struct WorkingItem {
    float dist;
    uint32_t internal_id;
};
struct WorkingCmp {
    bool operator()(const WorkingItem& a, const WorkingItem& b) const {
        return a.dist < b.dist;
    }
};

/// Per-worker scratch for Vamana operations (visit marks, prune buffers,
/// search heaps). Owned by the caller (Builder pool worker or Searcher
/// pool worker) and passed by reference — NEVER a thread_local global.
struct VamanaTLS {
    std::vector<uint32_t> visited_flags;
    std::vector<Candidate> prune_buffer;    // robust_prune internal sort copy
    std::vector<Candidate> occlusion_set;
    std::vector<Candidate> prune_output;    // robust_prune output (capacity retained)
    std::vector<Candidate> search_result;   // beam_search build-path output (capacity retained)
    std::vector<Candidate> connect_buffer;  // connect_and_prune overflow candidate pool
    std::vector<Candidate> recip_targets;   // snapshot of selected for reciprocal loop
    std::vector<float> lut_buffer;  // Reusable PQ distance LUT (m*K floats).
    /// Per-anchor LUT for PQ-construct build: anchor_lut[s*K+cid] = cross_table[s*K*K + anchor_code[s]*K + cid].
    /// Built once per insert_build_from_code; 8KB (m=32,K=256), L1-resident.
    std::vector<float> anchor_lut;
    std::vector<uint8_t> removed_flags;  // robust_prune removed bitset (bytes, not bools)
    // Search-time heaps (capacity retained across queries). Owned here so
    // beam_search_into is allocation-free on the hot path without relying
    // on thread_local globals.
    std::vector<FrontierItem> frontier_heap;  // min-heap by dist
    std::vector<WorkingItem> working_heap;    // max-heap by dist
    std::mt19937 rng;
    uint32_t visit_token = 0;
    uint32_t search_count_for_resize = 0;  // last count_ search scratch was sized for

    void resize(uint32_t max_nodes);
    /// Size the reusable LUT scratch once per worker (m*K floats from the
    /// quantizer). Called by the Engine/Builder after the quantizer is trained.
    void resize_lut(uint32_t lut_size);
    /// Prepare the search heaps for a beam of width L (capacity retained
    /// across calls; clears contents). Equivalent to the former
    /// SearchScratch::prepare.
    void prepare_search(uint32_t L) {
        if (frontier_heap.capacity() < L + 1) frontier_heap.reserve(L + 1);
        if (working_heap.capacity() < L + 1) working_heap.reserve(L + 1);
        frontier_heap.clear();
        working_heap.clear();
    }
};

/// Per-build scratch and mutable state. Owned by VamanaCore when building;
/// null when the core is configured for search-only use (Layer 4 §2.3).
///
/// Bundles the flat-in-RAM build buffers (codes, nodes, raw FP16 vectors),
/// the sharded node-lock pool that protects neighbor-list mutations during
/// parallel construction, and the optional atomic progress signal used by
/// the T5 dynamic-L_build ramp.
struct BuildContext {
    const uint8_t* codes = nullptr;            // count × code_size
    uint8_t* nodes = nullptr;                  // count × node_size
    const float16_t* vecs = nullptr;           // count × dim raw FP16 vectors
    const float* fp32_vecs = nullptr;          // count × dim raw FP32 vectors (FP32 build mode)
    std::unique_ptr<Mutex[]> node_locks;       // sharded lock pool
    uint32_t num_locks = 0;
    std::atomic<uint32_t>* progress = nullptr;  // dynamic L_build signal

    static constexpr uint32_t kDefaultLocksFactor = 4;

    /// Allocate the sharded lock pool sized to the build parallelism.
    void init_locks(uint32_t factor = kDefaultLocksFactor) {
        num_locks = std::max(256u, std::thread::hardware_concurrency() * factor);
        node_locks = std::unique_ptr<Mutex[]>(new Mutex[num_locks]);
    }

    /// Acquire the lock for a node (sharded lock pool).
    Mutex* lock_for(uint32_t internal_id) {
        return &node_locks[internal_id % num_locks];
    }

    /// Pointer into the flat node buffer.
    uint8_t* node_ptr(uint32_t internal_id, uint32_t node_size) {
        return nodes + static_cast<size_t>(internal_id) * node_size;
    }
    const uint8_t* node_ptr(uint32_t internal_id, uint32_t node_size) const {
        return nodes + static_cast<size_t>(internal_id) * node_size;
    }

    /// Pointer into the raw FP16 vectors (FP16 prune hybrid).
    const float16_t* vec_ptr(uint32_t internal_id, Dim dim) const {
        return vecs + static_cast<size_t>(internal_id) * dim;
    }
};

/// Per-query parameters for beam_search / beam_search_into (Layer 4 §8.1).
///
/// Bundles the four optional pointers that previously made beam_search_into
/// an 8-param signature. Each field is independently optional (null) and
/// gates a different distance-computation mode:
///
///   **Search path** (called by search()):
///   - `query_lut`     — PQ lookup table (m*K floats). Always set.
///   - `query_fp16`    — when non-null AND the store exposes FP16 vectors
///                       for the candidate node (MemGraph ball nodes),
///                       distances use simd::l2sq_f16 instead of PQ lut_distance
///                       (hybrid FP16+PQ precision).
///
///   **Build path** (called by insert_build_core):
///   - `query_lut`     — generic PQ LUT (default build mode).
///   - `anchor_code`    — when non-null, distances are computed via direct
///                       code-to-code lookup (code_distance) instead of the
///                       materialized LUT. Skips the 32KB LUT gather.
///   - `anchor_lut`    — when non-null, takes priority over both query_lut
///                       and anchor_code: distances are gathered from this
///                       8KB per-anchor LUT (L1-resident) via lut_distance.
///
///   **Common**:
///   - `forced_entry_points` — when non-null, overrides the core's own
///                             entry_points_ (used by partitioned builds
///                             and tests).
struct BeamQuery {
    const float* query_lut = nullptr;
    const float16_t* query_fp16 = nullptr;
    const float* query_fp32 = nullptr;       // FP32 build mode: direct FP32 distance
    const uint8_t* anchor_code = nullptr;
    const float* anchor_lut = nullptr;
    const std::vector<uint32_t>* forced_entry_points = nullptr;
};

/// The Vamana graph core. Owns the flat-in-RAM node buffer and codes buffer
/// during build, and references BlockCache during search.
class VamanaCore {
public:
    VamanaCore(VamanaParams params, PqQuantizer& quantizer);
    ~VamanaCore();

    /// Compute the static node size for a given R.
    static uint32_t static_node_size(uint16_t R, uint32_t code_size);

    /// Prepare for building `count` nodes.
    void prepare_for_build(uint32_t count);

    /// Insert a node during parallel build (PQ-construct mode: LUT from PQ code).
    /// Each thread calls this for disjoint node-ID ranges.
    void insert_build_from_code(uint32_t internal_id, RowId row_id,
                                VamanaTLS& tls);

    /// Shared build-insert core: beam_search → robust_prune → connect_and_prune.
    /// The LUT is always built via build_code_lut from the node's own PQ code.
    /// The prune query vec is build_vec_ptr(internal_id) when build_vecs_ is set
    /// (FP16 prune hybrid); otherwise nullptr (PQ fallback).
    void insert_build_core(uint32_t internal_id, RowId row_id, VamanaTLS& tls);

    /// BeamSearch from entry points. Returns candidates.
    ///
    /// `q` bundles the query inputs (LUT + optional FP16/PQ-construct anchor/forced
    /// entry points). See BeamQuery for the distance-mode taxonomy.
    /// `early_exit_patience`: search-time early-exit patience (0 = disabled,
    /// UINT32_MAX = defer to params_.early_exit_patience). Build callers MUST
    /// pass 0 (build never early-exits — see beam_search_into docs).
    std::vector<Candidate> beam_search(const BeamQuery& q, uint32_t L,
                                        uint32_t io_limit, VamanaTLS& tls,
                                        uint32_t early_exit_patience = kDeferToParams) const;

    /// BeamSearch writing into `out` (cleared; capacity retained). Build path
    /// uses this to avoid per-insert heap allocation.
    ///
    /// `early_exit_patience`: search-time early-exit patience. The build path
    /// (insert_build_core) passes kNeverExit (UINT32_MAX) — build-time
    /// beam_search determines edge quality and must NEVER be truncated, since
    /// a shallower candidate pool permanently degrades the graph. Search-time
    /// callers pass the SearchConfig's patience (0 = disabled). The default
    /// kDeferToParams resolves to params_.early_exit_patience for back-compat.
    static constexpr uint32_t kDeferToParams = 0xFFFFFFFFu;  // resolve to params_
    static constexpr uint32_t kNeverExit = 0xFFFFFFFEu;      // build path: no early-exit
    void beam_search_into(std::vector<Candidate>& out, const BeamQuery& q,
                          uint32_t L, uint32_t io_limit,
                          VamanaTLS& tls,
                          uint32_t early_exit_patience = kDeferToParams) const;

    /// RobustPrune: select R neighbors from candidates with occlusion.
    /// Writes the kept candidates into `out` (cleared; capacity retained).
    /// `candidates` is sorted in place (assumed already sorted ascending when
    /// `presorted=true` — skips the redundant std::sort when callers pass
    /// beam_search output, which is already ascending).
    ///
    /// When `query_vec` is non-null AND build_vecs_ is set, the occlusion check
    /// uses FP16 L2-squared distances (computed directly from raw FP16 vectors)
    /// instead of PQ code_distance. This closes the recall gap caused by PQ
    /// distortion (~3.6%) flipping occlusion decisions. Both the query→candidate
    /// distance (buf[].dist) and the candidate→candidate distance (d(p,pp)) are
    /// recomputed as FP16 L2sq so both sides of the occlusion rule use the same
    /// metric. When `query_vec` is null, falls back to the existing PQ path
    /// (tests / untrained quantizers).
    void robust_prune_into(const std::vector<Candidate>& candidates,
                           std::vector<Candidate>& out, uint16_t R, float alpha,
                           VamanaTLS& tls, uint32_t max_occlusion_size,
                           bool presorted = false,
                           const float16_t* query_vec = nullptr) const;

    /// Connect new node to selected neighbors and prune reciprocal edges.
    void connect_and_prune(uint32_t new_internal_id,
                           const std::vector<Candidate>& selected,
                           VamanaTLS& tls);

    /// Compute entry points via k-means on PQ codes.
    void compute_entry_points();

    /// Search: top-k candidates. The caller supplies the per-worker `tls`
    /// (owned by the pool worker, NOT a thread_local global — eliminates
    /// cross-instance aliasing and lets a single Searcher fan queries out
    /// across multiple VamanaTLS instances).
    /// Search: top-k candidates. The caller supplies the per-worker `tls`
    /// (owned by the pool worker, NOT a thread_local global — eliminates
    /// cross-instance aliasing and lets a single Searcher fan queries out
    /// across multiple VamanaTLS instances). `q` bundles the LUT + optional
    /// FP16 query (see BeamQuery).
    std::vector<Candidate> search(const BeamQuery& q, uint32_t k,
                                   uint32_t L_search, uint32_t io_limit,
                                   VamanaTLS& tls,
                                   uint32_t early_exit_patience = kDeferToParams) const;

    // --- Node accessors (flat buffer layout) ---
    static RowId get_row_id(const uint8_t* node);
    static void set_row_id(uint8_t* node, RowId val);
    static uint32_t get_internal_id(const uint8_t* node);
    static void set_internal_id(uint8_t* node, uint32_t val);
    static uint16_t get_neighbor_count(const uint8_t* node);
    static void set_neighbor_count(uint8_t* node, uint16_t val);
    static uint32_t get_neighbor(const uint8_t* node, uint32_t i);
    static void set_neighbor(uint8_t* node, uint32_t i, uint32_t val);

    // --- Setters for build buffers (no-op when no BuildContext is installed) ---

    /// Allocate the BuildContext (holds flat build buffers + node locks).
    /// Required before any insert_build_*/connect_and_prune call. Idempotent
    /// — calling again reallocates the lock pool (existing buffers must be
    /// re-installed via set_build_codes/nodes/vecs).
    void init_build_context() {
        if (!build_ctx_) {
            build_ctx_ = std::make_unique<BuildContext>();
        }
        build_ctx_->init_locks();
    }

    /// True when this core has a BuildContext installed (build mode). False
    /// for search-only cores.
    bool has_build_context() const { return static_cast<bool>(build_ctx_); }

    void set_build_codes(const uint8_t* codes, uint32_t /*count*/) {
        if (!build_ctx_) init_build_context();
        build_ctx_->codes = codes;
    }

    void set_build_nodes(uint8_t* nodes) {
        if (!build_ctx_) init_build_context();
        build_ctx_->nodes = nodes;
    }

    void set_build_vecs(const float16_t* vecs) {
        if (!build_ctx_) init_build_context();
        build_ctx_->vecs = vecs;
    }
    void set_build_fp32_vecs(const float* vecs) {
        if (!build_ctx_) init_build_context();
        build_ctx_->fp32_vecs = vecs;
    }

    void clear_build_buffers() {
        if (build_ctx_) {
            build_ctx_->codes = nullptr;
            build_ctx_->nodes = nullptr;
            build_ctx_->vecs = nullptr;
        }
    }

    /// RAII scope guard: clears the build-vec pointer on destruction so the
    /// core doesn't hold a dangling pointer after the borrower's buffer goes
    /// away. Use around the lifetime of a raw_vecs_buffer handed to set_build_vecs:
    ///
    ///     {
    ///         core.set_build_vecs(buf);
    ///         VamanaCore::BuildVecLoan loan(*core);
    ///         parallel_construct(...);   // may throw — loan still clears
    ///     }                              // buf pointer cleared here
    ///
    /// This is a clarity/maintenance guard only: the current call sites are
    /// already correct by construction (parallel_construct completes before
    /// the owner's buffer is freed), but the loan documents the invariant in
    /// the type system and closes the window if future code throws between
    /// set_build_vecs and the manual set_build_vecs(nullptr).
    class BuildVecLoan {
    public:
        explicit BuildVecLoan(VamanaCore& core) : core_(core) {}
        ~BuildVecLoan() { core_.set_build_vecs(nullptr); }
        BuildVecLoan(const BuildVecLoan&) = delete;
        BuildVecLoan& operator=(const BuildVecLoan&) = delete;
    private:
        VamanaCore& core_;
    };

    /// Pointer to the raw FP16 vector for `internal_id` (FP16 prune).
    /// Only valid when set_build_vecs() has been called with a count×dim buffer.
    const float16_t* build_vec_ptr(uint32_t internal_id) const {
        return build_ctx_->vec_ptr(internal_id, params_.dim);
    }

    /// Install a NodeStore for read access (search path). When set, beam_search
    /// goes through store_->pin_node/pin_code. During build, Engine installs a
    /// FlatNodeStore over the flat buffers so the same code path is exercised.
    /// `store` is non-owning; the caller must keep it alive.
    void set_store(NodeStore* store) { store_ = store; }
    NodeStore* store() const { return store_; }

    uint32_t size() const { return count_; }
    const std::vector<uint32_t>& entry_points() const { return entry_points_; }
    void set_entry_points(std::vector<uint32_t> eps) { entry_points_ = std::move(eps); }

    /// Configure this core for search-only use (Layer 3: per-worker cores).
    /// Sets the node count + entry points that search() needs. The store must
    /// be installed separately via set_store(). Quantizer + params are set at
    /// construction. No build buffers are populated (search reads through the
    /// store, not the flat buffers).
    void set_search_state(uint32_t count, std::vector<uint32_t> entry_points) {
        count_ = count;
        entry_points_ = std::move(entry_points);
    }

    /// T5: install a progress signal for dynamic L_build. The pointer must
    /// outlive the build; it is read with a relaxed load inside the insert
    /// path. Pass nullptr (the default) to use a fixed L_build.
    void set_build_progress(std::atomic<uint32_t>* progress) {
        if (build_ctx_) build_ctx_->progress = progress;
    }

private:
    VamanaParams params_;
    PqQuantizer& quantizer_;

    uint32_t count_ = 0;
    uint32_t node_size_ = 0;
    uint32_t code_size_ = 0;

    // NodeStore for read access (search + build both go through this). When
    // null, beam_search falls back to the flat buffers directly (build path).
    NodeStore* store_ = nullptr;

    // Build-only state (Layer 4 §2.3). Null when the core is configured for
    // search-only use; installed via init_build_context() before any build
    // method is called.
    std::unique_ptr<BuildContext> build_ctx_;

    // Entry points (computed after construct).
    std::vector<uint32_t> entry_points_;
};

}  // namespace sextant
