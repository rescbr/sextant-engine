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

namespace sextant {

struct PqQuantizer;
struct ResolvedParams;  // defined in engine.hpp; only the factory needs the full type.

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SEXTANT_HAS_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define SEXTANT_HAS_AVX2 1
#endif

/// L2-squared distance between two FP16 vectors. FP32 accumulation (no overflow),
/// FP16 per-element precision. Used by the FP16 prune occlusion check and the
/// partitioned-build merge truncation.
inline float l2sq_f16(const float16_t* a, const float16_t* b, uint32_t dim) {
#if defined(SEXTANT_HAS_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vcvt_f32_f16(vld1_f16(a + i));
        float32x4_t vb = vcvt_f32_f16(vld1_f16(b + i));
        float32x4_t diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }
    float r = vaddvq_f32(acc);
    for (; i < dim; i++) { float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]); r += diff * diff; }
    return r;
#elif defined(SEXTANT_HAS_AVX2) && defined(__F16C__)
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
        __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float r = _mm_cvtss_f32(sum);
    for (; i < dim; i++) { float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]); r += diff * diff; }
    return r;
#else
    float r = 0.0f;
    for (uint32_t i = 0; i < dim; i++) { float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]); r += diff * diff; }
    return r;
#endif
}

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
    /// Per-anchor LUT for HDC build: anchor_lut[s*K+cid] = cross_table[s*K*K + anchor_code[s]*K + cid].
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

    /// Insert a node during parallel build (HDC mode: LUT from PQ code).
    /// Each thread calls this for disjoint node-ID ranges.
    void insert_build_from_code(uint32_t internal_id, RowId row_id,
                                VamanaTLS& tls);

    /// Shared build-insert core: beam_search → robust_prune → connect_and_prune.
    /// The LUT is always built via build_code_lut from the node's own PQ code.
    /// The prune query vec is build_vec_ptr(internal_id) when build_vecs_ is set
    /// (FP16 prune hybrid); otherwise nullptr (PQ fallback).
    void insert_build_core(uint32_t internal_id, RowId row_id, VamanaTLS& tls);

    /// BeamSearch from entry points. Returns candidates.
    /// When `hdc_anchor` is non-null, distances are computed via direct
    /// code-to-code lookup (code_distance) instead of the materialized LUT.
    /// This skips the 32KB LUT gather, trading scattered table reads for
    /// zero memcpy. Only valid in HDC build mode (anchor is a PQ code).
    ///
    /// When `anchor_lut` is non-null, it takes priority over both `query_lut`
    /// and `hdc_anchor`: distances are gathered from the 8KB per-anchor LUT
    /// (L1-resident) via lut_distance, which is contiguous and cache-friendly.
    std::vector<Candidate> beam_search(const float* query_lut, uint32_t L,
                                       uint32_t io_limit, VamanaTLS& tls,
                                       const std::vector<uint32_t>* forced_entry_points = nullptr,
                                       const uint8_t* hdc_anchor = nullptr,
                                       const float* anchor_lut = nullptr,
                                       const float16_t* query_fp16 = nullptr) const;

    /// BeamSearch writing into `out` (cleared; capacity retained). Build path
    /// uses this to avoid per-insert heap allocation.
    ///
    /// When `query_fp16` is non-null AND the store exposes FP16 vectors for a
    /// node (NodeStore::fp16_ptr), distances use l2sq_f16 instead of PQ
    /// lut_distance for that node (hybrid FP16+PQ precision).
    void beam_search_into(std::vector<Candidate>& out,
                          const float* query_lut, uint32_t L,
                          uint32_t io_limit, VamanaTLS& tls,
                          const std::vector<uint32_t>* forced_entry_points = nullptr,
                           const uint8_t* hdc_anchor = nullptr,
                          const float* anchor_lut = nullptr,
                          const float16_t* query_fp16 = nullptr) const;

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
    std::vector<Candidate> search(const float* query_lut, uint32_t k,
                                   uint32_t L_search, uint32_t io_limit,
                                   VamanaTLS& tls,
                                   const float16_t* query_fp16 = nullptr) const;

    // --- Node accessors (flat buffer layout) ---
    static RowId get_row_id(const uint8_t* node);
    static void set_row_id(uint8_t* node, RowId val);
    static uint32_t get_internal_id(const uint8_t* node);
    static void set_internal_id(uint8_t* node, uint32_t val);
    static uint16_t get_neighbor_count(const uint8_t* node);
    static void set_neighbor_count(uint8_t* node, uint16_t val);
    static uint32_t get_neighbor(const uint8_t* node, uint32_t i);
    static void set_neighbor(uint8_t* node, uint32_t i, uint32_t val);

    // --- Setters for build buffers ---
    void set_build_codes(const uint8_t* codes, uint32_t count);
    void set_build_nodes(uint8_t* nodes);
    void set_build_vecs(const float16_t* vecs) { build_vecs_ = vecs; }
    void clear_build_buffers();

    /// Pointer to the raw FP16 vector for `internal_id` (FP16 prune).
    /// Only valid when set_build_vecs() has been called with a count×dim buffer.
    const float16_t* build_vec_ptr(uint32_t internal_id) const {
        return build_vecs_ + static_cast<size_t>(internal_id) * params_.dim;
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
        build_progress_ = progress;
    }

private:
    VamanaParams params_;
    PqQuantizer& quantizer_;

    uint32_t count_ = 0;
    uint32_t node_size_ = 0;
    uint32_t code_size_ = 0;

    // Flat-in-RAM build buffers (Issue 13).
    const uint8_t* build_codes_ = nullptr;  // count × code_size
    uint8_t* build_nodes_ = nullptr;        // count × node_size
    const float16_t* build_vecs_ = nullptr;     // count × dim raw vectors (FP16)

    // NodeStore for read access (search + build both go through this). When
    // null, beam_search falls back to the flat buffers directly.
    NodeStore* store_ = nullptr;

    // Sharded lock pool (Issue 11) — protects node neighbor-list mutations.
    std::unique_ptr<Mutex[]> node_locks_;
    uint32_t num_locks_ = 0;

    // Entry points (computed after construct).
    std::vector<uint32_t> entry_points_;

    // T5: progress signal for dynamic L_build (optional). Set by
    // parallel_construct to &next_id; read with a relaxed load. Null in
    // paths without a global progress counter (partitioned/online/tests).
    std::atomic<uint32_t>* build_progress_ = nullptr;

    /// Get a pointer into the flat node buffer.
    uint8_t* node_ptr(uint32_t internal_id);
    const uint8_t* node_ptr(uint32_t internal_id) const;

    /// Acquire the lock for a node (sharded lock pool).
    Mutex* node_lock(uint32_t internal_id);
};

}  // namespace sextant
