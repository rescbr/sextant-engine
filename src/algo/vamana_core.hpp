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

/// Parameters for the Vamana graph.
struct VamanaParams {
    Dim dim = 0;
    uint16_t R = 64;              ///< Max graph degree
    uint16_t L = 100;             ///< Search beam width
    uint16_t L_build = 100;       ///< Build beam width
    float alpha = 1.2f;           ///< Prune distance threshold
    uint16_t inline_pq_count = 0; ///< Neighbor PQ codes inlined per node
    uint16_t n_entry_points = 16;
    uint32_t max_occlusion = 750; ///< RobustPrune occlusion set size

    /// Factory: build VamanaParams from resolved engine params.
    /// `R_override` (0 = use p.R) lets callers pick a per-shard R (e.g. 2R/3).
    /// `inline_pq` is 0 for build cores (flat layout) and the resolved
    /// inline_pq_count for search cores. n_entry_points is fixed at 16.
    static VamanaParams from_resolved(const ResolvedParams& p, Dim dim,
                                       uint16_t R_override = 0,
                                       uint16_t inline_pq = 0);
};

/// Per-thread-local scratch for Vamana operations (visit marks, prune buffers).
struct VamanaTLS {
    std::vector<uint32_t> visited_flags;
    std::vector<Candidate> prune_buffer;    // robust_prune internal sort copy
    std::vector<Candidate> occlusion_set;
    std::vector<Candidate> prune_output;    // robust_prune output (capacity retained)
    std::vector<Candidate> search_result;   // beam_search build-path output (capacity retained)
    std::vector<Candidate> connect_buffer;  // connect_and_prune overflow candidate pool
    std::vector<Candidate> recip_targets;   // snapshot of selected for reciprocal loop
    std::vector<float> lut_buffer;  // Reusable PQ distance LUT (m*K floats).
    /// Per-anchor LUT for SDC build: anchor_lut[s*K+cid] = cross_table[s*K*K + anchor_code[s]*K + cid].
    /// Built once per insert_build_from_code; 8KB (m=32,K=256), L1-resident.
    std::vector<float> anchor_lut;
    std::vector<uint8_t> removed_flags;  // robust_prune removed bitset (bytes, not bools)
    std::mt19937 rng;
    uint32_t visit_token = 0;

    void resize(uint32_t max_nodes);
    /// Size the reusable LUT scratch once per thread (m*K floats from the
    /// quantizer). Called by the Engine after the quantizer is trained.
    void resize_lut(uint32_t lut_size);
};

/// The Vamana graph core. Owns the flat-in-RAM node buffer and codes buffer
/// during build, and references BlockCache during search.
class VamanaCore {
public:
    VamanaCore(VamanaParams params, PqQuantizer& quantizer);
    ~VamanaCore();

    /// Compute the static node size for a given R and inline_pq_count.
    static uint32_t static_node_size(uint16_t R, uint16_t inline_pq_count,
                                     uint32_t code_size);

    /// Prepare for building `count` nodes.
    void prepare_for_build(uint32_t count);

    /// Insert a node during parallel build (from PQ code, SDC mode).
    /// Each thread calls this for disjoint node-ID ranges.
    void insert_build_from_code(uint32_t internal_id, RowId row_id,
                                VamanaTLS& tls);

    /// Insert a node during parallel build (from raw vector, ADC mode).
    void insert_build(uint32_t internal_id, RowId row_id, const float* vec,
                      VamanaTLS& tls);

    /// Shared build-insert core. When `adc_vec` is null the node is inserted
    /// in SDC mode (LUT built from its own PQ code); otherwise ADC mode
    /// (LUT built from the raw vector via preprocess_query).
    void insert_build_core(uint32_t internal_id, RowId row_id, VamanaTLS& tls,
                           const float* adc_vec);

    /// BeamSearch from entry points. Returns candidates.
    /// When `sdc_anchor` is non-null, distances are computed via direct
    /// code-to-code lookup (code_distance) instead of the materialized LUT.
    /// This skips the 32KB LUT gather, trading scattered table reads for
    /// zero memcpy. Only valid in SDC build mode (anchor is a PQ code).
    ///
    /// When `anchor_lut` is non-null, it takes priority over both `query_lut`
    /// and `sdc_anchor`: distances are gathered from the 8KB per-anchor LUT
    /// (L1-resident) via lut_distance, which is contiguous and cache-friendly.
    std::vector<Candidate> beam_search(const float* query_lut, uint32_t L,
                                       uint32_t io_limit, VamanaTLS& tls,
                                       const std::vector<uint32_t>* forced_entry_points = nullptr,
                                       const uint8_t* sdc_anchor = nullptr,
                                       const float* anchor_lut = nullptr) const;

    /// BeamSearch writing into `out` (cleared; capacity retained). Build path
    /// uses this to avoid per-insert heap allocation.
    void beam_search_into(std::vector<Candidate>& out,
                          const float* query_lut, uint32_t L,
                          uint32_t io_limit, VamanaTLS& tls,
                          const std::vector<uint32_t>* forced_entry_points = nullptr,
                          const uint8_t* sdc_anchor = nullptr,
                          const float* anchor_lut = nullptr) const;

    /// RobustPrune: select R neighbors from candidates with occlusion.
    /// Writes the kept candidates into `out` (cleared; capacity retained).
    /// `candidates` is sorted in place (assumed already sorted ascending when
    /// `presorted=true` — skips the redundant std::sort when callers pass
    /// beam_search output, which is already ascending).
    ///
    /// When `query_vec` is non-null AND build_vecs_ is set, the occlusion check
    /// uses FP16 L2-squared distances (computed from raw float vectors) instead
    /// of PQ code_distance. This closes the recall gap caused by PQ distortion
    /// (~3.6%) flipping occlusion decisions. Both the query→candidate distance
    /// (buf[].dist) and the candidate→candidate distance (d(p,pp)) are
    /// recomputed as FP16 L2sq so both sides of the occlusion rule use the same
    /// metric. When `query_vec` is null, falls back to the existing PQ path.
    void robust_prune_into(const std::vector<Candidate>& candidates,
                           std::vector<Candidate>& out, uint16_t R, float alpha,
                           VamanaTLS& tls, uint32_t max_occlusion_size,
                           bool presorted = false,
                           const float* query_vec = nullptr) const;

    /// Connect new node to selected neighbors and prune reciprocal edges.
    void connect_and_prune(uint32_t new_internal_id,
                           const std::vector<Candidate>& selected,
                           VamanaTLS& tls);

    /// Finalize inline PQ codes (reformat from build layout).
    void finalize_inline_codes();

    /// Compute entry points via k-means on PQ codes.
    void compute_entry_points();

    /// Search: top-k candidates.
    std::vector<Candidate> search(const float* query_lut, uint32_t k,
                                   uint32_t L_search, uint32_t io_limit) const;

    // --- Node accessors (flat buffer layout) ---
    static RowId get_row_id(const uint8_t* node);
    static void set_row_id(uint8_t* node, RowId val);
    static uint32_t get_internal_id(const uint8_t* node);
    static void set_internal_id(uint8_t* node, uint32_t val);
    static uint16_t get_neighbor_count(const uint8_t* node);
    static void set_neighbor_count(uint8_t* node, uint16_t val);
    static uint16_t get_inline_pq_count(const uint8_t* node);
    static void set_inline_pq_count(uint8_t* node, uint16_t val);
    static uint32_t get_neighbor(const uint8_t* node, uint32_t i);
    static void set_neighbor(uint8_t* node, uint32_t i, uint32_t val);

    // --- Setters for build buffers ---
    void set_build_codes(const uint8_t* codes, uint32_t count);
    void set_build_nodes(uint8_t* nodes);
    void set_build_vecs(const float* vecs) { build_vecs_ = vecs; }
    void clear_build_buffers();

    /// Pointer to the raw float vector for `internal_id` (ADC build mode).
    /// Only valid when set_build_vecs() has been called with a count×dim buffer.
    const float* build_vec_ptr(uint32_t internal_id) const {
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
    const float* build_vecs_ = nullptr;     // count × dim raw vectors (ADC mode)

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
