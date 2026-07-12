#pragma once

/// @file vamana_core.hpp
/// Vamana graph algorithm core: BeamSearch, RobustPrune, ConnectAndPrune.
///
/// Evolved from Sextant's exploratory phase:
/// - Storage backend replaced (flat-in-RAM buffers + ShardedLRUCache)
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
};

/// Per-thread-local scratch for Vamana operations (visit marks, prune buffers).
struct VamanaTLS {
    std::vector<uint32_t> visited_flags;
    std::vector<Candidate> prune_buffer;
    std::vector<Candidate> occlusion_set;
    std::mt19937 rng;
    uint32_t visit_token = 0;

    void resize(uint32_t max_nodes);
};

/// The Vamana graph core. Owns the flat-in-RAM node buffer and codes buffer
/// during build, and references ShardedLRUCache during search.
class VamanaCore {
public:
    VamanaCore(VamanaParams params, PqQuantizer& quantizer);
    ~VamanaCore();

    /// Compute the static node size for a given R and inline_pq_count.
    static uint32_t static_node_size(uint16_t R, uint16_t inline_pq_count,
                                     uint8_t code_size);

    /// Prepare for building `count` nodes.
    void prepare_for_build(uint32_t count);

    /// Insert a node during parallel build (from PQ code, SDC mode).
    /// Each thread calls this for disjoint node-ID ranges.
    void insert_build_from_code(uint32_t internal_id, RowId row_id,
                                VamanaTLS& tls);

    /// Insert a node during parallel build (from raw vector, ADC mode).
    void insert_build(uint32_t internal_id, RowId row_id, const float* vec,
                      VamanaTLS& tls);

    /// BeamSearch from entry points. Returns candidates.
    std::vector<Candidate> beam_search(const float* query_lut, uint32_t L,
                                       uint32_t io_limit, VamanaTLS& tls,
                                       const std::vector<uint32_t>* forced_entry_points = nullptr) const;

    /// RobustPrune: select R neighbors from candidates with occlusion.
    std::vector<Candidate> robust_prune(std::vector<Candidate> candidates,
                                        uint16_t R, float alpha,
                                        VamanaTLS& tls,
                                        uint32_t max_occlusion_size) const;

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
    void clear_build_buffers();

    /// Install a NodeStore for read access (search path). When set, beam_search
    /// goes through store_->pin_node/pin_code. During build, Engine installs a
    /// FlatNodeStore over the flat buffers so the same code path is exercised.
    /// `store` is non-owning; the caller must keep it alive.
    void set_store(NodeStore* store) { store_ = store; }
    NodeStore* store() const { return store_; }

    uint32_t size() const { return count_; }
    const std::vector<uint32_t>& entry_points() const { return entry_points_; }

private:
    VamanaParams params_;
    PqQuantizer& quantizer_;

    uint32_t count_ = 0;
    uint32_t node_size_ = 0;
    uint8_t code_size_ = 0;

    // Flat-in-RAM build buffers (Issue 13).
    const uint8_t* build_codes_ = nullptr;  // count × code_size
    uint8_t* build_nodes_ = nullptr;        // count × node_size

    // NodeStore for read access (search + build both go through this). When
    // null, beam_search falls back to the flat buffers directly.
    NodeStore* store_ = nullptr;

    // Sharded lock pool (Issue 11) — protects node neighbor-list mutations.
    std::unique_ptr<Mutex[]> node_locks_;
    uint32_t num_locks_ = 0;

    // Entry points (computed after construct).
    std::vector<uint32_t> entry_points_;

    /// Get a pointer into the flat node buffer.
    uint8_t* node_ptr(uint32_t internal_id);
    const uint8_t* node_ptr(uint32_t internal_id) const;

    /// Acquire the lock for a node (sharded lock pool).
    Mutex* node_lock(uint32_t internal_id);
};

}  // namespace sextant
