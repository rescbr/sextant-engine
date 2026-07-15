#pragma once

/// @file engine.hpp
/// Top-level engine facade. Orchestrates build, search, and insert.
///
/// The Engine owns the PqQuantizer, VamanaCore, and storage layer. It exposes
/// a simple API: build from a VectorSource, search with a query vector.

#include <sextant/types.hpp>
#include <sextant/vector_source.hpp>
#include <sextant/config.hpp>
#include <memory>
#include <string>
#include <vector>

namespace sextant {

class PqQuantizer;
class VamanaCore;
class NodeStore;
class FlatNodeStore;
class PagedNodeStore;
class MemGraph;

/// Adaptive parameters resolved from dataset/machine properties (Issue 37).
struct ResolvedParams {
    uint16_t R = 64;
    uint16_t L = 100;
    uint16_t L_build = 100;
    float alpha = 1.2f;
    BuildMode build_mode = BuildMode::SDC;
    uint16_t inline_pq_count = 0;
    uint16_t pq_m = 32;
    uint8_t pq_bits = 8;          ///< 0 = auto (resolved by reservoir probe in pass1)
    float pq_max_distortion = 0.0f;   ///< 0 = default (1.20); max acceptable PQ distortion
    uint32_t max_occlusion = 750;
    MetricKind metric = MetricKind::L2Sq;
    uint64_t build_ram_budget = 0;
    uint32_t num_threads = 0;
    uint32_t K = 1;             ///< Partition count (K>1 → partitioned build)
    float closure_factor = 1.033f;  ///< Shard overlap radius ratio
};

/// Auto-resolve parameters from dataset properties + machine properties.
/// If any field in `overrides` is non-zero/non-default, it takes precedence.
ResolvedParams resolve_params(uint64_t n_vectors, Dim dim,
                               const BuildConfig& overrides);

/// The Sextant engine. Owns the index state.
class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// Build an index from a vector source. Writes sidecar files.
    BuildResult build(VectorSource& source, const std::string& index_path,
                      const BuildConfig& config);

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
    uint64_t count() const { return count_; }
    Dim dim() const { return dim_; }

    /// True when the engine is in SSD-resident (paged) mode — flat RAM buffers
    /// are NOT loaded. Used by tests to verify the low-idle-RAM invariant.
    bool is_paged() const { return paged_store_ != nullptr; }

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
    /// Thread-local L1 cache hit/miss counters.
    uint64_t tl_hits() const;
    uint64_t tl_misses() const;
    /// Number of nodes held in the MemGraph neighborhood cache (0 if none).
    uint32_t memgraph_cached_count() const;
    /// True when flat buffers are resident in RAM (build or post-insert).
    bool has_flat_buffers() const { return nodes_buffer_ != nullptr; }

private:
    bool opened_ = false;
    std::string index_path_;
    uint64_t count_ = 0;
    Dim dim_ = 0;

    std::unique_ptr<PqQuantizer> quantizer_;
    std::unique_ptr<VamanaCore> core_;

    // Flat-in-RAM build buffers (owned by Engine during build).
    uint8_t* codes_buffer_ = nullptr;   // count × code_size
    uint8_t* nodes_buffer_ = nullptr;   // count × node_size
    uint32_t code_size_ = 0;
    uint32_t node_size_ = 0;

    // NodeStore backings. flat_store_ wraps the flat buffers (build + insert).
    // paged_store_ is the SSD-resident search backend. At most one is active
    // on the core at a time. memgraph_ sits atop paged_store_ when paged search
    // is active — it caches the entry-point BFS neighborhood in RAM and
    // delegates cold nodes to paged_store_.
    std::unique_ptr<FlatNodeStore> flat_store_;
    std::unique_ptr<PagedNodeStore> paged_store_;
    std::unique_ptr<MemGraph> memgraph_;
    uint64_t cache_size_override_ = 0;  ///< 0 = auto

    /// Params loaded by open() (used by flush() to persist post-insert state).
    /// Only meaningful when opened_ && params_loaded_.
    ResolvedParams loaded_params_;
    bool params_loaded_ = false;

    /// Build helpers.
    /// pass1 fills the reservoir, resolves pq_bits (if auto, via global probe),
    /// constructs + trains the quantizer. After return, code_size_ is valid.
    void pass1_sample_and_train(VectorSource& source,
                                 const ResolvedParams& params);
    void pass2_encode(VectorSource& source, const ResolvedParams& params);
    void parallel_construct(const ResolvedParams& params);
    void finalize_and_flush(const ResolvedParams& params);

    /// Partitioned build (K>1): partition → per-shard build → merge → flush.
    BuildResult build_partitioned(VectorSource& source,
                                   const std::string& index_path,
                                   const ResolvedParams& params);

    /// Flush sidecar files (Issue 32: ring-buffered).
    void flush_sidecars(const ResolvedParams& params);

    /// Load sidecar files for search.
    void load_sidecars();
};

}  // namespace sextant
