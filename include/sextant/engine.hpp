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

/// Adaptive parameters resolved from dataset/machine properties (Issue 37).
struct ResolvedParams {
    uint16_t R = 64;
    uint16_t L = 100;
    uint16_t L_build = 100;
    float alpha = 1.2f;
    uint16_t inline_pq_count = 0;
    uint8_t pq_m = 32;
    uint8_t pq_bits = 8;
    uint32_t max_occlusion = 750;
    MetricKind metric = MetricKind::L2Sq;
    uint64_t build_ram_budget = 0;
    uint32_t num_threads = 0;
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
    uint64_t count() const { return count_; }
    Dim dim() const { return dim_; }

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

    /// Params loaded by open() (used by flush() to persist post-insert state).
    /// Only meaningful when opened_ && params_loaded_.
    ResolvedParams loaded_params_;
    bool params_loaded_ = false;

    /// Build helpers.
    void pass1_sample_and_train(VectorSource& source,
                                 const ResolvedParams& params);
    void pass2_encode(VectorSource& source, const ResolvedParams& params);
    void parallel_construct(const ResolvedParams& params);
    void finalize_and_flush(const ResolvedParams& params);

    /// Flush sidecar files (Issue 32: ring-buffered).
    void flush_sidecars(const ResolvedParams& params);

    /// Load sidecar files for search.
    void load_sidecars();
};

}  // namespace sextant
