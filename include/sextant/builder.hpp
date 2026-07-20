#pragma once

/// @file builder.hpp
/// Builder — bulk-build + mutate an Index.
///
/// Owns the build pipeline (resolve → reservoir/encode → parallel HDC
/// construct → flush sidecars) and the post-build mutators (insert/flush).
/// Shares the Index with the caller (Searcher reads it; Estimator builds
/// temporary in-RAM Indices via Builder).
///
/// Phase B (this file): methods moved verbatim from Engine. The signatures
/// keep their former shapes so the diff is structural (Engine::foo → Builder::foo)
/// rather than algorithmic. Engine keeps thin delegating wrappers until Phase E.

#include "sextant/config.hpp"
#include "sextant/index.hpp"
#include "sextant/vector_source.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sextant {

class PqQuantizer;
class VamanaCore;

/// Builds and mutates an Index. Owns build-only scratch (entry-point
/// centroids); the heavy state (quantizer, core, buffers, stores) lives on
/// the Index and is mutated in place.
class Builder {
    friend class Engine;  // Phase B: Engine::build_mini_ (estimate path) reaches
                          // into the private build helpers until Phase D moves
                          // the Estimator onto Builder directly.
public:
    /// Build into `index` (which must outlive the Builder). The Index's
    /// metadata (count, dim, path) is populated from `source` / `index_path`.
    explicit Builder(Index& index);

    /// Build an index from a vector source. Auto-resolves params from
    /// `config` via resolve_params() and writes sidecar files.
    BuildResult build(VectorSource& source, const std::string& index_path,
                      const BuildConfig& config);

    /// Build overload taking a fully-resolved params struct directly. Skips
    /// resolve_params — the caller is responsible for producing a complete
    /// ResolvedParams. This is the path used by `sextant autobuild` and by
    /// Estimator's mini-builds.
    BuildResult build(VectorSource& source, const std::string& index_path,
                      const ResolvedParams& params);

    /// Insert a single vector into an already-opened Index (live insert).
    /// Grows the flat buffers by one, runs a mini-construct for the new node,
    /// and marks the Index dirty (flush() will persist).
    void insert(const float* vec, Dim dim, RowId row_id);

    /// Flush any pending state to disk (.meta + .manifest commit; regenerates
    /// .graph/.codes/.ball when flat buffers are dirty).
    void flush();

    // --- PQ selection probe (build-time, no Index mutation) ---
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
        std::vector<ProbedRow> all;  ///< every probed config (for display)
        std::string reason;          ///< selection rationale
    };
    /// Probe PQ (m, bits) selection on a sample. Pure — no Index mutation.
    static PqSelection probe_pq_config(const float* sample, uint64_t n, Dim dim,
                                        const ResolvedParams& params);

private:
    Index& index_;

    // Build-only scratch: FP32 k-means centroids for entry-point selection.
    // Empty outside pass1 → snap_entry_points_ → write_sidecars_.
    std::vector<float> entry_centroids_;

    // --- Build pipeline ---
    void pass1_sample_and_train(VectorSource& source, const ResolvedParams& params);
    void pass2_encode(VectorSource& source, const ResolvedParams& params);
    void parallel_construct(const ResolvedParams& params);

    /// Core construct loop: chunked work-stealing + T5 dynamic L_build +
    /// progress logger. Shared by K==1 (full graph) and K>1 (per-shard).
    void construct_into(VamanaCore& core, uint32_t count,
                        const std::function<RowId(uint32_t)>& row_id_at,
                        uint32_t lut_sz, uint32_t nthreads, const char* label);

    /// Unified build (K==1 fast path + K>1 partitioned): partition →
    /// per-shard build → merge → flush.
    BuildResult build_partitioned(VectorSource& source,
                                   const std::string& index_path,
                                   const ResolvedParams& params);

    /// BFS reorder of build IDs → disk positions (PageShuffle). Pure.
    struct BfsReorder {
        std::vector<uint32_t> order;  // order[new_pos] = old_id
        std::vector<uint32_t> remap;  // remap[old_id]  = new_pos
    };
    BfsReorder compute_bfs_reorder_(const ResolvedParams& params) const;

    /// Snap stored FP32 centroids to nearest data vectors (medoids) in
    /// index_.raw_vecs_buffer and set them as the core's entry points.
    void snap_entry_points_(const ResolvedParams& params);

    // --- Sidecar writers (shared by build + flush) ---
    void write_sidecars_(const std::string& index_path, const BfsReorder& bfs,
                         const ResolvedParams& params);
    void write_meta_file(const ResolvedParams& params,
                         const std::vector<uint32_t>& entry_points,
                         const std::pair<uint64_t, uint64_t>& uuid);
    void write_manifest_file(const ResolvedParams& params,
                             const std::pair<uint64_t, uint64_t>& uuid);
};

}  // namespace sextant
