#pragma once

/// @file builder.hpp
/// Builder — bulk-build + mutate an Index.
///
/// Owns the build pipeline (resolve → reservoir/encode → parallel PQ-construct
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
class FlatNodeStore;

/// Builds and mutates an Index. Owns build-only scratch (entry-point
/// centroids); the heavy state (quantizer, core, buffers, stores) lives on
/// the Index and is mutated in place.
class Builder {
    friend class Engine;     // Phase B bridge: Engine keeps thin delegators
                             // until Phase E deletes it.
    friend class Estimator;  // Phase D: Estimator reaches into the private
                             // build pipeline for in-RAM mini-builds without
                             // the sidecar flush step.
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
    /// .graph/.codes when flat buffers are dirty).
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

    /// Prepare codes only: run pass1 (reservoir + PQ train) and pass2
    /// (encode), leaving codes_buffer + raw_vecs_buffer populated on the
    /// Index without running the construct or flush pipeline. Used by
    /// build_partitioned and build_ivf to share the global train+encode
    /// step across all shards.
    void prepare_codes(VectorSource& source, const ResolvedParams& params);

    /// IVF-probe build: train the quantizer ONCE globally, encode all N
    /// vectors, partition into K shards, and flush each shard as a complete
    /// production Index under `<index_path>.shards/shard_NNNN/`. Also writes
    /// `centroids.bin` (K × dim × float16_t, decoded from the partition's PQ
    /// centroid codes) and a line-oriented `manifest` (ready/K/dim/
    /// n_probe_default/closure_factor). Each shard is openable unchanged via
    /// `Index::read(shard_prefix)`; the whole IVF index via
    /// `IVFIndex::read(<index_path>.shards)`.
    ///
    /// The quantizer is shared across all shards (never retrained per shard).
    /// Each shard is built by a fresh Builder + Index pair at R_shard = 2R/3.
    /// `n_probe_default` (0 → max(1, K/4)) is recorded in the manifest.
    BuildResult build_ivf(VectorSource& source, const std::string& index_path,
                          const ResolvedParams& params,
                          uint32_t n_probe_default = 0);

    /// BuildConfig overload: resolves params first (picks up the IVF K
    /// heuristic when --partition-count is not explicit), then forwards to
    /// the ResolvedParams overload. Mirrors Builder::build's two-overload
    /// pattern.
    BuildResult build_ivf(VectorSource& source, const std::string& index_path,
                          const BuildConfig& config,
                          uint32_t n_probe_default = 0) {
        index_.count = source.count();
        index_.dim = source.dim();
        return build_ivf(source, index_path,
                         resolve_params(index_.count, index_.dim, config),
                         n_probe_default);
    }

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

    /// Build one IVF shard and flush it as a complete production Index to
    /// `shard_prefix.*` (standard sidecars: .graph/.codes/.meta/
    /// .manifest). The shard Index is fully populated by the caller:
    ///   - a CLONE of the global quantizer (so write_meta_file can serialize
    ///     it independently, and the reopened shard reconstructs its own
    ///     VamanaCore at R_shard);
    ///   - materialized shard-local codes/nodes/vecs buffers (local
    ///     0..shard_n-1 ordering; each node's row_id = its global ID);
    ///   - `count`/`dim`/`code_size`/`node_size` set (node_size = R_shard);
    ///   - a fresh VamanaCore at R_shard wired to those buffers.
    ///
    /// `shard_params` carries R = R_shard (2R/3) so write_sidecars_ emits the
    /// final layout at R_shard and Index::read reopens the shard with the
    /// right node_size. This helper runs construct_into (on a fresh Builder
    /// bound to `shard_index`) → snap_entry_points_ → compute_bfs_reorder_ →
    /// write_sidecars_, reusing the entire flush pipeline unchanged. The
    /// quantizer is NEVER retrained here — `members` is only used to remap
    /// shard-local IDs → global row IDs during construct.
    void build_shard_into_(Index& shard_index,
                           const ResolvedParams& shard_params,
                           const std::vector<uint32_t>& members,
                           uint32_t shard_k, uint32_t K,
                           const std::string& shard_prefix);

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

    /// IVF Workstream A1/A3: sub-clustered entry points for a shard.
    /// Runs a small k-means (k' = n_sub) on the shard's PQ codes, snaps each
    /// sub-centroid to its top-`medoids_per_cluster` members (nearest by FP16
    /// L2sq), and sets the k'×M deduped medoid LOCAL IDs as the core's entry
    /// points. Also populates index_.sub_centroids (k' × dim FP16) and
    /// index_.sub_centroid_medoids (k' × M LOCAL IDs) so write_sidecars_ can
    /// emit the `.epc` sidecar for A2/A3. Falls back to snap_entry_points_
    /// (stride sampling) when the shard is too small. Returns k_sub used (0
    /// on fallback).
    uint32_t compute_sub_cluster_entry_points_(const ResolvedParams& params,
                                                uint32_t n_sub,
                                                uint32_t medoids_per_cluster);

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
