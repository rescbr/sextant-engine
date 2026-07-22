#pragma once

/// @file ivf_index.hpp
/// IVFIndex — K independent shard Indexes + FP16 routing centroids.
///
/// Replaces the merged-graph search path with K independent shard graphs.
/// Queries route to `n_probe` nearest shards by centroid distance; each
/// probed shard is a complete production Index opened via `Index::read`.
/// See `docs/ivf_probe_design.md`.
///
/// On-disk layout under `<prefix>.shards/`:
///   centroids.bin   — K × dim × float16_t (the routing centroids)
///   manifest.json   — { K, n_probe_default, closure_factor, dim }
///   shard_0001/     — standard Sextant index (.graph/.codes/.meta/.ball/.manifest)
///   ...
///   shard_K/
///
/// Each shard's Index is opened unchanged via `Index::read`; IVFIndex just
/// composes them and owns the centroids used for routing.

#include "sextant/config.hpp"
#include "sextant/index.hpp"
#include "sextant/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sextant {

/// Owning container for an IVF-probe index: K shard Indexes + FP16 centroids.
///
/// Lifetime: the shard Indexes and the centroids buffer are owned by this
/// object. The IVFSearcher holds a reference to it (mirroring the
/// Index/Searcher ownership pattern).
struct IVFIndex {
    /// K × dim FP16 routing centroids, row-major. K = shards.size().
    std::vector<float16_t> centroids;

    /// The shard dimensions (every shard has the same dim — recorded once).
    Dim dim = 0;

    /// K (partition count). Equal to shards.size(). Redundant but convenient.
    uint32_t K = 0;

    /// Default n_probe (max(1, K/4) at build time). The Searcher may override.
    uint32_t n_probe_default = 1;

    /// closure_factor the index was built with (for diagnostics).
    float closure_factor = 1.0f;

    /// Shard indexes. Each is a complete production Index opened via
    /// Index::read; null entries correspond to empty shards (skipped at
    /// build time — rare, but possible if k-means produces an empty cluster
    /// that wasn't reseeded).
    std::vector<std::unique_ptr<Index>> shards;

    /// IVF Workstream A1/A2: per-shard sub-cluster entry-point data.
    /// `shard_sub_centroids[k]` = k' × dim FP16 sub-centroids for shard k
    /// (loaded from the shard's `.epc` sidecar). `shard_sub_medoids[k]` =
    /// k' disk-position medoid IDs (one per sub-cluster). Empty when the
    /// shard was built without sub-clustering (legacy/merged-path fallback).
    /// A2 uses these to pick the query's closest sub-cluster and seed
    /// beam_search from that sub-cluster's medoid via forced_entry_points.
    std::vector<std::vector<float16_t>> shard_sub_centroids;
    std::vector<std::vector<uint32_t>> shard_sub_medoids;

    /// The on-disk directory prefix (`<prefix>.shards`).
    std::string path;

    IVFIndex() = default;
    ~IVFIndex();

    IVFIndex(const IVFIndex&) = delete;
    IVFIndex& operator=(const IVFIndex&) = delete;
    IVFIndex(IVFIndex&&) = delete;
    IVFIndex& operator=(IVFIndex&&) = delete;

    /// Open an IVF index from a `<prefix>.shards/` directory.
    ///
    /// Validates `manifest.json`, loads `centroids.bin` (K × dim FP16),
    /// and opens each shard via `Index::read(shard_path, cache_size_override)`.
    ///
    /// `cache_size_override` (0 = auto per shard) is passed through to each
    /// shard's Index::read. The total resident cache is roughly K × the
    /// per-shard auto size; callers needing a global budget should divide
    /// and pass an explicit per-shard override.
    static std::unique_ptr<IVFIndex> read(const std::string& shards_dir,
                                          uint64_t cache_size_override = 0);
};

}  // namespace sextant
