#pragma once

/// @file index.hpp
/// `Index` — the unit of handoff between Builder, Searcher, and Estimator.
///
/// An Index bundles everything needed to search a built graph:
///   - metadata (n_vectors, dim, resolved build params, entry points)
///   - the trained quantizer
///   - a VamanaCore configured for search
///   - a NodeStore (flat RAM, paged SSD, or MemGraph over paged)
///
/// Phase A (this file): introduce Index as a plain container. Engine still
/// owns build/open orchestration but routes state through Index. Later phases
/// (B, C, D) extract Builder/Searcher/Estimator and Index becomes the
/// handoff type between them.
///
/// Future Layer 2 work (deferred per plan):
///   - TOML manifest reader/writer (Index::read / Index::write)
///   - IndexStore interface (currently NodeStore + write helpers)
///   - streaming-write API (currently Engine::write_sidecars_)

#include "sextant/config.hpp"
#include "sextant/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sextant {

class PqQuantizer;
class VamanaCore;
class NodeStore;
class FlatNodeStore;
class PagedNodeStore;
class MemGraph;

/// Build/run metadata + the objects needed to serve queries.
///
/// Invariants:
///   - `quantizer` is always populated once an Index is non-empty.
///   - `core` is always configured with `params` and the quantizer.
///   - Exactly one store is active:
///       * `flat_store`   (RAM buffers owned by the Index — build + insert)
///       * `paged_store`  (disk + cache — search)
///     When `paged_store` is active, `memgraph` may sit atop it as the
///     entry-point neighborhood cache. The VamanaCore's store pointer is
///     set to whichever store is the *top* of the stack (memgraph if
///     present, else paged_store, else flat_store).
struct Index {
    // --- Metadata ---
    uint64_t count = 0;
    Dim dim = 0;
    uint32_t node_size = 0;
    uint32_t code_size = 0;
    ResolvedParams params{};
    std::vector<uint32_t> entry_points;
    std::string path;  ///< Index directory/path prefix (no extension).
    /// Cache budget resolved by Index::read (cache_size_override or auto from
    /// index size + physical RAM). Searcher splits this across its N workers.
    uint64_t cache_size_bytes = 0;

    // --- Index data + search core ---
    std::unique_ptr<PqQuantizer> quantizer;
    std::unique_ptr<VamanaCore> core;

    // At most one store stack is active. See invariants above.
    std::unique_ptr<FlatNodeStore> flat_store;
    std::unique_ptr<PagedNodeStore> paged_store;
    std::unique_ptr<MemGraph> memgraph;

    /// Flat-in-RAM build buffers (owned by Index when flat_store is active).
    /// Codes and nodes are aligned allocations; raw_vecs is FP16 (build prune).
    /// Null when the Index is in paged mode.
    uint8_t* codes_buffer = nullptr;
    uint8_t* nodes_buffer = nullptr;
    float16_t* raw_vecs_buffer = nullptr;
    float* fp32_vecs_buffer = nullptr;  ///< FP32 build mode only (SEXTANT_FP32_BUILD). Null otherwise.

    /// IVF shard sub-cluster entry-point centroids (A1/A2/A3, Workstream A).
    /// Populated ONLY by Builder::build_shard_into_ for IVF shards (empty for
    /// the merged path). Layout:
    ///   - `sub_centroids`: k' × dim FP16 (k-means centroids on PQ codes).
    ///   - `sub_centroid_medoids`: k' × sub_medoids_per_cluster shard-LOCAL IDs
    ///     (row-major: cluster c, slot s → index c*sub_medoids_per_cluster + s).
    ///     Each slot is one of the cluster's M members nearest the sub-centroid
    ///     (FP16 L2sq). M = sub_medoids_per_cluster (default 4 → 8×4=32 entry
    ///     points when k'=8).
    /// write_sidecars_ remaps the medoids to disk positions and emits a `.epc`
    /// sidecar; IVFIndex::read loads them for query-adaptive entry-point
    /// selection (A2). Null/empty for non-IVF indexes.
    std::vector<float16_t> sub_centroids;
    std::vector<uint32_t> sub_centroid_medoids;
    uint32_t sub_medoids_per_cluster = 0;  // M; 0 when sub_centroids is empty

    Index() = default;
    ~Index();

    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
    Index(Index&&) = delete;
    Index& operator=(Index&&) = delete;

    /// True when flat RAM buffers are resident (build or post-insert mode).
    bool has_flat_buffers() const { return nodes_buffer != nullptr; }

    /// True when the index is SSD-resident (paged search mode).
    bool is_paged() const { return paged_store != nullptr; }

    /// The top-of-stack store the VamanaCore should read through. Returns
    /// memgraph if present (it delegates to paged_store), else paged_store,
    /// else flat_store. Null if no store is installed.
    NodeStore* top_store() const;

    /// Read an index from sidecar files (.meta + .graph + .codes + .ball +
    /// .manifest). Replaces the former Engine::open / Engine::load_sidecars.
    ///
    /// Validates the atomic .manifest commit point, loads the quantizer +
    /// entry points + params from .meta, reconstructs the search VamanaCore,
    /// installs a PagedNodeStore over .graph/.codes, and builds a MemGraph
    /// (3-hop BFS neighborhood of the entry points) over the PagedNodeStore
    /// + optional .ball FP16 sidecar.
    ///
    /// `cache_size_override` (0 = auto from index size + physical RAM) sets
    /// the L2 (BlockCache) budget. The cache rebalance enable/disable flag
    /// lives on the Searcher, not here — Index doesn't know about adaptive
    /// rebalancing.
    static std::unique_ptr<Index> read(const std::string& path,
                                        uint64_t cache_size_override = 0);
};

}  // namespace sextant
