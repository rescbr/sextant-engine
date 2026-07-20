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
};

}  // namespace sextant
