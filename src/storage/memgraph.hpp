#pragma once

/// @file memgraph.hpp
/// MemGraph — an in-memory navigation graph that caches the entry-point
/// neighborhood for fast SSD-resident search.
///
/// The paged search path (PagedNodeStore + ShardedLRUCache) thrashes under
/// streaming graph access because graph search touches a wide, shifting set
/// of nodes. The access value is concentrated at the entry-point neighborhood
/// — the "highway" nodes that EVERY query traverses.
///
/// MemGraph is a two-tier NodeStore: it holds the BFS neighborhood (default
/// 3 hops) of the entry points in RAM, and delegates cold nodes to a backing
/// store (PagedNodeStore). The hot path is a single branch + direct-indexed
/// array lookup — no locks, no hash map, no LRU list — so it is dramatically
/// faster than the ShardedLRUCache path for the nodes it covers.
///
/// Construction reads the .graph sidecar (after the SidecarHeader) as a flat
/// run of node_size-byte records, BFS-walks the neighbor lists, and copies
/// the collected nodes and their PQ codes into compact RAM buffers.

#include "storage/node_store.hpp"
#include "storage/sidecar_header.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sextant {

/// In-RAM entry-point neighborhood cache over a backing NodeStore.
class MemGraph : public NodeStore {
public:
    /// Build a MemGraph from a flat node buffer + entry points.
    /// Collects all nodes within `num_hops` BFS hops of any entry point.
    /// Stores their neighbor lists and PQ codes in compact RAM buffers.
    MemGraph(const uint8_t* all_nodes, const uint8_t* all_codes,
             uint32_t node_size, uint8_t code_size, uint32_t total_count,
             const std::vector<uint32_t>& entry_points,
             uint32_t num_hops = 3);

    /// Open a MemGraph from sidecar files. Reads the BFS neighborhood from
    /// .graph and .codes into RAM. `graph_path`/`codes_path` are the full
    /// sidecar paths (each begins with a SidecarHeader).
    MemGraph(const std::string& graph_path, const std::string& codes_path,
             uint32_t node_size, uint8_t code_size, uint32_t total_count,
             const std::vector<uint32_t>& entry_points,
             uint32_t num_hops = 3);

    // NodeStore interface
    const uint8_t* pin_node(uint32_t id) override;
    void unpin_node(uint32_t) override;
    const uint8_t* pin_code(uint32_t id) override;
    void unpin_code(uint32_t) override;
    uint8_t* mutable_node(uint32_t) override;   // throws
    uint8_t* mutable_code(uint32_t) override;   // throws
    bool is_paged() const override { return true; }

    /// Set the backing store for nodes NOT in the MemGraph.
    /// pin_node/pin_code check MemGraph first, fall through to backing store.
    void set_backing(NodeStore* backing) { backing_ = backing; }

    /// How many nodes are in the MemGraph.
    uint32_t cached_count() const { return cached_count_; }

    /// True if `id` is in the MemGraph.
    bool is_cached(uint32_t id) const {
        return id < membership_.size() && membership_[id];
    }

private:
    // BFS to collect node IDs within num_hops of entry points. `nodes` points
    // at the first record (past any header); the caller handles file loading.
    void collect_neighborhood(const uint8_t* nodes, uint32_t total_count,
                              const std::vector<uint32_t>& entry_points,
                              uint32_t num_hops);

    // Copy collected node records + codes into the compact RAM buffers and
    // build the id_to_local_ / membership_ lookups.
    void materialize(const uint8_t* nodes, const uint8_t* codes);

    // Compact storage for cached nodes.
    std::vector<uint8_t> node_data_;   // cached_count × node_size
    std::vector<uint8_t> code_data_;   // cached_count × code_size
    std::vector<uint32_t> id_map_;     // cached local index → original ID
    std::vector<bool> membership_;     // total_count bits: is this ID cached?
    // Direct-indexed: original ID → local index in node_data_.
    // total_count entries; UINT32_MAX if not cached. O(1) lookup, no hashing.
    std::vector<uint32_t> id_to_local_;

    uint32_t node_size_;
    uint8_t code_size_;
    uint32_t total_count_;
    uint32_t cached_count_ = 0;
    NodeStore* backing_ = nullptr;

    // Collected during BFS, consumed by materialize().
    std::vector<uint32_t> collected_;
};

}  // namespace sextant
