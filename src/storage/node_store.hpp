#pragma once

/// @file node_store.hpp
/// NodeStore abstraction: decouples VamanaCore from the flat-RAM buffer layout.
///
/// Backends:
///   - FlatNodeStore: direct pointers into flat RAM buffers (build + small
///     indices at search). Zero-copy, O(1).
///   - PagedNodeStore: on-demand block reads from sidecar files via DirectFile,
///     cached in a ShardedLRUCache. This is the SSD-resident search path.
///   - MemGraph: topology-based RAM cache for the entry-point neighborhood.
///     Delegates cold-node misses to a backing PagedNodeStore.
///
/// VamanaCore goes through this interface for ALL node/code access (both build
/// and search). During build, Engine installs a FlatNodeStore over the flat
/// buffers; during search, Engine installs a MemGraph over a PagedNodeStore.

#include "storage/direct_io.hpp"
#include "storage/sharded_lru.hpp"
#include "storage/sidecar_header.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sextant {

/// Result of a pin operation. Tells the caller where the data came from,
/// which determines whether SSD-amortization techniques (PageSearch) are
/// worthwhile for this node.
struct PinResult {
    const uint8_t* data = nullptr;
    /// True if this pin required I/O (cache miss → SSD read or LRU fetch).
    /// False if served from RAM (flat buffer, MemGraph hit, or LRU hit).
    /// beam_search uses this to decide whether PageSearch is worth the
    /// CPU cost: only amortize when we actually paid for I/O.
    bool from_ssd = false;
};

/// Abstract interface for accessing graph nodes and PQ codes.
class NodeStore {
public:
    virtual ~NodeStore() = default;

    /// Read a graph node. Returns pointer + whether it caused I/O.
    virtual PinResult pin_node(uint32_t internal_id) = 0;

    /// Release a pinned node. No-op for flat; refcount for paged.
    virtual void unpin_node(uint32_t internal_id) = 0;

    /// Read a PQ code. Returns pointer + whether it caused I/O.
    virtual PinResult pin_code(uint32_t internal_id) = 0;

    /// Release a pinned code.
    virtual void unpin_code(uint32_t internal_id) = 0;

    /// Write access (build mode). Only FlatNodeStore supports this.
    virtual uint8_t* mutable_node(uint32_t internal_id) = 0;
    virtual uint8_t* mutable_code(uint32_t internal_id) = 0;

    /// True if this store can serve SSD reads (PagedNodeStore or MemGraph
    /// with a paged backing store). Used to gate DynamicWidth.
    virtual bool is_paged() const = 0;
};

/// Flat buffer backing. Used during build and for small indices at search.
class FlatNodeStore : public NodeStore {
public:
    FlatNodeStore(uint8_t* nodes, const uint8_t* codes,
                  uint32_t node_size, uint8_t code_size);

    PinResult pin_node(uint32_t id) override;
    void unpin_node(uint32_t) override {}
    PinResult pin_code(uint32_t id) override;
    void unpin_code(uint32_t) override {}
    uint8_t* mutable_node(uint32_t id) override;
    uint8_t* mutable_code(uint32_t id) override;
    bool is_paged() const override { return false; }

private:
    uint8_t* nodes_;
    const uint8_t* codes_;
    uint32_t node_size_;
    uint8_t code_size_;
};

/// LRU-paged backing — the SSD-resident search path.
class PagedNodeStore : public NodeStore {
public:
    PagedNodeStore(const std::string& graph_path,
                   const std::string& codes_path,
                   uint32_t node_size, uint8_t code_size,
                   uint32_t num_shards,
                   uint64_t cache_size_bytes);

    PinResult pin_node(uint32_t internal_id) override;
    void unpin_node(uint32_t) override;
    PinResult pin_code(uint32_t internal_id) override;
    void unpin_code(uint32_t) override;
    uint8_t* mutable_node(uint32_t) override;
    uint8_t* mutable_code(uint32_t) override;
    bool is_paged() const override { return true; }

    uint64_t graph_reads() const { return graph_reads_; }
    uint64_t code_reads() const { return code_reads_; }

private:
    DirectFile graph_file_;
    DirectFile codes_file_;
    uint32_t node_size_;
    uint8_t code_size_;
    uint32_t nodes_per_block_;
    uint32_t codes_per_block_;
    uint32_t graph_block_size_;
    uint32_t codes_block_size_;
    ShardedLRUCache cache_;
    static constexpr uint64_t kCodeKeyBit = 1ULL << 63;

    mutable std::mutex reads_mu_;
    uint64_t graph_reads_ = 0;
    uint64_t code_reads_ = 0;

    PinResult get_node_block(uint64_t block_idx);
    PinResult get_code_block(uint64_t block_idx);
};

}  // namespace sextant
