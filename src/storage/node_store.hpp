#pragma once

/// @file node_store.hpp
/// NodeStore abstraction: decouples VamanaCore from the flat-RAM buffer layout.
///
/// Two backends:
///   - FlatNodeStore: direct pointers into flat RAM buffers (build + small
///     indices at search). Zero-copy, O(1).
///   - PagedNodeStore: on-demand block reads from sidecar files via DirectFile,
///     cached in a ShardedLRUCache. This is the SSD-resident search path
///     (Issue 5/13/25): an index of 100M+ vectors is searchable with <100MB
///     of idle RAM because graph nodes and PQ codes are fetched block-by-block
///     and evicted under an LRU budget.
///
/// VamanaCore goes through this interface for ALL node/code access (both build
/// and search). During build, Engine installs a FlatNodeStore over the flat
/// buffers; during search (after open or post-flush), Engine installs a
/// PagedNodeStore over the sidecar files.

#include "storage/direct_io.hpp"
#include "storage/sharded_lru.hpp"
#include "storage/sidecar_header.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sextant {

/// Abstract interface for accessing graph nodes and PQ codes.
///
/// Pinning contract: pin_node/pin_code return a pointer whose lifetime is
/// governed by the store. For FlatNodeStore the pointer is valid for the
/// lifetime of the underlying buffer. For PagedNodeStore the pointer is valid
/// as long as the containing block stays in the cache; callers MUST copy any
/// data they need before issuing pin calls that could evict the block. In
/// practice beam_search copies neighbor ids and distances into local scratch
/// before advancing, so this is safe.
class NodeStore {
public:
    virtual ~NodeStore() = default;

    /// Read a graph node. Returns a pointer to node_size bytes (read-only).
    virtual const uint8_t* pin_node(uint32_t internal_id) = 0;

    /// Release a pinned node. No-op for Flat; refcount/deadline for paged.
    /// Phase 1: the LRU manages eviction; unpin is a no-op (documented).
    virtual void unpin_node(uint32_t internal_id) = 0;

    /// Read a PQ code. Returns a pointer to code_size bytes (read-only).
    virtual const uint8_t* pin_code(uint32_t internal_id) = 0;

    /// Release a pinned code.
    virtual void unpin_code(uint32_t internal_id) = 0;

    /// Write access (build mode). Only FlatNodeStore supports this.
    /// PagedNodeStore throws Error(NotImplemented).
    virtual uint8_t* mutable_node(uint32_t internal_id) = 0;
    virtual uint8_t* mutable_code(uint32_t internal_id) = 0;

    /// True if this store reads from disk in blocks (PagedNodeStore).
    /// PageSearch only pays off when a block I/O brings co-located nodes into
    /// the cache; for RAM-resident flat stores it is skipped.
    virtual bool is_paged() const = 0;
};

/// Flat buffer backing. Used during build and for small indices at search.
/// Pointers are direct into the caller-owned buffers (no copy, no lock).
class FlatNodeStore : public NodeStore {
public:
    /// `nodes` is count × node_size (writable). `codes` is count × code_size.
    FlatNodeStore(uint8_t* nodes, const uint8_t* codes,
                  uint32_t node_size, uint8_t code_size);

    const uint8_t* pin_node(uint32_t id) override;
    void unpin_node(uint32_t) override {}
    const uint8_t* pin_code(uint32_t id) override;
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
///
/// The .graph file is laid out (after the SidecarHeader) as a flat run of
/// node_size-byte records. We read it in fixed kBlockSize blocks, one LRU
/// entry per block. The .codes file is similarly flat (code_size per record),
/// read in its own block granularity. Both share the same cache capacity
/// budget; each has its own key space (graph blocks vs code blocks are
/// disambiguated by a high bit so they can't collide within a shard).
class PagedNodeStore : public NodeStore {
public:
    /// `cache_size_bytes` is the total LRU budget (split across shards).
    /// Block size is kBlockSize (256 KB).
    PagedNodeStore(const std::string& graph_path,
                   const std::string& codes_path,
                   uint32_t node_size, uint8_t code_size,
                   uint32_t num_shards,
                   uint64_t cache_size_bytes);

    const uint8_t* pin_node(uint32_t internal_id) override;
    void unpin_node(uint32_t) override;  // Phase 1: no-op
    const uint8_t* pin_code(uint32_t internal_id) override;
    void unpin_code(uint32_t) override;  // Phase 1: no-op
    uint8_t* mutable_node(uint32_t) override;
    uint8_t* mutable_code(uint32_t) override;
    bool is_paged() const override { return true; }

    /// Read counters for testing/diagnostics.
    uint64_t graph_reads() const { return graph_reads_; }
    uint64_t code_reads() const { return code_reads_; }

private:
    DirectFile graph_file_;
    DirectFile codes_file_;
    uint32_t node_size_;
    uint8_t code_size_;

    // Block layout derived from kBlockSize.
    uint32_t nodes_per_block_;   // kBlockSize / node_size
    uint32_t codes_per_block_;   // kBlockSize / code_size
    uint32_t graph_block_size_;  // nodes_per_block * node_size (≤ kBlockSize)
    uint32_t codes_block_size_;  // codes_per_block * code_size (≤ kBlockSize)

    ShardedLRUCache cache_;

    // Key spaces: graph blocks use [0, 2^63); code blocks use [2^63, 2^64).
    static constexpr uint64_t kCodeKeyBit = 1ULL << 63;

    mutable std::mutex reads_mu_;
    uint64_t graph_reads_ = 0;
    uint64_t code_reads_ = 0;

    /// Fetch a node block: cache lookup, else read from disk and insert.
    /// Returns a pointer to the block buffer (valid until eviction).
    const uint8_t* get_node_block(uint64_t block_idx);
    const uint8_t* get_code_block(uint64_t block_idx);
};

}  // namespace sextant
