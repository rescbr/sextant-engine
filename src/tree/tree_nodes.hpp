#pragma once

/// @file tree_nodes.hpp
/// On-disk node and leaf layouts for the hierarchical IVF tree.
///
/// All structures are page-aligned and designed for mmap'd read-only access
/// at search time. The layouts are quantizer-agnostic: the leaf stores raw
/// FastScan code bytes + row IDs + optional per-vector factors (RaBitQ), and
/// the codebook is stored separately (at codebook_page in the superblock).
///
/// ## Quantizer support
/// The tree supports PQ (4-bit/8-bit), PRQ (4-bit), and RaBitQ (4-bit + factors).
/// The leaf format is the same for all — the manifest's quantizer_type / m4 /
/// scan_pq_bits / prq_nsplits fields tell the search path how to interpret the
/// codes and rebuild the codebook.
///
/// ## Internal node layout (multi-page extent)
/// Each internal node stores its children inline:
///   [header: TreeNodeHeader]
///   [child_0: ChildEntry]   — centroid (FP16) + page pointer + extent len
///                             + filter summary (summary_size bytes)
///   [child_1: ChildEntry]
///   ...
///
/// ## Leaf layout (multi-page extent)
///   [header: TreeLeafHeader]
///   [filter_summary: summary_size bytes]   (0 when no schema)
///   [FastScan code blocks: n_blocks × block_bytes]
///   [row_ids: count × sizeof(RowId)]
///   [factors: count × n_factors × sizeof(float)]   (RaBitQ only)
///   [padding to page boundary]
///
/// ## Integrity
/// Every header begins with a 4-byte magic and ends with a CRC32 over all
/// preceding bytes. The magic distinguishes node vs leaf; the CRC enables fsck.

#include "tree/page_file.hpp"
#include "sextant/types.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <cstddef>

namespace sextant::tree {

// ===========================================================================
// Magic + CRC32
// ===========================================================================

/// Magic bytes for internal tree nodes ("TRNO").
inline constexpr uint32_t kTreeNodeMagic = 0x54524E4Fu;
/// Magic bytes for tree leaves ("TLEF").
inline constexpr uint32_t kTreeLeafMagic = 0x544C4546u;

/// IEEE 802.3 CRC32 (same polynomial as zlib's crc32). Table-based.
inline uint32_t compute_crc32(const void* data, std::size_t len) {
    static const std::array<uint32_t, 256> table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    const auto* p = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i)
        crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

/// Compute the CRC32 over the bytes preceding the `header_crc` field.
/// `header` points to the start of the header; `bytes_before_crc` is the
/// byte offset of the `header_crc` field (i.e. offsetof(..., header_crc)).
inline uint32_t header_crc(const void* header, std::size_t bytes_before_crc) {
    return compute_crc32(header, bytes_before_crc);
}

// ===========================================================================
// Internal node
// ===========================================================================

/// Header for an internal tree node. Stored at the start of the node's extent.
struct TreeNodeHeader {
    uint32_t magic;            // kTreeNodeMagic at offset 0
    uint32_t magic_pad;        // explicit pad (magic is 4B; n_children is 8B)
    uint64_t n_children;       // number of children in this node
    uint64_t extent_pages;     // total pages in this extent (for validation)
    uint16_t dim;              // centroid dimensionality
    uint16_t reserved;         // alignment
    uint32_t header_crc;       // CRC32 of bytes [0 .. offsetof(header_crc))
};
static_assert(sizeof(TreeNodeHeader) == 32);

/// A single child entry in an internal node. Centroid is inline (FP16).
/// The filter summary follows the centroid — summary_size bytes, schema-derived.
struct ChildEntry {
    // child_page and child_pages come first (accessed during routing).
    PageId   child_page;       // page index of the child's extent (uint64_t)
    uint64_t child_pages;      // extent length in pages
    uint16_t is_leaf;          // 1 = child is a leaf, 0 = internal node
    uint16_t reserved;
    // Centroid follows (inline FP16, dim elements). Not a fixed-size struct
    // member because dim varies. Accessed via offset arithmetic.
    // Layout: [child_page:8][child_pages:8][is_leaf:2][reserved:2]
    //         [centroid: dim × float16_t]
    //         [filter_summary: summary_size bytes]
};
static_assert(sizeof(ChildEntry) == 24);

/// Compute the byte size of a child entry (including inline centroid + summary).
inline uint32_t child_entry_size(uint16_t dim, uint32_t summary_size) {
    return sizeof(ChildEntry) + dim * sizeof(float16_t) + summary_size;
}

/// Compute the total byte size of an internal node extent.
inline uint64_t node_extent_bytes(uint16_t dim, uint64_t n_children,
                                  uint32_t summary_size) {
    const uint64_t header = sizeof(TreeNodeHeader);
    const uint64_t children =
        static_cast<uint64_t>(child_entry_size(dim, summary_size)) * n_children;
    return header + children;
}

/// Number of pages needed for an internal node.
inline uint64_t node_extent_pages(uint16_t dim, uint64_t n_children,
                                  uint32_t summary_size) {
    return (node_extent_bytes(dim, n_children, summary_size) + kPageSize - 1) /
           kPageSize;
}

// ===========================================================================
// Leaf
// ===========================================================================

/// Header for a leaf node. Stored at the start of the leaf's extent.
struct TreeLeafHeader {
    uint32_t magic;                  // kTreeLeafMagic
    uint64_t count;                  // live vector count
    uint64_t tombstone_count;        // deleted vector count (Phase 4; 0 now)
    uint16_t m4;                     // PQ subquantizers (block_bytes = m4 × 16)
    uint8_t  pq_bits;                // 4 or 8
    uint8_t  n_factors;              // 0 for PQ/PRQ, 2 for RaBitQ
    uint32_t block_bytes;            // bytes per FastScan block
    uint32_t codes_per_block;        // 32 for 4-bit, 16 for 8-bit
    uint64_t extent_pages;           // total pages in this extent
    // --- Phase A additions ---
    uint32_t summary_size;           // schema-determined; 0 = no summary region
    uint32_t n_filter_columns;       // from schema
    uint64_t filter_columns_offset;  // byte offset of filter column data
    PageId   payload_extent_page;    // kInvalidPage if no payload
    uint32_t payload_extent_pages;   // 0 if no payload
    uint8_t  summary_dirty;          // 1 = summary needs repair
    PageId   next_dirty;             // next dirty leaf (kInvalidPage = none)
    uint32_t header_crc;             // CRC32 of bytes [0 .. offsetof(header_crc))
};
static_assert(sizeof(TreeLeafHeader) == 96);

/// Byte offset of the filter summary within a leaf (right after the header).
inline uint32_t leaf_filter_offset() {
    return sizeof(TreeLeafHeader);
}

/// Byte offset of the first FastScan code block within a leaf.
inline uint64_t leaf_codes_offset(uint32_t summary_size) {
    return sizeof(TreeLeafHeader) + summary_size;
}

/// Byte offset of the row_ids array within a leaf.
inline uint64_t leaf_rowids_offset(uint32_t summary_size, uint64_t n_blocks,
                                   uint32_t block_bytes) {
    return leaf_codes_offset(summary_size) +
           static_cast<uint64_t>(n_blocks) * block_bytes;
}

/// Byte offset of the factors array within a leaf (RaBitQ only).
inline uint64_t leaf_factors_offset(uint32_t summary_size, uint64_t n_blocks,
                                    uint32_t block_bytes, uint32_t count) {
    return leaf_rowids_offset(summary_size, n_blocks, block_bytes) +
           static_cast<uint64_t>(count) * sizeof(RowId);
}

/// Compute the total byte size of a leaf extent.
/// `filter_cols_bytes` is the total bytes of the filter column region (after
/// factors). Defaults to 0 — the layout is then identical to the pre-Phase-C
/// format (no filter columns).
inline uint64_t leaf_extent_bytes(uint64_t count, uint16_t m4, uint8_t pq_bits,
                                  uint8_t n_factors, uint32_t summary_size,
                                  uint64_t filter_cols_bytes = 0) {
    const uint32_t cpb = (pq_bits == 4) ? 32 : 16;
    const uint32_t block_bytes = m4 * 16;  // [m][16] for both 4-bit and 8-bit
    const uint32_t n_blocks = (count + cpb - 1) / cpb;
    const uint64_t codes = static_cast<uint64_t>(n_blocks) * block_bytes;
    const uint64_t rowids = static_cast<uint64_t>(count) * sizeof(RowId);
    const uint64_t factors = static_cast<uint64_t>(count) * n_factors * sizeof(float);
    return leaf_codes_offset(summary_size) + codes + rowids + factors + filter_cols_bytes;
}

/// Number of pages needed for a leaf.
inline uint64_t leaf_extent_pages(uint64_t count, uint16_t m4, uint8_t pq_bits,
                                  uint8_t n_factors, uint32_t summary_size,
                                  uint64_t filter_cols_bytes = 0) {
    return static_cast<uint32_t>(
        (leaf_extent_bytes(count, m4, pq_bits, n_factors, summary_size,
                           filter_cols_bytes) +
         kPageSize - 1) / kPageSize);
}

// ===========================================================================
// Leaf extent table (indirection for mutable leaf extents)
// ===========================================================================

/// One entry in the leaf extent table. Maps a leaf_id to its physical
/// location in the page file. When a leaf grows (Phase H closure, Phase J
/// insert), only this entry changes — no parent pointer fixup needed.
struct LeafTableEntry {
    PageId   page;    // starting page of this leaf's extent
    uint32_t pages;   // extent length in pages
} __attribute__((packed));
static_assert(sizeof(LeafTableEntry) == 12);

}  // namespace sextant::tree
