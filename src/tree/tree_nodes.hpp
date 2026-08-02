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
///   [child_1: ChildEntry]
///   ...
///
/// ## Leaf layout (multi-page extent)
///   [header: TreeLeafHeader]
///   [FastScan code blocks: n_blocks × block_bytes]
///   [row_ids: count × sizeof(RowId)]
///   [factors: count × n_factors × sizeof(float)]   (RaBitQ only)
///   [padding to page boundary]

#include "tree/page_file.hpp"
#include "sextant/types.hpp"

#include <cstdint>
#include <cstring>

namespace sextant::tree {

/// Filter summary reserved in every leaf header (256 bytes).
/// Unused in Phase 1; reserved for Phase 5 (filtered search).
inline constexpr uint32_t kFilterSummarySize = 256;

// ===========================================================================
// Internal node
// ===========================================================================

/// Header for an internal tree node. Stored at the start of the node's extent.
struct TreeNodeHeader {
    uint64_t n_children;       // number of children in this node
    uint64_t extent_pages;     // total pages in this extent (for validation)
    uint16_t dim;              // centroid dimensionality
    uint16_t reserved;         // alignment
    uint32_t reserved2;        // alignment to 8 bytes
};
static_assert(sizeof(TreeNodeHeader) == 24);

/// A single child entry in an internal node. Centroid is inline (FP16).
/// The filter summary follows the centroid — reserved for Phase 5.
struct ChildEntry {
    // child_page and child_pages come first (accessed during routing).
    PageId   child_page;       // page index of the child's extent (uint64_t)
    uint64_t child_pages;      // extent length in pages
    uint16_t is_leaf;          // 1 = child is a leaf, 0 = internal node
    uint16_t reserved;
    // Centroid follows (inline FP16, dim elements). Not a fixed-size struct
    // member because dim varies. Accessed via offset arithmetic.
    // Layout: [child_page:4][child_pages:4][is_leaf:2][reserved:2]
    //         [centroid: dim × float16_t]
    //         [filter_summary: 256 bytes]
};
static_assert(sizeof(ChildEntry) == 24);

/// Compute the byte size of a child entry (including inline centroid + filter).
inline uint32_t child_entry_size(uint16_t dim) {
    return sizeof(ChildEntry) + dim * sizeof(float16_t) + kFilterSummarySize;
}

/// Compute the total byte size of an internal node extent.
inline uint64_t node_extent_bytes(uint16_t dim, uint64_t n_children) {
    const uint64_t header = sizeof(TreeNodeHeader);
    const uint64_t children = static_cast<uint64_t>(child_entry_size(dim)) * n_children;
    return header + children;
}

/// Number of pages needed for an internal node.
inline uint64_t node_extent_pages(uint16_t dim, uint64_t n_children) {
    return (node_extent_bytes(dim, n_children) + kPageSize - 1) / kPageSize;
}

// ===========================================================================
// Leaf
// ===========================================================================

/// Header for a leaf node. Stored at the start of the leaf's extent.
struct TreeLeafHeader {
    uint64_t count;            // live vector count
    uint64_t tombstone_count;  // deleted vector count (Phase 4; 0 in Phase 1)
    uint16_t m4;               // PQ subquantizers (block_bytes = m4 × 16 for 4-bit)
    uint8_t  pq_bits;          // 4 or 8
    uint8_t  n_factors;        // 0 for PQ/PRQ, 2 for RaBitQ (dp_mult, or_minus_c_l2sqr)
    uint32_t block_bytes;      // bytes per FastScan block (m4 × 16 for 4-bit, m4 × 256 for 8-bit)
    uint32_t codes_per_block;  // 32 for 4-bit, 16 for 8-bit
    uint64_t extent_pages;     // total pages in this extent
    // filter_summary follows after this struct (kFilterSummarySize bytes).
};
static_assert(sizeof(TreeLeafHeader) == 40);

/// Byte offset of the filter summary within a leaf (right after the header).
inline uint32_t leaf_filter_offset() {
    return sizeof(TreeLeafHeader);
}

/// Byte offset of the first FastScan code block within a leaf.
inline uint64_t leaf_codes_offset() {
    return sizeof(TreeLeafHeader) + kFilterSummarySize;
}

/// Byte offset of the row_ids array within a leaf.
inline uint64_t leaf_rowids_offset(uint64_t n_blocks, uint32_t block_bytes) {
    return leaf_codes_offset() + static_cast<uint64_t>(n_blocks) * block_bytes;
}

/// Byte offset of the factors array within a leaf (RaBitQ only).
inline uint64_t leaf_factors_offset(uint64_t n_blocks, uint32_t block_bytes,
                                    uint32_t count) {
    return leaf_rowids_offset(n_blocks, block_bytes) +
           static_cast<uint64_t>(count) * sizeof(RowId);
}

/// Compute the total byte size of a leaf extent.
inline uint64_t leaf_extent_bytes(uint64_t count, uint16_t m4, uint8_t pq_bits,
                                  uint8_t n_factors) {
    const uint32_t cpb = (pq_bits == 4) ? 32 : 16;
    const uint32_t block_bytes = m4 * 16;  // [m][16] for both 4-bit and 8-bit
    const uint32_t n_blocks = (count + cpb - 1) / cpb;
    const uint64_t codes = static_cast<uint64_t>(n_blocks) * block_bytes;
    const uint64_t rowids = static_cast<uint64_t>(count) * sizeof(RowId);
    const uint64_t factors = static_cast<uint64_t>(count) * n_factors * sizeof(float);
    return leaf_codes_offset() + codes + rowids + factors;
}

/// Number of pages needed for a leaf.
inline uint64_t leaf_extent_pages(uint64_t count, uint16_t m4, uint8_t pq_bits,
                                  uint8_t n_factors) {
    return static_cast<uint32_t>(
        (leaf_extent_bytes(count, m4, pq_bits, n_factors) + kPageSize - 1) / kPageSize);
}

}  // namespace sextant::tree
