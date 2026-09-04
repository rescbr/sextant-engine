#pragma once

/// @file tree_nodes.hpp
/// On-disk node and leaf layouts for the hierarchical IVF tree.
///
/// All structures are page-aligned and designed for mmap'd read-only access
/// at search time. The layouts are quantizer-agnostic: the leaf stores raw
/// FastScan code bytes + row IDs, and the codebook is stored separately
/// (at codebook_page in the superblock).
///
/// ## Quantizer support
/// The tree supports PQ (4-bit/8-bit) and PRQ (4-bit). The leaf format is
/// the same for both — the manifest's quantizer_type / m4 / scan_pq_bits /
/// prq_nsplits fields tell the search path how to interpret the codes and
/// rebuild the codebook.
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
/// Rounded up to 8-byte alignment so that every ChildEntry in the packed array
/// starts at an address that satisfies alignof(ChildEntry). Without this,
/// variable-length summaries (e.g. bloom filters with odd byte counts) would
/// misalign subsequent entries, causing UB on reinterpret_cast access.
inline uint32_t child_entry_size(uint16_t dim, uint32_t summary_size) {
    const uint32_t raw = sizeof(ChildEntry) + dim * sizeof(float16_t) + summary_size;
    constexpr uint32_t align = alignof(ChildEntry);  // 8
    return (raw + align - 1) & ~(align - 1);
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
    // --- Per-leaf residual PQ additions ---
    /// Leaf state: 0 = CODED (global PQ), 1 = ACCUMULATING (raw FP32),
    /// 2 = CODED_LOCAL (local per-leaf codebook).
    uint8_t  leaf_state = 0;
    /// Padding to align the following u32 fields.
    uint8_t  pad1 = 0;
    uint16_t pad2 = 0;
    /// Byte offset of the FP32 centroid within the leaf extent (0 = none).
    /// For CODED_LOCAL leaves: dim × 4 bytes at this offset.
    uint32_t centroid_offset = 0;
    /// Byte offset of the local codebook within the leaf extent (0 = none).
    /// For CODED_LOCAL leaves: m4 × K × sub_dim × 4 bytes at this offset.
    uint32_t codebook_offset = 0;
    /// Reserved for future use.
    uint32_t reserved2 = 0;
    uint32_t header_crc;             // CRC32 of bytes [0 .. offsetof(header_crc))
};
static_assert(sizeof(TreeLeafHeader) == 112);

/// Byte offset of the filter summary within a leaf (right after the header).
inline uint32_t leaf_filter_offset() {
    return sizeof(TreeLeafHeader);
}

/// Byte offset of the first FastScan code block within a leaf.
/// Aligned to 8 bytes so that subsequent row_ids and filter column arrays
/// (which use reinterpret_cast<const RowId*> etc.) are naturally aligned.
inline uint64_t leaf_codes_offset(uint32_t summary_size) {
    constexpr uint64_t align = 8;
    const uint64_t raw = sizeof(TreeLeafHeader) + summary_size;
    return (raw + align - 1) & ~(align - 1);
}

/// Byte offset of the row_ids array within a leaf.
inline uint64_t leaf_rowids_offset(uint32_t summary_size, uint64_t n_blocks,
                                   uint32_t block_bytes) {
    return leaf_codes_offset(summary_size) +
           static_cast<uint64_t>(n_blocks) * block_bytes;
}

/// Byte offset of the per-leaf uniform levels (lo: dim × fp16, then
/// steps: dim × fp16) in a CodedLocalScalar leaf.
inline uint64_t lsc_levels_offset(uint32_t summary_size) {
    return leaf_codes_offset(summary_size);
}

/// Byte offset of the flat packed-nibble source code in a CodedLocalScalar
/// leaf (the FastScan block region starts here; m4 = dim nibbles/vector).
inline uint64_t lsc_codes_offset(uint32_t summary_size, uint16_t dim) {
    return leaf_codes_offset(summary_size) + 2ull * dim * sizeof(float16_t);
}

/// Scalar leaves are PQ4 FastScan blocks with m4 = dim (one 4-bit nibble
/// per dim): 32 vectors/block, block_bytes = dim × 16.
inline uint32_t scalar_codes_per_block() { return 32; }
inline uint32_t scalar_block_bytes(uint16_t dim) {
    return static_cast<uint32_t>(dim) * 16;
}
inline uint32_t scalar_n_blocks(uint64_t count) {
    return static_cast<uint32_t>((count + 31) / 32);
}

/// Scalar leaves (FastScan block layout, InnerProduct trees) carry a
/// per-vector fp16 IP bias (||x|| / ||x̂||, RaBitQ-style) between the code
/// blocks and row_ids: the scan multiplies ⟨q, x̂⟩ by it to cancel the
/// per-vector reconstruction norm shrinkage. Absent in L2Sq trees.
inline uint64_t scalar_bias_bytes(uint64_t count, bool has_ip_bias) {
    return has_ip_bias ? count * sizeof(float16_t) : 0;
}

/// Byte offset of the row_ids array in a scalar_lm leaf
/// ([header][summary][blocks][ip_biases?][row_ids]).
inline uint64_t scalar_rowids_offset(uint32_t summary_size, uint16_t dim,
                                     uint64_t count, bool has_ip_bias) {
    const uint64_t blocks =
        static_cast<uint64_t>(scalar_n_blocks(count)) * scalar_block_bytes(dim);
    return leaf_codes_offset(summary_size) + blocks +
           scalar_bias_bytes(count, has_ip_bias);
}

/// Byte offset of the row_ids array in a CodedLocalScalar leaf
/// ([header][summary][lo][steps][blocks][ip_biases?][row_ids]).
inline uint64_t lsc_rowids_offset(uint32_t summary_size, uint16_t dim,
                                  uint64_t count, bool has_ip_bias) {
    const uint64_t blocks =
        static_cast<uint64_t>(scalar_n_blocks(count)) * scalar_block_bytes(dim);
    return lsc_codes_offset(summary_size, dim) + blocks +
           scalar_bias_bytes(count, has_ip_bias);
}

/// Compute the total byte size of a leaf extent.
/// `filter_cols_bytes` is the total bytes of the filter column region (after
/// row_ids). Defaults to 0 — the layout is then identical to the pre-Phase-C
/// format (no filter columns).
inline uint64_t leaf_extent_bytes(uint64_t count, uint16_t m4, uint8_t pq_bits,
                                  uint32_t summary_size,
                                  uint64_t filter_cols_bytes = 0) {
    const uint32_t cpb = (pq_bits == 4) ? 32 : 16;
    const uint32_t block_bytes = m4 * 16;  // [m][16] for both 4-bit and 8-bit
    const uint32_t n_blocks = (count + cpb - 1) / cpb;
    const uint64_t codes = static_cast<uint64_t>(n_blocks) * block_bytes;
    const uint64_t rowids = static_cast<uint64_t>(count) * sizeof(RowId);
    return leaf_codes_offset(summary_size) + codes + rowids + filter_cols_bytes;
}

/// Number of pages needed for a leaf.
inline uint64_t leaf_extent_pages(uint64_t count, uint16_t m4, uint8_t pq_bits,
                                   uint32_t summary_size,
                                   uint64_t filter_cols_bytes = 0) {
    return static_cast<uint32_t>(
        (leaf_extent_bytes(count, m4, pq_bits, summary_size,
                           filter_cols_bytes) +
         kPageSize - 1) / kPageSize);
}

// ===========================================================================
// Per-leaf residual PQ layout (LocalPqTreeIndex)
// ===========================================================================

/// Leaf state for per-leaf residual PQ trees.
enum class LeafState : uint8_t {
    /// Coded against a global codebook. Used by IVFTreeIndex.
    Coded = 0,
    /// Accumulating raw FP32 vectors (count < n_train). Searched by brute force.
    Accumulating = 1,
    /// Coded with a local per-leaf codebook + FP32 centroid. Searched via
    /// FastScan with a per-leaf LUT built from query_residual = query - centroid.
    CodedLocal = 2,
    /// Scalar-coded with per-leaf uniform levels: level_d(c) = lo_d +
    /// step_d·c (fp16 lo/steps stored in the leaf). Arithmetic scan with a
    /// per-leaf query transform; layout [header][summary][lo][steps]
    /// [code blocks (FastScan, m4=dim)][ip_biases?][row_ids][filters].
    CodedLocalScalar = 3,
};

/// Local-PQ codebook size in bytes: m4 × K × sub_dim × 4.
/// For PQ4 (K=16, dim=768, m4=192, sub_dim=4): 192 × 16 × 4 × 4 = 49152 bytes.
inline uint64_t local_codebook_bytes(uint16_t m4, uint16_t dim, uint8_t pq_bits) {
    const uint32_t K = (pq_bits == 4) ? 16 : 256;
    const uint32_t sub_dim = dim / m4;
    return static_cast<uint64_t>(m4) * K * sub_dim * sizeof(float);
}

/// FP32 centroid size in bytes: dim × 4.
inline uint64_t local_centroid_bytes(uint16_t dim) {
    return static_cast<uint64_t>(dim) * sizeof(float);
}

/// Byte offset of the FP32 centroid in a CodedLocal leaf.
/// Layout: [header][summary]→[centroid]. Aligned to 16 for NEON.
inline uint64_t local_centroid_offset(uint32_t summary_size) {
    constexpr uint64_t align = 16;
    const uint64_t raw = sizeof(TreeLeafHeader) + summary_size;
    return (raw + align - 1) & ~(align - 1);
}

/// Byte offset of the local codebook in a CodedLocal leaf.
/// Right after the FP32 centroid. Aligned to 16.
inline uint64_t local_codebook_offset(uint32_t summary_size, uint16_t dim) {
    constexpr uint64_t align = 16;
    const uint64_t raw = local_centroid_offset(summary_size) + local_centroid_bytes(dim);
    return (raw + align - 1) & ~(align - 1);
}

/// Byte offset of the first FastScan code block in a CodedLocal leaf.
/// Right after the codebook. Aligned to 8.
inline uint64_t local_codes_offset(uint32_t summary_size, uint16_t dim,
                                    uint16_t m4, uint8_t pq_bits) {
    constexpr uint64_t align = 8;
    const uint64_t raw = local_codebook_offset(summary_size, dim) +
                         local_codebook_bytes(m4, dim, pq_bits);
    return (raw + align - 1) & ~(align - 1);
}

/// Byte offset of the row_ids array in a CodedLocal leaf.
inline uint64_t local_rowids_offset(uint32_t summary_size, uint16_t dim,
                                     uint16_t m4, uint8_t pq_bits,
                                     uint64_t n_blocks, uint32_t block_bytes) {
    return local_codes_offset(summary_size, dim, m4, pq_bits) +
           n_blocks * block_bytes;
}

/// Total byte size of a CodedLocal leaf extent.
inline uint64_t local_coded_extent_bytes(uint64_t count, uint16_t dim,
                                          uint16_t m4, uint8_t pq_bits,
                                          uint32_t summary_size,
                                          uint64_t filter_cols_bytes = 0) {
    const uint32_t cpb = (pq_bits == 4) ? 32 : 16;
    const uint32_t block_bytes = m4 * 16;
    const uint32_t n_blocks = (count + cpb - 1) / cpb;
    const uint64_t codes = static_cast<uint64_t>(n_blocks) * block_bytes;
    const uint64_t rowids = count * sizeof(RowId);
    return local_codes_offset(summary_size, dim, m4, pq_bits) +
           codes + rowids + filter_cols_bytes;
}

/// Pages needed for a CodedLocal leaf.
inline uint64_t local_coded_extent_pages(uint64_t count, uint16_t dim,
                                          uint16_t m4, uint8_t pq_bits,
                                          uint32_t summary_size,
                                          uint64_t filter_cols_bytes = 0) {
    return (local_coded_extent_bytes(count, dim, m4, pq_bits,
                                     summary_size, filter_cols_bytes) +
            kPageSize - 1) / kPageSize;
}

/// Byte offset of raw FP32 vectors in an Accumulating leaf.
/// Layout: [header][summary]→[raw vectors]. Aligned to 16 for NEON.
inline uint64_t accum_vectors_offset(uint32_t summary_size) {
    constexpr uint64_t align = 16;
    const uint64_t raw = sizeof(TreeLeafHeader) + summary_size;
    return (raw + align - 1) & ~(align - 1);
}

/// Byte offset of row_ids in an Accumulating leaf.
inline uint64_t accum_rowids_offset(uint32_t summary_size, uint16_t dim,
                                     uint64_t count) {
    return accum_vectors_offset(summary_size) + count * dim * sizeof(float);
}

/// Total byte size of an Accumulating leaf extent.
inline uint64_t accum_extent_bytes(uint64_t count, uint16_t dim,
                                    uint32_t summary_size,
                                    uint64_t filter_cols_bytes = 0) {
    return accum_rowids_offset(summary_size, dim, count) +
           count * sizeof(RowId) + filter_cols_bytes;
}

/// Pages needed for an Accumulating leaf.
inline uint64_t accum_extent_pages(uint64_t count, uint16_t dim,
                                    uint32_t summary_size,
                                    uint64_t filter_cols_bytes = 0) {
    return (accum_extent_bytes(count, dim, summary_size, filter_cols_bytes) +
            kPageSize - 1) / kPageSize;
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
