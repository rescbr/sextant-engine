#pragma once

/// @file superblock.hpp
/// Superblock for the IVF tree file.
///
/// The superblock is the ONLY fixed-location structure in the file. Page 0
/// and page 1 both hold a superblock (active + shadow). Updates write to the
/// shadow, fdatasync, then flip the active pointer atomically (a single
/// overwrite of page 0 or 1). On open, we read both copies and pick the one
/// with the higher commit_seq (the last successfully-committed write).
///
/// The superblock stores:
///   - magic number (format identification)
///   - commit sequence number (for shadow comparison)
///   - root node pointer (page + extent length)
///   - tree metadata (depth, leaf count)
///   - allocator metadata (n_pages, free_list_head, n_free_pages, bitmap loc)
///   - codebook pointer
///   - TOML config blob location (offset + length within a dedicated page range)
///
/// All fields are little-endian (we write on ARM/x86, both LE).

#include "tree/page_file.hpp"

#include <cstdint>
#include <cstring>

namespace sextant::tree {

/// Magic identifying the Sextant tree file format.
/// "SEXTREE\0" — 8 bytes including null.
inline constexpr uint64_t kSuperblockMagic = 0x0045525454584553ull; // "SEXTREE\0"

/// The superblock occupies page 0 (active) and page 1 (shadow).
inline constexpr PageId kSuperblockPage = 0;
inline constexpr PageId kSuperblockShadowPage = 1;
inline constexpr PageId kFirstDataPage = 2;  // bitmap starts here by default

/// On-disk superblock layout. Exactly fits in one 4KB page (with padding).
/// Packed to ensure deterministic layout across compilers.
struct SuperblockDisk {
    uint64_t magic;                  // kSuperblockMagic
    uint64_t commit_seq;             // monotonically increasing; higher = newer

    // Root node
    PageId   root_node_page;         // page index of root extent
    uint32_t root_node_pages;        // extent length in pages

    // Tree metadata
    uint16_t depth;                  // tree depth (1 = flat, 2 = two-level, ...)
    uint16_t reserved1;              // alignment padding
    uint64_t n_leaves;               // total leaf count

    // Allocator state
    uint64_t n_pages;                // total pages in file
    PageId   free_list_head;         // first free page (kInvalidPage if empty)
    uint64_t n_free_pages;           // free list length

    // Bitmap
    PageId   alloc_bitmap_page;      // bitmap start page
    uint32_t alloc_bitmap_pages;     // number of bitmap pages

    // Codebook
    PageId   codebook_page;          // serialized quantizer blob start
    uint32_t codebook_pages;         // codebook extent length

    // TOML config blob
    PageId   config_page;            // TOML blob start page
    uint32_t config_pages;           // TOML blob extent length

    // PCA routing data (0 pages = no PCA routing)
    PageId   pca_page;               // PCA blob start (projection + centroids)
    uint32_t pca_pages;              // PCA blob extent length

    // Reserved for future use.
    // Fields total: 8+8 + 8+4 + 2+2+8 + 8+8+8 + 8+4 + 8+4 + 8+4 + 8+4 = 112 bytes
    uint8_t  reserved2[4096 - 128];
};
static_assert(sizeof(SuperblockDisk) == kPageSize,
              "SuperblockDisk must be exactly one page");

/// In-memory superblock wrapper with load/commit logic.
class Superblock {
public:
    Superblock() = default;

    /// Read both copies from the file, pick the one with the higher commit_seq.
    /// Throws if the magic doesn't match on either copy.
    void load(const PageFile& file);

    /// Write the superblock to the shadow page, fdatasync, then overwrite the
    /// active page. This is the atomic commit: a crash leaves either the old
    /// or the new superblock, never a torn write.
    ///
    /// Increments commit_seq before writing.
    void commit(PageFile& file);

    /// Initialize a fresh superblock (commit_seq = 0, all pointers invalid).
    /// Used by `format`.
    void init_fresh(PageId bitmap_page, uint32_t bitmap_pages);

    // --- Mutable accessors (for build/update) ---

    uint64_t commit_seq() const { return disk_.commit_seq; }
    PageId   root_node_page() const { return disk_.root_node_page; }
    uint32_t root_node_pages() const { return disk_.root_node_pages; }
    uint16_t depth() const { return disk_.depth; }
    uint64_t n_leaves() const { return disk_.n_leaves; }
    uint64_t n_pages() const { return disk_.n_pages; }
    PageId   free_list_head() const { return disk_.free_list_head; }
    uint64_t n_free_pages() const { return disk_.n_free_pages; }
    PageId   alloc_bitmap_page() const { return disk_.alloc_bitmap_page; }
    uint32_t alloc_bitmap_pages() const { return disk_.alloc_bitmap_pages; }
    PageId   codebook_page() const { return disk_.codebook_page; }
    uint32_t codebook_pages() const { return disk_.codebook_pages; }
    PageId   config_page() const { return disk_.config_page; }
    uint32_t config_pages() const { return disk_.config_pages; }
    PageId   pca_page() const { return disk_.pca_page; }
    uint32_t pca_pages() const { return disk_.pca_pages; }

    void set_root(PageId page, uint32_t pages) {
        disk_.root_node_page = page;
        disk_.root_node_pages = pages;
    }
    void set_depth(uint16_t d) { disk_.depth = d; }
    void set_n_leaves(uint64_t n) { disk_.n_leaves = n; }
    void set_n_pages(uint64_t n) { disk_.n_pages = n; }
    void set_free_list(PageId head, uint64_t n_free) {
        disk_.free_list_head = head;
        disk_.n_free_pages = n_free;
    }
    void set_bitmap(PageId page, uint32_t pages) {
        disk_.alloc_bitmap_page = page;
        disk_.alloc_bitmap_pages = pages;
    }
    void set_codebook(PageId page, uint32_t pages) {
        disk_.codebook_page = page;
        disk_.codebook_pages = pages;
    }
    void set_config(PageId page, uint32_t pages) {
        disk_.config_page = page;
        disk_.config_pages = pages;
    }
    void set_pca(PageId page, uint32_t pages) {
        disk_.pca_page = page;
        disk_.pca_pages = pages;
    }

private:
    SuperblockDisk disk_{};

    /// Read a superblock from the given page. Returns false if the page
    /// doesn't have a valid magic (e.g., never written).
    static bool read_copy(const PageFile& file, PageId page,
                          SuperblockDisk& out);

    /// Write the superblock to a single page.
    static void write_copy(PageFile& file, PageId page,
                           const SuperblockDisk& sb);
};

}  // namespace sextant::tree
