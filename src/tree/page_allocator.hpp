#pragma once

/// @file page_allocator.hpp
/// Extent-based page allocator for the IVF tree file.
///
/// Manages page allocation within a PageFile. Supports two allocation modes:
///   - Single page: O(1) from a free list.
///   - Contiguous extent (N pages): for leaves (sequential NVMe I/O) and
///     internal nodes (cache-locality). Uses a bitmap-based first-fit search.
///
/// The allocator state lives in the Superblock (free_list_head, n_pages,
/// n_free_pages). The bitmap lives in dedicated pages in the file. This class
/// is the in-memory working state that loads from / persists to the superblock.
///
/// Threading: NOT thread-safe. The build is single-writer. Phase 4 (dynamic
/// inserts) will add a mutex at the index level, not here.

#include "tree/page_file.hpp"

#include <cstdint>
#include <vector>

namespace sextant::tree {

/// In-memory allocator state. Persisted via the superblock + bitmap pages.
///
/// The bitmap has one bit per page in the file. Bit set = allocated.
/// The free list is a linked list of free single pages: each free page stores
/// the PageId of the next free page in its first 4 bytes. This avoids needing
/// a separate free-list data structure on disk — free pages self-describe.
///
/// For contiguous extents, the bitmap is searched first-fit. This is O(n_pages)
/// per allocation but acceptable for build-time (infrequent) and leaf-split
/// (rare). The design doc: "Start with simple free-list; add buddy if
/// fragmentation is measured."
class PageAllocator {
public:
    PageAllocator() = default;

    /// Initialize a fresh allocator over `file` with `n_pages` total pages,
    /// starting from an empty state. Pages 0..2 are reserved (superblock +
    /// shadow). The bitmap starts at `bitmap_page` and spans `bitmap_pages`.
    /// All pages after the bitmap are free.
    void init(PageFile& file, PageId bitmap_page, uint32_t bitmap_pages);

    /// Load allocator state from an existing file. Reads the superblock fields
    /// (`n_pages`, `free_list_head`, `n_free_pages`) and loads the bitmap into
    /// memory.
    void load(PageFile& file, PageId bitmap_page, uint32_t bitmap_pages,
              uint64_t n_pages, PageId free_list_head, uint32_t n_free_pages);

    /// Allocate a single page. Returns kInvalidPage if the file is full
    /// (caller should extend via `grow`).
    PageId alloc_page(PageFile& file);

    /// Allocate `count` contiguous pages (an extent). Returns the starting
    /// PageId, or kInvalidPage if no contiguous range is available.
    PageId alloc_extent(PageFile& file, uint32_t count);

    /// Free a single page (adds it to the free list).
    void free_page(PageFile& file, PageId page);

    /// Free a contiguous extent of `count` pages.
    void free_extent(PageFile& file, PageId start, uint32_t count);

    /// Extend the file by `extra_pages` pages and mark them free. Called when
    /// alloc_page/alloc_extent return kInvalidPage. Updates the bitmap to
    /// cover the new pages.
    void grow(PageFile& file, uint64_t extra_pages);

    /// Persist the bitmap to disk (writes the bitmap pages). The free-list
    /// head / n_free_pages / n_pages are read by the caller for the superblock.
    void flush_bitmap(PageFile& file) const;

    // --- Accessors for superblock serialization ---

    uint64_t n_pages() const { return n_pages_; }
    PageId free_list_head() const { return free_list_head_; }
    uint32_t n_free_pages() const { return n_free_pages_; }
    PageId bitmap_page() const { return bitmap_page_; }
    uint32_t bitmap_pages() const { return bitmap_pages_; }

private:
    /// Grow the in-memory bitmap to cover `n_pages` total pages.
    void grow_bitmap(uint64_t n_pages);

    /// Mark `count` pages starting at `start` as allocated (bit = 1) in both
    /// the in-memory bitmap and on disk.
    void mark_allocated(PageFile& file, PageId start, uint32_t count);

    /// Mark `count` pages starting at `start` as free (bit = 0).
    void mark_free(PageFile& file, PageId start, uint32_t count);

    /// Read the free-list "next" pointer stored in a free page (first 4 bytes).
    PageId read_free_next(const PageFile& file, PageId page) const;

    /// Write the free-list "next" pointer into a free page.
    void write_free_next(PageFile& file, PageId page, PageId next) const;

    // --- State (mirrors superblock fields) ---

    PageId bitmap_page_ = kInvalidPage;       ///< bitmap start page in file
    uint32_t bitmap_pages_ = 0;               ///< number of bitmap pages
    uint64_t n_pages_ = 0;                    ///< total pages in file
    PageId free_list_head_ = kInvalidPage;    ///< first free page (or nil)
    uint32_t n_free_pages_ = 0;               ///< free list length

    /// In-memory bitmap. One bit per page. Bit set = allocated.
    /// Grows as the file grows. Always covers [0, n_pages_).
    std::vector<uint8_t> bitmap_;

    /// Number of bits currently representable in the bitmap vector.
    /// (bitmap_.size() * 8). May exceed n_pages_ (trailing bits are zero).
    uint64_t bitmap_capacity_bits_ = 0;
};

}  // namespace sextant::tree
