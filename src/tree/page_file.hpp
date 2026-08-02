#pragma once

/// @file page_file.hpp
/// Buffered, page-aligned file I/O for the hierarchical IVF tree.
///
/// The tree uses a single file with 4KB page granularity. Writes are buffered
/// (kernel page cache) with explicit `fdatasync` at commit points — the design
/// doc measured O_DIRECT as 5.2× slower for the tree's sequential write pattern.
/// Reads at search time go through mmap (Phase 1); build/vacuum use pread/pwrite.
///
/// All offsets are page indices (not byte offsets). The caller never deals
/// with raw byte offsets — everything is page-granular.

#include <cstdint>
#include <cstddef>
#include <string>

namespace sextant::tree {

/// Page size in bytes. 4KB — aligned to NVMe LBA, OS page, and mmap granularity.
inline constexpr uint32_t kPageSize = 4096;

/// Page index type. The tree file is a linear array of pages.
/// uint64_t covers 2^64 × 4KB = effectively unlimited. Previous uint32_t
/// limited to 16TB — too tight for 10B+ workloads with fragmentation +
/// filter metadata.
using PageId = uint64_t;

/// Sentinel for "no page" (invalid / nil).
inline constexpr PageId kInvalidPage = 0xFFFFFFFFFFFFFFFFull;

/// Buffered, page-aligned file for the IVF tree.
///
/// One FD. All I/O is at page granularity (multiples of kPageSize). The file
/// grows via `ftruncate` — pages beyond the current size are zero-filled.
class PageFile {
public:
    PageFile() = default;

    /// Open `path` for read/write (create if it doesn't exist).
    /// Throws on failure.
    explicit PageFile(const std::string& path);
    ~PageFile();

    PageFile(const PageFile&) = delete;
    PageFile& operator=(const PageFile&) = delete;
    PageFile(PageFile&&) noexcept;
    PageFile& operator=(PageFile&&) noexcept;

    /// Read `n_pages` consecutive pages starting at `page`. `buf` must be at
    /// least `n_pages * kPageSize` bytes. Throws on short read or error.
    void read_pages(PageId page, uint32_t n_pages, void* buf) const;

    /// Write `n_pages` consecutive pages starting at `page`. `buf` must be at
    /// least `n_pages * kPageSize` bytes. Throws on error.
    void write_pages(PageId page, uint32_t n_pages, const void* buf);

    /// Read a single page.
    void read_page(PageId page, void* buf) const {
        read_pages(page, 1, buf);
    }

    /// Write a single page.
    void write_page(PageId page, const void* buf) {
        write_pages(page, 1, buf);
    }

    /// Flush dirty pages to disk (fdatasync). Throws on error.
    void sync();

    /// Truncate/extend the file to exactly `n_pages` pages. New pages are
    /// zero-filled by the OS. Throws on error.
    void truncate(uint64_t n_pages);

    /// Current file size in pages (file_size_bytes / kPageSize, rounded up).
    uint64_t num_pages() const;

    /// File descriptor (for mmap in Phase 1).
    int fd() const { return fd_; }

    /// True if the file is open.
    bool is_open() const { return fd_ >= 0; }

    /// Path passed to the constructor.
    const std::string& path() const { return path_; }

private:
    int fd_ = -1;
    std::string path_;
};

}  // namespace sextant::tree
