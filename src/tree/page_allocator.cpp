#include "page_allocator.hpp"

#include "sextant/error.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>

namespace sextant::tree {

namespace {

// Bit manipulation helpers for the allocation bitmap.
// Bit set = page allocated. Bit clear = page free.

inline bool bit_get(const std::vector<uint8_t>& bm, uint64_t bit) {
    return (bm[bit / 8] >> (bit % 8)) & 1u;
}

inline void bit_set(std::vector<uint8_t>& bm, uint64_t bit) {
    bm[bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
}

inline void bit_clear(std::vector<uint8_t>& bm, uint64_t bit) {
    bm[bit / 8] &= static_cast<uint8_t>(~(1u << (bit % 8)));
}

/// Find the first run of `count` consecutive zero bits in `bm`, starting from
/// bit `from`. Returns the bit index of the run start, or UINT64_MAX if none.
uint64_t find_free_run(const std::vector<uint8_t>& bm, uint64_t from,
                       uint64_t n_bits, uint32_t count) {
    uint64_t run_start = from;
    uint32_t run_len = 0;
    for (uint64_t i = from; i < n_bits; ++i) {
        if (bit_get(bm, i)) {
            run_start = i + 1;
            run_len = 0;
        } else {
            ++run_len;
            if (run_len >= count) {
                return run_start;
            }
        }
    }
    return UINT64_MAX;
}

}  // namespace

void PageAllocator::init(PageFile& file, PageId bitmap_page,
                         uint32_t bitmap_pages) {
    bitmap_page_ = bitmap_page;
    bitmap_pages_ = bitmap_pages;

    // The file starts with: page 0 (superblock), page 1 (shadow), page 2+
    // (bitmap). Everything after the bitmap is allocatable free space.
    const PageId first_free = bitmap_page + bitmap_pages;

    n_pages_ = first_free;  // file has no free pages yet
    free_list_head_ = kInvalidPage;
    n_free_pages_ = 0;

    // Grow the in-memory bitmap to cover the initial reserved pages.
    grow_bitmap(n_pages_);

    // Mark the reserved pages as allocated (superblock + shadow + bitmap).
    // These are never free — set bits directly without touching the free
    // counter (mark_allocated would decrement n_free_pages_, which is 0 here).
    for (uint32_t i = 0; i < first_free; ++i) {
        bit_set(bitmap_, i);
    }

    // The file may already be larger (if pre-truncated). Sync n_pages_.
    n_pages_ = std::max(n_pages_, file.num_pages());
    grow_bitmap(n_pages_);
}

void PageAllocator::load(PageFile& file, PageId bitmap_page,
                         uint32_t bitmap_pages, uint64_t n_pages,
                         PageId free_list_head, uint64_t n_free_pages) {
    bitmap_page_ = bitmap_page;
    bitmap_pages_ = bitmap_pages;
    n_pages_ = n_pages;
    free_list_head_ = free_list_head;
    n_free_pages_ = n_free_pages;

    // Loud-fail when the file outgrew its bitmap coverage (spec §4.1 rule 3:
    // open/fsck must fail on region/file-size mismatch, never silently
    // treat uncovered pages as free). This is the v1 failure mode: an
    // under-provisioned region makes every page past coverage look FREE on
    // disk, so insert/delete would hand out live leaf pages.
    if (bitmap_pages_ > 0) {
        const uint64_t covered_pages =
            static_cast<uint64_t>(bitmap_pages_) * kPageSize * 8;
        if (n_pages_ > covered_pages) {
            throw Error(ErrorCode::CorruptIndex,
                "PageAllocator::load: file has " +
                    std::to_string(n_pages_) +
                    " pages but the bitmap region (start " +
                    std::to_string(bitmap_page_) + ", " +
                    std::to_string(bitmap_pages_) +
                    " pages) only addresses " + std::to_string(covered_pages) +
                    " — file outgrew its bitmap (under-provisioned at "
                    "format time? rebuild with a larger --bitmap-pages)");
        }
    }

    // Load the bitmap from disk into memory.
    grow_bitmap(n_pages_);
    // The on-disk bitmap occupies bitmap_pages_ × kPageSize bytes, but
    // grow_bitmap sizes to just n_pages_ bits. Grow to hold the full on-disk
    // copy so read_pages doesn't overflow.
    const size_t on_disk_bytes =
        static_cast<size_t>(bitmap_pages_) * kPageSize;
    if (bitmap_.size() < on_disk_bytes)
        bitmap_.resize(on_disk_bytes, 0);
    bitmap_capacity_bits_ = static_cast<uint64_t>(bitmap_.size()) * 8;
    if (bitmap_pages_ > 0) {
        file.read_pages(bitmap_page_, bitmap_pages_, bitmap_.data());
    }
    spdlog::debug("PageAllocator: loaded n_pages={}, free={}, bitmap={} pages",
                  n_pages_, n_free_pages_, bitmap_pages_);
}

void PageAllocator::grow_bitmap(uint64_t n_pages) {
    const uint64_t needed_bytes = (n_pages + 7) / 8;
    if (bitmap_.size() < needed_bytes) {
        bitmap_.resize(needed_bytes, 0);
    }
    bitmap_capacity_bits_ = static_cast<uint64_t>(bitmap_.size()) * 8;
}

PageId PageAllocator::alloc_page(PageFile& file) {
    // 1. Try the free list (explicitly freed pages). The list is a HINT;
    //    the bitmap is the source of truth. alloc_extent(count>1) can
    //    reallocate free-listed pages via a bitmap run scan WITHOUT
    //    unlinking them; such pages get rewritten with leaf/node data,
    //    so their stored "next" is data, not a pointer (measured: depth-3
    //    split cascades crashed reading a garbage page id from a stale
    //    entry). On any stale entry, discard the whole chain and rebuild
    //    from the bitmap — correct, and the list repopulates on frees.
    while (free_list_head_ != kInvalidPage) {
        const PageId page = free_list_head_;
        if (bit_get(bitmap_, page)) {
            // Stale entry: page is allocated and may have been rewritten.
            clear_free_list();
            break;
        }
        free_list_head_ = read_free_next(file, page);
        --n_free_pages_;
        bit_set(bitmap_, page);
        return page;
    }

    // 2. No free list entries. Scan the bitmap for the first zero bit.
    const uint64_t found = find_free_run(bitmap_, 0, n_pages_, 1);
    if (found != UINT64_MAX) {
        bit_set(bitmap_, found);
        --n_free_pages_;
        return static_cast<PageId>(found);
    }

    // 3. Bitmap is full. Grow the file by one page.
    grow(file, 1);
    // The new page is at index n_pages_ - 1 (just added, bit = 0).
    const PageId page = static_cast<PageId>(n_pages_ - 1);
    bit_set(bitmap_, page);
    --n_free_pages_;
    return page;
}

PageId PageAllocator::alloc_extent(PageFile& file, uint32_t count) {
    if (count == 0) return kInvalidPage;
    if (count == 1) return alloc_page(file);

    // Search the bitmap for `count` consecutive free pages.
    const uint64_t found = find_free_run(bitmap_, 0, n_pages_, count);

    if (found != UINT64_MAX) {
        mark_allocated(file, static_cast<PageId>(found), count);
        // Note: the free list may contain stale entries pointing to pages now
        // covered by this extent. alloc_page validates each popped entry
        // against the bitmap (the source of truth) and skips stale ones.
        return static_cast<PageId>(found);
    }

    // No contiguous run found. Grow the file to make room.
    grow(file, count);

    // After grow, the new pages are at the end — contiguous by construction.
    const uint64_t new_found = find_free_run(bitmap_, 0, n_pages_, count);
    if (new_found == UINT64_MAX) {
        // Should not happen after grow.
        return kInvalidPage;
    }
    mark_allocated(file, static_cast<PageId>(new_found), count);
    return static_cast<PageId>(new_found);
}

void PageAllocator::free_page(PageFile& file, PageId page) {
    if (page == kInvalidPage) return;
    if (!bit_get(bitmap_, page)) {
        spdlog::warn("PageAllocator::free_page: page {} is already free", page);
        return;
    }
    bit_clear(bitmap_, page);
    // Push onto the free list head.
    write_free_next(file, page, free_list_head_);
    free_list_head_ = page;
    ++n_free_pages_;
}

void PageAllocator::free_extent(PageFile& file, PageId start, uint32_t count) {
    // Free each page in the extent. They go onto the free list individually.
    for (uint32_t i = 0; i < count; ++i) {
        const PageId page = start + i;
        if (bit_get(bitmap_, page)) {
            bit_clear(bitmap_, page);
            write_free_next(file, page, free_list_head_);
            free_list_head_ = page;
            ++n_free_pages_;
        }
    }
}

void PageAllocator::clear_free_list() {
    free_list_head_ = kInvalidPage;
    // Recount free pages from the bitmap (source of truth).
    n_free_pages_ = 0;
    for (uint64_t i = 0; i < n_pages_; ++i)
        if (!bit_get(bitmap_, i)) ++n_free_pages_;
}

void PageAllocator::grow(PageFile& file, uint64_t extra_pages) {
    const uint64_t new_n_pages = n_pages_ + extra_pages;

    grow_bitmap(new_n_pages);

    // Extend the file.
    n_pages_ = new_n_pages;
    file.truncate(n_pages_);

    // The new pages are free (bitmap bits = 0, already zero from grow_bitmap).
    // Track the count for the superblock's n_free_pages field.
    n_free_pages_ += extra_pages;
}

void PageAllocator::flush_bitmap(PageFile& file) const {
    if (bitmap_pages_ == 0) return;
    // Write the bitmap pages. The in-memory bitmap must fit the reserved
    // on-disk region — if it doesn't, the file outgrew the region sized at
    // format time and bits past the region would be silently dropped
    // (every such page would look FREE on disk). Fail loudly instead.
    const uint32_t bytes_to_write = bitmap_pages_ * kPageSize;
    if (bitmap_.size() > bytes_to_write) {
        const size_t uncovered_pages =
            (bitmap_.size() * 8 - static_cast<size_t>(bytes_to_write) * 8);
        throw Error(ErrorCode::CorruptIndex,
            "PageAllocator::flush_bitmap: file grew past the allocation "
            "bitmap region (" + std::to_string(bitmap_.size()) +
            " bytes in memory vs " + std::to_string(bytes_to_write) +
            " reserved; " + std::to_string(uncovered_pages) +
            " pages uncovered) — bitmap under-provisioned at format time");
    }
    if (bitmap_.size() < bytes_to_write) {
        // Pad with zeros.
        std::vector<uint8_t> padded(bytes_to_write, 0);
        std::memcpy(padded.data(), bitmap_.data(), bitmap_.size());
        file.write_pages(bitmap_page_, bitmap_pages_, padded.data());
    } else {
        file.write_pages(bitmap_page_, bitmap_pages_, bitmap_.data());
    }
}

void PageAllocator::mark_allocated(PageFile& file, PageId start,
                                   uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const PageId page = start + i;
        if (!bit_get(bitmap_, page)) {
            // Was free — adjust the counter.
            --n_free_pages_;
        }
        bit_set(bitmap_, page);
    }
}

void PageAllocator::mark_free(PageFile& file, PageId start, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const PageId page = start + i;
        if (bit_get(bitmap_, page)) {
            // Was allocated — adjust the counter.
            ++n_free_pages_;
        }
        bit_clear(bitmap_, page);
    }
}

PageId PageAllocator::read_free_next(const PageFile& file, PageId page) const {
    PageId next = kInvalidPage;
    uint8_t buf[kPageSize];
    file.read_page(page, buf);
    std::memcpy(&next, buf, sizeof(next));
    return next;
}

void PageAllocator::write_free_next(PageFile& file, PageId page,
                                    PageId next) const {
    // Read-modify-write: preserve the rest of the page (it may contain data
    // from a previously-freed allocation cycle, though free pages are
    // undefined). We zero the page and write the next pointer in the first
    // 4 bytes — free pages have no meaningful content.
    uint8_t buf[kPageSize];
    std::memset(buf, 0, kPageSize);
    std::memcpy(buf, &next, sizeof(next));
    file.write_page(page, buf);
}

}  // namespace sextant::tree
