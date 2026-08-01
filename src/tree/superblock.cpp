#include "superblock.hpp"

#include "sextant/error.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>

namespace sextant::tree {

void Superblock::init_fresh(PageId bitmap_page, uint32_t bitmap_pages) {
    std::memset(&disk_, 0, sizeof(disk_));
    disk_.magic = kSuperblockMagic;
    disk_.commit_seq = 0;
    disk_.root_node_page = kInvalidPage;
    disk_.root_node_pages = 0;
    disk_.depth = 0;
    disk_.n_leaves = 0;
    disk_.alloc_bitmap_page = bitmap_page;
    disk_.alloc_bitmap_pages = bitmap_pages;
    disk_.free_list_head = kInvalidPage;
    disk_.n_free_pages = 0;
    disk_.codebook_page = kInvalidPage;
    disk_.codebook_pages = 0;
    disk_.config_page = kInvalidPage;
    disk_.config_pages = 0;
    disk_.n_pages = bitmap_page + bitmap_pages;  // file covers superblocks + bitmap
}

bool Superblock::read_copy(const PageFile& file, PageId page,
                           SuperblockDisk& out) {
    file.read_page(page, &out);
    if (out.magic != kSuperblockMagic) {
        return false;
    }
    return true;
}

void Superblock::write_copy(PageFile& file, PageId page,
                            const SuperblockDisk& sb) {
    file.write_page(page, &sb);
}

void Superblock::load(const PageFile& file) {
    SuperblockDisk active{}, shadow{};
    bool active_ok = read_copy(file, kSuperblockPage, active);
    bool shadow_ok = read_copy(file, kSuperblockShadowPage, shadow);

    if (!active_ok && !shadow_ok) {
        throw Error(ErrorCode::CorruptIndex,
                    "Superblock::load: neither copy has valid magic on '" +
                        file.path() + "'");
    }

    // Pick the copy with the higher commit_seq. If only one is valid, use it.
    if (active_ok && shadow_ok) {
        disk_ = (active.commit_seq >= shadow.commit_seq) ? active : shadow;
    } else if (active_ok) {
        disk_ = active;
    } else {
        disk_ = shadow;
    }

    spdlog::debug("Superblock: loaded commit_seq={}, root=page {} ({} pages), "
                  "depth={}, n_leaves={}, n_pages={}",
                  disk_.commit_seq, disk_.root_node_page, disk_.root_node_pages,
                  disk_.depth, disk_.n_leaves, disk_.n_pages);
}

void Superblock::commit(PageFile& file) {
    // Increment the commit sequence.
    ++disk_.commit_seq;

    // Determine which copy is currently the shadow (the one NOT active).
    // The active copy is the one with the higher commit_seq (or page 0 on
    // the very first commit). We alternate: even commit_seq → page 0,
    // odd → page 1. This ensures the shadow is always the opposite of the
    // last-committed copy.
    const PageId shadow_page =
        (disk_.commit_seq % 2 == 0) ? kSuperblockPage : kSuperblockShadowPage;

    // 1. Write to shadow.
    write_copy(file, shadow_page, disk_);

    // 2. fdatasync — the shadow must be on disk before we flip.
    file.sync();

    spdlog::trace("Superblock: committed seq {} to page {}",
                  disk_.commit_seq, shadow_page);
}

}  // namespace sextant::tree
