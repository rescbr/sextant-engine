#pragma once

/// @file fsck.hpp
/// Filesystem check / repair for the IVF tree file.
///
/// The tree is the source of truth; the bitmap/free-list is a cache that can
/// be rebuilt. fsck validates the tree across five levels:
///
///   0. Superblock  — magic, commit_seq, root in bounds
///   1. Tree walk   — child pointers valid, magic/CRC ok, leaf counts
///   2. Accounting  — every page = tree ∪ aux ∪ free; no leaks/orphans
///   3. Leaf internals — magic, CRC, region sizes
///   4. Payload     — payload extents referenced once, in bounds
///   5. Referential — no duplicate row_ids (detect only)
///
/// With --repair, Level 2 fixes rebuild the bitmap/free-list from the tree
/// walk. Leaf contents are never modified.

#include "tree/page_file.hpp"

#include <cstdint>
#include <string>

namespace sextant::tree {

struct FsckResult {
    // Level 0: Superblock
    bool superblock_ok = false;
    uint64_t commit_seq = 0;
    PageId root_page = 0;

    // Level 1: Tree walk
    bool tree_walk_ok = false;
    uint64_t n_nodes = 0;        // internal nodes visited
    uint64_t n_leaves = 0;
    uint64_t total_leaf_count = 0;  // Σ leaf.count
    uint32_t max_depth_reached = 0;

    // Level 2: Page accounting
    bool page_accounting_ok = false;
    uint64_t total_allocated = 0;    // pages claimed by tree + aux
    uint64_t total_free = 0;         // pages in free set
    uint64_t total_pages = 0;        // file size in pages
    uint64_t leaked_pages = 0;       // allocated in bitmap but not in tree
    uint64_t orphan_pages = 0;       // in tree but not in bitmap

    // Level 3: Leaf internals
    bool leaf_internals_ok = false;
    uint64_t crc_failures = 0;
    uint64_t magic_failures = 0;

    // Level 4: Payload
    bool payload_ok = false;
    uint64_t payload_extents = 0;

    // Level 5: Referential
    bool referential_ok = false;
    uint64_t duplicate_row_ids = 0;

    // Overall
    bool all_ok() const;
    std::string summary() const;
};

/// Run fsck on a tree file. If `repair` is true, rebuild the bitmap/free-list
/// from the tree walk (Level 2 repair). Never modifies leaf contents.
FsckResult fsck(const std::string& path, bool repair = false);

}  // namespace sextant::tree
