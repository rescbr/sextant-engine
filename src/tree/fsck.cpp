#include "fsck.hpp"

#include "page_allocator.hpp"
#include "superblock.hpp"
#include "tree_manifest.hpp"
#include "tree_nodes.hpp"
#include "plane.hpp"

#include <sextant/error.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace sextant::tree {

namespace {

/// A page range claimed by the tree walk or auxiliary metadata.
struct PageRange {
    PageId start;
    uint32_t count;
};

/// Collect all allocated pages by walking the tree from root.
struct TreeWalkResult {
    std::vector<PageRange> node_ranges;     // internal nodes (incl. root)
    std::vector<PageRange> leaf_ranges;     // leaves
    std::vector<PageRange> payload_ranges;  // leaf payload extents
    uint64_t n_nodes = 0;
    uint64_t n_leaves = 0;
    uint64_t total_leaf_count = 0;
    uint64_t total_leaf_blocks = 0;  // Σ ceil(count/32) — plane sizing
    uint64_t plane_v2_leaves = 0;    // leaves with in-extent plane rows
    uint64_t plane_offset_bad = 0;   // plane_offset beyond extent bounds
    uint32_t max_depth = 0;
    uint64_t magic_failures = 0;
    uint64_t crc_failures = 0;
    bool pointers_ok = true;  // false if any child/leaf page is out of bounds
};

/// Read a page range into a buffer sized for npages pages.
std::vector<uint8_t> read_extent(const PageFile& file, PageId page,
                                 uint32_t npages) {
    std::vector<uint8_t> buf(static_cast<size_t>(npages) * kPageSize);
    file.read_pages(page, npages, buf.data());
    return buf;
}

/// Validate a leaf header's magic + CRC and accumulate its payload extent.
/// `physical_page` / `physical_pages` are the resolved page location (from
/// the leaf table).
void validate_leaf(const PageFile& file, PageId physical_page,
                   uint32_t physical_pages, TreeWalkResult& result) {
    result.leaf_ranges.push_back({physical_page, physical_pages});
    ++result.n_leaves;

    if (physical_page == kInvalidPage || physical_pages == 0) {
        ++result.magic_failures;
        result.pointers_ok = false;
        return;
    }

    auto lbuf = read_extent(file, physical_page, physical_pages);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(lbuf.data());

    if (lh->magic != kTreeLeafMagic) {
        ++result.magic_failures;
        return;
    }

    const uint32_t expected_crc =
        header_crc(lbuf.data(), offsetof(TreeLeafHeader, header_crc));
    if (lh->header_crc != expected_crc) {
        ++result.crc_failures;
        // CRC failure is detect-only: keep reading fields if plausible.
    }

    result.total_leaf_count += lh->count;
    result.total_leaf_blocks += (lh->count + 31) / 32;

    // Plane v2: the routing rows are a suffix of this extent. Bound-check
    // here (encoding-independent); the plane section cross-checks the
    // exact size once the block geometry is known.
    if (lh->plane_offset != 0) {
        ++result.plane_v2_leaves;
        if (lh->plane_offset >=
            static_cast<uint64_t>(physical_pages) * kPageSize) {
            ++result.plane_offset_bad;
        }
    }

    if (lh->payload_extent_page != kInvalidPage &&
        lh->payload_extent_pages > 0) {
        result.payload_ranges.push_back(
            {lh->payload_extent_page, lh->payload_extent_pages});
    }
}

/// DFS from root, collecting all allocated page ranges.
TreeWalkResult walk_tree(const PageFile& file, const Superblock& sb,
                         const TreeManifest& manifest) {
    TreeWalkResult result;

    // Load the leaf extent table for leaf_id → page resolution. Mandatory.
    std::vector<LeafTableEntry> leaf_table;
    if (sb.leaf_table_page() == kInvalidPage || sb.leaf_table_pages() == 0) {
        throw Error(ErrorCode::CorruptIndex,
            "superblock: missing leaf extent table");
    }
    {
        std::vector<uint8_t> blob(
            static_cast<size_t>(sb.leaf_table_pages()) * kPageSize);
        file.read_pages(sb.leaf_table_page(), sb.leaf_table_pages(),
                        blob.data());
        uint64_t n_entries = 0;
        std::memcpy(&n_entries, blob.data(), 8);
        leaf_table.resize(n_entries);
        std::memcpy(leaf_table.data(), blob.data() + 8,
                    n_entries * sizeof(LeafTableEntry));
    }

    struct StackEntry {
        PageId page;
        uint32_t pages;
        uint32_t depth;
    };
    std::vector<StackEntry> stack;
    stack.push_back({sb.root_node_page(), sb.root_node_pages(), 0});

    const uint32_t cesize =
        child_entry_size(manifest.dim, manifest.summary_size);
    const uint64_t npages_total = file.num_pages();

    while (!stack.empty()) {
        auto [page, npages, depth] = stack.back();
        stack.pop_back();

        if (page == kInvalidPage) continue;

        // Bounds check.
        if (page >= npages_total || page + npages > npages_total) {
            ++result.magic_failures;
            result.pointers_ok = false;
            continue;
        }

        result.node_ranges.push_back({page, npages});
        result.max_depth = std::max(result.max_depth, depth);

        auto buf = read_extent(file, page, npages);
        const auto* nh = reinterpret_cast<const TreeNodeHeader*>(buf.data());

        if (nh->magic != kTreeNodeMagic) {
            ++result.magic_failures;
            continue;  // can't walk further
        }

        const uint32_t expected_crc =
            header_crc(buf.data(), offsetof(TreeNodeHeader, header_crc));
        if (nh->header_crc != expected_crc) {
            ++result.crc_failures;
            // CRC failure is detect-only: continue walking.
        }

        const uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
        for (uint64_t j = 0; j < nh->n_children; ++j) {
            const auto* ce = reinterpret_cast<const ChildEntry*>(p);
            if (ce->child_page == kInvalidPage) {
                p += cesize;
                continue;
            }

            if (ce->is_leaf) {
                // Resolve leaf_id → physical page via leaf table.
                PageId leaf_id = ce->child_page;
                PageId lpage = leaf_id;
                uint32_t lpages = static_cast<uint32_t>(ce->child_pages);
                if (!leaf_table.empty() && leaf_id < leaf_table.size()) {
                    lpage = leaf_table[leaf_id].page;
                    lpages = leaf_table[leaf_id].pages;
                }
                validate_leaf(file, lpage, lpages, result);
            } else {
                stack.push_back(
                    {ce->child_page, static_cast<uint32_t>(ce->child_pages),
                     depth + 1});
                ++result.n_nodes;
            }
            p += cesize;
        }
    }

    return result;
}

/// Write a free-list "next" pointer into a free page (first 8 bytes),
/// matching PageAllocator::write_free_next's format.
void write_free_next(PageFile& file, PageId page, PageId next) {
    uint8_t buf[kPageSize];
    std::memset(buf, 0, kPageSize);
    std::memcpy(buf, &next, sizeof(next));
    file.write_page(page, buf);
}

}  // namespace

bool FsckResult::all_ok() const {
    return superblock_ok && tree_walk_ok && page_accounting_ok &&
           leaf_internals_ok && payload_ok && referential_ok;
}

std::string FsckResult::summary() const {
    std::string s;
    auto add = [&](const auto& arg) { s += std::string(arg); };
    add("fsck summary:\n");
    add("  Level 0 (superblock):  "); add(superblock_ok ? "OK" : "FAIL"); add("\n");
    add("    commit_seq="); add(std::to_string(commit_seq)); add("\n");
    add("    root_page="); add(std::to_string(root_page)); add("\n");
    add("  Level 1 (tree walk):   "); add(tree_walk_ok ? "OK" : "FAIL"); add("\n");
    add("    n_nodes="); add(std::to_string(n_nodes)); add("\n");
    add("    n_leaves="); add(std::to_string(n_leaves)); add("\n");
    add("    total_leaf_count="); add(std::to_string(total_leaf_count)); add("\n");
    add("    max_depth="); add(std::to_string(max_depth_reached)); add("\n");
    add("  Level 2 (accounting):  "); add(page_accounting_ok ? "OK" : "FAIL"); add("\n");
    add("    total_pages="); add(std::to_string(total_pages)); add("\n");
    add("    total_allocated="); add(std::to_string(total_allocated)); add("\n");
    add("    total_free="); add(std::to_string(total_free)); add("\n");
    add("    leaked_pages="); add(std::to_string(leaked_pages)); add("\n");
    add("    orphan_pages="); add(std::to_string(orphan_pages)); add("\n");
    add("  Level 3 (leaf internals): "); add(leaf_internals_ok ? "OK" : "FAIL"); add("\n");
    add("    magic_failures="); add(std::to_string(magic_failures)); add("\n");
    add("    crc_failures="); add(std::to_string(crc_failures)); add("\n");
    add("  Level 4 (payload):     "); add(payload_ok ? "OK" : "FAIL"); add("\n");
    add("    payload_extents="); add(std::to_string(payload_extents)); add("\n");
    add("  Level 5 (referential): "); add(referential_ok ? "OK" : "FAIL"); add("\n");
    add("    duplicate_row_ids="); add(std::to_string(duplicate_row_ids)); add("\n");
    return s;
}

FsckResult fsck(const std::string& path, bool repair) {
    FsckResult result;

    PageFile file(path);

    // -----------------------------------------------------------------------
    // Level 0: Superblock.
    // -----------------------------------------------------------------------
    Superblock sb;
    try {
        sb.load(file);
        result.commit_seq = sb.commit_seq();
        result.root_page = sb.root_node_page();
        result.superblock_ok =
            (sb.commit_seq() > 0) &&
            (result.root_page == kInvalidPage ||
             (result.root_page < file.num_pages()));
    } catch (const std::exception& e) {
        spdlog::warn("fsck: superblock load failed: {}", e.what());
        result.superblock_ok = false;
        return result;  // Can't continue without superblock.
    }

    // -----------------------------------------------------------------------
    // Load manifest (needed for child_entry_size).
    // -----------------------------------------------------------------------
    TreeManifest manifest;
    if (sb.config_page() != kInvalidPage && sb.config_pages() > 0) {
        std::string cfg_buf(static_cast<size_t>(sb.config_pages()) * kPageSize,
                            '\0');
        file.read_pages(sb.config_page(), sb.config_pages(), cfg_buf.data());
        // Trim null padding.
        auto nul = cfg_buf.find('\0');
        if (nul != std::string::npos) cfg_buf.resize(nul);
        try {
            manifest = manifest_from_toml(cfg_buf);
        } catch (const std::exception& e) {
            spdlog::warn("fsck: manifest parse failed: {}", e.what());
        }
    }

    // -----------------------------------------------------------------------
    // Level 1 + 3: Tree walk (collects magic/crc failures too).
    // -----------------------------------------------------------------------
    auto walk = walk_tree(file, sb, manifest);
    result.n_nodes = walk.n_nodes;
    result.n_leaves = walk.n_leaves;
    result.total_leaf_count = walk.total_leaf_count;
    result.max_depth_reached = walk.max_depth;
    result.magic_failures = walk.magic_failures;
    result.crc_failures = walk.crc_failures;
    result.tree_walk_ok = (walk.magic_failures == 0 && walk.pointers_ok);
    result.leaf_internals_ok =
        (walk.magic_failures == 0 && walk.crc_failures == 0);

    // -----------------------------------------------------------------------
    // Level 4: Payload (simplified — count extents, no orphan detection).
    // -----------------------------------------------------------------------
    result.payload_extents = walk.payload_ranges.size();
    result.payload_ok = true;

    // Level 5 deferred (detect-only, expensive).
    result.referential_ok = true;

    // -----------------------------------------------------------------------
    // Level 2: Page accounting.
    // -----------------------------------------------------------------------
    std::vector<PageRange> all_allocated;
    all_allocated.insert(all_allocated.end(), walk.node_ranges.begin(),
                         walk.node_ranges.end());
    all_allocated.insert(all_allocated.end(), walk.leaf_ranges.begin(),
                         walk.leaf_ranges.end());
    all_allocated.insert(all_allocated.end(), walk.payload_ranges.begin(),
                         walk.payload_ranges.end());

    // Auxiliary / structural pages.
    all_allocated.push_back({0, 2});  // superblock + shadow
    if (sb.alloc_bitmap_page() != kInvalidPage)
        all_allocated.push_back(
            {sb.alloc_bitmap_page(), sb.alloc_bitmap_pages()});
    if (sb.codebook_page() != kInvalidPage)
        all_allocated.push_back({sb.codebook_page(), sb.codebook_pages()});
    if (sb.config_page() != kInvalidPage)
        all_allocated.push_back({sb.config_page(), sb.config_pages()});
    if (sb.pca_page() != kInvalidPage)
        all_allocated.push_back({sb.pca_page(), sb.pca_pages()});
    if (sb.cardinality_page() != kInvalidPage)
        all_allocated.push_back(
            {sb.cardinality_page(), sb.cardinality_pages()});
    // Routing plane: register the extent and cross-check its sizing
    // against the walk (blocks = Σ ceil(count/32), alpha = 2×count).
    // v2 layout: the extent is header-only; the blocks live in the leaf
    // extents (every live leaf must carry a plane_offset, in bounds, and
    // large enough for its blocks + pv alphas).
    if (sb.plane_page() != kInvalidPage && sb.plane_pages() > 0) {
        all_allocated.push_back({sb.plane_page(), sb.plane_pages()});
        auto blob = read_extent(file, sb.plane_page(),
                                std::min<uint32_t>(sb.plane_pages(), 1));
        PlaneMeta pm;
        uint64_t blocks_bytes = 0, cb_bytes = 0, alpha_bytes = 0;
        uint8_t pflags = 0;
        if (parse_plane_header(blob.data(), blob.size(), &pm, &blocks_bytes,
                               &cb_bytes, &alpha_bytes, &pflags)) {
            const uint64_t bb = (pm.encoding == PlaneEncoding::B1G)
                ? static_cast<uint64_t>(pm.rank) * 4
                : static_cast<uint64_t>(pm.rank) * 16;
            if (pflags & 2) {
                if (walk.plane_v2_leaves != walk.n_leaves ||
                    walk.plane_offset_bad > 0)
                    ++result.magic_failures;
            } else {
                const bool blocks_ok =
                    blocks_bytes == walk.total_leaf_blocks * bb;
                const bool alpha_ok =
                    pm.encoding != PlaneEncoding::U4LM_PV ||
                    alpha_bytes == walk.total_leaf_count * 2;
                if (!blocks_ok || !alpha_ok) ++result.magic_failures;
            }
        } else {
            ++result.magic_failures;
        }
    }
    all_allocated.push_back(
        {sb.leaf_table_page(), sb.leaf_table_pages()});

    result.total_pages = file.num_pages();
    std::vector<bool> allocated(file.num_pages(), false);
    for (const auto& r : all_allocated) {
        for (uint32_t i = 0; i < r.count; ++i) {
            PageId p = r.start + i;
            if (p < allocated.size()) {
                allocated[p] = true;
            }
        }
    }

    // Read bitmap directly and cross-reference.
    std::vector<uint8_t> bitmap;
    if (sb.alloc_bitmap_page() != kInvalidPage && sb.alloc_bitmap_pages() > 0) {
        bitmap.assign(static_cast<size_t>(sb.alloc_bitmap_pages()) * kPageSize,
                      0);
        file.read_pages(sb.alloc_bitmap_page(), sb.alloc_bitmap_pages(),
                        bitmap.data());
    }
    auto is_bit_set = [&](PageId p) -> bool {
        if (bitmap.empty()) return false;
        const uint64_t byte = p / 8;
        if (byte >= bitmap.size()) return false;
        return (bitmap[byte] >> (p % 8)) & 1u;
    };

    result.total_allocated = 0;
    result.total_free = 0;
    result.leaked_pages = 0;
    result.orphan_pages = 0;
    for (uint64_t p = 0; p < file.num_pages(); ++p) {
        const bool in_bitmap = is_bit_set(p);
        const bool in_tree = (p < allocated.size() && allocated[p]);
        if (in_tree) ++result.total_allocated;
        if (in_bitmap && !in_tree) ++result.leaked_pages;
        if (in_tree && !in_bitmap) ++result.orphan_pages;
        if (!in_bitmap && !in_tree) ++result.total_free;
    }

    result.page_accounting_ok =
        (result.leaked_pages == 0 && result.orphan_pages == 0);

    // -----------------------------------------------------------------------
    // Level 2 repair: rebuild bitmap + free list from the tree walk.
    // -----------------------------------------------------------------------
    if (repair && !result.page_accounting_ok &&
        sb.alloc_bitmap_page() != kInvalidPage &&
        sb.alloc_bitmap_pages() > 0) {
        // The old bitmap extent may be under-provisioned (pre-fix trees
        // cover only a prefix of the file — writing the rebuilt bits
        // back there silently DROPS every page past its coverage, the
        // exact F3/F4 pattern). When it can't hold the full file,
        // RELOCATE: new larger extent at EOF, old pages freed.
        const uint64_t kBitsPerBitmapPage =
            static_cast<uint64_t>(kPageSize) * 8;
        const uint64_t bits_needed = file.num_pages();
        PageId bmp = sb.alloc_bitmap_page();
        uint32_t nbm = sb.alloc_bitmap_pages();
        const PageId old_bmp = bmp;
        bool relocated = false;
        if (static_cast<uint64_t>(nbm) * kPageSize <
            (bits_needed + 7) / 8) {
            nbm = static_cast<uint32_t>(bits_needed / kBitsPerBitmapPage + 2);
            bmp = file.num_pages();
            file.truncate(bmp + nbm);
            relocated = true;
        }
        // Rebuild bitmap: clear, then set every allocated page.
        std::vector<uint8_t> rebuilt(static_cast<size_t>(nbm) * kPageSize, 0);
        std::vector<bool> alloc2(file.num_pages(), false);
        for (const auto& r : all_allocated) {
            if (relocated && r.start == old_bmp) continue;  // old extent freed
            for (uint32_t i = 0; i < r.count; ++i) {
                PageId p = r.start + i;
                if (p / 8 < rebuilt.size()) {
                    rebuilt[p / 8] |= static_cast<uint8_t>(1u << (p % 8));
                    alloc2[p] = true;
                }
            }
        }
        if (relocated) {
            for (uint64_t p = bmp; p < bmp + nbm; ++p) {
                rebuilt[p / 8] |= static_cast<uint8_t>(1u << (p % 8));
                alloc2[p] = true;
            }
        }
        file.write_pages(bmp, nbm, rebuilt.data());
        sb.set_bitmap(bmp, nbm);
        sb.set_n_pages(file.num_pages());

        // Rebuild free list: every page not in the allocated set, chained LIFO
        // (matches PageAllocator::free_page ordering).
        PageId prev_free = kInvalidPage;
        result.total_free = 0;
        for (uint64_t p = file.num_pages(); p-- > 0;) {
            if (p < alloc2.size() && alloc2[p]) continue;
            // Free page — write next pointer, push to head.
            write_free_next(file, p, prev_free);
            prev_free = p;
            ++result.total_free;
        }

        // Update superblock + commit.
        sb.set_free_list(prev_free, result.total_free);
        sb.commit(file);
        file.sync();

        result.leaked_pages = 0;
        result.orphan_pages = 0;
        result.page_accounting_ok = true;
        spdlog::info("fsck: repaired bitmap + free list ({} free pages)",
                     result.total_free);
    }

    return result;
}

}  // namespace sextant::tree
