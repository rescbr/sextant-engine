#include <gtest/gtest.h>

#include "tree/page_file.hpp"
#include "tree/page_allocator.hpp"
#include "tree/superblock.hpp"
#include "tree/tree_manifest.hpp"
#include "sextant/error.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace sextant::tree {
namespace {

// ---------------------------------------------------------------------------
// Helper: create a unique temp file path
// ---------------------------------------------------------------------------
std::string temp_path(const char* suffix) {
    auto tmpl = std::string("/tmp/sextant_tree_test_XXXXXX") + suffix;
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = ::mkstemp(buf.data());
    EXPECT_NE(fd, -1);
    ::close(fd);
    ::unlink(buf.data());  // remove the file; we just want a unique name
    return std::string(buf.data());
}

// ===========================================================================
// PageFile
// ===========================================================================

TEST(TreePageFile, WriteReadRoundTrip) {
    std::string path = temp_path(".tree");
    constexpr uint32_t kNPg = 4;

    {
        PageFile f(path);
        f.truncate(kNPg);
        EXPECT_EQ(f.num_pages(), kNPg);

        // Write distinct patterns into each page.
        std::vector<uint8_t> buf(kPageSize);
        for (uint32_t p = 0; p < kNPg; ++p) {
            std::memset(buf.data(), static_cast<int>(p + 1), kPageSize);
            f.write_page(p, buf.data());
        }
        f.sync();
    }
    {
        PageFile f(path);
        std::vector<uint8_t> buf(kPageSize);
        for (uint32_t p = 0; p < kNPg; ++p) {
            f.read_page(p, buf.data());
            EXPECT_EQ(buf[0], static_cast<uint8_t>(p + 1));
            EXPECT_EQ(buf[kPageSize - 1], static_cast<uint8_t>(p + 1));
        }
    }
    std::filesystem::remove(path);
}

TEST(TreePageFile, MultiPageExtent) {
    std::string path = temp_path(".tree");
    constexpr uint32_t kNPg = 8;

    PageFile f(path);
    f.truncate(kNPg);

    // Write 4 pages at once starting at page 2.
    std::vector<uint8_t> wbuf(kPageSize * 4);
    for (size_t i = 0; i < wbuf.size(); ++i) {
        wbuf[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
    }
    f.write_pages(2, 4, wbuf.data());

    // Read back.
    std::vector<uint8_t> rbuf(kPageSize * 4);
    f.read_pages(2, 4, rbuf.data());
    EXPECT_EQ(0, std::memcmp(wbuf.data(), rbuf.data(), wbuf.size()));

    std::filesystem::remove(path);
}

TEST(TreePageFile, TruncateGrowsAndShrinks) {
    std::string path = temp_path(".tree");
    PageFile f(path);
    f.truncate(10);
    EXPECT_EQ(f.num_pages(), 10);
    f.truncate(3);
    EXPECT_EQ(f.num_pages(), 3);
    std::filesystem::remove(path);
}

// ===========================================================================
// PageAllocator
// ===========================================================================

class TreeAllocatorTest : public ::testing::Test {
protected:
    std::string path = temp_path(".tree");

    // Layout: page 0 = superblock, page 1 = shadow, page 2 = bitmap (1 page).
    static constexpr PageId kBitmapPage = 2;
    static constexpr uint32_t kBitmapPages = 1;  // 1 page = 32768 bits

    PageFile file{path};
    PageAllocator alloc;

    void SetUp() override {
        // Start with a file that has pages 0-2 reserved.
        file.truncate(kBitmapPage + kBitmapPages);
        alloc.init(file, kBitmapPage, kBitmapPages);
    }

    void TearDown() override {
        std::filesystem::remove(path);
    }
};

TEST_F(TreeAllocatorTest, InitReservesPages) {
    // Pages 0, 1 (superblocks), 2 (bitmap) should be allocated.
    EXPECT_EQ(alloc.n_pages(), 3u);  // bitmap_page + bitmap_pages
    EXPECT_EQ(alloc.n_free_pages(), 0u);
    EXPECT_EQ(alloc.free_list_head(), kInvalidPage);
}

TEST_F(TreeAllocatorTest, AllocSinglePage) {
    PageId p1 = alloc.alloc_page(file);
    EXPECT_NE(p1, kInvalidPage);
    EXPECT_EQ(p1, 3u);  // first allocatable page after bitmap

    PageId p2 = alloc.alloc_page(file);
    EXPECT_EQ(p2, 4u);
    EXPECT_EQ(alloc.n_pages(), 5u);  // grew by 2
}

TEST_F(TreeAllocatorTest, AllocExtent) {
    PageId ext = alloc.alloc_extent(file, 5);
    EXPECT_NE(ext, kInvalidPage);
    EXPECT_EQ(ext, 3u);  // starts right after bitmap

    // The extent covers pages 3-7. Next single alloc should be page 8.
    PageId p = alloc.alloc_page(file);
    EXPECT_EQ(p, 8u);
}

TEST_F(TreeAllocatorTest, FreeAndRealloc) {
    PageId p1 = alloc.alloc_page(file);  // page 3
    PageId p2 = alloc.alloc_page(file);  // page 4
    EXPECT_EQ(p1, 3u);
    EXPECT_EQ(p2, 4u);

    // Free p1 — it goes to the free list head.
    alloc.free_page(file, p1);
    EXPECT_EQ(alloc.free_list_head(), p1);
    EXPECT_EQ(alloc.n_free_pages(), 1u);

    // Alloc should reuse p1 (LIFO).
    PageId p3 = alloc.alloc_page(file);
    EXPECT_EQ(p3, 3u);
    EXPECT_EQ(alloc.n_free_pages(), 0u);
}

TEST_F(TreeAllocatorTest, FreeExtentAndRealloc) {
    PageId ext = alloc.alloc_extent(file, 3);  // pages 3-5
    EXPECT_EQ(ext, 3u);

    // Free the extent — each page goes to the free list.
    alloc.free_extent(file, ext, 3);
    EXPECT_EQ(alloc.n_free_pages(), 3u);

    // Allocating 3 contiguous pages should find them in the bitmap
    // (they're free again). The free list has stale entries but alloc_extent
    // uses the bitmap, and alloc_page validates against the bitmap.
    PageId ext2 = alloc.alloc_extent(file, 3);
    EXPECT_EQ(ext2, 3u);
}

TEST_F(TreeAllocatorTest, BitmapPersistence) {
    // Allocate some pages.
    alloc.alloc_page(file);  // page 3
    alloc.alloc_extent(file, 2);  // pages 4-5
    alloc.alloc_page(file);  // page 6

    // Flush the bitmap.
    alloc.flush_bitmap(file);
    file.sync();

    // Save allocator state.
    auto n_pages = alloc.n_pages();
    auto flh = alloc.free_list_head();
    auto nfp = alloc.n_free_pages();

    // Reload from disk.
    PageAllocator alloc2;
    alloc2.load(file, kBitmapPage, kBitmapPages, n_pages, flh, nfp);

    // The reloaded bitmap should agree: pages 0-6 are allocated.
    // Allocating a new page should give page 7.
    PageId p = alloc2.alloc_page(file);
    EXPECT_EQ(p, 7u);
}

// ===========================================================================
// Superblock
// ===========================================================================

TEST(TreeSuperblock, InitFreshAndLoad) {
    std::string path = temp_path(".tree");

    {
        PageFile f(path);
        f.truncate(3);  // superblock + shadow + bitmap

        Superblock sb;
        sb.init_fresh(/*bitmap_page=*/2, /*bitmap_pages=*/1);
        sb.commit(f);

        // Verify commit_seq.
        EXPECT_EQ(sb.commit_seq(), 1u);
    }
    {
        PageFile f(path);
        Superblock sb;
        sb.load(f);
        EXPECT_EQ(sb.commit_seq(), 1u);
        EXPECT_EQ(sb.alloc_bitmap_page(), 2u);
        EXPECT_EQ(sb.alloc_bitmap_pages(), 1u);
        EXPECT_EQ(sb.root_node_page(), kInvalidPage);  // no root yet
    }
    std::filesystem::remove(path);
}

TEST(TreeSuperblock, MultipleCommits) {
    std::string path = temp_path(".tree");
    PageFile f(path);
    f.truncate(3);

    Superblock sb;
    sb.init_fresh(2, 1);
    sb.commit(f);
    EXPECT_EQ(sb.commit_seq(), 1u);

    sb.set_root(42, 7);
    sb.set_depth(2);
    sb.commit(f);
    EXPECT_EQ(sb.commit_seq(), 2u);

    // Reload.
    Superblock sb2;
    sb2.load(f);
    EXPECT_EQ(sb2.commit_seq(), 2u);
    EXPECT_EQ(sb2.root_node_page(), 42u);
    EXPECT_EQ(sb2.root_node_pages(), 7u);
    EXPECT_EQ(sb2.depth(), 2u);

    std::filesystem::remove(path);
}

TEST(TreeSuperblock, ShadowSurvivesPartialWrite) {
    // Simulate a crash: after commit #1, we overwrite the ACTIVE page
    // (the one with the higher seq) with garbage. The shadow (commit #0)
    // should still be valid and loadable.
    std::string path = temp_path(".tree");

    PageFile f(path);
    f.truncate(3);

    Superblock sb;
    sb.init_fresh(2, 1);
    sb.commit(f);  // seq=1, written to shadow page 1

    // Corrupt page 0 (the "active" or old copy) with garbage.
    std::vector<uint8_t> garbage(kPageSize, 0xAA);
    f.write_page(0, garbage.data());

    // Should still load from the surviving shadow.
    Superblock sb2;
    sb2.load(f);
    EXPECT_EQ(sb2.commit_seq(), 1u);
    EXPECT_EQ(sb2.alloc_bitmap_page(), 2u);

    std::filesystem::remove(path);
}

TEST(TreeSuperblock, BadMagicThrows) {
    std::string path = temp_path(".tree");
    PageFile f(path);
    f.truncate(3);

    // Both pages are zero (no valid magic).
    Superblock sb;
    EXPECT_THROW(sb.load(f), sextant::Error);

    std::filesystem::remove(path);
}

// ===========================================================================
// TreeManifest (TOML round-trip)
// ===========================================================================

TEST(TreeManifest, RoundTrip) {
    TreeManifest m;
    m.dim = 768;
    m.m4 = 192;
    m.scan_pq_bits = 4;
    m.quantizer_type = "pq";
    m.prq_nsplits = 0;
    m.depth = 2;
    m.k_root = 128;
    m.leaf_capacity = 5000;
    m.n_leaves = 2000;
    m.n_probe_l0 = 16;
    m.n_probe_ln = 4;
    m.adaptive_probe_gap = 1.606f;
    m.median_lid = 13.21f;
    m.balance_factor = 4.0f;
    m.sub_shard_probe_pct = 50;

    std::string toml = manifest_to_toml(m);
    TreeManifest m2 = manifest_from_toml(toml);

    EXPECT_EQ(m2.dim, 768u);
    EXPECT_EQ(m2.m4, 192u);
    EXPECT_EQ(m2.scan_pq_bits, 4);
    EXPECT_EQ(m2.quantizer_type, "pq");
    EXPECT_EQ(m2.depth, 2);
    EXPECT_EQ(m2.k_root, 128u);
    EXPECT_EQ(m2.leaf_capacity, 5000u);
    EXPECT_EQ(m2.n_leaves, 2000u);
    EXPECT_EQ(m2.n_probe_l0, 16u);
    EXPECT_EQ(m2.n_probe_ln, 4u);
    EXPECT_FLOAT_EQ(m2.adaptive_probe_gap, 1.606f);
    EXPECT_FLOAT_EQ(m2.median_lid, 13.21f);
    EXPECT_FLOAT_EQ(m2.balance_factor, 4.0f);
    EXPECT_EQ(m2.sub_shard_probe_pct, 50u);
}

TEST(TreeManifest, MissingFieldThrows) {
    std::string bad_toml = R"(
[index]
dim = 768
m4 = 192
scan_pq_bits = 4
# missing quantizer_type is optional, but missing depth is fatal

[tree]
depth = 2
k_root = 128
# missing leaf_capacity
n_leaves = 2000
n_probe_l0 = 16
n_probe_ln = 4
)";
    EXPECT_THROW(manifest_from_toml(bad_toml), sextant::Error);
}

TEST(TreeManifest, MissingSectionThrows) {
    std::string bad_toml = R"(
[index]
dim = 768
)";
    EXPECT_THROW(manifest_from_toml(bad_toml), sextant::Error);
}

}  // namespace
}  // namespace sextant::tree
