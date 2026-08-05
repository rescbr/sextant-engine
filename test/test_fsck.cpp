#include <gtest/gtest.h>
#include "fbin_source.hpp"
#include "tree/fsck.hpp"
#include "tree/ivf_tree_index.hpp"
#include "tree/page_file.hpp"
#include "tree/superblock.hpp"
#include "sextant/config.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace sextant::tree {
namespace {

/// Write random clustered data to an fbin file.
std::string write_test_fbin(const std::string& name, uint64_t n, uint32_t dim,
                            uint32_t n_clusters, uint32_t seed) {
    auto path = (std::filesystem::temp_directory_path() / name).string();
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.5f);

    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }

    std::vector<float> data(n * dim);
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t ci = i % n_clusters;
        for (uint32_t d = 0; d < dim; ++d) {
            data[i * dim + d] = centers[ci][d] + noise(rng);
        }
    }

    FILE* f = std::fopen(path.c_str(), "wb");
    uint32_t header[2] = {static_cast<uint32_t>(n), dim};
    std::fwrite(header, sizeof(uint32_t), 2, f);
    std::fwrite(data.data(), sizeof(float), n * dim, f);
    std::fclose(f);
    return path;
}

/// Build a small tree and return its path.
std::string build_small_tree(const std::string& tree_path) {
    const uint64_t n = 2000;
    const uint32_t dim = 64;
    const std::string base_path =
        write_test_fbin("fsck_test_pq.fbin", n, dim, /*n_clusters=*/20,
                        /*seed=*/42);
    std::filesystem::remove(tree_path);

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 16;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.num_threads = 4;

    auto result = ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    EXPECT_EQ(result.n_vectors, n);
    EXPECT_TRUE(std::filesystem::exists(tree_path));
    return tree_path;
}

// ===========================================================================
// fsck: clean tree passes all levels
// ===========================================================================

TEST(TreeFsck, CleanTreePasses) {
    const std::string tree_path =
        (std::filesystem::temp_directory_path() / "fsck_clean.tree").string();
    build_small_tree(tree_path);

    auto result = fsck(tree_path);
    EXPECT_TRUE(result.superblock_ok);
    EXPECT_TRUE(result.tree_walk_ok);
    EXPECT_TRUE(result.leaf_internals_ok);
    EXPECT_TRUE(result.page_accounting_ok);
    EXPECT_TRUE(result.payload_ok);
    EXPECT_TRUE(result.referential_ok);
    EXPECT_TRUE(result.all_ok());

    // Sanity: the tree has leaves and vectors.
    EXPECT_GT(result.n_leaves, 0u);
    EXPECT_GT(result.total_leaf_count, 0u);
    EXPECT_EQ(result.magic_failures, 0u);
    EXPECT_EQ(result.crc_failures, 0u);
    EXPECT_EQ(result.leaked_pages, 0u);
    EXPECT_EQ(result.orphan_pages, 0u);

    // Summary should mention key fields.
    const auto s = result.summary();
    EXPECT_NE(s.find("total_leaf_count"), std::string::npos);
    EXPECT_NE(s.find("total_pages"), std::string::npos);
    EXPECT_NE(s.find("n_leaves"), std::string::npos);

    std::filesystem::remove(tree_path);
}

// ===========================================================================
// fsck: detects + repairs a corrupted bitmap
// ===========================================================================

TEST(TreeFsck, RepairsCorruptBitmap) {
    const std::string tree_path =
        (std::filesystem::temp_directory_path() / "fsck_repair.tree").string();
    build_small_tree(tree_path);

    // First, a clean fsck should pass.
    auto before = fsck(tree_path);
    ASSERT_TRUE(before.all_ok());

    // Corrupt the bitmap so that accounting breaks. We need to know which
    // pages are allocated vs free to create a real leak + orphan.
    PageId leaked = 0;   // a free page we'll falsely mark allocated
    PageId orphan = 0;   // an allocated page we'll falsely mark free
    {
        PageFile file(tree_path);
        Superblock sb;
        sb.load(file);

        const PageId bmp_page = sb.alloc_bitmap_page();
        const uint32_t bmp_pages = sb.alloc_bitmap_pages();
        ASSERT_NE(bmp_page, kInvalidPage);

        std::vector<uint8_t> bmp(static_cast<size_t>(bmp_pages) * kPageSize, 0);
        file.read_pages(bmp_page, bmp_pages, bmp.data());

        // Find the first free page (leak) and first allocated data page (orphan).
        const uint64_t total = file.num_pages();
        for (uint64_t p = kFirstDataPage; p < total && orphan == 0; ++p) {
            const bool set = (bmp[p / 8] >> (p % 8)) & 1u;
            if (set) orphan = p;
            else if (leaked == 0) leaked = p;
        }
        ASSERT_NE(orphan, 0u);  // there must be at least one allocated data page

        // Flip bits: leak a free page (if any), orphan an allocated page.
        if (leaked != 0)
            bmp[leaked / 8] ^= static_cast<uint8_t>(1u << (leaked % 8));
        bmp[orphan / 8] ^= static_cast<uint8_t>(1u << (orphan % 8));

        file.write_pages(bmp_page, bmp_pages, bmp.data());
        file.sync();
    }

    // Now fsck (read-only) should detect the discrepancy.
    auto corrupted = fsck(tree_path, /*repair=*/false);
    EXPECT_FALSE(corrupted.page_accounting_ok);
    EXPECT_GE(corrupted.leaked_pages + corrupted.orphan_pages, 1u);

    // Repair.
    auto repaired = fsck(tree_path, /*repair=*/true);
    EXPECT_TRUE(repaired.page_accounting_ok);
    EXPECT_EQ(repaired.leaked_pages, 0u);
    EXPECT_EQ(repaired.orphan_pages, 0u);
    EXPECT_TRUE(repaired.all_ok());

    // A follow-up read-only fsck should still pass (persisted repair).
    auto after = fsck(tree_path, /*repair=*/false);
    EXPECT_TRUE(after.all_ok());

    std::filesystem::remove(tree_path);
}

// ===========================================================================
// fsck: missing file fails gracefully
// ===========================================================================

TEST(TreeFsck, EmptyFileFailsGracefully) {
    // PageFile creates the file if absent; fsck should then report a bad
    // superblock rather than throwing.
    const std::string tree_path =
        (std::filesystem::temp_directory_path() / "fsck_empty.tree").string();
    std::filesystem::remove(tree_path);

    auto result = fsck(tree_path);
    EXPECT_FALSE(result.superblock_ok);
    EXPECT_FALSE(result.all_ok());

    std::filesystem::remove(tree_path);
}

}  // namespace
}  // namespace sextant::tree
