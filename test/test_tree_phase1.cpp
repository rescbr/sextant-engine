#include <gtest/gtest.h>

#include "tree/ivf_tree_index.hpp"
#include "sextant/config.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <spdlog/spdlog.h>

namespace sextant::tree {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write random clustered data to an fbin file.
std::string write_test_fbin(const std::string& name, uint64_t n, uint32_t dim,
                            uint32_t n_clusters, uint32_t seed) {
    auto path = (std::filesystem::temp_directory_path() / name).string();
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.5f);

    // Cluster centers.
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

/// Brute-force k-NN ground truth.
std::vector<RowId> brute_force_knn(const std::string& fbin_path,
                                   const float* query, uint32_t dim,
                                   uint32_t k) {
    FILE* f = std::fopen(fbin_path.c_str(), "rb");
    uint32_t header[2];
    std::fread(header, sizeof(uint32_t), 2, f);
    const uint64_t n = header[0];
    std::vector<float> data(n * dim);
    std::fread(data.data(), sizeof(float), n * dim, f);
    std::fclose(f);

    std::vector<std::pair<float, RowId>> dists(n);
    for (uint64_t i = 0; i < n; ++i) {
        float d = 0;
        for (uint32_t j = 0; j < dim; ++j) {
            const float diff = data[i * dim + j] - query[j];
            d += diff * diff;
        }
        dists[i] = {d, static_cast<RowId>(i)};
    }
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
    std::vector<RowId> result;
    for (uint32_t i = 0; i < k; ++i) result.push_back(dists[i].second);
    return result;
}

// ===========================================================================
// Build + Search: basic round-trip with PQ
// ===========================================================================

TEST(IVFTreeIndex, BuildAndSearchPQ) {
    const uint64_t n = 2000;
    const uint32_t dim = 64;
    const std::string base_path = write_test_fbin("tree_test_pq.fbin", n, dim,
                                                   /*n_clusters=*/20, /*seed=*/42);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_test_pq.tree").string();
    std::filesystem::remove(tree_path);  // clean slate

    // Build.
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

    auto result = IVFTreeIndex::build(base_path, tree_path, cfg);
    EXPECT_EQ(result.n_vectors, n);
    EXPECT_EQ(result.dim, dim);
    EXPECT_TRUE(std::filesystem::exists(tree_path));

    // Open.
    auto idx = IVFTreeIndex::open(tree_path);
    EXPECT_EQ(idx->dim(), dim);
    EXPECT_EQ(idx->m4(), 16u);
    EXPECT_EQ(idx->k_root(), 8u);

    // Search a query that's near one of the cluster centers.
    // Regenerate the same cluster centers.
    std::mt19937 rng(42);
    std::vector<std::vector<float>> centers(20);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }

    SearchConfig sconfig;
    sconfig.k = 10;
    sconfig.n_probe = 8;  // probe all root children

    uint32_t total_recall = 0;
    for (uint32_t trial = 0; trial < 10; ++trial) {
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            query[d] = centers[trial % 20][d];
        }

        auto gt = brute_force_knn(base_path, query.data(), dim, 10);
        auto results = idx->search(query.data(), 10, sconfig);
        EXPECT_LE(results.size(), 10u);

        // Compute recall@10.
        std::unordered_set<RowId> gt_set(gt.begin(), gt.end());
        uint32_t hits = 0;
        for (const auto& c : results) {
            if (gt_set.count(c.row_id)) ++hits;
        }
        total_recall += hits;
    }

    // With 20 well-separated clusters and probing all 8 root children,
    // recall should be decent (PQ quantization loses some).
    const float recall = float(total_recall) / (10 * 10);
    spdlog::info("IVFTreeIndex::BuildAndSearchPQ: recall@10 = {:.3f}", recall);
    EXPECT_GT(recall, 0.05f);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Build: verify tree structure (depth, n_leaves)
// ===========================================================================

TEST(IVFTreeIndex, TreeStructure) {
    const uint64_t n = 5000;
    const uint32_t dim = 32;
    const std::string base_path = write_test_fbin("tree_test_struct.fbin",
                                                   n, dim, 50, 99);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_test_struct.tree").string();

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 16;
    cfg.leaf_capacity = 500;  // → n_leaves = 10
    cfg.num_threads = 4;

    IVFTreeIndex::build(base_path, tree_path, cfg);
    auto idx = IVFTreeIndex::open(tree_path);

    // With n=5000, leaf_cap=500: n_leaves=10. k_root=16.
    // depth=1 (n_leaves < k_root).
    EXPECT_EQ(idx->depth(), 1u);
    EXPECT_EQ(idx->k_root(), 16u);
    EXPECT_EQ(idx->n_leaves(), 10u);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Build: depth=2 when n_leaves > k_root
// ===========================================================================

TEST(IVFTreeIndex, DepthTwoTree) {
    const uint64_t n = 20000;
    const uint32_t dim = 32;
    const std::string base_path = write_test_fbin("tree_test_depth2.fbin",
                                                   n, dim, 100, 77);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_test_depth2.tree").string();

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 16;
    cfg.leaf_capacity = 500;  // → n_leaves = 40
    cfg.num_threads = 4;

    IVFTreeIndex::build(base_path, tree_path, cfg);
    auto idx = IVFTreeIndex::open(tree_path);

    // n_leaves=40 > k_root=16 → depth=2.
    EXPECT_EQ(idx->depth(), 2u);
    EXPECT_EQ(idx->k_root(), 16u);
    EXPECT_EQ(idx->n_leaves(), 40u);

    // Search should work on depth=2.
    SearchConfig sconfig;
    sconfig.k = 10;
    sconfig.n_probe = 4;

    std::vector<float> query(dim, 0.5f);
    auto results = idx->search(query.data(), 10, sconfig);
    EXPECT_GE(results.size(), 1u);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

}  // namespace
}  // namespace sextant::tree
