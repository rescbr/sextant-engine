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

    auto result = IVFTreeIndex::build_streaming_pca(base_path, tree_path, cfg);
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
    // k_root comfortably larger than the expected leaf count so the streaming
    // build collapses to depth=1 (root children point directly to leaves).
    // leaf_capacity >= N guarantees each cluster flushes at most one leaf,
    // so n_leaves <= (non-empty clusters) <= k_root.
    cfg.k_root = 32;
    cfg.leaf_capacity = 5000;  // >= N → ≤ 1 leaf per cluster
    cfg.num_threads = 4;
    cfg.closure_multiplier = 0.0f;  // no boundary replication → stable leaf count

    IVFTreeIndex::build_streaming_pca(base_path, tree_path, cfg);
    auto idx = IVFTreeIndex::open(tree_path);

    // n_leaves (≤ k_root) forces depth=1: root children are leaves.
    EXPECT_EQ(idx->depth(), 1u);
    EXPECT_EQ(idx->k_root(), 32u);
    EXPECT_LE(idx->n_leaves(), 32u);

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
    cfg.leaf_capacity = 500;  // n=20000 → many leaves (> k_root) → depth=2
    cfg.num_threads = 4;
    cfg.closure_multiplier = 0.0f;  // no boundary replication → stable leaf count

    IVFTreeIndex::build_streaming_pca(base_path, tree_path, cfg);
    auto idx = IVFTreeIndex::open(tree_path);

    // n_leaves > k_root → depth=2 (root → L2 internal nodes → leaves).
    EXPECT_EQ(idx->depth(), 2u);
    EXPECT_EQ(idx->k_root(), 16u);
    EXPECT_GT(idx->n_leaves(), 16u);

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

// ===========================================================================
// Rerank: decoding top-W PQ codes to FP32 and re-sorting by exact distance
// should not hurt (and usually improves) recall vs. raw PQ-approx distances.
// ===========================================================================

TEST(IVFTreeIndex, RerankImprovesRecall) {
    const uint64_t n = 4000;
    const uint32_t dim = 64;
    const std::string base_path = write_test_fbin("tree_test_rerank.fbin", n,
                                                   dim, /*n_clusters=*/30,
                                                   /*seed=*/99);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_test_rerank.tree").string();
    std::filesystem::remove(tree_path);

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 16;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 400;
    cfg.num_threads = 4;

    IVFTreeIndex::build_streaming_pca(base_path, tree_path, cfg);
    auto idx = IVFTreeIndex::open(tree_path);

    // Regenerate the cluster centers to build near-centroid queries.
    std::mt19937 rng(99);
    std::vector<std::vector<float>> centers(30);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }

    SearchConfig base_sconfig;
    base_sconfig.k = 10;
    base_sconfig.n_probe = 8;
    base_sconfig.fastscan_W = 60;  // tight shortlist → rerank re-orders more

    auto recall_run = [&](bool rerank) {
        SearchConfig sc = base_sconfig;
        sc.rerank = rerank;
        uint32_t total_hits = 0;
        for (uint32_t trial = 0; trial < 20; ++trial) {
            std::vector<float> query(dim);
            for (uint32_t d = 0; d < dim; ++d) {
                query[d] = centers[trial % 30][d];
            }
            auto gt = brute_force_knn(base_path, query.data(), dim, 10);
            auto results = idx->search(query.data(), 10, sc);
            std::unordered_set<RowId> gt_set(gt.begin(), gt.end());
            for (const auto& c : results) {
                if (gt_set.count(c.row_id)) ++total_hits;
            }
        }
        return float(total_hits) / (20.0f * 10.0f);
    };

    const float recall_no_rerank = recall_run(false);
    const float recall_rerank    = recall_run(true);
    spdlog::info("RerankImprovesRecall: no-rerank recall@10 = {:.3f}, "
                 "rerank recall@10 = {:.3f}", recall_no_rerank, recall_rerank);

    // Rerank never hurts recall (it uses strictly more information); allow a
    // tiny tolerance for ties on already-saturated queries.
    EXPECT_GE(recall_rerank + 1e-6f, recall_no_rerank);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Depth-3 tree (build_streaming_pca): root → L1 → L2 → leaves.
// Forces depth-3 with a small dataset by lowering k_root_max_depth2.
// ===========================================================================

TEST(IVFTreeIndex, DepthThreeTreeStreamingPca) {
    const uint64_t n = 30000;
    const uint32_t dim = 64;
    const uint32_t n_clusters = 80;
    const std::string base_path = write_test_fbin("tree_test_depth3.fbin",
                                                   n, dim, n_clusters,
                                                   /*seed=*/123);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_test_depth3.tree").string();

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 16;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 64;                 // fine-grained cluster count
    cfg.k_root_max_depth2 = 16;      // 64 > 16 → depth-3, k_l1 = 16
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;        // keep the test fast
    cfg.num_threads = 4;

    auto result = IVFTreeIndex::build_streaming_pca(base_path, tree_path, cfg);
    EXPECT_EQ(result.n_vectors, n);

    auto idx = IVFTreeIndex::open(tree_path);
    // k_root (fine centroids) is preserved; depth-3 was selected.
    ASSERT_EQ(idx->depth(), 3u);
    EXPECT_EQ(idx->k_root(), 64u);
    EXPECT_GT(idx->n_leaves(), 0u);

    // Regenerate cluster centers (seed must match write_test_fbin).
    std::mt19937 rng(123);
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }

    // --- Routing-quality check (cluster-membership recall) ---
    // The depth-3 descent (root → L1 → L2 → leaves) must route a query placed
    // at cluster center `ci` to leaves holding that cluster's vectors. We
    // measure this by the fraction of returned row_ids that belong to cluster
    // `ci` (row_id % n_clusters == ci). This is independent of PQ-quantization
    // ranking noise and validates the multi-level routing correctness.
    //
    // Probe a few L1 nodes + several L2/leaves to exercise the full descent.
    SearchConfig sconfig;
    sconfig.k = 10;
    sconfig.n_probe = 4;
    sconfig.n_probe_ln = 8;
    sconfig.rerank = false;  // routing quality, not PQ-rerank quality

    uint32_t cluster_hits = 0;
    uint32_t total_returned = 0;
    const uint32_t trials = 20;
    for (uint32_t trial = 0; trial < trials; ++trial) {
        const uint32_t ci = trial % n_clusters;
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d) query[d] = centers[ci][d];
        auto results = idx->search(query.data(), 10, sconfig);
        EXPECT_LE(results.size(), 10u);
        EXPECT_GE(results.size(), 1u);  // depth-3 must route to ≥1 leaf
        for (const auto& c : results) {
            ++total_returned;
            if (static_cast<uint32_t>(c.row_id) % n_clusters == ci)
                ++cluster_hits;
        }
    }
    const float cluster_recall = (total_returned > 0)
        ? float(cluster_hits) / total_returned : 0.0f;
    spdlog::info("IVFTreeIndex::DepthThreeTree: cluster-membership = {:.3f}",
                 cluster_recall);
    // The 3-level descent must land in the correct cluster most of the time.
    EXPECT_GT(cluster_recall, 0.5f);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Depth-3 vs depth-2 parity: with the same data + k_root, depth-3 (extra L1
// level) must not regress routing quality vs the equivalent depth-2 tree.
// Both use build_streaming_pca; only k_root_max_depth2 differs.
// ===========================================================================

TEST(IVFTreeIndex, DepthThreeParityWithDepthTwo) {
    const uint64_t n = 30000;
    const uint32_t dim = 64;
    const uint32_t n_clusters = 80;
    const std::string base_path = write_test_fbin("tree_test_parity.fbin",
                                                   n, dim, n_clusters, 321);
    auto build_and_measure = [&](uint32_t k_root_max_depth2) -> float {
        const auto tree_path = (std::filesystem::temp_directory_path() /
            (std::string("tree_test_parity_") +
             std::to_string(k_root_max_depth2) + ".tree")).string();
        IVFTreeIndex::BuildConfig cfg;
        cfg.params.metric = MetricKind::L2Sq;
        cfg.params.quantizer_type = "pq";
        cfg.params.pq4_m = 16;
        cfg.params.scan_pq_bits = 4;
        cfg.params.closure_epsilon = -1.0f;
        cfg.k_root = 64;
        cfg.k_root_max_depth2 = k_root_max_depth2;
        cfg.leaf_capacity = 500;
        cfg.pca_dims = 16;
        cfg.max_lloyd_passes = 2;
        cfg.num_threads = 4;
        IVFTreeIndex::build_streaming_pca(base_path, tree_path, cfg);
        auto idx = IVFTreeIndex::open(tree_path);

        std::mt19937 rng(321);
        std::vector<std::vector<float>> centers(n_clusters);
        for (auto& c : centers) {
            c.resize(dim);
            for (float& v : c)
                v = std::uniform_real_distribution<float>(-10, 10)(rng);
        }
        SearchConfig sc;
        sc.k = 10; sc.n_probe = 4; sc.n_probe_ln = 8; sc.rerank = false;
        uint32_t hits = 0, total = 0;
        for (uint32_t t = 0; t < 20; ++t) {
            const uint32_t ci = t % n_clusters;
            std::vector<float> q(dim);
            for (uint32_t d = 0; d < dim; ++d) q[d] = centers[ci][d];
            auto res = idx->search(q.data(), 10, sc);
            for (const auto& c : res) {
                ++total;
                if (static_cast<uint32_t>(c.row_id) % n_clusters == ci) ++hits;
            }
        }
        std::filesystem::remove(tree_path);
        return total > 0 ? float(hits) / total : 0.0f;
    };

    const float d2 = build_and_measure(/*k_root_max_depth2=*/100000);  // → depth-2
    const float d3 = build_and_measure(/*k_root_max_depth2=*/16);      // → depth-3
    spdlog::info("DepthThreeParity: depth-2 cluster-recall={:.3f}, "
                 "depth-3 cluster-recall={:.3f}", d2, d3);
    // Depth-3 routing quality must be within 15pp of depth-2 (the extra level
    // adds one FP16 routing hop, which is slightly lossy but comparable).
    EXPECT_GE(d3 + 0.15f, d2);
    // Both should route to the correct cluster reasonably well.
    EXPECT_GT(d2, 0.3f);
    EXPECT_GT(d3, 0.3f);

    std::filesystem::remove(base_path);
}

}  // namespace
}  // namespace sextant::tree
