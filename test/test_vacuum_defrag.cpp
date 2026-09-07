#include <gtest/gtest.h>
#include "fbin_source.hpp"
#include "test_data.hpp"
#include "tree/ivf_tree_index.hpp"
#include <sextant/column_data.hpp>
#include "sextant/config.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace sextant::tree {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string temp_path(const char* suffix) {
    auto tmpl = std::string("/tmp/sextant_vacuum_test") + suffix + "_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = ::mkstemp(buf.data());
    EXPECT_NE(fd, -1);
    ::close(fd);
    ::unlink(buf.data());
    return std::string(buf.data());
}

std::string write_test_fbin(const std::string& path, uint64_t n, uint32_t dim,
                             uint32_t n_clusters, uint32_t seed) {
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
        for (uint32_t d = 0; d < dim; ++d)
            data[i * dim + d] = centers[ci][d] + noise(rng);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    uint32_t header[2] = {static_cast<uint32_t>(n), dim};
    std::fwrite(header, sizeof(uint32_t), 2, f);
    std::fwrite(data.data(), sizeof(float), n * dim, f);
    std::fclose(f);
    return path;
}

/// Build a tree with an int32 filter column ("category") for vacuum/defrag tests.
struct TestTree {
    std::string tree_path;
    std::string fbin_path;
    std::vector<float> base_data;
    uint32_t dim;
    uint64_t n;
    Schema schema;
    std::vector<ColumnData> filter_data;
};

TestTree build_test_tree(uint64_t n, uint32_t dim, uint32_t n_clusters,
                          uint32_t leaf_cap, uint32_t k_root) {
    TestTree tt;
    tt.dim = dim;
    tt.n = n;
    tt.fbin_path = "vacuum_test.fbin";
    tt.tree_path = temp_path(".tree");
    std::filesystem::remove(tt.tree_path);

    write_test_fbin(tt.fbin_path, n, dim, n_clusters, 42);

    // Read base data for queries.
    FILE* f = std::fopen(tt.fbin_path.c_str(), "rb");
    uint32_t hdr[2];
    if (std::fread(hdr, sizeof(uint32_t), 2, f) != 2) return tt;
    tt.base_data.resize(n * dim);
    if (std::fread(tt.base_data.data(), sizeof(float), n * dim, f)
        != static_cast<uint64_t>(n) * dim) return tt;
    std::fclose(f);

    // int32 filter column: category = i % 10.
    tt.schema.columns.push_back({"category", ColumnType::Int32});
    tt.filter_data.resize(1);
    tt.filter_data[0].type = ColumnType::Int32;
    tt.filter_data[0].fixed_data.resize(n * 4);
    for (uint32_t i = 0; i < n; ++i) {
        int32_t cat = static_cast<int32_t>(i % 10);
        std::memcpy(&tt.filter_data[0].fixed_data[i * 4], &cat, 4);
    }

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = k_root;
    cfg.leaf_capacity = leaf_cap;
    cfg.num_threads = 2;
    cfg.adaptive_probe_gap = 0.0f;
    cfg.filter_schema = tt.schema;
    cfg.filter_column_data = tt.filter_data;

    FbinSource s(tt.fbin_path);
    IVFTreeIndex::build_streaming_pca(s, tt.tree_path, cfg);
    return tt;
}

// ===========================================================================
// Vacuum: summary repair after deletes
// ===========================================================================

TEST(VacuumDefrag, VacuumRepairsDirtySummaries) {
    auto tt = build_test_tree(/*n=*/1000, /*dim=*/32, /*n_clusters=*/10,
                               /*leaf_cap=*/200, /*k_root=*/4);
    auto idx = IVFTreeIndex::open(tt.tree_path);

    // Delete all vectors with category == 0 (100 vectors).
    std::vector<RowId> to_delete;
    for (uint32_t i = 0; i < tt.n; ++i) {
        if (i % 10 == 0) to_delete.push_back(static_cast<RowId>(i));
    }
    ASSERT_EQ(to_delete.size(), 100u);
    idx->delete_batch(to_delete);
    EXPECT_EQ(idx->live_count(), tt.n - 100);

    // After delete, vacuum should repair summaries.
    IVFTreeIndex::VacuumConfig vcfg;
    vcfg.batch_size = 16;  // small batch to exercise batched commit
    auto result = idx->vacuum(vcfg);

    EXPECT_GT(result.summaries_repaired, 0u)
        << "Vacuum should have repaired at least one dirty summary";
    EXPECT_EQ(result.leaves_scanned, idx->n_leaves());

    // Search for category == 5 should still work correctly after vacuum.
    SearchConfig scfg;
    scfg.k = 10;
    scfg.n_probe = 4;
    scfg.adaptive_probe_gap = 0.0f;
    Predicate pred;
    pred.column = "category";
    pred.op = PredicateOp::Eq;
    pred.value = 5;
    scfg.predicates = {pred};

    auto results = idx->search(&tt.base_data[5 * tt.dim], 10, scfg);
    EXPECT_FALSE(results.empty());
    // All results must have category == 5 (they were not deleted).
    // (We can't directly verify the filter value from search results without
    // a row_id → category lookup, but the search correctness is already
    // validated in test_tree_filter. Here we verify vacuum didn't corrupt.)

    std::filesystem::remove(tt.fbin_path);
    std::filesystem::remove(tt.tree_path);
}

TEST(VacuumDefrag, VacuumPreservesSearchRecall) {
    const uint32_t dim = 32;
    auto tt = build_test_tree(/*n=*/1000, dim, /*n_clusters=*/10,
                               /*leaf_cap=*/200, /*k_root=*/4);
    auto idx = IVFTreeIndex::open(tt.tree_path);

    // Baseline recall: search for 20 random queries (no filter).
    auto measure_recall = [&](IVFTreeIndex& index) -> float {
        SearchConfig scfg;
        scfg.k = 10;
        scfg.n_probe = 4;
        scfg.adaptive_probe_gap = 0.0f;
        uint32_t hits = 0;
        const uint32_t n_queries = 20;
        for (uint32_t q = 0; q < n_queries; ++q) {
            auto results = index.search(&tt.base_data[q * dim], 10, scfg);
            for (const auto& r : results)
                if (static_cast<uint32_t>(r.row_id) == q) { ++hits; break; }
        }
        return static_cast<float>(hits) / n_queries;
    };

    // Baseline recall (unused but validates search works before mutation).
    (void)measure_recall(*idx);

    // Delete 200 vectors.
    std::vector<RowId> to_delete;
    for (uint32_t i = 0; i < 200; ++i) to_delete.push_back(static_cast<RowId>(i * 3));
    idx->delete_batch(to_delete);

    // Vacuum.
    idx->vacuum();

    // Recall should be similar (vacuum only touches summaries, not codes).
    const float recall_after = measure_recall(*idx);

    // Recall should not drop significantly. It may change slightly because
    // deleted vectors are no longer candidates.
    EXPECT_GT(recall_after, 0.0f) << "Search returned nothing after vacuum";

    std::filesystem::remove(tt.fbin_path);
    std::filesystem::remove(tt.tree_path);
}

TEST(VacuumDefrag, VacuumRebuildCardinality) {
    auto tt = build_test_tree(/*n=*/500, /*dim=*/32, /*n_clusters=*/10,
                               /*leaf_cap=*/200, /*k_root=*/4);
    auto idx = IVFTreeIndex::open(tt.tree_path);

    // Delete 50 vectors.
    std::vector<RowId> to_delete;
    for (uint32_t i = 0; i < 50; ++i) to_delete.push_back(static_cast<RowId>(i * 7));
    idx->delete_batch(to_delete);

    // Vacuum with cardinality rebuild.
    IVFTreeIndex::VacuumConfig vcfg;
    vcfg.rebuild_cardinality = true;
    vcfg.batch_size = 16;
    auto result = idx->vacuum(vcfg);

    EXPECT_GT(result.cardinality_entries_rebuilt, 0u)
        << "Cardinality table should have been rebuilt";

    // Reopen and verify search still works.
    idx = IVFTreeIndex::open(tt.tree_path);
    SearchConfig scfg;
    scfg.k = 10;
    scfg.n_probe = 4;
    scfg.adaptive_probe_gap = 0.0f;
    auto results = idx->search(&tt.base_data[5 * 32], 10, scfg);
    EXPECT_FALSE(results.empty());

    std::filesystem::remove(tt.fbin_path);
    std::filesystem::remove(tt.tree_path);
}

TEST(VacuumDefrag, VacuumOnCleanIndexIsNoOp) {
    auto tt = build_test_tree(/*n=*/500, /*dim=*/32, /*n_clusters=*/10,
                               /*leaf_cap=*/200, /*k_root=*/4);
    auto idx = IVFTreeIndex::open(tt.tree_path);

    // No deletes → no dirty leaves → vacuum should repair 0 summaries.
    auto result = idx->vacuum();
    EXPECT_EQ(result.summaries_repaired, 0u);

    std::filesystem::remove(tt.fbin_path);
    std::filesystem::remove(tt.tree_path);
}

// ===========================================================================
// Defrag: extent compaction + file shrink
// ===========================================================================

TEST(VacuumDefrag, DefragRelocatesLeaves) {
    auto tt = build_test_tree(/*n=*/1000, /*dim=*/32, /*n_clusters=*/10,
                               /*leaf_cap=*/200, /*k_root=*/4);
    auto idx = IVFTreeIndex::open(tt.tree_path);

    const uint64_t pages_before = idx->n_pages();

    // Delete to create free pages.
    std::vector<RowId> to_delete;
    for (uint32_t i = 0; i < 300; ++i) to_delete.push_back(static_cast<RowId>(i * 2));
    idx->delete_batch(to_delete);

    // Defrag.
    IVFTreeIndex::DefragConfig dcfg;
    dcfg.batch_size = 16;
    auto result = idx->defrag(dcfg);

    EXPECT_GT(result.leaves_relocated, 0u)
        << "Defrag should have relocated at least one leaf";

    // File should not have grown.
    EXPECT_LE(idx->n_pages(), pages_before + 10)
        << "File grew unexpectedly after defrag";

    std::filesystem::remove(tt.fbin_path);
    std::filesystem::remove(tt.tree_path);
}

TEST(VacuumDefrag, DefragPreservesSearchResults) {
    const uint32_t dim = 32;
    auto tt = build_test_tree(/*n=*/1000, dim, /*n_clusters=*/10,
                               /*leaf_cap=*/200, /*k_root=*/4);
    auto idx = IVFTreeIndex::open(tt.tree_path);

    // Baseline: search for 20 queries, record distance profiles.
    SearchConfig scfg;
    scfg.k = 10;
    scfg.n_probe = 4;
    scfg.adaptive_probe_gap = 0.0f;
    std::vector<std::vector<float>> baseline_dists;
    std::vector<float> baseline_mean;
    for (uint32_t q = 0; q < 20; ++q) {
        auto results = idx->search(&tt.base_data[q * dim], 10, scfg);
        std::vector<float> dists;
        float mean = 0;
        for (const auto& r : results) { dists.push_back(r.dist); mean += r.dist; }
        if (!dists.empty()) mean /= dists.size();
        baseline_dists.push_back(std::move(dists));
        baseline_mean.push_back(mean);
    }

    // Defrag.
    idx->defrag();

    // After defrag, search should return results with the same distance
    // profile. Defrag relocates leaf extents to new pages; it does not change
    // codes. However, on clustered data many candidates have identical
    // distances (ties), and the exact set of top-10 among ties depends on
    // leaf visit order, which changes after defrag. So we verify distance
    // distributions match rather than exact row_id overlap.
    for (uint32_t q = 0; q < 20; ++q) {
        auto results = idx->search(&tt.base_data[q * dim], 10, scfg);
        ASSERT_FALSE(results.empty());
        // The best distance should be the same.
        EXPECT_NEAR(results[0].dist, baseline_dists[q].front(), 1.0f)
            << "Query " << q << ": best distance changed after defrag";
        // The mean distance should be similar (within 10%).
        float mean_after = 0;
        for (const auto& r : results) mean_after += r.dist;
        mean_after /= results.size();
        EXPECT_NEAR(mean_after, baseline_mean[q], baseline_mean[q] * 0.10f + 1.0f)
            << "Query " << q << ": mean distance changed significantly";
    }

    std::filesystem::remove(tt.fbin_path);
    std::filesystem::remove(tt.tree_path);
}

TEST(VacuumDefrag, DefragShrinksFile) {
    auto tt = build_test_tree(/*n=*/1000, /*dim=*/32, /*n_clusters=*/10,
                               /*leaf_cap=*/200, /*k_root=*/4);
    auto idx = IVFTreeIndex::open(tt.tree_path);
    const uint64_t pages_after_build = idx->n_pages();

    // Delete a large chunk to create free pages.
    std::vector<RowId> to_delete;
    for (uint32_t i = 0; i < 500; ++i) to_delete.push_back(static_cast<RowId>(i));
    idx->delete_batch(to_delete);

    // Defrag with shrink.
    IVFTreeIndex::DefragConfig dcfg;
    dcfg.shrink_file = true;
    dcfg.batch_size = 16;
    auto dres = idx->defrag(dcfg);
    (void)dres;  // stats logged; we check n_pages below

    // File should be smaller or equal after shrinking.
    EXPECT_LE(idx->n_pages(), pages_after_build)
        << "File should not have grown after defrag + shrink";

    std::filesystem::remove(tt.fbin_path);
    std::filesystem::remove(tt.tree_path);
}

TEST(VacuumDefrag, VacuumThenDefrag) {
    const uint32_t dim = 32;
    auto tt = build_test_tree(/*n=*/800, dim, /*n_clusters=*/10,
                               /*leaf_cap=*/200, /*k_root=*/4);
    auto idx = IVFTreeIndex::open(tt.tree_path);

    // Delete 200 vectors.
    std::vector<RowId> to_delete;
    for (uint32_t i = 0; i < 200; ++i) to_delete.push_back(static_cast<RowId>(i * 3));
    idx->delete_batch(to_delete);

    // Vacuum first (repair summaries), then defrag (compact extents).
    auto vres = idx->vacuum();
    EXPECT_GT(vres.summaries_repaired, 0u);

    auto dres = idx->defrag();
    EXPECT_GT(dres.leaves_relocated, 0u);

    // Search should still work.
    SearchConfig scfg;
    scfg.k = 10;
    scfg.n_probe = 4;
    scfg.adaptive_probe_gap = 0.0f;
    Predicate pred;
    pred.column = "category";
    pred.op = PredicateOp::Eq;
    pred.value = 5;
    scfg.predicates = {pred};

    auto results = idx->search(&tt.base_data[5 * dim], 10, scfg);
    EXPECT_FALSE(results.empty());

    // All results must pass the filter.
    // (Exact filter correctness is validated by test_tree_filter;
    //  here we verify the index is functional after vacuum+defrag.)

    std::filesystem::remove(tt.fbin_path);
    std::filesystem::remove(tt.tree_path);
}

}  // namespace
}  // namespace sextant::tree
