#include <gtest/gtest.h>
#include "fbin_source.hpp"
#include "tree/ivf_tree_index.hpp"
#include "tree/tree_nodes.hpp"
#include "tree/page_file.hpp"
#include "tree/superblock.hpp"
#include "tree/filter_scan.hpp"  // eval_predicate_geo, haversine_km, summary_may_match
#include <sextant/column_data.hpp>
#include <sextant/error.hpp>
#include "sextant/config.hpp"
#include "sextant/schema.hpp"
#include "sextant/types.hpp"
#include "mem_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace sextant::tree {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write random clustered data to an fbin file. Returns the path.
/// row_id i is assigned to cluster (i % n_clusters).
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
        const uint32_t ci = static_cast<uint32_t>(i % n_clusters);
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

/// Build a std::vector<ColumnData> for a schema with:
///   col 0: int32  "year"
///   col 1: string "category"
/// row_id i → year = 2000 + (i % 50), category = "cat_<i % 10>".
/// These are global (indexed by row_id 0..N-1), matching the build's expectations.
std::vector<ColumnData> make_filter_data_int32_string(uint64_t n) {
    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});
    schema.columns.push_back({"category", ColumnType::String});

    std::vector<ColumnData> cols(2);
    cols[0].type = ColumnType::Int32;
    cols[1].type = ColumnType::String;

    // Int32 "year".
    cols[0].fixed_data.resize(n * 4);
    for (uint64_t i = 0; i < n; ++i) {
        const int32_t year = 2000 + static_cast<int32_t>(i % 50);
        std::memcpy(&cols[0].fixed_data[i * 4], &year, 4);
    }

    // String "category".
    for (uint64_t i = 0; i < n; ++i) {
        std::string val = "cat_" + std::to_string(i % 10);
        cols[1].str_offsets.push_back(
            static_cast<uint32_t>(cols[1].str_data.size()));
        cols[1].str_lengths.push_back(static_cast<uint16_t>(val.size()));
        cols[1].str_data.insert(cols[1].str_data.end(),
                                val.data(), val.data() + val.size());
    }
    return cols;
}

/// Walk the tree file to find the FIRST leaf extent. Returns its (page, pages)
/// and fills `out_buf` with the leaf's bytes. Brute-force scans data pages for
/// the leaf magic — the leaf header itself stores summary_size, so we learn
/// everything once found.
struct LeafExtent { PageId page; uint64_t pages; };
LeafExtent find_first_leaf(const std::string& tree_path,
                           std::vector<uint8_t>& out_buf) {
    PageFile file(tree_path);
    Superblock sb;
    sb.load(file);

    const uint64_t n_pages = sb.n_pages();
    std::vector<uint8_t> page_buf(kPageSize);
    for (PageId p = kFirstDataPage; p < n_pages; ++p) {
        file.read_pages(p, 1, page_buf.data());
        uint32_t magic;
        std::memcpy(&magic, page_buf.data(), 4);
        if (magic == kTreeLeafMagic) {
            TreeLeafHeader lh;
            std::memcpy(&lh, page_buf.data(), sizeof(lh));
            out_buf.resize(static_cast<size_t>(lh.extent_pages) * kPageSize);
            file.read_pages(p, lh.extent_pages, out_buf.data());
            return {p, lh.extent_pages};
        }
    }
    return {kInvalidPage, 0};
}

// ===========================================================================
// Phase C: build with filter columns and verify on-disk leaf layout.
// ===========================================================================

TEST(TreeFilterColumns, BuildAndVerifyLeafLayout) {
    const uint64_t n = 2000;
    const uint32_t dim = 32;
    const uint32_t n_clusters = 20;
    const std::string base_path = write_test_fbin("tree_filter_base.fbin",
                                                   n, dim, n_clusters, 7);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_filter.tree").string();
    std::filesystem::remove(tree_path);

    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});
    schema.columns.push_back({"category", ColumnType::String});

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = make_filter_data_int32_string(n);

    auto result = ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    EXPECT_EQ(result.n_vectors, n);

    auto idx = IVFTreeIndex::open(tree_path);
    EXPECT_GT(idx->n_leaves(), 0u);

    // --- Verify a leaf's header + summary + filter column region ---
    std::vector<uint8_t> leaf_buf;
    LeafExtent le = find_first_leaf(tree_path, leaf_buf);
    ASSERT_NE(le.page, kInvalidPage);

    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_buf.data());
    EXPECT_EQ(lh->magic, kTreeLeafMagic);
    EXPECT_EQ(lh->n_filter_columns, 2u);
    EXPECT_GT(lh->summary_size, 0u);
    // filter_columns_offset points past factors (factors = 0 for PQ).
    const uint64_t fc_off_expected =
        leaf_factors_offset(lh->summary_size,
                            (static_cast<uint32_t>(lh->count) +
                             lh->codes_per_block - 1) / lh->codes_per_block,
                            lh->block_bytes,
                            static_cast<uint32_t>(lh->count));
    EXPECT_EQ(lh->filter_columns_offset, fc_off_expected);

    // --- Verify the summary region: numeric min/max for "year" ---
    const uint8_t* summary = leaf_buf.data() + leaf_filter_offset();
    uint8_t n_numeric = summary[0];
    ASSERT_EQ(n_numeric, 1u);  // only "year" is numeric
    uint8_t col_id = summary[1];
    EXPECT_EQ(col_id, 0u);  // "year" is column 0
    double min_val, max_val;
    std::memcpy(&min_val, summary + 2, 8);
    std::memcpy(&max_val, summary + 10, 8);
    // year values are 2000..2049; min ≥ 2000, max ≤ 2049.
    EXPECT_GE(min_val, 2000.0);
    EXPECT_LE(min_val, 2049.0);
    EXPECT_GE(max_val, 2000.0);
    EXPECT_LE(max_val, 2049.0);
    EXPECT_LE(min_val, max_val);

    // --- Verify the filter column data region: int32 "year" column ---
    // Layout after factors: [int32 year: count × 4][string category: ...]
    const uint32_t count = static_cast<uint32_t>(lh->count);
    const uint8_t* fc = leaf_buf.data() + lh->filter_columns_offset;
    // The "year" column is the first fixed-width column.
    for (uint32_t i = 0; i < count; ++i) {
        int32_t v;
        std::memcpy(&v, fc + i * 4, 4);
        EXPECT_GE(v, 2000);
        EXPECT_LE(v, 2049);
    }

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Phase C: search still works with filter columns present (no predicates).
// ===========================================================================

TEST(TreeFilterColumns, SearchUnaffectedByFilterColumns) {
    const uint64_t n = 3000;
    const uint32_t dim = 48;
    const uint32_t n_clusters = 30;
    const std::string base_path = write_test_fbin("tree_filter_search.fbin",
                                                   n, dim, n_clusters, 11);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_filter_search.tree").string();
    std::filesystem::remove(tree_path);

    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 12;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = make_filter_data_int32_string(n);

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);

    // Regenerate cluster centers (seed must match write_test_fbin).
    std::mt19937 rng(11);
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }

    SearchConfig sconfig;
    sconfig.k = 10;
    sconfig.n_probe = 8;
    sconfig.n_probe_ln = 8;

    uint32_t total_returned = 0;
    for (uint32_t trial = 0; trial < 10; ++trial) {
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d)
            query[d] = centers[trial % n_clusters][d];
        auto results = idx->search(query.data(), 10, sconfig);
        EXPECT_LE(results.size(), 10u);
        EXPECT_GE(results.size(), 1u);
        total_returned += static_cast<uint32_t>(results.size());
    }
    EXPECT_GT(total_returned, 0u);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Phase C: no filter columns → layout identical to pre-Phase-C (offset = 0).
// ===========================================================================

TEST(TreeFilterColumns, NoFilterColumnsPreservesLayout) {
    const uint64_t n = 1000;
    const uint32_t dim = 32;
    const std::string base_path = write_test_fbin("tree_filter_none.fbin",
                                                   n, dim, 10, 5);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_filter_none.tree").string();
    std::filesystem::remove(tree_path);

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    // No filter_schema, no filter_column_data.

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);

    std::vector<uint8_t> leaf_buf;
    LeafExtent le = find_first_leaf(tree_path, leaf_buf);
    ASSERT_NE(le.page, kInvalidPage);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_buf.data());
    EXPECT_EQ(lh->n_filter_columns, 0u);
    EXPECT_EQ(lh->summary_size, 0u);
    EXPECT_EQ(lh->filter_columns_offset, 0u);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Phase C: schema with int32 + int64 + float + bool (all numeric summary).
// ===========================================================================

TEST(TreeFilterColumns, AllNumericSummary) {
    const uint64_t n = 1500;
    const uint32_t dim = 32;
    const std::string base_path = write_test_fbin("tree_filter_numeric.fbin",
                                                   n, dim, 15, 9);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_filter_numeric.tree").string();
    std::filesystem::remove(tree_path);

    Schema schema;
    schema.columns.push_back({"a", ColumnType::Int32});
    schema.columns.push_back({"b", ColumnType::Int64});
    schema.columns.push_back({"c", ColumnType::Float});
    schema.columns.push_back({"d", ColumnType::Bool});

    std::vector<ColumnData> cols(4);
    cols[0].type = ColumnType::Int32;
    cols[1].type = ColumnType::Int64;
    cols[2].type = ColumnType::Float;
    cols[3].type = ColumnType::Bool;
    cols[0].fixed_data.resize(n * 4);
    cols[1].fixed_data.resize(n * 8);
    cols[2].fixed_data.resize(n * 4);
    cols[3].fixed_data.resize(n * 1);
    for (uint64_t i = 0; i < n; ++i) {
        const int32_t a = static_cast<int32_t>(i % 100);
        const int64_t b = static_cast<int64_t>(i * 7);
        const float c = static_cast<float>(i) * 1.5f;
        const uint8_t d = (i % 2 == 0) ? 1 : 0;
        std::memcpy(&cols[0].fixed_data[i * 4], &a, 4);
        std::memcpy(&cols[1].fixed_data[i * 8], &b, 8);
        std::memcpy(&cols[2].fixed_data[i * 4], &c, 4);
        cols[3].fixed_data[i] = d;
    }

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = std::move(cols);

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);

    std::vector<uint8_t> leaf_buf;
    LeafExtent le = find_first_leaf(tree_path, leaf_buf);
    ASSERT_NE(le.page, kInvalidPage);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_buf.data());
    EXPECT_EQ(lh->n_filter_columns, 4u);
    EXPECT_GT(lh->summary_size, 0u);

    // Summary: 3 numeric columns (int32, int64, float; bool has no summary entry).
    const uint8_t* summary = leaf_buf.data() + leaf_filter_offset();
    EXPECT_EQ(summary[0], 3u);  // n_numeric

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Validation: oversize strings/sets must throw (not silently truncate)
// ===========================================================================

TEST(TreeFilterColumns, ValidationRejectsOversizeString) {
    MemSourceBuilder b(/*dim=*/4);
    Schema schema;
    schema.columns.push_back({"s", ColumnType::String});
    b.set_schema(schema);

    float vec[4] = {0, 0, 0, 0};
    b.add_vector(vec, 0);

    // 65536 bytes > uint16 max (65535).
    std::string big(65536, 'x');
    EXPECT_THROW(b.set_string(0, big), sextant::Error);
}

TEST(TreeFilterColumns, ValidationRejectsOversizeSetCount) {
    MemSourceBuilder b(/*dim=*/4);
    Schema schema;
    schema.columns.push_back({"tags", ColumnType::Set});
    b.set_schema(schema);

    float vec[4] = {0, 0, 0, 0};
    b.add_vector(vec, 0);

    // 256 elements > uint8 max (255).
    std::vector<std::string_view> elems(256, "x");
    EXPECT_THROW(b.set_set(0, elems), sextant::Error);
}

TEST(TreeFilterColumns, ValidationRejectsOversizeSetElement) {
    MemSourceBuilder b(/*dim=*/4);
    Schema schema;
    schema.columns.push_back({"tags", ColumnType::Set});
    b.set_schema(schema);

    float vec[4] = {0, 0, 0, 0};
    b.add_vector(vec, 0);

    // Single element > uint16 max.
    std::string big(65536, 'y');
    std::vector<std::string_view> elems = {std::string_view(big)};
    EXPECT_THROW(b.set_set(0, elems), sextant::Error);
}

// ===========================================================================
// Phase D: filtered search — predicates are applied during the scan.
// ===========================================================================

TEST(TreeFilterColumns, FilteredSearchInt32Equality) {
    const uint64_t n = 3000;
    const uint32_t dim = 48;
    const uint32_t n_clusters = 30;
    const std::string base_path = write_test_fbin("tree_filter_eq.fbin",
                                                   n, dim, n_clusters, 23);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_filter_eq.tree").string();
    std::filesystem::remove(tree_path);

    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});
    schema.columns.push_back({"category", ColumnType::String});

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 12;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = make_filter_data_int32_string(n);

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);

    // Regenerate cluster centers (seed must match write_test_fbin).
    std::mt19937 rng(23);
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }

    // Ground truth: year = 2000 + (i % 50). year==2020 ⇒ (i % 50) == 20.
    auto year_matches = [&](int64_t row_id) {
        return (static_cast<uint64_t>(row_id) % 50) == 20;
    };

    const uint32_t k = 10;
    SearchConfig base_sconfig;
    base_sconfig.k = k;
    base_sconfig.n_probe = 8;
    base_sconfig.n_probe_ln = 8;

    // (1) Filtered search: year == 2020. ALL results must satisfy the predicate.
    SearchConfig fsconfig = base_sconfig;
    Predicate pred;
    pred.column = "year";
    pred.op = PredicateOp::Eq;
    pred.value = 2020.0;
    fsconfig.predicates.push_back(pred);

    uint32_t filtered_returned = 0;
    for (uint32_t trial = 0; trial < 10; ++trial) {
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d)
            query[d] = centers[trial % n_clusters][d];
        auto results = idx->search(query.data(), k, fsconfig);
        for (const auto& r : results) {
            // Every returned candidate must pass the predicate.
            EXPECT_TRUE(year_matches(r.row_id))
                << "row_id " << r.row_id << " failed year==2020 filter";
        }
        filtered_returned += static_cast<uint32_t>(results.size());
    }

    // (2) Unfiltered search: no predicates. Should return ≥ as many candidates
    //     overall (the filter rejects some), and results need not match year.
    uint32_t unfiltered_returned = 0;
    for (uint32_t trial = 0; trial < 10; ++trial) {
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d)
            query[d] = centers[trial % n_clusters][d];
        auto results = idx->search(query.data(), k, base_sconfig);
        unfiltered_returned += static_cast<uint32_t>(results.size());
    }
    EXPECT_GE(unfiltered_returned, filtered_returned);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

TEST(TreeFilterColumns, FilteredSearchStringEquality) {
    const uint64_t n = 3000;
    const uint32_t dim = 48;
    const uint32_t n_clusters = 30;
    const std::string base_path = write_test_fbin("tree_filter_streq.fbin",
                                                   n, dim, n_clusters, 29);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_filter_streq.tree").string();
    std::filesystem::remove(tree_path);

    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});
    schema.columns.push_back({"category", ColumnType::String});

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 12;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = make_filter_data_int32_string(n);

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);

    // Regenerate cluster centers (seed must match write_test_fbin).
    std::mt19937 rng(29);
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }

    // Ground truth: category = "cat_" + (i % 10). "cat_3" ⇒ (i % 10) == 3.
    auto cat_matches = [&](int64_t row_id) {
        return (static_cast<uint64_t>(row_id) % 10) == 3;
    };

    const uint32_t k = 10;
    SearchConfig fsconfig;
    fsconfig.k = k;
    fsconfig.n_probe = 8;
    fsconfig.n_probe_ln = 8;
    Predicate pred;
    pred.column = "category";
    pred.op = PredicateOp::Eq;
    pred.str_value = "cat_3";
    fsconfig.predicates.push_back(pred);

    for (uint32_t trial = 0; trial < 10; ++trial) {
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d)
            query[d] = centers[trial % n_clusters][d];
        auto results = idx->search(query.data(), k, fsconfig);
        for (const auto& r : results) {
            EXPECT_TRUE(cat_matches(r.row_id))
                << "row_id " << r.row_id << " failed category==cat_3 filter";
        }
    }

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Phase E: internal node (root + L2) child summaries are populated.
// Verifies bottom-up summary propagation: root child entries and the L2 node
// child entries they point to must carry non-zero filter summaries.
// ===========================================================================
TEST(TreeFilterColumns, InternalNodeSummariesPopulated) {
    const uint64_t n = 2000;
    const uint32_t dim = 32;
    const uint32_t n_clusters = 20;
    const std::string base_path = write_test_fbin("tree_filter_intnode.fbin",
                                                   n, dim, n_clusters, 7);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_filter_intnode.tree").string();
    std::filesystem::remove(tree_path);

    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});
    schema.columns.push_back({"category", ColumnType::String});

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;        // < k_root_max_depth2 (512) → depth=2
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = make_filter_data_int32_string(n);

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();

    // Open the file directly to inspect the root node extent.
    PageFile file(tree_path);
    Superblock sb;
    sb.load(file);
    const PageId root_page = sb.root_node_page();
    const uint32_t root_pages = sb.root_node_pages();
    ASSERT_NE(root_page, kInvalidPage);
    ASSERT_GT(root_pages, 0u);

    std::vector<uint8_t> root_buf(static_cast<size_t>(root_pages) * kPageSize);
    file.read_pages(root_page, root_pages, root_buf.data());

    const auto* rh = reinterpret_cast<const TreeNodeHeader*>(root_buf.data());
    ASSERT_EQ(rh->magic, kTreeNodeMagic);
    const uint32_t cesize = child_entry_size(dim, schema.summary_size());
    const uint32_t summary_off = sizeof(ChildEntry) + dim * sizeof(float16_t);

    // --- Verify each root child's summary is populated (numeric min/max) ---
    // The summary region must NOT be all zeros; its numeric min/max for "year"
    // must lie within the global data range [2000, 2049].
    uint32_t populated_root_children = 0;
    for (uint32_t c = 0; c < rh->n_children; ++c) {
        const uint8_t* p = root_buf.data() + sizeof(TreeNodeHeader)
                           + static_cast<uint64_t>(c) * cesize;
        const auto* ce = reinterpret_cast<const ChildEntry*>(p);
        if (ce->child_page == kInvalidPage) continue;  // empty child
        const uint8_t* summary = p + summary_off;
        const uint8_t n_numeric = summary[0];
        ASSERT_EQ(n_numeric, 1u);  // "year" is the only numeric column
        const uint8_t col_id = summary[1];
        EXPECT_EQ(col_id, 0u);
        double min_val, max_val;
        std::memcpy(&min_val, summary + 2, 8);
        std::memcpy(&max_val, summary + 10, 8);
        EXPECT_GE(min_val, 2000.0);
        EXPECT_LE(min_val, 2049.0);
        EXPECT_GE(max_val, 2000.0);
        EXPECT_LE(max_val, 2049.0);
        EXPECT_LE(min_val, max_val);
        ++populated_root_children;
    }
    EXPECT_GT(populated_root_children, 0u);

    // --- Descend into the first non-empty root child (an L2 node) and verify
    // its child entries (leaves) also carry populated summaries ---
    for (uint32_t c = 0; c < rh->n_children; ++c) {
        const uint8_t* p = root_buf.data() + sizeof(TreeNodeHeader)
                           + static_cast<uint64_t>(c) * cesize;
        const auto* ce = reinterpret_cast<const ChildEntry*>(p);
        if (ce->child_page == kInvalidPage) continue;
        ASSERT_EQ(ce->is_leaf, 0u);  // depth=2: root child is an internal node

        std::vector<uint8_t> node_buf(
            static_cast<size_t>(ce->child_pages) * kPageSize);
        file.read_pages(ce->child_page, ce->child_pages, node_buf.data());
        const auto* nh = reinterpret_cast<const TreeNodeHeader*>(
            node_buf.data());
        ASSERT_EQ(nh->magic, kTreeNodeMagic);

        uint32_t populated_leaves = 0;
        for (uint32_t j = 0; j < nh->n_children; ++j) {
            const uint8_t* cp = node_buf.data() + sizeof(TreeNodeHeader)
                                + static_cast<uint64_t>(j) * cesize;
            const auto* lce = reinterpret_cast<const ChildEntry*>(cp);
            if (lce->child_page == kInvalidPage) continue;
            const uint8_t* summary = cp + summary_off;
            // Leaf summaries must be populated (Phase C) and now also copied
            // into the L2 child entry (Phase E propagation).
            EXPECT_EQ(summary[0], 1u);  // n_numeric
            double min_val, max_val;
            std::memcpy(&min_val, summary + 2, 8);
            std::memcpy(&max_val, summary + 10, 8);
            EXPECT_GE(min_val, 2000.0);
            EXPECT_LE(max_val, 2049.0);
            EXPECT_LE(min_val, max_val);
            ++populated_leaves;
        }
        EXPECT_GT(populated_leaves, 0u);
        break;  // only inspect the first non-empty L2 node
    }

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Phase D: cardinality table is built, serialized, and loaded at open time.
// Verifies the global selectivity table round-trips and produces a reasonable
// estimate for a known-frequency string column.
// ===========================================================================
TEST(TreeFilterColumns, CardinalityTableSerializedAndLoaded) {
    const uint64_t n = 3000;
    const uint32_t dim = 48;
    const uint32_t n_clusters = 30;
    const std::string base_path = write_test_fbin("tree_filter_card.fbin",
                                                   n, dim, n_clusters, 81);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                   "tree_filter_card.tree").string();
    std::filesystem::remove(tree_path);

    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});
    schema.columns.push_back({"category", ColumnType::String});

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 12;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = make_filter_data_int32_string(n);

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);

    // The cardinality table must have been serialized and loaded.
    ASSERT_FALSE(idx->cardinality().empty());
    EXPECT_EQ(idx->cardinality().n_vectors(), n);

    // category = "cat_" + (i % 10). Each value appears exactly n/10 times, so
    // selectivity for any single value is ~0.1. category is schema column 1.
    const float sel = idx->cardinality().selectivity_string(1, "cat_3");
    EXPECT_NEAR(sel, 0.1f, 0.02f);

    // A value that never appears should report ~0 selectivity.
    EXPECT_NEAR(idx->cardinality().selectivity_string(1, "cat_99"), 0.0f, 1e-6f);

    // search() with the equality predicate must still return only matching rows.
    std::mt19937 rng(81);
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }
    auto cat_matches = [&](int64_t row_id) {
        return (static_cast<uint64_t>(row_id) % 10) == 3;
    };

    const uint32_t k = 10;
    SearchConfig fsconfig;
    fsconfig.k = k;
    fsconfig.n_probe = 8;
    fsconfig.n_probe_ln = 8;
    Predicate pred;
    pred.column = "category";
    pred.op = PredicateOp::Eq;
    pred.str_value = "cat_3";
    fsconfig.predicates.push_back(pred);

    for (uint32_t trial = 0; trial < 5; ++trial) {
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d)
            query[d] = centers[trial % n_clusters][d];
        auto results = idx->search(query.data(), k, fsconfig);
        for (const auto& r : results) {
            EXPECT_TRUE(cat_matches(r.row_id))
                << "row_id " << r.row_id << " failed category==cat_3 filter";
        }
    }

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Numeric selectivity: extreme low selectivity triggers brute-force fallback
// ===========================================================================

TEST(TreeFilterColumns, NumericLowSelectivityBruteForceFallback) {
    // Build a tree where year values are spread across [2000..2099].
    // Predicate year == 2099 matches ~1% (1 value out of 100).
    // This should trigger the brute-force fallback (<1% estimated selectivity
    // from root summaries) and return correct results.
    const uint64_t n = 5000;
    const uint32_t dim = 32;

    auto base_path = write_test_fbin("tree_bf_numeric.fbin", n, dim, 50, 88);
    auto tree_path = (std::filesystem::temp_directory_path() /
                      "tree_bf_numeric.tree").string();

    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});

    std::vector<ColumnData> cols(1);
    cols[0].type = ColumnType::Int32;
    cols[0].fixed_data.resize(n * 4);
    for (uint64_t i = 0; i < n; ++i) {
        const int32_t year = 2000 + static_cast<int32_t>(i % 100);
        std::memcpy(&cols[0].fixed_data[i * 4], &year, 4);
    }

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.k_root = 16;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = std::move(cols);

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);

    // Predicate: year == 2099. Only i % 100 == 99 rows match (~1%).
    SearchConfig scfg;
    scfg.k = 10;
    scfg.n_probe = 16;
    scfg.n_probe_ln = 8;
    Predicate pred;
    pred.column = "year";
    pred.op = PredicateOp::Eq;
    pred.value = 2099.0;
    scfg.predicates.push_back(pred);

    auto results = idx->search(
        &std::vector<float>(dim, 0.5f)[0], 10, scfg);

    // Every result must have year == 2099 (i.e., row_id % 100 == 99).
    for (const auto& c : results) {
        EXPECT_EQ(c.row_id % 100, 99)
            << "row_id " << c.row_id << " has wrong year";
    }

    // Should return some results (brute-force scans all matching leaves).
    EXPECT_GE(results.size(), 1u);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Phase E: opaque payload round-trip.
// Build with has_payload=true + per-row blobs, search, fetch_payload, verify.
// ===========================================================================

TEST(TreeFilterColumns, PayloadRoundTrip) {
    const uint64_t n = 2000;
    const uint32_t dim = 32;
    const uint32_t n_clusters = 20;
    const std::string base_path = write_test_fbin(
        "tree_payload_rt.fbin", n, dim, n_clusters, 31337);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                    "tree_payload_rt.tree").string();
    std::filesystem::remove(tree_path);

    // Schema with a filter column + payload flag.
    Schema schema;
    schema.columns.push_back({"year", ColumnType::Int32});
    schema.has_payload = true;

    // Build known payload blobs: payload = "payload_<row_id>" (variable length
    // so offsets are exercised). Packed into payload_data with N+1 offsets.
    std::vector<uint8_t> payload_data;
    std::vector<uint32_t> payload_offsets(n + 1);
    for (uint64_t i = 0; i < n; ++i) {
        payload_offsets[i] = static_cast<uint32_t>(payload_data.size());
        std::string blob = "payload_" + std::to_string(i);
        payload_data.insert(payload_data.end(), blob.begin(), blob.end());
    }
    payload_offsets[n] = static_cast<uint32_t>(payload_data.size());

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    cfg.filter_schema = schema;
    cfg.filter_column_data = make_filter_data_int32_string(n);
    cfg.payload_data = payload_data.data();
    cfg.payload_offsets = payload_offsets.data();

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);
    EXPECT_GT(idx->n_leaves(), 0u);

    // Regenerate cluster centers (seed must match write_test_fbin).
    std::mt19937 rng(31337);
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution<float>(-10, 10)(rng);
    }

    SearchConfig sconfig;
    sconfig.k = 10;
    sconfig.n_probe = 8;
    sconfig.n_probe_ln = 8;

    uint32_t verified = 0;
    for (uint32_t trial = 0; trial < 10; ++trial) {
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d)
            query[d] = centers[trial % n_clusters][d];

        std::vector<std::pair<const uint8_t*, uint32_t>> locs;
        auto results = idx->search(query.data(), 10, sconfig, &locs);
        ASSERT_EQ(results.size(), locs.size());
        for (uint32_t ri = 0; ri < results.size(); ++ri) {
            const auto blob = idx->fetch_payload(locs[ri].first, locs[ri].second);
            const std::string expected =
                "payload_" + std::to_string(results[ri].row_id);
            EXPECT_EQ(blob.size(), expected.size())
                << "row_id " << results[ri].row_id << " payload size mismatch";
            EXPECT_TRUE(std::equal(blob.begin(), blob.end(), expected.begin()))
                << "row_id " << results[ri].row_id << " payload content mismatch";
            ++verified;
        }
    }
    EXPECT_GT(verified, 0u);

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// Phase E: fetch_payload on a leaf with no payload returns empty.
// ===========================================================================

TEST(TreeFilterColumns, PayloadAbsentReturnsEmpty) {
    const uint64_t n = 1000;
    const uint32_t dim = 32;
    const std::string base_path = write_test_fbin(
        "tree_payload_none.fbin", n, dim, 10, 5);
    const std::string tree_path = (std::filesystem::temp_directory_path() /
                                    "tree_payload_none.tree").string();
    std::filesystem::remove(tree_path);

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "pq";
    cfg.params.pq4_m = 8;
    cfg.params.scan_pq_bits = 4;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 500;
    cfg.pca_dims = 16;
    cfg.max_lloyd_passes = 2;
    cfg.num_threads = 4;
    // No payload.

    ([&]{ FbinSource s(base_path); return IVFTreeIndex::build_streaming_pca(s, tree_path, cfg); })();
    auto idx = IVFTreeIndex::open(tree_path);

    SearchConfig sconfig;
    sconfig.k = 5;
    std::vector<float> query(dim, 0.5f);
    std::vector<std::pair<const uint8_t*, uint32_t>> locs;
    auto results = idx->search(query.data(), 5, sconfig, &locs);
    for (const auto& loc : locs) {
        // With no payload extent, fetch must return empty.
        EXPECT_TRUE(idx->fetch_payload(loc.first, loc.second).empty());
    }

    std::filesystem::remove(base_path);
    std::filesystem::remove(tree_path);
}

// ---------------------------------------------------------------------------
// Geo predicate unit tests (GeoBox, GeoRadius)
// ---------------------------------------------------------------------------
// These exercise eval_predicate_geo, haversine_km, and summary_may_match
// directly, without building a full index.

/// Build a Float ColumnView over a small lat/lng array.
ColumnView make_float_view(const std::vector<float>& vals) {
    ColumnView v;
    v.type = ColumnType::Float;
    v.fixed_width = 4;
    // Persist the data: leak intentionally (test-only, process-exit reclaims).
    auto* buf = new float[vals.size()];
    std::memcpy(buf, vals.data(), vals.size() * sizeof(float));
    v.fixed_base = reinterpret_cast<const uint8_t*>(buf);
    return v;
}

TEST(GeoPredicate, HaversineSanity) {
    // NYC → London is ~5570 km.
    const double d = haversine_km(40.7128, -74.0060, 51.5074, -0.1278);
    EXPECT_NEAR(d, 5570.0, 50.0);
    // Zero distance to self.
    EXPECT_NEAR(haversine_km(37.0, -122.0, 37.0, -122.0), 0.0, 1e-6);
    // 1° latitude ≈ 111 km.
    EXPECT_NEAR(haversine_km(0.0, 0.0, 1.0, 0.0), 111.0, 1.0);
}

TEST(GeoPredicate, EvalGeoBox) {
    // lat in [37.0, 38.0], lng in [-123.0, -121.0].
    Predicate pred;
    pred.op = PredicateOp::GeoBox;
    pred.value = 37.0;   pred.value2 = -123.0;  // min_lat, min_lng
    pred.value3 = 38.0;  pred.value4 = -121.0;  // max_lat, max_lng
    pred.geo_lng_column = "lng";

    std::vector<float> lats = {37.5f, 38.5f, 37.5f, 37.5f};
    std::vector<float> lngs = {-122.0f, -122.0f, -124.0f, -122.0f};
    ColumnView lat_col = make_float_view(lats);
    ColumnView lng_col = make_float_view(lngs);

    EXPECT_TRUE (eval_predicate_geo(lat_col, lng_col, 0, pred));  // inside
    EXPECT_FALSE(eval_predicate_geo(lat_col, lng_col, 1, pred));  // lat too high
    EXPECT_FALSE(eval_predicate_geo(lat_col, lng_col, 2, pred));  // lng too low
    EXPECT_TRUE (eval_predicate_geo(lat_col, lng_col, 3, pred));  // inside
}

TEST(GeoPredicate, EvalGeoRadius) {
    // Center at (37.0, -122.0), radius 10 km.
    Predicate pred;
    pred.op = PredicateOp::GeoRadius;
    pred.value = 37.0;     // center lat
    pred.value2 = -122.0;  // center lng
    pred.radius_km = 10.0;
    pred.geo_lng_column = "lng";

    // ~0.05° lat ≈ 5.5 km (inside). ~0.2° ≈ 22 km (outside).
    std::vector<float> lats = {37.05f, 37.2f};
    std::vector<float> lngs = {-122.0f, -122.0f};
    ColumnView lat_col = make_float_view(lats);
    ColumnView lng_col = make_float_view(lngs);

    EXPECT_TRUE (eval_predicate_geo(lat_col, lng_col, 0, pred));  // ~5.5 km
    EXPECT_FALSE(eval_predicate_geo(lat_col, lng_col, 1, pred));  // ~22 km
}

TEST(GeoPredicate, SummaryMayMatchGeoBox) {
    // Build a summary with two numeric columns: lat (col 0), lng (col 1).
    // Summary format: [n_numeric][col_id, min f64, max f64]... then n_string(0),
    // n_set(0).
    std::vector<uint8_t> sum;
    auto push_u8 = [&](uint8_t x) { sum.push_back(x); };
    auto push_f64 = [&](double x) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&x);
        sum.insert(sum.end(), p, p + 8);
    };
    push_u8(2);                 // 2 numeric entries
    push_u8(0); push_f64(36.0); push_f64(39.0);  // col 0 (lat): [36, 39]
    push_u8(1); push_f64(-124.0); push_f64(-120.0);  // col 1 (lng): [-124, -120]
    push_u8(0);  // n_string
    push_u8(0);  // n_set

    Schema schema;
    schema.columns.push_back({"lat", ColumnType::Float});
    schema.columns.push_back({"lng", ColumnType::Float});

    // Box overlapping both ranges → may match.
    Predicate overlap;
    overlap.column = "lat";
    overlap.op = PredicateOp::GeoBox;
    overlap.value = 37.0; overlap.value2 = -123.0;
    overlap.value3 = 38.0; overlap.value4 = -121.0;
    overlap.geo_lng_column = "lng";

    // Box completely north of lat range → no match.
    Predicate north;
    north.column = "lat";
    north.op = PredicateOp::GeoBox;
    north.value = 40.0; north.value2 = -123.0;
    north.value3 = 41.0; north.value4 = -121.0;
    north.geo_lng_column = "lng";

    // Box lat-ok but east of lng range → no match.
    Predicate east;
    east.column = "lat";
    east.op = PredicateOp::GeoBox;
    east.value = 37.0; east.value2 = -119.0;
    east.value3 = 38.0; east.value4 = -118.0;
    east.geo_lng_column = "lng";

    EXPECT_TRUE (summary_may_match(sum.data(), sum.size(), schema,
                                    {overlap}, {0}, {1}));
    EXPECT_FALSE(summary_may_match(sum.data(), sum.size(), schema,
                                    {north}, {0}, {1}));
    EXPECT_FALSE(summary_may_match(sum.data(), sum.size(), schema,
                                    {east}, {0}, {1}));
}

TEST(GeoPredicate, SummaryMayMatchGeoRadius) {
    // Same summary as above: lat [36,39], lng [-124,-120].
    std::vector<uint8_t> sum;
    auto push_u8 = [&](uint8_t x) { sum.push_back(x); };
    auto push_f64 = [&](double x) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&x);
        sum.insert(sum.end(), p, p + 8);
    };
    push_u8(2);
    push_u8(0); push_f64(36.0); push_f64(39.0);
    push_u8(1); push_f64(-124.0); push_f64(-120.0);
    push_u8(0); push_u8(0);

    Schema schema;
    schema.columns.push_back({"lat", ColumnType::Float});
    schema.columns.push_back({"lng", ColumnType::Float});

    // Center inside, radius 100 km → bounding box overlaps → may match.
    Predicate inside;
    inside.column = "lat";
    inside.op = PredicateOp::GeoRadius;
    inside.value = 37.5; inside.value2 = -122.0;
    inside.radius_km = 100.0;
    inside.geo_lng_column = "lng";

    // Center far away (Alaska), radius 10 km → no overlap → prune.
    Predicate far;
    far.column = "lat";
    far.op = PredicateOp::GeoRadius;
    far.value = 65.0; far.value2 = -150.0;
    far.radius_km = 10.0;
    far.geo_lng_column = "lng";

    EXPECT_TRUE (summary_may_match(sum.data(), sum.size(), schema,
                                    {inside}, {0}, {1}));
    EXPECT_FALSE(summary_may_match(sum.data(), sum.size(), schema,
                                    {far}, {0}, {1}));
}

}  // namespace
}  // namespace sextant::tree
