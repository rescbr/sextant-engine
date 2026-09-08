// Tests for subtree-major batch search (IVFTreeIndex::search_batch).
//
// Core contract: identical inputs (queries, k, config) must produce the
// same results as per-query search() — batch-of-one degenerates to the
// query-major path byte-for-byte, larger batches differ only in I/O
// coalescing, never in results. Also covers: duplicate queries inside a
// batch, hot-set (leaf cache) interaction, batch observability counters,
// and config rejection for feedback probing.

#include <gtest/gtest.h>

#include "tree/ivf_tree_index.hpp"

#include <sextant/error.hpp>
#include <sextant/vector_source.hpp>
#include "fbin_source.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <vector>

namespace {

/// Clustered fbin fixture (same recipe as test_leaf_cache).
std::string write_test_fbin(const std::string& path, uint32_t n, uint32_t dim,
                            uint32_t n_clusters, uint32_t seed = 7) {
    std::mt19937 rng(seed);
    const float spread = 0.05f;
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c)
            v = std::uniform_real_distribution(-1.f, 1.f)(rng);
    }
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (uint32_t i = 0; i < n; ++i) {
        const auto& c = centers[i % n_clusters];
        for (uint32_t d = 0; d < dim; ++d)
            data[static_cast<size_t>(i) * dim + d] =
                c[d] + std::uniform_real_distribution(-spread, spread)(rng);
    }
    std::ofstream f(path, std::ios::binary);
    uint32_t hdr[2] = {n, dim};
    f.write(reinterpret_cast<const char*>(hdr), 8);
    f.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(float));
    f.close();
    return path;
}

struct BatchFixture {
    std::filesystem::path dir;
    std::string tree_path;
    uint32_t dim = 32;

    BatchFixture() {
        dir = std::filesystem::temp_directory_path() / "batch_search_int";
        std::filesystem::create_directories(dir);
        const std::string base =
            write_test_fbin((dir / "base.fbin").string(), 20000, dim, 30);
        tree_path = (dir / "tree").string();
        sextant::FbinSource s(base);
        sextant::tree::IVFTreeIndex::BuildConfig cfg;
        cfg.k_root = 8;
        cfg.leaf_capacity = 1000;
        cfg.pca_dims = 0;  // FP16 routing — simplest deterministic path
        cfg.num_threads = 4;
        cfg.closure_multiplier = 0.0f;
        sextant::tree::IVFTreeIndex::build_streaming_pca(s, tree_path, cfg);
    }
    ~BatchFixture() { std::filesystem::remove_all(dir); }

    /// nq random queries near the cluster centers.
    std::vector<float> make_queries(uint32_t nq, uint32_t seed = 11) const {
        std::mt19937 rng(seed);
        std::vector<float> q(static_cast<size_t>(nq) * dim);
        for (uint32_t i = 0; i < nq; ++i) {
            const uint32_t c = std::uniform_int_distribution(
                0u, 29u)(rng);
            for (uint32_t d = 0; d < dim; ++d)
                q[static_cast<size_t>(i) * dim + d] =
                    std::uniform_real_distribution(-1.f, 1.f)(rng) * 0.1f +
                    (c % 2 ? 0.6f : -0.6f);
        }
        return q;
    }
};

const BatchFixture& fixture() {
    static BatchFixture f;
    return f;
}

sextant::SearchConfig base_config() {
    sextant::SearchConfig sc;
    sc.k = 10;
    sc.n_probe = 4;
    sc.n_probe_ln = 4;
    sc.adaptive_probe_gap = 0.0f;
    sc.fastscan_W = 1000;
    return sc;
}

void expect_same_results(const std::vector<sextant::Candidate>& a,
                         const std::vector<sextant::Candidate>& b,
                         bool exact_order) {
    ASSERT_EQ(a.size(), b.size());
    if (exact_order) {
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i].row_id, b[i].row_id) << "position " << i;
        }
    } else {
        std::set<int64_t> ra, rb;
        for (const auto& c : a) ra.insert(c.row_id);
        for (const auto& c : b) rb.insert(c.row_id);
        EXPECT_EQ(ra, rb);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Result equivalence vs per-query search().
// ---------------------------------------------------------------------------

TEST(BatchSearch, BatchOfOneMatchesSearchExactly) {
    const auto& fx = fixture();
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path);
    auto sc = base_config();
    sc.search_threads = 1;  // bit-exact comparison needs the serial paths
    const auto queries = fx.make_queries(1);
    const auto single = idx->search(queries.data(), 10, sc);
    ASSERT_FALSE(single.empty());
    std::vector<std::vector<sextant::Candidate>> out;
    idx->search_batch(queries.data(), 1, 10, sc, out);
    ASSERT_EQ(out.size(), 1u);
    expect_same_results(single, out[0], /*exact_order=*/true);
}

TEST(BatchSearch, LargerBatchesMatchSearch) {
    const auto& fx = fixture();
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path);
    for (uint32_t nq : {2u, 17u, 64u}) {
        auto sc = base_config();
        sc.search_threads = 1;
        const auto queries = fx.make_queries(nq);
        std::vector<std::vector<sextant::Candidate>> out;
        idx->search_batch(queries.data(), nq, 10, sc, out);
        ASSERT_EQ(out.size(), nq);
        for (uint32_t i = 0; i < nq; ++i) {
            const auto single = idx->search(
                queries.data() + static_cast<size_t>(i) * fx.dim, 10, sc);
            expect_same_results(single, out[i], /*exact_order=*/true);
        }
    }
}

TEST(BatchSearch, ParallelSweepMatchesSearchRowSets) {
    const auto& fx = fixture();
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path);
    auto sc = base_config();
    sc.search_threads = 4;
    const uint32_t nq = 48;
    const auto queries = fx.make_queries(nq);
    std::vector<std::vector<sextant::Candidate>> out;
    idx->search_batch(queries.data(), nq, 10, sc, out);
    ASSERT_EQ(out.size(), nq);
    // Serial per-query baseline: row-id SETS must match exactly; ordering
    // may differ on pq_dist ties under parallelism (same contract as
    // search()'s parallel scan path).
    auto sc1 = sc;
    sc1.search_threads = 1;
    for (uint32_t i = 0; i < nq; ++i) {
        const auto single = idx->search(
            queries.data() + static_cast<size_t>(i) * fx.dim, 10, sc1);
        expect_same_results(single, out[i], /*exact_order=*/false);
    }
}

TEST(BatchSearch, DuplicateQueriesInsideBatch) {
    const auto& fx = fixture();
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path);
    auto sc = base_config();
    sc.search_threads = 1;
    const auto one = fx.make_queries(1);
    std::vector<float> queries;
    for (uint32_t i = 0; i < 9; ++i)
        queries.insert(queries.end(), one.begin(), one.end());
    const auto single = idx->search(one.data(), 10, sc);
    std::vector<std::vector<sextant::Candidate>> out;
    idx->search_batch(queries.data(), 9, 10, sc, out);
    ASSERT_EQ(out.size(), 9u);
    for (auto& r : out) expect_same_results(single, r, /*exact_order=*/true);
    // Coalescing contract: 9 identical queries must sweep each probed leaf
    // exactly ONCE (leaves_unique == leaf_scans / 9).
    const auto bs = idx->batch_stats().snapshot_and_reset();
    ASSERT_GT(bs.leaf_scans, 0u);
    EXPECT_EQ(bs.leaf_scans, 9u * bs.leaves_unique);
}

TEST(BatchSearch, RerankOnParity) {
    const auto& fx = fixture();
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path);
    auto sc = base_config();
    sc.search_threads = 1;
    sc.rerank = 1;
    sc.adaptive_w_gap = 0;  // plain top-k for a clean comparison
    const uint32_t nq = 8;
    const auto queries = fx.make_queries(nq);
    std::vector<std::vector<sextant::Candidate>> out;
    idx->search_batch(queries.data(), nq, 10, sc, out);
    for (uint32_t i = 0; i < nq; ++i) {
        const auto single = idx->search(
            queries.data() + static_cast<size_t>(i) * fx.dim, 10, sc);
        expect_same_results(single, out[i], /*exact_order=*/true);
    }
}

// ---------------------------------------------------------------------------
// Hot-set (leaf cache) interaction + observability.
// ---------------------------------------------------------------------------

TEST(BatchSearch, HotSetParityAndAccounting) {
    const auto& fx = fixture();
    // Cache large enough to hold every leaf of the fixture.
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path, 8ull << 20);
    auto sc = base_config();
    sc.search_threads = 2;
    const uint32_t nq = 16;
    const auto queries = fx.make_queries(nq);
    std::vector<std::vector<sextant::Candidate>> out;
    idx->search_batch(queries.data(), nq, 10, sc, out);  // warm the hot set

    auto sc1 = sc;
    sc1.search_threads = 1;
    for (uint32_t i = 0; i < nq; ++i) {
        const auto single = idx->search(
            queries.data() + static_cast<size_t>(i) * fx.dim, 10, sc1);
        expect_same_results(single, out[i], /*exact_order=*/false);
    }
    // Second pass over the same batch: every unique leaf must hit the
    // hot-set (no fills).
    (void)idx->search_stats().snapshot_and_reset();
    idx->search_batch(queries.data(), nq, 10, sc, out);
    const auto s = idx->search_stats().snapshot_and_reset();
    EXPECT_GT(s.cache_hits, 0u);
    EXPECT_EQ(s.cache_misses, 0u);
    EXPECT_EQ(s.cache_bytes_filled, 0u);
}

TEST(BatchSearch, BatchStatsCounters) {
    const auto& fx = fixture();
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path);
    (void)idx->batch_stats().snapshot_and_reset();
    auto sc = base_config();
    sc.search_threads = 2;
    const uint32_t nq = 32;
    const auto queries = fx.make_queries(nq);
    std::vector<std::vector<sextant::Candidate>> out;
    idx->search_batch(queries.data(), nq, 10, sc, out);
    const auto bs = idx->batch_stats().snapshot_and_reset();
    EXPECT_EQ(bs.batches, 1u);
    EXPECT_EQ(bs.queries, nq);
    EXPECT_GT(bs.leaves_unique, 0u);
    EXPECT_GE(bs.leaf_scans, bs.leaves_unique);
    EXPECT_GT(bs.bytes_unique, 0u);
    // Query-major accounting stays comparable: per-query stats count every
    // query's own probe set.
    const auto s = idx->search_stats().snapshot_and_reset();
    EXPECT_EQ(s.queries, nq);
    EXPECT_GE(s.leaves_probed, bs.leaf_scans);
    EXPECT_GT(s.bytes_touched, bs.bytes_unique);
}

TEST(BatchSearch, EmptyBatch) {
    const auto& fx = fixture();
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path);
    auto sc = base_config();
    std::vector<std::vector<sextant::Candidate>> out;
    idx->search_batch(nullptr, 0, 10, sc, out);
    EXPECT_TRUE(out.empty());
}

TEST(BatchSearch, FeedbackConfigRejected) {
    const auto& fx = fixture();
    auto idx = sextant::tree::IVFTreeIndex::open(fx.tree_path);
    auto sc = base_config();
    sc.feedback.mode = sextant::FeedbackProbe::Mode::Fixed;
    sc.feedback.fixed_fraction = 0.5f;
    sc.n_probe = 0;  // feedback requires n_probe == 0
    const auto queries = fx.make_queries(2);
    std::vector<std::vector<sextant::Candidate>> out;
    EXPECT_THROW(idx->search_batch(queries.data(), 2, 10, sc, out),
                 sextant::Error);
}
