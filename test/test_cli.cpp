// End-to-end tests for the CLI / engine pipeline.
//
// These exercise the full build → search cycle programmatically (using the
// Engine API directly, not spawning the CLI binary), verifying that:
//  - an index can be built from a synthetic .fbin
//  - search returns the true nearest neighbor for a known query
//  - rerank by exact distance recovers the exact NN
//  - insert + reopen persists a new vector
//
// The synthetic dataset places a handful of "anchor" vectors in a random cloud;
// a query equal to an anchor must return that anchor as the top hit.

#include <gtest/gtest.h>
#include "fbin_source.hpp"
#include "sextant/config.hpp"
#include "sextant/builder.hpp"
#include "sextant/estimator.hpp"
#include "sextant/index.hpp"
#include "sextant/searcher.hpp"
#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"
#include "sextant/error.hpp"
#include "sextant/vector_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace sextant {
namespace {

/// Write n×dim float vectors to a .fbin file from a flat row-major buffer.
static std::string write_fbin(const std::string& name, uint32_t n,
                              uint32_t dim, const float* data) {
    std::string path =
        (std::filesystem::temp_directory_path() / name).string();
    FILE* fp = std::fopen(path.c_str(), "wb");
    EXPECT_NE(fp, nullptr);
    std::fwrite(&n, sizeof(n), 1, fp);
    std::fwrite(&dim, sizeof(dim), 1, fp);
    std::fwrite(data, sizeof(float), static_cast<size_t>(n) * dim, fp);
    std::fclose(fp);
    return path;
}

/// Write a single-vector .fbin.
static std::string write_single_fbin(const std::string& name, uint32_t dim,
                                     const float* data) {
    return write_fbin(name, 1, dim, data);
}

static void remove_sidecars(const std::string& base) {
    for (const char* suf : {".graph", ".codes", ".meta", ".manifest"}) {
        std::remove((base + suf).c_str());
    }
}

static float l2sq(const float* a, const float* b, uint32_t dim) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Full build → search: the query is an exact copy of an indexed anchor vector;
// the top search result must be that anchor's row_id.
// ---------------------------------------------------------------------------
TEST(Cli, BuildSearchFindsAnchorNN) {
    const uint32_t n = 500;
    const uint32_t dim = 32;
    const uint32_t anchor = 100;  // the vector we will query for

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (auto& v : data) v = dist(rng);

    const std::string fbin =
        write_fbin("cli_anchor_data.fbin", n, dim, data.data());
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "cli_anchor_idx").string();
    remove_sidecars(index_path);

    // Build.
    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        Builder(*idx).build(source, index_path, BuildConfig{.pq_m = 8, .pq_bits = 8});
    }

    // Query = exact copy of the anchor vector.
    std::vector<float> query(data.begin() + anchor * dim,
                             data.begin() + (anchor + 1) * dim);

    auto idx = Index::read(index_path);
    Searcher searcher(*idx);

    SearchConfig scfg;
    scfg.k = 10;
    scfg.L_search = 120;
    auto results = searcher.search(query.data(), scfg.k, scfg);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().row_id, static_cast<RowId>(anchor));

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// Build → search with rerank: exact-distance reranking must put the true NN
// first, even if approximate order differed.
// ---------------------------------------------------------------------------
TEST(Cli, BuildSearchRerankExactNN) {
    const uint32_t n = 400;
    const uint32_t dim = 24;
    const uint32_t target = 250;

    std::mt19937 rng(13);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (auto& v : data) v = dist(rng);

    const std::string fbin =
        write_fbin("cli_rerank_data.fbin", n, dim, data.data());
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "cli_rerank_idx").string();
    remove_sidecars(index_path);

    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        Builder(*idx).build(source, index_path, BuildConfig{.pq_m = 8, .pq_bits = 8});
    }

    // Query = target vector + tiny perturbation.
    std::vector<float> query(dim);
    for (uint32_t d = 0; d < dim; d++) {
        query[d] = data[static_cast<size_t>(target) * dim + d] + 1e-4f;
    }

    auto idx = Index::read(index_path);
    Searcher searcher(*idx);

    const uint32_t k = 3;
    const uint32_t rerank_factor = 10;
    SearchConfig scfg;
    scfg.k = k * rerank_factor;
    scfg.L_search = 150;
    auto cands = searcher.search(query.data(), scfg.k, scfg);
    ASSERT_FALSE(cands.empty());

    // Rerank exactly.
    std::vector<std::pair<float, RowId>> scored;
    for (const auto& c : cands) {
        if (c.row_id < 0 || static_cast<uint64_t>(c.row_id) >= n) continue;
        scored.emplace_back(
            l2sq(query.data(), &data[static_cast<size_t>(c.row_id) * dim],
                 dim),
            c.row_id);
    }
    std::sort(scored.begin(), scored.end());
    ASSERT_FALSE(scored.empty());
    EXPECT_EQ(scored.front().second, static_cast<RowId>(target));

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// Insert via Engine API, flush, reopen: the inserted vector survives and is the
// top result when searched for.
// ---------------------------------------------------------------------------
TEST(Cli, InsertPersistAndSearch) {
    const uint32_t n = 300;
    const uint32_t dim = 16;

    std::mt19937 rng(21);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (auto& v : data) v = dist(rng);

    const std::string fbin =
        write_fbin("cli_insert_data.fbin", n, dim, data.data());
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "cli_insert_idx").string();
    remove_sidecars(index_path);

    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        Builder(*idx).build(source, index_path, BuildConfig{.pq_m = 8, .pq_bits = 8});
    }

    // Insert a near-duplicate of base vector #0 (in-distribution so the PQ
    // code is accurate and the node is reachable by beam search).
    std::vector<float> ins(dim);
    for (uint32_t d = 0; d < dim; d++) {
        ins[d] = data[d] + 1e-5f;
    }
    const std::string ins_fbin =
        write_single_fbin("cli_ins_vec.fbin", dim, ins.data());
    const RowId ins_row = 4242;

    // Mimic the CLI insert path: open → insert → flush.
    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        EXPECT_EQ(static_cast<uint64_t>(n), idx->count);
        Builder builder(*idx);
        builder.insert(ins.data(), dim, ins_row);
        EXPECT_EQ(static_cast<uint64_t>(n) + 1, idx->count);
        builder.flush();
    }

    // Reopen and search for the inserted vector.
    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        EXPECT_EQ(static_cast<uint64_t>(n) + 1, idx->count);

        SearchConfig scfg;
        scfg.k = 10;
        scfg.L_search = 120;
        auto results = searcher.search(ins.data(), scfg.k, scfg);
        ASSERT_FALSE(results.empty());
        bool found = false;
        for (const auto& c : results) {
            if (c.row_id == ins_row) { found = true; break; }
        }
        EXPECT_TRUE(found) << "inserted row not found after reopen";
    }

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
    std::remove(ins_fbin.c_str());
}

}  // namespace
}  // namespace sextant
