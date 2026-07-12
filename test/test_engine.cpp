#include <gtest/gtest.h>
#include "engine/fbin_source.hpp"
#include "sextant/config.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/vector_source.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace sextant {
namespace {

/// Write n×dim random float vectors to a .fbin file. Returns the path.
static std::string write_random_fbin(const std::string& name, uint32_t n,
                                     uint32_t dim, uint64_t seed = 42) {
    std::string path =
        (std::filesystem::temp_directory_path() / name).string();
    FILE* fp = std::fopen(path.c_str(), "wb");
    EXPECT_NE(fp, nullptr);
    std::fwrite(&n, sizeof(n), 1, fp);
    std::fwrite(&dim, sizeof(dim), 1, fp);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> row(dim);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            row[d] = dist(rng);
        }
        std::fwrite(row.data(), sizeof(float), dim, fp);
    }
    std::fclose(fp);
    return path;
}

/// Remove all four sidecar files for a given index path.
static void remove_sidecars(const std::string& base) {
    for (const char* suf : {".graph", ".codes", ".meta", ".manifest"}) {
        std::string p = base + suf;
        std::remove(p.c_str());
    }
}

// ---------------------------------------------------------------------------
// Build a small index and verify all four sidecar files exist.
// ---------------------------------------------------------------------------
TEST(Engine, BuildWritesAllSidecars) {
    const uint32_t n = 1000;
    const uint32_t dim = 128;
    const std::string fbin =
        write_random_fbin("engine_build_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_build_idx").string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        ASSERT_EQ(n, source.count());
        ASSERT_EQ(dim, source.dim());

        BuildConfig cfg;
        BuildResult result = engine.build(source, index_path, cfg);
        EXPECT_EQ(static_cast<uint64_t>(n), result.n_vectors);
        EXPECT_EQ(dim, result.dim);
    }

    // All four sidecar files must exist and be non-empty.
    for (const char* suf : {".graph", ".codes", ".meta", ".manifest"}) {
        std::string p = index_path + suf;
        std::error_code ec;
        ASSERT_TRUE(std::filesystem::exists(p, ec))
            << "missing sidecar: " << p;
        EXPECT_GT(std::filesystem::file_size(p, ec), 0u)
            << "empty sidecar: " << p;
    }

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// Manifest is the atomic commit point: it must exist and be non-empty.
// ---------------------------------------------------------------------------
TEST(Engine, ManifestAtomicity) {
    const uint32_t n = 500;
    const uint32_t dim = 64;
    const std::string fbin =
        write_random_fbin("engine_manifest_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_manifest_idx")
            .string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        BuildConfig cfg;
        engine.build(source, index_path, cfg);
    }

    const std::string manifest = index_path + ".manifest";
    std::error_code ec;
    ASSERT_TRUE(std::filesystem::exists(manifest, ec));
    EXPECT_GT(std::filesystem::file_size(manifest, ec), 0u);
    // No leftover .tmp file.
    EXPECT_FALSE(std::filesystem::exists(manifest + ".tmp", ec));

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// open() after build() loads the index; search() returns ≤ k candidates with
// valid row_ids.
// ---------------------------------------------------------------------------
TEST(Engine, OpenAndSearch) {
    const uint32_t n = 800;
    const uint32_t dim = 64;
    const std::string fbin =
        write_random_fbin("engine_search_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_search_idx").string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        BuildConfig cfg;
        engine.build(source, index_path, cfg);
    }

    {
        Engine engine;
        engine.open(index_path);
        EXPECT_TRUE(engine.is_open());
        EXPECT_EQ(static_cast<uint64_t>(n), engine.count());
        EXPECT_EQ(dim, engine.dim());

        // Search with one of the indexed vectors (recall should find it).
        std::vector<float> query(dim);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (uint32_t d = 0; d < dim; d++) query[d] = dist(rng);

        SearchConfig scfg;
        scfg.k = 10;
        scfg.L_search = 100;
        auto results = engine.search(query.data(), scfg.k, scfg);
        EXPECT_LE(results.size(), static_cast<size_t>(scfg.k));
        for (const auto& c : results) {
            EXPECT_GE(c.row_id, 0);
            EXPECT_LT(c.row_id, static_cast<RowId>(n));
        }
    }

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// search() before open() throws.
// ---------------------------------------------------------------------------
TEST(Engine, SearchBeforeOpenThrows) {
    Engine engine;
    std::vector<float> q(16, 0.0f);
    SearchConfig scfg;
    EXPECT_THROW(engine.search(q.data(), 10, scfg), Error);
}

// ---------------------------------------------------------------------------
// insert(): after build+open, insert a new vector and verify it is findable.
//
// We insert a near-duplicate of an existing base vector (same distribution so
// the PQ code is accurate), then search for it — the inserted row must appear
// among the top results alongside the original.
// ---------------------------------------------------------------------------
TEST(Engine, InsertAfterBuildIsFindable) {
    const uint32_t n = 800;
    const uint32_t dim = 64;
    const std::string fbin =
        write_random_fbin("engine_insert_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_insert_idx").string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        BuildConfig cfg;
        engine.build(source, index_path, cfg);
    }

    // Read base vector #0 to build an in-distribution inserted near-duplicate.
    std::vector<float> base0(dim);
    {
        FILE* fp = std::fopen(fbin.c_str(), "rb");
        ASSERT_NE(fp, nullptr);
        uint32_t hn, hd;
        std::fread(&hn, sizeof(hn), 1, fp);
        std::fread(&hd, sizeof(hd), 1, fp);
        std::fseek(fp, static_cast<long>(8 + 0 * dim * sizeof(float)), SEEK_SET);
        std::fread(base0.data(), sizeof(float), dim, fp);
        std::fclose(fp);
    }
    std::vector<float> new_vec(dim);
    for (uint32_t d = 0; d < dim; d++) new_vec[d] = base0[d] + 1e-5f;

    const RowId new_row_id = 123456;
    {
        Engine engine;
        engine.open(index_path);
        EXPECT_EQ(static_cast<uint64_t>(n), engine.count());

        engine.insert(new_vec.data(), dim, new_row_id);
        EXPECT_EQ(static_cast<uint64_t>(n) + 1, engine.count());

        // Searching for the inserted vector must find it in the top results.
        SearchConfig scfg;
        scfg.k = 20;
        scfg.L_search = 150;
        auto results = engine.search(new_vec.data(), scfg.k, scfg);
        ASSERT_FALSE(results.empty());
        bool found = false;
        for (const auto& c : results) {
            if (c.row_id == new_row_id) { found = true; break; }
        }
        EXPECT_TRUE(found) << "inserted row_id not in search results";

        engine.flush();
    }

    // Reopen: the inserted vector must still be present and findable.
    {
        Engine engine;
        engine.open(index_path);
        EXPECT_EQ(static_cast<uint64_t>(n) + 1, engine.count());

        SearchConfig scfg;
        scfg.k = 20;
        scfg.L_search = 150;
        auto results = engine.search(new_vec.data(), scfg.k, scfg);
        ASSERT_FALSE(results.empty());
        bool found = false;
        for (const auto& c : results) {
            if (c.row_id == new_row_id) { found = true; break; }
        }
        EXPECT_TRUE(found) << "inserted row_id not found after reopen";
    }

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// insert() before open() throws.
// ---------------------------------------------------------------------------
TEST(Engine, InsertBeforeOpenThrows) {
    Engine engine;
    std::vector<float> v(16, 0.0f);
    EXPECT_THROW(engine.insert(v.data(), 16, 1), Error);
}

// ---------------------------------------------------------------------------
// insert() with a dimension mismatch throws.
// ---------------------------------------------------------------------------
TEST(Engine, InsertDimMismatchThrows) {
    const uint32_t n = 200;
    const uint32_t dim = 32;
    const std::string fbin =
        write_random_fbin("engine_insert_dim_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_insert_dim_idx")
            .string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        engine.build(source, index_path, BuildConfig{});
    }

    {
        Engine engine;
        engine.open(index_path);
        std::vector<float> bad(dim + 1, 0.0f);
        EXPECT_THROW(engine.insert(bad.data(), dim + 1, 7), Error);
    }

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// Rerank: when the caller recomputes exact distances from the base vectors,
// the true (exact) nearest neighbor must rank first.
// ---------------------------------------------------------------------------
TEST(Engine, SearchRerankFindsExactNN) {
    const uint32_t n = 600;
    const uint32_t dim = 32;
    const std::string fbin =
        write_random_fbin("engine_rerank_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_rerank_idx")
            .string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        engine.build(source, index_path, BuildConfig{});
    }

    // Read all base vectors so we can compute exact distances for rerank and
    // determine the ground-truth NN.
    std::vector<float> base(static_cast<size_t>(n) * dim);
    {
        FILE* fp = std::fopen(fbin.c_str(), "rb");
        ASSERT_NE(fp, nullptr);
        uint32_t hdr_n = 0, hdr_dim = 0;
        std::fread(&hdr_n, sizeof(hdr_n), 1, fp);
        std::fread(&hdr_dim, sizeof(hdr_dim), 1, fp);
        ASSERT_EQ(hdr_n, n);
        ASSERT_EQ(hdr_dim, dim);
        std::fread(base.data(), sizeof(float),
                   static_cast<size_t>(n) * dim, fp);
        std::fclose(fp);
    }

    Engine engine;
    engine.open(index_path);

    // Query = base vector #42 + tiny noise → its exact NN is row 42.
    std::vector<float> query(dim);
    for (uint32_t d = 0; d < dim; d++) {
        query[d] = base[static_cast<size_t>(42) * dim + d] + 1e-4f;
    }

    // Over-fetch candidates for rerank.
    const uint32_t k = 5;
    const uint32_t rerank_factor = 10;
    SearchConfig scfg;
    scfg.k = k * rerank_factor;
    scfg.L_search = 150;
    auto cands = engine.search(query.data(), scfg.k, scfg);
    ASSERT_FALSE(cands.empty());

    // Rerank by exact L2-sq distance against the base vectors.
    std::vector<std::pair<float, RowId>> scored;
    scored.reserve(cands.size());
    for (const auto& c : cands) {
        if (c.row_id < 0 || static_cast<uint64_t>(c.row_id) >= n) continue;
        const float* bv = &base[static_cast<size_t>(c.row_id) * dim];
        float dist = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            const float diff = query[d] - bv[d];
            dist += diff * diff;
        }
        scored.emplace_back(dist, c.row_id);
    }
    std::sort(scored.begin(), scored.end());

    // The rerank top-1 must be the exact nearest neighbor (row 42).
    ASSERT_FALSE(scored.empty());
    EXPECT_EQ(scored.front().second, static_cast<RowId>(42));

    // Sanity: the exact NN distance is ~ (dim * (1e-4)^2) ≈ 3.2e-7.
    EXPECT_NEAR(scored.front().first, 0.0f, 1e-3f);

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

}  // namespace
}  // namespace sextant
