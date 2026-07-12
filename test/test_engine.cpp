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

}  // namespace
}  // namespace sextant
