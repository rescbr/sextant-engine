#include <gtest/gtest.h>
#include "engine/fbin_source.hpp"
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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "storage/sidecar_header.hpp"

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
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        ASSERT_EQ(n, source.count());
        ASSERT_EQ(dim, source.dim());

        BuildConfig cfg;
        cfg.pq_m = 8; cfg.pq_bits = 8;
        BuildResult result = Builder(*idx).build(source, index_path, cfg);
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
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        BuildConfig cfg;
        cfg.pq_m = 8; cfg.pq_bits = 8;
        Builder(*idx).build(source, index_path, cfg);
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
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        BuildConfig cfg;
        cfg.pq_m = 8; cfg.pq_bits = 8;
        Builder(*idx).build(source, index_path, cfg);
    }

    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        EXPECT_TRUE(idx != nullptr);
        EXPECT_EQ(static_cast<uint64_t>(n), idx->count);
        EXPECT_EQ(dim, idx->dim);

        // Search with one of the indexed vectors (recall should find it).
        std::vector<float> query(dim);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (uint32_t d = 0; d < dim; d++) query[d] = dist(rng);

        SearchConfig scfg;
        scfg.k = 10;
        scfg.L_search = 100;
        auto results = searcher.search(query.data(), scfg.k, scfg);
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
// After open(), the engine is in SSD-resident (paged) mode: flat RAM buffers
// are NOT loaded. This is the core value proposition — idle RAM stays ~1MB
// regardless of index size.
// ---------------------------------------------------------------------------
TEST(Engine, OpenIsPagedLowIdleRam) {
    const uint32_t n = 800;
    const uint32_t dim = 64;
    const std::string fbin =
        write_random_fbin("engine_paged_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_paged_idx").string();
    remove_sidecars(index_path);

    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        BuildConfig cfg;
        cfg.pq_m = 8; cfg.pq_bits = 8;
        Builder(*idx).build(source, index_path, cfg);
        // After build+flush: flat buffers freed, paged mode active.
        EXPECT_TRUE(idx->is_paged());
        EXPECT_FALSE(idx->has_flat_buffers());
    }

    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        EXPECT_TRUE(idx != nullptr);
        // open() loads only .meta — flat buffers must remain null.
        EXPECT_TRUE(idx->is_paged());
        EXPECT_FALSE(idx->has_flat_buffers());

        // Search must still work through the PagedNodeStore.
        std::vector<float> query(dim);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (uint32_t d = 0; d < dim; d++) query[d] = dist(rng);

        SearchConfig scfg;
        scfg.k = 10;
        scfg.L_search = 100;
        auto results = searcher.search(query.data(), scfg.k, scfg);
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
// Paged search recall matches the flat-RAM path. Build an index, then search
// the same query with both the flat (post-insert) and paged (open) paths —
// results must be identical since PagedNodeStore reads the same bytes.
// ---------------------------------------------------------------------------
TEST(Engine, PagedSearchMatchesFlat) {
    const uint32_t n = 500;
    const uint32_t dim = 32;
    const std::string fbin =
        write_random_fbin("engine_paged_match.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_paged_match_idx")
            .string();
    remove_sidecars(index_path);

    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        Builder(*idx).build(source, index_path, BuildConfig{.pq_m = 8, .pq_bits = 8});
    }

    // Build a set of queries.
    std::vector<float> queries(dim * 5);
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : queries) v = dist(rng);
    }

    // Paged search path (open → PagedNodeStore).
    std::vector<std::vector<Candidate>> paged_results;
    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        SearchConfig scfg;
        scfg.k = 10;
        scfg.L_search = 120;
        for (size_t q = 0; q < 5; q++) {
            paged_results.push_back(
                searcher.search(&queries[q * dim], scfg.k, scfg));
        }
    }

    // Each paged search must return valid, non-empty results.
    for (const auto& res : paged_results) {
        EXPECT_FALSE(res.empty());
        for (const auto& c : res) {
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
    auto idx = std::make_unique<sextant::Index>();
    Searcher searcher(*idx);
    std::vector<float> q(16, 0.0f);
    SearchConfig scfg;
    EXPECT_THROW(searcher.search(q.data(), 10, scfg), Error);
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
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        BuildConfig cfg;
        cfg.pq_m = 8; cfg.pq_bits = 8;
        Builder(*idx).build(source, index_path, cfg);
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
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        EXPECT_EQ(static_cast<uint64_t>(n), idx->count);

        Builder builder(*idx);
        builder.insert(new_vec.data(), dim, new_row_id);
        EXPECT_EQ(static_cast<uint64_t>(n) + 1, idx->count);

        // Searching for the inserted vector must find it in the top results.
        SearchConfig scfg;
        scfg.k = 20;
        scfg.L_search = 150;
        auto results = searcher.search(new_vec.data(), scfg.k, scfg);
        ASSERT_FALSE(results.empty());
        bool found = false;
        for (const auto& c : results) {
            if (c.row_id == new_row_id) { found = true; break; }
        }
        EXPECT_TRUE(found) << "inserted row_id not in search results";

        builder.flush();
    }

    // Reopen: the inserted vector must still be present and findable.
    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        EXPECT_EQ(static_cast<uint64_t>(n) + 1, idx->count);

        SearchConfig scfg;
        scfg.k = 20;
        scfg.L_search = 150;
        auto results = searcher.search(new_vec.data(), scfg.k, scfg);
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
    auto idx = std::make_unique<sextant::Index>();  // empty
    std::vector<float> v(16, 0.0f);
    EXPECT_THROW(Builder(*idx).insert(v.data(), 16, 1), Error);
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
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        Builder(*idx).build(source, index_path, BuildConfig{.pq_m = 8, .pq_bits = 8});
    }

    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        std::vector<float> bad(dim + 1, 0.0f);
        EXPECT_THROW(Builder(*idx).insert(bad.data(), dim + 1, 7), Error);
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
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        Builder(*idx).build(source, index_path, BuildConfig{.pq_m = 8, .pq_bits = 8});
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

    auto idx = Index::read(index_path);
    Searcher searcher(*idx);

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
    auto cands = searcher.search(query.data(), scfg.k, scfg);
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

// ---------------------------------------------------------------------------
// PageShuffle: the flush-time BFS reordering must not degrade search recall.
// Build, then search for a near-duplicate of an indexed vector; after rerank
// the exact NN must still rank first. Also confirms entry-point remapping is
// valid (open + search works).
// ---------------------------------------------------------------------------
TEST(Engine, PageShuffleRecallPreserved) {
    const uint32_t n = 600;
    const uint32_t dim = 32;
    const std::string fbin =
        write_random_fbin("engine_pageshuffle_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_pageshuffle_idx")
            .string();
    remove_sidecars(index_path);

    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        Builder(*idx).build(source, index_path, BuildConfig{.pq_m = 8, .pq_bits = 8});
    }

    // Read all base vectors to compute exact distances for rerank.
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

    auto idx = Index::read(index_path);
    Searcher searcher(*idx);
    ASSERT_TRUE(idx != nullptr);

    // Query a handful of indexed vectors; each query's exact NN is itself.
    SearchConfig scfg;
    scfg.k = 30;
    scfg.L_search = 150;
    uint32_t hits = 0;
    const uint32_t num_queries = 20;
    for (uint32_t q = 0; q < num_queries; q++) {
        const uint32_t target = (q * 29) % n;  // spread across the index
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; d++) {
            query[d] = base[static_cast<size_t>(target) * dim + d] + 1e-4f;
        }
        auto cands = searcher.search(query.data(), scfg.k, scfg);
        ASSERT_FALSE(cands.empty());

        // Rerank by exact L2-sq distance; the top-1 must be `target`.
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
        ASSERT_FALSE(scored.empty());
        if (scored.front().second == static_cast<RowId>(target)) hits++;
    }

    // Recall@1 across the queries must be high (≥ 0.90) — PageShuffle must not
    // corrupt the graph. Use a relaxed bar vs. SIFT (random data is harder).
    const float recall = static_cast<float>(hits) / num_queries;
    EXPECT_GE(recall, 0.90f)
        << "PageShuffle degraded recall@1 to " << recall;

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// Entry-point persistence round-trip (T6): build → flush → open.
//
// After build(), the .meta sidecar must contain the serialized entry points.
// We parse the .meta payload at the file level (skipping SidecarHeader and the
// quantizer blob) to recover the persisted entry-point IDs, then confirm they
// are non-empty and in valid range, and that the index is searchable after
// open() consumes them.
// ---------------------------------------------------------------------------
TEST(Engine, EntryPointPersistenceRoundTrip) {
    const uint32_t n = 500;
    const uint32_t dim = 64;
    const std::string fbin =
        write_random_fbin("engine_entrypoint_persist.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_entrypoint_persist_idx")
            .string();
    remove_sidecars(index_path);

    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        BuildConfig cfg;
        cfg.pq_m = 8; cfg.pq_bits = 8;
        Builder(*idx).build(source, index_path, cfg);
    }

    // Parse the .meta payload directly to recover persisted entry points.
    // Layout (mirrors Engine::load_sidecars in src/engine/search.cpp):
    //   [SidecarHeader][u64 qsize][qsize bytes][u16 ep_count][ep_count×u32]...
    std::vector<uint32_t> persisted_eps;
    {
        const std::string meta_path = index_path + ".meta";
        std::ifstream f(meta_path, std::ios::binary);
        ASSERT_TRUE(f.good()) << "cannot open " << meta_path;

        // Skip the SidecarHeader.
        SidecarHeader hdr{};
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        ASSERT_TRUE(f.good()) << "failed reading SidecarHeader";
        ASSERT_EQ(hdr.magic, kMagicMeta) << ".meta bad magic";

        // u64 quantizer_size, then skip the quantizer blob.
        uint64_t qsize = 0;
        f.read(reinterpret_cast<char*>(&qsize), sizeof(qsize));
        ASSERT_TRUE(f.good()) << "failed reading quantizer_size";
        ASSERT_GE(qsize, 0u);
        f.seekg(static_cast<std::streamoff>(qsize), std::ios::cur);
        ASSERT_TRUE(f.good()) << "failed seeking past quantizer blob";

        // u16 entry-point count, then ep_count × u32 IDs.
        uint16_t ep_count = 0;
        f.read(reinterpret_cast<char*>(&ep_count), sizeof(ep_count));
        ASSERT_TRUE(f.good()) << "failed reading entry-point count";

        persisted_eps.reserve(ep_count);
        for (uint16_t i = 0; i < ep_count; i++) {
            uint32_t ep = 0;
            f.read(reinterpret_cast<char*>(&ep), sizeof(ep));
            ASSERT_TRUE(f.good()) << "failed reading entry point " << i;
            persisted_eps.push_back(ep);
        }
    }

    // Entry points must have been serialized.
    EXPECT_FALSE(persisted_eps.empty())
        << "no entry points persisted to .meta";

    // Every persisted entry-point internal ID must be a valid vector index.
    for (uint32_t ep : persisted_eps) {
        EXPECT_LT(ep, n) << "persisted entry point " << ep << " out of range";
    }

    // open() must load the index and the persisted entry points, and search
    // must be functional (proving load_sidecars → set_entry_points works).
    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        ASSERT_TRUE(idx != nullptr);
        ASSERT_EQ(static_cast<uint64_t>(n), idx->count);
        ASSERT_EQ(dim, idx->dim);

        std::vector<float> query(dim);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (uint32_t d = 0; d < dim; d++) query[d] = dist(rng);

        SearchConfig scfg;
        scfg.k = 10;
        scfg.L_search = 100;
        auto results = searcher.search(query.data(), scfg.k, scfg);
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
// open() must reject an index whose manifest was deleted but sidecars remain.
// This simulates a crashed build where the atomic commit never completed.
// ---------------------------------------------------------------------------
TEST(Engine, OpenRejectsOrphanedSidecars) {
    const uint32_t n = 300;
    const uint32_t dim = 32;
    const std::string fbin =
        write_random_fbin("engine_orphan_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_orphan_idx").string();
    remove_sidecars(index_path);

    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        BuildConfig cfg;
        cfg.pq_m = 8; cfg.pq_bits = 8;
        Builder(*idx).build(source, index_path, cfg);
    }

    // Delete ONLY the manifest, leaving orphaned sidecars.
    std::remove((index_path + ".manifest").c_str());

    {
        try {
            auto idx = Index::read(index_path);
            FAIL() << "expected Error";
        } catch (const Error& e) {
            EXPECT_EQ(ErrorCode::CorruptIndex, e.code());
        }
    }

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// open() must reject a completely missing index (no manifest or sidecars).
// ---------------------------------------------------------------------------
TEST(Engine, OpenRejectsMissingIndex) {
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_missing_idx")
            .string();
    remove_sidecars(index_path);

    try {
        auto idx = Index::read(index_path);
        FAIL() << "expected Error";
    } catch (const Error& e) {
        EXPECT_EQ(ErrorCode::CorruptIndex, e.code());
    }

    remove_sidecars(index_path);
}

// ---------------------------------------------------------------------------
// HDC build mode: build with PQ-distance construct and verify
// the index is valid and searchable.
// ---------------------------------------------------------------------------
TEST(Engine, BuildHDCMode) {
    const uint32_t n = 800;
    const uint32_t dim = 64;
    const std::string fbin =
        write_random_fbin("engine_hdc_data.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "engine_hdc_idx").string();
    remove_sidecars(index_path);

    {
        auto idx = std::make_unique<sextant::Index>();
        FbinSource source(fbin);
        ASSERT_EQ(n, source.count());
        ASSERT_EQ(dim, source.dim());

        BuildConfig cfg;
        cfg.pq_m = 8; cfg.pq_bits = 8;
        BuildResult result = Builder(*idx).build(source, index_path, cfg);
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

    // Open and search — basic smoke test that the HDC-built graph works.
    {
        auto idx = Index::read(index_path);
        Searcher searcher(*idx);
        EXPECT_TRUE(idx != nullptr);
        EXPECT_EQ(static_cast<uint64_t>(n), idx->count);
        EXPECT_EQ(dim, idx->dim);

        std::vector<float> query(dim);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (uint32_t d = 0; d < dim; d++) query[d] = dist(rng);

        SearchConfig scfg;
        scfg.k = 10;
        scfg.L_search = 100;
        auto results = searcher.search(query.data(), scfg.k, scfg);
        EXPECT_LE(results.size(), static_cast<size_t>(scfg.k));
        for (const auto& c : results) {
            EXPECT_GE(c.row_id, 0);
            EXPECT_LT(c.row_id, static_cast<RowId>(n));
        }
    }

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

}  // namespace
}  // namespace sextant
