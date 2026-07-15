#include <gtest/gtest.h>
#include "engine/fbin_source.hpp"
#include "sextant/config.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/vector_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace sextant {
namespace {

/// Write n×dim random float vectors to a .fbin file. The vectors are drawn
/// from a small number of cluster centers so nearest-neighbor structure is
/// meaningful (pure uniform noise has no real NN signal at low N).
static std::string write_clustered_fbin(const std::string& name, uint32_t n,
                                        uint32_t dim, uint32_t n_clusters,
                                        uint64_t seed = 7) {
    std::string path =
        (std::filesystem::temp_directory_path() / name).string();
    FILE* fp = std::fopen(path.c_str(), "wb");
    EXPECT_NE(fp, nullptr);
    std::fwrite(&n, sizeof(n), 1, fp);
    std::fwrite(&dim, sizeof(dim), 1, fp);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    // Cluster centers.
    std::vector<std::vector<float>> centers(n_clusters,
        std::vector<float>(dim));
    for (auto& c : centers) {
        for (auto& v : c) v = u(rng) * 5.0f;
    }
    std::normal_distribution<float> noise(0.0f, 0.3f);
    std::uniform_int_distribution<uint32_t> pick(0, n_clusters - 1);
    std::vector<float> row(dim);
    for (uint32_t i = 0; i < n; i++) {
        const auto& c = centers[pick(rng)];
        for (uint32_t d = 0; d < dim; d++) row[d] = c[d] + noise(rng);
        std::fwrite(row.data(), sizeof(float), dim, fp);
    }
    std::fclose(fp);
    return path;
}

/// Read all vectors from a .fbin into a flat float buffer (n×dim).
static std::vector<float> read_fbin_vectors(const std::string& path,
                                            uint32_t& n_out,
                                            uint32_t& dim_out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    EXPECT_NE(fp, nullptr);
    std::fread(&n_out, sizeof(n_out), 1, fp);
    std::fread(&dim_out, sizeof(dim_out), 1, fp);
    std::vector<float> data(static_cast<size_t>(n_out) * dim_out);
    std::fread(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
    return data;
}

/// Remove all four sidecar files for a given index path.
static void remove_sidecars(const std::string& base) {
    for (const char* suf : {".graph", ".codes", ".meta", ".manifest"}) {
        std::string p = base + suf;
        std::remove(p.c_str());
    }
}

/// Brute-force exact k-NN by L2-sq distance. Returns sorted (dist, row_id).
static std::vector<std::pair<float, RowId>> brute_force_knn(
    const std::vector<float>& base, uint32_t n, uint32_t dim,
    const float* query, uint32_t k) {
    std::vector<std::pair<float, RowId>> scored;
    scored.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        const float* v = &base[static_cast<size_t>(i) * dim];
        float d = 0.0f;
        for (uint32_t dd = 0; dd < dim; dd++) {
            const float diff = query[dd] - v[dd];
            d += diff * diff;
        }
        scored.emplace_back(d, static_cast<RowId>(i));
    }
    std::partial_sort(scored.begin(),
                      scored.begin() + std::min<size_t>(k, scored.size()),
                      scored.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
    if (scored.size() > k) scored.resize(k);
    return scored;
}

/// Build an index with the given config and measure recall@k against brute
/// force, using the first `n_queries` base vectors as queries.
static float measure_recall(const std::string& fbin_path, uint32_t n,
                            uint32_t dim, const std::vector<float>& base,
                            const std::string& index_path,
                            const BuildConfig& cfg, uint32_t k,
                            uint32_t n_queries, uint32_t L_search) {
    remove_sidecars(index_path);
    {
        Engine engine;
        FbinSource source(fbin_path);
        engine.build(source, index_path, cfg);
    }
    Engine engine;
    engine.open(index_path);
    EXPECT_EQ(static_cast<uint64_t>(n), engine.count());

    uint32_t hits = 0;
    uint32_t total = 0;
    SearchConfig scfg;
    scfg.k = k;
    scfg.L_search = L_search;
    scfg.rerank_factor = 1;

    for (uint32_t q = 0; q < n_queries; q++) {
        const float* query = &base[static_cast<size_t>(q) * dim];
        auto truth = brute_force_knn(base, n, dim, query, k);
        auto results = engine.search(query, k, scfg);

        std::vector<RowId> truth_ids;
        truth_ids.reserve(truth.size());
        for (auto& t : truth) truth_ids.push_back(t.second);

        for (const auto& c : results) {
            if (std::find(truth_ids.begin(), truth_ids.end(), c.row_id) !=
                truth_ids.end()) {
                hits++;
            }
        }
        total += k;
    }
    remove_sidecars(index_path);
    return total > 0 ? static_cast<float>(hits) / total : 0.0f;
}

// ---------------------------------------------------------------------------
// Build a partitioned index (K=4) and verify all sidecars are written and
// the index opens and searches successfully.
// ---------------------------------------------------------------------------
TEST(Partition, PartitionedBuildProducesValidIndex) {
    const uint32_t n = 2000;
    const uint32_t dim = 64;
    const std::string fbin =
        write_clustered_fbin("partition_build_data.fbin", n, dim, 20);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "partition_build_idx").string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        BuildConfig cfg;
        cfg.R = 32;
        cfg.pq_m = 16; cfg.pq_bits = 8;
        // per_vec ≈ code_size(16) + node_size(144) = 160.
        // budget = 80000 → max_per_partition = 500 → K = ceil(2000/500) = 4.
        cfg.build_ram_budget = 80000;
        BuildResult result = engine.build(source, index_path, cfg);
        EXPECT_EQ(static_cast<uint64_t>(n), result.n_vectors);
        EXPECT_EQ(dim, result.dim);
    }

    for (const char* suf : {".graph", ".codes", ".meta", ".manifest"}) {
        std::string p = index_path + suf;
        std::error_code ec;
        ASSERT_TRUE(std::filesystem::exists(p, ec))
            << "missing sidecar: " << p;
        EXPECT_GT(std::filesystem::file_size(p, ec), 0u)
            << "empty sidecar: " << p;
    }

    {
        Engine engine;
        engine.open(index_path);
        EXPECT_TRUE(engine.is_open());
        EXPECT_EQ(static_cast<uint64_t>(n), engine.count());
        std::vector<float> q(dim, 0.0f);
        SearchConfig scfg;
        scfg.k = 10;
        scfg.L_search = 100;
        auto results = engine.search(q.data(), 10, scfg);
        EXPECT_LE(results.size(), static_cast<size_t>(10));
        for (const auto& c : results) {
            EXPECT_GE(c.row_id, 0);
            EXPECT_LT(c.row_id, static_cast<RowId>(n));
        }
    }

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// Merged graph connectivity: read the .graph sidecar directly and run a BFS
// over the adjacency lists (treating edges as undirected). The merged graph
// must be a single connected component — no isolated nodes, no disconnected
// shards. This directly verifies the merge + closure_factor overlap.
// ---------------------------------------------------------------------------
TEST(Partition, MergedGraphIsConnected) {
    const uint32_t n = 1500;
    const uint32_t dim = 32;
    const std::string fbin =
        write_clustered_fbin("partition_conn_data.fbin", n, dim, 10);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "partition_conn_idx").string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        BuildConfig cfg;
        cfg.R = 32;
        cfg.pq_m = 16; cfg.pq_bits = 8;
        cfg.inline_pq_count = 0;  // deterministic node_size = build layout
        cfg.build_ram_budget = 80000;  // forces K ≈ 3
        engine.build(source, index_path, cfg);
    }

    // Read the .graph sidecar header (64 bytes) + node bodies. We set
    // inline_pq_count=0 so node_size = ((16 + R*4 + 7) & ~7) = 144 for R=32.
    // Neighbor layout within a node:
    //   [row_id 8][internal_id 4][neighbor_count 2][inline_pq_count 2]
    //   [neighbors R×4]
    constexpr uint32_t R = 32;
    const uint32_t node_stride = (16u + R * 4u + 7u) & ~7u;  // 144
    std::vector<uint8_t> graph_raw;
    uint64_t n_in_file = 0;
    {
        FILE* fp = std::fopen((index_path + ".graph").c_str(), "rb");
        ASSERT_NE(fp, nullptr);
        // Header: magic(8) ver(4) hs(4) off(8) ts(8) uuid(16) n_vectors(8) dim(4) ca(4)
        uint8_t hdr[64];
        ASSERT_EQ(std::fread(hdr, 1, 64, fp), 64u);
        std::memcpy(&n_in_file, hdr + 48, 8);  // n_vectors offset in SidecarHeader
        ASSERT_EQ(n_in_file, static_cast<uint64_t>(n));
        std::fseek(fp, 0, SEEK_END);
        const long fsize = std::ftell(fp);
        std::fseek(fp, 64, SEEK_SET);
        // The node data is n × node_stride, possibly padded to kDiskAlign.
        const size_t data_region = static_cast<size_t>(fsize) - 64;
        ASSERT_GE(data_region, static_cast<size_t>(n) * node_stride)
            << "graph file too small";
        graph_raw.resize(static_cast<size_t>(n) * node_stride);
        ASSERT_EQ(std::fread(graph_raw.data(), 1, graph_raw.size(), fp),
                  graph_raw.size());
        std::fclose(fp);
    }

    // Build adjacency: for each node, read neighbor_count + neighbor IDs.
    // Neighbors are stored as uint32 IDs (global row_ids in the merged graph).
    auto read_u16 = [&](uint32_t node_off, uint32_t field) -> uint16_t {
        uint16_t v;
        std::memcpy(&v, graph_raw.data() + node_off + field, 2);
        return v;
    };
    auto read_u32 = [&](uint32_t node_off, uint32_t i) -> uint32_t {
        uint32_t v;
        std::memcpy(&v, graph_raw.data() + node_off + 16 + i * 4, 4);
        return v;
    };

    // Every node must have ≥1 neighbor.
    uint32_t zero_deg = 0;
    for (uint32_t gid = 0; gid < n; gid++) {
        const uint32_t off = gid * node_stride;
        if (read_u16(off, 12) == 0) zero_deg++;
    }
    EXPECT_EQ(zero_deg, 0u) << zero_deg << " nodes have zero neighbors";

    // BFS from node 0 over directed edges (the merged graph stores directed
    // edges from each shard's connect_and_prune; reachability under directed
    // traversal is the strict connectivity check).
    std::vector<uint8_t> visited(n, 0);
    std::vector<uint32_t> frontier;
    frontier.reserve(n);
    frontier.push_back(0);
    visited[0] = 1;
    uint32_t reached = 1;
    while (!frontier.empty()) {
        const uint32_t cur = frontier.back();
        frontier.pop_back();
        const uint32_t off = cur * node_stride;
        const uint16_t deg = read_u16(off, 12);
        for (uint16_t i = 0; i < deg; i++) {
            const uint32_t nb = read_u32(off, i);
            if (nb < n && !visited[nb]) {
                visited[nb] = 1;
                reached++;
                frontier.push_back(nb);
            }
        }
    }
    // The merged graph must be (weakly, and here strongly in the directed
    // sense from node 0) connected: ≥95% of nodes reachable.
    EXPECT_GE(reached, static_cast<uint32_t>(n * 0.95))
        << "graph not connected: reached " << reached << "/" << n
        << " from node 0";

    // Verify boundary nodes (those replicated across shards) have cross-shard
    // edges. We can't directly identify boundary nodes here, but a connected
    // graph with no isolated nodes after K-way merge proves cross-shard edges
    // exist. As an extra gate: the largest connected component == n.
    EXPECT_EQ(reached, n)
        << "largest reachable set from node 0 is " << reached << ", want " << n;

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// Primary partition-correctness gate: K=1 vs K=4 recall parity.
// Build the same clustered dataset monolithically and partitioned; the
// partitioned recall@10 must be within ~5% of the monolithic recall.
// ---------------------------------------------------------------------------
TEST(Partition, K1VsK4RecallParity) {
    const uint32_t n = 2000;
    const uint32_t dim = 64;
    const uint32_t k = 10;
    const uint32_t n_queries = 100;
    const std::string fbin =
        write_clustered_fbin("partition_recall_data.fbin", n, dim, 25);
    uint32_t rn, rdim;
    auto base = read_fbin_vectors(fbin, rn, rdim);
    ASSERT_EQ(rn, n);
    ASSERT_EQ(rdim, dim);

    const std::string idx_mono =
        (std::filesystem::temp_directory_path() / "partition_recall_mono").string();
    const std::string idx_part =
        (std::filesystem::temp_directory_path() / "partition_recall_part").string();

    // Monolithic (K=1): large RAM budget.
    BuildConfig cfg_mono;
    cfg_mono.R = 32;
    cfg_mono.pq_m = 16; cfg_mono.pq_bits = 8;
    cfg_mono.build_ram_budget = static_cast<uint64_t>(1) << 40;  // 1 TiB → K=1
    const float recall_mono = measure_recall(fbin, n, dim, base, idx_mono,
                                             cfg_mono, k, n_queries, 300);

    // Partitioned (K=4): budget = 500×160 → max_per_partition=500 → K=ceil(2000/500)=4.
    BuildConfig cfg_part;
    cfg_part.R = 32;
    cfg_part.pq_m = 16; cfg_part.pq_bits = 8;
    cfg_part.build_ram_budget = 80000;
    const float recall_part = measure_recall(fbin, n, dim, base, idx_part,
                                             cfg_part, k, n_queries, 300);

    std::fprintf(stderr, "[partition-test] recall mono(K=1)=%.4f part(K=4)=%.4f\n",
                 recall_mono, recall_part);

    // Both should achieve meaningful recall on structured data.
    EXPECT_GT(recall_mono, 0.5f) << "monolithic recall too low";
    EXPECT_GT(recall_part, 0.5f) << "partitioned recall too low";

    // Partition-correctness gate: within 5% (absolute).
    EXPECT_NEAR(recall_mono, recall_part, 0.05f)
        << "partitioned recall diverges from monolithic by > 5%";

    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// K sweep: K=1,2,4 — recall should degrade gracefully (monotonic, no cliff).
// ---------------------------------------------------------------------------
TEST(Partition, KSweepRecallDegradesGracefully) {
    const uint32_t n = 2000;
    const uint32_t dim = 64;
    const uint32_t k = 10;
    const uint32_t n_queries = 80;
    const std::string fbin =
        write_clustered_fbin("partition_sweep_data.fbin", n, dim, 25);
    uint32_t rn, rdim;
    auto base = read_fbin_vectors(fbin, rn, rdim);
    ASSERT_EQ(rn, n);
    ASSERT_EQ(rdim, dim);

    // per_vec ≈ 160 bytes. For K targets:
    //   K=1: max_per_partition >= n → budget >= 2000*160
    //   K=2: max_per_partition = 1000 → budget = 1000*160
    //   K=4: max_per_partition = 500  → budget = 500*160
    const uint64_t per_vec = 160;
    struct Probe { uint32_t target_K; uint64_t budget; float recall; };
    std::vector<Probe> probes = {
        {1, n * per_vec + 1024, 0.0f},
        {2, (n / 2) * per_vec, 0.0f},
        {4, (n / 4) * per_vec, 0.0f},
    };

    for (auto& pr : probes) {
        BuildConfig cfg;
        cfg.R = 32;
        cfg.pq_m = 16; cfg.pq_bits = 8;
        cfg.build_ram_budget = pr.budget;
        std::string idx =
            (std::filesystem::temp_directory_path() /
             ("partition_sweep_K" + std::to_string(pr.target_K))).string();
        pr.recall = measure_recall(fbin, n, dim, base, idx, cfg, k,
                                   n_queries, 300);
        std::fprintf(stderr,
                     "[partition-test] K=%u budget=%llu recall=%.4f\n",
                     pr.target_K,
                     static_cast<unsigned long long>(pr.budget), pr.recall);
        EXPECT_GT(pr.recall, 0.4f)
            << "K=" << pr.target_K << " recall too low";
    }

    // Graceful degradation: the worst (K=4) must not be more than 10% below
    // the best (K=1). No cliff.
    const float best = probes.front().recall;   // K=1
    const float worst = probes.back().recall;   // K=4
    EXPECT_LE(best - worst, 0.10f)
        << "recall cliff: best(K=1)=" << best << " worst(K=4)=" << worst;

    std::remove(fbin.c_str());
}

}  // namespace
}  // namespace sextant
