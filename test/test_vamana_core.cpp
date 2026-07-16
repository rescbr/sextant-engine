#include <gtest/gtest.h>
#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"

#include <algorithm>
#include <cstring>
#include <queue>
#include <vector>

namespace sextant {
namespace {

// ---------------------------------------------------------------------------
// Fixtures / helpers
// ---------------------------------------------------------------------------

/// Build a minimal VamanaCore with small params for unit tests.
/// The PQ quantizer may be a stub (returns 0 distances); tests that depend on
/// real distance semantics are guarded so they only assert structural
/// invariants that hold regardless of the distance function.
struct TestGraph {
    VamanaParams params;
    PqQuantizer quant;
    std::unique_ptr<VamanaCore> core;
    std::vector<uint8_t> codes;   // count × code_size
    std::vector<uint8_t> nodes;   // count × node_size

    TestGraph(uint32_t count, uint16_t R = 8, uint16_t L = 16,
              uint16_t inline_pq = 0)
        : params{/*.dim=*/16, R, L, /*L_build=*/L, /*alpha=*/1.2f, inline_pq,
                 /*n_entry_points=*/4, /*max_occlusion=*/750},
          quant(MetricKind::L2Sq, /*dim=*/16, /*m=*/4, /*bits=*/4) {
        core = std::make_unique<VamanaCore>(params, quant);
        const uint32_t cs = quant.code_size();
        const uint32_t ns = VamanaCore::static_node_size(R, inline_pq,
                                                          static_cast<uint8_t>(cs));
        codes.assign(static_cast<size_t>(count) * cs, 0);
        nodes.assign(static_cast<size_t>(count) * ns, 0);
        core->set_build_codes(codes.data(), count);
        core->set_build_nodes(nodes.data());
        core->prepare_for_build(count);
    }
};

// ---------------------------------------------------------------------------
// Node size
// ---------------------------------------------------------------------------

TEST(VamanaCore, NodeSize) {
    // R=64, inline_pq=0, code_size=32: (16 + 64*4 + 7) & ~7 = 272
    EXPECT_EQ(272u, VamanaCore::static_node_size(64, 0, 32));

    // R=64, inline_pq=64, code_size=32: 272 + 64*32 = 2320
    EXPECT_EQ(2320u, VamanaCore::static_node_size(64, 64, 32));
}

// ---------------------------------------------------------------------------
// Node accessor round-trips
// ---------------------------------------------------------------------------

TEST(VamanaCore, NodeAccessors) {
    TestGraph g(/*count=*/4, /*R=*/4);
    // node 0 layout: header + 4 neighbor slots.
    uint8_t* node = g.nodes.data();

    g.core->set_row_id(node, 123456);
    EXPECT_EQ(123456, g.core->get_row_id(node));

    g.core->set_internal_id(node, 42u);
    EXPECT_EQ(42u, g.core->get_internal_id(node));

    g.core->set_neighbor_count(node, 3);
    EXPECT_EQ(3u, g.core->get_neighbor_count(node));

    g.core->set_inline_pq_count(node, 7);
    EXPECT_EQ(7u, g.core->get_inline_pq_count(node));

    g.core->set_neighbor(node, 0, 10u);
    g.core->set_neighbor(node, 1, 20u);
    g.core->set_neighbor(node, 2, 30u);
    EXPECT_EQ(10u, g.core->get_neighbor(node, 0));
    EXPECT_EQ(20u, g.core->get_neighbor(node, 1));
    EXPECT_EQ(30u, g.core->get_neighbor(node, 2));
}

// ---------------------------------------------------------------------------
// Small graph build: structural invariants
//
// Even with a stub quantizer (all distances 0), the build must preserve:
//   - every node's degree ≤ R
//   - no self-loops
//   - all neighbor ids are in [0, count)
//   - the graph is weakly connected (BFS from node 0 via undirected edges
//     reaches every node) — because each inserted node connects to the
//     entry point, forming a star.
// ---------------------------------------------------------------------------

TEST(VamanaCore, BuildStructuralInvariants) {
    const uint32_t N = 50;
    const uint16_t R = 8;
    TestGraph g(N, R, /*L=*/16);

    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build_from_code(i, /*row_id=*/static_cast<RowId>(1000 + i),
                                       tls);
    }
    EXPECT_EQ(N, g.core->size());

    // Degree + range + self-loop checks.
    for (uint32_t id = 0; id < N; id++) {
        const uint8_t* node = g.nodes.data() +
                              static_cast<size_t>(id) * g.core->static_node_size(
                                  g.params.R, g.params.inline_pq_count,
                                  static_cast<uint8_t>(g.quant.code_size()));
        const uint16_t deg = g.core->get_neighbor_count(node);
        ASSERT_LE(deg, R) << "node " << id << " exceeds degree R";
        for (uint16_t j = 0; j < deg; j++) {
            const uint32_t nb = g.core->get_neighbor(node, j);
            EXPECT_NE(nb, id) << "self-loop on node " << id;
            EXPECT_LT(nb, N) << "neighbor id out of range on node " << id;
        }
    }
}

TEST(VamanaCore, BuildGraphConnected) {
    const uint32_t N = 30;
    const uint16_t R = 8;
    TestGraph g(N, R, /*L=*/16);

    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build_from_code(i, static_cast<RowId>(i), tls);
    }

    // Build undirected adjacency from the directed neighbor lists.
    const uint32_t ns = g.core->static_node_size(
        g.params.R, g.params.inline_pq_count,
        static_cast<uint8_t>(g.quant.code_size()));
    std::vector<std::vector<uint32_t>> adj(N);
    for (uint32_t id = 0; id < N; id++) {
        const uint8_t* node = g.nodes.data() + static_cast<size_t>(id) * ns;
        const uint16_t deg = g.core->get_neighbor_count(node);
        for (uint16_t j = 0; j < deg; j++) {
            const uint32_t nb = g.core->get_neighbor(node, j);
            adj[id].push_back(nb);
            adj[nb].push_back(id);
        }
    }

    // BFS from node 0 — must reach all N nodes.
    std::vector<bool> seen(N, false);
    std::queue<uint32_t> q;
    q.push(0);
    seen[0] = true;
    uint32_t reached = 0;
    while (!q.empty()) {
        const uint32_t u = q.front();
        q.pop();
        reached++;
        for (uint32_t v : adj[u]) {
            if (!seen[v]) {
                seen[v] = true;
                q.push(v);
            }
        }
    }
    EXPECT_EQ(N, reached) << "graph not connected from node 0";
}

// ---------------------------------------------------------------------------
// ADC build path (insert_build from raw vector) — structural invariant.
// ---------------------------------------------------------------------------

TEST(VamanaCore, InsertBuildFromVector) {
    const uint32_t N = 20;
    const uint16_t R = 6;
    TestGraph g(N, R, /*L=*/12);

    std::vector<float16_t> vec(g.params.dim, static_cast<float16_t>(0));
    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build(i, static_cast<RowId>(i), vec.data(), tls);
    }
    EXPECT_EQ(N, g.core->size());

    const uint32_t ns = g.core->static_node_size(
        g.params.R, g.params.inline_pq_count,
        static_cast<uint8_t>(g.quant.code_size()));
    for (uint32_t id = 0; id < N; id++) {
        const uint8_t* node = g.nodes.data() + static_cast<size_t>(id) * ns;
        const uint16_t deg = g.core->get_neighbor_count(node);
        ASSERT_LE(deg, R);
        ASSERT_EQ(id, g.core->get_internal_id(node));
        ASSERT_EQ(static_cast<RowId>(id), g.core->get_row_id(node));
    }
}

// ---------------------------------------------------------------------------
// set_entry_points / entry_points round-trip — the setter must install the
// exact vector and the getter must return it (T6: entry-point persistence).
// ---------------------------------------------------------------------------

TEST(VamanaCore, SetEntryPointsRoundTrip) {
    const uint32_t N = 50;
    TestGraph g(N);
    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build_from_code(i, static_cast<RowId>(i), tls);
    }

    // insert_build_from_code seeds entry_points_ with the last inserted id;
    // set_entry_points must overwrite it with the exact persisted vector.
    const std::vector<uint32_t> persisted = {3u, 17u, 42u, 49u};
    g.core->set_entry_points(std::vector<uint32_t>(persisted));
    const auto& eps = g.core->entry_points();
    ASSERT_EQ(eps.size(), persisted.size());
    for (size_t i = 0; i < persisted.size(); i++) {
        EXPECT_EQ(eps[i], persisted[i]);
    }

    // Overwriting with an empty set is allowed (clears entry points).
    g.core->set_entry_points({});
    EXPECT_TRUE(g.core->entry_points().empty());
}

// ---------------------------------------------------------------------------
// compute_entry_points — populates entry_points_ with evenly-spread ids.
// ---------------------------------------------------------------------------

TEST(VamanaCore, ComputeEntryPoints) {
    const uint32_t N = 100;
    TestGraph g(N);
    // Populate nodes minimally so count_ is set.
    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build_from_code(i, static_cast<RowId>(i), tls);
    }

    g.core->compute_entry_points();
    const auto& eps = g.core->entry_points();
    ASSERT_LE(eps.size(), static_cast<size_t>(g.params.n_entry_points));
    ASSERT_GT(eps.size(), 0u);
    for (uint32_t ep : eps) {
        EXPECT_LT(ep, N);
    }
    // Entry points should be distinct.
    auto sorted = eps;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted.end(), std::unique(sorted.begin(), sorted.end()));
}

// ---------------------------------------------------------------------------
// search — returns at most k candidates; ids resolved to row_ids.
// ---------------------------------------------------------------------------

TEST(VamanaCore, SearchReturnsAtMostK) {
    const uint32_t N = 40;
    TestGraph g(N);
    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build_from_code(i, static_cast<RowId>(2000 + i), tls);
    }

    const uint32_t lut_sz = g.quant.lut_size();
    std::vector<float> lut(lut_sz, 0.0f);
    const uint32_t k = 5;
    auto results = g.core->search(lut.data(), k, /*L_search=*/16,
                                  /*io_limit=*/0);
    EXPECT_LE(results.size(), static_cast<size_t>(k));
    // Every returned row_id must map to a node we inserted.
    for (const auto& c : results) {
        EXPECT_GE(c.row_id, 2000);
        EXPECT_LT(c.row_id, 2000 + static_cast<RowId>(N));
    }
}

// ---------------------------------------------------------------------------
// beam_search with forced entryPoints — starts from the given node.
// ---------------------------------------------------------------------------

TEST(VamanaCore, BeamSearchForcedEntry) {
    const uint32_t N = 25;
    TestGraph g(N);
    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build_from_code(i, static_cast<RowId>(i), tls);
    }

    const uint32_t lut_sz = g.quant.lut_size();
    std::vector<float> lut(lut_sz, 0.0f);
    std::vector<uint32_t> eps = {5};
    auto results = g.core->beam_search(lut.data(), /*L=*/10, /*io_limit=*/0,
                                       tls, &eps);
    // At minimum, the forced entry point is in the working set.
    ASSERT_GE(results.size(), 1u);
    // With the stub quantizer all distances are 0; the entry point should be
    // among the returned internal_ids.
    bool found = false;
    for (const auto& c : results) {
        if (static_cast<uint32_t>(c.row_id) == 5u) {
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// finalize_inline_codes — no crash, preserves neighbor counts.
// ---------------------------------------------------------------------------

TEST(VamanaCore, FinalizeInlineCodes) {
    const uint32_t N = 15;
    const uint16_t R = 4;
    const uint16_t inline_pq = 4;
    // Use inline_pq > 0 so finalize actually does work.
    TestGraph g(N, R, /*L=*/8, inline_pq);

    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build_from_code(i, static_cast<RowId>(i), tls);
    }
    // Should not throw; neighbor counts unchanged.
    g.core->finalize_inline_codes();

    const uint32_t ns = g.core->static_node_size(
        g.params.R, g.params.inline_pq_count,
        static_cast<uint8_t>(g.quant.code_size()));
    for (uint32_t id = 0; id < N; id++) {
        const uint8_t* node = g.nodes.data() + static_cast<size_t>(id) * ns;
        const uint16_t deg = g.core->get_neighbor_count(node);
        EXPECT_LE(deg, R);
    }
}

// ---------------------------------------------------------------------------
// DynamicWidth: beam_search with the two-phase width logic must still return
// valid results. The stub quantizer (all-zero distances) exercises the early
// convergence path — after 5 pops with no improvement, the beam widens to L.
// Verifies the search doesn't crash and returns candidates within degree/R.
// ---------------------------------------------------------------------------

TEST(VamanaCore, BeamSearchDynamicWidth) {
    const uint32_t N = 30;
    const uint16_t R = 6;
    TestGraph g(N, R, /*L=*/16);
    VamanaTLS tls;
    tls.resize(N);
    tls.resize_lut(g.quant.lut_size());
    for (uint32_t i = 0; i < N; i++) {
        g.core->insert_build_from_code(i, static_cast<RowId>(i), tls);
    }

    const uint32_t lut_sz = g.quant.lut_size();
    std::vector<float> lut(lut_sz, 0.0f);
    // L larger than R so DynamicWidth's L_current starts at max(R, L/4)=R.
    auto results = g.core->beam_search(lut.data(), /*L=*/32, /*io_limit=*/0, tls);
    ASSERT_GE(results.size(), 1u);
    // All returned candidates must be valid internal ids.
    for (const auto& c : results) {
        EXPECT_GE(c.row_id, 0);
        EXPECT_LT(static_cast<uint32_t>(c.row_id), N);
    }
    // With the stub quantizer (all distances 0), candidates are unsorted-by-
    // distance but structurally valid. Result count must not exceed L.
    EXPECT_LE(results.size(), 32u);
}

}  // namespace
}  // namespace sextant
