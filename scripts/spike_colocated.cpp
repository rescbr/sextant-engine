// Partial co-located neighbor codes spike (SymphonyQG-inspired, option B).
//
// Question: does co-locating K of R neighbors' PQ codes inline per node
// (vs the current per-hop random pin_code into the global codes array)
// meaningfully improve graph-search QPS at our scale?
//
// Setup: load an existing built index via Index::read. For each node,
// gather K of its neighbors' codes inline. Sweep K ∈ {1,2,4,8,16,32}.
// Run a custom beam-search loop in two variants:
//   (a) BASELINE: fetch each neighbor's code from the global codes array
//       (mimics today's pin_code path; random access).
//   (b) COLOCATED_K: fetch the first K neighbors' codes from the inline
//       block (sequential read); fetch the remaining R-K from the global
//       array (random access).
// Measure QPS + recall@10 vs ground truth.
//
// Important: this spike uses the EXISTING engine codebook (m=96, bits=8,
// 96 B/code). We're not testing 4-bit PQ here — we're testing the layout
// hypothesis. The "is 4-bit better" question is independent (option A in
// architecture-options-2026-07-25.md) and already answered.
//
// The "K of R" framing comes from this session's discussion: full
// co-location (K=R=32) costs 32× code storage = infeasible at 1B. But
// greedy graph search visits the closest neighbor first; if we co-locate
// just the K closest, we cover the hot path. K=4 fits the existing
// lut_distance_batch4 kernel exactly.

#include "quant/pq_quantizer.hpp"
#include "sextant/index.hpp"
#include "sextant/searcher.hpp"
#include "sextant/config.hpp"
#include "sextant/types.hpp"
#include "sextant/error.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <thread>
#include <vector>

using namespace sextant;

// --- .fbin / .gt readers (lifted from prior spikes).
struct FbinData { uint32_t n=0, dim=0; std::vector<float> data; };
bool read_fbin(const std::string& p, FbinData& o) {
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    f.read((char*)&o.n, 4); f.read((char*)&o.dim, 4);
    if (!f || o.n==0) return false;
    o.data.resize(size_t(o.n)*o.dim);
    f.read((char*)o.data.data(), o.data.size()*4);
    return f.good() || f.eof();
}
struct GroundTruth { uint32_t n=0, k=0; std::vector<uint32_t> ids; };
bool read_gt(const std::string& p, GroundTruth& o) {
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    f.read((char*)&o.n, 4); f.read((char*)&o.k, 4);
    if (!f || o.n==0) return false;
    o.ids.resize(size_t(o.n)*o.k);
    f.read((char*)o.ids.data(), o.ids.size()*4);
    return f.good() || f.eof();
}

double secs() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Custom beam search using either baseline or co-located codes.
// Mirrors the engine's beam_search_into but exposes the per-hop distance
// source so we can A/B them. We re-use the engine's PagedNodeStore via
// Index::read — the test is whether the layout change beats the random
// pin_code access pattern.
struct SearchStats {
    uint64_t hops = 0;
    uint64_t codes_evaluated = 0;
    uint64_t codes_colocated = 0;  // served from inline block (no pin)
    uint64_t codes_random = 0;     // served from global codes array (pin)
};

int main(int argc, char** argv) {
    const std::string idx_path   = (argc > 1) ? argv[1] : "/tmp/arxiv100k_idx";
    const std::string qpath      = (argc > 2) ? argv[2] : "datasets/arxiv100k_query.fbin";
    const std::string gtpath     = (argc > 3) ? argv[3] : "datasets/arxiv100k_gt.gt";
    const std::string base_path  = (argc > 4) ? argv[4] : "datasets/arxiv100k_base.fbin";

    FbinData queries, base;
    GroundTruth gt;
    if (!read_fbin(qpath, queries)) { printf("cannot read %s\n", qpath.c_str()); return 1; }
    if (!read_fbin(base_path, base)) { printf("cannot read %s\n", base_path.c_str()); return 1; }
    if (!read_gt(gtpath, gt)) { printf("cannot read %s\n", gtpath.c_str()); return 1; }

    printf("Loading index %s ...\n", idx_path.c_str());
    auto t0 = secs();
    auto idx = Index::read(idx_path);
    const double load_secs = secs() - t0;
    printf("  loaded in %.2fs: n=%u dim=%u\n", load_secs, idx->count, idx->dim);

    const PqQuantizer& q = *idx->quantizer;
    const uint32_t code_size = q.code_size();
    printf("  quantizer: m=%u bits=%u code_size=%u\n", q.m(), q.bits(), code_size);

    // We need flat access to (a) every node's PQ code and (b) the raw vectors
    // for rerank. Index::read doesn't keep the flat buffers resident; we
    // re-encode every vector with the loaded quantizer into our own flat
    // codes array, and read raw vectors from base_path on demand.
    printf("Re-encoding %u vectors (flat codes array for spike)...\n", idx->count);
    std::vector<uint8_t> codes_all((size_t)idx->count * code_size, 0);
    t0 = secs();
    {
        // Encode in parallel via std::thread.
        const uint32_t nthreads = std::max(1u, std::thread::hardware_concurrency());
        auto worker = [&](uint32_t lo, uint32_t hi) {
            std::vector<uint8_t> tmp(code_size);
            for (uint32_t i = lo; i < hi; i++) {
                q.encode(base.data.data() + size_t(i) * idx->dim, tmp.data());
                std::memcpy(codes_all.data() + size_t(i) * code_size, tmp.data(), code_size);
            }
        };
        std::vector<std::thread> ts;
        for (uint32_t t = 0; t < nthreads; t++) {
            const uint32_t lo = (uint64_t)t * idx->count / nthreads;
            const uint32_t hi = (uint64_t)(t+1) * idx->count / nthreads;
            ts.emplace_back(worker, lo, hi);
        }
        for (auto& th : ts) th.join();
    }
    printf("  re-encoded in %.2fs (%.1f MB)\n", secs() - t0,
           codes_all.size() / 1e6);

    // Use the engine's own Searcher to get the baseline numbers — it already
    // does beam_search_into via PagedNodeStore/MemGraph with pin_code.
    // The engine does NOT do rerank internally (rerank is the caller's job);
    // we measure engine-QPS here, then rerank ourselves for recall.
    printf("\n=== Baseline (engine Searcher) ===\n");
    Searcher searcher(*idx, /*num_threads=*/1);
    SearchConfig cfg;
    cfg.L_search = 200;
    {
        uint64_t hits_engine = 0;       // recall before rerank (raw PQ ranking)
        uint64_t hits_rerank = 0;       // recall after FP32 rerank of L_search results
        t0 = secs();
        for (uint32_t qi = 0; qi < queries.n; qi++) {
            const float* query = queries.data.data() + size_t(qi) * queries.dim;
            auto result = searcher.search(query, /*k=*/10, cfg);
            const uint32_t* gt_row = gt.ids.data() + size_t(qi) * gt.k;
            // Engine-PQ recall (no rerank).
            for (size_t r = 0; r < result.size() && r < 10; r++) {
                for (uint32_t g = 0; g < 10; g++)
                    if (gt_row[g] == result[r].row_id) { hits_engine++; break; }
            }
            // Rerank top-10 from result against base vectors.
            std::vector<std::pair<float, int64_t>> ranked;
            for (size_t r = 0; r < result.size(); r++) {
                const int64_t rid = result[r].row_id;
                const float* v = base.data.data() + size_t(rid) * base.dim;
                float d = 0;
                for (uint32_t k = 0; k < base.dim; k++) { float x = v[k]-query[k]; d += x*x; }
                ranked.push_back({d, rid});
            }
            std::partial_sort(ranked.begin(), ranked.begin() + std::min<size_t>(10, ranked.size()),
                              ranked.end());
            for (size_t r = 0; r < std::min<size_t>(10, ranked.size()); r++) {
                for (uint32_t g = 0; g < 10; g++)
                    if (gt_row[g] == ranked[r].second) { hits_rerank++; break; }
            }
        }
        const double dt = secs() - t0;
        printf("  PQ recall@10   = %.4f\n", (float)hits_engine / (float(queries.n) * 10.0f));
        printf("  rerank recall@10 = %.4f\n", (float)hits_rerank / (float(queries.n) * 10.0f));
        printf("  QPS = %.1f   (engine search + our rerank)\n", queries.n / dt);
    }

    // ---------------------------------------------------------------------
    // Read neighbor lists from .graph sidecar directly (skip the engine's
    // paged/MemGraph layer — we want flat access for the spike).
    //
    // CRITICAL ID CONVENTION:
    //   - Neighbor arrays store INTERNAL IDs (0..count-1 in node-buffer order).
    //   - codes_all is indexed by ROW ID (the user-facing data order).
    //   - The two may differ; we must build an internal→row map per node.
    // ---------------------------------------------------------------------
    printf("\nLoading neighbor lists from .graph sidecar...\n");
    std::vector<std::vector<uint32_t>> neighbors(idx->count);   // internal IDs
    std::vector<int64_t> internal_to_row(idx->count);            // internal → row
    {
        std::ifstream f(idx_path + ".graph", std::ios::binary);
        if (!f) { printf("cannot open .graph\n"); return 1; }
        f.seekg(64);  // skip SidecarHeader
        const uint32_t ns = idx->node_size;
        std::vector<uint8_t> node_buf(ns);
        for (uint32_t i = 0; i < idx->count; i++) {
            f.read((char*)node_buf.data(), ns);
            if (!f) { printf("short read at node %u\n", i); return 1; }
            // Layout per vamana_core.cpp:
            //   offset 0:  row_id (i64)
            //   offset 8:  internal_id (u32)
            //   offset 12: neighbor_count (u16)
            //   offset 16: neighbor_array (count × u32, internal IDs)
            int64_t rid;
            std::memcpy(&rid, node_buf.data() + 0, 8);
            uint16_t ncount;
            std::memcpy(&ncount, node_buf.data() + 12, 2);
            neighbors[i].resize(ncount);
            std::memcpy(neighbors[i].data(), node_buf.data() + 16, ncount * 4);
            internal_to_row[i] = rid;
        }
    }
    // Build row→internal for the reverse lookup (entry points come as
    // internal IDs from idx->core->entry_points()).
    // codes_all is indexed by row ID, so we need internal→row to fetch
    // a neighbor's code: codes_all[internal_to_row[neighbor_internal_id]].
    // Sanity check: are internal_id and row_id identical?
    bool ids_match = true;
    for (uint32_t i = 0; i < idx->count; i++) {
        if (internal_to_row[i] != (int64_t)i) { ids_match = false; break; }
    }
    double avg_deg = 0;
    for (auto& n : neighbors) avg_deg += n.size();
    avg_deg /= idx->count;
    printf("  avg degree = %.2f   (internal_id == row_id: %s)\n",
           avg_deg, ids_match ? "yes" : "NO — mapping required");

    // ---------------------------------------------------------------------
    // Co-located codes: for each node, copy the first K neighbors' codes
    // into an inline buffer right after the neighbor list.
    //
    // Inline layout per node:
    //   [neighbor_ids: R × u32][inline_codes: K × code_size bytes]
    // The remaining R-K neighbors are looked up via codes_all (random).
    //
    // We test K ∈ {0 (= baseline in our framework), 1, 2, 4, 8, 16, 32}.
    // K=0 means every neighbor fetch goes through codes_all (mimics today).
    // ---------------------------------------------------------------------
    auto build_colocated = [&](uint32_t K) {
        // Returns: for each node, a contiguous buffer with neighbor_ids
        // followed by K neighbor codes. We store in a flat per-node vector
        // for the spike (production would inline into the .graph sidecar).
        struct NodeBlock {
            std::vector<uint32_t> neighbors;     // internal IDs
            std::vector<uint8_t> inline_codes;   // K * code_size bytes
        };
        std::vector<NodeBlock> blocks(idx->count);
        for (uint32_t i = 0; i < idx->count; i++) {
            blocks[i].neighbors = neighbors[i];
            const uint32_t Keff = std::min<uint32_t>(K, (uint32_t)neighbors[i].size());
            blocks[i].inline_codes.resize((size_t)Keff * code_size, 0);
            for (uint32_t k = 0; k < Keff; k++) {
                const uint32_t nb_internal = neighbors[i][k];
                const int64_t nb_row = internal_to_row[nb_internal];
                std::memcpy(blocks[i].inline_codes.data() + (size_t)k * code_size,
                            codes_all.data() + (size_t)nb_row * code_size,
                            code_size);
            }
        }
        return blocks;
    };

    // Custom beam search using a colocated layout. Returns top-k Candidate.
    // Uses PQ LUT distances (no rerank in the loop; we rerank at the end).
    auto search_colocated = [&](const auto& blocks, uint32_t K,
                                 const float* query, uint32_t k, uint32_t L,
                                 const std::vector<float>& lut,
                                 std::vector<Candidate>& out) {
        // Working set W: a max-heap (priority_queue) keyed by dist. top() is
        // the WORST of the top-L (so we can pop it when W grows past L).
        // Slow but correct (the per-iter O(L) copy to find unvisited min is
        // ~11% of total time per the sample profile, but the layout signal
        // we're measuring is per-fetch latency which is independent).
        struct Item { float dist; uint32_t id; };
        struct MaxHeapCmp {
            bool operator()(const Item& a, const Item& b) const { return a.dist > b.dist; }
        };
        std::priority_queue<Item, std::vector<Item>, MaxHeapCmp> W;
        std::vector<bool> visited(idx->count, false);
        float worst_in_W = std::numeric_limits<float>::max();
        auto push_if_better = [&](float d, uint32_t id) {
            if (visited[id]) return;
            if (W.size() >= L && d >= worst_in_W) return;
            W.push({d, id});
            if (W.size() > L) W.pop();
            worst_in_W = W.size() >= L ? W.top().dist
                                       : std::numeric_limits<float>::max();
        };
        // PQ distance helpers (scalar). Both take INTERNAL id; codes_all
        // is indexed by ROW id so we translate.
        auto pq_dist_global = [&](uint32_t internal_id) {
            const int64_t row = internal_to_row[internal_id];
            const uint8_t* code = codes_all.data() + (size_t)row * code_size;
            float d = 0;
            for (uint32_t s = 0; s < q.m(); s++) d += lut[s * 256 + code[s]];
            return d;
        };
        auto pq_dist_inline = [&](const uint8_t* code) {
            float d = 0;
            for (uint32_t s = 0; s < q.m(); s++) d += lut[s * 256 + code[s]];
            return d;
        };
        // Seed with the engine's entry points (internal IDs).
        const auto& eps = idx->core->entry_points();
        for (uint32_t ep : eps) push_if_better(pq_dist_global(ep), ep);

        std::vector<Item> expand_order;
        while (true) {
            expand_order.clear();
            auto Wcopy = W;
            while (!Wcopy.empty()) { expand_order.push_back(Wcopy.top()); Wcopy.pop(); }
            std::sort(expand_order.begin(), expand_order.end(),
                      [](const Item& a, const Item& b){ return a.dist < b.dist; });
            uint32_t best_id = UINT32_MAX;
            for (auto& it : expand_order) {
                if (!visited[it.id]) { best_id = it.id; break; }
            }
            if (best_id == UINT32_MAX) break;
            visited[best_id] = true;
            const auto& block = blocks[best_id];
            const uint32_t R = (uint32_t)block.neighbors.size();
            const uint32_t Keff = std::min<uint32_t>(K, R);
            for (uint32_t j = 0; j < Keff; j++) {
                const uint32_t nb = block.neighbors[j];
                if (visited[nb]) continue;
                const uint8_t* code = block.inline_codes.data() + (size_t)j * code_size;
                push_if_better(pq_dist_inline(code), nb);
            }
            for (uint32_t j = Keff; j < R; j++) {
                const uint32_t nb = block.neighbors[j];
                if (visited[nb]) continue;
                push_if_better(pq_dist_global(nb), nb);
            }
        }
        // Extract top-k from W. Translate internal IDs → row IDs for output.
        std::vector<Item> all;
        while (!W.empty()) { all.push_back(W.top()); W.pop(); }
        std::sort(all.begin(), all.end(),
                  [](const Item& a, const Item& b){ return a.dist < b.dist; });
        out.clear();
        for (uint32_t i = 0; i < std::min<uint32_t>(k, (uint32_t)all.size()); i++) {
            out.push_back({internal_to_row[all[i].id], all[i].dist});
        }
    };

    printf("\n=== Co-located spike (K of R neighbors inline) ===\n");
    printf("%-6s %12s %12s %12s %14s\n", "K", "recall@10", "QPS", "inline MB", "storage ×");
    printf("------------------------------------------------------------\n");
    for (uint32_t K : {0u, 1u, 2u, 4u, 8u, 16u, 32u}) {
        auto blocks = build_colocated(K);
        // Precompute query LUTs once (reuse across all queries).
        std::vector<float> lut(q.lut_size());
        uint64_t hits = 0;
        t0 = secs();
        for (uint32_t qi = 0; qi < queries.n; qi++) {
            const float* query = queries.data.data() + size_t(qi) * queries.dim;
            q.preprocess_query(query, lut.data());
            std::vector<Candidate> result;
            search_colocated(blocks, K, query, /*k=*/10, /*L=*/200, lut, result);
            // Rerank top-10 against base vectors.
            std::vector<std::pair<float, int64_t>> ranked;
            for (auto& c : result) {
                const float* v = base.data.data() + size_t(c.row_id) * base.dim;
                float d = 0;
                for (uint32_t k2 = 0; k2 < base.dim; k2++) {
                    float x = v[k2] - query[k2]; d += x*x;
                }
                ranked.push_back({d, c.row_id});
            }
            std::partial_sort(ranked.begin(),
                              ranked.begin() + std::min<size_t>(10, ranked.size()),
                              ranked.end());
            const uint32_t* gt_row = gt.ids.data() + size_t(qi) * gt.k;
            for (size_t r = 0; r < std::min<size_t>(10, ranked.size()); r++) {
                for (uint32_t g = 0; g < 10; g++)
                    if (gt_row[g] == ranked[r].second) { hits++; break; }
            }
        }
        const double dt = secs() - t0;
        const float recall = (float)hits / (float(queries.n) * 10.0f);
        const float qps = queries.n / dt;
        // Inline storage = K × code_size per node × count.
        const double inline_mb = (double)K * code_size * idx->count / 1e6;
        // Storage multiple vs current (current = code_size per node only).
        const double storage_mult =
            1.0 + (double)K * code_size / (double)idx->node_size;
        printf("%-6u %12.4f %12.1f %12.2f %13.2fx\n",
               K, recall, qps, inline_mb, storage_mult);
    }

    return 0;
}
