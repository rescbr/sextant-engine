// Spike: routing-loss decomposition through the IVF tree.
//
// Delivered recall = P(neighbor's leaf is probed)            [routing]
//                   × P(neighbor survives the scan top-W)    [scan/quantizer]
//                   × P(neighbor survives rerank top-k)      [merge]
// This harness measures each factor separately using the debug accessors on
// IVFTreeIndex (visited_leaf_pages out-param + debug_leaf_row_ids), so a
// recall deficit can be attributed to routing vs quantization vs merge.
//
// Protocol per query:
//  1. search(k=W, fastscan_W=W, visited_leaf_pages) — the returned set of W
//     deduped candidates equals the scan heap (rerank re-sorts but W==k keeps
//     every member), giving scan containment directly.
//  2. search(k=10, fastscan_W=W, visited_leaf_pages) — delivered recall@10.
//  3. routing recall: GT top-10 neighbors whose home leaf (any leaf storing
//     them, closure may replicate) intersects the visited set.
//
// Usage: spike_routing <index.tree> <base.fbin> <query.fbin> <gt.gt|gtmm>
//                      [k] [W] [n_probe] [n_probe_ln]
// Defaults: k=10 W=100 n_probe=8 n_probe_ln=8 (the leaderboard config).

#include "../src/tree/ivf_tree_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace sextant;
using tree::IVFTreeIndex;
using tree::PageId;
using tree::kInvalidPage;

static std::vector<float> load_fbin(const char* path, uint32_t& n, uint32_t& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s failed\n", path); std::exit(1); }
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&d), 4);
    std::vector<float> v(static_cast<size_t>(n) * d);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * 4);
    return v;
}

static bool load_gt(const char* path, std::vector<int32_t>& ids, uint32_t& k) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    uint32_t n;
    if (magic != 0x4D4D5447) {  // GTMM magic (LE)
        std::fprintf(stderr, "%s: missing GTMM magic\n", path);
        return false;
    }
    // [magic][n][k][metric:u8] then per query [ids k×i32][dists k×f32].
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&k), 4);
    char metric; f.read(&metric, 1);
    ids.resize(static_cast<size_t>(n) * k);
    std::vector<float> dists(k);
    for (uint32_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(&ids[static_cast<size_t>(i) * k]),
               k * 4);
        f.read(reinterpret_cast<char*>(dists.data()), k * 4);
    }
    return f.good() || f.eof();
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s index.tree base.fbin query.fbin gt [k] [W] [np] [np_ln]\n",
            argv[0]);
        return 1;
    }
    const uint32_t k = argc > 5 ? std::atoi(argv[5]) : 10;
    const uint32_t W = argc > 6 ? std::atoi(argv[6]) : 100;
    const uint32_t np = argc > 7 ? std::atoi(argv[7]) : 8;
    const uint32_t np_ln = argc > 8 ? std::atoi(argv[8]) : 8;

    uint32_t nb, db, nq, dq, gtk;
    auto base = load_fbin(argv[2], nb, db);
    auto query = load_fbin(argv[3], nq, dq);
    std::vector<int32_t> gt;
    if (!load_gt(argv[4], gt, gtk)) {
        std::fprintf(stderr, "read %s failed\n", argv[4]); return 1;
    }
    std::printf("base=%u dim=%u queries=%u gt_k=%u  k=%u W=%u np=%u np_ln=%u\n",
                nb, db, nq, gtk, k, W, np, np_ln);

    auto index = IVFTreeIndex::open(argv[1]);

    // --- Build row_id -> leaf-id membership over ALL leaves ---
    const auto leaf_info = index->debug_leaf_info();
    const uint32_t n_leaves = static_cast<uint32_t>(leaf_info.size());
    std::unordered_map<int64_t, std::vector<uint32_t>> home;
    uint64_t total_stored = 0;
    for (uint32_t leaf = 0; leaf < n_leaves; ++leaf) {
        if (leaf_info[leaf].page == kInvalidPage) continue;
        for (RowId rid : index->debug_leaf_row_ids(leaf)) {
            home[rid].push_back(leaf);
            ++total_stored;
        }
    }
    std::printf("leaves=%u stored=%llu (replication %.3fx)\n", n_leaves,
                static_cast<unsigned long long>(total_stored),
                static_cast<double>(total_stored) / nb);

    SearchConfig cfg;
    cfg.n_probe = np;
    cfg.n_probe_ln = np_ln;
    cfg.fastscan_W = W;
    cfg.rerank = true;  // match the CLI default (code-decode rerank on)

    // Aggregators (fractions of GT top-k neighbors, averaged over queries).
    double s_any = 0, s_route = 0, s_scan = 0, s_deliv = 0;
    double s_route_cond = 0;  // scan containment given routed
    uint32_t hard_routed = 0; // queries where every neighbor routed

    std::vector<PageId> visited;
    std::unordered_set<int64_t> scan_set, top_set;
    for (uint32_t q = 0; q < nq; ++q) {
        const float* qv = query.data() + static_cast<size_t>(q) * db;
        // Pass 1: k == W so the returned deduped set == the scan shortlist.
        auto shortlist = index->search(qv, W, cfg, nullptr, nullptr, nullptr,
                                       &visited);
        scan_set.clear();
        for (const auto& c : shortlist) scan_set.insert(c.row_id);
        // Pass 2: delivered top-k.
        auto top = index->search(qv, k, cfg, nullptr, nullptr, nullptr,
                                 &visited);
        top_set.clear();
        for (const auto& c : top) top_set.insert(c.row_id);

        std::unordered_set<PageId> vset(visited.begin(), visited.end());
        double any = 0, routed = 0, scanned = 0, deliv = 0;
        for (uint32_t j = 0; j < k; ++j) {
            const int32_t gt_id = gt[static_cast<size_t>(q) * gtk + j];
            auto it = home.find(gt_id);
            const bool in_index = it != home.end();
            bool r = false;
            if (in_index) {
                for (uint32_t leaf : it->second)
                    if (vset.count(leaf_info[leaf].page)) { r = true; break; }
            }
            const bool s = scan_set.count(gt_id) > 0;
            const bool d = top_set.count(gt_id) > 0;
            any += in_index; routed += r; scanned += s; deliv += d;
        }
        s_any += any / k; s_route += routed / k;
        s_scan += scanned / k; s_deliv += deliv / k;
        if (routed == k) ++hard_routed;
        if (routed > 0) s_route_cond += scanned / routed;
    }
    s_any /= nq; s_route /= nq; s_scan /= nq; s_deliv /= nq;
    s_route_cond /= nq;

    std::printf(
        "\n--- decomposition (fraction of exact top-%u) ---\n"
        "in index      : %.4f   (sanity: must be 1.0)\n"
        "routed        : %.4f   (neighbor's leaf probed)\n"
        "scan@W=%-4u   : %.4f   (neighbor in scan shortlist)\n"
        "delivered     : %.4f   (final top-%u after rerank)\n"
        "scan | routed : %.4f\n"
        "queries fully routed: %u / %u\n",
        k, s_any, s_route, W, s_scan, s_deliv, k, s_route_cond, hard_routed,
        nq);
    std::printf(
        "\nloss attribution (pp):\n"
        "routing loss : %.2f  (in-index − routed)\n"
        "scan loss    : %.2f  (routed − scan@W; quantizer/TBL within probed leaves)\n"
        "merge loss   : %.2f  (scan@W − delivered; rerank/W cut)\n",
        100 * (s_any - s_route), 100 * (s_route - s_scan),
        100 * (s_scan - s_deliv));
    return 0;
}
