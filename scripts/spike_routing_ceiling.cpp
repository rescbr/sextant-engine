// Spike: routing CEILING decomposition — what would perfect leaf ranking buy?
//
// The engine's measured f-grid (probe_fraction) shows recall ≈ coverage at
// depth 2: ranking leaves by PCA-32 centroid distance buys almost nothing
// over random order. This spike separates the two possible causes:
//
//   metric loss   — PCA-32 (or centroid distance generally) mis-ranks
//                   leaves relative to full-dim exact centroid distance.
//   structure loss — even with EXACT distances, the cluster assignment
//                   spreads a query's true neighbors across most leaves
//                   (nothing a better metric can fix; needs deeper
//                   trees / reassignment / replication).
//
// Protocol (per fixture, k=10 GT):
//   For each query, rank leaves under three orders:
//     a. random          — the coverage baseline (what f measures)
//     b. exact centroid   — L2 distance to the leaf's mean in full dim
//     c. member-min       — distance to the nearest ACTUAL member (the
//                            ideal leaf ranking; upper bounds any metric
//                            on centroids)  [subset of queries]
//   Then compute recall-vs-coverage curves: for each prefix fraction φ of
//   the ranked leaf list (by page-extent weight, matching the engine's
//   leaf-coverage contract), the fraction of GT@10 neighbors whose home
//   leaf set intersects the prefix.
//
// Read-out:
//   b ≈ a  → structure loss dominates; go deeper/reassign.
//   c >> b → centroid is a poor leaf summary; per-leaf summaries or
//            multi-probing by margin, not depth, is the lever.
//   c ≈ b >> a → the metric is fine and structure is fine but PCA loses
//            it (compare with the engine's PCA f-grid separately).
//
// Usage: spike_routing_ceiling <index.tree> <base.fbin> <query.fbin> <gt.gtmm>
//                              [nq] [member-min queries]

#include "../src/tree/ivf_tree_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
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

static bool load_gtmm(const char* path, std::vector<int32_t>& ids, uint32_t& k) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = 0, n = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x4D4D5447) return false;
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&k), 4);
    char metric; f.read(&metric, 1);
    ids.resize(static_cast<size_t>(n) * k);
    std::vector<float> dists(k);
    for (uint32_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(&ids[static_cast<size_t>(i) * k]), k * 4);
        f.read(reinterpret_cast<char*>(dists.data()), k * 4);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s index.tree base.fbin query.fbin gt.gtmm [nq] [mm_q]\n",
            argv[0]);
        return 1;
    }
    const uint32_t nq = argc > 5 ? std::atoi(argv[5]) : 200;
    const uint32_t mm_q = argc > 6 ? std::atoi(argv[6]) : 50;

    uint32_t nb, db, nqt, dq, gtk;
    auto base = load_fbin(argv[2], nb, db);
    auto query = load_fbin(argv[3], nqt, dq);
    std::vector<int32_t> gt;
    if (!load_gtmm(argv[4], gt, gtk) || gtk < 10) {
        std::fprintf(stderr, "gt load failed / k<10\n"); return 1;
    }
    const uint32_t gt_stride = gtk;  // file row width (may be > 10)
    gtk = 10;
    std::printf("base=%u dim=%u queries=%u(used %u) gt_k=10 mm_queries=%u\n",
                nb, db, nqt, nq, mm_q);

    auto index = IVFTreeIndex::open(argv[1]);
    const auto leaf_info = index->debug_leaf_info();
    const uint32_t L = static_cast<uint32_t>(leaf_info.size());

    // Leaf membership + extent weights (pages, matching the engine's
    // leaf-coverage contract) + full-dim centroids.
    std::vector<std::vector<RowId>> members(L);
    std::vector<double> centroid(L * db, 0.0);
    std::vector<uint64_t> weight(L, 0);
    uint64_t total_w = 0, stored = 0;
    std::unordered_map<int64_t, std::vector<uint32_t>> home;
    for (uint32_t l = 0; l < L; ++l) {
        if (leaf_info[l].page == kInvalidPage) continue;
        members[l] = index->debug_leaf_row_ids(l);
        weight[l] = leaf_info[l].pages;
        total_w += weight[l];
        stored += members[l].size();
        for (RowId rid : members[l]) {
            const float* v = &base[static_cast<size_t>(rid) * db];
            for (uint32_t d = 0; d < db; ++d) centroid[l * db + d] += v[d];
            home[rid].push_back(l);
        }
    }
    for (uint32_t l = 0; l < L; ++l) {
        const size_t m = members[l].size();
        if (m) for (uint32_t d = 0; d < db; ++d) centroid[l * db + d] /= m;
    }
    // Per-leaf RADIUS: max member distance to the leaf mean — pairs with
    // the centroid into an admissible lower bound d(q,c) - r on the
    // distance to the nearest member (ball-tree pruning, no noise).
    std::vector<float> radius(L, 0.0f);
    for (uint32_t l = 0; l < L; ++l) {
        const double* c = &centroid[l * db];
        for (RowId rid : members[l]) {
            const float* v = &base[static_cast<size_t>(rid) * db];
            double d2 = 0;
            for (uint32_t d = 0; d < db; ++d) {
                const double diff = v[d] - c[d];
                d2 += diff * diff;
            }
            radius[l] = std::max(radius[l], static_cast<float>(std::sqrt(d2)));
        }
    }
    {
        // Sanity: GT ids must resolve into the home map.
        uint64_t found = 0, tot = 0, distinct = 0, oob = 0;
        std::unordered_map<int64_t, uint32_t> seen;
        for (auto& kv : home) { ++distinct; seen[kv.first] = 1; }
        for (uint32_t qi = 0; qi < std::min(nq, nqt); ++qi)
            for (uint32_t g = 0; g < 10; ++g) {
                const int32_t rid = gt[static_cast<size_t>(qi) * gt_stride + g];
                ++tot;
                if (rid < 0 || rid >= static_cast<int32_t>(nb)) ++oob;
                else if (home.count(rid)) ++found;
            }
        std::printf("[sanity] home distinct=%llu of nb=%u; GT ids found %llu/%llu (oob %llu)\n",
                    static_cast<unsigned long long>(distinct), nb,
                    static_cast<unsigned long long>(found),
                    static_cast<unsigned long long>(tot),
                    static_cast<unsigned long long>(oob));
        // Cross-check with row 0: print its home leaves and the leaf means'
        // distance to base row 0 (should be small vs other rows' distance).
    }
    std::printf("leaves=%u stored=%llu (%.3fx) total_pages=%llu\n", L,
                static_cast<unsigned long long>(stored),
                static_cast<double>(stored) / nb,
                static_cast<unsigned long long>(total_w));

    // --- Per-leaf ANCHOR SETS (plan A, rev 2): A sub-MEANS per leaf from
    // a recursive 2-means split (dominant-direction seed + reassignment),
    // NOT FPS members — peripheral anchors underperform the plain mean
    // (measured); dense-mode sub-means are the deployable approximation
    // of member-min. A leaf ranks by min L2 to its anchors.
    const uint32_t ANCHORS[] = {4, 16, 64, 256};
    constexpr size_t NA = 3;
    std::vector<std::vector<float>> anchor_sets[NA];  // [a][leaf][k*db]
    for (size_t ai = 0; ai < NA; ++ai) {
        const uint32_t A = ANCHORS[ai];
        anchor_sets[ai].resize(L);
        for (uint32_t l = 0; l < L; ++l) {
            const auto& mem = members[l];
            if (mem.size() < 2) continue;
            const double* c = &centroid[l * db];
            std::vector<float> pts(mem.size() * db);
            for (size_t i = 0; i < mem.size(); ++i)
                std::memcpy(&pts[i * db],
                            &base[static_cast<size_t>(mem[i]) * db],
                            db * sizeof(float));
            // Dominant direction of this leaf (power iteration on the
            // centered members), used to seed the first split.
            auto dom_dir = [&](const std::vector<double>& m) {
                std::vector<double> dir(db, 0.0);
                for (uint32_t d = 0; d < db; ++d)
                    dir[d] = pts[db + d] - m[d];
                for (int it = 0; it < 8; ++it) {
                    std::vector<double> nd(db, 0.0);
                    for (size_t i = 0; i < mem.size(); ++i)
                        for (uint32_t d = 0; d < db; ++d)
                            nd[d] += (pts[i * db + d] - m[d]) * dir[d];
                    double nrm = 1e-30;
                    for (uint32_t d = 0; d < db; ++d) nrm += nd[d] * nd[d];
                    nrm = std::sqrt(nrm);
                    for (uint32_t d = 0; d < db; ++d) dir[d] = nd[d] / nrm;
                }
                return dir;
            };
            std::vector<uint32_t> assign(mem.size(), 0);
            {
                const auto dir = dom_dir(std::vector<double>(c, c + db));
                for (size_t i = 0; i < mem.size(); ++i) {
                    double proj = 0;
                    for (uint32_t d = 0; d < db; ++d)
                        proj += (pts[i * db + d] - c[d]) * dir[d];
                    assign[i] = proj > 0 ? 1 : 0;
                }
            }
            // Grow to A clusters by splitting the largest along its
            // dominant direction.
            while (std::accumulate(assign.begin(), assign.end(), 0u,
                                   [&](uint32_t acc, uint32_t) { return acc + 1; }) >= 0) {
                uint32_t ncl = 0;
                for (uint32_t a : assign) ncl = std::max(ncl, a + 1);
                if (ncl >= A) break;
                std::vector<uint64_t> cnt(ncl, 0);
                for (uint32_t a : assign) ++cnt[a];
                uint32_t big = 0;
                for (uint32_t ci = 0; ci < ncl; ++ci)
                    if (cnt[ci] > cnt[big]) big = ci;
                std::vector<double> bm(db, 0.0);
                for (size_t i = 0; i < mem.size(); ++i)
                    if (assign[i] == big)
                        for (uint32_t d = 0; d < db; ++d) bm[d] += pts[i * db + d];
                for (uint32_t d = 0; d < db; ++d) bm[d] /= cnt[big];
                const auto dir = dom_dir(bm);
                for (size_t i = 0; i < mem.size(); ++i)
                    if (assign[i] == big) {
                        double proj = 0;
                        for (uint32_t d = 0; d < db; ++d)
                            proj += (pts[i * db + d] - bm[d]) * dir[d];
                        if (proj > 0) assign[i] = ncl;
                    }
            }
            // Two Lloyd passes that actually update the assignment.
            for (int it = 0; it < 2; ++it) {
                uint32_t ncl = 0;
                for (uint32_t a : assign) ncl = std::max(ncl, a + 1);
                std::vector<std::vector<double>> cs(
                    ncl, std::vector<double>(db, 0.0));
                std::vector<uint64_t> cnt(ncl, 0);
                for (size_t i = 0; i < mem.size(); ++i) {
                    ++cnt[assign[i]];
                    for (uint32_t d = 0; d < db; ++d)
                        cs[assign[i]][d] += pts[i * db + d];
                }
                for (uint32_t ci = 0; ci < ncl; ++ci)
                    if (cnt[ci])
                        for (uint32_t d = 0; d < db; ++d) cs[ci][d] /= cnt[ci];
                for (size_t i = 0; i < mem.size(); ++i) {
                    double bd2 = 1e30; uint32_t ba = 0;
                    for (uint32_t ci = 0; ci < ncl; ++ci) {
                        double d2 = 0;
                        for (uint32_t d = 0; d < db; ++d) {
                            const double diff = pts[i * db + d] - cs[ci][d];
                            d2 += diff * diff;
                        }
                        if (d2 < bd2) { bd2 = d2; ba = ci; }
                    }
                    assign[i] = ba;
                }
            }
            uint32_t ncl = 0;
            for (uint32_t a : assign) ncl = std::max(ncl, a + 1);
            std::vector<std::vector<double>> cs(ncl,
                                                std::vector<double>(db, 0.0));
            std::vector<uint64_t> cnt(ncl, 0);
            for (size_t i = 0; i < mem.size(); ++i) {
                ++cnt[assign[i]];
                for (uint32_t d = 0; d < db; ++d)
                    cs[assign[i]][d] += pts[i * db + d];
            }
            auto& store = anchor_sets[ai][l];
            store.reserve(ncl * db);
            for (uint32_t ci = 0; ci < ncl; ++ci) {
                if (!cnt[ci]) continue;
                for (uint32_t d = 0; d < db; ++d)
                    store.push_back(static_cast<float>(cs[ci][d] / cnt[ci]));
            }
        }
    }

    // Prefix fractions at which to report containment (by page weight).
    const double fracs[] = {0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0};
    constexpr size_t NF = sizeof(fracs) / sizeof(fracs[0]);
    std::vector<std::vector<double>> hit(4 + NA + 3, std::vector<double>(NF, 0.0));
    uint32_t q_used = std::min(nq, nqt);
    std::vector<uint32_t> leaf_order;

    for (uint32_t qi = 0; qi < q_used; ++qi) {
        const float* q = &query[static_cast<size_t>(qi) * db];
        // GT leaves for this query, weighted per NEIGHBOR (the engine's
        // routing metric: a neighbor is routed if ANY home leaf is probed;
        // closure replication gives some neighbors several homes).
        std::vector<uint32_t> gt_leaf_sets;  // flat: 10 neighbors × their homes
        std::vector<uint32_t> gt_offsets{0};
        for (uint32_t g = 0; g < gtk; ++g) {
            int32_t rid = gt[static_cast<size_t>(qi) * gt_stride + g];
            auto it = home.find(rid);
            if (it != home.end())
                for (uint32_t l : it->second) gt_leaf_sets.push_back(l);
            gt_offsets.push_back(
                static_cast<uint32_t>(gt_leaf_sets.size()));
        }
        const uint32_t n_nb =
            static_cast<uint32_t>(gt_offsets.size() - 1);
        if (gt_leaf_sets.empty()) continue;
        std::vector<char> is_gt(L, 0);
        for (uint32_t l : gt_leaf_sets) is_gt[l] = 1;

        // --- b. exact-centroid order ---
        std::vector<std::pair<float, uint32_t>> cd(L);
        for (uint32_t l = 0; l < L; ++l) {
            double d2 = 0;
            const double* c = &centroid[l * db];
            for (uint32_t d = 0; d < db; ++d) {
                const double diff = q[d] - c[d];
                d2 += diff * diff;
            }
            cd[l] = {static_cast<float>(d2), l};
        }
        std::sort(cd.begin(), cd.end());
        std::vector<uint32_t> centroid_order(L);
        for (uint32_t i = 0; i < L; ++i) centroid_order[i] = cd[i].second;

        // --- bb. admissible bound order: d(q,c) - r (ball-tree) ---
        std::vector<std::pair<float, uint32_t>> bo(L);
        for (uint32_t l = 0; l < L; ++l)
            bo[l] = {std::sqrt(cd[l].first) - radius[l], l};
        std::sort(bo.begin(), bo.end());
        std::vector<uint32_t> bound_order(L);
        for (uint32_t i = 0; i < L; ++i) bound_order[i] = bo[i].second;

        // --- a. random order (deterministic per query) ---
        leaf_order.resize(L);
        std::iota(leaf_order.begin(), leaf_order.end(), 0u);
        uint32_t s = qi * 2654435761u + 1;
        for (uint32_t i = L - 1; i > 0; --i) {
            s = s * 1664525u + 1013904223u;
            const uint32_t j = s % (i + 1);
            std::swap(leaf_order[i], leaf_order[j]);
        }

        // Containment vs prefix fraction under each order, per neighbor:
        // neighbor i is covered when any leaf in gt_leaf_sets[offsets[i]..]
        // has appeared in the prefix.
        auto walk = [&](const std::vector<uint32_t>& order, size_t oi) {
            std::vector<uint8_t> done(n_nb, 0);
            uint64_t cum = 0, covered = 0;
            size_t fi = 0;
            for (uint32_t l : order) {
                cum += weight[l];
                if (is_gt[l]) {
                    // mark neighbors having this leaf as a home
                    for (uint32_t i = 0; i < n_nb; ++i) {
                        if (done[i]) continue;
                        for (uint32_t j = gt_offsets[i]; j < gt_offsets[i + 1]; ++j)
                            if (gt_leaf_sets[j] == l) { done[i] = 1; ++covered; break; }
                    }
                }
                while (fi < NF &&
                       static_cast<double>(cum) >= fracs[fi] * total_w - 1e-9) {
                    hit[oi][fi] += static_cast<double>(covered) / n_nb;
                    ++fi;
                }
            }
            while (fi < NF) { hit[oi][fi] += 1.0; ++fi; }
        };

        walk(leaf_order, 0);
        walk(centroid_order, 1);
        walk(bound_order, 6 + NA);  // bound column (every query)

        // --- z. oracle order (GT leaves first) — harness self-check:
        // must read ~1.0 from the smallest fraction onward.
        {
            std::vector<uint32_t> order;
            order.reserve(L);
            for (uint32_t l = 0; l < L; ++l) if (is_gt[l]) order.push_back(l);
            for (uint32_t l = 0; l < L; ++l) if (!is_gt[l]) order.push_back(l);
            walk(order, 3);
        }

        // --- c. member-min (ideal) + PARTIAL-DIM variants, on a subset.
        // partial-D is plan D's proxy: rank leaves by the best IP over
        // members restricted to the first PD dims — what a coarse 4-bit
        // code sweep over PD dims would approximate (codes exist for
        // every member; here we use fp32 vectors as the ceiling).
        if (qi < mm_q) {
            std::vector<float> best(L, -std::numeric_limits<float>::max());
            const uint32_t PD1 = db / 12;   // 128 at 1536
            const uint32_t PD2 = db / 4;    // 384 at 1536
            std::vector<float> pbest1(L, -std::numeric_limits<float>::max());
            std::vector<float> pbest2(L, -std::numeric_limits<float>::max());
            for (uint32_t l = 0; l < L; ++l) {
                for (RowId rid : members[l]) {
                    const float* v = &base[static_cast<size_t>(rid) * db];
                    float ip = 0, ip1 = 0, ip2 = 0;
                    for (uint32_t d = 0; d < db; ++d) {
                        const float qq = q[d] * v[d];
                        ip += qq;
                        if (d < PD1) ip1 += qq;
                        if (d < PD2) ip2 += qq;
                    }
                    best[l] = std::max(best[l], ip);
                    pbest1[l] = std::max(pbest1[l], ip1);
                    pbest2[l] = std::max(pbest2[l], ip2);
                }
            }
            // partial-1/12 order (plan D target)
            {
                std::vector<std::pair<float, uint32_t>> po(L);
                for (uint32_t l = 0; l < L; ++l) po[l] = {-pbest1[l], l};
                std::sort(po.begin(), po.end());
                std::vector<uint32_t> order(L);
                for (uint32_t i = 0; i < L; ++i) order[i] = po[i].second;
                walk(order, 4 + NA);
            }
            // partial-1/4 order (fallback resolution)
            {
                std::vector<std::pair<float, uint32_t>> po(L);
                for (uint32_t l = 0; l < L; ++l) po[l] = {-pbest2[l], l};
                std::sort(po.begin(), po.end());
                std::vector<uint32_t> order(L);
                for (uint32_t i = 0; i < L; ++i) order[i] = po[i].second;
                walk(order, 5 + NA);
            }
            std::vector<std::pair<float, uint32_t>> mo(L);
            for (uint32_t l = 0; l < L; ++l) mo[l] = {-best[l], l};
            std::sort(mo.begin(), mo.end());
            std::vector<uint32_t> order(L);
            for (uint32_t i = 0; i < L; ++i) order[i] = mo[i].second;
            walk(order, 2);

            // --- anchors: min L2 over the leaf's A anchors ---
            for (size_t ai = 0; ai < NA; ++ai) {
                std::vector<std::pair<float, uint32_t>> ao(L);
                for (uint32_t l = 0; l < L; ++l) {
                    const auto& anc = anchor_sets[ai][l];
                    float bestd = std::numeric_limits<float>::max();
                    const size_t cnt = anc.size() / db;
                    for (size_t aidx = 0; aidx < cnt; ++aidx) {
                        const float* v = &anc[aidx * db];
                        float d2 = 0;
                        for (uint32_t d = 0; d < db; ++d) {
                            const float diff = q[d] - v[d];
                            d2 += diff * diff;
                        }
                        bestd = std::min(bestd, d2);
                    }
                    ao[l] = {bestd, l};
                }
                std::sort(ao.begin(), ao.end());
                std::vector<uint32_t> aorder(L);
                for (uint32_t i = 0; i < L; ++i) aorder[i] = ao[i].second;
                walk(aorder, 4 + ai);
            }
        }
    }

    // --- Engine points: (probed page fraction, per-neighbor routing
    // containment) under the engine's own PCA routing, np = 1/2/4/8.
    {
        std::unordered_map<uint64_t, uint64_t> page_weight;
        for (uint32_t l = 0; l < L; ++l)
            if (leaf_info[l].page != kInvalidPage)
                page_weight[leaf_info[l].page] = weight[l];
        for (uint32_t np : {1u, 2u, 4u, 8u}) {
            SearchConfig cfg;
            cfg.k = 10;
            cfg.n_probe = np;
            cfg.n_probe_ln = 8;
            cfg.fastscan_W = 1000;
            double cont_sum = 0, frac_sum = 0, delivered_sum = 0;
            uint32_t used = 0;
            for (uint32_t qi = 0; qi < q_used; ++qi) {
                const float* q = &query[static_cast<size_t>(qi) * db];
                // recompute per-neighbor homes for this query
                std::vector<std::vector<uint32_t>> nb_homes;
                for (uint32_t g = 0; g < 10; ++g) {
                    std::vector<uint32_t> h;
                    auto it = home.find(
                        gt[static_cast<size_t>(qi) * gt_stride + g]);
                    if (it != home.end()) h = it->second;
                    nb_homes.push_back(std::move(h));
                }
                uint32_t have = 0;
                for (auto& h : nb_homes) if (!h.empty()) ++have;
                if (!have) continue;
                std::vector<PageId> visited;
                auto res = index->search(q, 10, cfg, nullptr, nullptr,
                                         nullptr, &visited);
                // delivered: returned ids that are GT top-10
                uint32_t del = 0;
                for (const auto& c : res)
                    for (uint32_t g = 0; g < 10; ++g)
                        if (static_cast<int32_t>(c.row_id) ==
                            gt[static_cast<size_t>(qi) * gt_stride + g]) {
                            ++del; break;
                        }
                delivered_sum += static_cast<double>(del) / 10;
                uint64_t w = 0;
                std::unordered_set<uint64_t> vp(visited.begin(),
                                                visited.end());
                for (auto pg : vp) {
                    auto it2 = page_weight.find(pg);
                    if (it2 != page_weight.end()) w += it2->second;
                }
                uint32_t cov = 0;
                for (auto& h : nb_homes) {
                    bool hit = false;
                    for (uint32_t l : h) {
                        auto it3 = page_weight.find(leaf_info[l].page);
                        if (it3 != page_weight.end() && vp.count(it3->first)) {
                            hit = true; break;
                        }
                    }
                    if (hit) ++cov;
                }
                // dedup: a leaf id may repeat through homes; containment
                // counts each neighbor once.
                cont_sum += static_cast<double>(cov) / have;
                frac_sum += static_cast<double>(w) / total_w;
                ++used;
            }
            std::printf("[engine] np=%u: probed frac=%.3f  routing=%.4f  delivered=%.4f  (%u queries)\n",
                        np, frac_sum / used, cont_sum / used,
                        delivered_sum / used, used);
        }
    }

    std::printf("\n%-8s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s  (ideal rows over %u queries)\n",
                "frac", "random", "centrd", "mbrmin", "oracle",
                "an04", "an16", "an64", "a256", "bound", "p1/12", "p1/4", mm_q);
    for (size_t fi = 0; fi < NF; ++fi) {
        std::printf("%-8.2f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n",
                    fracs[fi],
                    hit[0][fi] / q_used, hit[1][fi] / q_used,
                    hit[2][fi] / std::min(mm_q, q_used),
                    hit[3][fi] / q_used,
                    hit[4][fi] / std::min(mm_q, q_used),
                    hit[5][fi] / std::min(mm_q, q_used),
                    hit[6][fi] / std::min(mm_q, q_used),
                    hit[7][fi] / std::min(mm_q, q_used),
                    hit[6 + NA][fi] / q_used,
                    hit[4 + NA][fi] / std::min(mm_q, q_used),
                    hit[5 + NA][fi] / std::min(mm_q, q_used));
    }
    return 0;
}
