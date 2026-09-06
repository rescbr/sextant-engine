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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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
    int base_err = 0;

    uint32_t nb, db, nqt, dq, gtk;
    // Base is mmap'd read-only: 10M-scale corpora (29 GB at cohere-10m)
    // must not live on the heap next to the plane projections.
    const float* base = [](const char* path, uint32_t& n, uint32_t& d,
                           int& err) -> const float* {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) { err = 1; return nullptr; }
        struct stat st;
        if (::fstat(fd, &st) != 0 || st.st_size < 8) {
            ::close(fd); err = 1; return nullptr;
        }
        void* m = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) { err = 1; return nullptr; }
        const auto* hdr = static_cast<const uint32_t*>(m);
        n = hdr[0]; d = hdr[1];
        if (static_cast<uint64_t>(n) * d * 4 + 8 >
            static_cast<uint64_t>(st.st_size)) { err = 1; return nullptr; }
        return reinterpret_cast<const float*>(hdr + 2);
    }(argv[2], nb, db, base_err);
    if (base_err) {
        std::fprintf(stderr, "base mmap failed\n"); return 1;
    }
    auto query = load_fbin(argv[3], nqt, dq);
    std::vector<int32_t> gt;
    if (!load_gtmm(argv[4], gt, gtk) || gtk < 10) {
        std::fprintf(stderr, "gt load failed / k<10\n"); return 1;
    }
    const uint32_t gt_stride = gtk;  // file row width (may be > 10)
    gtk = 10;
    std::printf("base=%u dim=%u queries=%u(used %u) gt_k=10 mm_queries=%u\n",
                nb, db, nqt, nq, mm_q);

    // Clustered-ingest artifacts (milestone-2): SPIKE_BASIS_ORDER (row
    // order file, i32) makes the basis train along that ingest order;
    // SPIKE_INVORDER maps row_id -> position in that order, enabling
    // query filtering by neighbor ingest position (SPIKE_GT_POS tercile).
    std::vector<int32_t> basis_order, inv_order;
    if (const char* o = std::getenv("SPIKE_BASIS_ORDER")) {
        std::ifstream f(o, std::ios::binary);
        if (!f) { std::fprintf(stderr, "open %s failed\n", o); return 1; }
        basis_order.resize(nb);
        f.read(reinterpret_cast<char*>(basis_order.data()),
               static_cast<std::streamsize>(nb * 4));
        std::fprintf(stderr, "[order] loaded %u-row ingest order\n", nb);
    }
    if (const char* o = std::getenv("SPIKE_INVORDER")) {
        std::ifstream f(o, std::ios::binary);
        if (!f) { std::fprintf(stderr, "open %s failed\n", o); return 1; }
        inv_order.resize(nb);
        f.read(reinterpret_cast<char*>(inv_order.data()),
               static_cast<std::streamsize>(nb * 4));
    }
    const int gt_pos = [] { const char* e = std::getenv("SPIKE_GT_POS");
                            return e ? std::atoi(e) : -1; }();

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
    std::vector<uint64_t> stored_per_leaf(L, 0);
#pragma omp parallel for schedule(dynamic)
    for (uint32_t l = 0; l < L; ++l) {
        if (leaf_info[l].page == kInvalidPage) continue;
        members[l] = index->debug_leaf_row_ids(l);
        weight[l] = leaf_info[l].pages;
        stored_per_leaf[l] = members[l].size();
        double* c = &centroid[l * db];
        for (RowId rid : members[l]) {
            const float* v = &base[static_cast<size_t>(rid) * db];
            for (uint32_t d = 0; d < db; ++d) c[d] += v[d];
        }
    }
    for (uint32_t l = 0; l < L; ++l) {
        if (leaf_info[l].page == kInvalidPage) continue;
        total_w += weight[l];
        stored += stored_per_leaf[l];
        for (RowId rid : members[l]) home[rid].push_back(l);
    }
    for (uint32_t l = 0; l < L; ++l) {
        const size_t m = members[l].size();
        if (m) for (uint32_t d = 0; d < db; ++d) centroid[l * db + d] /= m;
    }
    // --- PCA projections of members (plan-D plane proxy): rank leaves
    // by max over members of IP in the top-P PCA directions. Basis via
    // power iteration + deflation on a 20K-row sample covariance.
    const uint32_t PW[] = {32, 128};
    constexpr size_t NP = 2;
    std::vector<std::vector<float>> proj_members[NP];  // [p][leaf][m*P]
    std::vector<std::vector<double>> pca_basis;
    std::vector<double> pca_mean;
    {
        // Basis-training window knobs (milestone-2 drift sim):
        //   SPIKE_BASIS_PREFIX  fraction of the sample that trains the
        //                       basis (0.05 = early ingest prefix)
        //   SPIKE_BASIS_OFFSET  fraction of nb where the window starts
        //                       (0.5 = mid-corpus ingest window)
        const float bf = [] { const char* e = std::getenv("SPIKE_BASIS_PREFIX");
                              return e ? std::atof(e) : 1.0f; }();
        const float bo = [] { const char* e = std::getenv("SPIKE_BASIS_OFFSET");
                              return e ? std::atof(e) : 0.0f; }();
        const uint32_t SAMPLE = std::min<uint32_t>(20000, nb);
        const uint32_t off = static_cast<uint32_t>(
            std::min<double>(bo, 0.99) * (nb - SAMPLE));
        const uint32_t TRAIN = std::max(1000u,
            static_cast<uint32_t>(bf * SAMPLE));
        std::fprintf(stderr, "[pca] basis trains on rows [%u, %u) of %u\n",
                     off, off + TRAIN, nb);
        pca_mean.assign(db, 0.0);
        auto& mean = pca_mean;
        auto train_row = [&](uint32_t i) {
            const uint32_t r = basis_order.empty() ? off + i
                : static_cast<uint32_t>(basis_order[off + i]);
            return static_cast<size_t>(r) * db;
        };
        for (uint32_t i = 0; i < TRAIN; ++i)
            for (uint32_t d = 0; d < db; ++d)
                mean[d] += base[train_row(i) + d] / TRAIN;
        std::vector<double> cov(db * db, 0.0);
        for (uint32_t i = 0; i < TRAIN; ++i) {
            const float* v = &base[train_row(i)];
            for (uint32_t r = 0; r < db; ++r) {
                const double vr = v[r] - mean[r];
                for (uint32_t c2 = r; c2 < db; ++c2)
                    cov[r * db + c2] += vr * (v[c2] - mean[c2]) / SAMPLE;
            }
        }
        for (uint32_t r = 0; r < db; ++r)
            for (uint32_t c2 = 0; c2 < r; ++c2)
                cov[r * db + c2] = cov[c2 * db + r];
        auto& basis = pca_basis;
        std::vector<double> resid = cov;
        const uint32_t PMAX = 128;
        for (uint32_t e = 0; e < PMAX; ++e) {
            std::vector<double> v(db, 1.0 / std::sqrt((double)db));
            for (int it = 0; it < 40; ++it) {
                std::vector<double> w(db, 0.0);
                for (uint32_t r = 0; r < db; ++r) {
                    double acc = 0;
                    for (uint32_t c2 = 0; c2 < db; ++c2) acc += resid[r * db + c2] * v[c2];
                    w[r] = acc;
                }
                double nrm = 1e-30;
                for (uint32_t d = 0; d < db; ++d) nrm += w[d] * w[d];
                nrm = std::sqrt(nrm);
                for (uint32_t d = 0; d < db; ++d) v[d] = w[d] / nrm;
            }
            basis.push_back(v);
            for (uint32_t r = 0; r < db; ++r)
                for (uint32_t c2 = 0; c2 < db; ++c2)
                    resid[r * db + c2] -= v[r] * v[c2] * 0;  // deflate below
            // deflate: resid -= lambda * v v^T (lambda = v^T resid v)
            double lambda = 0;
            for (uint32_t r = 0; r < db; ++r)
                for (uint32_t c2 = 0; c2 < db; ++c2) lambda += v[r] * resid[r * db + c2] * v[c2];
            for (uint32_t r = 0; r < db; ++r)
                for (uint32_t c2 = 0; c2 < db; ++c2)
                    resid[r * db + c2] -= lambda * v[r] * v[c2];
        }
        for (size_t pi = 0; pi < NP; ++pi) {
            const uint32_t P = PW[pi];
            proj_members[pi].resize(L);
            for (uint32_t l = 0; l < L; ++l) {
                const auto& mem = members[l];
                if (mem.empty()) continue;
                auto& store = proj_members[pi][l];
                store.resize(mem.size() * P);
#pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < mem.size(); ++i) {
                    const float* v = &base[static_cast<size_t>(mem[i]) * db];
                    for (uint32_t e = 0; e < P; ++e) {
                        const auto& b = basis[e];
                        double acc = 0;
                        for (uint32_t d = 0; d < db; ++d) acc += (v[d] - mean[d]) * b[d];
                        store[i * P + e] = static_cast<float>(acc);
                    }
                }
            }
        }
        std::fprintf(stderr, "[pca] basis ready (32/128 of %u dims)\n", db);
    }

    // i8-quantized plane variants (deployable bytes points): global
    // per-dim scales over sampled members, int8 projections, integer
    // dot. i8@128 = 128 B/vec; i8@64 = 64 B/vec.
    const uint32_t I8R[] = {128, 64};
    constexpr size_t NI8 = 2;
    std::vector<std::vector<float>> i8_scale(NI8);
    std::vector<std::vector<std::vector<int8_t>>> proj_i8(NI8);
    {
        const size_t SRC = 1;  // the 128-d projections (prefix for 64)
        for (size_t ii = 0; ii < NI8; ++ii) {
            const uint32_t P = I8R[ii];
            i8_scale[ii].assign(P, 1e-30f);
            for (uint32_t l = 0; l < L; l += std::max(1u, L / 32)) {
                const auto& pm = proj_members[SRC][l];
                const size_t cnt = pm.size() / 128;
                for (size_t i = 0; i < cnt; i += std::max<size_t>(1, cnt / 64))
                    for (uint32_t e = 0; e < P; ++e)
                        i8_scale[ii][e] = std::max(
                            i8_scale[ii][e], std::abs(pm[i * 128 + e]));
            }
            for (uint32_t e = 0; e < P; ++e)
                i8_scale[ii][e] = 127.0f / i8_scale[ii][e];
            proj_i8[ii].resize(L);
#pragma omp parallel for schedule(dynamic)
            for (uint32_t l = 0; l < L; ++l) {
                const auto& pm = proj_members[SRC][l];
                const size_t cnt = pm.size() / 128;
                if (!cnt) continue;
                auto& st = proj_i8[ii][l];
                st.resize(cnt * P);
                for (size_t i = 0; i < cnt; ++i)
                    for (uint32_t e = 0; e < P; ++e) {
                        int v = static_cast<int>(
                            std::lround(pm[i * 128 + e] * i8_scale[ii][e]));
                        st[i * P + e] = static_cast<int8_t>(
                            std::clamp(v, -127, 127));
                    }
            }
        }
        std::fprintf(stderr, "[pca] i8 planes ready (128/64 dims)\n");
    }

    // Per-leaf RADIUS: max member distance to the leaf mean — pairs with
    // the centroid into an admissible lower bound d(q,c) - r on the
    // distance to the nearest member (ball-tree pruning, no noise).
    // SPIKE_NO_BOUND skips this leaf-ordered pass (the one random-access
    // mmap phase left); the bound column is falsified anyway.
    std::vector<float> radius(L, 0.0f);
    if (std::getenv("SPIKE_NO_BOUND")) {
        std::fprintf(stderr, "[bound] skipped (SPIKE_NO_BOUND)\n");
    } else
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
    // Anchor family already measured+falsified; SPIKE_NO_ANCHORS=1
    // skips the (expensive) k-means precompute for corpus-2 runs that
    // only need the plane/rank columns.
    static const bool no_anchors = std::getenv("SPIKE_NO_ANCHORS") != nullptr;
    const uint32_t ANCHORS_arr[] = {4, 16, 64, 256};
    std::vector<uint32_t> ANCHORS_v(
        no_anchors ? ANCHORS_arr : ANCHORS_arr,
        no_anchors ? ANCHORS_arr : ANCHORS_arr + 4);
    if (no_anchors) ANCHORS_v.clear();
    const auto& ANCHORS = ANCHORS_v;
    constexpr size_t NA_MAX = 4;
    const size_t NA = ANCHORS.size();
    std::vector<std::vector<float>> anchor_sets[NA_MAX];  // [a][leaf][k*db]
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
    std::vector<std::vector<double>> hit(18, std::vector<double>(NF, 0.0));
    uint32_t q_used = std::min(nq, nqt);
    uint32_t q_scored = 0;    // queries passing the GT_POS filter
    uint32_t q_scored_mm = 0; // ... and inside the member-min subset
    std::vector<uint32_t> leaf_order;

    // --- adaptive (best-first) stopping study (SPIKE_ADAPTIVE): probe
    // leaves in descending i8-plane score (the observable stage-1 sweep
    // estimate) and stop when the next leaf's best-member IP estimate
    // falls below alpha x the best score seen. No GT in the rule — GT
    // only scores the outcome. Fixed-fraction rows on the same ordering
    // are the control: the gap is what per-query adaptivity buys.
    static const bool adaptive = std::getenv("SPIKE_ADAPTIVE") != nullptr;
    const float AD_ALPHAS[] = {0.0f, 0.3f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    constexpr size_t NAL = 7;
    constexpr size_t NAFF = 3;
    const double AFF_FR[NAFF] = {0.05, 0.10, 0.20};
    std::vector<std::vector<double>> ad_frac(NAL + NAFF), ad_cont(NAL + NAFF);

    // Batched member-min / partial-dim precompute: one pass over the
    // mmap'd base in ROW order (sequential readahead) scoring ALL mm
    // queries at once. Per-query leaf-ordered scans are random-access
    // page-fault storms at 10M scale (~20x slower). home is the exact
    // inverse of the per-leaf member lists, so the max is identical.
    const uint32_t MMQ = std::min(mm_q, nqt);
    const uint32_t PD1 = db / 12;   // 128 at 1536
    const uint32_t PD2 = db / 4;    // 384 at 1536
    std::vector<float> mm_best((size_t)L * MMQ,
        -std::numeric_limits<float>::max());
    std::vector<float> mm_p1((size_t)L * MMQ,
        -std::numeric_limits<float>::max());
    std::vector<float> mm_p2((size_t)L * MMQ,
        -std::numeric_limits<float>::max());
    {
        std::vector<const float*> mq(MMQ);
        for (uint32_t qi2 = 0; qi2 < MMQ; ++qi2)
            mq[qi2] = &query[static_cast<size_t>(qi2) * db];
        std::fprintf(stderr, "[mm] batched member-min pass (%u queries)\n", MMQ);
#pragma omp parallel for schedule(dynamic, 2048)
        for (size_t r = 0; r < nb; ++r) {
            auto it = home.find(static_cast<int64_t>(r));
            if (it == home.end() || it->second.empty()) continue;
            const float* v = &base[r * db];
            for (uint32_t l : it->second) {
                float* b1 = &mm_best[(size_t)l * MMQ];
                float* b2 = &mm_p1[(size_t)l * MMQ];
                float* b3 = &mm_p2[(size_t)l * MMQ];
                for (uint32_t qi2 = 0; qi2 < MMQ; ++qi2) {
                    const float* qv = mq[qi2];
                    float ip = 0, ip1 = 0, ip2 = 0;
                    for (uint32_t d = 0; d < db; ++d) {
                        const float qq = qv[d] * v[d];
                        ip += qq;
                        if (d < PD1) ip1 += qq;
                        if (d < PD2) ip2 += qq;
                    }
                    if (ip > b1[qi2]) b1[qi2] = ip;
                    if (ip1 > b2[qi2]) b2[qi2] = ip1;
                    if (ip2 > b3[qi2]) b3[qi2] = ip2;
                }
            }
        }
        std::fprintf(stderr, "[mm] batched member-min done\n");
    }

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
        // Tercile filter: score only queries whose neighbors' mean
        // position in the clustered ingest order falls in tercile
        // gt_pos (0=early ... 2=late). Requires the inverse order.
        if (gt_pos >= 0) {
            if (inv_order.empty()) {
                std::fprintf(stderr, "SPIKE_GT_POS needs SPIKE_INVORDER\n");
                return 1;
            }
            double meanpos = 0; uint32_t cnt = 0;
            for (uint32_t g = 0; g < 10; ++g) {
                const int32_t rid =
                    gt[static_cast<size_t>(qi) * gt_stride + g];
                if (rid >= 0 && rid < static_cast<int32_t>(nb)) {
                    meanpos += inv_order[rid]; ++cnt;
                }
            }
            if (cnt) meanpos /= cnt * nb;  // in [0,1)
            const int tercile = static_cast<int>(meanpos * 3.0);
            if (tercile != gt_pos) continue;
        }
        ++q_scored;
        if (qi < mm_q) ++q_scored_mm;
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
        walk(bound_order, 8);  // bound column (every query)

        if (adaptive) {
            const uint32_t P = I8R[0];
            std::vector<int8_t> qpi(P);
            {
                std::vector<float> qp(P);
                for (uint32_t e = 0; e < P; ++e) {
                    double acc = 0;
                    const auto& bvec = pca_basis[e];
                    for (uint32_t d = 0; d < db; ++d)
                        acc += (q[d] - pca_mean[d]) * bvec[d];
                    qp[e] = static_cast<float>(acc);
                }
                for (uint32_t e = 0; e < P; ++e) {
                    int v = static_cast<int>(
                        std::lround(qp[e] * i8_scale[0][e]));
                    qpi[e] = static_cast<int8_t>(std::clamp(v, -127, 127));
                }
            }
            std::vector<float> score(L, -std::numeric_limits<float>::max());
#pragma omp parallel for schedule(dynamic)
            for (uint32_t l = 0; l < L; ++l) {
                const auto& pm = proj_i8[0][l];
                const size_t cnt = pm.size() / P;
                if (!cnt) continue;
                int32_t best = -0x7fffffff;
                for (size_t i = 0; i < cnt; ++i) {
                    int32_t acc = 0;
                    for (uint32_t e = 0; e < P; ++e)
                        acc += qpi[e] * pm[i * P + e];
                    if (acc > best) best = acc;
                }
                score[l] = static_cast<float>(best);
            }
            std::vector<std::pair<float, uint32_t>> so(L);
            for (uint32_t l = 0; l < L; ++l) so[l] = {-score[l], l};
            std::sort(so.begin(), so.end());
            // one ordered walk: cumulative weight per position, and each
            // neighbor's first-covered position
            std::vector<uint64_t> cumw(L);
            std::vector<uint32_t> first_pos(n_nb, L);
            uint64_t cw = 0;
            for (uint32_t pos = 0; pos < L; ++pos) {
                const uint32_t l = so[pos].second;
                cw += weight[l];
                cumw[pos] = cw;
                if (is_gt[l])
                    for (uint32_t i = 0; i < n_nb; ++i) {
                        if (first_pos[i] != L) continue;
                        for (uint32_t j = gt_offsets[i];
                             j < gt_offsets[i + 1]; ++j)
                            if (gt_leaf_sets[j] == l) { first_pos[i] = pos; break; }
                    }
            }
            const double s0 = score[so[0].second];
            auto cont_at = [&](uint32_t cut) {
                uint32_t c = 0;
                for (uint32_t i = 0; i < n_nb; ++i)
                    if (first_pos[i] <= cut) ++c;
                return static_cast<double>(c) / n_nb;
            };
            auto frac_at = [&](uint32_t cut) {
                return cut >= L ? 1.0
                    : static_cast<double>(cumw[cut]) / total_w;
            };
            for (size_t ai = 0; ai < NAL; ++ai) {
                uint32_t cut = 0;
                while (cut < L && score[so[cut].second] >= AD_ALPHAS[ai] * s0)
                    ++cut;
                if (cut) --cut;
                ad_frac[ai].push_back(frac_at(cut));
                ad_cont[ai].push_back(cont_at(cut));
            }
            for (size_t fi = 0; fi < NAFF; ++fi) {
                uint32_t cut = 0;
                while (cut < L - 1 && frac_at(cut) < AFF_FR[fi]) ++cut;
                ad_frac[NAL + fi].push_back(frac_at(cut));
                ad_cont[NAL + fi].push_back(cont_at(cut));
            }
        }

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
        if (qi < MMQ) {
            // per-leaf member-min / partial-dim scores for this query,
            // precomputed in the batched row-order pass above
            std::vector<float> best(L), pbest1(L), pbest2(L);
            for (uint32_t l = 0; l < L; ++l) {
                best[l] = mm_best[(size_t)l * MMQ + qi];
                pbest1[l] = mm_p1[(size_t)l * MMQ + qi];
                pbest2[l] = mm_p2[(size_t)l * MMQ + qi];
            }
            // PCA-P member orders (plane proxy): max projected IP
            for (size_t pi = 0; pi < NP; ++pi) {
                const uint32_t P = PW[pi];
                std::vector<float> qp(P);
                for (uint32_t e = 0; e < P; ++e) {
                    const auto& b = pca_basis[e];
                    double acc = 0;
                    for (uint32_t d = 0; d < db; ++d)
                        acc += (q[d] - pca_mean[d]) * b[d];
                    qp[e] = static_cast<float>(acc);
                }
                std::vector<float> pbest(L, -std::numeric_limits<float>::max());
#pragma omp parallel for schedule(dynamic)
                for (uint32_t l = 0; l < L; ++l) {
                    const auto& pm = proj_members[pi][l];
                    const size_t cnt = pm.size() / P;
                    for (size_t i = 0; i < cnt; ++i) {
                        float ip = 0;
                        for (uint32_t e = 0; e < P; ++e)
                            ip += qp[e] * pm[i * P + e];
                        pbest[l] = std::max(pbest[l], ip);
                    }
                }
                std::vector<std::pair<float, uint32_t>> po(L);
                for (uint32_t l = 0; l < L; ++l) po[l] = {-pbest[l], l};
                std::sort(po.begin(), po.end());
                std::vector<uint32_t> order(L);
                for (uint32_t i = 0; i < L; ++i) order[i] = po[i].second;
                walk(order, 11 + pi);
                if (pi == 1) {  // i8 uses the 128-d projections
                for (size_t ii = 0; ii < NI8; ++ii) {
                    const uint32_t P = I8R[ii];
                    std::vector<int8_t> qpi(P);
                    for (uint32_t e = 0; e < P; ++e) {
                        int v = static_cast<int>(
                            std::lround(qp[e] * i8_scale[ii][e]));
                        qpi[e] = static_cast<int8_t>(
                            std::clamp(v, -127, 127));
                    }
                    std::vector<float> rbest(
                        L, -std::numeric_limits<float>::max());
#pragma omp parallel for schedule(dynamic)
                    for (uint32_t l = 0; l < L; ++l) {
                        const auto& pm = proj_i8[ii][l];
                        const size_t cnt = pm.size() / P;
                        for (size_t i = 0; i < cnt; ++i) {
                            int32_t acc = 0;
                            for (uint32_t e = 0; e < P; ++e)
                                acc += qpi[e] * pm[i * P + e];
                            rbest[l] = std::max(rbest[l],
                                                static_cast<float>(acc));
                        }
                    }
                    std::vector<std::pair<float, uint32_t>> ro(L);
                    for (uint32_t l = 0; l < L; ++l)
                        ro[l] = {-rbest[l], l};
                    std::sort(ro.begin(), ro.end());
                    std::vector<uint32_t> rorder(L);
                    for (uint32_t i = 0; i < L; ++i)
                        rorder[i] = ro[i].second;
                    walk(rorder, 16 + ii);
                }
                // Rank sweep from the NESTED 128-d projections: prefix
                }
                // sums give every rank <= 128 for free.
                if (pi == 1) {
                    for (uint32_t RANK : {48u, 64u, 96u}) {
                        std::vector<float> rbest(
                            L, -std::numeric_limits<float>::max());
#pragma omp parallel for schedule(dynamic)
                        for (uint32_t l = 0; l < L; ++l) {
                            const auto& pm = proj_members[pi][l];
                            const size_t cnt = pm.size() / 128;
                            for (size_t i = 0; i < cnt; ++i) {
                                float ip = 0;
                                for (uint32_t e = 0; e < RANK; ++e)
                                    ip += qp[e] * pm[i * 128 + e];
                                rbest[l] = std::max(rbest[l], ip);
                            }
                        }
                        std::vector<std::pair<float, uint32_t>> ro(L);
                        for (uint32_t l = 0; l < L; ++l)
                            ro[l] = {-rbest[l], l};
                        std::sort(ro.begin(), ro.end());
                        std::vector<uint32_t> rorder(L);
                        for (uint32_t i = 0; i < L; ++i)
                            rorder[i] = ro[i].second;
                        // slot by rank: 48->+3, 64->+4, 96->+5 past the
                        // two plane columns
                        walk(rorder, RANK == 48u ? 13
                                          : RANK == 64u ? 14 : 15);
                    }
                }
            }
            // partial-1/12 order (plan D target)
            {
                std::vector<std::pair<float, uint32_t>> po(L);
                for (uint32_t l = 0; l < L; ++l) po[l] = {-pbest1[l], l};
                std::sort(po.begin(), po.end());
                std::vector<uint32_t> order(L);
                for (uint32_t i = 0; i < L; ++i) order[i] = po[i].second;
                walk(order, 9);;
            }
            // partial-1/4 order (fallback resolution)
            {
                std::vector<std::pair<float, uint32_t>> po(L);
                for (uint32_t l = 0; l < L; ++l) po[l] = {-pbest2[l], l};
                std::sort(po.begin(), po.end());
                std::vector<uint32_t> order(L);
                for (uint32_t i = 0; i < L; ++i) order[i] = po[i].second;
                walk(order, 10);
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

    std::fprintf(stderr, "[gt] scored %u of %u presented queries\n", q_scored, q_used);

    // --- Engine points: (probed page fraction, per-neighbor routing
    // containment) under the engine's own PCA routing, np = 1/2/4/8.
    {
        std::unordered_map<uint64_t, uint64_t> page_weight;
        for (uint32_t l = 0; l < L; ++l)
            if (leaf_info[l].page != kInvalidPage)
                page_weight[leaf_info[l].page] = weight[l];
        for (uint32_t np : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u}) {
            SearchConfig cfg;
            cfg.k = 10;
            cfg.n_probe = np;
            cfg.n_probe_ln = 64;  // cover all leaves per root child on
                                  // fine trees (cap-1250 has ~10/child)
            cfg.fastscan_W = 1000;
            double cont_sum = 0, frac_sum = 0, delivered_sum = 0;
            uint32_t used = 0;
            for (uint32_t qi = 0; qi < q_used; ++qi) {
                const float* q = &query[static_cast<size_t>(qi) * db];
                // NOTE: cfg below leaves rerank OFF (engine default), so
                // `delivered` measures 4-bit PQ-ordering quality — the
                // quantization ceiling, verified bit-identical to
                // `tree-search --no-rerank` (0.6611 @np256, 10M shape).
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

    std::printf("\n%-8s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s  (ideal rows over %u queries)\n",
                "frac", "random", "centrd", "mbrmin", "oracle",
                no_anchors ? "  --" : "an04",
                no_anchors ? "  --" : "an16",
                no_anchors ? "  --" : "an64",
                no_anchors ? "  --" : "a256",
                "bound", "p1/12", "p1/4", "pca32", "p128", "p048", "p064", "p096", "i8-128", "i8-064", mm_q);
    for (size_t fi = 0; fi < NF; ++fi) {
        std::printf("%-8.2f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n",
                    fracs[fi],
                    hit[0][fi] / q_scored, hit[1][fi] / q_scored,
                    hit[2][fi] / std::min(mm_q, q_scored_mm),
                    hit[3][fi] / q_scored,
                    NA > 0 ? hit[4][fi] / std::min(mm_q, q_scored_mm) : 0.0,
                    NA > 1 ? hit[5][fi] / std::min(mm_q, q_scored_mm) : 0.0,
                    NA > 2 ? hit[6][fi] / std::min(mm_q, q_scored_mm) : 0.0,
                    NA > 3 ? hit[7][fi] / std::min(mm_q, q_scored_mm) : 0.0,
                    hit[8][fi] / q_scored,
                    hit[9][fi] / std::min(mm_q, q_scored_mm),
                    hit[10][fi] / std::min(mm_q, q_scored_mm),
                    hit[11][fi] / std::min(mm_q, q_scored_mm),
                    hit[12][fi] / std::min(mm_q, q_scored_mm),
                    hit[13][fi] / std::min(mm_q, q_scored_mm),
                    hit[14][fi] / std::min(mm_q, q_scored_mm),
                    hit[15][fi] / std::min(mm_q, q_scored_mm),
                    hit[16][fi] / std::min(mm_q, q_scored_mm),
                    hit[17][fi] / std::min(mm_q, q_scored_mm));
    }
    if (adaptive && !ad_frac[0].empty()) {
        std::printf("\n[adaptive] i8-128 score-ordered probing, per-query stop rule\n");
        std::printf("%-12s %10s %12s\n", "rule", "mean frac", "mean cont");
        for (size_t ai = 0; ai < NAL; ++ai) {
            double mf = 0, mc = 0;
            for (double v : ad_frac[ai]) mf += v;
            for (double v : ad_cont[ai]) mc += v;
            const size_t n = ad_frac[ai].size();
            std::printf("alpha=%-6.2f %10.4f %12.4f  (%zu queries)\n",
                        AD_ALPHAS[ai], mf / n, mc / n, n);
        }
        for (size_t fi = 0; fi < NAFF; ++fi) {
            double mf = 0, mc = 0;
            for (double v : ad_frac[NAL + fi]) mf += v;
            for (double v : ad_cont[NAL + fi]) mc += v;
            const size_t n = ad_frac[NAL + fi].size();
            std::printf("fixed=%-6.2f %10.4f %12.4f  (%zu queries)\n",
                        AFF_FR[fi], mf / n, mc / n, n);
        }
    }
    return 0;
}
