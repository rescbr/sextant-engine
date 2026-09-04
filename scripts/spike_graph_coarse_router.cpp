// Spike: graph-coarse router vs flat routing for IVF-list-scan.
//
// Validates the hypothesis that a Vamana graph over K centroids is a faster
// router than flat O(K) centroid distance at large K, without losing routing
// recall — and that the end-to-end graph-coarse + IVF-scan-fine beats flat
// routing at K≥1024.
//
// Pluggable coarse-graph distance (the variable under test):
//   - FP16-exact: simd::l2sq_f16(query, centroid). Production-realistic for
//     the coarse graph (centroids are few; no need to compress).
//   - PQ-approx: train a PQ codebook on the K centroids, search via PQ LUT.
//     Mirrors what VamanaCore-as-is would do. Measures whether PQ-on-centroids
//     hurts routing recall.
//
// For each (K, n_probe, distance mode):
//   - Build the coarse graph over K centroids (Vamana robust_prune + connect).
//   - Route each query three ways: flat, graph-FP16, graph-PQ.
//   - Scan the selected shards, top-W, FP32 rerank, recall@10.
//   - Report: routing recall (vs oracle), routing µs/query, end-to-end QPS,
//     codes scanned/query, MB/query.
//
// arxiv-nomic 1.34M, K ∈ {256, 1024, 4096}.

#include "quant/pq_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/types.hpp"
#include "util/fp16.hpp"

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace sextant;

// --- Minimal .fbin / .gt readers -------------------------------------------

struct FbinData {
    uint32_t n = 0, dim = 0;
    std::vector<float> data;
};
bool read_fbin(const std::string& path, FbinData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&out.n), 4);
    f.read(reinterpret_cast<char*>(&out.dim), 4);
    if (!f.good() || out.n == 0 || out.dim == 0) return false;
    out.data.resize(size_t(out.n) * out.dim);
    f.read(reinterpret_cast<char*>(out.data.data()),
           std::streamsize(out.data.size() * sizeof(float)));
    return f.good() || f.eof();
}

struct GroundTruth { uint32_t n = 0, k = 0; std::vector<uint32_t> ids; };
bool read_gt(const std::string& p, GroundTruth& o) {
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    constexpr uint32_t kGtMagic = 0x4D4D5447u;  // "GTMM" LE
    uint32_t magic = 0; f.read((char*)&magic, 4);
    if (magic != kGtMagic) return false;
    f.read((char*)&o.n, 4); f.read((char*)&o.k, 4);
    char metric = 0; f.read(&metric, 1); (void)metric;
    if (!f || o.n == 0) return false;
    o.ids.resize(size_t(o.n) * o.k);
    for (uint32_t i = 0; i < o.n; ++i) {
        f.read((char*)&o.ids[size_t(i) * o.k], size_t(o.k) * 4);
        f.seekg(size_t(o.k) * 4, std::ios::cur);  // dists
    }
    return !f.fail() || f.eof();
}

// --- Timing ----------------------------------------------------------------

using Clock = std::chrono::steady_clock;
template <typename D> double secs(D d) {
    return std::chrono::duration<double>(d).count();
}

// --- k-means (for centroids; same as the other spikes) ---------------------

struct KmeansResult {
    std::vector<float> centroids;  // K × dim
};
KmeansResult kmeans(const float* data, uint32_t n, uint32_t dim, uint32_t K,
                    uint32_t iters, uint64_t seed) {
    std::mt19937_64 rng(seed);
    // k-means++ init.
    std::vector<float> cents(static_cast<size_t>(K) * dim);
    {
        std::uniform_int_distribution<uint32_t> pick(0, n - 1);
        const uint32_t first = pick(rng);
        std::memcpy(cents.data(), data + size_t(first) * dim,
                    size_t(dim) * sizeof(float));
    }
    std::vector<float> d2(n);
    for (uint32_t k = 1; k < K; k++) {
        const float* prev = cents.data() + size_t(k - 1) * dim;
        for (uint32_t i = 0; i < n; i++) {
            const float* v = data + size_t(i) * dim;
            float dd = 0;
            for (uint32_t d = 0; d < dim; d++) {
                const float x = v[d] - prev[d];
                dd += x * x;
            }
            if (k == 1 || dd < d2[i]) d2[i] = dd;
        }
        double sum = 0;
        for (float v : d2) sum += v;
        std::uniform_real_distribution<double> u(0, sum);
        double target = u(rng);
        double acc = 0;
        uint32_t chosen = n - 1;
        for (uint32_t i = 0; i < n; i++) {
            acc += d2[i];
            if (acc >= target) { chosen = i; break; }
        }
        std::memcpy(cents.data() + size_t(k) * dim,
                    data + size_t(chosen) * dim,
                    size_t(dim) * sizeof(float));
    }
    // Lloyd iterations.
    std::vector<uint32_t> assign(n, 0);
    for (uint32_t it = 0; it < iters; it++) {
        // Assign.
        for (uint32_t i = 0; i < n; i++) {
            const float* v = data + size_t(i) * dim;
            float best = std::numeric_limits<float>::max();
            uint32_t bk = 0;
            for (uint32_t k = 0; k < K; k++) {
                const float* c = cents.data() + size_t(k) * dim;
                float dd = 0;
                for (uint32_t d = 0; d < dim; d++) {
                    const float x = v[d] - c[d];
                    dd += x * x;
                }
                if (dd < best) { best = dd; bk = k; }
            }
            assign[i] = bk;
        }
        // Update.
        std::vector<float> newc(static_cast<size_t>(K) * dim, 0);
        std::vector<uint32_t> counts(K, 0);
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t k = assign[i];
            const float* v = data + size_t(i) * dim;
            for (uint32_t d = 0; d < dim; d++)
                newc[size_t(k) * dim + d] += v[d];
            counts[k]++;
        }
        for (uint32_t k = 0; k < K; k++) {
            if (counts[k] > 0) {
                for (uint32_t d = 0; d < dim; d++)
                    newc[size_t(k) * dim + d] /= counts[k];
            } else {
                // Reseed empty cluster to a random point.
                std::uniform_int_distribution<uint32_t> pick(0, n - 1);
                std::memcpy(newc.data() + size_t(k) * dim,
                            data + size_t(pick(rng)) * dim,
                            size_t(dim) * sizeof(float));
            }
        }
        cents = std::move(newc);
    }
    return {std::move(cents)};
}

// --- 4-bit PQ LUT + FastScan kernel (identical to spike_pq4_recall) ---------

static inline void pq4_block32(const uint8_t* code_block, const uint8_t* lut4,
                               uint32_t m, uint32_t out[32]) {
    uint16x8_t acc_lo_a = vdupq_n_u16(0);
    uint16x8_t acc_lo_b = vdupq_n_u16(0);
    uint16x8_t acc_hi_a = vdupq_n_u16(0);
    uint16x8_t acc_hi_b = vdupq_n_u16(0);
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (uint32_t s = 0; s < m; s++) {
        const uint8x16_t lut_v = vld1q_u8(lut4 + s * 16);
        const uint8x16_t c = vld1q_u8(code_block + s * 16);
        const uint8x16_t clo = vandq_u8(c, mask4);
        const uint8x16_t chi = vshrq_n_u8(c, 4);
        const uint8x16_t rlo = vqtbl1q_u8(lut_v, clo);
        const uint8x16_t rhi = vqtbl1q_u8(lut_v, chi);
        acc_lo_a = vaddq_u16(acc_lo_a, vmovl_u8(vget_low_u8(rlo)));
        acc_lo_b = vaddq_u16(acc_lo_b, vmovl_u8(vget_high_u8(rlo)));
        acc_hi_a = vaddq_u16(acc_hi_a, vmovl_u8(vget_low_u8(rhi)));
        acc_hi_b = vaddq_u16(acc_hi_b, vmovl_u8(vget_high_u8(rhi)));
    }
    vst1q_u32(out +  0, vmovl_u16(vget_low_u16 (acc_lo_a)));
    vst1q_u32(out +  4, vmovl_u16(vget_high_u16(acc_lo_a)));
    vst1q_u32(out +  8, vmovl_u16(vget_low_u16 (acc_lo_b)));
    vst1q_u32(out + 12, vmovl_u16(vget_high_u16(acc_lo_b)));
    vst1q_u32(out + 16, vmovl_u16(vget_low_u16 (acc_hi_a)));
    vst1q_u32(out + 20, vmovl_u16(vget_high_u16(acc_hi_a)));
    vst1q_u32(out + 24, vmovl_u16(vget_low_u16 (acc_hi_b)));
    vst1q_u32(out + 28, vmovl_u16(vget_high_u16(acc_hi_b)));
}

static void build_lut4(const PqQuantizer& q, const float* query, uint8_t* lut4) {
    const uint32_t m = q.m();
    std::vector<float> f32_lut(q.lut_size());
    q.preprocess_query(query, f32_lut.data());
    float max_span = 0.0f;
    std::vector<float> seg_min(m);
    for (uint32_t s = 0; s < m; s++) {
        float* row = f32_lut.data() + s * 16;
        float mn = row[0];
        for (uint32_t c = 1; c < 16; c++) if (row[c] < mn) mn = row[c];
        seg_min[s] = mn;
        for (uint32_t c = 0; c < 16; c++) {
            const float span = row[c] - mn;
            if (span > max_span) max_span = span;
        }
    }
    const float A = (max_span > 0) ? 15.0f / max_span : 0.0f;
    for (uint32_t s = 0; s < m; s++) {
        const float* row = f32_lut.data() + s * 16;
        const float mn = seg_min[s];
        for (uint32_t c = 0; c < 16; c++) {
            const int q4 = (int)((row[c] - mn) * A + 0.5f);
            lut4[s * 16 + c] = (uint8_t)std::max(0, std::min(15, q4));
        }
    }
}

// --- Coarse graph (Vamana-style, standalone) -------------------------------
//
// Nodes are centroids (id 0..K-1). The graph is built with Vamana's
// robust_prune rule (alpha * d(p, pp) <= d(query, pp)) using exact pairwise
// FP16 centroid distances — at K≤4096 we can afford O(K²/2) candidate
// generation per node at build time (16M ops at K=4096, ~seconds).
//
// Routing uses beam search over this graph with a pluggable distance function
// (FP16-exact OR PQ-approx — see main). Same graph structure for both; only
// the search-time distance differs.

struct CoarseGraph {
    // Fixed-degree adjacency (Vamana prunes to R; we store as vector for simplicity).
    std::vector<std::vector<uint32_t>> neighbors;  // K nodes
    std::vector<uint32_t> entry_points;             // typically 1-4 medoids
};

// Distance function: (query, centroid_id) -> float (smaller = nearer).
using CoarseDistFn = std::function<float(const float*, uint32_t)>;

// Beam search over the coarse graph. Returns the top-L explored candidates
// sorted ascending by distance. Used for routing (take top-n_probe).
static void coarse_beam_search(
        const CoarseGraph& g, uint32_t entry, const float* query,
        uint32_t L, const CoarseDistFn& dist_fn,
        std::vector<std::pair<float, uint32_t>>& out) {
    // Simple beam search: frontier (sorted vector) + visited set + working
    // top-L (max-heap by dist → front is the worst). Small L → vector beats heap.
    std::unordered_set<uint32_t> visited;
    visited.reserve(2 * L);
    std::vector<std::pair<float, uint32_t>> frontier;
    frontier.reserve(L + 1);
    auto working_cmp = [](const auto& a, const auto& b) {
        return a.first < b.first;  // max-heap
    };
    std::vector<std::pair<float, uint32_t>> working;
    working.reserve(L + 1);

    const float d0 = dist_fn(query, entry);
    frontier.push_back({d0, entry});
    working.push_back({d0, entry});

    while (!frontier.empty()) {
        // Pop the nearest unvisited frontier node.
        std::pair<float, uint32_t> cur = {std::numeric_limits<float>::max(), 0};
        uint32_t cur_idx = 0;
        for (uint32_t i = 0; i < frontier.size(); i++) {
            if (frontier[i].first < cur.first) {
                cur = frontier[i];
                cur_idx = i;
            }
        }
        frontier.erase(frontier.begin() + cur_idx);
        if (visited.count(cur.second)) continue;
        visited.insert(cur.second);

        for (uint32_t nb : g.neighbors[cur.second]) {
            if (visited.count(nb)) continue;
            const float d = dist_fn(query, nb);
            if (working.size() < L) {
                working.push_back({d, nb});
                std::push_heap(working.begin(), working.end(), working_cmp);
                frontier.push_back({d, nb});
            } else if (d < working.front().first) {
                std::pop_heap(working.begin(), working.end(), working_cmp);
                working.back() = {d, nb};
                std::push_heap(working.begin(), working.end(), working_cmp);
                frontier.push_back({d, nb});
            }
        }
    }
    std::sort(working.begin(), working.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    out = std::move(working);
}

// Build a coarse graph over K centroids using exact FP16 pairwise distances
// and Vamana's robust_prune (alpha=1.2) at R=32. Entry point = centroid 0.
static CoarseGraph build_coarse_graph_fp16(
        uint32_t K, const float16_t* centroids_fp16, Dim dim, uint16_t R,
        float alpha) {
    CoarseGraph g;
    g.neighbors.resize(K);
    g.entry_points.push_back(0);
    #pragma omp parallel for
    for (int i = 0; i < int(K); i++) {
        std::vector<std::pair<float, uint32_t>> all(K);
        const float16_t* ci = centroids_fp16 + size_t(i) * dim;
        for (uint32_t j = 0; j < K; j++) {
            all[j] = {simd::l2sq_f16(ci, centroids_fp16 + size_t(j) * dim, dim), j};
        }
        std::sort(all.begin(), all.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        // Robust prune: take candidates in order, skip occluded ones.
        std::vector<uint32_t> selected;
        for (uint32_t idx = 0; idx < K && selected.size() < R; idx++) {
            if (all[idx].second == uint32_t(i)) continue;
            bool keep = true;
            for (uint32_t s : selected) {
                const float16_t* cs = centroids_fp16 + size_t(s) * dim;
                const float16_t* cpp =
                    centroids_fp16 + size_t(all[idx].second) * dim;
                const float d_p_pp = simd::l2sq_f16(cs, cpp, dim);
                if (alpha * d_p_pp < all[idx].first) { keep = false; break; }
            }
            if (keep) selected.push_back(all[idx].second);
        }
        g.neighbors[i] = std::move(selected);
    }
    // Reciprocal connect: ensure undirected (add i to each neighbor's list).
    for (uint32_t i = 0; i < K; i++) {
        for (uint32_t nb : g.neighbors[i]) {
            if (nb >= K) continue;
            auto& nbn = g.neighbors[nb];
            bool found = false;
            for (uint32_t x : nbn) if (x == i) { found = true; break; }
            if (!found) nbn.push_back(i);
        }
    }
    return g;
}

// --- Flat routing baseline (O(K) centroid distance) ------------------------

 static void flat_route(const float16_t* centroids_fp16, uint32_t K, Dim dim,
                        const float16_t* query_fp16,
                        uint32_t n_probe, std::vector<uint32_t>& out) {
    std::vector<std::pair<float, uint32_t>> cd(K);
    for (uint32_t k = 0; k < K; k++) {
        const float16_t* c = centroids_fp16 + size_t(k) * dim;
        cd[k] = {simd::l2sq_f16(query_fp16, c, dim), k};
    }
    if (n_probe < K) {
        std::partial_sort(cd.begin(), cd.begin() + n_probe, cd.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
    } else {
        std::sort(cd.begin(), cd.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    out.clear();
    for (uint32_t i = 0; i < std::min<uint32_t>(n_probe, K); i++) out.push_back(cd[i].second);
}

// --- Oracle routing (brute-force true nearest centroids) -------------------
// Same as flat but returns ALL K sorted — used to measure routing recall.

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);  // unbuffered so progress is visible
    const std::string base_path = (argc > 1) ? argv[1] : "datasets/arxiv_nomic_base.fbin";
    const std::string query_path = (argc > 2) ? argv[2] : "datasets/arxiv_nomic_query.fbin";
    const std::string gt_path = (argc > 3) ? argv[3] : "datasets/arxiv_nomic_gt.gtmm";

    FbinData db, queries;
    GroundTruth gt;
    if (!read_fbin(base_path, db)) { printf("cannot read %s\n", base_path.c_str()); return 1; }
    if (!read_fbin(query_path, queries)) { printf("cannot read %s\n", query_path.c_str()); return 1; }
    if (!read_gt(gt_path, gt)) { printf("cannot read %s\n", gt_path.c_str()); return 1; }
    printf("Loaded: db %u×%u, queries %u, gt %u×%u\n",
           db.n, db.dim, queries.n, gt.n, gt.k);

    const uint32_t m4 = 192;  // dim/4 for arxiv-nomic (768/4)

    // Train the 4-bit PQ codebook for the LEAF scan (centroids × dim).
    PqQuantizer pq4(MetricKind::L2Sq, db.dim, m4, 4, 42);
    const uint32_t train_n = std::min<uint32_t>(20000, db.n);
    pq4.train(db.data.data(), train_n);

    for (uint32_t K : {256u, 1024u, 4096u}) {
        printf("\n========== K = %u ==========\n", K);

        // k-means on a 50k sample → K centroids.
        const uint32_t sample_n = std::min<uint32_t>(50000, db.n);
        auto km = kmeans(db.data.data(), sample_n, db.dim, K, 8, 42);

        // Cast centroids to FP16 for routing distances.
        std::vector<float16_t> centroids_fp16(static_cast<size_t>(K) * db.dim);
        for (uint32_t k = 0; k < K; k++) {
            cast_fp32_to_fp16(km.centroids.data() + size_t(k) * db.dim,
                              centroids_fp16.data() + size_t(k) * db.dim, db.dim);
        }

        // Assign all N vectors to nearest centroid (parallel).
        std::vector<uint32_t> assign(db.n, 0);
        {
            const uint32_t nthreads = std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> ts;
            for (uint32_t t = 0; t < nthreads; t++) {
                const uint32_t lo = (uint64_t)t * db.n / nthreads;
                const uint32_t hi = (uint64_t)(t + 1) * db.n / nthreads;
                ts.emplace_back([&, lo, hi]() {
                    for (uint32_t i = lo; i < hi; i++) {
                        const float* v = db.data.data() + size_t(i) * db.dim;
                        float best = std::numeric_limits<float>::infinity();
                        uint32_t bk = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            const float* c = km.centroids.data() + size_t(k) * db.dim;
                            float dd = 0;
                            for (uint32_t d = 0; d < db.dim; d++) {
                                const float x = v[d] - c[d];
                                dd += x * x;
                            }
                            if (dd < best) { best = dd; bk = k; }
                        }
                        assign[i] = bk;
                    }
                });
            }
            for (auto& th : ts) th.join();
        }
        std::vector<std::vector<uint32_t>> shard_members(K);
        for (uint32_t i = 0; i < db.n; i++) shard_members[assign[i]].push_back(i);

        // Build per-shard 4-bit FastScan block layout.
        std::vector<std::vector<uint8_t>> shard_blocks(K);
        {
            std::vector<uint8_t> nibbles(static_cast<size_t>(db.n) * m4, 0);
            std::vector<uint8_t> packed(pq4.code_size());
            for (uint32_t i = 0; i < db.n; i++) {
                pq4.encode(db.data.data() + size_t(i) * db.dim, packed.data());
                for (uint32_t s = 0; s < m4; s++) {
                    nibbles[static_cast<size_t>(i) * m4 + s] =
                        (uint8_t)((packed[s / 2] >> ((s % 2) * 4)) & 0xF);
                }
            }
            for (uint32_t k = 0; k < K; k++) {
                const auto& members = shard_members[k];
                const uint32_t sz = static_cast<uint32_t>(members.size());
                const uint32_t n_blocks = (sz + 31) / 32;
                shard_blocks[k].resize(static_cast<size_t>(n_blocks) * m4 * 16, 0);
                for (uint32_t b = 0; b < n_blocks; b++) {
                    for (uint32_t s = 0; s < m4; s++) {
                        for (uint32_t lane = 0; lane < 16; lane++) {
                            const uint32_t v0 = b * 32 + lane;
                            const uint32_t v1 = b * 32 + 16 + lane;
                            const uint8_t lo = (v0 < sz)
                                ? nibbles[static_cast<size_t>(members[v0]) * m4 + s] : 0;
                            const uint8_t hi = (v1 < sz)
                                ? nibbles[static_cast<size_t>(members[v1]) * m4 + s] : 0;
                            shard_blocks[k][((size_t)b * m4 + s) * 16 + lane] =
                                (uint8_t)((hi << 4) | lo);
                        }
                    }
                }
            }
        }

        // PQ-on-centroids: train a PQ codebook on the K centroids, encode them,
        // build a LUT per query. This is the "PQ-approx" coarse distance.
        // sub_dim must be reasonable; use m_coarse such that dim/m_coarse = 4-8.
        const uint16_t m_coarse = std::max<uint16_t>(8u, static_cast<uint16_t>(db.dim / 8));
        PqQuantizer pq_coarse(MetricKind::L2Sq, db.dim, m_coarse, 8, 123);
        pq_coarse.train(km.centroids.data(), K);
        std::vector<uint8_t> centroid_codes(static_cast<size_t>(K) * pq_coarse.code_size());
        for (uint32_t k = 0; k < K; k++) {
            pq_coarse.encode(km.centroids.data() + size_t(k) * db.dim,
                             centroid_codes.data() + size_t(k) * pq_coarse.code_size());
        }
        std::vector<float> coarse_lut(pq_coarse.lut_size());

        // Distance closures.
        // FP16-exact search distance: cast the float* query to FP16 on the fly
        // and compute exact L2sq vs the centroid. (The cast is per-call; for
        // the spike's small query count this is fine. Production would cast
        // once per query and reuse.)
        CoarseDistFn dist_fp16 = [&](const float* query, uint32_t cid) -> float {
            float16_t qbuf[2048];
            cast_fp32_to_fp16(query, qbuf, db.dim);
            return simd::l2sq_f16(qbuf,
                                  centroids_fp16.data() + size_t(cid) * db.dim,
                                  db.dim);
        };
        // PQ-approx search distance: caller must call pq_coarse.preprocess_query
        // into coarse_lut before invoking; this gathers the per-centroid code.
        CoarseDistFn dist_pq = [&](const float* query, uint32_t cid) -> float {
            (void)query;  // LUT already built by caller.
            return pq_coarse.lut_distance(
                centroid_codes.data() + size_t(cid) * pq_coarse.code_size(),
                coarse_lut.data());
        };
        // Build the coarse graph (FP16, robust_prune at R=32, alpha=1.2).
        auto t0 = Clock::now();
        CoarseGraph g_fp16 = build_coarse_graph_fp16(
            K, centroids_fp16.data(), db.dim, /*R=*/32, /*alpha=*/1.2f);
        auto t1 = Clock::now();
        printf("# Built FP16 coarse graph (R=32) in %.2fs\n", secs(t1 - t0));

        // For each query: route three ways, measure routing recall + time.
        // Routing recall = fraction of oracle top-n_probe centroids found.
        printf("\n%-10s %-10s %-12s %-12s %-12s %-12s %-12s %-12s\n",
               "router", "n_probe", "rt_recall", "rt_us/q", "e2e_recall", "QPS",
               "codes/qry", "MB/qry");
        printf("----------------------------------------------------------------------------------------\n");

        // Pre-cast queries to FP16.
        std::vector<float16_t> queries_fp16(static_cast<size_t>(queries.n) * db.dim);
        for (uint32_t i = 0; i < queries.n; i++) {
            cast_fp32_to_fp16(queries.data.data() + size_t(i) * db.dim,
                              queries_fp16.data() + size_t(i) * db.dim, db.dim);
        }

        // Pre-compute oracle nearest centroids per query (for routing recall).
        std::vector<std::vector<uint32_t>> oracle(queries.n);
        #pragma omp parallel for
        for (int qi = 0; qi < int(queries.n); qi++) {
            std::vector<std::pair<float, uint32_t>> cd(K);
            for (uint32_t k = 0; k < K; k++) {
                cd[k] = {simd::l2sq_f16(queries_fp16.data() + size_t(qi) * db.dim,
                          centroids_fp16.data() + size_t(k) * db.dim, db.dim), k};
            }
            std::sort(cd.begin(), cd.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            oracle[qi].reserve(K);
            for (auto& [d, id] : cd) oracle[qi].push_back(id);
        }

        // Scan helper: given a set of shard ids, scan each, collect top-W by
        // 4-bit distance, return (dist, global_id) pairs.
        auto scan_shards = [&](const std::vector<uint32_t>& shard_ids,
                                const uint8_t* lut4,
                                std::vector<std::pair<uint32_t, uint32_t>>& ranked) {
            ranked.clear();
            for (uint32_t sid : shard_ids) {
                if (sid >= K) continue;
                const auto& blocks = shard_blocks[sid];
                const auto& members = shard_members[sid];
                const uint32_t sz = static_cast<uint32_t>(members.size());
                const uint32_t nb = (sz + 31) / 32;
                for (uint32_t b = 0; b < nb; b++) {
                    uint32_t out[32];
                    pq4_block32(blocks.data() + (size_t)b * m4 * 16, lut4, m4, out);
                    for (uint32_t j = 0; j < 32; j++) {
                        const uint32_t local = b * 32 + j;
                        if (local >= sz) continue;
                        ranked.push_back({out[j], members[local]});
                    }
                }
            }
        };

        for (uint32_t n_probe : {8u, 16u, 32u}) {
            if (n_probe > K) continue;
            const uint32_t W = 300;
            std::vector<uint8_t> lut4(static_cast<size_t>(m4) * 16);

            // --- Flat routing ---
            {
                uint64_t rt_hits = 0, rt_total = 0;
                uint64_t e2e_hits = 0, e2e_total = 0;
                uint64_t codes_scanned = 0;
                double rt_time = 0, e2e_time = 0;
                std::vector<uint32_t> shards;
                std::vector<std::pair<uint32_t, uint32_t>> ranked;
                std::vector<std::pair<float, uint32_t>> exact;
                auto e2e_t0 = Clock::now();
                for (uint32_t qi = 0; qi < queries.n; qi++) {
                    const float* q = queries.data.data() + size_t(qi) * db.dim;
                    const float16_t* qfp16 =
                        queries_fp16.data() + size_t(qi) * db.dim;
                    auto rt0 = Clock::now();
                    flat_route(centroids_fp16.data(),
                               K, db.dim, qfp16, n_probe, shards);
                    auto rt1 = Clock::now();
                    rt_time += secs(rt1 - rt0);
                    // Routing recall vs oracle.
                    for (uint32_t i = 0; i < n_probe; i++) {
                        for (uint32_t j = 0; j < n_probe; j++) {
                            if (shards[i] == oracle[qi][j]) { rt_hits++; break; }
                        }
                    }
                    rt_total += n_probe;
                    // Scan.
                    build_lut4(pq4, q, lut4.data());
                    scan_shards(shards, lut4.data(), ranked);
                    codes_scanned += ranked.size();
                    // top-W + rerank.
                    if (ranked.size() > W) {
                        std::nth_element(ranked.begin(), ranked.begin() + W,
                                         ranked.end(),
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
                        ranked.resize(W);
                    }
                    std::sort(ranked.begin(), ranked.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    exact.clear();
                    for (const auto& [d, gid] : ranked) {
                        const float* v = db.data.data() + size_t(gid) * db.dim;
                        float dd = 0;
                        for (uint32_t k = 0; k < db.dim; k++) {
                            const float x = v[k] - q[k];
                            dd += x * x;
                        }
                        exact.push_back({dd, gid});
                    }
                    std::sort(exact.begin(), exact.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    const uint32_t* gt_row = gt.ids.data() + size_t(qi) * gt.k;
                    for (uint32_t r = 0; r < std::min<uint32_t>(10, exact.size()); r++) {
                        for (uint32_t g = 0; g < 10; g++) {
                            if (gt_row[g] == exact[r].second) { e2e_hits++; break; }
                        }
                    }
                    e2e_total += 10;
                }
                auto e2e_t1 = Clock::now();
                e2e_time = secs(e2e_t1 - e2e_t0);
                printf("%-10s %-10u %-12.4f %-12.2f %-12.4f %-12.1f %-12.0f %-12.1f\n",
                       "flat", n_probe,
                       double(rt_hits) / rt_total,
                       rt_time * 1e6 / queries.n,
                       double(e2e_hits) / e2e_total,
                       queries.n / e2e_time,
                       double(codes_scanned) / queries.n,
                       double(codes_scanned) * (m4 / 2) / queries.n / 1e6);
            }

            // --- Graph routing (FP16-exact) ---
            {
                uint64_t rt_hits = 0, rt_total = 0;
                uint64_t e2e_hits = 0, e2e_total = 0;
                uint64_t codes_scanned = 0;
                double rt_time = 0, e2e_time = 0;
                std::vector<std::pair<float, uint32_t>> search_out;
                std::vector<std::pair<uint32_t, uint32_t>> ranked;
                std::vector<std::pair<float, uint32_t>> exact;
                auto e2e_t0 = Clock::now();
                for (uint32_t qi = 0; qi < queries.n; qi++) {
                    const float* q = queries.data.data() + size_t(qi) * db.dim;
                    auto rt0 = Clock::now();
                    coarse_beam_search(g_fp16, g_fp16.entry_points[0], q,
                                       std::max(n_probe * 4u, 64u), dist_fp16,
                                       search_out);
                    auto rt1 = Clock::now();
                    rt_time += secs(rt1 - rt0);
                    std::vector<uint32_t> shards;
                    for (uint32_t i = 0; i < std::min<uint32_t>(n_probe, search_out.size()); i++) {
                        shards.push_back(search_out[i].second);
                    }
                    for (uint32_t i = 0; i < shards.size(); i++) {
                        for (uint32_t j = 0; j < n_probe; j++) {
                            if (shards[i] == oracle[qi][j]) { rt_hits++; break; }
                        }
                    }
                    rt_total += shards.size();
                    build_lut4(pq4, q, lut4.data());
                    scan_shards(shards, lut4.data(), ranked);
                    codes_scanned += ranked.size();
                    if (ranked.size() > W) {
                        std::nth_element(ranked.begin(), ranked.begin() + W,
                                         ranked.end(),
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
                        ranked.resize(W);
                    }
                    std::sort(ranked.begin(), ranked.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    exact.clear();
                    for (const auto& [d, gid] : ranked) {
                        const float* v = db.data.data() + size_t(gid) * db.dim;
                        float dd = 0;
                        for (uint32_t k = 0; k < db.dim; k++) {
                            const float x = v[k] - q[k];
                            dd += x * x;
                        }
                        exact.push_back({dd, gid});
                    }
                    std::sort(exact.begin(), exact.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    const uint32_t* gt_row = gt.ids.data() + size_t(qi) * gt.k;
                    for (uint32_t r = 0; r < std::min<uint32_t>(10, exact.size()); r++) {
                        for (uint32_t g = 0; g < 10; g++) {
                            if (gt_row[g] == exact[r].second) { e2e_hits++; break; }
                        }
                    }
                    e2e_total += 10;
                }
                auto e2e_t1 = Clock::now();
                e2e_time = secs(e2e_t1 - e2e_t0);
                printf("%-10s %-10u %-12.4f %-12.2f %-12.4f %-12.1f %-12.0f %-12.1f\n",
                       "graph-fp16", n_probe,
                       double(rt_hits) / rt_total,
                       rt_time * 1e6 / queries.n,
                       double(e2e_hits) / e2e_total,
                       queries.n / e2e_time,
                       double(codes_scanned) / queries.n,
                       double(codes_scanned) * (m4 / 2) / queries.n / 1e6);
            }

            // --- Graph routing (PQ-approx) ---
            // Same graph structure (FP16-built), but search with PQ-on-centroids
            // distance. Measures whether PQ-on-centroids degrades routing recall.
            {
                uint64_t rt_hits = 0, rt_total = 0;
                uint64_t e2e_hits = 0, e2e_total = 0;
                uint64_t codes_scanned = 0;
                double rt_time = 0, e2e_time = 0;
                std::vector<std::pair<float, uint32_t>> search_out;
                std::vector<std::pair<uint32_t, uint32_t>> ranked;
                std::vector<std::pair<float, uint32_t>> exact;
                auto e2e_t0 = Clock::now();
                for (uint32_t qi = 0; qi < queries.n; qi++) {
                    const float* q = queries.data.data() + size_t(qi) * db.dim;
                    auto rt0 = Clock::now();
                    pq_coarse.preprocess_query(q, coarse_lut.data());
                    coarse_beam_search(g_fp16, g_fp16.entry_points[0], q,
                                       std::max(n_probe * 4u, 64u), dist_pq,
                                       search_out);
                    auto rt1 = Clock::now();
                    rt_time += secs(rt1 - rt0);
                    std::vector<uint32_t> shards;
                    for (uint32_t i = 0; i < std::min<uint32_t>(n_probe, search_out.size()); i++) {
                        shards.push_back(search_out[i].second);
                    }
                    for (uint32_t i = 0; i < shards.size(); i++) {
                        for (uint32_t j = 0; j < n_probe; j++) {
                            if (shards[i] == oracle[qi][j]) { rt_hits++; break; }
                        }
                    }
                    rt_total += shards.size();
                    build_lut4(pq4, q, lut4.data());
                    scan_shards(shards, lut4.data(), ranked);
                    codes_scanned += ranked.size();
                    if (ranked.size() > W) {
                        std::nth_element(ranked.begin(), ranked.begin() + W,
                                         ranked.end(),
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
                        ranked.resize(W);
                    }
                    std::sort(ranked.begin(), ranked.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    exact.clear();
                    for (const auto& [d, gid] : ranked) {
                        const float* v = db.data.data() + size_t(gid) * db.dim;
                        float dd = 0;
                        for (uint32_t k = 0; k < db.dim; k++) {
                            const float x = v[k] - q[k];
                            dd += x * x;
                        }
                        exact.push_back({dd, gid});
                    }
                    std::sort(exact.begin(), exact.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    const uint32_t* gt_row = gt.ids.data() + size_t(qi) * gt.k;
                    for (uint32_t r = 0; r < std::min<uint32_t>(10, exact.size()); r++) {
                        for (uint32_t g = 0; g < 10; g++) {
                            if (gt_row[g] == exact[r].second) { e2e_hits++; break; }
                        }
                    }
                    e2e_total += 10;
                }
                auto e2e_t1 = Clock::now();
                e2e_time = secs(e2e_t1 - e2e_t0);
                printf("%-10s %-10u %-12.4f %-12.2f %-12.4f %-12.1f %-12.0f %-12.1f\n",
                       "graph-pq", n_probe,
                       double(rt_hits) / rt_total,
                       rt_time * 1e6 / queries.n,
                       double(e2e_hits) / e2e_total,
                       queries.n / e2e_time,
                       double(codes_scanned) / queries.n,
                       double(codes_scanned) * (m4 / 2) / queries.n / 1e6);
            }
        }
    }
    return 0;
}
