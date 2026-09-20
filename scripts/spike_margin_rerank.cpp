// Hybrid spike: IVF routing + 4-bit FastScan per shard + FP32 rerank.
//
// Question: at arxiv-nomic 1.34M scale, what recall-QPS Pareto does the
// "route to n_probe shards, 4-bit-FastScan each shard, rerank" pattern give?
// How does it compare to (a) brute-force global scan (no routing), and
// (b) the current production graph-inside-IVF (archived measurement).
//
// What this spike IS: an algorithm-level Pareto measurement. We do our own
// simple k-means IVF (k-means++ init, 10 iterations) on the dataset, encode
// each shard's codes at 4-bit PQ (m=192), scan with the FastScan kernel,
// rerank with exact FP32 L2.
//
// What this spike is NOT: production-quality. K-means is single-threaded,
// encoding is on-the-fly per-shard (not persisted), no paging. The numbers
// are an algorithmic upper bound — the production paged version can only be
// slower (paging adds IO). But the *recall* and the *relative QPS across
// (K, n_probe, W)* are honest.
//
// Build: ninja -C build spike_hybrid_ivf

#include "quant/pq_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace sextant;

// --- .fbin / .gt readers (same as spike_pq4_recall).

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

// --- 4-bit FastScan kernel (32 codes/block, identical to spike_pq4_recall).
static inline void pq4_block32(const uint8_t* code_block, const uint8_t* lut4,
                               uint32_t m, uint32_t out[32]) {
    uint16x8_t acc_lo_a = vdupq_n_u16(0), acc_lo_b = vdupq_n_u16(0);
    uint16x8_t acc_hi_a = vdupq_n_u16(0), acc_hi_b = vdupq_n_u16(0);
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (uint32_t s = 0; s < m; s++) {
        const uint8x16_t lut_v = vld1q_u8(lut4 + s * 16);
        const uint8x16_t c = vld1q_u8(code_block + s * 16);
        const uint8x16_t rlo = vqtbl1q_u8(lut_v, vandq_u8(c, mask4));
        const uint8x16_t rhi = vqtbl1q_u8(lut_v, vshrq_n_u8(c, 4));
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

// --- Simple k-means for IVF routing centroids (single-threaded, good enough).
// Returns K centroids and the assignment of each vector to a centroid.
struct KmeansResult {
    std::vector<float> centroids;  // K * dim
    std::vector<uint32_t> assign;  // N
};
KmeansResult kmeans(const float* data, uint32_t n, uint32_t dim, uint32_t K,
                    uint32_t iters = 10, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    // k-means++ init.
    std::vector<float> cents(K * dim);
    {
        // Pick first centroid at random.
        const uint32_t first = rng() % n;
        std::memcpy(cents.data(), data + size_t(first)*dim, dim*4);
        std::vector<float> d2(n, std::numeric_limits<float>::infinity());
        for (uint32_t k = 1; k < K; k++) {
            // Update d2 from newly-added centroid k-1.
            const float* c = cents.data() + (k-1)*dim;
            for (uint32_t i = 0; i < n; i++) {
                const float* v = data + size_t(i)*dim;
                float d = 0; for (uint32_t d2i=0; d2i<dim; d2i++){float x=v[d2i]-c[d2i];d+=x*x;}
                if (d < d2[i]) d2[i] = d;
            }
            // Sample next centroid proportional to d2.
            float sum = std::accumulate(d2.begin(), d2.end(), 0.0f);
            float r = (rng() / float(rng.max())) * sum;
            float acc = 0; uint32_t pick = n - 1;
            for (uint32_t i = 0; i < n; i++) {
                acc += d2[i];
                if (acc >= r) { pick = i; break; }
            }
            std::memcpy(cents.data() + k*dim, data + size_t(pick)*dim, dim*4);
        }
    }
    // Lloyd iterations.
    std::vector<uint32_t> assign(n, 0);
    std::vector<float> new_cents(K * dim, 0);
    std::vector<uint32_t> counts(K, 0);
    for (uint32_t it = 0; it < iters; it++) {
        // Assign.
        for (uint32_t i = 0; i < n; i++) {
            const float* v = data + size_t(i)*dim;
            float best = std::numeric_limits<float>::infinity();
            uint32_t bk = 0;
            for (uint32_t k = 0; k < K; k++) {
                const float* c = cents.data() + k*dim;
                float d = 0; for (uint32_t di=0; di<dim; di++){float x=v[di]-c[di];d+=x*x;}
                if (d < best) { best = d; bk = k; }
            }
            assign[i] = bk;
        }
        // Update.
        std::fill(new_cents.begin(), new_cents.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (uint32_t i = 0; i < n; i++) {
            const float* v = data + size_t(i)*dim;
            float* nc = new_cents.data() + assign[i]*dim;
            for (uint32_t di = 0; di < dim; di++) nc[di] += v[di];
            counts[assign[i]]++;
        }
        for (uint32_t k = 0; k < K; k++) {
            if (counts[k] > 0) {
                for (uint32_t di = 0; di < dim; di++)
                    cents[k*dim + di] = new_cents[k*dim + di] / counts[k];
            }
        }
    }
    return {std::move(cents), std::move(assign)};
}

int main(int argc, char** argv) {
    std::string base_path  = "datasets/arxiv_nomic_base.fbin";
    std::string query_path = "datasets/arxiv_nomic_query.fbin";
    std::string gt_path    = "datasets/arxiv_nomic_gt.gtmm";
    bool apply_jlt = false;
    uint64_t jlt_seed = 12345;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--jlt") apply_jlt = true;
        else if (s == "--jlt-seed" && i + 1 < argc) jlt_seed = strtoull(argv[++i], nullptr, 10);
        else if (base_path == "datasets/arxiv_nomic_base.fbin" && i == 1) base_path = s;
        else if (query_path == "datasets/arxiv_nomic_query.fbin" && i == 2) query_path = s;
        else if (gt_path == "datasets/arxiv_nomic_gt.gtmm" && i == 3) gt_path = s;
    }

    FbinData db, queries;
    GroundTruth gt;
    if (!read_fbin(base_path, db)) { printf("cannot read %s\n", base_path.c_str()); return 1; }
    if (!read_fbin(query_path, queries)) { printf("cannot read %s\n", query_path.c_str()); return 1; }
    if (!read_gt(gt_path, gt)) { printf("cannot read %s\n", gt_path.c_str()); return 1; }
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered so background runs show progress

    // Optional JLT (random orthogonal rotation). L2 distance is rotation-
    // invariant, so applying R to both db and queries leaves rerank distances
    // unchanged but mixes dimensions across PQ segments. Tests whether random
    // rotation gives PQ enough uniformity for margin-based rerank (which raw
    // PQ lacks — see spike_margin_rerank E1 result).
    //
    // NOTE: this is NOT OPQ. OPQ uses PCA-based rotation to balance variance
    // across segments. JLT uses random rotation to make the ERROR distribution
    // uniform across codes (the property RaBitQ exploits). The two have
    // different goals — OPQ minimizes reconstruction error; JLT uniformizes
    // error distribution. We previously tested OPQ (0pp on arxiv-nomic);
    // this is the first JLT-then-PQ test.
    if (apply_jlt) {
        printf("Applying JLT (random orthogonal rotation, seed=%llu)...\n",
               (unsigned long long)jlt_seed);
        const uint32_t D = db.dim;
        // Generate random orthogonal matrix via Gram-Schmidt on random Gaussians.
        std::mt19937_64 rng(jlt_seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> R(D * D);
        for (auto& v : R) v = gauss(rng);
        // Modified Gram-Schmidt.
        for (uint32_t i = 0; i < D; i++) {
            for (uint32_t j = 0; j < i; j++) {
                float dot = 0;
                for (uint32_t k = 0; k < D; k++) dot += R[i*D + k] * R[j*D + k];
                for (uint32_t k = 0; k < D; k++) R[i*D + k] -= dot * R[j*D + k];
            }
            float nrm = 0;
            for (uint32_t k = 0; k < D; k++) nrm += R[i*D + k] * R[i*D + k];
            nrm = std::sqrt(nrm);
            for (uint32_t k = 0; k < D; k++) R[i*D + k] /= nrm;
        }
        // Apply R to db and queries in place. R is row-major; result[i] = sum_j R[i,j] * v[j].
        auto rotate = [&](FbinData& fd) {
            std::vector<float> tmp(fd.dim);
            for (uint32_t v = 0; v < fd.n; v++) {
                const float* src = fd.data.data() + size_t(v) * fd.dim;
                for (uint32_t i = 0; i < fd.dim; i++) {
                    float s = 0;
                    for (uint32_t j = 0; j < fd.dim; j++) s += R[i*fd.dim + j] * src[j];
                    tmp[i] = s;
                }
                std::memcpy(fd.data.data() + size_t(v) * fd.dim, tmp.data(),
                            fd.dim * sizeof(float));
            }
        };
        auto t0 = std::chrono::steady_clock::now();
        rotate(db);
        rotate(queries);
        auto t1 = std::chrono::steady_clock::now();
        printf("  JLT applied in %.1fs (db=%u, queries=%u)\n",
               std::chrono::duration<double>(t1-t0).count(), db.n, queries.n);
    }
    printf("Loaded: db %u×%u (%.0f MB), queries %u, gt %u×%u\n",
           db.n, db.dim, db.data.size()*4/1e6, queries.n, gt.n, gt.k);

    // --- Step 1: IVF k-means on a sample.
    const uint32_t K = 64;  // IVF shard count
    printf("\n[K=%u] Running k-means on 50k sample for IVF centroids...\n", K);
    const uint32_t sample_n = std::min<uint32_t>(50000, db.n);
    auto t0 = std::chrono::steady_clock::now();
    auto km = kmeans(db.data.data(), sample_n, db.dim, K, 10, 42);
    auto t1 = std::chrono::steady_clock::now();
    printf("  k-means: %.1fs\n", std::chrono::duration<double>(t1-t0).count());

    // --- Step 2: assign ALL db vectors to shards (full pass).
    printf("Assigning all %u vectors to shards...\n", db.n);
    t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<uint32_t>> shard_members(K);
    {
        // Reuse the assignment from k-means if sample_n == db.n; else reassign.
        std::vector<uint32_t> full_assign(db.n, 0);
        for (uint32_t i = 0; i < db.n; i++) {
            const float* v = db.data.data() + size_t(i)*db.dim;
            float best = std::numeric_limits<float>::infinity();
            uint32_t bk = 0;
            for (uint32_t k = 0; k < K; k++) {
                const float* c = km.centroids.data() + k*db.dim;
                float d = 0; for (uint32_t di=0; di<db.dim; di++){float x=v[di]-c[di];d+=x*x;}
                if (d < best) { best = d; bk = k; }
            }
            full_assign[i] = bk;
        }
        for (uint32_t i = 0; i < db.n; i++) shard_members[full_assign[i]].push_back(i);
        t1 = std::chrono::steady_clock::now();
    }
    uint32_t max_shard = 0, min_shard = UINT32_MAX;
    for (auto& s : shard_members) {
        max_shard = std::max(max_shard, (uint32_t)s.size());
        min_shard = std::min(min_shard, (uint32_t)s.size());
    }
    printf("  assign: %.1fs, shard sizes min=%u max=%u avg=%u\n",
           std::chrono::duration<double>(t1-t0).count(),
           min_shard, max_shard, db.n / K);

    // --- Step 3: train a 4-bit PQ codebook on a sample, encode every vector.
    const uint32_t m4 = 192;  // 4-bit PQ segment count
    printf("Training 4-bit PQ (m=%u) on 20k sample...\n", m4);
    PqQuantizer pq4(MetricKind::L2Sq, db.dim, m4, 4, 42);
    t0 = std::chrono::steady_clock::now();
    pq4.train(db.data.data(), std::min<uint32_t>(20000, db.n));
    t1 = std::chrono::steady_clock::now();
    printf("  PQ train: %.1fs\n", std::chrono::duration<double>(t1-t0).count());

    printf("Encoding all %u vectors at 4-bit (flat layout for rerank lookup)...\n", db.n);
    t0 = std::chrono::steady_clock::now();
    // Flat [N][m] nibble array (one byte per segment for easy indexing).
    std::vector<uint8_t> nibbles((size_t)db.n * m4, 0);
    {
        std::vector<uint8_t> packed(pq4.code_size());
        for (uint32_t i = 0; i < db.n; i++) {
            pq4.encode(db.data.data() + size_t(i)*db.dim, packed.data());
            for (uint32_t s = 0; s < m4; s++)
                nibbles[(size_t)i * m4 + s] = (uint8_t)((packed[s/2] >> ((s%2)*4)) & 0xF);
        }
    }
    t1 = std::chrono::steady_clock::now();
    printf("  encode: %.1fs (%.0f MB)\n",
           std::chrono::duration<double>(t1-t0).count(),
           nibbles.size() / 1e6);

    // --- Step 4: for each shard, precompute the segment-major block layout.
    // Each shard's codes are packed into 32-code blocks for the FastScan kernel.
    printf("Precomputing per-shard block layout...\n");
    std::vector<std::vector<uint8_t>> shard_blocks(K);
    std::vector<std::vector<uint32_t>> shard_block_ids(K);  // vector id per lane
    t0 = std::chrono::steady_clock::now();
    for (uint32_t k = 0; k < K; k++) {
        const auto& members = shard_members[k];
        const uint32_t sz = (uint32_t)members.size();
        const uint32_t n_blocks = (sz + 31) / 32;
        shard_blocks[k].resize((size_t)n_blocks * m4 * 16, 0);
        shard_block_ids[k].resize((size_t)n_blocks * 32, 0);
        for (uint32_t b = 0; b < n_blocks; b++) {
            for (uint32_t s = 0; s < m4; s++) {
                for (uint32_t lane = 0; lane < 16; lane++) {
                    const uint32_t v0_idx = b * 32 + lane;
                    const uint32_t v1_idx = b * 32 + 16 + lane;
                    uint8_t lo = 0, hi = 0;
                    if (v0_idx < sz) {
                        const uint32_t gid = members[v0_idx];
                        lo = nibbles[(size_t)gid * m4 + s];
                        shard_block_ids[k][b*32 + lane] = gid;
                    }
                    if (v1_idx < sz) {
                        const uint32_t gid = members[v1_idx];
                        hi = nibbles[(size_t)gid * m4 + s];
                        shard_block_ids[k][b*32 + 16 + lane] = gid;
                    }
                    shard_blocks[k][((size_t)b * m4 + s) * 16 + lane] = (uint8_t)((hi << 4) | lo);
                }
            }
        }
    }
    t1 = std::chrono::steady_clock::now();
    printf("  block layout: %.1fs\n", std::chrono::duration<double>(t1-t0).count());

    // --- Step 5: query helper — build 4-bit LUT.
    auto build_lut4 = [&](const float* query, uint8_t* lut4, float* A_out) {
        std::vector<float> f32_lut(pq4.lut_size());
        pq4.preprocess_query(query, f32_lut.data());
        float max_span = 0;
        std::vector<float> seg_min(m4);
        for (uint32_t s = 0; s < m4; s++) {
            float* row = f32_lut.data() + s*16;
            float mn = row[0];
            for (uint32_t c = 1; c < 16; c++) if (row[c] < mn) mn = row[c];
            seg_min[s] = mn;
            for (uint32_t c = 0; c < 16; c++) {
                float span = row[c] - mn;
                if (span > max_span) max_span = span;
            }
        }
        float A = (max_span > 0) ? 15.0f / max_span : 0;
        *A_out = A;
        for (uint32_t s = 0; s < m4; s++) {
            const float* row = f32_lut.data() + s*16;
            float mn = seg_min[s];
            for (uint32_t c = 0; c < 16; c++) {
                int q4 = (int)((row[c] - mn) * A + 0.5f);
                lut4[s*16 + c] = (uint8_t)std::max(0, std::min(15, q4));
            }
        }
    };

    // --- Step 6: Compare rerank strategies at matched average-rerank budget.
    //
    // Three strategies:
    //   (a) TOP-W: rerank fixed top-W by PQ estimate (today's behavior).
    //   (b) MARGIN: rerank any candidate with pq_dist <= pq_best + margin
    //       (RaBitQ-style selective rerank, but with empirical margin
    //       instead of theoretical bound).
    //   (c) TOP-W ∪ MARGIN: rerank top-W AND any within margin (covers
    //       both gaps — true NN that ranks outside W but within margin).
    //
    // Compare at MATCHED average rerank count per query (the real cost
    // driver). If MARGIN beats TOP-W at matched avg-rerank → the
    // selection rule itself helps PQ.
    const uint32_t n_query = std::min<uint32_t>(200, queries.n);
    printf("\n=== Rerank strategy comparison (K=%u, m4=%u, %u queries, JLT=%s) ===\n",
           K, m4, n_query, apply_jlt ? "ON" : "OFF");

    // First, collect per-query (pq_dist, global_id) arrays for each (n_probe)
    // — we'll reuse them across strategies to avoid re-scanning.
    printf("Collecting PQ scan results per n_probe...\n");
    std::map<uint32_t, std::vector<std::vector<std::pair<float, uint32_t>>>> per_query_ranked;
    for (uint32_t n_probe : {1u, 2u, 4u, 8u}) {
        auto& vecs = per_query_ranked[n_probe];
        vecs.resize(n_query);
        for (uint32_t qi = 0; qi < n_query; qi++) {
            const float* query = queries.data.data() + size_t(qi)*db.dim;
            std::vector<std::pair<float, uint32_t>> cdists(K);
            for (uint32_t k = 0; k < K; k++) {
                const float* c = km.centroids.data() + k*db.dim;
                float d = 0; for (uint32_t di=0; di<db.dim; di++){float x=query[di]-c[di];d+=x*x;}
                cdists[k] = {d, k};
            }
            std::partial_sort(cdists.begin(), cdists.begin()+n_probe, cdists.end());
            std::vector<uint8_t> lut4((size_t)m4 * 16);
            float A;
            build_lut4(query, lut4.data(), &A);
            auto& ranked = vecs[qi];
            for (uint32_t p = 0; p < n_probe; p++) {
                const uint32_t k = cdists[p].second;
                const auto& blocks = shard_blocks[k];
                const auto& ids = shard_block_ids[k];
                const uint32_t sz = (uint32_t)shard_members[k].size();
                const uint32_t n_blocks = (sz + 31) / 32;
                for (uint32_t b = 0; b < n_blocks; b++) {
                    uint32_t out[32];
                    pq4_block32(blocks.data() + (size_t)b * m4 * 16,
                                lut4.data(), m4, out);
                    for (uint32_t j = 0; j < 32; j++) {
                        const uint32_t lane = b * 32 + j;
                        if (lane < sz) ranked.push_back({(float)out[j], ids[lane]});
                    }
                }
            }
            // Sort ascending by pq_dist once.
            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& a, const auto& b){return a.first<b.first;});
        }
    }

    // Rerank helper: given a query and a list of gids to rerank, return
    // recall@10 hits and count.
    auto rerank_and_score = [&](uint32_t qi,
                                 const std::vector<uint32_t>& gids_to_rerank,
                                 uint32_t& hits_out) {
        const float* query = queries.data.data() + size_t(qi)*db.dim;
        std::vector<std::pair<float, uint32_t>> exact;
        exact.reserve(gids_to_rerank.size());
        for (uint32_t gid : gids_to_rerank) {
            const float* v = db.data.data() + size_t(gid)*db.dim;
            float d = 0;
            for (uint32_t di=0; di<db.dim; di++){float x=v[di]-query[di];d+=x*x;}
            exact.push_back({d, gid});
        }
        std::partial_sort(exact.begin(),
                          exact.begin() + std::min<size_t>(10, exact.size()),
                          exact.end());
        const uint32_t* gt_row = gt.ids.data() + size_t(qi)*gt.k;
        uint32_t hits = 0;
        for (size_t r = 0; r < std::min<size_t>(10, exact.size()); r++) {
            for (uint32_t g = 0; g < 10; g++)
                if (gt_row[g] == exact[r].second) { hits++; break; }
        }
        hits_out = hits;
        return exact.size();  // rerank count
    };

    // Strategy evaluators. Each takes (qi, ranked_for_query) → returns
    // (rerank_count, hits). We compute average rerank count and recall.
    auto run_strategy = [&](const char* name, uint32_t n_probe, auto select_gids) {
        // select_gids: lambda(uint32_t qi, const auto& ranked) -> std::vector<uint32_t>
        uint64_t total_hits = 0;
        uint64_t total_rerank = 0;
        auto t0 = std::chrono::steady_clock::now();
        const auto& vecs = per_query_ranked[n_probe];
        for (uint32_t qi = 0; qi < n_query; qi++) {
            auto gids = select_gids(qi, vecs[qi]);
            uint32_t hits;
            total_rerank += rerank_and_score(qi, gids, hits);
            total_hits += hits;
        }
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const float recall = (float)total_hits / (float(n_query) * 10.0f);
        const float avg_rerank = (float)total_rerank / n_query;
        const float qps = n_query / secs;
        printf("  %-30s np=%u  recall@10=%.4f  avg-rerank=%.1f  QPS=%.1f\n",
               name, n_probe, recall, avg_rerank, qps);
        return std::make_pair(recall, avg_rerank);
    };

    // Sweep W (baseline top-W) for each n_probe.
    printf("\n--- Strategy (a): TOP-W (today's behavior) ---\n");
    for (uint32_t n_probe : {1u, 2u, 4u, 8u}) {
        for (uint32_t W : {10u, 30u, 100u, 300u}) {
            run_strategy(("TOP-W=" + std::to_string(W)).c_str(), n_probe,
                         [&](uint32_t /*qi*/, const auto& r) {
                             std::vector<uint32_t> gids;
                             const uint32_t Weff = std::min<uint32_t>(W, (uint32_t)r.size());
                             gids.reserve(Weff);
                             for (uint32_t i = 0; i < Weff; i++) gids.push_back(r[i].second);
                             return gids;
                         });
        }
    }

    // Sweep MARGIN. Note: pq_dist is in quantized units (sum of m4 × uint4
    // values). The "best" pq_dist for a query is r[0].first. We rerank any
    // candidate with pq_dist <= best + margin.
    printf("\n--- Strategy (b): MARGIN (RaBitQ-style selective rerank) ---\n");
    for (uint32_t n_probe : {1u, 2u, 4u, 8u}) {
        for (float margin : {0.0f, 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f}) {
            char buf[64];
            snprintf(buf, sizeof(buf), "MARGIN=%.0f", margin);
            run_strategy(buf, n_probe,
                         [&](uint32_t /*qi*/, const auto& r) -> std::vector<uint32_t> {
                             if (r.empty()) return {};
                             const float best = r[0].first;
                             const float thresh = best + margin;
                             std::vector<uint32_t> gids;
                             for (auto& it : r) {
                                 if (it.first > thresh) break;
                                 gids.push_back(it.second);
                             }
                             return gids;
                         });
        }
    }

    // Strategy (c): TOP-W ∪ MARGIN. For each query, rerank top-W OR
    // within margin (whichever is larger). Useful to see if margin catches
    // candidates top-W misses.
    printf("\n--- Strategy (c): TOP-W ∪ MARGIN (covers both gaps) ---\n");
    for (uint32_t n_probe : {1u, 2u, 4u, 8u}) {
        for (uint32_t W : {30u, 100u}) {
            for (float margin : {100.0f, 500.0f}) {
                char buf[64];
                snprintf(buf, sizeof(buf), "W=%u ∪ M=%.0f", W, margin);
                run_strategy(buf, n_probe,
                             [&](uint32_t /*qi*/, const auto& r) -> std::vector<uint32_t> {
                                 if (r.empty()) return {};
                                 const float best = r[0].first;
                                 const float thresh = best + margin;
                                 std::vector<uint32_t> gids;
                                 std::unordered_set<uint32_t> seen;
                                 const uint32_t Weff = std::min<uint32_t>(W, (uint32_t)r.size());
                                 for (uint32_t i = 0; i < Weff; i++) {
                                     gids.push_back(r[i].second);
                                     seen.insert(r[i].second);
                                 }
                                 for (auto& it : r) {
                                     if (it.first > thresh) break;
                                     if (!seen.count(it.second)) {
                                         gids.push_back(it.second);
                                         seen.insert(it.second);
                                     }
                                 }
                                 return gids;
                             });
            }
        }
    }

    return 0;
}
