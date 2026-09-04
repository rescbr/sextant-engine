// Validation (2): Large-K routing recall + bytes-scanned Pareto.
//
// Question: at arxiv-nomic 1.34M, how does K ∈ {64, 256, 1024} affect:
// (a) routing recall at fixed n_probe? (expected: drops slightly, boundary)
// (b) bytes scanned per query at fixed recall? (the paging-determining metric)
// (c) QPS at fixed recall? (CPU-determining; less important for paging)
//
// The paging-relevant metric is (b): bytes scanned per query at recall 0.99.
// Smaller = better for 1B-scale sequential SSD reads.

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
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using namespace sextant;

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

struct KmeansResult { std::vector<float> centroids; };
KmeansResult kmeans(const float* data, uint32_t n, uint32_t dim, uint32_t K,
                    uint32_t iters = 8, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<float> cents(K * dim);
    const uint32_t first = rng() % n;
    std::memcpy(cents.data(), data + size_t(first)*dim, dim*4);
    std::vector<float> d2(n, std::numeric_limits<float>::infinity());
    for (uint32_t k = 1; k < K; k++) {
        const float* c = cents.data() + (k-1)*dim;
        for (uint32_t i = 0; i < n; i++) {
            const float* v = data + size_t(i)*dim;
            float d = 0; for (uint32_t d2i=0; d2i<dim; d2i++){float x=v[d2i]-c[d2i];d+=x*x;}
            if (d < d2[i]) d2[i] = d;
        }
        float sum = std::accumulate(d2.begin(), d2.end(), 0.0f);
        float r = (rng() / float(rng.max())) * sum;
        float acc = 0; uint32_t pick = n - 1;
        for (uint32_t i = 0; i < n; i++) { acc += d2[i]; if (acc >= r) { pick = i; break; } }
        std::memcpy(cents.data() + k*dim, data + size_t(pick)*dim, dim*4);
    }
    std::vector<float> new_cents(K * dim, 0);
    std::vector<uint32_t> counts(K, 0);
    for (uint32_t it = 0; it < iters; it++) {
        // Parallel Lloyd assignment + per-thread accumulators (avoids the
        // critical section on new_cents/counts during the reduction).
        std::fill(new_cents.begin(), new_cents.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        const uint32_t nthreads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::vector<float>> local_new(nthreads, std::vector<float>(K*dim, 0.0f));
        std::vector<std::vector<uint32_t>> local_counts(nthreads, std::vector<uint32_t>(K, 0));
        auto worker = [&](uint32_t tid, uint32_t lo, uint32_t hi) {
            std::vector<float>& ln = local_new[tid];
            std::vector<uint32_t>& lc = local_counts[tid];
            for (uint32_t i = lo; i < hi; i++) {
                const float* v = data + size_t(i)*dim;
                float best = std::numeric_limits<float>::infinity(); uint32_t bk = 0;
                for (uint32_t k = 0; k < K; k++) {
                    const float* c = cents.data() + k*dim;
                    float d = 0; for (uint32_t di=0; di<dim; di++){float x=v[di]-c[di];d+=x*x;}
                    if (d < best) { best = d; bk = k; }
                }
                float* nc = ln.data() + bk*dim;
                for (uint32_t di = 0; di < dim; di++) nc[di] += v[di];
                lc[bk]++;
            }
        };
        std::vector<std::thread> ts;
        for (uint32_t t = 0; t < nthreads; t++) {
            const uint32_t lo = (uint64_t)t * n / nthreads;
            const uint32_t hi = (uint64_t)(t+1) * n / nthreads;
            ts.emplace_back(worker, t, lo, hi);
        }
        for (auto& th : ts) th.join();
        for (uint32_t t = 0; t < nthreads; t++) {
            for (uint32_t k = 0; k < K*dim; k++) new_cents[k] += local_new[t][k];
            for (uint32_t k = 0; k < K; k++) counts[k] += local_counts[t][k];
        }
        for (uint32_t k = 0; k < K; k++)
            if (counts[k] > 0)
                for (uint32_t di = 0; di < dim; di++)
                    cents[k*dim + di] = new_cents[k*dim + di] / counts[k];
    }
    return {std::move(cents)};
}

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

int main(int argc, char** argv) {
    // Line-buffer stdout so progress is visible in background runs.
    setvbuf(stdout, nullptr, _IOLBF, 0);

    const std::string base_path = (argc > 1) ? argv[1] : "datasets/arxiv_nomic_base.fbin";
    const std::string query_path = (argc > 2) ? argv[2] : "datasets/arxiv_nomic_query.fbin";
    const std::string gt_path = (argc > 3) ? argv[3] : "datasets/arxiv_nomic_gt.gtmm";

    FbinData db, queries; GroundTruth gt;
    if (!read_fbin(base_path, db)) { printf("cannot read %s\n", base_path.c_str()); return 1; }
    if (!read_fbin(query_path, queries)) { printf("cannot read %s\n", query_path.c_str()); return 1; }
    if (!read_gt(gt_path, gt)) { printf("cannot read %s\n", gt_path.c_str()); return 1; }
    printf("Loaded: db %u×%u, queries %u\n", db.n, db.dim, queries.n);

    const uint32_t m4 = 192;
    printf("Training 4-bit PQ (m=%u) on 20k sample...\n", m4);
    PqQuantizer pq4(MetricKind::L2Sq, db.dim, m4, 4, 42);
    pq4.train(db.data.data(), std::min<uint32_t>(20000, db.n));

    // Encode all DB once (flat nibbles).
    printf("Encoding all %u vectors...\n", db.n);
    std::vector<uint8_t> nibbles((size_t)db.n * m4, 0);
    {
        std::vector<uint8_t> pk(pq4.code_size());
        for (uint32_t i = 0; i < db.n; i++) {
            pq4.encode(db.data.data() + size_t(i)*db.dim, pk.data());
            for (uint32_t s = 0; s < m4; s++)
                nibbles[(size_t)i * m4 + s] = (uint8_t)((pk[s/2] >> ((s%2)*4)) & 0xF);
        }
    }

    auto build_lut4 = [&](const float* query, uint8_t* lut4) {
        std::vector<float> f32(pq4.lut_size());
        pq4.preprocess_query(query, f32.data());
        float max_span = 0; std::vector<float> seg_min(m4);
        for (uint32_t s = 0; s < m4; s++) {
            float* row = f32.data() + s*16; float mn = row[0];
            for (uint32_t c = 1; c < 16; c++) if (row[c] < mn) mn = row[c];
            seg_min[s] = mn;
            for (uint32_t c = 0; c < 16; c++) { float sp = row[c]-mn; if (sp > max_span) max_span = sp; }
        }
        const float A = (max_span > 0) ? 15.0f / max_span : 0;
        for (uint32_t s = 0; s < m4; s++) {
            const float* row = f32.data() + s*16; float mn = seg_min[s];
            for (uint32_t c = 0; c < 16; c++) {
                int q4 = (int)((row[c]-mn)*A + 0.5f);
                lut4[s*16+c] = (uint8_t)std::max(0, std::min(15, q4));
            }
        }
    };

    const uint32_t n_query = std::min<uint32_t>(200, queries.n);
    const uint32_t BYTES_PER_VEC = m4 / 2;  // 4-bit packed

    printf("\n=== Large-K Pareto (m4=%u, 4-bit, %u bytes/vec) ===\n", m4, BYTES_PER_VEC);
    printf("Each (K, n_probe, W) line: recall@10 / QPS / codes scanned / MB scanned per query\n");
    printf("%-6s %-4s %-6s %10s %10s %12s %12s\n",
           "K", "np", "W", "recall", "QPS", "codes/qry", "MB/qry");
    printf("-----------------------------------------------------------------------\n");

    for (uint32_t K : {64u, 256u, 1024u}) {
        // k-means + assign + per-shard block layout.
        auto t0 = std::chrono::steady_clock::now();
        const uint32_t sample_n = std::min<uint32_t>(50000, db.n);
        auto km = kmeans(db.data.data(), sample_n, db.dim, K, 8, 42);
        // Parallel assignment: each vector's nearest centroid is independent.
        // Use a flat assignment array + per-thread shard buffers to avoid the
        // critical section on shard_members.push_back.
        std::vector<uint32_t> assign(db.n, 0);
        const uint32_t nthreads = std::max(1u, std::thread::hardware_concurrency());
        auto assign_worker = [&](uint32_t lo, uint32_t hi) {
            for (uint32_t i = lo; i < hi; i++) {
                const float* v = db.data.data() + size_t(i)*db.dim;
                float best = std::numeric_limits<float>::infinity(); uint32_t bk = 0;
                for (uint32_t k = 0; k < K; k++) {
                    const float* c = km.centroids.data() + k*db.dim;
                    float d = 0; for (uint32_t di=0; di<db.dim; di++){float x=v[di]-c[di];d+=x*x;}
                    if (d < best) { best = d; bk = k; }
                }
                assign[i] = bk;
            }
        };
        {
            std::vector<std::thread> ts;
            for (uint32_t t = 0; t < nthreads; t++) {
                const uint32_t lo = (uint64_t)t * db.n / nthreads;
                const uint32_t hi = (uint64_t)(t+1) * db.n / nthreads;
                ts.emplace_back(assign_worker, lo, hi);
            }
            for (auto& th : ts) th.join();
        }
        std::vector<std::vector<uint32_t>> shard_members(K);
        for (uint32_t i = 0; i < db.n; i++) shard_members[assign[i]].push_back(i);
        // Per-shard block layout.
        std::vector<std::vector<uint8_t>> shard_blocks(K);
        for (uint32_t k = 0; k < K; k++) {
            const auto& members = shard_members[k];
            const uint32_t sz = (uint32_t)members.size();
            const uint32_t n_blocks = (sz + 31) / 32;
            shard_blocks[k].resize((size_t)n_blocks * m4 * 16, 0);
            for (uint32_t b = 0; b < n_blocks; b++) {
                for (uint32_t s = 0; s < m4; s++) {
                    for (uint32_t lane = 0; lane < 16; lane++) {
                        const uint32_t v0 = b*32+lane, v1 = b*32+16+lane;
                        uint8_t lo = (v0 < sz) ? nibbles[(size_t)members[v0]*m4+s] : 0;
                        uint8_t hi = (v1 < sz) ? nibbles[(size_t)members[v1]*m4+s] : 0;
                        shard_blocks[k][((size_t)b*m4+s)*16+lane] = (uint8_t)((hi<<4)|lo);
                    }
                }
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        printf("# K=%u setup: %.1fs, shard avg=%u min=%u max=%u\n",
               K, std::chrono::duration<double>(t1-t0).count(),
               db.n/K,
               std::accumulate(shard_members.begin(), shard_members.end(), UINT32_MAX,
                               [](uint32_t a, const auto& b){return std::min(a,(uint32_t)b.size());}),
               std::accumulate(shard_members.begin(), shard_members.end(), 0u,
                               [](uint32_t a, const auto& b){return std::max(a,(uint32_t)b.size());}));

        // Pick n_probe values appropriate per K (we want to span comparable bytes-scanned).
        std::vector<uint32_t> nps;
        if (K == 64) nps = {1u, 2u, 4u, 8u};
        else if (K == 256) nps = {4u, 8u, 16u, 32u};
        else nps = {8u, 16u, 32u, 64u};  // K=1024

        for (uint32_t n_probe : nps) {
            for (uint32_t W : {100u, 300u}) {
                std::vector<uint8_t> lut4((size_t)m4 * 16);
                std::vector<std::pair<float,uint32_t>> ranked;
                std::vector<std::pair<float,uint32_t>> exact;
                uint64_t hits = 0;
                uint64_t codes_scanned = 0;
                auto qt0 = std::chrono::steady_clock::now();
                for (uint32_t qi = 0; qi < n_query; qi++) {
                    const float* query = queries.data.data() + size_t(qi)*db.dim;
                    std::vector<std::pair<float,uint32_t>> cd(K);
                    for (uint32_t k = 0; k < K; k++) {
                        const float* c = km.centroids.data() + k*db.dim;
                        float d = 0; for (uint32_t di=0; di<db.dim; di++){float x=query[di]-c[di];d+=x*x;}
                        cd[k] = {d, k};
                    }
                    std::partial_sort(cd.begin(), cd.begin()+n_probe, cd.end());
                    build_lut4(query, lut4.data());
                    ranked.clear();
                    for (uint32_t p = 0; p < n_probe; p++) {
                        const uint32_t k = cd[p].second;
                        const auto& blocks = shard_blocks[k];
                        const uint32_t sz = (uint32_t)shard_members[k].size();
                        const uint32_t nb = (sz + 31) / 32;
                        codes_scanned += sz;
                        for (uint32_t b = 0; b < nb; b++) {
                            uint32_t out[32];
                            pq4_block32(blocks.data() + (size_t)b*m4*16, lut4.data(), m4, out);
                            for (uint32_t j = 0; j < 32; j++) {
                                const uint32_t lane = b*32+j;
                                if (lane < sz) ranked.push_back({(float)out[j], shard_members[k][lane]});
                            }
                        }
                    }
                    if (ranked.size() > W)
                        std::nth_element(ranked.begin(), ranked.begin()+W, ranked.end());
                    const uint32_t Weff = std::min<uint32_t>(W, (uint32_t)ranked.size());
                    std::sort(ranked.begin(), ranked.begin()+Weff);
                    exact.resize(Weff);
                    for (uint32_t i = 0; i < Weff; i++) {
                        const uint32_t gid = ranked[i].second;
                        const float* v = db.data.data() + size_t(gid)*db.dim;
                        float d = 0; for (uint32_t di=0; di<db.dim; di++){float x=v[di]-query[di];d+=x*x;}
                        exact[i] = {d, gid};
                    }
                    std::partial_sort(exact.begin(), exact.begin()+std::min<uint32_t>(10,Weff), exact.end());
                    const uint32_t* gt_row = gt.ids.data() + size_t(qi)*gt.k;
                    for (uint32_t r = 0; r < std::min<uint32_t>(10,Weff); r++)
                        for (uint32_t g = 0; g < 10; g++)
                            if (gt_row[g] == exact[r].second) { hits++; break; }
                }
                auto qt1 = std::chrono::steady_clock::now();
                const double secs = std::chrono::duration<double>(qt1-qt0).count();
                const float recall = (float)hits / (float(n_query) * 10.0f);
                const float qps = n_query / secs;
                const float codes_qry = (float)codes_scanned / n_query;
                const float mb_qry = codes_qry * BYTES_PER_VEC / (1024.0f * 1024.0f);
                printf("%-6u %-4u %-6u %10.4f %10.1f %12.0f %12.2f\n",
                       K, n_probe, W, recall, qps, codes_qry, mb_qry);
            }
        }
        printf("\n");
    }

    return 0;
}
