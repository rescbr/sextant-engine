// Three-way precision comparison for IVF-list-scan:
//   A: 4-bit PQ (m=192, freshly trained) — fastest scan, +50% disk vs E
//   C: 8-bit PQ (m=96, scanned via split-table) — no storage change, slowest scan
//   E: 8-bit PQ stored (m=96) + on-the-fly 8→4 collapse (m=192 derived 4-bit)
//      — no storage change vs current, recall between A and C
//
// Question: at matched recall targets on arxiv-nomic 1.34M, what's the QPS
// of each? Does 4-bit's 4.5x kernel speed survive the recall adjustment
// (more rerank needed)? Is option E good enough to skip storing a separate
// 4-bit codebook?
//
// All three use the SAME IVF shards (K=64, fixed k-means assignment) —
// only the per-shard scan precision varies. This isolates the precision
// effect from routing differences.

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
#include <utility>
#include <vector>

using namespace sextant;

// --- Readers (lifted from prior spikes).
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

// --- K-means (same as spike_hybrid_ivf).
struct KmeansResult { std::vector<float> centroids; std::vector<uint32_t> assign; };
KmeansResult kmeans(const float* data, uint32_t n, uint32_t dim, uint32_t K,
                    uint32_t iters = 10, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<float> cents(K * dim);
    {
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
    }
    std::vector<uint32_t> assign(n, 0);
    std::vector<float> new_cents(K * dim, 0);
    std::vector<uint32_t> counts(K, 0);
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t i = 0; i < n; i++) {
            const float* v = data + size_t(i)*dim;
            float best = std::numeric_limits<float>::infinity(); uint32_t bk = 0;
            for (uint32_t k = 0; k < K; k++) {
                const float* c = cents.data() + k*dim;
                float d = 0; for (uint32_t di=0; di<dim; di++){float x=v[di]-c[di];d+=x*x;}
                if (d < best) { best = d; bk = k; }
            }
            assign[i] = bk;
        }
        std::fill(new_cents.begin(), new_cents.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (uint32_t i = 0; i < n; i++) {
            const float* v = data + size_t(i)*dim;
            float* nc = new_cents.data() + assign[i]*dim;
            for (uint32_t di = 0; di < dim; di++) nc[di] += v[di];
            counts[assign[i]]++;
        }
        for (uint32_t k = 0; k < K; k++) {
            if (counts[k] > 0)
                for (uint32_t di = 0; di < dim; di++)
                    cents[k*dim + di] = new_cents[k*dim + di] / counts[k];
        }
    }
    return {std::move(cents), std::move(assign)};
}

// --- 4-bit FastScan kernel (32 codes/block).
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

// Build a segment-major block layout from a flat [N][m] nibble array.
// Block b, segment s, byte k: low=v(b*32+k), high=v(b*32+16+k).
static std::vector<uint8_t> pack_blocks_4bit(const std::vector<uint8_t>& nibbles,
                                              uint32_t m, uint32_t n) {
    const uint32_t n_blocks = (n + 31) / 32;
    std::vector<uint8_t> blocks((size_t)n_blocks * m * 16, 0);
    for (uint32_t b = 0; b < n_blocks; b++) {
        for (uint32_t s = 0; s < m; s++) {
            for (uint32_t k = 0; k < 16; k++) {
                const uint32_t v0 = b * 32 + k, v1 = b * 32 + 16 + k;
                uint8_t lo = (v0 < n) ? nibbles[(size_t)v0 * m + s] : 0;
                uint8_t hi = (v1 < n) ? nibbles[(size_t)v1 * m + s] : 0;
                blocks[((size_t)b * m + s) * 16 + k] = (uint8_t)((hi << 4) | lo);
            }
        }
    }
    return blocks;
}

// 8-bit segment-major block layout from flat [N][m] byte array.
// Block b, segment s, lane j: byte at offset (b*m + s)*16 + j = vector (b*16+j) seg s.
static std::vector<uint8_t> pack_blocks_8bit(const std::vector<uint8_t>& bytes,
                                              uint32_t m, uint32_t n) {
    const uint32_t n_blocks = (n + 15) / 16;
    std::vector<uint8_t> blocks((size_t)n_blocks * m * 16, 0);
    for (uint32_t b = 0; b < n_blocks; b++) {
        for (uint32_t s = 0; s < m; s++) {
            for (uint32_t j = 0; j < 16; j++) {
                const uint32_t v = b * 16 + j;
                if (v < n) blocks[((size_t)b * m + s) * 16 + j] = bytes[(size_t)v * m + s];
            }
        }
    }
    return blocks;
}

// --- LUT builders.
// 4-bit LUT (FAISS NormTableScaler, unclamped, u32 accumulator).
static float build_lut4(const PqQuantizer& q, const float* query, uint8_t* lut4) {
    const uint32_t m = q.m();
    std::vector<float> f32(q.lut_size());
    q.preprocess_query(query, f32.data());
    float max_span = 0; std::vector<float> seg_min(m);
    for (uint32_t s = 0; s < m; s++) {
        float* row = f32.data() + s*16; float mn = row[0];
        for (uint32_t c = 1; c < 16; c++) if (row[c] < mn) mn = row[c];
        seg_min[s] = mn;
        for (uint32_t c = 0; c < 16; c++) { float sp = row[c]-mn; if (sp > max_span) max_span = sp; }
    }
    const float A = (max_span > 0) ? 15.0f / max_span : 0;
    for (uint32_t s = 0; s < m; s++) {
        const float* row = f32.data() + s*16; float mn = seg_min[s];
        for (uint32_t c = 0; c < 16; c++) {
            int q4 = (int)((row[c]-mn) * A + 0.5f);
            lut4[s*16 + c] = (uint8_t)std::max(0, std::min(15, q4));
        }
    }
    return A;
}

// 8-bit LUT for the split-table kernel (uint8, global scale).
static void build_lut8(const PqQuantizer& q, const float* query, uint8_t* lut8) {
    const uint32_t m = q.m();
    std::vector<float> f32(q.lut_size());
    q.preprocess_query(query, f32.data());
    float max_span = 0; std::vector<float> seg_min(m);
    for (uint32_t s = 0; s < m; s++) {
        float* row = f32.data() + s*256; float mn = row[0];
        for (uint32_t c = 1; c < 256; c++) if (row[c] < mn) mn = row[c];
        seg_min[s] = mn;
        for (uint32_t c = 0; c < 256; c++) { float sp = row[c]-mn; if (sp > max_span) max_span = sp; }
    }
    float A = (max_span > 0) ? 255.0f / max_span : 0;
    // No u16-overflow clamp needed (split-table kernel uses u32 accum).
    for (uint32_t s = 0; s < m; s++) {
        const float* row = f32.data() + s*256; float mn = seg_min[s];
        for (uint32_t c = 0; c < 256; c++) {
            int q8 = (int)((row[c]-mn) * A + 0.5f);
            lut8[s*256 + c] = (uint8_t)std::max(0, std::min(255, q8));
        }
    }
}

// --- Precision variants are wired inline in main() below (no abstract base —
// each has different scan/LUT signatures, not worth unifying).

int main(int argc, char** argv) {
    const std::string base_path = (argc > 1) ? argv[1] : "datasets/arxiv_nomic_base.fbin";
    const std::string query_path = (argc > 2) ? argv[2] : "datasets/arxiv_nomic_query.fbin";
    const std::string gt_path = (argc > 3) ? argv[3] : "datasets/arxiv_nomic_gt.gtmm";

    FbinData db, queries; GroundTruth gt;
    if (!read_fbin(base_path, db)) { printf("cannot read %s\n", base_path.c_str()); return 1; }
    if (!read_fbin(query_path, queries)) { printf("cannot read %s\n", query_path.c_str()); return 1; }
    if (!read_gt(gt_path, gt)) { printf("cannot read %s\n", gt_path.c_str()); return 1; }
    printf("Loaded: db %u×%u (%.0f MB), queries %u\n",
           db.n, db.dim, db.data.size()*4/1e6, queries.n);

    // --- IVF routing (shared across all variants).
    const uint32_t K = 64;
    printf("\n[K=%u] k-means on 50k sample...\n", K);
    auto t0 = std::chrono::steady_clock::now();
    const uint32_t sample_n = std::min<uint32_t>(50000, db.n);
    auto km = kmeans(db.data.data(), sample_n, db.dim, K, 8, 42);
    auto t1 = std::chrono::steady_clock::now();
    printf("  k-means: %.1fs. Assigning all vectors...\n",
           std::chrono::duration<double>(t1-t0).count());
    std::vector<std::vector<uint32_t>> shard_members(K);
    t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < db.n; i++) {
        const float* v = db.data.data() + size_t(i)*db.dim;
        float best = std::numeric_limits<float>::infinity(); uint32_t bk = 0;
        for (uint32_t k = 0; k < K; k++) {
            const float* c = km.centroids.data() + k*db.dim;
            float d = 0; for (uint32_t di=0; di<db.dim; di++){float x=v[di]-c[di];d+=x*x;}
            if (d < best) { best = d; bk = k; }
        }
        shard_members[bk].push_back(i);
    }
    t1 = std::chrono::steady_clock::now();
    printf("  assign: %.1fs\n", std::chrono::duration<double>(t1-t0).count());

    // --- Train all three quantizers.
    const uint32_t m4 = 192, m8 = 96;
    printf("Training quantizers on 20k sample...\n");
    PqQuantizer pq4_trained(MetricKind::L2Sq, db.dim, m4, 4, 42);
    PqQuantizer pq8(MetricKind::L2Sq, db.dim, m8, 8, 42);
    pq4_trained.train(db.data.data(), std::min<uint32_t>(20000, db.n));
    pq8.train(db.data.data(), std::min<uint32_t>(20000, db.n));

    // For option E (8→4 collapse): build map8to4[s][256] by k-means clustering
    // the 256 8-bit centroids into 16 groups per segment, with the 4-bit LUT
    // computed at the 4-bit cluster medoid. The 4-bit code for an 8-bit code
    // is map8to4[s][8bit_code].
    printf("Building 8→4 collapse map for option E...\n");
    std::vector<uint8_t> map8to4((size_t)m8 * 256, 0);
    for (uint32_t s = 0; s < m8; s++) {
        // Get the 256 8-bit centroids for this segment as 8-dim points.
        std::vector<float> cents_8(256 * 8);
        const uint32_t sub_dim = db.dim / m8;
        for (uint32_t c = 0; c < 256; c++) {
            const float* src = pq8.codebook() + (size_t)s * 256 * sub_dim + c * sub_dim;
            for (uint32_t d = 0; d < sub_dim; d++) cents_8[c*8 + d] = src[d];
        }
        // 1D index k-means into 16 clusters over the 256 centroids.
        // For simplicity use a random init + few Lloyd iters in 8-dim space.
        std::mt19937_64 rng(42 + s);
        std::vector<float> cents_4(16 * 8);
        for (uint32_t k = 0; k < 16; k++) {
            const uint32_t pick = rng() % 256;
            std::memcpy(cents_4.data() + k*8, cents_8.data() + pick*8, 8*4);
        }
        for (uint32_t iter = 0; iter < 5; iter++) {
            std::vector<uint8_t> assign(256);
            for (uint32_t c = 0; c < 256; c++) {
                float best = std::numeric_limits<float>::infinity(); uint32_t bk = 0;
                for (uint32_t k = 0; k < 16; k++) {
                    float d = 0;
                    for (uint32_t d2 = 0; d2 < 8; d2++) {
                        float x = cents_8[c*8+d2] - cents_4[k*8+d2]; d += x*x;
                    }
                    if (d < best) { best = d; bk = k; }
                }
                assign[c] = bk;
            }
            std::vector<float> sum(16*8, 0); std::vector<uint32_t> cnt(16, 0);
            for (uint32_t c = 0; c < 256; c++) {
                for (uint32_t d2 = 0; d2 < 8; d2++) sum[assign[c]*8+d2] += cents_8[c*8+d2];
                cnt[assign[c]]++;
            }
            for (uint32_t k = 0; k < 16; k++)
                if (cnt[k] > 0)
                    for (uint32_t d2 = 0; d2 < 8; d2++)
                        cents_4[k*8+d2] = sum[k*8+d2] / cnt[k];
            if (iter == 4) for (uint32_t c = 0; c < 256; c++) map8to4[s*256 + c] = assign[c];
        }
    }
    // For option E's LUT, we need a 4-bit PQ quantizer that produces the
    // collapsed 4-bit codes. Easiest: synthesize a "virtual" 4-bit codebook
    // = the 16 cluster medoids per segment, stuff it into a real PqQuantizer
    // so we can reuse preprocess_query + build_lut4.
    PqQuantizer pq4_collapse(MetricKind::L2Sq, db.dim, m8, 4, 42);  // m8 segments!
    // Overwrite its codebook with the collapse medoids so segments align with pq8.
    {
        // Access via the public codebook() — read-only. We need to re-train it
        // to overwrite; or build a parallel "manual" LUT in the scan path.
        // Simpler: in the scan, build_lut4 from pq8's float LUT directly,
        // grouping by map8to4. We'll handle E specially in the scan.
    }

    // --- Encode all DB vectors three ways (flat layouts for easy re-blocking).
    printf("Encoding all %u vectors (4-bit trained, 8-bit, 8-bit for E)...\n", db.n);
    t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> nibbles_A((size_t)db.n * m4, 0);  // option A
    std::vector<uint8_t> bytes_C((size_t)db.n * m8, 0);    // option C (and E source)
    {
        std::vector<uint8_t> pk4(pq4_trained.code_size());
        std::vector<uint8_t> pk8(pq8.code_size());
        for (uint32_t i = 0; i < db.n; i++) {
            const float* v = db.data.data() + size_t(i)*db.dim;
            pq4_trained.encode(v, pk4.data());
            pq8.encode(v, pk8.data());
            for (uint32_t s = 0; s < m4; s++)
                nibbles_A[i*m4 + s] = (uint8_t)((pk4[s/2] >> ((s%2)*4)) & 0xF);
            for (uint32_t s = 0; s < m8; s++)
                bytes_C[i*m8 + s] = pk8[s];
        }
    }
    t1 = std::chrono::steady_clock::now();
    printf("  encode: %.1fs\n", std::chrono::duration<double>(t1-t0).count());

    // Per-shard block layouts for each variant.
    auto build_shard_layouts = [&]() {
        struct Layouts {
            std::vector<std::vector<uint8_t>> blocks_A, blocks_C;
            // For E, we use blocks_C but collapse to 4-bit at scan time (no
            // separate layout). We rebuild blocks_E from the 8-bit codes
            // collapsed via map8to4, so E scans 4-bit blocks derived from C.
            std::vector<std::vector<uint8_t>> blocks_E;
        } L;
        L.blocks_A.resize(K); L.blocks_C.resize(K); L.blocks_E.resize(K);
        for (uint32_t k = 0; k < K; k++) {
            const auto& members = shard_members[k];
            const uint32_t sz = (uint32_t)members.size();
            // A: gather nibbles_A for member vectors, pack.
            std::vector<uint8_t> nib(sz * m4);
            for (uint32_t i = 0; i < sz; i++)
                std::memcpy(nib.data() + (size_t)i*m4, nibbles_A.data() + (size_t)members[i]*m4, m4);
            L.blocks_A[k] = pack_blocks_4bit(nib, m4, sz);
            // C: gather bytes_C, pack 8-bit.
            std::vector<uint8_t> byt(sz * m8);
            for (uint32_t i = 0; i < sz; i++)
                std::memcpy(byt.data() + (size_t)i*m8, bytes_C.data() + (size_t)members[i]*m8, m8);
            L.blocks_C[k] = pack_blocks_8bit(byt, m8, sz);
            // E: collapse bytes_C through map8to4, pack 4-bit.
            std::vector<uint8_t> e_nib(sz * m8);  // m8 segments (not m4!)
            for (uint32_t i = 0; i < sz; i++) {
                const uint8_t* src = bytes_C.data() + (size_t)members[i]*m8;
                for (uint32_t s = 0; s < m8; s++) e_nib[i*m8 + s] = map8to4[s*256 + src[s]];
            }
            L.blocks_E[k] = pack_blocks_4bit(e_nib, m8, sz);  // m8-seg 4-bit blocks
        }
        return L;
    };
    printf("Building per-shard block layouts...\n");
    t0 = std::chrono::steady_clock::now();
    auto L = build_shard_layouts();
    t1 = std::chrono::steady_clock::now();
    printf("  layouts: %.1fs\n", std::chrono::duration<double>(t1-t0).count());

    // --- Scan + measure for each variant.
    auto run_variant = [&](const std::string& name,
                           const std::vector<std::vector<uint8_t>>& shard_blocks,
                           uint32_t m_seg, bool is_4bit,
                           auto build_lut_fn, auto scan_block_fn,
                           uint32_t n_query) {
        printf("\n--- %s (m=%u, %s) ---\n", name.c_str(), m_seg,
               is_4bit ? "4-bit" : "8-bit");
        printf("%-6s %-6s %10s %10s\n", "np", "W", "recall@10", "QPS");
        for (uint32_t n_probe : {2u, 4u, 8u}) {
            for (uint32_t W : {30u, 100u, 300u}) {
                std::vector<uint8_t> lut((size_t)m_seg * (is_4bit ? 16 : 256));
                std::vector<std::pair<float,uint32_t>> ranked;
                std::vector<std::pair<float,uint32_t>> exact;
                uint64_t hits = 0;
                auto qt0 = std::chrono::steady_clock::now();
                for (uint32_t qi = 0; qi < n_query; qi++) {
                    const float* query = queries.data.data() + size_t(qi)*db.dim;
                    // Route.
                    std::vector<std::pair<float,uint32_t>> cd(K);
                    for (uint32_t k = 0; k < K; k++) {
                        const float* c = km.centroids.data() + k*db.dim;
                        float d = 0; for (uint32_t di=0; di<db.dim; di++){float x=query[di]-c[di];d+=x*x;}
                        cd[k] = {d, k};
                    }
                    std::partial_sort(cd.begin(), cd.begin()+n_probe, cd.end());
                    // LUT.
                    build_lut_fn(query, lut.data());
                    // Scan.
                    ranked.clear();
                    for (uint32_t p = 0; p < n_probe; p++) {
                        const uint32_t k = cd[p].second;
                        const auto& blocks = shard_blocks[k];
                        const uint32_t sz = (uint32_t)shard_members[k].size();
                        const uint32_t codes_per_block = is_4bit ? 32 : 16;
                        const uint32_t nb = (sz + codes_per_block - 1) / codes_per_block;
                        const auto& ids = shard_members[k];
                        for (uint32_t b = 0; b < nb; b++) {
                            uint32_t out[32];
                            scan_block_fn(blocks.data() + (size_t)b * m_seg * 16,
                                          lut.data(), m_seg, out);
                            for (uint32_t j = 0; j < codes_per_block; j++) {
                                const uint32_t lane = b * codes_per_block + j;
                                if (lane < sz) ranked.push_back({(float)out[j], ids[lane]});
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
                printf("%-6u %-6u %10.4f %10.1f\n", n_probe, W, recall, qps);
            }
        }
    };

    const uint32_t n_query = std::min<uint32_t>(200, queries.n);

    // --- Variant A: 4-bit trained.
    run_variant("A: 4-bit trained (m=192)", L.blocks_A, m4, /*is_4bit=*/true,
                [&](const float* q, uint8_t* lut) { build_lut4(pq4_trained, q, lut); },
                [&](const uint8_t* blk, const uint8_t* lut, uint32_t m, uint32_t* out) {
                    pq4_block32(blk, lut, m, out);
                }, n_query);

    // --- Variant C: 8-bit scanned via split-table.
    run_variant("C: 8-bit split-table (m=96)", L.blocks_C, m8, /*is_4bit=*/false,
                [&](const float* q, uint8_t* lut) { build_lut8(pq8, q, lut); },
                [&](const uint8_t* blk, const uint8_t* lut, uint32_t m, uint32_t* out) {
                    // 8-bit split-table kernel from simd_kernels.hpp. Out is 16
                    // uint32 (not 32); the scan loop above reads codes_per_block
                    // (16 for 8-bit) values, so this works.
                    simd::fastscan_block16(blk, lut, m, (uint16_t)0xFFFF, out);
                }, n_query);

    // --- Variant E: 8-bit stored, collapsed to 4-bit on the fly.
    // Build LUT specially: take pq8's float LUT, group 256 entries into 16
    // via map8to4 (min per group), quantize that to 4-bit.
    run_variant("E: 8-bit→4-bit collapse (m=96 segs)", L.blocks_E, m8, /*is_4bit=*/true,
                [&](const float* q, uint8_t* lut) {
                    // Get float LUT from pq8 (256 entries per seg), collapse to 16.
                    std::vector<float> f32(pq8.lut_size());
                    pq8.preprocess_query(q, f32.data());
                    // For each segment, find min/max over 16 groups (defined by map8to4).
                    // The 4-bit LUT entry for cluster k = min of (float LUT[c] for c with map[c]==k).
                    // Then quantize the 16 group-mins to [0,15] using global max_span across segs.
                    std::vector<float> group_min((size_t)m8 * 16);
                    for (uint32_t s = 0; s < m8; s++) {
                        for (uint16_t k = 0; k < 16; k++) group_min[s*16+k] = std::numeric_limits<float>::max();
                        for (uint16_t c = 0; c < 256; c++) {
                            const uint8_t k = map8to4[s*256 + c];
                            const float v = f32[s*256 + c];
                            if (v < group_min[s*16+k]) group_min[s*16+k] = v;
                        }
                    }
                    // Global scale over all group_min spans.
                    float max_span = 0; std::vector<float> seg_min(m8);
                    for (uint32_t s = 0; s < m8; s++) {
                        float mn = group_min[s*16];
                        for (uint16_t k = 1; k < 16; k++) if (group_min[s*16+k] < mn) mn = group_min[s*16+k];
                        seg_min[s] = mn;
                        for (uint16_t k = 0; k < 16; k++) {
                            float sp = group_min[s*16+k] - mn; if (sp > max_span) max_span = sp;
                        }
                    }
                    const float A = (max_span > 0) ? 15.0f / max_span : 0;
                    for (uint32_t s = 0; s < m8; s++) {
                        for (uint16_t k = 0; k < 16; k++) {
                            int q4 = (int)((group_min[s*16+k]-seg_min[s])*A + 0.5f);
                            lut[s*16+k] = (uint8_t)std::max(0, std::min(15, q4));
                        }
                    }
                },
                [&](const uint8_t* blk, const uint8_t* lut, uint32_t m, uint32_t* out) {
                    pq4_block32(blk, lut, m, out);
                }, n_query);

    return 0;
}
