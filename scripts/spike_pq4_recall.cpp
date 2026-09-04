// Spike: 4-bit PQ + FP32 rerank recall probe on arxiv100k.
//
// Front-loads the critical uncertainty: can FP32 rerank of top-W recover the
// 4-bit PQ recall collapse (0.75 per our docs)? If yes, the 5.6× kernel speed
// win is realizable; if no, all 4-bit-FastScan variants die together and we
// abandon the approach.
//
// This is a BRUTE-FORCE scan (no graph, no IVF) — isolates the
// 4-bit-PQ-plus-rerank recall ceiling from any algorithmic concern. We're
// measuring the recall *given a complete scan*, not the algorithm's ability
// to find candidates.
//
// Output: recall@10 + QPS at W ∈ {10, 30, 100, 300, 1000} rerank depths.

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
#include <vector>

using namespace sextant;

// --- Minimal .fbin / .gt readers (lifted from pq_explore.cpp / benchmark.cpp).

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

// --- 4-bit FastScan kernel (FAISS-style, 32 codes/block).
// code_block: [m][16] segment-major, 2 codes packed per byte (lo=even lane,
//             hi=odd lane). Lane j (j<16) is the low nibble of byte j;
//             lane 16+j is the high nibble of byte j.
// lut4:       [m][16] uint8 LUT.
// out:        32 uint32 distances.
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

// Build the 4-bit LUT for a query: m segments × 16 entries.
// Uses the FAISS NormTableScaler approach (per-query GLOBAL scale, not
// per-segment) so cross-segment distances compose correctly. Per-segment
// mins are subtracted (and summed into the offset, but for argmin we drop it).
//
// IMPORTANT: uses the natural scale A = 15 / max_span WITHOUT clamping, because
// the kernel accumulates into uint32 (not uint16). At m=192 segments × 15 max
// per segment = 2880 max sum; A can grow up to ~2.3M before u32 overflow at
// m=192 — generous headroom for tight LUTs.
static float g_lut4_scale;  // communicated to scan for debugging

static void build_lut4(const PqQuantizer& q, const float* query, uint8_t* lut4) {
    const uint32_t m = q.m();
    std::vector<float> f32_lut(q.lut_size());
    q.preprocess_query(query, f32_lut.data());

    // Per-segment min, plus global max_span.
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
    // A = 15 / max_span, no clamp (u32 accumulator has wide headroom).
    const float A = (max_span > 0) ? 15.0f / max_span : 0.0f;
    g_lut4_scale = A;

    for (uint32_t s = 0; s < m; s++) {
        const float* row = f32_lut.data() + s * 16;
        const float mn = seg_min[s];
        for (uint32_t c = 0; c < 16; c++) {
            const int q4 = (int)((row[c] - mn) * A + 0.5f);
            lut4[s * 16 + c] = (uint8_t)std::max(0, std::min(15, q4));
        }
    }
}

// Encode the whole dataset to 4-bit block layout: [n_blocks][m][16] bytes.
// Each block holds 32 vectors; per segment s, 16 bytes pack vectors (2k, 2k+1)
// into byte k (low nibble = vector 2k's segment-s code, high = 2k+1's).
static std::vector<uint8_t> encode_dataset_4bit(const PqQuantizer& q,
                                                const FbinData& db) {
    const uint32_t m = q.m();
    const uint32_t n_blocks = (db.n + 31) / 32;
    std::vector<uint8_t> blocks((size_t)n_blocks * m * 16, 0);

    // First, encode every vector to its m nibbles into a flat [N][m] array
    // (one byte per segment, value in [0,15]).
    std::vector<uint8_t> nibbles((size_t)db.n * m, 0);
    const uint32_t cs = q.code_size();  // = m/2 at bits=4 (packed)
    std::vector<uint8_t> packed(cs, 0);
    for (uint32_t i = 0; i < db.n; i++) {
        q.encode(db.data.data() + size_t(i) * db.dim, packed.data());
        // Unpack: segment s code = (packed[s/2] >> ((s%2)*4)) & 0xF.
        for (uint32_t s = 0; s < m; s++) {
            nibbles[(size_t)i * m + s] =
                (uint8_t)((packed[s / 2] >> ((s % 2) * 4)) & 0xF);
        }
    }

    // Transpose into segment-major blocks. The kernel extracts lanes as:
    //   out[0..15]   = low  nibbles of bytes 0..15  → vectors 0..15
    //   out[16..31]  = high nibbles of bytes 0..15  → vectors 16..31
    // So byte k of segment s holds: low nibble = vector (b*32 + k),
    // high nibble = vector (b*32 + 16 + k). (NOT 2k/2k+1 — that would
    // deinterleave the output order.)
    for (uint32_t b = 0; b < n_blocks; b++) {
        for (uint32_t s = 0; s < m; s++) {
            for (uint32_t k = 0; k < 16; k++) {
                const uint32_t v0 = b * 32 + k;          // low nibble
                const uint32_t v1 = b * 32 + 16 + k;     // high nibble
                uint8_t lo = (v0 < db.n) ? nibbles[(size_t)v0 * m + s] : 0;
                uint8_t hi = (v1 < db.n) ? nibbles[(size_t)v1 * m + s] : 0;
                blocks[((size_t)b * m + s) * 16 + k] = (uint8_t)((hi << 4) | lo);
            }
        }
    }
    return blocks;
}

int main(int argc, char** argv) {
    const std::string base_path = (argc > 1) ? argv[1] : "datasets/arxiv100k_base.fbin";
    const std::string query_path = (argc > 2) ? argv[2] : "datasets/arxiv100k_query.fbin";
    const std::string gt_path = (argc > 3) ? argv[3] : "datasets/arxiv100k_gt.gtmm";

    FbinData db, queries;
    GroundTruth gt;
    if (!read_fbin(base_path, db)) { printf("cannot read %s\n", base_path.c_str()); return 1; }
    if (!read_fbin(query_path, queries)) { printf("cannot read %s\n", query_path.c_str()); return 1; }
    if (!read_gt(gt_path, gt)) { printf("cannot read %s\n", gt_path.c_str()); return 1; }
    printf("Loaded: db %u×%u, queries %u, gt %u×%u\n",
           db.n, db.dim, queries.n, gt.n, gt.k);

    // Train 4-bit PQ codebook. m=192 to keep code size = 96 B (= current 8-bit).
    const uint32_t m4 = 192;
    PqQuantizer q4(MetricKind::L2Sq, db.dim, m4, /*bits=*/4, /*seed=*/42);
    // Train on a 20k sample (PQ doesn't need full data).
    const uint32_t train_n = std::min<uint32_t>(20000, db.n);
    printf("Training 4-bit PQ (m=%u, K=16) on %u samples...\n", m4, train_n);
    auto t0 = std::chrono::steady_clock::now();
    q4.train(db.data.data(), train_n);
    auto t1 = std::chrono::steady_clock::now();
    printf("  trained in %.2fs\n", std::chrono::duration<double>(t1 - t0).count());

    // Encode the whole DB at 4-bit, in block layout.
    printf("Encoding %u vectors at 4-bit...\n", db.n);
    t0 = std::chrono::steady_clock::now();
    auto blocks = encode_dataset_4bit(q4, db);
    t1 = std::chrono::steady_clock::now();
    printf("  encoded in %.2fs (%.1f MB)\n",
           std::chrono::duration<double>(t1 - t0).count(),
           blocks.size() / 1e6);

    const uint32_t n_blocks = (db.n + 31) / 32;

    // Sweep rerank depths W.
    printf("\n%-6s %12s %12s %12s\n", "W", "recall@10", "QPS", "avg scan");
    printf("-----------------------------------------------\n");

    // DEBUG: probe one query — what does the 4-bit scan return as top-10?
    {
        const float* query = queries.data.data();
        std::vector<uint8_t> lut4((size_t)m4 * 16);
        build_lut4(q4, query, lut4.data());
        std::vector<uint32_t> pq_dists(db.n);
        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t out[32];
            pq4_block32(blocks.data() + (size_t)b * m4 * 16, lut4.data(), m4, out);
            for (uint32_t j = 0; j < 32; j++) {
                const uint32_t v = b * 32 + j;
                if (v < db.n) pq_dists[v] = out[j];
            }
        }
        // Compare: scan-distance vs reference (pq4.lut_distance over unpacked).
        std::vector<uint8_t> packed(q4.code_size());
        std::vector<float> ref_lut(q4.lut_size());
        q4.preprocess_query(query, ref_lut.data());
        for (uint32_t i = 0; i < std::min<uint32_t>(20, db.n); i++) {
            q4.encode(db.data.data() + size_t(i)*db.dim, packed.data());
            const float ref_d = q4.lut_distance(packed.data(), ref_lut.data());
            // The scan returns an int that's the SUM of (entry-min_s)*A — not
            // directly comparable to ref_d (which is the float LUT). Just print.
            if (i < 5) printf("  v=%u  scan_dist=%u  ref_float=%.3f\n",
                              i, pq_dists[i], ref_d);
            (void)ref_d;
        }
        // Top-10 by scan distance.
        std::vector<uint32_t> idx(db.n);
        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin()+10, idx.end(),
                         [&](uint32_t a, uint32_t b){return pq_dists[a]<pq_dists[b];});
        std::sort(idx.begin(), idx.begin()+10,
                  [&](uint32_t a, uint32_t b){return pq_dists[a]<pq_dists[b];});
        printf("  scan top-10: ");
        for (int i = 0; i < 10; i++) printf("%u ", idx[i]);
        printf("\n  GT top-10:   ");
        for (int i = 0; i < 10; i++) printf("%u ", gt.ids[i]);
        printf("\n  LUT scale A=%.4f\n", g_lut4_scale);
    }
    printf("\n");

    for (uint32_t W : {10u, 30u, 100u, 300u, 1000u, 3000u}) {
        // For each query: brute-force 4-bit scan, take top-W by 4-bit distance,
        // rerank with exact FP32 L2, take top-10, measure recall@10 vs GT.
        std::vector<uint8_t> lut4((size_t)m4 * 16);
        std::vector<uint32_t> pq_dists(db.n);
        std::vector<float> exact_dists(W);

        uint64_t total_hits = 0;
        t0 = std::chrono::steady_clock::now();
        for (uint32_t qi = 0; qi < queries.n; qi++) {
            const float* query = queries.data.data() + size_t(qi) * db.dim;
            build_lut4(q4, query, lut4.data());

            // 4-bit scan: process all blocks, write 32 distances per block.
            for (uint32_t b = 0; b < n_blocks; b++) {
                uint32_t out[32];
                pq4_block32(blocks.data() + (size_t)b * m4 * 16,
                            lut4.data(), m4, out);
                for (uint32_t j = 0; j < 32; j++) {
                    const uint32_t v = b * 32 + j;
                    if (v < db.n) pq_dists[v] = out[j];
                }
            }

            // Take top-W by pq_dist (smaller = closer for L2).
            // Simple partial sort: nth_element then sort the first W.
            std::vector<uint32_t> idx(db.n);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + W, idx.end(),
                             [&](uint32_t a, uint32_t b) { return pq_dists[a] < pq_dists[b]; });
            std::sort(idx.begin(), idx.begin() + W,
                      [&](uint32_t a, uint32_t b) { return pq_dists[a] < pq_dists[b]; });

            // Exact FP32 rerank of top-W.
            for (uint32_t i = 0; i < W; i++) {
                const float* v = db.data.data() + size_t(idx[i]) * db.dim;
                float d = 0;
                for (uint32_t k = 0; k < db.dim; k++) {
                    const float diff = v[k] - query[k];
                    d += diff * diff;
                }
                exact_dists[i] = d;
            }
            // Top-10 by exact distance.
            std::vector<uint32_t> top_idx(W);
            std::iota(top_idx.begin(), top_idx.end(), 0);
            std::partial_sort_copy(top_idx.begin(), top_idx.end(),
                                   top_idx.begin(), top_idx.begin() + std::min<uint32_t>(10, W),
                                   [&](uint32_t a, uint32_t b) { return exact_dists[a] < exact_dists[b]; });

            // Recall@10 vs GT.
            const uint32_t* gt_row = gt.ids.data() + size_t(qi) * gt.k;
            for (uint32_t r = 0; r < std::min<uint32_t>(10, W); r++) {
                const uint32_t found = idx[top_idx[r]];
                for (uint32_t g = 0; g < 10; g++) {
                    if (gt_row[g] == found) { total_hits++; break; }
                }
            }
        }
        t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const float recall = (float)total_hits / (float(queries.n) * 10.0f);
        const float qps = queries.n / secs;
        printf("%-6u %12.4f %12.1f %12.3f M\n",
               W, recall, qps,
               float(db.n) / (qps * 1e6f));
    }

    return 0;
}
