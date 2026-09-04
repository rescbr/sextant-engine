#pragma once

/// @file coder_util.hpp
/// Shared leaf-coder kernels moved verbatim from ivf_tree_index.cpp:
/// FastScan block (de)interleave, heap-min helpers, and the f32 Lloyd
/// split loop used by the non-global-PQ families. Bit-identical commit 1.

#include "../leaf_coder.hpp"
#include "../tree_nodes.hpp"
#include "simd_kernels.hpp"
#include "util/fp16.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sextant::tree::coders {

/// Extract a standard packed PQ code from a leaf's FastScan block layout
/// (moved verbatim from ivf_tree_index.cpp).
inline void extract_code_from_leaf(const uint8_t* leaf_ptr, uint32_t local_idx,
                                   uint32_t summary_size, uint16_t m4,
                                   uint8_t pq_bits, uint32_t codes_per_block,
                                   uint32_t block_bytes, uint8_t* out,
                                   uint64_t codes_base_offset = 0) {
    const uint32_t block = local_idx / codes_per_block;
    const uint32_t slot  = local_idx % codes_per_block;
    const uint64_t base_off = (codes_base_offset > 0)
        ? codes_base_offset : leaf_codes_offset(summary_size);
    const uint8_t* codes_base = leaf_ptr + base_off;
    const uint8_t* blk = codes_base + static_cast<uint64_t>(block) * block_bytes;

    if (pq_bits == 8) {
        // 16 vectors/block; segment s code at blk[s * 16 + slot].
        const uint32_t byte_idx = slot;
        for (uint16_t s = 0; s < m4; ++s) out[s] = blk[s * 16 + byte_idx];
    } else {
        // 32 vectors/block; slot < 16 uses lo nibble, slot >= 16 uses hi.
        const uint32_t byte_idx = slot % 16;
        const bool is_hi = (slot >= 16);
        for (uint16_t s = 0; s < m4; ++s) {
            const uint8_t nib = is_hi
                ? static_cast<uint8_t>(blk[s * 16 + byte_idx] >> 4)
                : static_cast<uint8_t>(blk[s * 16 + byte_idx] & 0x0F);
            const uint32_t byte_off = s / 2;
            const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
            out[byte_off] = static_cast<uint8_t>(
                ((s % 2 == 0) ? (out[byte_off] & 0xF0u) : (out[byte_off] & 0x0Fu))
                | (nib << shift));
        }
    }
}

/// Interleave one compact code into a FastScan block at lane `j` (the
/// inverse of extract_code_from_leaf; the loop body of the old flush /
/// split / insert block writers).
inline void write_code_to_fs_block(uint8_t* blk, uint32_t j, uint16_t m4,
                                   uint8_t pq_bits, uint32_t code_size,
                                   const uint8_t* code) {
    if (pq_bits == 4) {
        const uint8_t nbi = j % 16;
        const bool hi = (j >= 16);
        for (uint16_t s = 0; s < m4; ++s) {
            uint8_t nib = (s / 2 < code_size)
                ? ((s % 2 == 0) ? (code[s / 2] & 0x0F) : (code[s / 2] >> 4)) : 0;
            if (hi) blk[s * 16 + nbi] |= static_cast<uint8_t>(nib << 4);
            else    blk[s * 16 + nbi] |= nib;
        }
    } else {
        for (uint16_t s = 0; s < m4; ++s)
            blk[s * 16 + j] = (s < code_size) ? code[s] : 0;
    }
}

/// Overwrite lanes whose valid_mask bit is 0 with the 0xFFFFFFFF sentinel.
inline void u32_mask_sentinel32(uint32_t* v, uint32_t valid_mask) {
#if defined(__aarch64__)
    for (int r = 0; r < 4; ++r) {
        const int32x4_t sh = {-(4 * r), -(4 * r + 1), -(4 * r + 2),
                              -(4 * r + 3)};
        const uint32x4_t bit = vandq_u32(
            vshlq_u32(vdupq_n_u32(valid_mask), sh), vdupq_n_u32(1));
        const uint32x4_t invalid = vceqq_u32(bit, vdupq_n_u32(0));
        vst1q_u32(v + 4 * r,
                  vorrq_u32(vld1q_u32(v + 4 * r), invalid));
    }
#else
    for (uint32_t j = 0; j < 32; ++j)
        if (!((valid_mask >> j) & 1u)) v[j] = 0xFFFFFFFFu;
#endif
}

inline uint32_t u32_min32(const uint32_t* v) {
#if defined(__aarch64__)
    const uint32x4_t a = vminq_u32(vld1q_u32(v),      vld1q_u32(v + 4));
    const uint32x4_t b = vminq_u32(vld1q_u32(v + 8),  vld1q_u32(v + 12));
    const uint32x4_t c = vminq_u32(vld1q_u32(v + 16), vld1q_u32(v + 20));
    const uint32x4_t d = vminq_u32(vld1q_u32(v + 24), vld1q_u32(v + 28));
    return vminvq_u32(vminq_u32(vminq_u32(a, b), vminq_u32(c, d)));
#else
    uint32_t m = 0xFFFFFFFFu;
    for (uint32_t j = 0; j < 32; ++j) m = std::min(m, v[j]);
    return m;
#endif
}

inline uint32_t u32_min16(const uint32_t* v) {
#if defined(__aarch64__)
    const uint32x4_t a = vminq_u32(vld1q_u32(v),     vld1q_u32(v + 4));
    const uint32x4_t b = vminq_u32(vld1q_u32(v + 8), vld1q_u32(v + 12));
    return vminvq_u32(vminq_u32(a, b));
#else
    uint32_t m = 0xFFFFFFFFu;
    for (uint32_t j = 0; j < 16; ++j) m = std::min(m, v[j]);
    return m;
#endif
}

/// The f32 Lloyd (K=2) split loop shared by local_pq / scalar_lm /
/// local_scalar (moved verbatim from split_leaf_). Produces the two groups
/// and the two FP16 centroids.
inline void lloyd_split_f32(const float* vecs, uint32_t count, uint16_t dim,
                            std::vector<uint32_t>& group0,
                            std::vector<uint32_t>& group1,
                            std::vector<float16_t>& cent0_fp16,
                            std::vector<float16_t>& cent1_fp16) {
    // Seeds: first vector, then the farthest vector from it.
    auto l2 = [&](uint32_t a, uint32_t b) {
        float s = 0.f;
        for (uint16_t d = 0; d < dim; ++d) {
            const float diff = vecs[static_cast<size_t>(a) * dim + d] -
                               vecs[static_cast<size_t>(b) * dim + d];
            s += diff * diff;
        }
        return s;
    };
    uint32_t s1 = 0;
    float best = -1.f;
    for (uint32_t i = 1; i < count; ++i) {
        const float d = l2(0, i);
        if (d > best) { best = d; s1 = i; }
    }
    std::vector<float> c0(vecs, vecs + dim);
    std::vector<float> c1(vecs + static_cast<size_t>(s1) * dim,
                          vecs + static_cast<size_t>(s1 + 1) * dim);
    std::vector<uint8_t> assign(count, 0);
    for (int iter = 0; iter < 10; ++iter) {
        bool changed = false;
        for (uint32_t i = 0; i < count; ++i) {
            const float* v = vecs + static_cast<size_t>(i) * dim;
            float d0 = 0.f, d1 = 0.f;
            for (uint16_t d = 0; d < dim; ++d) {
                const float e0 = v[d] - c0[d];
                const float e1 = v[d] - c1[d];
                d0 += e0 * e0;
                d1 += e1 * e1;
            }
            const uint8_t a = (d0 <= d1) ? 0 : 1;
            if (a != assign[i]) { assign[i] = a; changed = true; }
        }
        // Recompute centroids.
        std::fill(c0.begin(), c0.end(), 0.f);
        std::fill(c1.begin(), c1.end(), 0.f);
        uint32_t n0 = 0, n1 = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const float* v = vecs + static_cast<size_t>(i) * dim;
            auto& c = assign[i] ? c1 : c0;
            (assign[i] ? n1 : n0)++;
            for (uint16_t d = 0; d < dim; ++d) c[d] += v[d];
        }
        if (n0 > 0) for (uint16_t d = 0; d < dim; ++d) c0[d] /= static_cast<float>(n0);
        if (n1 > 0) for (uint16_t d = 0; d < dim; ++d) c1[d] /= static_cast<float>(n1);
        if (!changed && iter > 0) break;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (assign[i] == 0) group0.push_back(i);
        else group1.push_back(i);
    }
    cent0_fp16.resize(dim);
    cent1_fp16.resize(dim);
    cast_fp32_to_fp16(c0.data(), cent0_fp16.data(), dim);
    cast_fp32_to_fp16(c1.data(), cent1_fp16.data(), dim);
}

}  // namespace sextant::tree::coders

// ===========================================================================
// Scalar block-scan kernels (scalar_lm / local_scalar).
//
// Layout: FastScan PQ4 blocks with m4 = dim (one 4-bit nibble per dim),
// 32 vectors per block, block layout [dim][16]: byte (d*16 + j%16) holds
// dim-d nibbles of vectors j<16 (lo nibble) and j>=16 (hi nibble). One
// vld1q_u8 per (block, dim) feeds 32 vectors — 16x fewer nibble-unpack
// operations than the old 4-vector-at-a-time flat kernels.
//
// The kernels are separate functions ON PURPOSE (same reason as commit 1):
// one body holding all modes blew up register pressure.
// ===========================================================================

namespace sextant::tree::coders {

struct ScalarScanCtx {
    // Leaf state
    const uint8_t* codes = nullptr;   // FastScan block base (dim*16 B/block)
    uint32_t count = 0;
    uint32_t block_bytes = 0;         // dim * 16
    const float16_t* ip_bias = nullptr;  // count entries or null
    // Query transform (arith / i8 kernels)
    const float* a_uni = nullptr;     // dim entries (kernels loop d < dim)
    float c0 = 0.f;
    const float* query = nullptr;     // raw query (LM gather kernel)
    uint16_t dim = 0;
    // Kernel selection. i8_mode > 0 selects the LUT FastScan kernel
    // (1 = single LUT, 2 = dual hi/lo LUT); mode 0 = block-arith f32.
    int i8_mode = 0;                  // 0/1/2
    // LUT FastScan operands (mode >= 1). lut rows are [dim][16], the exact
    // layout simd::pq4_block32 consumes (m4 = dim, one 4-bit code per dim).
    const uint8_t* lut_hi = nullptr;  // dim*16 bytes
    const uint8_t* lut_lo = nullptr;  // dim*16 bytes (mode 2 only)
    float lut_inv_ah = 0.f;           // dots = acc_hi*inv_ah
    float lut_inv_al = 0.f;           //     + (acc_lo - lut_lo_off)*inv_al
    float lut_lo_off = 0.f;           //     + lut_b
    float lut_b = 0.f;
    bool slm_arith = true;
    bool slm_shaped = false;
    // Lloyd-Max gather kernel inputs
    const float* levels = nullptr;    // row-major K/dim
    uint32_t K = 16;
    // Shared-shape tables
    const uint8_t* shape_u8 = nullptr;  // NEON TBL (16 bytes)
    const float* shape_f32 = nullptr;   // scalar fallback
};

namespace detail {

/// Heap-push one block's 32 scores. Padding lanes beyond `count` never
/// enter the heap (their code nibbles are zero, but they are simply not
/// pushed).
inline void scalar_block_heap_push(RawScanHeap& heap, const ScalarScanCtx& c,
                                   uint32_t base, const float* dots) {
    const uint32_t nv = std::min(32u, c.count - base);
    for (uint32_t j = 0; j < nv; ++j) {
        const float score = dots[j] + c.c0;
        const float dist = c.ip_bias
            ? -(score * static_cast<float>(c.ip_bias[base + j]))
            : -score;
        const uint32_t pq_dist = f32_to_dist_key(dist);
        if (!heap_full(heap)) heap_push(heap, pq_dist, base + j);
        else if (pq_dist < heap_front(heap))
            heap_replace_top(heap, pq_dist, base + j);
    }
}

}  // namespace detail

// ===========================================================================
// LUT FastScan scan (the codebase's proven pattern).
//
// The scalar block layout (m4 = dim, one 4-bit nibble per dim) is
// structurally identical to a PQ4 FastScan layout with dim subquantizers,
// so the scan is a per-(query|leaf) dim x 16 byte LUT + simd::pq4_block32:
//
//   term(d,n) = a_d * f(n)      a_d = q_d*step_d, f = nibble value
//                               (identity, or the LM shape table)
//   LUT_hi[d][n] = round((term - min_d(term)) * A_h),  A_h = E / T
//   resid        = (term - min_d) - LUT_hi/A_h,       |resid| <= 0.5/A_h
//   LUT_lo[d][n] = round(resid * A_l) + E/2,           A_l = E * A_h
//   dots = acc_hi/A_h + (acc_lo - dim*E/2)/A_l + sum_d(min_d)
//
// E = 65535/dim caps each LUT entry so pq4_block32's internal u16
// accumulators (per-lane max = dim * E) cannot overflow at any dim. The
// +E/2 offset keeps lo entries unsigned; it is subtracted once per vector.
// The per-dim min sum (lut_b) is NOT droppable: the per-vector ip_bias
// multiplies the score, so a constant offset shifts IPs with unequal bias.
//
// Dual (mode 2) leaves ~0.5/A_l per-dim rounding noise (~0.4*T worst-case
// over dim=1536 dims); single (mode 1) leaves ~0.5/A_h (~18*T worst-case) —
// the hi/lo split mirrors the flat dual-SDOT decomposition.
// ===========================================================================

struct ScalarLut {
    std::vector<uint8_t> hi, lo;  // dim*16 each (lo empty unless dual)
    float inv_ah = 0.f;
    float inv_al = 0.f;
    float lo_off = 0.f;  // dim * (E/2), subtracted from the lo accumulator
    float b = 0.f;       // sum of per-dim minima (added back exactly)
    bool dual = false;
};


/// SIMD LUT builder (NEON). Validated against the scalar reference
/// (deleted afterwards): hi tables BIT-IDENTICAL (same op order — mul+add,
/// never fused FMA, floor(x+0.5) via vrndm — and the same constants); lo
/// entries may flip by ±1 when the residual lands exactly on a rounding
/// boundary (a compiler-contraction last-bit difference; one lo unit is
/// 1/(E·A_h) of a hi unit, i.e. score noise 3 orders below hi resolution);
/// b may differ by 1 ulp (lane-tree summation order — runtime-only
/// constant, never persisted). `a` must be zero-padded to a multiple of 4
/// (both coders already pad a_uni to the 16-dim kernel width); padding
/// lanes contribute mn = span = 0 and are inert. Full batches only —
/// no scalar tail.
inline bool scalar_build_lut(const float* a, uint32_t dim, const float* shape,
                             bool dual, ScalarLut& out) {
    if (dim == 0 || dim > 65535) return false;
    const uint32_t E = std::min(255u, 65535u / dim);
    const float half = static_cast<float>(E) * 0.5f;
    out.hi.assign(static_cast<size_t>(dim) * 16, 0);
    out.dual = dual;
    if (dual) out.lo.assign(static_cast<size_t>(dim) * 16, 0);

    // Shape extrema (closed form): per-dim min/max term = a_d·s_min/s_max
    // for a_d >= 0, swapped for a_d < 0. Null shape = identity f(n) = n.
    float s_min = 0.f, s_max = 15.f;
    if (shape) {
        s_min = shape[0];
        s_max = shape[0];
        for (uint32_t n = 1; n < 16; ++n) {
            s_min = std::min(s_min, shape[n]);
            s_max = std::max(s_max, shape[n]);
        }
    }
    [[maybe_unused]] const float span_k = s_max - s_min;

#if defined(__aarch64__)
    // ---- Pass 1 (batches of 4 dims; padded lanes are inert):
    // B = sum of per-dim minima, T = max per-dim span.
    float T = 0.f, B = 0.f;
    {
        const uint32_t pd = (dim + 3) & ~3u;
        float32x4_t acc_b = vdupq_n_f32(0.f);
        float32x4_t acc_t = vdupq_n_f32(0.f);
        for (uint32_t d = 0; d < pd; d += 4) {
            const float32x4_t ad = vld1q_f32(a + d);
            const float32x4_t lo_t = vmulq_n_f32(ad, s_min);
            const float32x4_t hi_t = vmulq_n_f32(ad, s_max);
            const float32x4_t mn = vbslq_f32(vcltzq_f32(ad), hi_t, lo_t);
            // span computed as (ad*s_max) - (ad*s_min) with the sign
            // folded into the subtract order — same two products and one
            // subtract as the reference, so T (and thus A_h) matches
            // bit-for-bit; the closed form |ad|*(s_max-s_min) differs by
            // 1 ulp and shifted every downstream constant.
            const float32x4_t span = vabsq_f32(
                vsubq_f32(vmulq_n_f32(ad, s_min), vmulq_n_f32(ad, s_max)));
            acc_b = vaddq_f32(acc_b, mn);
            acc_t = vmaxq_f32(acc_t, span);
        }
        B = vaddvq_f32(acc_b);
        T = vmaxvq_f32(acc_t);
    }
    out.b = B;
    if (T <= 0.f) {
        out.inv_ah = out.inv_al = 0.f;
        out.lo_off = 0.f;
        return true;
    }
    const float A_h = static_cast<float>(E) / T;
    const float inv_ah_h = 1.0f / A_h;
    const float A_l = static_cast<float>(E) * A_h;
    out.inv_ah = 1.0f / A_h;
    out.inv_al = dual ? 1.0f / A_l : 0.f;
    out.lo_off = dual ? static_cast<float>(dim) * half : 0.f;

    // ---- Pass 2: 16 entries per dim — four f32x4 quadrants per row,
    // narrowed to one u8x16 store per (row, LUT).
    {
        const float32x4_t vAh = vdupq_n_f32(A_h);
        const float32x4_t vHlf = vdupq_n_f32(half);
        const float32x4_t vC05 = vdupq_n_f32(0.5f);
        const float32x4_t vInvAh = vdupq_n_f32(inv_ah_h);
        const float32x4_t vAl = vdupq_n_f32(A_l);
        alignas(16) static const float nid[16] = {0,1,2,3,4,5,6,7,
                                                  8,9,10,11,12,13,14,15};
        for (uint32_t d = 0; d < dim; ++d) {
            const float ad = a[d];
            const float mn = ad < 0.f ? ad * s_max : ad * s_min;
            uint8_t* hrow = out.hi.data() + static_cast<size_t>(d) * 16;
            uint8_t* lrow = dual ? out.lo.data() + static_cast<size_t>(d) * 16
                                 : nullptr;
            const float32x4_t vAd = vdupq_n_f32(ad);
            const float32x4_t vMn = vdupq_n_f32(mn);
            int32x4_t hq[4];
            int32x4_t lq[4];
            for (uint32_t g = 0; g < 4; ++g) {
                const float32x4_t nf = shape
                    ? vld1q_f32(shape + g * 4)
                    : vld1q_f32(nid + g * 4);
                // v = ad*f(n) - mn ; h = floor(v*Ah + 0.5)
                const float32x4_t v = vsubq_f32(vmulq_f32(vAd, nf), vMn);
                const float32x4_t hf = vrndmq_f32(
                    vaddq_f32(vmulq_f32(v, vAh), vC05));
                hq[g] = vcvtq_s32_f32(hf);
                if (dual) {
                    // l = floor((v - h*invAh)*Al + half + 0.5)
                    const float32x4_t r0 = vsubq_f32(v,
                        vmulq_f32(hf, vInvAh));
                    const float32x4_t r2 = vaddq_f32(vaddq_f32(
                        vmulq_f32(r0, vAl), vHlf), vC05);
                    lq[g] = vcvtq_s32_f32(vrndmq_f32(r2));

                }
            }
            const uint16x8_t h01 = vcombine_u16(vqmovun_s32(hq[0]),
                                                vqmovun_s32(hq[1]));
            const uint16x8_t h23 = vcombine_u16(vqmovun_s32(hq[2]),
                                                vqmovun_s32(hq[3]));
            vst1q_u8(hrow, vcombine_u8(vqmovn_u16(h01),
                                       vqmovn_u16(h23)));
            if (dual) {
                const uint16x8_t l01 = vcombine_u16(vqmovun_s32(lq[0]),
                                                    vqmovun_s32(lq[1]));
                const uint16x8_t l23 = vcombine_u16(vqmovun_s32(lq[2]),
                                                    vqmovun_s32(lq[3]));
                vst1q_u8(lrow, vcombine_u8(vqmovn_u16(l01),
                                           vqmovn_u16(l23)));
            }
        }
    }
    return true;
#else
    // Portable fallback for non-NEON builds (same algorithm).
    float T = 0.f, B = 0.f;
    for (uint32_t d = 0; d < dim; ++d) {
        const float ad = a[d];
        const float mn = ad < 0.f ? ad * s_max : ad * s_min;
        B += mn;
        const float span = std::fabs(ad) * span_k;
        if (span > T) T = span;
    }
    out.b = B;
    if (T <= 0.f) {
        out.inv_ah = out.inv_al = 0.f;
        out.lo_off = 0.f;
        return true;
    }
    const float A_h = static_cast<float>(E) / T;
    const float A_l = static_cast<float>(E) * A_h;
    out.inv_ah = 1.0f / A_h;
    out.inv_al = dual ? 1.0f / A_l : 0.f;
    out.lo_off = dual ? static_cast<float>(dim) * half : 0.f;
    for (uint32_t d = 0; d < dim; ++d) {
        const float ad = a[d];
        const float mn = ad < 0.f ? ad * s_max : ad * s_min;
        uint8_t* hrow = out.hi.data() + static_cast<size_t>(d) * 16;
        uint8_t* lrow = dual ? out.lo.data() + static_cast<size_t>(d) * 16
                             : nullptr;
        for (uint32_t n = 0; n < 16; ++n) {
            const float t = ad * (shape ? shape[n]
                                        : static_cast<float>(n));
            const float v = t - mn;
            const float h = std::floor(v * A_h + 0.5f);
            hrow[n] = static_cast<uint8_t>(h);
            if (dual)
                lrow[n] = static_cast<uint8_t>(std::floor(
                    (v - h * (1.0f / A_h)) * A_l + half + 0.5f));
        }
    }
    return true;
#endif
}

/// LUT FastScan kernel: one simd::pq4_block32 per (block, LUT) — 2 table
/// lookups per (dim, vector) with u8 entries (dual: two 32-lane passes).
inline void scalar_scan_block_lut(const ScalarScanCtx& c,
                                  RawScanHeap& heap) {
    const uint32_t dim = c.dim;
    const uint32_t n_blocks = (c.count + 31) / 32;
    const uint32_t bb = c.block_bytes;
    const bool dual = c.i8_mode >= 2 && c.lut_lo;
    alignas(16) float dots[32];
    uint32_t acc_h[32], acc_l[32];
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = c.codes + static_cast<uint64_t>(b) * bb;
        simd::pq4_block32(blk, c.lut_hi, dim, acc_h);
        if (dual) {
            simd::pq4_block32(blk, c.lut_lo, dim, acc_l);
            for (uint32_t j = 0; j < 32; ++j)
                dots[j] = static_cast<float>(acc_h[j]) * c.lut_inv_ah +
                          (static_cast<float>(acc_l[j]) - c.lut_lo_off) *
                              c.lut_inv_al +
                          c.lut_b;
        } else {
            for (uint32_t j = 0; j < 32; ++j)
                dots[j] =
                    static_cast<float>(acc_h[j]) * c.lut_inv_ah + c.lut_b;
        }
        detail::scalar_block_heap_push(heap, c, b * 32, dots);
    }
}

/// BLOCK-ARITH f32: per dim, one 16-byte load unpacks the dim-d nibbles of
/// all 32 vectors (lo nibble = vectors 0..15, hi = 16..31). Nibbles widen
/// u8->u16->u32->f32 and FMLA against a broadcast a_uni[d]. Accumulators
/// are 8 x float32x4 (4 even + 4 odd quads, lane = vector) — comfortably
/// inside the register budget, no spills.
inline void scalar_scan_block_arith(const ScalarScanCtx& c,
                                    RawScanHeap& heap) {
    const uint32_t dim = c.dim;
    const uint32_t n_blocks = (c.count + 31) / 32;
    const uint8_t* codes = c.codes;
    const uint32_t bb = c.block_bytes;
    const float* a_uni = c.a_uni;
    alignas(16) float dots[32];
#if defined(__aarch64__)
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    const bool shaped = c.slm_shaped;
    const uint8x16_t f_tbl = shaped ? vld1q_u8(c.shape_u8) : vdupq_n_u8(0);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = codes + static_cast<uint64_t>(b) * bb;
        float32x4_t ae[4], ao[4];
        for (int r = 0; r < 4; ++r) {
            ae[r] = vdupq_n_f32(0.f);
            ao[r] = vdupq_n_f32(0.f);
        }
        for (uint32_t d = 0; d < dim; ++d) {
            const uint8x16_t cv = vld1q_u8(blk + d * 16);
            const uint8x16_t clo = vandq_u8(cv, mask4);
            const uint8x16_t chi = vshrq_n_u8(cv, 4);
            const uint8x16_t ne = shaped ? vqtbl1q_u8(f_tbl, clo) : clo;
            const uint8x16_t no = shaped ? vqtbl1q_u8(f_tbl, chi) : chi;
            const float32x4_t a = vdupq_n_f32(a_uni[d]);
            const uint16x8_t e0 = vmovl_u8(vget_low_u8(ne));
            const uint16x8_t e1 = vmovl_u8(vget_high_u8(ne));
            const uint16x8_t o0 = vmovl_u8(vget_low_u8(no));
            const uint16x8_t o1 = vmovl_u8(vget_high_u8(no));
            ae[0] = vfmaq_f32(ae[0], a,
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(e0))));
            ae[1] = vfmaq_f32(ae[1], a,
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(e0))));
            ae[2] = vfmaq_f32(ae[2], a,
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(e1))));
            ae[3] = vfmaq_f32(ae[3], a,
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(e1))));
            ao[0] = vfmaq_f32(ao[0], a,
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(o0))));
            ao[1] = vfmaq_f32(ao[1], a,
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(o0))));
            ao[2] = vfmaq_f32(ao[2], a,
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(o1))));
            ao[3] = vfmaq_f32(ao[3], a,
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(o1))));
        }
        for (int r = 0; r < 4; ++r) {
            vst1q_f32(dots + 4 * r, ae[r]);
            vst1q_f32(dots + 16 + 4 * r, ao[r]);
        }
        detail::scalar_block_heap_push(heap, c, b * 32, dots);
    }
#else
    // Portable fallback: dim-major scalar walk over the same blocks.
    const float* ftbl = c.shape_f32;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = codes + static_cast<uint64_t>(b) * bb;
        std::fill(dots, dots + 32, 0.f);
        for (uint32_t d = 0; d < dim; ++d) {
            const uint8_t* row = blk + d * 16;
            const float a = a_uni[d];
            for (uint32_t j = 0; j < 16; ++j) {
                const uint8_t byte = row[j];
                dots[j] += a * (c.slm_shaped ? ftbl[byte & 0xF]
                                             : static_cast<float>(byte & 0xF));
                dots[16 + j] += a *
                    (c.slm_shaped ? ftbl[byte >> 4]
                                  : static_cast<float>(byte >> 4));
            }
        }
        detail::scalar_block_heap_push(heap, c, b * 32, dots);
    }
#endif
}

/// BLOCK-GATHER (Lloyd-Max mode 0, the non-arithmetic family): rare path;
/// dim-major scalar walk over the blocks (sequential 16-byte rows, two
/// vectors per byte).
inline void scalar_scan_block_gather(const ScalarScanCtx& c,
                                     RawScanHeap& heap) {
    const uint32_t dim = c.dim;
    const uint32_t K = c.K;
    const float* q = c.query;
    const float* levels = c.levels;
    const uint32_t n_blocks = (c.count + 31) / 32;
    alignas(16) float dots[32];
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = c.codes + static_cast<uint64_t>(b) * c.block_bytes;
        std::fill(dots, dots + 32, 0.f);
        for (uint32_t d = 0; d < dim; ++d) {
            const uint8_t* row = blk + d * 16;
            const float qd = q[d];
            const float* lrow = levels + static_cast<size_t>(d) * K;
            for (uint32_t j = 0; j < 16; ++j) {
                const uint8_t byte = row[j];
                dots[j] += qd * lrow[byte & 0xF];
                dots[16 + j] += qd * lrow[byte >> 4];
            }
        }
        detail::scalar_block_heap_push(heap, c, b * 32, dots);
    }
}

/// Scan one scalar leaf (FastScan blocks, m4 = dim) into `heap`.
inline void scalar_scan_leaf(const ScalarScanCtx& c, RawScanHeap& heap) {
    if (c.i8_mode) scalar_scan_block_lut(c, heap);
    else if (c.slm_arith) scalar_scan_block_arith(c, heap);
    else scalar_scan_block_gather(c, heap);
}

/// Order-preserving u32 encoding of a float score (the old heap path's
/// dist_bits trick, factored for reuse).
inline uint32_t f32_to_dist_key(float dist) {
    uint32_t bits;
    std::memcpy(&bits, &dist, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

}  // namespace sextant::tree::coders
