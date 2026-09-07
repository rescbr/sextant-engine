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
// Scalar flat-nibble scan kernel (shared by scalar_lm / local_scalar).
// Moved verbatim from ivf_tree_index.cpp's scan_one_leaf; the family state
// that used to live in captured locals rides on ScalarScanCtx.
// ===========================================================================

namespace sextant::tree::coders {

struct ScalarScanCtx {
    // Leaf state
    const uint8_t* codes = nullptr;   // flat code rows
    uint32_t count = 0;
    uint32_t cs = 0;                  // bytes per code row
    const float16_t* ip_bias = nullptr;  // count entries or null
    // Query transform (arith kernels)
    const float* a_uni = nullptr;     // padded to 16 dims
    float c0 = 0.f;
    const float* query = nullptr;     // raw query (LM gather kernel)
    uint16_t dim = 0;
    // Kernel selection
    int i8_mode = 0;                  // 0/1/2
    const int8_t* a8 = nullptr;       // i8 operands (mode >= 1 / 2)
    const int8_t* a8_lo = nullptr;
    float i8_inv = 0.f;
    bool slm_arith = true;
    bool slm_shaped = false;
    // Lloyd-Max gather kernel inputs
    const float* levels = nullptr;    // row-major K/dim
    uint32_t K = 16;
    // Shared-shape tables
    const uint8_t* shape_u8 = nullptr;  // NEON TBL (16 bytes)
    const float* shape_f32 = nullptr;   // scalar fallback
};

/// Scan one scalar leaf into `heap` (bounded top-W max-heap). The zero
/// code row used by tail batches is a thread-local scratch (per-worker, so
/// parallel leaf scans never share mutable state).
///
/// The three kernels are separate functions ON PURPOSE: one body holding
/// all three blew up register pressure (accumulator spills around the
/// fmla/zip loop cost ~20% on the arith path).
namespace detail {

struct ScalarRowPtrs {
    const uint8_t* cp[4];
    uint32_t nv = 0;
};

inline ScalarRowPtrs scalar_rows(const ScalarScanCtx& c, uint32_t i,
                                 const uint8_t* pad_row) {
    ScalarRowPtrs r;
    r.nv = std::min(4u, c.count - i);
    for (uint32_t v = 0; v < r.nv; ++v)
        r.cp[v] = c.codes + (uint64_t)(i + v) * c.cs;
    for (uint32_t v = r.nv; v < 4; ++v) r.cp[v] = pad_row;
    return r;
}

inline void scalar_heap_push4(RawScanHeap& heap, const ScalarScanCtx& c,
                              const uint32_t i, const uint32_t nv,
                              const float* dots) {
    // Heap push — real vectors only (padding never enters).
    for (uint32_t v = 0; v < nv; ++v) {
        const float score = dots[v] + c.c0;
        float dist = c.ip_bias
            ? -(score * static_cast<float>(c.ip_bias[i + v]))
            : -score;
        const uint32_t pq_dist = f32_to_dist_key(dist);
        if (!heap_full(heap)) {
            heap_push(heap, pq_dist, i + v);
        } else if (pq_dist < heap_front(heap)) {
            heap_replace_top(heap, pq_dist, i + v);
        }
    }
}

}  // namespace detail

inline void scalar_scan_i8(const ScalarScanCtx& c, RawScanHeap& heap) {
    [[maybe_unused]] const uint32_t dim = c.dim;
    const uint32_t cs = c.cs;
    static thread_local std::vector<uint8_t> pad_row_buf;
    if (pad_row_buf.size() < cs) pad_row_buf.assign(cs, 0);
    [[maybe_unused]] const uint8_t* pad_row = pad_row_buf.data();
#if defined(__ARM_FEATURE_DOTPROD)
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded_i8 = padded / 2 <= cs;
    const uint32_t d16i = fully_padded_i8 ? padded : dim / 16 * 16;
    const int8_t* a8p = c.a8;
    const int8_t* a8lop = c.i8_mode >= 2 ? c.a8_lo : nullptr;
    const float lo_w = a8lop ? (1.0f / 127.0f) : 0.f;
    for (uint32_t i = 0; i < c.count; i += 4) {
        const auto r = detail::scalar_rows(c, i, pad_row);
        float dots[4] = {0, 0, 0, 0};
        int32x4_t acc8[4];
        int32x4_t acc8lo[4];
        for (auto& x : acc8) x = vdupq_n_s32(0);
        if (a8lop) for (auto& x : acc8lo) x = vdupq_n_s32(0);
        for (uint32_t d = 0; d < d16i; d += 16) {
            const int8x16_t a8v = vld1q_s8(a8p + d);
            const int8x16_t a8lov = a8lop ? vld1q_s8(a8lop + d) : a8v;
            for (uint32_t v = 0; v < 4; ++v) {
                const uint8x8_t b = vld1_u8(r.cp[v] + d / 2);
                const uint8x8_t lo = vand_u8(b, vdup_n_u8(0x0F));
                const uint8x8_t hi = vshr_n_u8(b, 4);
                const uint8x8x2_t z = vzip_u8(lo, hi);
                const int8x16_t c8 = vreinterpretq_s8_u8(
                    vcombine_u8(z.val[0], z.val[1]));
                acc8[v] = vdotq_s32(acc8[v], a8v, c8);
                if (a8lop)
                    acc8lo[v] = vdotq_s32(acc8lo[v], a8lov, c8);
            }
        }
        for (uint32_t v = 0; v < 4; ++v)
            dots[v] = (static_cast<float>(vaddvq_s32(acc8[v]))
                       + lo_w * static_cast<float>(vaddvq_s32(acc8lo[v])))
                      * c.i8_inv;
        if (!fully_padded_i8) {
            for (uint32_t d = d16i; d < dim; ++d) {
                const float a8d = static_cast<float>(c.a8[d])
                    + (a8lop ? static_cast<float>(c.a8_lo[d]) / 127.0f : 0.f);
                for (uint32_t v = 0; v < 4; ++v) {
                    const uint8_t byte = r.cp[v][d / 2];
                    const uint8_t nib = (d % 2 == 0)
                        ? (byte & 0xF) : ((byte >> 4) & 0xF);
                    dots[v] += a8d * nib * c.i8_inv;
                }
            }
        }
        detail::scalar_heap_push4(heap, c, i, r.nv, dots);
    }
#else
    (void)heap;
#endif
}

inline void scalar_scan_arith(const ScalarScanCtx& c, RawScanHeap& heap) {
    const uint32_t dim = c.dim;
    const uint32_t cs = c.cs;
    static thread_local std::vector<uint8_t> pad_row_buf;
    if (pad_row_buf.size() < cs) pad_row_buf.assign(cs, 0);
    const uint8_t* pad_row = pad_row_buf.data();
    [[maybe_unused]] std::vector<HeapEntry>& h = *heap.h;
    [[maybe_unused]] const uint32_t W = heap.w;
    [[maybe_unused]] const uint32_t leaf_slot = heap.leaf_slot;
    const float* a_uni = c.a_uni;
    const float16_t* ip_bias = c.ip_bias;
    const float c0 = c.c0;
    const uint32_t count = c.count;
    const uint8_t* codes = c.codes;
#if defined(__aarch64__)
    const uint8x16_t f_tbl = c.slm_shaped
        ? vld1q_u8(c.shape_u8) : vdupq_n_u8(0);
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded = padded / 2 <= cs;
    const uint32_t d16 = fully_padded ? padded : dim / 16 * 16;
    for (uint32_t i = 0; i < count; i += 4) {
        const uint32_t nv = std::min(4u, count - i);
        const uint8_t* cp[4];
        for (uint32_t v = 0; v < nv; ++v)
            cp[v] = codes + (uint64_t)(i + v) * cs;
        for (uint32_t v = nv; v < 4; ++v) cp[v] = pad_row;

        float dots[4] = {0, 0, 0, 0};
        // Arithmetic decode: dot += (q·step)·f[code] — no gather.
        // NEON: 4 vectors × 16 dims per iteration. 8 code bytes unpack
        // to 16 dim-ordered nibbles (vzip lo/hi); for the shared-shape
        // quantizer they index the 16-byte f table via one TBL, then
        // widen to f32 and FMLA against a_uni.
        float32x4_t acc[4][4];
        for (auto& row : acc)
            for (auto& x : row) x = vdupq_n_f32(0);
        for (uint32_t d = 0; d < d16; d += 16) {
            const float32x4_t a0 = vld1q_f32(&a_uni[d]);
            const float32x4_t a1 = vld1q_f32(&a_uni[d + 4]);
            const float32x4_t a2 = vld1q_f32(&a_uni[d + 8]);
            const float32x4_t a3 = vld1q_f32(&a_uni[d + 12]);
            for (uint32_t v = 0; v < 4; ++v) {
                const uint8x8_t b = vld1_u8(cp[v] + d / 2);
                const uint8x8_t lo = vand_u8(b, vdup_n_u8(0x0F));
                const uint8x8_t hi = vshr_n_u8(b, 4);
                const uint8x8x2_t z = vzip_u8(lo, hi);
                if (c.slm_shaped) {
                    const uint8x16_t fv = vqtbl1q_u8(
                        f_tbl, vcombine_u8(z.val[0], z.val[1]));
                    const uint16x8_t w0 = vmovl_u8(vget_low_u8(fv));
                    const uint16x8_t w1 = vmovl_u8(vget_high_u8(fv));
                    acc[v][0] = vfmaq_f32(acc[v][0], a0,
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0))));
                    acc[v][1] = vfmaq_f32(acc[v][1], a1,
                        vcvtq_f32_u32(vmovl_u16(vget_high_u16(w0))));
                    acc[v][2] = vfmaq_f32(acc[v][2], a2,
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1))));
                    acc[v][3] = vfmaq_f32(acc[v][3], a3,
                        vcvtq_f32_u32(vmovl_u16(vget_high_u16(w1))));
                } else {
                    const uint16x8_t w0 = vmovl_u8(z.val[0]);
                    const uint16x8_t w1 = vmovl_u8(z.val[1]);
                    acc[v][0] = vfmaq_f32(acc[v][0], a0,
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0))));
                    acc[v][1] = vfmaq_f32(acc[v][1], a1,
                        vcvtq_f32_u32(vmovl_u16(vget_high_u16(w0))));
                    acc[v][2] = vfmaq_f32(acc[v][2], a2,
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1))));
                    acc[v][3] = vfmaq_f32(acc[v][3], a3,
                        vcvtq_f32_u32(vmovl_u16(vget_high_u16(w1))));
                }
            }
        }
        for (uint32_t v = 0; v < 4; ++v) {
            dots[v] = vaddvq_f32(acc[v][0]) + vaddvq_f32(acc[v][1]) +
                      vaddvq_f32(acc[v][2]) + vaddvq_f32(acc[v][3]);
        }
        if (!fully_padded) {
            const float* ftbl = c.shape_f32;
            for (uint32_t d = d16; d < dim; ++d) {
                for (uint32_t v = 0; v < 4; ++v) {
                    const uint8_t byte = cp[v][d / 2];
                    const uint8_t nib =
                        (d % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
                    dots[v] += a_uni[d] *
                        (c.slm_shaped ? ftbl[nib] : float(nib));
                }
            }
        }
        // Heap push — real vectors only (padding never enters).
        for (uint32_t v = 0; v < nv; ++v) {
            const float score = dots[v] + c0;
            float dist = ip_bias
                ? -(score * static_cast<float>(ip_bias[i + v]))
                : -score;
            uint32_t dist_bits;
            std::memcpy(&dist_bits, &dist, sizeof(dist_bits));
            uint32_t pq_dist = (dist_bits & 0x80000000u)
                ? ~dist_bits : (dist_bits | 0x80000000u);
            if (h.size() < W) {
                h.push_back({pq_dist, leaf_slot, i + v});
                if (h.size() == W)
                    std::make_heap(h.begin(), h.end(), heap_entry_less);
            } else if (pq_dist < h[0].pq_dist) {
                heap_replace(h, pq_dist, leaf_slot, i + v);
            }
        }
    }
#else
    const uint32_t d4 = dim / 4 * 4;
    for (uint32_t i = 0; i < count; i += 4) {
        const uint32_t nv = std::min(4u, count - i);
        const uint8_t* cp[4];
        for (uint32_t v = 0; v < nv; ++v)
            cp[v] = codes + (uint64_t)(i + v) * cs;
        for (uint32_t v = nv; v < 4; ++v) cp[v] = pad_row;
        float dots[4] = {0, 0, 0, 0};
        for (uint32_t d = 0; d < d4; d += 4) {
            const float a4[4] = {a_uni[d], a_uni[d + 1],
                                 a_uni[d + 2], a_uni[d + 3]};
            const float* ftbl = c.shape_f32;
            for (uint32_t v = 0; v < nv; ++v) {
                const uint8_t b0 = cp[v][d / 2], b1 = cp[v][d / 2 + 1];
                if (c.slm_shaped) {
                    dots[v] += a4[0] * ftbl[b0 & 0xF]
                             + a4[1] * ftbl[(b0 >> 4) & 0xF]
                             + a4[2] * ftbl[b1 & 0xF]
                             + a4[3] * ftbl[(b1 >> 4) & 0xF];
                } else {
                    dots[v] += a4[0] * (b0 & 0xF)
                             + a4[1] * ((b0 >> 4) & 0xF)
                             + a4[2] * (b1 & 0xF)
                             + a4[3] * ((b1 >> 4) & 0xF);
                }
            }
        }
        for (uint32_t v = 0; v < nv; ++v) {
            const float score = dots[v] + c0;
            float dist = ip_bias
                ? -(score * static_cast<float>(ip_bias[i + v]))
                : -score;
            const uint32_t pq_dist = f32_to_dist_key(dist);
            if (!heap_full(heap)) heap_push(heap, pq_dist, i + v);
            else if (pq_dist < heap_front(heap))
                heap_replace_top(heap, pq_dist, i + v);
        }
    }
#endif
}

inline void scalar_scan_gather(const ScalarScanCtx& c, RawScanHeap& heap) {
    const uint32_t dim = c.dim;
    const uint32_t cs = c.cs;
    const uint32_t d4 = dim / 4 * 4;
    static thread_local std::vector<uint8_t> pad_row_buf;
    if (pad_row_buf.size() < cs) pad_row_buf.assign(cs, 0);
    const uint8_t* pad_row = pad_row_buf.data();
    // Lloyd-Max gather kernel: batch-4, amortized query loads.
    const float* levels = c.levels;
    const uint32_t K = c.K;
    const float* query = c.query;
    for (uint32_t i = 0; i < c.count; i += 4) {
        const auto r = detail::scalar_rows(c, i, pad_row);
        float dots[4] = {0, 0, 0, 0};
        for (uint32_t d = 0; d < d4; d += 4) {
            float q4[4] = {query[d], query[d+1], query[d+2], query[d+3]};
            for (uint32_t v = 0; v < 4; ++v) {
                uint8_t b0 = r.cp[v][d/2], b1 = r.cp[v][d/2+1];
                dots[v] += q4[0]*levels[d*K+(b0&0xF)]
                         + q4[1]*levels[(d+1)*K+((b0>>4)&0xF)]
                         + q4[2]*levels[(d+2)*K+(b1&0xF)]
                         + q4[3]*levels[(d+3)*K+((b1>>4)&0xF)];
            }
        }
        for (uint32_t d = d4; d < dim; ++d) {
            for (uint32_t v = 0; v < 4; ++v) {
                uint8_t byte = r.cp[v][d/2];
                uint8_t nib = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
                dots[v] += query[d] * levels[d*K + nib];
            }
        }
        detail::scalar_heap_push4(heap, c, i, r.nv, dots);
    }
}

inline void scalar_scan_leaf(const ScalarScanCtx& c, RawScanHeap& heap) {
    if (c.i8_mode) scalar_scan_i8(c, heap);
    else if (c.slm_arith) scalar_scan_arith(c, heap);
    else scalar_scan_gather(c, heap);
}

/// Order-preserving u32 encoding of a float score (the old heap path's
/// dist_bits trick, factored for reuse).
inline uint32_t f32_to_dist_key(float dist) {
    uint32_t bits;
    std::memcpy(&bits, &dist, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

}  // namespace sextant::tree::coders
