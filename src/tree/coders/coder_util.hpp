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
        } else if (heap_should_replace(heap, pq_dist, i + v)) {
            heap_replace_top(heap, pq_dist, i + v);
        }
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Arch-dispatched dot kernels for the scalar scan family.
//
// Each computes dots[4] for one batch of up to 4 code rows (cp[4]; a zero
// pad row is supplied for v >= nv — pad dots are computed but never pushed).
// A `_ref` scalar twin is always compiled so tests can cross-validate the
// SIMD path against it in the same binary.
//
// i8 kernel semantics (mirrors the NEON DOTPROD path): dots[v] =
// (S + w_lo * S_lo) * i8_inv where S = sum_d a8[d]*nib_d and
// S_lo = sum_d a8_lo[d]*nib_d are exact integer sums, w_lo = 1/127 in dual
// mode. AVX-512 VNNI is unsigned-only; the signed a8 x unsigned-nibble dot
// uses sum(a*c) = dpbusd(a^0x80, c) - 128*dpbusd(ones, c)  (a^0x80 == a+128
// mod 256, so the bias term removes exactly).
//
// arith kernel semantics: dots[v] = sum_d a_uni[d] * f[nib_d] where f is
// the 16-byte shared-shape table (u8) or the identity (nib as float).
// a_uni is zero-padded to the 16-dim kernel width by scan_setup.
// ---------------------------------------------------------------------------

inline void scalar_i8_dots4_ref(const ScalarScanCtx& c,
                                const uint8_t* const* cp, float dots[4]) {
    const uint32_t dim = c.dim;
    const int8_t* a8 = c.a8;
    const int8_t* a8_lo = c.i8_mode >= 2 ? c.a8_lo : nullptr;
    for (uint32_t v = 0; v < 4; ++v) {
        int64_t acc = 0, acc_lo = 0;
        for (uint32_t d = 0; d < dim; ++d) {
            const uint8_t byte = cp[v][d / 2];
            const uint32_t nib = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            acc += static_cast<int64_t>(a8[d]) * nib;
            if (a8_lo) acc_lo += static_cast<int64_t>(a8_lo[d]) * nib;
        }
        dots[v] = (static_cast<float>(acc)
                   + (a8_lo ? static_cast<float>(acc_lo) / 127.0f : 0.0f))
                  * c.i8_inv;
    }
}

inline void scalar_arith_dots4_ref(const ScalarScanCtx& c,
                                   const uint8_t* const* cp, float dots[4]) {
    const uint32_t dim = c.dim;
    const float* a_uni = c.a_uni;
    const float* ftbl = c.shape_f32;
    for (uint32_t v = 0; v < 4; ++v) {
        float acc = 0;
        for (uint32_t d = 0; d < dim; ++d) {
            const uint8_t byte = cp[v][d / 2];
            const uint32_t nib = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            acc += a_uni[d] * (c.slm_shaped ? ftbl[nib] : float(nib));
        }
        dots[v] = acc;
    }
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) \
    && defined(__AVX512BW__)
#define SEXTANT_HAS_AVX512_SCAN 1
#include <immintrin.h>
#endif

#if defined(SEXTANT_HAS_AVX512_SCAN)
/// 16 nibbles of one 16-dim chunk of row `v`, dim-ordered, as __m128i.
inline __m128i avx512_row_nibbles(const uint8_t* row, uint32_t d) {
    const __m128i b = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(row + d / 2));
    const __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4),
                                     _mm_set1_epi8(0x0F));
    // 8 code bytes hold 16 dims. Dims 0-7 come from bytes 0-3, dims 8-15
    // from bytes 4-7 (bytes 8-15 of the register are zero padding).
    const __m128i lo_hi4 = _mm_srli_si128(lo, 4);
    const __m128i hi_hi4 = _mm_srli_si128(hi, 4);
    return _mm_unpacklo_epi64(_mm_unpacklo_epi8(lo, hi),
                              _mm_unpacklo_epi8(lo_hi4, hi_hi4));
}

/// Signed 8-bit a8 (16 dims) dotted with the row's nibbles, exact, via the
/// unsigned-VNNI bias identity. Returns the 16 per-dim products as i32.
/// All 4 rows' nibbles for one 16-dim chunk, row v in 128-bit quarter v.
inline __m512i avx512_nibbles4(const uint8_t* const* cp, uint32_t d) {
    const __m128i n0 = avx512_row_nibbles(cp[0], d);
    const __m128i n1 = avx512_row_nibbles(cp[1], d);
    const __m128i n2 = avx512_row_nibbles(cp[2], d);
    const __m128i n3 = avx512_row_nibbles(cp[3], d);
    return _mm512_inserti32x4(
        _mm512_inserti32x4(
            _mm512_inserti32x4(_mm512_castsi128_si512(n0), n1, 1),
            n2, 2),
        n3, 3);
}
#endif  // AVX512

inline void scalar_i8_dots4(const ScalarScanCtx& c,
                            const uint8_t* const* cp, float dots[4]) {
#if defined(SEXTANT_HAS_AVX512_SCAN)
    const uint32_t dim = c.dim;
    const uint32_t cs = c.cs;
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded = padded / 2 <= cs;
    const uint32_t d16i = fully_padded ? padded : dim / 16 * 16;
    const int8_t* a8p = c.a8;
    const int8_t* a8lop = c.i8_mode >= 2 ? c.a8_lo : nullptr;
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i xor80 = _mm512_set1_epi8(-128);
    __m512i acc = _mm512_setzero_si512();   // quarter v: row v lane sums
    __m512i acclo = _mm512_setzero_si512();
    for (uint32_t d = 0; d < d16i; d += 16) {
        const __m512i nibs = avx512_nibbles4(cp, d);
        const __m512i corr = _mm512_slli_epi32(
            _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, nibs), 7);
        const __m128i a128 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(a8p + d));
        const __m512i biased = _mm512_xor_si512(
            _mm512_broadcast_i32x4(a128), xor80);
        acc = _mm512_add_epi32(
            acc, _mm512_sub_epi32(
                     _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                         biased, nibs),
                     corr));
        if (a8lop) {
            const __m128i alo128 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(a8lop + d));
            const __m512i lbiased = _mm512_xor_si512(
                _mm512_broadcast_i32x4(alo128), xor80);
            acclo = _mm512_add_epi32(
                acclo, _mm512_sub_epi32(
                           _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                               lbiased, nibs),
                           corr));
        }
    }
    alignas(64) int32_t lanes[16];
    alignas(64) int32_t lanes_lo[16];
    _mm512_store_si512(reinterpret_cast<__m512i*>(lanes), acc);
    if (a8lop)
        _mm512_store_si512(reinterpret_cast<__m512i*>(lanes_lo), acclo);
    for (uint32_t v = 0; v < 4; ++v) {
        int64_t s = int64_t(lanes[v * 4]) + lanes[v * 4 + 1] +
                    lanes[v * 4 + 2] + lanes[v * 4 + 3];
        float lo = 0.0f;
        if (a8lop) {
            int64_t sl = int64_t(lanes_lo[v * 4]) + lanes_lo[v * 4 + 1] +
                         lanes_lo[v * 4 + 2] + lanes_lo[v * 4 + 3];
            lo = static_cast<float>(sl) / 127.0f;
        }
        dots[v] = (static_cast<float>(s) + lo) * c.i8_inv;
    }
    if (!fully_padded) {
        for (uint32_t d = d16i; d < dim; ++d) {
            const float a8d = static_cast<float>(a8p[d]) +
                (a8lop ? static_cast<float>(a8lop[d]) / 127.0f : 0.0f);
            for (uint32_t v = 0; v < 4; ++v) {
                const uint8_t byte = cp[v][d / 2];
                const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                                 : ((byte >> 4) & 0xF);
                dots[v] += a8d * nib * c.i8_inv;
            }
        }
    }
    return;
#endif
    scalar_i8_dots4_ref(c, cp, dots);
}

inline void scalar_arith_dots4(const ScalarScanCtx& c,
                               const uint8_t* const* cp, float dots[4]) {
#if defined(SEXTANT_HAS_AVX512_SCAN)
    const uint32_t dim = c.dim;
    const uint32_t cs = c.cs;
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded = padded / 2 <= cs;
    const uint32_t d16 = fully_padded ? padded : dim / 16 * 16;
    const __m128i f_tbl = c.slm_shaped
        ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(c.shape_u8))
        : _mm_setzero_si128();
    const float* ftbl = c.shape_f32;
    __m512 acc[4];
    for (uint32_t v = 0; v < 4; ++v) acc[v] = _mm512_setzero_ps();
    for (uint32_t d = 0; d < d16; d += 16) {
        const __m512 a = _mm512_loadu_ps(c.a_uni + d);
        for (uint32_t v = 0; v < 4; ++v) {
            __m128i n = avx512_row_nibbles(cp[v], d);
            if (c.slm_shaped) n = _mm_shuffle_epi8(f_tbl, n);
            const __m512 nf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(n));
            acc[v] = _mm512_fmadd_ps(nf, a, acc[v]);
        }
    }
    for (uint32_t v = 0; v < 4; ++v) dots[v] = _mm512_reduce_add_ps(acc[v]);
    if (!fully_padded) {
        for (uint32_t d = d16; d < dim; ++d) {
            for (uint32_t v = 0; v < 4; ++v) {
                const uint8_t byte = cp[v][d / 2];
                const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                                 : ((byte >> 4) & 0xF);
                dots[v] += c.a_uni[d] *
                    (c.slm_shaped ? ftbl[nib] : float(nib));
            }
        }
    }
    return;
#endif
    scalar_arith_dots4_ref(c, cp, dots);
}

inline void scalar_scan_i8(const ScalarScanCtx& c, RawScanHeap& heap) {
    const uint32_t cs = c.cs;
    static thread_local std::vector<uint8_t> pad_row_buf;
    if (pad_row_buf.size() < cs) pad_row_buf.assign(cs, 0);
    const uint8_t* pad_row = pad_row_buf.data();
    for (uint32_t i = 0; i < c.count; i += 4) {
        const auto r = detail::scalar_rows(c, i, pad_row);
        float dots[4];
        scalar_i8_dots4(c, r.cp, dots);
        detail::scalar_heap_push4(heap, c, i, r.nv, dots);
    }
}

inline void scalar_scan_arith(const ScalarScanCtx& c, RawScanHeap& heap) {
    const uint32_t cs = c.cs;
    static thread_local std::vector<uint8_t> pad_row_buf;
    if (pad_row_buf.size() < cs) pad_row_buf.assign(cs, 0);
    const uint8_t* pad_row = pad_row_buf.data();
    for (uint32_t i = 0; i < c.count; i += 4) {
        const auto r = detail::scalar_rows(c, i, pad_row);
        float dots[4];
        scalar_arith_dots4(c, r.cp, dots);
        detail::scalar_heap_push4(heap, c, i, r.nv, dots);
    }
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
