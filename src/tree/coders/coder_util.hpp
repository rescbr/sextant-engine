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

/// Buffered top-W admission (replaces the per-candidate heap branch).
/// Candidates buffer per LEAF (raw f32 dists); the flush vectorizes the
/// monotone f32->u32 key transform + threshold filter (AVX-512) and only
/// the survivors (~W out of thousands once the heap fills) touch the
/// branchy heap. Bit-exact with the incremental path: heaps are
/// sentinel-prefilled, so the bounded survivor set is the W smallest by
/// heap_entry_less INDEPENDENT of arrival order (see heap_init).
struct ScalarCandBuf {
    std::vector<float> dist;
    std::vector<uint32_t> idx;
    void reset() { dist.clear(); idx.clear(); }
};

inline void scalar_cand_push4(ScalarCandBuf& b, const ScalarScanCtx& c,
                              const uint32_t i, const uint32_t nv,
                              const float* dots) {
    for (uint32_t v = 0; v < nv; ++v) {
        const float score = dots[v] + c.c0;
        b.dist.push_back(c.ip_bias
            ? -(score * static_cast<float>(c.ip_bias[i + v]))
            : -score);
        b.idx.push_back(i + v);
    }
}

inline void scalar_cand_flush(ScalarCandBuf& b, RawScanHeap& heap) {
    const size_t n = b.dist.size();
    if (!heap_full(heap)) {
        // Cold start: fill (front is the sentinel 0xFFFFFFFF until full).
        for (size_t j = 0; j < n; ++j)
            heap_push(heap, f32_to_dist_key(b.dist[j]), b.idx[j]);
        b.reset();
        return;
    }
    // Threshold taken once: the max-heap front only DECREASES during the
    // flush, so a stale threshold merely admits a few candidates the
    // guarded replace then rejects.
    const uint32_t thr = heap_front(heap);
#if defined(__AVX512F__)
    const __m512i sgn = _mm512_set1_epi32(static_cast<int>(0x80000000u));
    size_t j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m512i bits = _mm512_castps_si512(
            _mm512_loadu_ps(&b.dist[j]));
        const __m512i keys = _mm512_xor_si512(
            bits, _mm512_or_si512(_mm512_srai_epi32(bits, 31), sgn));
        // Strictly-smaller lanes take the guarded replace; EQUAL keys
        // need the (dist, slot, idx) tie-break, so route them through
        // the scalar check too (rare).
        uint32_t k16[16];
        _mm512_storeu_si512(k16, keys);
        __mmask16 m = _mm512_cmplt_epu32_mask(
            keys, _mm512_set1_epi32(thr));
        while (m) {
            const int l = __builtin_ctz(m);
            m &= m - 1;
            heap_replace_top(heap, k16[l], b.idx[j + l]);
        }
        m = _mm512_cmpeq_epu32_mask(keys, _mm512_set1_epi32(thr));
        while (m) {
            const int l = __builtin_ctz(m);
            m &= m - 1;
            if (heap_should_replace(heap, k16[l], b.idx[j + l]))
                heap_replace_top(heap, k16[l], b.idx[j + l]);
        }
    }
    for (; j < n; ++j) {
        const uint32_t key = f32_to_dist_key(b.dist[j]);
        if (heap_should_replace(heap, key, b.idx[j]))
            heap_replace_top(heap, key, b.idx[j]);
    }
#else
    for (size_t j = 0; j < n; ++j) {
        const uint32_t key = f32_to_dist_key(b.dist[j]);
        if (heap_should_replace(heap, key, b.idx[j]))
            heap_replace_top(heap, key, b.idx[j]);
    }
#endif
    b.reset();
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
            const int64_t g = c.slm_shaped ? c.shape_u8[nib]
                                           : static_cast<int64_t>(nib);
            acc += static_cast<int64_t>(a8[d]) * g;
            if (a8_lo) acc_lo += static_cast<int64_t>(a8_lo[d]) * g;
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
    // Shared-shape: nibbles map through the u8 f table (pshufb per
    // 128-bit lane) before the dpbusd — the affine f correction
    // (fmin·Σa_d, 1/S) is folded into c0/i8_inv at setup time.
    // dpbusd operand roles: FIRST unsigned, SECOND signed. The g-mapped
    // nibbles (0..255) take the unsigned slot; a8 (int8) broadcasts take
    // the signed slot — the product is then EXACT, no bias correction.
    const __m512i g_tbl = c.slm_shaped
        ? _mm512_broadcast_i32x4(_mm_loadu_si128(
              reinterpret_cast<const __m128i*>(c.shape_u8)))
        : _mm512_setzero_si512();
    __m512i acc = _mm512_setzero_si512();   // quarter v: row v lane sums
    __m512i acclo = _mm512_setzero_si512();
    for (uint32_t d = 0; d < d16i; d += 16) {
        __m512i nibs = avx512_nibbles4(cp, d);
        if (c.slm_shaped) nibs = _mm512_shuffle_epi8(g_tbl, nibs);
        const __m128i a128 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(a8p + d));
        acc = _mm512_dpbusd_epi32(acc, nibs,
                                  _mm512_broadcast_i32x4(a128));
        if (a8lop) {
            const __m128i alo128 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(a8lop + d));
            acclo = _mm512_dpbusd_epi32(acclo, nibs,
                                        _mm512_broadcast_i32x4(alo128));
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
                const uint8_t g = c.slm_shaped ? c.shape_u8[nib] : nib;
                dots[v] += a8d * g * c.i8_inv;
            }
        }
    }
    return;
#endif
    scalar_i8_dots4_ref(c, cp, dots);
}

/// 4-QUERY batched i8 kernel: one nibble unpack per 16-dim chunk shared by
/// 4 query contexts (same leaf → same dim/cs), one dpbusd chain per query
/// against its own a8 broadcast. Per query, the accumulator sequence is
/// IDENTICAL to scalar_i8_dots4 (same nibs values, same dpbusd order), so
/// dots[q][v] is bit-identical to a per-query scalar_i8_dots4 call.
inline void scalar_i8_dots4_q4(const ScalarScanCtx* const c4[4],
                               const uint8_t* const* cp,
                               float dots[4][4]) {
#if defined(SEXTANT_HAS_AVX512_SCAN)
    const ScalarScanCtx& c0 = *c4[0];
    const uint32_t dim = c0.dim;
    const uint32_t cs = c0.cs;
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded = padded / 2 <= cs;
    const uint32_t d16i = fully_padded ? padded : dim / 16 * 16;
    const __m512i g_tbl = c0.slm_shaped
        ? _mm512_broadcast_i32x4(_mm_loadu_si128(
              reinterpret_cast<const __m128i*>(c0.shape_u8)))
        : _mm512_setzero_si512();
    __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(),
                      _mm512_setzero_si512(), _mm512_setzero_si512()};
    __m512i acclo[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(),
                        _mm512_setzero_si512(), _mm512_setzero_si512()};
    for (uint32_t d = 0; d < d16i; d += 16) {
        __m512i nibs = avx512_nibbles4(cp, d);  // unpack ONCE, shared
        if (c0.slm_shaped) nibs = _mm512_shuffle_epi8(g_tbl, nibs);
        for (uint32_t q = 0; q < 4; ++q) {
            const ScalarScanCtx& cq = *c4[q];
            const __m128i a128 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(cq.a8 + d));
            acc[q] = _mm512_dpbusd_epi32(
                acc[q], nibs, _mm512_broadcast_i32x4(a128));
            if (cq.i8_mode >= 2 && cq.a8_lo) {
                const __m128i alo128 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(cq.a8_lo + d));
                acclo[q] = _mm512_dpbusd_epi32(
                    acclo[q], nibs, _mm512_broadcast_i32x4(alo128));
            }
        }
    }
    // Epilogue: fold each 128-bit lane's four dwords into the row-v total
    // with SIMD shuffles only (row v lives in lane v). Single store; the
    // four row totals are read from the folded lanes.
    alignas(64) int32_t sums[16];
    alignas(64) int32_t sums_lo[16];
    for (uint32_t q = 0; q < 4; ++q) {
        const ScalarScanCtx& cq = *c4[q];
        const bool lo = cq.i8_mode >= 2 && cq.a8_lo;
        __m512i t = _mm512_add_epi32(acc[q], _mm512_shuffle_epi32(
                                         acc[q], _MM_SHUFFLE(2, 3, 0, 1)));
        t = _mm512_add_epi32(t, _mm512_shuffle_epi32(
                                 t, _MM_SHUFFLE(1, 0, 3, 2)));
        __m512i u = t;
        if (lo) {
            u = _mm512_add_epi32(
                acclo[q], _mm512_shuffle_epi32(acclo[q],
                                               _MM_SHUFFLE(2, 3, 0, 1)));
            u = _mm512_add_epi32(u, _mm512_shuffle_epi32(
                                     u, _MM_SHUFFLE(1, 0, 3, 2)));
            _mm512_store_si512(reinterpret_cast<__m512i*>(sums_lo), u);
        }
        _mm512_store_si512(reinterpret_cast<__m512i*>(sums), t);
        if (lo) {
            const __m128 f = _mm_add_ps(
                _mm_cvtepi32_ps(_mm_setr_epi32(sums[0], sums[4], sums[8],
                                               sums[12])),
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_setr_epi32(
                              sums_lo[0], sums_lo[4], sums_lo[8],
                              sums_lo[12])),
                           _mm_set1_ps(1.0f / 127.0f)));
            _mm_storeu_ps(&dots[q][0], _mm_mul_ps(f, _mm_set1_ps(cq.i8_inv)));
        } else {
            const __m128 f = _mm_mul_ps(
                _mm_cvtepi32_ps(_mm_setr_epi32(sums[0], sums[4], sums[8],
                                               sums[12])),
                _mm_set1_ps(cq.i8_inv));
            _mm_storeu_ps(&dots[q][0], f);
        }
    }
    if (!fully_padded) {
        // SIMD tail: masked loads build zero-padded rows; zero nibbles
        // contribute nothing to dpbusd, so the padded chunk is exact.
        const uint32_t rem = dim - d16i;           // 1..15 dims
        const uint32_t nbytes = (rem + 1) / 2;     // 1..8 bytes
        auto row_masked = [&](const uint8_t* row) {
            const __m128i b = _mm_maskz_loadu_epi8(
                static_cast<__mmask16>((1u << nbytes) - 1u),
                row + d16i / 2);
            const __m128i m = _mm_set1_epi8(0x0F);
            const __m128i l = _mm_and_si128(b, m);
            const __m128i h = _mm_and_si128(_mm_srli_epi16(b, 4), m);
            const __m128i l4 = _mm_srli_si128(l, 4);
            const __m128i h4 = _mm_srli_si128(h, 4);
            return _mm_unpacklo_epi64(_mm_unpacklo_epi8(l, h),
                                      _mm_unpacklo_epi8(l4, h4));
        };
        const __m512i nibs = _mm512_inserti32x4(
            _mm512_inserti32x4(
                _mm512_inserti32x4(
                    _mm512_castsi128_si512(row_masked(cp[0])),
                    row_masked(cp[1]), 1),
                row_masked(cp[2]), 2),
            row_masked(cp[3]), 3);
        const __m512i nibs_g =
            c0.slm_shaped ? _mm512_shuffle_epi8(g_tbl, nibs) : nibs;
        const __mmask8 ma = static_cast<__mmask8>((1u << rem) - 1u);
        auto fold = [](__m512i v) {
            __m512i t = _mm512_add_epi32(
                v, _mm512_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
            return _mm512_add_epi32(
                t, _mm512_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
        };
        for (uint32_t q = 0; q < 4; ++q) {
            const ScalarScanCtx& cq = *c4[q];
            const bool lo = cq.i8_mode >= 2 && cq.a8_lo;
            const __m128i a128 = _mm_maskz_loadu_epi8(ma, cq.a8 + d16i);
            __m512i tq = _mm512_dpbusd_epi32(
                _mm512_setzero_si512(), nibs_g,
                _mm512_broadcast_i32x4(a128));
            __m512i ulo = tq;
            if (lo) {
                const __m128i alo = _mm_maskz_loadu_epi8(ma,
                                                         cq.a8_lo + d16i);
                ulo = _mm512_dpbusd_epi32(_mm512_setzero_si512(), nibs_g,
                                          _mm512_broadcast_i32x4(alo));
            }
            _mm512_store_si512(reinterpret_cast<__m512i*>(sums), fold(tq));
            if (lo) _mm512_store_si512(reinterpret_cast<__m512i*>(sums_lo),
                                       fold(ulo));
            const __m128 f0 = _mm_cvtepi32_ps(
                _mm_setr_epi32(sums[0], sums[4], sums[8], sums[12]));
            const __m128 f = lo
                ? _mm_add_ps(f0, _mm_mul_ps(
                    _mm_cvtepi32_ps(_mm_setr_epi32(
                        sums_lo[0], sums_lo[4], sums_lo[8], sums_lo[12])),
                    _mm_set1_ps(1.0f / 127.0f)))
                : f0;
            _mm_storeu_ps(&dots[q][0], _mm_add_ps(
                _mm_loadu_ps(&dots[q][0]),
                _mm_mul_ps(f, _mm_set1_ps(cq.i8_inv))));
        }
    }
    return;
#endif
    for (uint32_t q = 0; q < 4; ++q)
        scalar_i8_dots4_ref(*c4[q], cp, dots[q]);
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
    static thread_local detail::ScalarCandBuf cbuf;
    for (uint32_t i = 0; i < c.count; i += 4) {
        const auto r = detail::scalar_rows(c, i, pad_row);
        float dots[4];
        scalar_i8_dots4(c, r.cp, dots);
        detail::scalar_cand_push4(cbuf, c, i, r.nv, dots);
    }
    detail::scalar_cand_flush(cbuf, heap);
}

inline void scalar_scan_arith(const ScalarScanCtx& c, RawScanHeap& heap) {
    const uint32_t cs = c.cs;
    static thread_local std::vector<uint8_t> pad_row_buf;
    if (pad_row_buf.size() < cs) pad_row_buf.assign(cs, 0);
    const uint8_t* pad_row = pad_row_buf.data();
    static thread_local detail::ScalarCandBuf cbuf;
    for (uint32_t i = 0; i < c.count; i += 4) {
        const auto r = detail::scalar_rows(c, i, pad_row);
        float dots[4];
        scalar_arith_dots4(c, r.cp, dots);
        detail::scalar_cand_push4(cbuf, c, i, r.nv, dots);
    }
    detail::scalar_cand_flush(cbuf, heap);
}

inline void scalar_scan_gather(const ScalarScanCtx& c, RawScanHeap& heap) {
    static thread_local detail::ScalarCandBuf cbuf;
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
        detail::scalar_cand_push4(cbuf, c, i, r.nv, dots);
    }
    detail::scalar_cand_flush(cbuf, heap);
}

inline void scalar_scan_leaf(const ScalarScanCtx& c, RawScanHeap& heap) {
    if (c.i8_mode) scalar_scan_i8(c, heap);
    else if (c.slm_arith) scalar_scan_arith(c, heap);
    else scalar_scan_gather(c, heap);
}

/// Multi-query i8 scan: ONE pass over the leaf's code rows serving up to 4
/// query contexts (same leaf → same codes/count/cs). Per query the
/// candidate push/flush sequence is identical to scalar_scan_i8 (all rows
/// pushed, one flush), and the buffered admission is arrival-order
/// independent anyway — heaps end up bit-identical to per-query scans.
inline void scalar_scan_i8_batch(const ScalarScanCtx* const c4[4],
                                 uint32_t qn, RawScanHeap* const heaps[4]) {
    const ScalarScanCtx& c0 = *c4[0];
    static thread_local std::vector<uint8_t> pad_row_buf;
    if (pad_row_buf.size() < c0.cs) pad_row_buf.assign(c0.cs, 0);
    // Reused per-worker buffers (capacity persists across leaves, like the
    // single-query kernel's thread-local cbuf).
    static thread_local detail::ScalarCandBuf cbufs[4];
    for (uint32_t q = 0; q < qn; ++q) cbufs[q].reset();
    for (uint32_t i = 0; i < c0.count; i += 4) {
        const auto r = detail::scalar_rows(c0, i, pad_row_buf.data());
        float dots[4][4];
        scalar_i8_dots4_q4(c4, r.cp, dots);
        for (uint32_t q = 0; q < qn; ++q)
            detail::scalar_cand_push4(cbufs[q], *c4[q], i, r.nv, dots[q]);
    }
    for (uint32_t q = 0; q < qn; ++q)
        detail::scalar_cand_flush(cbufs[q], *heaps[q]);
}

/// Batched dispatch for up to 4 query contexts on one leaf. Uses the
/// shared-unpack i8 kernel when every context agrees on i8_mode and
/// slm_shaped (the normal case — one coder, one leaf); otherwise falls
/// back to per-query scalar_scan_leaf (zero behavior change either way).
inline void scalar_scan_leaf_batch(const ScalarScanCtx* const c4[4],
                                   uint32_t qn, RawScanHeap* const heaps[4]) {
    if (qn > 0 && c4[0]->i8_mode) {
        bool uniform = true;
        for (uint32_t q = 0; q < qn; ++q)
            if (!c4[q]->i8_mode ||
                c4[q]->slm_shaped != c4[0]->slm_shaped) uniform = false;
        if (uniform) {
            scalar_scan_i8_batch(c4, qn, heaps);
            return;
        }
    }
    for (uint32_t q = 0; q < qn; ++q)
        scalar_scan_leaf(*c4[q], *heaps[q]);
}

/// Fused decode+L2² rerank for arithmetic (uniform / shared-shape) 4-bit
/// rows: with level_d[k] = L0_d + step_d·g[k] (g = identity for uniform,
/// the u8-quantized shared-shape offsets otherwise — the SAME table/scale
/// the scan kernel uses),
///   ||q - level(c)||² = A + Σ B_d·g[c_d] + Σ C_d·g[c_d]²
/// A/B/C are per-query constants (levels are global), so rerank is one
/// shuffle-dot pass instead of a 768-dim gather decode + l2sq.
/// `b`/`c` must be zero-padded to the 16-dim kernel width.
inline float scalar_arith_dist1(const uint8_t* row, uint32_t dim, uint32_t cs,
                                float a, const float* b, const float* c,
                                const uint8_t* gu8, const float* g) {
#if defined(SEXTANT_HAS_AVX512_SCAN)
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded = padded / 2 <= cs;
    const uint32_t d16 = fully_padded ? padded : dim / 16 * 16;
    const __m128i g_tbl = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(gu8));
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    for (uint32_t d = 0; d < d16; d += 16) {
        const __m512 bb = _mm512_loadu_ps(b + d);
        const __m512 cc = _mm512_loadu_ps(c + d);
        __m128i n = avx512_row_nibbles(row, d);
        n = _mm_shuffle_epi8(g_tbl, n);
        const __m512 gf =
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(n));
        acc1 = _mm512_fmadd_ps(gf, bb, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_mul_ps(gf, gf), cc, acc2);
    }
    float dist = a + _mm512_reduce_add_ps(acc1) + _mm512_reduce_add_ps(acc2);
    if (!fully_padded) {
        for (uint32_t d = d16; d < dim; ++d) {
            const uint8_t byte = row[d / 2];
            const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                             : ((byte >> 4) & 0xF);
            const float gf = g[nib];
            dist += b[d] * gf + c[d] * gf * gf;
        }
    }
    return dist;
#else
    float dist = a;
    for (uint32_t d = 0; d < dim; ++d) {
        const uint8_t byte = row[d / 2];
        const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                         : ((byte >> 4) & 0xF);
        const float gf = g[nib];
        dist += b[d] * gf + c[d] * gf * gf;
    }
    return dist;
#endif
}

/// Order-preserving u32 encoding of a float score (the old heap path's
/// dist_bits trick, factored for reuse).
inline uint32_t f32_to_dist_key(float dist) {
    uint32_t bits;
    std::memcpy(&bits, &dist, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

}  // namespace sextant::tree::coders
