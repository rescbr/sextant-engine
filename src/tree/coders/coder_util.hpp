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
    // Sweep-local expanded staging (ARM value): when true, `codes` points
    // at PRE-EXPANDED 1-byte-per-dim rows (g-map already baked by
    // scalar_expand_codes, tail lanes zero) and `cs` is the expanded row
    // stride. Only the i8 kernels honor it; ip_bias keeps pointing into
    // the PACKED leaf (it is addressed separately from codes).
    bool expanded = false;
    const float16_t* ip_bias = nullptr;  // count entries or null
    // L2 serving: ip_bias entries hold ||x_hat|^2 and the score is combined
    // as bias - 2*score (= ||q - x_hat||^2 minus the query constant) instead
    // of the IP metric's multiplicative -(score * bias).
    bool bias_is_normsq = false;
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
        float dist;
        if (c.ip_bias) {
            const float b = static_cast<float>(c.ip_bias[i + v]);
            dist = c.bias_is_normsq ? (b - 2.f * score) : -(score * b);
        } else {
            dist = -score;
        }
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
        float dist;
        if (c.ip_bias) {
            const float b = static_cast<float>(c.ip_bias[i + v]);
            dist = c.bias_is_normsq ? (b - 2.f * score) : -(score * b);
        } else {
            dist = -score;
        }
        b.dist.push_back(dist);
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
    const uint32_t thr [[maybe_unused]] = heap_front(heap);
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
// NEON (SEXTANT_HAS_NEON_DOTSCAN): unshaped nibbles are positive s8 lanes
// so plain SDOT is already the exact product; shaped uses USDOT under
// FEAT_I8MM or the exact two-SDOT bias split otherwise (see neon_gdot_s32).
// All accumulation is integer, and the zero-masked tail folds into the same
// i32 accumulators, so every NEON path is bit-identical to the _ref twin
// for ANY dim (integer sums are associative; float() runs once on the exact
// total).
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
#elif defined(__ARM_FEATURE_DOTPROD)
// NEON DotProd scan path (SDOT) for the i8 dot kernels below. Guarded by
// __ARM_FEATURE_DOTPROD (NOT by the meson simd_target string): a 'neon'
// -march=native machine without FEAT_DOTPROD still compiles → _ref.
// FEAT_I8MM (USDOT) is an inner refinement for the shaped g-table dot;
// i8mm without dotprod is not a real configuration, so the outer guard is
// dotprod alone.
//
// SVE2 analysis (why there is no sve2_dots4 in src/simd/sve2_kernels.cpp):
// this kernel does NO table gather — the SVE2 win in scalar_dot_u4_sve2 is
// the svld1_gather; here the cost is nibble unpack + 4-way byte dots, both
// of which SVE2 svdot does at exactly NEON SDOT throughput per 128 bits.
// Neoverse-V2 (c4a) runs SVE at 2×128-bit, so svdot at VL=256 would need
// 32 nibble lanes unpacked per chunk (4× the vzip work) for the same dot
// throughput the NEON loop already gets with unrolled 128-bit vectors —
// no vector-width or gather advantage for THIS kernel. Engine TUs build
// with -march=native, so sve2 builds get this NEON path directly; the
// explicitly -march=armv8-a+sve2 sve2 TU never includes this header.
#define SEXTANT_HAS_NEON_DOTSCAN 1
#include <arm_neon.h>
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

#if defined(SEXTANT_HAS_NEON_DOTSCAN)
/// 16 nibbles of one 16-dim chunk of row `v`, dim-ordered, as uint8x16_t.
/// 8 code bytes hold 16 dims: uint8x8 zip interleaves the lo/hi nibbles of
/// bytes 0-3 into dims 0-7 (val[0]) and bytes 4-7 into dims 8-15 (val[1]).
inline uint8x16_t neon_row_nibbles(const uint8_t* row, uint32_t d) {
    const uint8x8_t b = vld1_u8(row + d / 2);
    const uint8x8_t lo = vand_u8(b, vdup_n_u8(0x0F));
    const uint8x8_t hi = vshr_n_u8(b, 4);
    const uint8x8x2_t z = vzip_u8(lo, hi);
    return vcombine_u8(z.val[0], z.val[1]);
}

/// Tail chunk (rem = dim % 16 ∈ 1..15): the (rem+1)/2 real bytes are
/// copied into a zeroed 8-byte scratch (zero-pad, the house FHM-remainder
/// pattern) and expanded like neon_row_nibbles — dims beyond the copy are
/// naturally zero, EXCEPT the odd-dim high nibble of the last byte, which
/// the byte copy alone would leave live. The lane mask returned by
/// neon_tail_mask zeroes it (and must be applied AFTER the shaped g-table
/// lookup: a zeroed nibble INDEX maps to g_tbl[0], which need not be 0).
inline uint8x16_t neon_row_nibbles_tail(const uint8_t* row, uint32_t d,
                                        uint32_t rem, uint8_t scratch[8]) {
    const uint32_t nbytes = (rem + 1) / 2;   // 1..8
    memset(scratch, 0, 8);
    memcpy(scratch, row + d / 2, nbytes);
    const uint8x8_t b = vld1_u8(scratch);
    const uint8x8_t lo = vand_u8(b, vdup_n_u8(0x0F));
    const uint8x8_t hi = vshr_n_u8(b, 4);
    const uint8x8x2_t z = vzip_u8(lo, hi);
    return vcombine_u8(z.val[0], z.val[1]);
}

/// 0xFF for the leading `rem` (1..15) nibble lanes, 0 beyond.
inline uint8x16_t neon_tail_mask(uint32_t rem) {
    static const uint8_t iota16[16] = {0,  1,  2,  3,  4,  5,  6,  7,
                                       8,  9,  10, 11, 12, 13, 14, 15};
    return vcltq_u8(vld1q_u8(iota16),
                    vdupq_n_u8(static_cast<uint8_t>(rem)));
}

/// acc += Σ_j nib[j]·a[j] as i32 (SDOT). Unshaped nibbles are 0..15, i.e.
/// POSITIVE s8 lanes, so the signed×signed SDOT reproduces the exact
/// integer product the AVX-512 dpbusd computes — no bias trick needed.
inline int32x4_t neon_nibdot_s32(int32x4_t acc, uint8x16_t nib, int8x16_t a) {
    return vdotq_s32(acc, vreinterpretq_s8_u8(nib), a);
}

/// acc += Σ_j g[j]·a[j], g unsigned 0..255 (shaped g-table output).
/// With FEAT_I8MM: USDOT (u8×s8→i32) — the exact analog of dpbusd,
/// single instruction, exact.
/// Without i8mm (dotprod-only cores, e.g. Neoverse-N1): the exact
/// two-SDOT bias split
///     g = (g & 0x7F) + 128·(g >> 7)
///     Σ a·g = SDOT(a, g&0x7F) + 128·SDOT(a, g>>7)
/// g>>7 ∈ {0,1} is a positive s8 lane; the <<7 scale is exact in i32 and
/// every per-lane sum stays far below i32 overflow, so the split is exact.
inline int32x4_t neon_gdot_s32(int32x4_t acc, uint8x16_t g, int8x16_t a) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    return vusdotq_s32(acc, g, a);
#else
    const int8x16_t gl = vreinterpretq_s8_u8(vandq_u8(g, vdupq_n_u8(0x7F)));
    const int8x16_t gh = vreinterpretq_s8_u8(vshrq_n_u8(g, 7));
    const int32x4_t lo = vdotq_s32(acc, gl, a);
    const int32x4_t hi = vdotq_s32(vdupq_n_s32(0), gh, a);
    return vaddq_s32(lo, vshlq_n_s32(hi, 7));
#endif
}

/// Horizontal fold + the shared float epilogue. The i32 lane sums and the
/// vaddvq reduction are EXACT integers (integer addition is associative,
/// so lane order is irrelevant), and float(i32) rounds the same value the
/// _ref twin's float(i64) rounds — bit-identical for any magnitude.
inline float neon_i8_finish(int32x4_t s, int32x4_t s_lo, bool has_lo,
                            float inv) {
    const int32_t t = vaddvq_s32(s);
    float lo = 0.0f;
    if (has_lo)
        lo = static_cast<float>(vaddvq_s32(s_lo)) / 127.0f;
    return (static_cast<float>(t) + lo) * inv;
}
#endif  // NEON DOTPROD

/// Expand packed 4-bit flat-nibble code rows into 1-byte-per-dim rows,
/// baking the shaped g-table (when `shaped`) at expand time. Tail lanes
/// (dims >= dim up to the 16-dim padded width) are written as ZERO — the
/// same value the packed kernels mask in — so expanded kernels need no
/// tail handling. Returns the expanded row stride (= dim padded to 16).
/// Bit-exact twin of the packed nibble decode: a zeroed nibble INDEX
/// reads g_tbl[0] (need not be 0), so the tail mask applies AFTER the tbl
/// lookup, exactly like neon_row_nibbles_tail consumers.
inline uint32_t scalar_expand_codes(const uint8_t* codes, uint32_t count,
                                    uint32_t cs, uint32_t dim, bool shaped,
                                    const uint8_t* shape_u8, uint8_t* out) {
    const uint32_t padded = (dim + 15) / 16 * 16;
#if defined(SEXTANT_HAS_NEON_DOTSCAN)
    const uint32_t d16i = dim / 16 * 16;
    const uint8x16_t g_tbl = shaped
        ? vld1q_u8(shape_u8) : vdupq_n_u8(0);
    for (uint32_t r = 0; r < count; ++r, codes += cs, out += padded) {
        for (uint32_t d = 0; d < d16i; d += 16) {
            uint8x16_t nib = neon_row_nibbles(codes, d);
            if (shaped) nib = vqtbl1q_u8(g_tbl, nib);
            vst1q_u8(out + d, nib);
        }
        if (d16i < dim) {
            const uint32_t rem = dim - d16i;
            uint8_t scratch[8];
            uint8x16_t nib =
                neon_row_nibbles_tail(codes, d16i, rem, scratch);
            if (shaped) nib = vqtbl1q_u8(g_tbl, nib);  // mask AFTER tbl
            vst1q_u8(out + d16i, vandq_u8(nib, neon_tail_mask(rem)));
        }
    }
#else
    for (uint32_t r = 0; r < count; ++r, codes += cs, out += padded) {
        for (uint32_t d = 0; d < padded; ++d) {
            uint8_t g = 0;
            if (d < dim) {
                const uint8_t byte = codes[d / 2];
                const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                                 : (byte >> 4);
                g = shaped ? shape_u8[nib] : nib;
            }
            out[d] = g;
        }
    }
#endif
    return padded;
}

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
#elif defined(SEXTANT_HAS_NEON_DOTSCAN)
    // NEON SDOT/USDOT. One 128-bit vector per row; 16 dims per chunk = one
    // vdotq per (row, chunk). ALL accumulation stays integer — the tail
    // chunk's nibbles are zero-masked so it folds into the SAME i32
    // accumulators and the float epilogue runs exactly once, matching the
    // _ref expression bit-for-bit for ANY dim (the AVX-512 single kernel's
    // scalar float tail above is last-ulp-off on non-multiple-of-16 dims;
    // this path is exact everywhere).
    const uint32_t dim = c.dim;
    const uint32_t cs = c.cs;
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded = padded / 2 <= cs;
    const uint32_t d16i = fully_padded ? padded : dim / 16 * 16;
    const int8_t* a8p = c.a8;                       // zero-padded to `padded`
    const int8_t* a8lop = c.i8_mode >= 2 ? c.a8_lo : nullptr;
    const bool shaped = c.slm_shaped;
    // Shaped: nibbles map through the 16-entry u8 f table (vqtbl1q — the
    // exact 16-entry TBL analog of pshufb) before the dot; the affine f
    // correction is folded into c0/i8_inv at setup time, as on AVX-512.
    const uint8x16_t g_tbl = shaped
        ? vld1q_u8(c.shape_u8) : vdupq_n_u8(0);
    int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0),
                        vdupq_n_s32(0), vdupq_n_s32(0)};
    int32x4_t acclo[4] = {vdupq_n_s32(0), vdupq_n_s32(0),
                          vdupq_n_s32(0), vdupq_n_s32(0)};
    for (uint32_t d = 0; d < d16i; d += 16) {
        const int8x16_t a = vld1q_s8(a8p + d);
        const int8x16_t al = a8lop ? vld1q_s8(a8lop + d) : vdupq_n_s8(0);
        for (uint32_t v = 0; v < 4; ++v) {
            uint8x16_t nib = neon_row_nibbles(cp[v], d);
            // Unshaped nibbles (0..15) dot directly via SDOT; shaped map
            // through g first and dot via USDOT / the bias split.
            acc[v] = shaped
                ? neon_gdot_s32(acc[v], vqtbl1q_u8(g_tbl, nib), a)
                : neon_nibdot_s32(acc[v], nib, a);
            if (a8lop)
                acclo[v] = shaped
                    ? neon_gdot_s32(acclo[v], vqtbl1q_u8(g_tbl, nib), al)
                    : neon_nibdot_s32(acclo[v], nib, al);
        }
    }
    if (!fully_padded) {
        // Tail chunk (rem = 1..15): zero-masked nibbles + the zero-padded
        // a8/a8_lo tail lanes fold into the same integer accumulators.
        const uint32_t rem = dim - d16i;
        const uint8x16_t m = neon_tail_mask(rem);
        uint8_t scratch[4][8];
        const int8x16_t a = vld1q_s8(a8p + d16i);
        const int8x16_t al = a8lop ? vld1q_s8(a8lop + d16i) : vdupq_n_s8(0);
        for (uint32_t v = 0; v < 4; ++v) {
            uint8x16_t nib = neon_row_nibbles_tail(cp[v], d16i, rem,
                                                  scratch[v]);
            if (shaped) nib = vqtbl1q_u8(g_tbl, nib);  // mask AFTER tbl
            nib = vandq_u8(nib, m);
            acc[v] = shaped
                ? neon_gdot_s32(acc[v], nib, a)
                : neon_nibdot_s32(acc[v], nib, a);
            if (a8lop)
                acclo[v] = shaped
                    ? neon_gdot_s32(acclo[v], nib, al)
                    : neon_nibdot_s32(acclo[v], nib, al);
        }
    }
    for (uint32_t v = 0; v < 4; ++v)
        dots[v] = neon_i8_finish(acc[v], acclo[v], a8lop != nullptr,
                                 c.i8_inv);
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
    // 2-chunk unroll: even/odd accumulator halves break the per-query
    // dpbusd dependency chain (latency ~2x its issue slot on Zen5) and
    // overlap the next chunk's nibble unpack with the current dots.
    // Integer accumulation reassociates exactly — bit-identical results.
    __m512i acc1[4] = {acc[0], acc[1], acc[2], acc[3]};
    __m512i acclo1[4] = {acclo[0], acclo[1], acclo[2], acclo[3]};
    uint32_t d = 0;
    for (; d + 32 <= d16i; d += 32) {
        __m512i nibs = avx512_nibbles4(cp, d);  // unpack ONCE, shared
        __m512i nibs1 = avx512_nibbles4(cp, d + 16);
        if (c0.slm_shaped) {
            nibs = _mm512_shuffle_epi8(g_tbl, nibs);
            nibs1 = _mm512_shuffle_epi8(g_tbl, nibs1);
        }
        for (uint32_t q = 0; q < 4; ++q) {
            const ScalarScanCtx& cq = *c4[q];
            const __m128i a128 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(cq.a8 + d));
            const __m128i a1281 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(cq.a8 + d + 16));
            acc[q] = _mm512_dpbusd_epi32(
                acc[q], nibs, _mm512_broadcast_i32x4(a128));
            acc1[q] = _mm512_dpbusd_epi32(
                acc1[q], nibs1, _mm512_broadcast_i32x4(a1281));
            if (cq.i8_mode >= 2 && cq.a8_lo) {
                const __m128i alo128 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(cq.a8_lo + d));
                const __m128i alo1281 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(cq.a8_lo + d + 16));
                acclo[q] = _mm512_dpbusd_epi32(
                    acclo[q], nibs, _mm512_broadcast_i32x4(alo128));
                acclo1[q] = _mm512_dpbusd_epi32(
                    acclo1[q], nibs1, _mm512_broadcast_i32x4(alo1281));
            }
        }
    }
    for (; d < d16i; d += 16) {
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
    for (uint32_t q = 0; q < 4; ++q) {
        acc[q] = _mm512_add_epi32(acc[q], acc1[q]);
        acclo[q] = _mm512_add_epi32(acclo[q], acclo1[q]);
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
        // __mmask16, NOT __mmask8: rem is 9..14 for real tail dims (cs =
        // ceil(dim/2) makes rem=15 fully-padded), and an 8-bit mask
        // silently zeroed a8 lanes 8..15, dropping tail dims 8..14.
        // Caught by test_dots4_kernel (q4-vs-ref diverged for rem >= 9).
        const __mmask16 ma = static_cast<__mmask16>((1u << rem) - 1u);
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
#elif defined(SEXTANT_HAS_NEON_DOTSCAN)
    // 4-QUERY batched NEON kernel: one nibble unpack per (row, chunk) shared
    // by all 4 queries (same leaf → same dim/cs/shape), 4 independent
    // accumulator sets for ILP. Per (query, row) the integer lane order is
    // IDENTICAL to scalar_i8_dots4 above (same nibble expansion, same vdotq
    // sequence), so dots[q][v] is bit-identical to a per-query
    // scalar_i8_dots4 call — and the integer-only tail keeps that exact for
    // non-multiple-of-16 dims as well.
    const ScalarScanCtx& c0 = *c4[0];
    const uint32_t dim = c0.dim;
    const uint32_t cs = c0.cs;
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded = padded / 2 <= cs;
    const uint32_t d16i = fully_padded ? padded : dim / 16 * 16;
    const bool shaped = c0.slm_shaped;
    const uint8x16_t g_tbl = shaped
        ? vld1q_u8(c0.shape_u8) : vdupq_n_u8(0);
    int32x4_t acc[4][4], acclo[4][4];   // [query][row]
    for (uint32_t q = 0; q < 4; ++q)
        for (uint32_t v = 0; v < 4; ++v) {
            acc[q][v] = vdupq_n_s32(0);
            acclo[q][v] = vdupq_n_s32(0);
        }
    for (uint32_t d = 0; d < d16i; d += 16) {
        uint8x16_t nib[4];              // unpack ONCE per row, shared
        for (uint32_t v = 0; v < 4; ++v) {
            nib[v] = neon_row_nibbles(cp[v], d);
            if (shaped) nib[v] = vqtbl1q_u8(g_tbl, nib[v]);
        }
        for (uint32_t q = 0; q < 4; ++q) {
            const ScalarScanCtx& cq = *c4[q];
            const int8x16_t a = vld1q_s8(cq.a8 + d);
            const bool lo = cq.i8_mode >= 2 && cq.a8_lo;
            const int8x16_t al = lo ? vld1q_s8(cq.a8_lo + d) : vdupq_n_s8(0);
            for (uint32_t v = 0; v < 4; ++v) {
                acc[q][v] = shaped
                    ? neon_gdot_s32(acc[q][v], nib[v], a)
                    : neon_nibdot_s32(acc[q][v], nib[v], a);
                if (lo)
                    acclo[q][v] = shaped
                        ? neon_gdot_s32(acclo[q][v], nib[v], al)
                        : neon_nibdot_s32(acclo[q][v], nib[v], al);
            }
        }
    }
    // (A 2-chunk even/odd-accumulator unroll was tried here and REVERTED:
    // measured 64.6 vs 67.5 QPS on c4a/V2 culturaX f=1.0 — the [4][4]
    // accumulator arrays already sit at the 32-q-register budget and the
    // doubled chains spill. The AVX-512 unroll +6% does not carry over.)
    if (!fully_padded) {
        // Zero-masked tail chunk shared by all queries; a8/a8_lo tails are
        // zero-padded buffers, so the full 16-lane loads are safe.
        const uint32_t rem = dim - d16i;           // 1..15
        const uint8x16_t m = neon_tail_mask(rem);
        uint8_t scratch[4][8];
        uint8x16_t nib[4];
        for (uint32_t v = 0; v < 4; ++v) {
            nib[v] = neon_row_nibbles_tail(cp[v], d16i, rem, scratch[v]);
            if (shaped) nib[v] = vqtbl1q_u8(g_tbl, nib[v]);  // then mask
            nib[v] = vandq_u8(nib[v], m);
        }
        for (uint32_t q = 0; q < 4; ++q) {
            const ScalarScanCtx& cq = *c4[q];
            const int8x16_t a = vld1q_s8(cq.a8 + d16i);
            const bool lo = cq.i8_mode >= 2 && cq.a8_lo;
            const int8x16_t al = lo ? vld1q_s8(cq.a8_lo + d16i)
                                    : vdupq_n_s8(0);
            for (uint32_t v = 0; v < 4; ++v) {
                acc[q][v] = shaped
                    ? neon_gdot_s32(acc[q][v], nib[v], a)
                    : neon_nibdot_s32(acc[q][v], nib[v], a);
                if (lo)
                    acclo[q][v] = shaped
                        ? neon_gdot_s32(acclo[q][v], nib[v], al)
                        : neon_nibdot_s32(acclo[q][v], nib[v], al);
            }
        }
    }
    for (uint32_t q = 0; q < 4; ++q) {
        const ScalarScanCtx& cq = *c4[q];
        const bool lo = cq.i8_mode >= 2 && cq.a8_lo;
        for (uint32_t v = 0; v < 4; ++v)
            dots[q][v] = neon_i8_finish(acc[q][v], acclo[q][v], lo,
                                        cq.i8_inv);
    }
    return;
#endif
    for (uint32_t q = 0; q < 4; ++q)
        scalar_i8_dots4_ref(*c4[q], cp, dots[q]);
}

/// 4-QUERY batched i8 kernel over PRE-EXPANDED rows (see ScalarScanCtx::
/// expanded / scalar_expand_codes): codes are 1 byte/dim with the g-map
/// baked and tail lanes zero, so this is a pure dot stream — no nibble
/// unpack, no tbl, no tail masking. ARM-only value (x86 Zen5 measured the
/// expanded layout SLOWER: dpbusd is load-bound and the unpack is already
/// amortized over 4 queries); kept portable via a scalar twin so the
/// bit-exactness tests run on any host.
inline void scalar_i8_dots4_q4_expanded(const ScalarScanCtx* const c4[4],
                                        const uint8_t* const* cp,
                                        float dots[4][4]) {
#if defined(SEXTANT_HAS_NEON_DOTSCAN)
    const ScalarScanCtx& c0 = *c4[0];
    const uint32_t dim = c0.dim;
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool shaped = c0.slm_shaped;
    int32x4_t acc[4][4], acclo[4][4];   // [query][row]
    for (uint32_t q = 0; q < 4; ++q)
        for (uint32_t v = 0; v < 4; ++v) {
            acc[q][v] = vdupq_n_s32(0);
            acclo[q][v] = vdupq_n_s32(0);
        }
    for (uint32_t d = 0; d < padded; d += 16) {
        uint8x16_t g[4];               // load ONCE per row, shared
        for (uint32_t v = 0; v < 4; ++v) g[v] = vld1q_u8(cp[v] + d);
        for (uint32_t q = 0; q < 4; ++q) {
            const ScalarScanCtx& cq = *c4[q];
            const int8x16_t a = vld1q_s8(cq.a8 + d);
            const bool lo = cq.i8_mode >= 2 && cq.a8_lo;
            const int8x16_t al = lo ? vld1q_s8(cq.a8_lo + d) : vdupq_n_s8(0);
            for (uint32_t v = 0; v < 4; ++v) {
                acc[q][v] = shaped
                    ? neon_gdot_s32(acc[q][v], g[v], a)
                    : neon_nibdot_s32(acc[q][v], g[v], a);
                if (lo)
                    acclo[q][v] = shaped
                        ? neon_gdot_s32(acclo[q][v], g[v], al)
                        : neon_nibdot_s32(acclo[q][v], g[v], al);
            }
        }
    }
    for (uint32_t q = 0; q < 4; ++q) {
        const ScalarScanCtx& cq = *c4[q];
        const bool lo = cq.i8_mode >= 2 && cq.a8_lo;
        for (uint32_t v = 0; v < 4; ++v)
            dots[q][v] = neon_i8_finish(acc[q][v], acclo[q][v], lo,
                                        cq.i8_inv);
    }
    return;
#endif
    // Portable scalar twin — mirrors scalar_i8_dots4_ref's exact float
    // epilogue; expanded tail lanes are zero so looping dim-only is exact.
    for (uint32_t q = 0; q < 4; ++q) {
        const ScalarScanCtx& cq = *c4[q];
        const int8_t* a8 = cq.a8;
        const int8_t* a8_lo = cq.i8_mode >= 2 ? cq.a8_lo : nullptr;
        for (uint32_t v = 0; v < 4; ++v) {
            int64_t acc = 0, acc_lo = 0;
            for (uint32_t d = 0; d < cq.dim; ++d) {
                const int64_t g = cp[v][d];
                acc += static_cast<int64_t>(a8[d]) * g;
                if (a8_lo) acc_lo += static_cast<int64_t>(a8_lo[d]) * g;
            }
            dots[q][v] =
                (static_cast<float>(acc)
                 + (a8_lo ? static_cast<float>(acc_lo) / 127.0f : 0.0f))
                * cq.i8_inv;
        }
    }
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
    const bool expanded = c0.expanded;
    for (uint32_t i = 0; i < c0.count; i += 4) {
        const auto r = detail::scalar_rows(c0, i, pad_row_buf.data());
        float dots[4][4];
        if (expanded)
            scalar_i8_dots4_q4_expanded(c4, r.cp, dots);
        else
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
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // NEON float-FMA rerank: 16 dims/chunk, same structure as the AVX-512
    // branch (g-mapped nibbles -> FMA with b/c). b/c are zero-padded to
    // `padded` by the setup, so tail lanes contribute exact zeros — only
    // the (rare) not-fully-padded tail runs scalar, as on x86. Fused FMA
    // may differ from the pure-scalar fallback in the last ulp (the same
    // is true of the AVX-512 branch); integer scan distances are exact.
    const uint32_t padded = (dim + 15) / 16 * 16;
    const bool fully_padded = padded / 2 <= cs;
    const uint32_t d16 = fully_padded ? padded : dim / 16 * 16;
    const uint8x16_t g_tbl = vld1q_u8(gu8);
    const uint8x16_t m15 = vdupq_n_u8(0x0F);
    float32x4_t acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0);
    for (uint32_t d = 0; d < d16; d += 16) {
        // 8 code bytes hold dims d..d+15 (even dim = lo nibble, odd = hi).
        const uint8x16_t bx =
            vcombine_u8(vld1_u8(row + d / 2), vdup_n_u8(0));
        const uint8x16_t nib = vzip1q_u8(vandq_u8(bx, m15),
                                         vshrq_n_u8(bx, 4));
        const uint8x16_t gf8 = vqtbl1q_u8(g_tbl, nib);
        // u8x16 -> f32x4 lanes, twice per 8 dims.
        const uint16x8_t g16lo = vmovl_u8(vget_low_u8(gf8));
        const uint16x8_t g16hi = vmovl_u8(vget_high_u8(gf8));
        const float32x4_t gf0 = vcvtq_f32_s32(vreinterpretq_s32_u32(
            vmovl_u16(vget_low_u16(g16lo))));
        const float32x4_t gf1 = vcvtq_f32_s32(vreinterpretq_s32_u32(
            vmovl_u16(vget_high_u16(g16lo))));
        const float32x4_t gf2 = vcvtq_f32_s32(vreinterpretq_s32_u32(
            vmovl_u16(vget_low_u16(g16hi))));
        const float32x4_t gf3 = vcvtq_f32_s32(vreinterpretq_s32_u32(
            vmovl_u16(vget_high_u16(g16hi))));
        const float* bp = b + d;
        const float* cp = c + d;
        acc1 = vfmaq_f32(acc1, gf0, vld1q_f32(bp));
        acc1 = vfmaq_f32(acc1, gf1, vld1q_f32(bp + 4));
        acc1 = vfmaq_f32(acc1, gf2, vld1q_f32(bp + 8));
        acc1 = vfmaq_f32(acc1, gf3, vld1q_f32(bp + 12));
        acc2 = vfmaq_f32(acc2, gf0, vmulq_f32(gf0, vld1q_f32(cp)));
        acc2 = vfmaq_f32(acc2, gf1, vmulq_f32(gf1, vld1q_f32(cp + 4)));
        acc2 = vfmaq_f32(acc2, gf2, vmulq_f32(gf2, vld1q_f32(cp + 8)));
        acc2 = vfmaq_f32(acc2, gf3, vmulq_f32(gf3, vld1q_f32(cp + 12)));
    }
    float dist = a + vaddvq_f32(acc1) + vaddvq_f32(acc2);
    if (!fully_padded) {
        for (uint32_t d = d16; d < dim; ++d) {
            const uint8_t byte = row[d / 2];
            const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                             : ((byte >> 4) & 0xF);
            const float gf = static_cast<float>(gu8[nib]);
            dist += b[d] * gf + c[d] * gf * gf;
        }
    }
    return dist;
#else
    // gf must be the RAW u8 table value (gu8[nib]) — the 1/sg scale is
    // already folded into b/c by the setup. Using g[nib] (= gu8/sg) here
    // divided by sg a SECOND time (found on c4a: scalar_shape rerank
    // distances ~470x off; the fallback never runs under AVX-512, so x86
    // never saw it). Uniform (sg=1) was unaffected.
    float dist = a;
    for (uint32_t d = 0; d < dim; ++d) {
        const uint8_t byte = row[d / 2];
        const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                         : ((byte >> 4) & 0xF);
        const float gf = static_cast<float>(gu8[nib]);
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
