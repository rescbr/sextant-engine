#pragma once

/// @file simd_kernels.hpp
/// Hand-written SIMD distance/dot kernels used across the engine, benchmark,
/// and tools. Consolidated here so every site gets the same optimized paths.
///
/// Why not NumKong for these? NumKong's `_f32` kernels (nk_sqeuclidean_f32,
/// nk_dot_f32, nk_dots_packed_f32) accumulate in f64 (output `nk_f64_t`) for
/// bit-for-bit reference matching. On NEON f64 is 2-wide vs f32's 4-wide —
/// half the throughput. For ranking and LUT construction, f32 precision is
/// more than sufficient. These kernels accumulate in f32 and are ~2-3× faster
/// than NumKong's f64-accumulating equivalents at the shapes we care about
/// (dim=768 FP32, sub_dim=8 PQ, 1×K×sub_dim matvec).
///
/// Ports: NEON today (Apple Silicon, c4a Axion Neoverse-V2 in AArch64 mode).
/// SVE and AVX2 variants follow the same shape; add when profiling on those
/// targets shows the NEON path is the bottleneck (V2 has SVE2 — 4-wide f32
/// with predicate; AVX2 has 8-wide f32).

#include <cstdint>
#include <sextant/types.hpp>

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SEXTANT_HAS_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define SEXTANT_HAS_AVX2 1
#endif

namespace sextant {
namespace simd {

// ---------------------------------------------------------------------------
// FP16 kernels (FHM on ARM, F16C on x86)
// ---------------------------------------------------------------------------

/// L2-squared distance between two FP16 vectors. FP32 accumulation (no
/// overflow), FP16 per-element precision. Used by the FP16 prune occlusion
/// check, the partitioned-build merge truncation, and IVF routing.
///
/// ARM NEON path: prefers FEAT_FHM (`vfmlalq_low/high_f16`) when available —
/// the FHM instructions fuse FP16→FP32 widening with multiply-accumulate and
/// process 8 FP16 elements per pair of instructions, ~halving the instruction
/// count of the 4-wide `vcvt_f32_f16` + FP32 `vfmaq` path. This is the #1 hot
/// spot in beam_search (profiled at ~94% of search time via the inlined
/// convert+FMA at +7520 in beam_search_into). The FHM function carries a
/// per-function `target` attribute so it compiles even when the global -march
/// doesn't include fp16fml; the call dispatch is a compile-time constant.
/// Validated bit-equivalent to the 4-wide path (max rel err 9.8e-5 over 10k
/// random 768-dim trials — the only difference is FP32 accumulation order).
inline float l2sq_f16(const float16_t* a, const float16_t* b, uint32_t dim);

#if defined(SEXTANT_HAS_NEON)
/// FEAT_FHM 8-wide L2-squared. Per-function target attribute (numkong pattern)
/// enables fp16fml codegen without a global -march change.
__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml")))
inline float l2sq_f16_fhm_(const float16_t* a, const float16_t* b, uint32_t dim) {
    float32x4_t acc_lo = vdupq_n_f32(0.0f);
    float32x4_t acc_hi = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        float16x8_t va = vld1q_f16(reinterpret_cast<const __fp16*>(a + i));
        float16x8_t vb = vld1q_f16(reinterpret_cast<const __fp16*>(b + i));
        float16x8_t diff = vsubq_f16(va, vb);
        acc_lo = vfmlalq_low_f16(acc_lo, diff, diff);
        acc_hi = vfmlalq_high_f16(acc_hi, diff, diff);
    }
    // Remainder (0-7 elements): zero-pad into a full 8-lane FP16 vector and run
    // one more FHM pair. Padding with 0 contributes (0-0)^2 = 0 to the sum, so
    // the result stays exact for any dim (not just dim % 8 == 0). This avoids a
    // scalar tail loop (the dim=768 production case is always %8==0 anyway).
    if (i < dim) {
        __fp16 tmp_a[8] = {0}, tmp_b[8] = {0};
        const uint32_t rem = dim - i;
        for (uint32_t j = 0; j < rem; j++) {
            tmp_a[j] = a[i + j];
            tmp_b[j] = b[i + j];
        }
        float16x8_t va = vld1q_f16(tmp_a);
        float16x8_t vb = vld1q_f16(tmp_b);
        float16x8_t diff = vsubq_f16(va, vb);
        acc_lo = vfmlalq_low_f16(acc_lo, diff, diff);
        acc_hi = vfmlalq_high_f16(acc_hi, diff, diff);
    }
    return vaddvq_f32(vaddq_f32(acc_lo, acc_hi));
}
#endif

inline float l2sq_f16(const float16_t* a, const float16_t* b, uint32_t dim) {
#if defined(SEXTANT_HAS_NEON)
    // FEAT_FHM is available on all our aarch64 targets (Apple Silicon, c4a
    // Axion). The per-function target attribute on l2sq_f16_fhm_ makes this
    // safe regardless of the global -march setting.
    return l2sq_f16_fhm_(a, b, dim);
#elif defined(SEXTANT_HAS_AVX2) && defined(__F16C__)
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
        __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float r = _mm_cvtss_f32(sum);
    for (; i < dim; i++) { float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]); r += diff * diff; }
    return r;
#else
    float r = 0.0f;
    for (uint32_t i = 0; i < dim; i++) { float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]); r += diff * diff; }
    return r;
#endif
}

/// Inner product (dot product) between two FP16 vectors. The IP counterpart to
/// `l2sq_f16`: same FHM 8-wide structure, but skips the subtract (one fewer
/// FP16 vector op per 8 elements). For L2-normalized data, IP ranking is
/// equivalent to L2sq ranking on TRUE distances (`‖q−x‖² = 2 − 2⟨q,x⟩`), so
/// this is the cheaper drop-in for normalized-data configs. Used by the same
/// FP16 paths (routing, rerank, occlusion) when the configured metric is
/// InnerProduct.
inline float dot_f16(const float16_t* a, const float16_t* b, uint32_t dim);

#if defined(SEXTANT_HAS_NEON)
/// FEAT_FHM 8-wide dot product. Mirrors `l2sq_f16_fhm_` without the subtract.
__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml")))
inline float dot_f16_fhm_(const float16_t* a, const float16_t* b, uint32_t dim) {
    float32x4_t acc_lo = vdupq_n_f32(0.0f);
    float32x4_t acc_hi = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        float16x8_t va = vld1q_f16(reinterpret_cast<const __fp16*>(a + i));
        float16x8_t vb = vld1q_f16(reinterpret_cast<const __fp16*>(b + i));
        acc_lo = vfmlalq_low_f16(acc_lo, va, vb);
        acc_hi = vfmlalq_high_f16(acc_hi, va, vb);
    }
    // Remainder (0-7 elements): zero-pad into a full 8-lane FP16 vector and run
    // one more FHM pair. Padding with 0 contributes 0*a = 0 to the sum, so the
    // result stays exact for any dim (not just dim % 8 == 0).
    if (i < dim) {
        __fp16 tmp_a[8] = {0}, tmp_b[8] = {0};
        const uint32_t rem = dim - i;
        for (uint32_t j = 0; j < rem; j++) {
            tmp_a[j] = a[i + j];
            tmp_b[j] = b[i + j];
        }
        float16x8_t va = vld1q_f16(tmp_a);
        float16x8_t vb = vld1q_f16(tmp_b);
        acc_lo = vfmlalq_low_f16(acc_lo, va, vb);
        acc_hi = vfmlalq_high_f16(acc_hi, va, vb);
    }
    return vaddvq_f32(vaddq_f32(acc_lo, acc_hi));
}
#endif

inline float dot_f16(const float16_t* a, const float16_t* b, uint32_t dim) {
#if defined(SEXTANT_HAS_NEON)
    return dot_f16_fhm_(a, b, dim);
#elif defined(SEXTANT_HAS_AVX2) && defined(__F16C__)
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
        __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float r = _mm_cvtss_f32(sum);
    for (; i < dim; i++) { r += static_cast<float>(a[i]) * static_cast<float>(b[i]); }
    return r;
#else
    float r = 0.0f;
    for (uint32_t i = 0; i < dim; i++) { r += static_cast<float>(a[i]) * static_cast<float>(b[i]); }
    return r;
#endif
}

/// Dispatch helper: distance between two FP16 vectors under a metric.
/// Centralizes the metric switch so call sites don't repeat it. Returns a value
/// ordered consistently with a min-heap (L2sq: ascending; IP: negated, so
/// "smaller" = higher dot product = nearer).
inline float dist_f16(MetricKind metric, const float16_t* a, const float16_t* b, uint32_t dim) {
    switch (metric) {
    case MetricKind::InnerProduct:
        return -dot_f16(a, b, dim);
    case MetricKind::L2Sq:
    default:
        return l2sq_f16(a, b, dim);
    }
}

// ---------------------------------------------------------------------------
// FP32 kernels (NEON FMA 4-wide / AVX2 FMA 8-wide)
// ---------------------------------------------------------------------------

/// L2-squared distance between two FP32 vectors. NEON FMA (4-wide) or AVX2
/// FMA (8-wide). Used by the FP32 build mode (SEXTANT_FP32_BUILD=1) where
/// exact distances are preferred over PQ LUT for graph construction, and by
/// the rerank paths in the benchmark and estimator. f32-accumulated — ~2× the
/// throughput of NumKong's f64-accumulating `nk_sqeuclidean_f32`.
inline float l2sq_f32(const float* a, const float* b, uint32_t dim) {
#if defined(SEXTANT_HAS_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }
    float r = vaddvq_f32(acc);
    for (; i < dim; i++) { float diff = a[i] - b[i]; r += diff * diff; }
    return r;
#elif defined(SEXTANT_HAS_AVX2)
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float r = _mm_cvtss_f32(sum);
    for (; i < dim; i++) { float diff = a[i] - b[i]; r += diff * diff; }
    return r;
#else
    float r = 0.0f;
    for (uint32_t i = 0; i < dim; i++) { float diff = a[i] - b[i]; r += diff * diff; }
    return r;
#endif
}

/// Inner product (dot product) between two FP32 vectors. The IP counterpart to
/// `l2sq_f32`: same FMA structure, without the subtract. Used by rerank and the
/// FP32 build mode when the configured metric is InnerProduct. f32-accumulated
/// — ~2× the throughput of NumKong's f64-accumulating `nk_dot_f32`.
inline float dot_f32(const float* a, const float* b, uint32_t dim) {
#if defined(SEXTANT_HAS_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb);
    }
    float r = vaddvq_f32(acc);
    for (; i < dim; i++) { r += a[i] * b[i]; }
    return r;
#elif defined(SEXTANT_HAS_AVX2)
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float r = _mm_cvtss_f32(sum);
    for (; i < dim; i++) { r += a[i] * b[i]; }
    return r;
#else
    float r = 0.0f;
    for (uint32_t i = 0; i < dim; i++) { r += a[i] * b[i]; }
    return r;
#endif
}

/// Dispatch helper: distance between two FP32 vectors under a metric.
/// Returns a value ordered consistently with a min-heap (L2sq: ascending;
/// IP: negated, so "smaller" = higher dot product = nearer).
inline float dist_f32(MetricKind metric, const float* a, const float* b, uint32_t dim) {
    switch (metric) {
    case MetricKind::InnerProduct:
        return -dot_f32(a, b, dim);
    case MetricKind::L2Sq:
    default:
        return l2sq_f32(a, b, dim);
    }
}

// ---------------------------------------------------------------------------
// PQ matvec: one query-sub vs K centroids of depth sub_dim (f32-accumulated)
// ---------------------------------------------------------------------------

/// Compute `out[c] = <query, centroids[c]>` for c in 0..K, f32-accumulated.
/// `query` points to `sub_dim` floats. `centroids` is row-major K × sub_dim.
/// `out` is K floats. K must be > 0; sub_dim must be > 0.
///
/// Used by `PqQuantizer::preprocess_query_as` to build the PQ LUT in m matvecs
/// instead of m*K per-entry NumKong calls. ~3× faster than the original
/// per-entry path and ~1.8× faster than `nk_dots_packed_f32` (which
/// accumulates in f64) at K=256, sub_dim=8.
///
/// The implementations trade off ILP vs register pressure for the shallow
/// depths (sub_dim 4-16) typical of PQ. At sub_dim=8, K=256: 2 fmla per
/// centroid, 64 batches of 4 centroids using 4 independent FMA chains.
inline void pq_matvec_f32(const float* query, const float* centroids,
                          float* out, uint32_t K, uint32_t sub_dim) {
#if defined(SEXTANT_HAS_NEON)
    if (sub_dim == 8 && (K % 4) == 0) {
        // Fast path: depth 8 (4 lanes × 2 stages), 4-centroid batches.
        // 4 independent FMA chains hide latency; 2 query loads amortized
        // across 4×2 centroid loads per batch.
        const float32x4_t q0 = vld1q_f32(query);
        const float32x4_t q1 = vld1q_f32(query + 4);
        uint32_t c = 0;
        for (; c + 4 <= K; c += 4) {
            const float* cen0 = centroids + (c + 0) * 8;
            const float* cen1 = centroids + (c + 1) * 8;
            const float* cen2 = centroids + (c + 2) * 8;
            const float* cen3 = centroids + (c + 3) * 8;
            float32x4_t c0_lo = vld1q_f32(cen0);
            float32x4_t c1_lo = vld1q_f32(cen1);
            float32x4_t c2_lo = vld1q_f32(cen2);
            float32x4_t c3_lo = vld1q_f32(cen3);
            float32x4_t acc0 = vmulq_f32(q0, c0_lo);
            float32x4_t acc1 = vmulq_f32(q0, c1_lo);
            float32x4_t acc2 = vmulq_f32(q0, c2_lo);
            float32x4_t acc3 = vmulq_f32(q0, c3_lo);
            float32x4_t c0_hi = vld1q_f32(cen0 + 4);
            float32x4_t c1_hi = vld1q_f32(cen1 + 4);
            float32x4_t c2_hi = vld1q_f32(cen2 + 4);
            float32x4_t c3_hi = vld1q_f32(cen3 + 4);
            acc0 = vfmaq_f32(acc0, q1, c0_hi);
            acc1 = vfmaq_f32(acc1, q1, c1_hi);
            acc2 = vfmaq_f32(acc2, q1, c2_hi);
            acc3 = vfmaq_f32(acc3, q1, c3_hi);
            out[c + 0] = vaddvq_f32(acc0);
            out[c + 1] = vaddvq_f32(acc1);
            out[c + 2] = vaddvq_f32(acc2);
            out[c + 3] = vaddvq_f32(acc3);
        }
        return;
    }
#elif defined(SEXTANT_HAS_AVX2)
    if (sub_dim == 8 && (K % 8) == 0) {
        // Fast path: depth 8 (8 lanes × 1 stage via broadcast-FMA, plus the
        // remaining 4 lanes via a second __m128). 8-centroid batches.
        const __m256 q_full = _mm256_loadu_ps(query);
        for (uint32_t c = 0; c < K; c += 8) {
            for (uint32_t i = 0; i < 8; i++) {
                __m256 cen = _mm256_loadu_ps(centroids + (c + i) * 8);
                __m256 prod = _mm256_mul_ps(q_full, cen);
                __m128 lo = _mm256_castps256_ps128(prod);
                __m128 hi = _mm256_extractf128_ps(prod, 1);
                __m128 sum = _mm_add_ps(lo, hi);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                out[c + i] = _mm_cvtss_f32(sum);
            }
        }
        return;
    }
#endif
    // Scalar fallback (any sub_dim or K; no SIMD or shape mismatch).
    for (uint32_t c = 0; c < K; c++) {
        const float* cen = centroids + c * sub_dim;
        float acc = 0.0f;
        for (uint32_t d = 0; d < sub_dim; d++) acc += query[d] * cen[d];
        out[c] = acc;
    }
}

}  // namespace simd
}  // namespace sextant
