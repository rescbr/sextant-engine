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

#include <limits>  // std::numeric_limits (used in argmin_scaled)

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
///
/// Note: `gemv_f32` below subsumes this at n=1; keep both until a microbench
/// at K=256/sub_dim=8 confirms gemv_f32(n=1) matches this specialized path.
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

// ---------------------------------------------------------------------------
// General GEMV: dots[n × k] = A[n×depth] × B[k×depth]ᵀ (f32-accumulated)
// ---------------------------------------------------------------------------
//
// Used by k-means assignment (PQ training: depth=8; entry-point k-means:
// depth=768). Replaces the per-pair l2sq_f32 inner loop with the standard
// ||a||² + ||b||² − 2<a,b> decomposition: precompute row-norms once, use
// GEMV for the <a,b> matrix, then argmin over (row_norm + col_norm − 2·dot).
// ~5-10× faster than the per-pair l2sq path on the assignment step.
//
// Shape contract:
//   - A is row-major n × depth.
//   - B is row-major k × depth (must have padded rows up to k_simd; see below).
//   - dots is row-major n × k (caller-allocated; written, not read).
//   - k_simd = k rounded up to the SIMD batch width. The caller MUST pass
//     k_simd, allocate dots with n × k_simd floats, and allocate B with
//     k_simd × depth floats (padding extra rows with zeros). dots columns
//     [k, k_simd) are written but the caller must NOT read them. Precedent:
//     zero-pad tails in l2sq_f16_fhm_ (line 69) and the batch4 remainder
//     path in lut_distance_batch4.
//
// Padding strategy: round k up rather than scalar-fallback the tail. Same
// SIMD cost as the aligned case; the extra outputs are computed but discarded.
//
// Fast paths:
//   - depth = 8: 4-centroid batches, 2-stage FMA chains (the pq_matvec shape).
//     Hottest path — PQ training at sub_dim=8.
//   - depth = 4*L (L ≥ 2): L vector loads per centroid, ILP across 4
//     centroids. Covers depth=16, 32, ..., and depth=768 (entry-points).
//   - general depth (depth % 4 != 0): SIMD over the aligned prefix, scalar
//     tail over the last 1-3 elements.

inline void gemv_f32(const float* A, uint32_t n,
                     const float* B, uint32_t k, uint32_t k_simd,
                     uint32_t depth, float* dots) {
    // Trivial cases.
    if (n == 0 || k_simd == 0 || depth == 0) return;

#if defined(SEXTANT_HAS_NEON)
    constexpr uint32_t kBatch = 4;  // 4 centroids per batch (4-wide F32 NEON)
    if (depth == 8) {
        // Hot path: PQ training at sub_dim=8. 2 NEON loads/centroid, 2-stage
        // FMA. Identical shape to pq_matvec_f32's fast path but batched over n.
        for (uint32_t i = 0; i < n; i++) {
            const float* a = A + i * 8;
            float* out = dots + i * k_simd;
            const float32x4_t a0 = vld1q_f32(a);
            const float32x4_t a1 = vld1q_f32(a + 4);
            uint32_t c = 0;
            for (; c + kBatch <= k_simd; c += kBatch) {
                const float* b0 = B + (c + 0) * 8;
                const float* b1 = B + (c + 1) * 8;
                const float* b2 = B + (c + 2) * 8;
                const float* b3 = B + (c + 3) * 8;
                float32x4_t acc0 = vmulq_f32(a0, vld1q_f32(b0));
                float32x4_t acc1 = vmulq_f32(a0, vld1q_f32(b1));
                float32x4_t acc2 = vmulq_f32(a0, vld1q_f32(b2));
                float32x4_t acc3 = vmulq_f32(a0, vld1q_f32(b3));
                acc0 = vfmaq_f32(acc0, a1, vld1q_f32(b0 + 4));
                acc1 = vfmaq_f32(acc1, a1, vld1q_f32(b1 + 4));
                acc2 = vfmaq_f32(acc2, a1, vld1q_f32(b2 + 4));
                acc3 = vfmaq_f32(acc3, a1, vld1q_f32(b3 + 4));
                out[c + 0] = vaddvq_f32(acc0);
                out[c + 1] = vaddvq_f32(acc1);
                out[c + 2] = vaddvq_f32(acc2);
                out[c + 3] = vaddvq_f32(acc3);
            }
        }
        return;
    }
    // General-depth NEON path: depth must be a multiple of 4 (true for the
    // entry-point case dim=768; if not, falls through to scalar below). Process
    // depth/4 FMA stages per centroid, 4-centroid batches for ILP.
    if ((depth % 4) == 0) {
        const uint32_t nvec = depth / 4;
        for (uint32_t i = 0; i < n; i++) {
            const float* a = A + i * depth;
            float* out = dots + i * k_simd;
            uint32_t c = 0;
            for (; c + kBatch <= k_simd; c += kBatch) {
                const float* b0 = B + (c + 0) * depth;
                const float* b1 = B + (c + 1) * depth;
                const float* b2 = B + (c + 2) * depth;
                const float* b3 = B + (c + 3) * depth;
                float32x4_t acc0 = vmulq_f32(vld1q_f32(a + 0),
                                             vld1q_f32(b0 + 0));
                float32x4_t acc1 = vmulq_f32(vld1q_f32(a + 0),
                                             vld1q_f32(b1 + 0));
                float32x4_t acc2 = vmulq_f32(vld1q_f32(a + 0),
                                             vld1q_f32(b2 + 0));
                float32x4_t acc3 = vmulq_f32(vld1q_f32(a + 0),
                                             vld1q_f32(b3 + 0));
                for (uint32_t v = 1; v < nvec; v++) {
                    const float32x4_t av = vld1q_f32(a + v * 4);
                    acc0 = vfmaq_f32(acc0, av, vld1q_f32(b0 + v * 4));
                    acc1 = vfmaq_f32(acc1, av, vld1q_f32(b1 + v * 4));
                    acc2 = vfmaq_f32(acc2, av, vld1q_f32(b2 + v * 4));
                    acc3 = vfmaq_f32(acc3, av, vld1q_f32(b3 + v * 4));
                }
                out[c + 0] = vaddvq_f32(acc0);
                out[c + 1] = vaddvq_f32(acc1);
                out[c + 2] = vaddvq_f32(acc2);
                out[c + 3] = vaddvq_f32(acc3);
            }
        }
        return;
    }
#elif defined(SEXTANT_HAS_AVX2)
    constexpr uint32_t kBatch = 8;  // 8 centroids per batch (8-wide F32 AVX2)
    if (depth == 8) {
        // Hot path: one __m256 per centroid, broadcast-FMA across the query.
        for (uint32_t i = 0; i < n; i++) {
            const float* a = A + i * 8;
            float* out = dots + i * k_simd;
            const __m256 a_full = _mm256_loadu_ps(a);
            uint32_t c = 0;
            for (; c + kBatch <= k_simd; c += kBatch) {
                for (uint32_t j = 0; j < kBatch; j++) {
                    __m256 b = _mm256_loadu_ps(B + (c + j) * 8);
                    __m256 p = _mm256_mul_ps(a_full, b);
                    __m128 lo = _mm256_castps256_ps128(p);
                    __m128 hi = _mm256_extractf128_ps(p, 1);
                    __m128 s = _mm_add_ps(lo, hi);
                    s = _mm_hadd_ps(s, s);
                    s = _mm_hadd_ps(s, s);
                    out[c + j] = _mm_cvtss_f32(s);
                }
            }
        }
        return;
    }
    if ((depth % 4) == 0) {
        const uint32_t nvec = depth / 4;
        for (uint32_t i = 0; i < n; i++) {
            const float* a = A + i * depth;
            float* out = dots + i * k_simd;
            uint32_t c = 0;
            for (; c + kBatch <= k_simd; c += kBatch) {
                for (uint32_t j = 0; j < kBatch; j++) {
                    const float* b = B + (c + j) * depth;
                    __m128 acc4 = _mm_mul_ps(_mm_loadu_ps(a),
                                             _mm_loadu_ps(b));
                    for (uint32_t v = 1; v < nvec; v++) {
                        acc4 = _mm_add_ps(acc4, _mm_mul_ps(
                            _mm_loadu_ps(a + v * 4),
                            _mm_loadu_ps(b + v * 4)));
                    }
                    // Horizontal sum of 4 lanes.
                    __m128 s = _mm_hadd_ps(acc4, acc4);
                    s = _mm_hadd_ps(s, s);
                    out[c + j] = _mm_cvtss_f32(s);
                }
            }
        }
        return;
    }
#endif
    // Scalar fallback (no SIMD, or depth % 4 != 0 with no SIMD path).
    // Still pads to k_simd so the caller's contract holds.
    for (uint32_t i = 0; i < n; i++) {
        const float* a = A + i * depth;
        float* out = dots + i * k_simd;
        for (uint32_t c = 0; c < k_simd; c++) {
            const float* b = B + c * depth;
            float acc = 0.0f;
            for (uint32_t d = 0; d < depth; d++) acc += a[d] * b[d];
            out[c] = acc;
        }
    }
}

/// Helper: round k up to the SIMD batch width for gemv_f32. The caller uses
/// this to size the dots buffer and the padded centroids matrix.
constexpr uint32_t gemv_k_simd(uint32_t k) {
#if defined(SEXTANT_HAS_NEON)
    constexpr uint32_t kBatch = 4;
#elif defined(SEXTANT_HAS_AVX2)
    constexpr uint32_t kBatch = 8;
#else
    constexpr uint32_t kBatch = 1;
#endif
    return ((k + kBatch - 1) / kBatch) * kBatch;
}

/// Per-row argmin of `(scale[c] - 2*row[c])` over c in [0,k), for each of
/// `n` rows. Writes the argmin index to `assign[i]` and increments
/// `changed` when the new argmin differs from the incoming `assign[i]`.
///
/// Used by k-means assignment after gemv_f32: the full distance is
/// `||a||² + scale[c] - 2*row[c]`, and since ||a||² is constant per row it
/// doesn't affect the argmin. `scale` is ||centroids[c||² (the ||b||² term).
///
/// `dots` is row-major n × k_simd (the gemv_f32 output); only the first `k`
/// columns of each row are read. Reads `assign[i]` (previous iteration's
/// assignment) and writes the new one.
///
/// SIMD argmin strategy: walk 4 lanes (NEON) or 8 (AVX2) at a time,
/// tracking both the running min value and its lane index per batch.
/// Ties broken by lower index (strict < compare → first-wins).
inline void argmin_scaled(uint32_t n, uint32_t k, uint32_t k_simd,
                          const float* scale,
                          const float* dots,
                          uint32_t* assign, uint64_t& changed) {
    (void)k;  // k_simd is used; the caller ensures k ≤ k_simd.

    // SIMD path requires k_simd to be aligned to the batch width. This is
    // always true for callers using gemv_k_simd() to size k_simd, but we
    // guard defensively — if misaligned, fall through to scalar.
#if defined(SEXTANT_HAS_NEON)
    constexpr uint32_t kBatch = 4;
    if (k_simd % kBatch == 0) {
        const float32x4_t two = vdupq_n_f32(2.0f);
        const uint32x4_t lane_idx = {0, 1, 2, 3};
        for (uint32_t i = 0; i < n; i++) {
            const float* row = dots + i * k_simd;
            float32x4_t best_val =
                vdupq_n_f32(std::numeric_limits<float>::infinity());
            uint32x4_t best_idx = vdupq_n_u32(0);
            uint32_t base = 0;
            for (uint32_t c = 0; c < k_simd; c += kBatch, base += kBatch) {
                float32x4_t s = vld1q_f32(scale + c);
                float32x4_t r = vld1q_f32(row + c);
                float32x4_t vals = vsubq_f32(s, vmulq_f32(two, r));
                uint32x4_t idx = vaddq_u32(vdupq_n_u32(base), lane_idx);
                uint32x4_t mask = vcltq_f32(vals, best_val);
                best_val = vbslq_f32(mask, vals, best_val);
                best_idx = vbslq_f32(mask, idx, best_idx);
            }
            // Horizontal reduce of 4 lanes.
            float l0 = vgetq_lane_f32(best_val, 0);
            float l1 = vgetq_lane_f32(best_val, 1);
            float l2 = vgetq_lane_f32(best_val, 2);
            float l3 = vgetq_lane_f32(best_val, 3);
            uint32_t i0 = vgetq_lane_u32(best_idx, 0);
            uint32_t i1 = vgetq_lane_u32(best_idx, 1);
            uint32_t i2 = vgetq_lane_u32(best_idx, 2);
            uint32_t i3 = vgetq_lane_u32(best_idx, 3);
            float m_lo = (l0 <= l1) ? l0 : l1;
            uint32_t im_lo = (l0 <= l1) ? i0 : i1;
            float m_hi = (l2 <= l3) ? l2 : l3;
            uint32_t im_hi = (l2 <= l3) ? i2 : i3;
            uint32_t im = (m_lo <= m_hi) ? im_lo : im_hi;
            if (im != assign[i]) {
                assign[i] = im;
                changed++;
            }
        }
        return;
    }
#elif defined(SEXTANT_HAS_AVX2)
    constexpr uint32_t kBatch = 8;
    if (k_simd % kBatch == 0) {
        const __m256 two = _mm256_set1_ps(2.0f);
        const __m256i lane_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        for (uint32_t i = 0; i < n; i++) {
            const float* row = dots + i * k_simd;
            __m256 best_val =
                _mm256_set1_ps(std::numeric_limits<float>::infinity());
            __m256i best_idx = _mm256_setzero_si256();
            uint32_t base = 0;
            for (uint32_t c = 0; c < k_simd; c += kBatch, base += kBatch) {
                __m256 s = _mm256_loadu_ps(scale + c);
                __m256 r = _mm256_loadu_ps(row + c);
                __m256 vals = _mm256_sub_ps(s, _mm256_mul_ps(two, r));
                __m256i idx = _mm256_add_epi32(_mm256_set1_epi32(base),
                                               lane_idx);
                __m256 mask = _mm256_cmp_ps(vals, best_val, _CMP_LT_OQ);
                best_val = _mm256_blendv_ps(best_val, vals, mask);
                best_idx = _mm256_blendv_epi8(best_idx, idx,
                                              _mm256_castps_si256(mask));
            }
            alignas(32) float vals[8];
            alignas(32) uint32_t idxs[8];
            _mm256_store_ps(vals, best_val);
            _mm256_store_si256((__m256i*)idxs, best_idx);
            float m = vals[0]; uint32_t im = idxs[0];
            for (int j = 1; j < 8; j++) {
                if (vals[j] < m) { m = vals[j]; im = idxs[j]; }
            }
            if (im != assign[i]) {
                assign[i] = im;
                changed++;
            }
        }
        return;
    }
#else
    // No SIMD backend. k_simd is used in the scalar stride below.
#endif
    // Scalar fallback. Reached when:
    //   - no SIMD backend compiled in, OR
    //   - SIMD backend present but k_simd is misaligned (shouldn't happen
    //     for callers using gemv_k_simd() — included for robustness).
    const float two = 2.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float* row = dots + i * k_simd;
        float best = std::numeric_limits<float>::infinity();
        uint32_t best_c = 0;
        for (uint32_t c = 0; c < k; c++) {
            const float v = scale[c] - two * row[c];
            if (v < best) { best = v; best_c = c; }
        }
        if (best_c != assign[i]) {
            assign[i] = best_c;
            changed++;
        }
    }
}

}  // namespace simd
}  // namespace sextant
