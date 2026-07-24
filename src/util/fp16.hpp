#pragma once

/// @file fp16.hpp
/// SIMD-accelerated FP32 → FP16 array conversion.
///
/// Used at build time (encode pass, partitioned shard copy, entry-point
/// k-means centroid snap) where we cast large streams of FP32 vectors to
/// FP16 for the FP16 prune. The scalar fallback
/// (per-element `static_cast<float16_t>`) maxes one core at ~2 GB/s and is
/// audible as coil whine on laptops; the SIMD paths hit ~16 GB/s.
///
/// These are pure elementwise casts with no inter-element dependencies —
/// the easiest possible SIMD target. Round-to-nearest-even matches the
/// hardware instruction semantics on both platforms.
///
/// Platform support:
///   - ARM NEON (`vcvt_f16_f32`): defined whenever `<arm_neon.h>` is usable,
///     which is every arm64 target. Always available on our GCP Axion build.
///   - x86 F16C (`_mm256_cvtps_ph`): available on every Intel/AMD server
///     since Haswell (2013) / Piledriver (2012). Detected via `__F16C__`
///     which is defined by `-mf16c` OR by `-march=native` on any modern CPU.
///     gcp_bench.sh passes `-mf16c` explicitly on amd64.
///   - Scalar fallback: only on truly ancient x86 without F16C, or if
///     someone explicitly passes `-mno-f16c`. This is a perf regression;
///     the static_assert below flags it at build time.

#include <cstdint>
#include <cstddef>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SEXTANT_FP16_HAS_NEON 1
#elif defined(__F16C__)
#include <immintrin.h>
#define SEXTANT_FP16_HAS_F16C 1
#endif

namespace sextant {

/// Cast `n` FP32 values at `src` to `dst` (FP16). `n` need not be a multiple
/// of the SIMD width; the tail is handled element-by-element.
inline void cast_fp32_to_fp16(const float* src, float16_t* dst, size_t n) {
#if defined(SEXTANT_FP16_HAS_NEON)
    // vcvt_f16_f32 converts a 4-lane FP32 vector to a 4-lane FP16 vector
    // (single instruction, FCVT).
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vst1_f16(dst + i, vcvt_f16_f32(v));
    }
    for (; i < n; i++) {
        dst[i] = static_cast<float16_t>(src[i]);
    }
#elif defined(SEXTANT_FP16_HAS_F16C)
    // _mm256_cvtps_ph converts 8 FP32 → 8 FP16 in one instruction (VCVTPS2PH).
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
    }
    if (i + 4 <= n) {
        __m128 v = _mm_loadu_ps(src + i);
        __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), h);
        i += 4;
    }
    for (; i < n; i++) {
        dst[i] = static_cast<float16_t>(src[i]);
    }
#else
// Scalar fallback — no SIMD FP16 cast available. This is a real perf
// regression (~8× slower than the SIMD paths). If you see this, add
// -mf16c (x86) or build for an arm64 target.
#warning "FP32→FP16 cast using scalar fallback; add -mf16c or target arm64"
    for (size_t i = 0; i < n; i++) {
        dst[i] = static_cast<float16_t>(src[i]);
    }
#endif
}

}  // namespace sextant

