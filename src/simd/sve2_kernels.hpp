#pragma once

/// @file sve2_kernels.hpp
/// SVE2-accelerated decode-dot kernel for scalar_lloydmax.
///
/// This is the ONLY translation unit that includes <arm_sve.h> (at file
/// scope — clang 18 cannot handle the SVE header inside a function body).
/// It is compiled as a separate static_library with -march=armv8-a+sve2 in
/// meson.build, ONLY when SVE2 is detected at configure time. On non-SVE2
/// hardware (e.g. Apple M4, which has NEON+I8MM but no SVE), this TU is
/// excluded entirely and the dispatcher falls back to the NEON kernel
/// (scalar_dot_u4_float in simd_kernels.hpp).
///
/// Runtime dispatch: simd_kernels.hpp's scalar_dot_u4_sve2() calls
/// scalar_dot_u4_sve2_impl() when SEXTANT_HAS_SVE2_KERNEL is defined,
/// otherwise it calls scalar_dot_u4_float (NEON). The compile-time macro
/// avoids any indirect-call overhead.

#include <cstdint>

namespace sextant::simd {

#if defined(SEXTANT_HAS_SVE2_KERNEL)
/// SVE2 decode-dot: decode 4-bit codes via gather-load of per-dim levels,
/// FMA with query. Vector-length-agnostic (uses svcntw() lane count).
/// `levels` is [dim][K] floats; `code` is (dim*4+7)/8 packed-nibble bytes.
float scalar_dot_u4_sve2_impl(const float* query,
                               const float* levels,
                               const uint8_t* code,
                               uint32_t dim, uint32_t K);
#endif

}  // namespace sextant::simd
