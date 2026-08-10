/// SVE2 decode-dot kernel for scalar_lloydmax.
///
/// COMPILED ONLY WHEN SVE2 IS DETECTED (meson guards this TU with
/// -march=armv8-a+sve2). Do NOT include from non-SVE2 code — the dispatcher
/// in simd_kernels.hpp handles the fallback to the NEON kernel.
///
/// The kernel decodes 4-bit packed-nibble codes into per-dim level values,
/// then FMA with the query. The dominant cost in the NEON path is the
/// scalar level-table load per dim (levels[d*K + nibble]); SVE's
/// svld1_gather_u32index_f32 issues a hardware-prefetched vector gather that
/// the Neoverse-V2 memory subsystem services far more efficiently than
/// scalar loads. Nibble extraction stays scalar (cheap, sequential byte
/// reads); the win is the float gather + FMA vectorization.
///
/// Tested on c4a (Neoverse-V2). NOT compilable on Apple M4 (no SVE).

#include "sve2_kernels.hpp"

#if defined(SEXTANT_HAS_SVE2_KERNEL) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#include <cstdint>

namespace sextant::simd {

float scalar_dot_u4_sve2_impl(const float* query,
                               const float* levels,
                               const uint8_t* code,
                               uint32_t dim, uint32_t K) {
    svfloat32_t vacc = svdup_n_f32(0.0f);
    const uint32_t vl = svcntw();   // active lanes per SVE vector (e.g. 4 on 128-bit)

    // Stage buffer: per-lane (dd * K + nibble) indices for the current tile.
    // VL is known at runtime but bounded (≤ 16 for 512-bit SVE); a small
    // fixed buffer suffices.
    uint32_t idx_buf[16];

    uint32_t d = 0;
    while (d < dim) {
        const uint32_t lane_n = (d + vl <= dim) ? vl : (dim - d);
        const svbool_t pg = svwhilelt_b32_u32(d, dim);

        // Scalar nibble decode into the index buffer (sequential byte reads,
        // cheap). Lane i → dim dd = d + i.
        for (uint32_t i = 0; i < lane_n; ++i) {
            const uint32_t dd = d + i;
            const uint8_t byte = code[dd >> 1];
            const uint32_t nib = (dd & 1u) ? (byte >> 4) : (byte & 0x0F);
            idx_buf[i] = dd * K + nib;
        }

        // Vector gather of decoded levels + query, FMA.
        svuint32_t idx = svld1_u32(pg, idx_buf);
        svfloat32_t q = svld1_f32(pg, query + d);
        svfloat32_t lv = svld1_gather_u32index_f32(pg, levels, idx);
        vacc = svmla_f32_x(pg, vacc, q, lv);

        d += vl;
    }

    // Horizontal sum across all lanes.
    return svaddv_f32(svptrue_b32(), vacc);
}

}  // namespace sextant::simd

#endif  // SEXTANT_HAS_SVE2_KERNEL && __ARM_FEATURE_SVE
