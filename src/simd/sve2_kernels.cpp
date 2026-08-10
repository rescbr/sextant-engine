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
    const uint32_t vl = svcntw();   // active lanes per SVE vector (e.g. 4 on 128-bit V2)

    // Stage buffer: per-lane (dd * K + nibble) indices for the current tile.
    // VL is runtime-constant; allocate once. SVE f32 VL is at most 16 (512-bit).
    // Size defensively to 16 + assert; V2 (128-bit) uses VL=4.
    uint32_t idx_buf[16];
    if (vl > 16) return 0.0f;  // unreachable on current hardware; defensive

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

/// SVE2 batch decode-dot for scalar_lloydmax main scan.
///
/// For each vector i in [0, count): decode its packed-nibble codes, gather
/// the per-dim levels, and FMA with the query into an accumulator, then
/// horizontal-reduce to out[i]. Same nibble-decode + gather-FMA inner loop
/// as scalar_dot_u4_sve2_impl (the verified rerank kernel); only the outer
/// loop (per-vector instead of single) differs.
///
/// Compiles ONLY on SVE2 hardware (Neoverse-V2 / c4a). The dispatcher in
/// simd_kernels.hpp falls back to the NEON kernel (scalar_dot_u4_float per
/// vector) on non-SVE2 hardware (e.g. Apple M4).
void scalar_dot_u4_sve2_batch(const float* query,
                                const float* levels,
                                const uint8_t* codes, uint32_t count,
                                uint32_t dim, uint32_t K, uint32_t code_size,
                                float* out) {
    const uint32_t vl = svcntw();   // active lanes per SVE vector (4 on 128-bit V2)
    // Stage buffer: per-lane (dd * K + nibble) indices for the current tile.
    // SVE f32 VL is at most 16 (512-bit). Size defensively to 16 + assert.
    uint32_t idx_buf[16];
    if (vl > 16) {
        // Unreachable on current hardware; defensive — fall back to zeroed out.
        for (uint32_t i = 0; i < count; ++i) out[i] = 0.0f;
        return;
    }

    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* code = codes + static_cast<uint64_t>(i) * code_size;
        svfloat32_t vacc = svdup_n_f32(0.0f);

        uint32_t d = 0;
        while (d < dim) {
            const uint32_t lane_n = (d + vl <= dim) ? vl : (dim - d);
            const svbool_t pg = svwhilelt_b32_u32(d, dim);

            // Scalar nibble decode into the index buffer (sequential byte
            // reads, cheap). Lane j -> dim dd = d + j.
            for (uint32_t j = 0; j < lane_n; ++j) {
                const uint32_t dd = d + j;
                const uint8_t byte = code[dd >> 1];
                const uint32_t nib = (dd & 1u) ? (byte >> 4) : (byte & 0x0F);
                idx_buf[j] = dd * K + nib;
            }

            // Vector gather of decoded levels + query, FMA.
            svuint32_t idx = svld1_u32(pg, idx_buf);
            svfloat32_t q = svld1_f32(pg, query + d);
            svfloat32_t lv = svld1_gather_u32index_f32(pg, levels, idx);
            vacc = svmla_f32_x(pg, vacc, q, lv);

            d += vl;
        }

        out[i] = svaddv_f32(svptrue_b32(), vacc);
    }
}

}  // namespace sextant::simd

#endif  // SEXTANT_HAS_SVE2_KERNEL && __ARM_FEATURE_SVE
