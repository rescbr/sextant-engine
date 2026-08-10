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
/// Processes vectors in groups of 4 concurrently. This mirrors the scalar
/// batch-4 loop (ivf_tree_index.cpp scan_one_leaf) and the NEON batch-4
/// kernel (scalar_dot_u4_float_batch): 4 independent accumulator chains per
/// tile, sharing the per-tile query load. The previous per-vector SVE2 batch
/// re-gathered the query for every vector, losing the query-load amortization
/// that makes the scalar/NEON batch-4 loop fast — on V2 (VL=4) the per-vector
/// gather overhead exceeded the scalar batch-4 win (95 QPS vs 216 QPS).
///
/// Per group of 4 vectors and per dim-tile (VL dims):
///   1 query gather  (shared across all 4 vectors),
///   4 level gathers (one per vector, hardware-prefetched by V2),
///   4 svmla         (4 independent accumulator chains → full FMA throughput).
///
/// The 4 independent accumulators fill the SVE FMA pipeline (no data
/// dependency between them), and the query gather is amortized 4×. The level
/// gathers — 4× per vector per tile in the scalar loop, the expensive part —
/// become SVE hardware-prefetched gathers.
///
/// The nibble decode is identical to scalar_dot_u4_sve2_impl (the verified
/// rerank kernel); only the 4-vector batching is new. Each vector's dot is
/// the same sum over d of query[d]*levels[d*K + nibble_i_d] — horizontally
/// reduced per tile then summed, matching the rerank kernel's lane order.
///
/// Compiles ONLY on SVE2 hardware (Neoverse-V2 / c4a). The dispatcher in
/// simd_kernels.hpp falls back to the NEON kernel (scalar_dot_u4_float_batch)
/// on non-SVE2 hardware (e.g. Apple M4).
void scalar_dot_u4_sve2_batch(const float* query,
                                const float* levels,
                                const uint8_t* codes, uint32_t count,
                                uint32_t dim, uint32_t K, uint32_t code_size,
                                float* out) {
    const uint32_t vl = svcntw();   // active lanes per SVE vector (4 on 128-bit V2)
    // 4 index buffers, one per concurrent vector, each VL entries. SVE f32 VL
    // is at most 16 (512-bit). Size defensively to 16 + assert; V2 uses VL=4.
    uint32_t idx_buf[4][16];
    if (vl > 16) {
        // Unreachable on current hardware; defensive — fall back to zeroed out.
        for (uint32_t i = 0; i < count; ++i) out[i] = 0.0f;
        return;
    }

    // Full groups of 4 vectors: 4 independent accumulators, shared query load.
    uint32_t i = 0;
    for (; i + 3 < count; i += 4) {
        const uint8_t* cp[4] = {
            codes + static_cast<uint64_t>(i)     * code_size,
            codes + static_cast<uint64_t>(i + 1) * code_size,
            codes + static_cast<uint64_t>(i + 2) * code_size,
            codes + static_cast<uint64_t>(i + 3) * code_size,
        };
        svfloat32_t vacc0 = svdup_n_f32(0.0f);
        svfloat32_t vacc1 = svdup_n_f32(0.0f);
        svfloat32_t vacc2 = svdup_n_f32(0.0f);
        svfloat32_t vacc3 = svdup_n_f32(0.0f);

        uint32_t d = 0;
        while (d < dim) {
            const uint32_t lane_n = (d + vl <= dim) ? vl : (dim - d);
            const svbool_t pg = svwhilelt_b32_u32(d, dim);

            // Scalar nibble decode: 4 vectors, each VL lanes (sequential byte
            // reads, cheap). Lane j -> dim dd = d + j.
            for (uint32_t j = 0; j < lane_n; ++j) {
                const uint32_t dd = d + j;
                const uint8_t shift = (dd & 1u) ? 4 : 0;
                idx_buf[0][j] = dd * K + ((cp[0][dd >> 1] >> shift) & 0x0F);
                idx_buf[1][j] = dd * K + ((cp[1][dd >> 1] >> shift) & 0x0F);
                idx_buf[2][j] = dd * K + ((cp[2][dd >> 1] >> shift) & 0x0F);
                idx_buf[3][j] = dd * K + ((cp[3][dd >> 1] >> shift) & 0x0F);
            }

            // Shared query gather for this tile.
            const svfloat32_t q = svld1_f32(pg, query + d);

            // 4 level gathers + 4 independent FMAs.
            svuint32_t idx0 = svld1_u32(pg, idx_buf[0]);
            svuint32_t idx1 = svld1_u32(pg, idx_buf[1]);
            svuint32_t idx2 = svld1_u32(pg, idx_buf[2]);
            svuint32_t idx3 = svld1_u32(pg, idx_buf[3]);
            vacc0 = svmla_f32_x(pg, vacc0, q,
                                svld1_gather_u32index_f32(pg, levels, idx0));
            vacc1 = svmla_f32_x(pg, vacc1, q,
                                svld1_gather_u32index_f32(pg, levels, idx1));
            vacc2 = svmla_f32_x(pg, vacc2, q,
                                svld1_gather_u32index_f32(pg, levels, idx2));
            vacc3 = svmla_f32_x(pg, vacc3, q,
                                svld1_gather_u32index_f32(pg, levels, idx3));

            d += vl;
        }

        out[i]     = svaddv_f32(svptrue_b32(), vacc0);
        out[i + 1] = svaddv_f32(svptrue_b32(), vacc1);
        out[i + 2] = svaddv_f32(svptrue_b32(), vacc2);
        out[i + 3] = svaddv_f32(svptrue_b32(), vacc3);
    }

    // Tail (count not multiple of 4): fall back to the verified single-vector
    // rerank kernel. Correct and avoids a second predicated path.
    for (; i < count; ++i) {
        const uint8_t* code = codes + static_cast<uint64_t>(i) * code_size;
        out[i] = scalar_dot_u4_sve2_impl(query, levels, code, dim, K);
    }
}

}  // namespace sextant::simd

#endif  // SEXTANT_HAS_SVE2_KERNEL && __ARM_FEATURE_SVE
