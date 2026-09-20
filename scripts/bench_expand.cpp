// Microbench: does pre-expanding (and g-mapping) the 4-bit codes into
// 1 byte/dim pay off? Measures, on identical logical work:
//   A) the REAL engine kernel  scalar_i8_dots4_q4 over packed nibbles
//   B) pure NEON USDOT stream   over pre-expanded g-mapped bytes
//   C) pure SVE2 svdot stream   over pre-expanded bytes (256-bit SVE HW)
// Same rows, same queries, same dot count. Shaped family (g-table), the
// culturaX/cohere configuration.
#include "tree/coders/coder_util.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#elif defined(SEXTANT_HAS_AVX512_SCAN)
#include <immintrin.h>
#endif

using namespace sextant::tree::coders;

[[maybe_unused]] static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    const uint32_t dim = 768;
    const uint32_t cs = dim / 2;               // packed bytes/row
    const uint32_t n_rows = 1u << 16;          // 64k rows
    [[maybe_unused]] const double run_s = 1.0;

    // Synthetic codes + g-table.
    std::vector<uint8_t> rows(static_cast<size_t>(n_rows) * cs);
    srand(1);
    for (auto& b : rows) b = rand() & 0xFF;
    uint8_t g_tbl[16];
    for (uint32_t k = 0; k < 16; ++k) g_tbl[k] = static_cast<uint8_t>(rand());

    // Pre-expanded g-mapped bytes (what a cache-layer fill would store).
    std::vector<uint8_t> expanded(static_cast<size_t>(n_rows) * dim);
    for (uint32_t r = 0; r < n_rows; ++r)
        for (uint32_t d = 0; d < dim; ++d) {
            const uint8_t byte = rows[r * cs + d / 2];
            const uint8_t nib = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            expanded[static_cast<size_t>(r) * dim + d] = g_tbl[nib];
        }

    // 4 query contexts (shaped, single-precision i8 mode 1).
    std::vector<ScalarScanCtx> ctx(4);
    std::vector<std::vector<int8_t>> a8(4, std::vector<int8_t>(dim));
    for (uint32_t q = 0; q < 4; ++q) {
        for (uint32_t d = 0; d < dim; ++d)
            a8[q][d] = static_cast<int8_t>((rand() % 255) - 127);
        ctx[q].a8 = a8[q].data();
        ctx[q].a8_lo = nullptr;
        ctx[q].i8_mode = 1;
        ctx[q].i8_inv = 1.0f;
        ctx[q].dim = dim;
        ctx[q].cs = cs;
        ctx[q].slm_shaped = true;
        ctx[q].shape_u8 = g_tbl;
    }
    [[maybe_unused]] const ScalarScanCtx* c4[4] = {&ctx[0], &ctx[1], &ctx[2], &ctx[3]};

    // Row pointer groups of 4 (like scalar_rows does).
    std::vector<const uint8_t*> rptr(n_rows);
    for (uint32_t r = 0; r < n_rows; ++r) rptr[r] = &rows[r * cs];
    std::vector<const uint8_t*> eptr(n_rows);
    for (uint32_t r = 0; r < n_rows; ++r) eptr[r] = &expanded[r * dim];

    float sink = 0;
    [[maybe_unused]] uint64_t iters;

#if defined(__aarch64__) || defined(__ARM_NEON)
    // --- A: real kernel over packed rows ---
    iters = 0;
    {
        const double t0 = now_s();
        while (now_s() - t0 < run_s) {
            for (uint32_t base = 0; base + 4 <= n_rows; base += 4) {
                const uint8_t* cp[4] = {rptr[base], rptr[base + 1],
                                        rptr[base + 2], rptr[base + 3]};
                float dots[4][4];
                scalar_i8_dots4_q4(c4, cp, dots);
                sink += dots[0][0];
            }
            ++iters;
        }
        const double dt = now_s() - t0;
        printf("A real-kernel packed : %8.1f Mrows/s (%.2f GMAC/s)\n",
               iters * (n_rows / 4.0) / dt / 1e6,
               iters * static_cast<double>(n_rows) * dim / dt / 1e9);
    }

    // --- B: pure NEON USDOT over pre-expanded rows ---
    iters = 0;
    {
        const int8x16_t aq[4] = {vld1q_s8(a8[0].data()), vld1q_s8(a8[1].data()),
                                 vld1q_s8(a8[2].data()), vld1q_s8(a8[3].data())};
        const double t0 = now_s();
        while (now_s() - t0 < run_s) {
            for (uint32_t base = 0; base + 4 <= n_rows; base += 4) {
                int32x4_t acc[4][4];
                for (uint32_t q = 0; q < 4; ++q)
                    for (uint32_t v = 0; v < 4; ++v)
                        acc[q][v] = vdupq_n_s32(0);
                for (uint32_t d = 0; d < dim; d += 16) {
                    for (uint32_t v = 0; v < 4; ++v) {
                        const uint8x16_t g =
                            vld1q_u8(eptr[base + v] + d);
                        for (uint32_t q = 0; q < 4; ++q) {
                            const int8x16_t a =
                                vld1q_s8(a8[q].data() + d);
                            acc[q][v] = neon_gdot_s32(acc[q][v], g, a);
                            (void)aq;
                        }
                    }
                }
                int32_t s = 0;
                for (uint32_t q = 0; q < 4; ++q)
                    for (uint32_t v = 0; v < 4; ++v)
                        s += vaddvq_s32(acc[q][v]);
                sink += static_cast<float>(s);
            }
            ++iters;
        }
        const double dt = now_s() - t0;
        printf("B NEON usdot expanded: %8.1f Mrows/s (%.2f GMAC/s)\n",
               iters * (n_rows / 4.0) / dt / 1e6,
               iters * static_cast<double>(n_rows) * dim / dt / 1e9);
    }

#if defined(__ARM_FEATURE_SVE)
    // --- C: pure SVE svdot over pre-expanded rows (all 16 dots) ---
    iters = 0;
    {
        const double t0 = now_s();
        const uint32_t vl = svcntb();   // bytes per SVE vector
        printf("SVE vector bytes: %u\n", vl);
        while (now_s() - t0 < run_s) {
            for (uint32_t base = 0; base + 4 <= n_rows; base += 4) {
                svint32_t a00 = svdup_n_s32(0), a01 = svdup_n_s32(0);
                svint32_t a02 = svdup_n_s32(0), a03 = svdup_n_s32(0);
                svint32_t a10 = svdup_n_s32(0), a11 = svdup_n_s32(0);
                svint32_t a12 = svdup_n_s32(0), a13 = svdup_n_s32(0);
                svint32_t a20 = svdup_n_s32(0), a21 = svdup_n_s32(0);
                svint32_t a22 = svdup_n_s32(0), a23 = svdup_n_s32(0);
                svint32_t a30 = svdup_n_s32(0), a31 = svdup_n_s32(0);
                svint32_t a32 = svdup_n_s32(0), a33 = svdup_n_s32(0);
                for (uint32_t d = 0; d < dim; d += vl) {
                    const svbool_t pg = svwhilelt_b8(d, dim);
                    const svuint8_t g0 = svld1_u8(pg, eptr[base] + d);
                    const svuint8_t g1 = svld1_u8(pg, eptr[base + 1] + d);
                    const svuint8_t g2 = svld1_u8(pg, eptr[base + 2] + d);
                    const svuint8_t g3 = svld1_u8(pg, eptr[base + 3] + d);
                    const svint8_t q0 = svld1_s8(pg, a8[0].data() + d);
                    const svint8_t q1 = svld1_s8(pg, a8[1].data() + d);
                    const svint8_t q2 = svld1_s8(pg, a8[2].data() + d);
                    const svint8_t q3 = svld1_s8(pg, a8[3].data() + d);
                    // Exact u8-s8 dot without i8mm:
                    // g = (g&0x7F) + 128*(g>>7); both halves positive s8.
                    #define SVE_GDOT(acc, g, q)                                   \
                        do {                                                       \
                            const svint8_t gl = svreinterpret_s8_u8(              \
                                svand_u8_x(pg, g, svdup_n_u8(0x7F)));             \
                            const svint8_t gh = svreinterpret_s8_u8(               \
                                svlsr_n_u8_x(pg, g, 7));                           \
                            acc = svdot_s32(acc, gl, q);                           \
                            acc = svmla_s32_m(                                    \
                                pg, acc, svdup_n_s32(128),                        \
                                svdot_s32(svdup_n_s32(0), gh, q));                \
                        } while (0)
                    SVE_GDOT(a00, g0, q0);
                    SVE_GDOT(a01, g0, q1);
                    SVE_GDOT(a02, g0, q2);
                    SVE_GDOT(a03, g0, q3);
                    SVE_GDOT(a10, g1, q0);
                    SVE_GDOT(a11, g1, q1);
                    SVE_GDOT(a12, g1, q2);
                    SVE_GDOT(a13, g1, q3);
                    SVE_GDOT(a20, g2, q0);
                    SVE_GDOT(a21, g2, q1);
                    SVE_GDOT(a22, g2, q2);
                    SVE_GDOT(a23, g2, q3);
                    SVE_GDOT(a30, g3, q0);
                    SVE_GDOT(a31, g3, q1);
                    SVE_GDOT(a32, g3, q2);
                    SVE_GDOT(a33, g3, q3);
                    #undef SVE_GDOT
                }
                int32_t s = 0;
                for (const svint32_t* p : {&a00, &a01, &a02, &a03, &a10,
                                           &a11, &a12, &a13, &a20, &a21,
                                           &a22, &a23, &a30, &a31, &a32,
                                           &a33})
                    s += svaddv_s32(svptrue_b32(), *p);
                sink += static_cast<float>(s);
            }
            ++iters;
        }
        const double dt = now_s() - t0;
        printf("C SVE usdot expanded : %8.1f Mrows/s (%.2f GMAC/s)\n",
               iters * (n_rows / 4.0) / dt / 1e6,
               iters * static_cast<double>(n_rows) * dim / dt / 1e9);
    }
#endif

#elif defined(SEXTANT_HAS_AVX512_SCAN)
    // x86: real packed kernel vs pure-dpbusd over expanded bytes.
    iters = 0;
    {
        const double t0 = now_s();
        while (now_s() - t0 < run_s) {
            for (uint32_t base = 0; base + 4 <= n_rows; base += 4) {
                const uint8_t* cp[4] = {rptr[base], rptr[base + 1],
                                        rptr[base + 2], rptr[base + 3]};
                float dots[4][4];
                scalar_i8_dots4_q4(c4, cp, dots);
                sink += dots[0][0];
            }
            ++iters;
        }
        const double dt = now_s() - t0;
        printf("A real-kernel packed : %8.1f Mrows/s (%.2f GMAC/s)\n",
               iters * (n_rows / 4.0) / dt / 1e6,
               iters * static_cast<double>(n_rows) * dim / dt / 1e9);
    }
    iters = 0;
    {
        const double t0 = now_s();
        while (now_s() - t0 < run_s) {
            for (uint32_t base = 0; base + 4 <= n_rows; base += 4) {
                __m512i acc[4][4];
                for (uint32_t q = 0; q < 4; ++q)
                    for (uint32_t v = 0; v < 4; ++v)
                        acc[q][v] = _mm512_setzero_si512();
                for (uint32_t d = 0; d < dim; d += 32) {
                    // row bytes occupy quarters? No: expanded rows are
                    // contiguous; process 32 dims = one zmm per row.
                    const __m512i g0 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(eptr[base] + d));
                    const __m512i g1 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(eptr[base + 1] + d));
                    const __m512i g2 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(eptr[base + 2] + d));
                    const __m512i g3 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(eptr[base + 3] + d));
                    for (uint32_t q = 0; q < 4; ++q) {
                        const __m512i a = _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(
                                a8[q].data() + d));
                        acc[q][0] = _mm512_dpbusd_epi32(acc[q][0], g0, a);
                        acc[q][1] = _mm512_dpbusd_epi32(acc[q][1], g1, a);
                        acc[q][2] = _mm512_dpbusd_epi32(acc[q][2], g2, a);
                        acc[q][3] = _mm512_dpbusd_epi32(acc[q][3], g3, a);
                    }
                }
                int32_t s = 0;
                alignas(64) int32_t lanes[16];
                for (uint32_t q = 0; q < 4; ++q)
                    for (uint32_t v = 0; v < 4; ++v) {
                        _mm512_store_si512(
                            reinterpret_cast<__m512i*>(lanes), acc[q][v]);
                        s += lanes[0] + lanes[8] + lanes[16 % 16];
                    }
                sink += static_cast<float>(s);
            }
            ++iters;
        }
        const double dt = now_s() - t0;
        printf("B AVX512 dpbusd expanded: %8.1f Mrows/s (%.2f GMAC/s)\n",
               iters * (n_rows / 4.0) / dt / 1e6,
               iters * static_cast<double>(n_rows) * dim / dt / 1e9);
    }
#endif
    printf("(sink %f)\n", sink);
    return 0;
}
