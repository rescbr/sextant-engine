/// @file test_dots4_kernel.cpp
/// Correctness tests for the arch-dispatched i8 scan dot kernels
/// (`scalar_i8_dots4` / `scalar_i8_dots4_q4`) against their bit-exact
/// `_ref` scalar twins.
///
/// Kernel math is INTEGER end to end (i32 dot accumulation, one float
/// epilogue), so kernel-vs-ref agreement is bit-exact by construction on
/// every path that keeps the tail chunk inside the integer accumulators
/// (NEON: zero-masked tail lanes; AVX-512 q4: masked loads). The AVX-512
/// single kernel instead finishes non-multiple-of-16 dims with a scalar
/// float tail — last-ulp rounding differences vs `_ref` there, so on x86
/// the tail-dim assertions use a tight relative tolerance while multiple-
/// of-16 dims stay bit-exact. On NEON (SEXTANT_HAS_NEON_DOTSCAN) the
/// integer-only tail makes EVERY dim bit-exact.
///
/// Fixture dims: 768/96 = multiple of 16 (full chunks only), 726/764/33 =
/// tail-path dims (rem = 6/12/1).

#include <gtest/gtest.h>
#include "tree/coders/coder_util.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace sextant {
namespace tree {
namespace coders {
namespace {

// Which fast path this binary compiled in (from coder_util.hpp).
#if defined(SEXTANT_HAS_NEON_DOTSCAN)
// NEON SDOT/USDOT: integer-only tail → bit-exact for every dim.
#define DOTS4_TAIL_EXACT 1
#elif defined(SEXTANT_HAS_AVX512_SCAN)
// AVX-512: scalar (single) / float-add (q4) tail → last-ulp-off on tails.
#define DOTS4_TAIL_EXACT 0
#endif

struct Fixture {
    std::vector<uint8_t> codes;    // 4 rows × cs bytes
    std::vector<int8_t> a8;        // zero-padded to 16-dim chunks
    std::vector<int8_t> a8_lo;
    std::vector<uint8_t> shape;    // 16 bytes
    ScalarScanCtx c;
};

Fixture make_fixture(std::mt19937& rng, uint32_t dim, int mode, bool shaped,
                     float inv, const uint8_t* shared_shape = nullptr) {
    Fixture f;
    const uint32_t cs = (dim + 1) / 2;
    f.codes.resize(4 * cs);
    for (auto& b : f.codes) b = static_cast<uint8_t>(rng());
    const uint32_t padded = (dim + 15) / 16 * 16;
    f.a8.assign(padded, 0);
    f.a8_lo.assign(padded, 0);
    for (uint32_t d = 0; d < dim; ++d) {
        f.a8[d] = static_cast<int8_t>(rng() % 256 - 128);
        f.a8_lo[d] = static_cast<int8_t>(rng() % 256 - 128);
    }
    f.shape.resize(16);
    if (shared_shape) {
        // q4 contract: one LEAF → ONE shared shape table for all 4 query
        // contexts (the kernel reads c4[0].slm_shaped / shape_u8 for all).
        for (int i = 0; i < 16; ++i) f.shape[i] = shared_shape[i];
    } else {
        for (auto& b : f.shape) b = static_cast<uint8_t>(rng());
    }
    f.c.codes = f.codes.data();
    f.c.count = 4;
    f.c.cs = cs;
    f.c.dim = static_cast<uint16_t>(dim);
    f.c.i8_mode = mode;
    f.c.a8 = f.a8.data();
    if (mode >= 2) f.c.a8_lo = f.a8_lo.data();
    f.c.i8_inv = inv;
    f.c.slm_shaped = shaped;
    if (shaped) f.c.shape_u8 = f.shape.data();
    return f;
}

const uint32_t kDims[] = {768, 726, 764, 96, 33};

void row_ptrs(const Fixture& f, const uint8_t* cp[4]) {
    for (int v = 0; v < 4; ++v) cp[v] = f.codes.data() + v * f.c.cs;
}

// Independent naive recompute (double accumulation, explicit nibble decode)
// used to validate the FIXTURE and the _ref twin — always runs, even with
// no fast path compiled in.
void naive(const Fixture& f, double dots[4]) {
    for (int v = 0; v < 4; ++v) {
        double acc = 0, acc_lo = 0;
        for (uint32_t d = 0; d < f.c.dim; ++d) {
            const uint8_t byte = f.codes[v * f.c.cs + d / 2];
            const uint32_t nib = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            const double g = f.c.slm_shaped ? f.shape[nib]
                                            : static_cast<double>(nib);
            acc += static_cast<double>(f.a8[d]) * g;
            if (f.c.i8_mode >= 2)
                acc_lo += static_cast<double>(f.a8_lo[d]) * g;
        }
        dots[v] = (acc + acc_lo / 127.0) * f.c.i8_inv;
    }
}

void expect_match(float got, float want, bool bit_exact) {
    if (bit_exact) {
        EXPECT_EQ(got, want) << " got=" << got << " want=" << want;
    } else {
        // Last-ulp float-order difference (AVX-512 scalar tail): the exact
        // integer content is identical; only the rounding sequence differs.
        EXPECT_NEAR(got, want, std::abs(want) * 1e-5f + 1e-4f)
            << " got=" << got << " want=" << want;
    }
}

bool tail_exact(uint32_t dim) {
#if defined(DOTS4_TAIL_EXACT)
    return dim % 16 == 0 || DOTS4_TAIL_EXACT;
#else
    return false;
#endif
}

TEST(Dots4Kernel, RefSanity) {
    // Always runs: validates the fixtures + the _ref twins against an
    // independent double-precision recompute (exact integer sums, so the
    // agreement is limited only by float/int64→float rounding).
    for (uint32_t dim : kDims)
        for (int mode : {1, 2})
            for (bool shaped : {false, true}) {
                std::mt19937 rng(dim * 31 + mode * 7 + shaped);
                Fixture f = make_fixture(rng, dim, mode, shaped, 1.0f / 254.0f);
                double nd[4];
                naive(f, nd);
                float rd[4];
                const uint8_t* cp[4];
                row_ptrs(f, cp);
                scalar_i8_dots4_ref(f.c, cp, rd);
                for (int v = 0; v < 4; ++v)
                    EXPECT_NEAR(rd[v], nd[v], std::abs(nd[v]) * 1e-6 + 1e-3)
                        << "dim=" << dim << " mode=" << mode
                        << " shaped=" << shaped << " v=" << v;
            }
}

#if defined(SEXTANT_HAS_AVX512_SCAN) || defined(SEXTANT_HAS_NEON_DOTSCAN)
// Fast path present: assert kernel == _ref (bit-exact where the path is
// integer through the tail; tolerance on the AVX-512 float-tail dims).
TEST(Dots4Kernel, SingleKernelVsRef) {
    for (uint32_t dim : kDims)
        for (int mode : {1, 2})
            for (bool shaped : {false, true}) {
                std::mt19937 rng(dim * 131 + mode * 17 + shaped);
                Fixture f = make_fixture(rng, dim, mode, shaped, 1.0f / 254.0f);
                float rd[4], kd[4];
                const uint8_t* cp[4];
                row_ptrs(f, cp);
                scalar_i8_dots4_ref(f.c, cp, rd);
                scalar_i8_dots4(f.c, cp, kd);
                const bool exact = tail_exact(dim);
                for (int v = 0; v < 4; ++v) {
                    EXPECT_TRUE(exact || std::isfinite(kd[v]));
                    expect_match(kd[v], rd[v], exact);
                }
            }
}

TEST(Dots4Kernel, Q4KernelVsRef) {
    for (uint32_t dim : kDims)
        for (int mode : {1, 2})
            for (bool shaped : {false, true}) {
                std::mt19937 rng(dim * 977 + mode * 13 + shaped);
                // 4 distinct query contexts (different a8 / inv), same
                // leaf → one shared shape table and code rows (cp).
                Fixture f0 = make_fixture(rng, dim, mode, shaped,
                                          1.0f / 100.0f);
                std::vector<Fixture> fx;
                fx.push_back(std::move(f0));
                for (int q = 1; q < 4; ++q)
                    fx.push_back(make_fixture(rng, dim, mode, shaped,
                                              1.0f / (100.0f + q),
                                              fx[0].shape.data()));
                const uint8_t* cp[4];
                row_ptrs(fx[0], cp);
                float kq[4][4];
                const ScalarScanCtx* c4[4] = {&fx[0].c, &fx[1].c, &fx[2].c,
                                              &fx[3].c};
                scalar_i8_dots4_q4(c4, cp, kq);
                const bool exact = tail_exact(dim);
                for (int q = 0; q < 4; ++q) {
                    float rd[4];
                    scalar_i8_dots4_ref(fx[q].c, cp, rd);
                    for (int v = 0; v < 4; ++v)
                        expect_match(kq[q][v], rd[v], exact);
                }
            }
}

TEST(Dots4Kernel, Q4KernelVsFourSingles) {
    // Contract: q4 == 4 × scalar_i8_dots4 (shared nibbles, per-query a8).
    // Mixed i8_modes across the 4 queries exercise the per-query a8_lo
    // branches of both kernels (identical conditions in both → equality).
    for (uint32_t dim : kDims)
        for (bool shaped : {false, true}) {
            std::mt19937 rng(dim * 6151 + shaped);
            const int modes[4] = {1, 2, 2, 1};
            std::vector<Fixture> fx;
            fx.push_back(make_fixture(rng, dim, modes[0], shaped,
                                      1.0f / 200.0f));
            for (int q = 1; q < 4; ++q)
                fx.push_back(make_fixture(rng, dim, modes[q], shaped,
                                          1.0f / (200.0f + q),
                                          fx[0].shape.data()));
            const uint8_t* cp[4];
            row_ptrs(fx[0], cp);
            float kq[4][4], singles[4][4];
            const ScalarScanCtx* c4[4] = {&fx[0].c, &fx[1].c, &fx[2].c,
                                          &fx[3].c};
            scalar_i8_dots4_q4(c4, cp, kq);
            for (int q = 0; q < 4; ++q)
                scalar_i8_dots4(fx[q].c, cp, singles[q]);
            const bool exact = tail_exact(dim);
            for (int q = 0; q < 4; ++q)
                for (int v = 0; v < 4; ++v)
                    expect_match(kq[q][v], singles[q][v], exact);
        }
}
#endif  // fast path present

}  // namespace
}  // namespace coders
}  // namespace tree
}  // namespace sextant
