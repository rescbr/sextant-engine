/// Cross-validation tests for the x86 SIMD ports (AVX-512) against their
/// always-compiled scalar `_ref` twins. Run on the 9950X (Zen 5) where the
/// AVX-512 branches are live; on other arches the dispatcher IS the ref, so
/// these tests trivially pass (and still exercise the ref itself).
///
/// Expectations:
///   - integer kernels (fastscan_many, scalar_i8_dots4): BIT-EXACT match
///   - float kernels (scalar_arith_dots4, u4 decode-dot): accumulation
///     order differs, so relative tolerance

#include <gtest/gtest.h>

#include "simd_kernels.hpp"
#include "tree/coders/coder_util.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace sextant;
using sextant::tree::coders::ScalarScanCtx;

struct Rng {
    std::mt19937 g{12345};
    std::uniform_real_distribution<float> uf{-1.0f, 1.0f};
    std::uniform_int_distribution<int> ui8{0, 255};
};

// ---------------------------------------------------------------------------
// fastscan_many: AVX-512 vpermi2b 4-bank shuffle vs scalar reference.
// ---------------------------------------------------------------------------
TEST(SimdX86, FastscanManyMatchesRef) {
    Rng r;
    for (uint32_t m : {1u, 2u, 8u, 33u}) {
        for (uint32_t n_blocks : {1u, 3u, 4u, 5u, 17u, 64u}) {
            std::vector<uint8_t> lut8(size_t(m) * 256);
            for (auto& b : lut8) b = uint8_t(r.ui8(r.g));
            std::vector<uint8_t> blocks(size_t(n_blocks) * m * 16);
            for (auto& b : blocks) b = uint8_t(r.ui8(r.g));
            std::vector<uint16_t> masks(n_blocks, 0xFFFF);
            for (uint32_t b = 0; b < n_blocks; b++) {
                // Random valid prefixes + some all-invalid blocks.
                masks[b] = (b % 5 == 4) ? 0u
                    : uint16_t((1u << (1 + r.ui8(r.g) % 16)) - 1);
            }
            std::vector<uint32_t> got(size_t(n_blocks) * 16);
            std::vector<uint32_t> want(size_t(n_blocks) * 16);
            simd::fastscan_many(blocks.data(), n_blocks, lut8.data(), m,
                                masks.data(), got.data());
            // Reference: the scalar fallback body, inlined here (it is also
            // compiled as the #else arm; replicate for explicitness).
            std::memset(want.data(), 0, want.size() * sizeof(uint32_t));
            for (uint32_t b = 0; b < n_blocks; b++) {
                for (uint32_t s = 0; s < m; s++) {
                    const uint8_t* lut_row = lut8.data() + s * 256;
                    const uint8_t* code_row =
                        blocks.data() + size_t(b) * m * 16 + s * 16;
                    for (uint32_t j = 0; j < 16; j++)
                        want[size_t(b) * 16 + j] += lut_row[code_row[j]];
                }
                for (uint32_t j = 0; j < 16; j++) {
                    if (!((masks[b] >> j) & 1u))
                        want[size_t(b) * 16 + j] = 0xFFFFFFFFu;
                }
            }
            for (size_t i = 0; i < got.size(); i++) {
                ASSERT_EQ(got[i], want[i])
                    << "m=" << m << " n_blocks=" << n_blocks << " i=" << i;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// scalar_i8_dots4: VNNI bias-identity dot vs scalar reference (exact).
// ---------------------------------------------------------------------------
ScalarScanCtx make_i8_ctx(Rng& r, uint32_t dim, uint32_t cs, int mode) {
    ScalarScanCtx c{};
    c.dim = uint16_t(dim);
    c.cs = cs;
    c.count = 8;
    c.i8_mode = mode;
    c.i8_inv = 1.0f / 127.0f;
    static std::vector<int8_t> a8, a8_lo;
    a8.assign(dim, 0);
    a8_lo.assign(dim, 0);
    for (uint32_t d = 0; d < dim; d++) {
        a8[d] = int8_t(r.ui8(r.g) % 255 - 127);
        a8_lo[d] = int8_t(r.ui8(r.g) % 255 - 127);
    }
    c.a8 = a8.data();
    c.a8_lo = mode >= 2 ? a8_lo.data() : nullptr;
    return c;
}

TEST(SimdX86, ScalarI8DotsMatchRef) {
    Rng r;
    for (uint32_t dim : {16u, 32u, 48u, 768u}) {
        for (int mode : {1, 2}) {
            // cs large enough for fully-padded reads, and tight (tail path).
            for (uint32_t cs : {uint32_t(dim / 2), uint32_t((dim + 15) / 16 * 16 / 2)}) {
                auto c = make_i8_ctx(r, dim, cs, mode);
                std::vector<uint8_t> codes(size_t(c.count) * cs, 0);
                for (auto& b : codes) b = uint8_t(r.ui8(r.g));
                const uint8_t* cp[4] = {codes.data(), codes.data() + cs,
                                        codes.data() + 2 * cs,
                                        codes.data() + 3 * cs};
                float got[4], want[4];
                tree::coders::scalar_i8_dots4(c, cp, got);
                tree::coders::scalar_i8_dots4_ref(c, cp, want);
                for (int v = 0; v < 4; v++) {
                    EXPECT_NEAR(got[v], want[v],
                                std::abs(want[v]) * 1e-6f + 1e-6f)
                        << "dim=" << dim << " mode=" << mode << " cs=" << cs
                        << " v=" << v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// scalar_arith_dots4: FMA dot vs scalar reference (float tolerance).
// ---------------------------------------------------------------------------
TEST(SimdX86, ScalarArithDotsMatchRef) {
    Rng r;
    for (uint32_t dim : {16u, 32u, 768u, 1536u}) {
        for (bool shaped : {false, true}) {
            const uint32_t padded = (dim + 15) / 16 * 16;
            const uint32_t cs = dim / 2;
            ScalarScanCtx c{};
            c.dim = uint16_t(dim);
            c.cs = cs;
            c.count = 8;
            c.slm_arith = true;
            c.slm_shaped = shaped;
            static std::vector<float> a_uni, shape_f32;
            static std::vector<uint8_t> shape_u8;
            a_uni.assign(padded, 0.0f);
            shape_f32.assign(16, 0.0f);
            shape_u8.assign(16, 0);
            for (uint32_t d = 0; d < dim; d++) a_uni[d] = r.uf(r.g);
            // The SIMD path reads the u8 table (NEON-equivalent
            // semantics); the scalar ref reads shape_f32. They must agree.
            for (int i = 0; i < 16; i++) {
                shape_u8[i] = uint8_t(r.ui8(r.g));
                shape_f32[i] = float(shape_u8[i]);
            }
            c.a_uni = a_uni.data();
            c.shape_f32 = shape_f32.data();
            c.shape_u8 = shaped ? shape_u8.data() : nullptr;
            std::vector<uint8_t> codes(size_t(c.count) * cs, 0);
            for (auto& b : codes) b = uint8_t(r.ui8(r.g));
            const uint8_t* cp[4] = {codes.data(), codes.data() + cs,
                                    codes.data() + 2 * cs,
                                    codes.data() + 3 * cs};
            float got[4], want[4];
            tree::coders::scalar_arith_dots4(c, cp, got);
            tree::coders::scalar_arith_dots4_ref(c, cp, want);
            for (int v = 0; v < 4; v++) {
                EXPECT_NEAR(got[v], want[v],
                            std::abs(want[v]) * 1e-5f + 1e-5f)
                    << "dim=" << dim << " shaped=" << shaped << " v=" << v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// u4 decode-dot (gather) vs scalar reference accumulation.
// ---------------------------------------------------------------------------
TEST(SimdX86, U4DecodeDotMatchesRef) {
    Rng r;
    for (uint32_t dim : {16u, 17u, 100u, 768u}) {
        const uint32_t K = 16;
        std::vector<float> query(dim), levels(size_t(dim) * K);
        std::vector<uint8_t> code((dim + 1) / 2);
        for (auto& q : query) q = r.uf(r.g);
        for (auto& l : levels) l = r.uf(r.g);
        for (auto& b : code) b = uint8_t(r.ui8(r.g));
        const float got = simd::scalar_dot_u4_float(
            query.data(), levels.data(), code.data(), dim, K);
        float want = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            const uint8_t byte = code[d / 2];
            const uint32_t nib = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            want += query[d] * levels[size_t(d) * K + nib];
        }
        EXPECT_NEAR(got, want, std::abs(want) * 1e-5f + 1e-5f)
            << "dim=" << dim;
    }
}


// ---------------------------------------------------------------------------
// scalar_arith_dist1: fused rerank (A + B.g + C.g^2) vs manual reference
// (audit F9: recall parity was the only indirect evidence — this is the
// direct one).
// ---------------------------------------------------------------------------
TEST(SimdX86, ScalarArithDist1MatchesRef) {
    Rng r;
    for (uint32_t dim : {16u, 48u, 768u}) {
        for (bool shaped : {false, true}) {
            const uint32_t padded = (dim + 15) / 16 * 16;
            const uint32_t cs = dim / 2;
            std::vector<float> b(padded, 0.f), c(padded, 0.f);
            for (uint32_t d = 0; d < dim; d++) {
                b[d] = r.uf(r.g);
                c[d] = r.uf(r.g);
            }
            std::vector<uint8_t> gu8(16);
            std::vector<float> g(16);
            for (int i = 0; i < 16; i++) {
                gu8[i] = uint8_t(r.ui8(r.g));
                g[i] = float(gu8[i]);
            }
            if (!shaped) {
                // uniform: g = identity over 16 levels
                for (int i = 0; i < 16; i++) { gu8[i] = uint8_t(i); g[i] = float(i); }
            }
            const float a = r.uf(r.g);
            std::vector<uint8_t> row(cs, 0);
            for (auto& by : row) by = uint8_t(r.ui8(r.g));
            const float got = tree::coders::scalar_arith_dist1(
                row.data(), dim, cs, a, b.data(), c.data(),
                gu8.data(), g.data());
            float want = a;
            for (uint32_t d = 0; d < dim; d++) {
                const uint8_t byte = row[d / 2];
                const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                                 : ((byte >> 4) & 0xF);
                want += b[d] * g[nib] + c[d] * g[nib] * g[nib];
            }
            EXPECT_NEAR(got, want, std::abs(want) * 1e-5f + 1e-5f)
                << "dim=" << dim << " shaped=" << shaped;
        }
    }
}

// ---------------------------------------------------------------------------
// scalar_scan_gather (Lloyd-Max kernel): full heap vs per-row manual dot.
// (audit F9: never directly cross-validated.)
// ---------------------------------------------------------------------------
TEST(SimdX86, ScalarScanGatherMatchesRef) {
    Rng r;
    for (uint32_t dim : {16u, 20u, 768u}) {
        const uint32_t K = 16;
        const uint32_t cs = (dim + 1) / 2;
        const uint32_t count = 37;  // exercises the batch-4 tail
        std::vector<float> levels(size_t(dim) * K);
        for (auto& v : levels) v = r.uf(r.g) * 2.f - 1.f;
        std::vector<float> query(dim);
        for (auto& q : query) q = r.uf(r.g) * 2.f - 1.f;
        std::vector<uint8_t> codes(size_t(count) * cs, 0);
        for (auto& by : codes) by = uint8_t(r.ui8(r.g));

        ScalarScanCtx c{};
        c.dim = uint16_t(dim);
        c.cs = cs;
        c.count = count;
        c.codes = codes.data();
        c.levels = levels.data();
        c.K = K;
        c.query = query.data();
        c.slm_arith = false;  // gather path
        c.c0 = 0.f;

        std::vector<sextant::tree::HeapEntry> heap;
        sextant::tree::RawScanHeap rh{&heap, count, 0};
        tree::coders::scalar_scan_gather(c, rh);
        ASSERT_EQ(heap.size(), count);

        // Manual reference: dot(query, levels[nibble]) per row.
        // The kernel accumulates in batch-of-4 groups (different fp
        // ordering than the sequential reference), so keys may differ by
        // a few ulps of the order-preserving u32 encoding — compare per
        // row with a tiny key tolerance instead of exact equality.
        ASSERT_EQ(heap.size(), count);
        std::unordered_map<uint32_t, uint32_t> got_by_idx;
        for (const auto& e : heap) got_by_idx[e.local_idx] = e.pq_dist;
        for (uint32_t i = 0; i < count; i++) {
            float dot = 0.f;
            for (uint32_t d = 0; d < dim; d++) {
                const uint8_t byte = codes[size_t(i) * cs + d / 2];
                const uint8_t nib = (d % 2 == 0) ? (byte & 0xF)
                                                 : ((byte >> 4) & 0xF);
                dot += query[d] * levels[size_t(d) * K + nib];
            }
            const uint32_t want_key =
                sextant::tree::f32_to_dist_key(-dot);
            const auto it = got_by_idx.find(i);
            ASSERT_NE(it, got_by_idx.end()) << "row " << i << " missing";
            const int64_t delta =
                std::llabs(int64_t(it->second) - int64_t(want_key));
            EXPECT_LE(delta, 64) << "dim=" << dim << " row=" << i
                                 << " got=" << it->second
                                 << " want=" << want_key;
        }
    }
}

}  // namespace

// Regression (2026-09-07): argmin_scaled SIMD paths iterated the zero-padded
// tail columns [k, k_simd); a padded column's value 0 - 2*0 = 0 won the
// argmin whenever every real column was positive, assigning points to
// nonexistent centroids. k=13 -> k_simd=16 on AVX2 (8-wide), 16 on NEON
// (4-wide) reproduces on both.
TEST(SimdX86, ArgminScaledPaddingPoisoned) {
    const uint32_t k = 13, n = 64;
    const uint32_t ks = simd::gemv_k_simd(k);
    ASSERT_GT(ks, k) << "fixture requires a padded tail";
    std::vector<float> scale(ks, 0.f), dots((size_t)n * ks, 0.f);
    for (uint32_t c = 0; c < k; ++c) scale[c] = 10.f + (float)c;
    for (auto& d : dots) d = -1.f;  // all real vals positive
    std::vector<uint32_t> got(n, 999), want(n, 0);
    uint64_t changed = 0;
    simd::argmin_scaled(n, k, ks, scale.data(), dots.data(), got.data(),
                        changed);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_LT(got[i], k) << "assigned to a padded (nonexistent) centroid";
        EXPECT_EQ(got[i], want[i]);
    }
    // And the normal case still agrees with the scalar reference.
    std::mt19937 rng(1);
    for (auto& d : dots) d = rng() % 1000 / 1000.f - 0.5f;
    for (uint32_t c = 0; c < k; ++c) scale[c] = rng() % 1000 / 1000.f;
    uint64_t ch2 = 0;
    simd::argmin_scaled(n, k, ks, scale.data(), dots.data(), got.data(), ch2);
    for (uint32_t i = 0; i < n; ++i) {
        float best = 1e30f; uint32_t bc = 0;
        for (uint32_t c = 0; c < k; ++c) {
            const float v = scale[c] - 2.f * dots[(size_t)i * ks + c];
            if (v < best) { best = v; bc = c; }
        }
        EXPECT_EQ(got[i], bc) << "argmin diverged from scalar reference";
    }
}
