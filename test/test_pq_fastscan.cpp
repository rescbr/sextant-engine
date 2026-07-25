/// @file test_pq_fastscan.cpp
/// Tests for the PQ-FastScan split-table kernel (`simd::fastscan_block16`)
/// and the uint8 LUT quantizer (`simd::quantize_lut_u8`,
/// `PqQuantizer::build_fastscan_lut`).
///
/// Design (see `docs/fastscan_research_2026-07-25.md`): the 256-entry uint8
/// LUT row is split into 4 banks of 64 bytes; four `vqtbl4q_u8` calls with
/// `code>>6` bank-routing cover all 256 codes exactly. The only error vs the
/// float-LUT path is the uint8 quantization (≤0.5pp recall per FAISS).

#include <gtest/gtest.h>
#include "quant/pq_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/error.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace sextant {
namespace {

// ---------------------------------------------------------------------------
// Test RNG + synthetic data (mirrors test_pq_quantizer.cpp conventions).
// ---------------------------------------------------------------------------

class Rng {
public:
    explicit Rng(uint64_t seed) : rng_(seed) {}
    float uniform(float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        return d(rng_);
    }
    uint32_t u32_below(uint32_t n) {
        std::uniform_int_distribution<uint32_t> d(0, n - 1);
        return d(rng_);
    }
private:
    std::mt19937_64 rng_;
};

std::vector<float> make_clustered_data(uint64_t n, uint32_t dim,
                                       uint32_t n_clusters, uint64_t seed) {
    Rng rng(seed);
    std::vector<float> centers(size_t(n_clusters) * dim);
    for (auto& v : centers) v = rng.uniform(-10.0f, 10.0f);
    std::vector<float> data(size_t(n) * dim);
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t c = rng.u32_below(n_clusters);
        for (uint32_t d = 0; d < dim; d++) {
            data[i * dim + d] = centers[c * dim + d] + noise(gen);
        }
    }
    return data;
}

/// Encode 16 vectors into the segment-major block layout `[m][16]` used by
/// `fastscan_block16`. `vecs` is an array of 16 pointers to dim-float vectors.
/// Out block: byte `s*16 + j` holds segment `s` of code `j`.
void encode_block16(const PqQuantizer& q, const float* vecs[16],
                    uint8_t* out_block) {
    const uint32_t m = q.m();
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> contiguous(size_t(16) * cs, 0);
    for (uint32_t j = 0; j < 16; j++) {
        q.encode(vecs[j], contiguous.data() + size_t(j) * cs);
    }
    // Transpose [j][s] -> [s][j] (only valid for bits=8 where byte s = code[s]).
    for (uint32_t s = 0; s < m; s++) {
        for (uint32_t j = 0; j < 16; j++) {
            out_block[s * 16 + j] = contiguous[size_t(j) * cs + s];
        }
    }
}

// ===========================================================================
// simd::fastscan_block16 — kernel correctness (no quantizer involved).
// ===========================================================================

TEST(FastScanBlock16, ExhaustiveAllCodesAllLanes) {
    // For every (lane, code) pair: place `code` in `lane`, all-zero codes
    // elsewhere, verify the kernel computes the exact scalar sum. This is
    // the in-engine mirror of the pre-plan exhaustive verification.
    const uint32_t M = 96, K = 256;
    std::vector<uint8_t> lut8(size_t(M) * K);
    // Distinct (s,c) values so any routing error surfaces immediately.
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t c = 0; c < K; c++)
            lut8[s * K + c] = static_cast<uint8_t>((s * 7 + c * 13) & 0xFF);

    int fail = 0;
    for (uint32_t lane = 0; lane < 16; lane++) {
        for (uint32_t code_val = 0; code_val < 256; code_val++) {
            std::vector<uint8_t> block(size_t(M) * 16, 0);
            for (uint32_t s = 0; s < M; s++)
                block[s * 16 + lane] = static_cast<uint8_t>(code_val);
            uint32_t out[16];
            simd::fastscan_block16(block.data(), lut8.data(), M, 0xFFFF, out);
            uint32_t expected = 0;
            for (uint32_t s = 0; s < M; s++)
                expected += lut8[s * K + code_val];
            if (out[lane] != expected) {
                if (fail < 5) {
                    ADD_FAILURE() << "lane=" << lane << " code=" << code_val
                                  << " expected " << expected << " got " << out[lane];
                }
                fail++;
            }
        }
    }
    EXPECT_EQ(0, fail) << "exhaustive kernel check failed";
}

TEST(FastScanBlock16, ValidityMaskInvalidatesLanes) {
    const uint32_t M = 8, K = 256;
    std::vector<uint8_t> lut8(size_t(M) * K);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t c = 0; c < K; c++)
            lut8[s * K + c] = static_cast<uint8_t>((s + c) & 0xFF);

    std::vector<uint8_t> block(size_t(M) * 16, 0);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t j = 0; j < 16; j++)
            block[s * 16 + j] = static_cast<uint8_t>(j * 7);

    const uint16_t mask = 0x7BDF;  // bits 4, 9, 14 cleared
    uint32_t out[16];
    simd::fastscan_block16(block.data(), lut8.data(), M, mask, out);

    for (uint32_t j = 0; j < 16; j++) {
        const bool valid = (mask >> j) & 1u;
        if (valid) {
            uint32_t expected = 0;
            for (uint32_t s = 0; s < M; s++)
                expected += lut8[s * K + block[s * 16 + j]];
            EXPECT_EQ(expected, out[j]) << "valid lane " << j;
        } else {
            EXPECT_EQ(0xFFFFFFFFu, out[j]) << "invalid lane " << j;
        }
    }
}

TEST(FastScanBlock16, AllInvalidMaskYieldsAllSentinel) {
    const uint32_t M = 4, K = 256;
    std::vector<uint8_t> lut8(size_t(M) * K, 0);
    std::vector<uint8_t> block(size_t(M) * 16, 0);
    uint32_t out[16];
    simd::fastscan_block16(block.data(), lut8.data(), M, /*valid_mask=*/0x0000, out);
    for (uint32_t j = 0; j < 16; j++) {
        EXPECT_EQ(0xFFFFFFFFu, out[j]) << "lane " << j;
    }
}

// ===========================================================================
// simd::quantize_lut_u8 — quantization correctness + monotonicity.
// ===========================================================================

TEST(QuantizeLutU8, MaxEntryIs255AtMaxSpan) {
    // LUT with a single max value, rest at min: A=255/max_span should map
    // the max to 255 (or clamped). Span = 100.
    const uint32_t M = 4, K = 16;
    std::vector<float> lut_f32(M * K);
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t c = 0; c < K; c++)
            lut_f32[s * K + c] = static_cast<float>(c) * 6.25f;  // spans 0..93.75

    std::vector<uint8_t> lut8(M * K);
    float A, B;
    simd::quantize_lut_u8(lut_f32.data(), M, K, lut8.data(), &A, &B);
    // Each segment's max entry (c=K-1) should round to 255.
    for (uint32_t s = 0; s < M; s++) {
        EXPECT_EQ(255, lut8[s * K + (K - 1)]) << "segment " << s;
        EXPECT_EQ(0,   lut8[s * K + 0])       << "segment " << s;
    }
    EXPECT_GT(A, 0.0f);
    EXPECT_FLOAT_EQ(0.0f, B);  // min of every segment is 0
}

TEST(QuantizeLutU8, BIsSumOfPerSegmentMins) {
    // Distinct per-segment mins so B isn't trivially 0.
    const uint32_t M = 3, K = 8;
    std::vector<float> lut_f32(M * K);
    const float seg_offset[3] = {10.0f, 20.0f, 30.0f};
    for (uint32_t s = 0; s < M; s++)
        for (uint32_t c = 0; c < K; c++)
            lut_f32[s * K + c] = seg_offset[s] + static_cast<float>(c);
    std::vector<uint8_t> lut8(M * K);
    float A, B;
    simd::quantize_lut_u8(lut_f32.data(), M, K, lut8.data(), &A, &B);
    // Each segment's min is at c=0; the per-segment spans are all 7 (K-1).
    // B = 10 + 20 + 30 = 60.
    EXPECT_FLOAT_EQ(60.0f, B);
    // All segments have the same span (7), so A = 255/7 ≈ 36.43.
    EXPECT_NEAR(255.0f / 7.0f, A, 0.01f);
}

TEST(QuantizeLutU8, PreserveArgminWithinSegment) {
    // The argmin within each segment's LUT row should be preserved by the
    // uint8 quantization (small relative errors shouldn't change ranking).
    const uint32_t M = 8, K = 256;
    Rng rng(123);
    std::vector<float> lut_f32(M * K);
    for (auto& v : lut_f32) v = rng.uniform(0.0f, 100.0f);
    // Force a clear argmin per segment at c=0 (set others ≥1).
    for (uint32_t s = 0; s < M; s++) {
        lut_f32[s * K + 0] = 0.0f;
        for (uint32_t c = 1; c < K; c++)
            lut_f32[s * K + c] = std::max(lut_f32[s * K + c], 1.0f);
    }
    std::vector<uint8_t> lut8(M * K);
    float A, B;
    simd::quantize_lut_u8(lut_f32.data(), M, K, lut8.data(), &A, &B);
    for (uint32_t s = 0; s < M; s++) {
        // uint8 min should also be at c=0.
        uint8_t mn = lut8[s * K];
        uint32_t argmin = 0;
        for (uint32_t c = 1; c < K; c++) {
            if (lut8[s * K + c] < mn) { mn = lut8[s * K + c]; argmin = c; }
        }
        EXPECT_EQ(0u, argmin) << "segment " << s;
    }
}

// ===========================================================================
// End-to-end: PqQuantizer::build_fastscan_lut vs float LUT distance parity.
// ===========================================================================

/// Build a real trained quantizer with realistic m/K/dim.
std::unique_ptr<PqQuantizer> make_trained_quantizer(uint32_t dim, uint16_t m,
                                                    uint8_t bits, MetricKind metric,
                                                    uint64_t seed) {
    auto q = std::make_unique<PqQuantizer>(metric, dim, m, bits, seed);
    const uint64_t n = std::max<uint64_t>(5000, m * 50);  // enough samples to train
    auto data = make_clustered_data(n, dim, /*n_clusters=*/50, seed + 1);
    q->train(data.data(), n);
    return q;
}

TEST(BuildFastScanLut, DistanceErrorWithinTolerance) {
    const uint32_t dim = 64;
    const uint16_t m = 16;
    auto q = make_trained_quantizer(dim, m, /*bits=*/8, MetricKind::L2Sq, /*seed=*/42);

    // Random query + 16 random code vectors.
    Rng rng(7);
    std::vector<float> query(dim);
    for (auto& v : query) v = rng.uniform(-5.0f, 5.0f);
    std::vector<float> vecs(size_t(16) * dim);
    for (auto& v : vecs) v = rng.uniform(-5.0f, 5.0f);

    // Float LUT + float distances (the reference).
    std::vector<float> lut_f32(q->lut_size());
    q->preprocess_query(query.data(), lut_f32.data());

    // uint8 LUT.
    std::vector<uint8_t> lut8(q->fastscan_lut_bytes());
    float A, B;
    q->build_fastscan_lut(query.data(), lut8.data(), &A, &B);

    // Encode 16 vectors both ways: contiguous (for float lut_distance) and
    // segment-major block (for fastscan_block16).
    const uint32_t cs = q->code_size();
    std::vector<uint8_t> codes_contig(size_t(16) * cs, 0);
    std::vector<uint8_t> block(size_t(m) * 16, 0);
    std::vector<const float*> vec_ptrs(16);
    for (uint32_t j = 0; j < 16; j++) {
        vec_ptrs[j] = vecs.data() + size_t(j) * dim;
        q->encode(vec_ptrs[j], codes_contig.data() + size_t(j) * cs);
    }
    encode_block16(*q, vec_ptrs.data(), block.data());

    // Run the kernel.
    uint32_t out_u32[16];
    simd::fastscan_block16(block.data(), lut8.data(), m, 0xFFFF, out_u32);

    // Reconstruct float distance: dist ≈ (acc / A) + B.
    // Compare against the float-LUT distance.
    float max_abs_err = 0.0f;
    for (uint32_t j = 0; j < 16; j++) {
        const float float_dist = q->lut_distance(codes_contig.data() + size_t(j) * cs,
                                                 lut_f32.data());
        const float fastscan_dist = (float)out_u32[j] / A + B;
        const float err = std::abs(fastscan_dist - float_dist);
        max_abs_err = std::max(max_abs_err, err);
    }
    // Tolerance: uint8 quantization error per entry ≈ 1/A. Over m=16 segments
    // the worst-case sum-error is m/A. Allow 2× that for safety margin.
    const float tolerance = 2.0f * (float)m / A;
    EXPECT_LT(max_abs_err, tolerance)
        << "max abs err " << max_abs_err << " exceeds tolerance " << tolerance
        << " (A=" << A << ", B=" << B << ")";
}

TEST(BuildFastScanLut, ArgminRankAgreement) {
    // Over many random blocks, the argmin (nearest of 16 codes) under
    // FastScan should match the float-LUT argmin in almost all cases.
    // Allow ≤1 rank-1 disagreement per 20 trials (5%) due to quantization
    // noise on near-tied distances.
    const uint32_t dim = 96;
    const uint16_t m = 24;
    auto q = make_trained_quantizer(dim, m, /*bits=*/8, MetricKind::L2Sq, /*seed=*/99);

    const int TRIALS = 200;
    int disagree = 0;
    Rng rng(2024);
    std::vector<float> query(dim);
    std::vector<float> vecs(size_t(16) * dim);
    std::vector<const float*> vec_ptrs(16);

    std::vector<float> lut_f32(q->lut_size());
    std::vector<uint8_t> lut8(q->fastscan_lut_bytes());
    const uint32_t cs = q->code_size();
    std::vector<uint8_t> codes_contig(size_t(16) * cs, 0);
    std::vector<uint8_t> block(size_t(m) * 16, 0);

    for (int t = 0; t < TRIALS; t++) {
        for (auto& v : query) v = rng.uniform(-5.0f, 5.0f);
        for (auto& v : vecs) v = rng.uniform(-5.0f, 5.0f);

        q->preprocess_query(query.data(), lut_f32.data());
        float A, B;
        q->build_fastscan_lut(query.data(), lut8.data(), &A, &B);

        for (uint32_t j = 0; j < 16; j++) {
            vec_ptrs[j] = vecs.data() + size_t(j) * dim;
            q->encode(vec_ptrs[j], codes_contig.data() + size_t(j) * cs);
        }
        encode_block16(*q, vec_ptrs.data(), block.data());

        uint32_t out_u32[16];
        simd::fastscan_block16(block.data(), lut8.data(), m, 0xFFFF, out_u32);

        // Float argmin + fastscan argmin.
        uint32_t argmin_float = 0;
        float best_float = std::numeric_limits<float>::infinity();
        uint32_t argmin_fs = 0;
        uint32_t best_fs = 0xFFFFFFFFu;
        for (uint32_t j = 0; j < 16; j++) {
            const float d = q->lut_distance(codes_contig.data() + size_t(j) * cs,
                                            lut_f32.data());
            if (d < best_float) { best_float = d; argmin_float = j; }
            if (out_u32[j] < best_fs) { best_fs = out_u32[j]; argmin_fs = j; }
        }
        if (argmin_float != argmin_fs) disagree++;
    }

    // Allow up to 5% disagreement (10/200). In practice expect 0-2.
    EXPECT_LE(disagree, TRIALS / 20)
        << "argmin disagreement " << disagree << "/" << TRIALS
        << " exceeds 5% tolerance";

    // Diagnostic: log the actual disagreement count so we know the noise
    // floor (helps decide if/when uint8 LUT quantization ever needs revisiting).
    std::cout << "[BuildFastScanLut.ArgminRankAgreement] disagreements: "
              << disagree << "/" << TRIALS << "\n";
}

}  // namespace
}  // namespace sextant
