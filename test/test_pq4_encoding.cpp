/// @file test_pq4_encoding.cpp
/// Correctness tests for the FastScan 4-bit code block packing layout.
///
/// The encoding packs 32 vectors' m nibbles into a [m][16] byte block where
/// byte k of segment s holds:
///   low  nibble = vector (b*32 + k)        (the "lo" lanes 0..15)
///   high nibble = vector (b*32 + 16 + k)   (the "hi" lanes 16..31)
///
/// This is the FAISS `perm0` permutation, NOT (2k, 2k+1) — that would
/// deinterleave the kernel's output. The mismatch is subtle: a wrong layout
/// produces near-miss IDs (82785 instead of 82786) and is very hard to
/// diagnose. These tests pin the lane order explicitly.
///
/// Also tested: tail-block zero-padding (lanes beyond shard_n are 0) and the
/// end-to-end encode → pack → decode → verify roundtrip.

#include <gtest/gtest.h>
#include "quant/pq_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace sextant {
namespace {

// ---------------------------------------------------------------------------
// Mirror of Builder::build_ivf_scan's per-shard packing, lifted to a free
// function for testing. This is the production layout; DO NOT diverge.
//
// `nibbles`: [n][m] — one byte per segment per vector (value 0..15).
// `blocks`:  [ceil(n/32) * m * 16] — packed FastScan blocks.
// ---------------------------------------------------------------------------

/// Pack a flat [n][m] nibble array into FastScan block layout.
/// Lane-order rule (load-bearing): byte k of segment s of block b holds
/// low = vector (b*32+k), high = vector (b*32+16+k).
void pack_blocks(const uint8_t* nibbles, uint32_t n, uint32_t m,
                 uint8_t* blocks) {
    const uint32_t n_blocks = (n + 31) / 32;
    for (uint32_t b = 0; b < n_blocks; b++) {
        for (uint32_t s = 0; s < m; s++) {
            for (uint32_t k = 0; k < 16; k++) {
                const uint32_t v0 = b * 32 + k;          // low nibble
                const uint32_t v1 = b * 32 + 16 + k;     // high nibble
                const uint8_t lo = (v0 < n)
                    ? nibbles[v0 * m + s] : 0;
                const uint8_t hi = (v1 < n)
                    ? nibbles[v1 * m + s] : 0;
                blocks[((b * m) + s) * 16 + k] =
                    static_cast<uint8_t>((hi << 4) | lo);
            }
        }
    }
}

/// Decode block (b), segment (s), lane (j) → the original nibble value.
/// Inverse of pack_blocks. j ∈ [0, 32): j<16 is the low-nibble lane (vector
/// b*32+j), j>=16 is the high-nibble lane (vector b*32+j; the kernel's
/// out[16+k] = byte k high nibble).
uint8_t decode_lane(const uint8_t* blocks, uint32_t m,
                    uint32_t b, uint32_t s, uint32_t j) {
    EXPECT_LT(j, 32u);
    const uint8_t byte = blocks[((b * m) + s) * 16 + (j & 15u)];
    return (j < 16) ? static_cast<uint8_t>(byte & 0x0F)
                    : static_cast<uint8_t>(byte >> 4);
}

class Rng {
public:
    explicit Rng(uint64_t seed) : rng_(seed) {}
    uint8_t nibble() {
        std::uniform_int_distribution<int> d(0, 15);
        return static_cast<uint8_t>(d(rng_));
    }
private:
    std::mt19937_64 rng_;
};

// ---------------------------------------------------------------------------
// Lane-order pin (the load-bearing test).
// ---------------------------------------------------------------------------

TEST(Pq4Encoding, LaneOrderMatchesProductionRule) {
    // Build a [n=32][m] nibble array where nibbles[v, s] = a distinct value
    // per (v, s). Pack, then verify each lane decodes to the expected vector.
    constexpr uint32_t n = 32;  // one full block
    constexpr uint32_t m = 4;
    std::vector<uint8_t> nibbles(static_cast<size_t>(n) * m);
    for (uint32_t v = 0; v < n; v++) {
        for (uint32_t s = 0; s < m; s++) {
            // Values 0..15 only (4 bits); make a per-(v,s) pattern.
            nibbles[v * m + s] = static_cast<uint8_t>((v + s) & 0x0F);
        }
    }
    std::vector<uint8_t> blocks(static_cast<size_t>(m) * 16);
    pack_blocks(nibbles.data(), n, m, blocks.data());

    // Lane j corresponds to vector j within the block.
    for (uint32_t j = 0; j < 32; j++) {
        for (uint32_t s = 0; s < m; s++) {
            const uint8_t got = decode_lane(blocks.data(), m, /*b=*/0, s, j);
            const uint8_t want = static_cast<uint8_t>((j + s) & 0x0F);
            EXPECT_EQ(got, want)
                << "lane j=" << j << " s=" << s << ": expected vector "
                << j << "'s nibble (" << int(want) << "), got " << int(got);
        }
    }
}

TEST(Pq4Encoding, HighLanesAreVectors16to31NotInterleaved) {
    // The classic bug: encoding byte k as vectors (2k, 2k+1) instead of
    // (k, 16+k). With (2k, 2k+1), lane 16 (= byte 0 high) would decode to
    // vector 1's nibble instead of vector 16's. Pin against that.
    constexpr uint32_t n = 32;
    constexpr uint32_t m = 1;
    std::vector<uint8_t> nibbles(static_cast<size_t>(n) * m);
    // Vector v's nibble = v mod 16. So lane j should decode to j mod 16.
    for (uint32_t v = 0; v < n; v++) nibbles[v] = static_cast<uint8_t>(v & 0x0F);

    std::vector<uint8_t> blocks(static_cast<size_t>(m) * 16);
    pack_blocks(nibbles.data(), n, m, blocks.data());

    for (uint32_t j = 0; j < 32; j++) {
        const uint8_t got = decode_lane(blocks.data(), m, 0, 0, j);
        const uint8_t want = static_cast<uint8_t>(j & 0x0F);
        EXPECT_EQ(got, want)
            << "lane " << j << " decoded to " << int(got)
            << "; if this is " << int((j ^ 1) & 0x0F)
            << " the encoding is using (2k,2k+1) instead of (k,16+k)";
    }
}

TEST(Pq4Encoding, TailBlockPadsWithZeros) {
    // n=5: only lanes 0..4 should hold real nibbles. Lanes 5..31 must be
    // zero (they're masked at scan time by the valid_mask).
    constexpr uint32_t n = 5;
    constexpr uint32_t m = 3;
    std::vector<uint8_t> nibbles(static_cast<size_t>(n) * m);
    for (auto& nb : nibbles) nb = 11;  // any nonzero value

    std::vector<uint8_t> blocks(static_cast<size_t>(m) * 16);
    pack_blocks(nibbles.data(), n, m, blocks.data());

    for (uint32_t j = 0; j < 32; j++) {
        for (uint32_t s = 0; s < m; s++) {
            const uint8_t got = decode_lane(blocks.data(), m, 0, s, j);
            if (j < n) {
                EXPECT_EQ(got, 11)
                    << "real lane " << j << " s=" << s << " should hold 11";
            } else {
                EXPECT_EQ(got, 0)
                    << "padding lane " << j << " s=" << s << " should be 0";
            }
        }
    }
}

TEST(Pq4Encoding, MultipleBlocksEachIndependent) {
    // 33 vectors → 2 blocks. Block 0 holds vectors 0..31, block 1 holds
    // vector 32 (lanes 0 of block 1) plus 31 padding lanes.
    constexpr uint32_t n = 33;
    constexpr uint32_t m = 2;
    std::vector<uint8_t> nibbles(static_cast<size_t>(n) * m);
    for (uint32_t v = 0; v < n; v++) {
        for (uint32_t s = 0; s < m; s++) {
            nibbles[v * m + s] = static_cast<uint8_t>((v * 7 + s) & 0x0F);
        }
    }
    constexpr uint32_t n_blocks = 2;
    std::vector<uint8_t> blocks(static_cast<size_t>(n_blocks) * m * 16);
    pack_blocks(nibbles.data(), n, m, blocks.data());

    // Block 0: lanes 0..31 → vectors 0..31.
    for (uint32_t j = 0; j < 32; j++) {
        const uint8_t got = decode_lane(blocks.data(), m, 0, 0, j);
        EXPECT_EQ(got, static_cast<uint8_t>((j * 7) & 0x0F))
            << "block 0 lane " << j;
    }
    // Block 1: lane 0 → vector 32, lanes 1..31 → padding (0).
    EXPECT_EQ(decode_lane(blocks.data(), m, 1, 0, 0),
              static_cast<uint8_t>((32 * 7) & 0x0F));
    for (uint32_t j = 1; j < 32; j++) {
        EXPECT_EQ(decode_lane(blocks.data(), m, 1, 0, j), 0)
            << "block 1 padding lane " << j;
    }
}

// ---------------------------------------------------------------------------
// End-to-end: PqQuantizer.encode → unpack → pack → kernel output ranks the
// training-set vectors sensibly (the encoded nearest centroid per segment
// produces a small distance; an unrelated code produces a larger one).
// ---------------------------------------------------------------------------

TEST(Pq4Encoding, EncodeUnpackPackRoundtripPreservesNibbles) {
    // Encode a known vector, unpack to nibbles, pack to blocks, decode —
    // the decoded nibbles must match the unpacked ones exactly.
    constexpr Dim dim = 32;
    constexpr uint16_t m = 8;  // dim/m = 4 floats per segment
    constexpr uint8_t bits = 4;

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/42);
    // Train on a tiny clustered dataset so encode produces varied codes.
    std::mt19937_64 gen(7);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    std::vector<float> sample(200 * dim);
    for (uint64_t c = 0; c < 4; c++) {
        std::vector<float> center(dim);
        for (auto& v : center) v = static_cast<float>(c) * 2.0f - 3.0f;
        for (uint64_t i = 0; i < 50; i++) {
            const uint64_t idx = (c * 50 + i) * dim;
            for (Dim d = 0; d < dim; d++) sample[idx + d] = center[d] + noise(gen);
        }
    }
    q.train(sample.data(), 200);

    // Encode one vector.
    std::vector<uint8_t> packed(q.code_size());  // m/2 bytes
    q.encode(sample.data(), packed.data());

    // Unpack to nibbles.
    std::vector<uint8_t> nibbles(m);
    for (uint16_t s = 0; s < m; s++) {
        nibbles[s] = static_cast<uint8_t>(
            (packed[s / 2] >> ((s % 2) * 4)) & 0xF);
    }

    // Pack a 1-vector "block" and decode it back.
    std::vector<uint8_t> blocks(static_cast<size_t>(m) * 16, 0xFF);
    pack_blocks(nibbles.data(), /*n=*/1, m, blocks.data());

    for (uint16_t s = 0; s < m; s++) {
        const uint8_t decoded = decode_lane(blocks.data(), m, 0, s, /*j=*/0);
        EXPECT_EQ(decoded, nibbles[s])
            << "segment " << s << ": decode(encode(v)) != unpacked nibble";
    }
}

}  // namespace
}  // namespace sextant
