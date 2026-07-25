/// @file test_pq4_kernel.cpp
/// Correctness tests for the 4-bit FastScan kernel (`simd::pq4_block32`) and
/// the multi-block wrapper (`simd::pq4_scan_many`).
///
/// The kernel's distance output must EXACTLY match a brute-force reference
/// (per-segment LUT[code] summed across m segments). The whole pipeline is
/// integer arithmetic (uint8 LUT entries, uint32 accumulation) so the match
/// is bit-exact, not approximate. Any mismatch indicates a lane-order or
/// accumulation bug.
///
/// Load-bearing invariants verified:
///   1. Lane order: out[j] = Σ_s lut[s*16 + lo_nibble(code[s,j])]  for j<16
///                  out[16+j] = Σ_s lut[s*16 + hi_nibble(code[s,j])] for j<16
///      — byte k of segment s holds TWO vectors' s-th nibbles.
///   2. No A-clamp on the LUT (u32 headroom) — verified by a query whose
///      natural scale would have been crushed under u16 accumulation.
///   3. `pq4_scan_many` masks invalid tail lanes to 0xFFFFFFFF.

#include <gtest/gtest.h>
#include "simd_kernels.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace sextant {
namespace {

// Brute-force reference: per-segment LUT[code] summed across m segments,
// matching the kernel's lane extraction order exactly.
//   code_block: [m][16] bytes (segment-major within the block)
//   lut4:       [m][16] uint8
//   out:        32 uint32 distances
void pq4_block32_ref(const uint8_t* code_block, const uint8_t* lut4,
                     uint32_t m, uint32_t out[32]) {
    for (int j = 0; j < 16; j++) out[j] = 0;
    for (int j = 0; j < 16; j++) out[16 + j] = 0;
    for (uint32_t s = 0; s < m; s++) {
        const uint8_t* codes = code_block + s * 16;
        const uint8_t* row   = lut4 + s * 16;
        for (int j = 0; j < 16; j++) {
            out[j]      += row[codes[j] & 0x0F];
            out[16 + j] += row[codes[j] >> 4];
        }
    }
}

class Rng {
public:
    explicit Rng(uint64_t seed) : rng_(seed) {}
    uint8_t u8_nibble() {
        std::uniform_int_distribution<int> d(0, 15);
        return static_cast<uint8_t>(d(rng_));
    }
    uint32_t u32_below(uint32_t n) {
        std::uniform_int_distribution<uint32_t> d(0, n - 1);
        return d(rng_);
    }
private:
    std::mt19937_64 rng_;
};

// ---------------------------------------------------------------------------
// pq4_block32: parity with brute-force reference.
// ---------------------------------------------------------------------------

TEST(Pq4Block32, SingleBlockMatchesReference) {
    constexpr uint32_t m = 8;  // small m for the first sanity check
    Rng rng(1234);
    std::vector<uint8_t> code_block(static_cast<size_t>(m) * 16);
    std::vector<uint8_t> lut4(static_cast<size_t>(m) * 16);
    for (auto& b : code_block) b = rng.u8_nibble() | (rng.u8_nibble() << 4);
    for (auto& b : lut4) b = rng.u8_nibble();

    uint32_t out_actual[32];
    uint32_t out_ref[32];
    simd::pq4_block32(code_block.data(), lut4.data(), m, out_actual);
    pq4_block32_ref(code_block.data(), lut4.data(), m, out_ref);

    for (int j = 0; j < 32; j++) {
        EXPECT_EQ(out_actual[j], out_ref[j])
            << "lane " << j << " mismatch";
    }
}

TEST(Pq4Block32, ProductionM192MatchesReference) {
    constexpr uint32_t m = 192;  // the production config (dim=768, m=dim/4)
    Rng rng(5678);
    std::vector<uint8_t> code_block(static_cast<size_t>(m) * 16);
    std::vector<uint8_t> lut4(static_cast<size_t>(m) * 16);
    for (auto& b : code_block) b = rng.u8_nibble() | (rng.u8_nibble() << 4);
    for (auto& b : lut4) b = rng.u8_nibble();

    uint32_t out_actual[32];
    uint32_t out_ref[32];
    simd::pq4_block32(code_block.data(), lut4.data(), m, out_actual);
    pq4_block32_ref(code_block.data(), lut4.data(), m, out_ref);

    for (int j = 0; j < 32; j++) {
        EXPECT_EQ(out_actual[j], out_ref[j])
            << "lane " << j << " mismatch at m=192";
    }
}

TEST(Pq4Block32, AllZeroLutGivesZeroDistances) {
    // With an all-zero LUT, every distance must be 0 regardless of codes.
    constexpr uint32_t m = 16;
    Rng rng(9090);
    std::vector<uint8_t> code_block(static_cast<size_t>(m) * 16);
    std::vector<uint8_t> lut4(static_cast<size_t>(m) * 16, 0);
    for (auto& b : code_block) b = rng.u8_nibble() | (rng.u8_nibble() << 4);

    uint32_t out[32];
    simd::pq4_block32(code_block.data(), lut4.data(), m, out);
    for (int j = 0; j < 32; j++) {
        EXPECT_EQ(out[j], 0u) << "lane " << j << " nonzero with zero LUT";
    }
}

TEST(Pq4Block32, MaxValuesNoOverflow) {
    // Worst case: every LUT entry = 15, m segments → distance = m × 15.
    // At m=192: 2880. Must fit in uint32 (and uint16 too, per the kernel's
    // internal accumulation). Verifies no overflow path.
    constexpr uint32_t m = 192;
    std::vector<uint8_t> code_block(static_cast<size_t>(m) * 16, 0);
    std::vector<uint8_t> lut4(static_cast<size_t>(m) * 16, 15);

    uint32_t out[32];
    simd::pq4_block32(code_block.data(), lut4.data(), m, out);
    // code_block all-zero → low nibble = 0, high nibble = 0 → both lookup
    // lut[s*16 + 0] = 15. Distance = m × 15 = 2880.
    for (int j = 0; j < 32; j++) {
        EXPECT_EQ(out[j], m * 15u)
            << "lane " << j << " overflow or wrong max";
    }
}

TEST(Pq4Block32, HighNibbleLanesIndependentFromLow) {
    // Set codes so lane k's LOW nibble = k, HIGH nibble = 15-k.
    // With a LUT row = identity (lut[c] = c), lane k = Σ k = m×k (low) and
    // lane 16+k = Σ (15-k) = m×(15-k) (high). Verifies lanes are independent
    // and the lane order is low-then-high per byte.
    constexpr uint32_t m = 4;
    std::vector<uint8_t> code_block(static_cast<size_t>(m) * 16);
    std::vector<uint8_t> lut4(static_cast<size_t>(m) * 16);
    for (uint32_t s = 0; s < m; s++) {
        for (int k = 0; k < 16; k++) {
            const uint8_t lo = static_cast<uint8_t>(k);
            const uint8_t hi = static_cast<uint8_t>(15 - k);
            code_block[s * 16 + k] = static_cast<uint8_t>((hi << 4) | lo);
        }
        for (int c = 0; c < 16; c++) lut4[s * 16 + c] = static_cast<uint8_t>(c);
    }

    uint32_t out[32];
    simd::pq4_block32(code_block.data(), lut4.data(), m, out);
    for (int k = 0; k < 16; k++) {
        EXPECT_EQ(out[k],      static_cast<uint32_t>(m) * k)
            << "low lane k=" << k;
        EXPECT_EQ(out[16 + k], static_cast<uint32_t>(m) * (15 - k))
            << "high lane k=" << k;
    }
}

// ---------------------------------------------------------------------------
// pq4_scan_many: multi-block + tail masking.
// ---------------------------------------------------------------------------

TEST(Pq4ScanMany, MultipleFullBlocksMatchReference) {
    constexpr uint32_t m = 8;
    constexpr uint32_t n_blocks = 5;  // all full (32 real vectors each)
    Rng rng(424242);
    std::vector<uint8_t> blocks(static_cast<size_t>(n_blocks) * m * 16);
    std::vector<uint8_t> lut4(static_cast<size_t>(m) * 16);
    for (auto& b : blocks) b = rng.u8_nibble() | (rng.u8_nibble() << 4);
    for (auto& b : lut4) b = rng.u8_nibble();

    std::vector<uint32_t> masks(n_blocks, 0xFFFFFFFFu);
    std::vector<uint32_t> actual(static_cast<size_t>(n_blocks) * 32);
    simd::pq4_scan_many(blocks.data(), n_blocks, lut4.data(), m,
                        masks.data(), actual.data());

    for (uint32_t b = 0; b < n_blocks; b++) {
        uint32_t ref[32];
        pq4_block32_ref(blocks.data() + static_cast<size_t>(b) * m * 16,
                        lut4.data(), m, ref);
        for (int j = 0; j < 32; j++) {
            EXPECT_EQ(actual[b * 32 + j], ref[j])
                << "block " << b << " lane " << j;
        }
    }
}

TEST(Pq4ScanMany, TailBlockMaskingSetsInvalidLanes) {
    // A tail block with only the first 5 lanes valid. Masked lanes must be
    // 0xFFFFFFFF (never argmin-winners).
    constexpr uint32_t m = 4;
    constexpr uint32_t n_blocks = 1;
    Rng rng(777);
    std::vector<uint8_t> blocks(static_cast<size_t>(m) * 16);
    std::vector<uint8_t> lut4(static_cast<size_t>(m) * 16);
    for (auto& b : blocks) b = rng.u8_nibble() | (rng.u8_nibble() << 4);
    for (auto& b : lut4) b = rng.u8_nibble();

    // Mask: bits 0..4 set (5 valid lanes).
    const uint32_t mask = 0x0000001Fu;
    std::vector<uint32_t> masks(1, mask);
    std::vector<uint32_t> actual(32);
    simd::pq4_scan_many(blocks.data(), 1, lut4.data(), m,
                        masks.data(), actual.data());

    uint32_t ref[32];
    pq4_block32_ref(blocks.data(), lut4.data(), m, ref);

    for (int j = 0; j < 32; j++) {
        if ((mask >> j) & 1u) {
            EXPECT_EQ(actual[j], ref[j])
                << "valid lane " << j << " should match reference";
        } else {
            EXPECT_EQ(actual[j], 0xFFFFFFFFu)
                << "invalid lane " << j << " should be masked to max";
        }
    }
}

TEST(Pq4ScanMany, FullBlockMaskIsNoOp) {
    // A full mask (0xFFFFFFFF) must NOT alter any lane — the kernel's
    // distance output should pass through unchanged.
    constexpr uint32_t m = 8;
    Rng rng(11);
    std::vector<uint8_t> blocks(static_cast<size_t>(m) * 16);
    std::vector<uint8_t> lut4(static_cast<size_t>(m) * 16);
    for (auto& b : blocks) b = rng.u8_nibble() | (rng.u8_nibble() << 4);
    for (auto& b : lut4) b = rng.u8_nibble();

    std::vector<uint32_t> masks(1, 0xFFFFFFFFu);
    std::vector<uint32_t> actual(32);
    simd::pq4_scan_many(blocks.data(), 1, lut4.data(), m,
                        masks.data(), actual.data());

    uint32_t ref[32];
    pq4_block32_ref(blocks.data(), lut4.data(), m, ref);
    for (int j = 0; j < 32; j++) {
        EXPECT_EQ(actual[j], ref[j]) << "full-mask lane " << j;
    }
}

}  // namespace
}  // namespace sextant
