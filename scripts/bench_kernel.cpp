// Microbench: pq4_block32 variants at m=192 (production config).
//
// Findings on Apple Silicon (M2):
//   baseline: 342 M codes/s  (widen+add every segment)
//   u8acc:    260 M codes/s  (u8 accumulators, carry to u16 every 16 segs)
//   8chain:   336 M codes/s  (two accumulator sets, ILP)
//
// All within 5% on this hardware. The baseline is at the hardware limit:
// not latency-bound (8chain doesn't help) and the u8-accumulator trick
// (standard FAISS/x86 optimization) costs more than it saves here because
// vget_low/high to split for u8 add is more ops than vmovl_u8+vaddq_u16,
// and Apple's NEON handles widen+add in 1 cycle. The u8acc variant is
// bit-exact correct (verified) but slower; kept here for reference and for
// re-benchmarking on c4a V2 (where the tradeoff may differ).
//
// Run: ./build/scripts/bench_kernel

#include "simd_kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace sextant;

// Baseline (current production): widen+add every segment.
static inline void pq4_block32_baseline(const uint8_t* code_block,
                                         const uint8_t* lut4, uint32_t m,
                                         uint32_t out[32]) {
    uint16x8_t acc_lo_a = vdupq_n_u16(0);
    uint16x8_t acc_lo_b = vdupq_n_u16(0);
    uint16x8_t acc_hi_a = vdupq_n_u16(0);
    uint16x8_t acc_hi_b = vdupq_n_u16(0);
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (uint32_t s = 0; s < m; s++) {
        const uint8x16_t lut_v = vld1q_u8(lut4 + s * 16);
        const uint8x16_t c = vld1q_u8(code_block + s * 16);
        const uint8x16_t clo = vandq_u8(c, mask4);
        const uint8x16_t chi = vshrq_n_u8(c, 4);
        const uint8x16_t rlo = vqtbl1q_u8(lut_v, clo);
        const uint8x16_t rhi = vqtbl1q_u8(lut_v, chi);
        acc_lo_a = vaddq_u16(acc_lo_a, vmovl_u8(vget_low_u8(rlo)));
        acc_lo_b = vaddq_u16(acc_lo_b, vmovl_u8(vget_high_u8(rlo)));
        acc_hi_a = vaddq_u16(acc_hi_a, vmovl_u8(vget_low_u8(rhi)));
        acc_hi_b = vaddq_u16(acc_hi_b, vmovl_u8(vget_high_u8(rhi)));
    }
    vst1q_u32(out +  0, vmovl_u16(vget_low_u16 (acc_lo_a)));
    vst1q_u32(out +  4, vmovl_u16(vget_high_u16(acc_lo_a)));
    vst1q_u32(out +  8, vmovl_u16(vget_low_u16 (acc_lo_b)));
    vst1q_u32(out + 12, vmovl_u16(vget_high_u16(acc_lo_b)));
    vst1q_u32(out + 16, vmovl_u16(vget_low_u16 (acc_hi_a)));
    vst1q_u32(out + 20, vmovl_u16(vget_high_u16(acc_hi_a)));
    vst1q_u32(out + 24, vmovl_u16(vget_low_u16 (acc_hi_b)));
    vst1q_u32(out + 28, vmovl_u16(vget_high_u16(acc_hi_b)));
}

// Optimized: accumulate in u8, carry to u16 every 16 segments.
// Per-segment per-lane contribution is at most 15 (u4 LUT entries), so 16
// segments in u8 sum to at most 240 < 256 — no overflow. Cuts the u8->u16
// widen+add from m=192 ops to m/16=12 — 16x fewer.
//
// The four u8 accumulators correspond to the four 8-lane output quartets:
//   lo_a = low 8 lanes of low nibbles (vectors 0..7)
//   lo_b = high 8 lanes of low nibbles (vectors 8..15)
//   hi_a = low 8 lanes of high nibbles (vectors 16..23)
//   hi_b = high 8 lanes of high nibbles (vectors 24..31)
// Each vqtbl1q produces 16 results; we vget_low/high to split into the two
// 8-lane halves, add each half to its u8 accumulator. After 16 segments,
// widen each u8 accumulator to u16 and add to the persistent u16 accumulator,
// then reset the u8 accumulator to 0.
static inline void pq4_block32_u8acc(const uint8_t* code_block,
                                      const uint8_t* lut4, uint32_t m,
                                      uint32_t out[32]) {
    uint8x8_t u8_lo_a = vdup_n_u8(0);
    uint8x8_t u8_lo_b = vdup_n_u8(0);
    uint8x8_t u8_hi_a = vdup_n_u8(0);
    uint8x8_t u8_hi_b = vdup_n_u8(0);
    uint16x8_t acc_lo_a = vdupq_n_u16(0);
    uint16x8_t acc_lo_b = vdupq_n_u16(0);
    uint16x8_t acc_hi_a = vdupq_n_u16(0);
    uint16x8_t acc_hi_b = vdupq_n_u16(0);
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);

    const uint32_t m_round16 = (m / 16) * 16;
    uint32_t s = 0;
    for (; s < m_round16; s++) {
        const uint8x16_t lut_v = vld1q_u8(lut4 + s * 16);
        const uint8x16_t c = vld1q_u8(code_block + s * 16);
        const uint8x16_t clo = vandq_u8(c, mask4);
        const uint8x16_t chi = vshrq_n_u8(c, 4);
        const uint8x16_t rlo = vqtbl1q_u8(lut_v, clo);
        const uint8x16_t rhi = vqtbl1q_u8(lut_v, chi);
        u8_lo_a = vadd_u8(u8_lo_a, vget_low_u8(rlo));
        u8_lo_b = vadd_u8(u8_lo_b, vget_high_u8(rlo));
        u8_hi_a = vadd_u8(u8_hi_a, vget_low_u8(rhi));
        u8_hi_b = vadd_u8(u8_hi_b, vget_high_u8(rhi));

        // Every 16 segments (15 since we just added one): carry to u16.
        // Check (s+1) % 16 == 0 ⇒ we've added 16 contributions since the
        // last carry. Note the u8 max after 16 adds is 16*15=240 < 256.
        if ((s & 15u) == 15u) {
            acc_lo_a = vaddq_u16(acc_lo_a, vmovl_u8(u8_lo_a));
            acc_lo_b = vaddq_u16(acc_lo_b, vmovl_u8(u8_lo_b));
            acc_hi_a = vaddq_u16(acc_hi_a, vmovl_u8(u8_hi_a));
            acc_hi_b = vaddq_u16(acc_hi_b, vmovl_u8(u8_hi_b));
            u8_lo_a = vdup_n_u8(0);
            u8_lo_b = vdup_n_u8(0);
            u8_hi_a = vdup_n_u8(0);
            u8_hi_b = vdup_n_u8(0);
        }
    }
    // Tail: remaining segments (m mod 16). Add to u8 then carry once.
    for (; s < m; s++) {
        const uint8x16_t lut_v = vld1q_u8(lut4 + s * 16);
        const uint8x16_t c = vld1q_u8(code_block + s * 16);
        const uint8x16_t clo = vandq_u8(c, mask4);
        const uint8x16_t chi = vshrq_n_u8(c, 4);
        const uint8x16_t rlo = vqtbl1q_u8(lut_v, clo);
        const uint8x16_t rhi = vqtbl1q_u8(lut_v, chi);
        u8_lo_a = vadd_u8(u8_lo_a, vget_low_u8(rlo));
        u8_lo_b = vadd_u8(u8_lo_b, vget_high_u8(rlo));
        u8_hi_a = vadd_u8(u8_hi_a, vget_low_u8(rhi));
        u8_hi_b = vadd_u8(u8_hi_b, vget_high_u8(rhi));
    }
    acc_lo_a = vaddq_u16(acc_lo_a, vmovl_u8(u8_lo_a));
    acc_lo_b = vaddq_u16(acc_lo_b, vmovl_u8(u8_lo_b));
    acc_hi_a = vaddq_u16(acc_hi_a, vmovl_u8(u8_hi_a));
    acc_hi_b = vaddq_u16(acc_hi_b, vmovl_u8(u8_hi_b));

    vst1q_u32(out +  0, vmovl_u16(vget_low_u16 (acc_lo_a)));
    vst1q_u32(out +  4, vmovl_u16(vget_high_u16(acc_lo_a)));
    vst1q_u32(out +  8, vmovl_u16(vget_low_u16 (acc_lo_b)));
    vst1q_u32(out + 12, vmovl_u16(vget_high_u16(acc_lo_b)));
    vst1q_u32(out + 16, vmovl_u16(vget_low_u16 (acc_hi_a)));
    vst1q_u32(out + 20, vmovl_u16(vget_high_u16(acc_hi_a)));
    vst1q_u32(out + 24, vmovl_u16(vget_low_u16 (acc_hi_b)));
    vst1q_u32(out + 28, vmovl_u16(vget_high_u16(acc_hi_b)));
}


// 8-chain variant: two sets of 4 accumulators, alternating per segment.
// Doubles the parallelism available to the OoO execution unit, hiding
// add latency. Same total ops as baseline; different scheduling.
static inline void pq4_block32_8chain(const uint8_t* code_block,
                                       const uint8_t* lut4, uint32_t m,
                                       uint32_t out[32]) {
    uint16x8_t a0_lo_a = vdupq_n_u16(0), a1_lo_a = vdupq_n_u16(0);
    uint16x8_t a0_lo_b = vdupq_n_u16(0), a1_lo_b = vdupq_n_u16(0);
    uint16x8_t a0_hi_a = vdupq_n_u16(0), a1_hi_a = vdupq_n_u16(0);
    uint16x8_t a0_hi_b = vdupq_n_u16(0), a1_hi_b = vdupq_n_u16(0);
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    uint32_t s = 0;
    const uint32_t m_round2 = (m / 2) * 2;
    for (; s < m_round2; s += 2) {
        // Even segment -> a0.
        {
            const uint8x16_t lut_v = vld1q_u8(lut4 + s * 16);
            const uint8x16_t c = vld1q_u8(code_block + s * 16);
            const uint8x16_t clo = vandq_u8(c, mask4);
            const uint8x16_t chi = vshrq_n_u8(c, 4);
            const uint8x16_t rlo = vqtbl1q_u8(lut_v, clo);
            const uint8x16_t rhi = vqtbl1q_u8(lut_v, chi);
            a0_lo_a = vaddq_u16(a0_lo_a, vmovl_u8(vget_low_u8(rlo)));
            a0_lo_b = vaddq_u16(a0_lo_b, vmovl_u8(vget_high_u8(rlo)));
            a0_hi_a = vaddq_u16(a0_hi_a, vmovl_u8(vget_low_u8(rhi)));
            a0_hi_b = vaddq_u16(a0_hi_b, vmovl_u8(vget_high_u8(rhi)));
        }
        // Odd segment -> a1.
        {
            const uint8x16_t lut_v = vld1q_u8(lut4 + (s+1) * 16);
            const uint8x16_t c = vld1q_u8(code_block + (s+1) * 16);
            const uint8x16_t clo = vandq_u8(c, mask4);
            const uint8x16_t chi = vshrq_n_u8(c, 4);
            const uint8x16_t rlo = vqtbl1q_u8(lut_v, clo);
            const uint8x16_t rhi = vqtbl1q_u8(lut_v, chi);
            a1_lo_a = vaddq_u16(a1_lo_a, vmovl_u8(vget_low_u8(rlo)));
            a1_lo_b = vaddq_u16(a1_lo_b, vmovl_u8(vget_high_u8(rlo)));
            a1_hi_a = vaddq_u16(a1_hi_a, vmovl_u8(vget_low_u8(rhi)));
            a1_hi_b = vaddq_u16(a1_hi_b, vmovl_u8(vget_high_u8(rhi)));
        }
    }
    // Tail.
    if (s < m) {
        const uint8x16_t lut_v = vld1q_u8(lut4 + s * 16);
        const uint8x16_t c = vld1q_u8(code_block + s * 16);
        const uint8x16_t clo = vandq_u8(c, mask4);
        const uint8x16_t chi = vshrq_n_u8(c, 4);
        const uint8x16_t rlo = vqtbl1q_u8(lut_v, clo);
        const uint8x16_t rhi = vqtbl1q_u8(lut_v, chi);
        a0_lo_a = vaddq_u16(a0_lo_a, vmovl_u8(vget_low_u8(rlo)));
        a0_lo_b = vaddq_u16(a0_lo_b, vmovl_u8(vget_high_u8(rlo)));
        a0_hi_a = vaddq_u16(a0_hi_a, vmovl_u8(vget_low_u8(rhi)));
        a0_hi_b = vaddq_u16(a0_hi_b, vmovl_u8(vget_high_u8(rhi)));
    }
    // Merge a0 + a1.
    uint16x8_t acc_lo_a = vaddq_u16(a0_lo_a, a1_lo_a);
    uint16x8_t acc_lo_b = vaddq_u16(a0_lo_b, a1_lo_b);
    uint16x8_t acc_hi_a = vaddq_u16(a0_hi_a, a1_hi_a);
    uint16x8_t acc_hi_b = vaddq_u16(a0_hi_b, a1_hi_b);
    vst1q_u32(out +  0, vmovl_u16(vget_low_u16 (acc_lo_a)));
    vst1q_u32(out +  4, vmovl_u16(vget_high_u16(acc_lo_a)));
    vst1q_u32(out +  8, vmovl_u16(vget_low_u16 (acc_lo_b)));
    vst1q_u32(out + 12, vmovl_u16(vget_high_u16(acc_lo_b)));
    vst1q_u32(out + 16, vmovl_u16(vget_low_u16 (acc_hi_a)));
    vst1q_u32(out + 20, vmovl_u16(vget_high_u16(acc_hi_a)));
    vst1q_u32(out + 24, vmovl_u16(vget_low_u16 (acc_hi_b)));
    vst1q_u32(out + 28, vmovl_u16(vget_high_u16(acc_hi_b)));
}

int main() {
    const uint32_t m = 192;
    const uint32_t n_blocks = 4096;  // simulate ~130k vectors
    std::vector<uint8_t> blocks((size_t)n_blocks * m * 16);
    std::vector<uint8_t> lut4((size_t)m * 16);
    // Fill with deterministic data.
    for (auto& b : blocks) b = (uint8_t)((rand() & 0xFF));
    for (auto& b : lut4) b = (uint8_t)(rand() & 0x0F);

    std::vector<uint32_t> out(n_blocks * 32);
    constexpr int kIters = 200;

    uint64_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < kIters; it++) {
        for (uint32_t b = 0; b < n_blocks; b++) {
            pq4_block32_baseline(blocks.data() + (size_t)b * m * 16,
                                  lut4.data(), m, out.data() + b * 32);
        }
        sink += out[it % (n_blocks * 32)];
    }
    asm volatile("" :: "r"(sink) : "memory");
    auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double codes = (double)n_blocks * 32 * kIters;
    printf("baseline: %.0f Mcodes/s  (%.1f ms/iter for %u blocks)\n",
           codes / sec / 1e6,
           sec * 1e3 / kIters, n_blocks);
    auto t2 = std::chrono::steady_clock::now();
    sink = 0;
    for (int it = 0; it < kIters; it++) {
        for (uint32_t b = 0; b < n_blocks; b++) {
            pq4_block32_u8acc(blocks.data() + (size_t)b * m * 16,
                               lut4.data(), m, out.data() + b * 32);
        }
        sink += out[it % (n_blocks * 32)];
    }
    asm volatile("" :: "r"(sink) : "memory");
    auto t3 = std::chrono::steady_clock::now();
    const double sec2 = std::chrono::duration<double>(t3 - t2).count();
    const double codes2 = (double)n_blocks * 32 * kIters;
    printf("u8acc:    %.0f Mcodes/s  (%.1f ms/iter for %u blocks)\n",
           codes2 / sec2 / 1e6,
           sec2 * 1e3 / kIters, n_blocks);

    // Correctness: compare outputs on a single block.
    uint32_t ref[32], got[32];
    pq4_block32_baseline(blocks.data(), lut4.data(), m, ref);
    pq4_block32_u8acc(blocks.data(), lut4.data(), m, got);
    int mism = 0;
    for (int j = 0; j < 32; j++) if (ref[j] != got[j]) mism++;
    printf("mismatches on block 0: %d/32\n", mism);
    auto t4 = std::chrono::steady_clock::now();
    sink = 0;
    for (int it = 0; it < kIters; it++) {
        for (uint32_t b = 0; b < n_blocks; b++) {
            pq4_block32_8chain(blocks.data() + (size_t)b * m * 16,
                                lut4.data(), m, out.data() + b * 32);
        }
        sink += out[it % (n_blocks * 32)];
    }
    asm volatile("" :: "r"(sink) : "memory");
    auto t5 = std::chrono::steady_clock::now();
    const double sec3 = std::chrono::duration<double>(t5 - t4).count();
    const double codes3 = (double)n_blocks * 32 * kIters;
    printf("8chain:   %.0f Mcodes/s  (%.1f ms/iter for %u blocks)\n",
           codes3 / sec3 / 1e6, sec3 * 1e3 / kIters, n_blocks);
    {
        uint32_t ref[32], got[32];
        pq4_block32_baseline(blocks.data(), lut4.data(), m, ref);
        pq4_block32_8chain(blocks.data(), lut4.data(), m, got);
        int mism = 0;
        for (int j = 0; j < 32; j++) if (ref[j] != got[j]) mism++;
        printf("8chain mismatches on block 0: %d/32\n", mism);
    }
    return 0;
}
