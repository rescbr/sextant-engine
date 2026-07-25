// Microbench: PQ LUT distance kernels.
//
// Compares three paths for the PQ-ADC filter inner loop (computing distances
// from a query LUT to a batch of PQ codes):
//   (A) float batch4   — current production path (`lut_distance_batch4`):
//                         4 codes/iter, float LUT, scalar gather (Mac NEON)
//                         or SVE2 gather (c4a). m gather rounds per 4 codes.
//   (B) float batch16  — control: same algorithm as (A) but at 16 codes/iter.
//                         Isolates the batch-width effect (register reuse,
//                         amortised LUT-row pointer setup) from the shuffle
//                         effect. Tells us how much of (C)'s win comes from
//                         batching wider vs from eliminating gather.
//   (C) split-table    — `simd::fastscan_block16`: 4-bank `vqtbl4q_u8` over
//                         uint8 LUT. 16 codes/iter, zero gather. The new path.
//
// Config: m=96 segments, K=256 centroids (our arxiv-nomic production config).
// Reports codes/sec and relative speedup. Run on Mac NEON AND c4a (V2 SVE2)
// to compare — the (A) path gets SVE2 on c4a but (C) is baseline NEON both
// places, so the relative gap differs.
//
// Build: ninja -C build bench_fastscan

#include "quant/pq_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/types.hpp"

#include <arm_neon.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sextant;

static constexpr uint32_t M = 96;    // segments (arxiv-nomic production)
static constexpr uint32_t K = 256;   // centroids per segment (8-bit codes)
static constexpr uint32_t BLOCK = 16;  // FastScan batch width

int main() {
    const uint32_t N_CODES = 1'000'000;  // total codes to evaluate
    const uint32_t N_BLOCKS = (N_CODES + BLOCK - 1) / BLOCK;
    const uint32_t N_QUADS = (N_CODES + 3) / 4;  // for batch4

    // Allocate LUTs and codes.
    // Float LUT (m × K) — used by (A) and (B).
    std::vector<float> lut_f32((size_t)M * K);
    for (auto& v : lut_f32) v = (rand() / float(RAND_MAX)) * 100.0f;

    // uint8 LUT (m × K) — used by (C). Build via quantize_lut_u8 so it's the
    // real production quantization, not random bytes.
    std::vector<uint8_t> lut_u8((size_t)M * K);
    float A, B;
    simd::quantize_lut_u8(lut_f32.data(), M, K, lut_u8.data(), &A, &B);

    // Codes in three layouts:
    //   - contiguous [N][m]   for (A) batch4 and (B) batch16 (scalar gather).
    //   - segment-major [N_BLOCKS][m][16] for (C) split-table.
    std::vector<uint8_t> codes_contig((size_t)N_CODES * M);
    for (auto& c : codes_contig) c = rand() & 0xFF;

    std::vector<uint8_t> codes_block((size_t)N_BLOCKS * M * BLOCK, 0);
    for (uint32_t b = 0; b < N_BLOCKS; b++) {
        for (uint32_t s = 0; s < M; s++) {
            for (uint32_t j = 0; j < BLOCK; j++) {
                const uint32_t code_idx = b * BLOCK + j;
                codes_block[((size_t)b * M + s) * BLOCK + j] =
                    (code_idx < N_CODES) ? codes_contig[(size_t)code_idx * M + s] : 0;
            }
        }
    }

    // Output buffers (kept warm to defeat dead-code elimination).
    std::vector<float>   out_f32(N_CODES, 0.0f);
    std::vector<uint32_t> out_u32(N_CODES, 0);

    const int ITERS = 50;  // repeat the whole sweep to amortise timer overhead

    auto report = [](const char* name, double secs, uint32_t n_codes) {
        const double codes_per_sec = (double)n_codes * ITERS / secs;
        printf("  %-22s %8.2f M codes/s  (%6.1f ns/code)\n",
               name, codes_per_sec / 1e6, secs * 1e9 / ((double)n_codes * ITERS));
        return codes_per_sec;
    };

    printf("PQ LUT distance microbench (m=%u, K=%u, N=%u codes, %d iters)\n",
           M, K, N_CODES, ITERS);
    printf("LUT: float32 (A,B) + uint8 quantized; A=%.4f B=%.4f\n\n", A, B);

    // -------------------------------------------------------------------------
    // (A) float batch4 — mirror the current production `lut_distance_batch4`
    //     NEON path. 4 codes/iter, 4×scalar gather + vld1q_f32 + vaddq_f32.
    // -------------------------------------------------------------------------
    double a_secs;
    {
        float sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < ITERS; it++) {
            for (uint32_t q = 0; q < N_QUADS; q++) {
                const uint8_t* c0 = codes_contig.data() + (size_t)(q*4+0) * M;
                const uint8_t* c1 = codes_contig.data() + (size_t)(q*4+1) * M;
                const uint8_t* c2 = codes_contig.data() + (size_t)(q*4+2) * M;
                const uint8_t* c3 = codes_contig.data() + (size_t)(q*4+3) * M;
                float32x4_t vacc = vdupq_n_f32(0.0f);
                for (uint32_t s = 0; s < M; s++) {
                    const float* row = lut_f32.data() + s * K;
                    float vals[4] = { row[c0[s]], row[c1[s]], row[c2[s]], row[c3[s]] };
                    vacc = vaddq_f32(vacc, vld1q_f32(vals));
                }
                // Store 4 distances back to the output buffer (warm).
                if (q*4+3 < N_CODES) {
                    out_f32[q*4+0] = vgetq_lane_f32(vacc, 0);
                    out_f32[q*4+1] = vgetq_lane_f32(vacc, 1);
                    out_f32[q*4+2] = vgetq_lane_f32(vacc, 2);
                    out_f32[q*4+3] = vgetq_lane_f32(vacc, 3);
                }
                if ((q & 0x3FF) == 0) sink += vaddvq_f32(vacc) * 1e-30f;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        a_secs = std::chrono::duration<double>(t1 - t0).count();
        const double cps = report("(A) float batch4", a_secs, N_CODES);
        printf("                        [sink=%.3f]\n", (double)sink);
        (void)cps;
    }

    // -------------------------------------------------------------------------
    // (B) float batch16 — control: same float gather as (A) but 16-wide.
    //     Two uint32x4 / float32x4 accumulators covering 16 codes.
    // -------------------------------------------------------------------------
    double b_secs;
    {
        float sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < ITERS; it++) {
            for (uint32_t b = 0; b < N_BLOCKS; b++) {
                const uint8_t* cb_base = codes_contig.data() + (size_t)b * BLOCK * M;
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);
                for (uint32_t s = 0; s < M; s++) {
                    const float* row = lut_f32.data() + s * K;
                    // Contiguous layout [N][m]: code j of this block at
                    // cb_base + j*M + s. Strided access — the realistic cost
                    // of gather on contiguous code storage.
                    alignas(16) float v0[4] = {
                        row[cb_base[0*M + s]], row[cb_base[1*M + s]],
                        row[cb_base[2*M + s]], row[cb_base[3*M + s]] };
                    alignas(16) float v1[4] = {
                        row[cb_base[4*M + s]], row[cb_base[5*M + s]],
                        row[cb_base[6*M + s]], row[cb_base[7*M + s]] };
                    alignas(16) float v2[4] = {
                        row[cb_base[8*M + s]], row[cb_base[9*M + s]],
                        row[cb_base[10*M + s]], row[cb_base[11*M + s]] };
                    alignas(16) float v3[4] = {
                        row[cb_base[12*M + s]], row[cb_base[13*M + s]],
                        row[cb_base[14*M + s]], row[cb_base[15*M + s]] };
                    acc0 = vaddq_f32(acc0, vld1q_f32(v0));
                    acc1 = vaddq_f32(acc1, vld1q_f32(v1));
                    acc2 = vaddq_f32(acc2, vld1q_f32(v2));
                    acc3 = vaddq_f32(acc3, vld1q_f32(v3));
                }
                if ((b & 0x3FF) == 0) {
                    sink += vaddvq_f32(acc0) + vaddvq_f32(acc1) +
                            vaddvq_f32(acc2) + vaddvq_f32(acc3);
                }
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        b_secs = std::chrono::duration<double>(t1 - t0).count();
        report("(B) float batch16", b_secs, N_CODES);
        printf("                        [sink=%.3f]\n", (double)sink);
    }

    // -------------------------------------------------------------------------
    // (C) split-table FastScan — single-block API. Reloads LUT per block.
    //     Kept for comparison; NOT the production path.
    // -------------------------------------------------------------------------
    double c_secs;
    {
        uint64_t sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < ITERS; it++) {
            for (uint32_t b = 0; b < N_BLOCKS; b++) {
                const uint8_t* block = codes_block.data() + (size_t)b * M * BLOCK;
                uint32_t out[BLOCK];
                simd::fastscan_block16(block, lut_u8.data(), M, 0xFFFF, out);
                if ((b & 0x3FF) == 0) {
                    for (uint32_t j = 0; j < BLOCK; j++) sink += out[j];
                }
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        c_secs = std::chrono::duration<double>(t1 - t0).count();
        report("(C) split-table/block", c_secs, N_CODES);
        printf("                        [sink=%llu]\n", (unsigned long long)sink);
    }

    // -------------------------------------------------------------------------
    // (D) split-table FastScan — segment-major many-blocks API.
    //     The production path. m outer passes; one segment's tables live in
    //     registers across all blocks. LUT-load cost amortises.
    // -------------------------------------------------------------------------
    // We measure at several batch sizes to show the amortisation curve: the
    // graph-search batch is 4-1024 codes (i.e. 1-64 blocks of 16).
    printf("\n  (D) split-table/many  (segment-major, by batch size):\n");
    for (uint32_t batch_blocks : {1u, 4u, 16u, 64u, 256u, N_BLOCKS}) {
        const uint32_t nb = batch_blocks;
        const uint32_t n_codes_in_batch = nb * BLOCK;
        // Reusable buffers.
        std::vector<uint16_t> masks(nb, 0xFFFF);
        std::vector<uint32_t> psum((size_t)nb * 16, 0);
        // For batch_blocks == N_BLOCKS the whole codes_block fits; for smaller
        // batches we just reuse the first nb blocks each iter (same data,
        // fair timing).
        uint64_t sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        // For each timing iter, sweep the whole codes_block in chunks of nb.
        // Pad the buffer so the last chunk is always full (avoid OOB).
        const uint32_t nb_padded = ((N_BLOCKS + nb - 1) / nb) * nb;
        std::vector<uint8_t> codes_block_padded((size_t)nb_padded * M * BLOCK, 0);
        std::memcpy(codes_block_padded.data(), codes_block.data(),
                    (size_t)N_BLOCKS * M * BLOCK);
        const uint32_t chunks = nb_padded / nb;
        for (int it = 0; it < ITERS; it++) {
            for (uint32_t o = 0; o < chunks; o++) {
                const uint8_t* blk_ptr = codes_block_padded.data() +
                    (size_t)o * nb * M * BLOCK;
                simd::fastscan_many(blk_ptr, nb, lut_u8.data(), M,
                                    masks.data(), psum.data());
                if ((o & 0xF) == 0) {
                    for (uint32_t j = 0; j < nb * 16; j++) sink += psum[j];
                }
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        // Throughput: count real codes (not padding) — equals nb * BLOCK per chunk
        // times chunks times ITERS, minus padding tail. For simplicity use
        // nb*BLOCK*chunks*ITERS; padding is ≤ nb-1 blocks per iter, negligible
        // at the larger batches (and visible only at nb == N_BLOCKS where it's 0).
        const uint64_t total_codes = (uint64_t)nb * BLOCK * chunks * ITERS;
        const double cps = (double)total_codes / secs;
        printf("    batch=%4u codes (%2u blocks): %8.2f M codes/s  [sink=%llu]\n",
               n_codes_in_batch, nb, cps / 1e6, (unsigned long long)sink);
        if (batch_blocks == N_BLOCKS) {
            printf("  --- (D)/(A) at full sweep: %.2fx ---\n", a_secs / secs);
        }
    }

    printf("\n--- summary ---\n");
    printf("  (B)/(A) = %.2fx   (batch-width 4 -> 16, float gather)\n", a_secs / b_secs);
    printf("  (C)/(A) = %.2fx   (per-block split-table — LUT-reload-bound)\n", a_secs / c_secs);
    printf("\nPass criterion: (D)/(A) at batch>=64 blocks >= 1.5x\n");
    return 0;
}
