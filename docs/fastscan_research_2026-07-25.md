# PQ-FastScan for 8-bit codes — feasibility research (2026-07-25)

> **Status: historical (July 2026 flat-graph era) — kept for reference; the 4-bank `vqtbl4q_u8` design carried into the tree path: src/tree/plane.hpp, leaf_coder.hpp.**

> Decided design: **4-bank `vqtbl4q_u8` split-table** (Quicker-ADC port).
> This doc records the analysis that selected it, the alternatives ruled
> out, and the empirical verifications. Lives behind
> `~/.local/state/maki/plans/measured-improved-iguana.md`.

## The constraint that shapes everything

NEON's `vqtbl*` family covers at most 64 bytes (4 registers × 16 bytes),
and indices ≥ table-size return 0. FAISS's stock FastScan sidesteps this
by training PQ at **4 bits (K=16)** so the LUT row fits in 16 entries
(one `vqtbl1q_u8` table). For 8-bit PQ (K=256, our config) the LUT row
is 256 entries and the shuffle trick "doesn't fit" — FAISS ships no
8-bit FastScan kernel.

Our workload cannot drop to 4-bit:
`docs/optimization_levers_and_attribution.md:66` documents 4-bit recall
collapsing to 0.72-0.76 (navigation broken; 16 centroids/segment
insufficient at our recall target 0.9931).

## Approaches ruled out (do not revisit)

### 1. Switch to 4-bit PQ + rerank (FAISS pattern)
Recall collapses per above. Rerank would need a large shortlist → more
graph hops → net loss. Also requires a second codebook (+storage).

### 2. Nibble-split additive decomposition
Decompose `LUT[code] = L_lo[code & 15] + L_hi[code >> 4]` so two 16-entry
tables replace the 256-entry one. **Tested empirically: 46% pairwise argmin
disagreement, max abs error 58940/65535.** Fails because the PQ distance
LUT does NOT factor additively — the bilinear term `−2⟨q_s, c⟩` prevents
any such decomposition. (An arbitrary 256-entry table admits no exact
additive decomposition into two 16-entry tables; this is a math fact,
not an implementation issue.)

### 3. `vqtbl4q_u8` as a single 512-byte table
The "4" in `vqtbl4q` is 4 registers × 16 bytes = **64 bytes** total, not
4×256. Verified empirically: indices ≥64 return 0. Cannot cover a 256-entry
uint16 (512-byte) or uint8 (256-byte) LUT row in one call.

### 4. UDOT / i8mm low-rank matmul
Would require the LUT to factor as `U·V` with V code-dependent but cheap.
The per-segment PQ LUT has rank ≈ m=96 across segments (each segment's
codebook is an independent k-means) — no useful low-rank structure exists.
ScaNN's matmul speedup comes from a *different* structure (4-bit code
packing with LUT16 shuffle), not a low-rank LUT. UDOT is not load-bearing
for this problem.

## The winning approach: 4-bank split-table (Quicker-ADC port)

From André et al., arXiv:1812.09162 §3.2 ("Quicker ADC"). Originally an
AVX512-VBMI `vpermi2b` (7-bit, 128-byte coverage) technique; ported here
to NEON `vqtbl4q_u8` (6-bit, 64-byte coverage) by using **four** tables.

### Math
- LUT row `L_s[0..255]` split into 4 banks: `T_b[c] = L_s[64·b + c]`,
  `b ∈ {0,1,2,3}`. Each `T_b` is 64 bytes = one `uint8x16x4_t` table-quad
  (4 NEON registers concatenated).
- Per code `c`: bank `b = c >> 6`, in-bank index `i = c & 63`,
  lookup `T_b[i]`. The other 3 banks return 0 for this code's lane.
- Per 16 codes (one `uint8x16_t` register): compute `b16 = c>>6`,
  `i16 = c&63`; for each bank `bk`, mask `(b16 == bk) ? i16 : 0xFF`,
  `vqtbl4q_u8(T_bk, masked_idx)`, OR all four results. Each code hits
  exactly one bank → the OR is exact.

### Correctness
**Exact over all 256 codes.** Verified exhaustively (0 failures across
256 codes). The only error in the full pipeline is the uint8 LUT
quantization (FAISS-documented ≤0.5pp recall loss; within our 0.005
budget).

### Storage
**Zero per-code increase.** Same 96 bytes/code (m=96 segments × 1 byte).

### Expected throughput
- Mac NEON microbench (pre-plan, m=96/K=256/16-wide): **158 vs 61 M codes/s
  = 2.6× over scalar gather.**
- V2 model: ~4× (V2's `vqtbl4q` V01-pipe throughput is higher than Apple's
  for the 4-reg form). To be confirmed in Phase 4 c4a microbench.

The win is NOT from gather latency (the existing `lut_distance_simd_findings.md`
shows gather is only ~3% of cycles on V2 — pipelines well). The win is
**eliminating the scalar index arithmetic** that is 86% of the current loop:
the `s*K + code[s]` address computation collapses to the implicit tbl index.

## SIMD primitives survey (AArch64, V2 + Apple M-series)

Confirmed via ARM SWOG 109898 (Neoverse V2), LLVM
`AArch64SchedNeoverseV2.td`, and Lemire et al. arXiv:2503.01662 Table 2:

| Instruction | Latency | Throughput | Coverage |
|---|---|---|---|
| `vqtbl1q_u8` | 2 | ~2/cyc | 16 bytes (FAISS uses this for 4-bit) |
| `vqtbl2q_u8` | 2 | ~2/cyc | 32 bytes |
| `vqtbl3q_u8` | 4 | 1/cyc | 48 bytes |
| **`vqtbl4q_u8`** | 4 | ~2/3/cyc (V01 pair) | **64 bytes — our workhorse** |
| `vqtbx{1,2,3,4}q_u8` | 2-6 | up to 4/cyc | OOB leaves dst unchanged (TBX vs TBL zeros) |
| `UDOT`/`SDOT` | 3 | ~2/cyc | 4×4 i8 dot → u32 (not useful here — see above) |
| SVE2 `svld1_gather_u32index_f32` | 6-7 | ~0.3-1.3/cyc (4 uOps) | the current path's primitive |

`vqtbl4q_u8` is baseline AArch64 NEON — no SVE, no feature flag. Works
identically on c4a (V2) and Apple M-series. **This is a portability win
over the current path**, which only gets its SVE2 gather speedup on c4a.

## Tolerance analysis (why uint8 LUT is safe)

End goal is *argmin over codes* (to drive graph traversal + rerank
shortlist), not absolute distance. The FP16 rerank pass re-evaluates
the shortlist exactly. So the PQ LUT only needs to (a) keep true top-k
inside the rerank pool and (b) order the pool well enough to avoid
extra graph hops.

FAISS's uint8 LUT path is the existence proof: quantizing each row's
`[min,max]` span to 256 levels costs ~0.5pp recall@1. Our config
(recall 0.9931 at rr=10) has 0.005 budget — comfortable.

FAISS `NormTableScaler` mechanism (ported in Phase 1.1):
- Per-query global `(A, B)` where `A = 255 / max_span` (one scale across
  all m segments — preserves cross-segment comparability).
- `B = Σ_s min(LUT[s][·])` (sum of per-segment mins).
- Each entry: `lut8[s][c] = round((LUT_f32[s][c] - min_s) * A)`.
- Clamp `A` so `m × 255 × A < 65535` (avoid uint16 accumulator overflow;
  m=96 × 255 = 24480, leaving ~2.7× headroom).
- Final distance: `dist ≈ (uint16_acc) / A + B`.

## What's adaptive vs static

- **Per-query:** LUT quantization scale (A, B) — free, in `preprocess_query`.
- **Per-build:** codebook structure is unchanged. (Sorted-codebook and
  additive-correction ideas were considered and rejected — no global
  centroid ordering makes the LUT monotonic across queries; per-code
  residual bytes are affordable but unnecessary given D1's exactness.)
- **Per-code:** unchanged. 96 bytes/code, no auxiliary structure needed.

## Open risks (monitor in Phase 1/4)

1. **`vqtbl4q` throughput on V2 (V01 pipe)** is the real ceiling. If
   derated throughput lands below 1.5× over gather on c4a, fall back to
   float-batch4 (still selectable via `--pq-eval float`).
2. **Register pressure.** 4 banks × 4 registers = 16 NEON regs for tables.
   Mitigation: segment-pair tiling (load 8 table-quads covering 2 segments,
   accumulate, evict) — keeps live table-regs at 8.
3. **uint8 overflow management.** The FAISS scaler math must be ported
   exactly to avoid uint16 accumulator overflow over m=96 segments.

## References

- Quicker-ADC: André et al., "Quicker ADC: Splitting the Pareto-Frontier
  of Vector Quantization and Search" (arXiv:1812.09162). The split-table
  technique, §3.2 Fig 1.
- FAISS source: `faiss/impl/pq4_fast_scan.cpp`, `simdlib_neon.h`
  (`lookup_2_lanes`), `quantize_lut.cpp` (`round_uint8_per_column` =
  the `NormTableScaler` approach).
- FAISS wiki: "Fast accumulation of PQ and AQ codes (FastScan)".
- Lemire et al., "Faster 8-bit Shuffle Filters on AArch64" (arXiv:2503.01662)
  — measured `tbl` throughputs on Apple + V2 cores.
- ARM SWOG 109898 (Neoverse V2 software optimization guide).
- Existing `docs/lut_distance_simd_findings.md` — independently confirms
  the gather-is-3%-index-math-is-86% finding that motivates this work.
