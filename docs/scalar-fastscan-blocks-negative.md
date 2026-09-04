# Scalar FastScan blocks — negative result (2026-09-04)

## What was tried

Commit 2 of the LeafCoder refactor: switch the scalar families
(`scalar_lloydmax/uniform/shape`, `local_scalar`) from flat packed-nibble
code rows to the pq-style FastScan 32-vector interleaved block layout
(`m4 = dim`), and scan with the existing `pq4_block32` LUT kernel —
plus a NEON LUT builder with a hi/lo dual split to preserve precision.

## Result (c4a / Graviton Neoverse-V2, clang-19, dbpedia 933K×1536,
## local_scalar, 50 queries, back-to-back)

| config                        | routed  | exhaustive | recall@10 (raw) |
|-------------------------------|---------|------------|-----------------|
| flat + dual-SDOT (mode 2)     | 16.4 ms | 66.4 ms    | 0.842           |
| blocks + LUT-dual             | 35.3 ms | 148 ms     | 0.838           |

**2.1× slower.** Reverted; the work lives on branch
`fastscan-scalar-blocks` (block layout, dual-LUT kernels, NEON LUT
builder — all validated correct, just slower).

## Why it loses — the axis mismatch

SDOT (`vdotq_s32`) reduces over *consecutive operand bytes*. For a dot
product that means consecutive **dims of one vector** — i.e. the flat row
layout, where 8 code bytes unpack to 16 dims of a single vector and two
`vdotq` (hi + lo/127) produce a 15-mantissa-bit partial dot. The block
layout puts 32 vectors of *one dim* in each 16-byte group: the wrong axis
for SDOT, forcing the TBL/widen pattern instead (2 TBL + widening adds
per dim per 32-vector block, plus saturation management). On V2 the SDOT
throughput is exceptional; the TBL path cannot compete at `m4 = dim`.

The original FastScan motivations did not apply here anyway:
- the rerank nibble-unpack lever (6.8 pp) was PQ-specific — scalar
  rerank decodes flat rows directly;
- `LeafFilterLayout`'s block assumption was already fixed by the coder
  `geometry()` in commit 1.

## The opportunity: grouped-dim scalar quantization

The obvious next idea is *not* to give up on LUTs for scalar codes, but
to change what a "subquantizer" covers. Per-dim scalar quantization is
exactly PQ with `sub_dim = 1`. If we group **G=2 dims per codebook**
(learned per leaf, K=16 centroids over the 2-dim residual pairs), we get:

- same code size (4 bits/dim — 8 bits per pair);
- LUT count halves (`m4 = dim/2` → 2× fewer TBL iterations per block);
- **better recall per bit**: a joint 2-dim codebook captures the
  correlation between paired dims that two independent uniform rulers
  cannot — this is the classic PQ-vs-scalar-quantization tradeoff at the
  smallest possible group size;
- and mechanically it is just `local_pq` with `sub_dim=2` — the FastScan
  path for that already exists and is measured fast.

Pairing adjacent dims only pays off if the pairs are roughly
independent of each other; an OPQ-style rotation (Ge et al., TPAMI 2014)
could choose the pairing. Literature anchoring this direction:

- André, Kermarrec, Le Scouarnec — *Cache locality is not enough:
  high-performance NN search with product quantization fast scan*
  (VLDB 2015). The original in-register 16-entry u8 LUT + shuffle
  (pshufb / TBL) formulation; 4–6× over PQ scan.
- André, Kermarrec, Le Scouarnec — *Quicker ADC* (TPAMI 2019).
  **Irregular Product Quantization**: groups subquantizers of different
  bit widths (e.g. 12×{6,6,4}) to align codes to word boundaries —
  the direct precedent for "grouping" as a design axis, and for
  mapping input dims to subquantizers non-uniformly.
- Matsui et al. — *4-bit PQ on ARM* (arXiv:2203.02505). The
  two-128-bit-register TBL trick our NEON kernels use; 10× over naive
  PQ on ARM.
- Jégou, Douze, Schmid — *Product Quantization for NN Search*
  (TPAMI 2011): the sub_dim tradeoff (quantization error falls as
  sub_dim grows at fixed bits, for correlated sub-vectors).

Gating note (measured this session): per-dim LUT machinery loses to
dual-SDOT on V2, so the grouped variant must be **spiked before any
layout commitment** — win condition is roughly `≥2× fewer TBL passes
+ joint-pair recall gain` beating `2 vdotq per 16 dims` at equal recall.
The per-leaf training cost of `dim/2` tiny codebooks also needs a
budget (single pass of 2-dim k-means over leaf vectors; likely fine at
leaf sizes ~4K but must be measured on the flush path).
