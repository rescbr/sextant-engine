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

**Correction (2026-09-04, post-review): the arithmetic first.** A group
of G dims with a K=16 codebook costs 4 bits per *group* — i.e.
**4/G bits per dim**, not 4. G=2/K=16 is 2 bits/dim; keeping 4 bits/dim
at G=2 requires K=256 per pair, whose 8-bit LUT lookup costs
~2×TBL+blend per group on NEON and re-loses to dual-SDOT on the same
axis argument as above. So grouped LUT scanning is not "same recall,
fewer passes" — it is a **bits-vs-scan-time frontier**: G-fold cheaper
block scans (m4 = dim/G) bought with G-fold fewer bits per dim, with
joint G-dim codebooks recovering part of the lost precision (the
Jégou sub_dim tradeoff) and rerank recovering the rest. This is the
same tradeoff that motivates production RaBitQ-1-bit-shortlist +
rerank pipelines (Infino's menu; rejected for our dynamic tree, but
the frontier point is testable cheaply).

The cheapest test needs NO new code: `local_pq` with `m4 = dim/2`
(sub_dim=2, K=16, 2 bits/dim joint pairs) exercises the whole path —
per-leaf pair codebooks, FastScan blocks at m4=768, existing kernels.
Compare recall/latency against `local_scalar` (4 bits/dim rulers) and
`local_pq` at other m4; if the 2 b/dim point lands near the 4 b/dim
recall at materially lower scan cost, the frontier is worth climbing
with G>2 and anisotropic weighting. Literature anchoring the direction:

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

## Spike result (2026-09-04, same evening): FALSIFIED

Ran the frontier with zero new code — `local_pq` trees at sub_dim=2 and
sub_dim=4 on dbpedia 933K×1536, vs the `local_scalar` 4 b/dim reference
(50 queries, c4a, back-to-back):

| config            | bits/dim | bytes/vec | exh. ms | exh. raw R@10 | exh. rerank R@10 |
|-------------------|----------|-----------|---------|---------------|------------------|
| local_scalar      | 4        | 768       | 233     | 0.934         | 1.000            |
| local_pq sub_dim=2| 2        | 384       | 42      | **0.140**     | 0.174            |
| local_pq sub_dim=4| 1        | 192       | 24      | **0.124**     | 0.166            |

The frontier is a **cliff, not a slope**: 2 b/dim recall collapses to
0.14 (vs 0.93 at 4 b/dim), and joint 2-dim pair codebooks recover
essentially nothing (0.140 vs 0.124 at 1 b/dim — the Jégou sub_dim
gain is absent at G=2 on this data). Rerank cannot rescue it: the scan
ceiling (0.14) caps the shortlist, so even exact rerank tops out at
0.17. Root cause is the same phenomenon as the earlier "locality
falsified" finding: at d=1536 embeddings the per-dim correlation
structure is too weak for joint small-group codebooks to exploit — the
recall elbow is driven almost entirely by bits, and 4 b/dim is below
the knee for scan-only use.

Conclusion: scalar stays at 4 b/dim flat + dual-SDOT (mode 2), with
the regime-(b) dial (decode-only / exact_rerank_base) already covering
the low-bytes point properly. The G=2/G=4 LUT points are dominated on
both axes for this workload class.

(The spike also flushed out and fixed a real bug: LocalPqCoder::rerank
used a fixed 64-byte code buffer — stack overflow for any m4 > 128,
i.e. the default m4=dim/4 on dim>512.)
