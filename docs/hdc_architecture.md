# Sextant Distance Computation Architecture

## Standard PQ-ADC + FP32 rerank

Sextant uses the standard PQ-ANN distance architecture:
1. **PQ-ADC** for graph navigation (asymmetric: query unquantized, database PQ-quantized).
2. **FP32 rerank** for final top-k selection (standard practice in DiskANN/FAISS/ScaNN).

The former three-tier "HDC" design (FP16 ball → PQ ADC → FP32 rerank) was
**retired** on 2026-07-24. The FP16 ball tier — which was presented as Sextant's
novel contribution — turned out to actively hurt recall (premature convergence)
and QPS (FP16 compute more expensive than PQ-ADC LUT sum). See
`results/p2.3-noball/` and commit `2dca950`.

**This is not novel.** ADC + rerank is the standard PQ-ANN architecture used
by DiskANN, FAISS, ScaNN, and others. The "HDC" framing was based on the FP16
ball tier providing a precision advantage; with that tier removed, there's no
hybrid — it's plain ADC navigation + rerank, same as everyone else.

What IS ours in this area: the **f32-accumulated SIMD kernels** (`simd::`
namespace) and the **pq_matvec_f32** LUT construction kernel, which are ~3×
faster than NumKong's f64-accumulating equivalents at our shapes. That's an
implementation optimization, not an architectural contribution.

## Build-time FP16 prune

During graph construction, `robust_prune_into` uses FP16 L2sq distances
(`simd::l2sq_f16`) for the occlusion check instead of PQ code-to-code
distances. This produces a graph where edges reflect true vector proximity.

The FP16 prune has no recall benefit (measured ±0.6pp vs PQ-only prune),
but makes the build **36% faster** (better-sorted candidates → fewer
occlusion checks). Kept for build speed.

The FP16 vectors (`raw_vecs_buffer`) are loaded once at the start of
construction and freed after the graph is built. They are NOT stored in
any index sidecar.

## Search-time: PQ-ADC everywhere

```
Query arrives as FP32
       │
       ├──► preprocess_query(query_fp32)  →  PQ LUT (per-query, m × K floats)
       │                                   Built via simd::pq_matvec_f32 (f32-accumulated
       │                                   matvec, ~3× faster than per-entry loop).
       ▼
   beam_search(query_lut)
       │
       │   For EACH node visited during beam search:
       │
       └──► lut_distance(node_pq_code, query_lut)
            Per-query LUT lookup — m-segment gather, summed via
            lut_distance_batch4 (4-wide SIMD gather). Used for ALL nodes.
            No FP16 tier, no precision switchback.
       │
       ▼
   beam_search returns fetch_k candidates (ranked by PQ-ADC distance)
       │
       ▼
   Rerank (caller responsibility — benchmark tool or application)
       │
       └──► simd::dist_f32(metric, query_fp32, base_fp32, dim)
            Exact f32 distance on the original base vectors, dispatched by
            the configured metric (L2sq or IP). Corrects PQ-ADC ranking
            errors among the fetch_k candidates. Returns the final top-k.
```

### Why PQ-ADC only (no FP16 tier)?

The FP16 ball tier was measured to hurt both recall and QPS:

| Config | recall (with ball) | QPS (with ball) | recall (no ball) | QPS (no ball) |
|--------|-------------------|-----------------|-----------------|---------------|
| L2sq rr=1 | 0.29 | 2498 | **0.47** | **5400** |
| L2sq rr=10 | 0.72 | 805 | **0.95** | **2192** |

The FP16 tier's true distances caused **premature search convergence** — the
search found a locally-optimal neighborhood and stopped exploring. PQ-ADC-only
search explores more broadly → broader candidate pool → rerank finds more true
neighbors. PQ-ADC is also cheaper per eval (LUT sum vs FP16 dim-wide compute).

At c4a 1.34M: recall within 2pp (neutral), QPS slightly higher without ball.

### Metric selection

The `--metric {l2sq,ip}` flag controls:
1. **PQ-ADC LUT construction** (`preprocess_query`): L2sq or IP entries.
2. **Rerank** (`benchmark.cpp`): `simd::dist_f32` dispatches by metric.

For L2-normalized data, both metrics are rank-equivalent on TRUE distances.
The difference is in the PQ-ADC estimator: L2sq-ADC includes the `||x̃||²`
per-code term; IP-ADC doesn't. At m=96/bits=8 on arxiv-nomic 1.34M:
- L2sq-ADC ceiling: 0.99 recall (rr=10).
- IP-ADC ceiling: 0.84 recall (rr=10), at 2.2× the QPS.

IP is Pareto-competitive at recall targets ≤ 0.84.

## SIMD kernels

All distance computation uses hand-written f32-accumulated kernels in
`src/simd_kernels.hpp` (namespace `sextant::simd`). These replaced NumKong's
f64-accumulating kernels (which accumulate in `nk_f64_t` — 2-wide on NEON,
half the f32 throughput). See `docs/optimization_levers_and_attribution.md`
for the full attribution.
