# Sextant Distance Computation Architecture

## HDC: Hybrid Distance Calculation

Sextant uses a **Hybrid Distance Calculation (HDC)** approach that combines
Product Quantization (PQ) for cheap navigation with FP16 vector compute for
precise distance evaluation. This replaces the traditional PQ-literature
distinction of SDC (Symmetric Distance Computation) and ADC (Asymmetric
Distance Computation) — Sextant is neither purely symmetric nor purely
asymmetric; it is a hybrid that uses the right precision for each phase.

The HDC principle: **compute beats gather**. PQ ADC distances require
scattered LUT lookups (data-dependent, pipeline-stalling). FP16 vector
distances are sequential loads (prefetcher-friendly). When FP16 vectors
are available, sequential FP16 compute is both faster and more accurate
than PQ gather. Sextant uses FP16 wherever the vectors are in RAM, and
PQ wherever only codes are available.

## Build-time HDC

During graph construction, each node insertion does two distance computations:

1. **Beam search** (navigation): PQ code-to-code distances via
   `code_distance` or per-anchor `lut_distance`. This finds candidate
   neighbors cheaply — PQ codes are 96 bytes/vector, and the cross-distance
   table enables fast approximate distances without touching the raw vectors.

2. **Robust prune** (edge selection): FP16 L2sq distances via `l2sq_f16`.
   This precisely determines which edges to keep — FP16 vectors are loaded
   sequentially (prefetcher-friendly), and the FP16 precision (≈3 decimal
   digits) is sufficient for the occlusion check.

The FP16 vectors (`raw_vecs_buffer_`) are loaded once at the start of
construction and freed after the graph is built. They exist only during
the build pass — they are NOT stored in the index sidecars (except for
the MemGraph ball in `.vecs`, see below).

## Search-time HDC: three-tier distance

At search time, the query arrives as FP32. The engine converts it to both
a PQ LUT and an FP16 copy, then uses three precision tiers during beam search:

```
Query arrives as FP32
       │
       ├──► preprocess_query(query_fp32)  →  PQ LUT (per-query, m × 2^bits floats)
       │
       └──► convert query_fp32 → query_fp16  →  FP16 copy of the query
       │
       ▼
   beam_search(pq_lut, query_fp16)
       │
       │   For EACH node visited during beam search:
       │
       ├─ Tier 1: Is the node in the MemGraph ball AND .vecs loaded?
       │   │
       │   ├─ YES: l2sq_f16(query_fp16, node_fp16, dim)
       │   │        Sequential FP16 compute — prefetcher-friendly, no LUT
       │   │        gather, no cache pin/unpin. Used for ~147K entry-point
       │   │        neighborhood nodes (the approach phase).
       │   │
       │   └─ NO:  Continue to Tier 2.
       │
       ├─ Tier 2: PQ ADC distance via lut_distance(node_pq_code, query_lut)
       │           Per-query LUT lookup — m-segment gather from the ADC table.
       │           Used for all non-ball nodes (the converge phase, ~1.2M nodes
       │           at production scale). This is the bulk of the distance
       │           computations.
       │
       ▼
   beam_search returns fetch_k candidates (ranked by FP16 or PQ distance)
       │
       ▼
   Tier 3: Rerank (caller responsibility — benchmark tool or application)
       │
       └─  l2sq_distance(query_fp32, base_fp32) — exact FP32 L2sq
            on the original base vectors. Corrects PQ/FP16 ranking errors
            among the fetch_k candidates. Returns the final top-k.

Precision hierarchy: FP16 (Tier 1) > PQ (Tier 2) > FP32 (Tier 3, exact).
```

### Why three tiers?

Each tier trades precision for cost:

- **FP16 (Tier 1)**: most precise navigation, highest per-node cost (1.5KB
  sequential load per vector). Used only for the MemGraph ball — the
  entry-point neighborhood traversed by every query during the approach
  phase. Ball size is bounded by `entry_points × R^hops` (~147K at
  R=32, 3 hops, 16 entry points), independent of dataset size.

- **PQ ADC (Tier 2)**: less precise navigation, lowest per-node cost (96
  bytes per code + LUT gather). Used for the majority of nodes beyond the
  ball. The LUT is built once per query (O(m × 2^bits)) and reused for
  all nodes. This is where most distance computations happen.

- **FP32 rerank (Tier 3)**: exact distance, highest cost (3KB per vector
  from the original base data). Applied only to the final fetch_k
  candidates (k × rerank, e.g. 100 × 10 = 1000). This is the rerank
  "tax" — it corrects PQ's ranking errors but requires reading the full-
  precision base vectors.

### The rerank tax

The rerank step (Tier 3) is the dominant throughput bottleneck at k=100.
At full scale (1.34M arxiv-nomic), PQ-only recall@100 = 0.73 (Tier 2
ceiling). To achieve recall@100 ≥ 0.99, the search oversamples
fetch_k = k × rerank = 1000 candidates and reranks them with exact FP32
distances. This requires ~3MB of base-vector reads per query.

Raising the PQ-only ceiling (via anisotropic codebook training — AHDC)
reduces the rerank multiplier needed, directly improving throughput.

## AHDC: Anisotropic HDC (planned)

The anisotropic variant modifies PQ codebook training to weight
reconstruction error by its impact on nearest-neighbor ranking. This
raises the PQ-only recall ceiling (Tier 2), reducing the rerank tax
(Tier 3). See `docs/quantization_improvements.md` for the full analysis
of OPQ and ScaNN-style anisotropic quantization approaches.
