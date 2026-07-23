# IVF Phase 3 — Closing Design & Findings

**Status:** Consolidation, 2026-07-23. Closes the Phase 3 optimization arc
(A: entry points, B: parallelism, C: traversal intelligence) plus the routing
study, K sweep, and FHM win.

## What IVF is for

IVF is **not** a single-node QPS optimization. The merged graph beats it on QPS
at matched recall (~1754 vs ~1200 QPS at recall 0.91), and nothing in this
phase closed that gap — because the gap is structural (the n_probe multiplier
on beam_search cost), not a fixable inefficiency. IVF's value is three things
the merged path structurally cannot provide:

### 1. The 1B-scale lever (the primary value)

The merged path is RAM-bound: the full graph + codes + ball must be paged and
cached as one structure. At 1B vectors the merged index is ~500 GB (see
`ivf_probe_design.md` scale table) — far beyond RAM, and the paged search path
hits I/O limits that no amount of caching fixes at that scale.

IVF's K shards are each **independently sized** — each shard is a complete,
self-contained `Index` that fits the per-shard RAM/SSD budget. The partition
is the unit of scale: you grow N by adding shards (K↑) without each shard
becoming unsearchable. At SIFT1B (K=512), each shard holds ~2M vectors —
comfortably searchable with the same paged path that struggles at 1B monolithic.

The K sweep we ran (`docs/ivf_k_sweep.md`) is literally the scale-out curve:
K=8 (coarse, big shards) → K=64 (fine, small shards). At each K the per-shard
search is the same beam_search; only the shard count and routing change. This
is how DiskANN-style systems scale, and it's why IVF exists in the architecture.

### 2. The recall ceiling

Merged is PQ-navigation-capped at ~0.9084 on arxiv-nomic (concentration of
measure + PQ's per-subspace independence). IVF exceeds this at np=8 across
every K measured: 0.9558 (K=8) to 0.9801 (K=32). Searching N shards gives N×
candidate pool diversity that a single frontier can't match. For applications
where recall > 0.91 matters, IVF is the only path.

### 3. The distributed path

K = node count. Each shard lives on a different host. The routing step
becomes request dispatch: a query fans out to its n_probe nearest nodes.
Multi-probe (`SearchConfig::multiprobe_ratio`) is the **fan-out policy** —
unambiguous queries (deep in one Voronoi cell) dispatch to 1-2 nodes; boundary
queries dispatch to more. This is the natural request-routing pattern for
distributed ANN, and it's why multi-probe shipped despite being QPS-neutral
on single-node throughput.

## What this phase accomplished

| change | commit | effect |
|--------|--------|--------|
| FHM `l2sq_f16` (`vfmlalq`) | `2a4538c` | +33% QPS local, +4% c4a merged. Rising tide for both paths. |
| A1 sub-clustered entry points | `282142b` | 8 k-means medoids per shard (replaces stride sampling). Modest recall help. |
| Multi-probe | `f9a5f8b` | Recall lever + distributed dispatch primitive. ~10% QPS per +1pp recall. |
| Route quality tool | `eac0d78`, `1ee1383` | Measured routing recall (closure-aware) + multi-probe tradeoffs. |
| K sweep doc | — | Operating-point curve; K=16-32 practical range. |

## What was tested and rejected (with reasons)

- **A2/A3** (entry-point selection / scaling): neutral-to-negative. The default
  multi-start over A1's medoids is already good; entry-point selection was
  never the bottleneck.
- **B1** (two-level parallelism): validated negative for throughput. The
  benchmark is throughput-bound (saturates cores with query-fan-out); B1 trades
  query-fan-out for shard-fan-out on a fixed budget → net loss. May help
  latency-bound serving (unmeasured).
- **C1** (frontier-saturation skip): dead end. Redundant with DynamicWidth +
  early-exit — the wasteful tail evals it targets are already suppressed. The
  27% PQ-eval chunk is productive work.
- **FP32 routing**: identical to FP16. Centroids are PQ-decoded; precision
  isn't the limiter.
- **FHT-OPQ** (prior phase): hurt recall. But this was random rotation, NOT
  real OPQ — see below.

## Why the IVF/merged QPS gap is structural (not fixable)

The c4a profile (merged, 8t) is diffuse:
- PQ neighbor eval: 27% (productive — DynamicWidth + early-exit already cut waste)
- beam_search heap/loop: 18% (scales with L, which drives recall)
- FP16 ball: 12% (post-FHM; was higher)
- rerank: 12% (query engine's job, fixed by `k×rerank`)
- libc/merge/misc: 31%

No single hot spot. IVF does `n_probe ×` this work per query; merged does `1×`.
The ~1.5× QPS gap is the n_probe multiplier on a diffuse, already-optimized
cost base. The remaining QPS levers are operating-point knobs (n_probe,
patience, K, multi-probe ratio) trading recall for speed — not algorithmic wins.

## OPQ: already tested at scale — 0pp (do NOT revisit)

OPQ (PCA rotation — the real data-adaptive SVD rotation, not the earlier FHT
random rotation) **was implemented and tested at full 1.34M scale** (commits
`7a39697`, `f7d4977`; documented in `docs/quantization_findings.md`). Results:
- arxiv100k: PQ-only recall@100 0.8955 → 0.9058 (+1.03pp)
- **arxiv-nomic 1.34M: 0.7300 → 0.7300 (0pp)**

The +1pp on arxiv100k is a small-scale artifact that vanishes at production
scale. The PQ-only ceiling (0.73) is NOT a codebook-quality problem — it's a
structural concentration-of-measure problem: at large N the k-th NN distance
shrinks (tighter neighborhoods), so PQ's fixed absolute error becomes
relatively larger regardless of codebook quality. A 7.9% MSE improvement
doesn't change the fundamental geometry.

**All three quantization-improvement approaches were tested and failed at
production scale** (`docs/quantization_findings.md`):
1. Per-vector direction proxy (ScaNN λ weighting): recall *degraded*.
2. Per-subspace covariance weighting (scale-transform): −0.56pp.
3. OPQ (PCA rotation): 0pp at scale.

The rerank tax (12.4%) is therefore **irreducible via quantizer changes** on
this dataset — rerank is how both paths get from the 0.73 PQ ceiling to the
0.91 target, and no codebook improvement raises that ceiling at scale.

**Do not revisit OPQ or anisotropic.** The ceiling is structural. The only
known way ScaNN reaches ~0.90 PQ-only is via its full anisotropic *training*
pipeline (not just rotation), which is a fundamentally different and much
heavier quantizer — out of scope for this architecture.

## Recommended operating points (from the K sweep + multi-probe)

| use case | K | n_probe | multi-probe | patience | notes |
|----------|---|---------|-------------|----------|-------|
| Single-node, recall ~0.91 | 32 | 4 | 1.0 (off) | 0 | closest to merged's operating point |
| Single-node, high recall | 32 | 8 | 1.0 (off) | 0 | recall 0.93+, beats merged's ceiling |
| Distributed / latency-bound | 16-32 | 4 | 1.05 | tuned | multi-probe = fan-out policy |
| 1B-scale single-node | 256-512 | 8-16 | 1.0 | tuned | shard = scale unit; K from RAM budget |

## Stop condition

Phase 3 is complete. The IVF path is production-viable for its intended roles
(1B scale, high recall, distributed). The single-node QPS gap to merged is
structural and documented. The next algorithmic lever is OPQ (separate task),
not further IVF micro-optimization.

---

## Appendix: complete improvement-lever inventory

Every lever identified across the session, ranked by estimated promise, with
status. This is the exhaustive answer to "what else can improve things."

### Tier 1 — the remaining structural levers (memory-access-bound)

These are the levers the profile actually supports. The constraint is memory
access pattern / RAM bandwidth (the sublinear thread scaling: 4.3× at 8t).
Past wins came from trading RAM access for compute (FHM fused convert+multiply
to cut loads; Direct-SDC eliminated the LUT gather memcpy). The same principle
applies here:

**Heap data structure (18.4% of beam_search) — TESTED, NEGATIVE.** Replaced
the frontier binary heap with a bucket queue (O(1) push/pop). The bucket
queue DID reduce heap self-time (18.4% → 16.9% on c4a), but the approximate
pop ordering (within a bucket, any element) made the search explore nodes in
slightly worse distance order → more hops → more PQ evals (lut_distance_batch4
rose 19.7% → 27.6%). Net: −6.7% QPS. Same lesson as the frontier-saturation-
skip: the frontier's pop precision matters; approximate ordering degrades
convergence and the extra evals cost more than the heap savings. Header kept
at `src/algo/bucket_queue.hpp`; beam_search wiring reverted.

**Neighbor-list copy (part of libc 12.9%).** Each node expansion memcpys the
neighbor list (R×4 bytes = 256B at R=64) out of the pinned node into a stack
buffer to enable batch4. At thousands of expansions/query this is real traffic.
Eliminating it means **processing neighbors in-place from the pinned node** —
but batch4 needs contiguous code pointers, and `pin_code` returns a per-node
pointer (scattered). The trade: gather the 4 code pointers from the in-place
neighbor list (4 loads) and batch4 them directly, skipping the memcpy. This is
the same gather pattern `lut_distance_batch4` already uses internally.

**PQ batch4 entry-point multi-start (7.1% scalar chunk).** The entry-point
seeding evaluates 8 (A1) entry points via scalar `lut_distance` one-at-a-time.
Batching into `lut_distance_batch4` (2 calls of 4) is a pure code cleanup —
same compute, fewer call overheads, better instruction-level parallelism. No
recall risk.

**Batch4 remainder (scalar tail).** When `pq_n % 4 != 0`, the last 1-3
neighbors go through scalar `lut_distance`. A masked SVE2 op processes the
remainder at batch4 throughput. Small but free.

### Tier 2 — tested, dead ends (do NOT revisit)

- **OPQ (PCA rotation)**: implemented (`7a39697`), 0pp at 1.34M scale. The PQ
  ceiling is structural (concentration of measure), not codebook-quality.
- **Anisotropic PQ** (both per-vector and per-subspace): tested (`820d3cd`,
  `26b54da`), recall degraded or flat.
- **A2/A3** (entry-point selection/count): neutral-to-negative.
- **B1** (two-level parallelism): throughput-negative (memory-bandwidth wall).
- **C1** (frontier-saturation skip): redundant with DynamicWidth + early-exit.
- **FP32 routing**: identical to FP16 (PQ-decoded centroids).
- **FP16 rerank**: out of scope (query engine's job per `rerank_ownership.md`)
  AND increases storage (FP16 base vectors). Rejected on both grounds.
- **FHT-OPQ**: random rotation, hurt recall (distinct from PCA-OPQ above).

### The honest meta-conclusion

The diffuse profile (no symbol >20%) means **the engine is well-optimized at
the algorithm level**. Every quantization-side lever (OPQ, anisotropic) has
been tested and fails at production scale due to concentration of measure —
the 0.73 PQ ceiling is structural, so the rerank tax is irreducible via the
quantizer.

The remaining levers are **memory-access-pattern optimizations** (heap → bucket
queue, eliminate neighbor memcpy, batch the scalar tails) — the same class of
RAM→compute trade that produced the FHM win. These help both merged and IVF
equally. They won't close the IVF/merged n_probe multiplier gap (that's
structural), but they'd lift the whole performance baseline.

