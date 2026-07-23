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

## OPQ: revisit (the one open algorithmic lever)

The prior OPQ test (`t13a_opq_findings.md`) used **FHT random rotation** — a
fixed random orthogonal transform. It hurt recall because arxiv-nomic
embeddings are already variance-balanced (normalized/spherized), so random
rotation adds noise, especially from the 256 zero-pad dims (768→1024) diluting
the signal.

**That was not OPQ.** The actual OPQ paper (Ge et al., "Optimized Product
Quantization," TPAMI 2013) uses **data-adaptive rotation learned via SVD /
alternating minimization** — R is chosen to balance variance across subspaces
*for the specific dataset*. `docs/quantization_improvements.md` already flags
this distinction (Approach 1, "Why we tried OPQ before and it didn't help") and
recommends revisiting at k=100.

### Why it matters now
- The PQ-only ceiling is 0.73 at k=100 (concentration of measure). This is what
  forces rerank=2 (the 12% rerank tax). If OPQ lifts the ceiling to 0.80-0.85
  (literature estimate), rerank could drop to ~1.5 or the recall target could
  be hit at lower L.
- OPQ is the one untested algorithmic lever that attacks the **root cause**
  (PQ distortion) rather than the symptoms (eval count, routing, parallelism).
- It helps BOTH merged and IVF equally (same quantizer).

### What to do differently this time
1. **SVD-based rotation**, not FHT. Learn R from the data covariance (Jacobi
   eigendecomposition of the d×d covariance, one-time at training).
2. **Test at k=100** (the regime where plain PQ is weak; OPQ's benefit is
   largest at low PQ-only recall).
3. **Watch the zero-pad issue.** FHT padded 768→1024 and the 256 zero dims
   diluted the signal. The SVD rotation should be on the native 768-dim
   covariance, no padding.
4. **Measure distortion**, not just recall — the `probe_pq_config` machinery
   already reports per-config distortion; OPQ should lower it vs plain PQ at
   the same (m, bits).

### Marked for revisit
`docs/quantization_improvements.md` §"Phase 1: Revisit OPQ at k=100" is the
entry point. This is a follow-up task, not a Phase 3 deliverable — but it's
the most promising untested lever for raising the recall ceiling (and thus
reducing the rerank tax) across both search paths.

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

### Tier 1 — promising, unexplored or under-explored

**OPQ (SVD rotation, not FHT).** The PQ-only ceiling is 0.73 at k=100, which
forces the rerank=2 tax (12.4% of query time). Real OPQ (data-adaptive SVD
rotation) is estimated to lift the ceiling to 0.80-0.85, reducing rerank
multiplier and raising recall for BOTH paths. The prior FHT test was random
rotation, not OPQ. **Highest-promise untested lever** — attacks the root cause
(PQ distortion). See `docs/quantization_improvements.md`.

**ScaNN-style anisotropic quantization.** Replaces isotropic MSE with a
weighted objective that minimizes error along the query direction. Estimated
10-25% recall improvement (larger than OPQ alone). More complex (modified
k-means objective) but compatible with the ADC infrastructure. The approach
that achieves SOTA on VIBE.

**FP16 rerank.** Rerank (12.4%) is memory-bound on FP32 base-vector loads.
Loading/storing FP16 base vectors for rerank would halve the memory traffic.
The rerank is the query engine's job (`rerank_ownership.md`), but the engine
could expose an FP16 base path. Risk: rerank precision (FP16 has ~3 decimal
digits; for rerank's fine-grained re-sorting this may cost recall). Needs A/B.

### Tier 2 — real but small

**Batch the entry-point multi-start scan.** The multi-start seeding
(`beam_search_into:349-383`) evaluates entry points via scalar `lut_distance`,
contributing to the 7.1% scalar-PQ chunk. With A1's 8 entry points, batching
these into one `lut_distance_batch4` call (2 batches of 4) would cut ~half of
that 7%. Small, easy, no recall risk. Pure code-cleanup win.

**Batch4 remainder handling.** The PQ batch4 loop's scalar tail (lines 686-701)
fires when `pq_n % 4 != 0`. With R=64 and partial visits, remainders are
common. A masked batch4 (process the final 1-3 neighbors via a masked SVE2
op instead of scalar fallback) would recover the tail. Small.

**`preprocess_query` caching across similar queries.** The LUT rebuild is 4.5%
(O(m·K) per query). Already amortized in IVF (shared across n_probe shards).
For workloads with repeated/near-duplicate queries, a small LUT cache could
help — but most ANN workloads have unique queries, so limited applicability.

### Tier 3 — structural, not fixable without redesign

**Heap operations (18.4%).** The frontier/W heaps scale with L, which drives
recall. A bucket-queue (O(1) push by quantized distance) could help, but the
distance distribution is wide (needs log-scale or adaptive bucketing) and the
gain is bounded — DynamicWidth + early-exit already suppress most wasteful
pushes. Invasive for uncertain gain.

**neighbors_buf memcpy (part of libc 12.9%).** Copying the neighbor list out
of each pinned node enables batch processing. Eliminating it requires
restructuring pin semantics (process in-place). Risky, load-bearing.

**libc memcpy/memset residual.** Heap sifting, vector resizing, pin buffers.
Death by a thousand cuts; no single concentration.

### Tier 4 — tested, dead ends (documented for reference)

- **A2/A3** (entry-point selection/count): neutral-to-negative.
- **B1** (two-level parallelism): throughput-negative (memory-bandwidth wall).
- **C1** (frontier-saturation skip): redundant with DynamicWidth + early-exit.
- **FP32 routing**: identical to FP16 (PQ-decoded centroids).
- **FHT-OPQ**: hurt recall (random rotation ≠ real OPQ).

### The honest meta-conclusion

The diffuse profile (no symbol >20%) means **the engine is well-optimized** —
there is no concentrated inefficiency left to exploit. The Tier 1 levers (OPQ,
anisotropic, FP16 rerank) are all **quantization-side** improvements that raise
recall at the source, reducing the downstream rerank/eval pressure. They help
both merged and IVF equally. The Tier 2 levers are small code-cleanups.

The IVF/merged QPS gap is not closeable by engine micro-optimization — it's the
n_probe structural multiplier. The path to higher QPS at matched recall runs
through the quantizer (OPQ/anisotropic → less rerank needed → lower fetch_k →
less of everything), not through beam_search or routing.

