# Build/Search Quantizer Coupling — The Decoupling Lever

**Status:** Design insight, 2026-07-23. Surfaced by the 4-bit PQ experiment.

## The insight

The 4-bit experiment showed 4-bit PQ collapses recall not just at search time
but also degrades the **graph topology**. The reason: the build's beam_search
(which finds the candidate pool for pruning) uses the same PQ quantizer as
search. So 4-bit PQ → worse build-time navigation → worse candidate pool →
worse edges (even though robust_prune itself uses FP16).

This reveals a design coupling we've treated as fixed but isn't: **build and
search share one quantizer**, so quantizer quality affects graph topology AND
search navigation identically.

## Current architecture (coupled)

```
Build:
  beam_search (find candidates) ──── PQ LUT distance (anchor_lut) ──┐
  robust_prune (select edges)   ──── FP16 L2sq (build_vec_ptr)    ──┤
                                                                    │
Search:                                                             ├─ same quantizer
  beam_search (navigate graph) ──── PQ LUT distance (query_lut)   ──┤
  (rerank happens in the query engine, FP32)                       │
                                                                    │
Stored codes (.codes) ──────────── encoded with this quantizer ────┘
```

The build beam_search and search beam_search use the SAME PQ LUT mechanism.
robust_prune is already decoupled (FP16). The stored codes are shared.

## The decoupling opportunity

**Build the graph with high-precision distance; search with low-precision PQ.**
The graph topology would be high-quality (good edges, well-connected), and
only search-time navigation would use the cheap codes. This separates "graph
quality" from "search speed."

### Option A: FP16 beam_search during build (simplest)
The build already loads FP16 raw vectors (`raw_vecs_buffer`) for robust_prune.
What if the build's beam_search ALSO used `l2sq_f16` instead of the PQ anchor
LUT? Then graph construction would be fully FP16-quality — no PQ coupling at
all for topology. Search still uses PQ (8-bit) for speed.

**Cost:** build beam_search would call `l2sq_f16` (dim=768, FHM-accelerated)
instead of `lut_distance` (m=96 lookups). ~8× more compute per eval during
build. But build is one-time, and FHM makes l2sq_f16 fast (~192 NEON FMLA for
dim=768). Estimated build-time increase: ~2-3× (beam_search is the dominant
build cost). At arxiv-nomic's ~90s build, that's ~200-270s — acceptable for a
one-time cost if it meaningfully improves graph quality.

**Benefit:** graph edges selected from FP16-accurate candidate pools, not
PQ-navigated ones. The graph would be as good as if PQ didn't exist. Search
then uses 8-bit PQ (current behavior) — no search-time cost change. This might
improve recall at fixed R (better edges = fewer dead-ends = more reachable
NNs), OR allow lower R at fixed recall (sparser but better-connected graph).

**Risk:** may not help if the current 8-bit PQ build navigation is already good
enough that FP16 doesn't change the candidate pools meaningfully. Needs
measurement. The 4-bit experiment suggests build navigation quality matters
(4-bit build → worse graph), but whether 8-bit→FP16 is a meaningful step up is
unknown.

### Option B: Dual-quantizer (build-only high-precision quantizer)
Build with a temporary high-precision quantizer (e.g. m=192/bits=8, or even
FP16 codes), store codes from the search quantizer (m=96/bits=8). Two codebooks;
the build keeps its own codes in RAM temporarily; `.codes` stores the search
codes.

**Cost:** higher build RAM (two code sets during construction), build-time
encoding overhead. More complex serialization (build-only codebook not stored).

**Benefit:** could use 4-bit PQ at search (fast) while building with 8-bit
(graph quality preserved). This is the path that would make 4-bit search viable
— the thing the 4-bit experiment failed at.

**Risk:** significant architectural change (two quantizers, dual encoding,
build pipeline restructure). The payoff depends on whether 4-bit search on a
good graph holds recall — unknown.

## Which to pursue first

**Option A (FP16 build beam_search) is the cheaper experiment.** It's a
build-time-only change (no format break, no search-path change, no dual
quantizer). If it improves graph quality measurably (better recall at fixed R,
or lower R at fixed recall), that's a win for BOTH merged and IVF at 8-bit. If
it doesn't help, we've learned the 8-bit PQ build navigation is already
sufficient — also valuable.

Option B is the heavier architectural bet worth pursuing only if Option A shows
build quality matters AND we have a reason to want 4-bit search (e.g. a
memory-bandwidth-constrained deployment where the 2× code-size reduction is
worth the complexity).

## Connection to the 4-bit result
The 4-bit experiment failed because BOTH build and search used 4-bit. If the
build had used 8-bit (or FP16) and only search used 4-bit, the graph would be
good and the question reduces to: can 4-bit PQ *navigate* a good graph well
enough to find the right regions? That's a different (and more answerable)
question than "can 4-bit do everything." Option B would answer it.

## Status
Option A tested (2026-07-23) — **NEGATIVE RESULT.** FP16 build does not improve
graph quality on arxiv100k. See results below.

## Option A results (FP16 beam_search during build)

arxiv100k, merged, R=26, L_build=100, pq_m=96/8-bit, 8 threads:

| build mode | recall@100 | QPS | R̄ | clustering | dead_ends | build time |
|------------|-----------|-----|----|----|-----------|------------|
| PQ (default) | 0.9242 | 6186 | 53.38 | 0.0992 | 0.0000 | 48s |
| FP16 | 0.9189 | 6248 | 53.71 | 0.1040 | 0.0000 | 48s |

**FP16 build gives slightly LOWER recall (−0.5pp) and slightly higher QPS (+1%).**
Graph stats are nearly identical.

### Why it doesn't help
8-bit PQ navigation is already good enough at this scale — the candidate pools
found by PQ beam_search are not meaningfully improved by FP16. Moreover, FP16
distances have FP16 precision (3 decimal digits) while PQ LUT distances
accumulate per-subspace in FP32, so the build's candidate ranking is actually
*less precise* with FP16 than with the PQ LUT. FP16 isn't "exact" — it has its
own quantization error, which on this dataset slightly degrades edge selection.

### Why build time didn't increase
The predicted 2-3× build-time slowdown didn't materialize (48s vs 48s). The
estimator's mini-builds (which also use FP16 under the env var) dominate the
analyze phase, and FHM makes l2sq_f16 fast enough that the per-eval cost is
comparable to the PQ LUT path at these sample sizes.

### Conclusion
Option A is dead. The build/search coupling through the shared quantizer is not
a meaningful lever at 8-bit PQ — the 8-bit PQ navigation is already sufficient
for graph construction. Option B (dual-quantizer for 4-bit search) is the only
remaining decoupling path, and it's architecturally heavy. The 0.73 PQ recall
ceiling remains structural for per-subspace PQ; the only escape is a different
quantizer architecture (ScaNN-style, documented separately).

### Bug fixed during the experiment
The `dist_to` lambda was restructured to not short-circuit on `store_` when
`precise_vec` returns null. FlatNodeStore has `store_ != null` but no
`precise_vec` (returns null), so the old code would skip the `build_ctx_->vecs`
fallback. This was a latent bug that only surfaces when `query_fp16` is set in
build mode (the FP16 build experiment). Fixed in commit 3c9c17f.
