# ScaNN-style Anisotropic Quantizer — Rearchitecture Scope

**Status:** Design scoping, 2026-07-23. NOT started; documented to define the
effort and risk for a future decision.

## Why this exists

The PQ-only recall ceiling is 0.73 at k=100 on arxiv-nomic 1.34M — structural
for **per-subspace PQ at (m=96, bits=8)**, NOT for quantization in general.
ScaNN reaches ~0.90 PQ-only on similar datasets via a fundamentally different
quantizer: a single full-vector codebook trained with an anisotropic objective.
This is the only known quantization-side escape from the 0.73 ceiling. All
parameter-tweak approaches (OPQ PCA rotation, per-vector/per-subspace anisotropic
weighting) have been tested and fail at production scale
(`docs/quantization_findings.md`).

This doc scopes what building the ScaNN-style quantizer would require.

## What ScaNN does differently

| | Sextant (current) | ScaNN |
|---|---|---|
| **codebook** | m=96 sub-codebooks × K=256 centroids × 8-dim each | 1 codebook × K=256-4096 centroids × full dim (768) |
| **code per vector** | 96 bytes (1 byte × 96 sub-centroids) | log2(K) bits = 8-12 bits (~1-1.5 bytes) |
| **distance** | LUT gather: sum 96 precomputed sub-distances | direct full-vector L2sq/dot against the assigned centroid |
| **training** | independent k-means per subspace | anisotropic k-means (weighted by ranking impact) |
| **per-query prep** | build 96×256 LUT (96KB, the 4.5% preprocess_query) | none (distance is direct) |

The code-size reduction (~70-100×) is the appeal: far less memory traffic per
candidate, no LUT build, no gather scatter. The anisotropic training is what
raises the recall ceiling.

## Rearchitecture scope (blast radius)

### 1. Quantizer layer — REWRITE (~1,700 lines)
Replace `PqQuantizer` with a new `AnisotropicQuantizer` (or parallel class):
- New codebook layout: K × dim (vs m × K × sub_dim).
- New training: anisotropic k-means objective (weight quantization error by
  impact on dot-product ranking). This is NOT standard k-means — it's a
  modified assignment step that accounts for the query-direction weighting.
- New encode: assign each vector to its nearest centroid (full-vector L2sq),
  store the centroid ID (log2(K) bits).
- New distance: direct dim-wide L2sq or dot against the centroid. No LUT.
- Serialization: new codebook format, new code format (bit-packed IDs).

### 2. On-disk format — BREAKING
- `.codes` sidecar shrinks ~70-100× (1-1.5 bytes/vector vs 96 bytes). At 1.34M,
  codes go from 124MB to ~1-2MB. At 1B, from ~96GB to ~1-1.5GB.
- `code_size` propagates to `node_size`, `.graph` layout, `static_node_size`.
  Every reader/writer touches it.
- The format is NOT stabilized (project rule allows this), so a clean break is
  acceptable — but every sidecar reader/writer must update in lockstep.

### 3. Hot path — RESTRUCTURE
- `lut_distance` / `lut_distance_batch4` (27% of profile): REPLACED by direct
  full-vector distance. The SVE2 batch4 gather machinery becomes irrelevant.
- `preprocess_query` (4.5%): REMOVED (no LUT to build).
- `beam_search_into`'s distance evaluation (the `dist_to` lambda, the FP16/PQ
  split, `pin_codes`): restructures around the new distance call. The FP16-ball
  tier (MemGraph) stays; the PQ tier is replaced by the anisotropic distance.
- HDC build mode (anchor LUT, cross-distance table): PQ-specific — either
  removed or paralleled.

### 4. Coupling — 56 call sites across 17 files
`code_size` appears 76 times in the hot path. Every node layout, every buffer
allocation, every BFS/pin operation assumes the current code layout. Files:
- `include/sextant/{builder,config,index,ivf_searcher,searcher}.hpp`
- `src/algo/vamana_core.{cpp,hpp}`
- `src/engine/{builder,estimator,index,ivf_searcher,partition,probe,searcher}.cpp`
- `src/quant/pq_quantizer.{cpp,hpp}`
- `src/storage/node_store.{cpp,hpp}`

### 5. Validation
Every benchmark, every test, both merged and IVF paths. High regression risk
for the merged path (which currently works well). The IVF path inherits
whatever the quantizer does.

## Estimated effort

**2-3 weeks of focused work:**
- Week 1: new quantizer class + anisotropic training (the algorithm is
  non-trivial; ScaNN's exact objective isn't fully documented in the paper).
- Week 2: format migration + hot-path rewrite (beam_search, pin, node layout).
- Week 3: validation, tuning K (number of full-vector centroids), regression
  testing on merged + IVF.

## Risk and uncertainty

- **The payoff is genuinely uncertain.** ScaNN's published ~0.90 PQ-only depends
  on their FULL training pipeline (which is sophisticated, includes learned
  weighting, and isn't fully reproducible from the paper alone). A partial
  implementation may not reach 0.90 — it could land at 0.78-0.85, which would
  reduce but not eliminate the rerank tax.
- **The 12.4% rerank tax reduction is the main prize.** If PQ-only reaches 0.85,
  rerank=2 might suffice where rerank=10 was needed — cutting rerank work 5×.
- **Memory-traffic reduction is a secondary prize.** 70-100× smaller codes
  means far less bandwidth per candidate — directly attacks the memory-wall
  scaling limit (4.3× at 8 threads).
- **Destabilizes the merged path.** The merged path currently works well; a
  quantizer rewrite risks regressions there. Mitigation: keep PqQuantizer as
  a parallel option, gate via config.

## When to pursue this

This is a research project, not an optimization. Pursue it when:
1. The 0.73 PQ ceiling is the blocking issue for a real product requirement
   (e.g. a customer needs recall@100 ≥ 0.95 without rerank=10).
2. The memory-bandwidth wall (4.3× scaling at 8t) is the blocking QPS limiter
   and the code-size reduction is worth the format break.
3. There's appetite for a 2-3 week focused effort with uncertain payoff.

Do NOT pursue it as one of several parallel optimization levers — the blast
radius is too wide for parallel work, and the payoff is too uncertain to
prioritize over the localized memory-access levers (memcpy elimination,
batch4 remainder, bucket queue).

## Alternatives that avoid the full rearchitecture

- **Add more PQ segments (higher m).** m=192 or m=256 doubles code size but
  raises per-subspace resolution. Untested at scale; may hit the same
  concentration-of-measure wall as OPQ. Cheaper to try than full ScaNN.
- **4-bit PQ with doubled m.** m=192/bits=4 keeps code size at 96 bytes but
  uses more, finer subspaces. The `probe_pq_config` machinery tested 4-bit
  variants; recall was lower at the same code size.
- **Hybrid: PQ for navigation + a lightweight second quantizer for refinement**
  (SymphonyQG-style). Avoids the FP32 rerank bandwidth without a full
  quantizer rewrite. Intermediate effort.
