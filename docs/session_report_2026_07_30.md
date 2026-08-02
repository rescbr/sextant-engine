# Session Report: 2026-07-30 / 2026-07-31

## Overview

This session attacked the query throughput bottleneck of the Sextant ANN
engine from multiple angles: rerank optimization, scan-bandwidth analysis,
k-means acceleration, closure theory, and sub-shard architecture. The
cumulative result is a **1.8× QPS improvement at matched recall** at 10M
scale on c4a, with a clear path to higher recall targets via Panorama +
rerank tuning.

---

## 1. Selective Rerank Investigation

### Approach A: PQ Error Bounds — NO-GO (proven)

Spike: `scripts/spike_pq_bounds.cpp`

Tested whether a statistical or Cauchy-Schwarz error bound on PQ distance
estimates could shrink the W=300 shortlist (scan-side selective rerank).

**Root cause (proven, not hypothesized):** PQ residuals are structurally
non-isotropic. The residual `r = o - c` lives in the centroid's Voronoi
cell, creating systematic alignment with data/query directions. Isotropy
test shows tails **23-1024× heavier than Gaussian** at z=3-4.

| Bound | IP (se100k) | L2sq (arxiv100k) |
|-------|-------------|------------------|
| STAT (free) | 100% over-prune, 0 recall | 32% prune, 4.85% false-prune |
| CS‖r‖ (+4 B) | 0% prune (too loose) | 0% prune (too loose) |
| E1 margin | catastrophic | catastrophic |

This definitively confirms: "PQ cannot borrow RaBitQ's selective-rerank
trick." The structural mechanism (Voronoi cell geometry) is now proven.

### Approach B: Panorama Progressive Rerank — VALIDATED, needs exploration

Spike: `scripts/spike_panorama_rerank.cpp`
Implementation: `tools/benchmark.cpp` (`--panorama-levels` flag)

Panorama bounds the **exact FP32 distance** (not the lossy PQ estimate).
Partition D dims into L levels, compute distance incrementally, prune via
Cauchy-Schwarz on remaining dims. Zero recall loss.

| Scale | Metric | QPS improvement | Note |
|-------|--------|-----------------|------|
| 99k (se100k) | IP | +7% (5475→5861) | p99 reduced 34% |
| 10M (Sphere) | IP | **0%** | Rerank is <3% of cycles at np=32 |

At 10M with np=32, W=300: rerank is negligible (2.9% of cycles per c4a
profile). Panorama's savings are invisible.

**BUT: Panorama becomes relevant at higher recall targets.** Current recall
at 10M is 0.67-0.68. Pushing to 0.85+ requires W=1000+ (10× more rerank
work), at which point rerank becomes 10-15% of cycles. Panorama's 2× rerank
speedup would then save 5-7% of total cycles. Combined with sub-shards
(reducing scan cost), Panorama + higher W is the path to production recall.

**Also fixed:** benchmark timing bug — p50/p99 historically excluded rerank
cost (timestamp captured before `process_results`). Now honest.

### c4a profiling at 10M (Sphere IP, K=1024, np=32, 8 threads)

| Component | % cycles |
|-----------|----------|
| on_block lambda (pq4_block32 SIMD + heap) | 67.5% |
| search_body_ (routing, merge) | 7.1% |
| dedup hash map clear | 4.6% |
| process_results (rerank) | 2.9% |
| dot_f16 (centroid routing) | 2.1% |

**Verdict: scan is COMPUTE-BOUND** (47% in `pq4_block32`), not memory-bound.
The lever is fewer codes scanned, not faster memory access.

---

## 2. K=4096 Validation

| Config | Recall@10 | QPS |
|--------|-----------|-----|
| K=1024 np=32 (baseline) | 0.6817 | 407 |
| K=4096 np=64 | 0.6531 | **652** (+60%) |
| K=4096 np=96 | 0.7113 | 478 (+18%) |

At matched recall ~0.68: ~580 QPS → **+43% QPS**. The np ∝ √K law holds;
fine-K boundary losses cost ~3pp (routing coverage, not closure).

---

## 3. Partition K-Means Optimization

Three commits accelerating the build's k-means phase:

1. **SIMD batch4 assignment** (`cadd8ee`): Uses the existing
   `code_distance_batch4` (SVE2/NEON gather-load) to evaluate 4 centroids
   per vector simultaneously. 1.5× faster on macOS NEON; more on c4a SVE2.

2. **Centroid padding** (`5350956`): Pad centroids to a multiple of 4,
   eliminating the scalar tail branch.

3. **Heap-buffer-overflow fix** (`e181559`): The batch4 padding made
   `centroids.size() = K_padded`, but the closure pass sized `dists(K)`.
   Fixed to `dists(centroids.size())`. Caught by ASAN on arxiv-nomic 1.34M.

---

## 4. Closure Theory

### Analytical derivation (`docs/closure_factor_derivation.md`)

Paper-ready derivation of K-dependent closure factor. Key results:

- **c is K-independent at leading order.** The shell count `μ(c) = c^{d_eff} − 1`
  is K-free because both cell radius and inter-centroid gap scale with the
  same power of K.
- **The 2.86pp K=4096 recall loss is routing coverage**, not closure.
- **Defensive c(K):** `c(K) = c₀·(K/K₀)^{1/(d_eff(d_eff−1))}` (exponent 0.05
  at d_eff=5). c(4096)=1.107. Validated empirically — recall unchanged.
- **SPANN's absolute-margin scheme is structurally superior** to ratio-based
  closure. ε/d₁ → ∞ as K grows.

### Implementation

- **Defensive c(K)** in `resolve_params.cpp` — auto-scales with K.
- **Absolute-margin closure** (`--closure-epsilon`): SPANN-style `d_best + ε`.
  `-1` = auto-compute from mean NN distance.
- Estimator propagation: `sub_shard_threshold`, `closure_epsilon` now flow
  through `estimate_config` to `ResolvedParams`.

---

## 5. Sub-Shard Architecture (the big win)

Design: `docs/sub_shard_design.md`

### Concept

Split large/skewed IVF shards into smaller sub-shards with two-level routing.
After coarse routing to np shards, route within each to the top-N sub-shards
via FP16 sub-centroid distances. Scan only those.

### Implementation (Phase 1 + Phase 2)

- **Storage:** `shard_NNNN/sub_MMMM/` directories (one file per sub-shard)
- **Build:** after partitioning, shards exceeding `target_sub_size` are split
  via local batch4-optimized k-means. Adaptive S (no cap): `S = ceil(shard_n / target)`.
- **Search:** two-level routing (coarse FP16 → sub-centroid FP16 `dist_f16`).
  `--sub-shard-n-probe` overrides at search time.
- **Closure:** sub-shard partition uses `closure_factor=1.05` to replicate
  boundary vectors across sub-shards.

### Results

**Local (se100k, 99k, K=64, threshold=1000):**

| sub_np | Recall@10 | QPS |
|--------|-----------|-----|
| 3 | 0.9675 | 9,173 |
| 4 | 0.9690 | 8,394 |

vs flat K=64 np=8: recall=0.9275, QPS=1,910 → **4.8× QPS at matched recall**

**c4a 10M (Sphere IP, K=1024, threshold=5000):**

| Config | Recall@10 | QPS | vs Flat |
|--------|-----------|-----|---------|
| Flat K=1024 np=32 | 0.6817 | 453 | baseline |
| Sub-shard sub_np=2 | 0.6179 | **1,050** | +131% |
| Sub-shard sub_np=3 | 0.6570 | **924** | +104% |
| Sub-shard sub_np=4 | 0.6689 | **834** | +84% |

At matched recall ~0.67: **1.8× QPS**. Build overhead: 742s vs 630s (18%).

### Architecture debt noted

- `builder.cpp` interweaves file I/O with algorithm logic. Future refactor:
  separate Encoder (pure), Packer (pure), Writer (I/O), Reader (I/O).
- PQ training is single-threaded (`PqQuantizer::train`). Future: parallelize
  per-subspace (m=96 independent k-means runs).
- Thread count discrepancy: `--threads 4` showed 9 threads during reservoir
  sampling. Needs audit.

---

## 6. Next Steps

### Immediate: Panorama × rerank for production recall

Current recall at 10M is 0.67-0.68. Production targets need 0.85+. The path:

1. **Increase W** (rerank shortlist) from 300 to 1000+. This raises recall
   but increases rerank cost from 3% to 10-15% of cycles.
2. **Enable Panorama** (`--panorama-levels 64`) to cut the increased rerank
   cost by ~2×. Panorama's benefit scales with W — more candidates = more
   pruning opportunities.
3. **Combine with sub-shards** (sub_np=3) to keep scan cost low. The total
   pipeline: sub-shard scan (fast) → large W shortlist → Panorama rerank
   (fast) → high recall.

This combination hasn't been benchmarked yet. The expected Pareto: sub-shards
cut scan by 2×, Panorama cuts rerank by 2×, larger W recovers recall. Net:
higher recall at similar or better QPS than current flat+W=300.

### Medium-term

- Phase 3: adaptive S auto-tuning (per-shard S based on size distribution)
- Phase 2 (live updates): split/merge API for insert/delete
- Parallelize PQ training (per-subspace parallelism)
- Refactor file I/O out of builder.cpp

### Long-term

- 100M+ scale validation
- GPU offload for scan (`docs/gpu-future-option.md`)
- SPANN-style absolute-margin closure as the default (replace ratio-based)

---

## Commit log (22 commits)

| Commit | Description |
|--------|-------------|
| `d64455b` | spike: PQ error bounds — NEGATIVE RESULT |
| `1aab867` | spike: Panorama progressive rerank — POSITIVE |
| `42a781e` | benchmark: Panorama flag + timing fix |
| `f5e7372` | gcp_bench: fix --ivf flag + Panorama A/B |
| `f42c3fa` | docs: selective rerank findings + next steps |
| `cadd8ee` | partition: SIMD batch4 k-means assignment |
| `5350956` | partition: pad centroids to multiple of 4 |
| `58dde20` | docs: analytical c(K) derivation |
| `eaf679e` | partition: defensive c(K) + absolute-margin closure |
| `55a546e` | feat: sub-shard Phase 1 — build-time routing |
| `9962113` | feat: sub-shard Phase 2 — closure + search override |
| `c6ed359` | feat: remove sub-shard S cap |
| `c2e614f` | fix: sub-shard k-means serial (avoid nested pool) |
| `94da56a` | fix: propagate sub_shard params through estimator |
| `e181559` | fix: heap-buffer-overflow in closure pass |
| `0d960b3` | docs: reorder sub-shard phases to 1→3→2 |
| `429b42e` | docs: sub-shard design document |
| + others | gcp_bench fixes, memory updates |
