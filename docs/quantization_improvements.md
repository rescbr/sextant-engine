# Quantization Improvements: Raising the PQ-Only Recall Ceiling

## The problem

At full scale (1.34M arxiv-nomic), PQ-only recall@100 = **0.73** with our
plain PQ (m=96, 8-bit). This means 27% of the true top-100 are misranked by
PQ distance. The rerank step (oversampling k×rerank candidates and re-sorting
by exact distance) compensates — but at rerank=10, that's fetch_k=1000
candidates per query, requiring 3MB of base-vector reads. This rerank tax is
the dominant throughput bottleneck (~5× overhead vs no-rerank).

The VIBE leaders (SymphonyQG ~3200 QPS, ScaNN ~3200 QPS at recall@100=0.95)
don't pay this tax. Their quantization raises the PQ-only ceiling high enough
that they either skip rerank or use a minimal oversample (rerank=2). Closing
the throughput gap requires raising our PQ-only recall ceiling.

## Why plain PQ recall is low

Plain PQ splits the vector into m sub-vectors, each quantized independently
with k-means into 2^bits centroids. Two factors limit recall:

1. **Subspace independence assumption.** PQ assumes the sub-vectors are
   independent — that quantizing each subspace separately is near-optimal.
   In practice, real embeddings have inter-dimensional correlations. PQ
   ignores these, losing information. At m=96/768-dim, each 8-dim subspace
   is quantized to 256 centroids — fine-grained within the subspace, but
   cross-subspace structure is lost.

2. **Isotropic quantization error.** Plain PQ minimizes reconstruction
   error (MSE) equally in all directions. But for nearest-neighbor ranking,
   errors along the query direction matter more than errors perpendicular
   to it. Plain PQ doesn't know the query direction at train time, so it
   spreads error isotropically — wasting bits on directions that don't
   affect ranking.

## Approach 1: OPQ (Optimized Product Quantization)

**Paper:** Ge et al., "Optimized Product Quantization" (TPAMI 2013).
**Used by:** SymphonyQG, many ANN systems.

### What it does
OPQ adds a **learned rotation matrix** R (d×d) applied before PQ splitting.
Instead of quantizing x directly, it quantizes R·x. The rotation is
optimized (via gradient descent or alternating minimization) to make the
rotated sub-vectors as independent as possible — aligning the PQ split
boundaries with the data's principal components.

### Why it helps recall
- The rotation decorrelates the sub-vectors, making the independence
  assumption closer to true → each subspace's k-means is more effective.
- The split boundaries align with the data geometry (eigenvectors) →
  less information lost at the split points.
- Typical improvement: 5-15% recall at the same m/bits, or equivalent
  recall at ~30% fewer bytes.

### Implementation cost
- **Training**: learn the d×d rotation matrix (one SVD + alternating
  minimization, ~seconds on a 20K sample). The existing k-means machinery
  is reused.
- **Encoding**: apply R before encoding (one matrix-vector multiply per
  vector, O(d²) — cheap at d=768, ~0.6ms/vector but parallelizable).
- **Search**: apply R to the query once (same O(d²) multiply), then
  proceed with standard ADC. No search-path architecture change.
- **Storage**: the rotation matrix (d×d×4 = 2.4MB at d=768) is stored in
  .meta alongside the codebook. Negligible.

### Why we tried OPQ before and it didn't help
Previous attempt (documented in t13-opq-direction.md / t13a_opq_findings.md)
found OPQ gave minimal improvement on our datasets. Likely reasons:
- The test was at k=10, where plain PQ already has high recall (~0.99 with
  rerank). OPQ's benefit is largest when PQ-only recall is LOW — exactly
  the k=100 regime where PQ-only recall is 0.73. The cost/benefit shifts
  dramatically at k=100.
- The rotation may not have been well-optimized (the alternating
  minimization may have needed more iterations or a better initialization).
- Worth revisiting at k=100 with the mini-build measurement infrastructure.

### Estimated impact on PQ-only recall@100
Plain PQ ceiling: 0.73. With OPQ: likely **0.80-0.85** (based on literature
for similar dimensionality). This would reduce the rerank multiplier from
10 to ~4-5 — meaningful but not transformative.

---

## Approach 2: ScaNN's Anisotropic Quantization

**Paper:** Guo et al., "Accelerating Large-Scale Inference with Anisotropic
Vector Quantization" (NeurIPS 2020).
**Used by:** ScaNN (Google's production ANN library).

### What it does
ScaNN replaces the isotropic MSE objective with an **anisotropic** objective
that weights quantization error by its impact on dot-product (inner product)
ranking. The key insight: for a query q and database vector x, the dot product
⟨q, x⟩ is approximated by ⟨q, x̂⟩ where x̂ is the quantized version. The
error is ⟨q, x - x̂⟩ = ⟨q, ε⟩ — the component of the quantization error
ALONG the query direction. ScaNN's training minimizes a weighted MSE where
the weight is proportional to how much the error affects the dot product
ranking.

More precisely, ScaNN decomposes each vector x into:
- A **norm component** (projection onto the query direction): quantized
  with high precision (this dominates the distance ranking).
- A **perpendicular component**: quantized with lower precision (doesn't
  affect ranking as much).

This is implemented via a **learned lookup table structure** where the ADC
computation prioritizes the dimensions that matter for ranking.

### Why it helps recall
- For L2-normalized vectors (which arxiv-nomic is), L2 distance and inner
  product are equivalent: ||q-x||² = 2 - 2⟨q,x⟩. So ranking by ⟨q,x⟩ is
  the same as ranking by L2 distance. ScaNN's anisotropic objective directly
  optimizes for this ranking.
- The isotropic MSE of plain PQ wastes bits on perpendicular directions
  that don't affect the ranking. ScaNN reallocates those bits to the
  parallel direction → higher effective precision where it matters.
- Typical improvement: 10-25% recall over plain PQ at the same m/bits,
  especially at high recall levels where the neighborhood is tightly packed.

### Implementation cost
- **Training**: ScaNN's anisotropic k-means requires a modified objective
  (weighted by the query-direction projection). The training is more
  expensive than plain k-means but similar in structure. Google's implementation
  uses a two-stage approach (warm-start from plain PQ, then fine-tune).
- **Encoding**: same as plain PQ (lookup table indexing), but the tables
  are structured differently (anisotropic weighting baked into the codebook).
- **Search**: the ADC computation is slightly different (the anisotropic
  weighting affects the LUT construction, not the per-code lookup). The
  per-query LUT is built once, then distance is the same gather+sum.
- **Storage**: same as plain PQ (no extra matrix needed — the anisotropy
  is baked into the codebook).

### Why it's the more promising direction
- ScaNN achieves ~3200 QPS at recall@100=0.95 on arxiv-nomic — it's the
  proven approach for this exact benchmark.
- The anisotropic objective directly targets the ranking problem (not just
  reconstruction), which is what matters for recall.
- No per-query rotation overhead (unlike OPQ which needs R·q per query).
- The LUT structure is compatible with our existing ADC infrastructure
  (the change is in codebook training, not in the search path).

### Estimated impact on PQ-only recall@100
Plain PQ ceiling: 0.73. With ScaNN-style anisotropic: likely **0.85-0.92**
(based on ScaNN's published results on similar datasets). This would reduce
the rerank multiplier from 10 to ~2-3 — a **3-5× rerank bandwidth reduction**,
which directly translates to throughput improvement.

---

## Approach 3: OPQ + Anisotropic (combined)

The two approaches are complementary:
- OPQ decorrelates sub-vectors (geometric alignment).
- Anisotropic weighting prioritizes ranking-relevant dimensions.

Combined (OPQ rotation → anisotropic PQ on the rotated space): potentially
**0.88-0.95** PQ-only recall@100. At that level, rerank=1 (no rerank) might
suffice for recall@100=0.90, and rerank=2 for 0.95 — eliminating the rerank
tax almost entirely.

SymphonyQG uses OPQ + reordering (a form of anisotropic structuring),
achieving the best of both. This is likely the recipe to replicate.

---

## Recommendation

**STATUS (2026-07-23): all three approaches below have been prototyped and
measured. Summary of what's dead vs alive.**

### What's DEAD (do not revisit as parameter tweaks)

**OPQ via PCA rotation (`4580c88`):** +1.03pp on arxiv100k, **0pp at 1.34M
scale** (0.7300 → 0.7300). This implemented the rotation half of OPQ — PCA
eigendecomposition of the data covariance, R = V^T, rotate training data +
queries to the eigenbasis before PQ splitting. This is the standard "OPQ via
PCA initialization." It is NOT the full OPQ algorithm (Ge et al., TPAMI 2013),
which **alternates** between fixing R and re-optimizing codebooks (k-means on
rotated space), then fixing codebooks and re-optimizing R, iterating to
convergence (typically 10-25 iterations). The PCA-init version is a lower
bound on what full OPQ achieves.

However: the reason PCA-init gave 0pp at scale is NOT primarily "the rotation
wasn't iterated." The first-order limit is the per-subspace-independence
assumption itself (256 centroids per 8-dim subspace can't represent recall@100
at 1.34M). Cross-subspace decorrelation — what both PCA-init and full
alternating OPQ improve — is a second-order effect. The 7.9% MSE improvement
from PCA rotation didn't move recall at scale; a few more percent MSE from
iterating is unlikely to either on arxiv-nomic (already variance-balanced/
spherized embeddings with weak cross-subspace correlation).

Status: **PCA-init OPQ is DEAD (validated). Full alternating OPQ is formally
untested but low expected value (<30% chance of meaningful gain at 1.34M given
the PCA-init result + the dataset's decorrelation profile).** Not worth the
implementation effort vs the architectural alternatives below.

**Per-vector anisotropic proxy (`5651d1d`):** recall *degraded* monotonically
with λ. Per-vector direction ≠ expected query direction.

**Per-subspace covariance scale-transform (`5997d7d`):** −0.56pp. The global
anisotropy is spread across 96 subspaces; each 8-dim sub-covariance is nearly
isotropic, so per-subspace weighting can't redistribute error.

### What's ALIVE but architecturally heavy (genuine Tier-1 lever)

**ScaNN-style full-vector anisotropic quantization.** The reason ScaNN reaches
~0.90 PQ-only (vs our 0.73) is that it **does not use per-subspace PQ for
search-time distance**. It uses a single full-vector quantizer trained with an
anisotropic objective that weights quantization error by its impact on ranking.
This escapes the per-subspace-independence limit entirely — the ceiling is
structural for *our* quantizer (m=96/bits=8 PQ), not for quantization in
general.

Why the prior anisotropic attempts didn't capture this: they applied anisotropic
*weighting* to the existing per-subspace PQ. ScaNN's mechanism is a different
*codebook structure* (full-vector, not per-subspace) trained end-to-end with
the anisotropic objective. That's a major architectural change — different
quantizer, different training pipeline, different search-time distance — not a
parameter tweak.

This is the only known quantization-side escape from the 0.73 ceiling. It's
untested in Sextant and would be a substantial implementation effort. If
pursued, it would help BOTH merged and IVF equally (same quantizer) and could
reduce or eliminate the 12% rerank tax.

**SymphonyQG-style multi-level reordering.** Uses a coarse quantizer for
initial ranking + a finer one for refinement, avoiding the FP32 rerank
bandwidth. Also architecturally heavy; avoids rerank rather than improving PQ.

### The honest framing
The PQ-only ceiling (0.73) is structural for **per-subspace PQ at (m=96,
bits=8)**. It is NOT structural for quantization in general — ScaNN and
SymphonyQG escape it via fundamentally different quantizer architectures that
we have not built. The dead ends above are dead because they're parameter
tweaks to the existing quantizer; the alive lever requires building a different
quantizer.
them. This is SymphonyQG's recipe.

---

## Data points for calibration

From mini10k (R=32, L=2000, k=100, no rerank):
| Config | Distortion | recall@100 |
|--------|-----------|------------|
| m=64/4 | 0.1270 | 0.7904 |
| m=64/8 | 0.0761 | 0.8372 |
| m=96/4 | 0.0916 | 0.8176 |
| m=96/8 | 0.0402 | 0.8509 |
| m=128/4| 0.0662 | 0.8297 |
| m=128/8| 0.0272 | 0.8610 |
| m=192/4| 0.0399 | 0.8430 |
| m=192/8| 0.0160 | 0.8658 |
| m=256/4| 0.0302 | 0.8526 |
| m=256/8| 0.0101 | 0.8682 |

Pearson r(distortion, recall) = -0.98. Distortion is a valid ranking proxy.

From full-scale (1.34M, R=32, L=1000, k=100):
- m=96/8 PQ-only ceiling: 0.7301 (vs 0.8509 on mini10k — the scale gap)
- m=96/8 with rerank=10: 0.9931

The scale gap: PQ-only recall DROPS at scale (0.85 → 0.73) because
concentration of measure makes neighborhoods tighter, amplifying PQ's
fixed absolute error relative to the shrinking inter-neighbor distances.
Rerank recovers (0.73 → 0.99) because the graph visits the right nodes
(the denser full-scale graph navigates better); rerank just fixes the
PQ ranking errors among the visited nodes.
