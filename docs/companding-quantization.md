# Scalar Quantization for Anisotropic Embeddings: Research Report

**Date**: 2026-08-08, critically revised 2026-08-31  
**Status**: Implemented, validated, and re-benchmarked after the
`adaptive_probe_gap` bug fix (git f17c42a). This revision separates numbers that
are valid (measured without the tree, or post-fix) from those that were tainted,
and adds what the gap-bug investigation taught us about *methodology* — which
changes how quantizer research should proceed.

> **What was wrong before:** every recall number measured *through the IVF tree*
> before 2026-08-31 was depressed by the gap bug (probing pruned to 4-10 of 26
> leaves). Damage scaled with quantizer quality — scalar_lm lost 40pp, PQ4 lost
> 12pp, local_pq lost nothing. Numbers measured **in isolation** (Python spike,
> `spike_lm_recall`) were correct all along. Corrected tables below;
> operating curves in `sweep-operating-curves.md`.

## 1. Problem

Cohere embed v3 produces 768-dimensional normalized embeddings with
extreme anisotropy. Standard Product Quantization (PQ4) degrades
significantly on this data compared to normal embeddings.

### Eigenvalue spectrum (Cohere 100k vs arxiv-nomic) — valid, tree-independent

| Metric | Cohere v3 | arxiv-nomic |
|--------|-----------|-------------|
| eig0 / eig1 ratio | 57.3× | 1.3× |
| PC0 variance share | 52.3% | ~5% |
| Intrinsic dim (top-32) | 1.5 | 15.5 |
| Per-dim kurtosis | 35.3 | 1.0 |
| k-means residual norm (k=256) | 0.514 | — |

Cohere's PC0 is a DC component: 57× more variance than PC1, carrying
no neighbor-discriminating information (removing it preserves 100%
of neighbors). The remaining 43% of variance is an isotropic noise
floor spread across 767 dimensions.

### Impact on PQ4 recall — corrected

| Dataset | PQ4 m=96 | PQ4 m=192 | QPS (m=96) | B/vec |
|---------|----------|-----------|------------|-------|
| arxiv-nomic 100k | 53.7% | — | ~6.2k | 48/96 |
| Cohere 100k | 36.2% | 51.6% | ~6.1k | 48/96 |

(Old report: 67.5%/43.0% — gap-tainted and measured at different m.)
PQ4 is degraded on Cohere vs arxiv, and — critically — is **quantizer-bound**:
recall saturates by n-probe=4, so no amount of probing helps. The ceiling is
the 4-bit codes themselves.

## 2. Investigation

### Phase 1: Per-leaf residual PQ (negative result) — valid

Hypothesis: per-leaf codebooks trained on residuals (vec - centroid)
would reduce distortion below the neighbor spread.

Result: k-means cannot partition Cohere embeddings into tight clusters.
Residual norm remains ~0.5 at any k (16 to 4096). The isotropic noise
floor is unpartitionable. Implemented and tested; recall identical to
global PQ.

### Phase 2: Signal processing framing — valid

Reframed as a signal extraction problem: each embedding dimension is a
frequency band with a DC bias and wideband noise.

#### Techniques tested and rejected

| Technique | Source field | Result |
|-----------|-------------|--------|
| Spectral whitening (1/√λ scaling) | DSP/NLP | Destroys neighbor structure (30.9% survive) |
| Graph diffusion (MAGIC-style) | Biology | Destroys neighbors (44.6% preserved at t=1) |
| PCA rotation + companding | — | Worse (CLT makes dims Gaussian, kills companding) |
| Wiener shrinkage + companding | Astronomy | Worse (distorts per-dim distribution) |
| Water-filling bit allocation | Info theory | Marginal (similar variance per dim) |
| Dense-sparse decomposition | LLM compression | Poor (39% — signal spread across all dims) |
| Per-vector adaptive companding | ADPCM | No help (all vectors have same distribution shape) |
| Inter-dimension DPCM | Telecom | Dead end (contrastive training makes dims isotropic) |
| Power-law companding (p=1/3) | Panter-Dite | Worse than A-law at 4-bit |
| Dead-zone quantizer | Heavy-tail theory | Worse than Lloyd-Max |

**Note for future work:** the "PCA rotation + companding" rejection is more
important than it looked — see §7.E.

#### Key discovery: companding

μ-law/A-law companding (ITU-T G.711, 1972 telephony) applies a logarithmic
transform before uniform quantization. This redistributes quantization
levels: fine resolution near zero (where 90%+ of coordinate values cluster),
coarser for outliers.

Cohere's per-dimension distribution is super-Laplacian (kurtosis 35.3 vs
Gaussian's 3.0). Uniform quantization wastes levels on the tails.
Companding addresses this — exactly the problem it was designed for in
1972 for speech signals (same dynamic range structure).

### Phase 3: Companding results (Cohere 100k, Python spike) — VALID, and now confirmed

These were measured by brute force over the whole dataset, **no tree** —
so they were never gap-tainted. The previous revision of this report
distrusted them ("may not be directly comparable"); that distrust was
misplaced — the C++ isolated measurement (`spike_lm_recall`, 2026-08-31)
reproduces the ceiling at 92.21%:

| Method | Bits/dim | B/vec | Recall@10 |
|--------|----------|-------|-----------|
| PQ4 m=192 (baseline) | 0.5 | 96 | ~43-52% |
| μ-law uniform | 4 | 384 | 81.7% |
| A-law uniform | 4 | 384 | 83.0% |
| A-law + Lloyd-Max | 4 | 384 | 89.6% |
| **Raw Lloyd-Max** | **4** | **384** | **91.1%** (C++ re-measure: 92.2%) |

**Lesson: when tree-through numbers and isolated numbers disagree, trust the
isolated ones and suspect the tree.** The gap bug hid exactly this way.

### Phase 4: Lloyd-Max is the answer — revised with new evidence

Raw Lloyd-Max (1D k-means per dimension, no companding) beats all
companded variants. Companding is a heuristic approximation of the
optimal scalar quantizer; Lloyd-Max computes the optimum directly.

Training quality (2026-08-31 update): the C++ trainer had three defects —
the quantile init read **unsorted** data, empty clusters were never
reseeded, and restarts jittered by ±0.1% of range (all converged to the
same basin). All fixed (sorted init, max-error reseed, spaced-random
restarts, 3→10 restarts). Result: **recall unchanged (+0.05pp), MSE
improved.** Interpretation: at 4-bit, MSE quality is *not* the binding
constraint — the quantizer already sits at its information ceiling
(92.2%). Better MSE optimizes a proxy that no longer moves recall; only
changing the *objective* (ranking loss, §7.A) or the *bits* can.

## 3. Implementation

### Architecture: decode-then-dot

Instead of PQ's LUT-based FastScan, scalar Lloyd-Max decodes each
vector's 4-bit codes to float (via per-dim level lookup) and computes
the dot product directly. This avoids the FastScan LUT quantization
dynamic range bug (see below).

### Why not FastScan

FastScan's `quantize_lut_u4` uses one global scale: `A = 15/max_span`.
For m=768 (one subquantizer per dimension), per-segment spans vary by
80× (the DC dimension dominates). The global scale crushes 98% of LUT
values to zero. This is the same dynamic range problem as the embeddings
themselves, appearing in the distance computation.

The decode-then-dot approach sidesteps this entirely: no LUT, no
quantization of distance values, float precision throughout.

### SIMD kernels — updated with measured outcomes (2026-08-31)

| Platform | Kernel | Status |
|----------|--------|--------|
| Apple M4 | NEON batch-4 float (scan), scalar batch-4 unrolled | **Active** — the hand-unrolled scalar batch-4 beat all vectorized attempts |
| c4a (Neoverse-V2) | SVE2 gather (rerank only) | Active — correct (byte-identical), modest win (W≈300 entries) |
| c4a (Neoverse-V2) | SVE2 gather (main scan) | **Measured dead end** — per-vector: 95 QPS; 4-vector concurrent: 65 QPS; scalar batch-4 baseline: 216 QPS |

The SVE2 scan result is worth internalizing: the levels table (48 KB) is
L2-resident with predictable *scalar* access in the batch-4 loop. SVE2 gather
latency — even with 4 independent FMA chains and hardware prefetch — exceeds
scalar L1-hit level loads on Neoverse-V2. **Gather-based decode-dot does not
beat amortized scalar lookups on current ARM cores.** Any future quantizer
designed for *sequential* code→value mapping (no LUT gather) would change this
calculus — a SIMD-research reason to prefer uniform-step quantizers, if recall
allows.

I8MM (`vmmlaq_s32`) was implemented and tested but is slower than
batch-4 float due to per-dimension lookup overhead.

### Properties

- **Centroid-independent**: codes computed against global per-dim levels.
  Inserts/deletes don't invalidate existing codes.
- **Storage**: 384 B/vec (4-bit codes) + 49 KB shared levels table.
- **No training at insert time**: levels trained once on a sample.

## 4. Benchmarks — corrected (post gap-fix, 2026-08-31)

All via the `sweep` subcommand (bit-exact vs standalone), M4 8 threads,
n-probe-ln=8 (full leaf probing), W=100:

| Dataset | Method | B/vec | Recall@10 | QPS |
|---------|--------|-------|-----------|-----|
| Cohere | Scalar LM | 384 | **92.2%** | ~371 |
| Cohere | PQ4 m=192 | 96 | 51.6% | ~7.8k |
| Cohere | PQ4 m=96 | 48 | 36.2% | ~6.1k |
| arxiv | Scalar LM | 384 | **94.5%** | ~371 |
| arxiv | PQ4 m=96 | 48 | 53.7% | ~6.2k |

### The ceiling result

Cohere scalar_lm through the tree at full probing = **92.2%**, exactly equal
to the isolated-quantizer measurement (0.9221). **The tree adds zero recall
loss.** Consequences:

1. Quantizer research can iterate entirely in the isolated harness
   (`scripts/spike_lm_recall.cpp`) — the tree is a faithful carrier.
2. Remaining headroom (7.8pp to 100%) belongs to the quantizer, not the
   tree. Qdrant reports 97.3% on Cohere at 4-bit (static HNSW + TurboQuant),
   suggesting ~5pp of it is reachable at this bitrate.
3. QPS differences between quantizers are then the only tree-side variable.

### Methodological rules (learned the hard way)

1. **Isolate before judging through the tree.** The gap bug damaged better
   quantizers more (scalar_lm −40pp, PQ4 −12pp, local_pq ±0) — tree-through
   comparisons can *invert* conclusions about quantizer quality.
2. **Self-lookup test**: query with base vectors; a sound index must return
   each at distance ≈ 2×quantization-MSE×dim (~0.004 here). Fastest sanity
   check; it caught the gap bug's signature in minutes.
3. **Check probe saturation**: if recall stops improving with n-probe before
   reaching the isolated ceiling, the tree is losing candidates (bug or
   pruning). If it saturates *at* the isolated ceiling, the quantizer is bound.
4. **W (rerank shortlist) is a non-knob** for these quantizers — flat from
   W=100. Historical claims that large W was needed were gap-bug artifacts.

## 5. Why an IVF tree (not HNSW) — unchanged

HNSW dominates RAM-resident benchmarks (2-3× QPS, 97%+ recall). But its
random-access graph traversal fails on disk. At billion scale:

- HNSW+INT8 needs ~900 GB RAM (~$6,500/month on AWS).
- IVF tree needs ~8 GB RAM + ~500 GB NVMe (~$185/month). 35× cheaper.

The industry knows this: Milvus recommends IVF for n>100M. DiskHIVF
(NeurIPS 2025) designs hierarchical IVF for disk. FAISS's 1.5T
proof-of-concept uses IVF + OnDiskInvertedLists.

Our IVF tree is designed for disk-resident search from the ground up:
sequential leaf scan (NVMe-friendly), filter pruning at centroid level,
no graph overhead. Scalar Lloyd-Max quantization with centroid-independent
codes enables dynamic inserts without re-quantization.

## 6. Industry context — updated positioning

- **Qdrant TurboQuant (2026)**: Hadamard rotation + Lloyd-Max + RaBitQ
  renormalization. **97.3% on Cohere at 4-bit** — the number to beat; ~5pp
  above our 92.2%. Their stack is static HNSW; ours is dynamic IVF. Their
  result proves the headroom exists at this bitrate.
- **Elasticsearch OSQ**: per-vector optimized quantiles + correction term.
- **NVQ (JVector)**: per-subvector nonlinear quantizer (learned companding).
- **RaBitQ (SIGMOD 2024)**: centroid-dependent; incompatible with dynamic
  IVF (tested, removed).

### What's novel in our work

The companding investigation path — 1972 telecom techniques applied to
embedding quantization, and the proof that Lloyd-Max on raw values beats
companding — plus the negative-result table (§2 Phase 2). Also: the
demonstration that a *tree-side* recall bug can masquerade as a quantizer
quality problem for weeks, and the isolation methodology that detects it.

## 7. Future research directions — revised with constraints

### A. Learnable companding / ranking-loss optimization (LCQ-style) — the main recall lever

Lloyd-Max optimizes MSE; we showed MSE no longer moves recall (§2 Phase 4).
The remaining 7.8pp gap needs an objective that preserves neighbor *ordering*.
Per-dim piecewise-linear compressor trained via SGD on neighbor-ranking loss.
Headroom bounded by Qdrant's 97.3% → realistic target ~5pp.
**Evaluate in the isolated harness first** (rule §4.1).

### B. Discrimination-aware bit allocation

Allocate bits by per-dim neighbor-discrimination importance, not variance.
Gate: per-dim discrimination must vary >2× (measure with the isolated spike).
At fixed total bits this trades tail dims for discriminative ones — the first
thing to try for the middle-ground frontier (below).

### C. ScaNN-style anisotropic loss

Replace MSE with score-aware loss during level optimization. Penalize
quantization error parallel to the vector. Synergistic with A (same trainer,
different loss).

### D. SVE2 gather kernel — RESOLVED, negative

Implemented and measured (two designs). Gather latency > scalar L1 hits on
Neoverse-V2 for this access pattern. Do not revisit unless the quantizer
changes to sequential code→value mapping (§3 SIMD note).

### E. Dimensionality reduction for smaller codes — has a known trap

Motivation: the bytes/recall frontier (48B/36% ↔ 384B/92.2%) is unmapped;
smaller codes are the billion-scale economics question.

**The trap:** the obvious approach — PCA-project then quantize — collides
with our own Phase 2 result: rotation makes dims Gaussian (CLT), kurtosis
35.3 → ~3, and Gaussian dims waste Lloyd-Max levels (the companding advantage
evaporates; that row in the rejection table was measuring exactly this).
A rotated-then-Lloyd-Max quantizer loses the property that makes Lloyd-Max
win on this data.

Viable alternatives:
- **Raw-dim selection** (no rotation): keep the highest-discrimination raw
  dims, drop tail dims. Preserves per-dim distributions; code length scales
  down; recall floor set by dropped dims' information. Cheap to evaluate in
  the isolated spike (zero new machinery).
- **Partial rotation**: rotate only within near-degenerate subspaces, leaving
  the heavy-tailed structure intact. Harder; needs a spike to validate.
- First experiment: isolated recall vs #dims kept curve for raw-dim selection
  — bounds the whole direction in an afternoon.

### F. Bound the 4-bit headroom (new, cheap, do first)

Measure isolated recall of **8-bit** scalar LM (the quantizer supports it;
only the tree scan path rejects it). That number is the ceiling of all 4-bit
work: if 8-bit gives ~97%, ranking-loss training (A) can realistically reach
it; if 8-bit gives ~93%, the 4-bit information limit is nearly exhausted and
research effort should shift to E (more bits on fewer dims) instead.

## 8. References

- ITU-T G.711 (1972). Pulse code modulation (μ-law/A-law).
- Panter & Dite (1951). Quantization distortion in PCM.
- Lloyd (1957) / Max (1960). Least squares quantization (Lloyd-Max).
- Mu & Viswanath (ICLR 2018). All-but-the-top.
- Gao et al. (EMNLP 2021). SimCSE (anisotropy theory).
- Guo et al. (ICML 2020). ScaNN (anisotropic vector quantization).
- Yamamoto (CVPR 2021). Learnable Companding Quantization (LCQ).
- Gao & Cheng (SIGMOD 2024). RaBitQ.
- Pleshkov (2026). Qdrant TurboQuant + RaBitQ on Cohere.
- Elastic (2024-2025). Optimized Scalar Quantization (OSQ/BBQ).
- Tepper & Willke (2025). NVQ (non-uniform vector quantization).
- Zandieh et al. (2026). TurboQuant (arXiv:2504.19874).
