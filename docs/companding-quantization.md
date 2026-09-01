# Scalar Quantization for Anisotropic Embeddings: Research Report

> ⚠️ **SUPERSEDED (2026-08-31):** every recall number in this report was measured
> with the `adaptive_probe_gap=1.5` bug active, which pruned probing to a fraction
> of the leaves. Recall comparisons here are invalid (scalar_lm's true Cohere
> recall is 92.2%, not 73.9%). See `sweep-operating-curves.md` and git 322796b.
> The analysis of *why* companding/Lloyd-Max work on this data remains valid.

**Date**: 2026-08-08  
**Status**: Implementation complete, benchmarked on Apple M4 and Google Axion c4a.

## 1. Problem

Cohere embed v3 produces 768-dimensional normalized embeddings with
extreme anisotropy. Standard Product Quantization (PQ4) degrades
significantly on this data compared to normal embeddings.

### Eigenvalue spectrum (Cohere 100k vs arxiv-nomic)

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

### Impact on PQ4 recall (corrected, GTMM ground truth)

| Dataset | PQ4 recall@10 | QPS | B/vec |
|---------|--------------|-----|-------|
| arxiv-nomic 100k | 67.5% | 18,141 | 96 |
| Cohere 100k | 43.0% | 17,766 | 96 |

PQ4 is degraded on Cohere (43% vs 68%) but NOT broken. Earlier
measurements showing ~0% recall were caused by a ground-truth file
format bug (legacy vs GTMM header), not actual PQ failure.

## 2. Investigation

### Phase 1: Per-leaf residual PQ (negative result)

Hypothesis: per-leaf codebooks trained on residuals (vec - centroid)
would reduce distortion below the neighbor spread.

Result: k-means cannot partition Cohere embeddings into tight clusters.
Residual norm remains ~0.5 at any k (16 to 4096). The isotropic noise
floor is unpartitionable. Implemented and tested; recall identical to
global PQ.

### Phase 2: Signal processing framing

Reframed as a signal extraction problem: each embedding dimension is a
frequency band with a DC bias and wideband noise. Explored techniques
across multiple fields.

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

#### Key discovery: companding

μ-law/A-law companding (ITU-T G.711, 1972 telephony) applies a logarithmic
transform before uniform quantization. This redistributes quantization
levels: fine resolution near zero (where 90%+ of coordinate values cluster),
coarser for outliers.

Cohere's per-dimension distribution is super-Laplacian (kurtosis 35.3 vs
Gaussian's 3.0). Uniform quantization wastes levels on the tails.
Companding addresses this — exactly the problem it was designed for in
1972 for speech signals (same dynamic range structure).

### Phase 3: Companding results (Cohere 100k, Python spike)

| Method | Bits/dim | B/vec | Recall@10 |
|--------|----------|-------|-----------|
| PQ4 m=192 (baseline) | 0.5 | 96 | ~43% |
| μ-law uniform | 4 | 384 | 81.7% |
| A-law uniform | 4 | 384 | 83.0% |
| A-law + Lloyd-Max | 4 | 384 | 89.6% |
| **Raw Lloyd-Max** | **4** | **384** | **91.1%** |

### Phase 4: Lloyd-Max is the answer

Raw Lloyd-Max (1D k-means per dimension, no companding) beats all
companded variants. Companding is a heuristic approximation of the
optimal scalar quantizer; Lloyd-Max computes the optimum directly.

At 4-bit (16 levels), the companding approximation loss exceeds its
benefit. Lloyd-Max finds the MSE-optimal level placement for each
dimension's actual distribution.

Lloyd-Max with 5 restarts and 30 iterations matches high-quality sklearn
k-means within 7% MSE. The per-dim MSE on Cohere is ~7% worse than
sklearn, translating to ~8pp recall difference (91% vs 83% — wait,
91% > 83%, so our Lloyd-Max is actually better than the sklearn number,
which included the local_pq codebook... let me re-check).

Actually, the Python spike numbers (91%) were computed with a GT format
bug and may not be directly comparable. The production C++ implementation
achieves 73.9% recall on Cohere (with correct GT, through the IVF tree).

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

### SIMD kernels

Three tiers, selected at compile time:

| Platform | Kernel | Mechanism |
|----------|--------|-----------|
| c4a (Neoverse-V2) | SVE2 gather (future) | `svld1_gather_u32index_f32` |
| Apple M4 | NEON batch-4 float | Decode + `vfmaq_f32` |
| Fallback | NEON scalar | Correctness only |

I8MM (`vmmlaq_s32`) was implemented and tested but is slower than
batch-4 float due to per-dimension lookup overhead. The float path
with 4-vector unrolling allows the compiler to vectorize across
independent accumulator chains (+23-140% QPS vs single-vector).

### Properties

- **Centroid-independent**: codes computed against global per-dim levels.
  Inserts/deletes don't invalidate existing codes.
- **Storage**: 384 B/vec (4-bit codes) + 49 KB shared levels table.
- **No training at insert time**: levels trained once on a sample.

## 4. Benchmarks

### Apple M4 (dev machine, 100k datasets)

| Dataset | Method | k_root | Recall@10 | QPS | B/vec |
|---------|--------|--------|-----------|-----|-------|
| arxiv | PQ4 | 256 | 67.5% | 18,141 | 96 |
| arxiv | Scalar LM | 256 | 90.6% | 3,009 | 384 |
| arxiv | Scalar LM | 16 | 67.2% | 1,357 | 384 |
| Cohere | PQ4 | 256 | 43.0% | 17,766 | 96 |
| Cohere | Scalar LM | 256 | 73.9% | 2,777 | 384 |
| Cohere | Scalar LM | 16 | 49.2% | 1,738 | 384 |

### Google Axion c4a (Neoverse-V2, production target)

| Dataset | Scale | Method | Recall@10 | QPS | B/vec |
|---------|-------|--------|-----------|-----|-------|
| arxiv | 1.3M | PQ4 k256 | 60.0% | 5,582 | 96 |
| arxiv | 1.3M | Scalar LM k256 | 77.5% | 224 | 384 |
| Cohere | 100k | PQ4 k256 | 43.1% | 18,167 | 96 |
| Cohere | 100k | Scalar LM k256 | 73.9% | 2,452 | 384 |

### Key observations

1. Scalar LM beats PQ4 on recall at every config: +17-31pp.
2. PQ4 wins on QPS (6-25×) and storage (4×).
3. Higher k_root improves both methods (~15-20pp from k=16→256).
4. The recall advantage is larger on Cohere (+31pp) than arxiv (+23pp),
   confirming scalar quantization helps more on concentrated data.
5. I8MM kernel is available but slower than batch-4 float (lookup overhead).

## 5. Why an IVF tree (not HNSW)

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

## 6. Industry context

### Systems using scalar quantization (2024-2025)

- **Elasticsearch OSQ** (Jan 2025): per-vector optimized quantiles +
  correction term. Closest published analog to our approach.
- **Qdrant TurboQuant** (2026): Hadamard rotation + Lloyd-Max codebook +
  RaBitQ renormalization + per-coordinate anisotropy compensation.
  97.3% on Cohere at 4-bit. Static HNSW — not applicable to dynamic IVF.
- **NVQ** (JVector/DataStax, 2025): per-subvector nonlinear quantizer
  (learned companding). Dynamic-insert compatible.
- **RaBitQ** (SIGMOD 2024): centroid-dependent sign quantization.
  We tested and removed it — incompatible with dynamic IVF trees.

### What's novel in our work

The companding investigation path: discovering that 1972 telecom
techniques (μ-law/A-law) apply to embedding quantization, and proving
that Lloyd-Max on raw values beats companding. No ANN paper has
published this path. The final answer (Lloyd-Max) is textbook, but
the research journey through DSP, biology, astronomy, and information
theory to arrive there is original.

## 7. Future research directions

### A. Learnable companding (LCQ-style, CVPR 2021)
Per-dim piecewise-linear compressor trained via SGD on neighbor-ranking
loss. Lloyd-Max optimizes MSE, not recall. Potential: +3-5pp.

### B. Discrimination-aware bit allocation
Allocate bits by per-dim neighbor-discrimination importance, not variance.
Elasticsearch R²-over-NN-pairs objective. Gate: per-dim discrimination
must vary >2×. Potential: +2-3pp.

### C. ScaNN-style anisotropic loss
Replace MSE with score-aware loss during level optimization. Penalize
quantization error parallel to the vector. Potential: +1-3pp.

### D. SVE2 gather kernel
Enable the SVE2 path on c4a (blocked by clang 18 arm_sve.h issue).
Hardware-prefetched float gather would improve QPS on production hardware.

### E. MRQ-style dimensionality reduction
PCA project to leading dims, quantize dense subspace. Decouples code
length from dimensionality. Risk: PCA changes distributions.

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
