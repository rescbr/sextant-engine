# Recall/QPS Operating Curves (post gap-fix)

**Date:** 2026-08-31 · **Machine:** Apple M4, 8 threads · **Datasets:** cohere 100k, arxiv 100k
(d=768, unit-norm) · **Measured with:** the `sweep` subcommand (W-sharing: one scan per
n-probe serves all W values; bit-exact vs standalone runs)

> **Context:** all numbers in `companding-quantization.md` and earlier benchmark docs were
> measured with the `adaptive_probe_gap=1.5` bug active (pruned probing to 4-10 of 26
> leaves; damage scaled with quantizer quality — scalar_lm lost 40pp). This document
> supersedes them for recall comparisons. See git 322796b.

## n-probe sweep (n-probe-ln=8, W=100)

| np | cohere scalar_lm | QPS | arxiv scalar_lm | QPS |
|----|------------------|-----|-----------------|-----|
| 1  | 72.8%            | 826 | 76.8%           | 1193|
| 2  | 85.9%            | 509 | 88.8%           | 722 |
| 4  | **91.1%**        | **366** | **93.8%**  | **388** |
| 8  | 92.2%            | 224*| 94.5%           | 201*|

\* shared pass ran at W=1000; a standalone W=100 search measures ≈371 QPS.

## Leaf-probe sweep (np=8, W=100) — the dataset discriminator

| ln | cohere | QPS | arxiv | QPS |
|----|--------|-----|-------|-----|
| 1  | 31.6%  | 1231| 32.1% | 1115|
| 2  | 57.0%  | 622 | 63.3% | 578 |
| 4  | 76.1%  | 461 | **94.5%** | 370 |
| 8 (all) | **92.2%** | 371 | 94.5% | 361 |

## PQ4 m=96 reference (48 B/vec) — quantizer-bound

np=8: cohere **36.2%** @ 6.1k QPS, arxiv **53.7%** @ 6.2k QPS. Recall saturates by
np=4: the ceiling is the 4-bit codes themselves, not probing. (cohere PQ4 m=192:
51.6% from an earlier A/B.)

## Findings

1. **W is a non-knob.** Recall is flat from W=100 across both datasets and both
   quantizer families (W=30 slightly low on cohere at np≥4). The historical
   "W=5000 needed for scalar_lm" was a gap-bug artifact, not physics.

2. **Root-level sweet spot is dataset-agnostic: np=4, W=100** → 91-94% at
   ~370 QPS. np=8 buys ~1pp for ~1.8× QPS.

3. **The datasets differ only at the leaf level.** arxiv reaches *full* recall at
   ln=4 — leaf assignment works, real cluster structure. cohere needs all leaves
   (noise-dominated, neighbors scatter). But cohere at full probing **exactly hits
   the quantizer ceiling** (92.2% = the isolated-quantizer measurement 0.9221).
   Conclusion: Cohere's noise costs **QPS, not recall** — it forces exhaustive leaf
   scanning. Leaf-assignment quality is therefore a *throughput* lever
   (closure/replication tuning), not a recall lever.

4. **Two operating classes.** scalar_lm (384 B/vec): probe-bound, keeps improving
   with probing, >91% recall at ~370 QPS. PQ4 m=96 (48 B/vec): quantizer-bound
   at 36-54%, 6-8k QPS, probing irrelevant. 8× bytes buys 2.5× recall.

## Operating recommendations

| Data profile | Config | Expect |
|--------------|--------|--------|
| Clustered (arxiv-like) | np=4, ln=4, W=100 | ~94% @ ~370 QPS |
| Noise-dominated (cohere-like) | np=4, ln=all, W=100 | ~91% @ ~370 QPS |
| QPS-first, recall-tolerant | PQ4 m=96, np=4 | 36-54% @ ~7k QPS |

## Open frontier

- **Bytes/recall middle ground unmapped**: PQ8, PQ4 m=192, or scalar_lm on
  PCA-reduced dims. Between 48B/36% and 384B/92% the knee is unknown — at
  billion scale this is the deployment-economics question (48 GB vs 384 GB
  at 1B vectors).
- **Cohere closure tuning**: raise replication so fewer leaves must be scanned
  (QPS win; recall already at ceiling).
