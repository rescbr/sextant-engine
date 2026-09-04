# Test / benchmark datasets

Committed fixtures (used by the test suite):

- `mini10k_{base,query}.fbin` + `mini10k_gt.gtmm` — 10,000 × 768-dim
- `mini20k_{base,query}.fbin` + `mini20k_gt.gtmm` — 20,000 × 768-dim

Both are **uniform random samples without replacement** of
`arxiv100k_base.fbin` (numpy `default_rng(42).permutation`, original row
order preserved; 10K is a prefix of the 20K sample). Ground truth is
exact squared-L2 top-100 (GTMM: metric=0, per-query interleaved
[ids_k][dists_k]). Sample statistics verified against theory (top-10
overlap with the full-100K GT = 0.103 / 0.212 ≈ Binomial(10, 0.1/0.2)).

History note: the original mini GT files were byte-copies of
`arxiv100k_gt.gt` (ids up to 99921 — invalid for both bases) and the
bases were an unrecorded non-random subset; both were regenerated
2026-09-04.

Everything else in this directory (arxiv100k, arxiv_nomic, cohere_100k,
gist1m, sift1m, se100k, se_base_full, msmarco/) is a local benchmark
corpus — gitignored. All GT files are `.gtmm`; the legacy flat `.gt`
layout is unsupported by the engine.

Regenerating the minis from arxiv100k: uniform sample with seed 42,
then brute-force squared-L2 top-100 GT.
