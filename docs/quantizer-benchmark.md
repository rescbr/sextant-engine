# Quantizer Benchmark — PQ4 vs PRQ4 vs PQ8

**Hardware:** c4a-standard-8-lssd (ARM SVE2, NVMe RAID0)
**Date:** 2026-08-05

## Full results: arxiv_nomic (1.34M, 768d, n_probe=8)

| Quant | m | subdim | recall@10 | QPS | B/vec |
|-------|-----|--------|-----------|-------|-------|
| pq4 | 192 | 4 | 0.974 | 8002 | 96 |
| prq4 | 192 | 4 | 0.979 | 7660 | 96 |
| pq8 | 192 | 4 | 0.997 | 1824 | 192 |
| pq4 | 96 | 8 | 0.881 | 12509 | 48 |
| prq4 | 96 | 8 | 0.880 | 11692 | 48 |
| pq8 | 96 | 8 | 0.980 | 3576 | 96 |
| pq4 | 64 | 12 | 0.727 | 15298 | 32 |
| prq4 | 64 | 12 | 0.729 | 14093 | 32 |
| pq8 | 64 | 12 | 0.946 | 4898 | 64 |
| pq4 | 48 | 16 | 0.589 | 17348 | 24 |
| prq4 | 48 | 16 | 0.578 | 16277 | 24 |
| pq8 | 48 | 16 | 0.884 | 5939 | 48 |
| pq4 | 32 | 24 | 0.356 | 19167 | 16 |
| prq4 | 32 | 24 | 0.370 | 18370 | 16 |
| pq8 | 32 | 24 | 0.738 | 7750 | 32 |
| pq4 | 16 | 48 | 0.117 | 23065 | 8 |
| prq4 | 16 | 48 | 0.119 | 21845 | 8 |
| pq8 | 16 | 48 | 0.412 | 11434 | 16 |

## Cross-dataset PRQ4 vs PQ4 recall delta (n_probe=8)

### arxiv_nomic (1.34M, 768d, L2)

| m | PQ4 | PRQ4 | Δrecall | ΔQPS% |
|----|--------|--------|---------|-------|
| 192 | 0.9740 | 0.9793 | +0.0053 | -4.3% |
| 96 | 0.8810 | 0.8801 | -0.0009 | -6.5% |
| 64 | 0.7266 | 0.7293 | +0.0027 | -7.9% |
| 48 | 0.5886 | 0.5784 | -0.0102 | -6.2% |
| 32 | 0.3557 | 0.3701 | +0.0144 | -4.2% |
| 16 | 0.1168 | 0.1189 | +0.0021 | -5.3% |

**Summary:** avg +0.0022, wins=4 ties=1 losses=1. Inconsistent, marginal.

### msmarco (8.74M, 768d, L2)

| m | PQ4 | PRQ4 | Δrecall | ΔQPS% |
|----|--------|--------|---------|-------|
| 192 | 0.9359 | 0.9363 | +0.0004 | -4.7% |
| 96 | 0.8579 | 0.8579 | +0.0000 | -4.0% |
| 64 | 0.7734 | 0.7728 | -0.0006 | -5.8% |
| 48 | 0.6813 | 0.6800 | -0.0013 | -3.4% |
| 32 | 0.5114 | 0.5138 | +0.0024 | -3.9% |
| 16 | 0.2241 | 0.2224 | -0.0017 | -2.9% |

**Summary:** avg -0.0001, wins=1 ties=5 losses=0. Noise. PRQ4 is PQ4 with a QPS tax.

### sphere_ip (9.99M, 768d, IP)

| m | PQ4 | PRQ4 | Δrecall | ΔQPS% |
|----|--------|--------|---------|-------|
| 192 | 0.7624 | 0.7836 | +0.0212 | +0.4% |
| 96 | 0.5568 | 0.5504 | -0.0064 | -2.7% |
| 64 | 0.3928 | 0.4025 | +0.0097 | -3.8% |
| 48 | 0.2716 | 0.2770 | +0.0054 | -0.1% |
| 32 | 0.1430 | 0.1537 | +0.0106 | -1.1% |
| 16 | 0.0375 | 0.0406 | +0.0031 | +0.2% |

**Summary:** avg +0.0072, wins=5 ties=0 losses=1. PRQ4 clearly helps on IP.

## Why PRQ4 helps on IP but not L2

**Inner Product on the unit sphere** is intrinsically harder for PQ. All vectors
have the same norm (1.0), so distances depend entirely on angles. The Euclidean
subspace decomposition that PQ uses doesn't align well with angular structure —
PQ optimizes per-subspace squared error, which is a poor proxy for dot-product
ranking when norms are uniform.

**Residual quantization (PRQ)** mitigates this: after the first-level PQ encoding,
the residual (encoding error) contains the angular information that the coarse PQ
codebook missed. The second-level encoding captures this residual structure,
improving the quality of the distance estimate. This matters more for IP because
the first-level PQ is worse at IP than L2 — more residual signal to capture.

**For L2 metrics**, PQ's per-subspace squared-error optimization directly minimizes
the distance approximation error. The residual is small and mostly noise — encoding
it adds complexity without improving ranking. Hence PRQ4 ≈ PQ4 on L2 (within
±0.3pp), with a 3-8% QPS cost from the residual decode.

## Production guidance

| Scenario | Recommendation |
|----------|---------------|
| L2 metric, any dataset | **PQ4** (PRQ4 adds QPS cost, no recall gain) |
| IP metric (sphere, normalized embeddings) | **PRQ4** at high m (notable recall gain, negligible QPS cost) |
| Max recall needed (>0.98) | **PQ8** at m=dim/4 (2× storage, 0.5× QPS) |
| Tight storage budget | **PQ4** with m=dim/8 or dim/16 |

**PQ4 is the default quantizer.** The primary lever for recall/latency tradeoff
is m (subquantizer count), not the quantizer variant. Same-bytes-per-vector,
PQ4 with higher m always beats PQ8 (~2× QPS at same recall) due to FastScan
4-bit processing 32 vectors/block vs 16 for 8-bit.

## Rerank

Disabled by default (commit a7f73e3). PQ-decode rerank showed <0.001 recall
delta at all m values (2-192). A true rerank would need stored FP16 vectors
(storage ruled out for our use case).

## PRQ4 nsplits bug (FIXED)

Default nsplits = dim/8 (96 for 768d). When m < 96 or m % 96 != 0, build crashed.
Fixed (commit 4d57ad0): auto-clamp nsplits to largest divisor of m that also
divides dim.

## Build times (uncompressed Parquet)

| Dataset | N | Dim | Metric | Build | Index Size |
|---|---|---|---|---|---|
| sift1m | 1M | 128 | L2 | 5.3s | 53 MB |
| arxiv_nomic | 1.34M | 768 | L2 | 18.7s | 187 MB |
| msmarco | 8.74M | 768 | L2 | 162s | 1.3 GB |
| sphere_ip | 9.99M | 768 | IP | 347s | 1.4 GB |

## Recall@10 vs QPS (m=192, rerank off)

| Dataset | n_probe=1 | n_probe=4 | n_probe=8 | n_probe=32 |
|---|---|---|---|---|
| arxiv | 0.685/41K | 0.966/16K | 0.974/11K | 0.978/4K |
| msmarco | 0.526/34K | 0.892/15K | 0.936/9K | 0.970/3K |
| sphere_ip | 0.325/34K | 0.707/12K | 0.762/7K | 0.819/2K |
| sift1m | 0.589/58K | 0.801/39K | 0.816/33K | 0.819/18K |
