# Session Report: 2026-07-31

## Overview

This session focused on build pipeline optimization, flag defaults, and
the critical combined benchmark (task #1). Five commits shipped.

## Commits

| Commit | Description |
|--------|-------------|
| `8bf0036` | fix: respect --threads in PQ training + skip entry-point k-means for scan path |
| `2cd8770` | feat: evidence-backed default flags from benchmark analysis |
| `4e1cf76` | fix: estimator not propagating quantizer_type/prq params to build |

## 1. PQ Training Thread Fix

**Problem:** `PqQuantizer::train()` and `ProductResidualQuantizer::train()`
used `std::thread::hardware_concurrency()` for thread count, ignoring
`--threads`. With `--threads 4` on a 10-core machine, PQ training spawned
10 threads (oversubscription). On c4a inside an already-parallel build
context, this caused 8+8=16 threads on 8 cores.

**Fix:** Added `set_num_threads()` to PqQuantizer. Wired from
`pass1_sample_and_train` and `build_ivf_scan`. Also parallelized
`build_cross_distance_table()` across subspaces (was the ~14s serial
post-PQ-training stretch).

**Verified via `sample`:** `--threads 4` → exactly 4 worker threads
(396% CPU), down from 10 threads (367% CPU).

## 2. Entry-Point K-means Eliminated for Scan Path

**Problem:** `pass1_sample_and_train` computed entry-point centroids via
serial k-means on 256k FP32 samples (~14s at 100% CPU). The scan path
called this via `prepare_routing_codes` but never used the results
(entry_centroids_ is only consumed by graph-path sidecars).

**Fix:** Added `compute_entry_points` flag to `pass1_sample_and_train`.
Scan path passes `false`. For the graph path, parallelized the k-means
dist/assign loops and switched double→simd::l2sq_f32.

**Result:** The ~14s serial stretch is completely gone from scan builds.

## 3. Adaptive Sub-shard Auto-Tuning (Task #2)

**Auto-threshold** (`sub_shard_threshold=0`):
- Computes `threshold = max(2048, mean_shard_size / 4)` from partition output
- Only enables if `max_shard >= 2 × threshold`

**Auto-sub_np** (probe percentage):
- Manifest stores `sub_probe_pct` (0-100)
- Search computes `sub_np = max(1, ceil(n_sub × pct / 100))` per shard
- Default 50% (probe half the sub-shards)

**Removed all backward-compat manifest fallbacks** (format not stabilized).

## 4. Evidence-Backed Default Flags

Four changes from `docs/flag_analysis.md`:

| Flag | Old | New | Evidence |
|------|-----|-----|----------|
| partition_balance_factor | 0 | 4 | SPANN λ; zero search-time cost |
| closure_epsilon | 0 (ratio) | -1 (auto) | K-robust absolute margin |
| n_probe_default | K/4 | 2·√K | np∝√K law (3 datasets) |
| prq_nsplits auto | dim/32 | dim/8 | Measured best M_sub=2 |

PQ remains default quantizer (PRQ adds 18% build overhead, zero recall
gain on low-LID text). PRQ encode optimization tracked as future work.

## 5. Estimator Quantizer Propagation Bug

**Problem:** The estimator's `finalize_` function did not propagate
`quantizer_type`, `prq_nsplits`, `prq_beam_size`, etc. from resolved
params to the returned `ResolvedParams`. This caused `autobuild
--quantizer prq` to silently fall back to plain PQ.

**Found during task #1** — the build log showed "training 4-bit PQ"
instead of "training PRQ". Fixed by adding the missing propagations.

## 6. Task #1: Combined PRQ + Sub-shards + Panorama Benchmark

**Dataset:** Sphere IP 10M (high-LID, hardest dataset)
**Build:** PRQ(nsplits=96, beam=5) + auto sub-shards + K=1024
**Build time:** 2961s (~49 min) on c4a-standard-8-lssd

### Build commands
```
sextant autobuild \
  --input sphere_ip_base.fbin --index sextant_combined \
  --metric ip --partition-count 1024 \
  --quantizer prq --prq-nsplits 96 --prq-beam-size 5 \
  --sub-shard-threshold 0 --threads 8
```

### Benchmark commands
```
sextant_bench \
  --index=sextant_combined \
  --queries=sphere_ip_query.fbin \
  --base-data=sphere_ip_base.fbin \
  --ground-truth=sphere_ip_gt.gt \
  --topk=10 --threads=8 --rerank=10 \
  --fastscan-w=1000 --sub-shard-n-probe=100 --n-probe=96
```

### Results

| Config | Recall@10 | QPS |
|--------|-----------|-----|
| np64 W300 sub50% (auto default) | 0.449 | 541 |
| np64 W300 sub3 | 0.466 | 556 |
| np64 W300 sub5 | 0.580 | 357 |
| np64 W300 sub100 (scan all) | 0.630 | 265 |
| np64 W1000 sub100 | 0.648 | 72 |
| np64 W1000 sub100 pan8 | 0.648 | 70 |
| np96 W1000 sub100 | 0.686 | 46 |
| np96 W1000 sub100 pan8 | 0.686 | 46 |

### Key findings

1. **Panorama is 0% gain at 10M, even at W=1000.** The session hypothesis
   (Panorama helps when rerank becomes 10-15% of cycles at high W) is
   **disproved**. Even at W=1000, rerank is <5% of cycles on the scan path.
   Panorama's rerank speedup is invisible.

2. **Sub-shard probe=50% is too aggressive** for high-LID data. It costs
   18pp recall vs scan-all (0.449 vs 0.630). The auto default should
   probably be higher (75-100%) for high-LID datasets.

3. **Sub-shards are SLOWER than flat at 10M Sphere.** Best op-point
   (np96 W1000 sub100: 0.686 @ 46 QPS) is 10× slower than prior flat PQ
   (np32: 0.682 @ 453 QPS). The replication overhead from closure=1.05
   exceeds the scan savings.

4. **PRQ doesn't help here.** The recall bottleneck is routing coverage
   (np), not quantization quality. At np=64, PRQ recall (0.648) ≈ PQ
   recall (prior 0.682 at np=32). The 4-bit PQ quantization is not the
   limiting factor.

### Conclusion

The combined PRQ + sub-shards + Panorama path does NOT improve over flat
PQ at 10M Sphere. The path to higher recall is **higher np** (routing
coverage), not better quantization or sub-shard routing. Panorama can be
considered dead for the scan path at this scale.

## Next Steps

- The auto sub_probe_pct=50% default is too low for high-LID; consider
  auto-tuning based on LID or raising to 75%
- Sub-shard replication overhead needs investigation (closure=1.05 may
  be too aggressive at K=1024)
- The main lever for recall is np, not quantization or rerank
- Python rewrite of gcp_bench.sh (design ready in docs/gcp_bench_rewrite.md)
- PRQ encode path optimization (18% overhead, tracked in memory)
