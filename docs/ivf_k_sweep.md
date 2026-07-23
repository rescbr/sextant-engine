# IVF K Operating-Point Sweep

**Status:** Findings, 2026-07-23. arxiv-nomic 1.34M, c4a (Neoverse-V2), 8 threads.
**Sweep:** K ∈ {8, 16, 32, 64}, all built with R=64, L_build=128, pq_m=96/8-bit,
closure_factor=1.0330 (the estimator resolved consistently across this run, so
the sweep isolates K).

## Results (L=400, rr=2, pat=0 — the clean recall/QPS point)

| K  | np | routing recall@100 | realized recall | QPS  | oracle@np4 |
|----|----|--------------------|-----------------|------|------------|
| 8  | 4  | 0.9926             | 0.9500          | 788  | 0.9979     |
| 8  | 8  | 1.0000             | 0.9558          | 440  |            |
| 16 | 4  | 0.9752             | 0.9535          | 820  | 0.9897     |
| 16 | 8  | 0.9973             | 0.9713          | 471  |            |
| 32 | 4  | 0.9500             | 0.9422          | 855  | 0.9765     |
| 32 | 8  | 0.9912             | 0.9801          | 460  |            |
| 64 | 4  | 0.9168             | 0.9180          | 963  | 0.9533     |
| 64 | 8  | 0.9770             | 0.9759          | 490  |            |

**Merged baseline** (reference): 0.9084 @ 1754 QPS.

## The tradeoff curve

As K increases (finer partitioning, more/smaller shards):

| trend | np=4 | np=8 |
|-------|------|------|
| routing recall | drops (0.99 → 0.92) | drops (1.00 → 0.98) |
| realized recall | drops slightly (0.95 → 0.92) | rises (0.96 → 0.98) |
| QPS | rises (788 → 963) | roughly flat (440-490) |

**Why routing recall drops with K:** more centroids = more Voronoi cells =
more boundaries = more boundary ambiguity per probe. closure_factor covers less
when cells are small.

**Why realized recall at np=8 rises with K:** finer shards search better (the
shard graph is smaller and denser per-vector), and np=8 on K=64 probes 8 of 64
shards (12.5%) vs 8 of 8 (100%) at K=8. The diversity of probed regions
matters more than exhaustive coverage of few regions.

## Key insights

### 1. Routing recall ≠ realized recall
K=8 np=4: routing recall 0.9926 but realized only 0.9500. The gap is **shard
search quality**, not routing — the big shards (R=64, ~168K vectors each) have
more ground to cover and beam_search misses some NNs. At K=64 np=8, routing
recall 0.9770 ≈ realized 0.9759 — they match because finer shards search more
thoroughly. **The realized-recall limit shifts from shard-search-quality (coarse
K) to routing-coverage (fine K) as K grows.**

### 2. The recall ceiling is IVF's real advantage
Merged is hard-capped at ~0.9084 (PQ navigation ceiling on this dataset). Every
IVF K at np=8 exceeds this: 0.9558 (K=8) to 0.9801 (K=32). IVF's value
proposition is **recall the merged path structurally cannot reach**, at the cost
of QPS.

### 3. The QPS gap to merged persists across K
At realized recall ~0.92 (near merged's 0.9084), the best IVF QPS is ~963
(K=64 np=4) vs merged's 1754. The ~0.55× ratio is the n_probe multiplier on
beam_search cost — present at every K. **No K closes the QPS gap at matched
recall.** K trades routing recall against per-shard cost; it doesn't eliminate
the n_probe work multiplier.

### 4. K=16-32 is the practical operating range
- K=8: routing is great but shards are big and expensive; np=8 barely helps.
- K=64: cheap shards but poor routing recall at low np; needs np=8+ to reach
  the recall ceiling.
- K=16/32: the sweet spot where np=8 reaches recall 0.97+ at ~460-471 QPS.

## Distributed Sextant implication

K determines the **node count** (one shard per node). The K sweep directly
informs the distributed design:

- **K=8 (8 nodes):** each node holds ~168K vectors. Per-node index is large;
  routing puts most queries on 1-2 nodes (good locality). But recall ceiling at
  np=8 is only 0.9558, and scaling beyond 8 nodes requires repartitioning.
- **K=64 (64 nodes):** each node holds ~21K vectors. Small per-node indexes
  (fast search), high fan-out (more nodes contacted per query). Needs multi-probe
  to recover boundary recall at low dispatch fan-out.
- **K=16-32:** the likely production range. ~42K-84K vectors/node, np=8 reaches
  recall 0.97+, and multi-probe can drop the effective fan-out for unambiguous
  queries (most queries dispatch to 4-5 nodes, boundary queries to a few more).

**Multi-probe becomes the distributed dispatch policy:** the ratio determines
how many nodes a query fans out to. Unambiguous queries (deep in one Voronoi
cell) fan out to 1-2 nodes; boundary queries fan out to more. This is exactly
the request-routing pattern a distributed ANN system wants.

## Confound: estimator R-selection variability

This run resolved R=64/L_build=128 for all K. A prior K=32 run (the FHM
benchmark) resolved R=46/L_build=100 and got QPS 1185 (vs 855 here). The
estimator's R selection isn't fully stable across runs — likely sensitivity to
the mini-build sampling. The K=32 QPS difference (855 vs 1185) is largely the
R=64 vs R=46 difference (denser graphs = more neighbor evals). **Treat absolute
QPS as ±15%; the relative K trend is clean.** Worth stabilizing the estimator's
R selection if we commit to a production K.

## Reusing the sweep

The sweep script (`/tmp/k_sweep.sh` on the VM, run via profile-vm) builds each K
to `$SSD/ivf_K<K>.shards` and runs:
- `sextant_bench` at np=4 and np=8 (L=400, rr=2, pat=0)
- `route_quality` (closure-aware routing recall vs oracle)

To re-run with different graph params, pin R/L_build via explicit autobuild
flags. To add more K values, extend the loop.
