# What Bounds IVF Routing Recall — Consolidated Technical Report

*The routing arc, 2026-09-03 → 2026-09-06. Measurements: sextant engine
+ retrievalbench harness + `scripts/spike_routing_ceiling`. Corpora:
dbpedia OpenAI (933K × 1536, cosine, exact brute-force GT), arxiv-nomic
(100K × 768), cohere-10M (10M × 768, unit-normalized, Zilliz). All
columns are regenerable from one spike run per corpus; artifacts and
logs on `gs://<bench-bucket>/`.*

---

## 0. Executive summary

We asked what bounds the recall of `probe_fraction` routing — the data,
the routing metric, or the leaf summaries — and enumerated the design
space with leaf-ranking decompositions at 100K, 933K, and 10M vectors,
across a 10× leaf-granularity range, three corpora, and one adaptive
mechanism. Answers, each measured:

1. **Structure is not the bound.** Oracle containment is 1.0 by φ=0.10
   on every corpus; neighbor leaves are few and findable. "Recall tracks
   coverage" is a property of our routing, not the data.
2. **The engine already sits on the achievable free-routing curve.**
   PCA-32 centroid routing tracks exact full-dim leaf means at matched
   coverage (within a few points at 933K; exactly, through np=256 at
   10M).
3. **The only routing lever that beats centroids is per-member content
   (a resident plane), and its value is corpus- and rank-dependent** —
   +15pp @5% on dbpedia, ~0 on arxiv, +2pp on cohere-10M. At billion
   scale it fails the DRAM budget in every case and is cut.
4. **Routing is invariant to leaf granularity** (1025→9888 leaves:
   identical containment) — it is determined by the root partition. Leaf
   size is a scan-economics knob with a mild optimum at the default.
5. **Containment at fixed corpus fraction does not degrade with N**;
   the apparent 933K→10M improvement is a corpus effect.
6. **Sequential (probe-then-stop-on-score) adaptivity buys nothing** at
   matched mean fraction — the score gap does not localize GT leaves
   beyond what the ranking encodes.
7. Past np≈64 at 10M, **the binding constraint is scan/rerank ordering,
   not routing** (routing 0.996 while no-rerank delivered recall sits at
   0.66; the same tree reaches 0.989 end-to-end with rerank + adaptive-τ).

Net architectural consequence: the shipped design (root-partition
routing + probe_fraction + scan + exact/adaptive rerank) is the right
family; remaining headroom lives in the scan, not the router.

## 1. The question

An IVF-style index routes each query to a subset of its leaves; recall
is capped by *routing containment* — the probability that a true
neighbor's leaf is probed — before scan and rerank even matter. We
shipped `probe_fraction` (53f9451: probe budget as a corpus fraction)
and observed recall track coverage almost exactly: f = 0.5 held 0.99
recall@10 across a 9.3× corpus growth. This report asks what bounds
that curve — data structure, routing metric, or leaf-summary form — and
what, if anything, can beat it. It is bounded above by the project's
objective function: recall, latency, DRAM, and excess storage per
dollar at 100M–1B vectors with a routing budget of ≈ ≤4 B/vec (codes
live on NAND).

## 2. Method: measure leaf rankings, not end-to-end recall

For each query we rank all leaves under competing orderings and
measure, per GT@10 neighbor, whether *any* home leaf appears within a
page-weighted prefix fraction φ of the ordering (the same weight the
engine's leaf-coverage contract uses). This isolates routing from scan
and rerank. The orderings enumerate the design space:

| ordering | what it stands for |
|---|---|
| random | the coverage baseline (what f buys with no routing signal) |
| exact centroid | L2 to the full-dim leaf mean — deployable, query-independent, 0 B/vec |
| anchors (A = 4..256) | sub-means from recursive 2-means per leaf — richer query-independent summaries |
| ball bound `d(q,c) − r` | admissible lower bound (cover-tree style) |
| partial-dim member | max IP over members in a contiguous subspace |
| **PCA-P member** | max IP over members projected to the top-P principal directions — the "plane" |
| member-min (oracle-ish) | distance to the nearest actual member — upper bound for any content-aware method |
| oracle | GT leaves first — harness self-check |
| engine points | (probed fraction, containment) at n_probe = 1..256 |

Spike infrastructure (evolved over the arc): the base corpus is
mmap'd (10M×768 does not fit the heap beside the plane arrays); every
O(N·d) loop is OpenMP leaf-parallel with SVE kernels; and member-min is
computed for all probe queries in ONE row-order pass over the file —
per-query leaf-ordered scans are random-4K page-fault storms, ~20×
slower. Batched output verified bit-identical to the serial original.

Methodological traps that cost real time and are worth recording:
GT row width (k=100) ≠ graded k (10) — wrong stride reads as random;
gtmm rows are id-block + dist-block, NOT element-interleaved; and gtest
failure greps must match two spaces (`FAILED  ]`) — use exit codes.

## 3. The routing frontier (dbpedia-933K, 246 leaves)

Containment vs prefix fraction φ:

| φ | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|
| random | 0.08 | 0.15 | 0.29 | 0.39 |
| centroid | 0.77 | 0.88 | 0.94 | 0.97 |
| anchors A=4 / 16 / 64 / 256 | 0.79 / 0.81 / 0.84 / 0.83 | 0.88 / 0.92 / 0.92 / 0.92 | 0.96 / 0.97 / 0.97 / 0.97 | 0.98 / 0.99 / 0.99 / 0.99 |
| ball bound | 0.10 | 0.22 | 0.35 | 0.55 |
| contiguous 128-d / 384-d member | 0.83 / 0.84 | 0.92 / 0.94 | 0.97 / 0.99 | 0.99 / 1.00 |
| PCA-32 / 48 / 64 / 96 / 128 member | 0.74 / 0.80 / 0.83 / 0.92 / **0.94** | 0.89 / 0.92 / 0.97 / 0.99 / **0.99** | 0.97 / 0.98 / 1.00 / 1.00 / **1.00** | 0.99 / 0.99 / 1.00 / 1.00 / **1.00** |
| member-min (full-dim) | 1.00 | 1.00 | 1.00 | 1.00 |
| engine (PCA-32 centroids) | φ = 0.020 → 0.56, 0.039 → 0.73, 0.075 → 0.84, 0.147 → 0.92 | | | |

**F1 — Structure is not the bound.** Oracle reaches 1.0 by φ = 0.10
(neighbors concentrate in ~25 of 246 leaves).

**F2 — The engine sits on the centroid curve.** Containment at matched
coverage is within a few points of exact full-dim leaf means; PCA-32
and the depth-2 hierarchy cost little.

**F3 — Query-independent summaries saturate at ~0.84 @ 5%.** Anchors
stop improving past A ≈ 64; containment depends on whether a summary
lands near the query's neighbor, and mode-sampling is a lottery. FPS
member-anchors are actively worse (0.65 @ 5%) — peripheral points miss
the dense mode the mean already captures.

**F4 — Admissible bounds fail on sphere-shell data.** The ball bound
ranks *below random*: embedding clusters are thin shells, radii are
large and uniform, and the bound systematically prioritizes diffuse
leaves. Correct for pruning, anti-informative for ranking.

## 4. The bytes curve (deployable planes)

Global i8 quantization of the PCA projections (dbpedia-933K):

| structure | B/vec | @5% | @10% | @20% |
|---|---|---|---|---|
| centroid | 0 | 0.77 | 0.88 | 0.94 |
| anchors A=64 | ~1 | 0.84 | 0.92 | 0.97 |
| i8 PCA-64 | 64 | 0.83 | 0.97 | 0.99 |
| **i8 PCA-128** | **128** | **0.92** | **0.98** | **1.00** |
| fp32 PCA-128 | 512 | 0.94 | 0.99 | 1.00 |

**F5 — Content-aware ranking escapes the summary plateau only with
rank.** Contiguous subspaces land at the same 0.83–0.84 plateau
(projection noise inflates per-member maxima); PCA escapes it only
above ~96 dims. Below that rank a subspace — however selected — is
mostly noise.

**F6 — The frontier is a bytes-vs-containment curve.** 0 B/vec → 0.77;
~1 B/vec → 0.84; 64 → 0.83; 128 → 0.92; 512 → 0.94; full-dim → 1.00.
i8 quantization costs ~2pp at a quarter of the bytes. The two-stage
implication (sweep + fraction scan): ~0.98 recall at ~0.25× flat total
vs probe-fraction's 0.99 at 0.5× — a 2× scan reduction *at mid scale*.

**F7 — What guards the last points.** member-min's 1.00 @ 5% requires
full-dim evaluation of actual members per query — which IS the scan.
This bounds every IVF routing proposal, including ours.

## 5. Corpus dependence (arxiv-nomic, 100K × 768, 25 leaves)

Containment @ 5/10/20/30%: centroid 0.55/0.73/0.89/0.95 · member-min
0.56/0.72/0.92/0.98 · PCA-128 0.54/0.70/0.90/0.99 · PCA-32
0.49/0.67/0.85/0.95. **There is no plane win on arxiv because there is
no gap** — the centroid already sits at the content-aware ceiling. The
dbpedia win is a property of that corpus's spectrum. The adaptive-rank
rule (smallest P meeting an explained-variance target) is justified in
this one comparison: it selects a corpus-appropriate plane, and on
arxiv-like data correctly indicates the plane buys ~nothing.

## 6. Ingest drift (milestone 2, dbpedia cluster-ordered ingest)

Cluster-ordered ingest (sorted by leading 4 PCA components), i8@128
containment @5/10/20% split by neighbor tercile in the ingest order:

| basis | early tercile (seen) | late tercile (unseen) |
|---|---|---|
| 2%-prefix (clustered ingest) | 0.89 / 0.94 / 0.95 | **0.67 / 0.83 / 0.91** |
| full-sample (≈ refresh) | 0.92 / 0.95 / 0.98 | **0.81 / 0.86 / 0.96** |

Clustered ingest costs **14pp @ 5% on unseen clusters** (~3pp on seen);
a full-sample refresh recovers it (0.67 → 0.81). member-min = 1.0 in
both terciles — geometry unchanged; the damage is purely the basis.
Homogeneous ingest drifts ~nil even with a 0.1%-prefix basis, so basis
versioning has a measured job precisely in the clustered case.

## 7. Scale: coverage vs N (cohere-10M, 2864 leaves)

Does containment at fixed *corpus fraction* degrade as N grows? No —
0.893/0.956/0.987 @ 5/10/20% for the centroid ordering. The comparison
against dbpedia's 0.77 @ 5% looks like a scale effect but is NOT (§8
proves it): it is a corpus effect.

| ordering (B/vec) | 5% | 10% | 20% |
|---|---|---|---|
| centroid (0) | 0.893 | 0.956 | 0.987 |
| member-min (ideal) | 1.000 | 1.000 | 1.000 |
| i8 PCA-128 (128) | 0.914 | 0.960 | 0.992 |
| i8 PCA-64 (64) | 0.832 | 0.930 | 0.984 |
| PCA-32 fp (engine proxy) | 0.776 | 0.906 | 0.968 |

Engine sweep on the same tree (PCA-32 routing, 1000 queries):

| n_probe | probed frac | routing containment | delivered (no rerank) |
|---|---|---|---|
| 16 | 0.016 | 0.715 | 0.536 |
| 32 | 0.032 | 0.827 | 0.593 |
| 64 | 0.064 | 0.918 | 0.634 |
| 128 | 0.128 | 0.970 | 0.652 |
| 256 | 0.255 | 0.996 | 0.661 |

## 8. Granularity invariance (three trees, same corpus)

cohere-10M, k_root=1024, leaf-capacity 20000 / auto(5000) / 1250:

| leaves | avg vectors/leaf | centroid @5/10/20% | engine routing @np64 |
|---|---|---|---|
| 1025 | 9756 | 0.892/0.952/0.986 | 0.917 @6.4% probed |
| 2864 | 3492 | 0.893/0.956/0.987 | 0.918 @6.4% |
| 9888 | 1011 | 0.893/0.952/0.987 | 0.917 @6.4% |

**Containment at fixed corpus fraction is INVARIANT across a 10× leaf
count range.** Routing is the ROOT partition (identical k-means at
k_root=1024); leaf capacity only changes scan granularity. End-to-end
(`tree-search`, probe-fraction grid, adaptive-w-gap 2.0, recall@10 at
f = 0.05/0.10/0.20/0.50):

| tree | recall@10 | QPS @f=0.10 | QPS @f=0.50 |
|---|---|---|---|
| 1025 leaves | 0.878/0.937/0.965/0.973 | 56.8 (8t) | 11.6 (8t) |
| 2864 leaves | 0.881/0.948/0.979/0.989 | 28.5 (4t) | 5.4 (4t) |
| 9888 leaves | 0.879/0.938/0.965/0.973 | 25.5 (4t) | 4.6 (4t) |

Leaf size is a scan-economics knob: recall within ~1.5pp across the
range, mild optimum at the ~3.5K vectors/leaf default, and finer leaves
cost QPS. Corollaries: (a) the dbpedia-vs-cohere containment gap is
corpus, not scale or granularity; (b) `n_probe_ln` must scale with
leaves-per-root-child — a stale ln=8 made the 9888-leaf tree look 17pp
worse until corrected to 64.

## 9. Sequential (adaptive) probing — measured, negative

The one untried mechanism family: probe in score order and stop on an
observable signal, rather than committing f up front. Rule studied:
probe leaves in descending i8-128 plane score, stop when the next
leaf's score < α × best score seen (the stage-1 sweep score is itself
observable — no GT leakage; GT only scores outcomes). cohere-10M,
1000 queries, fixed-fraction control on the SAME ordering:

| rule | mean fraction | containment |
|---|---|---|
| α=0.80 | 0.014 | 0.745 |
| α=0.70 | 0.052 | 0.926 |
| α=0.60 | 0.152 | 0.987 |
| fixed 0.05 | 0.050 | 0.937 |
| fixed 0.10 | 0.100 | 0.975 |
| fixed 0.20 | 0.200 | 0.995 |

At matched mean fraction, score-relative stopping is a wash or slightly
worse — the score gap does not localize GT leaves beyond what the
ranking already encodes. The untested remainder of the family is
scan-feedback stopping (stop when the reranked k-th result stops
improving), which needs real per-leaf scans.

## 10. Scan-side reconciliation

The spike's `delivered` column (0.661 @ np256 with routing at 0.996)
is the **4-bit PQ-ordering ceiling**, not a scan defect: the spike's
SearchConfig leaves `rerank` off, and `tree-search --no-rerank`
reproduces 0.6611 bit-identically. With rerank + adaptive-w-gap 2.0 the
same tree reaches **0.989 recall@10 @ f=0.50** end-to-end. Past np≈64,
the binding constraint is scan/rerank ordering (W, quantizer, adaptive
τ) — routing is effectively solved at 25% probed fraction.

## 11. Consequences for the architecture

1. **The shipped design is validated, not indicted.** Centroid-family
   routing is within a few points of the best query-independent
   structure; the engine's PCA-32 routing tracks the exact-centroid
   curve at matched coverage through np=256.
2. **The resident plane is cut at billion scale.** Its edge is
   corpus-dependent (+15pp dbpedia, ~0 arxiv, +2pp cohere-10M @5%) and
   128 B/vec = 128 GB DRAM at 1B against a ≤4 B/vec budget — a
   rejection on budget grounds regardless of corpus. It survives only
   as the mid-scale (250K–fewM, spectrally-gapped, NAND-tier) option
   with the adaptive-rank rule (F5/F6).
3. **Leaf size is a scan-economics knob.** Keep the default capacity
   (~3.5–5K vectors/leaf); size `n_probe_ln` to leaves-per-root-child;
   expect routing containment to be a property of corpus + k_root,
   invariant from 1K to 10K leaves.
4. **Drift has a measured address**: clustered ingest degrades
   plane/basis containment by 14pp @5% on unseen clusters; full-sample
   refresh recovers it; homogeneous ingest needs no early versioning.
5. **Remaining headroom is in the scan, not the router**: delivered
   recall at fixed routing (0.66 no-rerank → 0.989 with adaptive
   rerank) dwarfs any remaining routing gain (≤2pp).

## 12. Reproducing

```
# canonical dbpedia triple (regenerated 2026-09-06; the original GT was
# in harness row order and is quarantined)
#   gs://.../dbpedia_933k_{base.fbin, gt_v2.gtmm, tree_v2.tree}
spike_routing_ceiling dbpedia_933k_tree_v2.tree dbpedia_933k_base.fbin \
    dbpedia_933k_query.fbin dbpedia_933k_gt_v2.gtmm 200 20

# 10M + granularity + adaptive (env: SPIKE_NO_ANCHORS=1,
# SPIKE_NO_BOUND=1, SPIKE_ADAPTIVE=1, OMP_NUM_THREADS=8)
spike_routing_ceiling cohere_10m_shape{,_cap1250,_cap20000}.tree \
    cohere_10m_norm.fbin cohere_10m_query_norm.fbin cohere_10m_gt.gtmm \
    1000 50

# end-to-end f-grid
sextant tree-search --index <tree> --query <q> --ground-truth <gt> \
    --probe-fraction 0.05,0.10,0.20,0.50 --adaptive-w-gap 2.0 --threads 4
```

Artifacts (GCS, `gs://<bench-bucket>/`):
`cohere/spike_routing_ceiling_10m.log`, `spike_engine_sweep_10m.log`,
`spike_adaptive_10m.log`, `dbpedia_933k_gt_v2.gtmm`,
`dbpedia_933k_tree_v2.tree`, `spike_dbpedia_v2.log`, clustered-ingest
perm/invperm/base + README. Trees for the granularity sweep on the VM
at `/mnt/ssd/data/cohere10m/`.

*Spike: `scripts/spike_routing_ceiling.cpp` (mmap + OpenMP + batched
member-min; `SPIKE_NO_ANCHORS`, `SPIKE_NO_BOUND`, `SPIKE_ADAPTIVE`,
basis-window knobs). Engine: `probe_fraction` (53f9451), rerank +
`adaptive_w_gap`. During the arc the same spike work also surfaced and
fixed three engine bugs (pca_dims=0 zero-dim k-means; PageAllocator
stale free-list pointers; stale-mmap parent scan) and a ~1.5× build
throughput win (double-buffered source prefetch + `--chunk-vectors`) —
see git log cda348a, e5f46a4.*
