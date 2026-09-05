# What Bounds IVF Routing Recall: A Decomposition

*Technical report. Measurements: sextant engine + retrievalbench
harness; dbpedia OpenAI embeddings (933K × 1536, cosine, exact
brute-force ground truth) and arxiv-nomic (100K × 768). Reproduce with
`scripts/spike_routing_ceiling` (all columns in this report are
regenerable from one run per corpus).*

---

## 1. The question

An IVF-style index routes each query to a subset of its leaves; recall
is capped by *routing containment* — the probability that a true
neighbor's leaf is probed — before the scan and rerank stages even
matter. We shipped `probe_fraction` (a corpus-fraction probe budget)
and observed that recall tracked coverage fraction almost exactly:
f = 0.5 held 0.99 recall@10 across a 9.3× corpus growth, and the
relationship looked structural. This report asks what actually bounds
that curve: the data's cluster structure, the routing metric, or the
form of the leaf summary — and what, if anything, can beat it.

## 2. Method: measure leaf rankings, not end-to-end recall

For each query we rank all leaves under competing orderings and
measure, per GT@10 neighbor, whether *any* home leaf appears within a
page-weighted prefix fraction φ of the ordering (the same weight the
engine's leaf-coverage contract uses). This isolates routing from scan
and rerank, and the orderings enumerate the design space:

| ordering | what it stands for |
|---|---|
| random | the coverage baseline (what f buys with no routing signal) |
| exact centroid | L2 to the full-dim leaf mean — deployable, query-independent |
| anchors (A = 4..256) | sub-means from recursive 2-means per leaf — richer query-independent summaries |
| ball bound `d(q,c) − r` | admissible lower bound (cover-tree style) |
| partial-dim member | max IP over members in a contiguous subspace (128/384 dims) |
| **PCA-P member** | max IP over members projected to the top-P principal directions |
| member-min (oracle-ish) | distance to the nearest actual member — upper bound for any content-aware method |
| oracle | GT leaves first — harness self-check |
| engine points | (probed fraction, containment) at np = 1/2/4/8 |

One methodological note that cost us an hour and is worth recording:
the ground-truth row width (k = 100) can exceed the graded k (10);
indexing GT with the graded stride silently corrupts every query after
the first, and every ordering then reads as random.

## 3. Results (dbpedia 933K, 246 leaves)

Containment vs prefix fraction:

| φ | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|
| random | 0.08 | 0.15 | 0.29 | 0.39 |
| centroid | 0.77 | 0.88 | 0.94 | 0.97 |
| anchors A=4 / 16 / 64 / 256 | 0.79 / 0.81 / 0.84 / 0.83 | 0.88 / 0.92 / 0.92 / 0.92 | 0.96 / 0.97 / 0.97 / 0.97 | 0.98 / 0.99 / 0.99 / 0.99 |
| ball bound | 0.10 | 0.22 | 0.35 | 0.55 |
| contiguous 128-d / 384-d member | 0.83 / 0.84 | 0.92 / 0.94 | 0.97 / 0.99 | 0.99 / 1.00 |
| **PCA-32 / 48 / 64 / 96 / 128 member** | 0.74 / 0.80 / 0.83 / 0.92 / **0.94** | 0.89 / 0.92 / 0.97 / 0.99 / **0.99** | 0.97 / 0.98 / 1.00 / 1.00 / **1.00** | 0.99 / 0.99 / 1.00 / 1.00 / **1.00** |
| member-min (full-dim) | 1.00 | 1.00 | 1.00 | 1.00 |
| engine (PCA-32 centroids) | — | — | — | 0.92 @ φ=0.147 |

Engine points: φ = 0.020 → 0.56, 0.039 → 0.73, 0.075 → 0.84,
0.147 → 0.92.

## 4. Findings

**F1 — Structure is not the bound.** The oracle reaches 1.0 by φ = 0.10:
neighbors concentrate in ~25 of 246 leaves. "Recall tracks coverage" is
a property of our routing, not of the data.

**F2 — The engine sits on the centroid curve.** Its containment at
matched coverage is within a few points of exact full-dim leaf means;
the PCA-32 routing metric and the depth-2 hierarchy cost little. (We
could not A/B full-dim routing in the engine — `--pca-dims 0`
coerces to 32 at build — but the spike's centroid ordering bounds it.)

**F3 — Query-independent summaries saturate at ~0.84 @ 5%.** Sub-mean
anchors stop improving past A ≈ 64 (A = 256: 0.83). The mechanism:
containment is decided by whether a summary lands near the *query's*
neighbor; mode-sampling is a lottery, and more anchors buy linearly
fewer winning tickets. FPS-sampled member anchors are actively worse
(0.65 @ 5%) — peripheral points miss the dense mode the mean already
captures.

**F4 — Admissible bounds fail on sphere-shell data.** The ball bound
`d(q,c) − r` ranks *below random* (0.10 @ 5%): embedding clusters are
thin shells, radii are large and uniform, and the bound systematically
prioritizes diffuse leaves. Correct for pruning, anti-informative for
ranking.

**F5 — Content-aware ranking escapes the plateau, but only with rank.**
Contiguous subspaces (128-d, 384-d) land at the same 0.83–0.84 plateau
as summaries — projection noise inflates per-member maxima. PCA
projections escape it only with sufficient rank: 32 dims *worse* than
contiguous-384 (0.74 vs 0.84), 96–128 dims far better (0.92–0.94) at a
third of the bytes. Embedding variance concentrates in ~128 principal
directions; below that rank a subspace — however selected — is mostly
noise.

**F6 — The frontier is a bytes-vs-containment curve.** Reading F3 and
F5 together: ~0 B/vec (centroid) → 0.77; ~1 B/vec (anchors) → 0.84;
64 B/vec (PCA-32 fp16) → 0.74; 128 B/vec (PCA-64) → 0.83; 256 B/vec
(PCA-128) → 0.94; full-dim members → 1.00. (Second corpus pending;
the rank at which the PCA curve saturates is a property of the corpus
spectrum and should be selected adaptively — smallest P meeting an
explained-variance target — not hard-coded. The constant-128 hazard is
the same as the 2.7σ hazard.)

**F7 — What guards the last 6 points.** member-min's 1.00 @ 5% requires
full-dim evaluation of actual members per query — which is the scan.
The scan is not a workaround for weak routing; it *is* the only
content-aware mechanism at full rank. This bounds every IVF routing
proposal, including ours.

## 5. Consequences for the architecture

1. **The current design is validated, not indicted**: centroid-family
   routing is within a few points of the best query-independent
   structure, and the scan + fraction + exact-rerank stack is the
   mechanism the decomposition says matters.
2. **The one cheap upgrade with measured headroom is a resident PCA
   plane** at adaptive rank (96–128 on this corpus): stage 1 sweeps a
   128–256 B/vec resident projection ranking leaves by per-member max
   IP (containment 0.99 @ 10%), stage 2 fraction-scans the winners.
   Total ≈ 0.4× flat-scan at 0.99 recall vs probe-fraction's 0.5×
   (fp16 plane; an i8 plane halves stage 1). At billion scale, where
   the system is bytes-bound and the plane is the DRAM tier of a
   cold-store design, this is the compounding win.
3. **Rank must be adaptive** (explained-variance from the build
   reservoir), per F6 — a hard-coded plane rank repeats the 2.7σ
   mistake at larger stakes.
4. Sequential probing (probe-then-decide using scan feedback) is the
   one mechanism class not tested here; it sits between routing and
   scanning and is the natural next question.

## 6. Reproducing

```
# one tree, one GT, one run — every table in this report
sextant build-tree --input dbpedia_base.fbin --index dbpedia.tree \
    --quantizer local_scalar
spike_routing_ceiling dbpedia.tree dbpedia_base.fbin \
    dbpedia_query.fbin dbpedia_gt.gtmm 200 30
```

Spike: `scripts/spike_routing_ceiling.cpp`. Engine: `probe_fraction`
(fc13b47). Ground truth: exact brute force, normalized, IP, k=100
(`gs://<bench-bucket>/dbpedia_{100k,933k}_gt.gtmm`).
The A = 256 anchor column is slow to precompute (~25 min at 933K);
the `ANCHORS` array trims it.
