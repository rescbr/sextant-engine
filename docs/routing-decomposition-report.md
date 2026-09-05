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

### 4.1b The deployable bytes curve (i8 plane)

Global-scale int8 quantization of the PCA-128/64 projections (dbpedia-933K):

| structure | B/vec | @5% | @10% | @20% |
|---|---|---|---|---|
| centroid | 0 | 0.77 | 0.88 | 0.94 |
| anchors A=64 | ~1 | 0.84 | 0.92 | 0.97 |
| i8 PCA-64 | 64 | 0.83 | 0.97 | 0.99 |
| **i8 PCA-128** | **128** | **0.92** | **0.98** | **1.00** |
| fp32 PCA-128 | 512 | 0.94 | 0.99 | 1.00 |

i8 quantization costs ~2pp of containment at a quarter of the bytes.
End-to-end implication for the two-stage design: an i8@128 sweep costs
~0.15× a full code scan, so **0.98 recall at ~0.25× flat total
(sweep + f=0.10 stage-2)** versus probe-fraction's 0.99 at 0.5× —
roughly a 2× scan reduction at matched recall, with the 1.00@20%
ceiling point available.

### 4.1 Second corpus: arxiv-nomic (100K × 768, 25 leaves)

Containment @ 5/10/20/30%: centroid 0.55/0.73/0.89/0.95 · member-min
0.56/0.72/0.92/0.98 · PCA-128 0.54/0.70/0.90/0.99 · PCA-32
0.49/0.67/0.85/0.95 · anchors ≤ centroid (measured previously).

**There is no plane win on arxiv — because there is no gap**: the
centroid already sits at the content-aware ceiling. The dbpedia win
(0.77 → 0.94 @ 5%) is a property of that corpus's spectrum. This is the
adaptive-rank rule's justification in one comparison: the variance
target selects a corpus-appropriate plane (and, on arxiv-like data,
correctly indicates the plane buys ~nothing over centroid routing).
Corpus-dependence measured, not assumed.

### 4.3 Ten million vectors (cohere-10M, 2864 leaves) — coverage vs N

The scale question the 933K run could not answer: does containment at a
fixed *corpus fraction* degrade as N grows? It does not — it improves.

| ordering (B/vec) | 5% | 10% | 20% | (933K @5%) |
|---|---|---|---|---|
| centroid (0) | 0.893 | 0.956 | 0.987 | 0.77 |
| member-min (ideal) | 1.000 | 1.000 | 1.000 | 1.00 |
| i8 PCA-128 (128) | 0.914 | 0.960 | 0.992 | 0.92 (dbpedia) |
| i8 PCA-64 (64) | 0.832 | 0.930 | 0.984 | 0.83 |
| PCA-32 fp (engine proxy) | 0.776 | 0.906 | 0.968 | 0.74 |

Finer leaves make centroid ranking *more* discriminative (2864 vs 246
leaves), and the plane's edge over free centroids shrinks from ~15pp
(dbpedia) to +2pp @5% here. Engine-side sweep on the same tree confirms
the engine's PCA-32 routing sits on the exact-centroid curve at matched
coverage:

| np | probed frac | routing containment |
|---|---|---|
| 16 | 0.016 | 0.715 |
| 32 | 0.032 | 0.827 |
| 64 | 0.064 | 0.918 |
| 128 | 0.128 | 0.970 |
| 256 | 0.255 | 0.996 |

Delivered recall at np256 plateaus at 0.66 with routing at 0.996 — the
binding constraint past np64 is the scan/rerank side (W, quantizer,
adaptive-τ), not routing. Logs:
`gs://.../cohere/spike_routing_ceiling_10m.log`, `spike_engine_sweep_10m.log`.

## 5. Consequences for the architecture

1. **The current design is validated, not indicted**: centroid-family
   routing is within a few points of the best query-independent
   structure, and the scan + fraction + exact-rerank stack is the
   mechanism the decomposition says matters. At 10M this strengthens:
   centroid routing *improves* with leaf count (§4.3) and the engine
   tracks it at matched coverage through np=256 (0.996 routing).
2. **Leaf granularity is the free routing lever; the resident plane is
   cut at billion scale.** §4.3: finer leaves (246→2864) buy +12pp
   containment @5% at 0 B/vec, while the 128 B/vec plane's edge shrinks
   to +2pp — 128 GB DRAM at 1B for 2pp is not a trade, it is a
   rejection. The plane survives only as the mid-scale (250K–fewM,
   spectrally-gapped, NAND-tier) option, and §5 of the design doc now
   records it as a measured negative at scale.
3. **Rank must be adaptive** (explained-variance from the build
   reservoir), per F6 — a hard-coded plane rank repeats the 2.7σ
   mistake at larger stakes. (Now largely moot for 1B per (2).)
4. **Build-time leaf-count sizing rule** (new, from §4.3): pick
   k_root/depth so leaves stay fine enough that centroid routing holds
   containment ≥ target at probe_fraction f. Calibration points:
   246 leaves → 0.77 @5% (dbpedia-933K), 2864 leaves → 0.89 @5%
   (cohere-10M). One intermediate f-grid point (option 2) completes
   the curve; until then, ≥2K leaves at N ≥ 10M is the working rule.
5. Sequential probing (probe-then-decide using scan feedback) is the
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
