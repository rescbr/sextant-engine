# Adaptive Routing Plane — design document

Status: proposed (v1). Grounded in measurements from
[routing-decomposition-report.md](routing-decomposition-report.md);
verification milestones in §9. Owner: sextant engine.

---

## 1. Problem and opportunity

`probe_fraction` (fc13b47) made the probe budget a corpus fraction: recall
holds across corpus growth at ~f× flat-scan cost (0.99 @ f = 0.5). The
decomposition report shows where the remaining headroom lives:

- Query-independent leaf summaries (centroids, anchor sets) saturate at
  ~0.84 containment @ 5% coverage — the *cheap-structure frontier*.
- Content-aware ranking (per-member evaluation) reaches 0.99+ @ 10%,
  but only with sufficient PCA rank (96–128 on dbpedia; 32 is **worse**
  than contiguous subspaces).
- The frontier is a **bytes-vs-containment curve**: 0 B/vec → 0.77,
  ~1 B/vec → 0.84, 128 B/vec (PCA-64) → 0.83, 256 B/vec (PCA-128) →
  0.94, full member → 1.00.

Opportunity: a resident PCA plane at adaptive rank converts the 0.94
point into ~**2× scan reduction at matched recall** (0.99 @ total
~0.25–0.4× flat), and becomes the DRAM tier of a bytes-bound
cold-store architecture at billion scale.

Design principle carried throughout: **every constant is a measurement.**
Rank comes from the corpus spectrum; recall promises come from the
monitor; both are re-derived as the corpus moves. (Same principle as
the 2.7σ correction — see the blog draft §2.)

## 2. Architecture

Two-stage search replacing centroid routing:

```
stage 1 (resident):  sweep the plane — max over members, per leaf, of
                     query · projection(P rows) → leaf ranking
stage 2 (fraction):  existing scan path over the f-fraction winners
                     (dual-SDOT codes, τ-rerank, exact_rerank intact)
```

- **Plane layout**: per-row P×i8 projections (MEASURED: i8 costs ~2pp
  containment vs fp32 — report §4.2), row-id keyed, resident.
  i8@128 = 128 B/vec → containment 0.92/0.98/1.00 @ 5/10/20%; i8@64 =
  64 B/vec → 0.83/0.97/0.99.
- **Stage 1 kernel**: the plane is `N × P` contiguous — a plain SDOT
  sweep with a per-leaf max-reduction; i8 variant uses the existing
  dual-SDOT machinery. Stage-1 cost ≈ 0.15× the 4-bit code scan at i8@128
  (128 of 866 B/vec). Target operating point: 0.98 recall at
  ~0.25× flat total (sweep + f=0.10 stage-2) vs probe-fraction's
  0.99 at 0.5×.
- **Stage 2**: unchanged scan; `probe_fraction` is re-interpreted as
  the stage-2 cut over the stage-1 ranking. Hierarchy remains for
  filtered search and payload locality; it stops being the recall
  mechanism.
- Correctness posture: stage 1 is heuristic (containment, not
  guarantee); results remain exact-as-scanned; the τ-contract and
  exact rerank are untouched.

## 3. Adaptive rank

- **Rule**: smallest P with explained variance ≥ target (default
  ~0.90, configurable) on the current reservoir spectrum. dbpedia-1536
  lands at 96–128; **arxiv-768 measured: no plane gap at any rank**
  (centroid ≈ member-min ≈ plane) — the rule must also be allowed to
  answer "no plane" (variance target met at low P ⇒ centroid routing
  suffices), which the measured second corpus confirms matters.
- **Rank is a property of the data, re-derived at every basis
  refresh.** No hard-coded P anywhere in the serving path; the manifest
  records (P, basis version, variance curve).
- Hard-coded-rank is the 2.7σ mistake at larger stakes — the report's
  F6 exists to prevent it.

## 4. Mutation and lifecycle

Soundness: all plane state is derived, append-maintained, or heuristic.
Stale state can cost recall, never correctness.

- **Append**: project once (P·d flops) and append the plane entry in
  the same write ordering as codes + row-id. Leaf splits are
  transparent (entries are per-row; closure replication just means the
  row's projection serves each home leaf's max).
- **Delete**: tombstone + liveness bitmap, which stage 1 already
  consults. Physical reclamation of plane entries happens in
  vacuum/defrag passes that already rewrite rows.
- **Basis versioning**: the active basis serves while a refresh trains
  on the current reservoir; swap is atomic; rows re-project lazily
  during vacuum/defrag (no stop-the-world). Dual-version window: rows
  not yet re-projected are scored under the old basis and flagged —
  stage 1 tolerates the mixture because it ranks, not guarantees.
- **Drift signal (cheap, continuous)**: per-leaf mean residual
  ‖x − BBᵀx‖² updated incrementally on append. Rising residual = the
  variance curve is sliding = schedule refresh. This catches clustered
  ingest order (topic-by-topic loads) automatically.

## 5. Bootstrap (no seeding dataset required)

- **Phase 0 (N < threshold)**: no plane, no PCA. Full-dim fp16 centroid
  routing (requires fixing the `--pca-dims 0 → 32` coercion at
  `ivf_tree_index.cpp:499`); probe-all is the right answer at small N.
- **Phase 1 (first threshold)**: train basis + rank from the reservoir
  (at this N the reservoir may be the whole corpus — the best spectrum
  estimate it will ever get); project existing rows in one pass;
  version-swap; plane active.
- **Phase 2+**: §4 steady state.
- Early-basis bias from clustered ingest is absorbed by the drift
  signal firing refreshes during the first corpus-doubling; refreshes
  are cheap and versioned.

## 6. The recall monitor (first-class subsystem)

The estimator pattern promoted from build step to running service.

- **Tap**: log-sampled client queries + returned ids +
  `visited_leaf_pages` (already a production out-param) + τ-shortlist.
- **Shadow truth**: exact top-k on the reservoir snapshot for each
  sampled query. Budget-governed (≤0.5% of QPS, adaptive rate; CUSUM/
  EWMA change detection with sequential testing to avoid noise-chasing).
- **Per-stage attribution** (the decomposition, online): containment
  loss → basis/rank drift; scan loss → quantizer/assignment; rerank
  loss → W. Maintenance is targeted, not blanket.
- **SLA surface**: recall@k with live CI as an exported metric; breach
  → trigger (raise f, refresh basis, escalate).
- **Bridge to per-query budgets**: logged (query, per-stage outcome)
  pairs train a cheap predictor (e.g., stage-1 score-gap profile →
  needed coverage), enabling per-query f — the sequential-probing
  mechanism (report F7) arriving learned rather than hand-designed.
- **Cold start**: self-lookup queries (ingest vectors as queries) until
  real traffic exists; documented as a conservative lower bound.
- **Honest scaling limit**: exact-truth verification is exact up to
  ~10M rows; beyond, truth is computed against a bounded snapshot
  (deep-k truth degrades combinatorially at 1B — probability a given
  true neighbor is in a 2M-of-1B sample is ~0.002). Contract wording:
  *exact-truth-verified to N rows; snapshot-consistent beyond; proxy
  signals continuous.* No static recall claims.

## 7. Billion-scale posture

- Plane = DRAM tier (128–256 GB at 1B for P=128; i8/P=64 halves it)
  over cold codes. Append: one hot + one cold write, single ordering.
  Delete: one tombstone honored by both tiers. Basis refresh rewrites
  only the resident tier (minutes, shadow-versioned).
- Mutation story is structurally stronger than graph indexes
  (connectivity-critical inserts) or retraining IVF (O(N) reassign).
- Stage-1 bytes ARE the query cost — the plane makes routing an
  explicit, budgeted read.

## 8. Risks and open questions

| risk | status |
|---|---|
| rank curve on 2nd corpus (arxiv) | DONE: arxiv-768 has NO plane gap (centroid ≈ member-min ≈ plane) — adaptive rule correctly de-emphasizes the plane there; report §4.1 |
| i8 plane containment loss | MEASURED: ~2pp vs fp32 (0.92 vs 0.94 @5%; report §4.2) — i8@128 adopted as the layout |
| stage-1 sweep latency at small N | plane may lose to probe-all below ~250K; auto-fallback by N |
| dual-basis window complexity | bounded by vacuum cadence; measure mixture recall in prototype |
| monitor bias under adversarial workloads | per-tenant slicing; document |
| PCA-64 sweet spot | DONE: i8@64 0.83/0.97/0.99; i8@128 dominates at +64 B/vec |

## 9. Verification milestones

1. Spike: i8 plane + PCA-64 points on the bytes curve (dbpedia + arxiv).
2. Spike: basis-refresh simulation — train basis on first 20% of a
   shuffled-vs-clustered ingest, measure containment decay and refresh
   recovery (validates §4/§5 offline).
3. Engine prototype: plane blob at build; stage-1 in search; f as
   stage-2 cut; verify via the rb auto-row grid (recipe:
   probe-fraction-shipped memory; target 0.99 @ total ≤ 0.4× flat,
   stretch ≤ 0.25×).
4. Monitor MVP: tap + shadow truth + per-stage counters + one exported
   metric; drive a forced drift (biased appends) end-to-end and show
   detection → refresh → recovery.
5. Full sweep at 4 corpus sizes; update the report and blog §4.

Prerequisites: fix `--pca-dims 0` coercion (also unblocks phase 0 and
the engine-without-PCA measurement).
