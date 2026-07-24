# Plan: Per-Tier Metric Selection + ScaNN Anisotropic PQ

**Status:** Planned, 2026-07-24. Not started.
**Supersedes** `docs/anisotropic_rearchitecture.md` and the standalone
`docs/plans/anisotropic_pq_plan.md` (now Phase 2 below).

This is a two-phase plan. **Phase 1** (per-tier metric selection) is a
standalone low-risk optimization + measurement PR that ships first.
**Phase 2** (ScaNN anisotropic PQ) is the research bet that builds on Phase 1's
baseline and plumbing.

---

# Phase 1: Per-Tier Metric Selection (standalone PR)

## Motivation

For **L2-normalized data** (arxiv-nomic is normalized), the L2sq and IP metrics
are rank-equivalent on *true* distances: `‖q−x‖² = 2 − 2⟨q,x⟩` when `‖q‖=‖x‖=1`.
This creates an opportunity we have never exploited:

1. **Cost:** IP is cheaper than naive L2sq per distance eval
   (IP: 1 FMA/vector-width; naive L2sq: 1 SUB + 1 FMA/vector-width).
   Our L2sq is naive everywhere — no norm precomputation.
2. **Recall (PQ-ADC tier only):** The PQ-ADC estimator operates on the quantized
   reconstruction `x̃`, which is NOT normalized. L2sq-ADC has a `‖x̃‖²` per-code
   noise term (`‖q−x̃‖² = ‖q‖² − 2⟨q,x̃⟩ + ‖x̃‖²`); IP-ADC doesn't. IP-ADC may
   already beat our 0.73 recall ceiling without any anisotropic training.

Every recall number we have (the 0.73 ceiling, IVF curves, OPQ/anisotropic
experiments) used L2sq-ADC. **IP was plumbed in Phase 1 of the engine
(`design_decisions.md:871`) but never benchmarked on arxiv-nomic.** We don't
know the IP-ADC recall, and we don't know the QPS effect of switching the FP16
tier to IP. Phase 1 answers both.

## Goal

Make the distance metric a first-class per-tier choice, add a decomposed-L2sq
option, and measure the full Pareto frontier on arxiv-nomic. Ship the cheapest
valid configuration as the default for normalized data.

**Why ship this before anisotropic:**
- Raises the baseline QPS/recall we measure anisotropic against.
- Validates the `dot_f16` plumbing that anisotropic will need anyway.
- Answers "is IP actually cheaper here?" empirically.
- Low-risk, independent of the research bet — we get the win regardless.

## The three metric options

| Option | Per-eval cost | Notes |
|--------|--------------|-------|
| **L2sq** (current) | 1 SUB + 1 FMA per vector-width | Naive subtract-then-square. What we ship today. |
| **IP** | 1 FMA per vector-width | Cheapest. Valid ranking for normalized data (true dist) and for PQ-ADC (no `‖x̃‖²` noise). |
| **Decomposed L2sq** | 1 dot + 1 add per entry | `‖q−c‖² = ‖q‖² − 2⟨q,c⟩ + ‖c‖²` with `‖q‖²` per query, `‖c‖²` per centroid at training. L2sq semantics at IP cost. Useful where L2sq ranking is wanted but cost matters. |

## Per-tier applicability

| Tier | What it computes | Valid metrics | Storage cost for decomposed-L2sq |
|------|-----------------|---------------|----------------------------------|
| **PQ-ADC** (nav, 27%) | Approximate dist via quantized `x̃` | L2sq, IP, Decomposed-L2sq | `‖c‖²` per centroid: K=256 floats/subspace — trivial, fits in codebook. |
| **MemGraph FP16 ball** (~12%) | True dist on actual FP16 vectors | L2sq, IP | `‖x‖²` per base vector: 4-8 B/vec at billion scale — expensive, **disqualified**. |
| **Routing** (IVF centroid) | True dist to centroids | L2sq, IP | Centroid norms trivial. |
| **Rerank** (final top-k, 12%) | True dist on FP32 base | L2sq, IP | Base-vector norms expensive at scale, **disqualified**. |

So: decomposed-L2sq is only interesting for PQ-ADC and routing. FP16/rerank get
L2sq or IP (binary choice).

## The measurement matrix

Five configurations on arxiv-nomic 1.34M (normalized):

| Config | PQ-ADC | FP16/rerank | Purpose |
|--------|--------|-------------|---------|
| **A** (baseline) | L2sq | L2sq-FHM | Current 0.73 reference |
| **B** | IP | L2sq-FHM | Isolates PQ-IP recall effect |
| **C** | IP | IP-FHM | Full IP — shipping candidate for normalized data |
| **D** | Decomposed-L2sq | L2sq-FHM | L2sq semantics at IP cost (PQ tier) |
| **E** | Decomposed-L2sq | IP-FHM | Max-cheap L2sq-semantics path |

**What we learn:**
- A vs B: recall effect of PQ-ADC metric alone.
- B vs C: QPS effect of FP16 tier metric (recall identical for normalized data).
- B vs D: recall of IP vs decomposed-L2sq at same PQ cost.
- A vs D: sanity — decomposed-L2sq should match plain L2sq recall (both L2sq semantics).

**Shipping decision:** for normalized data, pick the config with best QPS at
target recall. Likely winner: **C (IP/IP-FHM)** if IP-ADC recall ≥ L2sq-ADC
recall; otherwise **E** for L2sq semantics at IP cost. Default stays A for
non-normalized data.

## Phase 1 sequencing

### P1.1: Wire `dot_f16` with FHM (FP16 tier)
Add `dot_f16` to `vamana_core.hpp` as the counterpart to `l2sq_f16`, using
NumKong's `nk_dot_f16_neonfhm` (`third_party/numkong/include/numkong/dot/neonfhm.h:80`).
Same header-only inline + target-attribute pattern as `l2sq_f16_fhm_`.
**Gate:** unit test: `dot_f16` matches `dot_f32` to FP16 precision.

### P1.2: Thread metric through FP16 tier call sites
The FP16 tier is currently hardcoded to `l2sq_f16` at: `vamana_core.cpp:284,676,638,891`,
`ivf_searcher.cpp:112,183`, `estimator.cpp:252`, plus occlusion checks in
`builder.cpp`. Thread the configured metric through these call sites so they
dispatch to `l2sq_f16` or `dot_f16` based on config.

**Scope refined during implementation:**
- **Search-time FP16 (threaded):** `vamana_core.cpp` dist_to lambda (MemGraph
  ball, FP32 build path) + batched FP16 eval; `ivf_searcher.cpp` routing (both
  centroid and sub-centroid). These go through the new `dist_f16`/`dist_f32`
  dispatch helpers driven by `quantizer.metric()`.
- **Build-time FP16 (left as L2sq):** `builder.cpp` prune/medoid and
  `vamana_core.cpp:838,891` (`robust_prune_into`). Build metric is always L2sq
  per Neyshabur-Srebro — graph construction optimizes L2sq geometry regardless
  of search metric. Threading here would add complexity without benefit.
- **Estimator rerank (left as L2sq):** `estimator.cpp:252` is a measurement
  utility (R/alpha selection), not production search. L2sq vs IP is
  rank-equivalent for normalized data; leave as L2sq.
- **Benchmark rerank (DONE — threaded):** `benchmark.cpp:186` (the production
  rerank tool). Added `dot_simd` and `rerank_dist(metric, ...)` dispatch helper.
  `RerankCtx` gains a `metric` field, populated from the loaded index's
  quantizer (`idx->quantizer->metric()` for single-index; first present shard
  for IVF — they share the trained quantizer). All four rerank distance sites
  (rerank-with-base-in-RAM, rerank-with-file-read, no-rerank top-k scoring, and
  proximity `d_target`) route through `rerank_dist`. Critical: the rerank is
  ~12% of measured query latency. Leaving it on L2sq while the engine runs IP
  would understate the IP QPS win and measure something that doesn't match a
  real production consumer (which reranks by IP in IP mode). For L2-normalized
  data both metrics produce the same final top-k row IDs, so recall is correct
  under either — but the *cost* of the rerank differs and affects the QPS number.
**Gate:** existing L2sq tests unchanged; new IP path passes equivalent recall
on normalized data.

### P1.3: Decomposed-L2sq in PQ-ADC
Extend `MetricKind` with `DecomposedL2Sq`. In `preprocess_query`, handle it:
precompute `‖q‖²` once per query, look up `‖c‖²` per centroid (stored in
codebook at training), compute `−2⟨q,c⟩ + ‖q‖² + ‖c‖²` per entry. Store
per-centroid norms in the codebook layout.
**Gate:** decomposed-L2sq recall ≈ plain-L2sq recall (sanity check on arxiv100k).

### P1.4: Verify `preprocess_query` has no scalar fallback
Microbench `dot_f32` and `l2sq_f32` at sub_dim=8 — at m×K = 24,576 tiny NumKong
calls, dispatch overhead may dominate. If significant, hoist to one `nk_dot_f32`
per subspace × all K centroids (matvec: K×8 codebook slice × 8-dim query-sub).
Same optimization applies to L2sq path.
**Gate:** preprocess_query IP within 10% of L2sq throughput.

### P1.5: Measurement matrix on c4a
Run configs A-E on arxiv-nomic 1.34M. Record recall@100 and QPS at multiple
rerank values. Document results in `docs/metric_per_tier_results.md`.
**Gate:** identify the winning config for normalized data. Update default if
config C or E dominates A on the Pareto curve.

## Phase 1 success criteria

- **Recall:** config B (PQ-IP) recall measured. If > 0.73, the metric switch
  alone lifts the ceiling — re-scope Phase 2's anisotropic target accordingly.
- **QPS:** config C (full IP) QPS measured vs A. Expected: small win (1-3%)
  from one fewer SUB per distance eval in the FP16 tier.
- **Plumbing:** `dot_f16`-FHM wired and validated; ready for Phase 2.
- **Default:** if C dominates A on Pareto for normalized data, ship C as the
  default (keep A as fallback for non-normalized data via `--metric l2sq`).

## Phase 1 estimated effort

- P1.1 (dot_f16): 0.5 day. NumKong already has the kernel.
- P1.2 (thread metric): 1 day. Mechanical, many call sites.
- P1.3 (decomposed-L2sq): 1-2 days. Codebook norm storage + preprocess change.
- P1.4 (microbench): 0.5 day. May trigger P1.4b (matvec hoist): +1 day if needed.
- P1.5 (c4a measurement): 1 day. Run matrix, document.

**Total: ~4-6 days standalone.**

## Phase 1 risks

- **Non-normalized data correctness:** IP and decomposed-L2sq rely on the
  `‖q−x‖² = 2 − 2⟨q,x⟩` rank equivalence, which requires `‖q‖=‖x‖=1`. The
  engine already assumes normalization (Neyshabur-Sreiro build decision). Keep
  L2sq as the default for any path that doesn't assert normalization. Document
  the assumption in the metric enum doc.
- **FP16 IP precision:** `dot_f16` accumulates in FHM (FP16 mul, FP32 acc).
  Should match `l2sq_f16` precision. Verify on the c4a measurement.

---

# Phase 2: ScaNN Anisotropic PQ (the research)

**Builds on Phase 1.** Phase 1 established the IP baseline (config B/C recall)
and validated the `dot_f16` plumbing. Phase 2's anisotropic training targets the
gap between Phase 1's IP-ADC recall and ScaNN's ~0.90.

## Mission

**Sextant: measuring to the stars, navigating from reality.**

The stars: ScaNN at 1049 QPS @ recall@100 ≥ 0.90 on arxiv-nomic (VIBE benchmark,
in-memory, 384GB RAM, single-core AVX-512). That's the recall ceiling reference.

Our reality: c4a-standard-8-lssd spot at **$0.08/hr**, 8 ARM cores, 30GB RAM,
local SSD. Disk-resident Vamana graph + PQ codes. At 1B scale the index is
~230GB — SSD-resident regardless of quantizer choice. We are not matching ScaNN's
in-memory QPS. We are raising our recall ceiling to cut the rerank tax that
dominates disk-resident throughput.

### Phase 1 finding (P1.5, measured): plain PQ + L2sq-ADC dominates plain PQ + IP-ADC

On arxiv-nomic 1.34M at production scale (c4a, R=32, pq_m=96/bits=8):
- L2sq-ADC ceiling: 0.9940 (with rerank=10).
- IP-ADC ceiling: 0.5629. Cannot reach 0.90 at any config.
- IP is 1.2-2.0× faster per eval but the recall gap dominates on the Pareto curve.

The IP-ADC recall collapse traces entirely to the `‖x̃‖²` per-code variance
(noise term in `‖q−x̃‖² = ‖q‖² − 2⟨q,x̃⟩ + ‖x̃‖²` that IP-ADC omits). On
normalized data at scale, this noise scrambles the top-100 ranking.

### Phase 2 hypothesis: anisotropic training asymmetrically helps IP-ADC

The `‖x̃‖²` variance is not fixed — it depends on the training objective.
Standard k-means minimizes reconstruction MSE isotropically, which leaves
`‖x̃‖²` free to vary. **ScaNN's anisotropic objective weights parallel
quantization error higher** (parallel = along the query direction). This is
exactly the error component that generates `‖x̃‖²` drift. Hypothesis:
anisotropic training tightens `‖x̃‖²` enough that IP-ADC recall matches
L2sq-ADC recall — at which point IP's per-eval cost advantage (1.4-1.6× QPS)
wins on the Pareto curve.

This requires measuring a 2×2 matrix:

|                  | L2sq-ADC         | IP-ADC           |
|------------------|------------------|------------------|
| **Plain PQ**     | A: 0.99 (P1.5)   | B: 0.56 (P1.5)   |
| **Anisotropic PQ** | C: ?           | D: ? (ScaNN)     |

The Phase 2 prize is **D vs C on the Pareto curve** (with A as baseline).
Three outcomes:
- D dominates C → IP wins (anisotropic closes the recall gap, IP's cost wins).
- C dominates D → L2sq wins even with anisotropic training.
- Recall tie → IP wins by cost.

Phase 2 MUST measure both C and D, not just D. The original plan assumed
"anisotropic PQ uses IP-ADC" — but P1.5 showed that's not the right default
for plain PQ, so the anisotropic effect on L2sq-ADC must be measured too.

## The reframe (correcting the prior scope)

`docs/anisotropic_rearchitecture.md` had two premises the research and math
disproved:

1. **"ScaNN uses a single full-vector codebook"** — FALSE. ScaNN uses per-subspace
   Product Quantization (m = dim/2 subspaces of 2-dim each) with an anisotropic
   training objective. The full-vector VQ is only the pedagogical derivation in
   §4 of the paper. The production "Asymmetric Hashing (AH)" stage is PQ.

2. **"Code-size reduction is the billion-scale prize"** — WEAK. At 1B vectors
   (dim=768, R=32): graph=128GB, codes@96B=96GB, meta=8GB. Total=232GB. The
   graph dominates. The real SSD-bandwidth lever is **rerank reads**: at
   rerank=10 (fetch_k=1000), each query reads 3.1MB of random FP32 base-vector
   fetches. Cutting rerank 10× → 2× saves 2.5MB/query.

**The actual prize: raise PQ-only recall at the SAME 96B/vec budget, so we can
cut the rerank multiplier and reduce SSD bandwidth per query 5×.**

### What we are NOT doing

- **NOT shrinking codes.** Same 96B/vec (m=96, 8-bit).
- **NOT pursuing RaBitQ.** Dim-wide codes (min 768 bits = 96B at dim=768). Only
  matches PQ recall at 4-bit (384B/vec, 4× our budget).
- **NOT matching ScaNN's m=384 config.** ScaNN uses 2-dim subspaces (m=384 at
  dim=768 = 384B/vec, 4× our budget). We start at m=96 (8-dim subspaces, 96B).
  Escalate to m=192 only if m=96 falls short.

## The anisotropic PQ algorithm (from ScaNN §3-4)

**Codebook structure:** Per-subspace PQ, identical to ours. m sub-codebooks,
K=256 centroids each, sub_dim = dim/m. No structural change.

**The anisotropic objective** (ScaNN Definition 3.1 + Theorem 3.4):
For each datapoint xᵢ and its quantized version x̃ᵢ, decompose the residual
r = xᵢ − x̃ᵢ into components parallel and perpendicular to xᵢ:

```
r∥ = (⟨r, xᵢ⟩ / ‖xᵢ‖²) · xᵢ        (along xᵢ)
r⊥ = r − r∥                         (orthogonal to xᵢ)

ℓ(xᵢ, x̃ᵢ) = η · ‖r∥‖² + ‖r⊥‖²
```

where **η = T² / (1 − T²)** and **T = 0.2** (ScaNN's default) gives **η ≈ 4.125**.

- T = 0 ⟹ η = 1 (recovers standard k-means / isotropic MSE).
- T = 1 ⟹ η = ∞ (pure parallel loss).

**Why this works (ScaNN Theorem 3.2-3.3):** Under the assumption that queries q
are uniformly distributed on the unit sphere, the expected dot-product ranking
error decomposes into parallel and perpendicular residual components, with
parallel error weighted ≥ perpendicular error. The "query direction" is handled
analytically — no proxy queries, no sampling. The decomposition is against xᵢ's
own direction.

**Training algorithm** (ScaNN §4, Definition 4.3 — Lloyd's algorithm with the
anisotropic loss):

1. **Init:** warm-start with standard L2 k-means (our existing `train()`).
2. **Assignment step:** for each xᵢ, assign to argmin over centroids c of
   `η · ‖r∥(xᵢ, c)‖² + ‖r⊥(xᵢ, c)‖²`. Brute-force over K=256 centroids.
3. **Update step:** fix assignments, update each centroid via the closed form
   (ScaNN Theorem 4.2):
   ```
   cⱼ* = ( I·Σhᵢ,⊥ + Σ [(hᵢ,∥ − hᵢ,⊥)/‖xᵢ‖²]·xᵢxᵢᵀ )⁻¹ · Σ hᵢ,∥·xᵢ
   ```
   Per-subspace d×d linear solve (d=8 at m=96 — cheap).
4. Repeat 2-3 to convergence (loss is monotonic non-increasing).

### Search-time distance — IP-ADC

ScaNN's objective minimizes the error in ⟨q, x̃⟩. The natural search-time
estimator is **IP-ADC**: `⟨q, x̃⟩ = Σ_s ⟨q_sub_s, c_sub_s⟩`. Phase 1 already
shipped IP-ADC for the PQ tier. Phase 2 reuses it.

## Architecture: inheritance (train virtual, hot path non-virtual)

**Decision (revised from the original templating plan):** Make
`PqQuantizer::train()` the only virtual method. `AnisotropicPqQuantizer`
subclasses `PqQuantizer` and overrides ONLY `train()`. All hot-path methods
(`lut_distance`, `code_distance`, `lut_distance_batch4`, `code_distance_batch4`,
`build_code_lut`, `preprocess_query`) stay non-virtual — `VamanaCore` calls them
monomorphically through `PqQuantizer&`, zero dispatch overhead, inlinable.

This achieves the same hot-path monomorphism as full templating at ~120 lines
instead of ~1300. `VamanaCore`, `Index`, and the 5 construction sites stay
unchanged. `Builder` constructs `AnisotropicPqQuantizer` when `--quantizer
anisotropic-pq` is set, owning it via the base `unique_ptr<PqQuantizer>`.

**Why not full `VamanaCore<TQuantizer>` templating:** the ScaNN paper confirms
anisotropy lives entirely in training; search is standard ADC. The hot path is
identical for both quantizers. Templating would be 1300 lines of mechanical
refactor (base class, impl header split, explicit instantiations, factory, 5
construction-site updates) for zero hot-path benefit. YAGNI. If a later phase
discovers the hot path needs polymorphism (e.g., anisotropic weighting at query
time), revisit then.

**Why this works:** C++ non-virtual interface (NVI) idiom. The base class's
non-virtual methods resolve at compile time to `PqQuantizer`'s implementation.
`AnisotropicPqQuantizer` inherits those methods unchanged — its only difference
is the trained codebook (produced by its virtual `train()` override). The hot
path sees a `PqQuantizer` with a differently-trained codebook, nothing more.

### The quantizer classes

`PqQuantizer` stays as-is (proven, shipped). One change: `train()` becomes
virtual (the only virtual method). All hot-path methods stay non-virtual.

`AnisotropicPqQuantizer` is a NEW subclass of `PqQuantizer`. It overrides ONLY
`train()` — everything else (encode, preprocess_query, lut_distance, batch4,
code_distance, build_code_lut, serialize, deserialize) is inherited unchanged.
The inherited methods operate on whatever codebook `train()` produced, so they
work identically — only the trained codebook bytes differ.
  the `metric_` flag to IP, which Phase 1 validated).
- `build_code_lut`, `code_distance*`, `build_cross_distance_table` — identical
  (PQ-construct build mode, IP metric).
- `decode_code`, `serialize`, `deserialize` — identical structure, different
  codebook bytes.

Shared structure means most of `AnisotropicPqQuantizer` is a copy of
`PqQuantizer` with a different `train()`. **Consider extracting shared code
into a `PqOps` helper or CRTP base.** Implementation-taste decision; defer to P2.2.

### Config and selection

- `BuildConfig::quantizer` enum: `{ pq, anisotropic_pq }`. Default: `pq`.
- CLI: `--quantizer anisotropic-pq`.
- `--anisotropy-threshold T` (float, default 0.2). Threads into `train()`.

### Builder/Searcher branching

After templating, Builder/Searcher construct either `VamanaCore<PqQuantizer>` or
`VamanaCore<AnisotropicPqQuantizer>` based on config. Branch once at
construction via a factory function returning `unique_ptr<VamanaCoreBase>`
(virtual outer API, virtual-free inner loop), or branch at Builder/Searcher
entry into two code paths. Explore both in P2.4.

## Phase 2 sequencing

### P2.1: Make PqQuantizer::train() virtual — minimal refactor
Make `PqQuantizer::train()` the only virtual method (add `virtual` keyword;
the hot-path methods stay non-virtual). This is the only change to the existing
code — ~1 line. All existing tests pass byte-identical. No behavior change.
**Risk:** none. Virtual dispatch on `train()` is called O(1) per build.
**Gate:** all tests green.

### P2.2: AnisotropicPqQuantizer skeleton — standard k-means
Implement `AnisotropicPqQuantizer` as a subclass of `PqQuantizer` overriding
ONLY `train()`. The first version copies `PqQuantizer::train` verbatim (NOT
anisotropic yet — just to validate the plumbing). Add `--quantizer
anisotropic-pq` config flag + CLI; `Builder` constructs the subclass when set.
Validate plumbing: train → encode → preprocess_query → lut_distance →
serialize → deserialize. The inherited hot-path methods should produce
identical results to `PqQuantizer` since the codebook is trained identically.
**Risk:** serialization format (same tag + flag, or new tag).
**Gate:** anisotropic-pq with standard-k-means produces recall identical to
plain pq on arxiv100k. Proves plumbing before research.

### P2.3: Anisotropic training — THE RESEARCH (measures both L2sq-ADC and IP-ADC)
Implement ScaNN's anisotropic Lloyd's algorithm in `train()`:
- Assignment step: argmin over c of `η·‖r∥(x,c)‖² + ‖r⊥(x,c)‖²`.
- Update step: closed-form per-subspace solve (Theorem 4.2).
- T = 0.2 default (η ≈ 4.125). Sweep T ∈ {0.1, 0.2, 0.3, 0.5} on arxiv100k.
- Warm-start from standard k-means (P2.2's train), iterate 10-25 Lloyd rounds.

**Measure BOTH metrics after training** (the 2×2 matrix from the mission section):
- Config C: anisotropic PQ + L2sq-ADC.
- Config D: anisotropic PQ + IP-ADC.
Both share the same trained codebook; only the search-time ADC differs. The
comparison is the core Phase 2 research question: does anisotropic training
close the `‖x̃‖²`-variance gap enough for IP-ADC to match L2sq-ADC recall?

Validate locally on arxiv100k first (fast iteration on the training algorithm),
then arxiv-nomic 1.34M on c4a.
**Target: anisotropic IP-ADC recall ≥ 0.85 on 1.34M, AND anisotropic IP-ADC
within 2-3pp of anisotropic L2sq-ADC** (so IP's cost advantage wins on Pareto).
**Risk:** the genuine research risk. ScaNN's exact finite-d η isn't fully
documented (we use the d→∞ limit). May require iteration on the objective.
**Gate:** anisotropic IP-ADC recall on 1.34M. If < 0.78, ceiling holds —
document and stop. If 0.78-0.85 but >5pp below anisotropic L2sq-ADC, IP still
loses on Pareto — document and decide whether to ship anisotropic-L2sq.
If ≥ 0.85 AND within 2-3pp of L2sq, escalate to P2.4 for the full Pareto curve.

### P2.4: beam_search integration — end-to-end Pareto curve (2×2 matrix)
Wire `AnisotropicPqQuantizer` into `VamanaCore<AnisotropicPqQuantizer>`.
Builder/Searcher branch on `quantizer` config. Measure end-to-end Pareto curve
(recall@100 vs QPS) on arxiv-nomic 1.34M on c4a at rerank ∈ {1, 2, 3, 5, 10}.

**Measure all four configs** of the 2×2 matrix:
- A: plain PQ + L2sq-ADC (Phase 1 baseline — already measured).
- B: plain PQ + IP-ADC (Phase 1 — already measured).
- C: anisotropic PQ + L2sq-ADC.
- D: anisotropic PQ + IP-ADC (the ScaNN target).

The C-vs-D Pareto comparison is the Phase 2 verdict. If D dominates C, IP wins
(anisotropic closed the gap, IP's cost wins). If C dominates D, L2sq wins even
with anisotropic training. Either way, the anisotropic recall lift (C or D over
A) is the rerank-tax reduction prize.

**Risk:** hot-path regression for the PQ path (mitigated by templating).
**Gate:** anisotropic Pareto curve dominates Phase 1's best config on arxiv-nomic.
If dominated at every rerank, negative result — document and stop.

### P2.5: (Conditional) m escalation
Only if P2.3 recall falls short of 0.85 at m=96:
- m=192 (192B/vec, 2× budget): repeat P2.3 sweep.
- If m=192 still falls short, concentration-of-measure ceiling holds even with
  anisotropic training. Document and stop.
Only if P2.4 shows anisotropic PQ winning: productionize — update `analyze` tool
default, update docs.

## Phase 2 success criteria

**Primary (the Pareto prize):** Configuration D (anisotropic PQ + IP-ADC) beats
Configuration C (anisotropic PQ + L2sq-ADC) on the recall@100-vs-QPS Pareto
curve on arxiv-nomic 1.34M. Measured at rerank ∈ {1, 2, 3, 5, 10}.

This is the actual research question: does anisotropic training close the
IP-vs-L2sq recall gap enough for IP's cost advantage to win? P1.5 showed IP
loses badly with plain PQ (0.56 vs 0.99 ceiling). If anisotropic training
lifts IP-ADC to within 2-3pp of L2sq-ADC, IP wins on Pareto.

**Secondary (recall lift, regardless of metric):** anisotropic PQ lifts the
PQ-only recall ceiling above 0.80 at k=100 on 1.34M (vs 0.73 plain). This is
the rerank-tax reduction prize — it matters for BOTH L2sq-ADC and IP-ADC
configs (C and D). Even if IP loses to L2sq on Pareto, an anisotropic-L2sq
recall lift reduces the rerank multiplier and improves disk-resident QPS.

**Aspirational checkpoints:**
- Anisotropic IP-ADC recall ≥ 0.85 on 1.34M ⟹ rerank=2-3 where rerank=10
  was needed ⟹ ~3-5× SSD bandwidth reduction per query.
- Anisotropic IP-ADC recall ≥ 0.90 on 1.34M ⟹ rerank=1 might suffice.

**Stop conditions:**
- Anisotropic IP-ADC recall < 0.78 at m=96 AND m=192 ⟹ the `‖x̃‖²` variance
  can't be controlled by training; ceiling holds for IP. Document and stop.
- Configuration C (anisotropic L2sq-ADC) dominates D at every rerank AND
  lifts the ceiling above 0.85 ⟹ L2sq wins, ship anisotropic-L2sq as the
  production quantizer.
- Configuration D dominates C at every rerank ⟹ IP wins, ship anisotropic-IP.

## Phase 2 estimated effort

- P2.1 (templating): 1-2 days. Pure refactor.
- P2.2 (skeleton): 1-2 days. Copy-adapt from PqQuantizer.
- P2.3 (anisotropic training): 3-5 days. The research.
- P2.4 (integration + Pareto): 1-2 days. Plumbing + c4a benchmarking.
- P2.5 (conditional): 2-3 days if needed.

**Phase 2 total: ~2 weeks focused.**

---

## Hardware and validation (both phases)

- **Local (Mac):** arxiv100k for fast iteration (P1.4 microbench, P2.3 T-sweep,
  debug the closed-form solve, validate against brute force).
- **c4a at 1.34M:** authoritative recall + QPS (P1.5 matrix, P2.3 final, P2.4).
  Use `gcp_bench.sh`. Build with `-g` (release default).
- **Billion-scale extrapolation:** not directly measurable. The rerank-reduction
  win at 1.34M is the proxy: if rerank drops 10× → 2× at matched recall, the
  SSD-bandwidth reduction at 1B scale is the same factor.

## Literature references

- **ScaNN:** Guo et al., "Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization" (arXiv:1908.10396v5). §3 (score-aware loss decomposition),
  §4 Theorem 4.2 (closed-form centroid update), §5 (T=0.2 experiments).
- **VIBE benchmark:** arxiv-nomic: ScaNN 1049 QPS, SymphonyQG 560 QPS, LoRANN
  >1600 QPS at recall@100 ≥ 0.90 (in-memory, single-core, 384GB RAM). Reference
  points, not targets.
- **RaBitQ:** Gao & Long (arXiv:2405.12497). Disqualified (dim-wide codes).
- **Prior Sextant findings:** `docs/quantization_findings.md` (OPQ/anisotropic
  parameter tweaks all fail at scale), `docs/anisotropic_rearchitecture.md`
  (superseded).

## Open questions (defer to implementation)

1. **Finite-d η (P2.3):** ScaNN uses the d→∞ limit for η. At sub_dim=8, the
   finite-d correction may matter. Check if the paper's recursive formula is
   reproducible. If not, use the limit and validate empirically.
2. **Warm-start convergence (P2.3):** how many Lloyd rounds does the anisotropic
   objective need from a standard-k-means init? Measure on arxiv100k.
3. **Shared code with PqQuantizer (P2.2):** extract a `PqOps` helper or CRTP for
   identical encode/LUT/distance methods? Decide based on duplication volume.
4. **Serialization tag (P2.2):** same magic as PqQuantizer + anisotropy flag, or
   new tag? On-disk format not stabilized (clean slate allowed).
5. **Decomposed-L2sq precision (P1.3):** accumulating `‖q‖² − 2⟨q,c⟩ + ‖c‖²` in
   FP32 may differ from naive L2sq in the last ULP. Verify recall is unaffected.
6. **`dot_f32` dispatch overhead (P1.4):** 24,576 tiny NumKong calls at sub_dim=8.
   If slow, hoist to matvec (one `nk_dot_f32` per subspace × all K centroids).
