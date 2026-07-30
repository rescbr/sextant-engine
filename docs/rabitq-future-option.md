# Option E: RaBitQ as the quantizer — investigated and rejected (2026-07-30)

> **Status: REJECTED for high-LID data.** FAISS spike confirms RaBitQ
> underperforms PQ4 on Sphere (LID 20.8): 0.357 vs 0.521 recall at 2×
> the storage and 3.6× lower QPS. The paper's advantage (beats PQ at
> 2× bits) does not transfer to high-LID, high-norm embeddings.
> Implementation exists (6 commits) but is parked. May revisit for
> low-LID datasets (SIFT, MSMARCO at LID ~14).
> RaBitQ's advantages are NOT portable to PQ without becoming RaBitQ.

## Why this is on the table

Three negative spikes this session established that PQ's structural
properties block the tricks that make RaBitQ/SymphonyQG fast:

1. **Co-located neighbor codes** (SymphonyQG-inspired, Option B): ~5-10%
   QPS for 4-22× storage overhead. Not worth it at billion scale. The
   win in SymphonyQG comes from RaBitQ's distance quality + the
   no-rerank-via-normalization trick, not co-location alone.

2. **Margin-based selective rerank on raw PQ** (E1): strictly worse than
   fixed top-W at matched avg-rerank budget. PQ's per-code error is
   non-uniform (codes near query's nearest centroids get small PQ
   distances regardless of true distance).

3. **Margin-based selective rerank on JLT-then-PQ** (E1.6): still fails.
   JLT rotation doesn't eliminate PQ's per-segment structure; only
   RaBitQ's segment-free 1-bit/dim encoding has the required uniformity.

**The structural conclusion: PQ cannot borrow RaBitQ's tricks without
becoming RaBitQ.** Eliminating segments entirely is the load-bearing
property. This makes the full RaBitQ spike (E2) the only remaining path
to selective rerank.

## What RaBitQ is (from arXiv:2405.12497, in academic_literature/rabitq/)

### Algorithm summary

**Index phase (per IVF shard):**
1. Compute centroid `c` of the shard.
2. For each data vector `o_r`: normalize `o = (o_r − c) / ‖o_r − c‖`.
   Precompute and store `‖o_r − c‖` (one float per vector).
3. Sample random orthogonal matrix `P` (production: Fast Hadamard
   Transform with random sign flips, O(D log D); spike: full random P
   via Gram-Schmidt, O(D²)).
4. For each `o`: compute `P⁻¹ · o`, take sign of each entry → D-bit
   code `x̄_b`. (Code length = smallest multiple of 64 ≥ D.)
5. Precompute `⟨ō, o⟩ = ⟨P·x̄, o⟩` per data vector (one float per
   vector; concentrates around ~0.8).

**Storage per vector (D=768):** D bits code (96 B) + ⟨ō,o⟩ (4 B) +
‖o_r−c‖ (4 B) = **104 B**. Comparable to our PQ-8 (96 B).

**Query phase (per query, per shard):**
1. Compute `‖q_r − c‖` (once per shard).
2. Compute `P⁻¹ · q_r`. Quantize to B_q bits per dim (B_q=4 in paper;
   Θ(log log D) suffices theoretically).
3. Build FastScan LUT from quantized `P⁻¹ · q_r` — same 4-bit nibble
   kernel as PQ-FastScan, but LUT entries are query·(±1/√D) per 4-dim
   slice, not query·centroid per segment.
4. Per data code: FastScan computes `⟨P·x̄, q_r⟩`. Divide by precomputed
   `⟨P·x̄, o⟩` → unbiased estimate of `⟨o, q⟩`.
5. Plug into distance formula with ‖o_r−c‖ and ‖q_r−c‖ → distance
   estimate with provable error bound `ε = O(1/√D)`.
6. **Selective rerank**: rerank only candidates whose lower bound
   `d̂ − ε` is less than current best exact distance. Provably safe
   (true NN always reranked with high probability).

### Why it beats PQ (per the paper, §experiments)

At default code length (D bits for RaBitQ, 2D bits for PQ-4-FastScan):
- RaBitQ has **consistently better accuracy** than PQ/OPQ at comparable
  efficiency, despite half the code length.
- PQ-4 has "disastrous accuracy" on MSong; degrades non-gracefully.
  RaBitQ is robust across datasets.
- Max relative error: RaBitQ ≤40%, PQ-4-FastScan ~100% on most datasets.

Mechanism: PQ's per-segment codes are biased (each segment only sees
its sub-vector; cross-segment correlations lost). RaBitQ after JLT
rotation sees the whole vector per bit — provably tighter error bound.

### GPU-IVF-RaBitQ cross-validation (arXiv:2602.23999)

The GPU-IVF-RaBitQ paper (in academic_literature/gpu_ivf_rabitq/)
directly compares IVF-RaBitQ vs IVF-PQ on CPU-equivalent workloads:
- IVF-RaBitQ beats IVF-PQ by **1.3-5.3× at recall 0.95** (even with
  PQ's refinement).
- "Without relying on reranking of raw vectors" — the no-rerank property.

## What Sextant would gain from Option E

1. **Selective rerank** (the main prize): per-query adaptive rerank
   count via error bounds. Easy queries rerank few; hard queries rerank
   many. Eliminates the W hyperparameter tuning that PQ needs.

2. **Better per-bit distance quality**: RaBitQ at D bits beats PQ at 2D
   bits. Could either (a) improve recall at fixed storage, or (b) match
   recall at half the storage (48 B/vec instead of 96 B).

3. **Theoretical robustness**: error bound holds regardless of data
   distribution. PQ's empirical performance varies; RaBitQ is provably
   bounded.

4. **SymphonyQG compatibility**: if we ever want graph + FastScan (which
   we punted on for Option A), RaBitQ is what makes it work. The
   no-rerank-via-normalization trick depends on RaBitQ's distance
   decomposition.

## What it costs (vs Option A)

1. **New quantizer implementation**: JLT/FHT (production: Fast Hadamard
   Transform; spike: full random P), D-bit codec, unbiased estimator,
   bound computation. ~2-3 weeks of careful work. The quantizer is the
   load-bearing piece; bugs here silently destroy recall.

2. **New LUT construction**: the FastScan kernel itself is the same
   (4-bit nibble lookup), but the LUT is built from the rotated query
   differently. One new code path in preprocess_query.

3. **Storage +104 B/vec** (vs 96 B for PQ-8, or 48 B for PQ-4). At 1B
   that's 104 GB — comparable to our current PQ-8, 2× our Option A.

4. **Build cost**: training is trivial (random rotation, sign take).
   Cheaper than PQ's k-means. Encoding is O(D²) per vector with full P,
   O(D log D) with FHT. At 1B with FHT: manageable.

5. **No rerank shortlist fixed-W**: the selective-rerank trick needs
   the error bound computed per candidate, which adds per-candidate
   work. Net win depends on how many candidates get pruned vs PQ's
   fixed-W.

## When to revisit

Any of:
1. **Option A underperforms in production.** If the 4-bit PQ + top-W
   rerank doesn't hold its 2.9× win at 1B scale (e.g., rerank cost
   dominates, or recall drops on harder datasets), RaBitQ's selective
   rerank becomes the natural fix.
2. **A batched-search use case emerges** that benefits from RaBitQ's
   better per-bit quality (especially relevant for GPU offload — see
   `docs/gpu-future-option.md`).
3. **SymphonyQG-style graph + FastScan becomes attractive.** RaBitQ is
   a prerequisite; without it the graph path can't escape the per-hop
   random access penalty.
4. **A confirmed need to halve storage.** RaBitQ at D bits matches PQ
   at 2D bits, so a RaBitQ codebase could ship at 48 B/vec with the
   same recall as our 96 B/vec PQ-8.

## Implementation sketch (when we get there)

### Phase 1: Spike (local, arxiv-nomic, ~1 week)

1. **FHT on ARM NEON** (D=768 → pad to 1024 for power-of-2). Random
   sign flip before transform. ~500 lines.
2. **Codec**: encode (rotate, sign take) + decode (reconstruct P·x̄
   from bits). Store `⟨ō,o⟩` and `‖o_r−c‖` per vector.
3. **LUT build**: quantize rotated query to B_q=4 bits/dim, build
   16-entry-per-4-dim-slice LUT.
4. **Estimator**: ⟨P·x̄, q_r⟩ / ⟨P·x̄, o⟩ → ⟨o,q⟩; plug into Eq (3)
   of method.tex for squared distance.
5. **Selective rerank**: per-candidate bound-based pruning.
6. **Bench**: recall-QPS Pareto vs Option A at matched code size on
   arxiv-nomic. The key metric — does selective rerank beat fixed-W?

### Phase 2: Production (if Phase 1 wins, ~2-3 weeks)

1. **Builder integration**: new quantizer type alongside PqQuantizer.
   Build path: k-means IVF shard → per-shard centroid → RaBitQ encode.
2. **Sidecar format**: `.rabitq` sidecar (D-bit codes packed) alongside
   existing `.codes`. Or replace `.codes` if we commit to RaBitQ-only.
3. **Searcher**: IVF-list-scan reuses the same FastScan kernel (4-bit
   nibble); only the LUT construction and post-processing differ.
4. **c4a measurement**: validate the win holds on V2 SVE2.
5. **1B-scale paging test**: confirm FHT encoding time is acceptable.

### Phase 3: SymphonyQG compatibility (optional, ~2 weeks)

If RaBitQ lands and we want graph + FastScan: implement the
no-rerank-via-normalization trick (§3.2 of symphonyqg paper). The
decomposition `⟨x̄, P⁻¹·q⟩ = (1/‖q_r−c‖)·(⟨x̄, P⁻¹·q_r⟩ − ⟨x̄, P⁻¹·c⟩)`
lets the LUT be query-only (shared across all hops). This is what makes
per-hop FastScan viable without per-hop LUT rebuild.

## Key references (all in academic_literature/)

- `rabitq/` — RaBitQ paper (arXiv:2405.12497). Method, estimator, bound.
- `symphonyqg/` — SymphonyQG paper (arXiv:2411.12229). Graph + RaBitQ +
  no-rerank-via-normalization. §3.2 has the LUT-once decomposition.
- `gpu_ivf_rabitq/` — GPU-IVF-RaBitQ (arXiv:2602.23999). Cross-validation
  of IVF-RaBitQ vs IVF-PQ on CPU-equivalent workloads. Reports 1.3-5.3×
  speedup at recall 0.95.
- `chen_2026/` — PIMCQG (arXiv:2605.25522). IVF-clustered RaBitQ with
  α=0.8 mult-free trick. May inform Phase 3.

## Prior Sextant analysis (corrected this session)

The earlier `docs/optimization_levers_and_attribution.md` claimed
"RaBitQ worse than PQ at matched bytes." **That claim was unsourced and
contradicts both the RaBitQ paper and the GPU-IVF-RaBitQ paper.** The
correction is in `memory/colocated-spike-2026-07-25.md` (§"CORRECTION
on RaBitQ vs PQ recall"). The docs file should be updated when Option E
is revisited.

---

## FAISS spike results (2026-07-30) — REJECTED for high-LID data

A controlled FAISS experiment on Sphere (100k subset, IP, LID 20.8)
confirms RaBitQ underperforms on our target data:

| Config | Storage | np=64 recall | np=64 QPS |
|--------|---------|-------------|-----------|
| FAISS RaBitQ 1-bit | 96 B | 0.357 | 8,093 |
| FAISS PQ4 | 48 B | 0.521 | 28,829 |
| Sextant PRQ s96 b5 + 8b LUT | 48 B | **0.945** | 79 |

(QPS from FAISS Python on 100k cache-resident; not comparable to c4a
10M scale, but ratios are informative.)

### Why RaBitQ fails on high-LID data

1. **1 bit/dim is too coarse for LID 20.8.** The O(1/√D) error bound
   (~3.6% at D=768) is theoretically bounded, but the constant factor
   is too large for the tight neighbor distributions in high-LID data.
   Neighbors and non-neighbors have similar sign-dot products after
   rotation — the discriminative power is insufficient.

2. **2× storage penalty.** RaBitQ at D bits (96B for D=768) should be
   compared to PQ at 2D bits (96B = PQ8). But even against PQ4 at 48B
   (half the storage), RaBitQ has *lower* recall. The sign hash wastes
   capacity that PQ's learned codebooks use more efficiently.

3. **No transfer from paper benchmarks.** RaBitQ's claims are validated
   on SIFT1M/GIST1M (LID ~10-12). At LID 20.8, the advantage inverts.

### When RaBitQ might still be useful

- **Low-LID datasets** (SIFT, MSMARCO at LID ~14): the paper's advantage
  may hold. Worth testing if we add low-LID datasets to the benchmark.
- **Selective rerank architecture**: RaBitQ's error-bound-based adaptive
  rerank (vs our fixed W=300) is architecturally superior and could be
  ported to PQ/PRQ if we had per-vector error bounds.
- **Graph + FastScan (SymphonyQG-style)**: RaBitQ is a prerequisite for
  the no-rerank-via-normalization trick. But we're on IVF-scan, not graph.

### Implementation status

6 commits (encode, IVF integration, search path, selective rerank,
thread-safe query state). Distance estimation has a magnitude bug
(est_ip ≈ 0.35 when true ≈ 40) — multiple fixes applied but core issue
remains. Parked: fixing it won't change the conclusion (RaBitQ < PQ4
on Sphere even in FAISS's correct implementation).
