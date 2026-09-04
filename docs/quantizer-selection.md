# Quantizer selection

How to pick `--quantizer` (or `sextant_build_opts.quantizer`). Measured on
dbpedia-1536 @ 933K, c4a; ceilings = decoded-ranking recall@10 of a perfect
quantized scan (the upper bound for any scan/rerank that never consults the
original vectors). See `scripts/spike_bits_elbow.cpp` for the methodology.

## Local quantizers (recommended default)

Rulers/codebooks are fitted **per leaf at flush time**, from the leaf's own
vectors. No global training sample, no frozen model: appends, deletes, leaf
splits and rebalances re-derive their quantizer naturally.

| quantizer | bytes/vec | ceiling recall@10 | notes |
|---|---|---|---|
| `local_scalar` (default) | ~770 B (4-bit + bias + id) | 0.948 | Per-leaf σ-ruler. Best recall-per-byte on embedding corpora; dual-SDOT scan kernel (3.5× f32). Scales: ceiling *improves* with corpus size (0.938→0.948 over 9.3× data). |
| `local_pq` | 24–48 B + id | (bytes-axis rows, unmeasured) | Per-leaf residual codebooks. Choose when footprint dominates and shortlists are re-scored exactly by the caller (`exact_rerank_base`) — containment is what it sells, not ranking precision. |

Use a local quantizer when: the corpus may drift, appends/updates are
expected, or you simply want the default that does not depend on a training
sample being representative.

## Global quantizers

One ruler/codebook for the whole tree, trained at build on a **20K random
reservoir sample** (Algorithm R over the source stream — the engine never
opens source files). What you must consider:

1. **The model is frozen at build.** Appended vectors are encoded against
   the build-time distribution. If the corpus drifts (new topics, new
   embedding model versions, time-ordered streams), a global ruler silently
   loses recall — the failure mode is invisible until measured. Retraining
   means rebuild (an explicit `optimize()`-style operation, as in
   ClickHouse/Vespa). Local quantizers do not have this failure mode.
2. **Sample representativeness is your recall budget.** 20K uniform vectors
   is statistically sound for stationary data (prefix vs random vs full
   corpus measured 0.950 / ~0.953 / 0.956 on dbpedia — order matters more
   than sample size).
3. **They exist for specific slots, not general use:**
   - `scalar_uniform` — global σ-ruler; the data-oblivious-adjacent
     baseline; fastest build, no codebook; ceiling decays with size
     (0.963 @100K → 0.928 @933K on dbpedia).
   - `scalar_lloydmax` — global Lloyd-Max per-dim; same caveats.
   - `pq` / `anisotropic-pq` — sub-space codebooks; for tiny byte budgets
     (32–96 B/vec) where the LUT scan shape is wanted.
   - `prq` — additive residual refinement of pq.

## Precision tiers for rerank (all quantizers)

The decoded ceiling is not the end of the story — final ranking precision is
set by what re-scores the shortlist:

| rerank source | bytes/vec extra | end-to-end recall | notes |
|---|---|---|---|
| decoded codes | 0 | = scan ceiling (0.948 @ local_scalar-4b) | default |
| 8-bit rerank codes (m=3.9σ) | ~1.5 KB | 0.997 | beats fp16 (0.992) at half the bytes; bias must be f32/scaled-u16 (fp16 bias costs ~2.4pp) |
| caller corpus (`exact_rerank_base`) | 0 in-tree | 1.000 (containment-bound) | the engine gathers your f32 vectors by row_id; the index/compressed-store contract |
