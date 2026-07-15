# Proximity in-band vs. id-recall@k: why the standard metric fails on modern data

## The problem with id-recall@k

**id-recall@k** — the fraction of true top-k ids returned by the search — has
been the ANN benchmark standard since SIFT and GIST. It measures set
intersection: `|GT_k ∩ R_k| / k`.

This metric is correct when the ground truth's top-k is **unambiguous** — i.e.,
when there is exactly one correct set of k nearest neighbors. On hand-crafted
feature vectors like SIFT (128d, 2003) and GIST (960d, 2009), this holds: the
k-th NN distance is distinct from the (k+1)-th, so the true top-k is well-defined.

**Modern embedding data breaks this assumption.** Transformer embeddings (768d+),
geocoding embeddings, and entity embeddings exhibit near-duplicate clustering:
many vectors sit at (nearly) the same location. For a query landing in a cluster,
dozens of vectors may be tied at the k-th NN distance. The ground truth must pick
exactly k ids, but which k is arbitrary — determined by implementation details
of the brute-force scan (argpartition tie-breaking, scan order, chunk size).

This has a concrete consequence: **even exact brute-force search cannot score
id-recall@k = 1.0** against such a ground truth. On a geocoding embedding dataset
(768d, 1.26M vectors), brute-force top-10 scores only **0.63 recall@10** against
the generated GT — not because brute force misses anything, but because it returns
10 different (equally-near) vectors than the 10 the GT arbitrarily marked.

When the metric itself is capped below 1.0 by the data's structure, not by the
index's quality, it ceases to be a useful measure of index quality.

## Why id-recall@k became the standard

Three factors, in order of importance:

1. **The benchmark datasets drove the choice.** SIFT and GIST are hand-crafted
   feature vectors with clean, well-separated neighbors. On these, id-recall@k
   IS the right metric — there's no ambiguity, and it perfectly captures search
   quality. The metric was designed for the data that existed.

2. **Simplicity.** id-recall@k is set intersection — trivially defined, trivially
   computed, no parameters or edge cases. A ratio-based metric like proximity
   requires defining a target radius (the k-th NN distance), handling degenerate
   cases (d=0 for co-located vectors), and interpreting a continuous ratio rather
   than a boolean match. Simpler metrics win in benchmark infrastructure.

3. **Institutional inertia.** ann-benchmarks (Aumüller et al., 2020) locked in
   the convention around 2015. Modern clustered embeddings arrived with the
   transformer era but the framework and community expectations were already set.
   Changing the metric would break comparability with a decade of published
   results — so the field keeps using a metric that's increasingly misapplied.

## Proximity in-band: a cluster-aware alternative

**Proximity in-band** measures the fraction of returned results whose true
distance is at or inside the true k-th NN distance (the "target radius"). Formally:

```
proximity = |{i ∈ R_k : d(result_i) ≤ d_k}| / k
```

where `d_k` is the true k-th NN distance.

This metric is **cluster-aware**: it doesn't care *which* co-located id was
returned, only that the result landed in the correct neighborhood. When 50
vectors are tied at the k-th NN distance, returning any 10 of them counts as
perfect proximity — which is correct, because they're all equally valid answers.

On non-clustered data (SIFT, GIST — ties@k ≈ 0), proximity and id-recall
converge: the target radius is sharp, so being at the right distance means
being the right id.

## Why there is no simple mapping between the two

The relationship between proximity and id-recall is **data-dependent** and can
go in either direction:

| Data characteristic | Proximity vs. id-recall | Why |
|---|---|---|
| Loose/no clustering (SIFT) | proximity > recall (+0.10–0.12) | Some in-band results are co-located non-GT vectors |
| Moderate clustering (SE mini-graph) | proximity > recall (+0.06–0.07) | Fewer but still some co-located non-GT vectors |
| Tight clustering (SE full-scale) | proximity < recall (−0.03 to −0.04) | Target radius is so tight that even correct results fall outside it due to PQ distortion |

The offset varies from −0.04 to +0.12 depending on the data's distance
distribution. There is no transferable formula. This is why we show both
metrics in the `analyze` table rather than converting between them.

## Design decision: proximity as the primary metric

Sextant uses **proximity in-band** as its primary quality metric for PQ config
selection. Rationale:

- **Most workloads** (RAG, search, recommendation, geocoding) need the right
  *neighborhood*, not the exact *id*. A search that returns 10 vectors from the
  correct cluster is successful even if the specific ids differ from an
  arbitrary ground truth.

- **id-recall@k is structurally broken** on clustered data. A metric that caps
  at 0.63 for perfect brute-force search is not measuring index quality.

- **Proximity degrades gracefully** on clustered data: it correctly reports
  high quality when the index finds the right neighborhoods, regardless of
  tie-breaking.

For the **niche workloads** where exact id-matching matters (entity resolution,
deduplication, record linkage), `analyze` provides `--id-recall-target` as an
override. These users know who they are and why they need exact ids.

## Compatibility with the literature

The ANN literature almost exclusively reports id-recall@k. Researchers and
engineers comparing Sextant to published results need this metric.
`docs/recall_targets.md` provides a full survey of recall targets across
workloads. The `sextant_bench` tool reports both metrics side by side.

The key insight: the literature's recall targets (0.90/0.95/0.99) are quality
bars set by researchers for algorithm comparison. They represent the user's
intent ("I need good results"), not a property of the id-recall metric itself.
When used as proximity targets, they still represent the same quality intent —
but proximity is the metric that actually measures it correctly on modern data.

## References

- Aumüller, Bernhardsson, Faithfull. "ANN-Benchmarks: A Benchmarking Tool for
  Approximate Nearest Neighbor Algorithms." *Information Systems*, 2020.
- Subramanya et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor
  Search." NeurIPS 2019.
- "ANN Search: Recall What Matters." arXiv:2606.04522, 2026.
- Christen, P. *Data Matching: Concepts and Techniques for Record Linkage,
  Entity Resolution, and Duplicate Detection.* Springer, 2012.
