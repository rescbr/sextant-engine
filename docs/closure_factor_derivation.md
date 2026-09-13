# K-Dependent Closure Factor for IVF Partitioning: Analytical Derivation

> **Status: historical (July 2026 flat-graph era) — kept for reference; `closure_factor` is live in src/engine/estimator.cpp and src/tree/tree_estimator.hpp.**

> **Status:** Derived 2026-07-31. Validates the `np ∝ √K` law's boundary-loss
> correction and identifies the absolute-margin closure scheme (SPANN) as
> structurally superior to the ratio-based scheme.

## Abstract

We derive the optimal closure (boundary replication) factor `c` for IVF
partitioning as a function of the shard count `K`. The central, non-obvious
result is that **at leading order, `c` is K-independent**: the ratio-based
threshold `c · d₁` (where `d₁` is the nearest-centroid distance) covers a
constant expected number of secondary centroids regardless of `K`, because
both the cell radius and the inter-centroid gap scale with the same power
of `K`. The recall loss observed when scaling `K` (2.86 pp at K=4096 vs
K=1024 on Sphere-10M) is therefore a **routing-coverage** effect (more
cells = more boundary ambiguity per probe), not a closure-replication
deficiency. We derive a defensive `c(K)` that grows slowly with `K` as
insurance against higher-order effects, and show that SPANN's absolute-
margin boundary posting is structurally K-robust where the ratio scheme is
K-fragile.

---

## 1. Setup

An IVF index partitions `N` vectors into `K` Voronoi cells via k-means on
PQ-compressed codes. During build, each vector is assigned to its nearest
centroid (primary membership) and optionally replicated to nearby centroids
via **closure**: a vector at distance `d₁` from its nearest centroid is also
assigned to all centroids within `c · d₁`, where `c ≥ 1` is the closure
factor.

The closure factor controls the **replication rate** `R` — the fraction of
vectors assigned to ≥2 cells. Higher `R` → better boundary recall but more
storage and scan cost.

The existing formula (`docs/design_decisions.md:518–553`):

$$c = (1 - f_{\text{target}})^{-1/d_{\text{eff}}}$$

where `f_target` is the target fraction of non-replicated vectors and
`d_eff` is the effective (manifold) dimensionality. At `f_target = 0.15`,
`d_eff = 5`: `c = 1.033`.

**Empirical anomaly:** at K=1024 on Sphere (10M × 768, IP), this yields
5.4% replication and recall@10 = 0.6817. At K=4096 with the same `c`,
recall drops to 0.6531 (−2.86 pp) despite the `np ∝ √K` law predicting
equal recall at matched codes-scanned. Is this a closure problem?

---

## 2. Geometric Model

### 2.1 Nearest-centroid distance

For `K` k-means centroids on a `d_eff`-dimensional manifold, the local
centroid intensity is `λ ~ K`. The typical nearest-neighbor distance
satisfies the volume-packing condition:

$$\lambda \cdot r_1^{d_{\text{eff}}} \sim 1 \implies d_1(K) \sim A \, K^{-1/d_{\text{eff}}}$$

As `K → ∞`, cells shrink and `d₁ → 0`. ✓

### 2.2 Gap to the second-nearest centroid

Under a local Poisson point process approximation (justified for k-means
centroids away from lattice artifacts), the expected count of centroids in
the shell `[d₁, r]` is:

$$\mathbb{E}[\text{centroids in } [d_1, r]] = \lambda \cdot V_{d_{\text{eff}}} \cdot (r^{d_{\text{eff}}} - d_1^{d_{\text{eff}}})$$

Setting this to 1 for the second-nearest centroid:

$$\lambda \cdot (d_2^{d_{\text{eff}}} - d_1^{d_{\text{eff}}}) \sim \lambda \cdot d_1^{d_{\text{eff}}} \sim 1$$

(The second equality uses `λ · d₁^{d_eff} ~ 1` from §2.1.) This gives:

$$\frac{d_2}{d_1} \to 2^{1/d_{\text{eff}}}$$

**This ratio is a constant, independent of K.** The second-nearest centroid
is always at a fixed multiple of the nearest distance, regardless of how
many centroids there are.

### 2.3 Implication for the ratio threshold

A vector is replicated if any centroid falls in `[d₁, c · d₁]`. The expected
count in this shell is:

$$\mu(c) = \lambda \cdot (c \cdot d_1)^{d_{\text{eff}}} - \lambda \cdot d_1^{d_{\text{eff}}} = c^{d_{\text{eff}}} - 1$$

where we used `λ · d₁^{d_eff} ~ 1`. **This is K-independent.** The replication
probability is:

$$R(c) = 1 - e^{-\mu(c)} = 1 - e^{-(c^{d_{\text{eff}}} - 1)}$$

At the boundary-volume level, the complementary result (`docs/design_decisions.md:541`):

$$R(c) = 1 - c^{-d_{\text{eff}}}$$

Both models agree at leading order: **the replication rate depends on `c`
and `d_eff` but NOT on `K`.**

---

## 3. Why the Ratio Scheme is K-Independent

### 3.1 The key scaling

The reason the `K`-dependence cancels is that the closure threshold `c · d₁`
scales identically to the inter-centroid spacing:

| Quantity | Scaling with K |
|----------|---------------|
| Cell radius / NN distance `d₁` | `K^{-1/d_eff}` |
| Shell volume `d₁^{d_eff}` | `K^{-1}` |
| Centroid intensity `λ` | `K` |
| Expected centroids in shell `λ · d₁^{d_eff}` | `K · K^{-1} = O(1)` |

The `K`-dependence cancels exactly. Adding more centroids makes the cells
smaller, but proportionally fills in the gaps — the shell `[d₁, c·d₁]` always
contains the same expected number of centroids.

### 3.2 The covering-radius counter-argument

One might argue that the covering radius (the maximum distance from any
point to its nearest centroid) scales as `K^{-1/(d_eff-1)}`, which decays
**slower** than `d₁ ~ K^{-1/d_eff}` (since `1/(d_eff−1) > 1/d_eff`).
This means the inter-centroid gap shrinks **faster** than `d₁`, so `d₂/d₁`
should **decrease** with K — replication should *increase*, not drop.

**There is no monotone geometric mechanism in the ratio model that makes
replication drop with K.** The ratio scheme is structurally K-robust at
leading order.

---

## 4. Where the Recall Loss Actually Comes From

Since closure replication doesn't degrade with K, the 2.86 pp recall loss at
K=4096 must come from a different mechanism. The answer is documented in
`docs/ivf_k_sweep.md:33–35` (doc archived 2026-09-13 to workspace `old-docs/`): **routing recall drops with K**.

At K=8…64 (measured), routing recall drops from 0.99 to 0.92 at matched
probe fraction. The mechanism: more centroids = more Voronoi cells = more
boundaries = more boundary ambiguity per probe. A query's true neighbors
are more likely to straddle a cell boundary at fine K, and if the query
isn't routed to ALL the right cells, those neighbors are missed.

This is a **routing-coverage** problem, not a **build-time closure** problem.
Increasing closure_factor can partially compensate (by placing boundary
vectors in more cells, increasing the chance that a probed cell contains
them), but it's a second-order effect. The first-order fix is more probes
(higher np) or multi-probe extension — which is what the `np ∝ √K` law
prescribes.

### 4.1 Calibration note

The measured replication at K=1024 is 5.4%, but the formula predicts 15%
(at `d_eff=5`). This discrepancy implies an effective dimension of ~2 for
this specific statistic on Sphere-768 — the data is more concentrated than
the `d_eff=5` manifold model assumes. The constant `d_eff=5` was tuned for
SIFT-like manifolds and is miscalibrated for Sphere. **Fix: measure `d_eff`
per-dataset from the actual replication rate, not hardcode it.**

---

## 5. Defensive `c(K)` Formula

Although the leading-order theory says `c` is K-independent, higher-order
effects (non-Poisson centroid tessellation, finite-sample boundary
fluctuations, k-means convergence artifacts) may introduce weak K-dependence.
As defensive insurance, we derive a `c(K)` that grows slowly:

The covering radius `ρ ~ K^{-1/(d_eff-1)}` shrinks slightly faster than
`d₁ ~ K^{-1/d_eff}`. The residual ratio `ρ/d₁ ~ K^{1/d_eff - 1/(d_eff-1)}`
grows as a negative power of K (since `1/d_eff > 1/(d_eff-1)` is false —
actually `1/d_eff < 1/(d_eff-1)`). The correction exponent is:

$$\alpha = \frac{1}{d_{\text{eff}} - 1} - \frac{1}{d_{\text{eff}}} = \frac{1}{d_{\text{eff}}(d_{\text{eff}}-1)}$$

The defensive formula:

$$\boxed{c(K) = c_0 \cdot \left(\frac{K}{K_0}\right)^{1 / (d_{\text{eff}}(d_{\text{eff}}-1))}}$$

Calibrated at `K₀ = 1024`, `c₀ = 1.033`:

| K | c(K) |
|---|------|
| 1024 | 1.0330 (calibration) |
| 4096 | 1.1071 |
| 16384 | 1.1866 |

At `d_eff = 5`, the exponent is `1/20 = 0.05`. The growth is slow — a
4× increase in K only raises `c` by ~7%.

**Prediction:** re-running K=4096 with `c = 1.107` will recover at most
~1 pp of the 2.86 pp loss (the closure-attributable fraction), confirming
routing as the dominant term.

---

## 6. SPANN's Absolute-Margin Scheme: Structural Superiority

SPANN (Chen et al., NeurIPS 2021) posts a vector to cell `b` iff:

$$\text{dist}(x, b) \le \text{dist}(x, \text{own\_centroid}) + \varepsilon$$

where `ε` is an **absolute** margin, not a ratio. This is structurally
K-robust:

- As `K` increases, `d₁ → 0`, so `d₁ + ε ≈ ε`.
- The absolute margin `ε` covers an increasing **fraction** of the cell
  radius `d₁`: `ε / d₁ ~ ε · K^{1/d_eff} → ∞`.
- Replication grows with K automatically — exactly what's needed to
  compensate for denser boundaries.

In contrast, the ratio scheme `c · d₁` couples the threshold to the
shrinking `d₁`, making it **K-fragile** at higher order: any miscalibration
in `d_eff` or finite-sample effects are amplified as `d₁ → 0`.

**Recommendation:** migrate from ratio-based closure to absolute-margin
closure (SPANN-style). The `c(K)` formula above is a stopgap; the absolute
scheme is the principled fix.

---

## 7. Summary of Results

| Claim | Evidence |
|-------|----------|
| `c` is K-independent at leading order | Shell count `μ(c) = c^{d_eff} − 1` is K-free (§2.3) |
| Recall loss at K=4096 is routing, not closure | `docs/ivf_k_sweep.md:33–35` (archived to `old-docs/`) + §4 |
| `d_eff = 5` is miscalibrated for Sphere | Predicted 15% replication, measured 5.4% (§4.1) |
| Defensive `c(K)` grows as `K^{0.05}` | Covering-radius correction (§5) |
| SPANN absolute-margin is structurally superior | `ε/d₁ → ∞` as K grows (§6) |

## References

- SPANN: Chen et al., "SPANN: Highly-efficient Billion-scale Approximate
  Nearest Neighbor Search", NeurIPS 2021.
- The `np ∝ √K` law: `docs/np-sqrt-k-law-2026-07-28.md` (this workspace).
- Existing closure derivation: `docs/design_decisions.md:518–553`.
- K-sweep routing recall drop: `docs/ivf_k_sweep.md:33–35` (archived 2026-09-13 to workspace `old-docs/`).
