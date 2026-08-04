# Phase H: Leaf-Level Closure — Negative Result

**Date:** 2026-08-04
**Status:** Implemented, benchmarked, NOT Pareto-improving. Candidate for removal.

## What it does

Post-hoc leaf-level boundary replication (SPANN-style closure at the leaf level).
After the streaming build writes all leaves and their centroids are known, a
second pass identifies vectors near leaf boundaries — within `leaf_closure_eps`
of a second leaf centroid in the same root cluster — and appends them to the
neighboring leaf via extent growth.

Controlled by `BuildConfig::leaf_closure_multiplier` (default 0 = off).
When >0, the closure epsilon per cluster is derived from the mean gap between
sibling leaf centroids within the cluster.

## Benchmark Results

### Sphere-IP 10M (LID ~20.8, local d_eff ~0.29)

| Config       | np=4 QPS | np=4 Recall | np=8 QPS | np=8 Recall | np=16 QPS | np=16 Recall |
|--------------|----------|-------------|----------|-------------|-----------|--------------|
| Baseline     | 8952     | 0.776       | 6281     | 0.836       | 4042      | 0.874        |
| LC mult=0.15 | 8357     | 0.779       | 5806     | 0.839       | 3728      | 0.876        |
| LC mult=0.30 | 8197     | 0.781       | 5685     | 0.839       | 3609      | 0.876        |

Recall gain: +0.3–0.5pp. QPS cost: −6% to −11%.

### MSMARCO 8.7M (LID ~14.6)

| Config       | np=4 QPS | np=4 Recall | np=8 QPS | np=8 Recall | np=16 QPS | np=16 Recall |
|--------------|----------|-------------|----------|-------------|-----------|--------------|
| Baseline     | 9720     | 0.892       | 7045     | 0.936       | 4649      | 0.959        |
| LC mult=0.15 | 9156     | 0.892       | 6606     | 0.936       | 4321      | 0.959        |
| LC mult=0.30 | 8902     | 0.894       | 6303     | 0.937       | 4088      | 0.960        |

Recall gain: +0.05–0.18pp. QPS cost: −6% to −12%.

## Why it doesn't help

The tree already has **root-level closure** (SPANN-style boundary replication at
the root cluster level): during the emission pass, each vector is assigned to its
nearest root centroid AND any centroid within `closure_epsilon` of the nearest
distance. This already captures the inter-cluster boundary vectors that matter
most for recall.

Leaf-level closure replicates vectors across leaves *within the same root
cluster*. But these intra-cluster boundaries are already well-served by the
root closure — the vectors are in the cluster, reachable by routing. The
marginal vectors that leaf closure finds (those near a leaf boundary but not
near a root boundary) are few and contribute negligibly to recall.

The QPS cost is inherent: closure-grown leaves have more codes to scan
(replication factor increases), so every FastScan block does more work for
no recall benefit.

This holds across both low-LID (Sphere-IP, ~0.29 local d_eff) and medium-LID
(MSMARCO, ~14.6) data. The hypothesis that higher LID would benefit more from
leaf closure is **falsified**.

## What IS valuable from this work (keep)

The leaf closure feature itself is a candidate for removal, but the
infrastructure built to support it is essential for Phase J (insert/delete):

1. **Leaf table indirection** (`LeafTableEntry`, superblock blob, search/fsck
   resolution): `ChildEntry.child_page` for leaf children stores a `leaf_id`
   (index into the leaf table) instead of a raw page pointer. Leaf growth
   updates only the table entry — no parent pointer fixup needed.

2. **`grow_leaf` primitive** (in `run_leaf_closure`): allocate new extent →
   copy old data → append new vectors → free old extent → update leaf table.
   Phase J insert reuses this exact mechanism.

3. **`PageAllocator::clear_free_list()`**: fixes free-list corruption when
   extents are freed and reallocated in interleaved order (the free list
   accumulates stale "next" pointers pointing to data-overwritten pages).

4. **Per-cluster d_eff measurement** (Phase I): computed during Lloyd
   refinement. Available for future search-time adaptation (per-subtree
   n_probe calibration).

5. **Debugging infrastructure**: `__cxa_throw` interception (throw-site
   stack traces for all exceptions), core dump support, debug page bounds
   checking.

## Removal plan (when ready)

- Remove `run_leaf_closure` function from `ivf_tree_index.cpp`
- Remove `leaf_closure_multiplier` from `BuildConfig` and CLI flags
- Remove `ClosureAppend` struct and `leaf_closure_appends` from context
- Remove the `run_leaf_closure(ctx, file, alloc)` call from the orchestrator
- **Keep**: leaf table indirection, grow_leaf mechanism (refactor into a
  standalone `grow_leaf` method for Phase J), clear_free_list, d_eff
- **Keep**: `--leaf-closure-mult` CLI flag parsing removed; leaf table
  infrastructure stays

Estimated removal: ~200 lines of `run_leaf_closure` + ~30 lines of context.
No format change needed — leaf table is always written (even without closure).
