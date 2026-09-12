# Plane v2: in-build encoding, per-leaf storage, and mutation

> Design note — 2026-09-12. Status: agreed direction; implementation
> sequenced as (1) parquet payload support, (2) plane v2, (3) CulturaX
> dev-cycle corpus. Supersedes the "immutable v1 plane" model for the
> default path; v1 format stays readable and `plane-attach` remains the
> post-hoc re-ranking tool for existing trees.

## Problem

The v1 routing plane is a post-hoc, immutable overlay:

1. `attach_plane` requires the corpus as a contiguous fp32 .fbin mmap —
   parquet corpora cannot use the default build at all (attach silently
   skips).
2. Attach re-reads the full corpus (~100 GB of I/O at 100M) after the
   build already streamed it 11 times.
3. `insert_batch` / `delete_batch` / `vacuum` / `defrag` hard-fail while
   a plane is attached ("v1 planes are immutable").
4. Mutation has no rebalancing anyway: split-on-overflow is the only
   structural response; centroids are frozen at build; deletes
   tombstone; defrag relocates pages but never re-clusters.

## Design

### 1. In-build plane encoding (one pass, any VectorSource)

Plane training and encoding ride the existing build phases:

- **Training**: the build's 20K sample already exists; train the PCA-128
  basis + encoding stats (b1g per-dim scales / u4lm codebooks) from it,
  in `train_quantizer_and_pca`, zero extra source passes.
- **Encoding**: the emission pass has every vector AND its target leaf
  in hand. The plane row (16 B b1g) appends to the target leaf's
  in-memory buffer — the same buffer, capacity, and flush cadence as the
  codes (+16 B/vector ≈ +80 MB at k_root=1024 × leaf_cap 5000; bounded
  RSS preserved).
- **Replication question** (measure, don't assume): closure replicates
  boundary vectors into multiple leaves. Match v1 (encode once, primary
  leaf only) or replicate rows too (+~15% plane bytes, but the sweep
  then scores every member the scan will actually see).

### 2. Per-leaf plane storage (the mutation enabler)

Plane rows move OUT of the single contiguous blob INTO each leaf's own
extent (prefix/suffix region):

- offsets stop being derived from cumulative counts; each leaf owns its
  row region and can grow/shrink independently.
- The sweep's per-leaf read now includes that leaf's plane rows in the
  same extent — page-ordered locality strictly improves; the separate
  plane-cache tier collapses into the leaf cache.
- v1 blob stays supported at open (format versioned).

### 3. Mutation semantics

- **Insert**: encode the plane row at insert time (basis RAM-resident;
  projection + nibble encode is trivial) and append alongside the code
  in the existing leaf-growth path.
- **Split**: rows redistribute with codes through the existing split
  machinery, extended by one region.
- **Delete**: tombstone conservatively — stale rows over-propose
  candidates (recall-safe, small QPS cost); vacuum compacts rows.
- **Rebalance** (new; explicitly scoped out of v1, in for v2):
  threshold-triggered incremental Lloyd —
  - recompute leaf centroids from live members during any pass that
    already reads the leaf;
  - bounded boundary-member reassignment per pass (top-k drifters) —
    amortized, never a full re-cluster;
  - merge leaves below a floor, split above a ceiling;
  - root-centroid refresh on a slower cadence (piggyback on compaction).
  Plane rows ride member moves (a reassignment is a row move between
  two leaf extents).

### 4. Empty index / bootstrap

- New create-empty entry point (no build required to start inserting).
- Below the plane threshold (~20K inserts): buffer vectors in RAM
  (bounded, same budget as the build sample), route by legacy centroid
  descent (or brute force while tiny).
- At threshold: auto-trigger basis training + backfill of buffered
  rows, then incremental encode from then on.
- Same threshold-trigger pattern as rebalance: maintenance passes are
  background "read leaf → transform → write leaf" work, no-ops on a
  static index. The benchmark path pays nothing.

## Interaction notes

- Single-stream sweep invariant unchanged (leaf-ordered, per-leaf
  extents preserve it).
- Plane cache tier (2026-09-12) becomes redundant under per-leaf
  storage; keep `--plane-cache-mb` for v1-format trees only.
- Build determinism: plane encode rides the deterministic emission
  order; byte-identical-build discipline applies unchanged.
- Benchmark protocol: v2 must reproduce the canonical table before any
  layout landing (cold/warm, recall bit-identical to v1-attached trees
  at same rank/encoding).

## Validation gates

1. Parquet payload: suite + payload round-trip test (build with payload
   column, search with payload output, byte-compare).
2. Plane v2 in-build: canonical cohere10m config must match
   v1-plane-attached tree recall/QPS (same rank/enc) before layout swap;
   after per-leaf layout, cold numbers must not regress (locality claim).
3. Mutation: insert/delete/split with plane attached; split-merge cycle;
   bounded reassignment cost; recall drift measurement (drifting
   distribution stress test) — the drift curve is the deliverable.
4. Bootstrap: empty → threshold → plane active; recall parity of
   backfilled rows vs batch-encoded rows on the same data.
