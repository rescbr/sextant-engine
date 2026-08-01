# Hierarchical IVF Tree Design (v3)

> **Status:** Design (plan mode). Not yet implemented.
> **Goal:** Replace flat K-shard partitioning with a hierarchical tree that
> partitions along the data manifold, enabling better routing quality at scale
> and dynamic growth (insert/delete via leaf splits).

---

## 1. Why a Tree?

### 1.1 The problem with flat partitioning

Flat k-means at K=1024 on Sphere-IP (d_eff≈2) produces:
- **Sliver cells** (32 cells per axis on a 2D manifold → every query near a boundary)
- **Pathological skew** (min=221, max=119845, 542× ratio)
- **Recall ceiling** ~0.68 (routing coverage bottleneck)

### 1.2 Why a tree fixes this

Top-down partitioning along the manifold:
- Level 1: K₁=16-256 coarse centroids. Boundaries well-separated.
- Level 2+: Split within regions. Closure works.
- Routing: O(K₁ + K₁·K₂) FP16 distances, flat scan, stateless, parallel.
- Depth grows with N (leaves stay ~5-10k). No fixed K.

---

## 2. Architecture

### 2.1 On-disk format: custom page-based single file

**Decision:** One file. Page-allocated B+tree. BeFS/LMDB-inspired.

- **Single file** → one FD, mmap'd, no FD exhaustion.
- **Page size = 4KB** (aligned to NVMe LBA, OS page, and mmap granularity).
- **Extent allocator:** free-list + bitmap. Supports both single-page and
  N-contiguous-page (extent) allocation. Extents are critical for:
  - Leaves (18-35 contiguous pages each) → sequential NVMe reads.
  - Internal nodes (4-7 pages) → cache-locality for routing.
  This mirrors how filesystems allocate blocks (extents vs single blocks).
- **Nothing is fixed except the superblock.** The root node, internal nodes,
  leaves, and codebook are all page-allocated and can be relocated. The
  superblock stores `root_node_page` and `codebook_page` as pointers. When
  a node grows or splits, it's reallocated to a new extent and the parent's
  pointer is updated. The root's pointer is updated in the superblock.
- **mmap read-only at search time** → zero-copy, OS page cache, no pread.
- **Buffered writes** for build/vacuum (`fdatasync` at commit). No O_DIRECT
  (measured 5.2× slower for sequential streams).

**NVMe I/O strategy: sequential 1MB+ transfers, not 4KB.**
NVMe sequential bandwidth (3-7 GB/s gen4) saturates at ~128KB-1MB transfer
sizes. Below 128KB, per-command overhead (~10-20μs) dominates. Our leaves
are 70-140KB individually — below saturation.

Three mechanisms to hit full bandwidth:
1. **Extent allocation + contiguous leaf layout:** within a level-1 region,
   lay out children's leaves contiguously on disk. Scanning np leaves in a
   region becomes one sequential read (hundreds of KB to several MB).
2. **`madvise(MADV_WILLNEED)` on the full extent** before scanning — the
   kernel prefetches in large blocks.
3. **Read-ahead hints to NVMe:** the kernel issues large read commands when
   access is sequential within a mapped region.

The layout decision: **build writes leaves in tree-traversal order** (depth-
first, so siblings are contiguous). This makes the common case (probing a
few level-1 regions) hit sequential NVMe reads.

**Crash safety:** Shadow superblock (page 0 = active, page 1 = shadow).
Updates write to the shadow, `fdatasync`, then flip the active pointer
atomically via a single 4KB `pwrite`. Phase 4 adds a journal for multi-page
leaf-split transactions.

**Staying decoupled from DuckDB:** The engine owns the on-disk format.
The DB layer calls `search(query)` → candidates. A custom format doesn't
increase coupling — it makes the engine self-contained.

### 2.2 Filtered search (future, but reserved in the format)

BeFS insight: the filesystem IS a database; attributes are first-class.

For vector search: "find NN of Q where tag='science' AND year>=2020."

**Mechanism:** Each leaf stores a **filter summary** in its header:
- For numeric attributes: min/max range of the subtree (8 bytes each).
- For categorical attributes: fixed-size bloom filter (e.g., 256 bits = 32 bytes).

During routing, after selecting candidate leaves by centroid distance:
1. Check each leaf's filter summary against the query predicate.
2. Skip leaves that can't match (prune entire subtrees).
3. Internal nodes carry the union of children's summaries → prune at higher
   levels (skip an entire level-1 region if none of its leaves can match).

**Filter summaries are cheap to maintain:**
- Min/max: O(1) compare-and-swap on leaf header (update on insert/split).
- Bloom filter: O(1) set-bit on insert. Fixed-size (known FPR). No resize.
- On split: partition the bloom filter by re-inserting only the splitting
  leaf's members into the two children's filters.

**On-disk reservation:** Leaf headers reserve 256 bytes for filter summaries
from day one (Phase 1), even if unused. This avoids a format migration when
filtered search is added (Phase 5+).

**This makes us more DB-like — and that's fine.** The engine already owns
the format, allocator, tree, and search. Adding predicate-aware pruning is
a natural extension. DuckDB still does SQL/joins/projections; we make the
vector search predicate-aware.

### 2.3 Tree structure

```
                    [root]
                   /  |  \
              [L1_0] [L1_1] [L1_2] ...    ← level 1: K₁ regions
              /  \
        [L2_0] [L2_1] ...                 ← level 2: K₂ children each
         /  \
     [leaf] [leaf]                        ← leaves: 5-10k vectors each
```

- **Internal nodes:** centroid (FP16) + child page pointers + filter summary.
- **Leaves:** filter header + PQ codes (FastScan blocks) + row IDs.
- **Depth:** adaptive (1 for small N, 2-3 for large N).

### 2.4 Branching factor

- **Leaf capacity:** `leaf_capacity = 5_000–10_000` vectors.
- **Split K:** `K_split = ceil(node_size / leaf_capacity)`, clamped [2, 16].
- **Root K₁:** `K₁ = clamp(N / leaf_capacity, 16, 256)` for 2-level trees.

| N | Depth | K₁ | K_split | Leaves | Leaf size |
|---|-------|----|---------|--------|-----------|
| 100k | 1 | 16-32 | — | 16-32 | 3k-6k |
| 1M | 2 | 32-64 | 2-4 | 64-256 | 4k-8k |
| 10M | 2-3 | 64-128 | 4-8 | 256-1024 | 5k-10k |
| 100M | 3 | 128-256 | 8-16 | 1k-4k | 5k-10k |
| 1B | 3-4 | 256-512 | 8-16 | 4k-16k | 5k-10k |

### 2.5 Routing and I/O dispatch

Pure flat scan at every level. No graph, no HNSW.

1. FP16 distance to all K₁ root centroids. Sort. Take top-`probe_l0`.
2. For each: FP16 distance to children. Sort. Take top-`probe_ln`.
3. **At this point we know the full list of leaf extents to scan.** Dispatch
   prefetch for all of them before scanning the first.
4. Scan leaves sequentially (FastScan), maintain top-W heap.

Cost: ~a few hundred FP16 distances (~2μs). Stateless, parallel.
adaptive_probe_gap (B) at every level — prune far branches.

**NVMe I/O dispatch:** After routing, the candidate leaf list is fully known.
This is a key advantage of tree routing over graph search (where the next
hop depends on the current node). Three dispatch strategies:

1. **`posix_fadvise(FADV_WILLNEED)` (Phase 1, all platforms):** After routing,
   call fadvise on each leaf extent. The kernel submits async read requests
   to the NVMe block layer, which processes them in parallel across hardware
   queues (64K queue depth). By the time we finish scanning leaf 1, leaves
   2-n are in the page cache or in flight. ~3 lines of code.

**Phase 1 uses fadvise everywhere.** On macOS, `posix_fadvise` is a no-op
(the kernel's unified buffer cache handles read-ahead automatically for
mmap'd regions), so the mmap + sequential scan pattern is already optimal.
On Linux/c4a, fadvise explicitly triggers async NVMe prefetch.

2. **`io_uring` (Phase 1.5, Linux/c4a only):** Submit all leaf reads as a
   batch of async I/O requests. Maximum NVMe parallelism with zero syscalls
   per request (SQ/CQ rings). Poll CQ as we scan — if the next leaf isn't
   ready, wait; otherwise continue. The theoretical maximum. Linux-only;
   port from the fadvise implementation.

### 2.6 Cache strategy

**Phase 1: OS page cache only.** The scan path's sequential leaf access
naturally benefits from the kernel page cache + read-ahead. The tree's
contiguous leaf layout + fadvise prefetch gives sequential NVMe reads.
No application-level cache — same as today's scan path.

**The existing W-TinyLFU BlockCache is for the graph path** (random-access
beam_search hops). The scan path doesn't use it and doesn't need it —
sequential streaming and OS page cache are the right tools.

**Phase 2 (if query skew is high): leaf-level W-TinyLFU cache — separate
phase for isolated testing.** In production, popular queries hit the same
leaves repeatedly. An application-level leaf cache with admission control
(reuse the existing `BlockCache` infrastructure, keyed by leaf extent)
prevents cold one-off scans from evicting hot leaves from the page cache.
The existing `CacheController` (adaptive hill-climber) extends to manage a
third segment: graph blocks + code blocks + tree leaves.

Deliberately a separate phase so the tree's routing and I/O can be
validated independently before adding a caching layer.

This is only relevant when:
- The leaf working set is small enough to fit in RAM (hot leaves only).
- Query skew creates a meaningful hot set (not all-uniform traffic).

Decision deferred to Phase 2 based on measured cache behavior.

### 2.7 Closure replication

At every level, boundary vectors replicated via absolute-margin (SPANN-style).
Multi-level closure necessary for routing correctness. `ε` derived from
per-level d_eff. Compounding: R_total = 1 - (1-R₁)(1-R₂) ≈ 28% at 15%/15%.

### 2.8 Build path: two strategies

**Top-down (quality-first, for dynamic builds):**
```
1. PQ train + encode all N → routing codes.
2. Level-1 k-means → K₁ centroids + closure.
3. For each region: gather members, if > leaf_capacity split (level-2 k-means),
   else write leaf. Recurse depth-limited.
4. Encode 4-bit scan codes per leaf.
```

**Bottom-up (speed-first, for bulk builds):**
```
1. PQ train + encode all N → routing codes.
2. ONE k-means at leaf granularity: K_leaf = ceil(N / leaf_capacity).
3. Cluster K_leaf leaf centroids into K₁ groups (k-means on K_leaf points —
   trivial, milliseconds). No gather/re-partition of vectors.
4. Level-1 centroid = mean of children's centroids.
```

Both available. Bulk build uses bottom-up; dynamic inserts use top-down.

### 2.9 Dynamic growth / shrink (vacuum)

**Online growth (inserts):**
- Route to leaf → append → split on overflow.
- Allocate pages from free list (O(1)). `ftruncate` to extend when empty.
- Multi-page updates journaled for crash safety (Phase 4).

**Online shrink (deletes):**
- Tombstone deleted vectors in leaf (marked invalid, space not reclaimed).
- **Vacuum** (background or on-demand): compact one leaf at a time — remove
  tombstones, rewrite pages, free old pages. If post-vacuum leaf is too
  small, merge with nearest sibling.
- This is BFS/BeOS defragmentation and `mdb_compact` applied per-leaf.

**File shrink strategies (three levels):**

1. **Opportunistic tail-shrink (safe, online):** If free pages happen to be
   at the end of the file (common after bulk-delete of recent inserts),
   update `n_pages` in the superblock and `ftruncate`. A crash between the
   superblock write and `ftruncate` leaves extra pages (harmless — the free
   list says they're unallocated). Safe because the trimmed pages are
   already free.

2. **Copy-to-new-file (safe, online, full reclaim):** Write a compacted
   copy to a new file in the background. In-flight queries keep reading the
   old mmap'd file. Atomically swap (update the path → reopen). Old file
   deleted after all readers drain. This is `mdb_compact` / `VACUUM FULL`
   without the exclusive lock. Reclaims ALL free space.

3. **In-place compaction (risky, not recommended):** Moving live pages into
   free holes while queries read the mmap'd file → corruption risk without
   copy-on-write or reader locks. Avoid.

**Recommendation:** Use (1) opportunistically (cheap, common case) and (2)
for scheduled maintenance (full reclaim, like nightly VACUUM). Never (3).

**Superblock fields for growth/shrink:**
- `free_list_head`: first free page index.
- `n_pages`: total pages (file_size / page_size).
- `n_free_pages`: free list length (decide reuse vs. extend vs. tail-shrink).

---

## 3. File layout

**Nothing is fixed except the superblock.** The root, internal nodes, leaves,
and codebook are page-allocated and relocatable. Growth = reallocate + update
parent pointer. This avoids the "what if the root grows" problem.

```
Page 0:     Superblock (active)
Page 1:     Superblock (shadow)
Page 2+:    Allocation bitmap (grows as file grows)
Page 3+:    Dynamically allocated:
            ├── Root node (at root_node_page, multi-page extent)
            ├── Internal nodes (at pages referenced by parent)
            ├── Leaves (multi-page extents, contiguous within regions)
            ├── Codebook (at codebook_page)
            └── Free pages (in the free list, scattered)
```

### Superblock (4KB)
```
magic (8 bytes)                      — "SEXTANT0"
active_superblock_page (4)           — 0 or 1 (which copy is live)
root_node_page (4)                   — page index of root node extent
root_node_pages (4)                  — extent length in pages
depth (2)                            — tree depth
n_leaves (4)                          — total leaf count
n_pages (4)                           — total pages allocated
n_free_pages (4)                      — free list length
free_list_head (4)                    — first free page index
alloc_bitmap_page (4)                 — bitmap start page
codebook_page (4)                     — codebook blob start
config_toml_offset (4) + len (4)     — TOML config blob location
[padding to 4KB]
```

### Internal node page (multi-page extent)
Each node stores its children inline (centroid + page pointer + filter summary):
```
[n_children: u16]
[child_0_centroid: dim × FP16]        — inline, cache-local
[child_0_page: u32]                   — child extent start page
[child_0_pages: u16]                  — child extent length
[child_0_filter: 256 bytes]           — reserved for filtered search
[child_1_centroid, page, pages, filter]
...
```
At K_split=16, dim=768: 16 × (1536 + 4 + 2 + 256) ≈ 28KB → 7 pages.

**Node growth:** When a node grows beyond its extent (e.g., root adds a
child after a leaf split):
1. Allocate a new, larger extent.
2. Copy node data to the new extent.
3. Update parent's child pointer (or superblock if root).
4. Free old extent.

### Leaf (multi-page extent)
```
[leaf_header]
  count: u32                          — live vector count
  tombstone_count: u32                — deleted vector count
  filter_summary: 256 bytes           — reserved for filtered search
[FastScan blocks: m4 × 16 bytes / 32 vectors]
[row_ids: n × 8 bytes]
[tombstone bitmap: n bits, pad to 8 bytes]
```

**Leaf layout on disk:** Written in tree-traversal order (depth-first),
so siblings within a level-1 region are contiguous. Scanning np leaves in
a region becomes one sequential NVMe read (1MB+).

### Centroid storage: inline in node pages
Each internal node carries its children's centroids directly. No separate
centroid pool. Centroid + child pointer + filter in the same cache line.
Revisit if node extents grow too large (> 7 pages at K_split=16).

---

## 4. Manifest: TOML

Using cpptoml (header-only). Stored as a blob in the superblock.

```toml
[index]
dim = 768
m4 = 192
scan_pq_bits = 4
quantizer_type = "pq"
prq_nsplits = 0

[tree]
depth = 2
k_root = 128
leaf_capacity = 5000
n_leaves = 2000
n_probe_l0 = 16
n_probe_ln = 4

[routing]
adaptive_probe_gap = 1.606
median_lid = 13.21

[partition]
balance_factor = 4.0
sub_shard_probe_pct = 50
```

---

## 5. Search path

```cpp
search(query, k, config, filter) {
    // 1. Route at level 0: FP16 dists to K₁ root centroids. Sort.
    //    Top n_probe_l0. Gap check. Filter check (prune regions).
    // 2. For each probed node:
    //    a. Dists to children. Sort. Top n_probe_ln. Gap + filter.
    //    b. Leaves → scan (with filter predicate). Internal → recurse.
    // 3. Merge, dedup, return top-W.
}
```

B (gap) + D (budget) at every level. Filter pruning at every level.
`madvise(MADV_WILLNEED)` before each leaf scan (NVMe prefetch).

---

## 6. Dependencies

### cpptoml
- https://github.com/skystrife/cpptoml — header-only.
- Replace line-oriented manifest. Wrap in `Manifest` struct.

### cpptrace
- https://github.com/jeremy-rifkin/cpptrace
- Stack traces on crash. Signal handler in main().
- Critical for c4a debugging.

---

## 7. Implementation plan

### Phase 0: Infrastructure ✅
- cpptoml integration (submodule + meson.build).
- cpptrace integration (native meson.build, execinfo+addr2line+cxxabi backends).
- `TreeManifest` TOML struct.
- Page allocator (free-list + bitmap).
- Superblock read/write (with shadow + atomic flip).
- 16 tests, all passing.

### Phase 1: Two-level tree (bulk build, bottom-up)
- `IVFTreeIndex` (in-memory + page-file).
- Bottom-up build: leaf k-means → centroid grouping → write file.
- Search: two-level routing + leaf scan (mmap).
- **Validation:** Sphere 10M. Target: recall ≥0.70 @ QPS ≥400.

### Phase 2: Dynamic depth
- Auto-choose depth from N.

### Phase 3: Multi-level closure (top-down build)
- Closure at every level. Measure replication vs recall.

### Phase 4: Dynamic insert/delete + vacuum
- Leaf split/merge. Journal for atomicity. Tombstones + vacuum.
- **Incremental Lloyd rebalancing:** When a batch of new vectors arrives,
  route each to its nearest root centroid, accumulate per-region. When a
  region's buffer is large enough (or on schedule), run local Lloyd on
  just that root child's subtree:
  1. Re-read the region's leaf centroids from disk
  2. Project new vectors + existing members to PCA space
  3. Run 3-5 Lloyd iterations (K_leaf centroids, not K_root)
  4. Rewrite affected leaf extents
  5. Update level-1 node's child pointers + PCA leaf centroids
  - The multi-pass streaming Lloyd from Phase 1 is the REUSABLE PRIMITIVE.
    Same code, scoped to a subtree instead of the full tree.
  - Early-exit: if new vectors don't shift the partition (convergence on
    pass 1-2), skip the rewrite — no I/O needed. Cheap when distribution
    is stable.
  - At 1B scale: each root child has ~N/K_root vectors. For K_root=512,
    that's ~2M vectors per region = 6GB. May need streaming even for the
    subtree, but it's 1/K_root of the full problem.
  - Spherical k-means: renormalize centroids to unit length each iteration
    (correct for IP on sphere data — the arithmetic mean drifts off-sphere).

### Phase 5: Filtered search
- Filter summary headers. Predicate-aware pruning at every level.

### Phase 6: Per-level d_eff calibration

---

## 8. Open questions

1. **Level-2 k-means on routing codes vs original vectors?** Needs measurement.

2. **Depth threshold for 10M.** K₁=128 → 16 children → 2 levels. Measure.

3. **Routing cost at deep levels.** ~384 FP16 evals. Trivial.

4. **Closure × gap interaction.** Complementary.

5. **Page size: 4KB.** Aligned to NVMe LBA, OS page, mmap. Leaves span
   18-35 pages. **Extent allocator** gives contiguous pages for sequential
   NVMe reads. Leaves written in tree-traversal order for sibling locality.

6. **FD limit: resolved.** Single-file mmap → one FD.

7. **Centroid storage: inline in node pages** (cache locality).

8. **Vacuum scheduling.** Background thread? On-demand? Free-page ratio?
   Decision deferred to Phase 4.

9. **Node reallocation cost.** When a node grows, it's copied to a new extent.
   At K_split=16 (~28KB), this is ~7 page copies. Rare (only on split).
   Acceptable.

10. **Extent fragmentation.** Over time, extent allocation + freeing creates
    gaps. The bitmap tracks free extents; a buddy allocator could coalesce.
    Start with simple free-list; add buddy if fragmentation is measured.
