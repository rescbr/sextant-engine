# Sextant — Remaining Tasks

> Status as of 2026-07-12. Phase 1 is functionally complete (build, search,
> insert, partitioned build, CLI, 71 tests passing) with search-path R&D
> applied (PageShuffle, PageSearch, MemGraph, DynamicWidth, PinResult).
> This document lists what remains, ordered by impact.

## SIFT-1M validated performance

| Metric | Target | Current | Status |
|---|---|---|---|
| Recall@10 | ≥ 0.95 | 0.9957 | ✅ |
| Search QPS (full cache) | — | ~2500 (single-thread); multi-threaded now available via `--threads` | ✅ |
| Search QPS (32MB cache) | — | ~480 (single-thread); macOS page cache masks true I/O | ⚠️ multi-threaded + batched pre-read now available |
| Build time | < 30s | **113.7s** (was 190s; PQ parallel + direct-SDC + work-stealing) | ❌ 3.8× over target; code_distance is 60% of construct |
| Idle RAM | < 100 MB | not measured | deferred — see note below |

**Note on idle RAM:** The `< 100 MB` gate is deferred until all performance
work is complete. MemGraph caches 34.5% of nodes (~83MB for SIFT-1M) in RAM
for search performance. At billion scale, MemGraph + LRU will use significantly
more than 100MB. The gate will be revisited after multi-threaded search and
build optimization are done — we will strike a balance between RAM usage and
performance rather than chasing an arbitrary number.

---

## HIGH PRIORITY — Performance

### T1: Multi-threaded search

**Problem:** The benchmark and CLI search run queries single-threaded. On a
10-core machine, this leaves 90% of CPU idle. Each query is independent — no
shared state mutation during search (MemGraph and PagedNodeStore are both
thread-safe via nsync locks).

**Solution:** Add a thread pool for concurrent query execution. The benchmark
tool (`tools/benchmark.cpp`) should spawn N threads, each pulling queries from
a shared queue. The CLI search command should accept `--threads N`.

**Expected impact:** ~10× QPS on multi-core (linear scaling until I/O-bound).

**Files:** `tools/benchmark.cpp`, `tools/sextant_cli.cpp`, possibly
`include/sextant/engine.hpp` (batch search API).

**Effort:** Small — the search path is already thread-safe. Just add the
dispatch layer.

### T2: Build optimization (190s → 30s target)

**Problem:** SIFT-1M build takes ~190s (target: <30s). DiskANN builds SIFT-1M
in ~300s on 8 cores; we're at 770s on 10 cores (extrapolated from 100K). The
build is CPU-bound (`user` time >> `wall` time).

**Diagnosis (from sextant-build-notes.md):**
- Per-node CPU time: ~5190μs across 10 threads
- Each `insert_build_from_code` does:
  1. `build_code_lut`: 32 memcpy's of 256 floats (fast, ~32KB copy)
  2. `beam_search(L=100)`: visits ~100 nodes × R=48 neighbors each =
     ~4800 `lut_distance` calls @ ~100ns = ~480μs
  3. `robust_prune`: O(R²) `code_distance` calls through a 2MB
     cross_distance_table (cache-cold strided access)
  4. `connect_and_prune`: 48 lock ops on the sharded Mutex pool (40 locks,
     10 threads → contention)

**Potential fixes (ordered by expected impact):**
1. **RobustPrune LUT optimization:** instead of strided `code_distance` calls
   through the 2MB cross_distance_table, build a per-anchor LUT (like
   `build_code_lut`) so the inner loop reads from an L2-resident m×K buffer.
   This is the single biggest expected win.
2. **Prefetch cross_distance_table rows:** software prefetch ahead of the
   RobustPrune inner loop.
3. **Parallelize the encode pass:** currently serial (pass 2 loops through
   all vectors single-threaded). This is ~10% of build time.
4. **Profile with samply:** the build is too slow for intuition — need hard
   data. Run samply on a 100K-vector build (77s) to identify hotspots.

**Files:** `src/algo/vamana_core.cpp` (RobustPrune, beam_search),
`src/quant/pq_quantizer.cpp` (code_distance, build_code_lut).

**Effort:** Medium — the RobustPrune LUT is algorithmic but well-understood.

### T3: Batched pre-read on cache miss

**Problem:** On a PagedNodeStore cache miss, we do one `pread` of 256KB.
The plan (Step 2) specifies: "on cache miss, pread requested block + N
neighbors (default 4 blocks = 1 MB) in one syscall." This exploits graph
locality — after PageShuffle, adjacent blocks on disk contain nearby graph
nodes.

**Solution:** In `PagedNodeStore::get_node_block`, on a miss, read the
requested block plus the next 3 blocks in a single `pread` (1MB total).
Insert all 4 into the LRU. This amortizes syscall overhead and exploits
NVMe sequential-read bandwidth.

**Files:** `src/storage/node_store.cpp` (get_node_block, get_code_block).

**Effort:** Small — straightforward extension of the existing read path.

---

## MEDIUM PRIORITY — Robustness / Completeness

### T4: Crash recovery / orphan detection

**Problem:** The masterplan (lines 281-292) specifies:
- Manifest present + sidecar files match UUIDs → valid.
- Manifest present but sidecar files missing → corrupted, throw.
- **Manifest absent + sidecar files present (crash during build) → throw an
  error listing the orphaned files.** Never delete silently.

Currently, `Engine::open()` does not check for orphaned files. A crashed
build leaves `.graph`, `.codes`, `.meta` without a `.manifest`. Opening
these should error with a clear message.

**Solution:** In `Engine::open()`, before loading sidecars, check:
1. If `.manifest` exists → proceed (validate UUIDs against sidecars).
2. If `.manifest` absent but sidecar files exist → throw Error listing them.
3. If `.manifest` absent and no sidecars → throw "index not found."

**Files:** `src/engine/search.cpp` (open/load_sidecars).

**Effort:** Small — filesystem checks + error messages.

### T5: ADC build mode — REMOVED

ADC (Asymmetric Distance Computation) build mode has been removed. The FP16
prune hybrid made it redundant (only +0.0003 recall at 1.4× build cost). The
`BuildMode` enum is retained with `SDC` as the sole value for API compat.
The `--build-mode` CLI flag is kept as a deprecated no-op (accepts `sdc` only,
errors on `adc`).

### T6: Entry point persistence

**Problem:** `Engine::open()` recomputes entry points via
`compute_entry_points()` instead of loading them from `.meta`. This works
(the computation is deterministic) but wastes time and produces a different
set if the algorithm changes.

**Solution:** The entry points ARE serialized to `.meta` (by flush). The
issue is that `VamanaCore::entry_points_` has no public setter, so `open()`
can't inject them. Add a `set_entry_points()` method or load them directly
during VamanaCore construction.

**Files:** `src/algo/vamana_core.hpp` (add setter or constructor param),
`src/engine/search.cpp`.

**Effort:** Small.

---

## LOW PRIORITY — Deferred / Scale-dependent

### T7: Async I/O pipeline (PipeSearch)

**Problem:** Each cache miss blocks on synchronous `pread`. The PipeSearch
technique (PipeANN, OSDI 2025) decouples I/O from compute: issue speculative
reads for frontier candidates while current reads are in flight.

**Why deferred:** Requires async I/O (io_uring on Linux, dispatch_io on
macOS). macOS `F_NOCACHE` doesn't give true direct I/O — the kernel page
cache masks the I/O bottleneck, making it impossible to validate on macOS.
Needs Linux with `O_DIRECT` for real testing.

**When to revisit:** After multi-threaded search (T1) and build optimization
(T2) are done, and when we have Linux validation infrastructure.

### T8: BIGANN-100M validation

**Problem:** SIFT-1M is validated but BIGANN-100M is not. The plan targets
Recall@10 ≥ 0.85, build < 15 min at 100M scale.

**Why deferred:** Requires downloading 13GB dataset. Build at 100M with
current 190s-per-1M speed would take ~5.3 hours. Needs build optimization
(T2) first. Also needs Linux for meaningful I/O benchmarks.

### T9: samply profiling infrastructure

**Problem:** The plan says "profile every build/search benchmark." We have
samply installed but no profiling scripts or automated profile collection.

**Solution:** Write `scripts/profile_build.sh` and
`scripts/profile_search.sh` that run samply on smaller datasets and save
profiles. Integrate into CI.

**Effort:** Small but needs the build optimization (T2) work to be meaningful.

---

## Summary table

| ID | Task | Priority | Effort | Status |
|---|---|---|---|---|
| T1 | Multi-threaded search | HIGH | Small | ✅ Done (benchmark + CLI `--threads`, lock-free atomic dispatch) |
| T2 | Build optimization (190s→30s) | HIGH | Medium | 🔶 Partial — PQ parallelized + direct-SDC + work-stealing (190s→114s); code_distance is 60% of construct |
| T3 | Batched pre-read (4 blocks/miss) | HIGH | Small | ✅ Done (`PagedNodeStore::batched_read`, 1MB/miss) |
| T4 | Crash recovery / orphan detection | MEDIUM | Small | ✅ Done (manifest + orphan checks in `Engine::open`) |
| T5 | ADC build mode CLI flag | MEDIUM | Medium | ✅ Done (`--alpha 1.5` triggers ADC, raw vectors loaded for construct) |
| T6 | Entry point persistence | MEDIUM | Small | ✅ Done (`set_entry_points` + end-to-end test) |
| T7 | Async I/O pipeline (PipeSearch) | LOW | Large | ❌ PageHeap abandoned — architecturally redundant with PageShuffle+PageSearch (see `docs/cache_architecture.md` Phase 3). Async overlap not worth pursuing |
| T8 | BIGANN-100M validation | LOW | Large | Not started (blocked on T2) |
| T9 | samply profiling scripts | LOW | Small | ✅ Done (`scripts/profile_build.sh`, `profile_search.sh`, `scripts/analyze_profile.py`) |

## What's done (for reference)

- ✅ Steps 1-12 (Phase 1 code: skeleton, storage, PQ, Vamana, build, search,
  insert, CLI, partitioned build)
- ✅ SIFT-1M Recall@10 = 0.9957 (CI gate ≥ 0.95)
- ✅ Partitioned build K-sweep (K=1,4,8,15,29 — graceful degradation)
- ✅ PageShuffle (BFS reordering at flush)
- ✅ PageSearch (per-pin from_ssd gating via PinResult)
- ✅ MemGraph (topology-based RAM cache, 2700 QPS on SIFT-1M)
- ✅ DynamicWidth (two-phase beam width)
- ✅ NodeStore abstraction (FlatNodeStore, PagedNodeStore, MemGraph)
- ✅ CLI: build, search, insert, benchmark, fvecs_to_fbin
- ✅ CLI flags: --R, --L, --alpha, --pq-m, --pq-bits, --metric, --threads,
  --build-ram, --inline-pq, --cache-size, --log-level, --explain
- ✅ Documentation: README, design_decisions.md, LICENSE (SSPL-1.0-only)
- ✅ 71 tests across 10 suites, all passing
- ✅ Multi-threaded search (T1): `--threads` on benchmark + CLI, lock-free dispatch
- ✅ Batched pre-read (T3): 4 blocks (1MB) per cache miss via `batched_read`
- ✅ Crash recovery (T4): orphan sidecar detection in `Engine::open`
- ✅ Entry point persistence (T6): `set_entry_points` + `.meta` round-trip
- ✅ samply profiling scripts (T9)
- ✅ PQ training parallelized (T2): 70%→4.5% of build time
- ✅ Direct-SDC distance (T2): eliminated LUT materialization, construct 153s→105s
- ✅ Hot-path read counters converted from mutex to relaxed atomics
- ✅ CTPL `thread_pool_tls` UB fix (null TLS deref when no init function)
- ✅ Per-phase build timing instrumentation
- ✅ Dynamic work-stealing for construct (was static partitioning)
- ✅ W-TinyLFU cache with adaptive hill-climbing (replaces plain LRU)
  - FrequencySketch (4-bit CountMinSketch, ported from Caffeine)
  - Three-list eviction: window → probation → protected
  - Hill-climber at BlockCache level (per-workload, not per-shard)
  - BlockBufferPool: per-shard, eliminates aligned_alloc/madvise churn
- ✅ Thread-local L1 cache (pointer-based, epoch-validated)
- ✅ Search path allocation fixes:
  - Thread_local VamanaTLS (was 4MB/query allocation)
  - Pre-reserved heaps via SearchScratch (no priority_queue reallocation)
- ✅ Search QPS: 1 thread 191→306, 4 threads 133→300 (32MB cache)
- ✅ Cache architecture design doc (`docs/cache_architecture.md`)
- ✅ samply profile analysis tool (`scripts/analyze_profile.py`)
- ✅ Adaptive PQ selection (Issue 46/47): auto `(m, bits)` via reservoir probe
  - Joint sweep of candidate (m, bits) configs on 20K reservoir subset
  - Recall floor + three cache bands (L2/L3/RAM) + max recall within band
  - 4-bit at high m auto-selected for clustered data (se_base: m=192/4, 0.97 recall, L2 table)
  - Cross-platform cache detection (`src/util/cache_info.{hpp,cpp}`): sysctl/sysfs
  - `--pq-bits auto` (CLI default), `--pq-recall-floor`, `--probe-sample`
  - `--explain` runs the probe and shows full comparison table (no build)
  - `pq_explore` tool: m/bits sweeps, per-segment diagnostics, per-partition analysis
  - `PqQuantizer::m_` widened to uint16_t (was uint8_t, capped at 255)
