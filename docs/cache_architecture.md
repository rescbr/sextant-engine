# Search Cache Architecture

> Design doc — 2026-07-12. Synthesizes W-TinyLFU eviction, thread-local
> L1 caching, and our existing MemGraph/PageShuffle/batched-pread stack.
>
> Status: Phase 1 (W-TinyLFU) and Phase 2 (TL-L1 + allocation fixes) are
> implemented. Phase 3 (DiskANN++ PageHeap) was investigated, prototyped,
> profiled, and **abandoned** — see below.

## The problems

1. **Multi-threaded cache thrashing** — 4 threads on 32MB shared cache were
   slower than 1 thread (133 vs 191 QPS). Two root causes:
   - Cross-thread eviction in the shared LRU.
   - Per-query allocation churn (4MB VamanaTLS + priority_queue reallocation).

2. **Low page utilization** — DiskANN++ showed plain caching gets only 10-20%
   hit rate because cached pages are "unused for node expansion." The bottleneck
   isn't how many pages you cache, it's how many *useful vertices* each cached
   page delivers to the search frontier. **Addressed by PageSearch + PageShuffle**
   (co-located nodes are scanned at fetch time). The DiskANN++ PageHeap approach
   was investigated and abandoned — see Phase 3 below.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              SEARCH QUERY                 │
                    └────────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
         L0         │           MemGraph (SSSP cache)           │
      (topology)    │  Entry-point BFS neighborhood in RAM      │
                    │  ~49% of nodes, shared, read-only         │
                    └────────────────────┬─────────────────────┘
                               miss (~51%) │
                    ┌────────────────────▼─────────────────────┐
         L1         │    Thread-local block cache (per-thread)   │
      (isolation)   │  Pointer-based, epoch-validated            │
                    │  8 slots, lock-free, FIFO replacement      │
                    │  Eliminates cross-thread L2 lock on hits   │
                    └────────────────────┬─────────────────────┘
                               miss       │
                    ┌────────────────────▼─────────────────────┐
         L2         │     BlockCache (W-TinyLFU eviction)        │
     (eviction)     │  Scan-resistant: speculative pre-read      │
      W-TinyLFU     │  blocks enter window, TinyLFU admission    │
                    │  gates entry to main (probation→protected) │
                    │  Hill-climbing adaptive window (per-cache) │
                    │  Per-shard BlockBufferPool (no madvise)    │
                    └────────────────────┬─────────────────────┘
                               miss       │
                    ┌────────────────────▼─────────────────────┐
         L3         │    Batched pre-read (4 blocks / pread)      │
      (I/O amort)   │  PageShuffle ensures BFS co-location       │
                    └───────────────────────────────────────────┘
```

### L0: MemGraph (unchanged)
Topology-aware SSSP cache. Shared, read-only. ~49% hit rate on SIFT-1M.

### L1: Thread-local block cache ✅ DONE

Pointer-based TLBlockCache with epoch validation. Each search thread gets
its own 8-slot private cache that stores pointers into L2's memory. A global
epoch counter detects stale pointers when L2 evicts blocks.

- **86% hit rate** on SIFT-1M (32MB L2, 1 thread).
- **Lock-free** lookup (linear scan of 32 key entries, ~4 cache lines).
- **No data copy** — stores pointers, not 256KB block copies (copy-based
  design was tried and abandoned: it thrashed the CPU L2/L3 cache).
- **Per-shard epoch validation** — an eviction in shard K only invalidates
  L1 entries from shard K (not all entries). This allows a larger L1
  capacity (32 slots) without the false-invalidation churn that crippled
  the global-epoch design.

### L2: BlockCache with W-TinyLFU eviction ✅ DONE

Replaces the original plain LRU. Adapted from Caffeine's W-TinyLFU design.

**Why W-TinyLFU over ARC/2Q:**
- ARC is patented by IBM.
- W-TinyLFU is Apache 2.0, self-tuning, near-optimal hit rate.

**Three-list eviction:**
```
Window (adaptive %, starts at 1%)    → plain LRU, no admission filter
  ↓ overflow (TinyLFU admission check)
Probation (20% of main)              → SLRU A1
  ↓ second access
Protected (80% of main)              → SLRU A2
```

**TinyLFU admission:** When the window overflows into a full main cache, the
candidate's historic frequency (via 4-bit CountMinSketch) is compared to the
probation victim's frequency. The more frequent one wins.

**Hill-climbing adaptive window:** Lives at the BlockCache level (NOT
per-shard). Aggregates hit/miss counters across all shards, samples over
10×total_capacity accesses, computes one window ratio, writes atomic limits
on all shards. Each shard lazily reconciles on its next operation (no
cross-shard lock acquisition → deadlock-free).

**BlockBufferPool:** Per-shard pool of pre-allocated 256KB aligned buffers.
Recycles on eviction instead of going through the system allocator. Eliminates
the `aligned_alloc`/`aligned_free` → `madvise` syscall churn that dominated
under high eviction rates.

### L3: Batched pre-read (existing, T3)
Reads 4 blocks (1MB) per miss. PageShuffle ensures BFS co-location. No change.

## Implementation history

### Phase 1: W-TinyLFU eviction for L2 ✅ DONE
- FrequencySketch: 4-bit CountMinSketch ported from Caffeine.
- Three-list eviction with TinyLFU admission.
- Hill-climbing adaptive window (cache-level, not per-shard).
- Benchmark: 4-thread 32MB cache 133→271 QPS.

**Files:** `src/storage/block_cache.hpp/.cpp`, `src/storage/frequency_sketch.hpp`,
`test/test_wtinylfu.cpp`

### Phase 2: Thread-local L1 + search path fixes ✅ DONE
- **TLBlockCache** (`src/storage/tl_cache.hpp`): pointer-based, epoch-validated.
- **BlockBufferPool** (`src/storage/block_buffer_pool.hpp`): per-shard buffer recycling.
- **Thread-local VamanaTLS**: search() reuses TLS across queries (was allocating
  4MB visited_flags per query — the #1 multi-threaded bottleneck).
- **Pre-reserved heaps** (`SearchScratch`): priority_queue replaced with manual
  binary heaps over thread_local scratch (no reallocation).
- Benchmark: 1 thread 191→306 QPS (+60%), 4 threads 133→300 QPS (+125%).
  Recall 0.9970.

**Files:** `src/storage/tl_cache.hpp`, `src/storage/block_buffer_pool.hpp/.cpp`,
`src/algo/vamana_core.cpp` (SearchScratch + thread_local TLS)

### Phase 3: DiskANN++ PageHeap — INVESTIGATED & ABANDONED

**What it is:** DiskANN++'s PageHeap — actively extract useful vertices from
 cached pages and feed them into the search candidate set via `Pop()`. Turns
 the cache from passive storage into an active source of candidates.

**Status: Fully prototyped, profiled, and abandoned (2026-07-12). Code reverted.**

The investigation proceeded in three phases, each revealing a distinct issue:

**Phase 3a — QPS dropped 50%:**
The initial implementation registered *code blocks* (8192 nodes each, since
`codes_per_block = kBlockSize / code_size = 262144 / 32 = 8192`). The plan had
assumed ~940 nodes/block, but that was *graph* blocks (`node_size = 208B`).
Result: 122M `lut_distance` calls, 99.97% unused. Fixed by registering
graph-block nodes capped at `params_.R`, but this revealed the deeper issue.

**Phase 3b — PageHeap became a no-op:**
After the fix, PageHeap pushed 0 candidates. Reason: PageSearch (which fires on
`from_ssd=true` blocks) already scans the same graph-block co-located nodes and
marks them visited. PageHeap and PageSearch scan the *same* nodes — PageHeap is
redundant.

**Phase 3c — Simulation proved no ceiling exists:**
A `--page-heap sim` mode was added (disable PageSearch, register ALL touched
blocks including cached ones, track candidate competitiveness without
injecting). Results (SIFT-1M, 32MB cache, 1000 queries):

| Mode | recall  | QPS   | graph_reads |
|------|---------|-------|-------------|
| off  | 0.9967  | 311.9 | 33311       |
| all  | 0.9971  | 264.8 | 36045 (+8%) |

- Only **3.5%** of popped candidates were competitive (closer than W's L-th).
- `graph_reads` **never decreased** — they increased or stayed flat.
- Recall differences were within noise.

**Why it's architecturally redundant:**

It's PageShuffle, not the dataset or macOS page cache. PageShuffle co-locates
graph neighbors in disk blocks, which means:
1. When a block is fetched (`from_ssd=true`), PageSearch immediately scans its
   co-located nodes — which ARE the graph neighbors.
2. By the time the block is cached (`from_ssd=false`), all useful nodes are
   already visited. Nothing remains for PageHeap to surface.

**Skewed datasets don't change this.** Popularity skew helps MemGraph (the
hot path is already in RAM). Distributional skew (tight clusters) doesn't
create a "warm but not hot" tier — if a node is accessed enough to stay cached,
it's either in MemGraph or was a one-off cold read that PageSearch handled.

**The three-tier design is complete:** MemGraph (hot path) → L1/L2 + PageSearch
(cold path with locality exploitation) → SSD. PageHeap sat in a gap that doesn't
exist. See `memory/pageheap-impl.md` for the full investigation notes.

## Performance summary (SIFT-1M, 32MB cache)

| Threads | Original (plain LRU) | After all fixes |
|---|---|---|
| 1 | 191 QPS | 322 QPS (+68%) |
| 4 | 133 QPS (thrashing) | 339 QPS (+154%) |
| Recall | 0.9970 | 0.9970 |

## Open questions / future work

1. ~~**Per-shard epoch:**~~ ✅ DONE. Per-shard epoch implemented. L1 capacity
   increased from 8→32 slots. Hit rate improved 69%→86% (1 thread).

2. ~~**Sync PageHeap:**~~ ❌ ABANDONED. Fully investigated, prototyped, and
   profiled. Architecturally redundant with PageShuffle + PageSearch — see
   Phase 3 above. Do not re-implement.

3. **W-TinyLFU window ratio at scale:** The hill-climber adapts, but the
   optimal ratio may differ at 100M/1B scale where the cache-to-graph ratio
   is much smaller.
