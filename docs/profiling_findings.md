# Sextant profiling findings — 2026-07-20 (PRE-FHM baseline)

> **Note (2026-07-23):** This is the PRE-optimization baseline profile. The
> session's FHM l2sq_f16 (+4%), memcpy elimination (+2%), and batch4 remainder
> (+2%) optimizations changed the profile significantly. For the current
> post-optimization profile, see the beam-search-profile memory entry and the
> bucket-queue experiment results in `ivf_phase3_closing.md`.

Profiled on GCP c4a-standard-8-lssd (Axion / Neoverse-V2, 8 vCPU), ARM SVE2
build at `-O3 -g -march=armv8-a+sve2` with clang 23. Dataset: arxiv_nomic
1.34M × 768. Index: R=32, pq_m=96, pq_bits=8 (autobuild), 536MB total
(.codes 124M + .graph 185M + .vecs 227M) — fits entirely in the BlockCache
(322MB) and phys RAM (30GB).

## Headline numbers (corrected from earlier estimates)

**Earlier `8t/8` estimates were wrong** — they conflated single-core perf with
multithread contention. Measured directly on this VM at L=400 rr=2 (recall 0.909):

| threads | QPS    | scaling vs 1t |
|---------|--------|---------------|
| 1       | 445    | 1.00×         |
| 8       | 1108   | 2.49× (31% of linear) |

At 8 threads we lose **~2452 QPS** (3560 ideal − 1108 actual) to contention.

## The real gap to VIBE peers (1-thread, not 8t/8)

| peer             | QPS (single-core) | gap to Sextant 1t (445) |
|------------------|-------------------|-------------------------|
| DiskANN          | 711               | 1.60×                   |
| SVS-LVQ          | 1280              | 2.88×                   |
| SVS-LeanVec      | 1944              | 4.37×                   |

The single-core gap to DiskANN is **small and tractable** (1.6×). LVQ/LeanVec
are further but they use better quantization — apples to oranges.

## Finding 1: 8t scaling is broken by atomic cache-line bouncing

### Evidence

Top cycle consumers at 8t (vs 1t):

| function | 1t % | 8t % | Δ |
|----------|------|------|---|
| `__aarch64_ldadd8_relax` (atomic fetch_add) | 1.40 | **30.61** | **+22×** |
| `TLBlockCache operator[]` (L1 hash map) | — | **9.81** | new |
| `__aarch64_ldadd8_acq_rel` (atomic RMW) | 1.08 | **6.77** | +6× |
| `__aarch64_cas4_acq` (CAS) | — | **3.27** | new |
| `PagedNodeStore::batched_read` | 0.61 | 3.60 | +6× |
| `nsync_mu_lock` + `nsync_mu_unlock` | — | **2.82** | new |
| `lut_distance` (PQ eval) | **30.25** | 5.02 | −6× (relative) |

**At 8t, ~52% of CPU cycles are spent in atomics + locks** (30.61 + 9.81 +
6.77 + 3.27 + 1.47 + 1.35 + 1.17 + 0.98).

### Root cause

`BlockCache::record_access(bool hit)` is called on **every cache lookup**
(5 call sites in `CacheShard`) and unconditionally does:

```cpp
void BlockCache::record_access(bool hit) {
    if (hit) hc_hits_in_sample_.fetch_add(1, std::memory_order_relaxed);
    else     hc_misses_in_sample_.fetch_add(1, std::memory_order_relaxed);
    maybe_climb();
}
```

`hc_hits_in_sample_` and `hc_misses_in_sample_` are **single atomic counters
shared across all shards and all threads**. Every core that touches the cache
bounces this cache line. At 1.34M-vector scale with ~2.3M L1 hits + 2.2M L1
misses per 1000-query bench, that's ~4.5M atomic ops competing for one line.

Call-graph confirms the chain (8t profile):
- `CacheShard::lookup` → `BlockCache::record_access` → `fetch_add` → `__aarch64_ldadd8_relax` (6.91%)
- `ScopedWriteLock` constructor → `nsync_mu_lock` (6.72%) — shard write lock itself

### Why this matters for the sports-car (1B scale) case

At 1B scale the index won't fit in RAM, so cache **misses** will dominate —
but `record_access` is called on **every** lookup (hit or miss). The atomic
contention is structural, not workload-dependent. Fixing it helps both the
warm-cache benchmark and the cold-cache production case.

### Recommended fixes (priority order)

1. **Make `record_access` thread-local.** Each thread accumulates hits/misses
   in a `thread_local uint64_t`, flushes to the shared atomic every N (e.g.
   1024) calls or at end of query. Drops the per-call atomic entirely.
   **Expected impact: recovers most of the 30.61% `ldadd8_relax` cost.**

2. **Eliminate the per-shard write lock for lookups.** Currently
   `CacheShard::lookup` takes a write lock even on pure reads (because LRU
   updates mutate the list). Switch to a concurrent hash map with epoch-based
   reclamation (like F14 or `folly::ConcurrentHashMap`), or move LRU tracking
   to a separate periodic pass. Drops `nsync_mu_lock` cost (~6.72%).

3. **Pad atomic counters to their own cache line.** Even if (1) and (2) are
   done, any remaining shared atomics should be `alignas(64)` to prevent
   false sharing with adjacent fields.

4. **Investigate `TLBlockCache` (L1) sharing.** The 9.81% in `_Map_base::operator[]`
   suggests the L1 thread-local cache isn't fully thread-local. Verify
   `TLBlockCache` is truly per-thread (not a shared map). If shared, fix.

## Finding 2: Single-core bottleneck is PQ LUT eval, not rerank

### Evidence

At 1t warm rr=2: `lut_distance` consumes **30.25%** of cycles. At 1t rr=10
(10× more rerank work), it's still **28.52%** — slightly smaller share but
still #1. The rerank work doesn't shift the bottleneck.

Top 1t warm rr=2 consumers:

| function | % cycles |
|----------|---------|
| `PqQuantizer::lut_distance` | 30.25 |
| `beam_search_into::$_3` (dist_to lambda) | 16.27 |
| `beam_search_into` (loop body) | 15.57 |
| `std::thread::_State_impl::_M_run` (harness) | 8.18 |
| `MemGraph::fp16_ptr` | 3.37 |
| `PqQuantizer::preprocess_query` | 2.81 |
| `nk_sqeuclidean_f32_sve` (FP16 dist) | 2.66 |
| `MemGraph::unpin_code` | 2.06 |
| `CacheShard::lookup` | 1.45 |
| atomics (ldadd8_*) | 2.48 |

### Recommended fixes (priority order)

1. **Audit `lut_distance` SIMD.** At 30% of cycles, even a 20% improvement
   here is +6% overall single-core QPS. Compare against numkong's optimized
   dot products — the SVE2 gather path in `lut_distance` may be leaving
   bandwidth on the table. Use `perf annotate` on the function to see which
   SVE2 instructions dominate. **High impact, medium effort.**

2. **Investigate the `dist_to` lambda (16.27%) + search loop (15.57%).**
   Together that's 32% in the beam_search machinery. Some of this is
   unavoidable (heap ops, neighbor iteration), but verify there's no
   accidental pointer-chase or redundant work. `MemGraph::unpin_code` at
   2.06% is suspicious — unpin should be a no-op in the warm-cache case.

3. **`MemGraph::fp16_ptr` at 3.37%.** This is the FP16 vector lookup for the
   MemGraph ball. Verify it's a direct array index (O(1)) — if it's doing
   any hash lookup or branch-heavy check, simplify.

## Finding 3: L1 cache hit rate is only 51%

The bench log reports `tl-l1: hit_rate=51.0%` — half of all lookups miss the
thread-local L1 and fall through to the L2 shard (with its mutex + atomic
`record_access`). Improving L1 hit rate would directly reduce contention.

### Recommended investigation

- Is the L1 eviction policy appropriate? (LRU vs FIFO vs size)
- Is the L1 sized correctly? (currently unknown — needs a config knob)
- Are the access patterns amenable to L1 caching at all? Graph traversal
  is inherently random-ish; the working set per query may exceed L1.

If L1 can't be improved, the contention fixes in Finding 1 become more
important (they make L2 lookups cheap).

## Finding 4: Hardware cache counters unavailable on GCP Axion

`perf stat -e L1-dcache-load-misses,LLC-load-misses,...` returns
`<not supported>` on c4a VMs. The Axion hypervisor doesn't expose the PMU
cache events. Same for `perf c2c` (needs SPE/PEBS, also unavailable).
`arm_spe_0` is missing from `/sys/bus/event_source/devices/`.

Workaround: infer cache behavior from function hot spots. The 8t profile
strongly implies cache-line bouncing (the `ldadd8_relax` cost only makes
sense if the line is being invalidated constantly).

## Finding 5: `MemGraph::unpin_code` is a real cost (2.06% 1t)

`unpin_code(uint32_t)` appears in the hot path at non-trivial cost. In the
warm-cache case, unpin should be a no-op (no I/O completion to signal, no
buffer to release). Either it's doing real work unnecessarily, or it's
inlined something expensive.

### Recommended investigation

Read the `MemGraph::unpin_code` implementation. If it's more than a return
statement, simplify for the warm-cache case. This won't move the needle
alone but it's free perf.

## Summary of recommendations

| lever | impact | effort | sports-car safe? |
|-------|--------|--------|------------------|
| **Thread-local `record_access` counters** | HIGH (recovers ~30% of 8t cycles) | LOW | YES (structural, helps any workload) |
| **Remove per-shard write lock on lookup** | HIGH (~6% of 8t cycles) | MEDIUM | YES (concurrent map or epoch-based reclamation) |
| Verify `TLBlockCache` is truly thread-local | MEDIUM (~10% of 8t cycles if not) | LOW | YES |
| **Audit `lut_distance` SIMD** | MEDIUM (~6% of 1t cycles per 20% gain) | MEDIUM | YES |
| Pad shared atomics to cache lines | LOW | TRIVIAL | YES |
| Improve L1 cache hit rate (51% → higher) | MEDIUM (reduces L2 contention) | MEDIUM | YES |
| Investigate `MemGraph::unpin_code` | LOW (~2% of 1t cycles) | LOW | YES |

## What we did NOT measure

- **Cold-cache profile** (small BlockCache to simulate 1B-scale miss path).
  Skipped because the warm-cache 8t profile already shows the bottleneck is
  atomic contention in `record_access`, which is called on every lookup
  regardless of hit/miss. The fix (thread-local counters) helps both cases.
  Can revisit if the fix doesn't extrapolate as expected.
- **Hardware cache-miss rates** — PMU cache events not exposed by Axion.
- **`perf c2c` false-sharing detection** — needs SPE, not exposed.

## Profile artifacts

- `results/profiling/1t_cycles_top.txt` — 1t warm rr=2, top cycle consumers
- `results/profiling/8t_cycles_top.txt` — 8t warm rr=2, top cycle consumers
- `results/profiling/1t_rr10_top.txt` — 1t warm rr=10, top cycle consumers
- `perf.data` files left on the VM (now deleted) — full call-graph data

## Recommended next plan

**Attack the multithread scaling first.** It's the largest single lever
(recovers up to ~2452 QPS at 8t) and the fix (thread-local counters) is
small, well-isolated, and production-safe. After that:

1. `record_access` → thread-local with periodic flush
2. `CacheShard::lookup` → drop the write lock for reads
3. Re-profile 8t, measure scaling ratio (target: >70% of linear)
4. Then revisit single-core: audit `lut_distance` SIMD, then EP follow-ups
