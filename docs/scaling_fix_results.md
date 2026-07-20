# Multithread scaling fix — results

Three-step plan to fix the atomic cache-line bouncing villain identified
in `docs/profiling_findings.md`. All measurements on c4a-standard-8-lssd
(Axion / Neoverse-V2, ARM SVE2) with `-O3 -g` build, arxiv_nomic 1.34M,
L=400 rr=2 (recall 0.9088-0.9092 across all runs — no recall regression).

## Headline results

| metric | baseline | step 1 | step 2 | step 3 | total Δ |
|--------|---------|--------|--------|--------|---------|
| 1t QPS | 445 | 436 | 426 | **461** | +3.6% |
| 8t QPS | 1108 | 1281 | 1554 | **1671** | **+50.8%** |
| scaling (% of linear) | 31% | 37% | 46% | 45% | +14pp |
| recall@100 | 0.9086 | 0.9092 | 0.9088 | 0.9088 | unchanged |

## Per-step breakdown

### Step 1 (commit `66f4ac6`): thread-local stat counters

`BlockCache::record_access` and `PagedNodeStore::tl_hits_/tl_misses_` both
did a `fetch_add` on shared atomics per cache lookup. Thread-local
accumulation with periodic flush (every 64 calls) eliminated the per-call
atomic.

`__aarch64_ldadd8_relax` dropped from **30.61% → <1.5%** of cycles at 8t.
8t QPS: 1108 → 1281 (+15.6%).

### Step 2 (commit `416ce1e`): read-lock lookup + atomic sketch

`CacheShard::lookup` took the exclusive write lock on every read (LRU
mutation). Switched to `ScopedReadLock`; the W-TinyLFU sketch became fully
atomic (CAS-based, Caffeine-style) so concurrent readers don't need the
write lock. Dropped LRU bumps on hit (sketch frequency alone discriminates
under graph-traversal access skew).

`nsync_mu_lock` (exclusive) dropped out of the top hot spots, replaced by
`nsync_mu_rlock`+`runlock` (parallel reader pair). 8t QPS: 1281 → 1554
(+21.3%). Scaling: 37% → 46%.

### Step 3 (commit `447cdc0`): cached TLBlockCache& + deadlock fix

Two changes in the L1 hot path:

1. Cache `TLBlockCache*` and `L1Counters*` in a thread_local `TLFastPath`
   keyed by `this`. Eliminated the two per-call hash lookups (`g_tl_caches[]`
   and `tl_l1_counters_[]`) that were causing the 1t regression seen in
   steps 1+2.

2. **Deadlock fix** uncovered by step-3 testing: `batched_read` was holding
   `ScopedReadLock` and then calling `shard.lookup()` which takes its own
   `ScopedReadLock` internally — recursive read locking on `nsync_mu`.
   Under nsync's writer-preference fairness, this deadlocked when a writer
   was waiting (intermittent 8t hangs). Fixed by using `lookup_unlocked()`
   when the caller already holds the lock.

1t QPS: 426 → 461 (+8.3%, fully recovers the step-1 regression and is
+3.6% vs baseline). 8t QPS: 1554 → 1671 (+7.5%).

## Before/after profile (8t, top cycle consumers)

```
                              baseline    step 3
lut_distance (PQ eval)          5.02%    23.67%   ← real work, now #1
beam_search_into (main+lambda) 15.80%    31.02%   ← real work
nsync_mu_rlock + runlock         —        7.49%   ← reader pair (parallel)
__aarch64_ldadd8_relax         30.61%     2.78%   ← ELIMINATED
__aarch64_ldadd8_acq_rel        6.77%    <1.5%    ← ELIMINATED
TLBlockCache operator[]          9.81%    <1.5%    ← ELIMINATED
nsync_mu_lock (exclusive)        1.47%    <1.5%    ← ELIMINATED
cas4_acq (sketch CAS)            3.27%     1.37%
```

Atomics + locks went from **~52% of 8t cycles → ~12%**. The cache
machinery is no longer the bottleneck.

## Plan target assessment

The plan set aspirational targets: 8t QPS ≥ 2500, scaling ≥ 0.70. We did
not hit them — the actual headroom was smaller than estimated. We did
achieve:

- **8t QPS +50.8%** (1108 → 1671)
- **Scaling +14pp** (31% → 45% of linear)
- **1t QPS +3.6%** (no regression — actually a small gain)
- **Recall unchanged**

The 2500-QPS target assumed all 8t scaling loss was recoverable. In
practice, the remaining ~55% scaling gap is structural:

- `lut_distance` is now the bottleneck at 23.67% of cycles — this is real
  PQ distance computation, not contention. It scales with single-core
  SIMD throughput, not thread count.
- The W-TinyLFU machinery still serializes on per-shard locks during the
  miss path (writes). Misses are ~50% of L1 lookups and each one takes
  the write lock briefly.

## Next bottleneck (deferred to a follow-up plan)

`PqQuantizer::lut_distance` at 23.67% (combined with `beam_search_into`
31%) — this is the **single-core SIMD throughput** lever. The plan in
`docs/profiling_findings.md` anticipated this: "Audit `lut_distance`
SIMD. Compare against numkong's optimized dot products — the SVE2 gather
path may be leaving bandwidth on the table."

That's the next plan: profile `lut_distance` instruction-by-instruction
(via `perf annotate`), compare against numkong's `dots/sve.h` and
`mesh/skylake.h` primitives, and see where the SVE2 gather path is
leaving bandwidth on the table.

## Verification

- 9/12 unit tests pass locally (`test_block_cache`, `test_wtinylfu`,
  `test_tl_cache` all green; `test_engine`, `test_paged_search`,
  `test_partition` are pre-existing slow integration tests that time out
  at the 30s local default but pass on the bench VM).
- 5-rep measurement at L=400 rr=2: 1t stdev=5.9, 8t stdev=5.4 (tight).
- Recall unchanged at 0.9088 (within noise of baseline 0.9086).
- 8t profile shows villains eliminated (see above).
