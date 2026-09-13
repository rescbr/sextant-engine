# Locking contracts & gotchas

> Reference for the locking disciplines in the storage layer. Distilled from
> the multithread-scaling work (see `docs/scaling_fix_results.md` and
> `docs/profiling_findings.md` — both archived 2026-09-13 to workspace
> `old-docs/`).

## nsync reader-writer locks

The codebase wraps nsync's `nsync_mu` in `include/sextant/sync.hpp`:

- `Mutex` — owns an `nsync_mu`, default-constructed to unlocked.
- `ScopedReadLock` — RAII shared (reader) lock. Multiple readers proceed
  in parallel.
- `ScopedWriteLock` — RAII exclusive (writer) lock. Mutually exclusive
  with readers AND other writers.

nsync's `nsync_mu` uses **writer-preference fairness**: if a writer is
waiting, new readers block behind it (so writers don't starve under
heavy read load). This is the right default for a cache where miss-path
writes must make progress, but it has one non-obvious consequence —
see "Recursive read-lock deadlock" below.

## CacheShard locking contract

`CacheShard` (in `src/storage/block_cache.hpp`) is the W-TinyLFU shard.
Its methods have different locking requirements:

| method | lock required | notes |
|--------|---------------|-------|
| `lookup(idx)` | none (takes own `ScopedReadLock` internally) | atomic sketch + atomic counters make the read path lock-tolerant |
| `lookup_unlocked(idx)` | caller holds read OR write lock | same body as `lookup()`, skips the internal lock |
| `insert(idx, ...)` | caller holds write lock | mutates LRU lists, entry status, map_ |
| `mark_dirty(idx)` | caller holds write lock | mutates entry's `dirty` flag |
| `maybe_adapt()` | caller holds write lock | mutates LRU lists to reconcile with atomic limits |

If you change `lookup()`, you MUST also consider `lookup_unlocked()` —
they share the same body via a private helper.

## Rule: read-locked functions need an `_unlocked` variant

Any function that takes `ScopedReadLock` internally MUST expose an
`_unlocked` variant for callers that already hold the lock. Otherwise
callers holding a read lock who call into the function trigger a
**recursive read lock** on the same `nsync_mu`, which deadlocks under
writer-preference fairness.

### The deadlock mechanism

Consider this call sequence (what `batched_read` originally did):

```cpp
{
    ScopedReadLock lock(shard.mutex());   // thread A holds read lock
    shard.lookup(idx);                    // lookup() internally does
                                          //   ScopedReadLock lock(mu_);
                                          //   ... ← BLOCKS here
}
```

Meanwhile thread B wants the write lock for `insert()`:

```cpp
{
    ScopedWriteLock lock(shard.mutex());  // thread B waits for readers
                                          //   to drain
    shard.insert(...);
}
```

Under nsync's writer-preference:
1. Thread B (writer) announces it's waiting. New readers block behind B.
2. Thread A (reader) tries to re-acquire the read lock inside `lookup()`.
   nsync sees a writer waiting → blocks A behind B.
3. A holds the existing read lock → B can never acquire the write lock.
4. **Deadlock.** A waits for B, B waits for A.

### Symptom

Intermittent multithreaded hangs. In the step-3 measurement this
manifested as: 3 of 5 bench runs complete normally (~3s each), then the
4th hangs for 12+ minutes with all worker threads in `S` state at 0% CPU.
No crash, no assertion, no obvious trigger — pure timing-dependent
deadlock.

### Fix

Use `lookup_unlocked()` when the caller already holds the lock:

```cpp
{
    ScopedReadLock lock(shard.mutex());
    shard.lookup_unlocked(idx);   // no internal lock, no recursion
}
```

The same rule applies to any future read-locked helper. If you add a
function that takes `ScopedReadLock` internally and a caller might
already hold the lock, add a `_unlocked` variant.

## Where each lock is taken

### Production code

`src/storage/node_store.cpp::batched_read()` (the L1 + L2 + I/O pipeline):

| step | lock | reason |
|------|------|--------|
| L1 lookup (hot path) | none | `TLBlockCache` is thread-local |
| L2 read-lookup | `ScopedReadLock` | parallel readers; calls `lookup_unlocked` |
| pread into staging | none | thread-local buffer |
| L2 insert (per block) | `ScopedWriteLock` | mutates LRU, map; calls `lookup_unlocked` to check for race |
| L2 final lookup | `ScopedWriteLock` | under the write lock from insert; calls `lookup_unlocked` |

### Test code

Tests often batch `insert` + `lookup` under a single `ScopedWriteLock` to
model atomic operations. Inside that scope, use `lookup_unlocked()` (not
`lookup()`) — same reason as production. See `test/test_block_cache.cpp`
and `test/test_wtinylfu.cpp` for the established pattern.

If you write a new test that holds the write lock and calls lookup, use
`lookup_unlocked()`. The non-`_unlocked` variant will deadlock under
nsync.

## Thread-local state and lifetime coupling

Several pieces of search state are `thread_local` to avoid per-call
allocation or cross-thread contention:

| state | owner | cleared on |
|-------|-------|-----------|
| `g_staging` (pread buffer) | `PagedNodeStore` (per-instance via `instance_id_`) | thread exit |
| `g_tl_caches` (TLBlockCache map) | `PagedNodeStore` (per-instance) | `~PagedNodeStore` on the destroying thread; other threads' entries leak harmlessly |
| `tl_l1_counters_` (L1 hit/miss accumulators) | `PagedNodeStore` (per-instance) | same |
| `tl_fast_` (cached TLBlockCache* + L1Counters*) | `PagedNodeStore` (per-instance) | `~PagedNodeStore` on the destroying thread if it points at us |
| `BlockCache::tl_hot_counters_` (record_access accumulators) | `BlockCache` (per-instance) | `~BlockCache` on the destroying thread |

All of these assume **`PagedNodeStore` and `BlockCache` outlive all search
threads** — true in production (engine-lifetime objects). The destructor
only clears the destroying thread's entries; other threads' entries
become stale but harmless because:

- The `instance_id_` key is monotonic, never reused, so stale entries
  can never match a future `PagedNodeStore`.
- The `tl_fast_` cache checks `owner == this` on every use, so a stale
  pointer to a destroyed store is caught and re-hashed on the next call
  from that thread.

If you ever write a test that creates and destroys `PagedNodeStore` or
`BlockCache` instances on a thread that has called `batched_read()` or
`record_access()`, call `flush_thread_local()` / `flush_l1_counters()`
first to drain pending samples. Otherwise samples leak (a few bytes,
invisible in practice).

## Profiling guidance

When profiling multithreaded code:

- **Run benches in background with `setsid ... </dev/null >log 2>&1 & disown`**,
  then poll the log. Inline `for` loops via `gcloud compute ssh --command`
  leak orphan processes when SSH times out — the bench keeps running on
  the VM, contaminating subsequent measurements.
- **`pkill -9 -f sextant_bench` between runs** to clear any orphans.
- **Take 5+ repetitions**, not single samples. c4a spot instances have
  meaningful neighbor-contention variance (stdev ~5-50 QPS depending on
  config).
- **Intermittent hangs are almost always deadlocks**, not perf regressions.
  If 3/5 runs complete fast and the 4th hangs forever, suspect locking.
  Check `ps -eo pid,etime,stat,cmd` — all threads in `S` state at 0% CPU
  confirms a deadlock.
