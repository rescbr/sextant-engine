# Benchmark rules — cold storage, always

**Status: BINDING.** Every performance number we publish or act on follows
these rules. A number produced outside them is not a sextant benchmark.

## The prime directive

**Never benchmark against a RAM-cached dataset.** We are building a system
whose economics are cold-storage economics: codes on NAND, routing in DRAM,
bytes-touched as the currency. A warm page cache / ARC fabricates exactly
the property under test. We are not measuring who has the best Linux cache
tuning — we are measuring our system, and our competitors' systems, doing
the work they would actually have to do.

This applies **even when benching against competitors**. If a competitor
benefits from a warm cache in a shared harness, the comparison is invalid —
either flush/cold-mount for both engines or don't publish the comparison.
(Our numbers will look conservative vs RAM-resident published benchmarks —
Infino/turbovec flat scans run warm. That is a feature, and material for
the blog: we bench the hard problem, not the cache.)

## Workstation harness (pastry/largefiles/sextant)

- Dataset lives on `/mnt/sextant` (ZFS: `recordsize=1M`, `compression=off`,
  `atime=off`).
- `primarycache=metadata` — the ARC may cache metadata but **not data**.
  This is deliberate: it makes cold reads structural, not a matter of
  discipline.
- **CAVEAT (measured 2026-09-07, see results/leaf_cache/):** on this box
  `primarycache=metadata` does NOT keep data out of RAM — it pushes it out
  of the 2 GB ARC into the *unbounded* Linux page cache. Repeated mmap
  passes retain the whole index (`Cached` grows by the file size; a 4.6 GB
  index warmed to 162 QPS "for free"). The old "mmap faults do not persist"
  claim was wrong in this direction and has been removed. Consequences:
  - **Cold numbers require an explicit `echo 3 > /proc/sys/vm/drop_caches`**
    (+ `zpool sync pastry`) before the measured pass, every time.
  - Cold mmap *scan* passes still pay per-fault disk reads (230 GB of 4 KB
    faults on a 1000-query f=0.05 pass = 33 s); the page cache only warms
    via the readahead/fadvise path. Both behaviors are reproducible.
  - The tree-path engine cache (`--cache-mb`, LeafExtentCache) keeps the
    page cache out of the measurement entirely (composability), which is
    part of why it is the reportable warm layer.
- ARC cap 2 GB (metadata-sized on purpose). Do not raise it to fit data.
- No `vmtouch`, no `cat file > /dev/null` pre-warm, no "run it twice and
  take the second number" for anything reported as cold-search/build.
- Repeated-query *search* runs are fine (a server legitimately serves many
  queries); the engine's own caches (userspace, explicitly DRAM-budgeted:
  graph `BlockCache`, tree `LeafExtentCache` via `tree-search --cache-mb`)
  are the only data caches in play — that is the real-world layer
  and it is part of the measured system.

## Categories we report

1. **Cold search** — first pass over queries, nothing warm. The headline
   number.
2. **Warm search** — subsequent passes; valid ONLY with the engine cache
   explicitly sized/named in the result (e.g. "BlockCache=512MB"), never
   as an unlabeled QPS.
3. **Build** — throughput and utilization measured cold-streaming from
   storage. No userspace corpus preload on any measured build. (A preload
   path may exist for dev convenience; numbers from it must be labeled
   `warm-fabricated` and never cited.)
4. **Kernel microbenchmarks** — cache-resident by nature; fine, they
   measure kernel ceilings, never presented as end-to-end.

## Transfer/checkpoint discipline (unchanged)

md5-verify all transfers; back-to-back runs, medians; label machine, data,
threads (16 physical cores for search), and SIMD target (avx512) on every
table. Exit codes, not output greps.

## Quiesce before measuring

Never benchmark immediately after writing to the same pool: ZFS dirty-data
write-back (flush of freshly built trees) competes with the reads and can
depress light-probe cells by 5×+. After any build onto the bench dataset:
`zpool sync pastry` (or the pool at hand), then a short settle, then
measure. Caught 2026-09-07: cold f=0.05 measured 271 QPS with write-back in
flight, 1553 QPS after quiesce, same binaries and tree.

ALSO: the first search per tree, even after sync + settle, runs 4–6× slower
than steady state (post-build pool transient; recall identical, reruns
stable across processes, so it is not data-cache warmth). Per tree, run one
DISCARDED warmup search before the first measured cell. Caught 2026-09-07
on the cohere-1M grid: first cells showed f=0.05 slower than f=0.10 for
every quantizer and a phantom "i8 4.5×" result.



## History

2026-09-07: rule established after nearly reporting build utilization from
a warm-ARC proposal and catching that `primarycache=metadata` is not a
degradation but the correct harness. Prior c4a numbers (ext4 VM, default
page cache) should be treated as warm-leaning where repeated passes were
involved; the cohere-10M matrix will be the first fully cold-protocol run.
