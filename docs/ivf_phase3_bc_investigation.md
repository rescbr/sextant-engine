# IVF Phase 3 — B & C Investigation Plan

**Status:** Investigation framing, 2026-07-23. Supersedes the B/C sections of
`~/.local/state/maki/plans/ivf-phase3-graph-intelligence.md` in light of the
routing study and c4a profiling results.

## Context that reshapes B and C

1. **A is done.** A1 (sub-clustered entry points) shipped. A2/A3 validated
   negative (entry-point selection/count don't help — the default multi-start
   over A1's medoids is already good).
2. **Routing is largely solved.** closure_factor (build-time Poisson-Voronoi
   overlap) lifts np=4 routing recall to 0.9500. Remaining headroom is 2.7pp,
   and multi-probe only recovers it at variable per-query cost. Routing is NOT
   the IVF/merged gap (see `docs/ivf_routing_analysis.md`).
3. **The gap is structural work-multiplicity.** IVF does `n_probe × per-shard
   beam_search work`; merged does `1 × beam_search work`. The c4a profile
   (merged, 8t) is diffuse: PQ eval 27%, heap/loop 18%, FP16-ball 12% (post-FHM),
   rerank 12% (out of scope), libc 13%. No single hot spot.
4. **Memory bandwidth is the wall.** 8 threads → 4.3× throughput (not 8×).
   FHM cut FP16 instruction count; FP16 loads are now the bottleneck. The
   sublinearity is the aarch64 memory subsystem, IVF-agnostic (merged hits the
   same 4.46× ceiling).

So B and C must be re-evaluated against two questions: (a) does this attack the
n_probe multiplier (the actual gap)? (b) does it help under the memory-bandwidth
ceiling, or does it just rearrange contention?

---

## Workstream B — Two-level parallelism (query × shard)

### The plan's premise, re-checked
B1 splits N cores into Q query-workers × S shard-workers. Each query-worker
overlaps its n_probe shard searches across S shard-workers via bare std::thread
+ latch (no nested pools, avoiding the Layer-3 nsync_mu regression).

### Why it likely regresses throughput
The benchmark is **throughput-bound**: it pushes all N queries via
`search_one_async`, saturating all cores with query-level parallelism. At
8 threads / np=4, all 8 workers are busy the whole time with different queries.

B1 at Q=4, S=2: 4 concurrent queries (down from 8), each with 2-way shard
overlap. Per-query latency drops ~1.4× (Amdahl: shard phase ~80% × 2-wide),
but concurrent queries halve → net throughput ≈ 0.5 × 1.4 = **0.7× — a loss**.

This only reverses if the serial shard loop leaves cores idle (memory stalls)
AND a shard-worker on a different core could use that stall time. But that
different core was already running another query's shard loop. The memory-
bandwidth wall (4.3× at 8t) means cores are already well-utilized; B1 adds
contention, not overlap.

### When B1 genuinely helps (out of benchmark scope)
- **Latency-bound serving** (single-query p50/p99): variable work doesn't hurt
  throughput, and overlapping n_probe shards cuts single-query latency ~1.4×.
  This is a real product feature for online serving but not what the QPS
  benchmark measures.
- **Low-concurrency regimes** where query-fan-out can't saturate cores (fewer
  than N concurrent queries).

### Investigation steps
1. **Measure, don't assume.** Implement B1 behind a config flag (num_shard_threads).
   Run the benchmark at np=4/8 with S=1/2/4. The hypothesis: S=1 (current) wins
   on QPS at 8 threads; S=2 wins on p50/p99 latency. Validate or kill with data.
2. **If S=2 regresses QPS as predicted**, document B1 as a latency feature
   (useful for serving, neutral-to-negative for batch throughput) and move on.
3. **Do NOT pursue B2 (adaptive S)** unless B1 shows a QPS win — adaptive logic
   adds complexity for a lever that likely doesn't move the target metric.

### Files (if implemented)
- `IVFSearcher`: `num_shard_threads_` (S); in `search_body_`, spawn S bare
  std::threads each searching a shard slice, joined via std::latch (C++23).
- `IVFWorkerState`: gains S VamanaTLS instances (one per shard-worker).

---

## Workstream C — Graph traversal intelligence (reduce evals per hop)

### The corrected target
The plan's C1 ("degree-ascending neighbor expansion") was based on two flawed
premises: (a) neighbor lists are distance-sorted (they're NOT — back-edges are
appended unsorted during reciprocal edge wiring), and (b) NDSearch's "degree-
ascending" is about traversal order (it's actually about SSD page layout).

**The real target, confirmed by the c4a profile: reduce PQ distance evals
during neighbor expansion.** PQ eval is 27% of beam_search — the biggest chunk
post-FHM. The current code evaluates ALL unvisited neighbors of every popped
node; many are N_rbu (read-but-unexplored — pushed to the frontier, never
popped). DynamicWidth narrows the beam to suppress N_rbu, but doesn't skip
evaluating them in the first place.

### C1 (revised) — Frontier-saturation early-termination
**Idea:** during a node's neighbor expansion, if K consecutive evaluated
neighbors all fail to enter W (the frontier is saturated for this node), skip
the rest. Works WITHOUT sorted neighbors — it's a heuristic that once the
frontier is full and recent neighbors aren't improving it, the rest probably
won't either.

**Tension with batch4:** PQ eval is batch4 (4 codes per SVE2 gather). Checking
after each neighbor wastes the batch. Resolution: check after each batch4 chunk
— if all 4 failed to enter W and W is full, skip the remaining batches for this
node. Coarse-grained (every 4 neighbors), preserves SIMD throughput.

**Expected impact:** cuts evals per expansion by ~20-40% (the tail of a node's
neighbors rarely beat a saturated frontier). At 27% of beam_search time, a 30%
eval reduction → ~8% beam_search speedup → ~5-6% end-to-end (beam_search is
~68% inclusive). Modest but real, and it helps BOTH merged and IVF equally
(rising tide).

**Risk:** recall. The skipped neighbors might include rare good ones. Must A/B
carefully — the skip threshold (K consecutive misses) and the "W full" guard
need tuning. Start conservative (K=8, only skip when W is full AND the node is
past the approach phase).

### C2 (plan's) — Coarse position per node (1 byte)
Store a 1-byte sub-cluster ID per node (from A1's k-means). When visiting a
node, check its sub-cluster against the query's; de-prioritize mismatched
neighbors. **Requires node-layout change (PagedNodeStore/MemGraph) → escalate
before implementing.** The signal is weaker now that A2/A3 showed sub-cluster
selection doesn't help at the entry-point level; unclear it helps inside
traversal either.

### C3 (plan's) — Per-neighbor distance annotation
Rejected (R×4 bytes/node doubles node size). Confirmed still not viable.

### Investigation steps for C1
1. **Prototype in beam_search_into.** Add the skip heuristic behind a config
   flag (e.g. `frontier_saturation_skip`). Measure evals/hop (via a counter)
   and recall on arxiv100k — does the skip cut evals without dropping recall?
2. **If local validates, run c4a.** The diffuse profile means a 30% eval cut
   in the 27% PQ chunk is ~8% beam_search speedup. Measure realized QPS gain.
3. **Tune the threshold.** K (consecutive misses before skip) from 4 to 16;
   guard on "W full" and "past approach phase" to protect recall.

### Files (if implemented)
- `src/algo/vamana_core.cpp` (`beam_search_into`): add the skip check after
  each batch4 chunk in the neighbor expansion loop. A build-time or search-time
  flag gates it.

---

## Sequencing recommendation

1. **C1 first** (frontier-saturation skip). It attacks the confirmed biggest
   beam_search chunk (PQ eval), helps both paths, and the prototype is small
   (a counter + a skip check in the expansion loop). Validate evals/hop + recall
   locally, then QPS on c4a.
2. **B1 as a latency feature** (not a throughput lever). Implement behind a
   flag, measure p50/p99 at np=4/8, document the tradeoff. Don't expect a QPS
   win; expect a serving-regime capability.
3. **C2 only if C1 insufficient AND the user approves the node-layout change.**

## Stop conditions
- C1 gives <3% QPS at matched recall → the diffuse profile has no concentrated
  lever left; accept the current state.
- B1 regresses QPS as predicted → ship as latency feature, document, move on.
- Both confirm the gap is intrinsic → IVF is the high-recall option (ceiling
  0.9912 at np8 vs merged 0.9084); position accordingly.
