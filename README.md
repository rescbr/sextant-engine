# Sextant

Sextant is an SSD-resident vector index engine that achieves AiSAQ-level
performance — fast build, compact index size, and near-zero idle RAM — on
commodity workstations. It owns the entire index storage layer: O_DIRECT
sidecar files plus an application-level LRU cache, so that query working sets
that exceed RAM still serve at SSD speed without depending on the OS page
cache. Sextant is built to integrate as a DuckDB extension (Phase 2); until
then it ships as a standalone C++ library and CLI.

## Novel contributions

Sextant makes four contributions over native AiSAQ:

1. **PQ-codes-only build.** Raw vectors are streamed through once to train a
   product quantizer; the graph is then constructed from PQ codes alone.
   Native AiSAQ loads full-precision vectors into RAM for the whole build. At
   100M vectors / dim-128 this is ~30 GB of build RAM for Sextant versus ~98 GB
   for native AiSAQ — roughly 3× less.

2. **Adaptive PQ selection.** The engine probes candidate `(m, bits)`
   configurations on the reservoir sample, measuring **PQ distortion** (median
   `|1 − pq_dist / true_dist|` over true neighbors) rather than row-id recall.
   Distortion is geometrically grounded and robust to near-duplicate clustering
   — a property that row-id recall lacks (see
   [Why not row-id recall?](#why-not-row-id-recall)). The engine selects the
   minimum-cost config whose distortion is within a configurable bound
   (`--pq-max-distortion`, default 0.05), automatically picking 4-bit at high
   `m` when the data distribution allows, and falling back to 8-bit only when
   necessary. The probe runs at zero extra I/O cost on the already-sampled
   reservoir. See [Adaptive PQ selection](#adaptive-pq-selection---pq-m-auto---pq-bits-auto)
   below.

3. **`inline_pq` build-RAM decoupling.** The graph is always constructed with
   `inline_pq = 0` (flat neighbor lists) and only reformatted into its
   search-optimized inlined-PQ layout at flush time. Build RAM is therefore
   independent of the inline layout chosen for search.

4. **`closure_factor` partition overlap.** For partitioned builds, shard
   overlap is controlled by a distance-ratio threshold
   `c = (1 − f_target)^{−1/d_eff}` rather than naive k-base replication. This
   cuts vector replication from ~100% (native AiSAQ's `k_base = 2`) to ~15%,
   saving ~6.7× wasted build compute on merges.

## Performance targets

| Metric | Native AiSAQ (SE 105M) | Sextant target | SIFT-1M validated (auto PQ) |
|---|---|---|---|
| Build time | 9.3 min | < 15 min (8-core) | **100s** ✅ |
| Index size | 4.9 GB | ~5–10 GB | 304 MB (inline_pq=0) |
| Recall@10 | 85% | ≥ 85% | **0.9975** ✅ |
| Search QPS (full cache, 1 thread) | — | hardware-dependent | **817** |
| Search QPS (full cache, 10 threads) | — | hardware-dependent | **2,036** |
| Search p50 latency | — | hardware-dependent | **0.657 ms** |

> The `< 100 MB idle RAM` target is deferred until performance work is
> complete. MemGraph caches the entry-point neighborhood in RAM for search
> performance. The final RAM budget will balance caching vs. idle footprint.
>
> Performance targets (build time, QPS under cache pressure) are claimed on
> **Linux only** — see [macOS limitation](#macos-limitation).

### SIFT-1M with adaptive PQ (everything on auto)

A full SIFT-1M build with `--pq-bits auto` (no manual `m` or `bits`) produces
the following probe trace and results:

```
pass 1:   probe m=16 bits=4  code=8B   table=16KB    distortion=0.1299
pass 1:   probe m=16 bits=8  code=16B  table=4096KB  distortion=0.0600
pass 1:   probe m=32 bits=4  code=16B  table=32KB    distortion=0.1043
pass 1:   probe m=32 bits=8  code=32B  table=8192KB  distortion=0.0366  ← SELECTED
pass 1:   probe m=64 bits=4  code=32B  table=64KB    distortion=0.0581
pass 1:   probe m=64 bits=8  code=64B  table=16384KB distortion=0.0116
pass 1:   probe m=128 bits=4 code=64B  table=128KB   distortion=0.0152
pass 1:   probe m=128 bits=8 code=128B table=32768KB distortion=0.0000
pass 1: selected m=32 bits=8 (distortion=0.0366) [min cost; max_distortion 0.05]
```

The cost model is simple: **`cost = m`** (the number of table gathers per
distance computation). `m` is the dominant cost factor in both build and search
— each `code_distance`/`lut_distance` call performs `m` lookups. Among configs
meeting the distortion bound (default 0.05), the engine picks the lowest `m`.
When two configs share the same `m` (e.g. 4-bit vs 8-bit), the smaller table
wins as a tie-break — since a 4-bit table is 256× smaller than 8-bit at the
same `m`, this naturally prefers 4-bit when distortion is adequate.

### SIFT-1M PQ config comparison

All three configs below produce excellent recall. The cost model selects the
cheapest one that meets the distortion bound — but the tradeoffs are instructive:

| Metric | m=32/8-bit (auto) | m=128/4-bit | m=128/8-bit |
|---|---|---|---|
| **Distortion** | 0.037 | 0.015 | 0.000 |
| **Recall@10** | 0.9975 | 0.9990 | 0.9990 |
| **Proximity (in-band)** | 0.998 | 1.000 | 1.000 |
| **Build time** | **100s** | 197s | 415s |
| **QPS (1 thread)** | **817** | 650 | 610 |
| **QPS (10 threads)** | **2,036** | 1,534 | 1,351 |
| Code size | **32 B/vec** | 64 B/vec | 128 B/vec |
| Table size | 8 MB | **128 KB** | 32 MB |
| Cost (m) | **32** | 128 | 128 |

**m=32/8-bit wins on cost** — fewest gathers (32), with distortion (0.037) well
within the 0.05 bound. The auto-selection correctly picks this. All three configs
achieve near-perfect proximity (in-band ≥ 0.998), confirming the index finds the
right neighborhoods.

> SIFT-1M has `ties@10 = 0.01` (virtually no near-duplicates), so recall@10 and
> proximity agree — both confirm excellent quality. On clustered data, proximity
> is the more reliable signal (see below).

**m=128/4-bit vs m=128/8-bit is the definitive case for 4-bit.** At the same
`m` and nearly identical distortion (0.015 vs 0.000) and recall (0.9990 both),
4-bit beats 8-bit on every other axis:

| Metric | m=128/4-bit | m=128/8-bit | 4-bit advantage |
|---|---|---|---|
| Build time | **197s** | 415s | **2.1× faster** |
| QPS (1 thread) | **650** | 610 | **1.07× faster** |
| QPS (10 threads) | **1,534** | 1,351 | **1.14× faster** |
| Code size | **64 B/vec** | 128 B/vec | **2× smaller** |
| Table size | **128 KB** | 32 MB | **256× smaller** |

The speed advantage comes from the 256× smaller table: at m=128, the 4-bit
table (128 KB) fits per-core L2, while the 8-bit table (32 MB) spills to RAM.
This is why 4-bit support exists: **when distortion is equal at a given m, 4-bit
is strictly superior.** The cost model (tie-break by table size) naturally
selects it. On SIFT, lower-`m` 8-bit configs are cheaper overall, so
auto-selection picks m=32/8-bit. On datasets where high `m` is needed (clustered
embeddings like geolocation data), 4-bit at that `m` is the clear winner.

Loosen the distortion bound (`--pq-max-distortion 0.10`) to admit faster configs
at the cost of end-to-end recall (e.g. m=16/8-bit: distortion 0.060, recall
~0.968, faster build/search). Tighten it (`--pq-max-distortion 0.03`) to force
higher-fidelity configs at the cost of build/search speed.

## Build modes

| Mode | Description | alpha | Recall | When to use |
|---|---|---|---|---|
| SDC (default) | Parallel construct from PQ codes | 1.2 | baseline | all builds |
| ADC (opt-in) | Parallel construct; re-reads raw vectors for precise LUTs | 1.5 | +0.5–2% | recall-critical |

Phase 1 defaults to SDC. The engine core supports both modes; the default CLI
defaults to SDC. (ADC is selectable through the library API / build config;
exposing `--build-mode` on the CLI is in progress.)

## `inline_pq` presets

`inline_pq` controls how many neighbors' PQ codes are inlined into each graph
node, trading `.graph` size against reads-per-hop:

| Preset | inline_pq | `.graph` size (100M) | reads/hop | Use case |
|---|---|---|---|---|
| `scale` | 0 | 27 GB | 2 | 1B+ vectors, limited SSD |
| `balanced` (default) | auto from SSD space | varies | 1–2 | general purpose |
| `performance` | R | 232 GB | 1 | <100M, latency-critical |

The default `balanced` preset derives `inline_pq` from available SSD space.
`inline_pq` is settable through the library build config today; the
`--inline-pq scale|balanced|performance|N` CLI flag is in progress.

## Auto-parameter resolver

Given only `N` (vector count) and `dim`, the engine resolves roughly a dozen
build parameters, including:

- **R (degree):** `N<100K → 32`, `<1M → 48`, `<10M → 64`, `<100M → 96`, else `128`
- **L (beam):** `max(100, 2 × R)`
- **PQ m / bits:** auto-resolved via a reservoir probe (see below)
- **K (partitions):** derived from the build-RAM budget (max per-shard RAM)
- **cache_size:** `min(graph_size × 0.50, physical_ram × 0.35)`
- **inline_pq, closure_factor, alpha, num_threads** …

Any field left at 0 / default in the build config is auto-resolved; an explicit
override wins and propagates to dependent parameters.

### Adaptive PQ selection (`--pq-m auto --pq-bits auto`)

When `pq_m` or `pq_bits` is left at `auto` (the CLI default for `--pq-bits`),
the engine probes candidate `(m, bits)` configurations on the pass-1 reservoir
sample and selects the best one based on measured **distortion** and cost:

1. **Distortion probe.** For each candidate `(m, bits)` (all divisors of `dim`
   with `sub_dim ∈ [1, 12]`, at both 4-bit and 8-bit), the engine trains a
   throwaway PQ on a 20K-vector subset of the reservoir, encodes it, and
   measures the **median distortion**: `|1 − pq_distance / true_distance|` over
   the true top-10 neighbors of 500 probe queries. Distortion is 0 when PQ
   perfectly preserves neighbor distances; 0.05 when the median estimate is off
   by 5%. It is direction-agnostic and geometrically grounded — see
   [Why not row-id recall?](#why-not-row-id-recall) below.

2. **Distortion bound.** Configs whose measured distortion exceeds
   `--pq-max-distortion` (default 0.05) are filtered out.

3. **Min-cost selection.** Among eligible configs, the engine picks the lowest
   `m` (the number of table gathers per distance computation — the dominant cost
   in build and search). When two configs share the same `m`, the smaller
   cross-distance table wins as a tie-break. Since a 4-bit table is 256× smaller
   than 8-bit at the same `m`, this naturally prefers 4-bit when distortion is
   adequate — no cache-size detection or residency factors needed.

This policy naturally selects **4-bit at high m** (excellent distortion with a
tiny table) when the data distribution allows it, and falls back to **8-bit**
only when 4-bit can't stay within the distortion bound. The probe runs entirely
on the in-RAM reservoir (zero extra dataset I/O), costing a few seconds against
a build that may take minutes.

Typical selections at the default distortion bound (0.05):

| Dataset | Selected | m | Distortion |
|---|---|---|---|
| SIFTsmall (128d, 10K) | m=32, 8-bit | 32 | 0.034 |
| SIFT (128d, 1M) | m=32, 8-bit | 32 | 0.037 |
| Geolocation embeddings (768d, 1.2M, clustered) | m=256, 8-bit | 256 | 0.031 |
| GIST (960d, continuous) | m=96, 8-bit | 96 | ~0.04 |

> On high-dimensional clustered data like geolocation embeddings, the tight
> default (0.05) forces high `m` (m=256, build ~190s). Since recall on such data
> is capped by tie structure (not PQ quality), users may safely loosen the bound
> (`--pq-max-distortion 0.10`) to select m=64/8-bit instead — build drops to
> ~80s, QPS nearly doubles, while recall@10 (0.56 → 0.58) and proximity
> in-band (0.87 → 0.92) barely change. See
> [Why not row-id recall?](#why-not-row-id-recall).

Use the `--explain` dry-run to see the full probe trace — every candidate config,
its measured distortion, tie diagnostics, and the selection rationale
— without building:

```sh
./build/tools/sextant build --input data.fbin --index myidx --explain
```

When `pq_m`/`pq_bits` are auto, `--explain` reads a random sample (default 20K
vectors via seek — no full dataset scan) and runs the probe. With explicit
`--pq-m`/`--pq-bits`, it stays instant (pure parameter print). The probe sample
size is controlled by `--probe-sample N`.


## Evaluation: recall, proximity, and ground truth

**Recall@k** is the standard ANN quality metric: for each query, compute the
true top-k nearest neighbors by brute force (exhaustive scan using the same
distance function the index uses — L2-squared by default), then measure what
fraction of those k ids appear in the index's top-k results. Averaged over all
queries. Recall@10 = 1.0 means the index returned the exact top-10 for every
query; 0.95 means it found 9.5 of the 10 true nearest neighbors on average.

Sextant reports recall@10 in `sextant_bench` for comparability with standard
ANN benchmarks (SIFT, GIST, etc.). On datasets with well-separated neighbors
(no near-duplicates), recall@10 is an excellent quality signal.

However, recall@10 has a subtle failure mode on **clustered data** — datasets
where many vectors sit at (nearly) the same location. The next section explains
why, and what Sextant measures instead.

## Why not row-id recall?

A natural question: why measure PQ quality with **distortion** instead of just
**recall** (fraction of true neighbors the PQ ranking recovers)? The answer is
that row-id recall is fragile on datasets with near-duplicate structure —
exactly the datasets that real-world vector indexes serve.

### The problem: tie-breaking makes "the top-10" arbitrary

Many real datasets contain groups of vectors at (nearly) the same location:
address or geolocation embeddings (multiple rows per location), document embeddings (near-
duplicate pages), entity embeddings (repeated records). For a query landing in
such a cluster, dozens or hundreds of base vectors may be within the 10th-NN
distance — all equally near, all legitimate results.

Ground-truth generators must pick exactly 10 ids for recall@10. When >10 vectors
are tied at the boundary, which 10 get picked is determined by implementation
detail (e.g. `argpartition`'s tie-breaking, or the order vectors appear in the
file). The "true top-10" is **arbitrary** — a different chunk size or scan order
produces a different ground truth.

This has a concrete consequence: **even exact brute-force search cannot score
recall@10 = 1.0** against such a ground truth. On a geolocation embedding dataset (768-dim,
1.26M vectors), brute-force top-10 scores only **0.63
recall@10** against the generated GT — not because brute force misses anything,
but because it returns 10 different (equally-near) vectors than the 10 the GT
arbitrarily marked. No index, however perfect, can exceed this ceiling.

### What we measure instead

The probe reports four diagnostics, only one of which drives selection:

| Metric | What it measures | Used for selection? |
|---|---|---|
| **distortion** | Median `\|1 − pq_dist / true_dist\|` over true neighbors. How much PQ distorts the distances it sees. Geometric, direction-agnostic, transfers across dataset classes. | **Yes** — the selection signal. |
| **band_recall** | Fraction of true neighbors for which the PQ top-30 shortlist contains *some* vector within the 10th-NN distance. Cluster-aware: doesn't care *which* co-located id PQ picked, only that it surfaced the right neighborhood. | No — diagnostic. |
| **ties@10** | Fraction of probe queries where more than 10 vectors fall within the 10th-NN distance. When high, recall@10 is structurally capped below 1.0 — the "top-10" is ambiguous regardless of index quality. | No — diagnostic. |
| **ties@30** | Same, but at the shortlist boundary (30th-NN distance). When high, even the PQ top-30 shortlist boundary is ambiguous — band_recall is the better signal. | No — diagnostic. |

**Distortion** is the selection signal because it isolates PQ's own property
(distance fidelity) from downstream effects (graph navigation, tie-breaking). A
config with low distortion preserves the geometry the graph needs; whether the
end-to-end system then achieves high row-id recall depends on the graph and the
data's tie structure, not on PQ.

### Interpretation guide

The `--explain` table shows all four columns. Here's how to read them:

```
  m    bits  code   table     distort  band_r  ties@10 ties@30  cost   band  note
  32   8    32B    8.00MB    0.0366  1.0000  0.01   0.02    61     RAM   ← SELECTED
```

- **distortion = 0.037**: PQ overestimates true-neighbor distances by 3.7% on
  the median. Well within the 0.05 bound — this config is eligible.
- **band_r = 1.0**: for every true neighbor, PQ's top-30 shortlist includes at
  least one vector within the 10th-NN radius. PQ is finding the right
  neighborhoods perfectly.
- **ties@10 = 0.07**: 7% of probe queries have >10 vectors tied at the 10th-NN
  distance. Low — most queries have a well-defined top-10.
- **ties@30 = 0.05**: 5% have >30 vectors tied at the 30th-NN distance. Also low.

Contrast with a heavily-clustered dataset where `ties@10 = 0.57` (57% of
queries have an ambiguous top-10): there, row-id recall@10 cannot exceed ~0.57
no matter how good the index is. The distortion and band_recall columns tell
you PQ is doing its job; the ties columns tell you the data itself caps
row-id recall. Without the tie diagnostics, a low recall number is
uninterpretable — it could mean PQ is bad, the graph is bad, or the data is
just tied. The tie columns disambiguate.

### What this means for benchmarking

The `sextant_bench` tool reports two quality metrics:

- **Recall@k** — fraction of true top-k ids found (set intersection). Reported
  for comparability with standard ANN benchmarks. Reliable when `ties@k` is low;
  structurally capped when it's high.

- **Proximity** — for each returned result, the ratio `d_target / d_result`
  where `d_target` is the true k-th NN distance. Reports the fraction of results
  at or inside the target radius (`in-band`), plus the distribution of
  out-of-band ratios so you can see how far misses land. This is **graded**
  (not boolean) and **cluster-aware**: it measures whether the index found the
  right neighborhood, regardless of which specific co-located row it returned.

On non-clustered data (SIFT, GIST — `ties@10` ≈ 0), both metrics agree.
On clustered data (geolocation, entity embeddings — `ties@10` can exceed 0.5),
proximity is the meaningful number: recall@10 is capped by tie structure, while
proximity tells you whether the index is actually finding the right locations.

> Proximity requires a ground-truth file with real distances. The `fvecs_to_fbin`
> converter computes them automatically when given `--base` and `--queries`; the
> SIFT-1M and SIFTsmall GTs distributed with Sextant include real distances.

## Metric handling and normalization

- The engine always **builds with L2** (squared Euclidean) — the Neyshabur–
  Srebro result guarantees an L2 index serves cosine/IP queries well.
- **Search supports L2 and inner product (IP).**
- **Cosine = L2-normalize then L2.** Normalization is the adapter's job (Phase
  2: DuckDB normalizes before handing vectors to the engine). The engine core
  never normalizes.
- Integer vectors (`uint8` / `int8`) are **cast to float** — no scaling, no
  normalization.

## Input formats

`.fbin` (float32), `.ibin` (int8), and `.bbin` / `.bvecs` (uint8) — all
little-endian:

```
[uint32 n][uint32 dim][n × dim × sizeof(T)]
```

Sextant does **not** use Parquet. The engine always reads full vectors, never
individual dimensions, so a columnar format buys nothing here and would break
O(1) range `pread` into the sidecar files. Phase 2's DuckDB layer handles
columnar data natively upstream of the engine. Use `tools/fvecs_to_fbin` to
convert `.fvecs`/`.ivecs` corpora.

## Search architecture

Sextant's search path uses a multi-tier architecture informed by DiskANN,
PipeANN (OSDI 2025), and OctopusANN (VLDB 2026):

```
┌─────────────────────────────────────────────┐
│                VamanaCore                    │
│         beam_search / search                │
│   (DynamicWidth, PageSearch, PinResult)     │
└──────────────────┬──────────────────────────┘
                   │ NodeStore interface
                   ▼
┌─────────────────────────────────────────────┐
│              MemGraph                        │
│  BFS neighborhood (3 hops) in RAM            │
│  Lock-free O(1) array lookup                 │
│  ~34.5% of nodes for SIFT-1M (83 MB)        │
└──────────┬──────────────┬───────────────────┘
           │ hit          │ miss
           ▼              ▼
     (RAM pointer)  ┌─────────────────────┐
                    │  PagedNodeStore      │
                    │  ShardedLRUCache     │
                    │  DirectFile (pread)  │
                    └─────────────────────┘
```

- **PageShuffle:** graph nodes are BFS-reordered at flush time so neighbors
  are co-located on disk. Halves I/O on SIFT-1M.
- **MemGraph:** caches the entry-point BFS neighborhood in RAM. These
  "highway" nodes are touched by every query. Lock-free O(1) lookup.
- **PageSearch:** on a cache miss (actual SSD read), computes distances for
  R co-located nodes in the same block — amortizes the I/O cost.
- **DynamicWidth:** two-phase beam width — small during approach phase
  (navigating toward query region), large during converge phase.
- **PinResult:** `pin_node` returns `{data, from_ssd}` — PageSearch fires
  only when `from_ssd=true` (actual I/O), never on MemGraph/LRU hits.

See `docs/design_decisions.md` Issues 40-45 for the full rationale.

## Storage layout

All sidecar files are little-endian; `{index}` is the index name/path prefix:

- `{index}.graph` — graph nodes, `nodes_per_block` packed per 256 KB block
- `{index}.codes` — PQ codes, page-aligned
- `{index}.meta` — entry points, medoids, build params, PQ codebook
- `{index}.manifest` — atomic commit point (written to a temp file, then
  renamed over the real one)

**Crash recovery:** opening an index requires a valid `.manifest`. Orphaned
sidecar files (`.graph`/`.codes`/`.meta` present with no matching
`.manifest`) cause an error on open.

## macOS limitation

macOS is a **development and validation platform only**. macOS `F_NOCACHE` is
advisory and is not enforced direct I/O the way Linux `O_DIRECT` is, so the
kernel page cache can mask true SSD behavior. The performance targets above
(build time, idle RAM) are claimed on **Linux only**. On macOS, validate
**correctness, recall, and relative profiling** — not absolute throughput.

## Quick start

```sh
./bootstrap.sh                                  # fetch nsync + NumKong submodules
meson setup build && meson compile -C build     # build (C++17-compatible)
meson test -C build                             # run the test suite

# 1. Find the best PQ config for your data (~30-60s)
./build/tools/sextant analyze --input data.fbin --proximity-target 0.95
# → outputs: "Build with: --pq-m 32 --pq-bits 8"

# 2. Build using the recommended config
./build/tools/sextant build --input data.fbin --index myidx --pq-m 32 --pq-bits 8
```

### `analyze` — find the best PQ config

```sh
./build/tools/sextant analyze --input data.fbin --proximity-target 0.95
./build/tools/sextant analyze --input data.fbin                      # no target → knee-based
```

### `build` — build an index

```sh
./build/tools/sextant build --input data.fbin --index myidx --pq-m 32 --pq-bits 8
./build/tools/sextant build --input data.fbin --index myidx          # auto-probe PQ config
./build/tools/sextant build --input data.fbin --index myidx --explain  # dry-run, no build
```

### `search` — query an index

```sh
./build/tools/sextant search --index myidx --query q.fbin \
    --base-data data.fbin --k 10 --rerank 10
```

### `insert` — add a single vector

```sh
./build/tools/sextant insert --index myidx --vector vec.fbin --row-id 42
```

### `sextant_bench` — benchmark recall / QPS / latency

```sh
./build/tools/sextant_bench --index myidx --queries q.fbin \
    --base-data data.fbin --ground-truth gt.gt --k 10
```

### `pq_explore` — PQ parameter sweeps

```sh
./build/tools/pq_explore --input data.fbin --m-sweep 32,64,128 --bits-sweep 4,8
```

### CLI flags

| Command | Flag | Description |
|---|---|---|
| `build` | `--input` | input `.fbin`/`.ibin`/`.bbin` (required) |
| | `--index` | index name/path prefix (required) |
| | `--R` | graph degree (auto if 0) |
| | `--L` | beam width (auto if 0) |
| | `--alpha` | prune threshold (default 1.2) |
| | `--pq-m` | PQ segments (`0` = auto, probed on reservoir) |
| | `--pq-bits` | PQ bits: `4`, `8`, or `auto` (default; probed on reservoir) |
| | `--pq-max-distortion` | max PQ distortion for auto selection (default 0.05; lower favors higher-fidelity PQ, higher admits faster low-m configs) |
| | `--probe-sample` | sample size for `--explain` PQ probe (default 20000) |
| | `--metric` | `l2sq` (default) or `ip` |
| | `--threads` | build threads (auto if 0) |
| | `--build-ram` | per-shard RAM budget in bytes (forces partitioning if small) |
| | `--explain` | print resolved params and exit (dry-run; runs PQ probe when m/bits auto) |
| `search` | `--index`, `--query` | index prefix, query `.fbin` (required) |
| | `--k` | results to return (default 10) |
| | `--L` | search beam width (default 200) |
| | `--rerank` | rerank factor (default 10) |
| | `--base-data` | base `.fbin` for full-precision rerank |
| | `--output` | output file (default: stdout) |
| `insert` | `--index`, `--vector` | index prefix, single-vector `.fbin` (required) |
| | `--row-id` | row ID for the inserted vector (required) |
| `analyze` | `--input` | input `.fbin` (required) |
| | `--pq-max-distortion` | max PQ distortion for eligibility (default 0.05; ignored when a target is set) |
| | `--proximity-target` | target proximity in-band; probes all configs, recommends cheapest meeting it (default: knee-based) |
| | `--id-recall-target` | override: target id-recall@k instead of proximity (for entity matching/dedup) |
| | `--id-recall-k` | k for --id-recall-target (default 10) |
| | `--probe-sample` | probe pool size (0 = auto-size from density ratio, default) |
| | `--threads` | threads for the mini-graph sweep (default: hardware_concurrency) |
| | `--metric` | `l2sq` (default) or `ip` |

### `analyze` — PQ config advisor

Run `sextant analyze` before building to find the best `(m, bits)` PQ config
for your dataset and workload. It builds mini Vamana graphs on an auto-sized
sample, sweeps no-rerank proximity at L=50/100/200 for each config, then
recommends a config and outputs the exact `sextant build` command.

**Target-driven mode** (recommended): pass `--proximity-target` with your
workload's quality requirement. The tool probes all configs and recommends the
cheapest whose mini-graph proximity meets the target:

```sh
# "I need 95% proximity for my search workload"
./build/tools/sextant analyze --input data.fbin --proximity-target 0.95
```

Proximity in-band is the primary quality metric — it measures whether the index
finds the right neighborhood, robust to near-duplicate clustering. For entity
matching/dedup where exact ids matter, use `--id-recall-target` instead. See
`docs/proximity_vs_recall.md` for why proximity is the better default.

Typical targets (see `docs/recall_targets.md` for the full literature survey):

| Target | Workload | Basis |
|---|---|---|
| 0.80 | RAG / retrieval | Downstream quality robust to recall ≥ 0.4 (arXiv:2606.04522) |
| 0.95 | General search / benchmark | ANN community convention (Aumüller et al., 2020) |
| 0.99 | Entity matching / dedup / geocoding | Missed matches unrecoverable (Christen, 2012; BlockingPy, 2025) |

**Knee-driven mode** (default, no target): uses `--pq-max-distortion` to
filter eligible configs, then finds the diminishing-returns knee via relative
proximity deltas. Useful when you don't have a specific recall target and want
the best cost/quality tradeoff.

**Scales to billions.** `analyze` reads a random sample (auto-sized: 10K–100K
vectors via seek, not a scan) and builds mini-graphs on that sample only.
The cost is **constant** regardless of dataset size — it takes the same ~30-60s
whether you have 1M or 1B vectors. This works because of the curse of
dimensionality: the k-th nearest-neighbor distance is nearly constant above
~1M vectors, so a small sample's neighborhood structure matches the full
dataset. The auto-sizer computes the exact density ratio and shows it in the
output; see `docs/sensitivity_probe.md` for the math.

**Run `analyze` before `build`.** The build path's distortion probe (which
selects PQ config automatically) measures PQ quality in isolation — it doesn't
know whether your data is PQ-sensitive or whether higher `m` is worth the
build cost. `analyze` measures end-to-end (PQ + graph + search) on a
representative sample and tells you exactly which config to use. A 30-60s
analysis that picks the right config can save a multi-hour build at the wrong
`m`, or prevent wasted spend on an unnecessarily high `m`.

By default the probe pool is **auto-sized** from the dataset's N and dim to
achieve a 10th-NN density ratio ≤ 1.05. The mini-graph no-rerank proximity is
a conservative lower bound on full-scale proximity (rerank + larger graph both
help), so if a config meets the target here, it will at full scale too. See
`docs/sensitivity_probe.md` for validation data and
`docs/proximity_vs_recall.md` for why proximity is the primary metric.

## Partitioned build

When the monolithic graph would exceed the build-RAM budget (default 50% of
physical RAM), Sextant partitions the dataset into K shards, builds each shard
independently, and merges:

```sh
# 8 GB per-shard RAM ceiling
./build/tools/sextant build --input big.fbin --index bigidx --build-ram 8589934592
```

`K = ceil(N / (budget / per_vec))`. Each shard builds at `2R/3` degree, then
merges with `closure_factor` overlap (~15% replication). Recall degrades
gracefully as K rises — SIFT-1M K-sweep: `K=1 → 0.9956`, `K=4 → 0.9933`,
`K=29 → 0.9882`.

## Project structure

```
sextant-engine/
├── include/sextant/     # public API (C++17-compatible)
├── src/
│   ├── storage/         # DirectFile, BlockAllocator, ShardedLRUCache,
│   │                    # NodeStore (Flat/Paged/MemGraph), PinResult
│   ├── quant/           # PQ quantizer (NumKong distance kernels)
│   ├── algo/            # Vamana core (BeamSearch, RobustPrune, DynamicWidth,
│   │                    # PageSearch)
│   ├── engine/          # build/search pipelines, partitioning, FbinSource,
│   │                    # PageShuffle (BFS reordering at flush)
│   ├── util/            # cache_info (cross-platform L2/L3 detection)
│   └── logging.cpp
├── test/                # 71 tests across 10 suites
├── tools/               # sextant CLI, sextant_bench, fvecs_to_fbin, pq_explore
├── docs/                # design_decisions.md, remaining_tasks.md
├── third_party/         # nsync, NumKong (submodules); CTPL, cmdline (committed)
├── scripts/             # download_datasets.sh
├── bootstrap.sh         # submodule init
├── meson.build
└── meson_options.txt    # -Dtests=enabled, -Dbenchmarks=disabled
```

## Dependencies and licenses

| Dependency | License | Source |
|---|---|---|
| nsync | Apache-2.0 | git submodule (native Meson wrapper) |
| NumKong | Apache-2.0 | git submodule (header-only) |
| spdlog | MIT | Meson WrapDB |
| googletest | BSD-3-Clause | Meson WrapDB |
| CTPL | Apache-2.0 | committed in `third_party/` |
| cmdline | BSD-3-Clause | committed in `third_party/` |

All are compatible with the project license below.

## License

SSPL-1.0-only. See [LICENSE](LICENSE).
