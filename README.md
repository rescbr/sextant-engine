# Sextant

Sextant is an SSD-resident vector index engine that achieves AiSAQ-level
performance — fast build, compact index size, and near-zero idle RAM — on
commodity workstations. It owns the entire index storage layer: O_DIRECT
sidecar files plus an application-level LRU cache, so that query working sets
that exceed RAM still serve at SSD speed without depending on the OS page
cache. Sextant is built to integrate as a DuckDB extension (Phase 2); until
then it ships as a standalone C++ library and CLI.

## Novel contributions

Sextant makes three contributions over native AiSAQ:

1. **PQ-codes-only build.** Raw vectors are streamed through once to train a
   product quantizer; the graph is then constructed from PQ codes alone.
   Native AiSAQ loads full-precision vectors into RAM for the whole build. At
   100M vectors / dim-128 this is ~30 GB of build RAM for Sextant versus ~98 GB
   for native AiSAQ — roughly 3× less.

2. **`inline_pq` build-RAM decoupling.** The graph is always constructed with
   `inline_pq = 0` (flat neighbor lists) and only reformatted into its
   search-optimized inlined-PQ layout at flush time. Build RAM is therefore
   independent of the inline layout chosen for search.

3. **`closure_factor` partition overlap.** For partitioned builds, shard
   overlap is controlled by a distance-ratio threshold
   `c = (1 − f_target)^{−1/d_eff}` rather than naive k-base replication. This
   cuts vector replication from ~100% (native AiSAQ's `k_base = 2`) to ~15%,
   saving ~6.7× wasted build compute on merges.

## Performance targets

| Metric | Native AiSAQ (SE 105M) | Sextant target | SIFT-1M validated |
|---|---|---|---|
| Build time | 9.3 min | < 15 min (8-core) | ~190s ⚠️ (optimization in progress) |
| Index size | 4.9 GB | ~5–10 GB | 208 MB (inline_pq=0) |
| Recall@10 | 85% | ≥ 85% | **0.9957** ✅ |
| Search QPS (full cache) | — | hardware-dependent | **~2500** (single-threaded) |
| Search p50 latency | — | hardware-dependent | **0.45 ms** |

> The `< 100 MB idle RAM` target is deferred until performance work is
> complete. MemGraph caches the entry-point neighborhood in RAM for search
> performance. The final RAM budget will balance caching vs. idle footprint.
>
> Performance targets (build time, QPS under cache pressure) are claimed on
> **Linux only** — see [macOS limitation](#macos-limitation).

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
- **PQ m:** `dim / 4`, clamped to `[4, 64]`
- **K (partitions):** derived from the build-RAM budget (max per-shard RAM)
- **cache_size:** `min(graph_size × 0.50, physical_ram × 0.35)`
- **inline_pq, closure_factor, alpha, num_threads** …

Any field left at 0 / default in the build config is auto-resolved; an explicit
override wins and propagates to dependent parameters. Use the `--explain`
dry-run to see exactly what will be used:

```sh
./build/tools/sextant build --input data.fbin --index myidx --explain
```

This prints all resolved parameters and exits without building.

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
meson test -C build                             # run the test suite (56 tests)

# Build an index (R/L/pq-m auto-resolved when omitted)
./build/tools/sextant build --input data.fbin --index myidx --R 48

# Search with rerank against full-precision base vectors
./build/tools/sextant search --index myidx --query q.fbin \
    --base-data data.fbin --k 10 --rerank 10

# Insert a single vector
./build/tools/sextant insert --index myidx --vector vec.fbin --row-id 42

# Benchmark recall / QPS / latency against ground truth
./build/tools/sextant_bench --index myidx --queries q.fbin \
    --base-data data.fbin --ground-truth gt.gt --k 10

# Dry-run parameter resolution
./build/tools/sextant build --input data.fbin --index myidx --explain
```

### CLI flags

| Command | Flag | Description |
|---|---|---|
| `build` | `--input` | input `.fbin`/`.ibin`/`.bbin` (required) |
| | `--index` | index name/path prefix (required) |
| | `--R` | graph degree (auto if 0) |
| | `--L` | beam width (auto if 0) |
| | `--alpha` | prune threshold (default 1.2) |
| | `--pq-m` | PQ segments (auto from dim if 0) |
| | `--pq-bits` | PQ bits (default 8) |
| | `--metric` | `l2sq` (default) or `ip` |
| | `--threads` | build threads (auto if 0) |
| | `--build-ram` | per-shard RAM budget in bytes (forces partitioning if small) |
| | `--explain` | print resolved params and exit (dry-run) |
| `search` | `--index`, `--query` | index prefix, query `.fbin` (required) |
| | `--k` | results to return (default 10) |
| | `--L` | search beam width (default 200) |
| | `--rerank` | rerank factor (default 10) |
| | `--base-data` | base `.fbin` for full-precision rerank |
| | `--output` | output file (default: stdout) |
| `insert` | `--index`, `--vector` | index prefix, single-vector `.fbin` (required) |
| | `--row-id` | row ID for the inserted vector (required) |

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
│   └── logging.cpp
├── test/                # 71 tests across 10 suites
├── tools/               # sextant CLI, sextant_bench, fvecs_to_fbin
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
