# Sextant

Sextant is an SSD-resident vector index engine that achieves AiSAQ-level
performance — fast build, compact index size, and near-zero idle RAM — on
commodity workstations. It owns the entire index storage layer: `O_DIRECT`
file I/O plus an engine-owned W-TinyLFU application cache, so that query
working sets which exceed RAM still serve at SSD speed without depending on
the OS page cache.

**Current architecture: a hierarchical IVF tree.** The engine began life as a
flat Vamana-graph engine (July 2026, still present and buildable — see
[Flat graph engine](#flat-graph-engine-legacy) below); since August 2026 the
heart of the system is a streaming-built, single-file IVF tree with a
PCA-based routing plane and per-leaf quantizers (`src/tree/`). Sextant ships
today as a **standalone C++ library + CLI**. A DuckDB extension integration
remains unstarted future work — there is no code for it in this repo.

## The tree engine

- **Streaming IVF build** (`src/tree/`): vectors are chunked through a
  two-level tree — a small root fanout (`--k-root`) over PCA-projected
  centroids, leaves capped at `--leaf-capacity` vectors (default 5000).
  Build RAM is bounded by chunk size, not corpus size. See
  [docs/hierarchical_tree_design.md](docs/hierarchical_tree_design.md) and
  [docs/hierarchical_tree_design.md](docs/hierarchical_tree_design.md).
- **Per-leaf quantizers** (`--quantizer`): each leaf stores quantized codes
  with a quantizer chosen per build. Local quantizers (per-leaf rulers /
  codebooks) are trained at leaf flush time and stay safe under appends and
  rebalancing; global quantizers train on a 20K random reservoir and are
  frozen at build. See
  [docs/quantizer-selection.md](docs/quantizer-selection.md).
- **Routing plane (stage-1)**: a PCA-projection plane of per-vector
  signatures that routes queries to candidate leaves before any leaf extent
  is read. On by default at build time (`--no-plane-attach` to skip;
  `--no-plane` at search to disable). At b1g encoding it costs 16 B/vector
  and buys ~+17% cold QPS at iso-recall (validated on cohere/arxiv/deep).
  See [docs/adaptive-routing-plane-design.md](docs/adaptive-routing-plane-design.md).
- **Rerank**: FP32 decode-rerank by default (`--no-rerank` to keep raw PQ
  distances), or **exact rerank** against the original base vectors via
  `--exact-rerank-base` (mmap'd read-only; harness-measured perfect recall
  and faster than decode-rerank).
- **Mutations**: batch insert (`tree-insert`), delete by row id
  (`tree-delete`), `tree-vacuum` (repair stale filter summaries after
  deletes), `tree-defrag` (compact fragmented leaf extents and shrink the
  file), and `fsck --repair 1` (rebuild bitmap/free-list from a tree walk).
  **Mutation commands require a plane-less tree** (`--no-plane-attach` at
  build): attached routing planes are immutable. `tree-delete` additionally
  requires a global PQ/PRQ quantizer (not the scalar/local families).
- **Filtered search**: `--filter column:op:value` predicates (repeatable,
  AND-ed; ops include `eq/ne/lt/le/gt/ge/between/prefix/in/not_in/
  contains/contains_any/contains_all/geo_radius/geo_box`) against filter
  columns ingested from a `--label-file` parquet.
- **Payloads**: with parquet input, `--payload-col` streams a string/binary
  column into per-leaf payload extents, fetched at search via
  `--with-payload`.
- **Batching**: `tree-search --batch-window N` routes a window of queries,
  inverts the probe sets, and sweeps each unique probed leaf once in page
  order — coalesced I/O at zero extra engine DRAM, results identical to
  per-query search.

## Quick start

```sh
./bootstrap.sh                                  # fetch nsync + NumKong submodules
meson setup build && meson compile -C build     # build (C++17-compatible)
meson test -C build                             # run the test suite (31 files)

# Build a tree index from .fbin
./build/tools/sextant build-tree --input data.fbin --index mytree

# ...or directly from parquet
./build/tools/sextant build-tree --input data.parquet --vector-col embedding \
    --index mytree --payload-col text

# Search (recall computed if --ground-truth given, GTMM format)
./build/tools/sextant tree-search --index mytree --query q.fbin --topk 10

# Mutate + maintain (mutation commands need a plane-less build: add --no-plane-attach
# above; tree-delete also needs --quantizer pq/prq/anisotropic_pq)
./build/tools/sextant tree-insert --index mytree --vectors more.fbin --start-row-id 1000000
./build/tools/sextant tree-delete  --index mytree --row-ids ids.txt
./build/tools/sextant tree-vacuum  --index mytree
./build/tools/sextant tree-defrag  --index mytree
./build/tools/sextant fsck        --file mytree          # add --repair 1 to rebuild the bitmap/free-list

# Parameter sweep: n-probe × fastscan-W recall/QPS grid with shared scans
# (--fastscan-w plus a probe spec, e.g. --probe-fraction, are required)
./build/tools/sextant sweep --index mytree --query q.fbin --ground-truth gt.gtmm \
    --probe-fraction 0.1 --fastscan-w 100,300
```

## CLI commands

| Command | Purpose |
|---|---|
| `build-tree` (alias `build-tree-pca`) | Build a hierarchical IVF tree index (single file). PCA streaming build; `.fbin` or parquet input. |
| `tree-search` | Search a tree index; computes recall with `--ground-truth`. |
| `tree-insert` / `tree-delete` | Batch-insert vectors (`.fbin`) / delete by row ID into a plane-less tree (v1 planes are immutable). |
| `tree-vacuum` | Repair stale filter summaries after deletes. |
| `tree-defrag` | Compact fragmented leaf extents + shrink the file. |
| `fsck` | Check a tree index file; `--repair 1` rebuilds the bitmap/free-list. |
| `plane-attach` | Attach a routing plane to an existing plane-less tree (`--enc b1g\|u4lm`, `--rank`, `--train-rows`; also needs `--index` and `--base`). |
| `sweep` | n-probe × W recall/QPS grid with shared scans (one scan per n-probe serves all W); requires `--fastscan-w` plus a probe spec (`--n-probe` or `--probe-fraction`); scan-feedback rows via `--feedback fixed:F\|stall:M\|kth:M`. |
| `trace` | Replay scan-feedback stop rules on an EngineTrace file. |
`tree-insert` / `tree-delete` / `tree-vacuum` / `tree-defrag` / `plane-attach`
are **internal/experimental** (help text says so): v1's public surface is the
immutable build -> search -> drop lifecycle. The flat Vamana-graph engine and
its `build` / `search` / `insert` / `analyze` / `autobuild` commands were
removed for the v1 library release — preserved at git tag `vamana-eol`.

### Key `build-tree` flags

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Base vectors, `.fbin` or `.parquet` (glob source supports patterns) |
| `--vector-col` | `embedding` | Vector column name in parquet |
| `--vector-fp16` | off | Vector column is a FIXED_LEN_BYTE_ARRAY of fp16 halves (`dim`×2 bytes/row); converted to fp32 on read |
| `--index` | (required) | Output tree file path |
| `--k-root` | 0 (auto) | Root branching factor |
| `--leaf-capacity` | 5000 | Max vectors per leaf |
| `--pq4-m` | 0 (dim/4) | PQ subquantizers |
| `--pq-bits` | 4 | Scan codebook bits (4 or 8) |
| `--quantizer` | `scalar_shape` | Quantizer family: `scalar_shape`, `scalar_lloydmax`, `scalar_uniform`, `pq`, `prq`, `local_pq`, `anisotropic_pq` |
| `--pca-dims` | 32 | PCA dimensions |
| `--max-lloyd-passes` | 10 | Max streaming Lloyd passes |
| `--closure-mult` | 0.15 | Closure epsilon multiplier |
| `--threads` | 0 (auto) | Build threads |
| `--filter-data` / `--label-file` | — | Filter column sidecar (`.fdat`) / parquet with filter columns |
| `--payload-col` | — | Parquet column streamed into per-leaf payload extents |
| `--plane-enc` | `b1g` | Plane encoding: `b1g` (16 B/vec) or `u4lm` (64 B/vec, hard corpora) |
| `--plane-rank` / `--plane-train-rows` | 128 / 20000 | Plane PCA rank / training sample |
| `--plane-layout` | `blob` | Plane block storage: `blob` (v1, contiguous extent) or `leaf` (v2, rides leaf extents) |
| `--no-plane-attach` | off | Skip the default routing-plane attachment |
| `--metrics-file` | — | Append build phase metrics as JSON lines |

### Key `tree-search` flags

| Flag | Default | Description |
|---|---|---|
| `--index`, `--query` | (required) | Tree file; queries `.fbin` or `.parquet` |
| `--topk` | 10 | K nearest neighbors |
| `--n-probe` / `--n-probe-ln` | 0 (manifest) | Root probe count / leaf probe count per root child |
| `--probe-fraction` | 0 (manifest, new trees 0.5) | Probe budget as a corpus fraction (scale-stable; dbpedia 100K→933K: f=0.25 ≈0.96, f=0.5 ≈0.99 recall@10) |
| `--fastscan-w` | 300 | Rerank shortlist per shard |
| `--no-plane` | off | Disable stage-1 plane routing (legacy centroid descent) |
| `--plane-pre-prune` | 0.25 | Two-stage routing: fraction of leaves surviving to the plane sweep (measured best; 0 = off) |
| `--no-rerank` | off | Use raw PQ distances (skip FP32 rerank) |
| `--exact-rerank-base` | — | Original base `.fbin`; rerank against true vectors (implies rerank) |
| `--adaptive-probe-gap` / `--adaptive-w-gap` | manifest / **auto (2.5 @ 384B codes)** | Geometric gap pruning / adaptive shortlist cut: results become a variable-length list cut at the first reranked distance gap past k — the caller re-scores. Auto tau is per scan-code size (2.5 ≥ 288B, 5.0 below). `0` = fixed top-k. Measured CulturaX synth queries: +4pp recall@10 at iso-QPS, 2.5x fewer caller rescans |
| `--threads` / `--search-threads` | auto / serial | Query-parallel / within-query leaf-parallel scan threads |
| `--scan-i8` | −1 (auto) | Scalar scan kernel i8 SDOT/VNNI mode (0 float FMA, 1 int8, 2 int8+dual residual) |
| `--batch-window` | 0 (per-query) | Subtree-major batch mode window size |
| `--cache-mb` | 0 (mmap) | Engine-owned W-TinyLFU leaf cache in MiB (warm results must be labeled with this size — see BENCHMARK_RULES) |
| `--cache-window-pct` | 1 | W-TinyLFU window as % of capacity (Caffeine default) |
| `--plane-cache-mb` | 0 (mmap) | Independent W-TinyLFU cache for plane row blocks |
| `--passes` | 1 | Run the query set N times; N≥2 discards the first pass as warmup |
| `--filter` | — | Filter predicate (repeatable, AND-ed) |
| `--with-payload` / `--payload-text` | off | Fetch payload blobs with results / print as text |

## Performance guide

Every recipe below assumes the measured-best defaults: `scalar_shape`
quantizer, b1g r128 plane auto-attached, plane routing with pre-prune 0.25,
engine caches off (zero DRAM). Zero flags = the validated config; each flag
is an explicit contract choice
([docs/blog-cold-path.md](docs/blog-cold-path.md), canonical-table-v3).

| Goal | Command sketch | Expected result |
|---|---|---|
| Fastest build | `build-tree corpus.fbin --threads 16` | Default tree (best quantizer + plane, streaming, bounded RSS). arxiv-1M × 768: ~28 s wall at 16 threads ([results/build_cpu_attribution_20260909.md](results/build_cpu_attribution_20260909.md)) |
| Highest throughput | `tree-search tree queries.fbin --batch-window 256 --threads 16 --probe-fraction 0.05` | Canonical numbers measured at bw 256, 16 threads, t8: f=.05 → **401 cold / 525 warm QPS** @ recall@10 0.8921 (b1g plane, 16 threads) |
| Balanced (default operating point) | `tree-search ... --batch-window 256 --probe-fraction 0.1` | **332 cold / 397 warm QPS @ 0.9174**; 462 MB leaf/q vs legacy f=.2's 917 MB at 260/276 @ 0.9202 |
| Warm serving | `tree-search ... --batch-window 256 --cache-mb 8192 --plane-cache-mb 512` | Zero-DRAM by default; caches never lose to mmap. Full residency: 397 → 457 QPS (+15%), +8–13% at half a gigabyte |
| Max recall | `--probe-fraction 0.2+`, or `--exact-rerank-base corpus.fbin` | Exact rerank = 1.000 recall (containment-bound) but the resident fp32 base is 6.5× the tree at 10M (30 GB vs 4.6 GB) — [docs/design_decisions.md](docs/design_decisions.md) |

Notes:

- **Build**: `--threads` defaults to 0 (auto = hardware_concurrency); the
  canonical numbers use 16. Build is parallel and streaming — bounded RSS
  (~224 MB peak over a 3.8 GB corpus read at 1B scale). Use
  `--no-plane-attach` only if you need mutation commands: attached v1
  planes are immutable.
- **Throughput/quality dial**: `--probe-fraction`. f=.05 vs f=.1 trades
  ~2.5 pp recall for +21% cold QPS (401 vs 332) at half the leaf bytes.
- **Batch window 256** is the deployment shape; canonical-table-v3 measured
  at `--batch-window 256`. `--search-threads` (0=serial) parallelizes the
  within-query leaf scan, orthogonal to `--threads`.
- **Rerank is on by default**; `--no-rerank` trades recall for a cheaper
  scan. Keep it on unless you re-score externally.
- **Measurement honesty**: follow [docs/BENCHMARK_RULES.md](docs/BENCHMARK_RULES.md)
  — cold means drop_caches + sync before measured passes; warm is valid
  only with engine caches and must be labeled with `--cache-mb`. QPS is
  per-pass throughput (`qcount × timed_passes / secs`); numbers from
  different `--passes` counts are not comparable.

## Quantizers

The `--quantizer` flag selects the leaf-code representation
([docs/quantizer-selection.md](docs/quantizer-selection.md)):

| Value | Kind | Notes |
|---|---|---|
| `scalar_shape` (default) | global per-dim shared-shape ruler | Best general default |
| `scalar_lloydmax` / `scalar_uniform` | global per-dim rulers | `scalar_lm` / `scalar_uniform` families; i8 SDOT/VNNI scan kernels |
| `pq` / `anisotropic_pq` | global PQ codebooks | k-means / covariance-anisotropic training on a 20K reservoir; frozen at build |
| `local_pq` | per-leaf residual codebooks | Smallest footprint |
| `prq` | product residual quantizer | `--prq-nsplits`, `--prq-encode-mode greedy\|beam\|icm`, `--prq-*` training knobs |

Global quantizers train on a 20K random reservoir and their rulers/codebooks
are **frozen at build**: appended vectors are encoded against the build-time
distribution. Pick them when the corpus is homogeneous and stable; use a
local quantizer when data drifts or appends are expected.

## Input formats

- `.fbin` (float32), `.ibin` (int8), `.bbin` / `.bvecs` (uint8) — all
  little-endian:

  ```
  [uint32 n][uint32 dim][n × dim × sizeof(T)]
  ```

- **Parquet** (vector and query inputs): backed by the vendored
  [carquet](subprojects/carquet) C parquet library (v0.7.1, MIT, Meson
  subproject) via `sources/parquet_source.cpp` and
  `sources/parquet_glob_source.cpp` (glob patterns). `tools/fbin_to_parquet`
  converts the other way. `--vector-col` names the embedding column;
  `--payload-col` streams an opaque payload column into leaf extents;
  `--label-file` joins filter columns by row position.

  **Vector column layout matters for build speed.** The build re-reads the
  corpus ~12× (sample, ~10 Lloyd passes, emission). Layout cost per pass on
  the CulturaX 1M corpus (2.9M×768, warm page cache, 16 threads):

  | layout | per-pass Lloyd | full build | corpus size |
  |---|---|---|---|
  | `list<float>` (fp32, zstd) | 2.9 s | 53 s | 9.5 GB |
  | FIXED_LEN_BYTE_ARRAY fp32, zstd | 1.9 s | 40 s | 9.5 GB |
  | FIXED_LEN_BYTE_ARRAY fp32, uncompressed | 1.1 s | 29 s | 12 GB |
  | FIXED_LEN_BYTE_ARRAY fp16, zstd (`--vector-fp16`) | 1.1 s | 29 s | 5.5 GB |
  | FIXED_LEN_BYTE_ARRAY fp16, uncompressed (`--vector-fp16`) | **0.7 s** | **24 s** | 7.9 GB |

  `list<float>` is fully supported (you don't control customer input) but
  pays zstd decode + def/rep level assembly on every pass — ~2.2× slower than
  fp16 FLBA and ~4.5 GB larger on disk. Recommended layout: a fixed-size
  binary (`dim`×2) fp16 column, uncompressed for the mmap zero-copy path.
  Recall is unaffected: fp16 storage measured recall@10 0.857 vs 0.855 fp32
  on the tie-saturated CulturaX corpus. `scripts/emb_to_flba.py` converts
  a `list<float>` corpus to any of the FLBA layouts.

Use `tools/fvecs_to_fbin` to convert `.fvecs`/`.ivecs` corpora.

## Caching, storage, and locking

- **Engine-owned caches, not the OS page cache.** The tree search path has a
  W-TinyLFU `LeafExtentCache` (`--cache-mb`) and an independent plane-row
  cache (`--plane-cache-mb`); both evict under real memory pressure and are
  sized explicitly. See
  [docs/cache_architecture.md](docs/cache_architecture.md) for the
  architecture and the measured failure modes of relying on the Linux page
  cache.
- **Benchmark discipline is binding.** All published numbers follow
  [docs/BENCHMARK_RULES.md](docs/BENCHMARK_RULES.md): cold storage, explicit
  `drop_caches` before measured passes, warm results labeled with the cache
  size.
- Storage I/O is `O_DIRECT`/mmap-based via `src/storage/`
  (`direct_io`, `block_allocator`, `code_stream`, `tl_cache`, …). Tree
  indices are a **single file** with extent allocation, a bitmap/free-list,
  and per-leaf extents.
- Cross-thread locking contracts live in
  [include/sextant/sync.hpp](include/sextant/sync.hpp) and
  [docs/locking_contracts.md](docs/locking_contracts.md).

## C API

`include/sextant/sextant_c.h` (implemented in `src/capi/sextant_c.cpp`) is a
plain-C99 ABI over the tree index — the sole seam the (post-release) DuckDB
extension will use. The v1 surface covers: `.fbin`/in-memory/streaming-push
builds (the push API accepts typed filter columns — Int32/Int64/Float/
String/Set — and per-row payload blobs), open/close, single-query search
with an optional conjunct list of predicates covering the full engine
surface: EQ/NEQ/LT/LE/GT/GE/BETWEEN on numerics, EQ/NEQ/PREFIX/IN/NOT_IN on
strings (IN/NOT_IN also on numerics, values passed as strings),
CONTAINS/CONTAINS_ANY/CONTAINS_ALL on Set columns, and GEO_RADIUS/GEO_BOX
(`column` = latitude column, `geo_lng_column` = longitude column, both
numeric),
shared-predicate batch search with row-major output, opaque leaf-location
output plus `sextant_fetch_payload()` / `sextant_fetch_vector()` for O(1)
result blob/decoded-vector fetch (locations stay valid until close on the
default read-only handles), and `sextant_index_count()`.

**Exception-safety contract:** every entry point catches all exceptions
internally — no exception ever crosses the C boundary (DuckDB TUs are
`-fno-exceptions`). Failures surface as negative returns with a message
written into the caller's `char* err` buffer. Enum values and struct layouts
are ABI-fixed; structs evolve by tail-append only.

```c
sextant_build_opts bo = sextant_default_build_opts();
sextant_filter_col_def cols[2] = {{"year", SEXTANT_COL_INT32},
                                  {"tags", SEXTANT_COL_SET}};
char err[256];
void* b = sextant_build_begin(&bo, dim, cols, 2, /*has_payload=*/1, err, sizeof err);
/* per-chunk Set input: row r's elements are indices offsets[r]..offsets[r+1]
   into elem_data/elem_lengths (strings are length + pointer, not NUL-terminated) */
sextant_set_values tags = {counts, offsets, elem_data, elem_lengths};
const void* col_ptrs[2] = {years, &tags};
sextant_build_push(b, vecs, n_rows, col_ptrs, payload_offs, payload_bytes,
                   err, sizeof err);
sextant_build_finish(b, "/tmp/t.tree", err, sizeof err);   /* frees b */

void* idx = sextant_open_index("/tmp/t.tree", err, sizeof err);
sextant_predicate p = {.column = "tags", .op = SEXTANT_PRED_CONTAINS,
                       .str_value = "sports"};   /* designated init zero-fills
                                                    the tail-appended fields */
void* leaf[10]; uint32_t slot[10]; uint64_t ids[10]; float d[10];
int32_t n = sextant_search2(idx, query, &sopts, &p, 1, ids, d, leaf, slot,
                            10, err, sizeof err);
if (n > 0) {
    uint8_t buf[64];
    uint32_t size = sextant_fetch_payload(idx, leaf[0], slot[0],
                                          buf, sizeof buf);  /* full size */
}
```

Smoke/QPS harnesses: `scripts/capi_smoke.cpp`, `scripts/capi_qps.cpp`; the
v1 surface is covered by `test/test_capi.cpp`.

## Release & compatibility (v1)

- **On-disk format**: not finalized. **No compatibility guarantees until
  v1.0** — a tree built by any pre-1.0 build may be unreadable by the next;
  rebuild indexes when upgrading. Reserved format space (`tombstone_count`,
  `LeafState::ACCUMULATING`) is held for v3 serving (tombstoning /
  raw-insert bootstrap) and is never written or read in v1.
- **C API** (`include/sextant/sextant_c.h`) is the stable external seam:
  ABI-fixed enums, tail-append-only structs, exception-proof boundary
  (see above).
- **Determinism**: search results use the total order (distance, row_id
  DESC) — identical inputs give identical results regardless of batching,
  scan order, or thread count. Multithreaded *builds* are not
  byte-identical (documented notfix; `--threads 1` for reproducible
  trees).
- **Platforms**: x86-64 with AVX-512 (VNNI) is the primary target.
  aarch64 is validated on Neoverse-V2 (NEON DotProd/i8mm; SVE2 builds
  take the NEON kernels — see the analysis at the dispatch site in
  `coder_util.hpp`). The 4-query batched scan kernel is bit-identical to
  the reference on both; on V2 the compiler's auto-vectorized reference
  matches the explicit NEON kernel, so the kernel's value there is
  determinism, not speed.
- **carquet** (parquet reader/writer, `subprojects/carquet`) is pinned as
  a submodule; development happens on the fork
  `github.com/rescbr/carquet` (upstream PR pending).

## Evaluation: recall, proximity, and ground truth

**Recall@k** is the standard ANN quality metric: for each query, compute the
true top-k nearest neighbors by brute force (exhaustive scan using the same
distance function the index uses — L2-squared by default), then measure what
fraction of those k ids appear in the index's top-k results. Averaged over
all queries. Ground truth is in GTMM format; `scripts/gen_ground_truth.py`
and `tools/sextant_filtered_gt` generate it (including filtered GT).

However, recall@k has a subtle failure mode on **clustered data** — datasets
where many vectors sit at (nearly) the same location. The next section
explains why, and what Sextant measures instead.

## Why not row-id recall?

A natural question: why measure PQ quality with **distortion** instead of
just **recall** (fraction of true neighbors the PQ ranking recovers)? The
answer is that row-id recall is fragile on datasets with near-duplicate
structure — exactly the datasets that real-world vector indexes serve.

### The problem: tie-breaking makes "the top-10" arbitrary

Many real datasets contain groups of vectors at (nearly) the same location:
address or geolocation embeddings (multiple rows per location), document
embeddings (near-duplicate pages), entity embeddings (repeated records). For
a query landing in such a cluster, dozens or hundreds of base vectors may be
within the 10th-NN distance — all equally near, all legitimate results.

Ground-truth generators must pick exactly 10 ids for recall@10. When >10
vectors are tied at the boundary, which 10 get picked is determined by
implementation detail. The "true top-10" is **arbitrary** — a different chunk
size or scan order produces a different ground truth.

This has a concrete consequence: **even exact brute-force search cannot score
recall@10 = 1.0** against such a ground truth. On a geolocation embedding
dataset (768-dim, 1.26M vectors), brute-force top-10 scores only **0.63
recall@10** against the generated GT — not because brute force misses
anything, but because it returns 10 different (equally-near) vectors than the
10 the GT arbitrarily marked. No index, however perfect, can exceed this
ceiling.

### What we measure instead

The PQ probe (`src/engine/probe.cpp`) reports four diagnostics, only one of
which drives selection:

| Metric | What it measures | Used for selection? |
|---|---|---|
| **distortion** | Median `\|1 − pq_dist / true_dist\|` over true neighbors. How much PQ distorts the distances it sees. Geometric, direction-agnostic, transfers across dataset classes. | **Yes** — the selection signal. |
| **band_recall** | Fraction of true neighbors for which the PQ top-30 shortlist contains *some* vector within the 10th-NN distance. Cluster-aware. | No — diagnostic. |
| **ties@10** | Fraction of probe queries where more than 10 vectors fall within the 10th-NN distance. When high, recall@10 is structurally capped below 1.0. | No — diagnostic. |
| **ties@30** | Same, at the shortlist boundary. When high, even the PQ top-30 shortlist boundary is ambiguous. | No — diagnostic. |

**Distortion** is the selection signal because it isolates PQ's own property
(distance fidelity) from downstream effects (graph navigation,
tie-breaking). See
[docs/proximity_vs_recall.md](docs/proximity_vs_recall.md) for the proximity
metric used in evaluation.

## Flat graph engine (legacy)

The original engine — a flat, partitioned Vamana graph on O_DIRECT sidecar
files (`.graph`/`.codes`/`.meta`/`.manifest`) with MemGraph RAM-cached
entry-point neighborhoods, PageShuffle, and PageSearch — is still present and
buildable (`build`, `search`, `insert` commands; `src/algo/`, `src/engine/`,
`src/storage/memgraph*`). It shares the quantizer stack (`src/quant/`), the
parameter resolver, and the PQ distortion probe with the tree engine. Two of
its design results remain load-bearing and are preserved below.

### PQ-codes-only build

Raw vectors are streamed through once to train a product quantizer; the
graph is then constructed from PQ codes alone. Native AiSAQ loads
full-precision vectors into RAM for the whole build. At 100M vectors /
dim-128 this is ~30 GB of build RAM for Sextant versus ~98 GB for native
AiSAQ — roughly 3× less. `--pq-segments` (renamed from `--pq-m`) sets `m`;
`--pq-bits auto` runs the distortion probe described above.

### `closure_factor` partition overlap

For partitioned flat builds, shard overlap is controlled by a distance-ratio
threshold `c = (1 − f_target)^{−1/d_eff}` rather than naive k-base
replication (`src/engine/estimator.cpp`). This cuts vector replication from
~100% (native AiSAQ's `k_base = 2`) to ~15%, saving ~6.7× wasted build
compute on merges. The SPANN-style `--closure-epsilon` (absolute margin,
default auto) supersedes it; see
[docs/closure_factor_derivation.md](docs/closure_factor_derivation.md).

## Benchmarks

Canonical current results live under `results/`:

- `results/plane_quant_20260911/` — plane × quantizer matrix (cohere10m,
  arxiv1m, dbpedia933k; warm/spread traffic)
- `results/fixes_20260909/`, `results/audit_20260909*` — post-audit fix
  validation
- `results/leaf_cache/` — engine cache vs page-cache behavior (the
  measurement that reshaped [docs/BENCHMARK_RULES.md](docs/BENCHMARK_RULES.md))
- `docs/blog-cold-path.md` carries the **canonical-table-v3** numbers
  (current-defaults trees, 2026-09-13)

### Early validation (flat engine, July 2026)

These SIFT-1M numbers date from the flat Vamana-graph engine and do **not**
describe the current tree path; they are kept as historical validation of
the PQ-codes-only build:

| Metric | Value (flat engine, SIFT-1M, auto PQ) |
|---|---|
| Build time | 100 s |
| Index size | 304 MB (inline_pq=0) |
| Recall@10 | 0.9975 |
| QPS (1 / 10 threads, full cache) | 817 / 2,036 |

## Metric handling and normalization

- The engine builds with **L2** (squared Euclidean) or **inner product**
  (`--metric l2sq|ip`). The Neyshabur–Srebro result guarantees an L2 index
  serves cosine/IP queries well; `analyze` warns when `--metric ip` is used
  on non-normalized data (recall collapse risk).
- **Cosine = L2-normalize then L2.** Normalization is the caller's job; the
  engine core never normalizes.
- Integer vectors (`uint8` / `int8`) are **cast to float** — no scaling, no
  normalization.

## Project structure

```
sextant-engine/
├── include/sextant/    # public headers (config, builder/searcher/estimator,
│                       # tree API, sextant_c.h C ABI, sync.hpp locking)
├── src/
│   ├── tree/           # IVF tree: ivf_tree build/mutate/search/index,
│   │                   # plane.cpp, leaf_coder.hpp, leaf_extent_cache,
│   │                   # tree_estimator, coders/, batch_scheduler
│   ├── engine/         # Builder/Estimator/Searcher/Index, resolve_params,
│   │                   # probe (distortion probe), partition
│   ├── capi/           # C ABI (sextant_c.cpp)
│   ├── quant/          # pq, anisotropic_pq, scalar_lloydmax, PRQ quantizers
│   ├── storage/        # O_DIRECT I/O, block allocator, memgraph, tl_cache,
│   │                   # code_stream (graph-path sidecars)
│   ├── algo/           # Vamana core (graph path)
│   ├── simd/           # kernels
│   └── util/
├── sources/            # fbin / parquet / parquet-glob vector sources
├── test/               # 31 test files (tree, quantizers, cache, storage, …)
├── tools/              # sextant CLI, sextant_bench, sextant_filtered_gt,
│                       # fvecs_to_fbin, fbin_to_parquet, pq_explore, analyze
├── scripts/            # benchmark drivers, spikes, ground-truth generators
├── docs/               # design docs + results narratives (see links above)
├── subprojects/
│   └── carquet/        # vendored C parquet library (v0.7.1, MIT)
├── results/            # canonical benchmark outputs
├── third_party/        # nsync, NumKong (submodules); CTPL, cmdline (committed)
├── bootstrap.sh        # submodule init
├── meson.build
└── meson_options.txt   # -Dtests=enabled, -Dbenchmarks=disabled
```

## Dependencies and licenses

| Dependency | License | Source |
|---|---|---|
| nsync | Apache-2.0 | git submodule (native Meson wrapper) |
| NumKong | Apache-2.0 | git submodule (header-only) |
| spdlog | MIT | Meson WrapDB |
| googletest | BSD-3-Clause | Meson WrapDB |
| carquet | MIT | vendored Meson subproject (`subprojects/carquet`, v0.7.1) |
| CTPL | Apache-2.0 | committed in `third_party/` |
| cmdline | BSD-3-Clause | committed in `third_party/` |

All are compatible with the project license below.

## macOS limitation

macOS is a **development and validation platform only**. macOS `F_NOCACHE`
is advisory and is not enforced direct I/O the way Linux `O_DIRECT` is, so
the kernel page cache can mask true SSD behavior. Performance claims are
made on **Linux only**. On macOS, validate **correctness, recall, and
relative profiling** — not absolute throughput.

## License

SSPL-1.0-only. See [LICENSE](LICENSE).
