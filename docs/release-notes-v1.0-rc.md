# sextant v1.0-rc release notes

## What v1 is

A vector-search index library for the batch/datalake regime:
**build → search → drop**. The index is an immutable materialized view
(one file, mmap-read at search time) over an embedded corpus, with typed
filter columns (Int32/Int64/Float/Bool/String/Set), opaque per-row
payloads, and filtered ANN search with cardinality-routed predicates.
Mutation (insert/delete/vacuum/defrag) is internal/experimental — the v3
seed — and the flat/Vamana graph stack is gone (preserved at git tag
`vamana-eol`).

Two consuming surfaces:

- **C++** (`IVFTreeIndex`): build_streaming_pca / open / search /
  search_batch / fetch_payload / fetch_vector / stats / fsck.
- **C** (`include/sextant/sextant_c.h`): the stable external ABI —
  full predicate surface, filtered + batch search, payload/vector fetch,
  streaming push build with filter columns and payloads. Exception-proof
  boundary; ABI-fixed enums; tail-append-only structs. This is the seam
  the future DuckDB extension consumes.

## Search semantics

- **Adaptive-W shortlists are the default**: post-rerank, search returns
  up to W>k ids whose distance stays within the adaptive gap of the k-th
  — measured ~117-row shortlists on RAG-shaped queries (tied content
  surfaced together). `--adaptive-w-gap 0` (or `adaptive_w_gap = 0`)
  selects the classic fixed top-k.
- **Deterministic tie-breaking**: final results are ordered by the total
  order (distance, row_id DESC). Identical inputs give identical results
  regardless of single-vs-batch invocation, scan order, or thread count.
  Ties resolve toward higher row_ids so freshly inserted rows stay
  findable among equal-distance content.
- **Filtered search**: conjunctive predicates over filter columns;
  low-selectivity (rare-value) queries route to an exact brute-force
  fallback that honors `exact_rerank_base`. Malformed predicates
  (unknown column, op/type mismatch, geo without a numeric longitude
  column) throw `InvalidParam` — never silently pass-all or
  zero-results.

## Fixed in this release cycle (selected)

- `delete_batch` on payload-bearing indexes silently corrupted the
  payload mapping — now refuses loudly.
- Batched-scan AVX-512 tail mask truncated to 8 bits dropped tail dims
  8–14 when `dim % 16 ∈ 9..14` (canonical 768-dim data unaffected).
- scalar_shape fused rerank double-applied the shape scale in the
  non-AVX-512 fallback (rerank actively degraded results; visible only
  on aarch64 where the fallback executes).
- Plane-on builds crashed with ftruncate EINVAL after search handles
  went read-only (the build's plane-tail reopen needed the writable
  path).
- Read-only search handles: `PageFile` opens `O_RDONLY` (no `O_CREAT`)
  for search — read-only mounts/files work, stray writes are
  structurally impossible.

## Operational notes

- **Input format**: `fixed_len_byte_array` fp16 vectors build ~2× faster
  than `list<float>` (and ~2× the search-side ingest). See the README
  vector-layout table.
- **Multithreaded builds are not byte-identical** (documented notfix);
  `--threads 1` for reproducible trees. Search is deterministic at any
  thread count.
- **ARM timeouts**: `test_quantizer_families` and
  `test_tree_insert_delete` exceed a 30s default on Neoverse-V2 class
  hardware; test timeouts are set to 180s.

## Platform validation

- x86-64 (Zen 5, AVX-512 + VNNI): primary target; 24/24 test suites;
  full CulturaX e2e (2.9M × 768, fp16-flba) ALL GATES PASS — warm
  self-query recall 0.9505 at f=0.1, synthetic RAG recall 0.8355,
  exact-rerank invariant exactly 1.0, rare-value singleton gate 1/1.
- aarch64 (Neoverse-V2 / c4a, clang 21, sve2 build target): 24/24
  suites; dots4 kernel bit-exact vs reference; scalar_uniform i8 scan
  2384 QPS vs 159 float on arxiv100k f=1.0. The explicit NEON scan
  kernels match the compiler-auto-vectorized reference on V2 — the
  i8 scan *mode* is where ARM throughput comes from (15×).
- SME2 (SMOPA i8 tiles): analyzed, not implemented — the scan is a
  natural matmul for the ZA accumulator, but hardware/toolchain
  maturity puts it post-v1. See the SVE2 analysis note at the dispatch
  site in `coder_util.hpp`.

## Carquet

Parquet I/O via the `carquet` submodule, pinned; fork at
`github.com/rescbr/carquet` (upstream PR pending).
