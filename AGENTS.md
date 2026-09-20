# AGENTS.md

Guidance for coding agents (and humans in a hurry) working in this repo.

## What this is

Sextant engine: a streaming-built, single-file hierarchical IVF tree for
ANN vector search (see `src/tree/`). C++20, meson build, x86-64-v3+ SIMD
(AVX-512/VNNI kernels; build with `-Dcpu_arch=x86-64-v3` or better).
Ships as a library + CLI (`tools/sextant`), consumed from SQL by the
sibling `sextant-duckdb` extension (statically linked via
`include/sextant/sextant_c.h`).

## Build & test

```sh
./bootstrap.sh                    # submodules (carquet, numkong)
meson setup build-x86 -Dcpu_arch=x86-64-v3   # release
ninja -C build-x86
meson test -C build-x86           # 24 suites, run sequentially
# ASAN/UBSAN debug build:
meson setup build-dbg -Dbuildtype=debug -Db_sanitize=address,undefined
MALLOC_PERTURB_=213 meson test -C build-dbg
```

Verification standard: BOTH the release and the ASAN suite must pass
before a change is considered done. Suites are CPU-heavy at the tail
(`test_tree_insert_delete`, `test_quantizer_families`) — expect ~25 min
release, ~15 min ASAN under perturbation.

## On-disk format contract

The tree manifest format is **v1** (`kManifestFormatVersion` in
`src/tree/tree_manifest.cpp`). Parsing accepts only v1 and fails loudly
on anything else — there is no legacy support, by doctrine. If you change
the manifest or any on-disk layout, bump the version and expect every
existing tree (including the DuckDB extension's committed fixture
`test/data/small.tree` in sextant-duckdb) to require a rebuild.

## Conventions

- Errors: throw `sextant::Error` with an `ErrorCode`; user-facing text
  states what failed and what to do (see existing open/deserialize paths).
- Benchmarks use the rules in `docs/BENCHMARK_RULES.md` (cold-path
  discipline: kernel caches dropped, ZFS synced). Datasets:
  `docs/benchmark_datasets.md`. Raw benchmark outputs are NOT in the
  repo — don't commit them.
- Historical context: several docs are marked "historical (flat-engine
  era)" — they explain live code, read them for rationale, don't cite
  their numbers as current.
- No CI. Tests are the contract; run them.

## Gotchas

- `meson test` output interleaves suite logs; failures go to
  `build-*/meson-logs/testlog.txt`.
- The superblock keeps two copies; `fsck` (CLI) validates a tree file if
  you suspect corruption during index work.
- Wrap builds (`subprojects/`) are fetched at setup — a stale
  `subprojects/packagecache/` causes confusing meson failures; delete it.
