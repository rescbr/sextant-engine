# Post-Layer-5 Profile — arxiv-nomic 1.34M @ 8t production point

**Captured:** 2026-07-21 on GCP `c4a-standard-8-lssd` (Axion / Neoverse-V2, arm64-sve2).
Binaries built from HEAD post-L5 (commits through `dcc0dd4`).

## Workload
- Index: arxiv-nomic 1.34M (R=32, L_build=100, pq_m=96/8-bit, K=1 monolithic)
- Build time: 203.7s (autobuild picked R=32 from `--proximity-target 0.95`)
- 1000 queries, topk=100, L=400, 8 threads

## Production-point baseline

| Config | QPS | recall@100 | TL-L1 hit | W-TinyLFU hit |
|---|---|---|---|---|
| L=400 rr=2 (production) | **1689.7** | 0.9086 | 94.2% | 99.4% |
| L=400 rr=1 (no rerank)  | **1917.5** | 0.7286 | 94.4% | 99.4% |

**Rerank tax:** 228 QPS (12%) to bridge the 0.73 → 0.91 PQ-only recall gap.

**Gap to VIBE leaders:** SymphonyQG / ScaNN ~3200 QPS @ recall@100=0.95 →
we are 1.9× off at recall 0.91 (easier target).

## Methodology note (the measurement artifact)

First profile attempt showed **~36% kernel page-cache cycles** (`__arch_copy_to_user`,
`__pi_clear_page`, `folio_*`). Suspected I/O bottleneck. Investigated:

- **W-TinyLFU hit rate is 99.4%** — the engine itself isn't reading much.
- The benchmark loads the **entire 3.85 GB base file** into a `std::vector<float>`
  for rerank (`benchmark.cpp:279-298`) — this `resize()` + `read()` is **one-time
  setup**, not per-query.
- Total wallclock 2.856s; `total search time: 0.521s` → setup is 80% of wallclock
  but 0% of production cost.

**Re-ran with `perf record --delay 2000`** to skip setup. Kernel cycles dropped
from ~36% → ~15% (residual is benchmark file reads + setup tail).

**Verdict: Sextant is CPU-bound at the production point. No I/O bottleneck.
A ram-disk mount would not change the picture** — our app-level LRU + the kernel
page cache already absorb the working set.

## Clean profile (rr=2, search-only via `--delay`)

| % | Function | Category |
|---|---|---|
| **29.2%** | `VamanaCore::beam_search_into` | graph traversal loop |
| **17.3%** | `PqQuantizer::lut_distance_batch4` | SIMD PQ distance (4-way SVE2 gather) |
| 9.95% | `nk_sqeuclidean_f32_sve` (×2) | rerank + benchmark proximity calc |
| 5.7% | `PqQuantizer::lut_distance` | single PQ distance (non-batched path) |
| 2.6% | `PqQuantizer::preprocess_query` | query LUT setup |
| 1.7% | `MemGraph::precise_vec` | FP16 ball lookup |
| 1.3% | `PagedNodeStore::batched_read` | cache-miss I/O |
| ~15% | kernel | benchmark file reads + residual |

## Clean profile (rr=1, search-only via `--delay`)

| % | Function | Category |
|---|---|---|
| **32.0%** | `VamanaCore::beam_search_into` | graph traversal loop |
| **17.4%** | `PqQuantizer::lut_distance_batch4` | SIMD PQ distance |
| 5.5% | `PqQuantizer::lut_distance` | single PQ distance |
| 4.1% | `nk_sqeuclidean_f32_sve` | benchmark proximity instrumentation (not engine) |
| 2.5% | `MemGraph::precise_vec` | FP16 ball lookup |
| 2.5% | `PqQuantizer::preprocess_query` | query LUT setup |
| 1.5% | `PagedNodeStore::batched_read` | cache-miss I/O |

## Conclusions

1. **Engine hot path is `beam_search_into` + `lut_distance*` = ~52-55%** at both
   rerank settings. This is the algorithmic core; gains here translate directly
   to QPS.
2. **Rerank costs 12% QPS** at this scale (bridges 0.73→0.91 recall). Raising
   the PQ-only ceiling eliminates this tax.
3. **No I/O bottleneck.** W-TinyLFU 99.4%, TL-L1 94%. Sextant is CPU-bound.
4. **`lut_distance_batch4` is paying off** (17% — the SVE2 4-way gather from
   Layer 4). Single `lut_distance` is only 5.5%; further batch widening may
   help but with diminishing returns.

## Levers to close the 1.9× gap to VIBE leaders

| Lever | Mechanism | Expected impact | Effort |
|---|---|---|---|
| Raise PQ-only recall ceiling | Better quantization | Eliminates 12% rerank tax, +1.0 pp recall ceiling → closer to 0.95 peer target | Large |
| `beam_search_into` SIMD wins | Tighter neighbor-list distance batching, prefetch ahead of LUT gather | 5-15% on the 32% hot spot | Medium |
| Cache-miss path | `batched_read` 1.3% — not worth attacking | <1% | Skip |
| `preprocess_query` | Already cheap (2.5%); the LUT materialization is the cost | 1-2% | Small |

### Note on "raise the PQ-only ceiling"

This index was built with **plain PQ (OPQ off)**. The `pq_opq` flag exists and
works, but per `docs/quantization_findings.md` OPQ was already validated at
1.34M scale: **+0 pp recall** (0.7300 → 0.7300). The +1.03pp gain on arxiv100k
does not transfer — concentration-of-measure at scale makes the ceiling a
structural problem, not a codebook problem. **Do not re-run with `--pq-opq 1`
expecting a recall change.**

The structural levers (none implemented):
- **ScaNN-style anisotropic objective** — prototyped, negative result (see
  `docs/quantization_findings.md` §1).
- **SymphonyQG-style two-level quantization** (coarse PQ → fine refinement
  without reading FP32 vectors) — untried. Would eliminate the FP32 rerank
  bandwidth cost entirely.
- **Higher m or 10-bit PQ** — increases code size (more I/O) but is the
  brute-force way to push the ceiling.

## Files
- `perf_rr1.data` — perf record at rerank=1, `--delay 2000` to exclude setup
- `perf_rr2.data` — perf record at rerank=2, `--delay 2000` to exclude setup
