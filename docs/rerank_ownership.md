# Rerank ownership — architectural decision (2026-07-21)

## Decision

**sextant-engine does NOT do reranking.** Rerank is owned by the query
engine (the DuckDB extension in production, the benchmark tool in
testing).

## Background

Rerank is the final step of approximate nearest neighbor search:

1. **Search engine** (sextant-engine): returns top-N candidates ranked by
   approximate distance (PQ LUT distance). Fast but imprecise due to PQ
   quantization error.
2. **Rerank**: re-score those N candidates using exact distance against
   the ORIGINAL vectors (FP32 or FP16), return the top-k.

The exact distance needs the original vectors, which are NOT in the
search index — the index has only PQ codes (96 bytes for arxiv-nomic 768-dim
vs 3072 bytes for FP32).

## Why rerank belongs in the query engine, not the search engine

### 1. Layering
sextant-engine should not know about the database. It receives a query
vector and returns approximate candidates. The database owns the "source
of truth" for rows and their original vectors. Pushing rerank into the
search engine would either (a) require it to store all original vectors
(doubling index size) or (b) call back into the database (couples layers).

### 2. Storage
Keeping original vectors out of the index keeps it small (PQ codes only).
The 1.34M × 768-dim arxiv-nomic index is ~330MB of PQ codes + graph; the
original FP16 vectors would be ~2GB. Storing them in the index for rerank
would more than double the on-disk footprint.

### 3. Vectorization
DuckDB is already a vectorized execution engine (vectorized batches, SIMD
over columns). The original vectors live in DuckDB tables (or are
accessible via scans). DuckDB's expression engine can compute
`l2sq(query, vec)` over a batch of candidates far more efficiently than a
serial rerank loop. Putting rerank in the search engine would bypass this.

### 4. Policy
Rerank has a query-time POLICY decision: how many candidates to rerank,
when to rerank, recall/latency tradeoff. This depends on user intent,
workload (k=10 vs k=100), and cost model (PQ selectivity, base-vector
retrieval cost). The search engine can't know the user's recall/latency
tradeoff — that's a query-planning decision.

## What each layer owns

```
┌─────────────────────────────────────────────────┐
│  SQL Client                                      │
├─────────────────────────────────────────────────┤
│  DuckDB (query engine)                           │
│   - owns tables, rows, types                     │
│   - owns execution (vectorized, SIMD)            │
│   - owns transaction semantics                   │
│   - RERANK: exact distance over candidate batch  │
├─────────────────────────────────────────────────┤
│  sextant-extension (DuckDB extension)            │
│   - SQL surface: SELECT ... ORDER BY dist(...)   │
│   - binds ANN search calls, manages row_id space │
│   - rerank POLICY (fetch_k, trigger conditions)  │
│   - coordinates: search → scan → rerank → limit  │
├─────────────────────────────────────────────────┤
│  sextant-engine (ANN index)                      │
│   - owns the graph, PQ codes, cache              │
│   - returns APPROXIMATE candidates only          │
│   - knows nothing about SQL, tables, or rows     │
│   - NO rerank, NO exact-distance computation     │
└─────────────────────────────────────────────────┘
```

## Current state (2026-07-21)

- `SearchConfig::rerank_factor` is a vestigial field (flagged as audit
  §9.3 "dead in the engine"). The engine's `search()` never reads it.
  Should be removed in a future cleanup.
- The search engine's `VamanaCore::search()` returns candidates ranked by
  PQ LUT distance, period.
- The benchmark tool (`tools/benchmark.cpp`) currently does rerank itself:
  loads base.fbin into RAM, computes scalar `l2sq_distance` per candidate.
  This is the RIGHT layer (caller of the search engine) but the WRONG
  implementation (serial scalar — should be SIMD + parallel).

## What this means for benchmark performance

At 8t on arxiv-nomic, the search engine scales well (workers process
queries in parallel). The benchmark's serial main-thread rerank is the
bottleneck: 8 worker threads sit in `futex_do_wait` while the single main
thread does `l2sq_distance` over 200 candidates × 1000 queries.

Fixes (all benchmark-side, not engine-side):
1. Move rerank INTO the worker (parallel rerank). Workers share base_all
   read-only.
2. SIMD the l2sq_distance (NEON/SVE2 over 768-dim vectors).
3. Both together.

## What this means for the DuckDB extension (Phase 2)

The extension will replace the benchmark's hand-rolled rerank with a
SQL-level rerank. Two patterns:

```sql
-- Option A: explicit rerank in SQL (composable)
SELECT row_id, vec
FROM items
WHERE row_id IN (
    SELECT ann_search('items_idx', :query, fetch_k => 200)
)
ORDER BY l2sq_distance(vec, :query)
LIMIT 100;

-- Option B: composite function (ergonomic, hides rerank step)
SELECT * FROM ann_search_rerank(
    'items_idx', :query, k => 100, fetch_k => 200
);
```

Option A is preferred — it uses DuckDB's existing vectorized machinery
for the rerank step and keeps the SQL composable.

## The wrinkle: FP16 in the search engine

The search engine DOES use FP16 vectors, but only for NAVIGATION (not
rerank):

- MemGraph caches FP16 "ball" vectors for the entry-point neighborhood
  (~150K of 1.34M for arxiv-nomic). Used by beam_search for the "approach
  phase" — direct FP16 L2sq instead of PQ LUT distance.
- The build path uses FP16 vectors for occlusion-check distances (more
  accurate than PQ).

These are distance-computation shortcuts during graph traversal, NOT
rerank. The distinction: navigation distances only need to be
*consistent* (correctly rank neighbors for the beam); rerank distances
need to be *exact* (true top-k).
