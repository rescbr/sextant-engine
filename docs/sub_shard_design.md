# Sub-Shard Design Document

> **Status:** Design phase. Not yet implemented.
> **Foundation:** absolute-margin closure (shipped), batch4 k-means (shipped),
> c(K) derivation (validated), K=4096 Pareto win (validated).

## Goal

Split large/skewed IVF shards into smaller sub-shards with two-level routing.
This reduces codes scanned per query (the profiling-confirmed bottleneck),
mitigates shard-size skew, and enables future live insert/delete.

## Architecture decisions (validated with user)

1. **Storage:** One file per sub-shard (`shard_NNNN/sub_MMMM/` directories)
2. **Routing:** Two-level — coarse (existing) → sub-centroid (FP16 dist_f16)
3. **Split trigger:** Size threshold (T_max vectors, default ~20k)
4. **FD management:** mmap all, raise ulimit

## On-disk layout

```
<prefix>.shards/
├── manifest              (existing + new fields, see below)
├── centroids.bin         (K × dim FP16, existing — coarse centroids)
├── subcentroids.bin      (NEW: Σ sub_shards × dim FP16, raw)
├── codebook4.bin         (existing, shared)
├── shard_0001/
│   ├── sub_0000/         (NEW directory level)
│   │   ├── .codes4       (SidecarHeader + FastScan blocks)
│   │   ├── .rowids       (SidecarHeader + n × int64)
│   │   └── .manifest     (atomic commit)
│   ├── sub_0001/
│   │   ├── .codes4
│   │   ├── .rowids
│   │   └── .manifest
│   └── shard.manifest    (NEW: sub-shard count for this shard)
├── shard_0002/
│   └── ... (shards with 1 sub-shard = backward compatible)
```

### Backward compatibility

A shard with a single sub-shard is identical to today's flat shard, just
nested one directory deeper. For backward compat, we can either:
- Check for `shard_NNNN/.codes4` (old format) vs `shard_NNNN/sub_0000/` (new)
- Or always use the new format (no backward compat needed — on-disk format
  is NOT stabilized per project conventions)

## Manifest changes

### Top-level manifest (additive)

```
ready
K
dim
n_probe_default
m4
scan_pq_bits
quantizer_type
prq_nsplits
sub_shard_n_probe     (NEW: default 1; sub-shards to probe per coarse shard)
sub_shard_threshold   (NEW: max vectors per sub-shard before split, 0=off)
```

Old readers stop after line 8 (they ignore unknown trailing lines).

### Per-shard `shard.manifest` (NEW)

```
ready
<sub_shard_count>
<sub_shard_0_centroid_offset>   (index into subcentroids.bin, in FP16 elements)
<sub_shard_1_centroid_offset>
...
```

This is read at index open time to build the sub-shard routing table.

## In-memory structures

### ScanSubShard (new)

```cpp
struct ScanSubShard {
    std::unique_ptr<CodeStream> codes;   // same as ScanShard::codes
    std::vector<RowId> row_ids;          // same as ScanShard::row_ids
    uint32_t count = 0;
};
```

### ScanShard (extended)

```cpp
struct ScanShard {
    // Existing fields (when sub_shards.size() == 1, same as today):
    std::unique_ptr<CodeStream> codes;
    std::vector<RowId> row_ids;
    std::vector<float> factors;
    uint32_t count = 0;

    // NEW: sub-shard support
    std::vector<ScanSubShard> sub_shards;     // 1+ sub-shards
    std::vector<uint32_t> sub_centroid_offsets; // into IVFScanIndex::sub_centroids
    bool has_sub_shards = false;               // false = use flat codes/row_ids
};
```

### IVFScanIndex (extended)

```cpp
struct IVFScanIndex {
    // ... existing fields ...
    std::vector<float16_t> sub_centroids;  // NEW: Σ sub_shards × dim FP16
    uint32_t sub_shard_n_probe = 1;        // NEW: sub-shards to probe per shard
    uint32_t sub_shard_threshold = 0;      // NEW: split threshold, 0=off
};
```

## Search path changes

In `IVFScanSearcher::search_body_`, after coarse routing selects np shards:

```cpp
for (uint32_t p = 0; p < n_probe_eff; p++) {
    const uint32_t c = w.cent_dists[p].second;
    auto& shard = index_.shards[c];
    if (!shard || !shard->codes) continue;

    if (shard->has_sub_shards && index_.sub_shard_n_probe > 0) {
        // --- Two-level routing: pick top sub-shards ---
        const uint16_t* q_fp16 = w.query_fp16.data();
        std::vector<std::pair<float, uint32_t>> sub_dists;
        for (uint32_t s = 0; s < shard->sub_shards.size(); s++) {
            const float16_t* sub_cent =
                index_.sub_centroids.data() +
                shard->sub_centroid_offsets[s] * index_.dim;
            const float d = simd::dist_f16(metric, q_fp16, sub_cent, index_.dim);
            sub_dists.push_back({d, s});
        }
        std::partial_sort(sub_dists.begin(),
                         sub_dists.begin() + sub_shard_n_probe,
                         sub_dists.end());
        // Scan only the top sub_shard_n_probe sub-shards
        for (uint32_t s = 0; s < sub_shard_n_probe; s++) {
            auto& ss = shard->sub_shards[sub_dists[s].second];
            scan_one_sub_shard(ss, ...);
        }
    } else {
        // --- Flat scan (existing path, unchanged) ---
        scan_one_shard(*shard, ...);
    }
}
```

The `scan_one_sub_shard` function is a factored version of the existing
on_block scan loop — same FastScan kernel, same heap, just operating on
a ScanSubShard instead of a ScanShard.

## Build path changes

In `build_ivf_scan`, after partitioning:

```cpp
for each shard k with members:
    if members.size() > sub_shard_threshold && sub_shard_threshold > 0:
        // --- Local k-means(S) on the shard's members ---
        // S = ceil(members.size() / target_sub_size)
        // Uses batch4-optimized partition_codes on just this shard's codes
        auto sub_assignment = partition_codes(
            *index_.quantizer, shard_codes, shard_n, code_size,
            S, /*closure=*/1.0f, /*iters=*/5, num_threads);
        // Write each sub-shard as sub_MMMM/ directory
        for each sub_shard s:
            write sub_MMMM/.codes4, .rowids, .manifest
            decode sub-centroid to FP16, append to subcentroids.bin
    else:
        // Single sub-shard (or flat shard)
        write sub_0000/ as today
```

The local k-means uses the existing `partition_codes` (with batch4 SIMD)
on just the shard's members. At S=4 and ~10k members, this is <1 second.

## Split/merge for live updates (future phase)

### Split (insert-triggered)

When a sub-shard exceeds T_max:
1. Read the sub-shard's codes + row_ids into memory
2. Run `partition_codes` with K=2 on just this sub-shard's data
3. Write two new `sub_MMMM/` directories
4. Update `shard.manifest` and `subcentroids.bin` atomically
5. Delete the old sub-shard directory

Cost: O(sub_shard_size × m) — <1 second for 20k vectors.

### Merge (delete-triggered)

When a sub-shard drops below T_min:
1. Read the sub-shard + nearest sibling's data
2. Concatenate (no re-partitioning needed)
3. Write merged `sub_MMMM/` directory
4. Update manifest + subcentroids atomically
5. Delete old directories

## Implementation phases

### Phase 1: Build-time sub-shards (no live updates)
- Extend `ScanShard`, `IVFScanIndex` with sub-shard fields
- Extend `build_ivf_scan` to split large shards at build time
- Extend `search_body_` with two-level routing
- Extend manifest format
- Test: build K=1024 with sub_shard_threshold=5000, measure recall/QPS

### Phase 2: Live split/merge
- Implement split/merge API
- Wire to insert/delete paths
- Test: insert batches, verify splits happen and recall is maintained

### Phase 3: Adaptive sub-shard count
- Auto-tune S (number of sub-shards) based on shard size
- Auto-tune sub_shard_n_probe based on recall target
