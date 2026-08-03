#!/usr/bin/env python3
"""Compute filtered recall using BACKWARDS validation.

Instead of: |results ∩ filtered_GT_top_k| / k (limited by GT depth)
We compute: |filtered_GT_top_k ∩ results| / |filtered_GT_top_k|

The difference: we check for each GT entry (that matches the filter) whether
it appears in the search results. This correctly handles cases where the
filtered GT extends beyond the GT file's depth — we just count what we CAN
verify.

For queries where fewer than k GT entries match the filter, we divide by
the number of matching GT entries (not k), giving a fair recall measure.

Usage:
  python3 compute_filtered_recall.py --results results.txt --gt gt.gt \
    --fdat filter.fdat --year-val 2050 --k 10
"""

import argparse
import struct
import sys


def read_gt(path):
    with open(path, "rb") as f:
        maybe_magic = struct.unpack("<I", f.read(4))[0]
        if maybe_magic == 0x4D4D5447:
            n, k = struct.unpack("<II", f.read(8))
            f.read(1)
        else:
            n = maybe_magic
            k = struct.unpack("<I", f.read(4))[0]
        gt = []
        for _ in range(n):
            row = struct.unpack(f"<{k}I", f.read(k * 4))
            gt.append(list(row))
    return gt, k


def read_fdat_year_values(path):
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x46444154
        n_rows = struct.unpack("<Q", f.read(8))[0]
        n_cols = struct.unpack("<I", f.read(4))[0]
        col_types = [struct.unpack("<B", f.read(1))[0] for _ in range(n_cols)]
        for _ in range(n_cols):
            name_len = struct.unpack("<H", f.read(2))[0]
            f.read(name_len)
        has_payload = struct.unpack("<B", f.read(1))[0]
        if has_payload:
            f.read(n_rows * 4 + 4)
        years = list(struct.unpack(f"<{n_rows}i", f.read(n_rows * 4)))
    return years


def read_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            qi = int(parts[0])
            rid = int(parts[1])
            results.setdefault(qi, []).append(rid)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--fdat", required=True)
    p.add_argument("--year-val", type=int, required=True)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    gt, gt_k = read_gt(args.gt)
    years = read_fdat_year_values(args.fdat)
    results = read_results(args.results)

    matching = set(i for i, y in enumerate(years) if y == args.year_val)

    total_recall = 0.0
    n_queries = 0
    n_shallow = 0  # queries where GT depth limits measurement

    for qi, res_row_ids in results.items():
        if qi >= len(gt):
            continue
        # Backwards: check which filtered GT entries appear in results.
        filtered_gt = [rid for rid in gt[qi] if rid in matching]
        filtered_gt_topk = filtered_gt[:args.k]

        if not filtered_gt_topk:
            continue

        res_set = set(res_row_ids[:args.k])
        hits = sum(1 for rid in filtered_gt_topk if rid in res_set)
        recall = hits / len(filtered_gt_topk)
        total_recall += recall
        n_queries += 1

        if len(filtered_gt_topk) < args.k:
            n_shallow += 1

    if n_queries == 0:
        print("No queries with matching GT found!", file=sys.stderr)
        sys.exit(1)

    avg_recall = total_recall / n_queries
    print(f"filtered_recall@{args.k}: {avg_recall:.4f} "
          f"({n_queries} queries, {n_shallow} GT-shallow)")


if __name__ == "__main__":
    main()
