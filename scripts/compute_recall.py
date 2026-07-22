"""Compute recall@k from a sextant search output TSV vs ground-truth .gt file.

Usage: python3 compute_recall.py <search_output.tsv> <ground_truth.gt> [k]

search_output.tsv: lines of "qi\\trow_id\\tdist" (qi is the query index).
ground_truth.gt: fbin format, n×k int32 (first 8 bytes = n, d).
"""
import struct
import sys
from collections import defaultdict


def read_gt(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        import array
        a = array.array("i")
        a.fromfile(f, n * d)
    return a, n, d


def read_search(path):
    # qi -> list of row_ids (in ranked order)
    results = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            qi = int(parts[0])
            rid = int(parts[1])
            results[qi].append(rid)
    return results


def main():
    search_path = sys.argv[1]
    gt_path = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    gt, gt_n, gt_d = read_gt(gt_path)
    results = read_search(search_path)

    total = 0
    hits = 0
    n_queries = 0
    for qi, preds in results.items():
        if qi >= gt_n:
            continue
        n_queries += 1
        truth = set(gt[qi * gt_d : qi * gt_d + min(k, gt_d)])
        pred_topk = preds[:k]
        pred_set = set(pred_topk)
        inter = len(truth & pred_set)
        hits += inter
        total += len(truth)
        # Per-query recall (for stats)

    if total == 0:
        print("ERROR: no overlapping queries")
        sys.exit(1)
    recall = hits / total
    print(f"queries evaluated: {n_queries}")
    print(f"recall@{k}: {recall:.4f}")


if __name__ == "__main__":
    main()
