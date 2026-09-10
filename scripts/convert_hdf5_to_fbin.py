#!/usr/bin/env python3
"""Convert an ann-benchmarks-style HDF5 to sextant .fbin/.gt format.

Streaming over the train set (chunked reads) to avoid OOM on multi-GB files.

The .gt output carries the metric the GT was computed under (ann-benchmarks
HDF5 convention: the top-level `distance` attribute). The sextant benchmark
verifies the .gt metric matches the build's --metric and hard-errors on
mismatch — this prevents silent wrong-recall artifacts like the Sphere
IP-vs-L2sq ceiling (where Weaviate shipped Sphere with IP-computed GT and
we evaluated under L2sq, capping recall at 0.57).

If the HDF5 lacks a `distance` attribute (some sources omit it, e.g.
Sphere), --metric is REQUIRED. The converter refuses to guess — a wrong
guess produces silently-bad recall and is exactly the bug this format
extension exists to prevent.
"""

import sys
import struct
import argparse
import numpy as np
import h5py

# Must match tools/benchmark.cpp kGtMagic.
GT_MAGIC = 0x4D4D5447  # "GTMM" little-endian

# Must match include/sextant/types.hpp MetricKind.
METRIC_BYTE = {"l2sq": 0, "euclidean": 0, "ip": 1, "inner_product": 1,
               "dot": 1, "cosine": 2, "angular": 2}

# ann-benchmarks HDF5 `distance` attribute spellings → our byte.
HDF5_DISTANCE_TO_BYTE = {
    "euclidean": 0,
    "square_euclidean": 0,
    "l2": 0,
    "inner_product": 1,
    "ip": 1,
    "dot": 1,
    "cosine": 2,
    "angular": 2,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input HDF5 (ann-benchmarks format)")
    ap.add_argument("output_prefix", help="Output prefix; writes "
                     "<prefix>_base.fbin, _query.fbin, _gt.gtmm")
    ap.add_argument("--metric", default=None,
                    help="Metric the GT was computed under: l2sq, ip, or "
                    "cosine. If omitted, the HDF5 `distance` attribute is "
                    "used; if that's missing too, the converter errors "
                    "(refuses to guess — wrong guesses cause silent "
                    "recall artifacts).")
    args = ap.parse_args()

    # Resolve metric: flag > HDF5 attr > error.
    if args.metric is not None:
        m = args.metric.lower()
        if m not in METRIC_BYTE:
            ap.error(f"--metric must be one of {sorted(METRIC_BYTE)} (got {args.metric!r})")
        metric_byte = METRIC_BYTE[m]
        metric_source = "--metric"
    else:
        metric_byte = None
        metric_source = None

    chunk = 100000  # train vectors per read

    with h5py.File(args.input, 'r') as f:
        n, dim = f['train'].shape
        q, dim_q = f['test'].shape
        gt_n, gt_k = f['neighbors'].shape

        # If we still don't have a metric, try the HDF5 attribute.
        if metric_byte is None:
            dist_attr = f.attrs.get("distance") if f.attrs else None
            if dist_attr is None:
                ap.error(
                    "No metric specified and HDF5 has no `distance` attribute. "
                    "Pass --metric {l2sq,ip,cosine} explicitly. The sextant "
                    "benchmark verifies the .gt metric matches the build's "
                    "--metric; a wrong guess here produces silently-bad recall "
                    "(e.g. Sphere ships with IP GT but looks like L2sq data).")
            dist_str = dist_attr.decode() if isinstance(dist_attr, bytes) else str(dist_attr)
            dist_str = dist_str.lower()
            if dist_str not in HDF5_DISTANCE_TO_BYTE:
                ap.error(f"HDF5 `distance` attribute {dist_attr!r} not recognized; "
                         f"pass --metric explicitly")
            metric_byte = HDF5_DISTANCE_TO_BYTE[dist_str]
            metric_source = f"HDF5 attr 'distance'={dist_str}"

        metric_name = {0: "l2sq", 1: "ip", 2: "cosine"}[metric_byte]
        print(f"train:    {n} x {dim}")
        print(f"test:     {q} x {dim_q}")
        print(f"neighbors:{gt_n} x {gt_k}")
        print(f"metric:   {metric_name} (from {metric_source})")

        # Write base (streaming — never materialize the full train set).
        base_path = f"{args.output_prefix}_base.fbin"
        with open(base_path, 'wb') as fout:
            fout.write(struct.pack('<II', n, dim))
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                block = f['train'][start:end]
                fout.write(block.astype(np.float32).tobytes())
        print(f"wrote {base_path}")

        # Write query (small, load all).
        query_path = f"{args.output_prefix}_query.fbin"
        test = f['test'][:]
        with open(query_path, 'wb') as fout:
            fout.write(struct.pack('<II', q, dim_q))
            fout.write(test.astype(np.float32).tobytes())
        print(f"wrote {query_path}")

        # Canonical GTMM (include/sextant/ground_truth.hpp): per-query
        # interleaved [ids k×u32][dists k×f32].
        gt_path = f"{args.output_prefix}_gt.gtmm"
        neighbors = f['neighbors'][:].astype(np.uint32)
        distances = f['distances'][:].astype(np.float32) if 'distances' in f else \
                    np.zeros_like(neighbors, dtype=np.float32)
        with open(gt_path, 'wb') as fout:
            fout.write(struct.pack('<I', GT_MAGIC))
            fout.write(struct.pack('<II', q, gt_k))
            fout.write(struct.pack('<B', metric_byte))
            for row in range(q):
                fout.write(neighbors[row].tobytes())
                fout.write(distances[row].tobytes())
        print(f"wrote {gt_path} (GTMM interleaved, metric={metric_name})")


if __name__ == '__main__':
    main()
