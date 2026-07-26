#!/usr/bin/env python3
"""Convert MSMARCO Snowflake HDF5 to sextant .fbin/.gt format (streaming).

Reads the HDF5 in chunks to avoid OOM on the 8.8M × 768 = 26GB train set.
"""

import sys
import struct
import numpy as np
import h5py

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.hdf5 output_prefix")
        sys.exit(1)

    hdf5_path = sys.argv[1]
    prefix = sys.argv[2]
    chunk = 100000  # vectors per read

    with h5py.File(hdf5_path, 'r') as f:
        n, dim = f['train'].shape
        q, dim_q = f['test'].shape
        gt_n, gt_k = f['neighbors'].shape

        print(f"train: {n} x {dim}")
        print(f"test:  {q} x {dim_q}")
        print(f"neighbors: {gt_n} x {gt_k}")

        # Write base (streaming)
        base_path = f"{prefix}_base.fbin"
        with open(base_path, 'wb') as fout:
            fout.write(struct.pack('<II', n, dim))
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                block = f['train'][start:end]
                fout.write(block.astype(np.float32).tobytes())
        print(f"wrote {base_path}")

        # Write query (small, load all)
        query_path = f"{prefix}_query.fbin"
        test = f['test'][:]
        with open(query_path, 'wb') as fout:
            fout.write(struct.pack('<II', q, dim_q))
            fout.write(test.astype(np.float32).tobytes())
        print(f"wrote {query_path}")

        # Write GT
        gt_path = f"{prefix}_gt.gt"
        neighbors = f['neighbors'][:]
        with open(gt_path, 'wb') as fout:
            fout.write(struct.pack('<II', q, gt_k))
            fout.write(neighbors.astype(np.uint32).tobytes())
        print(f"wrote {gt_path}")

if __name__ == '__main__':
    main()
