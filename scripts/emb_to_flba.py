#!/usr/bin/env python
"""Convert CulturaX emb corpus chunk_*.parquet to FLBA vector-column layout.

The source stores emb as list<float> (fp32, ZSTD). ParquetSource's fast path
wants FIXED_LEN_BYTE_ARRAY: fixed-size-list<float> casts to FLBA(dim*4).
--fp16 re-encodes the vectors to IEEE half precision and stores dim*2 bytes
as a binary FLBA (requires engine fp16 FLBA support).

Usage:
  emb_to_flba.py [--fp16] [--uncompressed] [--jobs N] SRC_GLOB OUT_DIR
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DIM = 768


def convert_one(args):
    src, dst, fp16, uncompressed = args
    table = pq.read_table(src)
    emb = table.column("emb").combine_chunks()
    n = len(table)
    arr = np.asarray(emb.flatten(), dtype=np.float32).reshape(n, DIM)
    if fp16:
        width, raw = DIM * 2, arr.astype(np.float16).tobytes()
    else:
        width, raw = DIM * 4, arr.tobytes()
    # fixed-size binary -> parquet FIXED_LEN_BYTE_ARRAY(width): a flat,
    # non-nested leaf — the ParquetSource fast path.
    vec_col = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(width), n, [None, pa.py_buffer(raw)])
    field = pa.field("emb", pa.binary(width))
    out = table.drop(["emb"]).add_column(
        table.schema.get_field_index("emb"), field, vec_col)
    pq.write_table(out, dst, compression=None if uncompressed else "zstd",
                   use_dictionary=False)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--uncompressed", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("src_glob")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    srcs = sorted(glob.glob(a.src_glob))
    jobs = [(s, os.path.join(a.out_dir, os.path.basename(s)), a.fp16, a.uncompressed)
            for s in srcs]
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, dst in enumerate(ex.map(convert_one, jobs)):
            if i % 10 == 0 or i == len(jobs) - 1:
                print(f"[{i+1}/{len(jobs)}] {dst}")


if __name__ == "__main__":
    main()
