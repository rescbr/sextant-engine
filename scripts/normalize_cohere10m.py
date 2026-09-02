#!/usr/bin/env python3
"""Normalize cohere 10M shards (glob order, matching nothing — the fbin
row order IS the row_id space) into one .fbin, and normalize the queries.
L2sq on normalized vectors == Zilliz cosine GT ranking."""
import glob, struct
import numpy as np
import pyarrow.parquet as pq

OUT = '/mnt/ssd/data/cohere_10m_norm.fbin'
QOUT = '/mnt/ssd/data/cohere_10m_query_norm.fbin'
N, D = 10_000_000, 768

shards = sorted(glob.glob('/mnt/ssd/data/train-*-of-10.parquet'))
assert len(shards) == 10, shards

with open(OUT, 'wb') as f:
    f.write(struct.pack('<II', N, D))
    for s in shards:
        pf = pq.ParquetFile(s)
        for batch in pf.iter_batches(batch_size=65536, columns=['emb']):
            # list<float> -> (n, 768) without a python round-trip
            col = batch.column(0)
            arr = np.asarray(col.flatten().to_pylist(), dtype=np.float32) \
                .reshape(len(col), -1)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr /= np.maximum(norms, 1e-12)
            f.write(arr.astype('<f4').tobytes())
        print(f'[norm] {s} done', flush=True)

t = pq.read_table('/mnt/ssd/data/test.parquet')
q = np.asarray(t.column('emb').to_pylist(), dtype=np.float32)
q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
with open(QOUT, 'wb') as f:
    f.write(struct.pack('<II', len(q), q.shape[1]))
    f.write(q.astype('<f4').tobytes())
print('[norm] queries normalized:', q.shape, flush=True)
