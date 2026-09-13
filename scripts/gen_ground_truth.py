#!/usr/bin/env python3
"""Generate query + ground-truth files for a .fbin dataset.

Streams the base file in chunks, computing brute-force top-k neighbors for a
random sample of query vectors. Vectorized with numpy.

Outputs:
  <prefix>_query.fbin — the query vectors
  <prefix>_gt.gtmm    — canonical GTMM ground truth (see include/sextant/ground_truth.hpp)

Usage:
    python3 scripts/gen_ground_truth.py --base data.fbin --prefix out --n-queries 10000 --k 10
"""
import argparse
import struct
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--n-queries", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=50000)
    args = ap.parse_args()

    with open(args.base, 'rb') as f:
        n, dim = struct.unpack('<II', f.read(8))
    print(f"Base: {n} vectors, dim={dim}")

    nq = min(args.n_queries, n)
    k = min(args.k, n - 1)
    rng = np.random.default_rng(args.seed)

    # Pick query indices.
    query_idx = np.sort(rng.choice(n, size=nq, replace=False))
    query_set = set(query_idx.tolist())

    # Pass 1: read query vectors and all norms (streaming).
    print("Pass 1: collecting query vectors + norms...")
    queries = np.empty((nq, dim), dtype=np.float32)
    norms = np.empty(n, dtype=np.float32)
    qi_map = {idx: i for i, idx in enumerate(query_idx)}
    vec_bytes = dim * 4
    with open(args.base, 'rb') as f:
        f.read(8)
        for i in range(n):
            vec = np.frombuffer(f.read(vec_bytes), dtype=np.float32)
            norms[i] = np.dot(vec, vec)
            if i in qi_map:
                queries[qi_map[i]] = vec
    query_norms = norms[query_idx]
    print(f"  {nq} queries loaded.")

    # Pass 2: streaming brute-force KNN with chunked dot-product matrix.
    # For each chunk of base vectors, compute dists to all queries (chunk × nq),
    # then merge into per-query top-k arrays.
    print(f"Pass 2: brute-force KNN ({nq} queries × {n} base, k={k})...")
    # Store top-k as (nq, k) arrays kept SORTED ASCENDING by distance
    # after every merge. The merge gate is the LAST column (the worst
    # retained distance). BUG FIXED 2026-09-13: this used to gate on
    # [:,0], which after the ascending sort is the BEST distance — the
    # top-k then froze at the first chunk's values (bias toward early
    # rows; catastrophic on clustered-order bases, silently wrong on
    # shuffled ones). Detected via engine-vs-exact decomposition: engine
    # exh-raw hit 9/10 of exact truth while the GT agreed with exact on
    # only 2-5/10.
    top_dists = np.full((nq, k), np.inf, dtype=np.float32)
    top_ids = np.full((nq, k), -1, dtype=np.int32)

    with open(args.base, 'rb') as f:
        f.read(8)
        base_idx = 0
        while base_idx < n:
            chunk_n = min(args.chunk_size, n - base_idx)
            # Read chunk.
            chunk = np.frombuffer(
                f.read(chunk_n * vec_bytes), dtype=np.float32
            ).reshape(chunk_n, dim)
            # Dot products: (chunk_n, nq) = chunk @ queries.T
            dots = chunk @ queries.T
            chunk_norms = norms[base_idx:base_idx + chunk_n]
            # L2sq(q, x) = norm(x) + norm(q) - 2*dot(q,x)
            dists = chunk_norms[:, None] + query_norms[None, :] - 2.0 * dots  # (chunk_n, nq)
            # Transpose to (nq, chunk_n) for per-query processing.
            dists = dists.T  # (nq, chunk_n)

            # For each query, find candidates better than current worst top-k.
            current_worst = top_dists[:, -1]  # (nq,) max retained dist (sorted asc)
            # Mask: which chunk entries beat the current worst for each query.
            better = dists < current_worst[:, None]  # (nq, chunk_n)

            for q in range(nq):
                candidates = np.where(better[q])[0]
                if len(candidates) == 0:
                    continue
                cand_dists = dists[q, candidates]
                cand_ids = (candidates + base_idx).astype(np.int32)
                # Merge with existing top-k.
                all_d = np.concatenate([top_dists[q], cand_dists])
                all_i = np.concatenate([top_ids[q], cand_ids])
                # Exclude self.
                mask = all_i != query_idx[q]
                all_d = all_d[mask]
                all_i = all_i[mask]
                if len(all_d) <= k:
                    order = np.argsort(all_d)
                else:
                    order = np.argpartition(all_d, k)[:k]
                    order = order[np.argsort(all_d[order])]
                top_dists[q] = all_d[order]
                top_ids[q] = all_i[order]

            base_idx += chunk_n
            print(f"  {base_idx}/{n} ({base_idx/n*100:.1f}%)", flush=True)

    print("Writing output...")
    query_path = f"{args.prefix}_query.fbin"
    with open(query_path, 'wb') as f:
        f.write(struct.pack('<II', nq, dim))
        f.write(queries.tobytes())

    # Canonical GTMM (see include/sextant/ground_truth.hpp): magic,
    # n, k, metric byte, then PER-QUERY interleaved [ids k×u32][dists k×f32]
    # sorted ascending. (The legacy headerless bulk layout is removed.)
    gt_path = f"{args.prefix}_gt.gtmm"
    with open(gt_path, 'wb') as f:
        f.write(struct.pack('<III', 0x4D4D5447, nq, k))
        f.write(struct.pack('<B', 0))  # metric: l2sq
        for q in range(nq):
            order = np.argsort(top_dists[q])
            f.write(top_ids[q][order].astype('<u4').tobytes())
            f.write(top_dists[q][order].astype('<f4').tobytes())

    print(f"Done: {query_path} ({os.path.getsize(query_path)/1e6:.1f} MB), "
          f"{gt_path} ({os.path.getsize(gt_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
