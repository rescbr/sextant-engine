#!/usr/bin/env python3
"""Synthetic GT-dispersion corpus for the IVF blind-spot study (F3).

Mimics the geocoder failure mode: queries whose true neighbors are
same-"name" vectors replicated across embedding-distant regions. The
dispersion knob s controls how far apart the replicas of a name sit:

  name base direction b_g (random unit vec, d-dim)
  replica center c_{g,r} = normalize(b_g + s * u_{g,r}),  u random unit
  members            = normalize(c_{g,r} + sigma * noise)

s=0: all replicas coincide (clustered GT, easy). Larger s: replicas drift
apart; GT@10 still contains cross-replica members while background names
sit ~orthogonal (high-d), so the GT grows scattered — recall-vs-f should
flatten toward the SP no_city shape.

Outputs (unit-norm f32, metric=ip):
  <prefix>_base.fbin    G*R*M vectors
  <prefix>_query.fbin   nq noisy members (address w/o context)
  <prefix>_gt.gtmm      brute-force topk
  <prefix>_meta.npz     name_id[row], replica_id[row] for base and queries
"""
import argparse
import struct
import numpy as np


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--names", type=int, default=800)
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--members", type=int, default=25)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--sigma-q", type=float, default=0.25)
    ap.add_argument("--dispersion", type=float, required=True,
                    help="replica ring angle in DEGREES (0=coincident)")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    G, R, M, d = a.names, a.replicas, a.members, a.dim
    n = G * R * M

    b = unit(rng.standard_normal((G, d)))
    # Replicas on a fixed-angle ring around b: orthonormal directions v_r
    # per name, centers c_r = cos(a)·b + sin(a)·v_r. All replicas are
    # equidistant from b (the ambiguous query never prefers one), and the
    # INTER-REPLICA angle grows monotonically with a — the dispersion
    # knob. (Random u directions let query noise pick a favorite replica,
    # making dispersion non-monotone in s; that construction is dead.)
    alpha = np.deg2rad(a.dispersion)
    V = rng.standard_normal((G, R, d))
    V, _ = np.linalg.qr(V.transpose(0, 2, 1))     # orthonormal cols per name
    V = V.transpose(0, 2, 1)                       # G,R,d, rows orthonormal
    centers = unit(np.cos(alpha) * b[:, None, :]
                    + np.sin(alpha) * V).astype(np.float64)
    # noise: sigma is a VECTOR-NORM fraction (unit direction), so member
    # angle ~ arctan(sigma) independent of d
    nz = unit(rng.standard_normal((G, R, M, d)))
    base = unit(centers[:, :, None, :] + a.sigma * nz).reshape(n, d).astype(np.float32)

    name_id = np.repeat(np.arange(G), R * M).astype(np.int32)
    replica_id = np.tile(np.repeat(np.arange(R), M), G).astype(np.int32)

    # queries: ambiguous "address without city" — near the NAME's base
    # direction b_g, equidistant from every replica. As s grows the GT
    # members (nearest same-name vectors) sit in many embedding-distant
    # leaves: the scattered-GT regime.
    qg = rng.integers(0, G, a.n_queries)
    q = unit(b[qg].astype(np.float64)
             + a.sigma_q * unit(rng.standard_normal((a.n_queries, d)))
             ).astype(np.float32)

    # brute-force GT (ip) — on fp16-ROUNDED vectors, matching the
    # engine's effective storage precision (rerank reads fp16). Exact-fp32
    # GT on this corpus reorders within the name pool at fp16-noise scale
    # (top-15 ip spread ~2e-3 ≈ fp16 ip error) and caps recall at ~0.7
    # regardless of f. The SP geocoder corpus did the same (fp16 golden
    # re-embed).
    base16 = base.astype(np.float16).astype(np.float32)
    q16 = q.astype(np.float16).astype(np.float32)
    ids = np.zeros((a.n_queries, a.topk), np.uint32)
    block = 20_000
    for qs in range(0, a.n_queries, block):
        qe = min(qs + block, a.n_queries)
        scores = q16[qs:qe] @ base16.T                       # nq_block × n
        ids[qs:qe] = np.argpartition(-scores, a.topk, axis=1)[:, :a.topk]
        del scores

    with open(a.prefix + "_base.fbin", "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(base.tobytes())
    with open(a.prefix + "_query.fbin", "wb") as f:
        f.write(struct.pack("<II", a.n_queries, d))
        f.write(q.tobytes())
    with open(a.prefix + "_gt.gtmm", "wb") as f:
        f.write(b"GTMM")
        f.write(struct.pack("<IIB", a.n_queries, a.topk, 1))
        for i in range(a.n_queries):
            f.write(ids[i].tobytes())
            f.write(np.zeros(a.topk, np.float32).tobytes())
    np.savez(a.prefix + "_meta.npz",
             name_id=name_id, replica_id=replica_id,
             q_name=qg.astype(np.int32),
             q_replica=np.full(a.n_queries, -1, np.int32),
             gt_ids=ids)

    cross = np.mean([np.mean((name_id[ids[i]] == qg[i])
                             & (replica_id[ids[i]] != -1))
                     for i in range(a.n_queries)])
    same = np.mean([np.mean(name_id[ids[i]] == qg[i])
                    for i in range(a.n_queries)])
    nrep = np.mean([len(set(replica_id[ids[i]][name_id[ids[i]] == qg[i]]))
                    for i in range(a.n_queries)])
    print(f"n={n} d={d} s={a.dispersion} same-name GT@10: {same:.3f}, "
          f"distinct replicas in GT: {nrep:.1f}")


if __name__ == "__main__":
    main()
