#!/usr/bin/env python3
"""Convert deep1b GT (.ivecs, top-100 ids) to Sextant GTMM.

GTMM: [magic "GTMM":4B][n:u32][k:u32][metric:u8][ids k*u32][dists k*f32]
per query, interleaved. Dists are not present in ivecs -> zeros (the CLI
recall check only uses ids). Subsamples the query count to the first n.
"""
import struct, sys

def read_ivecs(path):
    with open(path, "rb") as f:
        data = f.read()
    off, out = 0, []
    while off + 4 <= len(data):
        (d,) = struct.unpack_from("<i", data, off)
        off += 4
        vals = struct.unpack_from(f"<{d}i", data, off)
        off += d * 4
        off += (-off) % 4  # 4-byte record padding
        out.append(vals)
    return out

def main():
    src, dst, nq = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rows = read_ivecs(src)
    k = min(len(r) for r in rows[:nq])
    rows = rows[:nq]
    with open(dst, "wb") as f:
        f.write(b"GTMM")
        f.write(struct.pack("<IIB", len(rows), k, 0))
        for r in rows:
            f.write(struct.pack(f"<{k}I", *r[:k]))
            f.write(struct.pack(f"<{k}f", *([0.0] * k)))
    print(f"wrote {dst}: n={len(rows)} k={k}")

if __name__ == "__main__":
    main()
