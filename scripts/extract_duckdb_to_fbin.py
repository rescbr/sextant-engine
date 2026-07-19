#!/usr/bin/env python3
"""Stream a DuckDB vector table to .fbin without loading it all into RAM.

Uses LIMIT/OFFSET pagination at the SQL level so DuckDB only ever materializes
one page in memory. Each page is written immediately and discarded.

Usage:
    python3 scripts/extract_duckdb_to_fbin.py \
        --input  /path/to/se_base.duckdb \
        --table  base \
        --output /path/to/se_base.fbin
"""
import argparse
import struct
import sys
import os
import array

try:
    import duckdb
except ImportError:
    sys.stderr.write("pip install duckdb\n")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--table", default="base")
    ap.add_argument("--output", required=True)
    ap.add_argument("--vec-col", default="vec")
    ap.add_argument("--where", default="")
    ap.add_argument("--chunk-size", type=int, default=10000)
    args = ap.parse_args()

    con = duckdb.connect(args.input, read_only=True)
    where = f"WHERE {args.where}" if args.where else ""

    n = con.execute(f"SELECT COUNT(*) FROM {args.table} {where}").fetchone()[0]
    dim = len(con.execute(
        f"SELECT {args.vec_col} FROM {args.table} {where} LIMIT 1").fetchone()[0])
    print(f"Streaming {n} vectors (dim={dim}) → {args.output}")

    with open(args.output, 'wb') as f:
        f.write(struct.pack('<II', n, dim))
        written = 0
        offset = 0
        while offset < n:
            rows = con.execute(
                f"SELECT {args.vec_col} FROM {args.table} {where} "
                f"LIMIT {args.chunk_size} OFFSET {offset}"
            ).fetchall()  # one page only — small, freed each iteration
            if not rows:
                break
            buf = array.array('f')
            for row in rows:
                buf.extend(row[0])
            f.write(buf.tobytes())
            offset += len(rows)
            written += len(rows)
            del buf, rows  # explicit free before next query
            print(f"  {written}/{n} ({written/n*100:.1f}%)", flush=True)

    print(f"Done: {written} vectors, {os.path.getsize(args.output)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
