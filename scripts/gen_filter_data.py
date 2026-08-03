#!/usr/bin/env python3
"""Generate synthetic filter column data (.fdat) for benchmarking.

Creates a .fdat sidecar file with an int32 column ("year") and/or a string
column ("category") with known distributions, matching a base .fbin file's
row count.

The int32 column uses values [2000, 2000+n_values), uniformly distributed.
This gives known selectivity for equality predicates:
  year == X → selectivity = 1/n_values

Usage:
  python3 gen_filter_data.py --base data.fbin --output data.fdat \
      --int32-col year --int32-values 100 \
      --string-col category --string-cardinality 20

The .fdat format matches the C++ read_filter_data() in filter_data_io.cpp.
"""

import argparse
import struct
import sys
from pathlib import Path


def read_fbin_count(path):
    with open(path, "rb") as f:
        n, dim = struct.unpack("<II", f.read(8))
    return n


def write_fdat(path, n_rows, schema_cols, col_writers, has_payload=False,
               payload_data=b"", payload_offsets=None):
    """Write a .fdat file.

    schema_cols: list of (name, type_byte) tuples.
    col_writers: list of functions(f) that write column data to file handle f.
    """
    with open(path, "wb") as f:
        # Header.
        f.write(struct.pack("<I", 0x46444154))  # magic "FDAT"
        f.write(struct.pack("<Q", n_rows))
        f.write(struct.pack("<I", len(schema_cols)))
        for name, t in schema_cols:
            f.write(struct.pack("<B", t))
        for name, t in schema_cols:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)

        # Payload.
        f.write(struct.pack("<B", 1 if has_payload else 0))
        if has_payload:
            assert payload_offsets is not None
            assert len(payload_offsets) == n_rows + 1
            for off in payload_offsets:
                f.write(struct.pack("<I", off))
            f.write(payload_data)

        # Column data.
        for writer in col_writers:
            writer(f)


def gen_int32_column(n_rows, n_values, col_idx):
    """Generate int32 column with uniform distribution over [2000, 2000+n_values)."""
    import random
    rng = random.Random(42 + col_idx)
    values = [2000 + rng.randint(0, n_values - 1) for _ in range(n_rows)]

    def writer(f):
        for v in values:
            f.write(struct.pack("<i", v))

    return writer, values


def gen_string_column(n_rows, cardinality, col_idx, prefix="cat"):
    """Generate string column with given cardinality."""
    import random
    rng = random.Random(100 + col_idx)
    values = [f"{prefix}_{rng.randint(0, cardinality - 1)}" for _ in range(n_rows)]

    def writer(f):
        # offsets (u32), lengths (u16), data.
        offset = 0
        offsets = []
        lengths = []
        data = b""
        for v in values:
            vb = v.encode("utf-8")
            offsets.append(offset)
            lengths.append(len(vb))
            data += vb
            offset += len(vb)
        for o in offsets:
            f.write(struct.pack("<I", o))
        for l in lengths:
            f.write(struct.pack("<H", l))
        f.write(data)

    return writer, values


# Type IDs (must match ColumnType enum in schema.hpp).
TYPE_INT32 = 0
TYPE_INT64 = 1
TYPE_FLOAT = 2
TYPE_BOOL = 3
TYPE_STRING = 4
TYPE_SET = 5


def main():
    p = argparse.ArgumentParser(description="Generate synthetic .fdat filter data")
    p.add_argument("--base", required=True, help="Base .fbin file (reads row count)")
    p.add_argument("--output", required=True, help="Output .fdat file")
    p.add_argument("--int32-col", default="", help="Int32 column name")
    p.add_argument("--int32-values", type=int, default=100,
                   help="Number of distinct int32 values (selectivity = 1/n)")
    p.add_argument("--string-col", default="", help="String column name")
    p.add_argument("--string-cardinality", type=int, default=20,
                   help="Number of distinct string values")
    args = p.parse_args()

    n_rows = read_fbin_count(args.base)
    print(f"base file: {n_rows} rows", file=sys.stderr)

    schema_cols = []
    col_writers = []

    if args.int32_col:
        writer, _ = gen_int32_column(n_rows, args.int32_values, 0)
        schema_cols.append((args.int32_col, TYPE_INT32))
        col_writers.append(writer)
        print(f"int32 column '{args.int32_col}': {args.int32_values} distinct values, "
              f"selectivity = {1.0/args.int32_values:.4f}", file=sys.stderr)

    if args.string_col:
        offset = len(col_writers)
        writer, _ = gen_string_column(n_rows, args.string_cardinality, offset)
        schema_cols.append((args.string_col, TYPE_STRING))
        col_writers.append(writer)
        print(f"string column '{args.string_col}': {args.string_cardinality} distinct values, "
              f"selectivity = {1.0/args.string_cardinality:.4f}", file=sys.stderr)

    if not schema_cols:
        print("ERROR: must specify at least one column (--int32-col or --string-col)",
              file=sys.stderr)
        sys.exit(1)

    write_fdat(args.output, n_rows, schema_cols, col_writers)
    out_size = Path(args.output).stat().st_size
    print(f"wrote {args.output}: {out_size / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
