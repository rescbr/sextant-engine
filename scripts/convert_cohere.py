#!/usr/bin/env python3
"""
Convert Cohere 10M benchmark dataset from S3 to Sextant-compatible format.

Reads:
  s3://assets.zilliz.com/benchmark/cohere_large_10m/train-{00..09}-of-10.parquet
  s3://assets.zilliz.com/benchmark/cohere_large_10m/scalar_labels.parquet
  s3://assets.zilliz.com/benchmark/cohere_large_10m/test.parquet

Writes (to output dir):
  cohere_10m.parquet   — FIXED_LEN_BYTE_ARRAY(3072) + labels:BYTE_ARRAY, uncompressed
  cohere_10m_query.fbin — 1000×768 float32 queries
  cohere_10m_*.parquet  — downloaded GT files (unfiltered + filtered)

Requires: pip install duckdb pyarrow
"""
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import struct
import os
import sys
import subprocess

S3_BASE = "s3://assets.zilliz.com/benchmark/cohere_large_10m"
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ssd/cohere"
os.makedirs(OUTPUT, exist_ok=True)

con = duckdb.connect()
# Configure S3 access (anonymous — public bucket)
con.execute("SET s3_region='us-east-1';")
con.execute("SET s3_access_key_id='';")
con.execute("SET s3_secret_access_key='';")
con.execute("SET s3_url_style='path';")

N_SHARDS = 10
DIM = 768

# ============================================================
# Step 1: Convert train shards + labels → single FLBA parquet
# ============================================================

out_parquet = os.path.join(OUTPUT, "cohere_10m.parquet")
if not os.path.exists(out_parquet):
    print(f"[1/3] Converting {N_SHARDS} train shards + labels → {out_parquet}")
    print("  Building combined table in DuckDB...")

    # Read all train shards + join with labels
    # The emb column is large_list<float> → cast to fixed-size list for FLBA output
    # DuckDB can write FIXED_LIST as FIXED_LEN_BYTE_ARRAY in parquet
    train_files = [f"{S3_BASE}/train-{i:02d}-of-{N_SHARDS}.parquet"
                   for i in range(N_SHARDS)]

    # Create the combined table: id, emb (as fixed 768 list), labels
    # We process shard-by-shard to avoid loading 31GB into memory
    print("  Registering remote files...")
    con.execute(f"""
        CREATE VIEW train_all AS
        SELECT * FROM read_parquet({train_files})
    """)
    con.execute(f"""
        CREATE VIEW labels_all AS
        SELECT * FROM read_parquet('{S3_BASE}/scalar_labels.parquet')
    """)

    # Write combined parquet with FLBA embedding column
    # DuckDB's write_parquet with list<float> of fixed length → parquet FIXED_LEN_BYTE_ARRAY
    # Actually DuckDB writes list<float> as LIST, not FLBA. We need pyarrow.
    print("  Querying joined data in batches...")

    # Use pyarrow to write with explicit FLBA schema
    flba_type = pa.list_(pa.float32(), DIM)  # FixedSizeList → FLBA in parquet
    schema = pa.schema([
        ('id', pa.int64()),
        ('emb', flba_type),
        ('labels', pa.string()),
    ])

    # Query in batches
    result = con.execute(f"""
        SELECT t.id, t.emb, l.labels
        FROM train_all t
        LEFT JOIN labels_all l ON t.id = l.id
        ORDER BY t.id
    """)

    BATCH = 100_000
    writer = pq.ParquetWriter(
        out_parquet, schema,
        compression='NONE',
        use_dictionary=False,
    )

    total = 0
    while True:
        batch = result.fetch_arrow_table(BATCH)
        if batch.num_rows == 0:
            break
        # Ensure types match
        batch = batch.cast(schema)
        writer.write_table(batch)
        total += batch.num_rows
        print(f"  Written {total:,} rows")

    writer.close()
    print(f"  Done: {out_parquet} ({os.path.getsize(out_parquet) / 1e9:.1f} GB)")
else:
    print(f"[1/3] {out_parquet} already exists, skipping")

# ============================================================
# Step 2: Convert test.parquet → .fbin query file
# ============================================================

out_query = os.path.join(OUTPUT, "cohere_10m_query.fbin")
if not os.path.exists(out_query):
    print(f"\n[2/3] Converting test queries → {out_query}")
    queries = con.execute(f"""
        SELECT emb FROM read_parquet('{S3_BASE}/test.parquet') ORDER BY id
    """).fetch_arrow_table()

    n = queries.num_rows
    assert queries.column(0).type == pa.list_(pa.float32()), \
        f"Expected list<float>, got {queries.column(0).type}"
    dim = DIM

    with open(out_query, 'wb') as f:
        f.write(struct.pack('II', n, dim))
        for i in range(n):
            vals = queries.column(0)[i].values()
            f.write(struct.pack(f'{dim}f', *vals))

    print(f"  Done: {n} queries, {dim}d")
else:
    print(f"\n[2/3] {out_query} already exists, skipping")

# ============================================================
# Step 3: Download ground truth files from S3
# ============================================================

print(f"\n[3/3] Downloading ground truth files")

gt_files = [
    # Unfiltered
    ("neighbors.parquet", "cohere_10m_gt_all.parquet"),
    # String label filtered GTs
    ("neighbors_labels_label_0.1p.parquet", "cohere_10m_gt_label_0.1p.parquet"),
    ("neighbors_labels_label_0.2p.parquet", "cohere_10m_gt_label_0.2p.parquet"),
    ("neighbors_labels_label_0.5p.parquet", "cohere_10m_gt_label_0.5p.parquet"),
    ("neighbors_labels_label_1p.parquet",   "cohere_10m_gt_label_1p.parquet"),
    ("neighbors_labels_label_2p.parquet",   "cohere_10m_gt_label_2p.parquet"),
    ("neighbors_labels_label_5p.parquet",   "cohere_10m_gt_label_5p.parquet"),
    ("neighbors_labels_label_10p.parquet",  "cohere_10m_gt_label_10p.parquet"),
    ("neighbors_labels_label_20p.parquet",  "cohere_10m_gt_label_20p.parquet"),
    ("neighbors_labels_label_50p.parquet",  "cohere_10m_gt_label_50p.parquet"),
]

for s3_name, local_name in gt_files:
    local_path = os.path.join(OUTPUT, local_name)
    if os.path.exists(local_path):
        print(f"  [skip] {local_name}")
        continue
    s3_url = f"{S3_BASE}/{s3_name}"
    print(f"  Downloading {s3_name} → {local_name}")
    subprocess.run(["aws", "s3", "cp", "--profile", "rescbr",
                    s3_url, local_path], check=True)

print(f"\n✓ Conversion complete. All files in {OUTPUT}/")
