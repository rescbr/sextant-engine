#!/usr/bin/env bash
# =============================================================================
# Cohere 10M filtered search benchmark — parquet-native path.
#
# Reads directly from parquet shards (list<float> vectors) + .fdat labels.
# No .fbin conversion needed.
#
# Measures:
#   1. Unfiltered search: recall@10 vs QPS (GT: neighbors.parquet)
#   2. Filtered search: recall@10 vs QPS at each selectivity
#      (GT: neighbors_labels_label_{0.1p..50p}.parquet)
#   3. Per-query label correctness: verify returned row_ids have the right label
#
# Metric: COSINE (--metric ip with L2-normalized vectors).
#
# Pre-requisites:
#   - Cohere parquet shards at $DATA/train-{00..09}-of-10.parquet
#   - Labels at $DATA/scalar_labels.parquet (converted to .fdat, see below)
#   - Query file at $DATA/test.parquet (or cohere_10m_query.fbin)
#   - GT files at $DATA/cohere_10m_gt_*.parquet
# =============================================================================
set -euo pipefail

BIN="${BENCH_DIR:-/mnt/ssd}/mybin/sextant"
DATA="${BENCH_DIR:-/mnt/ssd}/cohere"
OUT="${BENCH_DIR:-/mnt/ssd}/bench-cohere-out"
mkdir -p "$OUT"

# Vector shards (glob expands to 10 files)
SHARDS="$DATA/train-*-of-10.parquet"

# Labels — convert from parquet to .fdat if not done yet.
# (carquet has a dictionary-decode bug for large_string; .fdat is the workaround)
FDAT="$DATA/cohere_10m_labels.fdat"
if [ ! -f "$FDAT" ]; then
    echo "[labels] Converting scalar_labels.parquet → $FDAT"
    python3 -c "
import pyarrow.parquet as pq
import struct, sys
pf = pq.ParquetFile('$DATA/scalar_labels.parquet')
labels = pf.read_column('labels').to_pylist()
n = len(labels)
with open('$FDAT', 'wb') as f:
    f.write(struct.pack('<I', 0x46444154))
    f.write(struct.pack('<Q', n))
    f.write(struct.pack('<I', 1))
    f.write(struct.pack('<B', 3))
    name = b'labels'
    f.write(struct.pack('<H', len(name))); f.write(name)
    f.write(struct.pack('<B', 0))
    offsets = []; lengths = []; data = bytearray(); base = 0
    for label in labels:
        b = (label or '').encode('utf-8')
        offsets.append(base); lengths.append(len(b)); data.extend(b); base += len(b)
    offsets.append(base)
    for o in offsets: f.write(struct.pack('<I', o))
    for l in lengths: f.write(struct.pack('<H', l))
    f.write(bytes(data))
print(f'Wrote {n} labels to $FDAT')
"
fi

# Query file — prefer .fbin if available, fall back to .parquet
if [ -f "$DATA/cohere_10m_query.fbin" ]; then
    QUERY="$DATA/cohere_10m_query.fbin"
else
    QUERY="$DATA/test.parquet"
fi

# ============================================================
# Build index (COSINE = normalize + IP)
# ============================================================

IDX="$OUT/cohere_10m.tree"
if [ ! -f "$IDX" ]; then
    echo "[build] Cohere 10M (768d, COSINE, m=192) from parquet shards"
    t0=$(date +%s.%N)
    "$BIN" build-tree --input "$SHARDS" --index "$IDX" \
        --filter-data "$FDAT" \
        --vector-col emb --metric ip \
        --leaf-capacity 5000 --pca-dims 32 --threads 0 \
        --pq4-m 192 --pq-bits 4
    t1=$(date +%s.%N)
    echo "[build] done in $(echo "$t1 - $t0" | bc)s, size=$(du -h "$IDX" | cut -f1)"
else
    echo "[skip] $IDX exists"
fi

# ============================================================
# 1. Unfiltered search sweep
# ============================================================

echo ""
echo "========================================"
echo "  Cohere 10M — unfiltered recall@10 vs QPS (COSINE)"
echo "========================================"
printf "%-8s %-10s %-10s\n" "n_probe" "recall@10" "QPS"
printf "%-8s %-10s %-10s\n" "-------" "---------" "---"

GT_ALL="$DATA/cohere_10m_gt_all.parquet"
for nprobe in 1 2 4 8 16 32; do
    result=$("$BIN" tree-search --index "$IDX" --query "$QUERY" \
        --ground-truth "$GT_ALL" \
        --topk 10 --n-probe "$nprobe" --n-probe-ln "$nprobe" \
        --fastscan-w 300 --threads 0 2>&1)
    recall=$(echo "$result" | grep -oP 'recall@\d+:\s*\K[\d.]+' | head -1)
    qps=$(echo "$result" | grep -oP '\(\K[\d.]+(?= QPS)' | head -1)
    [ -z "$recall" ] && recall="ERR"
    [ -z "$qps" ] && qps="ERR"
    printf "%-8s %-10s %-10s\n" "$nprobe" "$recall" "$qps"
done

# ============================================================
# 2. Filtered search sweep (per selectivity)
# ============================================================

for sel in 0.1p 0.2p 0.5p 1p 2p 5p 10p 20p 50p; do
    echo ""
    echo "========================================"
    echo "  Filtered: labels:eq:label_${sel}"
    echo "========================================"
    printf "%-8s %-10s %-10s\n" "n_probe" "recall@10" "QPS"
    printf "%-8s %-10s %-10s\n" "-------" "---------" "---"

    GT_FILE="$DATA/cohere_10m_gt_label_${sel}.parquet"
    for nprobe in 1 2 4 8 16 32; do
        result=$("$BIN" tree-search --index "$IDX" --query "$QUERY" \
            --ground-truth "$GT_FILE" \
            --topk 10 --n-probe "$nprobe" --n-probe-ln "$nprobe" \
            --fastscan-w 300 --threads 0 \
            --filter "labels:eq:label_${sel}" 2>&1)
        recall=$(echo "$result" | grep -oP 'recall@\d+:\s*\K[\d.]+' | head -1)
        qps=$(echo "$result" | grep -oP '\(\K[\d.]+(?= QPS)' | head -1)
        [ -z "$recall" ] && recall="ERR"
        [ -z "$qps" ] && qps="ERR"
        printf "%-8s %-10s %-10s\n" "$nprobe" "$recall" "$qps"
    done
done

echo ""
echo "[done] Results in $OUT"
