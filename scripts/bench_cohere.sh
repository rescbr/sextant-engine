#!/usr/bin/env bash
# =============================================================================
# Cohere 10M filtered search benchmark.
#
# Measures recall@10 vs QPS for:
#   1. Unfiltered search (GT: neighbors.parquet, 10K neighbors/query)
#   2. Filtered search at each selectivity level
#      (GT: neighbors_labels_label_{0.1p..50p}.parquet, variable neighbors)
#
# Pre-requisites: run convert_cohere.py first to produce the data files.
# =============================================================================
set -euo pipefail

BIN="${BENCH_DIR:-/mnt/ssd}/mybin/sextant"
DATA="${BENCH_DIR:-/mnt/ssd}/cohere"
OUT="${BENCH_DIR:-/mnt/ssd}/bench-cohere-out"
mkdir -p "$OUT"

QUERY="$DATA/cohere_10m_query.fbin"
PQT="$DATA/cohere_10m.parquet"

# ============================================================
# Build index
# ============================================================

IDX="$OUT/cohere_10m.tree"
if [ ! -f "$IDX" ]; then
    echo "[build] Cohere 10M (768d, L2, m=192)"
    t0=$(date +%s.%N)
    "$BIN" build-tree --input "$PQT" --index "$IDX" \
        --leaf-capacity 5000 --pca-dims 32 --threads 0 \
        --pq4-m 192 --pq-bits 4 --metric l2sq
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
echo "  Cohere 10M — unfiltered recall@10 vs QPS"
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
# 2. Filtered search sweep (string label equality)
# ============================================================

echo ""
echo "========================================"
echo "  Cohere 10M — filtered recall@10 vs QPS"
echo "  (filter: labels:eq:label_Xp)"
echo "========================================"
printf "%-12s %-8s %-10s %-10s\n" "selectivity" "n_probe" "recall@10" "QPS"
printf "%-12s %-8s %-10s %-10s\n" "-----------" "-------" "---------" "---"

# Each selectivity: label name + GT file
# label_50p = 50% of vectors match, etc.
declare -a LABELS=("label_0.1p" "label_0.5p" "label_1p" "label_5p" "label_10p" "label_20p" "label_50p")

for label in "${LABELS[@]}"; do
    # Map label name to GT file
    gt_file="$DATA/cohere_10m_gt_${label}.parquet"
    if [ ! -f "$gt_file" ]; then
        echo "  [skip] no GT for $label"
        continue
    fi

    for nprobe in 8 16 32; do
        result=$("$BIN" tree-search --index "$IDX" --query "$QUERY" \
            --ground-truth "$gt_file" \
            --topk 10 --n-probe "$nprobe" --n-probe-ln "$nprobe" \
            --fastscan-w 300 \
            --filter "labels:eq:${label}" \
            --threads 0 2>&1)
        recall=$(echo "$result" | grep -oP 'recall@\d+:\s*\K[\d.]+' | head -1)
        qps=$(echo "$result" | grep -oP '\(\K[\d.]+(?= QPS)' | head -1)
        [ -z "$recall" ] && recall="ERR"
        [ -z "$qps" ] && qps="ERR"
        printf "%-12s %-8s %-10s %-10s\n" "$label" "$nprobe" "$recall" "$qps"
    done
done

echo ""
echo "============================================"
echo " Cohere 10M benchmark complete."
echo " $(date)"
echo "============================================"
