#!/usr/bin/env bash
# =============================================================================
# Cohere 10M filtered search benchmark.
#
# Measures:
#   1. Unfiltered search: recall@10 vs QPS (GT: neighbors.parquet)
#   2. Filtered search: recall@10 vs QPS at each selectivity
#      (GT: neighbors_labels_label_{0.1p..50p}.parquet)
#   3. Per-query label correctness: verify returned row_ids have the right label
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

declare -a LABELS=("label_0.1p" "label_0.5p" "label_1p" "label_5p" "label_10p" "label_20p" "label_50p")

for label in "${LABELS[@]}"; do
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

# ============================================================
# 3. Per-query label correctness check
#    Run filtered search, output results to TSV, then verify
#    every returned row_id actually has the queried label.
# ============================================================

echo ""
echo "========================================"
echo "  Cohere 10M — per-query label correctness"
echo "========================================"

SCALARS="$DATA/../cohere_s3/scalar_labels.parquet"
if [ ! -f "$SCALARS" ]; then
    SCALARS="$BENCH_DIR/cohere_s3/scalar_labels.parquet"
fi

for label in label_1p label_5p label_20p; do
    echo "  --- $label (n_probe=16) ---"
    TSV="$OUT/cohere_${label}_results.tsv"
    "$BIN" tree-search --index "$IDX" --query "$QUERY" \
        --topk 10 --n-probe 16 --n-probe-ln 16 \
        --fastscan-w 300 \
        --filter "labels:eq:${label}" \
        --output "$TSV" \
        --threads 0 2>&1 | grep -E "recall|QPS"

    # Verify: for each returned row_id, check the label matches.
    if [ -f "$SCALARS" ] && [ -f "$TSV" ]; then
        echo "  Verifying label correctness via duckdb..."
        python3 << PYEOF
import duckdb
con = duckdb.connect()
# Load the TSV: qi, row_id, dist
con.execute(f"""
    CREATE TABLE results AS
    SELECT column0 as qi, column1 as row_id
    FROM read_csv_auto('{TSV}', delim='\t', header=false)
""")
# Join with scalar labels to check correctness
correctness = con.execute(f"""
    WITH r AS (SELECT DISTINCT row_id FROM results)
    SELECT
        count(*) as total_results,
        sum(CASE WHEN l.labels = '{label}' THEN 1 ELSE 0 END) as correct,
        sum(CASE WHEN l.labels != '{label}' THEN 1 ELSE 0 END) as wrong
    FROM r
    LEFT JOIN read_parquet('{SCALARS}') l ON r.row_id = l.id
""").fetchone()
    total, ok, bad = correctness
    pct = 100.0 * ok / total if total > 0 else 0
    print(f"  label={label}: {ok}/{total} correct ({pct:.1f}%), {bad} wrong")
    if bad > 0:
        # Show a few wrong ones
        wrong = con.execute(f"""
            WITH r AS (SELECT DISTINCT row_id FROM results)
            SELECT r.row_id, l.labels
            FROM r
            JOIN read_parquet('{SCALARS}') l ON r.row_id = l.id
            WHERE l.labels != '{label}'
            LIMIT 5
        """).fetchall()
        for rid, lbl in wrong:
            print(f"    WRONG: row_id={rid} has label='{lbl}' (expected '{label}')")
PYEOF
    else
        echo "  [skip] missing scalars or TSV for verification"
    fi
    echo ""
done

echo ""
echo "============================================"
echo " Cohere 10M benchmark complete."
echo " $(date)"
echo "============================================"
