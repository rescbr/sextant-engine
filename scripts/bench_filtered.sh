#!/usr/bin/env bash
set -euo pipefail

# Filtered search benchmark: recall vs selectivity curve.
# Runs on the c4a dev VM. Uses arxiv-nomic (1.34M, 768d) by default.
#
# Usage: SEXTANT=/mnt/ssd/bin/sextant GEN=gen_filter_data.py \
#        ./bench_filtered.sh <base.fbin> <query.fbin> <gt.gt> <output_dir>

BASE="${1:?base .fbin required}"
QUERY="${2:?query .fbin required}"
GT="${3:?ground-truth .gt required}"
OUT="${4:-/tmp/bench_filtered}"
SEXTANT="${SEXTANT:-sextant}"
GEN="${GEN:-gen_filter_data.py}"

mkdir -p "$OUT"

echo "═══ Filtered Search Benchmark ═══"
echo "base: $BASE"
echo "output: $OUT"
echo

INT32_VALUES_LIST="10 50 100 200 500 1000"

# ── Build baseline tree (no filter) ──────────────────────────────────────
TREE_NOFILT="$OUT/tree_nofilter.tree"
if [[ ! -f "$TREE_NOFILT" ]]; then
    echo "Building baseline tree (no filter)..."
    "$SEXTANT" build-tree-pca --input "$BASE" --index "$TREE_NOFILT" \
        --leaf-capacity 5000 --pca-dims 32 --max-lloyd-passes 8 \
        --threads 8 2>&1 | tail -1
fi

# ── Build filtered trees at each selectivity ─────────────────────────────
for NV in $INT32_VALUES_LIST; do
    FDAT="$OUT/filter_${NV}.fdat"
    TREE="$OUT/tree_${NV}.tree"
    if [[ ! -f "$TREE" ]]; then
        echo "Generating filter data (int32, $NV distinct values, selectivity=$(python3 -c "print(f'{1.0/$NV:.4f}')"))..."
        python3 "$GEN" --base "$BASE" --output "$FDAT" \
            --int32-col year --int32-values $NV
        echo "Building tree..."
        "$SEXTANT" build-tree-pca --input "$BASE" --index "$TREE" \
            --filter-data "$FDAT" \
            --leaf-capacity 5000 --pca-dims 32 --max-lloyd-passes 8 \
            --threads 8 2>&1 | tail -1
    fi
done

echo
echo "═══ Baseline (no filter) ═══"
"$SEXTANT" tree-search --index "$TREE_NOFILT" --query "$QUERY" \
    --ground-truth "$GT" --topk 10 --n-probe 0 --n-probe-ln 8 \
    --output "$OUT/results_nofilter.txt" 2>&1 | tee "$OUT/summary_nofilter.txt"

# ── Sweep selectivity ────────────────────────────────────────────────────
echo
echo "═══ Filtered Search Sweep ═══"
echo "selectivity,year_value,recall,QPS" | tee "$OUT/sweep_results.csv"

for NV in $INT32_VALUES_LIST; do
    TREE="$OUT/tree_${NV}.tree"
    SEL=$(python3 -c "print(f'{1.0/$NV:.6f}')")
    YEAR_VAL=$((2000 + NV / 2))

    RESULT_FILE="$OUT/results_${NV}.txt"
    "$SEXTANT" tree-search --index "$TREE" --query "$QUERY" \
        --ground-truth "$GT" --topk 10 --n-probe 0 --n-probe-ln 8 \
        --filter "year:eq:${YEAR_VAL}" \
        --output "$RESULT_FILE" 2>&1 | tee "$OUT/summary_${NV}.txt"

    RECALL=$(grep "recall@" "$OUT/summary_${NV}.txt" 2>/dev/null | awk '{print $2}' || echo "N/A")
    QPS=$(grep "QPS" "$OUT/summary_${NV}.txt" 2>/dev/null | grep -oP '[\d.]+(?= QPS)' || echo "N/A")
    echo "${SEL},${YEAR_VAL},${RECALL},${QPS}" | tee -a "$OUT/sweep_results.csv"
done

echo
echo "═══ Results saved to $OUT/sweep_results.csv ═══"
cat "$OUT/sweep_results.csv"
