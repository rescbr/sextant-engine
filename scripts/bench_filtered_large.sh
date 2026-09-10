#!/usr/bin/env bash
set -euo pipefail

# Filtered search benchmark + profiling on c4a.
# Sweeps selectivity on large datasets, computes filtered recall, profiles with perf.
#
# Usage: SEXTANT=... GEN=... PERF=1 ./bench_filtered_large.sh \
#          <base.fbin> <query.fbin> <gt.gt> <metric> <output_dir> <dataset_name>
# Example:
#   ./bench_filtered_large.sh sphere_ip_base.fbin sphere_ip_query.fbin sphere_ip_gt.gtmm ip /mnt/ssd/results sphere_10m

BASE="${1:?base .fbin required}"
QUERY="${2:?query .fbin required}"
GT="${3:?ground-truth .gt required}"
METRIC="${4:-l2sq}"
OUT="${5:-/tmp/bench}"
NAME="${6:-dataset}"
SEXTANT="${SEXTANT:-sextant}"
GEN="${GEN:-gen_filter_data.py}"
DO_PERF="${PERF:-0}"

mkdir -p "$OUT"

echo "═══ Filtered Benchmark: $NAME (metric=$METRIC) ═══"
echo "base: $BASE"

INT32_VALUES_LIST="10 50 100 500 1000 5000"

# ── Build baseline ────────────────────────────────────────────────────────
TREE_NOFILT="$OUT/${NAME}_nofilter.tree"
if [[ ! -f "$TREE_NOFILT" ]]; then
    echo "Building baseline (no filter)..."
    "$SEXTANT" build-tree-pca --input "$BASE" --index "$TREE_NOFILT" \
        --metric "$METRIC" --leaf-capacity 5000 --pca-dims 32 --max-lloyd-passes 8 \
        --threads 8 2>&1 | tail -1
fi

# ── Build filtered trees ─────────────────────────────────────────────────
for NV in $INT32_VALUES_LIST; do
    FDAT="$OUT/${NAME}_filter_${NV}.fdat"
    TREE="$OUT/${NAME}_tree_${NV}.tree"
    if [[ ! -f "$TREE" ]]; then
        echo "Generating filter ($NV distinct values, sel=$(python3 -c "print(f'{1.0/$NV:.4f}')"))..."
        python3 "$GEN" --base "$BASE" --output "$FDAT" \
            --int32-col year --int32-values $NV
        echo "Building tree..."
        "$SEXTANT" build-tree-pca --input "$BASE" --index "$TREE" \
            --filter-data "$FDAT" --metric "$METRIC" \
            --leaf-capacity 5000 --pca-dims 32 --max-lloyd-passes 8 \
            --threads 8 2>&1 | tail -1
    fi
done

# ── Baseline search ──────────────────────────────────────────────────────
echo
echo "── Baseline (no filter) ──"
"$SEXTANT" tree-search --index "$TREE_NOFILT" --query "$QUERY" \
    --ground-truth "$GT" --topk 10 --n-probe 0 --n-probe-ln 8 \
    --output "$OUT/${NAME}_results_nofilter.txt" 2>&1 | tee "$OUT/${NAME}_summary_nofilter.txt"

# ── Filtered sweep ───────────────────────────────────────────────────────
echo
echo "── Filtered Sweep ──"
echo "selectivity,year_val,recall_raw,qps,recall_filtered" | tee "$OUT/${NAME}_sweep.csv"

for NV in $INT32_VALUES_LIST; do
    TREE="$OUT/${NAME}_tree_${NV}.tree"
    FDAT="$OUT/${NAME}_filter_${NV}.fdat"
    SEL=$(python3 -c "print(f'{1.0/$NV:.6f}')")
    YEAR_VAL=$((2000 + NV / 2))
    RESULT_FILE="$OUT/${NAME}_results_${NV}.txt"

    # Optionally profile with perf on the FIRST run at 1% selectivity.
    PERF_WRAPPER=""
    if [[ "$DO_PERF" == "1" && "$NV" == "100" ]]; then
        PERF_WRAPPER="perf record -F 999 -g -- "
        echo "  [profiling with perf at NV=$NV]"
    fi

    $PERF_WRAPPER "$SEXTANT" tree-search --index "$TREE" --query "$QUERY" \
        --ground-truth "$GT" --topk 10 --n-probe 0 --n-probe-ln 8 \
        --filter "year:eq:${YEAR_VAL}" \
        --output "$RESULT_FILE" 2>&1 | tee "$OUT/${NAME}_summary_${NV}.txt"

    RECALL_RAW=$(grep "recall@" "$OUT/${NAME}_summary_${NV}.txt" 2>/dev/null | awk '{print $2}' || echo "N/A")
    QPS=$(grep "QPS" "$OUT/${NAME}_summary_${NV}.txt" 2>/dev/null | grep -oP '[\d.]+(?= QPS)' || echo "N/A")

    # Compute filtered recall.
    RECALL_FILT="N/A"
    if [[ -f "$RESULT_FILE" ]]; then
        RECALL_FILT=$(python3 /mnt/ssd/compute_filtered_recall.py \
            --results "$RESULT_FILE" --gt "$GT" --fdat "$FDAT" \
            --year-val "$YEAR_VAL" --k 10 2>/dev/null | grep -oP '[\d.]+' || echo "N/A")
    fi
    echo "${SEL},${YEAR_VAL},${RECALL_RAW},${QPS},${RECALL_FILT}" | tee -a "$OUT/${NAME}_sweep.csv"
done

# ── Perf report ──────────────────────────────────────────────────────────
if [[ "$DO_PERF" == "1" ]]; then
    echo
    echo "── Perf top hotspots (filtered search, 1% selectivity) ──"
    perf report --stdio --no-children -g none --percent-limit 1 2>/dev/null | head -40 \
        | tee "$OUT/${NAME}_perf_top.txt"
fi

echo
echo "═══ Done. Results: $OUT/${NAME}_sweep.csv ═══"
