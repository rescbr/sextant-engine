#!/usr/bin/env bash
# Benchmark: fbin vs parquet (zstd) vs parquet (uncompressed) build paths.
set -euo pipefail

SSD="${SSD:-/mnt/ssd}"
BIN="$SSD/bin"
DATA="$SSD"
OUT="$SSD/bench-out"
mkdir -p "$OUT"

BASE_FBIN="$DATA/arxiv_nomic_base.fbin"
QUERY_FBIN="$DATA/arxiv_nomic_query.fbin"
GT_FILE="$DATA/arxiv_nomic_gt.gt"
PARQUET_ZSTD="$OUT/arxiv_nomic_zstd.parquet"
PARQUET_RAW="$OUT/arxiv_nomic_raw.parquet"

SEXTANT="$BIN/sextant"
FBIN2PARQUET="$BIN/fbin_to_parquet"

build_and_time() {
    local label="$1" input="$2" tree="$3" extra_flags="${4:-}"
    local log="$OUT/build_${label}.log"
    echo "--- Building: $label ---"
    local t0 t1
    t0=$(date +%s.%N)
    $SEXTANT build-tree --input "$input" --index "$tree" \
        --leaf-capacity 5000 --pca-dims 32 --threads 0 \
        --log-level info $extra_flags 2>&1 | tee "$log" | tail -1
    t1=$(date +%s.%N)
    local secs=$(echo "$t1 - $t0" | bc)
    echo "  ${label} build: ${secs}s  (tree: $(du -h "$tree" | cut -f1))"
    echo "$secs" > "$OUT/time_${label}.txt"
    echo ""
}

search_recall() {
    local label="$1" tree="$2"
    echo "--- Search: $label ---"
    $SEXTANT tree-search --index "$tree" --query "$QUERY_FBIN" \
        --ground-truth "$GT_FILE" \
        --topk 10 --n-probe 8 --n-probe-ln 8 2>&1 | grep -E "(recall|QPS)" || true
    echo ""
}

echo "============================================================"
echo "Sextant build-path benchmark: fbin vs parquet-zstd vs parquet-raw"
echo "dataset: arxiv_nomic (1.34M, 768-dim)"
echo "============================================================"
echo ""

# --- Conversions ---
if [[ ! -f "$PARQUET_ZSTD" ]]; then
    echo "[convert] fbin → parquet (zstd)..."
    /usr/bin/time -v "$FBIN2PARQUET" "$BASE_FBIN" "$PARQUET_ZSTD" 2>&1 | \
        grep -E "(fbin_to_parquet|Elapsed)"
    echo "  size: $(du -h "$PARQUET_ZSTD" | cut -f1)"
    echo ""
fi

if [[ ! -f "$PARQUET_RAW" ]]; then
    echo "[convert] fbin → parquet (uncompressed)..."
    /usr/bin/time -v "$FBIN2PARQUET" "$BASE_FBIN" "$PARQUET_RAW" --uncompressed 2>&1 | \
        grep -E "(fbin_to_parquet|Elapsed)"
    echo "  size: $(du -h "$PARQUET_RAW" | cut -f1)"
    echo ""
fi

# --- Builds ---
build_and_time fbin    "$BASE_FBIN"    "$OUT/tree_fbin.sextant"
build_and_time pq-zstd "$PARQUET_ZSTD" "$OUT/tree_pq_zstd.sextant"
build_and_time pq-raw  "$PARQUET_RAW"  "$OUT/tree_pq_raw.sextant"

# --- Search ---
search_recall fbin    "$OUT/tree_fbin.sextant"
search_recall pq-zstd "$OUT/tree_pq_zstd.sextant"
search_recall pq-raw  "$OUT/tree_pq_raw.sextant"

# --- Summary ---
FBIN_T=$(cat "$OUT/time_fbin.txt")
ZSTD_T=$(cat "$OUT/time_pq-zstd.txt")
RAW_T=$(cat "$OUT/time_pq-raw.txt")

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
printf "  %-12s %8s  %8s\n" "source" "build" "vs-fbin"
printf "  %-12s %8ss %8s\n" "fbin"     "$FBIN_T" "1.00x"
printf "  %-12s %8ss %8s\n" "pq-zstd"  "$ZSTD_T" "$(echo "scale=2; $ZSTD_T / $FBIN_T" | bc)x"
printf "  %-12s %8ss %8s\n" "pq-raw"   "$RAW_T"  "$(echo "scale=2; $RAW_T / $FBIN_T" | bc)x"
echo ""
echo "  parquet-zstd:       $(du -h "$PARQUET_ZSTD" | cut -f1)"
echo "  parquet-uncompressed: $(du -h "$PARQUET_RAW" | cut -f1)"
echo "  fbin:               $(du -h "$BASE_FBIN" | cut -f1)"
echo "============================================================"
