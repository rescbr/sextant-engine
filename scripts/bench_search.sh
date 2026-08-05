#!/usr/bin/env bash
# =============================================================================
# Sextant large-scale search benchmark: recall@10 vs QPS Pareto sweep.
#
# Datasets: sift1m (1M/128d), arxiv_nomic (1.34M/768d), msmarco (8.7M/768d),
#           sphere_ip (10M/768d, Inner Product)
#
# All datasets are built from uncompressed Parquet (fast mmap build path).
# The Parquet files include synthetic metadata columns (category, year, score)
# used for the filtered-search sweep.
#
# Sweeps:
#   1. n_probe sweep (1,2,4,8,16,32) × rerank (on/off) → recall vs QPS Pareto
#   2. Filtered search (category eq) at 1% selectivity × n_probe
#
# Usage: BENCH_DIR=/path/to/data+bin bash bench_search.sh
# Expects:
#   $BENCH_DIR/bin/sextant
#   $BENCH_DIR/bin/fbin_to_parquet
#   $BENCH_DIR/<dataset>_base.parquet (or .fbin, auto-converted)
#   $BENCH_DIR/<dataset>_query.fbin
#   $BENCH_DIR/<dataset>_gt.gt
# =============================================================================
set -euo pipefail

BIN="${BENCH_DIR:-/mnt/local-ssd}/bin/sextant"
CONV="${BENCH_DIR:-/mnt/local-ssd}/bin/fbin_to_parquet"
DATA="${BENCH_DIR:-/mnt/local-ssd}"
OUT="${BENCH_DIR:-/mnt/local-ssd}/bench-search-out"
mkdir -p "$OUT"

# Convert fbin to uncompressed parquet if parquet doesn't exist yet.
ensure_parquet() {
    local name=$1
    local parquet="$DATA/${name}.parquet"
    if [ -f "$parquet" ]; then
        echo "[skip] $parquet exists"
        echo "$parquet"
        return
    fi
    local fbin="$DATA/${name}_base.fbin"
    if [ ! -f "$fbin" ]; then
        echo "[warn] No parquet or fbin for $name — skipping"
        echo ""
        return
    fi
    echo "[convert] $fbin → $parquet (uncompressed)"
    "$CONV" "$fbin" "$parquet" --uncompressed
    echo "$parquet"
}

# Build index from parquet.
build_index() {
    local name=$1 parquet=$2 dim=$3 metric=$4
    local idx="$OUT/${name}.tree"
    if [ -f "$idx" ]; then
        echo "[skip] $idx already exists"
        return
    fi
    local pq4_m=$((dim / 4))
    echo "[build] $name (dim=$dim, m=$pq4_m, metric=$metric)"
    local t0 t1
    t0=$(date +%s.%N)
    "$BIN" build-tree "$parquet" "$idx" \
        --leaf-capacity 5000 \
        --pca-dims 32 \
        --threads 0 \
        --pq4-m "$pq4_m" \
        --pq-bits 4 \
        --metric "$metric"
    t1=$(date +%s.%N)
    echo "[build] $name done in $(echo "$t1 - $t0" | bc)s"
    # Index file size.
    local fsize
    fsize=$(du -h "$idx" | cut -f1)
    echo "[build] index size: $fsize"
}

# Search sweep: n_probe × rerank.
search_sweep() {
    local name=$1 idx=$2 query=$3 gt=$4
    echo ""
    echo "========================================"
    echo "  $name — recall@10 vs QPS"
    echo "========================================"
    printf "%-8s %-8s %-12s %-12s\n" "n_probe" "rerank" "recall@10" "QPS"
    printf "%-8s %-8s %-12s %-12s\n" "-------" "------" "---------" "---"

    for nprobe in 1 2 4 8 16 32; do
        for rerank in "" "--no-rerank"; do
            local label="on"
            [ -n "$rerank" ] && label="off"
            local result
            result=$("$BIN" tree-search "$idx" "$query" \
                --ground-truth "$gt" \
                --topk 10 \
                --n-probe "$nprobe" \
                --n-probe-ln "$nprobe" \
                --fastscan-w 300 \
                $rerank \
                --threads 0 2>&1 || true)
            local recall qps
            recall=$(echo "$result" | grep -oP 'recall@\d+:\s*\K[\d.]+' | head -1)
            qps=$(echo "$result" | grep -oP 'QPS:\s*\K[\d.]+' | head -1)
            [ -z "$recall" ] && recall="ERR"
            [ -z "$qps" ] && qps="ERR"
            printf "%-8s %-8s %-12s %-12s\n" "$nprobe" "$label" "$recall" "$qps"
        done
    done
}

# Filtered search sweep. The parquet already has "category" (string) + "year"
# (int32) + "score" (float) columns. The tree built from parquet has these
# as filter columns. We sweep year==X at ~1% selectivity (year range 2000-2099).
filtered_sweep() {
    local name=$1 idx=$2 query=$3 gt=$4
    echo ""
    echo "========================================"
    echo "  $name — filtered search (year==2025, ~1% selectivity)"
    echo "========================================"
    printf "%-12s %-8s %-12s %-12s\n" "selectivity" "n_probe" "recall@10" "QPS"
    printf "%-12s %-8s %-12s %-12s\n" "-----------" "-------" "---------" "---"

    for nprobe in 8 16 32; do
        local result
        result=$("$BIN" tree-search "$idx" "$query" \
            --ground-truth "$gt" \
            --topk 10 \
            --n-probe "$nprobe" \
            --n-probe-ln "$nprobe" \
            --filter "year:eq:int32:2025" \
            --threads 0 2>&1 || true)
        local recall qps
        recall=$(echo "$result" | grep -oP 'recall@\d+:\s*\K[\d.]+' | head -1)
        qps=$(echo "$result" | grep -oP 'QPS:\s*\K[\d.]+' | head -1)
        [ -z "$recall" ] && recall="ERR"
        [ -z "$qps" ] && qps="ERR"
        printf "%-12s %-8s %-12s %-12s\n" "~1%" "$nprobe" "$recall" "$qps"
    done
}

# =============================================================================
# Main
# =============================================================================

echo "============================================"
echo " Sextant Search Benchmark (Parquet build)"
echo " $(date)"
echo " Machine: $(uname -m)"
echo "============================================"

# --- sift1m (1M, 128d, L2) ---
if [ -f "$DATA/sift1m_base.fbin" ] || [ -f "$DATA/sift1m.parquet" ]; then
    PQT=$(ensure_parquet "sift1m")
    if [ -n "$PQT" ]; then
        build_index "sift1m" "$PQT" 128 "l2sq"
        search_sweep "sift1m" "$OUT/sift1m.tree" "$DATA/sift1m_query.fbin" "$DATA/sift1m_gt.gt"
        filtered_sweep "sift1m" "$OUT/sift1m.tree" "$DATA/sift1m_query.fbin" "$DATA/sift1m_gt.gt"
    fi
fi

# --- arxiv_nomic (1.34M, 768d, L2) ---
if [ -f "$DATA/arxiv_nomic_base.fbin" ] || [ -f "$DATA/arxiv_nomic.parquet" ]; then
    PQT=$(ensure_parquet "arxiv_nomic")
    if [ -n "$PQT" ]; then
        build_index "arxiv" "$PQT" 768 "l2sq"
        search_sweep "arxiv" "$OUT/arxiv.tree" "$DATA/arxiv_nomic_query.fbin" "$DATA/arxiv_nomic_gt.gt"
        filtered_sweep "arxiv" "$OUT/arxiv.tree" "$DATA/arxiv_nomic_query.fbin" "$DATA/arxiv_nomic_gt.gt"
    fi
fi

# --- msmarco (8.7M, 768d, L2 or IP) ---
if [ -f "$DATA/msmarco_base.fbin" ] || [ -f "$DATA/msmarco.parquet" ]; then
    PQT=$(ensure_parquet "msmarco")
    if [ -n "$PQT" ]; then
        build_index "msmarco" "$PQT" 768 "l2sq"
        search_sweep "msmarco" "$OUT/msmarco.tree" "$DATA/msmarco_query.fbin" "$DATA/msmarco_gt.gt"
    fi
fi

# --- sphere_ip (10M, 768d, Inner Product) ---
if [ -f "$DATA/sphere_ip_base.fbin" ] || [ -f "$DATA/sphere_ip.parquet" ]; then
    PQT=$(ensure_parquet "sphere_ip")
    if [ -n "$PQT" ]; then
        build_index "sphere" "$PQT" 768 "ip"
        search_sweep "sphere" "$OUT/sphere.tree" "$DATA/sphere_ip_query.fbin" "$DATA/sphere_ip_gt.gt"
    fi
fi

echo ""
echo "============================================"
echo " Benchmark complete."
echo " Results in $OUT/"
echo " $(date)"
echo "============================================"
