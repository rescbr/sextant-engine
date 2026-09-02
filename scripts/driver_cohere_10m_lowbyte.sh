#!/usr/bin/env bash
# 10M low-byte tier validation: PQ8 m96 (96B) + m192 (192B) with adaptive-W.
# Question: does the 100k finding (deep-tau recovers weak quantizers to
# ~97-99% @ ~165 rescans) survive at 10M, where W=300 containment binds?
set -euo pipefail
SSD=/mnt/ssd
GCS_BUCKET=gs://<bench-bucket>
DATA=$SSD/data
TREES=$SSD/trees_lb
SXT=$SSD/sextant
mkdir -p "$TREES"
LOG=$SSD/driver_lb.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[10m-lb] $(date -u +%H:%M:%S) $*"; }
up() { gsutil cp "$LOG" "$GCS_BUCKET/cohere/driver_lb.log" 2>/dev/null || true; }
trap up EXIT

[[ -x "$SXT" ]] || { gsutil cp "$GCS_BUCKET/binaries/sextant" "$SXT"; chmod +x "$SXT"; }
for f in cohere_10m_norm.fbin cohere_10m_query_norm.fbin cohere_10m_gt.gtmm; do
    [[ -s "$DATA/$f" ]] || gsutil cp "$GCS_BUCKET/cohere/$f" "$DATA/$f"
done
log "inputs staged"

build() { local name="$1"; shift
    [[ -s "$TREES/$name.tree" ]] && { log "$name exists"; return; }
    log "building $name ($*)"
    local t0=$SECONDS
    $SXT build-tree --input "$DATA/cohere_10m_norm.fbin" \
        --index "$TREES/$name.tree" --threads 8 "$@"
    log "$name built in $((SECONDS-t0))s ($(du -h "$TREES/$name.tree" | cut -f1))"
    gsutil cp "$TREES/$name.tree" "$GCS_BUCKET/cohere/$name.norm.tree"
}
build cohere_10m_pq8m96  --quantizer pq --pq-bits 8 --pq4-m 96
build cohere_10m_pq8m192 --quantizer pq --pq-bits 8 --pq4-m 192

CSV=$SSD/bench_10m_lb.csv
[[ -f "$CSV" ]] || echo "tree,np,tau,run,recall,qps,mean_ids" > "$CSV"
row() { echo "$1" >> "$CSV"; gsutil cp "$CSV" "$GCS_BUCKET/cohere/bench_10m_lb.csv" >/dev/null 2>&1 || true; up; }

run_row() { local t="$1" np="$2"; shift 2
    for rep in 1 2; do
        local out recall qps mean tau
        out=$($SXT tree-search --index="$TREES/$t.tree" \
              --query="$DATA/cohere_10m_query_norm.fbin" \
              --ground-truth="$DATA/cohere_10m_gt.gtmm" \
              --topk=10 --fastscan-w=300 --threads=8 \
              --n-probe="$np" --n-probe-ln=4 "$@" 2>&1 || true)
        recall=$(echo "$out" | grep -oE 'recall@10: [0-9.]+' | grep -oE '[0-9.]+$' || echo NA)
        qps=$(echo "$out" | grep -oE '\([0-9.]+ QPS\)' | grep -oE '[0-9.]+' || echo NA)
        mean=$(echo "$out" | grep -oE 'mean results/query: [0-9.]+' | grep -oE '[0-9.]+$' || echo 10)
        tau=0
        [[ "$*" == *--adaptive-w-gap=* ]] && tau=$(echo "$*" | grep -oE '[0-9.]+$')
        row "$t,$np,$tau,$rep,$recall,$qps,$mean"
        log "$t np=$np tau=$tau rep$rep: r=$recall qps=$qps m=$mean"
    done
}

# Main operating point np256/ln4 + the QPS point np64, tau grid incl. deep
# cuts (tau=10 keeps most of the 300-heap -> measures the containment
# ceiling at 10M).
for t in cohere_10m_pq8m96 cohere_10m_pq8m192; do
    for np in 64 256; do
        run_row "$t" "$np"
        for tau in 3 5 10 20; do
            run_row "$t" "$np" --adaptive-w-gap=$tau
        done
    done
done

log "=== 10M low-byte COMPLETE ==="
echo done | gsutil cp - "$GCS_BUCKET/cohere/.bench_10m_lb_done"
up
