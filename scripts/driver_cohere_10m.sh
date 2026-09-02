#!/usr/bin/env bash
# 10M driver: normalize -> rebuild trees -> validate -> np/ln + adaptive-W
# matrix. Runs under nohup on the VM; all runs --threads=8.
set -euo pipefail
SSD=/mnt/ssd
GCS_BUCKET=gs://<bench-bucket>
DATA=$SSD/data
TREES=$SSD/trees2
SXT=$SSD/sextant
mkdir -p "$TREES"
LOG=/mnt/ssd/driver.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[10m] $(date -u +%H:%M:%S) $*"; }
up() { gsutil cp "$LOG" "$GCS_BUCKET/cohere/driver.log" 2>/dev/null || true; }
trap up EXIT

# 1) Normalize (base in shard-glob order + queries). Row order == the GTMM
#    mapping space, so the corrected GT stays valid.
if [[ ! -s "$DATA/cohere_10m_norm.fbin" ]]; then
    log "normalizing..."
    python3 /mnt/ssd/normalize_cohere10m.py
    log "normalize done ($(du -h "$DATA/cohere_10m_norm.fbin" | cut -f1))"
fi
Q="$DATA/cohere_10m_query_norm.fbin"
GT="$DATA/cohere_10m_gt.gtmm"

# 2) Rebuild both trees from the normalized fbin.
build() { local name="$1"; shift
    [[ -s "$TREES/$name.tree" ]] && { log "$name exists"; return; }
    log "building $name ($*)"
    local t0=$SECONDS
    $SXT build-tree --input "$DATA/cohere_10m_norm.fbin" \
        --index "$TREES/$name.tree" --threads 8 "$@"
    log "$name built in $((SECONDS-t0))s ($(du -h "$TREES/$name.tree" | cut -f1))"
    gsutil cp "$TREES/$name.tree" "$GCS_BUCKET/cohere/$name.norm.tree"
}
build cohere_10m_shape  --quantizer scalar_shape
build cohere_10m_pq8m384 --quantizer pq --pq-bits 8 --pq4-m 384

CSV=/mnt/ssd/bench_10m.csv
[[ -f "$CSV" ]] || echo "tree,np,ln,tau,run,recall,qps,mean_ids" > "$CSV"
row() { echo "$1" >> "$CSV"; gsutil cp "$CSV" "$GCS_BUCKET/cohere/bench_10m.csv" >/dev/null 2>&1 || true; up; }

run_row() { local t="$1" np="$2" ln="$3"; shift 3
    for rep in 1 2 3; do
        local out recall qps mean
        out=$($SXT tree-search --index="$TREES/$t.tree" --query="$Q" \
              --ground-truth="$GT" --topk=10 --fastscan-w=300 --threads=8 \
              --n-probe="$np" --n-probe-ln="$ln" "$@" 2>&1 || true)
        recall=$(echo "$out" | grep -oE 'recall@10: [0-9.]+' | grep -oE '[0-9.]+$' || echo NA)
        qps=$(echo "$out" | grep -oE '\([0-9.]+ QPS\)' | grep -oE '[0-9.]+' || echo NA)
        mean=$(echo "$out" | grep -oE 'mean results/query: [0-9.]+' | grep -oE '[0-9.]+$' || echo 10)
        local tau=0
        [[ "$*" == *--adaptive-w-gap=* ]] && tau=$(echo "$*" | grep -oE '[0-9.]+$')
        row "$t,$np,$ln,$tau,$rep,$recall,$qps,$mean"
        log "$t np=$np ln=$ln tau=$tau rep$rep: r=$recall qps=$qps m=$mean"
    done
}

# 3) Validation cell first: heavy probe must now show real recall.
log "=== validation ==="
run_row cohere_10m_shape 1024 16

# 4) np/ln frontier (tau=0) — 1 rep.
log "=== phase 1: np/ln frontier ==="
for t in cohere_10m_shape cohere_10m_pq8m384; do
    for ln in 4 16; do
        for np in 16 64 256 1024; do
            run_row "$t" "$np" "$ln"
        done
    done
done

# 5) adaptive-W at three operating points.
log "=== phase 2: adaptive-W ==="
for t in cohere_10m_shape cohere_10m_pq8m384; do
    for point in "64 4" "256 4" "256 16"; do
        set -- $point
        run_row "$t" "$1" "$2"
        for tau in 2 3 5; do
            run_row "$t" "$1" "$2" --adaptive-w-gap=$tau
        done
    done
done

log "=== 10M COMPLETE ==="
echo done | gsutil cp - "$GCS_BUCKET/cohere/.bench_10m_done"
up
