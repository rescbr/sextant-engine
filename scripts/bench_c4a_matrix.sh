#!/usr/bin/env bash
# =============================================================================
# c4a benchmark matrix (2026-09-01) — leaderboard re-bench + adaptive-W + np/ln
#
# Runs on a c4a-standard-8-lssd VM as a startup script. Idempotent per-row via
# markers; results appended as CSV lines to GCS (bench_matrix.csv) + full log.
#
# Matrix (100k, threads=8, k=10, medians of 3 runs):
#   per dataset × quantizer:
#     - baseline np8/ln8 W=300 fixed top-k     (leaderboard re-bench)
#     - adaptive-W tau in {2,3,5} (W=300)      (new: shortlist contract)
#     - arxiv only: np4/ln4 baseline + tau     (new operating point)
#     - cohere IP (scalar_lm, metric=ip)       (deficit + tau recovery)
# =============================================================================
set -euo pipefail

SSD="${SSD:-/mnt/ssd}"
GCS_BUCKET="gs://<bench-bucket>"
DATA="$SSD/data"
TREES="$SSD/trees"
OUT="$SSD/out"
LOG_FILE="/var/log/sextant-bench.log"

exec > >(tee -a "$LOG_FILE") 2>&1
log() { echo "[sextant-bench] $(date -u +%H:%M:%S) $*"; }
upload_log() { gsutil cp "$LOG_FILE" "$GCS_BUCKET/bench_matrix.log" 2>/dev/null || true; }
trap 'upload_log' EXIT

mount_local_ssd() {
    if mountpoint -q "$SSD"; then return; fi
    mkdir -p "$SSD"
    local root_dev
    root_dev=$(findmnt -no SOURCE / | sed -E 's/p?[0-9]+$//')
    local ssds=()
    for dev in /dev/nvme*n1; do
        [[ -b "$dev" ]] && [[ "$dev" != "$root_dev" ]] && ssds+=("$dev")
    done
    if [[ ${#ssds[@]} -eq 0 ]]; then log "no local SSD — boot disk"; return; fi
    if [[ ${#ssds[@]} -eq 1 ]]; then
        local d="${ssds[0]}"
        blkid "$d" >/dev/null 2>&1 || mkfs.ext4 -F "$d"
        mount -o noatime "$d" "$SSD"
    else
        apt-get install -y -qq mdadm >/dev/null 2>&1
        mdadm --stop /dev/md/ssdraid 2>/dev/null || true
        mdadm --zero-superblock "${ssds[@]}" 2>/dev/null || true
        mdadm --create /dev/md/ssdraid --level=0 --raid-devices=${#ssds[@]} "${ssds[@]}" --force
        sleep 2
        mkfs.ext4 -F /dev/md/ssdraid
        mount -o noatime /dev/md/ssdraid "$SSD"
    fi
    log "local SSD mounted at $SSD ($(df -BG --output=size "$SSD" | tail -1))"
}

mkdir -p "$SSD"
mount_local_ssd
mkdir -p "$DATA" "$TREES" "$OUT"

# --- fetch binaries + data (idempotent) ---
for b in sextant; do
    [[ -x "$SSD/$b" ]] || gsutil cp "$GCS_BUCKET/binaries/$b" "$SSD/$b"
    chmod +x "$SSD/$b"
done
SXT="$SSD/sextant"
log "binary: $($SXT --version 2>/dev/null | head -1 || echo unknown)"

for f in cohere_100k_base.fbin cohere_100k_query.fbin cohere_100k_gt.gtmm \
         arxiv100k_base.fbin arxiv100k_query.fbin arxiv100k_gt.gtmm; do
    [[ -s "$DATA/$f" ]] || gsutil cp "$GCS_BUCKET/$f" "$DATA/$f"
done
log "data ready"

# --- CSV helpers (append a row to local csv; sync to GCS each row) ---
CSV="$OUT/bench_matrix.csv"
CSV_GCS="$GCS_BUCKET/bench_matrix.csv"
[[ -f "$CSV" ]] || echo "dataset,quantizer,metric,np,ln,tau,run,recall,qps,mean_ids" > "$CSV"
row() {
    # Append to the local CSV; upload the FULL csv (gsutil has no append —
    # copying the whole small file each row is cheap and keeps GCS in sync).
    echo "$1" >> "$CSV"
    gsutil cp "$CSV" "$CSV_GCS" >/dev/null 2>&1 || true
    upload_log
}

# run_row <label-csv-prefix> <index-args...>
run_row() {
    local prefix="$1"; shift
    for rep in 1 2 3; do
        local out
        out=$("$SXT" tree-search "$@" --threads=8 2>&1 || true)
        local recall qps mean
        recall=$(echo "$out" | grep -oE 'recall@10: [0-9.]+' | grep -oE '[0-9.]+$' || echo NA)
        qps=$(echo "$out" | grep -oE '\([0-9.]+ QPS\)' | grep -oE '[0-9.]+' || echo NA)
        mean=$(echo "$out" | grep -oE 'mean results/query: [0-9.]+' | grep -oE '[0-9.]+$' || echo 10)
        row "${prefix},${rep},${recall},${qps},${mean}"
        log "${prefix} rep${rep}: recall=${recall} qps=${qps} mean=${mean}"
    done
}

# --- build trees (idempotent) ---
build_tree() { # <name> <base> <extra build flags...>
    local name="$1" base="$2"; shift 2
    [[ -s "$TREES/$name.tree" ]] && return 0
    log "building $name ($*)"
    $SXT build-tree --input "$DATA/$base" --index "$TREES/$name.tree" \
        --threads 8 "$@" 2>&1 | tail -1 || true
}

build_tree co_shape   cohere_100k_base.fbin --quantizer scalar_shape
build_tree co_uni     cohere_100k_base.fbin --quantizer scalar_uniform
build_tree co_slm     cohere_100k_base.fbin --quantizer scalar_lloydmax
build_tree co_pq8m384 cohere_100k_base.fbin --quantizer pq --pq-bits 8 --pq4-m 384
build_tree co_ip_slm  cohere_100k_base.fbin --quantizer scalar_lloydmax --metric ip
build_tree ax_shape   arxiv100k_base.fbin   --quantizer scalar_shape
build_tree ax_pq8m384 arxiv100k_base.fbin   --quantizer pq --pq-bits 8 --pq4-m 384

log "=== matrix start ==="

# --- cohere L2sq baselines + adaptive-W ---
for t in co_shape:scalar_shape co_uni:scalar_uniform co_slm:scalar_lloydmax; do
    name="${t%%:*}"; q="${t##*:}"
    run_row "cohere,$q,l2sq,8,8,0" \
        --index="$TREES/$name.tree" --query="$DATA/cohere_100k_query.fbin" \
        --ground-truth="$DATA/cohere_100k_gt.gtmm" --topk=10 \
        --n-probe=8 --n-probe-ln=8 --fastscan-w=300
    for tau in 2 3 5; do
        run_row "cohere,$q,l2sq,8,8,$tau" \
            --index="$TREES/$name.tree" --query="$DATA/cohere_100k_query.fbin" \
            --ground-truth="$DATA/cohere_100k_gt.gtmm" --topk=10 \
            --n-probe=8 --n-probe-ln=8 --fastscan-w=300 --adaptive-w-gap=$tau
    done
done

# --- cohere IP ---
run_row "cohere,scalar_lloydmax,ip,8,8,0" \
    --index="$TREES/co_ip_slm.tree" --query="$DATA/cohere_100k_query.fbin" \
    --ground-truth="$DATA/cohere_100k_gt.gtmm" --topk=10 \
    --n-probe=8 --n-probe-ln=8 --fastscan-w=300
for tau in 2 3 5; do
    run_row "cohere,scalar_lloydmax,ip,8,8,$tau" \
        --index="$TREES/co_ip_slm.tree" --query="$DATA/cohere_100k_query.fbin" \
        --ground-truth="$DATA/cohere_100k_gt.gtmm" --topk=10 \
        --n-probe=8 --n-probe-ln=8 --fastscan-w=300 --adaptive-w-gap=$tau
done

# --- cohere pq8 m384 (best 384B recall point) ---
run_row "cohere,pq8m384,l2sq,8,8,0" \
    --index="$TREES/co_pq8m384.tree" --query="$DATA/cohere_100k_query.fbin" \
    --ground-truth="$DATA/cohere_100k_gt.gtmm" --topk=10 \
    --n-probe=8 --n-probe-ln=8 --fastscan-w=300
for tau in 2 3 5; do
    run_row "cohere,pq8m384,l2sq,8,8,$tau" \
        --index="$TREES/co_pq8m384.tree" --query="$DATA/cohere_100k_query.fbin" \
        --ground-truth="$DATA/cohere_100k_gt.gtmm" --topk=10 \
        --n-probe=8 --n-probe-ln=8 --fastscan-w=300 --adaptive-w-gap=$tau
done

# --- arxiv: np8/ln8 + np4/ln4, baseline + adaptive-W ---
for npset in "8 8" "4 4"; do
    set -- $npset; np=$1; ln=$2
    run_row "arxiv,scalar_shape,l2sq,$np,$ln,0" \
        --index="$TREES/ax_shape.tree" --query="$DATA/arxiv100k_query.fbin" \
        --ground-truth="$DATA/arxiv100k_gt.gtmm" --topk=10 \
        --n-probe=$np --n-probe-ln=$ln --fastscan-w=300
    for tau in 2 3; do
        run_row "arxiv,scalar_shape,l2sq,$np,$ln,$tau" \
            --index="$TREES/ax_shape.tree" --query="$DATA/arxiv100k_query.fbin" \
            --ground-truth="$DATA/arxiv100k_gt.gtmm" --topk=10 \
            --n-probe=$np --n-probe-ln=$ln --fastscan-w=300 --adaptive-w-gap=$tau
    done
    done

    # --- arxiv pq8 m384 np8/ln8 baseline ---
    run_row "arxiv,pq8m384,l2sq,8,8,0" \
        --index="$TREES/ax_pq8m384.tree" --query="$DATA/arxiv100k_query.fbin" \
        --ground-truth="$DATA/arxiv100k_gt.gtmm" --topk=10 \
        --n-probe=8 --n-probe-ln=8 --fastscan-w=300
    run_row "arxiv,pq8m384,l2sq,8,8,3" \
        --index="$TREES/ax_pq8m384.tree" --query="$DATA/arxiv100k_query.fbin" \
        --ground-truth="$DATA/arxiv100k_gt.gtmm" --topk=10 \
        --n-probe=8 --n-probe-ln=8 --fastscan-w=300 --adaptive-w-gap=3

log "=== matrix complete ==="
echo done | gsutil cp - "$GCS_BUCKET/.bench_matrix_done"
upload_log
poweroff || true
