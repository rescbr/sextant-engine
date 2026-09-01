#!/usr/bin/env bash
# Rebuild the Cohere 10M trees with the current binary (2026-09-01 HEAD).
# The old cohere_10m.tree (2026-08-06) was built during the
# adaptive_probe_gap bug era and has been deleted from GCS.
set -euo pipefail

SSD="${SSD:-/mnt/ssd}"
GCS_BUCKET="gs://<bench-bucket>"
DATA="$SSD/data"
TREES="$SSD/trees"
LOG_FILE="/var/log/sextant-rebuild.log"

exec > >(tee -a "$LOG_FILE") 2>&1
log() { echo "[sextant-rebuild] $(date -u +%H:%M:%S) $*"; }
upload_log() { gsutil cp "$LOG_FILE" "$GCS_BUCKET/cohere/rebuild.log" 2>/dev/null || true; }
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
    log "local SSD mounted at $SSD"
}

mount_local_ssd
mkdir -p "$DATA" "$TREES"

[[ -x "$SSD/sextant" ]] || gsutil cp "$GCS_BUCKET/binaries/sextant" "$SSD/sextant"
chmod +x "$SSD/sextant"
SXT="$SSD/sextant"
log "binary: $($SXT --version 2>&1 | head -1 || true)"

[[ -s "$DATA/cohere_10m_query.fbin" ]] || gsutil cp "$GCS_BUCKET/cohere/cohere_10m_query.fbin" "$DATA/cohere_10m_query.fbin"

# Train shards direct from Zilliz S3 (public; 300+ MB/s measured). The
# bucket's cohere_10m.parquet is UNUSABLE: pyarrow fixed_size_list encoding,
# which carquet cannot read (see infra notes). The Aug-6 build read these
# shards with --vector-col emb — the proven path.
ZILLIZ="https://assets.zilliz.com/benchmark/cohere_large_10m"
for i in 00 01 02 03 04 05 06 07 08 09; do
    f="train-${i}-of-10.parquet"
    [[ -s "$DATA/$f" ]] || wget -q -O "$DATA/$f" "$ZILLIZ/$f"
    log "shard $f ready"
done
log "data ready ($(du -sh "$DATA" | cut -f1))"

build_and_upload() { # <name> <extra flags...>
    local name="$1"; shift
    if [[ -s "$TREES/$name.tree" ]]; then
        log "$name already built — skipping"
    else
        log "building $name ($*)"
        local t0=$SECONDS
        if ! $SXT build-tree --input "$DATA/train-*-of-10.parquet" \
            --index "$TREES/$name.tree" --vector-col emb --threads 8 "$@"; then
            log "$name BUILD FAILED"; upload_log; return 1
        fi
        log "$name built in $((SECONDS - t0))s ($(du -h "$TREES/$name.tree" | cut -f1))"
    fi
    # Sanity: self-check the tree opens and answers 10 queries.
    $SXT tree-search --index="$TREES/$name.tree" \
        --query="$DATA/cohere_10m_query.fbin" --topk=10 \
        --n-probe=8 --n-probe-ln=8 --fastscan-w=300 --threads=1 \
        --output=/dev/null 2>&1 | grep -E "opened|QPS" | head -2 || true
    gsutil cp "$TREES/$name.tree" "$GCS_BUCKET/cohere/$name.tree"
    log "$name uploaded to GCS"
}

build_and_upload cohere_10m_shape  --quantizer scalar_shape
build_and_upload cohere_10m_pq8m384 --quantizer pq --pq-bits 8 --pq4-m 384

log "=== rebuild complete ==="
echo done | gsutil cp - "$GCS_BUCKET/cohere/.rebuild_done"
upload_log
poweroff || true
