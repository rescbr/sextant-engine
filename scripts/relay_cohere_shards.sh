#!/usr/bin/env bash
# Relay the Zilliz cohere_10m shards into OUR GCS bucket, one shard at a
# time (download → upload → next), so progress is durable and future VMs
# never touch S3 again.
set -euo pipefail
SSD="${SSD:-/mnt/ssd}"
GCS_BUCKET="gs://<bench-bucket>"
LOG_FILE="/var/log/shard-relay.log"
exec > >(tee -a "$LOG_FILE") 2>&1
log() { echo "[shard-relay] $(date -u +%H:%M:%S) $*"; }
trap 'gsutil cp "$LOG_FILE" "$GCS_BUCKET/cohere/shard-relay.log" 2>/dev/null || true' EXIT

if ! mountpoint -q "$SSD"; then
    mkdir -p "$SSD"
    root_dev=$(findmnt -no SOURCE / | sed -E 's/p?[0-9]+$//')
    ssds=()
    for dev in /dev/nvme*n1; do
        [[ -b "$dev" ]] && [[ "$dev" != "$root_dev" ]] && ssds+=("$dev")
    done
    if [[ ${#ssds[@]} -eq 1 ]]; then
        blkid "${ssds[0]}" >/dev/null 2>&1 || mkfs.ext4 -F "${ssds[0]}"
        mount -o noatime "${ssds[0]}" "$SSD"
    elif [[ ${#ssds[@]} -gt 1 ]]; then
        apt-get install -y -qq mdadm >/dev/null 2>&1
        mdadm --stop /dev/md/ssdraid 2>/dev/null || true
        mdadm --zero-superblock "${ssds[@]}" 2>/dev/null || true
        mdadm --create /dev/md/ssdraid --level=0 --raid-devices=${#ssds[@]} "${ssds[@]}" --force
        sleep 2; mkfs.ext4 -F /dev/md/ssdraid
        mount -o noatime /dev/md/ssdraid "$SSD"
    fi
fi
mkdir -p "$SSD/relay"
cd "$SSD/relay"

ZILLIZ="https://assets.zilliz.com/benchmark/cohere_large_10m"
for i in 00 01 02 03 04 05 06 07 08 09; do
    f="train-${i}-of-10.parquet"
    if gsutil -q stat "$GCS_BUCKET/cohere/shards/$f"; then
        log "$f already in GCS — skip"
        continue
    fi
    log "downloading $f..."
    wget -q -O "$f" "$ZILLIZ/$f"
    log "uploading $f ($(du -h "$f" | cut -f1))..."
    gsutil cp "$f" "$GCS_BUCKET/cohere/shards/$f"
    rm -f "$f"   # keep SSD usage flat
    log "$f relayed"
done
log "=== relay complete ==="
echo done | gsutil cp - "$GCS_BUCKET/cohere/.shards_uploaded"
poweroff || true
