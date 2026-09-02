#!/usr/bin/env bash
# Build a row-position GTMM for cohere 10M.
#
# Why: Zilliz GT (neighbors.parquet) stores DATASET ids, and the train
# shards' id column is jumbled relative to file order. The trees were built
# from the shards in raw glob/file order, so row_id = concat position.
# This script reads each shard's id column, builds id -> position, maps the
# GT neighbors to positions, and writes a GTMM file (k=100) the CLI reads
# natively.
set -euo pipefail
SSD="${SSD:-/mnt/ssd}"
GCS_BUCKET="gs://<bench-bucket>"
LOG_FILE="/var/log/gt-fix.log"
exec > >(tee -a "$LOG_FILE") 2>&1
log() { echo "[gt-fix] $(date -u +%H:%M:%S) $*"; }
trap 'gsutil cp "$LOG_FILE" "$GCS_BUCKET/cohere/gt-fix.log" 2>/dev/null || true' EXIT

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
    fi
fi
mkdir -p "$SSD/data"
cd "$SSD/data"
for i in 00 01 02 03 04 05 06 07 08 09; do
    f="train-${i}-of-10.parquet"
    [[ -s "$f" ]] || gsutil cp "$GCS_BUCKET/cohere/shards/$f" "$f"
done
[[ -s neighbors.parquet ]] || gsutil cp "$GCS_BUCKET/cohere/cohere_10m_gt_all.parquet" neighbors.parquet
[[ -s test.parquet ]] || gsutil cp "$GCS_BUCKET/cohere/test.parquet" test.parquet
log "inputs staged ($(du -sh . | cut -f1))"

log "installing pyarrow..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-pip >/dev/null
pip3 install --break-system-packages -q pyarrow numpy
python3 - <<'EOF'
import pyarrow.parquet as pq
import numpy as np, struct, glob

# 1) id -> concat position (shards in glob order = the tree build order).
shards = sorted(glob.glob('train-*-of-10.parquet'))
assert len(shards) == 10, shards
pos = np.empty(10_000_000, dtype=np.int64)
off = 0
for s in shards:
    ids = pq.read_table(s, columns=['id']).column('id').to_pylist()
    pos[np.asarray(ids, dtype=np.int64)] = np.arange(off, off + len(ids))
    off += len(ids)
    print(f"[gt-fix] {s}: ids mapped (cum {off})", flush=True)
assert off == 10_000_000
# Sanity: ids must be a dense permutation of 0..N-1.
assert pos.min() >= 0 and len(np.unique(pos)) == 10_000_000

# 2) Map GT neighbors ids -> positions, write GTMM k=100. (Keep per-shard
#    offsets for the spot-check below.)
OFFS = []
gt = pq.read_table('neighbors.parquet')
qid = gt.column('id').to_pylist()
nb = gt.column('neighbors_id').to_pylist()
K = 100
n = len(nb)
with open('cohere_10m_gt.gtmm', 'wb') as f:
    f.write(struct.pack('<III', 0x4D4D5447, n, K))
    f.write(struct.pack('<B', 0))  # metric byte (l2sq)
    for row in nb:
        top = np.asarray(row[:K], dtype=np.int64)
        mapped = pos[top]
        f.write(mapped.astype('<u4').tobytes())
        f.write(np.zeros(K, dtype='<f4').tobytes())
print(f"[gt-fix] wrote GTMM: {n} queries x {K}", flush=True)

# 3) Quick verification: spot-check 3 (query, gt-neighbor) pairs with exact
# dot products against the shard data (queries from test.parquet in id order).
def vec_at(p):
    import bisect
    s = bisect.bisect_right(OFFS, p) - 1
    r = p - OFFS[s]
    tbl = pq.read_table(shards[s], columns=['emb'])
    return np.asarray(tbl.column('emb')[r].as_py(), dtype=np.float32)
test = pq.read_table('test.parquet')
assert test.column('id').to_pylist() == qid
rng = np.random.default_rng(0)
for qi in (0, 1, 2):
    q = np.asarray(test.column('emb')[qi].as_py(), dtype=np.float32)
    nb0 = pos[np.asarray(nb[qi][0], dtype=np.int64)]
    v = vec_at(nb0)
    vrand = vec_at(int(rng.integers(0, 10_000_000)))
    print(f"[gt-fix] q{qi}: dot(q, mapped-top1)={float(q @ v):.4f} "
          f"vs random={float(q @ vrand):.4f}", flush=True)
EOF
gsutil cp "$SSD/data/cohere_10m_gt.gtmm" "$GCS_BUCKET/cohere/cohere_10m_gt.gtmm"
log "=== gt-fix complete ==="
echo done | gsutil cp - "$GCS_BUCKET/cohere/.gt_fix_done"
poweroff || true
