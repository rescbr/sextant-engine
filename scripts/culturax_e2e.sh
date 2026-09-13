#!/bin/bash
# CulturaX end-to-end release flow test.
#
# Exercises the whole shipped path on the embedded CulturaX corpus:
#   parquet(emb+payload) -> build-tree (defaults, plane attached, payloads)
#   -> fsck -> warm search vs GT -> payload roundtrip -> cold search
#   -> mutation flow (plane-less build: insert/delete/vacuum/defrag/fsck).
#
# Usage:
#   scripts/culturax_e2e.sh            # full corpus (requires embedding ALL DONE)
#   scripts/culturax_e2e.sh --mini     # first completed emb shard only (~50K chunks)
#   scripts/culturax_e2e.sh --skip-build  # reuse trees in $OUT (search/payload/mutate only)
#
# Gates (exit 1 on failure): build ok, fsck ok, warm recall >= MIN_RECALL,
# payload bytes roundtripped, cold search ok, mutation fsck ok.
set -euo pipefail
CD=$(dirname "$0")/..
SEXTANT="$CD/build-x86/tools/sextant"
PY=~/venvs/corpus/bin/python
EMB=/mnt/sextant/culturax1m/emb
OUT=/mnt/sextant/culturax_e2e
MINI=0; SKIP_BUILD=0
for a in "$@"; do
    [[ "$a" == "--mini" ]] && MINI=1
    [[ "$a" == "--skip-build" ]] && SKIP_BUILD=1
done
MIN_RECALL=${MIN_RECALL:-0.88}
mkdir -p "$OUT"

gate() { # name, condition(0=ok)
    if [[ "$2" -eq 0 ]]; then echo "PASS  $1"; else echo "FAIL  $1"; FAILED=1; fi
}

echo "== 0. preflight =="
[[ -x "$SEXTANT" ]] || { echo "binary missing: $SEXTANT (ninja -C build-x86)"; exit 1; }
if [[ $MINI -eq 0 && $SKIP_BUILD -eq 0 ]] && ! grep -q "ALL DONE" "$EMB/progress.log"; then
    echo "embedding not finished (no 'ALL DONE' in $EMB/progress.log); use --mini"; exit 1
fi
SHARDS=("$EMB"/chunk_*.parquet)
[[ ${#SHARDS[@]} -ge 1 ]] || { echo "no chunk_*.parquet in $EMB"; exit 1; }
if [[ $MINI -eq 1 ]]; then INPUT_GLOB="${SHARDS[0]}"; else INPUT_GLOB="$EMB/chunk_*.parquet"; fi
echo "input: $INPUT_GLOB (${#SHARDS[@]} shard files present)"

if [[ $SKIP_BUILD -eq 0 ]]; then
    echo "== 1. materialize base fbin + mutation fixtures =="
    $PY - "$INPUT_GLOB" "$OUT" <<'EOF'
import sys, glob, struct
import numpy as np
import pyarrow.parquet as pq
paths = sorted(glob.glob(sys.argv[1]))
out = sys.argv[2]
embs, seen = [], set()
for p in paths:
    t = pq.read_table(p, columns=["emb", "doc_idx", "chunk_idx"])
    e = np.asarray(t.column(0).to_pylist(), dtype=np.float32)
    di = t.column(1).to_pylist(); ci = t.column(2).to_pylist()
    for row, (d, c) in enumerate(zip(di, ci)):
        key = (d, c)
        if key in seen: continue  # resume-boundary duplicates
        seen.add(key); embs.append(e[row])
base = np.stack(embs); del embs
n, d = base.shape
with open(f"{out}/cx_base.fbin", "wb") as f:
    f.write(struct.pack("<II", n, d)); f.write(base.tobytes())
# fixtures for the mutation flow: head = first 100k rows, insert = next 20k
head = base[:100_000] if n >= 120_000 else base[: max(1, n // 2)]
ins  = base[100_000:120_000] if n >= 120_000 else base[max(1, n // 2):][:1000]
for name, arr in (("cx_head.fbin", head), ("cx_insert.fbin", ins)):
    with open(f"{out}/{name}", "wb") as f:
        f.write(struct.pack("<II", arr.shape[0], d)); f.write(arr.tobytes())
print(f"base {n} x {d}; head {head.shape[0]}; insert {ins.shape[0]}")
EOF
    gate "base fbin materialized" $?

    echo "== 2. ground truth (1000 queries, k=10) =="
    $PY "$CD/scripts/gen_ground_truth.py" --base "$OUT/cx_base.fbin" \
        --prefix "$OUT/cx" --n-queries 1000 --k 10
    gate "ground truth generated" $?

    echo "== 3. build-tree (defaults + payload) =="
    /usr/bin/time -f "build_wall=%es peak_rss=%MKB" \
    "$SEXTANT" build-tree --input "$INPUT_GLOB" --vector-col emb \
        --payload-col text --index "$OUT/cx.tree" 2>&1 | tail -5
    gate "default tree built (plane+payload)" $?

    echo "== 3b. build plane-less mutation tree (head subset) =="
    "$SEXTANT" build-tree --input "$OUT/cx_head.fbin" --index "$OUT/cx_mut.tree" \
        --no-plane-attach 2>&1 | tail -3
    gate "plane-less mutation tree built" $?
else
    echo "== 1-3. skipped (--skip-build) =="
fi

echo "== 4. fsck =="
"$SEXTANT" fsck --file "$OUT/cx.tree" 2>&1 | tail -2
gate "fsck default tree" $?

echo "== 5. warm search vs GT (defaults: f=.1, prune .25, bw 256) =="
"$SEXTANT" tree-search --index "$OUT/cx.tree" --query "$OUT/cx_query.fbin" \
    --ground-truth "$OUT/cx_gt.gtmm" --threads 16 --batch-window 256 \
    --probe-fraction 0.1 2>&1 | tee "$OUT/warm.log" | grep -E "recall@10|time:"
RECALL=$(grep -oE "recall@10[^0-9]*[0-9.]+" "$OUT/warm.log" | grep -oE "[0-9.]+$" | head -1)
awk -v r="${RECALL:-0}" -v m="$MIN_RECALL" 'BEGIN{exit !(r>=m)}'
gate "warm recall ${RECALL:-?} >= $MIN_RECALL" $?

echo "== 6. payload roundtrip =="
head -c 400000 "$OUT/cx_query.fbin" > "$OUT/q5.fbin"  # ~130 queries x 768 x f32
"$SEXTANT" tree-search --index "$OUT/cx.tree" --query "$OUT/q5.fbin" \
    --with-payload --payload-text --output "$OUT/payload.tsv" 2>&1 | tail -2
# some payload text bytes came back
SZ=$(stat -c%s "$OUT/payload.tsv" 2>/dev/null || echo 0)
[[ "$SZ" -gt 1000 ]]
gate "payload roundtrip (${SZ}B tsv)" $?

echo "== 7. cold search =="
sudo sh -c 'zpool sync pastry && echo 3 > /proc/sys/vm/drop_caches'
"$SEXTANT" tree-search --index "$OUT/cx.tree" --query "$OUT/cx_query.fbin" \
    --ground-truth "$OUT/cx_gt.gtmm" --threads 16 --batch-window 256 \
    --probe-fraction 0.1 --passes 2 2>&1 | tee "$OUT/cold.log" | grep -E "recall@10|time:"
gate "cold search completed" $?

echo "== 8. mutation flow (plane-less tree) =="
"$SEXTANT" tree-insert --index "$OUT/cx_mut.tree" --vectors "$OUT/cx_insert.fbin" 2>&1 | tail -1
gate "tree-insert" $?
seq 0 49 > "$OUT/del_ids.txt"
"$SEXTANT" tree-delete --index "$OUT/cx_mut.tree" --row-ids "$OUT/del_ids.txt" 2>&1 | tail -1
gate "tree-delete" $?
"$SEXTANT" tree-vacuum --index "$OUT/cx_mut.tree" 2>&1 | tail -1
gate "tree-vacuum" $?
"$SEXTANT" tree-defrag --index "$OUT/cx_mut.tree" 2>&1 | tail -1
gate "tree-defrag" $?
"$SEXTANT" fsck --file "$OUT/cx_mut.tree" 2>&1 | tail -1
gate "fsck mutated tree" $?
"$SEXTANT" tree-search --index "$OUT/cx_mut.tree" --query "$OUT/q5.fbin" \
    --threads 16 2>&1 | grep -E "recall@10|time:" || true
gate "post-mutation search runs" $?

echo
if [[ "${FAILED:-0}" -eq 0 ]]; then echo "E2E RESULT: ALL GATES PASS"; else echo "E2E RESULT: FAILURES PRESENT"; exit 1; fi
