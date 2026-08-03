#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?base .fbin required}"
QUERY="${2:?query .fbin required}"
GT="${3:?ground-truth .gt required}"
METRIC="${4:-ip}"
SEXTANT="${SEXTANT:-sextant}"

MULTS="0.05 0.10 0.15 0.20 0.25 0.35 0.50"
N=$(python3 -c "import struct; f=open('$BASE','rb'); print(struct.unpack('<I',f.read(4))[0])")

echo "# N=$N, closure_mult,n_leaves,replication,qps_np8,recall_np8,qps_np32,recall_np32"
flush_log() { :; }  # no-op

for M in $MULTS; do
    TREE="/tmp/sweep_${M}.tree"
    BUILD_OUT=$("$SEXTANT" build-tree-pca --input "$BASE" --index "$TREE" \
        --metric "$METRIC" --leaf-capacity 5000 --pca-dims 32 \
        --max-lloyd-passes 8 --threads 8 --closure-mult "$M" 2>&1)
    N_LEAVES=$(echo "$BUILD_OUT" | sed -n 's/.*n_leaves=\([0-9]*\).*/\1/p' | head -1)
    if [ -z "$N_LEAVES" ]; then echo "$M,BUILD_FAILED,,,,," ; continue; fi
    REPL=$(python3 -c "print(f'{$N_LEAVES / ($N / 5000):.2f}')")

    R8=$("$SEXTANT" tree-search --index "$TREE" --query "$QUERY" \
        --ground-truth "$GT" --topk 10 --n-probe 8 --n-probe-ln 8 --threads 8 2>&1)
    QPS8=$(echo "$R8" | sed -n 's/.*(\([0-9.]*\) QPS).*/\1/p')
    REC8=$(echo "$R8" | sed -n 's/.*recall@[0-9]*: \([0-9.]*\).*/\1/p')

    R32=$("$SEXTANT" tree-search --index "$TREE" --query "$QUERY" \
        --ground-truth "$GT" --topk 10 --n-probe 32 --n-probe-ln 8 --threads 8 2>&1)
    QPS32=$(echo "$R32" | sed -n 's/.*(\([0-9.]*\) QPS).*/\1/p')
    REC32=$(echo "$R32" | sed -n 's/.*recall@[0-9]*: \([0-9.]*\).*/\1/p')

    echo "$M,$N_LEAVES,$REPL,$QPS8,$REC8,$QPS32,$REC32"
    rm -f "$TREE"
done
echo "# DONE"
