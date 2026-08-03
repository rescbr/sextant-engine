#!/usr/bin/env bash
set -euo pipefail

# Closure multiplier sweep: find the sweet spot between QPS and recall.

BASE="${1:?base .fbin required}"
QUERY="${2:?query .fbin required}"
GT="${3:?ground-truth .gt required}"
METRIC="${4:-ip}"
SEXTANT="${SEXTANT:-sextant}"

MULTS="0.05 0.10 0.15 0.20 0.25 0.35 0.50"

# Read N from fbin header.
N=$(python3 -c "
import struct
with open('$BASE','rb') as f: print(struct.unpack('<I',f.read(4))[0])
")
echo "# N=$N, closure_mult,n_leaves,replication,qps_np8,recall_np8,qps_np32,recall_np32"

for M in $MULTS; do
    TREE="/tmp/sweep_${M}.tree"
    BUILD_OUT=$("$SEXTANT" build-tree-pca --input "$BASE" --index "$TREE" \
        --metric "$METRIC" --leaf-capacity 5000 --pca-dims 32 \
        --max-lloyd-passes 8 --threads 8 --closure-mult "$M" 2>&1)
    N_LEAVES=$(echo "$BUILD_OUT" | grep -oP 'n_leaves=\K\d+' | head -1)
    if [ -z "$N_LEAVES" ]; then echo "$M,BUILD_FAILED,,,,," ; continue; fi
    REPL=$(python3 -c "print(f'{$N_LEAVES / ($N / 5000):.2f}')")
    
    R8=$("$SEXTANT" tree-search --index "$TREE" --query "$QUERY" \
        --ground-truth "$GT" --topk 10 --n-probe 8 --n-probe-ln 8 --threads 8 2>&1)
    QPS8=$(echo "$R8" | grep -oP '[\d.]+(?= QPS)' || echo "0")
    REC8=$(echo "$R8" | grep 'recall@' | awk '{print $2}' || echo "0")
    
    R32=$("$SEXTANT" tree-search --index "$TREE" --query "$QUERY" \
        --ground-truth "$GT" --topk 10 --n-probe 32 --n-probe-ln 8 --threads 8 2>&1)
    QPS32=$(echo "$R32" | grep -oP '[\d.]+(?= QPS)' || echo "0")
    REC32=$(echo "$R32" | grep 'recall@' | awk '{print $2}' || echo "0")
    
    echo "$M,$N_LEAVES,$REPL,$QPS8,$REC8,$QPS32,$REC32"
    rm -f "$TREE"
done
