#!/bin/bash
# Canonical cold smoke — cohere10m, 1000q, 16t, batch 256, prune .25
# (canonical-table-v2 protocol). Reference bands (box noise ±10%):
#   b1g  f=.1 : 286 cold QPS @ recall 0.9212
#   legacy f=.2 : 244 cold QPS @ recall 0.9254
# Usage: scripts/canonical_smoke.sh [--warm]
set -euo pipefail
CD=$(dirname "$0")/..
SEXTANT="$CD/build-x86/tools/sextant"
IDX=/mnt/sextant/cohere10m
Q=$IDX/cohere_10m_query_norm.fbin
GT=$IDX/cohere_10m_gt.gtmm

run() { # name tree extra-flags...
    local name=$1 tree=$2; shift 2
    if [[ ! "${1:-}" == "--warm" ]]; then
        sudo sh -c 'zpool sync pastry && echo 3 > /proc/sys/vm/drop_caches'
    fi
    echo "=== $name ==="
    "$SEXTANT" tree-search --index "$tree" --query "$Q" --ground-truth "$GT" \
        --threads 16 --batch-window 256 "$@" 2>&1 \
        | grep -E "^time:|recall@10"
}

MODE="${1:-}"
if [[ "$MODE" == "--warm" ]]; then WARM=1; fi

run "b1g f=.1 prune.25" "$IDX/cohere_10m_shape_b1g.tree" \
    --probe-fraction 0.1 --plane-pre-prune 0.25 ${WARM:+}
run "legacy f=.2 (manifest gap)" "$IDX/cohere_10m_shape.norm.tree" \
    --no-plane --probe-fraction 0.2
