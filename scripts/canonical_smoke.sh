#!/bin/bash
# Canonical cold smoke — cohere10m, 1000q, 16t, batch 256, prune .25
# (canonical-table-v3 protocol: current-defaults trees @ cfdfb03+, built
# 2026-09-13; v2 trees predate the defaults drift). Reference bands (box
# noise ±10%; cold on pastry/largefiles):
#   b1g  f=.1 : ~332 cold QPS @ recall 0.9174   (v2: 286 @ 0.9212)
#   legacy f=.2 : ~260 cold QPS @ recall 0.9202 (v2: 244 @ 0.9254)
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

run "b1g f=.1 prune.25" "$IDX/cohere_10m_b1g_v3.tree" \
    --probe-fraction 0.1 --plane-pre-prune 0.25 ${WARM:+}
run "legacy f=.2 (manifest gap)" "$IDX/cohere_10m_noplane_v3.tree" \
    --no-plane --probe-fraction 0.2
