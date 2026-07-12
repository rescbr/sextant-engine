#!/usr/bin/env bash
# Profile search with samply.
#
# Runs the benchmark tool under samply to capture where search time is spent
# (beam_search, LUT distance, cache lookup, pread, rerank).
#
# Usage:
#   ./scripts/profile_search.sh [index_prefix] [extra bench args...]
#
# Defaults to SIFT-1M at /tmp/sift_remeasure if present. Writes to profiles/.
# Requires: samply, a built sextant_bench tool.
set -euo pipefail

ENGINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ENGINE_ROOT/build/tools/sextant_bench"
PROFILES_DIR="$ENGINE_ROOT/profiles"
mkdir -p "$PROFILES_DIR"

INDEX="${1:-/tmp/sift_remeasure}"
shift $(( $# < 1 ? $# : 1 ))

if [[ ! -x "$BENCH" ]]; then
    echo "error: build the bench tool first (meson compile -C build)" >&2
    exit 1
fi

# Default dataset paths (override via args after index).
QUERY="$ENGINE_ROOT/datasets/sift1m_query.fbin"
BASE="$ENGINE_ROOT/datasets/sift1m_base.fbin"
GT="$ENGINE_ROOT/datasets/sift1m_gt.gt"

if [[ ! -d "$INDEX.manifest" && ! -f "$INDEX.manifest" ]]; then
    if [[ ! -f "$INDEX.graph" ]]; then
        echo "error: index not found: $INDEX (.graph missing)" >&2
        exit 1
    fi
fi

echo "[profile] searching '$INDEX' under samply..."
echo "[profile] profile will be saved to $PROFILES_DIR/"

export SAMPPLY_OUT_DIR="$PROFILES_DIR"
samply record --save-only -o "$PROFILES_DIR/search_profile.samply" \
    "$BENCH" \
        --index "$INDEX" \
        --queries "$QUERY" \
        --base-data "$BASE" \
        --ground-truth "$GT" \
        --limit 1000 \
        "$@"

echo "[profile] done. Open with: samply load $PROFILES_DIR/search_profile.samply"
