#!/usr/bin/env bash
# Profile the index build with samply.
#
# Produces a samply profile that can be opened in a browser for flamegraph
# analysis. The build is CPU-bound, so this targets the construct/encode passes.
#
# Usage:
#   ./scripts/profile_build.sh [input.fbin] [index_prefix] [extra cli args...]
#
# Defaults to SIFT-1M if present. Writes profile to profiles/.
# Requires: samply (cargo install samply), a built sextant CLI.
set -euo pipefail

ENGINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ENGINE_ROOT/build/tools/sextant"
PROFILES_DIR="$ENGINE_ROOT/profiles"
mkdir -p "$PROFILES_DIR"

INPUT="${1:-$ENGINE_ROOT/datasets/sift1m_base.fbin}"
INDEX="${2:-/tmp/sextant_build_profile}"
shift $(( $# < 2 ? $# : 2 ))

if [[ ! -x "$CLI" ]]; then
    echo "error: build the CLI first (meson compile -C build)" >&2
    exit 1
fi
if [[ ! -f "$INPUT" ]]; then
    echo "error: input not found: $INPUT" >&2
    exit 1
fi

rm -f "${INDEX}".*

echo "[profile] building '$INPUT' → '$INDEX' under samply..."
echo "[profile] profile will be saved to $PROFILES_DIR/"

# samply runs the command as a subprocess and captures the profile.
# On macOS it uses the mach timer; on Linux it uses perf_event_open.
export SAMPPLY_OUT_DIR="$PROFILES_DIR"
samply record --save-only -o "$PROFILES_DIR/build_profile.samply" \
    "$CLI" build --input "$INPUT" --index "$INDEX" --log-level warn "$@"

echo "[profile] done. Open with: samply load $PROFILES_DIR/build_profile.samply"
