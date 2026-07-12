#!/usr/bin/env bash
# Fetches SIFT-1M from the Texmex corpus and converts to Sextant's .fbin format.
#
# Usage:
#   ./scripts/download_datasets.sh sift1m [output_dir]
#
# If a local copy of the .fvecs/.ivecs files is already present under
# $LOCAL_SIFT_DIR (default: a sibling datasets directory), the script skips
# the download and just converts. This keeps CI fast when datasets are cached.
#
# Requirements: curl (or wget), tar, and a built `fvecs_to_fbin` tool.
set -euo pipefail

DATASET="${1:-}"
OUT_DIR="${2:-datasets}"
ENGINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FVECS_TO_FBIN="$ENGINE_ROOT/build/tools/fvecs_to_fbin"

TEXMEX_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
# A mirror over HTTPS is more reliable in CI; fall back to it if ftp fails.
TEXMEX_HTTP_MIRROR="http://corpus-texmex.irisa.fr/ftp/sift.tar.gz"

DEFAULT_LOCAL_DIR="../datasets/sift"
LOCAL_SIFT_DIR="${LOCAL_SIFT_DIR:-$DEFAULT_LOCAL_DIR}"

usage() {
    cat <<EOF
Usage: $0 <dataset> [output_dir]
Supported datasets:
  sift1m   SIFT-1M (1M base, 10k query, 10k×100 ground truth)
EOF
    exit 1
}

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }

convert_sift() {
    local src_dir="$1"
    mkdir -p "$OUT_DIR"
    "$FVECS_TO_FBIN" --input "$src_dir/sift_base.fvecs"      --output "$OUT_DIR/sift1m_base.fbin"
    "$FVECS_TO_FBIN" --input "$src_dir/sift_query.fvecs"     --output "$OUT_DIR/sift1m_query.fbin"
    "$FVECS_TO_FBIN" --input "$src_dir/sift_groundtruth.ivecs" --output "$OUT_DIR/sift1m_gt.gt" --gt
    echo "SIFT-1M ready in $OUT_DIR"
}

download_sift() {
    local work; work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    echo "Downloading SIFT-1M from Texmex..."
    if command -v curl >/dev/null 2>&1; then
        ( curl -fsSL "$TEXMEX_HTTP_MIRROR" -o "$work/sift.tar.gz" || \
          curl -fsSL "$TEXMEX_URL"        -o "$work/sift.tar.gz" )
    else
        need wget
        ( wget -q "$TEXMEX_HTTP_MIRROR" -O "$work/sift.tar.gz" || \
          wget -q "$TEXMEX_URL"        -O "$work/sift.tar.gz" )
    fi
    tar -xzf "$work/sift.tar.gz" -C "$work"
    # The archive extracts to a "sift/" subfolder.
    convert_sift "$work/sift"
}

fetch_sift1m() {
    if [[ -f "$LOCAL_SIFT_DIR/sift_base.fvecs" && \
          -f "$LOCAL_SIFT_DIR/sift_query.fvecs" && \
          -f "$LOCAL_SIFT_DIR/sift_groundtruth.ivecs" ]]; then
        echo "Found local SIFT copy at $LOCAL_SIFT_DIR; converting only."
        convert_sift "$LOCAL_SIFT_DIR"
    else
        download_sift
    fi
}

[[ -n "$DATASET" ]] || usage
[[ -x "$FVECS_TO_FBIN" ]] || { echo "build fvecs_to_fbin first (meson compile -C build)" >&2; exit 1; }

case "$DATASET" in
    sift1m|sift-1m|sift) fetch_sift1m ;;
    *) echo "unknown dataset: $DATASET" >&2; usage ;;
esac
