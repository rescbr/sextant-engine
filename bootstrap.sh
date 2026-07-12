#!/usr/bin/env bash
# bootstrap.sh — pre-build setup for sextant-engine.
#
# Initializes the nsync and NumKong git submodules. spdlog and googletest
# are fetched automatically by Meson via WrapDB (see subprojects/*.wrap).
#
# CTPL and cmdline are committed directly in third_party/ — not fetched here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '[bootstrap] %s\n' "$*"; }
fail() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

check_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "'$1' not found in PATH. $2"
}

log "checking prerequisites..."
check_cmd git    "Install from https://git-scm.com/"
check_cmd cmake  "Required for building nsync."
check_cmd meson  "Install with 'pip install meson' or your package manager."
check_cmd clang++ "Required compiler. Install Xcode (macOS) or LLVM (Linux)."

setup_submodule() {
    local name="$1" repo="$2" path="$3"
    if [ -d "$path/.git" ]; then
        log "updating $name submodule..."
        git -C "$SCRIPT_DIR" submodule update --init --recursive -- "$path" 2>/dev/null || true
    elif git -C "$SCRIPT_DIR" submodule status -- "$path" 2>/dev/null | grep -q .; then
        log "initializing $name submodule..."
        git -C "$SCRIPT_DIR" submodule update --init --recursive -- "$path"
    else
        log "adding $name submodule..."
        git -C "$SCRIPT_DIR" submodule add --force "$repo" "$path"
        git -C "$SCRIPT_DIR" submodule update --init --recursive -- "$path"
    fi
}

setup_submodule "nsync"   "https://github.com/google/nsync.git"         "third_party/nsync"
setup_submodule "numkong" "https://github.com/ashvardanian/NumKong.git"  "third_party/numkong"

log "done. next step: meson setup build"
log "  (spdlog and googletest are fetched automatically from Meson WrapDB)"
