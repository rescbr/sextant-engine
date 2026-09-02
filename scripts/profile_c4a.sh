#!/usr/bin/env bash
# Working perf recipe on GCP c4a (Ubuntu 24.04, kernel 6.8) — several traps:
#   1. kernel.perf_event_paranoid=4 by default: sysctl -w ...=1 (as root).
#   2. `perf stat` SEGFAULTS on this kernel/tools combo — use perf record only.
#   3. `perf record --call-graph dwarf` captures but perf report fails with
#      "failed to process sample" (dwarf unwind broken in 6.17 userspace).
#      Use FLAT profiles (-g omitted) — kernels are monolithic inlined lambdas
#      anyway, caller context adds nothing.
#   4. perf record dies with SIGPIPE (rc=141, unfinalized data headers) when
#      its stdout isn't consumed — always pipe: `perf record ... | tail -1`.
# Usage: profile_c4a.sh <tree> <query.fbin> <gt.gtmm> <tag> [extra tree-search args]
# Always --threads=8: we WANT contention in the picture.
set -euo pipefail
TREE=$1; Q=$2; GT=$3; TAG=$4; shift 4
SXT=${SXT:-./sextant}
sysctl -w kernel.perf_event_paranoid=1 >/dev/null
ARGS=(--index="$TREE" --query="$Q" --ground-truth="$GT" --topk=10
      --fastscan-w=300 --threads=8 --n-probe=8 --n-probe-ln=8 "$@")
# warmup (page cache, thread pool)
$SXT tree-search "${ARGS[@]}" >/dev/null 2>&1 || true
perf record -F 999 -o "$TAG.data" -- $SXT tree-search "${ARGS[@]}" 2>&1 | tail -1
perf report -i "$TAG.data" --stdio --no-children --percent-limit 0.5
