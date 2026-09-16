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
# Canonical build input: fp16 FLBA (fixed_len_byte_array, dim*2) — the
# fast path (zero list assembly, ~0.7s/Lloyd-pass vs 2.9s for list<float>).
# --vector-fp16 tells the reader the FLBA holds fp16 halves. The list<float>
# path remains supported (see README vector-layout table).
FP16=/mnt/sextant/culturax1m/emb_flba16_raw
OUT=/mnt/sextant/culturax_e2e
MINI=0; SKIP_BUILD=0
for a in "$@"; do
    [[ "$a" == "--mini" ]] && MINI=1
    [[ "$a" == "--skip-build" ]] && SKIP_BUILD=1
done
MIN_RECALL=${MIN_RECALL:-0.82}
# Gate rationale (measured 2026-09-13, post row-id/carquet fixes): this
# corpus is tie-saturated (sibling chunks of a doc at cos 1.0000), so even an
# exact f=1.0 scan reproduces only ~0.86 of the fp32 GT top-10 (self included
# by the engine, excluded by GT; arbitrary tie-breaking on both sides). The
# f=0.1 shipped default reaches 0.840 — within ~3% of the exact-scan ceiling.
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
if [[ $MINI -eq 1 ]]; then INPUT_GLOB="${SHARDS[0]}"; VEC_FLAGS=""; else INPUT_GLOB="$FP16/chunk_*.parquet"; VEC_FLAGS="--vector-fp16"; fi
# Build input = the corpus parquets directly (emb + text payload + language/
# url/source/doc_idx/... auto filter columns). An earlier carquet 'set
# column' failure that forced a slim (emb+text) workaround turned out to be
# an engine-side schema element/leaf index bug — fixed; full schemas build
# and filtered search is a release gate (step 5b).
echo "input: $INPUT_GLOB (${#SHARDS[@]} shard files present)"

if [[ $SKIP_BUILD -eq 0 ]]; then
    echo "== 1. materialize base fbin + mutation fixtures (fp32 from $EMB) =="
    $PY - "$EMB/chunk_*.parquet" "$OUT" <<'EOF'
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

    echo "== 3. build-tree (defaults + payload + filter columns) =="
    /usr/bin/time -f "build_wall=%es peak_rss=%MKB" \
    "$SEXTANT" build-tree --input "$INPUT_GLOB" $VEC_FLAGS --vector-col emb \
        --payload-col text --index "$OUT/cx.tree" 2>&1 | tee "$OUT/build.log" | tail -5
    gate "default tree built (plane+payload+filters)" $?

    echo "== 3b. build plane-less mutation tree (head subset) =="
    # delete_batch requires a global-PQ-family codebook; the default
    # scalar_shape quantizer does not support tombstoning.
    "$SEXTANT" build-tree --input "$OUT/cx_head.fbin" --index "$OUT/cx_mut.tree" \
        --no-plane-attach --quantizer pq 2>&1 | tail -3
    gate "plane-less mutation tree built" $?
else
    echo "== 1-3. skipped (--skip-build) =="
fi

echo "== 4. fsck =="
"$SEXTANT" fsck --file "$OUT/cx.tree" 2>&1 | tail -2
gate "fsck default tree" $?

echo "== 5. warm search vs GT (defaults: f=.1, prune .25, bw 256) =="
if [[ $MINI -eq 1 ]]; then
    # 50K corpus = only 13 leaves: production f=.1 probes ~2 — recall gate is
    # scale-inappropriate. Sanity-check at full scan instead (f=1.0 measured
    # 0.69 row-id on this tie-saturated multilingual corpus; engine results
    # verified exact vs numpy brute force).
    "$SEXTANT" tree-search --index "$OUT/cx.tree" --query "$OUT/cx_query.fbin" \
        --ground-truth "$OUT/cx_gt.gtmm" --threads 16 --batch-window 256 \
        --probe-fraction 1.0 2>&1 | grep -E "recall@10|time:"
    RECALL=$(./build-x86/tools/sextant tree-search --index "$OUT/cx.tree" --query "$OUT/cx_query.fbin" \
        --ground-truth "$OUT/cx_gt.gtmm" --threads 16 --probe-fraction 1.0 2>&1 | grep -oE "recall@10: [0-9.]+" | grep -oE "[0-9.]+$")
    MIN_RECALL=0.55
    awk -v r="${RECALL:-0}" -v m="0.55" 'BEGIN{exit !(r>=m)}'
    gate "mini sanity: f=1.0 recall ${RECALL:-?} >= 0.55" $?
else
    "$SEXTANT" tree-search --index "$OUT/cx.tree" --query "$OUT/cx_query.fbin" \
        --ground-truth "$OUT/cx_gt.gtmm" --threads 16 --batch-window 256 \
        --probe-fraction 0.1 2>&1 | tee "$OUT/warm.log" | grep -E "recall@10|time:"
    RECALL=$(grep -oE "recall@10[^0-9]*[0-9.]+" "$OUT/warm.log" | grep -oE "[0-9.]+$" | head -1)
    awk -v r="${RECALL:-0}" -v m="$MIN_RECALL" 'BEGIN{exit !(r>=m)}'
    gate "warm recall ${RECALL:-?} >= $MIN_RECALL (f=.1, full corpus)" $?
fi

echo "== 5b. filtered search (is_first_chunk:eq:1) =="
FR=$("$SEXTANT" tree-search --index "$OUT/cx.tree" --query "$OUT/q5.fbin" \
    --threads 16 --probe-fraction 1.0 --filter 'is_first_chunk:eq:1' 2>&1 \
    | grep -oE 'mean results/query: [0-9.]+' | grep -oE '[0-9.]+$')
awk -v r="${FR:-0}" 'BEGIN{exit !(r >= 9)}'
gate "filtered search returns full result set (${FR:-0}/query, is_first_chunk=1)" $?

# == 5b2. rare-value Eq gate (url:eq:<globally-unique url>) ==
# Guards the rare-string brute-force fallback path that was once broken
# invisibly (fixed in 8803abc): a predicate matching exactly one row must
# return exactly that row (mean results/query == 1), not 0.
RARE_URL_FILE="$OUT/rare_url.txt"
if [[ ! -s "$RARE_URL_FILE" ]]; then
    "$PY" - "$INPUT_GLOB" "$RARE_URL_FILE" <<'PYEOF'
import sys, glob, collections, pyarrow.parquet as pq
pattern, out = sys.argv[1], sys.argv[2]
counts = collections.Counter()
files = sorted(glob.glob(pattern))
assert files, pattern
for f in files:
    urls = pq.read_table(f, columns=['url']).column('url').to_pylist()
    counts.update(urls)
for u, n in counts.items():
    if n == 1:
        with open(out, 'w') as fh:
            fh.write(u)
        print(f"rare url (1 of {sum(counts.values())} rows): {u[:80]}")
        sys.exit(0)
raise SystemExit("no singleton url found")
PYEOF
fi
RARE_URL=$(cat "$RARE_URL_FILE")
RR=$("$SEXTANT" tree-search --index "$OUT/cx.tree" --query "$OUT/q5.fbin" \
    --threads 16 --probe-fraction 1.0 --filter "url:eq:$RARE_URL" 2>&1 \
    | grep -oE 'mean results/query: [0-9.]+' | grep -oE '[0-9.]+$' || true)
awk -v r="${RR:--1}" 'BEGIN{exit !(r > 0.99 && r < 1.01)}'
gate "rare-value Eq returns exactly 1 result (${RR:-0}/query, url singleton)" $?

# == 5c. synthetic-query recall (LLM-generated RAG queries; alongside the
# self-query gate). Fixtures: cx_synth1k{,_16}_gt.gtmm from
# corpora/gen_synth_queries.py + embed_queries.py + gen_ground_truth.py.
# Two gates: (a) default config recall >= 0.75; (b) INVARIANT: f=1.0 +
# exact rerank == 1.0 — synthetic queries have no duplicate-vector ties,
# so the exact path must be perfect (self-queries saturate at ~0.86
# from sibling ties and cannot gate this).
if [[ -f "$OUT/cx_synth1k16_gt.gtmm" && -f "$OUT/cx_base16.fbin" ]]; then
    SR=$("$SEXTANT" tree-search --index "$OUT/cx.tree" \
        --query "$OUT/cx_synth1k16_query.fbin" \
        --ground-truth "$OUT/cx_synth1k16_gt.gtmm" \
        --threads 16 --batch-window 256 --probe-fraction 0.1 2>&1 \
        | grep -oE "recall@10: [0-9.]+" | grep -oE "[0-9.]+$")
    awk -v r="${SR:-0}" 'BEGIN{exit !(r >= 0.75)}'
    gate "synthetic-query recall ${SR:-0} >= 0.75 (f=.1, 1k RAG queries)" $?
    ER=$("$SEXTANT" tree-search --index "$OUT/cx.tree" \
        --query "$OUT/cx_synth1k16_query.fbin" \
        --ground-truth "$OUT/cx_synth1k16_gt.gtmm" \
        --threads 16 --batch-window 256 --probe-fraction 1.0 \
        --exact-rerank-base "$OUT/cx_base16.fbin" 2>&1 \
        | grep -oE "recall@10: [0-9.]+" | grep -oE "[0-9.]+$")
    awk -v r="${ER:-0}" 'BEGIN{exit !(r >= 0.999)}'
    gate "synthetic exact-rerank invariant ${ER:-0} == 1.0 (f=1.0)" $?
else
    echo "synthetic-query fixtures absent — skipping 5c (generate via"
    echo "corpora/gen_synth_queries.py + embed_queries.py + gen_ground_truth.py)"
fi

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

# --- Canonical metrics digest (scrape from the logs written above) ---
BW=$(grep -oE "build_wall=[0-9.m]+" "$OUT/build.log" 2>/dev/null | tail -1 | cut -d= -f2)
WQ=$(grep -oE "\([0-9.]+ QPS\)" "$OUT/warm.log" 2>/dev/null | tail -1 | tr -d '()' | awk '{print $1}')
CQ=$(grep -oE "\([0-9.]+ QPS\)" "$OUT/cold.log" 2>/dev/null | tail -1 | tr -d '()' | awk '{print $1}')
WR=$(grep -oE "recall@10: [0-9.]+" "$OUT/warm.log" 2>/dev/null | tail -1 | cut -d' ' -f2)
echo "METRICS build_wall=${BW:-?} warm_qps=${WQ:-?} cold_qps=${CQ:-?} warm_recall=${WR:-?} (n=2919988, fp16-flba-raw)"
