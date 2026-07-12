#!/usr/bin/env python3
"""Analyze a samply profile (Firefox Profiler JSON format) with .syms.json sidecar.

Resolves raw frame addresses to function names using the symbol sidecar
produced by `samply record --unstable-presymbolicate`, then computes
self-time and inclusive-time breakdowns per function.

Usage:
    python3 scripts/analyze_profile.py profile.json[.gz]

Requires the profile to have been recorded with --unstable-presymbolicate
so that a profile.json.syms.json sidecar exists alongside it.

Example:
    samply record --unstable-presymbolicate -r 10000 -s -o profile.json.gz -- ./build/release/duckdb -f script.sql
    python3 scripts/analyze_profile.py profile.json.gz
"""
from __future__ import annotations

import bisect
import collections
import gzip
import json
import sys
from pathlib import Path


def _open_maybe_gzip(path: str):
	if path.endswith(".gz"):
		return gzip.open(path, "rt")
	return open(path, "rt")


def load_profile(profile_path: str) -> dict:
	with _open_maybe_gzip(profile_path) as f:
		return json.load(f)


def load_symbols(profile_path: str) -> dict | None:
	"""Load the .syms.json sidecar if it exists."""
	for p in [
		profile_path + ".syms.json",
		profile_path[:-3] + ".syms.json" if profile_path.endswith(".gz") else None,
	]:
		if p is None:
			continue
		try:
			with open(p) as f:
				return json.load(f)
		except FileNotFoundError:
			continue
	return None


def _norm_debug_id(s: str | None) -> str:
	"""Normalize a breakpad/debug id for matching by lowercasing and
	stripping non-hex separators. samply's debug_id is a UUID like
	'2c49e22a-3ca1-380d-bd16-83d289ccee38' (with hyphens); the profile's
	breakpadId is the same digits with an extra age suffix and no hyphens,
	e.g. '2C49E22A3CA1380DBD1683D289CCEE380'. Comparing only the hex digits
	makes both forms match."""
	if not s:
		return ""
	return "".join(c for c in s.lower() if c in "0123456789abcdef")


def build_syms_resolver(syms: dict, libs: list[dict]):
	"""Build a library-scoped symbol resolver from a samply .syms.json sidecar.

	### samply .syms.json format (authoritative: samply/src/shared/symbol_precog.rs)

	Each entry in `data[]` describes one library and contains:
	  - `debug_name`: library name, matches profile `libs[].debugName`.
	  - `debug_id`: UUID string; matches the hex digits of `libs[].breakpadId`.
	  - `symbol_table`: list of `{rva, size, symbol}` where `symbol` is an
	    index into the shared top-level `string_table`.
	  - `known_addresses`: list of `[rva, sym_table_idx]` pairs, sorted by
	    rva. **The second element is an index into `symbol_table`, NOT into
	    `string_table`** (a common source of bugs). These rvae are
	    library-relative offsets in the same coordinate space as the profile's
	    `frameTable.address`.

	### Resolution (matches samply's PrecogLibraySymbolMap::lookup_sync)
	  1. Select the library's data entry by matching `debug_name` (and
	     `debug_id`/breakpadId when available) against the frame's library.
	  2. Exact-match the frame address against that library's
	     `known_addresses` (binary search). On hit, the symbol is
	     `symbol_table[idx].symbol -> string_table`.
	  3. Fallback: find the `symbol_table` entry whose `[rva, rva+size)`
	     range contains the address. This catches addresses samply didn't
	     pre-resolve. Out-of-range addresses resolve to a raw hex name.

	All lookups are scoped to a single library — addresses from one library
	are never matched against another library's symbols.
	"""
	st = syms["string_table"]

	def _name_for_symbol(sym_entry) -> str:
		idx = sym_entry.get("symbol", -1)
		return st[idx] if 0 <= idx < len(st) else f"sym_{idx}"

	# Per-library resolver state.
	# lib_key -> (known_addrs: sorted list[int], known_sym_idx: list[int],
	#             sym_rvas: sorted list[int], sym_entries: list, debug_name)
	lib_resolvers: dict[str, tuple] = {}
	syms_debug_ids: list[tuple[str, tuple]] = []
	for entry in syms.get("data", []):
		debug_name = entry.get("debug_name", "?")
		known = entry.get("known_addresses", []) or []
		# known_addresses is sorted by rva in the sidecar; sort defensively.
		known_sorted = sorted(known, key=lambda p: p[0])
		known_addrs = [p[0] for p in known_sorted]
		known_sym = [p[1] for p in known_sorted]
		sym_table = entry.get("symbol_table", []) or []
		sym_sorted = sorted(sym_table, key=lambda s: s.get("rva", 0))
		sym_rvas = [s.get("rva", 0) for s in sym_sorted]
		state = (known_addrs, known_sym, sym_rvas, sym_sorted, debug_name)
		# Key by debug_name (reliable) and store the normalized debug_id for
		# prefix matching against the profile's breakpadId.
		lib_resolvers[debug_name] = state
		syms_debug_ids.append((_norm_debug_id(entry.get("debug_id")), state))

	# Map each profile lib index -> resolver key.
	# Prefer matching by debug_id (prefix match, since breakpadId appends an
	# age nibble to the debug_id), fall back to debugName/name.
	lib_key_for: list[str | None] = []
	for lib in libs:
		bid = _norm_debug_id(lib.get("breakpadId"))
		key = None
		for did, state in syms_debug_ids:
			if did and bid and (bid.startswith(did) or did.startswith(bid)):
				key = state[4]  # debug_name
				break
		if key is None:
			cand = lib.get("debugName") or lib.get("name")
			key = cand if cand in lib_resolvers else None
		lib_key_for.append(key)

	def resolve(addr: int, lib_idx: int | None) -> tuple[str, str]:
		key = lib_key_for[lib_idx] if lib_idx is not None and lib_idx < len(lib_key_for) else None
		if key is None:
			return (f"0x{addr:x}", "unknown")
		known_addrs, known_sym, sym_rvas, sym_sorted, debug_name = lib_resolvers[key]

		# 1. Exact match in known_addresses (samply's primary path).
		i = bisect.bisect_left(known_addrs, addr)
		if i < len(known_addrs) and known_addrs[i] == addr:
			sym_idx = known_sym[i]
			if 0 <= sym_idx < len(sym_sorted):
				return (_name_for_symbol(sym_sorted[sym_idx]), debug_name)

		# 2. Fallback: symbol_table range match.
		j = bisect.bisect_right(sym_rvas, addr) - 1
		if j >= 0:
			s = sym_sorted[j]
			rva, size = s.get("rva", 0), s.get("size", 0) or 0
			if size and rva <= addr < rva + size:
				return (_name_for_symbol(s), debug_name)

		return (f"0x{addr:x}", debug_name)

	return resolve


def analyze_thread(thread: dict, resolve) -> tuple[collections.Counter, collections.Counter, int]:
	frame_table = thread.get("frameTable", {})
	stack_table = thread.get("stackTable", {})
	samples = thread.get("samples", {})

	sample_stacks = samples.get("stack", [])
	sample_weights = samples.get("weight", [])
	frame_addresses = frame_table.get("address", [])
	frame_funcs = frame_table.get("func", [])
	stack_frame = stack_table.get("frame", [])
	stack_prefix = stack_table.get("prefix", [])

	if not sample_stacks or not frame_addresses:
		return collections.Counter(), collections.Counter(), 0

	# Determine each frame's library index via:
	#   frameTable.func -> funcTable.resource -> resourceTable.lib -> libs[]
	func_table = thread.get("funcTable", {})
	res_table = thread.get("resourceTable", {})
	func_resources = func_table.get("resource", [])
	res_libs = res_table.get("lib", [])

	frame_lib_idx: list[int | None] = []
	for fi in range(len(frame_addresses)):
		func = frame_funcs[fi] if fi < len(frame_funcs) else None
		lib_idx = None
		if func is not None and 0 <= func < len(func_resources):
			res = func_resources[func]
			if res is not None and 0 <= res < len(res_libs):
				lib_idx = res_libs[res]
		frame_lib_idx.append(lib_idx)

	# Pre-resolve all frames (library-scoped).
	frame_names = []
	frame_libs = []
	for fi, addr in enumerate(frame_addresses):
		name, lib = resolve(addr, frame_lib_idx[fi])
		frame_names.append(name)
		frame_libs.append(lib)

	self_time: collections.Counter = collections.Counter()
	incl_time: collections.Counter = collections.Counter()

	for i, sidx in enumerate(sample_stacks):
		w = abs(sample_weights[i]) if i < len(sample_weights) else 1.0

		if sidx is not None and 0 <= sidx < len(stack_frame):
			fidx = stack_frame[sidx]
			if 0 <= fidx < len(frame_names):
				self_time[(frame_names[fidx], frame_libs[fidx])] += w

		cur = sidx
		visited: set[int] = set()
		while cur is not None and 0 <= cur < len(stack_frame) and cur not in visited:
			visited.add(cur)
			fidx = stack_frame[cur]
			if 0 <= fidx < len(frame_names):
				incl_time[(frame_names[fidx], frame_libs[fidx])] += w
			cur = stack_prefix[cur] if cur < len(stack_prefix) else None

	total = sum(self_time.values())
	return self_time, incl_time, total


KEYWORDS = [
	"aisaq", "vindex", "quantizer", "pq_", "beam", "prune", "insert",
	"construct", "encode", "search", "block_store", "buffermanager",
	"pin", "block", "createindex", "finalize", "train", "rawdistance",
	"codedistance", "lut", "robust", "connect", "alloc", "write",
]


def is_vindex_func(name: str) -> bool:
	lower = name.lower()
	return any(kw in lower for kw in KEYWORDS)


def main():
	if len(sys.argv) < 2:
		print(f"Usage: {sys.argv[0]} <profile.json[.gz]>")
		sys.exit(1)

	profile_path = sys.argv[1]
	profile = load_profile(profile_path)
	syms = load_symbols(profile_path)

	if syms is None:
		print("ERROR: No .syms.json sidecar found.", file=sys.stderr)
		print("Re-record with: samply record --unstable-presymbolicate ...", file=sys.stderr)
		sys.exit(1)

	resolve = build_syms_resolver(syms, profile.get("libs", []))

	threads = profile.get("threads", [])

	# Analyze ALL threads
	results = []
	for ti, thread in enumerate(threads):
		st, it, tot = analyze_thread(thread, resolve)
		vindex_w = sum(w for (n, _), w in st.items() if is_vindex_func(n))
		results.append((ti, thread, st, it, tot, vindex_w))

	# Sort by total samples descending
	results.sort(key=lambda r: r[4], reverse=True)

	for rank, (ti, thread, self_time, incl_time, total, vindex_self) in enumerate(results):
		if total == 0:
			continue

		thread_name = thread.get("name", "?")
		vindex_pct = 100.0 * vindex_self / total if total > 0 else 0

		print(f"\n{'='*120}")
		print(f"Thread {ti}: name={thread_name}, samples={total:.0f}, vindex_self={vindex_pct:.1f}%")
		print(f"{'='*120}")

		if vindex_pct < 1.0 and rank > 0:
			continue

		# Top self time — no truncation
		print(f"\n--- TOP 30 SELF TIME ---")
		for (name, lib), w in self_time.most_common(30):
			pct = 100.0 * w / total
			lib_short = Path(lib).name if lib else ""
			print(f"  {pct:>6.1f}%  [{lib_short}]  {name}")

		# Top inclusive time for vindex functions
		print(f"\n--- TOP 30 INCLUSIVE TIME (vindex + DuckDB hot paths) ---")
		filtered = {k: v for k, v in incl_time.items() if is_vindex_func(k[0])}
		if not filtered:
			filtered = dict(incl_time)
		for (name, lib), w in sorted(filtered.items(), key=lambda x: -x[1])[:30]:
			pct = 100.0 * w / total
			lib_short = Path(lib).name if lib else ""
			print(f"  {pct:>6.1f}%  [{lib_short}]  {name}")

	# Summary across all threads
	print(f"\n{'='*120}")
	print(f"SUMMARY")
	print(f"{'='*120}")
	for ti, thread, self_time, incl_time, total, vindex_self in results:
		if total == 0:
			continue
		name = thread.get("name", "?")
		pct = 100.0 * vindex_self / total if total > 0 else 0
		print(f"  Thread {ti} ({name}): {total:.0f} samples, vindex={pct:.1f}%")


if __name__ == "__main__":
	main()
