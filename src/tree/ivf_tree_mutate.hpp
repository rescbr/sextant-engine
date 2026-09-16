#pragma once

// ===========================================================================
// INTERNAL mutation surface for IVFTreeIndex — v3 seed, NOT public API.
// ===========================================================================
//
// The v1 public C++ surface is the read-only pipeline:
//     build_streaming_pca / open / search / search_batch /
//     fetch_payload / fetch_vector / stats / fsck  (+ debug/diagnostics)
//
// Everything below mutates an already-built index and is internal:
//     insert_batch / delete_batch / vacuum / defrag / live_count /
//     attach_plane / attach_plane_from  (+ the InsertPoint, VacuumResult,
//     VacuumConfig, DefragResult, DefragConfig helper types)
//
// For now the declarations still live inside class IVFTreeIndex
// (src/tree/ivf_tree_index.hpp, in the block delimited by
// `// --- INTERNAL: mutation (v3 seed; not public API) ---`) because the
// ~1800-line implementation (src/tree/ivf_tree_mutate.cpp) shares private
// state and private helpers (leaf_table_, coder_, commit_mutable_, remap_,
// split_leaf_, ...) with the build/search implementation units — a mixin or
// free-function split would require either rewriting every member access in
// the implementation or widening the public/private boundary. When v3 lands,
// this header is the seam: move the declarations here first (mixin or
// IVFTreeIndex-taking free functions), then relocate the implementation.
//
// WHO MAY INCLUDE THIS HEADER:
//   - src/tree/ivf_tree_mutate.cpp (implementation)
//   - CLI mutation commands (tools/sextant_cli.cpp:
//     tree-insert / tree-delete / tree-vacuum / tree-defrag / plane-attach)
//   - tests that mutate (test_tree_insert_delete, test_vacuum_defrag,
//     test_quantizer_families, test_leaf_cache, test_tree_filter) and
//     scripts/plane_selftest.cpp
// Public consumers (C API, search/build consumers) must NOT include it.

#include "tree/ivf_tree_index.hpp"
