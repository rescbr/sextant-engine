#pragma once

/// @file scan_pool.hpp
/// Process-wide worker pool for search-time parallelism.
///
/// Replaces per-call `std::async` workers in the search paths (within-query
/// parallel scan, batch sweep route/sweep/finalize workers). std::async
/// spawns a fresh thread per task (libstdc++ does not pool), which costs
/// tens of microseconds per spawn — material for per-query parallel_scan at
/// high QPS and for small search_batch calls. A long-lived pool also keeps
/// worker TLS (SearchScratch arenas) warm.
///
/// Contract: ONLY external (caller) threads submit tasks and wait on the
/// futures. Pool threads never submit-and-block, so pool-exhaustion
/// deadlock is structurally impossible. Concurrent callers (DuckDB scan
/// threads, CLI query workers) share the pool; their tasks interleave.
///
/// Lifetime: intentionally leaked (created on first use, never destroyed).
/// Joining at static-destruction time would race exit ordering with mmap'd
/// index handles and the engine's own static state; an idle pool costs
/// nothing but parked threads.

#include <ctpl/ctpl_stl_tls.h>

#include <cstdint>

namespace sextant::tree {

/// Trivial per-thread state marker (workers keep their own SearchScratch
/// via function-local thread_local arenas).
struct ScanWorkerTag {};

/// The process-wide scan pool (lazily created, never destroyed).
ctpl::thread_pool_tls<ScanWorkerTag>& scan_pool();

/// Current pool thread count.
uint32_t scan_pool_threads();

/// Resize the pool. n == 0 selects hardware_concurrency. Returns the new
/// count. Safe to call before/while tasks are in flight (idle threads are
/// stopped; busy threads stop after their current task). Never shrinks to
/// zero.
uint32_t scan_pool_set_threads(uint32_t n);

}  // namespace sextant::tree
