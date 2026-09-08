// BatchScheduler: windowing policy on top of IVFTreeIndex::search_batch.
//
// The scheduler is the serving layer for subtree-major batch search.
// Callers submit single queries and get futures; a sweeper thread closes
// windows adaptively and dispatches each window as ONE search_batch call
// (every unique probed leaf read once, page-ordered). The engine coupling
// is deliberate: the close heuristics need index shape (leaves per query,
// unique-leaf count) and the arrival stream, and the batch primitive is
// engine-internal.
//
// Window close policy (never a fixed latency tax):
//   - idle-close:   no arrival for idle_close_us while the queue is
//                   nonempty → close now. A lonely query dispatches
//                   immediately at fanout-1 cost (byte-identical to the
//                   query-major path — no penalty, no benefit).
//   - size-close:   queue reached max_window_queries → close (bounds
//                   memory and per-window latency at high arrival rate).
//   - deadline:     the oldest queued query waited window_max_us → close
//                   regardless. Latency contract: P99 ≈ window_max_us +
//                   sweep time.
// While a sweep is in flight, arriving queries accumulate and dispatch
// when it completes — a busy sweep IS the window. (Mid-sweep joining,
// where a late query rides the in-flight sweep's reads for leaves not
// yet scanned, requires a mutable sweep work list and is deliberately
// not in v1; see docs/design_decisions.md.)
//
// Result cache (result_cache_entries > 0): exact-repeat queries (the
// zipf head) are resolved from a small LRU keyed on the full query
// vector + k, before routing. Orthogonal to the sweep; saves both the
// I/O and the scan compute for repeats.
//
// One SearchConfig per scheduler (construction time): batch coalescing
// assumes homogeneous queries. No env vars, no boolean-mode flags —
// behavior is driven entirely by the typed Config fields.

#pragma once

#include "tree/ivf_tree_index.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <mutex>
#include <vector>

namespace sextant::tree {

class BatchScheduler {
public:
    struct Config {
        /// Deadline bound: the oldest queued query never waits longer
        /// than this for its window to close. 0 = default (100 ms).
        uint64_t window_max_us = 100'000;
        /// Idle close: with queries queued, no new arrival for this long
        /// closes the window (lonely queries dispatch immediately).
        uint64_t idle_close_us = 200;
        /// Size close: cap on queries per window (bounds memory + sweep
        /// latency at high arrival rates). 0 = default (4096).
        uint32_t max_window_queries = 4096;
        /// Exact-repeat result cache capacity (entries). 0 = off.
        uint32_t result_cache_entries = 0;
        /// search_threads forwarded to search_batch.
        uint32_t search_threads = 1;
    };

    struct Stats {
        uint64_t windows = 0;
        uint64_t queries = 0;
        uint64_t cache_hits = 0;
        uint64_t sweep_ns = 0;  // total search_batch wall inside dispatch
        /// Per-query submit→dispatch-delay samples (ns), capped at
        /// max_delay_samples; sorted on snapshot for percentiles.
        std::vector<uint64_t> delay_ns;
    };

    /// `index` must outlive the scheduler. Starts the sweeper thread.
    /// (Config has no aggregate default-arg constructor quirk: pass one
    /// explicitly — zero fields keep their documented defaults.)
    BatchScheduler(const IVFTreeIndex* index, SearchConfig search_config,
                   Config config);
    ~BatchScheduler();

    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;

    /// Submit one query (dim() floats, copied). The returned future
    /// resolves with the query's top-k after its window's sweep.
    /// Optional per-query predicates (null = the scheduler's base
    /// config predicates; empty vector = unfiltered query).
    /// Thread-safe.
    std::future<std::vector<Candidate>> submit(
            const float* query, uint32_t k,
            const std::vector<Predicate>* predicates = nullptr);

    /// Drain the queue, stop the sweeper, resolve pending futures.
    void stop();

    /// Delta snapshot (per-query delay samples sorted ascending).
    Stats stats();

private:
    struct Entry {
        std::vector<float> query;
        uint32_t k;
        std::vector<Predicate> predicates;  // empty = base config's
        std::promise<std::vector<Candidate>> promise;
        std::chrono::steady_clock::time_point submitted;
    };

    void sweeper_loop_();
    void dispatch_(std::deque<Entry> window);

    const IVFTreeIndex* index_;
    const SearchConfig search_config_;
    const Config config_;
    const uint64_t window_max_ns_;
    const uint32_t max_window_queries_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Entry> queue_;
    bool stopped_ = false;
    std::thread sweeper_;

    // Result cache: LRU over (query bytes, k) → results.
    struct CacheEntry {
        std::vector<float> query;
        uint32_t k;
        std::vector<Predicate> predicates;
        std::vector<Candidate> results;
    };
    std::deque<CacheEntry> cache_;  // front = most recent
    Stats stats_;
    static constexpr size_t kMaxDelaySamples = 1u << 20;
};

}  // namespace sextant::tree
