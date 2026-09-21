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
// not in v1; see the flat-era design register, archived.)
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
        /// Max concurrently-sweeping windows (piggybacking via
        /// pipelining). 1 = serialized windows (classic close rules
        /// only); >=2 = a query arriving while sweeps are in flight AND
        /// a slot is free dispatches IMMEDIATELY — it overlaps the
        /// in-flight sweeps instead of waiting for the next window
        /// (shared-leaf reads dedup in the ARC/page cache). At capacity,
        /// arrivals accumulate into windows as before, so the
        /// latency-mode/throughput-mode blend emerges from the cap.
        /// 0 is treated as 1. Default 2.
        uint32_t max_inflight_windows = 2;
    };

    struct Stats {
        uint64_t windows = 0;
        uint64_t queries = 0;
        uint64_t cache_hits = 0;
        uint64_t sweep_ns = 0;  // total search_batch wall inside dispatch
        uint32_t peak_inflight = 0;  // max concurrent sweeps observed
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
    /// Optional per-request settings (all default to the scheduler's
    /// base config): predicates (empty vector = unfiltered query),
    /// max_delay_us (latency tolerance — see the deadline-class docs
    /// at the full overload), probe_fraction (recall depth — a
    /// bytes-priced quality knob). Thread-safe. Returns an INVALID future
    /// when the scheduler has been stopped (check future::valid()).

    /// Full overload with per-request latency tolerance (deadline
    /// class).
    /// `max_delay_us` is the QUEUEING delay the request accepts: the
    /// scheduler must DISPATCH it within that long of submission.
    /// Semantics:
    ///   - It bounds queueing only — the sweep itself has a physics
    ///     floor (~60 ms fanout-1 on NAND-backed indexes) no deadline
    ///     can compress; end-to-end ≈ max_delay_us + sweep when the
    ///     deadline is met. Values below the sweep floor are all
    ///     effectively "Immediate".
    ///   - Large values = "Batchable": the request coalesces into
    ///     windows (~50x fewer bytes/query). Small values = "Immediate":
    ///     it jumps the queue (deadline-sorted), dispatches on the next
    ///     free slot, and carries its own uncoalesced reads. The knob is
    ///     literally a per-request bytes price.
    ///   - Deadline-driven dispatches split off only the DUE prefix —
    ///     patient entries behind an urgent one keep coalescing.
    ///   - 0 = the scheduler's window_max_us (default class).
    ///   - Beyond the uncoalesced service capacity (urgent offered rate
    ///     above the immediate-class request-rate ceiling — ~21 QPS
    ///     aggregate on the reference box: ~214 MB/query, ~65 ms
    ///     singleton sweeps, measured directly as
    ///     `tree-search --batch-window 1` → uncoalesced_capacity_qps)
    ///     deadlines become arbitration, not guarantees — enforce with
    ///     admission control at the edge. Exposed as metrics:
    ///     read_stream_gbps (per-stream bandwidth ingredient) and
    ///     uncoalesced_capacity_qps (singleton-window measurement).
    std::future<std::vector<Candidate>> submit(
            const float* query, uint32_t k,
            const std::vector<Predicate>* predicates = nullptr,
            uint64_t max_delay_us = 0, float probe_fraction = 0.0f);

    /// Drain the queue, stop the sweeper, resolve pending futures.
    void stop();

    /// Delta snapshot (per-query delay samples sorted ascending).
    Stats stats();

private:
    struct Entry {
        std::vector<float> query;
        uint32_t k;
        std::vector<Predicate> predicates;  // empty = base config's
        float probe_fraction = 0.0f;        // 0 = base config's (RECALL
        // knob: deeper queries probe more leaves, priced in bytes; the
        // unique-leaf sweep still reads shared leaves once)
        std::chrono::steady_clock::time_point submitted;
        std::chrono::steady_clock::time_point deadline;  // submitted +
        // max_delay_us (default: window_max_us). The queue is kept
        // deadline-sorted; the close rule takes the front's deadline.
        std::promise<std::vector<Candidate>> promise;
    };

    void sweeper_loop_();
    void dispatch_(std::deque<Entry> window);

    const IVFTreeIndex* index_;
    const SearchConfig search_config_;
    const Config config_;
    const uint64_t window_max_ns_;
    const uint32_t max_window_queries_;

    mutable     std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Entry> queue_;
    bool stopped_ = false;
    std::thread sweeper_;
    // Pipelined dispatch: in-flight search_batch calls + their futures
    // (async futures block in their destructor — kept until reaped).
    uint32_t inflight_ = 0;
    const uint32_t max_inflight_;
    std::mutex fut_mu_;
    std::deque<std::future<void>> futs_;

    // Result cache: LRU over (query bytes, k) → results.
    struct CacheEntry {
        std::vector<float> query;
        uint32_t k;
        std::vector<Predicate> predicates;
        float probe_fraction = 0.0f;
        std::vector<Candidate> results;
    };
    std::deque<CacheEntry> cache_;  // front = most recent
    Stats stats_;
    static constexpr size_t kMaxDelaySamples = 1u << 20;
};

}  // namespace sextant::tree
