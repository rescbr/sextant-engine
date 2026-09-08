#include "batch_scheduler.hpp"

#include <algorithm>
#include <chrono>

namespace sextant::tree {

/// Field-wise predicate equality (Predicate has no operator==; the
/// result-cache key is query bytes + k + the full predicate list).
inline bool predicates_equal(const std::vector<Predicate>& a,
                             const std::vector<Predicate>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto& x = a[i];
        const auto& y = b[i];
        if (x.column != y.column || x.op != y.op ||
            x.geo_lng_column != y.geo_lng_column ||
            x.value != y.value || x.value2 != y.value2 ||
            x.value3 != y.value3 || x.value4 != y.value4 ||
            x.radius_km != y.radius_km || x.values != y.values) {
            return false;
        }
    }
    return true;
}

BatchScheduler::BatchScheduler(const IVFTreeIndex* index,
                               SearchConfig search_config, Config config)
    : index_(index),
      search_config_(std::move(search_config)),
      config_(config),
      window_max_ns_(config.window_max_us > 0
                         ? config.window_max_us * 1000ull
                         : 100'000'000ull),
      max_window_queries_(config.max_window_queries > 0
                              ? config.max_window_queries
                              : 4096),
      max_inflight_(std::max(
          1u, config.max_inflight_windows)) {
    sweeper_ = std::thread([this] { sweeper_loop_(); });
}

BatchScheduler::~BatchScheduler() { stop(); }

std::future<std::vector<Candidate>> BatchScheduler::submit(
        const float* query, uint32_t k,
        const std::vector<Predicate>* predicates) {
    Entry e;
    e.query.assign(query, query + index_->dim());
    e.k = k;
    if (predicates) e.predicates = *predicates;
    e.submitted = std::chrono::steady_clock::now();
    auto fut = e.promise.get_future();

    // Exact-repeat fast path: resolve from the result cache without
    // routing (saves the scan compute too, not just the I/O). The key
    // includes the predicate set — same query bytes under a different
    // filter is a different result.
    if (config_.result_cache_entries > 0) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            if (it->k == k && predicates_equal(it->predicates, e.predicates) &&
                it->query.size() == e.query.size() &&
                std::equal(it->query.begin(), it->query.end(),
                           e.query.begin())) {
                auto results = it->results;  // copy for this caller
                ++stats_.cache_hits;
                std::rotate(cache_.begin(), it, it + 1);  // LRU bump
                e.promise.set_value(std::move(results));
                return fut;  // resolved without queuing
            }
        }
    }

    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(std::move(e));
    }
    cv_.notify_one();
    return fut;
}

void BatchScheduler::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopped_ = true;
    }
    cv_.notify_all();
    if (sweeper_.joinable()) sweeper_.join();
    // Drain in-flight dispatches so pending futures resolve before the
    // scheduler dies (async futures block in their dtors anyway).
    std::lock_guard<std::mutex> flk(fut_mu_);
    for (auto& f : futs_)
        if (f.valid()) f.get();
    futs_.clear();
}

BatchScheduler::Stats BatchScheduler::stats() {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s = stats_;
    std::sort(s.delay_ns.begin(), s.delay_ns.end());
    stats_ = Stats{};
    return s;
}

void BatchScheduler::sweeper_loop_() {
    const auto idle =
        std::chrono::microseconds(config_.idle_close_us);
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
        if (queue_.empty()) {
            if (stopped_) return;
            cv_.wait(lk, [this] { return !queue_.empty() || stopped_; });
            continue;
        }
        // CAPACITY GATE — never dispatch beyond max_inflight_ concurrent
        // sweeps. While saturated, arrivals accumulate into the next
        // window; a completion (notified) frees a slot and the whole
        // accumulated window dispatches. Without this gate the idle
        // close below would dispatch every arrival as its own sweep —
        // unbounded concurrency, measured collapse at rate>=50 QPS.
        if (!stopped_ && inflight_ >= max_inflight_) {
            cv_.wait(lk, [this] {
                return inflight_ < max_inflight_ || stopped_;
            });
            continue;  // re-evaluate with the accumulated queue
        }
        // Close decision, classic rules: stop / size cap / oldest-entry
        // deadline / idle gap, whichever first. Every submit and every
        // dispatch completion notifies, so waits re-evaluate. With
        // max_inflight_ >= 2 this yields pipelined dispatch: a query
        // arriving while a sweep is in flight AND a slot is free closes
        // after the (short) idle gap and sweeps CONCURRENTLY — it
        // overlaps the in-flight sweep instead of waiting out the next
        // window (piggybacking via pipelining; shared-leaf reads dedup
        // in the ARC/page cache).
        bool close = stopped_ || queue_.size() >= max_window_queries_;
        if (!close) {
            const auto now = std::chrono::steady_clock::now();
            const auto deadline =
                queue_.front().submitted +
                std::chrono::nanoseconds(window_max_ns_);
            if (now >= deadline) {
                close = true;
            } else if (config_.idle_close_us > 0) {
                if (cv_.wait_until(lk, std::min(deadline, now + idle)) ==
                    std::cv_status::no_timeout)
                    continue;  // re-evaluate (arrival / completion)
                close = true;
            } else {
                if (cv_.wait_until(lk, deadline) ==
                    std::cv_status::no_timeout)
                    continue;
                close = true;
            }
        }
        (void)close;  // reaching here means close
        std::deque<Entry> window;
        window.swap(queue_);
        ++inflight_;
        stats_.peak_inflight =
            std::max(stats_.peak_inflight, inflight_);
        lk.unlock();
        {
            std::lock_guard<std::mutex> flk(fut_mu_);
            futs_.push_back(std::async(
                std::launch::async,
                [this, w = std::move(window)]() mutable {
                    dispatch_(std::move(w));
                    {
                        std::lock_guard<std::mutex> ilk(mu_);
                        --inflight_;
                    }
                    cv_.notify_all();
                }));
            // Reap finished dispatches (keeping every async future
            // alive — their dtors block until the task completes).
            for (auto it = futs_.begin(); it != futs_.end();) {
                if (it->wait_for(std::chrono::seconds(0)) ==
                    std::future_status::ready) {
                    it->get();
                    it = futs_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        lk.lock();
        // Loop: on stop, the now-empty queue exits on the next pass.
    }
}

void BatchScheduler::dispatch_(std::deque<Entry> window) {
    const uint32_t nq = static_cast<uint32_t>(window.size());
    const uint64_t dim = index_->dim();
    std::vector<float> queries;
    queries.reserve(static_cast<size_t>(nq) * dim);
    for (const auto& e : window)
        queries.insert(queries.end(), e.query.begin(), e.query.end());

    // search_batch takes one k per call: run with the window's max and
    // prefix-cut each query's (distance-sorted) results to its own k.
    // Per-query predicates ride along (empty = base config's).
    uint32_t k_eff = 0;
    for (const auto& e : window) k_eff = std::max(k_eff, e.k);
    std::vector<std::vector<Predicate>> win_preds;
    bool any_preds = false;
    for (const auto& e : window)
        any_preds = any_preds || !e.predicates.empty();
    if (any_preds) {
        win_preds.reserve(window.size());
        for (const auto& e : window) win_preds.push_back(e.predicates);
    }

    std::vector<std::vector<Candidate>> results;
    SearchConfig sc = search_config_;
    sc.search_threads = config_.search_threads;
    const auto t0 = std::chrono::steady_clock::now();
    index_->search_batch(
        queries.data(), nq, k_eff, sc, results,
        any_preds ? &win_preds : nullptr);
    const auto t1 = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lk(mu_);
    stats_.windows += 1;
    stats_.queries += nq;
    stats_.sweep_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
            .count();
    for (uint32_t i = 0; i < nq; ++i) {
        auto& e = window[i];
        if (stats_.delay_ns.size() < kMaxDelaySamples) {
            stats_.delay_ns.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t0 - e.submitted)
                    .count());
        }
        // Result-cache insert BEFORE moving the results into the promise
        // (capacity-bounded; stores the query's own k truncation).
        if (config_.result_cache_entries > 0) {
            if (cache_.size() >= config_.result_cache_entries)
                cache_.pop_back();
            CacheEntry ce;
            ce.query = e.query;
            ce.k = e.k;
            ce.predicates = e.predicates;
            ce.results = results[i];
            if (ce.results.size() > e.k) ce.results.resize(e.k);
            cache_.push_front(std::move(ce));
        }
        if (results[i].size() > e.k) results[i].resize(e.k);
        e.promise.set_value(std::move(results[i]));
    }
}

}  // namespace sextant::tree
