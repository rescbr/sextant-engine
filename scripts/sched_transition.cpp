// Scheduler regime-transition harness (2026-09-08).
//
// Measures the BatchScheduler's latency/throughput trade across arrival
// rates: open-loop submits at fixed inter-arrival spacing for each rate
// in a ladder, a collector thread resolves futures and records
// submit→result latency. Below the batching threshold the idle-close
// dispatches lonely queries immediately (latency ≈ sweep time at
// fanout 1); above it windows coalesce (latency ≈ window cadence,
// throughput bounded by the sweep capacity). The knee between the two
// regimes is the deliverable.
//
// Env knobs (scripts convention — engine code stays env-free):
//   SCHED_IDLE_US   idle-close gap (default 200)
//   SCHED_MAX_US    window deadline bound (default 100000)
//   SCHED_THREADS   search_threads inside search_batch (default 8)
//   SCHED_RATES     comma-separated rate ladder (QPS)
//   SCHED_DUR_US    per-rate measurement duration in seconds x1e6 (default 6s)
//
// usage: sched_transition <tree> <queries.fbin> [warmup-rate]
//
// Build (in-tree): ninja -C build-x86 scripts/sched_transition

#include "tree/ivf_tree_index.hpp"
#include "tree/batch_scheduler.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/null_sink.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Submitted {
    Clock::time_point t;
    std::future<std::vector<sextant::Candidate>> fut;
};

double percentile_ns(std::vector<uint64_t>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t idx = std::min(v.size() - 1,
                                static_cast<size_t>(p * (v.size() - 1)));
    return static_cast<double>(v[idx]) / 1e6;  // ms
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <tree> <queries.fbin> [warmup-rate]\n",
                     argv[0]);
        return 1;
    }
    // Silence engine info logs — the harness owns stdout.
    spdlog::set_default_logger(
        spdlog::create<spdlog::sinks::null_sink_st>("null"));

    // Queries (.fbin: [n u32][dim u32][n*dim f32]).
    uint32_t qn = 0, qdim = 0;
    {
        std::ifstream f(argv[2], std::ios::binary);
        if (!f.read(reinterpret_cast<char*>(&qn), 4) ||
            !f.read(reinterpret_cast<char*>(&qdim), 4)) {
            std::fprintf(stderr, "qhdr\n");
            return 1;
        }
    }
    std::vector<float> queries(static_cast<size_t>(qn) * qdim);
    {
        std::ifstream f(argv[2], std::ios::binary);
        f.seekg(8);
        if (f.read(reinterpret_cast<char*>(queries.data()),
                   static_cast<std::streamsize>(
                       queries.size() * sizeof(float)))) {
        } else {
            std::fprintf(stderr, "qread\n");
            return 1;
        }
    }

    auto idx = sextant::tree::IVFTreeIndex::open(argv[1]);
    if (qdim != idx->dim()) {
        std::fprintf(stderr, "query dim %u != index dim %u\n", qdim,
                     idx->dim());
        return 1;
    }

    const uint64_t idle_us =
        std::getenv("SCHED_IDLE_US") ? std::atoll(std::getenv("SCHED_IDLE_US")) : 200;
    const uint64_t max_us =
        std::getenv("SCHED_MAX_US") ? std::atoll(std::getenv("SCHED_MAX_US")) : 100000;
    const uint32_t threads =
        std::getenv("SCHED_THREADS") ? std::atoi(std::getenv("SCHED_THREADS")) : 8;
    const double dur_s =
        std::getenv("SCHED_DUR_US")
            ? std::atoll(std::getenv("SCHED_DUR_US")) / 1e6 : 6.0;
    std::vector<double> rates = {1, 2, 5, 10, 20, 50, 100, 200, 400};
    if (const char* r = std::getenv("SCHED_RATES")) {
        rates.clear();
        std::string s(r);
        size_t pos = 0;
        while ((pos = s.find(',')) != std::string::npos) {
            rates.push_back(std::atof(s.substr(0, pos).c_str()));
            s.erase(0, pos + 1);
        }
        rates.push_back(std::atof(s.c_str()));
    }

    // Warmup (optional rate point, discarded): page cache + thread pools.
    if (argc > 3) {
        const double wr = std::atof(argv[3]);
        sextant::SearchConfig wsc;
        wsc.k = 10;
        wsc.search_threads = threads;
        sextant::tree::BatchScheduler warm(
            idx.get(), wsc,
            {max_us, idle_us, 4096, 0, threads});
        const auto t0 = Clock::now();
        for (uint32_t i = 0;
             std::chrono::duration<double>(Clock::now() - t0).count() < 3.0;
             ++i) {
            warm.submit(&queries[(i % qn) * qdim], 10).wait();
            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<uint64_t>(1e6 / wr)));
        }
    }

    std::cout << "tree=" << argv[1] << " queries=" << qn
              << " threads=" << threads << " idle_us=" << idle_us
              << " max_us=" << max_us << " dur_s=" << dur_s << "\n";
    std::cout << "rate_qps achieved_qps submitted completed p50_ms p95_ms"
                 " p99_ms max_ms windows avg_win_q fanout\n";

    for (double rate : rates) {
        sextant::SearchConfig sc;
        sc.k = 10;
        sc.search_threads = threads;
        sextant::tree::BatchScheduler sched(idx.get(), sc,
                                            {max_us, idle_us, 4096, 0,
                                             threads});

        std::mutex qmu;
        std::deque<Submitted> pending;   // generator → collector
        std::atomic<bool> gen_done{false};
        std::vector<uint64_t> latencies_ns;
        std::mutex lat_mu;
        std::atomic<uint64_t> submitted{0}, completed{0};
        // Completions counted only inside the measurement window — the
        // post-deadline drain must not inflate achieved QPS under
        // overload (the backlog is latency, not throughput).
        std::atomic<bool> deadline_passed{false};

        // Collector: resolve futures, record submit→result latency.
        std::thread collector([&] {
            while (true) {
                Submitted s;
                {
                    std::unique_lock<std::mutex> lk(qmu);
                    if (pending.empty()) {
                        if (gen_done.load()) break;
                        lk.unlock();
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(50));
                        continue;
                    }
                    s = std::move(pending.front());
                    pending.pop_front();
                }
                (void)s.fut.get();
                const auto lat = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(Clock::now() - s.t)
                    .count();
                {
                    std::lock_guard<std::mutex> lk(lat_mu);
                    latencies_ns.push_back(lat);
                }
                if (!deadline_passed.load(std::memory_order_relaxed))
                    completed.fetch_add(1, std::memory_order_relaxed);
            }
        });

        // Generator: open-loop, fixed inter-arrival spacing.
        const auto t0 = Clock::now();
        const auto deadline =
            t0 + std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::duration<double>(dur_s));
        // Deadline watcher flips the flag; the collector stops counting.
        std::thread watch([&] {
            std::this_thread::sleep_until(deadline);
            deadline_passed.store(true, std::memory_order_relaxed);
        });
        watch.detach();
        uint32_t qi = 0;
        const auto step = std::chrono::nanoseconds(
            static_cast<uint64_t>(1e9 / rate));
        while (std::chrono::duration<double>(Clock::now() - t0).count() <
               dur_s) {
            const auto target =
                t0 + static_cast<std::chrono::nanoseconds>(
                         step * static_cast<int64_t>(submitted.load()));
            std::this_thread::sleep_until(target);
            {
                std::lock_guard<std::mutex> lk(qmu);
                pending.push_back({Clock::now(),
                                   sched.submit(
                                       &queries[(qi % qn) * qdim], 10)});
            }
            submitted.fetch_add(1, std::memory_order_relaxed);
            ++qi;
        }
        gen_done.store(true);
        collector.join();

        const auto st = sched.stats();
        const auto bst = idx->batch_stats().snapshot_and_reset();
        (void)idx->search_stats().snapshot_and_reset();
        const double achieved =
            static_cast<double>(completed.load()) / dur_s;
        std::ostringstream fanout;
        fanout << std::fixed << std::setprecision(1)
               << (st.windows ? static_cast<double>(bst.leaf_scans) /
                                     std::max<uint64_t>(1, bst.leaves_unique)
                              : 0.0);
        std::cout << std::fixed << std::setprecision(1) << rate << " "
                  << achieved << " " << submitted.load() << " "
                  << completed.load() << " " << std::setprecision(2)
                  << percentile_ns(latencies_ns, 0.50) << " "
                  << percentile_ns(latencies_ns, 0.95) << " "
                  << percentile_ns(latencies_ns, 0.99) << " "
                  << percentile_ns(latencies_ns, 1.00) << " "
                  << st.windows << " "
                  << (st.windows
                          ? static_cast<double>(st.queries) / st.windows
                          : 0.0)
                  << " " << fanout.str() << "\n";
        std::cout.flush();
        // Let the scheduler drain before destructing.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return 0;
}
