#pragma once

/// @file phase_timer.hpp
/// Wall/CPU/RSS/source accounting for build phases and search windows.
///
/// PhaseTimer captures process-wide snapshots (steady_clock, getrusage
/// RUSAGE_SELF = thread-sum CPU, /proc/self/statm current RSS, and the
/// VectorSource's monotonic wait/bytes counters). MetricsCollector turns a
/// start/stop pair into a PhaseMetrics record, stores it, and emits it to
/// the sink.
///
/// Contract: phases are SEQUENTIAL and started/stopped by one thread —
/// no locking. CPU accounting is process-wide, so overlapping phases would
/// double-count; the build pipeline and windowed search emission both
/// respect this. Per-query CPU under concurrent search is NOT attributable
/// via rusage (other threads' cycles leak in) — search windows get CPU
/// attribution at the caller/harness boundary, engine-side counters are
/// traffic counters only.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

#include "sextant/metrics.hpp"
#include "sextant/vector_source.hpp"

namespace sextant::metrics {

class MetricsCollector;

/// An in-flight phase measurement. Obtained from MetricsCollector::start();
/// completed by MetricsCollector::stop(). Copyable, cheap (a few PODs).
class PhaseTimer {
public:
    struct Snap {
        std::chrono::steady_clock::time_point t{};
        double cpu_seconds = 0;
        double source_wait_seconds = 0;
        uint64_t wait_count = 0;
        uint64_t bytes_read = 0;
        uint64_t rss_bytes = 0;
    };

private:
    friend class MetricsCollector;
    const char* phase_ = nullptr;
    Snap start_;
};

/// Accumulates phase records, emits them to a sink (optional), and produces
/// totals. One collector per build (or per search-window emitter).
class MetricsCollector {
public:
    MetricsCollector() = default;
    /// `source` may be null (no source-wait/bytes attribution).
    /// `sink` may be null (records still accumulate in records()).
    MetricsCollector(const VectorSource* source, MetricsSink* sink,
                     std::string source_label = {})
        : source_(source), sink_(sink), source_label_(std::move(source_label)) {}

    PhaseTimer start(const char* phase) {
        PhaseTimer t;
        t.phase_ = phase;
        t.start_ = snapshot();
        return t;
    }

    /// Completes the phase: computes deltas, stores + emits the record.
    /// Returns a reference into records() (stable until the next stop()).
    const PhaseMetrics& stop(PhaseTimer& t) {
        const PhaseTimer::Snap end = snapshot();
        PhaseMetrics m;
        m.phase = t.phase_ ? t.phase_ : "";
        m.source = source_label_;
        m.wall_seconds = std::chrono::duration<double>(end.t - t.start_.t).count();
        m.cpu_seconds = end.cpu_seconds - t.start_.cpu_seconds;
        m.source_wait_seconds =
            end.source_wait_seconds - t.start_.source_wait_seconds;
        m.wait_count = end.wait_count - t.start_.wait_count;
        m.bytes_read = end.bytes_read - t.start_.bytes_read;
        m.rss_bytes = std::max(end.rss_bytes, t.start_.rss_bytes);
        m.timestamp = now_seconds();
        records_.push_back(std::move(m));
        if (sink_) {
            try {
                sink_->emit(records_.back());
            } catch (...) {
                // Sinks must not throw; belt and suspenders — a monitoring
                // path must never take down a build.
            }
        }
        return records_.back();
    }

    const std::vector<PhaseMetrics>& records() const { return records_; }

    /// Sum record over all stored phases (rss = max, not sum).
    PhaseMetrics totals(const char* phase = "total") const {
        PhaseMetrics m;
        m.phase = phase;
        m.source = source_label_;
        m.timestamp = now_seconds();
        for (const PhaseMetrics& r : records_) {
            m.wall_seconds += r.wall_seconds;
            m.cpu_seconds += r.cpu_seconds;
            m.source_wait_seconds += r.source_wait_seconds;
            m.wait_count += r.wait_count;
            m.bytes_read += r.bytes_read;
            m.rss_bytes = std::max(m.rss_bytes, r.rss_bytes);
        }
        return m;
    }

    /// Process peak RSS (getrusage ru_maxrss, KiB on Linux) — monotone.
    static uint64_t peak_rss_bytes() {
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
        return static_cast<uint64_t>(ru.ru_maxrss);
#else
        return static_cast<uint64_t>(ru.ru_maxrss) * 1024ULL;
#endif
    }

private:
    static double now_seconds() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    static double rusage_cpu_seconds() {
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        return static_cast<double>(ru.ru_utime.tv_sec) +
               1e-6 * static_cast<double>(ru.ru_utime.tv_usec) +
               static_cast<double>(ru.ru_stime.tv_sec) +
               1e-6 * static_cast<double>(ru.ru_stime.tv_usec);
    }

    /// Current RSS via /proc/self/statm (second field, pages). Returns 0 if
    /// unavailable (non-Linux) — peak_rss_bytes() still works everywhere.
    static uint64_t current_rss_bytes() {
#if defined(__linux__)
        std::FILE* f = std::fopen("/proc/self/statm", "r");
        if (!f) return 0;
        unsigned long long total = 0, resident = 0;
        const int got = std::fscanf(f, "%llu %llu", &total, &resident);
        std::fclose(f);
        if (got != 2) return 0;
        const long page = sysconf(_SC_PAGESIZE);
        return resident * static_cast<unsigned long long>(page);
#else
        return 0;
#endif
    }

    PhaseTimer::Snap snapshot() {
        PhaseTimer::Snap s;
        s.t = std::chrono::steady_clock::now();
        s.cpu_seconds = rusage_cpu_seconds();
        s.rss_bytes = current_rss_bytes();
        if (source_) {
            s.source_wait_seconds = source_->wait_seconds();
            s.wait_count = source_->wait_count();
            s.bytes_read = source_->bytes_read();
        }
        return s;
    }

    const VectorSource* source_ = nullptr;
    MetricsSink* sink_ = nullptr;
    std::string source_label_;
    std::vector<PhaseMetrics> records_;
};

}  // namespace sextant::metrics
