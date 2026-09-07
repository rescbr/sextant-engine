#pragma once

/// @file metrics.hpp
/// Engine-owned build/search metrics: typed records + pluggable sinks.
///
/// Design contract (see docs/BENCHMARK_RULES.md + plan "build instrumentation"):
///  - Records are emitted ONE AT A TIME, with copy semantics, the moment they
///    exist. The engine holds no reference afterwards. Any push transport
///    (UDP, message queue, OTel) is therefore just a sink implementation —
///    none of those are shipped here by design.
///  - Metric names are stable, dotted, lowercase, unit-suffixed
///    (build.phase.wall_seconds, search.queries, ...). Field names in the
///    JSONL wire format EQUAL the metric names. Additive-only policy: new
///    fields may be appended; existing names/semantics never change.
///  - Values are raw seconds/bytes/counts. Percentages and throughputs are
///    derived at presentation time, never stored.
///  - Sinks must never throw or block the engine: emit() implementations
///    swallow/log their own failures. A monitoring path must not take down
///    a build or a search thread.
///  - TRIPWIRE (SPRM→TOML style): the wire format is a fixed set of flat
///    fields with engine-controlled (closed-vocabulary) strings. If records
///    ever need nesting or user-sourced strings, switch to a real JSON
///    serializer instead of extending the hand-rolled writer below.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

namespace sextant::metrics {

/// One metrics record: a build phase, a search window, or a total.
struct PhaseMetrics {
    /// Closed vocabulary. Build: sample|pca|kmeans|lloyd|super|stream|write|
    /// pass1|pass2|graph|total. Search: window|total.
    std::string phase;
    /// Closed vocabulary describing the emitter, e.g. "fbin", "parquet",
    /// "tree-search". Never user input.
    std::string source;
    double wall_seconds = 0;          ///< build.phase.wall_seconds
    double cpu_seconds = 0;           ///< build.phase.cpu_seconds (thread-sum)
    double source_wait_seconds = 0;   ///< build.phase.source_wait_seconds
    uint64_t wait_count = 0;          ///< build.phase.wait_count
    uint64_t bytes_read = 0;          ///< build.phase.bytes_read
    uint64_t rss_bytes = 0;           ///< build.phase.rss_bytes (phase peak)
    /// Monotonic seconds at emit time (steady_clock-based epoch), for
    /// time-series alignment when records from multiple processes are merged.
    double timestamp = 0;
};

/// Escape `s` into a JSON string body (without surrounding quotes).
/// Only the two string fields use this, and both are closed-vocabulary —
/// this is defense-in-depth, not load-bearing.
std::string json_escape(std::string_view s);

/// Serialize one record as a single JSON object (no trailing newline).
std::string to_json(const PhaseMetrics& m);

/// ---------------------------------------------------------------------------
/// Sinks
/// ---------------------------------------------------------------------------

class MetricsSink {
public:
    virtual ~MetricsSink() = default;
    /// Consume one record. Must not throw; log-and-continue on I/O errors.
    virtual void emit(const PhaseMetrics& record) = 0;
    /// True when the sink is done with a record once emit() returns
    /// (callback / UDP / flushed file). Lets wrapping sinks know when it is
    /// safe to recycle buffers.
    virtual bool synchronous() const { return true; }
};

/// Human-readable log line (the default sink). Appends to the log, never
/// replaces richer per-phase log messages.
class LogMetricsSink final : public MetricsSink {
public:
    void emit(const PhaseMetrics& r) override {
        const double util =
            r.wall_seconds > 0 ? r.cpu_seconds / r.wall_seconds : 0.0;
        const double mib = 1024.0 * 1024.0;
        spdlog::info("[metrics] phase={} wall={:.3f}s cpu={:.3f}s util={:.0f}% "
                     "src_wait={:.3f}s waits={} read={:.1f}MiB rss={:.1f}MiB",
                     r.phase, r.wall_seconds, r.cpu_seconds, 100.0 * util,
                     r.source_wait_seconds, r.wait_count,
                     static_cast<double>(r.bytes_read) / mib,
                     static_cast<double>(r.rss_bytes) / mib);
    }
};

/// JSON-lines sink. Flushes per record so `tail -F` / filebeat / telegraf
/// see records live — the .jsonl file IS a working streaming pipeline before
/// any UDP/queue work exists.
class JsonlMetricsSink final : public MetricsSink {
public:
    /// Opens `path` for writing ("w"). Falls back to logging an error once
    /// and becoming a no-op — never throws, never blocks the engine.
    explicit JsonlMetricsSink(const std::string& path)
        : file_(std::fopen(path.c_str(), "w")), path_(path) {
        if (!file_) {
            spdlog::error("[metrics] cannot open metrics file '{}': {}",
                          path, std::strerror(errno));
        }
    }
    /// Adopts an already-open FILE* (does not close it on destruction).
    explicit JsonlMetricsSink(std::FILE* f) : file_(f), owns_(false) {}
    ~JsonlMetricsSink() override {
        if (file_ && owns_) std::fclose(file_);
    }
    JsonlMetricsSink(const JsonlMetricsSink&) = delete;
    JsonlMetricsSink& operator=(const JsonlMetricsSink&) = delete;

    void emit(const PhaseMetrics& r) override {
        if (!file_) return;
        const std::string line = to_json(r);
        if (std::fputs(line.c_str(), file_) < 0 || std::fputc('\n', file_) == EOF
            || std::fflush(file_) != 0) {
            spdlog::warn("[metrics] write to '{}' failed: {}", path_,
                         std::strerror(errno));
        }
    }

private:
    std::FILE* file_ = nullptr;
    std::string path_;
    bool owns_ = true;
};

/// Callback sink — THE embedding point for future push transports (UDP
/// sendto, Kafka/Redis producer, OTel collector, in-proc ring buffer): all
/// are lambdas an operator writes; the engine never changes.
class FunctionMetricsSink final : public MetricsSink {
public:
    explicit FunctionMetricsSink(std::function<void(const PhaseMetrics&)> fn)
        : fn_(std::move(fn)) {}
    void emit(const PhaseMetrics& r) override {
        if (fn_) fn_(r);  // callback exceptions must not propagate: wrap if needed
    }

private:
    std::function<void(const PhaseMetrics&)> fn_;
};

/// Fan-out to N sinks (e.g. log + jsonl + callback). Non-owning.
class MultiMetricsSink final : public MetricsSink {
public:
    void add(MetricsSink* s) { sinks_.push_back(s); }
    void clear() { sinks_.clear(); }
    void emit(const PhaseMetrics& r) override {
        for (MetricsSink* s : sinks_) s->emit(r);
    }
    bool synchronous() const override {
        for (MetricsSink* s : sinks_)
            if (!s->synchronous()) return false;
        return true;
    }

private:
    std::vector<MetricsSink*> sinks_;
};

/// ---------------------------------------------------------------------------
/// JSON writer (~40 lines, hand-rolled by design — see header tripwire).
/// Doubles use %.17g: round-trip-safe, locale-independent in the C locale
/// the engine keeps (the engine never calls setlocale).
/// ---------------------------------------------------------------------------

inline std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

inline std::string to_json(const PhaseMetrics& m) {
    char buf[512];
    std::string out;
    out.reserve(384);
    out += "{\"phase\":\"";
    out += json_escape(m.phase);
    out += "\",\"source\":\"";
    out += json_escape(m.source);
    std::snprintf(buf, sizeof(buf),
                  "\",\"wall_seconds\":%.17g,\"cpu_seconds\":%.17g,"
                  "\"source_wait_seconds\":%.17g,\"wait_count\":%llu,"
                  "\"bytes_read\":%llu,\"rss_bytes\":%llu,"
                  "\"timestamp\":%.17g}",
                  m.wall_seconds, m.cpu_seconds, m.source_wait_seconds,
                  static_cast<unsigned long long>(m.wait_count),
                  static_cast<unsigned long long>(m.bytes_read),
                  static_cast<unsigned long long>(m.rss_bytes),
                  m.timestamp);
    out += buf;
    return out;
}

}  // namespace sextant::metrics
