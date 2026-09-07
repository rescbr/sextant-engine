#pragma once

/// @file engine_trace.hpp
/// Machine-written, machine-read trace sink for engine diagnostics.
///
/// NOT a logger: records are tagged `key=value` lines written by the
/// engine's instrumented paths (e.g. scan-feedback probing) and consumed
/// by the `sextant trace` CLI subcommand for offline analysis. Human
/// logging goes through spdlog (sextant/logging.hpp) — this class exists
/// because per-record diagnostic data (per-query probe traces) is data
/// egress, not logs: no timestamps, no log levels, stable single-line
/// format, versioned header.
///
/// Ownership: caller-owned, like SearchConfig::exact_rerank_base. The
/// harness creates it, passes it via SearchConfig::trace, and destroys
/// it. The engine never opens files on its own initiative.
///
/// Thread safety: one mutex around a buffered stream. Records are
/// written from query-parallel search threads. Diagnostic volumes are
/// modest (≤ ~1M lines/run); per-thread buffering would be
/// over-engineering at this scale.

#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace sextant {

class EngineTrace {
public:
    /// Open `path` for writing (truncates). Returns nullptr if the file
    /// cannot be opened. Writes a versioned header line immediately so a
    /// truncated/corrupt trace is detectable by the reader.
    static std::unique_ptr<EngineTrace> create(const std::string& path,
                                               std::string_view run_label);

    /// Append one record. MUST be a single line (no embedded newlines);
    /// the newline is appended here. Formatted variants below cover the
    /// common cases without call-site string building.
    void record(std::string_view line) {
        std::scoped_lock lock(mu_);
        out_ << line << '\n';
    }

    /// Append a record formatted `tag k1=v1 k2=v2 ...` via std::format
    /// (compile-time checked format string) — no hand-rolled concatenation.
    template <class... Args>
    void record_fmt(std::format_string<Args...> fmt_str, Args&&... args) {
        record(std::format(fmt_str, std::forward<Args>(args)...));
    }

    /// Flush the underlying stream (also done by the destructor).
    void flush() {
        std::scoped_lock lock(mu_);
        out_.flush();
    }

    ~EngineTrace() {
        std::scoped_lock lock(mu_);
        out_.flush();
    }

    EngineTrace(const EngineTrace&) = delete;
    EngineTrace& operator=(const EngineTrace&) = delete;

private:
    EngineTrace() = default;
    std::ofstream out_;
    std::mutex mu_;
};

}  // namespace sextant
