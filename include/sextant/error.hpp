#pragma once

/// @file error.hpp
/// Error handling for the Sextant engine.
///
/// The engine throws `sextant::Error` on unrecoverable failures. The CLI
/// catches and prints; the Phase 2 DuckDB adapter maps to DuckDB exceptions.

#include <cpptrace/cpptrace.hpp>

#include <stdexcept>
#include <string>
#include <cstdint>

namespace sextant {

enum class ErrorCode : uint8_t {
    IoError,
    CorruptIndex,
    InvalidParam,
    OutOfMemory,
    NotImplemented,
};

/// Engine exception type. Inherits std::runtime_error for compatibility.
/// Captures a stack trace at construction (throw) site so the caller can
/// see exactly where the error originated, not just where it was caught.
class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& msg)
        : std::runtime_error(msg), code_(code),
          trace_(cpptrace::generate_trace()) {}

    ErrorCode code() const noexcept { return code_; }
    const cpptrace::stacktrace& trace() const noexcept { return trace_; }

private:
    ErrorCode code_;
    cpptrace::stacktrace trace_;
};

}  // namespace sextant
