#pragma once

/// @file error.hpp
/// Error handling for the Sextant engine.
///
/// The engine throws `sextant::Error` on unrecoverable failures. The CLI
/// catches and prints; a future DuckDB adapter maps to DuckDB exceptions.
///
/// Stack traces at the throw site are captured by the `__cxa_throw`
/// interception in crash_handler.hpp (stored in thread-local
/// `last_throw_trace()`), not by this class.

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
class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& msg)
        : std::runtime_error(msg), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

}  // namespace sextant
