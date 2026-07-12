#pragma once

/// @file logging.hpp
/// Engine logging interface (spdlog-backed).

#include <string>

namespace sextant {

enum class LogLevel : uint8_t {
    Debug,
    Info,
    Warn,
    Error,
};

/// Initialize the global logger. Call once at engine startup.
/// Default level: Info. Overridable per-build.
void init_logging(LogLevel level = LogLevel::Info);

/// Set the log level at runtime.
void set_log_level(LogLevel level);

}  // namespace sextant
