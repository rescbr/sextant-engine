#pragma once

/// @file system.hpp
/// Platform abstractions shared across the engine (RAM detection, etc.).
/// Consolidated from previously-duplicated #ifdef blocks in build.cpp and
/// search.cpp (architecture audit §10.4).

#include <cstdint>

namespace sextant {

/// Physical RAM in bytes. Returns 0 if detection fails.
/// Platform-specific (sysctl on macOS, sysconf elsewhere) with a portable
/// fallback.
uint64_t physical_ram_bytes();

}  // namespace sextant
