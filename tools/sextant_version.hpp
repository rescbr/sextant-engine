/// Version reporting for the CLI tools. The git commit + dirty flag are captured
/// at meson configure time (see meson.build) into build/version.h; this header
/// turns those macros into a human-readable string used by `--version` and the
/// usage/help banner of sextant and sextant_bench.
#pragma once

#include <string>
#include <string_view>

#include "version.h"  // generated: SEXTANT_VERSION, SEXTANT_GIT_SHA, SEXTANT_GIT_DIRTY

namespace sextant {

/// The full version line, e.g. "sextant 0.1.0 (4433a8a)" or, for a dirty tree,
/// "sextant 0.1.0 (4433a8a, dirty)". `prog` is the program name shown first.
inline std::string version_string(std::string_view prog) {
    std::string s{prog};
    s += ' ';
    s += SEXTANT_VERSION;
    s += " (";
    s += SEXTANT_GIT_SHA;
    if (SEXTANT_GIT_DIRTY) {
        s += ", dirty";
    }
    s += ", ";
    s += SEXTANT_SIMD_TARGET;
    s += ')';
    return s;
}

}  // namespace sextant
