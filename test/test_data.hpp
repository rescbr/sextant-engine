// test_data.hpp — locate SIFTsmall and other test fixtures.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

namespace sextant {
namespace test {

/// Resolve the test data directory. Checks (in order):
///   1. SEXTANT_TEST_DATA env var (set by meson test).
///   2. "test/data" relative to the current working directory.
///   3. The source tree "test/data" (common when running the binary directly).
inline std::string test_data_dir() {
    if (const char* env = std::getenv("SEXTANT_TEST_DATA"); env && *env) {
        return env;
    }
    namespace fs = std::filesystem;
    const fs::path candidates[] = {
        fs::path("test") / "data",
        fs::path("../test/data"),
        fs::path(SEXTANT_SOURCE_DIR) / "test" / "data",
    };
    for (const auto& p : candidates) {
        if (fs::exists(p / "siftsmall_base.fbin")) return p.string();
    }
    return "test/data";
}

inline std::string siftsmall_base() {
    return (std::filesystem::path(test_data_dir()) / "siftsmall_base.fbin").string();
}

inline std::string siftsmall_query() {
    return (std::filesystem::path(test_data_dir()) / "siftsmall_query.fbin").string();
}

inline std::string siftsmall_gt() {
    return (std::filesystem::path(test_data_dir()) / "siftsmall_gt.gt").string();
}

}  // namespace test
}  // namespace sextant
