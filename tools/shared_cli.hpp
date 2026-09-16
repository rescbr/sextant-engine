#pragma once

/// Scan-feedback probe-spec parsing shared by `sextant sweep` and `sextant
/// trace` (tools/sextant_cli.cpp).
///
/// The build/autobuild/analyze common flag parser that used to live here was
/// removed with the flat/Vamana stack (v1; preserved at tag `vamana-eol`).

#include "sextant/config.hpp"

#include <cmdline/cmdline.h>

#include <charconv>
#include <cstdint>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace sextant_cli {

/// Parse one scan-feedback probe spec (SearchConfig::feedback):
///   fixed:F | stall:M | kth:M   optionally followed by ,min:N
/// Throws std::runtime_error with the flag name on malformed input.
inline sextant::FeedbackProbe parse_feedback_spec(const std::string& spec) {
    sextant::FeedbackProbe fb;
    std::string main = spec;
    if (const auto comma = spec.find(','); comma != std::string::npos) {
        main = spec.substr(0, comma);
        const std::string mins = spec.substr(comma + 1);
        if (mins.rfind("min:", 0) != 0)
            throw std::runtime_error("--feedback: expected ,min:N suffix");
        uint32_t mv = 0;
        auto [ptr, ec] = std::from_chars(mins.data() + 4, mins.data() + mins.size(), mv);
        if (ec != std::errc() || ptr != mins.data() + mins.size() || mv == 0)
            throw std::runtime_error("--feedback: bad min value");
        fb.min_blocks = mv;
    }
    const auto colon = main.find(':');
    if (colon == std::string::npos)
        throw std::runtime_error("--feedback: expected fixed:F | stall:M | kth:M");
    const std::string name = main.substr(0, colon);
    const std::string val = main.substr(colon + 1);
    if (name == "fixed") {
        float f = 0;
        auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), f);
        if (ec != std::errc() || ptr != val.data() + val.size()
            || f <= 0.0f || f > 1.0f)
            throw std::runtime_error("--feedback: fixed fraction must be in (0,1]");
        fb.mode = sextant::FeedbackProbe::Mode::Fixed;
        fb.fixed_fraction = f;
    } else {
        uint32_t m = 0;
        auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), m);
        if (ec != std::errc() || ptr != val.data() + val.size() || m == 0)
            throw std::runtime_error("--feedback: threshold M must be a positive integer");
        fb.mode = name == "stall" ? sextant::FeedbackProbe::Mode::Stall
                                  : sextant::FeedbackProbe::Mode::Kth;
        if (name != "stall" && name != "kth")
            throw std::runtime_error("--feedback: unknown mode '" + name + "'");
        fb.m = m;
    }
    return fb;
}

/// Parse a comma-separated list of feedback specs into (spec, parsed) rows.
inline std::vector<std::pair<std::string, sextant::FeedbackProbe>>
parse_feedback_list(const std::string& list) {
    std::vector<std::pair<std::string, sextant::FeedbackProbe>> rows;
    std::string item;
    std::istringstream iss(list);
    while (std::getline(iss, item, ',')) {
        if (item.empty()) continue;
        rows.emplace_back(item, parse_feedback_spec(item));
    }
    return rows;
}

}  // namespace sextant_cli
