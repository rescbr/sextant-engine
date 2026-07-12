#pragma once

/// @file build.hpp
/// Build pipeline: two-pass streaming + parallel construct.

#include <sextant/types.hpp>
#include <sextant/vector_source.hpp>
#include <string>

namespace sextant {

struct PqQuantizer;
struct VamanaCore;

struct BuildConfig {
    uint16_t R = 64;
    uint16_t L = 100;
    float alpha = 1.2f;
    uint16_t inline_pq_count = 0;
    uint8_t pq_m = 0;           ///< 0 = auto-resolve from dim
    uint8_t pq_bits = 8;
    uint64_t build_ram_budget = 0;  ///< 0 = auto (50% of physical RAM)
    MetricKind metric = MetricKind::L2Sq;
};

struct BuildResult {
    std::string index_path;
    uint64_t n_vectors = 0;
    Dim dim = 0;
    double build_time_sec = 0;
};

}  // namespace sextant
