#pragma once

/// @file config.hpp
/// Build and search configuration structs.

#include <sextant/types.hpp>
#include <cstdint>

namespace sextant {

/// Build configuration. Fields set to 0/default are auto-resolved.
struct BuildConfig {
    uint16_t R = 0;             ///< 0 = auto from N
    uint16_t L = 0;             ///< 0 = auto from R
    float alpha = 0.0f;         ///< 0 = auto (1.2 SDC)
    uint16_t inline_pq_count = 0xFFFF; ///< 0xFFFF = auto (balanced preset)
    uint8_t pq_m = 0;           ///< 0 = auto from dim
    uint8_t pq_bits = 8;
    uint64_t build_ram_budget = 0;  ///< 0 = auto (50% of physical RAM)
    MetricKind metric = MetricKind::L2Sq;
    uint32_t num_threads = 0;   ///< 0 = hardware_concurrency
};

/// Result of a build operation.
struct BuildResult {
    std::string index_path;
    uint64_t n_vectors = 0;
    Dim dim = 0;
    double build_time_sec = 0;
    uint16_t R = 0;
    uint16_t L_build = 0;
    uint8_t pq_m = 0;
};

/// Search configuration.
struct SearchConfig {
    uint32_t k = 10;
    uint32_t L_search = 200;
    uint32_t rerank_factor = 10;
    uint32_t io_limit = 0;  ///< 0 = unlimited (visit as many nodes as L_search allows)
};

}  // namespace sextant
