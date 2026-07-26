#pragma once

/// @file resolve_params.hpp
/// Parameter resolution header. Configs are in the public API (include/sextant/config.hpp).

#include "sextant/config.hpp"

#include <algorithm>
#include <cstdint>

namespace sextant {

/// Compute the recall-driven K floor for IVF builds. Shared by
/// resolve_params and estimate_config (was duplicated between them).
///
/// Scan path (merged_graph=false): target ~200k vectors/shard, K in [64, 8192].
/// Graph path (sharded_graph=true): target ~64k vectors/shard, K in [2, 256].
/// Non-IVF (neither): K=1 (monolithic).
inline uint32_t recall_driven_k_floor(uint64_t n_vectors, bool is_scan,
                                       bool is_ivf) {
    if (!is_scan && !is_ivf) return 1;
    if (is_scan) {
        constexpr uint64_t kTargetShardSize = 200'000;
        uint32_t k = static_cast<uint32_t>(
            (n_vectors + kTargetShardSize - 1) / kTargetShardSize);
        return std::clamp(k, 64u, 8192u);
    } else {
        constexpr uint64_t kTargetShardSize = 65536;
        uint32_t k = static_cast<uint32_t>(std::max<uint64_t>(2u,
            (n_vectors + kTargetShardSize - 1) / kTargetShardSize));
        return std::min<uint32_t>(k, 256u);
    }
}

/// K_max for the given build path. Scan: 8192. Graph: 512.
inline uint32_t k_max_for_path(bool is_scan) {
    return is_scan ? 8192u : 512u;
}

}  // namespace sextant
