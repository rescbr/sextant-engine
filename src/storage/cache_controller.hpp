#pragma once

/// @file cache_controller.hpp
/// Adaptive controller that rebalances the graph and code BlockCache sizes
/// based on observed hit/miss ratios.
///
/// PagedNodeStore owns two independent BlockCaches — one for graph (.graph)
/// blocks, one for code (.codes) blocks — so the two streams don't evict each
/// other. The CacheController monitors their relative miss rates and
/// periodically shifts capacity between them via BlockCache::resize().
///
/// The controller is NOT invoked from the hot search loop automatically
/// (that's a follow-up). PagedNodeStore::maybe_rebalance_caches() forwards to
/// it and can be called periodically by the Engine.

#include "storage/block_cache.hpp"

#include <algorithm>
#include <cstdint>

namespace sextant {

/// Monitors hit/miss ratios for graph and code caches and periodically
/// rebalances their relative sizes. Called after every N searches.
class CacheController {
public:
    CacheController(BlockCache& graph_cache, BlockCache& code_cache,
                    uint32_t num_shards, uint64_t total_budget_bytes,
                    uint32_t block_size)
        : graph_cache_(graph_cache),
          code_cache_(code_cache),
          num_shards_(num_shards),
          total_budget_bytes_(total_budget_bytes),
          block_size_(block_size),
          total_blocks_per_shard_(std::max(
              1u, static_cast<uint32_t>(total_budget_bytes /
                                        (static_cast<uint64_t>(num_shards) *
                                         block_size)))),
          graph_fraction_(0.5) {}

    /// Called periodically (e.g. after every 10000 searches). Shifts capacity
    /// between the two caches based on relative miss rates.
    void maybe_rebalance() {
        const CacheStats g = graph_cache_.stats();
        const CacheStats c = code_cache_.stats();

        const uint64_t g_hits =
            g.hits_window + g.hits_probation + g.hits_protected;
        const uint64_t c_hits =
            c.hits_window + c.hits_probation + c.hits_protected;

        // Deltas since the last call.
        const uint64_t dg_hits = g_hits - prev_graph_hits_;
        const uint64_t dg_misses = g.misses - prev_graph_misses_;
        const uint64_t dc_hits = c_hits - prev_code_hits_;
        const uint64_t dc_misses = c.misses - prev_code_misses_;

        prev_graph_hits_ = g_hits;
        prev_graph_misses_ = g.misses;
        prev_code_hits_ = c_hits;
        prev_code_misses_ = c.misses;

        // If either stream was idle since the last call, nothing to do.
        const uint64_t dg_total = dg_hits + dg_misses;
        const uint64_t dc_total = dc_hits + dc_misses;
        if (dg_total == 0 || dc_total == 0) {
            return;
        }

        const double graph_miss_rate =
            static_cast<double>(dg_misses) / static_cast<double>(dg_total);
        const double code_miss_rate =
            static_cast<double>(dc_misses) / static_cast<double>(dc_total);

        // Shift capacity toward the cache with the higher miss rate.
        if (graph_miss_rate > code_miss_rate * 1.5) {
            graph_fraction_ = std::min(MAX_FRACTION, graph_fraction_ + 0.05);
        } else if (code_miss_rate > graph_miss_rate * 1.5) {
            graph_fraction_ = std::max(MIN_FRACTION, graph_fraction_ - 0.05);
        }

        // Compute new per-shard capacities, clamped to [MIN, MAX] fractions.
        uint32_t graph_blocks = static_cast<uint32_t>(
            graph_fraction_ * total_blocks_per_shard_);
        uint32_t code_blocks = total_blocks_per_shard_ - graph_blocks;

        const uint32_t min_blocks = std::max(
            1u, static_cast<uint32_t>(MIN_FRACTION * total_blocks_per_shard_));
        const uint32_t max_blocks = static_cast<uint32_t>(
            MAX_FRACTION * total_blocks_per_shard_);

        graph_blocks = std::clamp(graph_blocks, min_blocks, max_blocks);
        code_blocks = std::clamp(code_blocks, min_blocks, max_blocks);

        graph_cache_.resize(graph_blocks);
        code_cache_.resize(code_blocks);
    }

    static constexpr double MIN_FRACTION = 0.15;  ///< 15% floor per cache
    static constexpr double MAX_FRACTION = 0.85;  ///< 85% ceiling per cache

private:
    BlockCache& graph_cache_;
    BlockCache& code_cache_;
    [[maybe_unused]] uint32_t num_shards_;             // for future diagnostics
    [[maybe_unused]] uint64_t total_budget_bytes_;     // for future diagnostics
    [[maybe_unused]] uint32_t block_size_;             // for future diagnostics
    /// Per-shard capacity baseline (total / num_shards / block_size).
    uint32_t total_blocks_per_shard_;
    /// Previous cumulative counters for delta computation.
    uint64_t prev_graph_hits_ = 0, prev_graph_misses_ = 0;
    uint64_t prev_code_hits_ = 0, prev_code_misses_ = 0;
    /// Current fraction of total budget assigned to graph (0..1).
    double graph_fraction_ = 0.5;
};

}  // namespace sextant
