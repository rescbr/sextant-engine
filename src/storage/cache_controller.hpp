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
/// Invoked from the Engine search path (every N searches, CAS-guarded so only
/// one thread rebalances at a time). PagedNodeStore::maybe_rebalance_caches()
/// forwards here.

#include "storage/block_cache.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

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
          graph_fraction_(0.5) {
        // Tunable without recompile: SEXTANT_CACHE_REBALANCE_THRESHOLD
        // (default 1.5). Raising to 2.0 makes the controller less trigger-happy
        // under noisy workloads.
        double thresh = 1.5;
        if (const char* env = std::getenv("SEXTANT_CACHE_REBALANCE_THRESHOLD")) {
            double parsed = std::atof(env);
            if (parsed >= 1.0 && parsed <= 10.0) thresh = parsed;  // sanity clamp
        }
        miss_rate_threshold_ = thresh;
    }

    /// Seed graph_fraction_ from the caches' actual current per-shard
    /// capacities. Called once after construction (the caches are split
    /// file-size-proportionally; without seeding, the first step would jump
    /// from the true initial split to a 0.5-based value — a spurious shift).
    void seed_fraction_from_caches() {
        const uint32_t g = graph_cache_.shard(0).capacity();
        const uint32_t c = code_cache_.shard(0).capacity();
        const uint32_t total = g + c;
        if (total > 0) {
            graph_fraction_ = static_cast<double>(g) / static_cast<double>(total);
            // Clamp to the legal band so the first step starts in-range.
            graph_fraction_ = std::clamp(graph_fraction_, MIN_FRACTION, MAX_FRACTION);
        }
    }

    /// Called periodically (e.g. after every 1000 searches). Shifts capacity
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

        // Hysteresis: require 2 consecutive agreeing samples before stepping.
        // Without this, each resize evicts entries and causes the misses that
        // trigger the reverse step next round (positive feedback / thrashing).
        enum class Signal { None, GraphNeedsMore, CodeNeedsMore };
        Signal sig = Signal::None;
        if (graph_miss_rate > code_miss_rate * miss_rate_threshold_) {
            sig = Signal::GraphNeedsMore;
        } else if (code_miss_rate > graph_miss_rate * miss_rate_threshold_) {
            sig = Signal::CodeNeedsMore;
        }

        bool step = false;
        if (sig == Signal::GraphNeedsMore) {
            if (++trend_graph_needs_more_ >= 2) {
                graph_fraction_ = std::min(MAX_FRACTION, graph_fraction_ + 0.05);
                step = true;
            }
            trend_code_needs_more_ = 0;
        } else if (sig == Signal::CodeNeedsMore) {
            if (++trend_code_needs_more_ >= 2) {
                graph_fraction_ = std::max(MIN_FRACTION, graph_fraction_ - 0.05);
                step = true;
            }
            trend_graph_needs_more_ = 0;
        } else {
            // Signal::None — workload balanced or below threshold. Reset trends
            // so a future direction must build 2 fresh agreements.
            trend_graph_needs_more_ = 0;
            trend_code_needs_more_ = 0;
        }

        spdlog::debug("[sextant] cache-rebalance: g_miss={:.4f} c_miss={:.4f} "
                      "ratio={:.2f} thresh={:.2f} sig={} trend_g={} trend_c={} "
                      "graph_frac={:.3f} step={}",
                      graph_miss_rate, code_miss_rate,
                      graph_miss_rate > 0 ? code_miss_rate / graph_miss_rate : 0.0,
                      miss_rate_threshold_, static_cast<int>(sig),
                      trend_graph_needs_more_, trend_code_needs_more_,
                      graph_fraction_, step);

        // Only resize if the fraction actually moved (avoid needless work).
        if (!step) return;

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
    /// Hysteresis trend counters: consecutive agreeing samples. A step fires
    /// only when the count reaches 2.
    int trend_graph_needs_more_ = 0;
    int trend_code_needs_more_ = 0;
    /// Miss-rate ratio that triggers a step (from env, default 1.5).
    double miss_rate_threshold_ = 1.5;
};

}  // namespace sextant
