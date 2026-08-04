#pragma once

/// @file partition.hpp
/// K-means partitioning on PQ codes, used by the graph builder (shard
/// partitioning) and the tree index (leaf split).
///
/// The pipeline has two stages:
/// 1. **kmeans_pq**: core k-means loop (centroid init → assignment → medoid
///    update → convergence). Returns medoid centroids (PQ codes) + assignment.
/// 2. **assign_with_closure**: builds the final shards with closure overlap,
///    balance capping, and RNG redundancy pruning (SPANN-style).
///
/// partition_codes chains the two stages for the graph builder's use case.
/// Leaf split calls kmeans_pq directly (no closure — the split produces two
/// disjoint leaves).

#include <sextant/types.hpp>
#include <cstdint>
#include <vector>

namespace sextant {

class PqQuantizer;

/// Result of the core k-means loop.
struct KMeansResult {
    /// centroids[k] = PQ code of cluster k's medoid (code_size bytes).
    /// Padded to a multiple of 4 (K_padded) for batch4 SIMD.
    std::vector<std::vector<uint8_t>> centroids;

    /// assign[i] = cluster index of vector i (nearest centroid).
    std::vector<uint32_t> assign;

    /// Number of real (non-padding) clusters.
    uint32_t K = 0;
};

/// Result of partitioning: for each shard, the list of global vector IDs
/// assigned to it (in the order they were assigned — shard-local index = the
/// position in this vector).
struct PartitionAssignment {
    /// shards_[k] = vector of global IDs assigned to shard k.
    std::vector<std::vector<uint32_t>> shards;

    /// centroids_[k] = PQ code (code_size bytes) for shard k's centroid.
    std::vector<std::vector<uint8_t>> centroids;

    /// The closure_factor used for overlap assignment.
    float closure_factor = 1.0f;

    /// Absolute margin (SPANN-style) used for overlap assignment. When >0,
    /// vectors within d_best + closure_epsilon of a centroid are replicated.
    /// 0 = use ratio-based (closure_factor × d_best) instead.
    float closure_epsilon = 0.0f;
};

/// Core k-means on PQ codes. Runs centroid init → assignment → medoid update
/// for `iterations` iterations (with early convergence at <1% change).
///
/// Centroids are medoids (actual data points selected as the best central
/// representative). Distances use the quantizer's PQ code_distance (symmetric
/// via the cross-distance table, batch4 SIMD).
///
/// @param quantizer   Trained PQ quantizer (provides code_distance, batch4).
/// @param codes       Flat PQ codes buffer (n × code_size).
/// @param n           Number of vectors.
/// @param code_size   Bytes per PQ code.
/// @param K           Number of clusters.
/// @param iterations  Max k-means iterations.
/// @param num_threads Parallelism (0 = hardware_concurrency).
/// @param seed        RNG seed for initial centroid selection.
KMeansResult kmeans_pq(const PqQuantizer& quantizer,
                       const uint8_t* codes, uint32_t n,
                       uint32_t code_size, uint32_t K,
                       uint32_t iterations = 10,
                       uint32_t num_threads = 0,
                       uint64_t seed = 0xC0DE1234ULL);

/// Build the final shard assignment with closure overlap, balance capping,
/// and RNG redundancy pruning. Takes the k-means output and replicates
/// boundary vectors into multiple shards.
///
/// @param km               Output of kmeans_pq.
/// @param closure_factor   Ratio threshold for multi-shard assignment.
/// @param closure_epsilon  Absolute margin (SPANN-style). Sentinel -1.0 =
///                         auto-compute from mean NN distance. 0 = ratio only.
/// @param balance_factor   Size-balancing strength (0 = off). Caps max_shard
///                         at roughly (1 + 1/balance_factor) × mean_size.
/// @param num_threads      Parallelism (0 = hardware_concurrency).
PartitionAssignment assign_with_closure(
    const PqQuantizer& quantizer,
    const uint8_t* codes, uint32_t n, uint32_t code_size,
    const KMeansResult& km,
    float closure_factor,
    float closure_epsilon = 0.0f,
    float balance_factor = 0.0f,
    uint32_t num_threads = 0);

/// Run k-means on PQ codes and assign vectors to shards with closure overlap.
/// Convenience wrapper: kmeans_pq → assign_with_closure.
///
/// @param quantizer   Trained PQ quantizer (provides PQ code_distance).
/// @param codes       Flat PQ codes buffer (n × code_size), indexed by global ID.
/// @param n           Number of vectors.
/// @param code_size   Bytes per PQ code.
/// @param K           Number of partitions (clusters).
/// @param closure_factor  Ratio threshold for multi-shard assignment.
/// @param iterations  K-means iterations (default 10).
/// @param num_threads Parallelism for the assignment + medoid passes
///                    (0 = hardware_concurrency). The assignment passes are
///                    embarrassingly parallel; the medoid pass parallelizes
///                    over the K clusters.
/// @param seed        RNG seed for initial centroid selection.
/// @param balance_factor  Size-balancing strength (0 = off; SPANN-style
///                    λ-regularization). When >0, oversized clusters spill
///                    excess vectors to their next-nearest centroid, bounding
///                    shard-size variance. Caps max_shard at roughly
///                    (1 + 1/balance_factor) × mean_size. Default 0 (off).
///                    See SPANN (arXiv:2111.08566) §"Hierarchical Balanced
///                    Clustering" — list-size balance bounds the worst-case
///                    disk-read pathology at paged-billion-scale.
/// @param closure_epsilon  Absolute margin for SPANN-style boundary posting.
///                    When >0, vectors within d_best + closure_epsilon of a
///                    centroid are replicated. This is structurally K-robust:
///                    as K grows and cells shrink, epsilon covers an increasing
///                    fraction of the cell radius. 0 = use ratio-based closure
///                    (closure_factor × d_best). See docs/closure_factor_derivation.md §6.
PartitionAssignment partition_codes(const PqQuantizer& quantizer,
                                     const uint8_t* codes, uint32_t n,
                                     uint32_t code_size, uint32_t K,
                                     float closure_factor,
                                     uint32_t iterations = 10,
                                     uint32_t num_threads = 0,
                                     uint64_t seed = 0xC0DE1234ULL,
                                     float balance_factor = 0.0f,
                                     float closure_epsilon = 0.0f);

}  // namespace sextant
