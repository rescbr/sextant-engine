#pragma once

/// @file partition.hpp
/// PartitionInfo: k-means partitioning on PQ codes (Step 11).
///
/// Partitions the dataset into K shards by running k-means on the PQ codes
/// using PQ code distance (symmetric distance via the quantizer's cross-distance table).
/// Each vector is assigned to its nearest centroid, plus any centroid within
/// `closure_factor × d_best` — this creates ~15% overlap so boundary nodes
/// appear in multiple shards, keeping the merged graph connected.
///
/// The centroids are PQ codes (not raw float vectors), so both k-means and
/// assignment use code-to-code distances exclusively.

#include <sextant/types.hpp>
#include <cstdint>
#include <vector>

namespace sextant {

class PqQuantizer;

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
};

/// Run k-means on PQ codes and assign vectors to shards with closure overlap.
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
PartitionAssignment partition_codes(const PqQuantizer& quantizer,
                                     const uint8_t* codes, uint32_t n,
                                     uint32_t code_size, uint32_t K,
                                     float closure_factor,
                                     uint32_t iterations = 10,
                                     uint32_t num_threads = 0,
                                     uint64_t seed = 0xC0DE1234ULL,
                                     float balance_factor = 0.0f);

}  // namespace sextant
