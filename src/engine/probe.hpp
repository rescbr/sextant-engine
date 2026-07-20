#pragma once

/// @file probe.hpp
/// Shared helpers for PQ-config probing and reservoir sampling.
///
/// Used by both Builder (probe_pq_config + pass1_sample_and_train) and
/// Estimator (estimate_config). Lives in src/engine/ because it's an
/// internal cross-cutting concern — not part of the public API.

#include "sextant/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sextant {

/// Probe pool size: we measure recall on a subset of the reservoir to keep
/// the brute-force KNN cheap. 20K is what pq_explore uses; stable enough.
inline constexpr uint32_t kProbePool = 20000;
inline constexpr uint32_t kProbeQueries = 500;
inline constexpr uint32_t kProbeTopk = 10;
inline constexpr uint32_t kProbeRecallK = 30;

/// Brute-force truth for the probe: per query, the top-kProbeTopk neighbor ids
/// (or top-truth_k if overridden), their true L2sq distances, and band counts
/// within the k-th and kProbeRecallK-th NN distances (tie bands — how many
/// vectors tie at the boundary, which cap achievable recall).
struct ProbeTruth {
    std::vector<std::vector<uint32_t>> ids;      // [qi] → top-k ids
    std::vector<std::vector<float>> dists;       // [qi] → top-k true L2sq
    std::vector<uint32_t> band_counts;           // [qi] → # vectors within k-th NN dist
    std::vector<uint32_t> band30_counts;         // [qi] → # vectors within kProbeRecallK-th NN dist
};

/// Compute brute-force true top-`truth_k` for a set of query indices within
/// the pool. Returns ids + true distances (ascending) + per-query band counts.
ProbeTruth compute_truth(const float* pool, uint32_t pool_n, Dim dim,
                         const std::vector<uint32_t>& qidx,
                         uint32_t truth_k = kProbeTopk);

/// Distortion + diagnostics measured for one (m, bits) config against truth.
struct ProbeScore {
    double distortion = 0.0;     // median |1 - pq_dist/true_dist| (≥0; 0 = perfect)
    double band_recall = 0.0;    // frac truth neighbors within band found in PQ top-K
    double tie_fraction = 0.0;   // frac queries whose band(@k) has >kProbeTopk tied vectors
    double tie30_fraction = 0.0; // frac queries whose band(@kProbeRecallK) has >kProbeRecallK tied
};

/// Measure PQ quality for a given (m, bits) config (distortion, band_recall,
/// tie fractions).
ProbeScore probe_recall(const float* pool, uint32_t pool_n, Dim dim,
                        MetricKind metric, uint16_t pq_m, uint8_t bits,
                        const ProbeTruth& truth,
                        const std::vector<uint32_t>& qidx);

/// Candidate m values: divisors of dim with sub_dim in [1, 12].
std::vector<uint16_t> candidate_ms(Dim dim);

/// A probed (m, bits) config with its measured quality and cost.
struct ProbedConfig {
    uint16_t m;
    uint8_t bits;
    uint32_t code_bytes;   // m * bits / 8
    uint32_t table_bytes;  // m * K^2 * 4 (cross-distance table)
    double distortion;       // median |1 - pq_dist/true_dist| (≥0; 0 = perfect)
    double band_recall;      // cluster-aware recall (diagnostic)
    double tie_fraction;     // frac queries with >topk tied at @k (diagnostic)
    double tie30_fraction;   // frac queries with >30 tied at @30 (diagnostic)
    double cost;           // m * residency_factor (lower = faster)
};

/// Cross-distance table size in bytes for a given (m, bits).
inline uint32_t pq_table_bytes(uint16_t m, uint8_t bits) {
    const uint32_t K = 1u << bits;
    return static_cast<uint32_t>(m) * K * K * sizeof(float);
}

/// Sweep (m, bits) candidate pairs, measure distortion, and select the
/// minimum-cost config whose distortion is within `max_distortion`.
/// When `fixed_m != 0` or `fixed_bits != 0`, the probe is restricted to that
/// m/bits but the quality gate still applies. `all_out` receives every probed
/// config (for display); `reason_out` receives the selection rationale.
ProbedConfig probe_best_config(const float* pool, uint32_t pool_n, Dim dim,
                                MetricKind metric,
                                const ProbeTruth& truth,
                                const std::vector<uint32_t>& qidx,
                                uint16_t fixed_m, uint8_t fixed_bits,
                                double max_distortion,
                                std::vector<ProbedConfig>& all_out,
                                std::string& reason_out);

/// Element type tag for draw_random_sample's cast logic.
enum class SampleElemType { Float32, Int8, Uint8 };

SampleElemType infer_sample_elem_type(const std::string& path);

/// Vitter-style reservoir sampling via seek-based random access. Reads `k`
/// vectors from `path` (an .fbin/.ibin/.bbin file) without loading the whole
/// file. Returns floats (cast from int8/uint8 if needed).
std::vector<float> draw_random_sample(const std::string& path,
                                       uint64_t n, Dim dim, uint64_t k);

}  // namespace sextant
