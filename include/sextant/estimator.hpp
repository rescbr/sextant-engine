#pragma once

/// @file estimator.hpp
/// Estimator — sample-driven BuildConfig auto-resolution.
///
/// Phase D: estimate_config + its mini-build and measurement helpers moved
/// out of the Engine god-object. The Estimator builds temporary in-RAM
/// Indices via Builder (no temp files — architecture decision in
/// refactor-architecture-decisions.md), measures graph topology + search
/// quality on them, and returns the optimal ResolvedParams +
/// EstimationDiagnostics.
///
/// Engine keeps a thin delegator (estimate_config) so existing callers
/// (CLI, analyze) don't change yet. Phase E deletes Engine.

#include "sextant/config.hpp"
#include "sextant/index.hpp"
#include "sextant/vector_source.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace sextant {

/// Sample-driven parameter estimator. Stateless — each estimate_config call
/// builds its own mini-Indices internally and discards them on return.
class Estimator {
public:
    /// Estimate the optimal build configuration from a data source. Samples
    /// the source (seek-based random sample of kEstimateSampleTarget vectors),
    /// builds one or more mini-Indices at candidate (R, alpha) settings, and
    /// measures graph topology (R̄, clustering, dead-ends) and search quality
    /// (recall + proximity at target_topk) to pick the cheapest config that
    /// meets the user's quality gate.
    ///
    /// Returns the resolved params + the measured signals that informed them
    /// (for display via `sextant analyze` / `--explain`).
    EstimateResult estimate_config(VectorSource& source,
                                    const BuildConfig& overrides);

private:
    /// Build a mini-Index in-RAM on `sample` (sample_n × dim floats) at the
    /// given params. Uses Builder friend access for the private pipeline
    /// (pass1/pass2/parallel_construct) — skips the sidecar flush step.
    /// `params` must have pq_m/pq_bits resolved (non-zero) — pass1 requires
    /// explicit PQ config.
    static std::unique_ptr<Index> build_mini_(const float* sample,
                                               uint64_t sample_n, Dim dim,
                                               const ResolvedParams& params);

    /// Measure graph topology (avg degree, dead-end fraction, clustering
    /// coefficient) from the mini-Index's node buffer. LID is NOT computed
    /// here (it comes from truth distances in estimate_config).
    static GraphStats measure_graph_stats_(const Index& mini);

    struct SearchQuality { double recall; double proximity; };
    /// Run production-style search on the mini-Index at L, return mean
    /// recall@k and mean proximity (in-band fraction) against `truth_ids`/
    /// `truth_dists`. Reranks search candidates by true L2sq distance
    /// computed from the FP32 `sample` buffer.
    static SearchQuality measure_search_(
        const Index& mini, const float* sample, uint64_t sample_n, Dim dim,
        const std::vector<std::vector<uint32_t>>& truth_ids,
        const std::vector<std::vector<float>>& truth_dists,
        const std::vector<uint32_t>& qidx,
        uint32_t L, uint32_t k, uint32_t rerank);

    friend class Engine;  // Phase D bridge: Engine::estimate_config delegates.
};

}  // namespace sextant
