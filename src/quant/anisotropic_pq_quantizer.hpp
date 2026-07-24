#pragma once

/// @file anisotropic_pq_quantizer.hpp
/// ScaNN-style anisotropic Product Quantization.
///
/// Subclass of PqQuantizer that overrides train() with ScaNN's anisotropic
/// Lloyd's algorithm. The objective (ScaNN §3, Theorem 3.4):
///
///   ℓ(x, x̃) = η · ‖r∥‖² + ‖r⊥‖²
///
/// where r = x − x̃ decomposed against x's own direction, and
/// η = T² / (1 − T²) ≈ 4.125 at T = 0.2. Parallel quantization error is
/// weighted higher than perpendicular — it's the component that affects
/// dot-product ranking. See docs/plans/metric_per_tier_plan.md Phase 2.
///
/// All hot-path methods (encode, preprocess_query, lut_distance, batch4,
/// code_distance, build_code_lut, serialize, deserialize) are inherited
/// UNCHANGED from PqQuantizer. Only the trained codebook differs. VamanaCore
/// calls the non-virtual hot-path methods monomorphically through PqQuantizer&
/// — zero dispatch overhead.

#include "pq_quantizer.hpp"
#include <sextant/types.hpp>

namespace sextant {

class AnisotropicPqQuantizer : public PqQuantizer {
public:
    /// Construct with the same params as PqQuantizer plus the anisotropy
    /// threshold T (default 0.2 from ScaNN §5). η = T²/(1−T²) ≈ 4.125.
    /// T = 0 ⟹ η = 1 (recovers standard L2 k-means, same as PqQuantizer).
    AnisotropicPqQuantizer(MetricKind metric, Dim dim, uint16_t m,
                           uint8_t bits = 8, float threshold = 0.2f,
                           uint64_t seed = 0xC0DE1234ULL)
        : PqQuantizer(metric, dim, m, bits, seed), threshold_(threshold) {}

    /// ScaNN's anisotropic Lloyd's algorithm. Overrides PqQuantizer::train().
    /// P2.2 skeleton: delegates to the base (standard k-means) so the plumbing
    /// can be validated before the research logic lands in P2.3.
    void train(const float* samples, uint64_t n) override;

    float threshold() const { return threshold_; }

private:
    float threshold_;
};

}  // namespace sextant
