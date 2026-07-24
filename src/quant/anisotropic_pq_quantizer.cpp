// AnisotropicPqQuantizer — ScaNN-style anisotropic PQ training.
//
// P2.2 skeleton: the train() override delegates to PqQuantizer::train()
// (standard k-means) so the plumbing (config flag, Builder construction,
// serialization round-trip, VamanaCore integration via the base class) can be
// validated end-to-end before the actual anisotropic Lloyd's algorithm lands
// in P2.3. The hot path is inherited unchanged from PqQuantizer.

#include "anisotropic_pq_quantizer.hpp"

namespace sextant {

void AnisotropicPqQuantizer::train(const float* samples, uint64_t n) {
    // P2.2: delegate to standard k-means. P2.3 will replace this body with
    // ScaNN's anisotropic Lloyd's algorithm (weighted parallel/perpendicular
    // residual decomposition, closed-form centroid update from Theorem 4.2).
    PqQuantizer::train(samples, n);
}

}  // namespace sextant
