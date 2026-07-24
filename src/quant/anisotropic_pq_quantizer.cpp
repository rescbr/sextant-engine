// AnisotropicPqQuantizer — ScaNN-style anisotropic PQ training.
//
// Implements ScaNN's anisotropic Lloyd's algorithm (§3-4 of the paper).
// Warm-starts from standard k-means (PqQuantizer::train), then refines each
// subspace's codebook with anisotropic rounds that weight parallel quantization
// error higher than perpendicular.

#include "anisotropic_pq_quantizer.hpp"
#include "simd_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <atomic>
#include <vector>

namespace sextant {

namespace {

// ---------------------------------------------------------------------------
// SIMD-optimized kernels for the anisotropic k-means inner loops.
// ---------------------------------------------------------------------------

/// Per-data-vector precomputed constants for the anisotropic assignment step.
/// s_i = ‖x_i‖², beta_i = (η-1)/s_i. Stored interleaved for cache locality.
struct AnisoDataPrecomp {
    float s;     // ‖x‖²
    float beta;  // (η-1)/‖x‖²
};

/// Per-centroid precomputed norm: cnorm_j = ‖c_j‖².

/// Anisotropic assignment: for each data vector, find the centroid minimizing
///   ℓ = s - 2u + cnorm + beta·(s - u)²
/// where u = ⟨x, c⟩, s = ‖x‖², cnorm = ‖c‖², beta = (η-1)/s.
///
/// Uses simd::dot_f32 for the dot products. The post-dot arithmetic is scalar
/// but cheap (5 FLOPs per centroid, vs the dot's ~2·sub_dim).
///
/// Outputs: assign[i] = best centroid index for data vector i.
void aniso_assign(const float* data, uint64_t n, uint32_t dim,
                  const float* centroids, uint32_t k,
                  const AnisoDataPrecomp* precomp,
                  const float* cnorms,
                  uint32_t* assign) {
    for (uint64_t i = 0; i < n; i++) {
        const float* x = data + i * dim;
        const float s = precomp[i].s;
        const float beta = precomp[i].beta;
        float best_loss = std::numeric_limits<float>::infinity();
        uint32_t best_c = 0;
        for (uint32_t c = 0; c < k; c++) {
            const float u = simd::dot_f32(x, centroids + c * dim, dim);
            const float diff = s - u;
            const float loss = s - 2.0f * u + cnorms[c] + beta * diff * diff;
            if (loss < best_loss) {
                best_loss = loss;
                best_c = c;
            }
        }
        assign[i] = best_c;
    }
}

/// Solve a d×d linear system A·x = b using Gauss-Jordan elimination with
/// partial pivoting. d is small (8-24), so scalar is faster than SIMD here.
/// Modifies A and b in place. Returns false if singular.
bool solve_linear_system(float* A, float* b, uint32_t d) {
    for (uint32_t col = 0; col < d; col++) {
        // Partial pivot: find the row with the largest |A[row][col]|.
        uint32_t pivot = col;
        float max_abs = std::fabs(A[col * d + col]);
        for (uint32_t row = col + 1; row < d; row++) {
            const float v = std::fabs(A[row * d + col]);
            if (v > max_abs) {
                max_abs = v;
                pivot = row;
            }
        }
        if (max_abs < 1e-12f) return false;  // singular

        // Swap pivot row into place.
        if (pivot != col) {
            for (uint32_t j = 0; j < d; j++) {
                std::swap(A[pivot * d + j], A[col * d + j]);
            }
            std::swap(b[pivot], b[col]);
        }

        // Eliminate below and above.
        const float inv_pivot = 1.0f / A[col * d + col];
        for (uint32_t row = 0; row < d; row++) {
            if (row == col) continue;
            const float factor = A[row * d + col] * inv_pivot;
            if (factor == 0.0f) continue;
            for (uint32_t j = col; j < d; j++) {
                A[row * d + j] -= factor * A[col * d + j];
            }
            b[row] -= factor * b[col];
        }
        // Normalize the pivot row.
        for (uint32_t j = col; j < d; j++) {
            A[col * d + j] *= inv_pivot;
        }
        b[col] *= inv_pivot;
    }
    return true;
}

/// Anisotropic update step (ScaNN Theorem 4.2, derived):
/// For each cluster j:
///   M = Σ_{i∈P_j} β_i · x_i · x_iᵀ    (d×d outer-product sum)
///   b = η · Σ_{i∈P_j} x_i              (d-vector; the 1+β·s=η simplification)
///   c_j = (|P_j|·I + M)⁻¹ · b
///
/// The outer-product accumulation uses NEON for the d=8 fast path (8 f32x4
/// vector multiplies per point per cluster). The d×d solve is scalar
/// Gauss-Jordan (d too small for SIMD to win).
///
/// Inputs: data, assign (from the assignment step), precomp (β_i), centroids
/// (warm-started; refined in place). η is the anisotropy weight.
void aniso_update(const float* data, uint64_t n, uint32_t dim,
                  const uint32_t* assign, uint32_t k,
                  const AnisoDataPrecomp* precomp,
                  float* centroids, float eta) {
    const uint32_t d = dim;
    // Per-cluster accumulators: M (d×d), sum_x (d), count.
    std::vector<float> M(size_t(k) * d * d, 0.0f);
    std::vector<float> sum_x(size_t(k) * d, 0.0f);
    std::vector<uint64_t> count(k, 0);

    // Accumulate outer products and sums.
    // For each data point, add β_i·x_i·x_iᵀ to its cluster's M, and x_i to
    // its cluster's sum_x.
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t c = assign[i];
        const float* x = data + i * dim;
        const float beta = precomp[i].beta;
        count[c]++;
        float* M_c = M.data() + size_t(c) * d * d;
        float* sx_c = sum_x.data() + size_t(c) * d;

#if defined(SEXTANT_HAS_NEON)
        if (d == 8) {
            // Fast path: d=8. Each row of the outer product is x[row]*x[0..7].
            // Use f32x4: two halves per row.
            const float32x4_t x0 = vld1q_f32(x);       // x[0..3]
            const float32x4_t x1 = vld1q_f32(x + 4);   // x[4..7]
            // sum_x accumulation (will scale by eta later).
            vst1q_f32(sx_c, vaddq_f32(vld1q_f32(sx_c), x0));
            vst1q_f32(sx_c + 4, vaddq_f32(vld1q_f32(sx_c + 4), x1));
            // Outer product: M[r][j] += beta * x[r] * x[j].
            // Process 4 rows at a time (rows 0-3 use x0 broadcast per lane).
            // Row group 0: scalar x[r] * f32x4 x0 + x1.
            for (uint32_t r = 0; r < 4; r++) {
                const float32x4_t br0 = vdupq_n_f32(beta * x[r]);
                float* Mrow = M_c + r * d;
                vst1q_f32(Mrow, vmlaq_f32(vld1q_f32(Mrow), br0, x0));
                vst1q_f32(Mrow + 4, vmlaq_f32(vld1q_f32(Mrow + 4), br0, x1));
            }
            for (uint32_t r = 4; r < 8; r++) {
                const float32x4_t br0 = vdupq_n_f32(beta * x[r]);
                float* Mrow = M_c + r * d;
                vst1q_f32(Mrow, vmlaq_f32(vld1q_f32(Mrow), br0, x0));
                vst1q_f32(Mrow + 4, vmlaq_f32(vld1q_f32(Mrow + 4), br0, x1));
            }
        } else
#endif
        {
            // Scalar fallback (any d).
            for (uint32_t j = 0; j < d; j++) sx_c[j] += x[j];
            for (uint32_t r = 0; r < d; r++) {
                const float br = beta * x[r];
                float* Mrow = M_c + r * d;
                for (uint32_t j = 0; j < d; j++) {
                    Mrow[j] += br * x[j];
                }
            }
        }
    }

    // Solve for each centroid: c_j = (|P_j|·I + M_j)⁻¹ · (η·sum_x_j).
    for (uint32_t c = 0; c < k; c++) {
        if (count[c] == 0) {
            // Empty cluster: leave the centroid as-is (warm-started).
            continue;
        }
        // Build A = count·I + M (modify a copy of M).
        float* M_c = M.data() + size_t(c) * d * d;
        for (uint32_t r = 0; r < d; r++) {
            M_c[r * d + r] += float(count[c]);
        }
        // b = η · sum_x.
        std::vector<float> b(d);
        const float* sx_c = sum_x.data() + size_t(c) * d;
        for (uint32_t j = 0; j < d; j++) b[j] = eta * sx_c[j];
        // Solve A·c = b. Solution overwrites b.
        if (solve_linear_system(M_c, b.data(), d)) {
            float* cen = centroids + c * d;
            for (uint32_t j = 0; j < d; j++) cen[j] = b[j];
        }
        // If singular, leave the centroid unchanged (rare with warm-start).
    }
}

/// Precompute per-data-vector constants: s_i = ‖x_i‖², β_i = (η-1)/s_i.
/// Uses simd::dot_f32 for the norms. SIMD-accelerated for d=8.
void precompute_aniso(const float* data, uint64_t n, uint32_t dim,
                      float eta, AnisoDataPrecomp* out) {
    const float eta_m1 = eta - 1.0f;
    for (uint64_t i = 0; i < n; i++) {
        const float s = simd::dot_f32(data + i * dim, data + i * dim, dim);
        out[i].s = s;
        out[i].beta = (s > 1e-12f) ? (eta_m1 / s) : 0.0f;
    }
}

/// Compute per-centroid norms: cnorm_j = ‖c_j‖².
void compute_cnorms(const float* centroids, uint32_t k, uint32_t dim,
                    float* out) {
    for (uint32_t c = 0; c < k; c++) {
        out[c] = simd::dot_f32(centroids + c * dim,
                               centroids + c * dim, dim);
    }
}

}  // namespace

void AnisotropicPqQuantizer::train(const float* samples, uint64_t n) {
    // Phase 1: warm-start with standard k-means (delegates to the base).
    // This produces a reasonable codebook; the anisotropic rounds refine it.
    PqQuantizer::train(samples, n);

    // Phase 2: anisotropic Lloyd refinement, per subspace.
    // Each subspace's K centroids are refined independently (same parallel
    // structure as the base train). The assignment + update use ScaNN's
    // anisotropic objective.
    const float T = threshold_;
    const float eta = (T > 0.0f) ? (T * T) / (1.0f - T * T) : 1.0f;
    if (eta <= 1.0f) return;  // T=0 → η=1 → standard k-means, nothing to do.

    // Handle OPQ rotation: if active, the training data is rotated. We need
    // to rotate the samples the same way the base train() did.
    // For simplicity, reuse the same rotation logic. The base train() already
    // applied the rotation and stored it; we access the rotated buffer by
    // re-rotating here if needed.
    // NOTE: for the common case (no OPQ), train_ptr = samples directly.
    const float* train_ptr = samples;
    std::vector<float> rotated;
    // Access the rotation from the base class (it was computed in base train()).
    // We re-apply it to get the rotated training data for the anisotropic rounds.
    // If no rotation, skip.
    // TODO: if OPQ is active, recompute rotated buffer. For now, OPQ + anisotropic
    // is not a supported combination; log a warning and skip the anisotropic rounds.
    if (has_rotation_) {
        // OPQ + anisotropic: skip the anisotropic refinement (rare combo).
        return;
    }

    // Parallelize across subspaces (same structure as base train).
    const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t n_threads = std::min(hw, static_cast<uint32_t>(m_));
    std::atomic<uint32_t> next_seg{0};

    auto worker = [&]() {
        std::vector<float> sub_buffer(size_t(n) * sub_dim_);
        std::vector<AnisoDataPrecomp> precomp(n);
        std::vector<float> cnorms(K_);
        std::vector<uint32_t> assign(n);

        uint32_t s;
        while ((s = next_seg.fetch_add(1, std::memory_order_relaxed)) < m_) {
            // Gather sub-vectors for this segment.
            for (uint64_t i = 0; i < n; i++) {
                std::memcpy(sub_buffer.data() + i * sub_dim_,
                            train_ptr + i * dim_ + s * sub_dim_,
                            sub_dim_ * sizeof(float));
            }

            float* book_out = codebook_.data() + size_t(s) * K_ * sub_dim_;

            // Precompute per-data-vector constants.
            precompute_aniso(sub_buffer.data(), n, sub_dim_, eta,
                             precomp.data());

            // Anisotropic Lloyd iterations.
            constexpr uint32_t kAnisoIters = 15;
            for (uint32_t iter = 0; iter < kAnisoIters; iter++) {
                // Recompute centroid norms (centroids change each iteration).
                compute_cnorms(book_out, K_, sub_dim_, cnorms.data());

                // Assignment step.
                aniso_assign(sub_buffer.data(), n, sub_dim_,
                             book_out, K_, precomp.data(),
                             cnorms.data(), assign.data());

                // Update step.
                aniso_update(sub_buffer.data(), n, sub_dim_,
                             assign.data(), K_, precomp.data(),
                             book_out, eta);
            }
        }
    };

    std::vector<std::thread> pool;
    for (uint32_t t = 0; t < n_threads; t++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    // Recompute the derived data (cross-distance table, centroid norms,
    // packed codebook) since the codebook changed.
    build_cross_distance_table();
    compute_centroid_sqnorms_();
}

}  // namespace sextant
