// PQ quantizer — full implementation.
// Evolved from Sextant's exploratory phase; DuckDB dependencies stripped,
// simsimd → NumKong adaptation.

#include "pq_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/error.hpp"

// NOTE on NumKong vs simd_kernels.hpp:
// NumKong's `_f32` kernels (nk_sqeuclidean_f32, nk_dot_f32, nk_dots_packed_f32)
// accumulate in f64 (output nk_f64_t) for bit-for-bit reference matching —
// 2-wide on NEON, half the f32 throughput. For PQ LUT construction and rerank
// we don't need that precision, so this file uses the hand-written f32 kernels
// from simd_kernels.hpp (~2-3x faster at our shapes). NumKong remains linked
// for breadth — exotic dtypes (bf16/e4m3), f64-accurate numerical work, and
// any future kernel we haven't hand-coded. Don't re-add <numkong/numkong.h>
// here unless a specific nk_ call is being introduced.

#include <algorithm>
#include <atomic>
#include <numeric>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

// SVE2 detection disabled: arm_sve.h has compatibility issues on some
// clang 18 / Linux distros. The SVE path only accelerates
// lut_distance_batch4 (not needed for scalar_lloydmax). NEON is always
// available on aarch64 and is sufficient for all current kernels.
// Re-enable SVE when the header compatibility issue is resolved.
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SEXTANT_HAS_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define SEXTANT_HAS_AVX2 1
#endif

namespace sextant {

namespace {

// NOTE: simd::l2sq_f32 and simd::dot_f32 come from simd_kernels.hpp (f32-accumulated,
// ~2x faster than NumKong's f64-accumulating variants). Used by training,
// encode, and the preprocess_query fallback paths.

/// Extract centroid id for slot `s` from a packed code.
inline uint32_t read_code(const uint8_t* code, uint8_t bits, uint32_t s) {
    if (bits == 8) {
        return static_cast<uint32_t>(code[s]);
    }
    // bits == 4 — two codes per byte, slot 0 = low nibble, slot 1 = high.
    const uint32_t byte_off = s / 2;
    const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
    return (static_cast<uint32_t>(code[byte_off]) >> shift) & 0x0Fu;
}

/// Write centroid id for slot `s` into a packed code.
inline void write_code(uint8_t* code, uint8_t bits, uint32_t s, uint32_t cid) {
    if (bits == 8) {
        code[s] = static_cast<uint8_t>(cid & 0xFFu);
        return;
    }
    const uint32_t byte_off = s / 2;
    const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
    const uint8_t nibble = static_cast<uint8_t>(cid & 0x0Fu);
    code[byte_off] = static_cast<uint8_t>(
        (code[byte_off] & ~(0x0Fu << shift)) | (nibble << shift));
}

/// Assign a float sub-vector to its nearest centroid in a codebook slot.
/// Always uses L2 for centroid assignment (better PQ approximation).
inline uint32_t nearest_centroid(const float* codebook, uint32_t s,
                                 const float* sub, uint32_t K,
                                 uint32_t sub_dim) {
    const float* slot_book = codebook + size_t(s) * K * sub_dim;
    uint32_t best = 0;
    float best_d = std::numeric_limits<float>::infinity();
    for (uint32_t c = 0; c < K; c++) {
        const float d = simd::l2sq_f32(sub, slot_book + c * sub_dim, sub_dim);
        if (d < best_d) {
            best_d = d;
            best = c;
        }
    }
    return best;
}

/// k-means++ seeding: pick the first centroid uniformly at random, then each
/// subsequent one with probability proportional to D(x)^2 (distance to the
/// nearest already-chosen centroid).
void kmeans_pp_seed(const float* data, uint64_t n, uint32_t dim, uint32_t k,
                    std::mt19937_64& rng, float* centroids_out) {
    std::vector<float> min_d2(n, std::numeric_limits<float>::infinity());

    // First centroid: uniform.
    std::uniform_int_distribution<uint64_t> first(0, n - 1);
    const uint64_t i0 = first(rng);
    std::memcpy(centroids_out, data + i0 * dim, dim * sizeof(float));

    for (uint32_t c = 1; c < k; c++) {
        const float* prev = centroids_out + (c - 1) * dim;
        double sum = 0.0;
        for (uint64_t i = 0; i < n; i++) {
            const float d = simd::l2sq_f32(data + i * dim, prev, dim);
            if (d < min_d2[i]) {
                min_d2[i] = d;
            }
            sum += min_d2[i];
        }

        if (sum <= 0.0) {
            // All remaining points coincide with an existing centroid —
            // fall back to a random sample to fill up to k.
            std::uniform_int_distribution<uint64_t> u(0, n - 1);
            std::memcpy(centroids_out + c * dim, data + u(rng) * dim,
                        dim * sizeof(float));
            continue;
        }

        std::uniform_real_distribution<double> pick(0.0, sum);
        double r = pick(rng);
        uint64_t chosen = n - 1;
        for (uint64_t i = 0; i < n; i++) {
            r -= min_d2[i];
            if (r <= 0.0) {
                chosen = i;
                break;
            }
        }
        std::memcpy(centroids_out + c * dim, data + chosen * dim,
                    dim * sizeof(float));
    }
}

/// k-means++ initialization + Lloyd iterations. Standard Lloyd's algorithm
/// with early stop at <1% reassignment. Output written to `centroids_out`
/// (k x dim floats, preallocated).
///
/// Assignment step uses the FAISS GEMM decomposition: distance(a,b) =
/// ||a||² + ||b||² − 2<a,b>. We precompute ||a||² once (data is invariant),
/// ||b||² per iter (centroids change), and the <a,b> matrix via
/// `simd::gemv_f32`. This replaces the per-pair l2sq_f32 inner loop and is
/// ~5-10× faster on the assignment step (the dominant cost).

/// Reusable scratch for kmeans_pp. Allocated once per worker thread and
/// passed to every kmeans_pp call to avoid re-allocating the dots matrix
/// (n × k_simd × sizeof(float) — up to 256MB at n=256K, k_simd=256) on
/// each of the m=96 PQ subspace calls. Hoisting this out dropped kmeans_pp
/// allocation overhead from ~12% to ~0% of build time.
struct KmeansScratch {
    std::vector<float> centroids_padded;  // k_simd × dim
    std::vector<float> dots;              // n × k_simd
    std::vector<float> cnorm_sq;          // k_simd
    std::vector<float> xnorm_sq;          // n

    /// Resize for a given (n, k_simd, dim). No-op if already the right size.
    void resize(uint64_t n, uint32_t k_simd, uint32_t dim) {
        if (centroids_padded.size() != size_t(k_simd) * dim) {
            centroids_padded.assign(size_t(k_simd) * dim, 0.0f);
        }
        if (dots.size() != size_t(n) * k_simd) {
            dots.assign(size_t(n) * k_simd, 0.0f);
        }
        if (cnorm_sq.size() != k_simd) cnorm_sq.assign(k_simd, 0.0f);
        if (xnorm_sq.size() != n) xnorm_sq.assign(n, 0.0f);
    }
};

void kmeans_pp(const float* data, uint64_t n, uint32_t dim, uint32_t k,
               uint64_t seed, uint32_t max_iters, float* centroids_out,
               KmeansScratch* scratch = nullptr) {
    if (k == 0 || dim == 0) {
        throw Error(ErrorCode::InvalidParam, "k-means: dim and k must be positive");
    }
    if (n == 0) {
        std::memset(centroids_out, 0, size_t(k) * dim * sizeof(float));
        return;
    }

    std::mt19937_64 rng(seed);

    if (n < k) {
        // Can't make more unique centroids than points — repeat samples.
        for (uint32_t c = 0; c < k; c++) {
            std::memcpy(centroids_out + c * dim,
                        data + (c % n) * dim, dim * sizeof(float));
        }
    } else {
        kmeans_pp_seed(data, n, dim, k, rng, centroids_out);
    }

    std::vector<uint32_t> assign(n, 0);
    std::vector<double> sum_vec(size_t(k) * dim, 0.0);
    std::vector<uint64_t> count(k, 0);

    // GEMV scratch: padded centroids + dots matrix. Use the caller-provided
    // scratch if available (avoid 96× re-allocation across PQ subspaces);
    // otherwise allocate locally (for one-off callers / tests).
    const uint32_t k_simd = simd::gemv_k_simd(k);
    KmeansScratch local_scratch;
    KmeansScratch& s = scratch ? *scratch : local_scratch;
    s.resize(n, k_simd, dim);

    // ||data[i||² — invariant across iterations, compute once.
    for (uint64_t i = 0; i < n; i++) {
        s.xnorm_sq[i] = simd::dot_f32(data + i * dim, data + i * dim, dim);
    }

    // Initialize padded centroids from centroids_out (copy k rows; padding
    // rows zero-filled by resize()).
    auto sync_centroids_padded = [&]() {
        std::memcpy(s.centroids_padded.data(), centroids_out,
                    size_t(k) * dim * sizeof(float));
        // Padding rows [k, k_simd) stay zero — gemv_f32 writes the extra
        // dots columns but the caller never reads them.
    };
    sync_centroids_padded();

    for (uint32_t iter = 0; iter < max_iters; iter++) {
        // 1) Assign step via GEMV decomposition.
        //    d(a,c) = ||a||² + ||c||² − 2<a,c>; argmin over c.
        //    a) Per-iter: recompute ||c||² for real centroids (padding rows
        //       stay zero; their "distance" contribution is xnorm_sq[i] +
        //       0 − 0 = xnorm_sq[i], which the argmin will not pick as long
        //       as some real centroid is closer — always true since the
        //       closest real centroid gives d ≤ xnorm_sq[i]).
        for (uint32_t c = 0; c < k; c++) {
            s.cnorm_sq[c] = simd::dot_f32(
                s.centroids_padded.data() + c * dim,
                s.centroids_padded.data() + c * dim, dim);
        }
        //    b) GEMV: dots[i*k_simd + c] = <data[i], centroids[c]>.
        simd::gemv_f32(data, static_cast<uint32_t>(n),
                       s.centroids_padded.data(), k, k_simd, dim,
                       s.dots.data());
        //    c) Argmin per row, SIMD over the k dimension.
        //    d(a,c) = ||a||² + ||c||² − 2<a,b>; since ||a||² is constant
        //    across c for a given row, drop it from the argmin: minimize
        //    (cnorm_sq[c] - 2*dots[i*k_simd+c]). Vectorize 4 lanes at a time
        //    (NEON) or 8 (AVX2); track both min value and its index.
        uint64_t changed = 0;
        simd::argmin_scaled(static_cast<uint32_t>(n), k, k_simd,
                            s.cnorm_sq.data(), s.dots.data(),
                            assign.data(), changed);

        // 2) Recompute centroids as the mean of their assigned points.
        std::fill(sum_vec.begin(), sum_vec.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);
        for (uint64_t i = 0; i < n; i++) {
            const uint32_t c = assign[i];
            count[c]++;
            const float* v = data + i * dim;
            double* acc = sum_vec.data() + size_t(c) * dim;
            for (uint32_t d = 0; d < dim; d++) {
                acc[d] += v[d];
            }
        }
        for (uint32_t c = 0; c < k; c++) {
            if (count[c] > 0) {
                double inv = 1.0 / double(count[c]);
                float* cen = centroids_out + c * dim;
                double* acc = sum_vec.data() + size_t(c) * dim;
                for (uint32_t d = 0; d < dim; d++) {
                    cen[d] = float(acc[d] * inv);
                }
            } else {
                // Empty cluster: reseed from the point farthest from its
                // assigned centroid. Distance is read from the GEMV output
                // (no extra distance compute).
                float worst = -1.0f;
                uint64_t worst_i = 0;
                for (uint64_t i = 0; i < n; i++) {
                    const float d = s.xnorm_sq[i] + s.cnorm_sq[assign[i]]
                                  - 2.0f * s.dots[i * k_simd + assign[i]];
                    if (d > worst) {
                        worst = d;
                        worst_i = i;
                    }
                }
                std::memcpy(centroids_out + c * dim,
                            data + worst_i * dim, dim * sizeof(float));
            }
        }
        sync_centroids_padded();

        // Early stop: <1% of points changed assignment.
        if (iter > 0 && changed * 100 < n) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Covariance-based anisotropic PQ training (scale-transform k-means)
// ---------------------------------------------------------------------------
//
// The previous prototype used a per-vector direction proxy (ScaNN-style
// d = ||x-c||² + λ·((x-c)·x̂)²), which *degraded* recall on arxiv100k. The
// data is genuinely anisotropic (top-10 dims carry 30.7% of variance), so the
// failure was in the formulation, not the premise.
//
// The correct formulation weights each *dimension* by its variance, using the
// subspace covariance. The anisotropic distance:
//
//     d_aniso(x, c) = Σ_d  w_d · (x_d - c_d)²
//
// is exactly standard L2sq in a scaled space where x'_d = √w_d · x_d. So we
// scale the sub-vectors, run the *unchanged* `kmeans_pp` on the scaled data,
// then unscale the centroids by dividing by √w_d. No new distance function,
// no SIMD changes — only data preprocessing.
//
// The weights come from the covariance eigenvalues:
//     w_d = eigenvalue_d / mean(eigenvalues)
// normalized to mean=1 (preserves overall distance scale), clamped to
// [0.1, 10] to avoid degenerate weights on near-zero-variance dims.

/// Jacobi eigenvalue algorithm for a small symmetric `dim`×`dim` matrix.
///
/// Computes all eigenvalues (and optionally eigenvectors) of a real symmetric
/// matrix via cyclic Jacobi rotations. Stable, ~40 lines, no external deps.
/// Cost is O(dim³) per sweep with ~5-10 sweeps to convergence — negligible at
/// sub_dim ≤ 16.
///
/// `a` is row-major `dim`×`dim` (modified in place to the diagonal form).
/// `eigvals` receives the `dim` eigenvalues (unordered). `eigvecs` (may be
/// null) receives the `dim`×`dim` eigenvector matrix as columns (row-major:
/// eigvecs[d*dim + j] is the d-th component of eigenvector j).
void jacobi_eigen(std::vector<double>& a, uint32_t dim,
                  std::vector<double>& eigvals,
                  std::vector<double>* eigvecs = nullptr) {
    // Initialize eigvecs to identity (if requested).
    if (eigvecs) {
        eigvecs->assign(size_t(dim) * dim, 0.0);
        for (uint32_t i = 0; i < dim; i++) (*eigvecs)[i * dim + i] = 1.0;
    }

    for (uint32_t sweep = 0; sweep < 50; sweep++) {
        // Sum of off-diagonal magnitudes — convergence criterion.
        double off = 0.0;
        for (uint32_t p = 0; p < dim; p++) {
            for (uint32_t q = p + 1; q < dim; q++) {
                off += std::fabs(a[p * dim + q]);
            }
        }
        if (off < 1e-12) break;

        for (uint32_t p = 0; p < dim; p++) {
            for (uint32_t q = p + 1; q < dim; q++) {
                const double apq = a[p * dim + q];
                if (std::fabs(apq) < 1e-15) continue;
                const double app = a[p * dim + p];
                const double aqq = a[q * dim + q];
                // Rotation angle: tan(2θ) = 2·apq / (app - aqq).
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (std::fabs(theta) > 1e15) {
                    t = 1.0 / (2.0 * theta);  // ~0 for huge denominator
                } else {
                    t = (theta >= 0.0 ? 1.0 : -1.0) /
                        (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                }
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                // Apply rotation to A: zero out a[p,q] and a[q,p].
                a[p * dim + p] = app - t * apq;
                a[q * dim + q] = aqq + t * apq;
                a[p * dim + q] = 0.0;
                a[q * dim + p] = 0.0;
                for (uint32_t i = 0; i < dim; i++) {
                    if (i == p || i == q) continue;
                    const double aip = a[i * dim + p];
                    const double aiq = a[i * dim + q];
                    a[i * dim + p] = c * aip - s * aiq;
                    a[p * dim + i] = a[i * dim + p];
                    a[i * dim + q] = s * aip + c * aiq;
                    a[q * dim + i] = a[i * dim + q];
                }
                // Accumulate eigenvectors: V = V · R(p,q,θ).
                if (eigvecs) {
                    for (uint32_t i = 0; i < dim; i++) {
                        const double vip = (*eigvecs)[i * dim + p];
                        const double viq = (*eigvecs)[i * dim + q];
                        (*eigvecs)[i * dim + p] = c * vip - s * viq;
                        (*eigvecs)[i * dim + q] = s * vip + c * viq;
                    }
                }
            }
        }
    }

    eigvals.resize(dim);
    for (uint32_t i = 0; i < dim; i++) eigvals[i] = a[i * dim + i];
}

/// Compute per-dimension anisotropic scale factors `√w_d` for one subspace.
///
/// `sub_buffer` is `n × sub_dim` row-major sub-vectors. Returns `scale`
/// (length `sub_dim`) where `scale[d] = √w_d`, `w_d = eigval_d / mean(eigval)`
/// clamped to [0.1, 10]. On failure (e.g. zero covariance), returns all 1.0
/// (identity → standard k-means).
std::vector<float> anisotropic_scale(const float* sub_buffer, uint64_t n,
                                     uint32_t sub_dim) {
    std::vector<float> scale(sub_dim, 1.0f);
    if (n < 2 || sub_dim == 0) return scale;

    // Mean of sub-vectors.
    std::vector<double> mean(sub_dim, 0.0);
    for (uint64_t i = 0; i < n; i++) {
        const float* sv = sub_buffer + i * sub_dim;
        for (uint32_t d = 0; d < sub_dim; d++) mean[d] += sv[d];
    }
    for (uint32_t d = 0; d < sub_dim; d++) mean[d] /= static_cast<double>(n);

    // Covariance (sub_dim × sub_dim).
    std::vector<double> cov(size_t(sub_dim) * sub_dim, 0.0);
    for (uint64_t i = 0; i < n; i++) {
        const float* sv = sub_buffer + i * sub_dim;
        for (uint32_t a = 0; a < sub_dim; a++) {
            const double da = static_cast<double>(sv[a]) - mean[a];
            for (uint32_t b = a; b < sub_dim; b++) {
                const double db = static_cast<double>(sv[b]) - mean[b];
                cov[a * sub_dim + b] += da * db;
            }
        }
    }
    const double inv_df = 1.0 / static_cast<double>(n - 1);
    for (uint32_t a = 0; a < sub_dim; a++) {
        cov[a * sub_dim + a] *= inv_df;
        for (uint32_t b = a + 1; b < sub_dim; b++) {
            cov[a * sub_dim + b] *= inv_df;
            cov[b * sub_dim + a] = cov[a * sub_dim + b];  // symmetric
        }
    }

    // Eigenvalues of the symmetric covariance.
    std::vector<double> eigvals;
    jacobi_eigen(cov, sub_dim, eigvals, /*eigvecs=*/nullptr);

    double mean_eig = 0.0;
    for (uint32_t d = 0; d < sub_dim; d++) {
        // Numerical safety: covariance is PSD, but tiny negative eigenvalues
        // can appear from roundoff — clamp to 0.
        eigvals[d] = std::max(eigvals[d], 0.0);
        mean_eig += eigvals[d];
    }
    mean_eig /= static_cast<double>(sub_dim);
    if (mean_eig < 1e-12) return scale;  // degenerate: leave as identity

    for (uint32_t d = 0; d < sub_dim; d++) {
        double w = eigvals[d] / mean_eig;
        w = std::clamp(w, 0.1, 10.0);
        scale[d] = static_cast<float>(std::sqrt(w));
    }
    return scale;
}

/// Compute the OPQ PCA rotation R (dim × dim, row-major float) from a training
/// sample.
///
/// `samples` is `n × dim` row-major. Returns R = V^T where C = V Λ V^T is the
/// eigendecomposition of the sample covariance. In the rotated basis (R·x),
/// dimensions are decorrelated and ordered by eigenvalue, which aligns the PQ
/// dimension-split with the principal components. On failure (degenerate
/// covariance, n < 2), returns an empty vector (caller treats as no rotation).
std::vector<float> compute_pca_rotation(const float* samples, uint64_t n,
                                        uint32_t dim) {
    if (n < 2 || dim == 0) return {};
    // Mean.
    std::vector<double> mean(dim, 0.0);
    for (uint64_t i = 0; i < n; i++) {
        const float* v = samples + i * dim;
        for (uint32_t d = 0; d < dim; d++) mean[d] += v[d];
    }
    for (uint32_t d = 0; d < dim; d++) mean[d] /= static_cast<double>(n);

    // Covariance (upper triangle, then mirror).
    std::vector<double> cov(size_t(dim) * dim, 0.0);
    for (uint64_t i = 0; i < n; i++) {
        const float* v = samples + i * dim;
        for (uint32_t a = 0; a < dim; a++) {
            const double da = static_cast<double>(v[a]) - mean[a];
            for (uint32_t b = a; b < dim; b++) {
                const double db = static_cast<double>(v[b]) - mean[b];
                cov[a * dim + b] += da * db;
            }
        }
    }
    const double inv_df = 1.0 / static_cast<double>(n - 1);
    bool nonzero = false;
    for (uint32_t a = 0; a < dim; a++) {
        cov[a * dim + a] *= inv_df;
        if (cov[a * dim + a] > 1e-15) nonzero = true;
        for (uint32_t b = a + 1; b < dim; b++) {
            cov[a * dim + b] *= inv_df;
            cov[b * dim + a] = cov[a * dim + b];
        }
    }
    if (!nonzero) return {};  // degenerate

    std::vector<double> eigvals, eigvecs;
    jacobi_eigen(cov, dim, eigvals, &eigvecs);

    // Build rotation R = V^T. jacobi_eigen stores eigvecs such that
    // eigvecs[d*dim + j] is the d-th component of eigenvector j — i.e. column j
    // of V is eigenvector j (V[d][j] = eigvecs[d*dim + j]). R = V^T means
    // R[r][c] = V[c][r] = eigvecs[c*dim + r]. Applying R to x projects x onto
    // each eigenvector: rotated[r] = Σ_c V[c][r]·x[c] = eigvec_r · x.
    std::vector<float> rotation(size_t(dim) * dim);
    for (uint32_t r = 0; r < dim; r++) {
        for (uint32_t c = 0; c < dim; c++) {
            rotation[r * dim + c] = static_cast<float>(eigvecs[c * dim + r]);
        }
    }
     return rotation;
 }

}  // namespace

std::vector<float> compute_pca_rotation_public(const float* samples, uint64_t n,
                                                uint32_t dim,
                                                std::vector<double>* eigvals_out) {
    if (n < 2 || dim == 0) return {};

    // --- Randomized SVD (Halko et al.) for top-k principal components ---
    // Instead of forming the full dim×dim covariance and eigendecomposing
    // (O(dim³) Jacobi = ~20s for dim=768), we use randomized projection:
    //   1. Center the data
    //   2. Project onto k random Gaussian directions → Y (n × k)
    //   3. QR-orthonormalize Y → Q (dim × k)
    //   4. Form B = Q^T Cov Q (k × k — tiny)
    //   5. Eigendecompose B (k × k — milliseconds)
    //   6. Eigenvectors of Cov ≈ Q × eigvecs(B)
    // Cost: O(n × dim × k) for projection, O(k³) for eigendecomposition.

    const uint32_t k = std::min(std::min(dim, 128u), static_cast<uint32_t>(n));
    const uint32_t p = std::min(dim, k + 16);  // oversample for accuracy

    // Mean.
    std::vector<double> mean(dim, 0.0);
    for (uint64_t i = 0; i < n; i++) {
        const float* v = samples + i * dim;
        for (uint32_t d = 0; d < dim; d++) mean[d] += v[d];
    }
    for (uint32_t d = 0; d < dim; d++) mean[d] /= static_cast<double>(n);

    // Random Gaussian matrix Ω (dim × p), seed for reproducibility.
    std::mt19937_64 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> omega(dim * p);
    for (auto& v : omega) v = gauss(rng);

    // Y = (X - mean) × Ω → n × p matrix.
    // Y[i][j] = Σ_d (x[i][d] - mean[d]) × omega[d][j]
    std::vector<double> Y(n * p, 0.0);
    for (uint64_t i = 0; i < n; i++) {
        const float* xi = samples + i * dim;
        for (uint32_t j = 0; j < p; j++) {
            double acc = 0.0;
            for (uint32_t d = 0; d < dim; d++)
                acc += (static_cast<double>(xi[d]) - mean[d]) * omega[d * p + j];
            Y[i * p + j] = acc;
        }
    }

    // QR via modified Gram-Schmidt on Y^T (p × n → orthonormal rows).
    // Q is stored as p × dim: Q[j][d] = orthogonalized basis vector j component d.
    // Actually, we need Q from Y^T where Y^T is p × dim... wait.
    // Y = X_c × Ω is n × p. We need the column space of Y, which lives in R^dim.
    // The column space of Y = column space of X_c × Ω ⊆ column space of X_c.
    // Y^T is p × n. We orthonormalize the COLUMNS of Y (n-dimensional vectors).
    // Q (n × p) has orthonormal columns spanning the same space as Y.
    // Then B = Q^T × X_c × X_c^T × Q (p × p) — but that's n-expensive.
    //
    // Simpler: compute the p × p Gram matrix G = Y^T × Y (p × p).
    // Eigendecompose G. The eigenvectors of the covariance restricted to the
    // subspace are: principal_dirs = Ω × eigvecs(G). Then renormalize.
    //
    // Actually even simpler (direct randomized PCA):
    // Y = X_c × Ω (n × p). Compute Y^T × Y = p × p (the projected covariance).
    // Eigendecompose the p × p matrix. Eigenvalues of Y^T Y / (n-1) ≈ top
    // eigenvalues of the covariance. Principal directions = Ω × eigvecs.
    // Then renormalize principal directions to unit length.

    // G = Y^T × Y / (n-1) → p × p symmetric.
    std::vector<double> G(p * p, 0.0);
    for (uint64_t i = 0; i < n; i++) {
        for (uint32_t a = 0; a < p; a++) {
            const double ya = Y[i * p + a];
            for (uint32_t b = a; b < p; b++)
                G[a * p + b] += ya * Y[i * p + b];
        }
    }
    const double inv_df = 1.0 / static_cast<double>(n - 1);
    for (uint32_t a = 0; a < p; a++) {
        G[a * p + a] *= inv_df;
        for (uint32_t b = a + 1; b < p; b++) {
            G[a * p + b] *= inv_df;
            G[b * p + a] = G[a * p + b];
        }
    }

    // Eigendecompose G (p × p — fast, Jacobi on ~80×80 takes <1ms).
    std::vector<double> geigvals, geigvecs;
    jacobi_eigen(G, p, geigvals, &geigvecs);
    geigvals.resize(p);
    for (uint32_t i = 0; i < p; i++) geigvals[i] = G[i * p + i];

    // Sort by descending eigenvalue.
    std::vector<uint32_t> order(p);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return geigvals[a] > geigvals[b]; });

    if (eigvals_out) {
        eigvals_out->resize(dim);
        for (uint32_t i = 0; i < p; i++)
            (*eigvals_out)[i] = geigvals[order[i]];
        for (uint32_t i = p; i < dim; i++)
            (*eigvals_out)[i] = 0.0;  // unknown tail
    }

    // Principal directions: direction[j][d] = Σ_k omega[d][k] × geigvecs[k][order[j]]
    // Then renormalize to unit length.
    std::vector<float> rotation(size_t(dim) * dim, 0.0f);
    for (uint32_t j = 0; j < k; j++) {
        const uint32_t src = order[j];
        double norm2 = 0.0;
        for (uint32_t d = 0; d < dim; d++) {
            double val = 0.0;
            for (uint32_t c = 0; c < p; c++)
                val += omega[d * p + c] * geigvecs[c * p + src];
            rotation[j * dim + d] = static_cast<float>(val);
            norm2 += val * val;
        }
        // Renormalize.
        if (norm2 > 1e-20) {
            const float inv_norm = 1.0f / static_cast<float>(std::sqrt(norm2));
            for (uint32_t d = 0; d < dim; d++)
                rotation[j * dim + d] *= inv_norm;
        }
    }
    // Fill remaining rows (k..dim) with zeros — not used for routing.
    return rotation;
}


// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PqQuantizer::PqQuantizer(MetricKind metric, Dim dim, uint16_t m, uint8_t bits,
                         uint64_t seed)
    : metric_(metric), dim_(dim), m_(m), bits_(bits), seed_(seed) {
    if (m_ == 0) {
        throw Error(ErrorCode::InvalidParam, "PQ 'm' must be > 0");
    }
    if (dim_ % static_cast<Dim>(m_) != 0) {
        throw Error(ErrorCode::InvalidParam, "PQ requires dim divisible by m");
    }
    if (bits_ != 4 && bits_ != 8) {
        throw Error(ErrorCode::InvalidParam, "PQ 'bits' must be 4 or 8");
    }

    K_ = 1u << bits_;
    sub_dim_ = dim_ / m_;
    codebook_.resize(static_cast<size_t>(m_) * K_ * sub_dim_, 0.0f);
}

uint32_t PqQuantizer::code_size() const {
    return (static_cast<uint32_t>(m_) * bits_ + 7) / 8;
}

uint32_t PqQuantizer::lut_size() const {
    return static_cast<uint32_t>(m_) * K_;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

void PqQuantizer::train(const float* samples, uint64_t n) {
    if (n == 0) {
        throw Error(ErrorCode::InvalidParam, "PQ train: no samples provided");
    }
    codebook_.assign(static_cast<size_t>(m_) * K_ * sub_dim_, 0.0f);

    // OPQ: compute the d×d PCA rotation from the training sample and apply it
    // before PQ k-means. The rotation decorrelates dimensions so the PQ split
    // aligns with the data's principal components (lower reconstruction MSE).
    // The same R is applied at encode/search time, so all of PQ operates in
    // the rotated basis. `train_ptr`/`train_n` point at the data k-means sees.
    const float* train_ptr = samples;
    std::vector<float> rotated;
    if (opq_enabled_ && !has_rotation_) {
        rotation_ = compute_pca_rotation(samples, n, dim_);
        has_rotation_ = !rotation_.empty();
    }
    if (has_rotation_) {
        const uint32_t dim = dim_;
        const float* R = rotation_.data();
        rotated.resize(size_t(n) * dim);
        // rotated[i][r] = Σ_c R[r*dim + c] · samples[i][c]  (one dot per row).
        // Parallelized across vectors (each is dim independent simd::dot_f32 calls).
        const uint32_t hw_rot = num_threads_ > 0
            ? num_threads_
            : std::max(1u, std::thread::hardware_concurrency());
        const uint32_t n_threads_rot = std::min(hw_rot, static_cast<uint32_t>(n));
        std::atomic<uint64_t> next{0};
        auto worker = [&]() {
            uint64_t i;
            while ((i = next.fetch_add(1, std::memory_order_relaxed)) < n) {
                const float* src = samples + i * dim;
                float* dst = rotated.data() + i * dim;
                for (uint32_t r = 0; r < dim; r++) {
                    dst[r] = simd::dot_f32(R + r * dim, src, dim);
                }
            }
        };
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < n_threads_rot; t++) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
        train_ptr = rotated.data();
    }

    // Each segment's k-means++ is fully independent (disjoint codebook
    // region, disjoint sub_buffer). Parallelize across segments.
    const uint32_t hw = num_threads_ > 0
        ? num_threads_
        : std::max(1u, std::thread::hardware_concurrency());
    const uint32_t n_threads = std::min(hw, static_cast<uint32_t>(m_));

    std::atomic<uint32_t> next_seg{0};
    auto worker = [&]() {
        std::vector<float> sub_buffer(size_t(n) * sub_dim_);
        std::vector<float> scaled_buffer;  // reused across segments if aniso
        KmeansScratch km_scratch;  // reused across segments — avoids 256MB
                                   // re-allocation per subspace k-means call.
        if (anisotropic_) {
            scaled_buffer.resize(size_t(n) * sub_dim_);
        }
        uint32_t s;
        while ((s = next_seg.fetch_add(1, std::memory_order_relaxed)) < m_) {
            // Gather sub-vectors for this segment.
            for (uint64_t i = 0; i < n; i++) {
                std::memcpy(sub_buffer.data() + i * sub_dim_,
                            train_ptr + i * dim_ + s * sub_dim_,
                            sub_dim_ * sizeof(float));
            }
            const uint64_t slot_seed =
                seed_ ^ (0x9E3779B97F4A7C15ULL * (uint64_t(s) + 1));
            float* book_out = codebook_.data() + size_t(s) * K_ * sub_dim_;
            if (anisotropic_) {
                // Covariance-based scale-transform: scale sub-vectors by √w_d
                // (w_d = eigval_d / mean(eigval), clamped), run standard
                // k-means on the scaled data, then unscale the centroids.
                const std::vector<float> scale = anisotropic_scale(
                    sub_buffer.data(), n, sub_dim_);
                const float* sp = scale.data();
                for (uint64_t i = 0; i < n; i++) {
                    const float* sv = sub_buffer.data() + i * sub_dim_;
                    float* dst = scaled_buffer.data() + i * sub_dim_;
                    for (uint32_t d = 0; d < sub_dim_; d++) {
                        dst[d] = sv[d] * sp[d];
                    }
                }
                kmeans_pp(scaled_buffer.data(), n, sub_dim_, K_, slot_seed,
                          /*max_iters=*/25, book_out, &km_scratch);
                // Unscale centroids back to the original data space.
                for (uint32_t c = 0; c < K_; c++) {
                    float* cen = book_out + size_t(c) * sub_dim_;
                    for (uint32_t d = 0; d < sub_dim_; d++) {
                        cen[d] /= sp[d];
                    }
                }
            } else {
                kmeans_pp(sub_buffer.data(), n, sub_dim_, K_, slot_seed,
                          /*max_iters=*/25, book_out, &km_scratch);
            }
        }
    };

    std::vector<std::thread> pool;
    for (uint32_t t = 0; t < n_threads; t++) {
        pool.emplace_back(worker);
    }
    for (auto& th : pool) th.join();

    build_cross_distance_table();
    compute_centroid_sqnorms_();
}

void PqQuantizer::compute_centroid_sqnorms_() {
    // Per-centroid ||c||² for the fast L2sq path (decomposition via matvec).
    // Layout matches the LUT/centroid iteration order: centroid_sqnorms_[s*K + c].
    // Uses dot_f32(c, c) — l2sq_f32(c, c) would always return 0 (it's ||a-b||²).
    centroid_sqnorms_.assign(static_cast<size_t>(m_) * K_, 0.0f);
    for (uint32_t s = 0; s < m_; s++) {
        const float* book = codebook_.data() + size_t(s) * K_ * sub_dim_;
        float* out = centroid_sqnorms_.data() + size_t(s) * K_;
        for (uint32_t c = 0; c < K_; c++) {
            const float* cen = book + c * sub_dim_;
            out[c] = simd::dot_f32(cen, cen, sub_dim_);
        }
    }
}

void PqQuantizer::build_cross_distance_table() {
    cross_distance_table_.assign(static_cast<size_t>(m_) * K_ * K_, 0.0f);
    // Always use L2SQ for the cross-distance table, regardless of the
    // declared search metric. This mirrors the reference DiskANN approach
    // (Neyshabur-Sreiro transform -> build with L2). For unit-normalized
    // data, L2 and IP rankings are identical, and PQ-L2 has better
    // approximation properties than PQ-IP — measured: building the graph
    // with IP cross-distances drops IP search recall 15pp (0.76→0.61 on
    // arxiv100k) because IP code-to-code distances have higher variance.
    // The L2sq PQ-construct produces a graph robust to both search metrics.
    //
    // Each subspace's table is independent — parallelize across m_.
    const uint32_t hw = num_threads_ > 0
        ? num_threads_
        : std::max(1u, std::thread::hardware_concurrency());
    const uint32_t n_threads = std::min(hw, static_cast<uint32_t>(m_));

    std::atomic<uint32_t> next_seg{0};
    auto worker = [&]() {
        uint32_t s;
        while ((s = next_seg.fetch_add(1, std::memory_order_relaxed)) < m_) {
            const float* book = codebook_.data() + size_t(s) * K_ * sub_dim_;
            float* table = cross_distance_table_.data() + size_t(s) * K_ * K_;
            for (uint32_t a = 0; a < K_; a++) {
                const float* ca = book + a * sub_dim_;
                table[a * K_ + a] = 0.0f;  // L2 diagonal is always 0
                for (uint32_t b = a + 1; b < K_; b++) {
                    const float* cb = book + b * sub_dim_;
                    const float d = simd::l2sq_f32(ca, cb, sub_dim_);
                    table[a * K_ + b] = d;
                    table[b * K_ + a] = d;  // symmetric
                }
            }
        }
    };

    std::vector<std::thread> pool;
    for (uint32_t t = 0; t < n_threads; t++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
}

void PqQuantizer::build_ip_cross_distance_table() const {
    if (!ip_cross_distance_table_.empty()) return;  // already built
    ip_cross_distance_table_.assign(
        static_cast<size_t>(m_) * K_ * K_, 0.0f);

    const uint32_t hw = num_threads_ > 0
        ? num_threads_
        : std::max(1u, std::thread::hardware_concurrency());
    const uint32_t n_threads = std::min(hw, static_cast<uint32_t>(m_));

    std::atomic<uint32_t> next_seg{0};
    auto worker = [&]() {
        uint32_t s;
        while ((s = next_seg.fetch_add(1, std::memory_order_relaxed)) < m_) {
            const float* book = codebook_.data() + size_t(s) * K_ * sub_dim_;
            float* table = ip_cross_distance_table_.data() + size_t(s) * K_ * K_;
            for (uint32_t a = 0; a < K_; a++) {
                const float* ca = book + a * sub_dim_;
                for (uint32_t b = 0; b < K_; b++) {
                    const float* cb = book + b * sub_dim_;
                    // Negative dot product (so smaller = more similar).
                    table[a * K_ + b] = -simd::dot_f32(ca, cb, sub_dim_);
                }
            }
        }
    };

    std::vector<std::thread> pool;
    for (uint32_t t = 0; t < n_threads; t++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
}

float PqQuantizer::code_distance_ip(const uint8_t* code_a,
                                     const uint8_t* code_b) const {
    if (!ip_cross_distance_table_.empty()) {
        float acc = 0.0f;
        const float* table_base = ip_cross_distance_table_.data();
        for (uint32_t s = 0; s < m_; s++) {
            const uint32_t ca = read_code(code_a, bits_, s);
            const uint32_t cb = read_code(code_b, bits_, s);
            acc += table_base[size_t(s) * K_ * K_ + ca * K_ + cb];
        }
        return acc;
    }
    // Fallback: compute from codebook directly.
    float acc = 0.0f;
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t ca = read_code(code_a, bits_, s);
        const uint32_t cb = read_code(code_b, bits_, s);
        const float* book = codebook_.data() + size_t(s) * K_ * sub_dim_;
        acc += -simd::dot_f32(book + ca * sub_dim_, book + cb * sub_dim_,
                              sub_dim_);
    }
    return acc;
}

void PqQuantizer::encode(const float* vec, uint8_t* code_out) const {
    std::memset(code_out, 0, code_size());
    const float* v = vec;
    std::vector<float> rotated;
    if (has_rotation_) {
        // Apply R: rotated[r] = Σ_c R[r*dim + c] · vec[c].
        rotated.resize(dim_);
        const float* R = rotation_.data();
        for (uint32_t r = 0; r < dim_; r++) {
            rotated[r] = simd::dot_f32(R + r * dim_, vec, dim_);
        }
        v = rotated.data();
    }
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t cid =
            nearest_centroid(codebook_.data(), s, v + s * sub_dim_,
                             K_, sub_dim_);
        write_code(code_out, bits_, s, cid);
    }
}

void PqQuantizer::decode_code(const uint8_t* code, float* out) const {
    // Gather segment centroids into `out` (rotated space if OPQ).
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t cid = read_code(code, bits_, s);
        const float* cen = codebook_.data() +
                           static_cast<size_t>(s) * K_ * sub_dim_ +
                           static_cast<size_t>(cid) * sub_dim_;
        std::memcpy(out + s * sub_dim_, cen, sub_dim_ * sizeof(float));
    }
    // Apply inverse rotation R^T (R is orthogonal): out[c] = Σ_r R[r*dim+c]·out[r].
    // Done in place via a temporary to avoid clobbering source mid-transform.
    if (has_rotation_) {
        std::vector<float> rotated(out, out + dim_);
        const float* R = rotation_.data();
        for (uint32_t c = 0; c < dim_; c++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < dim_; r++) {
                acc += R[r * dim_ + c] * rotated[r];
            }
            out[c] = acc;
        }
    }
}

void PqQuantizer::preprocess_query_as(MetricKind metric, const float* query, float* out) const {
    // PQ LUT: out[s * K + c] = d(query_sub_s, centroid[s][c]).
    //   L2SQ: ||q_sub - c||².
    //   IP:   -<q_sub, c>.
    //
    // Two paths:
    // (1) FAST (matvec): if the packed codebook is available, compute <q_sub, c>
    //     for all K centroids in one nk_dots_packed_f32 call per subspace, then
    //     derive LUT entries. ~2× faster than the per-entry path on profiled
    //     hardware (m matvecs vs m*K per-entry calls). L2sq uses the algebraic
    //     decomposition ||q-c||² = ||q||² - 2<q,c> + ||c||² (||c||² cached at
    //     train, ||q||² once per query-sub) — same ranking as the naive path,
    //     ~1e-6 FP32 accumulation difference.
    // (2) FALLBACK (per-entry): the original per-centroid loop. Used before the
    //     packed codebook is built (e.g. untrained quantizer in unit tests) and
    //     when OPQ rotation is active (rotation interacts with packing; not
    //     worth the complexity for the rarely-used OPQ path).
    const float* q = query;
    std::vector<float> rotated;
    if (has_rotation_) {
        rotated.resize(dim_);
        const float* R = rotation_.data();
        for (uint32_t r = 0; r < dim_; r++) {
            rotated[r] = simd::dot_f32(R + r * dim_, query, dim_);
        }
        q = rotated.data();
    }

    // Try the fast matvec path (no rotation, codebook populated). Uses a
    // custom f32-accumulating kernel (`simd::pq_matvec_f32`) purpose-built for the
    // PqQuantizer shape (1 query × K centroids × sub_dim depth). NumKong's
    // `nk_dots_packed_f32` accumulates in f64 (2-wide on NEON, half the f32
    // throughput) for reference-matching precision we don't need; the custom
    // kernel is ~1.8× faster on Apple M4 at K=256, sub_dim=8.
    if (!has_rotation_ && !codebook_.empty()) {
        float dots_stack[256];  // K ≤ 256 at 8-bit
        float* dots = dots_stack;
        for (uint32_t s = 0; s < m_; s++) {
            const float* q_sub = q + s * sub_dim_;
            const float* slot_book = codebook_.data() + size_t(s) * K_ * sub_dim_;
           simd::pq_matvec_f32(q_sub, slot_book, dots, K_, sub_dim_);
            float* row = out + s * K_;
            switch (metric) {
            case MetricKind::InnerProduct:
                for (uint32_t c = 0; c < K_; c++) row[c] = -dots[c];
                break;
            case MetricKind::L2Sq:
            default: {
                // ||q-c||² = ||q||² - 2<q,c> + ||c||². ||c||² cached in
                // centroid_sqnorms_; ||q||² computed once per subspace via
                // dot_f32(q, q) — l2sq_f32(q, q) would be 0 (it's ||a-b||²).
                const float q_sq = simd::dot_f32(q_sub, q_sub, sub_dim_);
                const float* cnorms = centroid_sqnorms_.data() + size_t(s) * K_;
                for (uint32_t c = 0; c < K_; c++) {
                    row[c] = q_sq - 2.0f * dots[c] + cnorms[c];
                }
                break;
            }
            }
        }
        return;
    }

    // Fallback: per-entry loop (used by untrained quantizers in unit tests and
    // when OPQ rotation is active).
    for (uint32_t s = 0; s < m_; s++) {
        const float* q_sub = q + s * sub_dim_;
        const float* slot_book =
            codebook_.data() + size_t(s) * K_ * sub_dim_;
        float* row = out + s * K_;
        for (uint32_t c = 0; c < K_; c++) {
            const float* cen = slot_book + c * sub_dim_;
            switch (metric) {
            case MetricKind::L2Sq:
                row[c] = simd::l2sq_f32(q_sub, cen, sub_dim_);
                break;
            case MetricKind::InnerProduct:
                row[c] = -simd::dot_f32(q_sub, cen, sub_dim_);
                break;
            }
        }
    }
}

void PqQuantizer::preprocess_query(const float* query, float* out) const {
    preprocess_query_as(metric_, query, out);
}

void PqQuantizer::build_fastscan_lut(const float* query,
                                     uint8_t* lut8,
                                     float* scale,
                                     float* offset) const {
    // Build the float LUT into a scratch buffer, then quantize. The float
    // LUT is the same one `preprocess_query` produces; we re-derive it here
    // rather than require the caller to pass it in (keeps the call site at
    // `Searcher::build_query_lut` simple). Cost is m×K×4 bytes scratch +
    // one preprocess_query call per query — negligible vs the search itself.
    //
    // Heap-alloc via std::vector (m×K can be large: m=96/K=256 → 96KB, too
    // large for stack). This is per-query, not per-batch.
    std::vector<float> lut_f32(m_ * K_);
    std::vector<float> seg_min(m_);
    preprocess_query(query, lut_f32.data());
    simd::quantize_lut_u8_scaled(lut_f32.data(), m_, K_, lut8, scale, offset,
                                  seg_min.data());
}

void PqQuantizer::build_fastscan_lut4(const float* query,
                                      uint8_t* lut4,
                                      float* scale_out) const {
    // 4-bit FastScan LUT (Option A scan path). The 4-bit kernel needs K=16
    // entries per segment; misuse catches any other bits_ configuration.
    if (bits_ != 4 || K_ != 16) {
        throw Error(ErrorCode::InvalidParam,
                    "build_fastscan_lut4: requires bits=4 (K=16); got bits=" +
                        std::to_string(bits_) + " (K=" + std::to_string(K_) +
                        "). Use build_fastscan_lut for 8-bit.");
    }
    std::vector<float> lut_f32(static_cast<size_t>(m_) * K_);
    std::vector<float> seg_min(m_);
    preprocess_query(query, lut_f32.data());
    // Use quantize_lut_u4 (scales to 0-15) not quantize_lut_u8 (scales to
    // 0-255). The pq4_block32 kernel accumulates into uint16 internally;
    // with m > 257 (e.g. m=768 scalar Lloyd-Max), 0-255 values overflow
    // uint16 (m × 255 > 65535). The u4 scheme (max m × 15 = 11520 at m=768)
    // is always safe.
    simd::quantize_lut_u4(lut_f32.data(), m_, K_, lut4, scale_out,
                           seg_min.data());
}

// ---------------------------------------------------------------------------
// LUT distance
// ---------------------------------------------------------------------------

float PqQuantizer::lut_distance(const uint8_t* code, const float* lut) const {
    float acc = 0.0f;
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t cid = read_code(code, bits_, s);
        acc += lut[s * K_ + cid];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// PQ code LUT (PQ-construct build mode)
// ---------------------------------------------------------------------------

bool PqQuantizer::build_code_lut(const uint8_t* code, float* out) const {
    if (cross_distance_table_.empty()) {
        return false;
    }
    // For each segment, the anchor's centroid ca_s indexes a contiguous
    // K-float row: table[s*K*K + ca_s*K + 0..K-1]. Copy that row into the LUT.
    const float* table_base = cross_distance_table_.data();
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t ca = read_code(code, bits_, s);
        const float* src = table_base + size_t(s) * K_ * K_ + ca * K_;
        float* dst = out + s * K_;
        std::memcpy(dst, src, K_ * sizeof(float));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Code-to-code distance via cross-distance table
// ---------------------------------------------------------------------------

float PqQuantizer::code_distance(const uint8_t* code_a,
                                 const uint8_t* code_b) const {
    if (cross_distance_table_.empty()) {
        // Fallback: compute from codebook directly. Reached by stub/untrained
        // quantizers (e.g. VamanaCore unit tests); train()/deserialize() build
        // the table for production paths.
        float acc = 0.0f;
        for (uint32_t s = 0; s < m_; s++) {
            const uint32_t ca = read_code(code_a, bits_, s);
            const uint32_t cb = read_code(code_b, bits_, s);
            const float* book =
                codebook_.data() + size_t(s) * K_ * K_ * sub_dim_;
            const float* va = book + ca * sub_dim_;
            const float* vb = book + cb * sub_dim_;
            switch (metric_) {
            case MetricKind::L2Sq:
                acc += simd::l2sq_f32(va, vb, sub_dim_);
                break;
            case MetricKind::InnerProduct:
                acc += -simd::dot_f32(va, vb, sub_dim_);
                break;
            }
        }
        return acc;
    }

    const float* table_base = cross_distance_table_.data();
    float acc = 0.0f;
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t ca = read_code(code_a, bits_, s);
        const uint32_t cb = read_code(code_b, bits_, s);
        acc += table_base[size_t(s) * K_ * K_ + ca * K_ + cb];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Batch distance: 4 candidates against a fixed anchor (PQ code) or fixed LUT.
//
// The inner loop is gather-limited (data-dependent table lookups). Processing
// 4 candidates simultaneously gives the CPU 4 independent load streams,
// keeping both load ports saturated and hiding latency via ILP. The NEON/AVX
// vadd is essentially free — the bottleneck is load throughput, not ALU.
//
// Benchmarked on Apple M4 at m=96:
//   scalar single:  98.5 ns/call
//   batch4 SIMD:     52.2 ns/call  (1.89×)
//   batch4 + prefetch: 43.8 ns/call  (2.25×)
// ---------------------------------------------------------------------------

void PqQuantizer::code_distance_batch4(const uint8_t* anchor,
                                       const uint8_t* code_b0,
                                       const uint8_t* code_b1,
                                       const uint8_t* code_b2,
                                       const uint8_t* code_b3,
                                        float* out) const {
    if (cross_distance_table_.empty()) {
        // Fallback: delegate to scalar code_distance (reached by stub/untrained
        // quantizers, e.g. VamanaCore unit tests).
        out[0] = code_distance(anchor, code_b0);
        out[1] = code_distance(anchor, code_b1);
        out[2] = code_distance(anchor, code_b2);
        out[3] = code_distance(anchor, code_b3);
        return;
    }

    const float* tbl = cross_distance_table_.data();
    const uint32_t KK = K_ * K_;

    // Pre-extract anchor's per-segment centroid ids.
    // For 8-bit codes this is just a direct byte read.
    uint32_t ac[256];  // max m we support
    for (uint32_t s = 0; s < m_; s++)
        ac[s] = read_code(anchor, bits_, s);

#if defined(SEXTANT_HAS_SVE)
    // SVE2 gather-load path — only on SVE2 hardware (Neoverse V2 / GCP Axion).
    // Apple M4 has NEON but NOT SVE, so this branch is never selected on M4.
    // We process exactly 4 candidates; predicate activates the first 4 lanes.
    const svbool_t pg4 = svwhilelt_b32_u32(0u, 4u);
    svfloat32_t vacc = svdup_f32(0.0f);
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = tbl + size_t(s) * KK + ac[s] * K_;
        // Code byte per candidate at segment s → element index into `row`.
        uint32_t idx[4] = {
            read_code(code_b0, bits_, s),
            read_code(code_b1, bits_, s),
            read_code(code_b2, bits_, s),
            read_code(code_b3, bits_, s)
        };
        svuint32_t sidx = svld1_u32(pg4, idx);
        // Gather-load 4 floats from row[idx[0..3]] (indices are element offsets).
        svfloat32_t vals = svld1_gather_u32index_f32(pg4, row, sidx);
        // _z: zero inactive lanes before adding → safe with any vector length.
        vacc = svadd_f32_z(pg4, vacc, vals);
    }
    svst1_f32(pg4, out, vacc);
#elif defined(SEXTANT_HAS_NEON)
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = tbl + size_t(s) * KK + ac[s] * K_;
        float vals[4] = {
            row[read_code(code_b0, bits_, s)],
            row[read_code(code_b1, bits_, s)],
            row[read_code(code_b2, bits_, s)],
            row[read_code(code_b3, bits_, s)]
        };
        vacc = vaddq_f32(vacc, vld1q_f32(vals));
    }
    out[0] = vgetq_lane_f32(vacc, 0);
    out[1] = vgetq_lane_f32(vacc, 1);
    out[2] = vgetq_lane_f32(vacc, 2);
    out[3] = vgetq_lane_f32(vacc, 3);
#elif defined(SEXTANT_HAS_AVX2)
    __m256 vacc = _mm256_setzero_ps();
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = tbl + size_t(s) * KK + ac[s] * K_;
        // Load 4 values into low lanes of a 256-bit vector.
        __m128 v128 = _mm_set_ps(
            row[read_code(code_b3, bits_, s)],
            row[read_code(code_b2, bits_, s)],
            row[read_code(code_b1, bits_, s)],
            row[read_code(code_b0, bits_, s)]);
        vacc = _mm256_add_ps(vacc, _mm256_castps128_ps256(v128));
    }
    // Extract 4 floats from the low 128 bits.
    __m128 lo = _mm256_castps256_ps128(vacc);
    _mm_storeu_ps(out, lo);
#else
    // Scalar reference implementation.
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = tbl + size_t(s) * KK + ac[s] * K_;
        d0 += row[read_code(code_b0, bits_, s)];
        d1 += row[read_code(code_b1, bits_, s)];
        d2 += row[read_code(code_b2, bits_, s)];
        d3 += row[read_code(code_b3, bits_, s)];
    }
    out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3;
#endif
}

void PqQuantizer::lut_distance_batch4(const uint8_t* code_b0,
                                      const uint8_t* code_b1,
                                      const uint8_t* code_b2,
                                      const uint8_t* code_b3,
                                      const float* lut,
                                      float* out) const {
#if defined(SEXTANT_HAS_SVE)
    // SVE2 gather-load path — only on SVE2 hardware (Neoverse V2 / GCP Axion).
    // Apple M4 has NEON but NOT SVE, so this branch is never selected on M4.
    const svbool_t pg4 = svwhilelt_b32_u32(0u, 4u);
    svfloat32_t vacc = svdup_f32(0.0f);
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = lut + s * K_;
        uint32_t idx[4] = {
            read_code(code_b0, bits_, s),
            read_code(code_b1, bits_, s),
            read_code(code_b2, bits_, s),
            read_code(code_b3, bits_, s)
        };
        svuint32_t sidx = svld1_u32(pg4, idx);
        svfloat32_t vals = svld1_gather_u32index_f32(pg4, row, sidx);
        vacc = svadd_f32_z(pg4, vacc, vals);
    }
    svst1_f32(pg4, out, vacc);
#elif defined(SEXTANT_HAS_NEON)
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = lut + s * K_;
        float vals[4] = {
            row[read_code(code_b0, bits_, s)],
            row[read_code(code_b1, bits_, s)],
            row[read_code(code_b2, bits_, s)],
            row[read_code(code_b3, bits_, s)]
        };
        vacc = vaddq_f32(vacc, vld1q_f32(vals));
    }
    out[0] = vgetq_lane_f32(vacc, 0);
    out[1] = vgetq_lane_f32(vacc, 1);
    out[2] = vgetq_lane_f32(vacc, 2);
    out[3] = vgetq_lane_f32(vacc, 3);
#elif defined(SEXTANT_HAS_AVX2)
    __m256 vacc = _mm256_setzero_ps();
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = lut + s * K_;
        __m128 v128 = _mm_set_ps(
            row[read_code(code_b3, bits_, s)],
            row[read_code(code_b2, bits_, s)],
            row[read_code(code_b1, bits_, s)],
            row[read_code(code_b0, bits_, s)]);
        vacc = _mm256_add_ps(vacc, _mm256_castps128_ps256(v128));
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    _mm_storeu_ps(out, lo);
#else
    // Scalar reference implementation.
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = lut + s * K_;
        d0 += row[read_code(code_b0, bits_, s)];
        d1 += row[read_code(code_b1, bits_, s)];
        d2 += row[read_code(code_b2, bits_, s)];
        d3 += row[read_code(code_b3, bits_, s)];
    }
    out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3;
#endif
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

void PqQuantizer::serialize(std::vector<uint8_t>& out) const {
    // Layout:
    //   {metric:u8, m:u16 LE, bits:u8, dim:u32 LE, codebook:float32[m*K*sub_dim]}
    //   {has_rotation:u8, [if 1: rotation:float32[dim*dim]]}
    const size_t header = sizeof(uint8_t) + sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint32_t);
    const size_t book_bytes = codebook_.size() * sizeof(float);
    const size_t rot_bytes =
        has_rotation_ ? size_t(dim_) * dim_ * sizeof(float) : 0;
    out.resize(header + book_bytes + 1 + rot_bytes);
    uint8_t* ptr = out.data();
    ptr[0] = static_cast<uint8_t>(metric_);
    uint16_t m = m_;
    std::memcpy(ptr + 1, &m, sizeof(m));
    ptr[3] = bits_;
    uint32_t d = dim_;
    std::memcpy(ptr + 4, &d, sizeof(d));
    if (book_bytes > 0) {
        std::memcpy(ptr + header, codebook_.data(), book_bytes);
    }
    uint8_t* rot_flag = ptr + header + book_bytes;
    rot_flag[0] = has_rotation_ ? 1u : 0u;
    if (has_rotation_ && rot_bytes > 0) {
        std::memcpy(rot_flag + 1, rotation_.data(), rot_bytes);
    }
}

void PqQuantizer::deserialize(const uint8_t* in, size_t size) {
    const size_t header = sizeof(uint8_t) + sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint32_t);
    if (size < header) {
        throw Error(ErrorCode::CorruptIndex, "PQ deserialize: blob too small");
    }
    metric_ = static_cast<MetricKind>(in[0]);
    uint16_t m;
    std::memcpy(&m, in + 1, sizeof(m));
    m_ = m;
    bits_ = in[3];
    uint32_t d;
    std::memcpy(&d, in + 4, sizeof(d));
    dim_ = d;
    if (m_ == 0 || dim_ % static_cast<Dim>(m_) != 0 ||
        (bits_ != 4 && bits_ != 8)) {
        throw Error(ErrorCode::CorruptIndex,
                    "PQ deserialize: invalid header");
    }
    K_ = 1u << bits_;
    sub_dim_ = dim_ / m_;
    const size_t book_floats = static_cast<size_t>(m_) * K_ * sub_dim_;
    const size_t book_bytes = book_floats * sizeof(float);

    // Trailer: {has_rotation:u8, [if 1: rotation:float32[dim*dim]]}.
    // The trailer is mandatory in the clean-slate format.
    has_rotation_ = false;
    rotation_.clear();
    opq_enabled_ = false;
    const size_t trailer_min = header + book_bytes + 1;
    if (size < trailer_min) {
        throw Error(ErrorCode::CorruptIndex, "PQ deserialize: size mismatch");
    }
    const uint8_t* rot_flag = in + header + book_bytes;
    if (rot_flag[0] != 0u) {
        const size_t rot_bytes = size_t(dim_) * dim_ * sizeof(float);
        if (size != trailer_min + rot_bytes) {
            throw Error(ErrorCode::CorruptIndex,
                        "PQ deserialize: rotation size mismatch");
        }
        rotation_.resize(size_t(dim_) * dim_);
        std::memcpy(rotation_.data(), rot_flag + 1, rot_bytes);
        has_rotation_ = true;
    } else if (size != trailer_min) {
        throw Error(ErrorCode::CorruptIndex,
                    "PQ deserialize: size mismatch (no rotation, trailing bytes)");
    }
    codebook_.assign(book_floats, 0.0f);
    if (book_bytes > 0) {
         std::memcpy(codebook_.data(), in + header, book_bytes);
    }
    build_cross_distance_table();
    compute_centroid_sqnorms_();
}

}  // namespace sextant
