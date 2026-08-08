// Test non-uniform quantization: give more bits to high-variance PCA dims.
// Hypothesis: Cohere's 57:1 anisotropy means a few dims carry all the signal.
// Allocate precision accordingly.
//
// Usage: ./spike_aniso_quant <base.fbin>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

struct Fbin { uint32_t n, dim; std::vector<float> data; };
Fbin read_fbin(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) exit(1);
    Fbin fb; fread(&fb.n, 4, 1, f); fread(&fb.dim, 4, 1, f);
    fb.data.resize((size_t)fb.n * fb.dim);
    fread(fb.data.data(), 4, fb.data.size(), f); fclose(f); return fb;
}

void jacobi_eigen(std::vector<double>& A, uint32_t dim,
                  std::vector<double>& V, std::vector<double>& eigvals) {
    V.assign((size_t)dim * dim, 0);
    for (uint32_t i = 0; i < dim; ++i) V[(size_t)i * dim + i] = 1;
    for (uint32_t sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (uint32_t r = 0; r < dim; ++r)
            for (uint32_t c = r + 1; c < dim; ++c) off += A[r*dim+c]*A[r*dim+c];
        if (off < 1e-16) break;
        for (uint32_t p = 0; p < dim; ++p) for (uint32_t q = p+1; q < dim; ++q) {
            double apq = A[p*dim+q]; if (std::fabs(apq) < 1e-15) continue;
            double app = A[p*dim+p], aqq = A[q*dim+q];
            double theta = 0.5 * std::atan2(2*apq, aqq - app);
            double c = std::cos(theta), s = std::sin(theta);
            for (uint32_t i = 0; i < dim; ++i) {
                double aip = A[i*dim+p], aiq = A[i*dim+q];
                A[i*dim+p] = c*aip - s*aiq; A[i*dim+q] = s*aip + c*aiq;
            }
            for (uint32_t i = 0; i < dim; ++i) {
                double api = A[p*dim+i], aqi = A[q*dim+i];
                A[p*dim+i] = c*api - s*aqi; A[q*dim+i] = s*api + c*aqi;
            }
            for (uint32_t i = 0; i < dim; ++i) {
                double vip = V[i*dim+p], viq = V[i*dim+q];
                V[i*dim+p] = c*vip - s*viq; V[i*dim+q] = s*vip + c*viq;
            }
        }
    }
    eigvals.resize(dim);
    for (uint32_t i = 0; i < dim; ++i) eigvals[i] = A[i*dim+i];
}

// Uniform scalar quantization of a 1D array to n_levels.
// Returns the quantized values.
std::vector<float> scalar_quantize(const std::vector<float>& data, uint32_t n_levels) {
    float lo = *std::min_element(data.begin(), data.end());
    float hi = *std::max_element(data.begin(), data.end());
    std::vector<float> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        float t = (data[i] - lo) / std::max(hi - lo, 1e-15f);
        out[i] = std::round(t * (n_levels - 1)) / (n_levels - 1) * (hi - lo) + lo;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    auto fb = read_fbin(argv[1]);
    uint32_t n = std::min(fb.n, 10000u), dim = fb.dim;

    std::mt19937 rng(42);
    std::vector<uint32_t> idx(fb.n);
    for (uint32_t i = 0; i < fb.n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    std::vector<float> sub((size_t)n * dim);
    for (uint32_t i = 0; i < n; ++i)
        memcpy(&sub[(size_t)i * dim], &fb.data[(size_t)idx[i] * dim], dim * 4);

    // Compute covariance + PCA
    std::vector<double> mean(dim, 0), cov((size_t)dim*dim, 0);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t d = 0; d < dim; ++d) mean[d] += sub[(size_t)i*dim+d];
    for (uint32_t d = 0; d < dim; ++d) mean[d] /= n;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t r = 0; r < dim; ++r) {
            double dr = sub[(size_t)i*dim+r] - mean[r];
            for (uint32_t c = r; c < dim; ++c)
                cov[(size_t)r*dim+c] += dr * (sub[(size_t)i*dim+c] - mean[c]);
        }
    for (uint32_t r = 0; r < dim; ++r)
        for (uint32_t c = r; c < dim; ++c) { cov[r*dim+c] /= n; cov[c*dim+r] = cov[r*dim+c]; }

    std::vector<double> V, eigvals;
    jacobi_eigen(cov, dim, V, eigvals);

    // Sort by eigenvalue descending
    std::vector<uint32_t> order(dim);
    for (uint32_t i = 0; i < dim; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return eigvals[a] > eigvals[b]; });

    // Project to PCA space
    std::vector<float> rotated((size_t)n * dim);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < dim; ++j) {
            double s = 0;
            for (uint32_t d = 0; d < dim; ++d)
                s += (sub[(size_t)i*dim+d] - mean[d]) * V[(size_t)d*dim+order[j]];
            rotated[(size_t)i*dim+j] = (float)s;
        }

    printf("=== Non-uniform bit allocation on PCA dims ===\n");
    printf("dim=%u, n=%u\n", dim, n);
    printf("Eigenvalues: ");
    double total_var = 0;
    for (uint32_t i = 0; i < 10; ++i) { printf("%.5f ", eigvals[order[i]]); total_var += eigvals[order[i]]; }
    double all_var = 0; for (uint32_t i = 0; i < dim; ++i) all_var += eigvals[i];
    printf("\nTop-10 variance fraction: %.1f%%\n\n", 100*total_var/all_var);

    // Test different bit allocation strategies:
    // Strategy A: uniform PQ4 (baseline) — 4 bits per dim
    // Strategy B: top-N dims get 8 bits, rest get 2 bits
    // Strategy C: top-N dims get 16 bits (float16), rest get 0 bits (truncated)
    // Strategy D: top-N dims get 8 bits, rest get 4 bits

    for (uint32_t top_n : {1u, 4u, 16u, 64u}) {
        for (uint32_t top_bits : {8u, 16u}) {
            uint32_t rest_bits = (top_bits == 16u) ? 0u : 2u;

            // Quantize each dimension independently
            std::vector<float> recon((size_t)n * dim, 0);
            double total_sq_error = 0;
            uint32_t total_bits = 0;

            for (uint32_t j = 0; j < dim; ++j) {
                std::vector<float> col(n);
                for (uint32_t i = 0; i < n; ++i) col[i] = rotated[(size_t)i*dim+j];

                uint32_t bits = (j < top_n) ? top_bits : rest_bits;
                uint32_t levels = bits == 0 ? 1 : (1u << bits);
                total_bits += bits;

                if (bits == 0) {
                    // Truncate: reconstruction = mean (which is ~0 in PCA space)
                    for (uint32_t i = 0; i < n; ++i) {
                        float err = col[i]; total_sq_error += (double)err * err;
                    }
                } else {
                    auto q = scalar_quantize(col, levels);
                    for (uint32_t i = 0; i < n; ++i) {
                        recon[(size_t)i*dim+j] = q[i];
                        float err = col[i] - q[i]; total_sq_error += (double)err * err;
                    }
                }
            }

            double rms = std::sqrt(total_sq_error / n);
            double bpv = (double)total_bits / dim;
            printf("top_%-2u @%2ubit rest@%ubit: RMS=%.4f  %5.1f bits/vec  %s\n",
                   top_n, top_bits, rest_bits, rms, bpv * dim / 8,
                   rms < 0.018 ? "OK" : rms < 0.04 ? "marginal" : "FAIL");
        }
    }

    // Also show: what if we just store the top-N PCA dims as FP16 and truncate?
    printf("\n--- FP16 on top-N PCA dims, truncate rest ---\n");
    for (uint32_t top_n : {1u, 2u, 4u, 8u, 16u, 32u}) {
        double total_sq_error = 0;
        for (uint32_t j = top_n; j < dim; ++j) {
            for (uint32_t i = 0; i < n; ++i) {
                float v = rotated[(size_t)i*dim+j];
                total_sq_error += (double)v * v;
            }
        }
        double rms = std::sqrt(total_sq_error / n);
        double bytes = top_n * 2; // FP16 per top dim
        // Variance retained
        double retained = 0;
        for (uint32_t j = 0; j < top_n; ++j) retained += eigvals[order[j]];
        printf("top_%-2u FP16: RMS=%.4f  %4.0f B/vec  var_retained=%.1f%%  %s\n",
               top_n, rms, bytes, 100*retained/all_var,
               rms < 0.018 ? "OK" : rms < 0.04 ? "marginal" : "FAIL");
    }

    printf("\nNeighbor spread: ~0.018\n");
    return 0;
}
