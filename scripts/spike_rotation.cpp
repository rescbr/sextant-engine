// Measure PQ distortion with and without PCA rotation on Cohere data.
// Tests the hypothesis: OPQ (rotation before PQ) dramatically reduces distortion
// on anisotropic data like Cohere.
//
// Also tests: what if we just use more bits on the high-variance dims?
//
// Usage: ./spike_rotation <base.fbin>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

struct Fbin { uint32_t n, dim; std::vector<float> data; };
Fbin read_fbin(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    Fbin fb; fread(&fb.n, 4, 1, f); fread(&fb.dim, 4, 1, f);
    fb.data.resize((size_t)fb.n * fb.dim);
    fread(fb.data.data(), 4, fb.data.size(), f); fclose(f); return fb;
}

// Jacobi eigendecomposition of symmetric matrix.
// Returns eigenvectors (columns) and eigenvalues.
void jacobi_eigen(std::vector<double>& A, uint32_t dim,
                  std::vector<double>& eigvecs, std::vector<double>& eigvals) {
    eigvecs.assign((size_t)dim * dim, 0);
    for (uint32_t i = 0; i < dim; ++i) eigvecs[(size_t)i * dim + i] = 1;
    eigvals.resize(dim);

    for (uint32_t sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (uint32_t r = 0; r < dim; ++r)
            for (uint32_t c = r + 1; c < dim; ++c)
                off += A[(size_t)r * dim + c] * A[(size_t)r * dim + c];
        if (off < 1e-16) break;

        for (uint32_t p = 0; p < dim; ++p) {
            for (uint32_t q = p + 1; q < dim; ++q) {
                double apq = A[(size_t)p * dim + q];
                if (std::fabs(apq) < 1e-15) continue;
                double app = A[(size_t)p * dim + p];
                double aqq = A[(size_t)q * dim + q];
                double theta = 0.5 * std::atan2(2 * apq, aqq - app);
                double c = std::cos(theta), s = std::sin(theta);

                for (uint32_t i = 0; i < dim; ++i) {
                    double aip = A[(size_t)i * dim + p];
                    double aiq = A[(size_t)i * dim + q];
                    A[(size_t)i * dim + p] = c * aip - s * aiq;
                    A[(size_t)i * dim + q] = s * aip + c * aiq;
                }
                for (uint32_t i = 0; i < dim; ++i) {
                    double api = A[(size_t)p * dim + i];
                    double aqi = A[(size_t)q * dim + i];
                    A[(size_t)p * dim + i] = c * api - s * aqi;
                    A[(size_t)q * dim + i] = s * api + c * aqi;
                }
                for (uint32_t i = 0; i < dim; ++i) {
                    double vip = eigvecs[(size_t)i * dim + p];
                    double viq = eigvecs[(size_t)i * dim + q];
                    eigvecs[(size_t)i * dim + p] = c * vip - s * viq;
                    eigvecs[(size_t)i * dim + q] = s * vip + c * viq;
                }
            }
        }
    }
    for (uint32_t i = 0; i < dim; ++i) eigvals[i] = A[(size_t)i * dim + i];
}

// Simulate PQ encoding: split dim into m subvectors of sub_dim, quantize each
// to K levels via k-means. Returns total RMS reconstruction error.
double pq_rms_error(const float* data, uint32_t n, uint32_t dim,
                    uint32_t m, uint32_t K, uint32_t iters = 10) {
    uint32_t sub_dim = dim / m;
    double total_sq_error = 0;

    for (uint32_t s = 0; s < m; ++s) {
        // Extract subvectors for this segment
        std::vector<float> sv((size_t)n * sub_dim);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t d = 0; d < sub_dim; ++d)
                sv[(size_t)i * sub_dim + d] = data[(size_t)i * dim + s * sub_dim + d];

        // k-means with K centroids
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> uni(0, n - 1);
        std::vector<float> cent((size_t)K * sub_dim);
        for (uint32_t c = 0; c < K; ++c)
            memcpy(&cent[(size_t)c * sub_dim], &sv[(size_t)uni(rng) * sub_dim], sub_dim * 4);

        for (uint32_t iter = 0; iter < iters; ++iter) {
            std::vector<uint32_t> assign(n);
            for (uint32_t i = 0; i < n; ++i) {
                float bd = 1e30f; uint32_t bc = 0;
                for (uint32_t c = 0; c < K; ++c) {
                    float d = 0;
                    for (uint32_t dd = 0; dd < sub_dim; ++dd) {
                        float df = sv[(size_t)i * sub_dim + dd] - cent[(size_t)c * sub_dim + dd];
                        d += df * df;
                    }
                    if (d < bd) { bd = d; bc = c; }
                }
                assign[i] = bc;
            }
            std::vector<float> ns((size_t)K * sub_dim, 0);
            std::vector<uint32_t> cnt(K, 0);
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t c = assign[i]; cnt[c]++;
                for (uint32_t dd = 0; dd < sub_dim; ++dd)
                    ns[(size_t)c * sub_dim + dd] += sv[(size_t)i * sub_dim + dd];
            }
            for (uint32_t c = 0; c < K; ++c)
                if (cnt[c] > 0)
                    for (uint32_t dd = 0; dd < sub_dim; ++dd)
                        cent[(size_t)c * sub_dim + dd] = ns[(size_t)c * sub_dim + dd] / cnt[c];
        }
        // Compute final error
        for (uint32_t i = 0; i < n; ++i) {
            float bd = 1e30f;
            for (uint32_t c = 0; c < K; ++c) {
                float d = 0;
                for (uint32_t dd = 0; dd < sub_dim; ++dd) {
                    float df = sv[(size_t)i * sub_dim + dd] - cent[(size_t)c * sub_dim + dd];
                    d += df * df;
                }
                if (d < bd) bd = d;
            }
            total_sq_error += bd;
        }
    }
    return std::sqrt(total_sq_error / n);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <base.fbin>\n", argv[0]); return 1; }
    auto fb = read_fbin(argv[1]);
    uint32_t n = std::min(fb.n, 10000u), dim = fb.dim;

    std::mt19937 rng(42);
    std::vector<uint32_t> idx(fb.n);
    for (uint32_t i = 0; i < fb.n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    std::vector<float> sub((size_t)n * dim);
    for (uint32_t i = 0; i < n; ++i)
        memcpy(&sub[(size_t)i * dim], &fb.data[(size_t)idx[i] * dim], dim * 4);

    printf("=== PQ distortion analysis (n=%u, dim=%u) ===\n\n", n, dim);

    // 1. PQ without rotation (original space)
    printf("--- Original space (no rotation) ---\n");
    for (auto [m, K, label] : std::vector<std::tuple<uint32_t,uint32_t,const char*>>{
        {192, 16, "PQ4 m=192"},
        {192, 256, "PQ8 m=192"},
        {96, 256, "PQ8 m=96"},
    }) {
        double rms = pq_rms_error(sub.data(), n, dim, m, K, 10);
        printf("  %-16s: RMS=%.4f  %s\n", label, rms,
               rms < 0.018 ? "OK" : rms < 0.04 ? "marginal" : "FAIL");
    }

    // 2. Compute PCA rotation
    printf("\n--- Computing PCA rotation ---\n");
    std::vector<double> cov((size_t)dim * dim, 0);
    std::vector<double> mean(dim, 0);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t d = 0; d < dim; ++d) mean[d] += sub[(size_t)i * dim + d];
    for (uint32_t d = 0; d < dim; ++d) mean[d] /= n;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t r = 0; r < dim; ++r) {
            double dr = sub[(size_t)i * dim + r] - mean[r];
            for (uint32_t c = r; c < dim; ++c) {
                double dc = sub[(size_t)i * dim + c] - mean[c];
                cov[(size_t)r * dim + c] += dr * dc;
            }
        }
    for (uint32_t r = 0; r < dim; ++r)
        for (uint32_t c = r; c < dim; ++c) {
            cov[(size_t)r * dim + c] /= n;
            cov[(size_t)c * dim + r] = cov[(size_t)r * dim + c];
        }

    std::vector<double> eigvecs, eigvals;
    jacobi_eigen(cov, dim, eigvecs, eigvals);

    // Sort eigenvalues descending
    std::vector<uint32_t> order(dim);
    for (uint32_t i = 0; i < dim; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return eigvals[a] > eigvals[b];
    });
    printf("  Top eigenvalues: ");
    for (uint32_t i = 0; i < 5; ++i) printf("%.6f ", eigvals[order[i]]);
    printf("...\n");

    // Rotate data: rotated = (x - mean) * R  (R = eigenvectors as columns, sorted)
    std::vector<float> rotated((size_t)n * dim);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < dim; ++j) {
            double s = 0;
            for (uint32_t d = 0; d < dim; ++d)
                s += (sub[(size_t)i * dim + d] - mean[d]) * eigvecs[(size_t)d * dim + order[j]];
            rotated[(size_t)i * dim + j] = (float)s;
        }

    // 3. PQ after PCA rotation (OPQ with PCA, not fully optimized but shows the effect)
    printf("\n--- PCA-rotated space ---\n");
    for (auto [m, K, label] : std::vector<std::tuple<uint32_t,uint32_t,const char*>>{
        {192, 16, "PQ4 m=192"},
        {192, 256, "PQ8 m=192"},
        {96, 256, "PQ8 m=96"},
    }) {
        double rms = pq_rms_error(rotated.data(), n, dim, m, K, 10);
        printf("  %-16s: RMS=%.4f  %s\n", label, rms,
               rms < 0.018 ? "OK" : rms < 0.04 ? "marginal" : "FAIL");
    }

    // 4. Variance per subvector group (to show why rotation helps)
    printf("\n--- Variance per PQ subvector group (m=192, sd=4) ---\n");
    printf("  Original space:\n    ");
    uint32_t m = 192, sd = dim / m;
    for (uint32_t s = 0; s < m; s += 32) {
        double var = 0;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t d = 0; d < sd; ++d) {
                double v = sub[(size_t)i * dim + s * sd + d];
                var += v * v;
            }
        var /= n * sd;
        printf("[%u]=%.4f ", s, var);
    }
    printf("\n  Rotated space:\n    ");
    for (uint32_t s = 0; s < m; s += 32) {
        double var = 0;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t d = 0; d < sd; ++d) {
                double v = rotated[(size_t)i * dim + s * sd + d];
                var += v * v;
            }
        var /= n * sd;
        printf("[%u]=%.4f ", s, var);
    }
    printf("\n");

    printf("\nNeighbor spread: ~0.018. Target: PQ RMS << 0.018\n");
    return 0;
}
