// Compare geometric structure of two embedding datasets.
// Measures: eigenvalue spectrum, intrinsic dimensionality, anisotropy,
// and k-means partition quality at various k.
//
// Usage: ./spike_geometry <base_a.fbin> <label_a> <base_b.fbin> <label_b>
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

// Eigenvalues of the covariance matrix via power iteration (top-k).
// Returns k eigenvalues (descending) of dim×dim covariance.
std::vector<double> top_eigenvalues(const float* data, uint32_t n, uint32_t dim, uint32_t k) {
    // Compute covariance matrix (dim × dim) — too big for dim=768? 768^2 doubles = 4.7MB, fine.
    std::vector<double> cov((size_t)dim * dim, 0);
    std::vector<double> mean(dim, 0);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t d = 0; d < dim; ++d)
            mean[d] += data[(size_t)i * dim + d];
    for (uint32_t d = 0; d < dim; ++d) mean[d] /= n;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t r = 0; r < dim; ++r) {
            double dr = data[(size_t)i * dim + r] - mean[r];
            for (uint32_t c = r; c < dim; ++c) {
                double dc = data[(size_t)i * dim + c] - mean[c];
                cov[(size_t)r * dim + c] += dr * dc;
            }
        }
    for (uint32_t r = 0; r < dim; ++r)
        for (uint32_t c = r; c < dim; ++c) {
            cov[(size_t)r * dim + c] /= n;
            cov[(size_t)c * dim + r] = cov[(size_t)r * dim + c]; // symmetric
        }

    // Power iteration for top-k eigenvalues
    std::vector<double> eigvals;
    std::vector<double> v(dim);
    std::vector<double> Av(dim);
    for (uint32_t ki = 0; ki < k; ++ki) {
        std::mt19937 rng(42 + ki);
        std::normal_distribution<double> normal(0, 1);
        for (uint32_t d = 0; d < dim; ++d) v[d] = normal(rng);
        for (uint32_t iter = 0; iter < 200; ++iter) {
            // Av = cov * v
            for (uint32_t r = 0; r < dim; ++r) {
                double s = 0;
                for (uint32_t c = 0; c < dim; ++c)
                    s += cov[(size_t)r * dim + c] * v[c];
                Av[r] = s;
            }
            double norm = 0; for (uint32_t d = 0; d < dim; ++d) norm += Av[d]*Av[d];
            norm = sqrt(norm);
            if (norm < 1e-15) break;
            for (uint32_t d = 0; d < dim; ++d) v[d] = Av[d] / norm;
        }
        // Rayleigh quotient
        double eig = 0;
        for (uint32_t r = 0; r < dim; ++r) {
            double s = 0;
            for (uint32_t c = 0; c < dim; ++c)
                s += cov[(size_t)r * dim + c] * v[c];
            eig += v[r] * s;
        }
        eigvals.push_back(eig);
        // Deflate: cov -= eig * v * v^T
        for (uint32_t r = 0; r < dim; ++r)
            for (uint32_t c = 0; c < dim; ++c)
                cov[(size_t)r * dim + c] -= eig * v[r] * v[c];
    }
    return eigvals;
}

void kmeans_residuals(const float* data, uint32_t n, uint32_t dim, uint32_t k,
                      double& out_res_norm, uint32_t iters = 15) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> uni(0, n - 1);
    std::vector<float> centroids((size_t)k * dim);
    for (uint32_t c = 0; c < k; ++c) {
        uint32_t idx = uni(rng);
        memcpy(&centroids[(size_t)c * dim], &data[(size_t)idx * dim], dim * 4);
    }
    std::vector<uint32_t> assign(n);
    for (uint32_t iter = 0; iter < iters; ++iter) {
        for (uint32_t i = 0; i < n; ++i) {
            const float* xi = &data[(size_t)i * dim];
            float bd = 1e30f; uint32_t bc = 0;
            for (uint32_t c = 0; c < k; ++c) {
                const float* cc = &centroids[(size_t)c * dim];
                float d = 0;
                for (uint32_t d2 = 0; d2 < dim; ++d2) { float df = xi[d2]-cc[d2]; d += df*df; }
                if (d < bd) { bd = d; bc = c; }
            }
            assign[i] = bc;
        }
        std::vector<float> ns((size_t)k * dim, 0); std::vector<uint32_t> cnt(k, 0);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t c = assign[i];
            for (uint32_t d = 0; d < dim; ++d) ns[(size_t)c*dim+d] += data[(size_t)i*dim+d];
            cnt[c]++;
        }
        for (uint32_t c = 0; c < k; ++c)
            if (cnt[c] > 0)
                for (uint32_t d = 0; d < dim; ++d) centroids[(size_t)c*dim+d] = ns[(size_t)c*dim+d]/cnt[c];
    }
    // Spherical centroids + residual norms
    std::vector<float> csum((size_t)k * dim, 0); std::vector<uint32_t> cnt(k, 0);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t c = assign[i];
        for (uint32_t d = 0; d < dim; ++d) csum[(size_t)c*dim+d] += data[(size_t)i*dim+d];
        cnt[c]++;
    }
    double rs = 0; uint32_t rc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t c = assign[i];
        if (cnt[c] < 2) continue;
        float cn = 0;
        for (uint32_t d = 0; d < dim; ++d) { float v = csum[(size_t)c*dim+d]/cnt[c]; cn += v*v; }
        cn = sqrt(cn); if (cn < 1e-10f) continue;
        float rn = 0;
        for (uint32_t d = 0; d < dim; ++d) {
            float cent = csum[(size_t)c*dim+d]/cnt[c]/cn;
            float df = data[(size_t)i*dim+d] - cent; rn += df*df;
        }
        rs += sqrt(rn); rc++;
    }
    out_res_norm = rs / std::max(1u, rc);
}

void analyze(const char* path, const char* label) {
    auto fb = read_fbin(path);
    uint32_t n = std::min(fb.n, 20000u), dim = fb.dim;

    // Subsample
    std::mt19937 rng(42);
    std::vector<uint32_t> idx(fb.n);
    for (uint32_t i = 0; i < fb.n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    std::vector<float> sub((size_t)n * dim);
    for (uint32_t i = 0; i < n; ++i)
        memcpy(&sub[(size_t)i * dim], &fb.data[(size_t)idx[i] * dim], dim * 4);

    printf("\n=== %s (n=%u, dim=%u, sampled %u) ===\n", label, fb.n, dim, n);

    // Eigenvalue spectrum (top 32)
    auto eigs = top_eigenvalues(sub.data(), n, dim, 32);
    double total_var = 0;
    for (uint32_t i = 0; i < dim; ++i) total_var += eigs[std::min(i, 31u)]; // approximation
    // Better: sum of all eigenvalues = trace of covariance
    // We only computed top 32, so report cumulative fraction

    double cum = 0;
    printf("Top-32 eigenvalues (variance explained):\n");
    for (uint32_t i = 0; i < 32; ++i) {
        cum += eigs[i];
        if (i < 10 || i % 5 == 0)
            printf("  eig[%2u] = %10.6f  cum=%6.2f%%\n", i, eigs[i], 100*cum/cum);
    }
    // Ratio: top-1 / top-32
    printf("  eig[0]/eig[31] = %.1f (anisotropy ratio)\n", eigs[0] / std::max(eigs[31], 1e-15));
    printf("  eig[0]/eig[1]  = %.1f\n", eigs[0] / std::max(eigs[1], 1e-15));

    // Intrinsic dimensionality: (sum eigenvalues)^2 / sum(eigenvalues^2)
    // For top-32 approximation:
    double s1 = 0, s2 = 0;
    for (uint32_t i = 0; i < 32; ++i) { s1 += eigs[i]; s2 += eigs[i]*eigs[i]; }
    double int_dim = s1 * s1 / std::max(s2, 1e-30);
    printf("  Intrinsic dim (top-32): %.1f (of %u full)\n", int_dim, dim);

    // K-means residual norms at different k
    printf("\nK-means partition quality (residual norm):\n");
    for (uint32_t k : {16u, 64u, 256u, 1024u}) {
        double rn;
        auto t0 = std::chrono::steady_clock::now();
        kmeans_residuals(sub.data(), n, dim, k, rn, 15);
        auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("  k=%5u: residual_norm=%.4f  (%.1fs)\n", k, rn, secs);
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <a.fbin> <label_a> <b.fbin> <label_b>\n", argv[0]);
        return 1;
    }
    analyze(argv[1], argv[2]);
    analyze(argv[3], argv[4]);
    return 0;
}
