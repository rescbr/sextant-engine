// Measure residual norms and estimated PQ distortion at different k values.
// Usage: ./spike_residual_k <base.fbin> [max_k]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

// Read fbin header + data
struct Fbin {
    uint32_t n, dim;
    std::vector<float> data;
};

Fbin read_fbin(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    Fbin fb;
    fread(&fb.n, 4, 1, f);
    fread(&fb.dim, 4, 1, f);
    fb.data.resize((size_t)fb.n * fb.dim);
    fread(fb.data.data(), 4, fb.data.size(), f);
    fclose(f);
    return fb;
}

// Simple k-means (Lloyd's) on float data, returns assignments + centroids
void kmeans(const float* data, uint32_t n, uint32_t dim, uint32_t k,
            uint32_t* assign, std::vector<float>& centroids, uint32_t iters = 20) {
    centroids.resize((size_t)k * dim);
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> uni(0, n - 1);
    // Init: random points
    for (uint32_t c = 0; c < k; ++c) {
        uint32_t idx = uni(rng);
        memcpy(&centroids[(size_t)c * dim], &data[(size_t)idx * dim], dim * 4);
    }

    std::vector<float> new_cent((size_t)k * dim, 0);
    std::vector<uint32_t> counts(k, 0);

    for (uint32_t iter = 0; iter < iters; ++iter) {
        // Assign
        for (uint32_t i = 0; i < n; ++i) {
            const float* xi = &data[(size_t)i * dim];
            float best_d = 1e30f;
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k; ++c) {
                const float* cc = &centroids[(size_t)c * dim];
                float d = 0;
                for (uint32_t d2 = 0; d2 < dim; ++d2) {
                    float diff = xi[d2] - cc[d2];
                    d += diff * diff;
                }
                if (d < best_d) { best_d = d; best_c = c; }
            }
            assign[i] = best_c;
        }
        // Update
        std::fill(new_cent.begin(), new_cent.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t c = assign[i];
            const float* xi = &data[(size_t)i * dim];
            for (uint32_t d = 0; d < dim; ++d)
                new_cent[(size_t)c * dim + d] += xi[d];
            counts[c]++;
        }
        for (uint32_t c = 0; c < k; ++c) {
            if (counts[c] > 0)
                for (uint32_t d = 0; d < dim; ++d)
                    centroids[(size_t)c * dim + d] = new_cent[(size_t)c * dim + d] / counts[c];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <base.fbin> [max_k=1024]\n", argv[0]); return 1; }
    uint32_t max_k = argc > 2 ? atoi(argv[2]) : 1024;
    uint32_t sample_n = 20000;

    auto fb = read_fbin(argv[1]);
    uint32_t n = fb.n, dim = fb.dim;
    fprintf(stderr, "Loaded %u x %u\n", n, dim);

    // Subsample
    std::mt19937 rng(42);
    std::vector<uint32_t> idx(n);
    for (uint32_t i = 0; i < n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    if (sample_n > n) sample_n = n;
    std::vector<float> sub((size_t)sample_n * dim);
    for (uint32_t i = 0; i < sample_n; ++i)
        memcpy(&sub[(size_t)i * dim], &fb.data[(size_t)idx[i] * dim], dim * 4);

    printf("dim=%u, %u subsample\n", dim, sample_n);
    printf("%-22s %10s %10s %10s %10s %10s\n", "config", "res_norm", "PQ4_RMS", "PQ8_RMS", "sim_err4", "sim_err8");
    printf("%s\n", std::string(80, '-').c_str());

    for (uint32_t k : {16u, 64u, 256u, 1024u, 4096u}) {
        if (k > max_k) break;
        std::vector<uint32_t> assign(sample_n);
        std::vector<float> centroids;
        auto t0 = std::chrono::steady_clock::now();
        kmeans(sub.data(), sample_n, dim, k, assign.data(), centroids, 15);
        auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // Compute residual norms (spherical centroid)
        std::vector<float> cluster_sum((size_t)k * dim, 0);
        std::vector<uint32_t> counts(k, 0);
        for (uint32_t i = 0; i < sample_n; ++i) {
            uint32_t c = assign[i];
            for (uint32_t d = 0; d < dim; ++d)
                cluster_sum[(size_t)c * dim + d] += sub[(size_t)i * dim + d];
            counts[c]++;
        }

        double res_norm_sum = 0;
        uint32_t res_count = 0;
        for (uint32_t c = 0; c < k; ++c) {
            if (counts[c] < 2) continue;
            // Spherical centroid
            float cnorm = 0;
            for (uint32_t d = 0; d < dim; ++d) {
                float v = cluster_sum[(size_t)c * dim + d] / counts[c];
                cnorm += v * v;
            }
            cnorm = sqrt(cnorm);
            if (cnorm < 1e-10f) continue;
            std::vector<float> cent(dim);
            for (uint32_t d = 0; d < dim; ++d)
                cent[d] = cluster_sum[(size_t)c * dim + d] / counts[c] / cnorm;

            for (uint32_t i = 0; i < sample_n; ++i) {
                if (assign[i] != c) continue;
                float rn = 0;
                for (uint32_t d = 0; d < dim; ++d) {
                    float diff = sub[(size_t)i * dim + d] - cent[d];
                    rn += diff * diff;
                }
                res_norm_sum += sqrt(rn);
                res_count++;
            }
        }
        double res_norm = res_norm_sum / std::max(1u, res_count);
        double sigma2 = res_norm * res_norm / dim;

        // PQ4 m=192 sd=4 K=16
        double pq4_rms = sqrt(192.0 * sigma2 * 4.0 * pow(16.0, -2.0/4.0));
        // PQ8 m=96 sd=8 K=256
        double pq8_rms = sqrt(96.0 * sigma2 * 8.0 * pow(256.0, -2.0/8.0));

        printf("k=%-5d                %10.4f %10.4f %10.4f %10.4f %10.4f  (%.1fs)\n",
               k, res_norm, pq4_rms, pq8_rms, pq4_rms, pq8_rms, secs);

        // Also print PQ8 m=192 sd=4
        double pq8_192 = sqrt(192.0 * sigma2 * 4.0 * pow(256.0, -2.0/4.0));
        printf("  PQ8 m=192 sd=4:       %10.4f\n", pq8_192);
        // PQ4 m=384 sd=2
        double pq4_384 = sqrt(384.0 * sigma2 * 2.0 * pow(16.0, -2.0/2.0));
        printf("  PQ4 m=384 sd=2:       %10.4f\n", pq4_384);
    }
    printf("\nNeighbor spread: ~0.018\n");
    printf("Rule: sim_err < spread → PQ can resolve neighbors\n");
    return 0;
}
