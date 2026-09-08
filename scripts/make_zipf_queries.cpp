// make_zipf_queries: sample rows of an existing .fbin query file under a
// Zipf-like distribution (rank-1 heaviest) to benchmark skewed query
// streams against the LeafExtentCache's W-TinyLFU admission control.
//
// Usage: make_zipf_queries <in.fbin> <out.fbin> [exponent=1.0] [seed=7]
// Output has the same row count as the input: out row i = in row
// zipf_sample(i). With exponent 0 this degenerates to uniform shuffling.
//
// Manual build (meson scripts target drops c_args):
//   clang++ -std=c++23 -O2 scripts/make_zipf_queries.cpp -o /tmp/mzq

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s in.fbin out.fbin [exp=1.0] [seed=7]\n",
                     argv[0]);
        return 1;
    }
    const double expo = argc > 3 ? std::atof(argv[3]) : 1.0;
    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) { perror("open in"); return 1; }
    uint32_t hdr[2] = {0, 0};
    if (std::fread(hdr, 4, 2, in) != 2) { std::fprintf(stderr, "hdr\n"); return 1; }
    const uint32_t n = hdr[0], dim = hdr[1];
    std::vector<float> data(static_cast<size_t>(n) * dim);
    if (std::fread(data.data(), 4, data.size(), in) != data.size()) {
        std::fprintf(stderr, "short read\n"); return 1;
    }
    std::fclose(in);

    // Zipf over ranks 1..n (harmonic normalization), deterministic seed.
    std::mt19937_64 rng(argc > 4 ? std::atoll(argv[4]) : 7);
    std::vector<double> weights(n);
    double h = 0;
    for (uint32_t r = 1; r <= n; ++r) h += 1.0 / std::pow(r, expo);
    double acc = 0;
    for (uint32_t r = 1; r <= n; ++r) {
        acc += (1.0 / std::pow(r, expo)) / h;
        weights[r - 1] = acc;  // CDF
    }
    std::vector<uint32_t> src_row(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (uint32_t i = 0; i < n; ++i) {
        const double x = u(rng);
        // binary search the CDF
        uint32_t lo = 0, hi = n - 1;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (weights[mid] < x) lo = mid + 1; else hi = mid;
        }
        src_row[i] = lo;
    }

    std::FILE* out = std::fopen(argv[2], "wb");
    std::fwrite(hdr, 4, 2, out);
    std::vector<float> row(dim);
    for (uint32_t i = 0; i < n; ++i) {
        std::fwrite(&data[static_cast<size_t>(src_row[i]) * dim], 4, dim, out);
    }
    std::fclose(out);
    std::fprintf(stderr, "wrote %s: %u rows x %u dims, zipf exp=%.2f\n",
                 argv[2], n, dim, expo);
    return 0;
}
