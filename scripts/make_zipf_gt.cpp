// make_zipf_gt: generate a row-aligned GTMM ground-truth file for a
// zipf-resampled query stream. MUST be used with the same exponent and
// seed as make_zipf_queries used for the queries — the sampling is
// recomputed deterministically (verified: same mt19937_64 seed + CDF
// binary search reproduces the query file byte-for-byte), and GT row i
// is copied from GT row src_row[i].
//
// Usage: make_zipf_gt <queries_norm.fbin> <gt.gtmm> <out.gtmm> [exp=1.0] [seed=7]
//
// The queries file is only read for its row count (the zipf domain).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <queries.fbin> <gt.gtmm> <out.gtmm> "
                     "[exp=1.0] [seed=7]\n", argv[0]);
        return 1;
    }
    const double expo = argc > 4 ? std::atof(argv[4]) : 1.0;
    const long long seed = argc > 5 ? std::atoll(argv[5]) : 7;

    uint32_t nq = 0;
    {
        std::FILE* qf = std::fopen(argv[1], "rb");
        if (!qf) { perror("open queries"); return 1; }
        uint32_t hdr[2] = {0, 0};
        if (std::fread(hdr, 4, 2, qf) != 2) { std::fprintf(stderr, "q hdr\n"); return 1; }
        nq = hdr[0];
        std::fclose(qf);
    }

    std::FILE* gf = std::fopen(argv[2], "rb");
    if (!gf) { perror("open gt"); return 1; }
    constexpr uint32_t kGtMagic = 0x4D4D5447u;  // "GTMM" LE
    uint32_t gh[3] = {0, 0, 0};
    if (std::fread(gh, 4, 3, gf) != 3 || gh[0] != kGtMagic) {
        std::fprintf(stderr, "gt missing GTMM magic\n"); return 1;
    }
    uint8_t metric = 0;
    if (std::fread(&metric, 1, 1, gf) != 1) { std::fprintf(stderr, "gt metric\n"); return 1; }
    const uint32_t n = gh[1], k = gh[2];
    if (n != nq) {
        std::fprintf(stderr, "row count mismatch: queries=%u gt=%u\n", nq, n);
        return 1;
    }
    // Per-query interleaved layout: [ids k×u32][dists k×f32] per row.
    const size_t row_bytes = static_cast<size_t>(k) * 8;
    std::vector<uint8_t> rows(static_cast<size_t>(n) * row_bytes);
    if (std::fread(rows.data(), 1, rows.size(), gf) != rows.size()) {
        std::fprintf(stderr, "gt short read\n"); return 1;
    }
    std::fclose(gf);

    // Same sampling as make_zipf_queries (exp/seed must match).
    std::mt19937_64 rng(seed);
    std::vector<double> weights(n);
    double h = 0;
    for (uint32_t r = 1; r <= n; r++) h += 1.0 / std::pow(r, expo);
    double acc = 0;
    for (uint32_t r = 1; r <= n; r++) {
        acc += (1.0 / std::pow(r, expo)) / h;
        weights[r - 1] = acc;
    }
    std::vector<uint32_t> src_row(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (uint32_t i = 0; i < n; i++) {
        const double x = u(rng);
        uint32_t lo = 0, hi = n - 1;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (weights[mid] < x) lo = mid + 1; else hi = mid;
        }
        src_row[i] = lo;
    }

    std::FILE* out = std::fopen(argv[3], "wb");
    if (!out) { perror("open out"); return 1; }
    std::fwrite(gh, 4, 3, out);
    std::fwrite(&metric, 1, 1, out);
    for (uint32_t i = 0; i < n; i++) {
        const size_t s = static_cast<size_t>(src_row[i]) * row_bytes;
        std::fwrite(&rows[s], 1, row_bytes, out);
    }
    std::fclose(out);
    std::fprintf(stderr, "wrote %s: %u rows x k=%u, zipf exp=%.2f seed=%lld\n",
                 argv[3], n, k, expo, seed);
    return 0;
}
