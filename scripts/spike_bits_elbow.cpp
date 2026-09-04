// spike_bits_elbow: the bytes-vs-decoded-ranking-recall elbow, offline.
//
// Measures the DECODED-RANKING CEILING per bit budget: rank the whole corpus
// by quantized dot (a perfect scan over the quantized vectors) and grade
// top-10 against exact f32 GT. This is exactly the ceiling the engine's
// decoded rerank can reach at that budget — the number that decides regime
// (a)'s rerank storage (8-bit codes vs fp16 region).
//
// Rows (global quantizers; local fit only helps, so global = lower bound):
//   sq4: 16 uniform levels, per-dim σ ruler (mean ± 2.7σ) — engine's
//        train_uniform / Infino SQ4 construction
//   sq8: 256 uniform levels, mean ± 3.3σ loading
//   fp16: half-precision storage (sanity: should read ~1.0)
// All rows carry the per-vector IP bias ||x||/||x̂|| (same correction as the
// engine's scan; without it IP ranking is systematically biased low).
//
// usage: spike_bits_elbow <base.fbin> <query.fbin> [nq=100]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "simd_kernels.hpp"
#include "util/fp16.hpp"

namespace {

struct Fbin {
    std::vector<float> vecs;
    uint32_t n = 0, dim = 0;
};

Fbin load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    uint32_t hdr[2];
    if (fread(hdr, 4, 2, f) != 2) exit(1);
    Fbin fb;
    fb.n = hdr[0]; fb.dim = hdr[1];
    fb.vecs.resize((size_t)fb.n * fb.dim);
    if (fread(fb.vecs.data(), 4, fb.vecs.size(), f) != fb.vecs.size()) exit(1);
    fclose(f);
    return fb;
}

}  // namespace

using sextant::simd::dot_f32;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: spike_bits_elbow <base.fbin> <query.fbin> [nq]\n");
        return 1;
    }
    Fbin base = load(argv[1]);
    Fbin query = load(argv[2]);
    uint32_t nq = argc > 3 ? uint32_t(atoi(argv[3])) : 100;
    nq = std::min(nq, query.n);
    const uint32_t n = base.n, dim = base.dim;
    const uint32_t k = 10;
    printf("base %u x %u, %u queries, k=%u\n", n, dim, nq, k);

    // ---- Exact GT (f32 brute, 8 threads) ----
    std::vector<std::vector<uint32_t>> gt(nq);
    {
        const uint32_t T = 8;
        std::vector<std::thread> ts;
        for (uint32_t t = 0; t < T; ++t) {
            ts.emplace_back([&, t] {
                for (uint32_t q = t; q < nq; q += T) {
                    std::vector<std::pair<float, uint32_t>> sc(n);
                    for (uint32_t i = 0; i < n; ++i)
                        sc[i] = {dot_f32(&query.vecs[(size_t)q * dim],
                                               &base.vecs[(size_t)i * dim], dim), i};
                    std::partial_sort(sc.begin(), sc.begin() + k, sc.end(),
                                      [](auto& a, auto& b) { return a.first > b.first; });
                    gt[q].resize(k);
                    for (uint32_t i = 0; i < k; ++i) gt[q][i] = sc[i].second;
                }
            });
        }
        for (auto& th : ts) th.join();
    }
    printf("exact GT done\n");

    // ---- Per-dim stats (mean, sigma) over the full corpus ----
    std::vector<float> mean(dim, 0.f), sigma(dim, 0.f);
    {
        std::vector<double> s(dim, 0.0), s2(dim, 0.0);
        for (size_t i = 0; i < base.vecs.size(); ++i) { s[i % dim] += base.vecs[i]; s2[i % dim] += double(base.vecs[i]) * base.vecs[i]; }
        for (uint32_t d = 0; d < dim; ++d) {
            mean[d] = float(s[d] / n);
            const double var = std::max(s2[d] / n - double(mean[d]) * mean[d], 0.0);
            sigma[d] = float(std::sqrt(var));
        }
    }

    auto run_row = [&](const char* name, uint32_t levels, float loading) {
        // Ruler: lo = mean - loading*sigma, uniform step to +loading*sigma.
        std::vector<float> lo(dim), st(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            lo[d] = mean[d] - loading * sigma[d];
            st[d] = (2.f * loading * sigma[d]) / float(levels - 1);
            if (st[d] < 1e-12f) st[d] = 1e-12f;
        }
        // Decode + per-vector IP bias, then brute top-10 vs GT.
        double hits = 0;
        auto score_query = [&](uint32_t q, std::vector<float>& dec) {
            const float* qv = &query.vecs[(size_t)q * dim];
            std::vector<std::pair<float, uint32_t>> sc(n);
            for (uint32_t i = 0; i < n; ++i) {
                const float* v = &base.vecs[(size_t)i * dim];
                float dot = 0.f, nx2 = 0.f, nh2 = 0.f;
                for (uint32_t d = 0; d < dim; ++d) {
                    int c = int((v[d] - lo[d]) / st[d] + 0.5f);
                    c = std::clamp(c, 0, int(levels - 1));
                    const float r = lo[d] + st[d] * c;
                    dec[d] = r;
                    dot += qv[d] * r;
                    nx2 += v[d] * v[d];
                    nh2 += r * r;
                }
                const float bias = std::sqrt(nx2 / std::max(nh2, 1e-30f));
                sc[i] = {dot * bias, i};
            }
            std::partial_sort(sc.begin(), sc.begin() + k, sc.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            uint32_t h = 0;
            for (uint32_t i = 0; i < k; ++i)
                for (uint32_t g : gt[q])
                    if (g == sc[i].second) { ++h; break; }
            return h;
        };
        // Parallel over queries, per-thread decode buffer.
        const uint32_t T = 8;
        std::vector<double> part(T, 0.0);
        std::vector<std::thread> ts;
        for (uint32_t t = 0; t < T; ++t) {
            ts.emplace_back([&, t] {
                std::vector<float> dec(dim);
                double h = 0;
                for (uint32_t q = t; q < nq; q += T) h += score_query(q, dec);
                part[t] = h;
            });
        }
        for (auto& th : ts) th.join();
        for (double h : part) hits += h;
        printf("%-5s levels=%3u bytes/vec=%5u: decoded-ceiling recall@10 = %.4f\n",
               name, levels, uint32_t((dim * uint32_t(std::ceil(std::log2(levels)))) / 8),
               hits / (double(nq) * k));
    };

    // Loading sweep: derived MSE-optimal for N=16 Gaussian is ~2.5 (mid-tread
    // uniform, saturation outside); the cited value is 2.7 (Infino blog).
    // Ranking recall is the real metric — let the corpus decide.
    if (getenv("SWEEP")) {
        for (float m : {2.1f, 2.3f, 2.5f, 2.7f, 3.0f, 3.3f})
            run_row("sq4", 16, m);
        for (float m : {3.3f, 3.9f, 4.5f})
            run_row("sq8", 256, m);
        return 0;
    }
    run_row("sq4", 16, 2.7f);
    run_row("sq8", 256, 3.3f);

    // ---- fp16 sanity row: half-precision storage, no ruler ----
    {
        std::vector<float16_t> half(base.vecs.size());
        for (size_t i = 0; i < base.vecs.size(); ++i) half[i] = float16_t(base.vecs[i]);
        double hits = 0;
        const uint32_t T = 8;
        std::vector<double> part(T, 0.0);
        std::vector<std::thread> ts;
        for (uint32_t t = 0; t < T; ++t) {
            ts.emplace_back([&, t] {
                std::vector<float> dec(dim);
                double h = 0;
                for (uint32_t q = t; q < nq; q += T) {
                    const float* qv = &query.vecs[(size_t)q * dim];
                    std::vector<std::pair<float, uint32_t>> sc(n);
                    for (uint32_t i = 0; i < n; ++i) {
                        for (uint32_t d = 0; d < dim; ++d)
                            dec[d] = static_cast<float>(
                                half[(size_t)i * dim + d]);
                        sc[i] = {dot_f32(qv, dec.data(), dim), i};
                    }
                    std::partial_sort(sc.begin(), sc.begin() + k, sc.end(),
                                      [](auto& a, auto& b) { return a.first > b.first; });
                    for (uint32_t i = 0; i < k; ++i)
                        for (uint32_t g : gt[q])
                            if (g == sc[i].second) { ++h; break; }
                }
                part[t] = h;
            });
        }
        for (auto& th : ts) th.join();
        for (double h : part) hits += h;
        printf("%-5s levels=--- bytes/vec=%5u: decoded-ceiling recall@10 = %.4f\n",
               "fp16", dim * 2, hits / (double(nq) * k));
    }
    return 0;
}
