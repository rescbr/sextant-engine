// Spike: shared code-shape companding for the uniform MAC kernel.
//
// level_d[k] = lo_d + step_d * f[k], f a single monotone 16-vector shared
// across dims. Scan stays gather-free: dot = c0 + Σ (q_d·step_d)·f[code_d]
// (f is 16 bytes, one NEON TBL per 16 nibbles). Question: how much of the
// uniform->LloydMax recall gap (88.7 -> 92.2 cohere) does shape expressivity
// recover, before any ranking-loss training?
//
// Shapes evaluated (per-dim (lo,step) least-squares fitted to Lloyd-Max
// levels in all cases):
//   linear    f[k] = k                          (= uniform, sanity)
//   pow-p     f[k] = (k/15)^p, p in grid
//   mu-X      μ-law compander, μ in grid
//   emp-N     empirical shape: alternating fit — f[k] = mean_d
//             (lm_d[k]-lo_d)/step_d, iterated N times (LS-optimal shared f)
//
// Usage: spike_shape <base.fbin> <query.fbin> [n_train] [k]
// Reports isolated recall@k for each shape.

#include "../src/quant/scalar_lloydmax_quantizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace sextant;

static std::vector<float> load_fbin(const char* path, uint32_t& n, uint32_t& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s failed\n", path); std::exit(1); }
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&d), 4);
    std::vector<float> v(static_cast<size_t>(n) * d);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * 4);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s base query [n_train] [k]\n", argv[0]);
        return 1;
    }
    uint32_t nb, dq, nq, d2;
    auto base = load_fbin(argv[1], nb, dq);
    auto query = load_fbin(argv[2], nq, d2);
    const uint32_t dim = dq;
    const uint32_t n_train = argc > 3 ? std::atoi(argv[3]) : 20000;
    const uint32_t k = argc > 4 ? std::atoi(argv[4]) : 10;
    const uint64_t ns = std::min<uint64_t>(n_train, nb);
    const uint32_t K = 16;
    std::printf("base=%u dim=%u queries=%u n_train=%llu k=%u\n", nb, dim, nq,
                static_cast<unsigned long long>(ns), k);

    // Lloyd-Max reference levels (1 restart — shape study, not final quality).
    ScalarLloydMaxQuantizer lm(MetricKind::L2Sq, dim, 4);
    auto t0 = std::chrono::steady_clock::now();
    lm.train(base.data(), ns, /*restarts=*/1, 30);
    const float* LV = lm.levels();
    std::printf("LM train (1 restart): %.1fs\n",
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count());

    // Evaluate one shape: per-dim LS fit of (lo, step) to the LM levels,
    // then encode/decode all base vectors with nearest level and measure
    // isolated recall vs exact full-dim top-k.
    auto eval_shape = [&](const std::string& name, std::vector<float> f) {
        // Per-dim LS fit: lm_d[k] ~ lo + step*f[k].
        std::vector<float> lo(dim), st(dim), cd(static_cast<size_t>(dim) * K);
        for (uint32_t d = 0; d < dim; ++d) {
            const float* lv = LV + static_cast<size_t>(d) * K;
            double sf = 0, sf2 = 0, s1 = 0, sfx = 0;
            for (uint32_t kk = 0; kk < K; ++kk) {
                sf += f[kk]; sf2 += double(f[kk]) * f[kk];
                s1 += lv[kk]; sfx += double(f[kk]) * lv[kk];
            }
            const double denom = K * sf2 - sf * sf;
            const double step = denom != 0 ? (K * sfx - sf * s1) / denom : 0.0;
            const double l0 = (s1 - step * sf) / K;
            lo[d] = static_cast<float>(l0);
            st[d] = static_cast<float>(step);
            for (uint32_t kk = 0; kk < K; ++kk)
                cd[static_cast<size_t>(d) * K + kk] =
                    static_cast<float>(l0 + step * f[kk]);
        }
        // Encode + decode every base vector (nearest level).
        std::vector<float> decoded(static_cast<size_t>(nb) * dim);
        for (uint32_t i = 0; i < nb; ++i) {
            const float* v = base.data() + static_cast<size_t>(i) * dim;
            float* out = decoded.data() + static_cast<size_t>(i) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                const float* lv = cd.data() + static_cast<size_t>(d) * K;
                float best = lv[0]; float bd = std::fabs(v[d] - lv[0]);
                for (uint32_t kk = 1; kk < K; ++kk) {
                    const float dd = std::fabs(v[d] - lv[kk]);
                    if (dd < bd) { bd = dd; best = lv[kk]; }
                }
                out[d] = best;
            }
        }
        // Isolated recall@k vs exact.
        uint64_t hits = 0, total = 0;
        for (uint32_t qi = 0; qi < nq; ++qi) {
            const float* qv = query.data() + static_cast<size_t>(qi) * dim;
            std::vector<std::pair<float, uint32_t>> exact(nb), quant(nb);
            for (uint32_t i = 0; i < nb; ++i) {
                float de = 0, dq2 = 0;
                const float* be = base.data() + static_cast<size_t>(i) * dim;
                const float* bq = decoded.data() + static_cast<size_t>(i) * dim;
                for (uint32_t d = 0; d < dim; ++d) {
                    de += (qv[d] - be[d]) * (qv[d] - be[d]);
                    dq2 += (qv[d] - bq[d]) * (qv[d] - bq[d]);
                }
                exact[i] = {de, i};
                quant[i] = {dq2, i};
            }
            std::partial_sort(exact.begin(), exact.begin() + k, exact.end());
            std::partial_sort(quant.begin(), quant.begin() + k, quant.end());
            for (uint32_t a = 0; a < k; ++a)
                for (uint32_t b = 0; b < k; ++b)
                    if (exact[a].second == quant[b].second) { hits++; break; }
            total += k;
        }
        std::printf("%-12s recall@%u: %.4f\n", name.c_str(), k,
                    static_cast<double>(hits) / total);
        return f;
    };

    // Shapes.
    std::vector<float> lin(K);
    for (uint32_t kk = 0; kk < K; ++kk) lin[kk] = float(kk);
    eval_shape("linear", lin);

    for (double p : {0.5, 0.7, 1.0, 1.5, 2.0, 3.0}) {
        std::vector<float> f(K);
        for (uint32_t kk = 0; kk < K; ++kk)
            f[kk] = float(std::pow(double(kk) / (K - 1), p));
        char name[32];
        std::snprintf(name, sizeof name, "pow-%.1f", p);
        eval_shape(name, f);
    }

    for (double mu : {2.0, 5.0, 15.0, 50.0, 255.0}) {
        std::vector<float> f(K);
        const double lmu = std::log1p(mu);
        for (uint32_t kk = 0; kk < K; ++kk) {
            const double t = 2.0 * kk / (K - 1) - 1.0;  // [-1, 1]
            f[kk] = float(std::copysign(std::log1p(mu * std::fabs(t)) / lmu, t));
        }
        char name[32];
        std::snprintf(name, sizeof name, "mu-%.0f", mu);
        eval_shape(name, f);
    }

    // Empirical shape: alternating LS fit against the LM table.
    // init f = linear; iterate: (1) per-dim (lo,step) fit, (2) f[k] =
    // mean_d (lm_d[k]-lo_d)/step_d. Converges to the best shared shape.
    std::vector<float> f = lin;
    std::vector<float> lo(dim), st(dim);
    for (int it = 0; it < 8; ++it) {
        for (uint32_t d = 0; d < dim; ++d) {
            const float* lv = LV + static_cast<size_t>(d) * K;
            double sf = 0, sf2 = 0, s1 = 0, sfx = 0;
            for (uint32_t kk = 0; kk < K; ++kk) {
                sf += f[kk]; sf2 += double(f[kk]) * f[kk];
                s1 += lv[kk]; sfx += double(f[kk]) * lv[kk];
            }
            const double denom = K * sf2 - sf * sf;
            const double step = denom != 0 ? (K * sfx - sf * s1) / denom : 0.0;
            st[d] = float(step);
            lo[d] = float((s1 - step * sf) / K);
        }
        std::vector<double> fn(K, 0.0);
        for (uint32_t d = 0; d < dim; ++d) {
            const float* lv = LV + static_cast<size_t>(d) * K;
            for (uint32_t kk = 0; kk < K; ++kk)
                fn[kk] += (lv[kk] - lo[d]) / (st[d] + 1e-30f);
        }
        for (uint32_t kk = 0; kk < K; ++kk) f[kk] = float(fn[kk] / dim);
        // Keep monotone (defensive): running max.
        for (uint32_t kk = 1; kk < K; ++kk)
            if (f[kk] < f[kk - 1]) f[kk] = f[kk - 1];
        char name[32];
        std::snprintf(name, sizeof name, "emp-%d", it + 1);
        if (it == 0 || it == 3 || it == 7) eval_shape(name, f);
    }
    return 0;
}
