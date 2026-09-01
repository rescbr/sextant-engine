// Spike: LCQ ranking-loss fine-tune of the shared shape f.
//
// Given fixed codes (from the trained scalar_shape quantizer), the scan
// score is LINEAR in f: D̂(q,x) = Σ_k f[k]·A_k(q,x), A_k = Σ_{d:code=k}
// q_d·step_d. Ranking loss over (query, true-neighbor, negative) triples
// is therefore a 16-variable monotone-constrained optimization — solvable
// by projected subgradient descent (isotonic projection after each step).
//
// Protocol:
//  1. train_shape on the sample (as shipped).
//  2. Sample vectors double as queries; exact top-k within the sample =
//     positives; random + hard (rank 50-500) = negatives.
//  3. Optimize f on the hinge loss, anchored to the LS f (λ regularizer).
//  4. Report isolated scan-ordering recall@10 before/after (f32 f, and the
//     u8-TBL quantized variant to see the rounding cost).
//
// Usage: spike_lcq <base.fbin> <query.fbin> [n_train] [k]

#include "../src/quant/scalar_lloydmax_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
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

// Isotonic (non-decreasing) projection: pool-adjacent-violators, min SSE.
static void isotonic(std::vector<float>& v) {
    const uint32_t n = static_cast<uint32_t>(v.size());
    std::vector<float> w(n);
    uint32_t idx = 0;
    for (uint32_t i = 0; i < n;) {
        uint32_t j = i;
        double sum = v[i];
        uint32_t cnt = 1;
        while (j + 1 < n && (sum / cnt > v[j + 1] || true)) {
            if (sum / cnt <= v[j + 1]) break;
            ++j; sum += v[j]; ++cnt;
            // merge-back check against previous block
            while (idx > 0 && w[idx - 1] > sum / cnt) {
                // fold into previous: handled by outer re-walk below
                break;
            }
        }
        const float val = static_cast<float>(sum / cnt);
        for (uint32_t t = i; t <= j; ++t) w[t] = val;
        i = j + 1;
    }
    // iterate to fixness (PAV in one pass is enough with proper impl; this
    // simple relabeling converges in a few passes for our near-monotone f).
    v = w;
    for (int pass = 0; pass < 8; ++pass) {
        bool bad = false;
        for (uint32_t i = 1; i < n; ++i)
            if (v[i] < v[i - 1]) { const float m = 0.5f * (v[i] + v[i - 1]);
                                   v[i] = v[i - 1] = m; bad = true; }
        if (!bad) break;
    }
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
    std::printf("base=%u dim=%u queries=%u sample=%llu k=%u\n", nb, dim, nq,
                static_cast<unsigned long long>(ns), k);

    // 1. Shipped shape quantizer.
    ScalarLloydMaxQuantizer q(MetricKind::L2Sq, dim, 4);
    q.train_shape(base.data(), ns);
    const uint32_t cs = q.code_size();
    const float* steps = q.steps();
    std::vector<float> f0(q.shape(), q.shape() + K);

    // 2. Encode the training sample (codes fixed from here on).
    std::vector<uint8_t> codes(static_cast<size_t>(ns) * cs);
    for (uint64_t i = 0; i < ns; ++i)
        q.encode(base.data() + i * dim, codes.data() + i * cs);

    // 3. Training triples from the sample: exact dot top-k = positives,
    //    ranks 50..500 = hard negatives.
    const uint32_t n_qtrain = 500;
    std::mt19937 rng(7);
    std::vector<uint32_t> qids(n_qtrain);
    for (uint32_t i = 0; i < n_qtrain; ++i) qids[i] = rng() % ns;
    struct Triple { uint32_t q, pos, neg; };
    std::vector<Triple> triples;
    {
        std::vector<std::pair<float, uint32_t>> sims(ns);
        for (uint32_t t = 0; t < n_qtrain; ++t) {
            const uint32_t qi = qids[t];
            const float* qv = base.data() + static_cast<uint64_t>(qi) * dim;
            for (uint64_t i = 0; i < ns; ++i) {
                float dot = 0;
                const float* xv = base.data() + i * dim;
                for (uint32_t d = 0; d < dim; ++d) dot += qv[d] * xv[d];
                sims[i] = {dot, static_cast<uint32_t>(i)};
            }
            std::partial_sort(sims.begin(), sims.begin() + 600, sims.end(),
                              std::greater<>());
            for (uint32_t p = 1; p <= k; ++p)  // skip self at rank 0
                for (int r = 0; r < 3; ++r) {
                    const uint32_t neg =
                        sims[50 + rng() % 450].second;
                    triples.push_back({qi, sims[p].second, neg});
                }
        }
    }
    std::printf("triples: %zu\n", triples.size());

    // 4. Precompute A_k(q,x) features per triple side: A[k] = Σ_{d:c=k}
    //    q_d·step_d. Two feature rows per triple (pos, neg).
    auto features = [&](uint32_t qi, uint32_t xi, float* A) {
        std::fill(A, A + K, 0.f);
        const float* qv = base.data() + static_cast<uint64_t>(qi) * dim;
        const uint8_t* code = codes.data() + static_cast<uint64_t>(xi) * cs;
        for (uint32_t d = 0; d < dim; ++d) {
            const uint8_t b = code[d / 2];
            const uint8_t c = (d % 2 == 0) ? (b & 0xF) : ((b >> 4) & 0xF);
            A[c] += qv[d] * steps[d];
        }
    };
    std::vector<float> Apos(triples.size() * K), Aneg(triples.size() * K);
    for (size_t t = 0; t < triples.size(); ++t) {
        features(triples[t].q, triples[t].pos, &Apos[t * K]);
        features(triples[t].q, triples[t].neg, &Aneg[t * K]);
    }

    // 5. Projected subgradient descent on f (16 vars).
    //    Loss per triple: max(0, m - (D̂neg - D̂pos)), D̂ = Σ f·A.
    //    + λ·(f - f0)² anchor to the reconstruction-optimal shape.
    auto evaluate = [&](const std::vector<float>& f) {
        double loss = 0;
        const double m = 1e-3;  // margin in dot units (tuned by scale)
        for (size_t t = 0; t < triples.size(); ++t) {
            double dp = 0, dn = 0;
            for (uint32_t kk = 0; kk < K; ++kk) {
                dp += f[kk] * Apos[t * K + kk];
                dn += f[kk] * Aneg[t * K + kk];
            }
            const double viol = m - (dn - dp);
            if (viol > 0) loss += viol;
        }
        return loss / triples.size();
    };
    // Scale check: report typical D̂ magnitudes once.
    {
        double dp = 0;
        for (uint32_t kk = 0; kk < K; kk++) dp += f0[kk] * Apos[kk];
        std::printf("baseline mean loss: %.3g  (|D̂pos| sample ~%.3g)\n",
                    evaluate(f0), std::fabs(dp));
    }

    std::vector<float> f = f0;
    const double lambda = 0.02;
    double lr = 5e-3;
    const double m = 0.01;  // margin (dot units; |D̂pos| ~ 0.9)
    for (int iter = 0; iter < 200; ++iter) {
        std::vector<double> grad(K, 0.0);
        double dn_viol = 0;
        for (size_t t = 0; t < triples.size(); ++t) {
            double dp = 0, dn = 0;
            for (uint32_t kk = 0; kk < K; ++kk) {
                dp += f[kk] * Apos[t * K + kk];
                dn += f[kk] * Aneg[t * K + kk];
            }
            if (m - (dn - dp) > 0) {
                ++dn_viol;
                for (uint32_t kk = 0; kk < K; ++kk)
                    grad[kk] -= (Aneg[t * K + kk] - Apos[t * K + kk]);
            }
        }
        for (uint32_t kk = 0; kk < K; ++kk)
            grad[kk] = grad[kk] / triples.size() +
                       lambda * (f[kk] - f0[kk]);
        for (uint32_t kk = 0; kk < K; ++kk)
            f[kk] -= static_cast<float>(lr * grad[kk]);
        isotonic(f);
        if (iter % 100 == 99) {
            double gn = 0;
            for (uint32_t kk = 0; kk < K; ++kk) gn += grad[kk] * grad[kk];
            std::printf("iter %d loss %.4g viol %.1f%% |grad| %.3g\n",
                        iter + 1, evaluate(f),
                        100.0 * dn_viol / triples.size(), std::sqrt(gn));
            lr *= 0.6;
        }
    }

    // 6. Shortlist containment — the metric that predicts tree recall:
    //    P(exact top-k ⊆ scan top-W). The tree reranks the top-W shortlist
    //    with FP32 vectors, so tree recall == this containment (up to ties).
    const std::vector<uint32_t> Ws = {10u, 30u, 50u, 100u};
    auto containment_multi = [&](const std::vector<float>& f, bool u8) {
        std::vector<double> out;
        float fmin = f[0], fmax = f[0];
        for (uint32_t kk = 0; kk < K; ++kk) {
            fmin = std::min(fmin, f[kk]);
            fmax = std::max(fmax, f[kk]);
        }
        const float S = fmax > fmin ? 255.f / (fmax - fmin) : 1.f;
        std::vector<float> fu(K);
        for (uint32_t kk = 0; kk < K; ++kk)
            fu[kk] = u8 ? std::lround((f[kk] - fmin) * S) / S + fmin : f[kk];
        std::vector<uint64_t> hits(Ws.size(), 0);
        std::vector<uint8_t> code(cs);
        std::vector<std::pair<float, uint32_t>> exact(nb), scan(nb);
        for (uint32_t qi = 0; qi < nq; ++qi) {
            const float* qv = query.data() + static_cast<size_t>(qi) * dim;
            std::vector<float> a(dim);
            for (uint32_t d = 0; d < dim; ++d) a[d] = qv[d] * steps[d];
            for (uint32_t i = 0; i < nb; ++i) {
                float de = 0, s = 0;
                const float* be = base.data() + static_cast<size_t>(i) * dim;
                for (uint32_t d = 0; d < dim; ++d) {
                    const float x = qv[d] - be[d];
                    de += x * x;
                }
                q.encode(be, code.data());
                for (uint32_t d = 0; d < dim; ++d) {
                    const uint8_t b = code[d / 2];
                    const uint8_t c =
                        (d % 2 == 0) ? (b & 0xF) : ((b >> 4) & 0xF);
                    s += a[d] * fu[c];
                }
                exact[i] = {de, i};
                scan[i] = {-s, i};
            }
            std::partial_sort(exact.begin(), exact.begin() + k, exact.end());
            std::partial_sort(scan.begin(), scan.begin() + 100, scan.end());
            for (uint32_t wi = 0; wi < Ws.size(); ++wi) {
                uint64_t h2 = 0;
                for (uint32_t x = 0; x < k; ++x)
                    for (uint32_t y = 0; y < Ws[wi]; ++y)
                        if (exact[x].second == scan[y].second) { h2++; break; }
                hits[wi] += h2;
            }
        }
        for (uint32_t wi = 0; wi < Ws.size(); ++wi)
            out.push_back(static_cast<double>(hits[wi]) / (nq * k));
        return out;
    };
    {
        auto c0 = containment_multi(f0, false);
        auto c1 = containment_multi(f, false);
        auto c2 = containment_multi(f0, true);
        auto c3 = containment_multi(f, true);
        for (uint32_t wi = 0; wi < Ws.size(); ++wi)
            std::printf("containment@%-3u f0: %.4f tuned: %.4f | u8 f0: %.4f "
                        "u8 tuned: %.4f\n", Ws[wi], c0[wi], c1[wi], c2[wi],
                        c3[wi]);
    }
    std::printf("f0:");  for (float x : f0) std::printf(" %.3f", x);
    std::printf("\nf :"); for (float x : f) std::printf(" %.3f", x);
    std::printf("\n");
    return 0;
}
