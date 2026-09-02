// Spike: does PER-LEAF scalar level fitting beat global levels on cohere?
// (task 7 gate — validate the RANGE hypothesis before building local_shape)
//
// The prior negative (per-leaf residual PQ) showed cohere can't be
// partitioned into tight clusters DIRECTIONALLY. local_shape bets on a
// different mechanism: per-leaf dynamic RANGE fit (1-D per dim). This
// spike tests exactly that, offline:
//   - partition base into K=26 clusters (Lloyd k-means, mirrors leaves)
//   - global baseline: one scalar_lloydmax quantizer (the shipped tier)
//   - local uniform: per-cluster train_uniform (arithmetic-scan family:
//     level_d(c) = lo_d + step_d·c — the implementable design)
//   - local lm: per-cluster train() (freeform Lloyd-Max — the upper bound
//     that would cost the dim×16 LUT scan)
// Scoring: decode-dot with each vector's OWN levels (what the scan would
// compute). Metrics: top-k, containment@W, tau2 — same as spike_pq2_tier.
//
// Usage: spike_local_shape <base.fbin> <query.fbin> [n_train] [k] [n_q] [K]

#include "../src/quant/scalar_lloydmax_quantizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace sextant;

static std::vector<float> load_fbin(const char* path, uint32_t& n, uint32_t& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s failed\n", path); std::exit(1); }
    uint32_t hdr[2];
    f.read(reinterpret_cast<char*>(hdr), 8);
    n = hdr[0]; d = hdr[1];
    std::vector<float> v(static_cast<size_t>(n) * d);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(v.size()) * 4);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s base.fbin query.fbin [n_train] [k] "
                             "[n_queries] [K]\n", argv[0]);
        return 1;
    }
    uint32_t nb, nq, dim, d2;
    auto base = load_fbin(argv[1], nb, dim);
    auto query = load_fbin(argv[2], nq, d2);
    const uint32_t n_train = argc > 3 ? std::atoi(argv[3]) : 20000;
    const uint32_t k = argc > 4 ? std::atoi(argv[4]) : 10;
    const uint32_t n_q = argc > 5 ? std::atoi(argv[5])
                                  : std::min<uint32_t>(nq, 300);
    const uint32_t K = argc > 6 ? std::atoi(argv[6]) : 26;
    std::printf("base=%u dim=%u queries=%u n_train=%u k=%u K=%u\n",
                nb, dim, n_q, n_train, k, K);

    // --- Partition: Lloyd k-means K=26 (a few passes, seeded by first K).
    std::vector<float> cents(static_cast<size_t>(K) * dim);
    for (uint32_t c = 0; c < K; ++c)
        std::copy_n(base.data() + static_cast<size_t>(c * 7919 % nb) * dim,
                    dim, cents.begin() + static_cast<size_t>(c) * dim);
    std::vector<uint32_t> assign(nb, 0);
    for (int iter = 0; iter < 8; ++iter) {
        std::vector<float> nsum(static_cast<size_t>(K) * dim, 0.f);
        std::vector<uint32_t> ncnt(K, 0);
        for (uint32_t i = 0; i < nb; ++i) {
            const float* v = base.data() + static_cast<size_t>(i) * dim;
            float bd = 1e30f; uint32_t bc = 0;
            for (uint32_t c = 0; c < K; ++c) {
                float d = 0;
                const float* cc = cents.data() + static_cast<size_t>(c) * dim;
                for (uint32_t dd = 0; dd < dim; ++dd) {
                    const float e = v[dd] - cc[dd];
                    d += e * e;
                }
                if (d < bd) { bd = d; bc = c; }
            }
            assign[i] = bc;
            for (uint32_t dd = 0; dd < dim; ++dd)
                nsum[static_cast<size_t>(bc) * dim + dd] += v[dd];
            ncnt[bc]++;
        }
        for (uint32_t c = 0; c < K; ++c)
            if (ncnt[c] > 0)
                for (uint32_t dd = 0; dd < dim; ++dd)
                    cents[static_cast<size_t>(c) * dim + dd] =
                        nsum[static_cast<size_t>(c) * dim + dd] / ncnt[c];
    }
    // Cluster membership.
    std::vector<std::vector<uint32_t>> members(K);
    for (uint32_t i = 0; i < nb; ++i) members[assign[i]].push_back(i);
    { uint32_t mn = nb, mx = 0; for (auto& m : members) { mn = std::min(mn, (uint32_t)m.size()); mx = std::max(mx, (uint32_t)m.size()); }
      std::printf("clusters: %u, size min=%u max=%u\n", K, mn, mx); }

    // --- Quantizer variants: decoded reconstruction per vector.
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> rec_global(static_cast<size_t>(nb) * dim);
    {
        ScalarLloydMaxQuantizer g(MetricKind::L2Sq, dim, 4);
        g.train(base.data(), std::min<uint64_t>(n_train, nb));
        std::vector<uint8_t> code(g.code_size());
        for (uint32_t i = 0; i < nb; ++i) {
            g.encode(base.data() + static_cast<size_t>(i) * dim, code.data());
            g.decode(code.data(), rec_global.data() +
                                       static_cast<size_t>(i) * dim);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("global LM train+enc: %.1fs\n",
                std::chrono::duration<double>(t1 - t0).count());

    auto fit_local = [&](bool uniform) {
        std::vector<float> rec(static_cast<size_t>(nb) * dim);
        for (uint32_t c = 0; c < K; ++c) {
            const auto& mem = members[c];
            if (mem.empty()) continue;
            std::vector<float> sample(mem.size() * dim);
            for (size_t i = 0; i < mem.size(); ++i)
                std::copy_n(base.data() + static_cast<size_t>(mem[i]) * dim,
                            dim, sample.begin() + static_cast<size_t>(i) * dim);
            ScalarLloydMaxQuantizer q(MetricKind::L2Sq, dim, 4);
            if (uniform) q.train_uniform(sample.data(), mem.size());
            else q.train(sample.data(), mem.size());
            std::vector<uint8_t> code(q.code_size());
            for (size_t i = 0; i < mem.size(); ++i) {
                q.encode(sample.data() + i * dim, code.data());
                q.decode(code.data(), rec.data() +
                                           static_cast<size_t>(mem[i]) * dim);
            }
        }
        return rec;
    };
    auto rec_luni = fit_local(true);
    auto rec_llm = fit_local(false);
    auto t2 = std::chrono::steady_clock::now();
    std::printf("local fits (uniform + LM): %.1fs\n",
                std::chrono::duration<double>(t2 - t1).count());

    // --- Scoring: dot(q, x̂) with the vector's own levels. IP ranking on
    // normalized data == cosine == the benchmark GT regime.
    struct Res { const char* name; uint64_t top, cont, tau_h; double ids; };
    std::vector<Res> results = {
        {"global-LM ", 0, 0, 0, 0}, {"local-uni ", 0, 0, 0, 0},
        {"local-LM  ", 0, 0, 0, 0},
    };
    uint64_t total = 0;
    const uint32_t W = 300;
    const float tau = 2.0f;
    for (uint32_t qi = 0; qi < n_q; ++qi) {
        const float* qv = query.data() + static_cast<size_t>(qi) * dim;
        std::vector<std::pair<float, uint32_t>> ex(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            float de = 0;
            const float* xv = base.data() + static_cast<size_t>(i) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                const float e = qv[d] - xv[d];
                de += e * e;
            }
            ex[i] = {de, i};
        }
        std::partial_sort(ex.begin(), ex.begin() + k, ex.end());
        std::vector<uint32_t> gt(k);
        for (uint32_t a = 0; a < k; ++a) gt[a] = ex[a].second;

        const float* recs[3] = {rec_global.data(), rec_luni.data(),
                                rec_llm.data()};
        for (int v = 0; v < 3; ++v) {
            std::vector<std::pair<float, uint32_t>> sc(nb);
            for (uint32_t i = 0; i < nb; ++i) {
                const float* dv = recs[v] + static_cast<size_t>(i) * dim;
                float dot = 0;
                for (uint32_t d = 0; d < dim; ++d) dot += qv[d] * dv[d];
                sc[i] = {-dot, i};
            }
            std::partial_sort(sc.begin(), sc.begin() + W, sc.end());
            for (uint32_t a = 0; a < k; ++a)
                for (uint32_t b = 0; b < k; ++b)
                    if (sc[b].second == gt[a]) { results[v].top++; break; }
            for (uint32_t a = 0; a < W; ++a)
                for (uint32_t b = 0; b < k; ++b)
                    if (sc[a].second == gt[b]) { results[v].cont++; break; }
            const float g = (sc[k - 1].first - sc[0].first) / (k - 1);
            const float thr = sc[k - 1].first + tau * std::max(g, 1e-9f);
            for (uint32_t a = 0; a < W; ++a) {
                if (sc[a].first > thr) break;
                ++results[v].ids;
                for (uint32_t b = 0; b < k; ++b)
                    if (sc[a].second == gt[b]) { results[v].tau_h++; break; }
            }
        }
        total += k;
    }
    for (const auto& r : results)
        std::printf("%s top-%u %.4f | contain@%u %.4f | tau2 %.4f @ %.1f ids\n",
                    r.name, k, static_cast<double>(r.top) / total,
                    W, static_cast<double>(r.cont) / total,
                    static_cast<double>(r.tau_h) / total,
                    r.ids / n_q);
    return 0;
}
