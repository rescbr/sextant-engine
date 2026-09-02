// Spike: does a per-vector IP bias scalar fix the cohere IP deficit?
// (task 2 validation — no format change, offline brute force)
//
// For each base vector we have the raw x, the reconstruction x̂ (slm 4-bit),
// and per-vector scalars ‖x‖, ‖x̂‖. Scan-ordering candidates:
//   plain   : ⟨q, x̂⟩                      (what the scan computes today)
//   rabitq  : ⟨q, x̂⟩ · ‖x‖/‖x̂‖           (RaBitQ/TurboVec bias, IP target)
//   normxh  : ⟨q, x̂⟩ / ‖x̂‖               (reconstruction-normalized; the
//                                         cosine-target degenerate of rabitq)
//   exact   : ⟨q, x⟩                      (oracle)
// Targets:
//   GT-ip   : top-k of ⟨q, x⟩             (raw IP)
//   GT-cos  : top-k of ⟨q, x⟩/(‖q‖‖x‖)    (cosine — our benchmark GT)
// Reports top-k recall and containment@W (shortlist contract) per candidate.
//
// Usage: spike_ip_bias <base.fbin> <query.fbin> [n_train] [k] [W] [n_queries]

#include "../src/quant/scalar_lloydmax_quantizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
                             "[W] [n_queries]\n", argv[0]);
        return 1;
    }
    uint32_t nb, nq, dim, d2;
    auto base = load_fbin(argv[1], nb, dim);
    auto query = load_fbin(argv[2], nq, d2);
    const uint32_t n_train = argc > 3 ? std::atoi(argv[3]) : 20000;
    const uint32_t k = argc > 4 ? std::atoi(argv[4]) : 10;
    const uint32_t W = argc > 5 ? std::atoi(argv[5]) : 100;
    const uint32_t n_q = argc > 6 ? std::atoi(argv[6])
                                  : std::min<uint32_t>(nq, 1000);
    std::printf("base=%u dim=%u queries=%u/%u n_train=%u k=%u W=%u\n",
                nb, dim, n_q, nq, n_train, k, W);

    // Norm stats (the deficit's premise: cohere is unnormalized).
    double nmin = 1e30, nmax = 0, nmean = 0;
    for (uint32_t i = 0; i < nb; ++i) {
        double s = 0;
        for (uint32_t d = 0; d < dim; ++d) {
            const double x = base[static_cast<size_t>(i) * dim + d];
            s += x * x;
        }
        const double n = std::sqrt(s);
        nmin = std::min(nmin, n); nmax = std::max(nmax, n); nmean += n;
    }
    std::printf("base norms: min=%.3f max=%.3f mean=%.3f\n",
                nmin, nmax, nmean / nb);

    // Train + encode + decode.
    ScalarLloydMaxQuantizer q(MetricKind::L2Sq, dim, /*bits=*/4);
    auto t0 = std::chrono::steady_clock::now();
    q.train(base.data(), std::min<uint64_t>(n_train, nb));
    auto t1 = std::chrono::steady_clock::now();
    std::printf("train: %.2fs\n",
                std::chrono::duration<double>(t1 - t0).count());
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes(static_cast<size_t>(nb) * cs);
    std::vector<float> decoded(static_cast<size_t>(nb) * dim);
    std::vector<float> nx(nb), nxh(nb);   // ‖x‖, ‖x̂‖ per vector
    t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < nb; ++i) {
        const float* xv = base.data() + static_cast<size_t>(i) * dim;
        q.encode(xv, codes.data() + static_cast<size_t>(i) * cs);
        float* dv = decoded.data() + static_cast<size_t>(i) * dim;
        q.decode(codes.data() + static_cast<size_t>(i) * cs, dv);
        double sx = 0, sxh = 0;
        for (uint32_t d = 0; d < dim; ++d) {
            sx += static_cast<double>(xv[d]) * xv[d];
            sxh += static_cast<double>(dv[d]) * dv[d];
        }
        nx[i] = static_cast<float>(std::sqrt(sx));
        nxh[i] = static_cast<float>(std::sqrt(std::max(sxh, 1e-30)));
    }
    t1 = std::chrono::steady_clock::now();
    std::printf("encode+decode+norms: %.2fs\n",
                std::chrono::duration<double>(t1 - t0).count());

    // Bias spread diagnostics.
    {
        double bmin = 1e30, bmax = 0, bsum = 0;
        std::vector<double> bs(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            bs[i] = static_cast<double>(nx[i]) / nxh[i];
            bmin = std::min(bmin, bs[i]); bmax = std::max(bmax, bs[i]);
            bsum += bs[i];
        }
        std::sort(bs.begin(), bs.end());
        std::printf("bias ||x||/||x_hat||: min=%.4f p1=%.4f med=%.4f p99=%.4f "
                    "max=%.4f mean=%.4f\n", bmin, bs[nb / 100],
                    bs[nb / 2], bs[nb - nb / 100], bmax, bsum / nb);
        double hmin = 1e30, hmax = 0;
        for (uint32_t i = 0; i < nb; ++i) {
            hmin = std::min(hmin, (double)nxh[i]);
            hmax = std::max(hmax, (double)nxh[i]);
        }
        std::printf("||x_hat||: min=%.4f max=%.4f\n", hmin, hmax);
    }

    // Per-query evaluation. dot = ⟨q, x̂⟩ computed once; candidates are
    // post-hoc monotone-ish transforms of it (per-vector scalar factors).
    struct Cand { const char* name; uint64_t top, cont; };
    Cand cands[] = {
        {"plain  ", 0, 0}, {"rabitq ", 0, 0}, {"normxh ", 0, 0},
        {"exact  ", 0, 0},
    };
    Cand ip_cands[] = {
        {"plain  ", 0, 0}, {"rabitq ", 0, 0}, {"exact  ", 0, 0},
    };
    uint64_t total = 0;
    for (uint32_t qi = 0; qi < n_q; ++qi) {
        const float* qv = query.data() + static_cast<size_t>(qi) * dim;
        double nq_norm = 0;
        for (uint32_t d = 0; d < dim; ++d) nq_norm +=
            static_cast<double>(qv[d]) * qv[d];
        nq_norm = std::sqrt(std::max(nq_norm, 1e-30));

        std::vector<std::pair<float, uint32_t>> s_plain(nb), s_rabitq(nb),
            s_normxh(nb), s_exact(nb), s_gtcp(nb), s_gtip(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            const float* dv = decoded.data() + static_cast<size_t>(i) * dim;
            const float* xv = base.data() + static_cast<size_t>(i) * dim;
            double dot_h = 0, dot_e = 0;
            for (uint32_t d = 0; d < dim; ++d) {
                dot_h += static_cast<double>(qv[d]) * dv[d];
                dot_e += static_cast<double>(qv[d]) * xv[d];
            }
            s_plain[i]  = {-static_cast<float>(dot_h), i};
            s_rabitq[i] = {-static_cast<float>(dot_h * nx[i] / nxh[i]), i};
            s_normxh[i] = {-static_cast<float>(dot_h / nxh[i]), i};
            s_exact[i]  = {-static_cast<float>(dot_e), i};
            s_gtcp[i]   = {-static_cast<float>(dot_e / (nq_norm * nx[i])), i};
            s_gtip[i]   = {-static_cast<float>(dot_e), i};
        }
        // Shortlists.
        const uint32_t SL = std::max(W, k);
        std::partial_sort(s_plain.begin(), s_plain.begin() + SL, s_plain.end());
        std::partial_sort(s_rabitq.begin(), s_rabitq.begin() + SL, s_rabitq.end());
        std::partial_sort(s_normxh.begin(), s_normxh.begin() + SL, s_normxh.end());
        std::partial_sort(s_exact.begin(), s_exact.begin() + SL, s_exact.end());
        std::partial_sort(s_gtcp.begin(), s_gtcp.begin() + k, s_gtcp.end());
        std::partial_sort(s_gtip.begin(), s_gtip.begin() + k, s_gtip.end());

        // GT sets.
        std::vector<uint32_t> gt_cp(k), gt_ip(k);
        for (uint32_t a = 0; a < k; ++a) {
            gt_cp[a] = s_gtcp[a].second;
            gt_ip[a] = s_gtip[a].second;
        }
        auto score = [&](std::pair<float, uint32_t>* s, const uint32_t* gt,
                         Cand& c) {
            std::vector<uint32_t> topk(k);
            for (uint32_t a = 0; a < k; ++a) topk[a] = s[a].second;
            for (uint32_t a = 0; a < k; ++a) {
                for (uint32_t b = 0; b < k; ++b)
                    if (gt[a] == topk[b]) { c.top++; break; }
            }
            for (uint32_t a = 0; a < SL; ++a) {
                for (uint32_t b = 0; b < k; ++b)
                    if (gt[b] == s[a].second) { c.cont++; break; }
            }
        };
        // cosine target
        score(s_plain.data(),  gt_cp.data(), cands[0]);
        score(s_rabitq.data(), gt_cp.data(), cands[1]);
        score(s_normxh.data(), gt_cp.data(), cands[2]);
        score(s_exact.data(),  gt_cp.data(), cands[3]);
        // raw-IP target
        score(s_plain.data(),  gt_ip.data(), ip_cands[0]);
        score(s_rabitq.data(), gt_ip.data(), ip_cands[1]);
        score(s_exact.data(),  gt_ip.data(), ip_cands[2]);
        total += k;
    }

    std::printf("\n=== target = COSINE (benchmark GT) ===\n");
    for (const auto& c : cands)
        std::printf("%s top-%u recall %.4f   contain@%u %.4f\n",
                    c.name, k, static_cast<double>(c.top) / total,
                    W, static_cast<double>(c.cont) / (total * (W >= k ? 1 : 1)));
    std::printf("\n=== target = RAW IP ===\n");
    for (const auto& c : ip_cands)
        std::printf("%s top-%u recall %.4f   contain@%u %.4f\n",
                    c.name, k, static_cast<double>(c.top) / total,
                    W, static_cast<double>(c.cont) / total);
    return 0;
}
