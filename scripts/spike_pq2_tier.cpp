// Spike: is a 2-bit PQ tier viable for the adaptive-W hybrid contract?
// (task 4a — extends the economics to 24B/vec at m=96)
//
// Trains PQ at 2/4 bits over a range of m on the same sample, then for
// each config measures — brute-force, no tree:
//   - quantizer-only top-k recall (decode ordering)
//   - shortlist containment@W (the hybrid contract: if the true top-k is
//     inside the top-W by quantizer score, exact rerank recovers it)
//   - mean ids for an adaptive-W-style tau cut (gap-based, self-calibrated
//     against the top-k region like the engine's tau)
// Reports recall-per-byte so the tiers land on the economics frontier.
//
// Usage: spike_pq2_tier <base.fbin> <query.fbin> [n_train] [k] [n_queries]

#include "../src/quant/pq_quantizer.hpp"

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
                             "[n_queries]\n", argv[0]);
        return 1;
    }
    uint32_t nb, nq, dim, d2;
    auto base = load_fbin(argv[1], nb, dim);
    auto query = load_fbin(argv[2], nq, d2);
    const uint32_t n_train = argc > 3 ? std::atoi(argv[3]) : 20000;
    const uint32_t k = argc > 4 ? std::atoi(argv[4]) : 10;
    const uint32_t n_q = argc > 5 ? std::atoi(argv[5])
                                  : std::min<uint32_t>(nq, 500);
    std::printf("base=%u dim=%u queries=%u/%u n_train=%u k=%u\n",
                nb, dim, n_q, nq, n_train, k);

    struct Cfg { uint32_t m, bits; };
    const Cfg cfgs[] = {
        {48, 4},   // 24B/vec — matched-byte rival of m96-2b
        {96, 4},   // 48B/vec — the shipped tier (baseline)
        {192, 4},  // 96B/vec
        {96, 8},   // 96B/vec
        {384, 8},  // 384B/vec — the max-recall tier
        {96, 2},   // 24B/vec — the target tier
        {192, 2},  // 48B/vec — 2-bit at matched bytes
        {384, 2},  // 96B/vec — 2-bit ceiling
        {48, 2},   // 12B/vec — floor
    };

    for (const auto& c : cfgs) {
        auto t0 = std::chrono::steady_clock::now();
        PqQuantizer q(MetricKind::L2Sq, dim, static_cast<uint16_t>(c.m),
                      static_cast<uint8_t>(c.bits), 42);
        q.train(base.data(), std::min<uint64_t>(n_train, nb));
        const uint32_t cs = q.code_size();
        std::vector<uint8_t> codes(static_cast<size_t>(nb) * cs);
        std::vector<float> decoded(static_cast<size_t>(nb) * dim);
        for (uint32_t i = 0; i < nb; ++i) {
            q.encode(base.data() + static_cast<size_t>(i) * dim,
                     codes.data() + static_cast<size_t>(i) * cs);
            q.decode_code(codes.data() + static_cast<size_t>(i) * cs,
                     decoded.data() + static_cast<size_t>(i) * dim);
        }
        auto t1 = std::chrono::steady_clock::now();

        // Float LUT per query (m x K partial dots) — same math as the
        // engine's LUT rerank / scan.
        const uint32_t K = q.K(), sub = q.sub_dim(), m = c.m;
        const float* cb = q.codebook();
        uint64_t hits_topk = 0, hits_cont = 0, total = 0;
        uint64_t tau_ids = 0, tau_hits = 0;
        const uint32_t W = 300;
        const float tau = 2.0f;
        for (uint32_t qi = 0; qi < n_q; ++qi) {
            const float* qv = query.data() + static_cast<size_t>(qi) * dim;
            std::vector<float> lut(static_cast<size_t>(m) * K);
            for (uint32_t sg = 0; sg < m; ++sg)
                for (uint32_t kc = 0; kc < K; ++kc) {
                    const float* cw = cb + (static_cast<size_t>(sg) * K + kc) * sub;
                    float dsum = 0.f;
                    for (uint32_t j = 0; j < sub; ++j)
                        dsum += qv[static_cast<size_t>(sg) * sub + j] * cw[j];
                    lut[static_cast<size_t>(sg) * K + kc] = dsum;
                }
            // Scores + exact GT.
            std::vector<std::pair<float, uint32_t>> sc(nb), ex(nb);
            for (uint32_t i = 0; i < nb; ++i) {
                const uint8_t* code = codes.data() + static_cast<size_t>(i) * cs;
                float acc = 0.f;
                for (uint32_t sg = 0; sg < m; ++sg) {
                    uint32_t idx = 0;
                    if (c.bits == 4) {
                        const uint8_t b = code[sg / 2];
                        idx = (sg % 2 == 0) ? (b & 0xF) : (b >> 4);
                    } else if (c.bits == 2) {
                        const uint8_t b = code[sg / 4];
                        idx = (b >> (2 * (sg % 4))) & 0x3;
                    } else {
                        idx = code[sg];
                    }
                    acc += lut[static_cast<size_t>(sg) * K + idx];
                }
                sc[i] = {-acc, i};
                float de = 0.f;
                const float* xv = base.data() + static_cast<size_t>(i) * dim;
                for (uint32_t d = 0; d < dim; ++d) {
                    const float e = qv[d] - xv[d];
                    de += e * e;
                }
                ex[i] = {de, i};
            }
            std::partial_sort(sc.begin(), sc.begin() + W, sc.end());
            std::partial_sort(ex.begin(), ex.begin() + k, ex.end());
            std::vector<uint32_t> gt(k);
            for (uint32_t a = 0; a < k; ++a) gt[a] = ex[a].second;
            for (uint32_t a = 0; a < k; ++a)
                for (uint32_t b = 0; b < k; ++b)
                    if (sc[b].second == gt[a]) { hits_topk++; break; }
            for (uint32_t a = 0; a < W; ++a)
                for (uint32_t b = 0; b < k; ++b)
                    if (sc[a].second == gt[b]) { hits_cont++; break; }
            // Adaptive-W-style cut on the (negated) quantizer scores.
            // Signal self-calibration mirrors the engine: gap measured
            // against the top-k region's own spread.
            const float g = (sc[k - 1].first - sc[0].first) /
                            std::max(1.0f * (k - 1), 1e-9f);
            const float thresh = sc[k - 1].first + tau * std::max(g, 1e-9f);
            uint32_t kept = 0;
            for (uint32_t a = 0; a < W; ++a) {
                if (sc[a].first > thresh) break;
                ++kept;
                for (uint32_t b = 0; b < k; ++b)
                    if (sc[a].second == gt[b]) { tau_hits++; break; }
            }
            tau_ids += kept;
            total += k;
        }
        const double bpv = static_cast<double>(cs);
        std::printf("m=%3u bits=%u %5.0fB/vec train+enc %.1fs | top-%u %.4f | "
                    "contain@%u %.4f | tau2 %.4f @ %.1f ids\n",
                    c.m, c.bits, bpv,
                    std::chrono::duration<double>(t1 - t0).count(),
                    k, static_cast<double>(hits_topk) / total,
                    W, static_cast<double>(hits_cont) / total,
                    static_cast<double>(tau_hits) / total,
                    static_cast<double>(tau_ids) / n_q);
    }
    return 0;
}
