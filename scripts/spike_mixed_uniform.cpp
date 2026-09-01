// Spike: mixed bit allocation (top-variance dims @ 8-bit, rest @ 4-bit) and
// uniform-vs-LloydMax scalar quantization, measured IN ISOLATION (no tree).
//
// Answers two questions before any engine integration:
//  1. Where does the mixed 4/8-bit recall frontier sit vs PQ8 at equal bytes?
//     (PQ8 references, cohere np8: 96B=57.9%, 192B=79.5%, 256B=86.3%,
//      384B=93.5%)
//  2. What recall does uniform (equidistant) levels give up vs Lloyd-Max?
//     Uniform codes decode with sequential level tables — potential QPS
//     lever (no Lloyd-Max gather pattern change, but simpler training and a
//     cheaper encode; the scan gather is identical, see notes).
//
// Methodology identical to spike_lm_recall: exact top-k on full original
// dims vs top-k on quantized vectors, overlap = recall@k.
//
// Usage: spike_mixed_uniform <base.fbin> <query.fbin> [n_train] [k] [mode] [dims8]
//   lm4       4-bit Lloyd-Max everywhere (baseline; matches spike_lm_recall)
//   uniform4  4-bit uniform everywhere
//   uniform8  8-bit uniform everywhere
//   mixed     top `dims8` variance dims @ 8-bit LM, rest @ 4-bit LM

#include "../src/quant/scalar_lloydmax_quantizer.hpp"

#include <algorithm>
#include <chrono>
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

// Uniform scalar quantizer: per-dim [min,max] -> K equidistant levels.
// Level table has the same dim*K layout as Lloyd-Max, so the scan kernel
// shape is unchanged — only the training and code assignment differ.
struct UniformQuant {
    uint32_t dim = 0, K = 0;
    std::vector<float> lo, step, levels;
    void train(const float* data, uint32_t n, uint32_t dim_, uint32_t bits) {
        dim = dim_; K = 1u << bits;
        lo.assign(dim, 0.f); step.assign(dim, 0.f);
        levels.assign(static_cast<size_t>(dim) * K, 0.f);
        for (uint32_t d = 0; d < dim; ++d) {
            float mn = data[d], mx = data[d];
            for (uint32_t i = 1; i < n; ++i) {
                const float x = data[static_cast<size_t>(i) * dim + d];
                mn = std::min(mn, x); mx = std::max(mx, x);
            }
            lo[d] = mn;
            step[d] = K > 1 ? (mx - mn) / (K - 1) : 0.f;
            for (uint32_t k = 0; k < K; ++k)
                levels[static_cast<size_t>(d) * K + k] = mn + step[d] * k;
        }
    }
    inline uint8_t encode1(const float* v, uint32_t d) const {
        const float t = (v[d] - lo[d]) / (step[d] + 1e-30f);
        return static_cast<uint8_t>(
            std::min<int>(static_cast<int>(K) - 1,
                          std::max(0, static_cast<int>(t + 0.5f))));
    }
    inline float decode1(uint32_t d, uint8_t c) const {
        return levels[static_cast<size_t>(d) * K + c];
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s base query [n_train] [k] "
                     "[lm4|uniform4|uniform8|mixed] [dims8]\n",
                     argv[0]);
        return 1;
    }
    uint32_t nb, dq, nq, d2;
    auto base = load_fbin(argv[1], nb, dq);
    auto query = load_fbin(argv[2], nq, d2);
    const uint32_t dim = dq;
    const uint32_t n_train = argc > 3 ? std::atoi(argv[3]) : 20000;
    const uint32_t k = argc > 4 ? std::atoi(argv[4]) : 10;
    const std::string mode = argc > 5 ? argv[5] : "mixed";
    const uint32_t dims8 = argc > 6 ? std::atoi(argv[6]) : dim / 2;
    const uint64_t ns = std::min<uint64_t>(n_train, nb);
    std::printf("base=%u dim=%u queries=%u mode=%s dims8=%u\n", nb, dim, nq,
                mode.c_str(), dims8);

    // Per-dim variance on the training sample.
    std::vector<double> var(dim, 0.0), mean(dim, 0.0);
    for (uint64_t i = 0; i < ns; ++i)
        for (uint32_t d = 0; d < dim; ++d) mean[d] += base[i * dim + d];
    for (uint32_t d = 0; d < dim; ++d) mean[d] /= ns;
    for (uint64_t i = 0; i < ns; ++i)
        for (uint32_t d = 0; d < dim; ++d) {
            const double x = base[i * dim + d] - mean[d];
            var[d] += x * x;
        }
    std::vector<uint32_t> order(dim);
    for (uint32_t d = 0; d < dim; ++d) order[d] = d;
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return var[a] > var[b]; });
    std::vector<uint8_t> bits_of(dim, 4);
    if (mode == "mixed")
        for (uint32_t i = 0; i < dims8 && i < dim; ++i) bits_of[order[i]] = 8;
    else if (mode == "uniform8")
        for (uint32_t d = 0; d < dim; ++d) bits_of[d] = 8;

    const bool lm = (mode == "lm4") || (mode == "mixed");

    // Decode every base vector: per bit-group submatrix -> quantizer -> dequant.
    std::vector<float> decoded(static_cast<size_t>(nb) * dim, 0.f);
    double bytes_per_vec = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (uint8_t b : {4u, 8u}) {
        std::vector<uint32_t> idx;
        for (uint32_t d = 0; d < dim; ++d)
            if (bits_of[d] == b) idx.push_back(d);
        const uint32_t od = static_cast<uint32_t>(idx.size());
        if (od == 0) continue;
        bytes_per_vec += static_cast<double>(od) * b / 8.0;
        std::vector<float> sub(static_cast<size_t>(ns) * od);
        for (uint64_t i = 0; i < ns; ++i)
            for (uint32_t j = 0; j < od; ++j)
                sub[static_cast<size_t>(i) * od + j] =
                    base[static_cast<size_t>(i) * dim + idx[j]];
        if (lm) {
            ScalarLloydMaxQuantizer q(MetricKind::L2Sq, od, b);
            q.train(sub.data(), static_cast<uint32_t>(ns));
            const uint32_t cs = q.code_size();
            const uint32_t K = q.K();
            const float* lv = q.levels();
            std::vector<uint8_t> code(cs);
            std::vector<float> dv(od);
            for (uint32_t i = 0; i < nb; ++i) {
                // ScalarLloydMaxQuantizer expects contiguous dim=od vectors;
                // re-pack this vector's selected dims.
                std::vector<float> one(od);
                for (uint32_t j = 0; j < od; ++j)
                    one[j] = base[static_cast<size_t>(i) * dim + idx[j]];
                q.encode(one.data(), code.data());
                q.decode(code.data(), dv.data());
                for (uint32_t j = 0; j < od; ++j)
                    decoded[static_cast<size_t>(i) * dim + idx[j]] = dv[j];
            }
            (void)cs; (void)K; (void)lv;
        } else {
            UniformQuant u;
            u.train(sub.data(), static_cast<uint32_t>(ns), od, b);
            for (uint32_t i = 0; i < nb; ++i)
                for (uint32_t j = 0; j < od; ++j) {
                    const uint32_t d = idx[j];
                    const uint8_t c = u.encode1(
                        base.data() + static_cast<size_t>(i) * dim, d);
                    decoded[static_cast<size_t>(i) * dim + d] = u.decode1(d, c);
                }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("encode+decode: %.2fs (%.0f B/vec)\n",
                std::chrono::duration<double>(t1 - t0).count(), bytes_per_vec);

    // Recall@k: exact (full dims) vs quantized, brute force.
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
            for (uint32_t b2 = 0; b2 < k; ++b2)
                if (exact[a].second == quant[b2].second) { hits++; break; }
        total += k;
    }
    std::printf("isolated recall@%u: %.4f  (%.0f B/vec)\n", k,
                static_cast<double>(hits) / total, bytes_per_vec);
    return 0;
}
