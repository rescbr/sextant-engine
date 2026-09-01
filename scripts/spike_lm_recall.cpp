// Spike: measure ScalarLloydMaxQuantizer quality in ISOLATION (no tree).
//
// Encodes base vectors, decodes them, brute-forces top-k from the decoded
// vectors, and compares against exact-FP32 ground truth. This isolates
// "how much recall does 4-bit Lloyd-Max quantization alone cost" from tree
// routing / probing / shortlist effects.
//
// Usage: ./spike_lm_recall <base.fbin> <query.fbin> [n_train] [k] [metric]

#include "../src/quant/scalar_lloydmax_quantizer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>

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
        std::fprintf(stderr, "usage: %s base.fbin query.fbin [n_train] [k]\n",
                     argv[0]);
        return 1;
    }
    uint32_t nb, dq, nq, d2;
    auto base = load_fbin(argv[1], nb, dq);
    auto query = load_fbin(argv[2], nq, d2);
    const uint32_t dim = dq;
    const uint32_t n_train = argc > 3 ? std::atoi(argv[3]) : 20000;
    const uint32_t k = argc > 4 ? std::atoi(argv[4]) : 10;
    std::printf("base=%u dim=%u queries=%u n_train=%u k=%u\n", nb, dim, nq,
                n_train, k);

    // Train on a sample.
    ScalarLloydMaxQuantizer q(MetricKind::L2Sq, dim, 4);
    auto t0 = std::chrono::steady_clock::now();
    q.train(base.data(), std::min<uint64_t>(n_train, nb));
    auto t1 = std::chrono::steady_clock::now();
    std::printf("train: %.2fs\n",
                std::chrono::duration<double>(t1 - t0).count());

    // Encode + decode ALL base vectors.
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes(static_cast<size_t>(nb) * cs);
    std::vector<float> decoded(static_cast<size_t>(nb) * dim);
    t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < nb; ++i) {
        q.encode(base.data() + static_cast<size_t>(i) * dim,
                 codes.data() + static_cast<size_t>(i) * cs);
        q.decode(codes.data() + static_cast<size_t>(i) * cs,
                 decoded.data() + static_cast<size_t>(i) * dim);
    }
    t1 = std::chrono::steady_clock::now();
    std::printf("encode+decode: %.2fs (%.1f MB codes)\n",
                std::chrono::duration<double>(t1 - t0).count(),
                codes.size() / 1e6);

    // MSE (informational).
    double mse = 0;
    for (size_t i = 0; i < decoded.size(); ++i)
        mse += static_cast<double>(base[i] - decoded[i]) *
               (base[i] - decoded[i]);
    mse /= decoded.size();
    std::printf("MSE: %.6g\n", mse);

    // Brute-force top-k: exact (base) vs decoded, count overlap.
    // Also replicates the tree scan pipeline: shortlist by -dot on decoded
    // (the scan's ordering), rerank by L2sq on decoded. If the -dot ordering
    // loses the true neighbors before the shortlist cut, that's the tree's
    // recall gap — even though the quantizer itself is fine.
    const uint32_t W_short = 300;
    uint64_t hits = 0, total = 0;
    uint64_t hits_pipe = 0;   // tree pipeline (shortlist by -dot, rerank L2sq)
    uint64_t hits_fix = 0;    // fixed scan (shortlist by L2sq on decoded)
    for (uint32_t qi = 0; qi < nq; ++qi) {
        const float* qv = query.data() + static_cast<size_t>(qi) * dim;
        std::vector<std::pair<float, uint32_t>> exact(nb), quant(nb),
            bydot(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            float de = 0, dq2 = 0, dot = 0, vnorm = 0;
            const float* be = base.data() + static_cast<size_t>(i) * dim;
            const float* bq = decoded.data() + static_cast<size_t>(i) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                de += (qv[d] - be[d]) * (qv[d] - be[d]);
                dq2 += (qv[d] - bq[d]) * (qv[d] - bq[d]);
                dot += qv[d] * bq[d];
                vnorm += bq[d] * bq[d];
            }
            exact[i] = {de, i};
            quant[i] = {dq2, i};
            bydot[i] = {-dot, i};          // scan ordering (IP-style)
            (void)vnorm;
        }
        std::partial_sort(exact.begin(), exact.begin() + k, exact.end());
        std::partial_sort(quant.begin(), quant.begin() + k, quant.end());
        // Shortlists of size W_short.
        std::partial_sort(bydot.begin(), bydot.begin() + W_short,
                          bydot.end());
        std::vector<std::pair<float, uint32_t>> pipe;  // rerank by L2sq
        for (uint32_t a = 0; a < W_short; ++a) {
            const uint32_t i = bydot[a].second;
            const float* bq = decoded.data() + static_cast<size_t>(i) * dim;
            float dd = 0;
            for (uint32_t d = 0; d < dim; ++d)
                dd += (qv[d] - bq[d]) * (qv[d] - bq[d]);
            pipe.push_back({dd, i});
        }
        std::partial_sort(pipe.begin(), pipe.begin() + k, pipe.end());
        // Fixed scan: shortlist by true L2sq on decoded, take top-k.
        std::vector<std::pair<float, uint32_t>> fixl(
            quant.begin(), quant.begin() + W_short);

        for (uint32_t a = 0; a < k; ++a) {
            for (uint32_t b = 0; b < k; ++b)
                if (exact[a].second == quant[b].second) { hits++; break; }
            for (uint32_t b = 0; b < k; ++b)
                if (exact[a].second == pipe[b].second) { hits_pipe++; break; }
            for (uint32_t b = 0; b < k; ++b)
                if (exact[a].second == fixl[b].second) { hits_fix++; break; }
        }
        total += k;
    }
    std::printf("quantizer-only recall@%u:        %.4f\n", k,
                static_cast<double>(hits) / total);
    std::printf("tree pipeline (shortlist -dot):  %.4f\n",
                static_cast<double>(hits_pipe) / total);
    std::printf("fixed scan (shortlist L2sq):     %.4f\n",
                static_cast<double>(hits_fix) / total);
    return 0;
}
