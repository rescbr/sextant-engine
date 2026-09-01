// Spike: measure ScalarLloydMaxQuantizer quality in ISOLATION (no tree).
//
// Encodes base vectors, decodes them, brute-forces top-k from the decoded
// vectors, and compares against exact-FP32 ground truth. This isolates
// "how much recall does 4-bit Lloyd-Max quantization alone cost" from tree
// routing / probing / shortlist effects.
//
// Usage: ./spike_lm_recall <base.fbin> <query.fbin> [n_train] [k] [bits]
//                        [keep_dims]
// bits:     scalar quantizer bit width (default 4).
// keep_dims: raw dimension selection — keep only the keep_dims highest-
//            variance dims, NO rotation (PCA is a measured trap: rotation
//            Gaussianizes marginals via CLT and destroys the Lloyd-Max
//            advantage on heavy-tailed data). Exact top-k is always computed
//            on the FULL original dims, so recall folds in the information
//            lost by dropping dims.

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
        std::fprintf(stderr,
                     "usage: %s base.fbin query.fbin [n_train] [k] [bits] "
                     "[keep_dims]\n",
                     argv[0]);
        return 1;
    }
    uint32_t nb, dq, nq, d2;
    auto base = load_fbin(argv[1], nb, dq);
    auto query = load_fbin(argv[2], nq, d2);
    const uint32_t dim = dq;
    const uint32_t n_train = argc > 3 ? std::atoi(argv[3]) : 20000;
    const uint32_t k = argc > 4 ? std::atoi(argv[4]) : 10;
    const uint32_t bits = argc > 5 ? std::atoi(argv[5]) : 4;
    const uint32_t keep = argc > 6 ? std::atoi(argv[6]) : dim;
    std::printf("base=%u dim=%u keep=%u queries=%u n_train=%u k=%u bits=%u\n",
                nb, dim, keep, nq, n_train, k, bits);

    // Raw dimension selection: keep the `keep` highest-variance dims.
    // Exact ground truth below always uses the full original dims.
    const float* train_base = base.data();
    const float* train_query = query.data();
    std::vector<float> sel_base, sel_query;
    std::vector<uint32_t> sel_idx;
    if (keep < dim) {
        std::vector<double> var(dim, 0.0), mean(dim, 0.0);
        const uint64_t ns = std::min<uint64_t>(n_train, nb);
        for (uint64_t i = 0; i < ns; ++i)
            for (uint32_t d = 0; d < dim; ++d) mean[d] += base[i * dim + d];
        for (uint32_t d = 0; d < dim; ++d) mean[d] /= ns;
        for (uint64_t i = 0; i < ns; ++i)
            for (uint32_t d = 0; d < dim; ++d) {
                const double x = base[i * dim + d] - mean[d];
                var[d] += x * x;
            }
        sel_idx.resize(dim);
        for (uint32_t d = 0; d < dim; ++d) sel_idx[d] = d;
        std::sort(sel_idx.begin(), sel_idx.end(), [&](uint32_t a, uint32_t b) {
            return var[a] > var[b];
        });
        sel_idx.resize(keep);
        std::sort(sel_idx.begin(), sel_idx.end());
        sel_base.resize(static_cast<size_t>(nb) * keep);
        sel_query.resize(static_cast<size_t>(nq) * keep);
        for (uint32_t i = 0; i < nb; ++i)
            for (uint32_t d = 0; d < keep; ++d)
                sel_base[static_cast<size_t>(i) * keep + d] =
                    base[static_cast<size_t>(i) * dim + sel_idx[d]];
        for (uint32_t i = 0; i < nq; ++i)
            for (uint32_t d = 0; d < keep; ++d)
                sel_query[static_cast<size_t>(i) * keep + d] =
                    query[static_cast<size_t>(i) * dim + sel_idx[d]];
        train_base = sel_base.data();
        train_query = sel_query.data();
    }
    const uint32_t qdim = keep;

    // Train on a sample.
    ScalarLloydMaxQuantizer q(MetricKind::L2Sq, qdim, bits);
    auto t0 = std::chrono::steady_clock::now();
    q.train(base.data(), std::min<uint64_t>(n_train, nb));
    auto t1 = std::chrono::steady_clock::now();
    std::printf("train: %.2fs\n",
                std::chrono::duration<double>(t1 - t0).count());

    // Encode + decode ALL base vectors.
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes(static_cast<size_t>(nb) * cs);
    std::vector<float> decoded(static_cast<size_t>(nb) * qdim);
    t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < nb; ++i) {
        q.encode(train_base + static_cast<size_t>(i) * qdim,
                 codes.data() + static_cast<size_t>(i) * cs);
        q.decode(codes.data() + static_cast<size_t>(i) * cs,
                 decoded.data() + static_cast<size_t>(i) * qdim);
    }
    t1 = std::chrono::steady_clock::now();
    std::printf("encode+decode: %.2fs (%.1f MB codes)\n",
                std::chrono::duration<double>(t1 - t0).count(),
                codes.size() / 1e6);

    // MSE (informational).
    double mse = 0;
    for (size_t i = 0; i < decoded.size(); ++i)
        mse += static_cast<double>(train_base[i] - decoded[i]) *
               (train_base[i] - decoded[i]);
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
        const float* qs = train_query + static_cast<size_t>(qi) * qdim;
        std::vector<std::pair<float, uint32_t>> exact(nb), quant(nb),
            bydot(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            float dq2 = 0, dot = 0;
            // exact: full original dims (true neighbors)
            float de = 0;
            const float* be = base.data() + static_cast<size_t>(i) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                const float x = qv[d] - be[d];
                de += x * x;
            }
            const float* bq = decoded.data() + static_cast<size_t>(i) * qdim;
            for (uint32_t d = 0; d < qdim; ++d) {
                dq2 += (qs[d] - bq[d]) * (qs[d] - bq[d]);
                dot += qs[d] * bq[d];
            }
            exact[i] = {de, i};
            quant[i] = {dq2, i};
            bydot[i] = {-dot, i};          // scan ordering (IP-style)
        }
        std::partial_sort(exact.begin(), exact.begin() + k, exact.end());
        std::partial_sort(quant.begin(), quant.begin() + k, quant.end());
        // Shortlists of size W_short.
        std::partial_sort(bydot.begin(), bydot.begin() + W_short,
                          bydot.end());
        std::vector<std::pair<float, uint32_t>> pipe;  // rerank by L2sq
        for (uint32_t a = 0; a < W_short; ++a) {
            const uint32_t i = bydot[a].second;
            const float* bq = decoded.data() + static_cast<size_t>(i) * qdim;
            float dd = 0;
            for (uint32_t d = 0; d < qdim; ++d)
                dd += (qs[d] - bq[d]) * (qs[d] - bq[d]);
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
