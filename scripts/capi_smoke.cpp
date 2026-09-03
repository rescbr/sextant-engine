// capi_smoke: exercise the C ABI end-to-end — build from .fbin (local_scalar),
// open, exhaustive (probe-all) vs routed search, recall vs fp32 ground truth
// computed inline. Mirrors what engines/sextant.rs will do from Rust.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>

#include "sextant/sextant_c.h"

namespace {

struct Fbin {
    std::vector<float> vecs;
    uint32_t n = 0, dim = 0;
};

Fbin load_fbin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); exit(1); }
    Fbin out;
    if (fread(&out.n, 4, 1, f) != 1 || fread(&out.dim, 4, 1, f) != 1) exit(1);
    out.vecs.resize(static_cast<size_t>(out.n) * out.dim);
    if (fread(out.vecs.data(), 4, out.vecs.size(), f) != out.vecs.size()) exit(1);
    fclose(f);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const char* base_path = argc > 1 ? argv[1] : "/tmp/cohere_100k_base_norm.fbin";
    const char* query_path = argc > 2 ? argv[2] : "/tmp/cohere_100k_query_norm.fbin";
    const char* quantizer = argc > 3 ? argv[3] : "local_scalar";
    const uint32_t k = argc > 4 ? uint32_t(atoi(argv[4])) : 10;
    const uint32_t nq = argc > 5 ? uint32_t(atoi(argv[5])) : 200;
    const char* tree_override = getenv("TREE");

    Fbin base = load_fbin(base_path);
    Fbin query = load_fbin(query_path);
    if (query.dim != base.dim) { fprintf(stderr, "dim mismatch\n"); return 1; }
    const uint32_t nq_final = std::min(nq, query.n);
    printf("base %u x %u, queries %u, k=%u, quantizer=%s\n",
           base.n, base.dim, nq_final, k, quantizer);

    // --- Build through the C ABI ---
    sextant_build_opts bo = sextant_default_build_opts();
    bo.quantizer = quantizer;
    if (getenv("KR")) bo.k_root = uint32_t(atoi(getenv("KR")));
    if (getenv("LEAF")) bo.leaf_capacity = uint32_t(atoi(getenv("LEAF")));
    const char* m = getenv("METRIC");
    bo.metric = (m && m[0] == "l"[0]) ? SEXTANT_METRIC_L2SQ : SEXTANT_METRIC_IP;
    bo.num_threads = 8;
    char err[512] = {0};
    const std::string tree_path = tree_override ? tree_override : "/tmp/capi_smoke.tree";
    auto t0 = std::chrono::steady_clock::now();
    if (!tree_override) {
        const char* mb = getenv("MEMBUILD");
        int rc = mb
            ? sextant_build_mem(base.vecs.data(), base.n, base.dim,
                                tree_path.c_str(), &bo, err, sizeof(err))
            : sextant_build_fbin(base_path, tree_path.c_str(), &bo, err, sizeof(err));
        if (rc != 0) { fprintf(stderr, "build failed: %s\n", err); return 1; }
    }
    double build_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("build: %.2fs\n", build_s);

    void* idx = sextant_open_index(tree_path.c_str(), err, sizeof(err));
    if (!idx) { fprintf(stderr, "open failed: %s\n", err); return 1; }
    printf("opened: dim=%u live=%llu\n", sextant_index_dim(idx),
           (unsigned long long)sextant_index_live_count(idx));

    // --- Exact IP ground truth (multi-threaded) ---
    std::vector<std::vector<uint32_t>> gt(nq);
    {
        const uint32_t T = 8;
        std::vector<std::thread> ts;
        for (uint32_t t = 0; t < T; ++t) {
            ts.emplace_back([&, t] {
                for (uint32_t q = t; q < nq_final; q += T) {
                    const float* qv = &query.vecs[size_t(q) * base.dim];
                    // top-k by IP via partial selection
                    std::vector<std::pair<float, uint32_t>> scores;
                    scores.reserve(base.n);
                    for (uint32_t i = 0; i < base.n; ++i) {
                        const float* v = &base.vecs[size_t(i) * base.dim];
                        float s = 0.f;
                        for (uint32_t d = 0; d < base.dim; ++d) s += qv[d] * v[d];
                        scores.emplace_back(s, i);
                    }
                    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                                      [](auto& a, auto& b) { return a.first > b.first; });
                    for (uint32_t i = 0; i < k; ++i) gt[q].push_back(scores[i].second);
                }
            });
        }
        for (auto& th : ts) th.join();
    }

    // --- Search: routed/exhaustive x rerank off/on ----------------------
    // Config via env: MODES (mask, default 15), NP/NPLN (0 = manifest),
    // W (0 = default), TAU (0 = off, requires rerank), ST (search threads),
    // EXHNP (1 = probe-all overrides NP/NPLN).
    auto env = [](const char* n, long d) { const char* v = getenv(n); return v ? atol(v) : d; };
    const uint32_t cfg_np = env("NP", 0), cfg_npln = env("NPLN", 0);
    const uint32_t cfg_W = env("W", 0), cfg_st = env("ST", 0);
    const float cfg_tau = env("TAU", 2);
    const int modes = env("MODES", 15);
    for (int mode = 0; mode < 4; ++mode) {
        if (!(modes & (1 << mode))) continue;
        sextant_search_opts so = sextant_default_search_opts();
        so.k = k;
        so.exhaustive = mode & 1;
        so.n_probe = cfg_np;
        so.n_probe_ln = cfg_npln;
        so.fastscan_W = cfg_W;
        so.rerank = mode & 2;
        so.adaptive_w_gap = (mode & 2) ? cfg_tau : 0.0f;
        so.search_threads = cfg_st;
        so.int8_scan = env("I8", -1);
        // EXACT=1: rerank against the loaded base vectors (the driver keeps
        // them in memory anyway for GT) — the exact-rerank A/B knob.
        if (env("EXACT", 0)) so.exact_rerank_base = base.vecs.data();
        std::vector<uint64_t> ids(65536);
        std::vector<float> dists(65536);
        double recall = 0.0, crecall = 0.0, erecall = 0.0, gsum = 0.0;
        auto t1 = std::chrono::steady_clock::now();
        for (uint32_t q = 0; q < nq_final; ++q) {
            const float* qv0 = &query.vecs[size_t(q) * base.dim];
            int32_t got = sextant_search(idx, qv0, &so, ids.data(), dists.data(),
                                         65536, err, sizeof(err));
            if (got < 0) { fprintf(stderr, "search failed (mode %d): %s\n", mode, err); return 1; }
            // Adaptive-W contract: exact fp32 rerank of the shortlist, top-k.
            uint32_t hits = 0;
            if (so.rerank && got > 0) {
                std::vector<std::pair<float, uint32_t>> rescore(got);
                for (int32_t i = 0; i < got; ++i) {
                    const float* v = &base.vecs[size_t(ids[i]) * base.dim];
                    float sc = 0.f;
                    for (uint32_t d = 0; d < base.dim; ++d) sc += qv0[d] * v[d];
                    rescore[i] = {-sc, static_cast<uint32_t>(ids[i])};
                }
                std::partial_sort(rescore.begin(), rescore.begin() + k, rescore.end());
                for (uint32_t i = 0; i < k && i < rescore.size(); ++i)
                    for (uint32_t g : gt[q])
                        if (g == rescore[i].second) { ++hits; break; }
            } else {
                for (int32_t i = 0; i < got && i < int32_t(k); ++i)
                    for (uint32_t g : gt[q])
                        if (g == ids[i]) { ++hits; break; }
            }
            recall += double(hits) / k;
            // Engine-order recall: hits among the FIRST k ids as returned
            // (the engine's own rerank ranking, no external rescore). The
            // gap to `recall` (exact rescore of the same shortlist) is the
            // headroom for rerank-precision work (fp16/exact rerank).
            uint32_t ehits = 0;
            for (int32_t i = 0; i < got && i < int32_t(k); ++i)
                for (uint32_t g : gt[q])
                    if (g == ids[i]) { ++ehits; break; }
            erecall += double(ehits) / k;
            uint32_t chits = 0;
            for (int32_t i = 0; i < got; ++i)
                for (uint32_t g : gt[q])
                    if (g == ids[i]) { ++chits; break; }
            crecall += double(chits) / k; gsum += got;
        }
        double ms = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t1).count() * 1000.0 / nq_final;
        printf("%s %s: recall@%u = %.4f, %.3f ms/query "
               "(containment %.4f, engine-order %.4f, shortlist %.1f)\n",
               mode & 1 ? "EXHAUSTIVE" : "ROUTED   ",
               mode & 2 ? "+rerank" : "raw    ", k, recall / nq_final, ms,
               crecall / nq_final, erecall / nq_final, gsum / nq_final);
    }

    sextant_close_index(idx);
    return 0;
}
