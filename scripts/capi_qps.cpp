// capi_qps: batch-throughput driver (the retrievalbench protocol gap —
// its warm p50 is single-query nq=1). T threads each loop the query set;
// reports total QPS + recall@k per operating point. Mirrors sextant_bench
// --threads=N batch semantics over the C ABI.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>
#include <atomic>
#include "sextant/sextant_c.h"

namespace {
struct Fbin {
    std::vector<float> vecs; uint32_t n = 0, dim = 0;
};
Fbin load(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) exit(1);
    Fbin o;
    if (fread(&o.n, 4, 1, f) != 1 || fread(&o.dim, 4, 1, f) != 1) exit(1);
    o.vecs.resize((size_t)o.n * o.dim);
    if (fread(o.vecs.data(), 4, o.vecs.size(), f) != o.vecs.size()) exit(1);
    fclose(f); return o;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: capi_qps <tree> <query.fbin> [threads=8] [k=10] "
            "[exact_base.fbin]\n"
            "  exact_base.fbin: rerank against the original corpus "
            "(exact_rerank_base) instead of decode.\n"
            "  Remaining knobs are env (bench-harness convention): "
            "EXH I8 W TAU NP ROUNDS.\n");
        return 1;
    }
    const char* tree = argv[1];
    Fbin q = load(argv[2]);
    // Optional 5th positional: rerank against the original corpus
    // (exact_rerank_base) instead of decode — the frontier's audit hatch.
    // EXACT=<base.fbin> env still works (legacy manual-run form).
    const char* exact_path = argc > 5 ? argv[5] : getenv("EXACT");
    std::vector<float> exact_base;
    if (exact_path) {
        Fbin b = load(exact_path);
        if (b.dim != q.dim) { fprintf(stderr, "dim mismatch\n"); return 1; }
        exact_base = std::move(b.vecs);
    }
    const uint32_t T = argc > 3 ? atoi(argv[3]) : 8;
    const uint32_t k = argc > 4 ? atoi(argv[4]) : 10;
    const uint32_t nq = q.n;
    auto env = [](const char* n, const char* d) { const char* v = getenv(n); return v ? v : d; };
    const int exhaustive = atoi(env("EXH", "0"));
    const int int8 = atoi(env("I8", "-1"));
    const uint32_t W = atoi(env("W", "300"));
    const float tau = atof(env("TAU", "2"));
    const uint32_t np = atoi(env("NP", "8"));
    const uint32_t rounds = atoi(env("ROUNDS", "3"));

    char err[512] = {0};
    void* idx = sextant_open_index(tree, err, sizeof err);
    if (!idx) { fprintf(stderr, "open: %s\n", err); return 1; }

    for (uint32_t r = 0; r < rounds; ++r) {
        std::atomic<uint64_t> done{0};
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        for (uint32_t t = 0; t < T; ++t) {
            ts.emplace_back([&] {
                sextant_search_opts so = sextant_default_search_opts();
                so.k = k; so.n_probe = np; so.n_probe_ln = 8;
                so.fastscan_W = W; so.rerank = 1;
                so.adaptive_w_gap = tau; so.int8_scan = int8;
                so.exhaustive = exhaustive; so.search_threads = 0;
                if (exact_base.data())
                    so.exact_rerank_base = exact_base.data();
                std::vector<uint64_t> ids(65536);
                std::vector<float> ds(65536);
                for (uint32_t i = r; i < nq * 2; ++i) {
                    const uint32_t qi = i % nq;
                    sextant_search(idx, &q.vecs[(size_t)qi * q.dim], &so,
                                   ids.data(), ds.data(), 65536, err, sizeof err);
                    done.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : ts) th.join();
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t total = done.load();
        // last round is steady-state
        if (r == rounds - 1)
            printf("threads=%u k=%u%s%s np=%u W=%u tau=%.1f: %.0f QPS (%.3f ms/query amortized, %llu queries)\n",
                   T, k, exhaustive ? " EXH" : "", exact_base.data() ? " EXACT" : "",
                   np, W, tau, total / s, s * 1000.0 / total,
                   (unsigned long long)total);
    }
    sextant_close_index(idx);
    return 0;
}
