// capi_arrival: mixed-arrival serving bench over the sextant_scheduler_*
// CAPI. Open-loop arrivals at a fixed rate ladder; a pool of submitter
// threads performs BLOCKING sextant_scheduler_submit calls (the P/Invoke
// form the .NET geocoder will use). Per rate: achieved QPS, end-to-end
// latency p50/p99, recall@k vs optional ground truth, and scheduler
// stats deltas (window coalescing factor, delay percentiles).
//
// usage: capi_arrival <tree> <queries.fbin> [threads=16] [k=10]
//        [gt.fbin (u32 ids, n*k, fbin header) | "-" for none]
//
// Env knobs (scripts convention — engine code stays env-free):
//   ARR_RATES   comma-separated QPS ladder (default 8,32,128,256)
//   ARR_DUR_S   per-rate duration in seconds (default 6)
//   ARR_IDLE_US scheduler idle_close_us (default 200)
//   ARR_MAX_US  scheduler window_max_us (default 100000)
//   ARR_INFLIGHT scheduler max_inflight_windows (default 2)
//   ARR_STHREADS scheduler search_threads (default = threads arg)
//   ARR_CACHE   result_cache_entries (default 0)
//   ARR_MIX_PCT % of requests URGENT (max_delay_us=1000; rest 0 = default)
//   ARR_LC / ARR_PC  leaf / plane cache MiB for sextant_open_index
//
// Cold-path discipline (docs/BENCHMARK_RULES.md) is the caller's job:
// drop kernel caches + ZFS sync before each invocation for cold numbers.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <atomic>
#include <mutex>
#include "sextant/sextant_c.h"

using namespace std::chrono_literals;

namespace {

struct Fbin {
    std::vector<float> vecs; uint32_t n = 0, dim = 0;
};
Fbin load_f32(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
    Fbin o;
    if (fread(&o.n, 4, 1, f) != 1 || fread(&o.dim, 4, 1, f) != 1) exit(1);
    o.vecs.resize((size_t)o.n * o.dim);
    if (fread(o.vecs.data(), 4, o.vecs.size(), f) != o.vecs.size()) exit(1);
    fclose(f); return o;
}
// Ground truth: fbin-header u32 id matrix [n][k].
std::vector<uint32_t> load_gt(const char* p, uint32_t n, uint32_t k) {
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
    uint32_t gn = 0, gk = 0;
    if (fread(&gn, 4, 1, f) != 1 || fread(&gk, 4, 1, f) != 1) exit(1);
    if (gn < n || gk < k) { fprintf(stderr, "gt too small\n"); exit(1); }
    std::vector<uint32_t> v((size_t)n * k);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
    fclose(f); return v;
}

double pct_ms(std::vector<uint64_t>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1,
                              static_cast<size_t>(p * (v.size() - 1)));
    return static_cast<double>(v[i]) / 1e6;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tree> <queries.fbin> [threads=16] "
                        "[k=10] [gt.fbin|'-']\n", argv[0]);
        return 1;
    }
    const char* tree = argv[1];
    Fbin q = load_f32(argv[2]);
    const uint32_t T = argc > 3 ? atoi(argv[3]) : 16;
    const uint32_t k = argc > 4 ? atoi(argv[4]) : 10;
    std::vector<uint32_t> gt;
    bool have_gt = false;
    if (argc > 5 && strcmp(argv[5], "-") != 0) {
        gt = load_gt(argv[5], q.n, k);
        have_gt = true;
    }

    auto env = [](const char* n, const char* d) {
        const char* v = getenv(n); return v ? v : d;
    };
    std::vector<double> rates;
    {
        char buf[256]; snprintf(buf, sizeof(buf), "%s", env("ARR_RATES", "8,32,128,256"));
        for (char* tok = strtok(buf, ","); tok; tok = strtok(nullptr, ","))
            rates.push_back(atof(tok));
    }
    const double dur_s = atof(env("ARR_DUR_S", "6"));
    const uint64_t idle_us = strtoull(env("ARR_IDLE_US", "200"), nullptr, 10);
    const uint64_t max_us = strtoull(env("ARR_MAX_US", "100000"), nullptr, 10);
    const uint32_t inflight = atoi(env("ARR_INFLIGHT", "2"));
    const uint32_t sthreads = atoi(env("ARR_STHREADS", argv[3]));
    const uint32_t cache_e = atoi(env("ARR_CACHE", "0"));
    const int mix_pct = atoi(env("ARR_MIX_PCT", "0"));
    const uint64_t lc_mib = strtoull(env("ARR_LC", "0"), nullptr, 10);
    const uint64_t pc_mib = strtoull(env("ARR_PC", "0"), nullptr, 10);

    char err[512] = {0};
    void* index = sextant_open_index(tree, lc_mib << 20, pc_mib << 20,
                                     err, sizeof(err));
    if (!index) { fprintf(stderr, "open: %s\n", err); return 1; }
    if (sextant_index_dim(index) != q.dim) {
        fprintf(stderr, "dim mismatch\n"); return 1;
    }

    sextant_search_opts base = sextant_default_search_opts();
    base.k = k;
    sextant_scheduler_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.base = &base;
    cfg.idle_close_us = idle_us;
    cfg.window_max_us = max_us;
    cfg.max_inflight_windows = inflight;
    cfg.search_threads = sthreads;
    cfg.result_cache_entries = cache_e;

    printf("# capi_arrival tree=%s nq=%u k=%u threads=%u idle_us=%lu "
           "max_us=%lu inflight=%u sthreads=%u cache=%u mix_pct=%d gt=%s\n",
           tree, q.n, k, T, idle_us, max_us, inflight, sthreads, cache_e,
           mix_pct, have_gt ? "on" : "off");
    printf("# rate_qps\tachieved\tlat_p50_ms\tlat_p99_ms\trecall@%u\t"
           "windows\tavg_win\tdelay_p50_us\tdelay_p99_us\tcache_hits\n", k);
    fflush(stdout);

    for (double rate : rates) {
        void* sched = sextant_scheduler_create(index, &cfg, err, sizeof(err));
        if (!sched) { fprintf(stderr, "create: %s\n", err); return 1; }

        std::atomic<uint64_t> next{0};
        std::atomic<uint64_t> done{0};
        std::atomic<uint64_t> hits{0};
        std::mutex lat_mu;
        std::vector<uint64_t> lat;  // end-to-end ns
        const auto t_start = std::chrono::steady_clock::now();
        const auto deadline =
            t_start + std::chrono::nanoseconds((uint64_t)(dur_s * 1e9));

        std::vector<std::thread> threads;
        for (uint32_t t = 0; t < T; ++t) {
            threads.emplace_back([&, t] {
                std::vector<uint64_t> local_lat;
                uint64_t local_hits = 0;
                for (uint64_t i = t; ; i += T) {
                    const auto now = std::chrono::steady_clock::now();
                    // Open-loop pacing: request i is due at t_start +
                    // i/rate; block-free sleep until then, bail past the
                    // measurement deadline.
                    const auto due = t_start +
                        std::chrono::nanoseconds((uint64_t)(1e9 * (double)i / rate));
                    if (due >= deadline) break;
                    if (now < due) std::this_thread::sleep_until(due);
                    const uint64_t qi = next.fetch_add(1) % q.n;
                    const uint64_t delay = (mix_pct > 0 &&
                        (int)((i * 997) % 100) < mix_pct) ? 1000 : 0;
                    uint64_t ids[64]; float d[64];
                    if (k > 64) return;
                    char e[512] = {0};
                    const int32_t n = sextant_scheduler_submit(
                        sched, &q.vecs[(size_t)qi * q.dim], k,
                        nullptr, 0, delay, 0.0f, ids, d, k, e, sizeof(e));
                    if (n < 0) { fprintf(stderr, "submit: %s\n", e); return; }
                    if (have_gt) {
                        uint32_t h = 0;
                        for (int32_t j = 0; j < n; ++j)
                            for (uint32_t g = 0; g < k; ++g)
                                if (gt[(size_t)qi * k + g] ==
                                    (uint32_t)ids[j]) { ++h; break; }
                        local_hits += h;
                    }
                    local_lat.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - due).count());
                }
                done += local_lat.size();
                hits += local_hits;
                { std::lock_guard<std::mutex> lk(lat_mu);
                  lat.insert(lat.end(), local_lat.begin(), local_lat.end()); }
            });
        }
        for (auto& th : threads) th.join();
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();

        sextant_scheduler_stats_t st;
        sextant_scheduler_stats(sched, &st);
        sextant_scheduler_stop(sched);

        const double achieved = done.load() / wall;
        const double recall = have_gt && done
            ? (double)hits.load() / ((double)done.load() * k) : -1;
        printf("%8.1f\t%8.1f\t%10.3f\t%10.3f\t%8.4f\t%7llu\t%7.1f\t"
               "%12.1f\t%12.1f\t%10llu\n",
               rate, achieved, pct_ms(lat, 0.50), pct_ms(lat, 0.99), recall,
               (unsigned long long)st.windows,
               st.windows ? (double)st.queries / st.windows : 0.0,
               st.delay_p50_ns / 1e3, st.delay_p99_ns / 1e3,
               (unsigned long long)st.cache_hits);
        fflush(stdout);
    }
    sextant_close_index(index);
    return 0;
}
