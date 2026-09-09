// Mid-sweep joining simulator (2026-09-09).
//
// Decides go/no-go on true mid-sweep joining (a late query attaching to
// an in-flight sweep for leaves not yet swept) vs the current policy
// (arrivals wait for the next window). Uses REAL per-query probe sets
// (search() visited_leaf_pages) and a sweep-time model calibrated from
// real search_batch walls at several window sizes.
//
// Model, per arriving query at rate R (fixed spacing, open loop):
//   - leaves(i): page-sorted unique probe positions mapped into the
//     global page order (the sweep order of search_batch).
//   - sweep of a window W: walks the UNION of its queries' leaf
//     positions in page order; wall = (fixed + per_leaf*|union| +
//     per_scan*|pairs|) / threads (99% busy measured — see
//     results/leaf_cache/read_layer_20260909.md).
//   - NEXT policy (current BatchScheduler, offline): a free slot
//     closes its window at max(slot-free, last-arrival + idle gap),
//     capped by the front's deadline (100 ms); latency = dispatch
//     wait + window sweep wall.
//   - JOIN policy: identical dispatch, but an in-flight sweep ADOPTS a
//     new arrival if every leaf in its probe set is still unswept
//     (rank in the sweep's sorted leaf list >= leaves swept so far);
//     the joiner completes at its LAST leaf's sweep time; leaves not
//     already in the sweep's union are single-read extra bytes and
//     extend the sweep.
//
// Env knobs (scripts convention): JOIN_RATES (ladder), JOIN_DUR_S (20),
// JOIN_IDLE_US (200), JOIN_INFLIGHT (2), JOIN_THREADS (8),
// JOIN_Q (probe-set queries, 300).
//
// usage: join_sim <tree> <queries.fbin>

#include "tree/ivf_tree_index.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/null_sink.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double percentile_ms(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1,
                      static_cast<size_t>(p * (v.size() - 1)))] * 1e3;
}

struct Calib {
    double fixed_s = 0;     // window overhead (route + finalize)
    double per_leaf_s = 0;  // sweep time per unique leaf (incl. read)
    double per_scan_s = 0;  // scan+harvest per (query, leaf) pair
};

// Least-squares fit wall = fixed + per_leaf*U + per_scan*S.
Calib fit_calib(const std::vector<std::array<double, 3>>& pts) {
    std::array<std::array<double, 4>, 3> a{};
    for (const auto& [w, u, s] : pts) {
        const double x[3] = {1.0, u, s};
        for (uint32_t r = 0; r < 3; ++r) {
            for (uint32_t c = 0; c < 3; ++c) a[r][c] += x[r] * x[c];
            a[r][3] += x[r] * w;
        }
    }
    for (uint32_t col = 0; col < 3; ++col) {
        uint32_t piv = col;
        for (uint32_t r = col + 1; r < 3; ++r)
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
        std::swap(a[col], a[piv]);
        if (std::abs(a[col][col]) < 1e-12) return {};
        for (uint32_t r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = a[r][col] / a[col][col];
            for (uint32_t c = col; c < 4; ++c) a[r][c] -= f * a[col][c];
        }
    }
    return {a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]};
}

struct Sweep {
    double start_s = 0;
    double wall_s = 0;             // extends when joiners add leaves
    std::vector<uint32_t> leaves;  // sorted universe positions
    double end_s() const { return start_s + wall_s; }
    double done_leaves(double t) const {
        return std::clamp((t - start_s) / wall_s *
                              static_cast<double>(leaves.size()),
                          0.0, static_cast<double>(leaves.size()));
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tree> <queries.fbin>\n", argv[0]);
        return 1;
    }
    spdlog::set_default_logger(
        spdlog::create<spdlog::sinks::null_sink_st>("null"));

    uint32_t qn = 0, qdim = 0;
    {
        std::ifstream f(argv[2], std::ios::binary);
        if (!f.read(reinterpret_cast<char*>(&qn), 4) ||
            !f.read(reinterpret_cast<char*>(&qdim), 4) || qn == 0) {
            std::fprintf(stderr, "qhdr\n");
            return 1;
        }
    }
    std::vector<float> queries(static_cast<size_t>(qn) * qdim);
    {
        std::ifstream f(argv[2], std::ios::binary);
        f.seekg(8);
        if (!f.read(reinterpret_cast<char*>(queries.data()),
                    static_cast<std::streamsize>(
                        queries.size() * sizeof(float)))) {
            std::fprintf(stderr, "qread\n");
            return 1;
        }
    }

    auto idx = sextant::tree::IVFTreeIndex::open(argv[1]);
    if (qdim != idx->dim()) {
        std::fprintf(stderr, "dim mismatch\n");
        return 1;
    }

    const uint32_t n_probe_q =
        std::getenv("JOIN_Q") ? std::atoi(std::getenv("JOIN_Q")) : 300;
    const uint32_t threads = std::getenv("JOIN_THREADS")
                                 ? std::atoi(std::getenv("JOIN_THREADS")) : 8;
    const double idle_s =
        std::getenv("JOIN_IDLE_US")
            ? std::atoll(std::getenv("JOIN_IDLE_US")) / 1e6 : 200e-6;
    const uint32_t max_inflight = std::getenv("JOIN_INFLIGHT")
                                      ? std::atoi(std::getenv("JOIN_INFLIGHT")) : 2;
    const double dur_s =
        std::getenv("JOIN_DUR_S") ? std::atof(std::getenv("JOIN_DUR_S")) : 20.0;
    const double deadline_s = 100e-3;  // SCHED_MAX_US default in harness
    std::vector<double> rates = {5, 10, 20, 50, 100, 200, 400};
    if (const char* r = std::getenv("JOIN_RATES")) {
        rates.clear();
        std::string s(r);
        size_t pos = 0;
        while ((pos = s.find(',')) != std::string::npos) {
            rates.push_back(std::atof(s.substr(0, pos).c_str()));
            s.erase(0, pos + 1);
        }
        rates.push_back(std::atof(s.c_str()));
    }
    const uint32_t nq = std::min<uint32_t>(n_probe_q, qn);

    sextant::SearchConfig sc;
    sc.k = 10;
    sc.search_threads = threads;

    // --- Real probe sets ---
    std::vector<std::vector<uint32_t>> leaf_pos(nq);
    {
        std::set<sextant::tree::PageId> uni;
        std::vector<std::vector<sextant::tree::PageId>> per_q(nq);
        for (uint32_t i = 0; i < nq; ++i) {
            std::vector<sextant::tree::PageId> pages;
            (void)idx->search(&queries[size_t(i) * qdim], 10, sc, nullptr,
                              nullptr, nullptr, &pages);
            std::sort(pages.begin(), pages.end());
            pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
            per_q[i] = std::move(pages);
            uni.insert(per_q[i].begin(), per_q[i].end());
        }
        const std::vector<sextant::tree::PageId> order(uni.begin(),
                                                       uni.end());
        for (uint32_t i = 0; i < nq; ++i) {
            leaf_pos[i].reserve(per_q[i].size());
            for (auto p : per_q[i])
                leaf_pos[i].push_back(
                    static_cast<uint32_t>(std::lower_bound(order.begin(),
                                                           order.end(), p) -
                                          order.begin()));
        }
        const double mean_leaves =
            std::accumulate(leaf_pos.begin(), leaf_pos.end(), 0.0,
                            [](double a, const auto& v) {
                                return a + static_cast<double>(v.size());
                            }) /
            nq;
        std::cout << "# probe universe leaves=" << order.size()
                  << " mean_leaves/query=" << mean_leaves << "\n";
    }

    // --- Calibrate from real search_batch walls ---
    Calib cal;
    const double per_leaf_serial_margin = 0;  // see join model note
    {
        std::vector<std::array<double, 3>> pts;
        std::vector<float> buf;
        std::vector<std::vector<sextant::Candidate>> out;
        for (uint32_t w : {1u, 8u, 64u, 512u}) {
            const uint32_t m = std::min(w, nq);
            buf.assign(static_cast<size_t>(m) * qdim, 0.f);
            for (uint32_t i = 0; i < m; ++i)
                std::copy_n(&queries[size_t(i) * qdim], qdim,
                            &buf[size_t(i) * qdim]);
            for (int rep = 0; rep < 2; ++rep) {
                const auto t0 = Clock::now();
                idx->search_batch(buf.data(), m, 10, sc, out);
                const double wall =
                    std::chrono::duration<double>(Clock::now() - t0).count();
                if (rep == 1) {
                    std::set<uint32_t> u;
                    double scans = 0;
                    for (uint32_t i = 0; i < m; ++i) {
                        u.insert(leaf_pos[i].begin(), leaf_pos[i].end());
                        scans += static_cast<double>(leaf_pos[i].size());
                    }
                    pts.push_back({wall, static_cast<double>(u.size()), scans});
                }
            }
        }
        cal = fit_calib(pts);
        if (const char* m = std::getenv("JOIN_LEAF_MULT"))
            cal.per_leaf_s *= std::atof(m);  // model colder sweeps
        std::cout << "# calib (wall_s, union, scans):";
        for (const auto& p : pts)
            std::cout << " (" << p[0] << "," << p[1] << "," << p[2] << ")";
        std::cout << "\n# fit: fixed=" << cal.fixed_s
                  << "s per_leaf=" << cal.per_leaf_s
                  << "s per_scan=" << cal.per_scan_s << "s (serial, /"
                  << threads << " in model)\n";
    }
    const double leaf_extra_serial =
        std::max(0.0, cal.per_leaf_s + per_leaf_serial_margin) / threads;
    const auto sweep_wall = [&](const std::vector<uint32_t>& members) {
        std::set<uint32_t> u;
        double scans = 0;
        for (uint32_t q : members) {
            u.insert(leaf_pos[q].begin(), leaf_pos[q].end());
            scans += static_cast<double>(leaf_pos[q].size());
        }
        return (std::max(0.0, cal.fixed_s) +
                std::max(0.0, cal.per_leaf_s) *
                    static_cast<double>(u.size()) +
                std::max(0.0, cal.per_scan_s) * scans) /
               std::max(1u, threads);
    };

    // --- Arrival replay ---
    const auto simulate = [&](double rate, bool join_on) {
        std::vector<Sweep> inflight;
        std::deque<std::pair<uint32_t, double>> queue;  // (query, arrival)
        std::vector<double> lat_s;
        uint64_t join_hits = 0, join_extra_leaves = 0, join_miss = 0;
        const double step = 1.0 / rate;
        const uint64_t n_arr =
            static_cast<uint64_t>(rate * dur_s) + 1;
        uint64_t arrived = 0;
        double last_arrival = -1e9;

        while (arrived < n_arr || !queue.empty() || !inflight.empty()) {
            // Earliest event: arrival, sweep end, or dispatch.
            const double ta = arrived < n_arr
                                  ? static_cast<double>(arrived) * step
                                  : 1e18;
            double te = 1e18;
            for (const auto& s : inflight) te = std::min(te, s.end_s());
            double td = 1e18;  // earliest window close
            if (!queue.empty() && inflight.size() < max_inflight) {
                td = std::max(last_arrival + idle_s,
                              queue.front().second);
                td = std::min(td, queue.front().second + deadline_s);
            }
            const double t = std::min({ta, te, td});
            if (t >= 1e18) break;

            // Retire finished sweeps.
            inflight.erase(
                std::remove_if(inflight.begin(), inflight.end(),
                               [&](const Sweep& s) { return t >= s.end_s(); }),
                inflight.end());

            // Arrival first (an arrival can postpone the idle close).
            if (ta == t) {
                const uint32_t q = static_cast<uint32_t>(arrived % nq);
                bool joined = false;
                if (join_on) {
                    for (auto& s : inflight) {
                        const double done = s.done_leaves(ta);
                        uint32_t extra = 0;
                        bool ok = true;
                        double max_rank = 0;
                        for (uint32_t lp : leaf_pos[q]) {
                            const auto it =
                                std::lower_bound(s.leaves.begin(),
                                                 s.leaves.end(), lp);
                            if (it != s.leaves.end() && *it == lp) {
                                const double rank =
                                    it - s.leaves.begin();
                                if (rank < done) { ok = false; break; }
                                max_rank = std::max(max_rank, rank);
                            } else {
                                ++extra;  // leaf not in sweep's union
                            }
                        }
                        if (!ok) continue;
                        // Adopt: completion at the joiner's last leaf.
                        double last_leaf_rank = 0;
                        {
                            std::vector<uint32_t> merged(s.leaves);
                            for (uint32_t lp : leaf_pos[q])
                                merged.push_back(lp);
                            std::sort(merged.begin(), merged.end());
                            merged.erase(std::unique(merged.begin(),
                                                     merged.end()),
                                         merged.end());
                            for (uint32_t lp : leaf_pos[q])
                                last_leaf_rank = std::max(
                                    last_leaf_rank,
                                    static_cast<double>(
                                        std::lower_bound(merged.begin(),
                                                         merged.end(), lp) -
                                        merged.begin()));
                            s.leaves = std::move(merged);
                        }
                        s.wall_s += static_cast<double>(extra) * leaf_extra_serial;
                        const double frac =
                            (last_leaf_rank + 1) /
                            static_cast<double>(s.leaves.size());
                        const double done_at =
                            s.start_s + frac * s.wall_s;
                        lat_s.push_back(std::max(0.0, done_at - ta));
                        joined = true;
                        ++join_hits;
                        join_extra_leaves += extra;
                        break;
                    }
                }
                if (!joined) {
                    if (join_on) ++join_miss;
                    queue.push_back({q, ta});
                }
                last_arrival = ta;
                ++arrived;
            }
            // Dispatch (only if no arrival happened exactly now that
            // reopened the window — the queue-close recomputes anyway).
            if (!queue.empty() && inflight.size() < max_inflight) {
                double close = std::max(last_arrival + idle_s,
                                        queue.front().second);
                close = std::min(close,
                                 queue.front().second + deadline_s);
                if (t >= close) {
                    Sweep s;
                    s.start_s = t;
                    std::vector<uint32_t> members;
                    for (const auto& [q, arr] : queue) {
                        members.push_back(q);
                        (void)arr;
                    }
                    s.wall_s = sweep_wall(members);
                    for (uint32_t q : members)
                        for (uint32_t lp : leaf_pos[q]) s.leaves.push_back(lp);
                    std::sort(s.leaves.begin(), s.leaves.end());
                    s.leaves.erase(
                        std::unique(s.leaves.begin(), s.leaves.end()),
                        s.leaves.end());
                    const double done = s.end_s();
                    for (const auto& [q, arr] : queue)
                        lat_s.push_back(std::max(0.0, done - arr));
                    inflight.push_back(std::move(s));
                    queue.clear();
                }
            }
        }
        (void)join_miss;
        std::cout << std::fixed << std::setprecision(1) << rate << " "
                  << (join_on ? "JOIN" : "NEXT") << " n=" << lat_s.size()
                  << " p50=" << percentile_ms(lat_s, 0.50)
                  << " p95=" << percentile_ms(lat_s, 0.95)
                  << " p99=" << percentile_ms(lat_s, 0.99)
                  << "ms joins=" << join_hits
                  << " extra_leaves=" << join_extra_leaves << "\n";
    };

    for (double rate : rates) {
        simulate(rate, /*join_on=*/false);
        simulate(rate, /*join_on=*/true);
    }
    return 0;
}
