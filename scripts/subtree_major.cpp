// subtree_major: 1B-access-pattern experiment at 10M scale.
//
// Question: at single-digit-% residency, caching can't be the protagonist —
// can QUERY-COALESCED (subtree-major) access order replace it? Compare:
//   query-major:   each query reads its own probed leaf extents (order of
//                  arrival; no cache = every extent re-read per query)
//   subtree-major: leaves processed once in page order; every query that
//                  probed a leaf evaluates against that one shared read
// Measures wall time + bytes read for both, cold, via pread.
//
// Collection phase opens the index WITH a leaf cache (fast; probe sets are
// deterministic and independent of the cache). Replay uses raw preads into
// a recycled buffer (no page-cache reliance beyond one pass).
//
// Manual build:
//   clang++ -std=c++23 -O2 -I include -I src -I <nsync-public> \
//     scripts/subtree_major.cpp build-x86/libsextant.a <libnsync.a> \
//     -lspdlog -lfmt -lpthread -o /tmp/subtree_major

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <thread>
#include <atomic>

#include "tree/ivf_tree_index.hpp"
#include "tools/fbin_io.hpp"
#include <cstring>

using namespace sextant;
using namespace sextant::tree;
using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tree> <queries.fbin> [threads=8]\n",
                     argv[0]);
        return 1;
    }
    const int nthreads = argc > 3 ? std::atoi(argv[3]) : 8;

    // --- Collect per-query probed leaf first-pages --------------------------
    sextant::fbin_io::FbinHeader qh;
    {
        std::FILE* f = std::fopen(argv[2], "rb");
        if (!f || std::fread(&qh, 4, 2, f) != 2) { std::fprintf(stderr, "qhdr\n"); return 1; }
        std::fclose(f);
    }
    std::vector<float> queries((size_t)qh.n * qh.dim);
    {
        std::FILE* f = std::fopen(argv[2], "rb");
        std::fseek(f, 8, SEEK_SET);
        if (std::fread(queries.data(), 4, queries.size(), f) !=
            queries.size()) { std::fprintf(stderr, "qread\n"); return 1; }
        std::fclose(f);
    }

    auto idx = IVFTreeIndex::open(argv[1], 4ull << 30 /* fast collection */);
    SearchConfig sc;
    sc.k = 10;
    sc.probe_fraction = 0.05;
    sc.adaptive_probe_gap = 0.0f;
    sc.fastscan_W = 1000;

    auto leaf_info = idx->debug_leaf_info();  // (page, pages, count)
    std::unordered_map<PageId, uint32_t> leaf_pages;
    for (auto& li : leaf_info)
        leaf_pages[li.page] = static_cast<uint32_t>(li.pages);

    std::vector<std::vector<PageId>> qleaves(qh.n);
    const auto tc0 = Clock::now();
    for (uint32_t qi = 0; qi < qh.n; ++qi) {
        std::vector<PageId> vis;
        idx->search(&queries[(size_t)qi * qh.dim], 10, sc, nullptr, nullptr,
                    nullptr, &vis);
        qleaves[qi] = std::move(vis);
    }
    std::fprintf(stderr, "[collect] %u queries in %.1fs\n", qh.n,
                 secs(tc0, Clock::now()));

    uint64_t total_visits = 0, total_bytes_qm = 0;
    std::unordered_set<PageId> uniq;
    for (auto& v : qleaves) {
        total_visits += v.size();
        for (PageId p : v) { uniq.insert(p); total_bytes_qm += (uint64_t)leaf_pages[p] * 4096; }
    }
    uint64_t total_bytes_sm = 0;
    for (PageId p : uniq) total_bytes_sm += (uint64_t)leaf_pages[p] * 4096;
    std::fprintf(stderr, "[sets] visits=%llu unique_leaves=%zu/%zu qm=%.1fGiB sm=%.1fGiB amplif=%.1fx\n",
                 (unsigned long long)total_visits, uniq.size(), leaf_info.size(),
                 total_bytes_qm / 1073741824.0, total_bytes_sm / 1073741824.0,
                 (double)total_bytes_qm / (double)total_bytes_sm);

    // --- Replay harness -------------------------------------------------------
    const int fd = ::open(argv[1], O_RDONLY);
    std::vector<uint8_t> buf(64ull << 20);

    auto read_extent = [&](PageId page, uint32_t pages) -> uint64_t {
        const uint64_t sz = (uint64_t)pages * 4096;
        if (sz > buf.size()) return sz;  // oversize: count only
        const off_t off = (off_t)page * 4096;
        uint64_t done = 0;
        while (done < sz) {
            ssize_t r = ::pread(fd, buf.data() + done, sz - done,
                                off + (off_t)done);
            if (r <= 0) break;
            done += r;
        }
        return sz;
    };

    // Query-major: queries in parallel (nthreads streams), each reads its
    // own extents in page order (best case for query-major).
    {
        std::atomic<uint64_t> bytes{0};
        std::atomic<uint32_t> next{0};
        const auto t0 = Clock::now();
        std::vector<std::thread> ths;
        for (int t = 0; t < nthreads; ++t) {
            ths.emplace_back([&] {
                std::vector<PageId> mine;
                while (true) {
                    const uint32_t qi = next.fetch_add(1);
                    if (qi >= qh.n) break;
                    mine = qleaves[qi];
                    std::sort(mine.begin(), mine.end());
                    for (PageId p : mine)
                        bytes += read_extent(p, leaf_pages[p]);
                }
            });
        }
        for (auto& th : ths) th.join();
        std::printf("query-major  : %.2fs  bytes=%.1fGiB  (%.1f GB/s)\n",
                    secs(t0, Clock::now()),
                    bytes.load() / 1073741824.0,
                    bytes.load() / 1e9 / secs(t0, Clock::now()));
    }

    // Subtree-major: each unique leaf read ONCE, in page order; queries
    // touching it are "evaluated" (counted). nthreads streams partition the
    // page-ordered leaf list into contiguous ranges (sequential streams).
    {
        std::vector<PageId> leaves(uniq.begin(), uniq.end());
        std::sort(leaves.begin(), leaves.end());
        std::unordered_map<PageId, int> fanout;
        for (auto& v : qleaves)
            for (PageId p : v) fanout[p]++;
        std::atomic<uint64_t> bytes{0}, evals{0};
        std::atomic<size_t> next_leaf{0};
        const auto t0 = Clock::now();
        std::vector<std::thread> ths;
        for (int t = 0; t < nthreads; ++t) {
            ths.emplace_back([&] {
                while (true) {
                    const size_t li = next_leaf.fetch_add(1);
                    if (li >= leaves.size()) break;
                    const PageId p = leaves[li];
                    bytes += read_extent(p, leaf_pages[p]);
                    evals += fanout[p];  // queries served by this one read
                }
            });
        }
        for (auto& th : ths) th.join();
        std::printf("subtree-major: %.2fs  bytes=%.1fGiB  evals=%llu  (%.1f GB/s)\n",
                    secs(t0, Clock::now()),
                    bytes.load() / 1073741824.0,
                    (unsigned long long)evals.load(),
                    bytes.load() / 1e9 / secs(t0, Clock::now()));
    }
    (void)total_visits;
    return 0;
}
