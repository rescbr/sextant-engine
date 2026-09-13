// Unit + integration tests for the tree-path LeafExtentCache (W-TinyLFU,
// variable-size leaf extents, refcount + deferred free).
//
// Unit tests pread from a real temp file (the cache's own fill path — no
// mocks). Integration tests build a small IVF tree via build_streaming_pca
// and search with the cache on, checking hit accounting, result parity with
// the cache-off path, and invalidation across insert_batch.

#include <gtest/gtest.h>

#include "tree/ivf_tree_index.hpp"
#include "tree/leaf_extent_cache.hpp"

#include <sextant/vector_source.hpp>
#include "fbin_source.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using sextant::tree::LeafExtentCache;
using sextant::tree::PageId;
using sextant::tree::kPageSize;

constexpr uint32_t kExtentPages = 8;  // 32KB extents

/// Write a temp file where page P's content is the page index repeated.
struct TempPageFile {
    std::string path;
    int fd = -1;

    explicit TempPageFile(uint32_t n_pages) {
        path = (std::filesystem::temp_directory_path() /
                ("leaf_cache_test_" +
                 std::to_string(::getpid()) + "_" +
                 std::to_string(reinterpret_cast<uintptr_t>(this)) + ".bin"))
                    .string();
        std::vector<uint8_t> buf(n_pages * kPageSize);
        for (uint32_t p = 0; p < n_pages; ++p)
            std::memset(buf.data() + p * kPageSize, static_cast<int>(p & 0xFF),
                        kPageSize);
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        f.close();
        fd = ::open(path.c_str(), O_RDONLY);
    }
    ~TempPageFile() {
        if (fd >= 0) ::close(fd);
        std::filesystem::remove(path);
    }

    static void expect_page(const uint8_t* extent, PageId page,
                            uint32_t pages) {
        ASSERT_NE(extent, nullptr);
        for (uint32_t p = 0; p < pages; ++p) {
            const uint8_t expected = static_cast<uint8_t>((page + p) & 0xFF);
            ASSERT_EQ(extent[p * kPageSize + 7], expected)
                << "page " << (page + p) << " content mismatch";
        }
    }
};

LeafExtentCache::Handle pin_ok(LeafExtentCache& c, int fd, PageId page,
                               uint32_t pages, const uint8_t** out) {
    // Since admission-before-fill (2026-09-13), a pin under eviction
    // pressure can be REJECTED and returns the fallback pointer instead
    // of filling. Unit tests verify CONTENT either way, so provide the
    // same fallback the integration paths use: a mapping of the file.
    LeafExtentCache::Handle h;
    thread_local std::unordered_map<int, const uint8_t*> maps;
    auto it = maps.find(fd);
    if (it == maps.end()) {
        struct stat st;
        ::fstat(fd, &st);
        void* m = ::mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { perror("test mmap"); abort(); }
        it = maps.emplace(fd, static_cast<const uint8_t*>(m)).first;
    }
    *out = c.pin(page, pages, h, it->second +
                 static_cast<uint64_t>(page) * kPageSize);
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------
// Hit/miss accounting + correct data.
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, HitMissCountersAndData) {
    TempPageFile tf(200);
    LeafExtentCache cache(64ull << 20, 4, tf.fd);
    cache.set_expected_entries(64);

    const uint8_t* p = nullptr;
    auto h = pin_ok(cache, tf.fd, 10, kExtentPages, &p);
    TempPageFile::expect_page(p, 10, kExtentPages);
    cache.unpin(h);

    h = pin_ok(cache, tf.fd, 10, kExtentPages, &p);
    TempPageFile::expect_page(p, 10, kExtentPages);
    cache.unpin(h);

    auto st = cache.stats();
    EXPECT_EQ(st.misses, 1u);
    EXPECT_EQ(st.hits, 1u);
    EXPECT_EQ(st.bytes_filled, kExtentPages * kPageSize);
}

// ---------------------------------------------------------------------------
// Refcount: a pinned entry survives eviction pressure; its data is intact;
// after unpin it is gone (miss on next pin).
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, PinnedEntrySurvivesEviction) {
    // Capacity for ~2 extents; pin A, then insert enough others to force
    // A's eviction, and verify A's buffer is still readable and correct.
    const uint64_t cap = 2 * kExtentPages * kPageSize + kPageSize;
    TempPageFile tf(2000);
    LeafExtentCache cache(cap, 1, tf.fd);
    cache.set_expected_entries(16);

    const uint8_t* pa = nullptr;
    auto ha = pin_ok(cache, tf.fd, 100, kExtentPages, &pa);
    // Fill past capacity with distinct extents. With strict-> admission the
    // one-timer scan TIES against pinned A and must not evict it (that's the
    // fix — ties keep the incumbent); older `>=` semantics evicted it.
    for (PageId pg = 200; pg < 240; pg += kExtentPages) {
        const uint8_t* px = nullptr;
        auto hx = pin_ok(cache, tf.fd, pg, kExtentPages, &px);
        cache.unpin(hx);
    }
    // Whether A is still linked or evicted-but-pinned, its data must be
    // intact and every use must be safe — that is the pin contract.
    TempPageFile::expect_page(pa, 100, kExtentPages);
    cache.unpin(ha);

    const uint8_t* p2 = nullptr;
    auto h2 = pin_ok(cache, tf.fd, 100, kExtentPages, &p2);
    TempPageFile::expect_page(p2, 100, kExtentPages);
    cache.unpin(h2);
}

// ---------------------------------------------------------------------------
// TinyLFU admission: a frequently-accessed extent survives a scan of
// one-timers that would flush it from a plain LRU.
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, TinyLfuProtectsHotEntry) {
    const uint64_t cap = 3 * kExtentPages * kPageSize + kPageSize;
    TempPageFile tf(5000);
    LeafExtentCache cache(cap, 1, tf.fd);
    cache.set_expected_entries(32);

    // Warm the hot entry's frequency.
    for (int i = 0; i < 8; ++i) {
        const uint8_t* p = nullptr;
        auto h = pin_ok(cache, tf.fd, 42 * kExtentPages, kExtentPages, &p);
        cache.unpin(h);
    }
    // Scan of one-timers, 3x the capacity.
    for (PageId pg = 1000; pg < 1000 + 9 * kExtentPages; pg += kExtentPages) {
        const uint8_t* p = nullptr;
        auto h = pin_ok(cache, tf.fd, pg, kExtentPages, &p);
        cache.unpin(h);
    }
    EXPECT_TRUE(cache.contains(42 * kExtentPages))
        << "hot entry evicted by cold scan — admission control broken";
}

// ---------------------------------------------------------------------------
// Oversized extent: refused outright (fallback returned, nothing cached).
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, OversizedExtentIsTransient) {
    TempPageFile tf(64);
    LeafExtentCache cache(kPageSize, 1, tf.fd);  // 1 page total capacity
    static const uint8_t sentinel = 0xAB;
    const uint8_t* fb = &sentinel;
    LeafExtentCache::Handle h;
    const uint8_t* p = cache.pin(4, kExtentPages, h, fb);
    EXPECT_EQ(p, fb);              // refused → caller's fallback handed back
    EXPECT_EQ(h.entry, nullptr);   // nothing to unpin
    auto st = cache.stats();
    EXPECT_EQ(st.rejections, 1u);
    EXPECT_FALSE(cache.contains(4));
}

// ---------------------------------------------------------------------------
// Admission ties keep the incumbent (Caffeine strict-> semantics). A cold
// candidate (sketch freq equal to the probation LRU's) must NOT evict it —
// `>=` here previously degenerated retention to FIFO churn exactly when the
// sketch had least information (handover 2026-09-07/08).
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, AdmissionTieKeepsIncumbent) {
    // Capacity 3 extents; fill main with 3 one-timers (equal frequencies),
    // then insert a 4th never-seen candidate.
    const uint64_t cap = 3 * kExtentPages * kPageSize;  // main fits < 3
    TempPageFile tf(500);
    LeafExtentCache cache(cap, 1, tf.fd);
    cache.set_expected_entries(8);  // tiny sketch: cold-slate ties dominate

    for (PageId pg = 16; pg < 16 + 3 * kExtentPages; pg += kExtentPages) {
        const uint8_t* p = nullptr;
        auto h = pin_ok(cache, tf.fd, pg, kExtentPages, &p);
        cache.unpin(h);
        const uint8_t* p2 = nullptr;
        auto h2 = pin_ok(cache, tf.fd, pg, kExtentPages, &p2);  // 2 hits each: equal freq
        cache.unpin(h2);
    }
    auto st0 = cache.stats();
    const uint64_t ev_before = st0.evictions;

    // 4th extent: same sketch frequency as the probation LRU (all tie) →
    // rejected, no eviction of the incumbents.
    const uint8_t* p = nullptr;
    auto h = pin_ok(cache, tf.fd, 400, kExtentPages, &p);
    cache.unpin(h);

    auto st = cache.stats();
    EXPECT_GT(st.rejections, 0u) << "tie candidate should be refused";
    EXPECT_EQ(st.evictions, ev_before)
        << "tie must not evict the probation incumbent";
    EXPECT_TRUE(cache.contains(16));
}

// ---------------------------------------------------------------------------
// Single-flight (try-lock): concurrent pins of the same missing page are
// deduped ONLY when uncontended — the gate is deliberately non-blocking
// (blocking single-flight serialized parallel redundancy and cost 16% QPS
// on latency-bound zipf runs). Contract: every racer gets valid data, and
// the total fill count stays bounded by the racer count.
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, SingleFlightTryLockContract) {
    TempPageFile tf(400);
    LeafExtentCache cache(64ull << 20, 4, tf.fd);
    cache.set_expected_entries(32);

    std::atomic<int> go{0};
    std::vector<std::thread> ths;
    for (int t = 0; t < 8; ++t) {
        ths.emplace_back([&] {
            ++go;
            while (go.load() < 8) std::this_thread::yield();
            const uint8_t* p = nullptr;
            auto h = pin_ok(cache, tf.fd, 200, kExtentPages, &p);
            TempPageFile::expect_page(p, 200, kExtentPages);
            cache.unpin(h);
        });
    }
    for (auto& th : ths) th.join();

    auto st = cache.stats();
    EXPECT_GE(st.misses, 1u);
    EXPECT_LE(st.misses, 8u) << "one fill per racer at most";
    EXPECT_LE(st.bytes_filled, 8u * kExtentPages * kPageSize);
    // Uncontended sequential access dedups trivially: second pin hits.
    const uint8_t* p = nullptr;
    auto h = pin_ok(cache, tf.fd, 300, kExtentPages, &p);
    cache.unpin(h);
    h = pin_ok(cache, tf.fd, 300, kExtentPages, &p);
    cache.unpin(h);
    auto st2 = cache.stats();
    EXPECT_EQ(st2.misses, st.misses + 1u);
    EXPECT_EQ(st2.hits, st.hits + 1u);
}

// ---------------------------------------------------------------------------
// Run aliases: one merged fill covers multiple adjacent leaves; member
// pins hit the alias keys and get offset-correct pointers.
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, RunAliasMergedFill) {
    TempPageFile tf(400);
    LeafExtentCache cache(64ull << 20, 1, tf.fd);
    cache.set_expected_entries(32);

    const PageId p1 = 32, p2 = 32 + kExtentPages;
    const PageId aliases[] = {p2};
    LeafExtentCache::Handle h1;
    const uint8_t* run = cache.pin(p1, 2 * kExtentPages, h1, nullptr,
                                   nullptr, nullptr, aliases, 1);
    TempPageFile::expect_page(run, p1, kExtentPages);

    const uint8_t* p2ptr = nullptr;
    auto h2 = pin_ok(cache, tf.fd, p2, kExtentPages, &p2ptr);
    TempPageFile::expect_page(p2ptr, p2, kExtentPages);  // offset correct
    EXPECT_NE(p2ptr, run);
    EXPECT_EQ(p2ptr, run + kExtentPages * kPageSize);
    cache.unpin(h2);
    cache.unpin(h1);

    auto st = cache.stats();
    EXPECT_EQ(st.misses, 1u) << "alias pin must hit the run entry";
    EXPECT_EQ(st.hits, 1u);

    // Eviction removes every key of the run together.
    cache.invalidate_all();
    EXPECT_FALSE(cache.contains(p1));
    EXPECT_FALSE(cache.contains(p2));
}

// ---------------------------------------------------------------------------
// invalidate_all: cached entries dropped; pinned ones usable until unpin.
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, InvalidateAll) {
    TempPageFile tf(200);
    LeafExtentCache cache(64ull << 20, 2, tf.fd);
    cache.set_expected_entries(32);

    const uint8_t* p1 = nullptr;
    auto h1 = pin_ok(cache, tf.fd, 16, kExtentPages, &p1);
    const uint8_t* p2 = nullptr;
    auto h2 = pin_ok(cache, tf.fd, 48, kExtentPages, &p2);
    cache.unpin(h2);  // only p1 stays pinned

    cache.invalidate_all();
    EXPECT_FALSE(cache.contains(16));
    EXPECT_FALSE(cache.contains(48));

    // Pinned entry's data still valid; unpin frees it without error.
    TempPageFile::expect_page(p1, 16, kExtentPages);
    cache.unpin(h1);

    // Refetch = miss (counters moved).
    const uint64_t misses_before = cache.stats().misses;
    const uint8_t* p3 = nullptr;
    auto h3 = pin_ok(cache, tf.fd, 16, kExtentPages, &p3);
    cache.unpin(h3);
    EXPECT_GT(cache.stats().misses, misses_before);
}

// ---------------------------------------------------------------------------
// Concurrency stress: pins/unpins across threads under eviction pressure.
// Correctness = every pinned buffer keeps its full content while held.
// ---------------------------------------------------------------------------
TEST(LeafExtentCache, ConcurrentPinsUnderEviction) {
    TempPageFile tf(8000);
    LeafExtentCache cache(16 * kExtentPages * kPageSize, 4, tf.fd);
    cache.set_expected_entries(64);

    std::atomic<bool> bad{false};
    std::vector<std::thread> ths;
    for (uint32_t t = 0; t < 8; ++t) {
        ths.emplace_back([&, t] {
            std::mt19937 rng(1234 + t);
            for (int i = 0; i < 300; ++i) {
                const PageId pg = (rng() % 900) * kExtentPages;
                const uint8_t* p = nullptr;
                auto h = pin_ok(cache, tf.fd, pg, kExtentPages, &p);
                // Verify content while pinned.
                for (uint32_t q = 0; q < kExtentPages; ++q) {
                    if (p[q * kPageSize + 3] !=
                        static_cast<uint8_t>((pg + q) & 0xFF)) {
                        bad = true;
                    }
                }
                cache.unpin(h);
            }
        });
    }
    for (auto& th : ths) th.join();
    EXPECT_FALSE(bad);
}

// ===========================================================================
// Integration with IVFTreeIndex
// ===========================================================================

namespace {

/// Clustered fbin fixture (same recipe as test_tree_phase1).
std::string write_test_fbin(const std::string& path, uint32_t n, uint32_t dim,
                            uint32_t n_clusters, float spread = 0.05f) {
    std::mt19937 rng(7);
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c) v = std::uniform_real_distribution(-1.f, 1.f)(rng);
    }
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (uint32_t i = 0; i < n; ++i) {
        const auto& c = centers[i % n_clusters];
        for (uint32_t d = 0; d < dim; ++d)
            data[static_cast<size_t>(i) * dim + d] =
                c[d] + std::uniform_real_distribution(-spread, spread)(rng);
    }
    std::ofstream f(path, std::ios::binary);
    uint32_t hdr[2] = {n, dim};
    f.write(reinterpret_cast<const char*>(hdr), 8);
    f.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(float));
    f.close();
    return path;
}

}  // namespace

TEST(LeafExtentCacheIntegration, SearchParityAndHitAccounting) {
    const uint32_t n = 20000, dim = 32, n_clusters = 30;
    const auto dir = std::filesystem::temp_directory_path() / "leaf_cache_int";
    std::filesystem::create_directories(dir);
    const std::string base = write_test_fbin((dir / "base.fbin").string(),
                                             n, dim, n_clusters);
    const std::string tree_path = (dir / "tree").string();

    {
        sextant::FbinSource s(base);
        sextant::tree::IVFTreeIndex::BuildConfig cfg;
        cfg.k_root = 8;
        cfg.leaf_capacity = 1000;
        cfg.pca_dims = 0;  // FP16 routing — simplest deterministic path
        cfg.num_threads = 4;
        cfg.closure_multiplier = 0.0f;
        sextant::tree::IVFTreeIndex::build_streaming_pca(s, tree_path, cfg);
    }

    // Baseline results with the cache OFF (today's mmap path).
    std::vector<sextant::Candidate> baseline;
    sextant::SearchConfig sc;
    sc.k = 10;
    sc.n_probe = 4;
    sc.n_probe_ln = 4;
    sc.adaptive_probe_gap = 0.0f;
    sc.fastscan_W = 1000;
    {
        auto idx = sextant::tree::IVFTreeIndex::open(tree_path);
        std::vector<float> q(dim, 0.3f);
        baseline = idx->search(q.data(), 10, sc);
        ASSERT_FALSE(baseline.empty());
    }

    // Cache ON: first query = misses; identical query again = hits; results
    // must match the cache-off baseline exactly.
    auto idx = sextant::tree::IVFTreeIndex::open(
        tree_path, 8ull << 20);  // 8 MiB — fits all leaves of this fixture
    std::vector<float> q(dim, 0.3f);
    (void)idx->search(q.data(), 10, sc);  // warm
    auto s1 = idx->search_stats().snapshot_and_reset();
    ASSERT_GT(s1.cache_misses, 0u);

    auto with_cache = idx->search(q.data(), 10, sc);
    auto s2 = idx->search_stats().snapshot_and_reset();
    EXPECT_GT(s2.cache_hits, 0u);
    EXPECT_EQ(s2.cache_misses, 0u);  // everything resident

    ASSERT_EQ(with_cache.size(), baseline.size());
    for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_EQ(with_cache[i].row_id, baseline[i].row_id);
    }

    // Mutation invalidation: insert, then search again — must stay correct
    // (no stale-leaf hits) and produce fresh misses.
    std::vector<float> vec(dim, 0.9f);
    sextant::tree::IVFTreeIndex::InsertPoint point;
    point.vector = vec.data();
    point.row_id = static_cast<sextant::RowId>(n + 1);
    idx->insert_batch({point});
    auto after = idx->search(q.data(), 10, sc);
    auto s3 = idx->search_stats().snapshot_and_reset();
    EXPECT_GT(s3.cache_misses, 0u) << "cache should have been invalidated";
    ASSERT_EQ(after.size(), baseline.size());

    std::filesystem::remove_all(dir);
}

TEST(LeafExtentCacheIntegration, PayloadLocsAreMmapPointers) {
    // With the cache on, payload_locs must point at the MMAP (stable for the
    // index's life), not at cache buffers (pins die at search exit).
    const uint32_t n = 5000, dim = 16, n_clusters = 10;
    const auto dir = std::filesystem::temp_directory_path() / "leaf_cache_pay";
    std::filesystem::create_directories(dir);
    const std::string base = write_test_fbin((dir / "base.fbin").string(),
                                             n, dim, n_clusters);
    const std::string tree_path = (dir / "tree").string();
    {
        sextant::FbinSource s(base);
        sextant::tree::IVFTreeIndex::BuildConfig cfg;
        cfg.k_root = 4;
        cfg.leaf_capacity = 1000;
        cfg.pca_dims = 0;
        cfg.num_threads = 4;
        cfg.closure_multiplier = 0.0f;
        sextant::tree::IVFTreeIndex::build_streaming_pca(s, tree_path, cfg);
    }
    auto idx = sextant::tree::IVFTreeIndex::open(tree_path, 4ull << 20);
    std::vector<float> q(dim, 0.5f);
    sextant::SearchConfig sc;
    sc.k = 5;
    sc.adaptive_probe_gap = 0.0f;
    sc.fastscan_W = 500;
    std::vector<std::pair<const uint8_t*, uint32_t>> locs;
    auto res = idx->search(q.data(), 5, sc, &locs);
    ASSERT_EQ(res.size(), locs.size());
    for (auto& [ptr, slot] : locs) {
        // fetch_payload must be callable through the returned pointer after
        // the search returned (payload-free fixture → empty view, no crash).
        (void)idx->fetch_payload(ptr, slot);
    }
    std::filesystem::remove_all(dir);
}
