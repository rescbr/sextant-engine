/// @file test_capi.cpp — DuckDB-ready C API v1 surface tests.
///
/// Covers the extension entry points end-to-end: streaming push build with
/// filter columns + payloads (multi-chunk push), filtered/batch search
/// parity against a brute-force reimplementation over the pushed data,
/// payload/vector fetch round-trip, index_count, error paths, and abort.

#include <gtest/gtest.h>

#include <sextant/sextant_c.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kDim = 32;
constexpr uint32_t kN = 2000;
constexpr uint32_t kClusters = 20;

/// Synthetic corpus: kN clustered vectors (same generator as the tree-filter
/// tests, kept in memory — everything is pushed via the C push API).
struct Corpus {
    std::vector<float> data;      // kN × kDim row-major
    std::vector<std::vector<float>> centers;

    explicit Corpus(uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> noise(0.0f, 0.5f);
        centers.resize(kClusters);
        for (auto& c : centers) {
            c.resize(kDim);
            for (float& v : c)
                v = std::uniform_real_distribution<float>(-10, 10)(rng);
        }
        data.resize(static_cast<size_t>(kN) * kDim);
        for (uint32_t i = 0; i < kN; ++i) {
            const uint32_t ci = i % kClusters;
            for (uint32_t d = 0; d < kDim; ++d)
                data[static_cast<size_t>(i) * kDim + d] =
                    centers[ci][d] + noise(rng);
        }
    }

    const float* row(uint32_t i) const {
        return data.data() + static_cast<size_t>(i) * kDim;
    }
};

/// Filter values: year = 2000 + (i % 50) (Int32), category = "cat_<i%10>"
/// (String), except row 1234 which gets the unique category "rare_zz".
/// Payload: "payload_<i>" (variable length).
int32_t year_of(uint32_t i) { return 2000 + static_cast<int32_t>(i % 50); }
std::string category_of(uint32_t i) {
    return i == 1234 ? std::string("rare_zz")
                     : "cat_" + std::to_string(i % 10);
}
std::string payload_of(uint32_t i) { return "payload_" + std::to_string(i); }

/// Set column: row i always has "tag_<i%3>"; multiples of 5 also have
/// "x5"; row 777 additionally has the unique "rare_tag".
std::vector<std::string> tags_of(uint32_t i) {
    std::vector<std::string> t{"tag_" + std::to_string(i % 3)};
    if (i % 5 == 0) t.push_back("x5");
    if (i == 777) t.push_back("rare_tag");
    return t;
}

/// Geo columns (Float): a deterministic lat/lng grid around the Bay Area.
float lat_of(uint32_t i) { return static_cast<float>(37.0 + (i % 90) * 0.1); }
float lng_of(uint32_t i) {
    return static_cast<float>(-122.0 + (i / 90 % 90) * 0.1);
}

/// Build the shared test index via the push API in 3 chunks.
/// Returns the tree path; the index must be opened by the caller.
std::string build_test_index(const Corpus& c, const std::string& name) {
    const std::string path =
        (std::filesystem::temp_directory_path() / name).string();
    std::filesystem::remove(path);

    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    bopts.quantizer = "local_scalar";
    bopts.k_root = 8;
    bopts.leaf_capacity = 500;
    bopts.pca_dims = 16;
    bopts.max_lloyd_passes = 2;
    bopts.num_threads = 4;

    sextant_filter_col_def cols[5] = {{"year", SEXTANT_COL_INT32},
                                      {"category", SEXTANT_COL_STRING},
                                      {"tags", SEXTANT_COL_SET},
                                      {"lat", SEXTANT_COL_FLOAT},
                                      {"lng", SEXTANT_COL_FLOAT}};
    void* b = sextant_build_begin(&bopts, kDim, cols, 5, 1, err, sizeof(err));
    EXPECT_NE(b, nullptr) << err;
    if (!b) return path;

    // 3 uneven chunks: 700 / 700 / 600.
    const uint32_t chunk_sizes[3] = {700, 700, 600};
    uint32_t row = 0;
    for (int chunk = 0; chunk < 3; ++chunk) {
        const uint32_t n = chunk_sizes[chunk];
        std::vector<int32_t> years(n);
        std::vector<const char*> strs(n);
        std::vector<uint32_t> lens(n);
        std::vector<std::string> keep(n);
        std::vector<float> lats(n), lngs(n);
        std::vector<const char*> eptr;
        std::vector<uint32_t> elen;
        std::vector<uint32_t> scnt(n), soff(n + 1);
        std::vector<std::string> keep_elems;
        std::vector<uint64_t> offs(n + 1);
        std::vector<uint8_t> pdata;
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t i = row + r;
            years[r] = year_of(i);
            keep[r] = category_of(i);
            strs[r] = keep[r].data();
            lens[r] = static_cast<uint32_t>(keep[r].size());
            lats[r] = lat_of(i);
            lngs[r] = lng_of(i);
            scnt[r] = static_cast<uint32_t>(tags_of(i).size());
            soff[r] = static_cast<uint32_t>(elen.size());
            for (const auto& t : tags_of(i)) {
                keep_elems.push_back(t);
                elen.push_back(static_cast<uint32_t>(t.size()));
            }
            offs[r] = pdata.size();
            const std::string blob = payload_of(i);
            pdata.insert(pdata.end(), blob.begin(), blob.end());
        }
        offs[n] = pdata.size();
        soff[n] = static_cast<uint32_t>(elen.size());
        // Fill element pointers only after keep_elems is fully grown —
        // push_back reallocations would dangle them otherwise.
        eptr.resize(elen.size());
        for (size_t e = 0; e < elen.size(); ++e)
            eptr[e] = keep_elems[e].data();
        sextant_str_values sv{strs.data(), lens.data()};
        sextant_set_values setv{scnt.data(), soff.data(), eptr.data(),
                                elen.data()};
        const void* fvals[5] = {years.data(), &sv, &setv, lats.data(),
                                lngs.data()};
        const int rc = sextant_build_push(b, c.row(row), n, fvals,
                                          offs.data(), pdata.data(),
                                          err, sizeof(err));
        if (rc != 0) {
            ADD_FAILURE() << err;
            sextant_build_abort(b);
            return path;
        }
        row += n;
    }
    EXPECT_EQ(row, kN);

    const int rc = sextant_build_finish(b, path.c_str(), err, sizeof(err));
    EXPECT_EQ(rc, 0) << err;
    return path;
}

/// Brute-force filtered top-k over the pushed corpus (exact L2).
std::vector<uint64_t> brute_force_topk(const Corpus& c, const float* query,
                                       uint32_t k,
                                       const std::set<uint64_t>& allowed) {
    std::vector<std::pair<float, uint64_t>> scored;
    scored.reserve(allowed.size());
    for (uint64_t i : allowed) {
        const float* v = c.row(static_cast<uint32_t>(i));
        float d = 0.0f;
        for (uint32_t dd = 0; dd < kDim; ++dd) {
            const float e = v[dd] - query[dd];
            d += e * e;
        }
        scored.emplace_back(d, i);
    }
    std::sort(scored.begin(), scored.end());
    std::vector<uint64_t> out;
    for (uint32_t i = 0; i < k && i < scored.size(); ++i)
        out.push_back(scored[i].second);
    return out;
}

struct CApiTest : public ::testing::Test {
    Corpus corpus{23};
    std::string path = build_test_index(corpus, "capi_v1.tree");
    void* index = nullptr;

    void SetUp() override {
        char err[512] = {0};
        index = sextant_open_index(path.c_str(), err, sizeof(err));
        ASSERT_NE(index, nullptr) << err;
        ASSERT_EQ(sextant_index_dim(index), kDim);
    }
    void TearDown() override {
        if (index) sextant_close_index(index);
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    /// Exhaustive filtered search under `pred`, asserting every returned row
    /// lies in `allowed`, the count matches brute force, and top-k recall vs
    /// the exact ranking over `allowed` is at least k/2 (decoded-distance
    /// ranking is lossy at 4-bit codes).
    void run_filtered_parity(const sextant_predicate& pred,
                             const std::set<uint64_t>& allowed) {
        ASSERT_FALSE(allowed.empty());
        sextant_search_opts opts = sextant_default_search_opts();
        opts.k = 10;
        opts.exhaustive = 1;
        opts.rerank = 1;
        opts.int8_scan = 0;  // float kernel (score-exact scan)
        // Exact rerank against the pushed corpus: removes the quantized
        // ranking error, so the result set must equal brute force exactly.
        opts.exact_rerank_base = corpus.data.data();
        for (uint32_t trial = 0; trial < 3; ++trial) {
            const float* query = corpus.centers[trial % kClusters].data();
            uint64_t ids[10];
            float dists[10];
            char err[512] = {0};
            const int32_t n = sextant_search_filtered(
                index, query, &opts, &pred, 1, ids, dists, 10,
                err, sizeof(err));
            ASSERT_GE(n, 0) << err;
            ASSERT_GT(n, 0);
            for (int32_t i = 0; i < n; ++i) {
                EXPECT_NE(allowed.count(ids[i]), 0u)
                    << "row " << ids[i] << " failed predicate";
            }
            const auto truth =
                brute_force_topk(corpus, query, opts.k, allowed);
            ASSERT_EQ(static_cast<size_t>(n), truth.size());
            std::set<uint64_t> got(ids, ids + n);
            std::set<uint64_t> want(truth.begin(), truth.end());
            EXPECT_EQ(got, want) << "exact-rerank top-k mismatch";
        }
    }
};

}  // namespace

// (a) Build via ≥3 pushes; count exposed; both filter columns searchable.
TEST_F(CApiTest, PushBuildAndCount) {
    EXPECT_EQ(sextant_index_count(index), kN);
}

// (a2) Tree UUID: 32 lowercase hex chars, stable across re-open, unique
// per build (the DuckDB extension's stale-sidecar guard relies on this).
TEST_F(CApiTest, TreeUuidIdentity) {
    char uuid[40] = {0};
    ASSERT_EQ(sextant_index_uuid(index, uuid, sizeof(uuid)), 32);
    ASSERT_EQ(uuid[32], '\0');
    for (int i = 0; i < 32; ++i) {
        EXPECT_TRUE((uuid[i] >= '0' && uuid[i] <= '9') ||
                    (uuid[i] >= 'a' && uuid[i] <= 'f'))
            << "non-hex char at " << i;
    }
    // v4 shape: version nibble 4 at offset 12, variant nibble 8-b at 16.
    EXPECT_EQ(uuid[12], '4');
    ASSERT_TRUE(uuid[16] == '8' || uuid[16] == '9' || uuid[16] == 'a' ||
                uuid[16] == 'b');

    // Stable across re-open of the same file.
    char err[512] = {0};
    void* again = sextant_open_index(path.c_str(), err, sizeof(err));
    ASSERT_NE(again, nullptr) << err;
    char uuid2[40] = {0};
    EXPECT_EQ(sextant_index_uuid(again, uuid2, sizeof(uuid2)), 32);
    EXPECT_STREQ(uuid, uuid2);
    sextant_close_index(again);

    // Unique per build: a second corpus builds a different identity.
    Corpus other{91};
    const std::string path2 =
        build_test_index(other, "capi_v1_uuid_other.tree");
    void* idx2 = sextant_open_index(path2.c_str(), err, sizeof(err));
    ASSERT_NE(idx2, nullptr) << err;
    char uuid3[40] = {0};
    EXPECT_EQ(sextant_index_uuid(idx2, uuid3, sizeof(uuid3)), 32);
    EXPECT_STRNE(uuid, uuid3);
    sextant_close_index(idx2);

    // Argument contract: nulls and too-small buffers are rejected.
    EXPECT_EQ(sextant_index_uuid(nullptr, uuid, sizeof(uuid)), -1);
    EXPECT_EQ(sextant_index_uuid(index, nullptr, sizeof(uuid)), -1);
    EXPECT_EQ(sextant_index_uuid(index, uuid, 32), -1);  // no room for NUL
}

// (a3) DuckDB-shaped concurrency: one shared handle, waves of short-lived
// worker threads (DuckDB creates/joins scan workers per query), mixed
// single/filtered/batch searches plus scan-pool fan-out, all concurrent.
// Correctness contract: no crash/deadlock, and results match the
// single-threaded reference exactly (search is deterministic per config).
TEST_F(CApiTest, ConcurrentSearchStress) {
    // Single-threaded reference results (k=5).
    sextant_search_opts ref_opts = sextant_default_search_opts();
    ref_opts.k = 5;
    std::vector<std::vector<uint64_t>> ref(kClusters);
    for (uint32_t c = 0; c < kClusters; ++c) {
        uint64_t ids[5];
        float dists[5];
        char err[512] = {0};
        ASSERT_EQ(sextant_search(index, corpus.centers[c].data(), &ref_opts,
                                 ids, dists, 5, err, sizeof(err)),
                  5)
            << err;
        ref[c].assign(ids, ids + 5);
    }

    // Scan pool fan-out exercises pool sharing across caller threads.
    ASSERT_GE(sextant_scan_pool_set_threads(8), 1u);
    sextant_search_opts par_opts = sextant_default_search_opts();
    par_opts.k = 5;
    par_opts.search_threads = 4;

    std::atomic<uint32_t> failures{0};
    std::atomic<uint64_t> total_searches{0};
    const uint32_t kWaves = 8;
    const uint32_t kWorkersPerWave = 6;

    for (uint32_t wave = 0; wave < kWaves; ++wave) {
        std::vector<std::thread> workers;
        workers.reserve(kWorkersPerWave);
        for (uint32_t w = 0; w < kWorkersPerWave; ++w) {
            workers.emplace_back([&, w] {
                for (uint32_t iter = 0; iter < 25; ++iter) {
                    const uint32_t c = (w + iter) % kClusters;
                    uint64_t ids[5];
                    float dists[5];
                    char err[512] = {0};
                    const sextant_search_opts* opts =
                        (iter % 2) ? &par_opts : &ref_opts;
                    const int32_t n = sextant_search(
                        index, corpus.centers[c].data(), opts, ids, dists,
                        5, err, sizeof(err));
                    if (n != 5) {
                        ++failures;
                        return;
                    }
                    // Same config as reference (search_threads does not
                    // change results; the sweep is bit-stable).
                    for (uint32_t i = 0; i < 5; ++i) {
                        if (ids[i] != ref[c][i]) {
                            ++failures;
                            return;
                        }
                    }
                    ++total_searches;
                }
            });
        }
        // Wave barrier: threads are joined before the next wave spawns —
        // the churn pattern DuckDB produces per query batch.
        for (auto& t : workers) t.join();
    }

    EXPECT_EQ(failures.load(), 0u);
    EXPECT_EQ(total_searches.load(),
              uint64_t{kWaves} * kWorkersPerWave * 25);

    // Batch API concurrently with singles on the same handle.
    {
        std::vector<float> queries;
        for (uint32_t c = 0; c < kClusters; ++c)
            queries.insert(queries.end(), corpus.centers[c].begin(),
                           corpus.centers[c].end());
        std::thread batch_thread([&] {
            sextant_search_opts bopts = sextant_default_search_opts();
            bopts.k = 5;
            char err[512] = {0};
            uint64_t ids[kClusters * 5];
            float dists[kClusters * 5];
            uint32_t n_per[kClusters];
            const int32_t n = sextant_search_batch(
                index, queries.data(), kClusters, &bopts, nullptr, 0,
                ids, dists, 5, n_per, err, sizeof(err));
            if (n < 0) ++failures;
        });
        std::thread single_thread([&] {
            uint64_t ids[5];
            float dists[5];
            char err[512] = {0};
            sextant_search_opts s = sextant_default_search_opts();
            s.k = 5;
            if (sextant_search(index, queries.data(), &s, ids, dists, 5,
                               err, sizeof(err)) != 5)
                ++failures;
        });
        batch_thread.join();
        single_thread.join();
        EXPECT_EQ(failures.load(), 0u);
    }
    // Restore default pool size so later tests are unaffected.
    sextant_scan_pool_set_threads(0);
}

// (b) Filtered-search parity: string Eq and int32 Eq.
TEST_F(CApiTest, FilteredSearchParity) {
    sextant_search_opts opts = sextant_default_search_opts();
    opts.k = 10;
    // Exhaustive scan: every leaf probed. NOTE predicate queries with
    // untracked cardinality columns take the brute-force filtered path,
    // which ranks by PQ/scalar-DECODED distance — so we assert predicate
    // semantics exactly and brute-force parity as a recall overlap, not
    // bit-exact equality.
    opts.exhaustive = 1;
    opts.rerank = 1;
    opts.int8_scan = 0;  // force the float kernel (score-exact scan)

    for (int variant = 0; variant < 2; ++variant) {
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        std::set<uint64_t> allowed;
        if (variant == 0) {  // category == "cat_3"
            pred.column = "category";
            pred.op = SEXTANT_PRED_EQ;
            pred.str_value = "cat_3";
            for (uint32_t i = 0; i < kN; ++i)
                if (category_of(i) == "cat_3") allowed.insert(i);
        } else {  // year == 2020
            pred.column = "year";
            pred.op = SEXTANT_PRED_EQ;
            pred.value = 2020.0;
            for (uint32_t i = 0; i < kN; ++i)
                if (year_of(i) == 2020) allowed.insert(i);
        }
        ASSERT_FALSE(allowed.empty());

        for (uint32_t trial = 0; trial < 5; ++trial) {
            const float* query = corpus.centers[trial % kClusters].data();
            uint64_t ids[10];
            float dists[10];
            char err[512] = {0};
            const int32_t n = sextant_search_filtered(
                index, query, &opts, &pred, 1, ids, dists, 10,
                err, sizeof(err));
            ASSERT_GE(n, 0) << err;
            ASSERT_GT(n, 0);

            // Distances ascending.
            for (int32_t i = 1; i < n; ++i)
                EXPECT_LE(dists[i - 1], dists[i]);

            std::set<uint64_t> got(ids, ids + n);
            for (int32_t i = 0; i < n; ++i) {
                // Every result satisfies the predicate.
                EXPECT_NE(allowed.count(ids[i]), 0u)
                    << "row " << ids[i] << " failed filter (variant "
                    << variant << ")";
            }
            // Brute-force parity: full result count and a recall overlap
            // with the exact top-k over `allowed` (decoded-distance ranking
            // is lossy at 4-bit codes; measured overlap here is 8-10/10).
            const auto truth =
                brute_force_topk(corpus, query, opts.k, allowed);
            ASSERT_EQ(static_cast<size_t>(n), truth.size());
            uint32_t overlap = 0;
            for (uint64_t id : truth) overlap += got.count(id) ? 1 : 0;
            EXPECT_GE(overlap, opts.k / 2)
                << "variant " << variant << " trial " << trial
                << " overlap " << overlap << "/" << truth.size();

            // Unfiltered search returns the full k as well (no filter
            // shrink) and never fewer rows than the filtered call.
            uint64_t unf_ids[10];
            float unf_d[10];
            const int32_t un = sextant_search_filtered(
                index, query, &opts, nullptr, 0, unf_ids, unf_d, 10,
                err, sizeof(err));
            ASSERT_GT(un, 0);
            EXPECT_GE(un, n);
        }
    }
}

// (c) Rare string Eq: exactly one row matches ("rare_zz" = row 1234).
TEST_F(CApiTest, RareStringEqSingleRow) {
    sextant_search_opts opts = sextant_default_search_opts();
    opts.k = 10;
    // Exhaustive scan: every leaf probed. NOTE predicate queries with
    // untracked cardinality columns take the brute-force filtered path,
    // which ranks by PQ/scalar-DECODED distance — so we assert predicate
    // semantics exactly and brute-force parity as a recall overlap, not
    // bit-exact equality.
    opts.exhaustive = 1;
    opts.rerank = 1;
    opts.int8_scan = 0;  // force the float kernel (score-exact scan)

    sextant_predicate pred;
    std::memset(&pred, 0, sizeof(pred));
    pred.column = "category";
    pred.op = SEXTANT_PRED_EQ;
    pred.str_value = "rare_zz";

    uint64_t ids[10];
    float dists[10];
    char err[512] = {0};
    const int32_t n = sextant_search_filtered(
        index, corpus.centers[0].data(), &opts, &pred, 1, ids, dists, 10,
        err, sizeof(err));
    ASSERT_GE(n, 0) << err;
    ASSERT_EQ(n, 1);
    EXPECT_EQ(ids[0], 1234u);
}

// (d) Batch search equals per-query search (same predicates).
TEST_F(CApiTest, BatchMatchesPerQuery) {
    sextant_search_opts opts = sextant_default_search_opts();
    opts.k = 10;

    sextant_predicate pred;
    std::memset(&pred, 0, sizeof(pred));
    pred.column = "year";
    pred.op = SEXTANT_PRED_BETWEEN;
    pred.value = 2005;
    pred.value2 = 2010;

    constexpr uint32_t kQ = 4;
    std::vector<float> queries(kQ * kDim);
    for (uint32_t q = 0; q < kQ; ++q)
        std::copy(corpus.centers[q].data(), corpus.centers[q].data() + kDim,
                  queries.begin() + q * kDim);

    constexpr uint32_t kCap = 10;
    std::vector<uint64_t> batch_ids(kQ * kCap);
    std::vector<float> batch_d(kQ * kCap);
    std::vector<uint32_t> per_q(kQ);
    char err[512] = {0};
    const int rc = sextant_search_batch(
        index, queries.data(), kQ, &opts, &pred, 1,
        batch_ids.data(), batch_d.data(), kCap, per_q.data(),
        err, sizeof(err));
    ASSERT_EQ(rc, 0) << err;

    for (uint32_t q = 0; q < kQ; ++q) {
        uint64_t ids[kCap];
        float dists[kCap];
        const int32_t n = sextant_search_filtered(
            index, queries.data() + q * kDim, &opts, &pred, 1,
            ids, dists, kCap, err, sizeof(err));
        ASSERT_GE(n, 0) << err;
        EXPECT_EQ(per_q[q], static_cast<uint32_t>(n));

        std::set<uint64_t> solo(ids, ids + n);
        std::set<uint64_t> batch(batch_ids.begin() + q * kCap,
                                 batch_ids.begin() + q * kCap + per_q[q]);
        EXPECT_EQ(solo, batch) << "query " << q;
        // Batch row-major layout: distances too.
        for (uint32_t i = 1; i < per_q[q]; ++i)
            EXPECT_LE(batch_d[q * kCap + i - 1], batch_d[q * kCap + i]);
    }
}

// (e) + (f) Payload fetch round-trip + vector fetch within fp16 tolerance.
TEST_F(CApiTest, PayloadAndVectorFetchRoundTrip) {
    sextant_search_opts opts = sextant_default_search_opts();
    opts.k = 10;
    opts.exhaustive = 1;

    for (uint32_t trial = 0; trial < 5; ++trial) {
        const float* query = corpus.centers[trial % kClusters].data();
        uint64_t ids[10];
        float dists[10];
        void* leafs[10];
        uint32_t slots[10];
        char err[512] = {0};
        const int32_t n = sextant_search2(
            index, query, &opts, nullptr, 0, ids, dists,
            leafs, slots, 10, err, sizeof(err));
        ASSERT_GT(n, 0) << err;

        for (int32_t i = 0; i < n; ++i) {
            // Payload round-trip (two-call protocol: first call sizes).
            uint8_t small[4];
            const uint32_t sized = sextant_fetch_payload(
                index, leafs[i], slots[i], small, sizeof(small));
            const std::string expected =
                payload_of(static_cast<uint32_t>(ids[i]));
            EXPECT_EQ(sized, expected.size());
            std::vector<uint8_t> buf(sized);
            const uint32_t sized2 = sextant_fetch_payload(
                index, leafs[i], slots[i], buf.data(),
                static_cast<uint32_t>(buf.size()));
            EXPECT_EQ(sized2, sized);
            EXPECT_EQ(std::string(buf.begin(), buf.end()), expected);

            // Vector round-trip: stored codes are lossy — 1e-2 relative.
            std::vector<float> out(kDim);
            const int vrc = sextant_fetch_vector(
                index, leafs[i], slots[i], out.data());
            ASSERT_EQ(vrc, 0);
            const float* truth = corpus.row(static_cast<uint32_t>(ids[i]));
            // 4-bit per-leaf scalar codes over a ~[-11,11] element range:
            // quantization step up to ~1.4, worst-case error ~0.7 + fp16
            // rounding (measured max ~1.35 on this corpus). 1.5 absolute
            // covers the step; fp16-storing quantizers would pass at ~1e-2
            // relative.
            for (uint32_t d = 0; d < kDim; ++d) {
                EXPECT_LT(std::fabs(out[d] - truth[d]), 1.5f)
                    << "row " << ids[i] << " dim " << d;
            }
        }
    }
}

// (g) Error paths: nulls, unknown op, IN with empty list, geo without
// longitude column, CONTAINS without an element.
TEST_F(CApiTest, ErrorPaths) {
    char err[512] = {0};
    sextant_search_opts opts = sextant_default_search_opts();
    float q[kDim] = {0};

    // Null handle / query / opts.
    uint64_t ids[10];
    float dists[10];
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search(nullptr, q, &opts, ids, dists, 10,
                             err, sizeof(err)), 0);
    EXPECT_NE(err[0], '\0');
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search(index, nullptr, &opts, ids, dists, 10,
                             err, sizeof(err)), 0);
    EXPECT_NE(err[0], '\0');
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search(index, q, nullptr, ids, dists, 10,
                             err, sizeof(err)), 0);
    EXPECT_NE(err[0], '\0');
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search_batch(nullptr, q, 1, &opts, nullptr, 0,
                                   ids, dists, 10, nullptr,
                                   err, sizeof(err)), 0);
    EXPECT_NE(err[0], '\0');
    // Fetch with null handle.
    EXPECT_EQ(sextant_fetch_payload(nullptr, nullptr, 0, nullptr, 0), 0u);
    EXPECT_LT(sextant_fetch_vector(nullptr, nullptr, 0, q), 0);

    // Unknown op value.
    sextant_predicate pred;
    std::memset(&pred, 0, sizeof(pred));
    pred.column = "year";
    pred.op = 999;
    pred.value = 2020;
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search_filtered(index, q, &opts, &pred, 1,
                                      ids, dists, 10, err, sizeof(err)), 0);
    EXPECT_NE(err[0], '\0');

    // IN with an empty values list → clear error.
    pred.op = SEXTANT_PRED_IN;
    pred.n_values = 0;
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search_filtered(index, q, &opts, &pred, 1,
                                      ids, dists, 10, err, sizeof(err)), 0);
    EXPECT_NE(std::string(err).find("n_values"), std::string::npos) << err;

    // Geo op without geo_lng_column → clear error (the engine would
    // otherwise silently yield no results).
    pred.op = SEXTANT_PRED_GEO_RADIUS;
    pred.value = 40.0;
    pred.value2 = -121.0;
    pred.radius_km = 10.0;
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search_filtered(index, q, &opts, &pred, 1,
                                      ids, dists, 10, err, sizeof(err)), 0);
    EXPECT_NE(err[0], '\0');

    // CONTAINS without an element → clear error.
    pred.op = SEXTANT_PRED_CONTAINS;
    pred.str_value = nullptr;
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search_filtered(index, q, &opts, &pred, 1,
                                      ids, dists, 10, err, sizeof(err)), 0);
    EXPECT_NE(err[0], '\0');

    // Same validation on the batch path.
    pred.op = SEXTANT_PRED_IN;
    pred.n_values = 0;
    uint32_t per_q = 0;
    std::memset(err, 0, sizeof(err));
    EXPECT_LT(sextant_search_batch(index, q, 1, &opts, &pred, 1,
                                    ids, dists, 10, &per_q,
                                    err, sizeof(err)), 0);
    EXPECT_NE(err[0], '\0');
}

// (h) build_begin without finish: abort leaves no file behind.
TEST(CApiBuild, AbortLeavesNoFile) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "capi_abort.tree").string();
    std::filesystem::remove(path);

    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    sextant_filter_col_def cols[1] = {{"year", SEXTANT_COL_INT32}};
    void* b = sextant_build_begin(&bopts, kDim, cols, 1, 0, err, sizeof(err));
    ASSERT_NE(b, nullptr) << err;

    std::vector<float> vecs(static_cast<size_t>(64) * kDim, 0.5f);
    std::vector<int32_t> years(64, 2000);
    const void* fvals[1] = {years.data()};
    ASSERT_EQ(sextant_build_push(b, vecs.data(), 64, fvals, nullptr, nullptr,
                                 err, sizeof(err)), 0)
        << err;

    sextant_build_abort(b);
    EXPECT_FALSE(std::filesystem::exists(path));

    // SET columns are accepted since v1.1 (empty-set NULL fill): build a
    // tiny index with a set column whose filter_values entry is NULL —
    // every row gets the empty set, so CONTAINS matches nothing.
    const std::string setpath =
        (std::filesystem::temp_directory_path() / "capi_setnull.tree")
            .string();
    std::filesystem::remove(setpath);
    sextant_filter_col_def setcol[1] = {{"tags", SEXTANT_COL_SET}};
    std::memset(err, 0, sizeof(err));
    void* sb = sextant_build_begin(&bopts, kDim, setcol, 1, 0,
                                   err, sizeof(err));
    ASSERT_NE(sb, nullptr) << err;
    EXPECT_EQ(sextant_build_push(sb, vecs.data(), 64, nullptr, nullptr,
                                 nullptr, err, sizeof(err)), 0)
        << err;  // NULL filter_values → all rows get empty sets
    EXPECT_EQ(sextant_build_finish(sb, setpath.c_str(), err, sizeof(err)), 0)
        << err;

    void* sidx = sextant_open_index(setpath.c_str(), err, sizeof(err));
    ASSERT_NE(sidx, nullptr) << err;
    {
        sextant_search_opts sopts = sextant_default_search_opts();
        sopts.k = 4;
        sopts.exhaustive = 1;
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "tags";
        p.op = SEXTANT_PRED_CONTAINS;
        p.str_value = "anything";
        uint64_t ids4[4];
        float d4[4];
        char e2[256] = {0};
        EXPECT_EQ(sextant_search_filtered(sidx, vecs.data(), &sopts, &p, 1,
                                          ids4, d4, 4, e2, sizeof(e2)), 0)
            << e2;
    }
    sextant_close_index(sidx);
    std::filesystem::remove(setpath);
}

// ===========================================================================
// v1.1 predicate ops: In/NotIn, Contains family, geo. Every assertion is
// checked against ground truth computed in-test over the pushed data.
// ===========================================================================

// (i) IN / NOT_IN: string membership, numeric-as-string membership, exclusion.
TEST_F(CApiTest, InAndNotIn) {
    // String IN: category in {"cat_3", "cat_7"}.
    {
        const char* vals[2] = {"cat_3", "cat_7"};
        const uint32_t lens[2] = {5, 5};
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "category";
        pred.op = SEXTANT_PRED_IN;
        pred.values = vals;
        pred.value_lengths = lens;
        pred.n_values = 2;
        std::set<uint64_t> allowed;
        for (uint32_t i = 0; i < kN; ++i) {
            const auto c = category_of(i);
            if (c == "cat_3" || c == "cat_7") allowed.insert(i);
        }
        ASSERT_GE(allowed.size(), 10u);
        run_filtered_parity(pred, allowed);
    }
    // Numeric IN via strings: year in {"2010", "2011", "2012"}.
    {
        const char* vals[3] = {"2010", "2011", "2012"};
        const uint32_t lens[3] = {4, 4, 4};
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "year";
        pred.op = SEXTANT_PRED_IN;
        pred.values = vals;
        pred.value_lengths = lens;
        pred.n_values = 3;
        std::set<uint64_t> allowed;
        for (uint32_t i = 0; i < kN; ++i)
            if (year_of(i) == 2010 || year_of(i) == 2011 || year_of(i) == 2012)
                allowed.insert(i);
        ASSERT_GE(allowed.size(), 10u);
        run_filtered_parity(pred, allowed);
    }
    // NOT_IN: exclude "cat_0".
    {
        const char* vals[1] = {"cat_0"};
        const uint32_t lens[1] = {5};
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "category";
        pred.op = SEXTANT_PRED_NOT_IN;
        pred.values = vals;
        pred.value_lengths = lens;
        pred.n_values = 1;
        std::set<uint64_t> allowed;
        for (uint32_t i = 0; i < kN; ++i)
            if (category_of(i) != "cat_0") allowed.insert(i);
        run_filtered_parity(pred, allowed);
    }
}

// (j) CONTAINS family: positive + negative against tags_of ground truth.
TEST_F(CApiTest, SetContainsOps) {
    // CONTAINS "tag_1" -> i % 3 == 1.
    {
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "tags";
        pred.op = SEXTANT_PRED_CONTAINS;
        pred.str_value = "tag_1";
        std::set<uint64_t> allowed;
        for (uint32_t i = 0; i < kN; ++i)
            if (i % 3 == 1) allowed.insert(i);
        run_filtered_parity(pred, allowed);
    }
    // CONTAINS_ANY {"x5", "rare_tag"} -> i % 5 == 0 || i == 777.
    {
        const char* vals[2] = {"x5", "rare_tag"};
        const uint32_t lens[2] = {2, 8};
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "tags";
        pred.op = SEXTANT_PRED_CONTAINS_ANY;
        pred.values = vals;
        pred.value_lengths = lens;
        pred.n_values = 2;
        std::set<uint64_t> allowed;
        for (uint32_t i = 0; i < kN; ++i)
            if (i % 5 == 0 || i == 777) allowed.insert(i);
        ASSERT_GE(allowed.size(), 10u);
        run_filtered_parity(pred, allowed);
    }
    // CONTAINS_ALL {"tag_2", "x5"} -> i % 3 == 2 && i % 5 == 0.
    {
        const char* vals[2] = {"tag_2", "x5"};
        const uint32_t lens[2] = {5, 2};
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "tags";
        pred.op = SEXTANT_PRED_CONTAINS_ALL;
        pred.values = vals;
        pred.value_lengths = lens;
        pred.n_values = 2;
        std::set<uint64_t> allowed;
        for (uint32_t i = 0; i < kN; ++i)
            if (i % 3 == 2 && i % 5 == 0) allowed.insert(i);
        ASSERT_GE(allowed.size(), 10u);
        run_filtered_parity(pred, allowed);
    }
    // Negative: no row contains "missing_tag" -> zero results.
    {
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "tags";
        pred.op = SEXTANT_PRED_CONTAINS;
        pred.str_value = "missing_tag";
        sextant_search_opts opts = sextant_default_search_opts();
        opts.k = 10;
        opts.exhaustive = 1;
        uint64_t ids[10];
        float dists[10];
        char err[512] = {0};
        const int32_t n = sextant_search_filtered(
            index, corpus.centers[0].data(), &opts, &pred, 1,
            ids, dists, 10, err, sizeof(err));
        ASSERT_GE(n, 0) << err;
        EXPECT_EQ(n, 0);
    }
}

// (k) GEO_RADIUS / GEO_BOX over the synthetic lat/lng grid.
namespace {
double haversine_km_test(double lat1, double lng1, double lat2, double lng2) {
    constexpr double kR = 6371.0;
    constexpr double kD2R = 3.14159265358979323846 / 180.0;
    const double dlat = (lat2 - lat1) * kD2R;
    const double dlng = (lng2 - lng1) * kD2R;
    const double a = std::sin(dlat / 2) * std::sin(dlat / 2) +
                     std::cos(lat1 * kD2R) * std::cos(lat2 * kD2R) *
                     std::sin(dlng / 2) * std::sin(dlng / 2);
    return 2 * kR * std::asin(std::sqrt(a));
}
}  // namespace

TEST_F(CApiTest, GeoPredicates) {
    const double clat = 40.5, clng = -121.4, radius = 55.0;
    std::set<uint64_t> allowed;
    double min_gap = 1e9;  // closest row distance to the radius boundary
    for (uint32_t i = 0; i < kN; ++i) {
        const double d = haversine_km_test(clat, clng, lat_of(i), lng_of(i));
        min_gap = std::min(min_gap, std::fabs(d - radius));
        if (d <= radius) allowed.insert(i);
    }
    // No float32 storage rounding may flip a row across the boundary.
    ASSERT_GT(min_gap, 0.2) << "fixture too close to the radius boundary";
    ASSERT_GE(allowed.size(), 10u);

    {
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "lat";           // latitude column
        pred.op = SEXTANT_PRED_GEO_RADIUS;
        pred.geo_lng_column = "lng";   // longitude column
        pred.value = clat;             // center lat
        pred.value2 = clng;            // center lng
        pred.radius_km = radius;
        run_filtered_parity(pred, allowed);
    }
    {
        // GEO_BOX [40, 41] lat x [-121.5, -120.5] lng (inclusive).
        sextant_predicate pred;
        std::memset(&pred, 0, sizeof(pred));
        pred.column = "lat";
        pred.op = SEXTANT_PRED_GEO_BOX;
        pred.geo_lng_column = "lng";
        pred.value = 40.0;     // min_lat
        pred.value2 = -121.5;  // min_lng
        pred.value3 = 41.0;    // max_lat
        pred.value4 = -120.5;  // max_lng
        std::set<uint64_t> boxed;
        for (uint32_t i = 0; i < kN; ++i) {
            const double la = lat_of(i), ln = lng_of(i);
            if (la >= 40.0 && la <= 41.0 && ln >= -121.5 && ln <= -120.5)
                boxed.insert(i);
        }
        ASSERT_GE(boxed.size(), 10u);
        run_filtered_parity(pred, boxed);
    }
}
