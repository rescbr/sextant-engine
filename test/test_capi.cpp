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
#include <functional>
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

    sextant_filter_col_def cols[5] = {{"year", SEXTANT_COL_INT32, 0},
                                      {"category", SEXTANT_COL_STRING, 0},
                                      {"tags", SEXTANT_COL_SET, 0},
                                      {"lat", SEXTANT_COL_FLOAT, 0},
                                      {"lng", SEXTANT_COL_FLOAT, 0}};
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
        index = sextant_open_index(path.c_str(), 0, 0, err, sizeof(err));
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
    void* again = sextant_open_index(path.c_str(), 0, 0, err, sizeof(err));
    ASSERT_NE(again, nullptr) << err;
    char uuid2[40] = {0};
    EXPECT_EQ(sextant_index_uuid(again, uuid2, sizeof(uuid2)), 32);
    EXPECT_STREQ(uuid, uuid2);
    sextant_close_index(again);

    // Unique per build: a second corpus builds a different identity.
    Corpus other{91};
    const std::string path2 =
        build_test_index(other, "capi_v1_uuid_other.tree");
    void* idx2 = sextant_open_index(path2.c_str(), 0, 0, err, sizeof(err));
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

// (b0) Filter-schema introspection: names + types round-trip.
TEST_F(CApiTest, FilterColumnIntrospection) {
    EXPECT_EQ(sextant_index_filter_col_count(index), 5u);
    struct Expect { const char* name; int type; };
    const Expect expected[5] = {
        {"year", SEXTANT_COL_INT32}, {"category", SEXTANT_COL_STRING},
        {"tags", SEXTANT_COL_SET},   {"lat", SEXTANT_COL_FLOAT},
        {"lng", SEXTANT_COL_FLOAT}};
    for (uint32_t i = 0; i < 5; ++i) {
        char name[128] = {0};
        int type = -1;
        EXPECT_EQ(sextant_index_filter_col(index, i, name, sizeof(name), &type, nullptr), 0);
        EXPECT_STREQ(name, expected[i].name);
        EXPECT_EQ(type, expected[i].type);
    }
    char name[128];
    EXPECT_LT(sextant_index_filter_col(index, 5, name, sizeof(name), nullptr, nullptr), 0);
    EXPECT_LT(sextant_index_filter_col(nullptr, 0, name, sizeof(name), nullptr, nullptr), 0);
}

// (b1) Nullable filter columns: SQL three-valued predicate semantics.
TEST(CApiNull, NullableFilterColumns) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "capi_null.tree").string();
    std::filesystem::remove(path);

    constexpr uint32_t kRows = 160;
    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    bopts.quantizer = "local_scalar";
    sextant_filter_col_def cols[2] = {{"year", SEXTANT_COL_INT32, 1},
                                      {"category", SEXTANT_COL_STRING, 1}};
    void* b = sextant_build_begin(&bopts, kDim, cols, 2, 0, err, sizeof(err));
    ASSERT_NE(b, nullptr) << err;

    std::vector<float> vecs(static_cast<size_t>(kRows) * kDim, 0.0f);
    for (uint32_t i = 0; i < kRows; ++i) {
        // 4 clusters like the shared corpus so searches are stable.
        const float base = (i % 4) * 0.5f;
        for (uint32_t j = 0; j < kDim; ++j) {
            vecs[static_cast<size_t>(i) * kDim + j] = base + 0.01f * j;
        }
    }
    std::vector<int32_t> years(kRows);
    std::vector<uint8_t> year_nulls(kRows, 0);
    std::vector<const char*> cat_str(kRows);
    std::vector<uint32_t> cat_len(kRows);
    std::vector<uint8_t> cat_nulls(kRows, 0);
    std::vector<std::string> cats(kRows);
    for (uint32_t i = 0; i < kRows; ++i) {
        years[i] = 1990 + static_cast<int32_t>(i % 40);
        year_nulls[i] = (i % 7 == 3) ? 1 : 0;
        cats[i] = "cat_" + std::to_string(i % 5);
        cat_str[i] = cats[i].data();
        cat_len[i] = static_cast<uint32_t>(cats[i].size());
        cat_nulls[i] = (i % 11 == 5) ? 1 : 0;
    }
    sextant_str_values cat_vals{cat_str.data(), cat_len.data()};
    const void* fvals[2] = {years.data(), &cat_vals};
    const uint8_t* fnulls[2] = {year_nulls.data(), cat_nulls.data()};
    ASSERT_EQ(sextant_build_push_validity(b, vecs.data(), kRows, fvals, fnulls,
                                          nullptr, nullptr, err, sizeof(err)), 0)
        << err;
    ASSERT_EQ(sextant_build_finish(b, path.c_str(), err, sizeof(err)), 0) << err;

    void* idx = sextant_open_index(path.c_str(), 0, 0, err, sizeof(err));
    ASSERT_NE(idx, nullptr) << err;

    // Introspection reports the nullable flags.
    ASSERT_EQ(sextant_index_filter_col_count(idx), 2u);
    char name[128];
    int type = -1, nullable = -1;
    EXPECT_EQ(sextant_index_filter_col(idx, 0, name, sizeof(name), &type, &nullable), 0);
    EXPECT_EQ(nullable, 1);
    EXPECT_EQ(sextant_index_filter_col(idx, 1, name, sizeof(name), &type, &nullable), 0);
    EXPECT_EQ(nullable, 1);

    // SQL-semantics allowed sets over the known row data.
    std::set<uint64_t> eq_allowed, neq_allowed, range_allowed, is_null_allowed,
        not_null_allowed, str_eq_allowed, str_is_null;
    for (uint32_t i = 0; i < kRows; ++i) {
        if (!year_nulls[i] && years[i] == 2010) eq_allowed.insert(i);
        if (!year_nulls[i] && years[i] != 2010) neq_allowed.insert(i);
        if (!year_nulls[i] && years[i] >= 2000 && years[i] <= 2015)
            range_allowed.insert(i);
        if (year_nulls[i]) is_null_allowed.insert(i);
        if (!year_nulls[i]) not_null_allowed.insert(i);
        if (!cat_nulls[i] && cats[i] == "cat_2") str_eq_allowed.insert(i);
        if (cat_nulls[i]) str_is_null.insert(i);
    }
    ASSERT_FALSE(is_null_allowed.empty());
    ASSERT_FALSE(str_is_null.empty());

    sextant_search_opts opts = sextant_default_search_opts();
    opts.k = 8;
    opts.exhaustive = 1;
    opts.rerank = 1;
    opts.int8_scan = 0;

    float query[kDim];
    for (uint32_t j = 0; j < kDim; ++j) query[j] = 0.5f + 0.01f * j;

    const auto check = [&](const sextant_predicate& pred,
                           const std::set<uint64_t>& allowed) {
        uint64_t ids[8];
        float dists[8];
        char qerr[512] = {0};
        const int32_t n = sextant_search_filtered(
            idx, query, &opts, &pred, 1, ids, dists, 8, qerr, sizeof(qerr));
        ASSERT_GE(n, 0) << qerr;
        ASSERT_GT(n, 0);
        for (int32_t i = 0; i < n; ++i) {
            EXPECT_NE(allowed.count(ids[i]), 0u)
                << "row " << ids[i] << " failed SQL predicate semantics";
        }
    };

    {
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "year"; p.op = SEXTANT_PRED_EQ; p.value = 2010.0;
        check(p, eq_allowed);
        p.op = SEXTANT_PRED_NEQ;
        check(p, neq_allowed);
        p.op = SEXTANT_PRED_BETWEEN; p.value = 2000.0; p.value2 = 2015.0;
        check(p, range_allowed);
        p.op = SEXTANT_PRED_IS_NULL;
        check(p, is_null_allowed);
        p.op = SEXTANT_PRED_IS_NOT_NULL;
        check(p, not_null_allowed);
        std::memset(&p, 0, sizeof(p));
        p.column = "category"; p.op = SEXTANT_PRED_EQ; p.str_value = "cat_2";
        check(p, str_eq_allowed);
        p.op = SEXTANT_PRED_IS_NULL;
        check(p, str_is_null);
    }

    // Pushing a NULL for a non-nullable column must be rejected.
    sextant_filter_col_def strict[1] = {{"year", SEXTANT_COL_INT32, 0}};
    void* sb = sextant_build_begin(&bopts, kDim, strict, 1, 0, err, sizeof(err));
    ASSERT_NE(sb, nullptr) << err;
    const uint8_t bad_nulls[kRows] = {1};
    const void* svals[1] = {years.data()};
    const uint8_t* snulls[1] = {bad_nulls};
    EXPECT_LT(sextant_build_push_validity(sb, vecs.data(), 4, svals, snulls,
                                          nullptr, nullptr, err, sizeof(err)), 0);
    sextant_build_abort(sb);

    sextant_close_index(idx);
}

// Tiny match set with cardinality tracking ON (extension default "auto"):
// selectivity 4/2000 < 1% must trigger the brute-force filtered path and
// return ALL matching rows, not a W-shortlist filter that misses most.
TEST(CApiCardinality, TinyMatchSetTracked) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "capi_card.tree").string();
    std::filesystem::remove(path);

    constexpr uint32_t kRows = 2000;
    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    bopts.quantizer = "local_scalar";
    bopts.cardinality = "auto";
    sextant_filter_col_def cols[1] = {{"tag", SEXTANT_COL_STRING, 1}};
    void* b = sextant_build_begin(&bopts, kDim, cols, 1, 0, err, sizeof(err));
    ASSERT_NE(b, nullptr) << err;

    std::vector<float> vecs(static_cast<size_t>(kRows) * kDim, 0.0f);
    for (uint32_t i = 0; i < kRows; ++i) {
        for (uint32_t j = 0; j < kDim; ++j) {
            vecs[static_cast<size_t>(i) * kDim + j] = static_cast<float>(
                static_cast<double>(static_cast<float>(
                    (i * 7919u + j * 104729u) % 1009u)) / 1009.0 * 0.02 +
                static_cast<double>(i % 8) * 0.3);
        }
    }
    std::vector<std::string> tags(kRows);
    std::vector<const char*> tag_str(kRows);
    std::vector<uint32_t> tag_len(kRows);
    for (uint32_t i = 0; i < kRows; ++i) {
        tags[i] = i < 4 ? "rare" : "common";
        tag_str[i] = tags[i].data();
        tag_len[i] = static_cast<uint32_t>(tags[i].size());
    }
    sextant_str_values tag_vals{tag_str.data(), tag_len.data()};
    const void* fvals[1] = {&tag_vals};
    std::vector<uint8_t> tag_nulls(kRows, 0);
    const uint8_t* fnulls[1] = {tag_nulls.data()};
    ASSERT_EQ(sextant_build_push_validity(b, vecs.data(), kRows, fvals,
                                          fnulls, nullptr, nullptr, err,
                                          sizeof(err)), 0) << err;
    ASSERT_EQ(sextant_build_finish(b, path.c_str(), err, sizeof(err)), 0) << err;

    void* idx = sextant_open_index(path.c_str(), 0, 0, err, sizeof(err));
    ASSERT_NE(idx, nullptr) << err;

    // Same serving opts as the extension's index scan (no exhaustive, no
    // int8, rerank on, adaptive-W tau like code_size < 288).
    sextant_search_opts opts = sextant_default_search_opts();
    opts.k = 10;
    opts.rerank = 1;
    opts.int8_scan = 0;
    opts.adaptive_w_gap = 5.0f;
    opts.search_threads = 8;

    float query[kDim];
    for (uint32_t j = 0; j < kDim; ++j) query[j] = 0.1f;

    sextant_predicate pred;
    std::memset(&pred, 0, sizeof(pred));
    pred.column = "tag";
    pred.op = SEXTANT_PRED_EQ;
    pred.str_value = "rare";

    uint64_t ids[10];
    float dists[10];
    char qerr[512] = {0};
    const int32_t n = sextant_search_filtered(
        idx, query, &opts, &pred, 1, ids, dists, 10, qerr, sizeof(qerr));
    ASSERT_GE(n, 0) << qerr;
    // The 4 rare rows (rowid 0..3, cluster 0 — the query's cluster).
    ASSERT_EQ(n, 4) << "tiny tracked match set must be served exhaustively";
    std::set<uint64_t> got(ids, ids + n);
    for (uint64_t r = 0; r < 4; ++r) {
        EXPECT_NE(got.count(r), 0u) << "missing rare row " << r;
    }

    sextant_close_index(idx);
    std::filesystem::remove(path);
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
    opts.int8_scan = -1; // auto: int8 kernel on AVX512 (extension default)

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

// (g) Request scheduler: lifecycle, per-query parity with sextant_search,
// stats reset semantics, and submit-after-stop rejection.
TEST_F(CApiTest, SchedulerLifecycleAndParity) {
    sextant_search_opts base = sextant_default_search_opts();
    base.k = 10;

    sextant_scheduler_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.base = &base;
    cfg.result_cache_entries = 4;

    char err[512] = {0};
    void* sched = sextant_scheduler_create(index, &cfg, err, sizeof(err));
    ASSERT_NE(sched, nullptr) << err;

    for (uint32_t trial = 0; trial < 4; ++trial) {
        const float* query = corpus.centers[trial % kClusters].data();
        uint64_t ref_ids[10];
        float ref_d[10];
        const int32_t nref = sextant_search(
            index, query, &base, ref_ids, ref_d, 10, err, sizeof(err));
        ASSERT_GE(nref, 0) << err;

        uint64_t ids[10];
        float d[10];
        const int32_t n = sextant_scheduler_submit(
            sched, query, 10, nullptr, 0, 0, 0.0f, ids, d, 10,
            err, sizeof(err));
        ASSERT_GE(n, 0) << err;
        EXPECT_EQ(n, nref);
        for (int32_t i = 0; i < n; ++i) {
            EXPECT_EQ(ids[i], ref_ids[i]) << "trial " << trial << " hit " << i;
            EXPECT_EQ(d[i], ref_d[i]);
        }
    }

    // Same query twice: exact-repeat cache hit, identical results.
    const float* q0 = corpus.centers[0].data();
    uint64_t a_ids[10], b_ids[10];
    float a_d[10], b_d[10];
    ASSERT_GE(sextant_scheduler_submit(sched, q0, 10, nullptr, 0, 0, 0.0f,
                                       a_ids, a_d, 10, err, sizeof(err)),
              0)
        << err;
    ASSERT_GE(sextant_scheduler_submit(sched, q0, 10, nullptr, 0, 0, 0.0f,
                                       b_ids, b_d, 10, err, sizeof(err)),
              0)
        << err;
    EXPECT_EQ(0, std::memcmp(a_ids, b_ids, sizeof(a_ids)));
    EXPECT_EQ(0, std::memcmp(a_d, b_d, sizeof(a_d)));

    // Snapshot: queries counted, at least one cache hit, percentiles sane;
    // a second snapshot with no traffic in between is all zeros (reset).
    sextant_scheduler_stats_t st;
    ASSERT_EQ(sextant_scheduler_stats(sched, &st), 0);
    EXPECT_GE(st.queries, 6u);
    EXPECT_GE(st.cache_hits, 1u);
    EXPECT_GE(st.windows, 1u);
    EXPECT_GE(st.delay_p50_ns, 0u);
    EXPECT_GE(st.delay_p99_ns, st.delay_p50_ns);

    sextant_scheduler_stats_t st2;
    ASSERT_EQ(sextant_scheduler_stats(sched, &st2), 0);
    EXPECT_EQ(st2.queries, 0u);
    EXPECT_EQ(st2.windows, 0u);
    EXPECT_EQ(st2.delay_p50_ns, 0u);

    sextant_scheduler_stop(sched);
    // The handle is freed by stop() — post-stop submit rejection is a
    // BatchScheduler contract, covered in test_batch_search.
    // NULL stop is a no-op.
    sextant_scheduler_stop(nullptr);
}

// (h) Quality-contract defaults: adaptive-W tau AUTO + plane pre-prune at
// CLI parity (geocoder validation 2026-09-21: gap-0 default cost ~10pp
// recall@10 on the 23M corpus).
TEST_F(CApiTest, DefaultSearchOptsQualityContract) {
    const sextant_search_opts o = sextant_default_search_opts();
    EXPECT_EQ(o.adaptive_w_gap, -1.0f);  // AUTO, resolved per code size
    EXPECT_EQ(o.plane_pre_prune, 0.25f);
    EXPECT_EQ(o.rerank, 1);

    // AUTO resolves through search without error (and the scheduler
    // inherits the same resolution via its base opts).
    sextant_search_opts opts = o;
    opts.k = 5;
    uint64_t ids[5];
    float d[5];
    char err[512] = {0};
    const float* q = corpus.centers[0].data();
    ASSERT_GE(sextant_search(index, q, &opts, ids, d, 5, err, sizeof(err)),
              0)
        << err;

    sextant_scheduler_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.base = &opts;
    void* sched = sextant_scheduler_create(index, &cfg, err, sizeof(err));
    ASSERT_NE(sched, nullptr) << err;
    uint64_t sids[5];
    float sd[5];
    const int32_t n = sextant_scheduler_submit(
        sched, q, 5, nullptr, 0, 0, 0.0f, sids, sd, 5, err, sizeof(err));
    ASSERT_GE(n, 0) << err;
    EXPECT_EQ(n, 5);
    for (int32_t i = 0; i < n; ++i) {
        EXPECT_EQ(sids[i], ids[i]);
        EXPECT_EQ(sd[i], d[i]);
    }
    sextant_scheduler_stop(sched);
}

// (g2) Scheduler validation: null handles / arguments rejected; NULL cfg
// means all-default config and must succeed.
TEST_F(CApiTest, SchedulerNullArgs) {
    char err[512] = {0};
    EXPECT_EQ(sextant_scheduler_create(nullptr, nullptr, err, sizeof(err)),
              nullptr);

    sextant_scheduler_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    void* sched = sextant_scheduler_create(index, &cfg, err, sizeof(err));
    ASSERT_NE(sched, nullptr) << err;  // NULL fields = defaults

    float q[kDim];
    std::copy(corpus.centers[0].begin(), corpus.centers[0].end(), q);
    uint64_t ids[10];
    float d[10];
    EXPECT_LT(sextant_scheduler_submit(nullptr, q, 10, nullptr, 0, 0, 0.0f,
                                       ids, d, 10, err, sizeof(err)),
              0);
    EXPECT_LT(sextant_scheduler_submit(sched, nullptr, 10, nullptr, 0, 0,
                                       0.0f, ids, d, 10, err, sizeof(err)),
              0);
    EXPECT_LT(sextant_scheduler_submit(sched, q, 10, nullptr, 0, 0, 0.0f,
                                       nullptr, d, 10, err, sizeof(err)),
              0);
    EXPECT_LT(sextant_scheduler_submit(sched, q, 0, nullptr, 0, 0, 0.0f,
                                       ids, d, 10, err, sizeof(err)),
              0);  // k == 0
    EXPECT_LT(sextant_scheduler_submit(sched, q, 10, nullptr, 0, 0, 0.0f,
                                       ids, d, 3, err, sizeof(err)),
              0);  // capacity < k
    sextant_scheduler_stats_t st;
    EXPECT_LT(sextant_scheduler_stats(sched, nullptr), 0);
    EXPECT_LT(sextant_scheduler_stats(nullptr, &st), 0);
    sextant_scheduler_stop(sched);
}

// (g3) Scheduler predicates: parity with sextant_search_filtered.
TEST_F(CApiTest, SchedulerPredicateParity) {
    sextant_search_opts base = sextant_default_search_opts();
    base.k = 10;

    sextant_scheduler_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.base = &base;
    cfg.idle_close_us = 5000;  // give the window a moment to gather

    char err[512] = {0};
    void* sched = sextant_scheduler_create(index, &cfg, err, sizeof(err));
    ASSERT_NE(sched, nullptr) << err;

    sextant_predicate pred;
    std::memset(&pred, 0, sizeof(pred));
    pred.column = "year";
    pred.op = SEXTANT_PRED_GE;
    pred.value = 2007;

    for (uint32_t trial = 0; trial < 3; ++trial) {
        const float* query = corpus.centers[trial % kClusters].data();
        uint64_t ref_ids[10];
        float ref_d[10];
        const int32_t nref = sextant_search_filtered(
            index, query, &base, &pred, 1, ref_ids, ref_d, 10,
            err, sizeof(err));
        ASSERT_GE(nref, 0) << err;

        uint64_t ids[10];
        float d[10];
        const int32_t n = sextant_scheduler_submit(
            sched, query, 10, &pred, 1, 0, 0.0f, ids, d, 10,
            err, sizeof(err));
        ASSERT_GE(n, 0) << err;
        ASSERT_EQ(n, nref);
        for (int32_t i = 0; i < n; ++i) {
            EXPECT_EQ(ids[i], ref_ids[i]) << "trial " << trial;
            EXPECT_EQ(d[i], ref_d[i]);
        }
    }
    sextant_scheduler_stop(sched);
}

// (g4) Concurrent submits with mixed latency classes and per-request probe
// depth: every result matches its single-query reference; urgent (small
// max_delay_us) requests still resolve.
TEST_F(CApiTest, SchedulerConcurrentMixed) {
    sextant_search_opts base = sextant_default_search_opts();
    base.k = 5;

    sextant_scheduler_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.base = &base;
    cfg.max_inflight_windows = 2;

    char err[512] = {0};
    void* sched = sextant_scheduler_create(index, &cfg, err, sizeof(err));
    ASSERT_NE(sched, nullptr) << err;

    // Reference results for each center (default probe depth).
    constexpr uint32_t kQ = 12;
    std::vector<std::vector<uint64_t>> ref(kQ);
    for (uint32_t q = 0; q < kQ; ++q) {
        uint64_t ids[5];
        float d[5];
        const int32_t n = sextant_search(
            index, corpus.centers[q % kClusters].data(), &base,
            ids, d, 5, err, sizeof(err));
        ASSERT_GE(n, 0) << err;
        ref[q].assign(ids, ids + n);
    }

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < 6; ++t) {
        threads.emplace_back([&, t] {
            for (uint32_t i = 0; i < kQ; ++i) {
                const uint32_t q = (t * kQ + i) % kQ;
                const uint64_t delay = (i % 2 == 0) ? 1000 : 50'000;
                uint64_t ids[5];
                float d[5];
                char e[512] = {0};
                const int32_t n = sextant_scheduler_submit(
                    sched, corpus.centers[q % kClusters].data(), 5,
                    nullptr, 0, delay, 0.0f, ids, d, 5, e, sizeof(e));
                if (n < 0 || static_cast<uint32_t>(n) != ref[q].size()) {
                    ++failures;
                    return;
                }
                for (int32_t j = 0; j < n; ++j) {
                    if (ids[j] != ref[q][j]
                        || (j > 0 && d[j - 1] > d[j])) {
                        ++failures;
                        return;
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(failures.load(), 0);

    sextant_scheduler_stats_t st;
    ASSERT_EQ(sextant_scheduler_stats(sched, &st), 0);
    EXPECT_EQ(st.queries, 6u * kQ);
    EXPECT_GE(st.windows, 1u);
    EXPECT_LE(st.delay_p99_ns, 60ull * 1000 * 1000);  // < window_max default

    sextant_scheduler_stop(sched);
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

// (g2) typed-domain predicate comparison. All fixed-width predicate
// comparisons happen in the COLUMN's type, not in double: SQL casts the
// constant to the column type before comparing. Regression guard for the
// old read_fixed_as_double widening, which was wrong twice:
//   - Int64 beyond 2^53: (double)(2^54 + 2) rounds to 2^54, so EQ on
//     2^54 falsely matched rows holding 2^54+2 (and ranges blurred).
//   - Float: (double)0.1f != 0.1, so EQ matched nothing and GT/LE
//     included/excluded rows against SQL float semantics.
TEST(CApiPred, TypedDomainComparisons) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "capi_typed_pred.tree")
            .string();
    std::filesystem::remove(path);

    constexpr uint32_t kRows = 512;
    // big_id = 2^54 + {0,2,4,6}: spacing 2, but double ulp at 2^54 is 4 —
    // 2^54+2 and 2^54+6 are NOT representable in double (they round).
    const auto big_id_of = [](uint32_t i) -> int64_t {
        return (1LL << 54) + static_cast<int64_t>(i % 4) * 2;
    };
    // lat = float(37.0 + k*0.1): mirrors SQL float-column semantics where
    // the constant is cast to FLOAT before comparing.
    const auto lat_of = [](uint32_t i) -> float {
        return static_cast<float>(37.0 + (i % 90) * 0.1);
    };

    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    bopts.quantizer = "local_scalar";
    bopts.k_root = 4;
    bopts.pca_dims = 8;
    sextant_filter_col_def cols[2] = {{"big_id", SEXTANT_COL_INT64, 1},
                                      {"lat", SEXTANT_COL_FLOAT, 1}};
    void* b = sextant_build_begin(&bopts, kDim, cols, 2, 0, err, sizeof(err));
    ASSERT_NE(b, nullptr) << err;

    std::vector<float> vecs(static_cast<size_t>(kRows) * kDim);
    for (uint32_t i = 0; i < kRows; ++i)
        for (uint32_t d = 0; d < kDim; ++d)
            vecs[static_cast<size_t>(i) * kDim + d] =
                0.01f * static_cast<float>((i * 31 + d * 7) % 97);
    std::vector<int64_t> bigs(kRows);
    std::vector<float> lats(kRows);
    for (uint32_t i = 0; i < kRows; ++i) {
        bigs[i] = big_id_of(i);
        lats[i] = lat_of(i);
    }
    const void* fvals[2] = {bigs.data(), lats.data()};
    ASSERT_EQ(sextant_build_push(b, vecs.data(), kRows, fvals, nullptr,
                                 nullptr, err, sizeof(err)), 0)
        << err;
    ASSERT_EQ(sextant_build_finish(b, path.c_str(), err, sizeof(err)), 0)
        << err;

    void* idx = sextant_open_index(path.c_str(), 0, 0, err, sizeof(err));
    ASSERT_NE(idx, nullptr) << err;

    const float* q = vecs.data();  // any query; predicates do the work
    sextant_search_opts sopts = sextant_default_search_opts();
    sopts.k = 8;
    sopts.exhaustive = 1;

    const auto check = [&](sextant_predicate& pred,
                           const std::function<bool(uint32_t)>& sql_ok,
                           const char* what, uint32_t k = 8) {
        uint64_t ids[8];
        float ds[8];
        char e[256] = {0};
        sopts.k = k;
        const int32_t n = sextant_search_filtered(idx, q, &sopts, &pred, 1,
                                                  ids, ds, k, e, sizeof(e));
        ASSERT_GE(n, 0) << what << ": " << e;
        uint32_t pool = 0;
        for (uint32_t i = 0; i < kRows; ++i) pool += sql_ok(i) ? 1 : 0;
        ASSERT_GE(pool, k) << what << ": fixture pool too small";
        EXPECT_EQ(static_cast<uint32_t>(n), k) << what;
        for (int32_t r = 0; r < n; ++r) {
            EXPECT_TRUE(sql_ok(static_cast<uint32_t>(ids[r])))
                << what << ": row " << ids[r]
                << " (big_id=" << big_id_of(static_cast<uint32_t>(ids[r]))
                << ", lat=" << lat_of(static_cast<uint32_t>(ids[r]))
                << ") failed the SQL predicate";
        }
    };

    // Int64 EQ beyond 2^53: old double-widening matched 2^54+2 rows too.
    {
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "big_id";
        p.op = SEXTANT_PRED_EQ;
        p.value = static_cast<double>(1LL << 54);
        check(p, [](uint32_t i) { return (i % 4) == 0; }, "int64 eq 2^54");
    }
    // Int64 GE at a representable boundary (2^54+4): excludes +0/+2 rows.
    {
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "big_id";
        p.op = SEXTANT_PRED_GE;
        p.value = static_cast<double>((1LL << 54) + 4);
        check(p, [](uint32_t i) { return (i % 4) >= 2; }, "int64 ge 2^54+4");
    }
    // Float EQ against a double constant 37.1 (not representable in
    // binary32): SQL compares lat == (float)37.1 — old widening matched
    // nothing.
    {
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "lat";
        p.op = SEXTANT_PRED_EQ;
        p.value = 37.1;
        check(p, [&](uint32_t i) { return lat_of(i) == 37.1f; },
              "float eq 37.1", 4);
    }
    // Float GT: SQL says lat > (float)37.1 — old widening compared
    // against the double and pulled boundary rows in.
    {
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "lat";
        p.op = SEXTANT_PRED_GT;
        p.value = 37.1;
        check(p, [&](uint32_t i) { return lat_of(i) > 37.1f; },
              "float gt 37.1", 4);
    }

    sextant_close_index(idx);
    std::filesystem::remove(path);
}

// (h) build_begin without finish: abort leaves no file behind.
TEST(CApiBuild, AbortLeavesNoFile) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "capi_abort.tree").string();
    std::filesystem::remove(path);

    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    sextant_filter_col_def cols[1] = {{"year", SEXTANT_COL_INT32, 0}};
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
    sextant_filter_col_def setcol[1] = {{"tags", SEXTANT_COL_SET, 0}};
    std::memset(err, 0, sizeof(err));
    void* sb = sextant_build_begin(&bopts, kDim, setcol, 1, 0,
                                   err, sizeof(err));
    ASSERT_NE(sb, nullptr) << err;
    EXPECT_EQ(sextant_build_push(sb, vecs.data(), 64, nullptr, nullptr,
                                 nullptr, err, sizeof(err)), 0)
        << err;  // NULL filter_values → all rows get empty sets
    EXPECT_EQ(sextant_build_finish(sb, setpath.c_str(), err, sizeof(err)), 0)
        << err;

    void* sidx = sextant_open_index(setpath.c_str(), 0, 0, err, sizeof(err));
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

//===--------------------------------------------------------------------===//
// Streaming-push staging tiers. Every fixture above fits the stager's
// default 256 MiB memory tier, which left the spill read path untested
// (the CulturaX-scale OOB in StagedPushSource::next was invisible to the
// suite). The builds below replay the same corpus under a tiny
// staging_bytes budget so chunks spill to the temp file: the builder's
// sample/PCA/Lloyd passes then run vector-only seeks over the spill and
// the emission pass reads the filter/payload blobs back.
//===--------------------------------------------------------------------===//

/// Bool filter column: row i is true iff i % 7 == 0.
bool flag_of(uint32_t i) { return i % 7 == 0; }

/// Push build parameterized on staging budget and corpus size. Schema
/// covers every column type incl. Bool; payloads like build_test_index.
std::string build_push_index(const float* data, const std::string& name,
                             uint64_t staging_bytes, uint32_t n_rows,
                             uint32_t chunk, uint32_t leaf_capacity,
                             uint32_t lloyd_passes) {
    const std::string path =
        (std::filesystem::temp_directory_path() / name).string();
    std::filesystem::remove(path);

    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    bopts.quantizer = "local_scalar";
    bopts.k_root = 8;
    bopts.leaf_capacity = leaf_capacity;
    bopts.pca_dims = 16;
    bopts.max_lloyd_passes = lloyd_passes;
    bopts.num_threads = 4;
    bopts.staging_bytes = staging_bytes;

    sextant_filter_col_def cols[6] = {{"year", SEXTANT_COL_INT32, 0},
                                      {"category", SEXTANT_COL_STRING, 0},
                                      {"tags", SEXTANT_COL_SET, 0},
                                      {"lat", SEXTANT_COL_FLOAT, 0},
                                      {"lng", SEXTANT_COL_FLOAT, 0},
                                      {"flag", SEXTANT_COL_BOOL, 0}};
    void* b = sextant_build_begin(&bopts, kDim, cols, 6, 1, err, sizeof(err));
    EXPECT_NE(b, nullptr) << err;
    if (!b) return path;

    for (uint32_t row = 0; row < n_rows; row += chunk) {
        const uint32_t n = std::min(chunk, n_rows - row);
        std::vector<int32_t> years(n);
        std::vector<uint8_t> flags(n);
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
            flags[r] = flag_of(i) ? 1 : 0;
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
        eptr.resize(elen.size());
        for (size_t e = 0; e < elen.size(); ++e)
            eptr[e] = keep_elems[e].data();
        sextant_str_values sv{strs.data(), lens.data()};
        sextant_set_values setv{scnt.data(), soff.data(), eptr.data(),
                                elen.data()};
        const void* fvals[6] = {years.data(), &sv, &setv, lats.data(),
                                lngs.data(), flags.data()};
        const int rc = sextant_build_push(b, data + static_cast<size_t>(row) * kDim,
                                          n, fvals, offs.data(), pdata.data(),
                                          err, sizeof(err));
        if (rc != 0) {
            ADD_FAILURE() << err;
            sextant_build_abort(b);
            return path;
        }
    }

    const int rc = sextant_build_finish(b, path.c_str(), err, sizeof(err));
    EXPECT_EQ(rc, 0) << err;
    return path;
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

//===--------------------------------------------------------------------===//
// Streaming-push staging tiers: forced spill parity incl. Bool columns
// and multi-chunk mixed-tier builds (regression closure for the
// CulturaX spill OOB — see build_push_index above).
//===--------------------------------------------------------------------===//

namespace {

/// Exact top-k row ids over `allowed` from raw row-major data.
std::set<uint64_t> brute_force_topk_ids(const std::set<uint64_t>& allowed,
                                        const float* query,
                                        const float* data, uint32_t k) {
    std::vector<std::pair<float, uint64_t>> scored;
    scored.reserve(allowed.size());
    for (uint64_t i : allowed) {
        const float* v = data + static_cast<size_t>(i) * kDim;
        float d = 0.0f;
        for (uint32_t dd = 0; dd < kDim; ++dd) {
            const float e = v[dd] - query[dd];
            d += e * e;
        }
        scored.emplace_back(d, i);
    }
    std::sort(scored.begin(), scored.end());
    std::set<uint64_t> out;
    for (uint32_t i = 0; i < k && i < scored.size(); ++i)
        out.insert(scored[i].second);
    return out;
}

/// Exhaustive + exact-rerank filtered search: with the corpus as rerank
/// base the result set is the deterministic top-k over `allowed`, so
/// independently built trees must agree with it exactly.
std::set<uint64_t> exact_topk(void* index, const float* query,
                              const float* rerank_base,
                              const std::vector<sextant_predicate>& preds,
                              const std::set<uint64_t>& allowed,
                              uint32_t k, const char* case_name = "") {
    sextant_search_opts opts = sextant_default_search_opts();
    opts.k = k;
    opts.exhaustive = 1;
    opts.rerank = 1;
    opts.int8_scan = 0;
    opts.exact_rerank_base = rerank_base;
    // Exact rerank only sees the decoded-distance shortlist, and W is
    // applied BEFORE predicates: with W = |allowed| the shortlist holds
    // only ~|allowed|^2/N predicate-matching rows (measured: n=3 of 10
    // on a 2000-row index with a 57-row allowed set). Size W to the
    // full row count so exhaustive + exact rerank is the true top-k.
    opts.fastscan_W = std::max<uint32_t>(
        static_cast<uint32_t>(sextant_index_count(index)), k);
    std::vector<uint64_t> ids(k);
    std::vector<float> dists(k);
    char err[512] = {0};
    const int32_t n = sextant_search_filtered(
        index, query, &opts, preds.data(),
        static_cast<uint32_t>(preds.size()), ids.data(), dists.data(), k,
        err, sizeof(err));
    EXPECT_GE(n, 0) << err;
    EXPECT_GT(n, 0);
    for (int32_t i = 0; i < n; ++i)
        EXPECT_NE(allowed.count(ids[i]), 0u) << "row " << ids[i] << " failed predicate";
    std::set<uint64_t> got(ids.begin(), ids.begin() + n);
    const auto truth = brute_force_topk_ids(allowed, query, rerank_base, k);
    EXPECT_EQ(got, truth) << "exact-rerank top-k mismatch (" << case_name << ")";
    return got;
}

}  // namespace

// (h) Forced spill (staging_bytes = 1 → every chunk hits the spill file
// at finish; the build's vector-only passes seek across it) must behave
// identically to the memory tier: same count and the same deterministic
// filtered results across a battery covering every column type incl.
// Bool, plus identical payload round-trips.
TEST_F(CApiTest, ForcedSpillMatchesMemoryTier) {
    const std::string mem_path = build_push_index(
        corpus.data.data(), "capi_spill_mem.tree", 0, kN, 700, 500, 2);
    const std::string spill_path = build_push_index(
        corpus.data.data(), "capi_spill_forced.tree", 1, kN, 700, 500, 2);

    char err[512] = {0};
    void* mem = sextant_open_index(mem_path.c_str(), 0, 0, err, sizeof(err));
    ASSERT_NE(mem, nullptr) << err;
    void* spill = sextant_open_index(spill_path.c_str(), 0, 0, err, sizeof(err));
    ASSERT_NE(spill, nullptr) << err;

    EXPECT_EQ(sextant_index_count(mem), kN);
    EXPECT_EQ(sextant_index_count(spill), kN);

    struct Case {
        const char* name;
        std::vector<sextant_predicate> preds;
        std::set<uint64_t> allowed;
    };
    std::vector<Case> cases;
    auto add = [&](const char* name, std::vector<sextant_predicate> preds,
                   const std::function<bool(uint32_t)>& keep) {
        Case c{name, std::move(preds), {}};
        for (uint32_t i = 0; i < kN; ++i)
            if (keep(i)) c.allowed.insert(i);
        ASSERT_GE(c.allowed.size(), 10u) << name;
        cases.push_back(std::move(c));
    };
    auto pred = [](const char* col, int op, double value) {
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = col;
        p.op = op;
        p.value = value;
        return p;
    };

    add("flag=1", {pred("flag", SEXTANT_PRED_EQ, 1)},
        [](uint32_t i) { return flag_of(i); });
    add("flag!=1", {pred("flag", SEXTANT_PRED_NEQ, 1)},
        [](uint32_t i) { return !flag_of(i); });
    {
        sextant_predicate p = pred("year", SEXTANT_PRED_BETWEEN, 2005);
        p.value2 = 2014;
        add("flag=1 AND year 2005..2014", {pred("flag", SEXTANT_PRED_EQ, 1), p},
            [](uint32_t i) {
                return flag_of(i) && i % 50 >= 5 && i % 50 <= 14;
            });
    }
    {
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "category";
        p.op = SEXTANT_PRED_EQ;
        p.str_value = "cat_3";
        add("cat_3 AND flag=0",
            {p, pred("flag", SEXTANT_PRED_EQ, 0)},
            [](uint32_t i) { return i != 1234 && i % 10 == 3 && !flag_of(i); });
    }
    {
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "tags";
        p.op = SEXTANT_PRED_CONTAINS;
        p.str_value = "x5";
        add("tags∋x5 AND flag=1", {p, pred("flag", SEXTANT_PRED_EQ, 1)},
            [](uint32_t i) { return i % 5 == 0 && flag_of(i); });
    }
    {
        const double clat = 40.5, clng = -121.4, radius = 55.0;
        sextant_predicate p;
        std::memset(&p, 0, sizeof(p));
        p.column = "lat";
        p.op = SEXTANT_PRED_GEO_RADIUS;
        p.geo_lng_column = "lng";
        p.value = clat;
        p.value2 = clng;
        p.radius_km = radius;
        add("geo+flag=1", {p, pred("flag", SEXTANT_PRED_EQ, 1)}, [&](uint32_t i) {
            return flag_of(i) &&
                   haversine_km_test(clat, clng, lat_of(i), lng_of(i)) <= radius;
        });
    }

    for (const auto& c : cases) {
        for (uint32_t trial = 0; trial < 3; ++trial) {
            const float* query = corpus.centers[trial % kClusters].data();
            const auto got_mem =
                exact_topk(mem, query, corpus.data.data(), c.preds, c.allowed, 10, c.name);
            const auto got_spill =
                exact_topk(spill, query, corpus.data.data(), c.preds, c.allowed, 10, c.name);
            EXPECT_EQ(got_mem, got_spill) << c.name << ": memory vs spill mismatch";
        }
    }

    // Payload round-trip parity on the spilled build.
    {
        sextant_search_opts opts = sextant_default_search_opts();
        opts.k = 5;
        opts.exhaustive = 1;
        uint64_t ids[5];
        float dists[5];
        void* leafs[5];
        uint32_t slots[5];
        const int32_t n = sextant_search2(
            spill, corpus.centers[0].data(), &opts, nullptr, 0, ids, dists,
            leafs, slots, 5, err, sizeof(err));
        ASSERT_GT(n, 0) << err;
        for (int32_t i = 0; i < n; ++i) {
            const std::string expected =
                payload_of(static_cast<uint32_t>(ids[i]));
            std::vector<uint8_t> buf(expected.size() + 1);
            const uint32_t sized = sextant_fetch_payload(
                spill, leafs[i], slots[i], buf.data(),
                static_cast<uint32_t>(buf.size()));
            EXPECT_EQ(sized, expected.size());
            EXPECT_EQ(std::string(buf.begin(), buf.begin() + sized), expected);
        }
    }

    sextant_close_index(mem);
    sextant_close_index(spill);
    std::error_code ec;
    std::filesystem::remove(mem_path, ec);
    std::filesystem::remove(spill_path, ec);
}

// (j) metrics_path: the push build writes JSON-lines phase metrics,
// one record per phase, flushed per record.
TEST_F(CApiTest, MetricsJsonlExport) {
    const std::string mpath =
        (std::filesystem::temp_directory_path() / "capi_metrics.jsonl").string();
    std::filesystem::remove(mpath);
    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    bopts.quantizer = "local_scalar";
    bopts.k_root = 8;
    bopts.leaf_capacity = 500;
    bopts.pca_dims = 16;
    bopts.max_lloyd_passes = 2;
    bopts.num_threads = 4;
    bopts.metrics_path = mpath.c_str();

    void* b = sextant_build_begin(&bopts, kDim, nullptr, 0, 0,
                                  err, sizeof(err));
    ASSERT_NE(b, nullptr) << err;
    ASSERT_EQ(sextant_build_push(b, corpus.data.data(), kN, nullptr,
                                 nullptr, nullptr, err, sizeof(err)),
              0)
        << err;
    const std::string path =
        (std::filesystem::temp_directory_path() / "capi_metrics.tree").string();
    std::filesystem::remove(path);
    ASSERT_EQ(sextant_build_finish(b, path.c_str(), err, sizeof(err)), 0)
        << err;

    FILE* f = std::fopen(mpath.c_str(), "r");
    ASSERT_NE(f, nullptr);
    char line[512];
    int phases = 0;
    bool saw_stream = false;
    while (std::fgets(line, sizeof(line), f)) {
        ++phases;
        if (std::strstr(line, "\"phase\":\"stream\"") ||
            std::strstr(line, "phase=stream"))
            saw_stream = true;
    }
    std::fclose(f);
    EXPECT_GE(phases, 3) << "expected several phase records";
    EXPECT_TRUE(saw_stream) << "no stream phase record";
    std::filesystem::remove(path);
    std::filesystem::remove(mpath);
}

// (h) L2 raw-sweep ||x|^2 bias: a near-origin cluster must not be buried
// by higher-norm clusters for off-center queries. Before the fix the
// sweep ranked by -q.x_hat (an IP surrogate missing the ||x||^2 term of
// L2), so exhaustive search returned the max-norm cluster (d~40) while
// the true neighbors sat at d~0.26. Regression: default AND exhaustive
// paths, NO exact_rerank_base (the decoded rerank must see the rows).
TEST_F(CApiTest, L2SweepNearZeroClusterNotBuried) {
    constexpr uint32_t kC = 8;  // cluster of row i is i % kC; cluster 0
                                 // sits at the origin (near-zero norm)
    std::vector<float> data(static_cast<size_t>(kN) * kDim);
    {
        std::mt19937 rng(90210);
        std::uniform_real_distribution<float> noise(0.0f, 0.02f);
        for (uint32_t i = 0; i < kN; ++i) {
            const uint32_t ci = i % kC;
            for (uint32_t d = 0; d < kDim; ++d) {
                const bool in_span = d >= 4 * ci && d < 4 * ci + 4;
                data[static_cast<size_t>(i) * kDim + d] =
                    (ci > 0 && in_span ? 0.3f : 0.0f) + noise(rng);
            }
        }
    }

    char err[512] = {0};
    sextant_build_opts bopts = sextant_default_build_opts();
    bopts.quantizer = "local_scalar";
    bopts.k_root = 4;  // leaves mix clusters (the burial precondition)
    bopts.leaf_capacity = 500;
    bopts.pca_dims = 16;
    bopts.max_lloyd_passes = 2;
    bopts.num_threads = 4;
    void* b = sextant_build_begin(&bopts, kDim, nullptr, 0, 0,
                                  err, sizeof(err));
    ASSERT_NE(b, nullptr) << err;
    ASSERT_EQ(sextant_build_push(b, data.data(), kN, nullptr, nullptr,
                                 nullptr, err, sizeof(err)),
              0)
        << err;
    const std::string tpath =
        (std::filesystem::temp_directory_path() / "capi_l2bias.tree")
            .string();
    std::filesystem::remove(tpath);
    ASSERT_EQ(sextant_build_finish(b, tpath.c_str(), err, sizeof(err)), 0)
        << err;

    void* idx = sextant_open_index(tpath.c_str(), 0, 0, err, sizeof(err));
    ASSERT_NE(idx, nullptr) << err;

    float q[kDim];
    for (uint32_t d = 0; d < kDim; ++d) q[d] = 0.1f;

    // Brute-force ground truth over the pushed rows.
    std::vector<std::pair<float, uint32_t>> all(kN);
    for (uint32_t i = 0; i < kN; ++i) {
        float s = 0;
        for (uint32_t d = 0; d < kDim; ++d) {
            const float e = q[d] - data[static_cast<size_t>(i) * kDim + d];
            s += e * e;
        }
        all[i] = {s, i};
    }
    std::sort(all.begin(), all.end());
    // Sanity: the true neighbors are the near-origin cluster (row i is
    // cluster i % kC). With 4-bit codes the decoded distances run ~0.2
    // below the true ~0.5 (coarse in-leaf rulers), so exact set equality
    // with brute force is not assertable — cluster identity is.
    for (uint32_t i = 0; i < 10; ++i) ASSERT_EQ(all[i].second % kC, 0u);
    const float brute_min_d = std::sqrt(all[0].first);

    for (int exhaustive = 0; exhaustive <= 1; ++exhaustive) {
        uint64_t ids[10];
        float ds[10];
        sextant_search_opts so = sextant_default_search_opts();
        so.k = 10;
        so.exhaustive = exhaustive;
        so.rerank = 1;
        const int n = sextant_search(idx, q, &so, ids, ds, 10,
                                     err, sizeof(err));
        ASSERT_EQ(n, 10) << err;
        for (int i = 0; i < n; ++i) {
            EXPECT_EQ(ids[i] % kC, 0u)
                << (exhaustive ? "exhaustive" : "default")
                << " path buried the near-zero cluster (id " << ids[i] << ")";
            // Decoded distance stays within the 4-bit quantization band
            // of the brute-force minimum (pre-fix exhaustive returned
            // d~40 from the max-norm cluster).
            EXPECT_LT(ds[i], brute_min_d + 0.4);
            EXPECT_GT(ds[i], brute_min_d - 0.4);
        }
    }
    sextant_close_index(idx);
    std::filesystem::remove(tpath);
}

// (i) More than one sealed staging chunk (kChunkRows = 32768): chunk 1
// seals mid-push and spills under the 1 MiB budget; the 232-row tail
// stays in memory → mixed tiers drained spill-first at emission. The
// build's sample/PCA/Lloyd passes reset and re-seek the spill
// vector-only across passes (the exact CulturaX code path).
TEST_F(CApiTest, MultiChunkSpillVectorPasses) {
    constexpr uint32_t kBigN = 33000;  // 32768 + 232 across 4 pushes
    std::vector<float> data(static_cast<size_t>(kBigN) * kDim);
    {
        std::mt19937 rng(5150);
        std::normal_distribution<float> noise(0.0f, 0.5f);
        for (uint32_t i = 0; i < kBigN; ++i) {
            const uint32_t ci = i % kClusters;
            for (uint32_t d = 0; d < kDim; ++d)
                data[static_cast<size_t>(i) * kDim + d] =
                    corpus.centers[ci][d] + noise(rng);
        }
    }

    const std::string path = build_push_index(
        data.data(), "capi_spill_multi.tree", 1 << 20, kBigN, 8250, 5000, 1);
    char err[512] = {0};
    void* index = sextant_open_index(path.c_str(), 0, 0, err, sizeof(err));
    ASSERT_NE(index, nullptr) << err;
    EXPECT_EQ(sextant_index_count(index), kBigN);

    // Unfiltered exhaustive + exact rerank: the vector stream through
    // the spill seeks must be intact — results are the exact top-k.
    for (uint32_t trial = 0; trial < 2; ++trial) {
        const float* query = corpus.centers[trial % kClusters].data();
        std::set<uint64_t> all;
        for (uint32_t i = 0; i < kBigN; ++i) all.insert(i);
        const auto got = exact_topk(index, query, data.data(), {}, all, 10);
        EXPECT_EQ(got.size(), 10u);
    }

    // Filtered parity incl. Bool over the mixed-tier build.
    {
        sextant_predicate fp;
        std::memset(&fp, 0, sizeof(fp));
        fp.column = "flag";
        fp.op = SEXTANT_PRED_EQ;
        fp.value = 1;
        sextant_predicate yp;
        std::memset(&yp, 0, sizeof(yp));
        yp.column = "year";
        yp.op = SEXTANT_PRED_BETWEEN;
        yp.value = 2000;
        yp.value2 = 2019;
        std::set<uint64_t> allowed;
        for (uint32_t i = 0; i < kBigN; ++i)
            if (flag_of(i) && i % 50 <= 19) allowed.insert(i);
        ASSERT_GE(allowed.size(), 10u);
        for (uint32_t trial = 0; trial < 2; ++trial) {
            const float* query = corpus.centers[trial % kClusters].data();
            exact_topk(index, query, data.data(), {fp, yp}, allowed, 10);
        }
    }

    sextant_close_index(index);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}
