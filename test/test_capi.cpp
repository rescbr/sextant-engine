/// @file test_capi.cpp — DuckDB-ready C API v1 surface tests.
///
/// Covers the extension entry points end-to-end: streaming push build with
/// filter columns + payloads (multi-chunk push), filtered/batch search
/// parity against a brute-force reimplementation over the pushed data,
/// payload/vector fetch round-trip, index_count, error paths, and abort.

#include <gtest/gtest.h>

#include <sextant/sextant_c.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <string>
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

    sextant_filter_col_def cols[2] = {{"year", SEXTANT_COL_INT32},
                                      {"category", SEXTANT_COL_STRING}};
    void* b = sextant_build_begin(&bopts, kDim, cols, 2, 1, err, sizeof(err));
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
        std::vector<uint64_t> offs(n + 1);
        std::vector<uint8_t> pdata;
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t i = row + r;
            years[r] = year_of(i);
            keep[r] = category_of(i);
            strs[r] = keep[r].data();
            lens[r] = static_cast<uint32_t>(keep[r].size());
            offs[r] = pdata.size();
            const std::string blob = payload_of(i);
            pdata.insert(pdata.end(), blob.begin(), blob.end());
        }
        offs[n] = pdata.size();
        sextant_str_values sv{strs.data(), lens.data()};
        const void* fvals[2] = {years.data(), &sv};
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
};

}  // namespace

// (a) Build via ≥3 pushes; count exposed; both filter columns searchable.
TEST_F(CApiTest, PushBuildAndCount) {
    EXPECT_EQ(sextant_index_count(index), kN);
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

// (g) Error paths: nulls, unknown op, reserved (NotImplemented) ops.
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

    // Reserved op (IN) → NotImplemented.
    pred.op = SEXTANT_PRED_IN;
    std::memset(err, 0, sizeof(err));
    const int32_t rc = sextant_search_filtered(
        index, q, &opts, &pred, 1, ids, dists, 10, err, sizeof(err));
    EXPECT_EQ(rc, -4);
    EXPECT_NE(err[0], '\0');
    EXPECT_NE(std::string(err).find("not implemented"), std::string::npos);

    // Reserved op via batch path as well.
    pred.op = SEXTANT_PRED_GEO_RADIUS;
    uint32_t per_q = 0;
    std::memset(err, 0, sizeof(err));
    EXPECT_EQ(sextant_search_batch(index, q, 1, &opts, &pred, 1,
                                   ids, dists, 10, &per_q,
                                   err, sizeof(err)), -4);
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

    // SET columns are rejected at build_begin (v1 NotImplemented).
    sextant_filter_col_def setcol[1] = {{"tags", SEXTANT_COL_SET}};
    std::memset(err, 0, sizeof(err));
    EXPECT_EQ(sextant_build_begin(&bopts, kDim, setcol, 1, 0,
                                  err, sizeof(err)), nullptr);
    EXPECT_NE(err[0], '\0');
}
