// test_quantizer_families.cpp — one build+search+recall+mutation pass per
// quantizer family, on the committed mini10k fixture (datasets/).
//
// Motivation: three bugs found by hand that per-family coverage would have
// caught mechanically — the LeafFilterLayout hardcoded global-PQ offsets
// (wrong rowids for scalar/local filtered scans), LocalPqCoder::rerank's
// fixed 64-byte code buffer (stack overflow for m4 > 128, i.e. the DEFAULT
// m4 = dim/4 on dim > 512), and the signed-narrowing LUT saturation.
//
// Recall floors are calibrated on mini10k (10K x 768, k_root=8, np=8 =
// probe-all, W=1000) with margin; they are regressions tripwires, not
// quality claims. Reference measurements at calibration time:
//   pq             0.699   local_scalar   0.946

#include <gtest/gtest.h>

#include "fbin_source.hpp"
#include "tree/ivf_tree_index.hpp"
#include "sextant/config.hpp"
#include "sextant/engine_trace.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace sextant::tree {
namespace {

// ---------------------------------------------------------------------------
// Fixtures: datasets/mini10k (committed; see datasets/README.md)
// ---------------------------------------------------------------------------

std::string datasets_dir() {
    namespace fs = std::filesystem;
    if (const char* env = std::getenv("SEXTANT_DATASETS"); env && *env)
        return env;
    const fs::path candidates[] = {
        fs::path("datasets"),
        fs::path(SEXTANT_SOURCE_DIR) / "datasets",
    };
    for (const auto& p : candidates)
        if (fs::exists(p / "mini10k_base.fbin")) return p.string();
    return "datasets";
}

struct MiniData {
    std::vector<float> base, query;
    std::vector<std::vector<uint32_t>> gt;  // 1000 x 100 ids
    uint64_t n = 0;
    uint32_t dim = 0, nq = 0;
};

const MiniData& mini() {
    static MiniData d = []() -> MiniData {
        MiniData m;
        auto fail = [](const std::string& what) {
            throw std::runtime_error("mini10k fixture: " + what);
        };
        auto read_fbin = [&fail](const std::string& p, std::vector<float>& out,
                            uint64_t& n, uint32_t& dim) {
            std::ifstream f(p, std::ios::binary);
            if (!f) fail("cannot open " + p);
            uint32_t hdr[2];
            f.read(reinterpret_cast<char*>(hdr), 8);
            n = hdr[0]; dim = hdr[1];
            out.resize(static_cast<size_t>(n) * dim);
            f.read(reinterpret_cast<char*>(out.data()),
                   static_cast<std::streamsize>(out.size() * sizeof(float)));
            if (!f) fail("short read on " + p);
        };
        const std::string dir = datasets_dir();
        uint64_t qn = 0; uint32_t qd = 0;
        read_fbin(dir + "/mini10k_base.fbin", m.base, m.n, m.dim);
        read_fbin(dir + "/mini10k_query.fbin", m.query, qn, qd);
        if (qd != m.dim) fail("dim mismatch");

        // GTMM: [GTMM][nq u32][k u32][metric u8] + per query [ids_k][dists_k]
        std::ifstream g(dir + "/mini10k_gt.gtmm", std::ios::binary);
        if (!g) fail("cannot open gtmm");
        constexpr uint32_t kGtMagic = 0x4D4D5447u;
        uint32_t magic = 0, nq = 0, k = 0;
        g.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != kGtMagic) fail("gt missing GTMM magic");
        g.read(reinterpret_cast<char*>(&nq), 4);
        g.read(reinterpret_cast<char*>(&k), 4);
        char metric = 0; g.read(&metric, 1);
        if (metric != 0) fail("metric must be L2Sq");
        if (nq != qn) fail("nq mismatch");
        m.gt.resize(nq);
        std::vector<uint32_t> row(k);
        for (uint32_t i = 0; i < nq; ++i) {
            g.read(reinterpret_cast<char*>(row.data()),
                   static_cast<std::streamsize>(k * sizeof(uint32_t)));
            m.gt[i].assign(row.begin(), row.end());
            g.seekg(static_cast<std::streamoff>(k) * sizeof(float),
                    std::ios::cur);
        }
        return m;
    }();
    return d;
}

std::string temp_tree() {
    auto tmpl = std::string("/tmp/sextant_qfam_XXXXXX");
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = ::mkstemp(buf.data());
    if (fd == -1) throw std::runtime_error("mkstemp failed");
    ::close(fd);
    ::unlink(buf.data());
    return std::string(buf.data()) + ".tree";
}

// ---------------------------------------------------------------------------
// The per-family pass
// ---------------------------------------------------------------------------

struct FamilySpec {
    const char* quantizer;
    double recall_floor;      // raw scan recall@10 (np=8/W=1000)
    bool supports_delete;     // global pq family only
};

double recall_at_10(const std::vector<Candidate>& res,
                    const std::vector<uint32_t>& gt_row) {
    std::unordered_set<uint32_t> truth(gt_row.begin(), gt_row.begin() + 10);
    uint32_t hits = 0;
    for (uint32_t i = 0; i < res.size() && i < 10; ++i)
        hits += truth.count(static_cast<uint32_t>(res[i].row_id));
    return static_cast<double>(hits) / 10.0;
}

void run_family(const FamilySpec& spec) {
    const bool verbose = getenv("QFAM_VERBOSE") != nullptr;
    const MiniData& m = mini();
    const std::string tree = temp_tree();
    std::filesystem::remove(tree);

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = spec.quantizer;
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 2000;
    cfg.num_threads = 2;
    cfg.adaptive_probe_gap = 0.0f;

    {
        FbinSource s(datasets_dir() + "/mini10k_base.fbin");
        auto br = IVFTreeIndex::build_streaming_pca(s, tree, cfg);
        (void)br;
    }
    auto idx = IVFTreeIndex::open(tree, 0, 1, 0, /*writable=*/true);
    ASSERT_TRUE(idx);
    // Closure replication stores a bounded fraction of vectors in more
    // than one leaf (none on the synthetic clustered fixtures, ~9% here),
    // so live_count() counts stored entries, not unique row_ids.
    const uint64_t base_live = idx->live_count();
    EXPECT_GE(base_live, m.n);
    EXPECT_LE(base_live, m.n + m.n / 8);

    // --- raw scan recall vs GT (np=8 of 8 leaves = probe-all, W=1000) ---
    SearchConfig sc;
    sc.k = 10;
    sc.n_probe = 8;
    sc.fastscan_W = 1000;
    sc.adaptive_probe_gap = 0.0f;
    double raw = 0;
    const uint32_t nq_scored = 100;  // deterministic prefix of the 1000
    for (uint32_t q = 0; q < nq_scored; ++q)
        raw += recall_at_10(idx->search(&m.query[q * m.dim], 10, sc), m.gt[q]);
    raw /= nq_scored;
    if (verbose)
        fprintf(stderr, "[qfam] %-16s raw recall@10 = %.4f (floor %.3f)\n",
                spec.quantizer, raw, spec.recall_floor);
    EXPECT_GE(raw, spec.recall_floor)
        << spec.quantizer << ": raw scan recall regressed";

    // --- rerank must not degrade top-10 membership ---
    sc.rerank = true;
    double rr = 0;
    for (uint32_t q = 0; q < nq_scored; ++q)
        rr += recall_at_10(idx->search(&m.query[q * m.dim], 10, sc), m.gt[q]);
    rr /= nq_scored;
    if (verbose)
        fprintf(stderr, "[qfam] %-16s rerank recall@10 = %.4f\n",
                spec.quantizer, rr);
    EXPECT_GE(rr + 1e-9, raw - 0.02)
        << spec.quantizer << ": rerank degraded results";

    // --- insert: live count + inserted vector searchable ---
    const uint32_t n_ins = 50;
    std::vector<IVFTreeIndex::InsertPoint> points;
    std::vector<float> store(n_ins * m.dim);
    for (uint32_t i = 0; i < n_ins; ++i) {
        // Copies of query 0 with small perturbation: guaranteed near a
        // searchable region and unique row_ids.
        for (uint32_t d = 0; d < m.dim; ++d)
            store[i * m.dim + d] =
                m.query[d] + 1e-3f * static_cast<float>(i + 1) *
                                 (d % 7 == 0 ? 1.f : 0.f);
        points.push_back({&store[i * m.dim],
                          static_cast<RowId>(m.n + i), {}, {}});
    }
    idx->insert_batch(points);
    EXPECT_EQ(idx->live_count(), base_live + n_ins);
    {
        // The inserted vectors are near-copies of query 0; under PRQ-level
        // quantization noise the exact self-match does not reliably win the
        // cluster, so the family-agnostic contract is: rerank ON, an
        // inserted id surfaces in the top-10.
        SearchConfig s2;
        s2.k = 10;
        s2.n_probe = 8;
        s2.fastscan_W = 300;
        s2.rerank = true;
        auto res = idx->search(&store[0], 10, s2);
        std::unordered_set<RowId> ids;
        for (const auto& c : res) ids.insert(c.row_id);
        bool any_inserted = false;
        for (uint32_t i = 0; i < n_ins; ++i)
            any_inserted |= ids.count(static_cast<RowId>(m.n + i)) > 0;
        EXPECT_TRUE(any_inserted)
            << spec.quantizer << ": no inserted vector searchable";
    }

    // --- delete: supported for the global pq family, throws otherwise ---
    if (spec.supports_delete) {
        idx->delete_batch({0, 1, 2});
        EXPECT_EQ(idx->live_count(), base_live + n_ins - 3);
    } else {
        EXPECT_THROW(idx->delete_batch({0}), std::exception)
            << spec.quantizer << ": delete_batch should be unsupported";
    }

    // --- split path: force growth past leaf capacity via more inserts ---
    {
        std::vector<IVFTreeIndex::InsertPoint> more;
        std::vector<float> store2(n_ins * m.dim);
        for (uint32_t i = 0; i < n_ins; ++i) {
            for (uint32_t d = 0; d < m.dim; ++d)
                store2[i * m.dim + d] =
                    m.query[m.dim + d] + 1e-3f * static_cast<float>(i);
            more.push_back({&store2[i * m.dim],
                            static_cast<RowId>(m.n + n_ins + i), {}, {}});
        }
        idx->insert_batch(more);  // may split leaves; must not throw
        EXPECT_EQ(idx->live_count(), base_live + 2 * n_ins -
                                      (spec.supports_delete ? 3 : 0));
    }

    std::filesystem::remove(tree);
}

// ---------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------

TEST(QuantizerFamilies, GlobalPq) {
    run_family({"pq", 0.55, true});
}
TEST(QuantizerFamilies, AnisotropicPq) {
    run_family({"anisotropic_pq", 0.52, true});
}
TEST(QuantizerFamilies, Prq) {
    run_family({"prq", 0.45, true});
}
TEST(QuantizerFamilies, LocalPq) {
    // m4 = dim/4 = 192 (default) — exercises the >128 rerank code path
    // that previously overran its buffer.
    run_family({"local_pq", 0.33, false});
}
TEST(QuantizerFamilies, ScalarLloydmax) {
    run_family({"scalar_lloydmax", 0.85, false});
}
TEST(QuantizerFamilies, ScalarUniform) {
    run_family({"scalar_uniform", 0.85, false});
}
TEST(QuantizerFamilies, ScalarShape) {
    run_family({"scalar_shape", 0.80, false});
}
TEST(QuantizerFamilies, LocalScalar) {
    // The default family; measured 0.946 at calibration.
    run_family({"local_scalar", 0.88, false});
}

// Per-leaf IP-bias path (scalar families under InnerProduct). Uses its own
// tree: builds with metric=ip and checks raw scan sanity — the bias fix
// from the exact-rerank work regressed here once.
TEST(QuantizerFamilies, LocalScalarInnerProductBias) {
    const MiniData& m = mini();
    const std::string tree = temp_tree();
    std::filesystem::remove(tree);
    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::InnerProduct;
    cfg.params.quantizer_type = "local_scalar";
    cfg.params.partition_balance_factor = 4.0f;
    cfg.params.closure_epsilon = -1.0f;
    cfg.k_root = 8;
    cfg.leaf_capacity = 2000;
    cfg.num_threads = 2;
    cfg.adaptive_probe_gap = 0.0f;
    {
        FbinSource s(datasets_dir() + "/mini10k_base.fbin");
        IVFTreeIndex::build_streaming_pca(s, tree, cfg);
    }
    auto idx = IVFTreeIndex::open(tree, 0, 1, 0, /*writable=*/true);
    ASSERT_TRUE(idx);
    EXPECT_GE(idx->live_count(), m.n);
    SearchConfig sc;
    sc.k = 10;
    sc.n_probe = 8;
    sc.fastscan_W = 1000;
    sc.adaptive_probe_gap = 0.0f;
    sc.rerank = true;  // raw scan dist is an order-preserving key, not -IP
    // IP GT = L2 GT when vectors are L2-normalized; they are not, so use a
    // self-consistency check instead: for each query, the scan's top-1 must
    // beat a random vector's IP by a wide margin on average.
    double top1 = 0, rnd = 0;
    for (uint32_t q = 0; q < 50; ++q) {
        const float* qv = &m.query[q * m.dim];
        auto res = idx->search(qv, 10, sc);
        ASSERT_FALSE(res.empty());
        top1 += -res[0].dist;  // engine returns -IP as dist
        const float* rv = &m.base[(q * 977u % m.n) * m.dim];
        double ip = 0;
        for (uint32_t d = 0; d < m.dim; ++d) ip += qv[d] * rv[d];
        rnd += ip;
    }
    top1 /= 50; rnd /= 50;
    EXPECT_GT(top1, rnd + 0.05) << "IP bias path: top-1 not beating random";
    std::filesystem::remove(tree);
}

// PCA-off builds (pca_dims == 0, previously silently coerced to 32):
// must build, open WITHOUT PCA state, and search at sane recall via
// full-dim fp16 centroid routing (spike-measured: within a few pp of
// the PCA path at matched coverage).
TEST(QuantizerFamilies, PcaDisabledBuild) {
    const MiniData& m = mini();
    const std::string tree = temp_tree();
    IVFTreeIndex::BuildConfig cfg;
    cfg.params.quantizer_type = "local_scalar";
    cfg.k_root = 8;
    cfg.leaf_capacity = 2000;
    cfg.adaptive_probe_gap = 0.0f;
    cfg.pca_dims = 0;  // the previously-ignored "disable" request
    {
        FbinSource s(datasets_dir() + "/mini10k_base.fbin");
        IVFTreeIndex::build_streaming_pca(s, tree, cfg);
    }
    auto idx = IVFTreeIndex::open(tree, 0, 1, 0, /*writable=*/true);
    ASSERT_TRUE(idx);
    EXPECT_EQ(idx->pca_dims_default(), 0u);
    SearchConfig sc;
    sc.k = 10;
    sc.n_probe = 8;  // probe-all
    sc.fastscan_W = 1000;
    sc.adaptive_probe_gap = 0.0f;
    sc.rerank = true;
    double r = 0;
    for (uint32_t q = 0; q < 100; ++q)
        r += recall_at_10(idx->search(&m.query[q * m.dim], 10, sc), m.gt[q]);
    r /= 100;
    if (const char* v = std::getenv("QFAM_VERBOSE"); v && *v)
        fprintf(stderr, "[qfam] pca-off recall@10 = %.4f\n", r);
    // Measured 0.812 vs 0.941 for the PCA-32 build on this fixture:
    // with PCA off, k-means runs in full dim where mini10k's noise
    // coordinates dilute the clustering — a fixture-dependent build-
    // quality difference (dbpedia-100K spike: PCA vs full-dim LEAF
    // RANKING differ by only ~3pp), not a routing bug. The contract
    // under test is that 0 is honored and the fp16 path is functional.
    EXPECT_GE(r, 0.78) << "pca-off build regressed";
    std::filesystem::remove(tree);
}

// Probe-fraction routing (leaf-coverage contract). Invariants:
//   - new builds persist the 0.5 manifest default;
//   - f = 1.0 selects every root child → identical results to probe-all;
//   - recall is monotone in f (tiny-fraction ≤ default ≤ f=1.0);
//   - an explicit absolute n_probe overrides the fraction (expert path).
TEST(QuantizerFamilies, ProbeFractionRouting) {
    const MiniData& m = mini();
    const std::string tree = temp_tree();
    IVFTreeIndex::BuildConfig cfg;
    cfg.params.quantizer_type = "local_scalar";
    cfg.k_root = 8;
    cfg.leaf_capacity = 2000;
    cfg.adaptive_probe_gap = 0.0f;
    {
        FbinSource s(datasets_dir() + "/mini10k_base.fbin");
        IVFTreeIndex::build_streaming_pca(s, tree, cfg);
    }
    auto idx = IVFTreeIndex::open(tree, 0, 1, 0, /*writable=*/true);
    ASSERT_TRUE(idx);
    EXPECT_FLOAT_EQ(idx->probe_fraction_default(), 0.5f);

    auto recall = [&](SearchConfig sc) {
        sc.k = 10;
        sc.fastscan_W = 1000;
        sc.adaptive_probe_gap = 0.0f;
        double r = 0;
        for (uint32_t q = 0; q < 100; ++q)
            r += recall_at_10(idx->search(&m.query[q * m.dim], 10, sc),
                              m.gt[q]);
        return r / 100;
    };
    SearchConfig all;   all.n_probe = 8;          // probe-all (legacy)
    SearchConfig f1;    f1.probe_fraction = 1.0f; // fraction probe-all
    SearchConfig def;                             // 0/0 → manifest 0.5
    SearchConfig tiny;  tiny.probe_fraction = 0.05f;
    const double r_all = recall(all);
    const double r_f1 = recall(f1);
    const double r_def = recall(def);
    const double r_tiny = recall(tiny);
    EXPECT_NEAR(r_f1, r_all, 1e-9) << "f=1.0 must equal probe-all exactly";
    EXPECT_LE(r_tiny, r_def + 0.02) << "recall should grow with f";
    EXPECT_LE(r_def, r_f1 + 1e-9);
    EXPECT_GT(r_def, 0.0);
    SearchConfig np1;   np1.n_probe = 1;          // absolute override wins
    EXPECT_LE(recall(np1), r_f1 + 1e-9);
    if (const char* v = std::getenv("QFAM_VERBOSE"); v && *v)
        fprintf(stderr,
                "[qfam] probe-fraction: all=%.4f f1=%.4f def(0.5)=%.4f "
                "tiny(0.05)=%.4f\n",
                r_all, r_f1, r_def, r_tiny);
    std::filesystem::remove(tree);
}

// Scan-feedback probing (SearchConfig::feedback):
//   - Fixed mode at f == a fraction-routing search must match its results
//     (identical child-selection semantics: same cumulative page budget);
//   - Fixed f=1.0 == probe-all;
//   - Stall/Kth modes run, stop no later than probe-all, and their traces
//     (SearchConfig::trace) are written, versioned, and parseable.
TEST(QuantizerFamilies, FeedbackProbing) {
    const MiniData& m = mini();
    const std::string tree = temp_tree();
    IVFTreeIndex::BuildConfig cfg;
    cfg.params.quantizer_type = "local_scalar";
    cfg.k_root = 8;
    cfg.leaf_capacity = 2000;
    cfg.adaptive_probe_gap = 0.0f;
    {
        FbinSource s(datasets_dir() + "/mini10k_base.fbin");
        IVFTreeIndex::build_streaming_pca(s, tree, cfg);
    }
    auto idx = IVFTreeIndex::open(tree, 0, 1, 0, /*writable=*/true);
    ASSERT_TRUE(idx);

    auto result_ids = [&](SearchConfig sc) {
        sc.k = 10;
        sc.fastscan_W = 1000;
        sc.adaptive_probe_gap = 0.0f;
        std::vector<std::vector<RowId>> ids(100);
        for (uint32_t q = 0; q < 100; ++q) {
            auto res = idx->search(&m.query[q * m.dim], 10, sc);
            ids[q].reserve(res.size());
            for (const auto& c : res) ids[q].push_back(c.row_id);
        }
        return ids;
    };
    auto recall = [&](const std::vector<std::vector<RowId>>& ids) {
        double r = 0;
        for (uint32_t q = 0; q < ids.size(); ++q) {
            std::unordered_set<uint32_t> truth(m.gt[q].begin(),
                                               m.gt[q].begin() + 10);
            for (uint32_t i = 0; i < ids[q].size() && i < 10; ++i)
                r += truth.count(static_cast<uint32_t>(ids[q][i]));
        }
        return r / (10.0 * ids.size());
    };

    // Fixed feedback == fraction routing at the same f (result-set equality
    // up to ordering; both scan the same leaf set with the same heap).
    SearchConfig frac;
    frac.probe_fraction = 0.5f;
    const auto ids_frac = result_ids(frac);
    SearchConfig fbfix;
    fbfix.feedback.mode = FeedbackProbe::Mode::Fixed;
    fbfix.feedback.fixed_fraction = 0.5f;
    const auto ids_fbfix = result_ids(fbfix);
    double r_fix = 0;
    for (uint32_t q = 0; q < ids_frac.size(); ++q) {
        std::vector<RowId> a = ids_frac[q], b = ids_fbfix[q];
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        EXPECT_EQ(a, b) << "fixed-feedback top-k must match fraction routing";
        std::unordered_set<uint32_t> truth(m.gt[q].begin(),
                                           m.gt[q].begin() + 10);
        for (uint32_t i = 0; i < b.size() && i < 10; ++i)
            r_fix += truth.count(static_cast<uint32_t>(b[i]));
    }
    r_fix /= 10.0 * ids_frac.size();
    EXPECT_GT(r_fix, 0.0);

    // Fixed f=1.0 == probe-all.
    SearchConfig all;
    all.n_probe = 8;
    const auto ids_all = result_ids(all);
    SearchConfig fb1;
    fb1.feedback.mode = FeedbackProbe::Mode::Fixed;
    fb1.feedback.fixed_fraction = 1.0f;
    const auto ids_fb1 = result_ids(fb1);
    for (uint32_t q = 0; q < ids_all.size(); ++q) {
        std::vector<RowId> a = ids_all[q], b = ids_fb1[q];
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        EXPECT_EQ(a, b) << "fixed-feedback f=1.0 must equal probe-all";
    }

    // Stall/Kth: run with a trace sink; results must be a subset-scan of
    // probe-all (recall ≤ probe-all + noise), trace file well-formed.
    const std::string trace_path =
        tree + ".fbtrace";
    {
        auto trace = EngineTrace::create(trace_path, "qfam feedback test");
        ASSERT_TRUE(trace);
        SearchConfig st;
        st.feedback.mode = FeedbackProbe::Mode::Stall;
        st.feedback.m = 1;
        st.feedback.min_blocks = 1;
        st.trace = trace.get();
        const auto ids_stall = result_ids(st);
        EXPECT_LE(recall(ids_stall), recall(ids_all) + 0.02)
            << "stall-mode recall must not exceed probe-all";

        SearchConfig kt;
        kt.feedback.mode = FeedbackProbe::Mode::Kth;
        kt.feedback.m = 2;
        kt.feedback.min_blocks = 1;
        kt.trace = trace.get();
        const auto ids_kth = result_ids(kt);
        EXPECT_LE(recall(ids_kth), recall(ids_all) + 0.02)
            << "kth-mode recall must not exceed probe-all";
    }
    {
        std::ifstream tf(trace_path);
        ASSERT_TRUE(tf.good());
        std::string first, line;
        std::getline(tf, first);
        EXPECT_EQ(first.substr(0, 18), "# sextant-trace v1");
        size_t fb_lines = 0;
        while (std::getline(tf, line))
            if (line.rfind("FB ", 0) == 0) ++fb_lines;
        EXPECT_GT(fb_lines, 0u) << "trace must contain FB records";
    }
    std::filesystem::remove(tree);
    std::filesystem::remove(trace_path);
}

// n_probe_ln auto-sizing: the manifest default must cover all leaves of
// a probed root child. A default below the real leaves-per-child (the old
// hardcoded 4) silently truncates probing and masquerades as a routing
// regression — measured 17pp containment loss on a 9888-leaf cohere-10M
// tree. Build with few root children and many leaves per child; probing
// ALL children via the count path must then match fraction probe-all.
TEST(QuantizerFamilies, ProbeLnAutoSizing) {
    const MiniData& m = mini();
    const std::string tree = temp_tree();
    IVFTreeIndex::BuildConfig cfg;
    cfg.params.quantizer_type = "local_scalar";
    cfg.k_root = 4;
    cfg.leaf_capacity = 400;  // ~25 leaves → ~6+ per root child (> 4)
    cfg.adaptive_probe_gap = 0.0f;
    {
        FbinSource s(datasets_dir() + "/mini10k_base.fbin");
        IVFTreeIndex::build_streaming_pca(s, tree, cfg);
    }
    auto idx = IVFTreeIndex::open(tree, 0, 1, 0, /*writable=*/true);
    ASSERT_TRUE(idx);
    ASSERT_GT(idx->n_leaves(), 4u * 4u)
        << "fixture should force more than 4 leaves per root child";
    EXPECT_GE(idx->n_probe_ln_default(),
              (idx->n_leaves() + idx->k_root() - 1) /
                  idx->k_root())
        << "manifest ln default must cover the average leaves-per-child";

    auto recall = [&](SearchConfig sc) {
        sc.k = 10;
        sc.fastscan_W = 1000;
        sc.adaptive_probe_gap = 0.0f;
        double r = 0;
        for (uint32_t q = 0; q < 100; ++q)
            r += recall_at_10(idx->search(&m.query[q * m.dim], 10, sc),
                              m.gt[q]);
        return r / 100;
    };
    SearchConfig np_all;  np_all.n_probe = 4;  // count path, manifest ln
    SearchConfig f1;      f1.probe_fraction = 1.0f;
    const double r_np = recall(np_all);
    const double r_f1 = recall(f1);
    EXPECT_NEAR(r_np, r_f1, 0.01)
        << "np=k_root with auto ln must match fraction probe-all "
           "(truncation would cost recall)";
    if (const char* v = std::getenv("QFAM_VERBOSE"); v && *v)
        fprintf(stderr, "[qfam] ln-auto: np=%d leaves=%u ln=%u r=%.4f "
                        "f1=%.4f\n",
                4, idx->n_leaves(), idx->n_probe_ln_default(), r_np, r_f1);
    std::filesystem::remove(tree);
}

}  // namespace
}  // namespace sextant::tree
