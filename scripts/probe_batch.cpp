// Probe: batch vs single per-query divergence under per-query probe
// fractions (the PerQueryProbeFractionParity ARM failure). Mirrors the
// test fixture (local_pq, 20k, dim 32) and prints per-query diffs.
#include "tree/ivf_tree_index.hpp"
#include "fbin_source.hpp"

#include <cstdio>
#include <set>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>

using namespace sextant;
using namespace sextant::tree;

// Mirrors test_batch_search.cpp's fixture EXACTLY (clustered, spread 0.05).
static void write_test_fbin(const std::string& path, uint32_t n, uint32_t dim,
                            uint32_t n_clusters, uint32_t seed = 7) {
    std::mt19937 rng(seed);
    const float spread = 0.05f;
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (float& v : c)
            v = std::uniform_real_distribution(-1.f, 1.f)(rng);
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
}

static std::vector<float> make_queries(uint32_t nq, uint32_t dim,
                                       uint32_t seed = 11) {
    std::mt19937 rng(seed);
    std::vector<float> q(static_cast<size_t>(nq) * dim);
    for (uint32_t i = 0; i < nq; ++i) {
        const uint32_t c = std::uniform_int_distribution(0u, 29u)(rng);
        for (uint32_t d = 0; d < dim; ++d)
            q[static_cast<size_t>(i) * dim + d] =
                std::uniform_real_distribution(-1.f, 1.f)(rng) * 0.1f +
                (c % 2 ? 0.6f : -0.6f);
    }
    return q;
}

int main() {
    setenv("SEXTANT_LOG_LEVEL", "error", 0);
    const uint32_t dim = 32, n = 20000, nq = 6;
    const std::string dir = "/tmp/probe_batch";
    std::filesystem::create_directories(dir);
    const std::string base = dir + "/base.fbin";
    const std::string tree = dir + "/tree";
    std::remove(tree.c_str());
    write_test_fbin(base, n, dim, 30);

    IVFTreeIndex::BuildConfig cfg;
    cfg.k_root = 8;
    cfg.leaf_capacity = 1000;
    cfg.pca_dims = 0;
    cfg.num_threads = 4;
    cfg.closure_multiplier = 0.0f;
    // (default global pq — matches the failing test's fixture())
    {
        FbinSource s(base);
        IVFTreeIndex::build_streaming_pca(s, tree, cfg);
    }
    auto idx = IVFTreeIndex::open(tree);

    std::vector<float> q = make_queries(nq, dim);

    SearchConfig bc;
    bc.k = 10;
    bc.n_probe = 0;
    bc.probe_fraction = 0.05f;
    bc.adaptive_probe_gap = 0.0f;
    bc.fastscan_W = 1000;
    bc.search_threads = 1;
    std::vector<float> fracs(nq);
    for (uint32_t i = 0; i < nq; ++i) fracs[i] = (i % 2 == 0) ? 0.4f : 0.0f;

    // Determinism check first: is EITHER path itself stable?
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<std::vector<Candidate>> o1, o2;
        idx->search_batch(q.data(), nq, 10, bc, o1, nullptr, &fracs);
        idx->search_batch(q.data(), nq, 10, bc, o2, nullptr, &fracs);
        bool same = o1[3].size() == o2[3].size();
        for (size_t j = 0; same && j < o1[3].size(); ++j)
            same &= o1[3][j].row_id == o2[3][j].row_id;
        SearchConfig qc3 = bc;
        std::vector<PageId> vis1, vis2;
        auto s1 = idx->search(q.data() + 3 * dim, 10, qc3, nullptr,
                              nullptr, nullptr, &vis1);
        auto s2 = idx->search(q.data() + 3 * dim, 10, qc3, nullptr,
                              nullptr, nullptr, &vis2);
        bool s_same = s1.size() == s2.size();
        for (size_t j = 0; s_same && j < s1.size(); ++j)
            s_same &= s1[j].row_id == s2[j].row_id;
        printf("rep%d: batch-stable=%d single-stable=%d vis1=[", rep, same,
               s_same);
        for (size_t j = 0; j < vis1.size(); ++j)
            printf("%s%llu", j ? "," : "", (unsigned long long)vis1[j]);
        printf("] vis2=[");
        for (size_t j = 0; j < vis2.size(); ++j)
            printf("%s%llu", j ? "," : "", (unsigned long long)vis2[j]);
        printf("]\n");
    }

    // Sequence-effect check: run ALL singles in batch order first
    // (sharing the thread_local scratch), THEN compare q3 again.
    std::vector<std::vector<Candidate>> seq(nq);
    for (uint32_t i = 0; i < nq; ++i) {
        SearchConfig qc = bc;
        if (fracs[i] > 0) qc.probe_fraction = fracs[i];
        seq[i] = idx->search(q.data() + i * dim, 10, qc);
    }

    // Isolate: batch WITHOUT per-query fraction overrides (uniform 0.05).
    std::vector<std::vector<Candidate>> uni;
    idx->search_batch(q.data(), nq, 10, bc, uni, nullptr, nullptr);
    for (uint32_t i = 0; i < nq; ++i) {
        SearchConfig qc = bc;
        auto single = idx->search(q.data() + i * dim, 10, qc);
        bool same = single.size() == uni[i].size();
        for (size_t j = 0; same && j < single.size(); ++j)
            same &= single[j].row_id == uni[i][j].row_id;
        printf("uniform-batch q%u == single: %d\n", i, same);
    }

    // Config bisect: does the q3 divergence survive W changes / rerank?
    for (uint32_t W : {1000u, 5000u}) {
        for (int rr = 0; rr <= 1; ++rr) {
            SearchConfig b2 = bc;
            b2.fastscan_W = W;
            b2.rerank = rr != 0;
            std::vector<std::vector<Candidate>> o2;
            idx->search_batch(q.data(), nq, 10, b2, o2, nullptr, nullptr);
            SearchConfig q2 = b2;
            auto single = idx->search(q.data() + 3 * dim, 10, q2);
            bool same = single.size() == o2[3].size();
            for (size_t j = 0; same && j < single.size(); ++j)
                same &= single[j].row_id == o2[3][j].row_id;
            printf("W=%u rerank=%d: q3 batch==single: %d\n", W, rr, same);
        }
    }

    std::vector<std::vector<Candidate>> out;
    idx->search_batch(q.data(), nq, 10, bc, out, nullptr, &fracs);

    // Map row ids to leaf pages; compare single's visited set.
    {
        auto leaves = idx->debug_leaf_info();
        auto row_leaf = [&](int64_t rid) {
            for (uint32_t l = 0; l < leaves.size(); ++l) {
                auto rids = idx->debug_leaf_row_ids(l);
                for (auto r : rids)
                    if (r == rid) return (unsigned long long)leaves[l].page;
            }
            return 0ULL;
        };
        printf("leaf(single top 11009)=%llu leaf(batch top 12539)=%llu "
               "n_leaves=%zu\n",
               (unsigned long long)row_leaf(11009),
               (unsigned long long)row_leaf(12539), leaves.size());
        std::vector<PageId> vis;
        SearchConfig qc = bc;
        idx->search(q.data() + 3 * dim, 10, qc, nullptr, nullptr, nullptr,
                    &vis);
        printf("single q3 vis: ");
        for (auto p : vis) printf("%llu ", (unsigned long long)p);
        printf("\n");
    }

    // Which single-query fraction reproduces the batch q3 answer?
    for (float f = 0.05f; f <= 0.45f; f += 0.05f) {
        SearchConfig qc = bc;
        qc.probe_fraction = f;
        auto single = idx->search(q.data() + 3 * dim, 10, qc);
        bool same = single.size() == out[3].size();
        for (size_t j = 0; same && j < single.size(); ++j)
            same &= single[j].row_id == out[3][j].row_id;
        printf("q3 single frac=%.2f == batch(0.05): %d (top=%lld)\n", f,
               same, single.empty() ? -1LL : (long long)single[0].row_id);
    }

    for (uint32_t i = 0; i < nq; ++i) {
        SearchConfig qc = bc;
        if (fracs[i] > 0) qc.probe_fraction = fracs[i];
        const auto single = idx->search(q.data() + i * dim, 10, qc);
        printf("q%u (frac=%.2f): batch=%zu single=%zu", i, fracs[i],
               out[i].size(), single.size());
        std::set<int64_t> ra, rb;
        for (const auto& c : single) ra.insert(c.row_id);
        for (const auto& c : out[i]) rb.insert(c.row_id);
        bool same_set = ra == rb, same_order = single.size() == out[i].size();
        if (same_order)
            for (size_t j = 0; j < single.size(); ++j)
                same_order &= single[j].row_id == out[i][j].row_id;
        printf(" %s %s", same_set ? "SET-OK" : "SET-DIFF",
               same_order ? "ORDER-OK" : "ORDER-DIFF");
        if (!same_set) {
            printf(" | s-only:");
            for (const auto& c : single)
                if (!rb.count(c.row_id)) printf(" %lld", (long long)c.row_id);
            printf(" b-only:");
            for (const auto& c : out[i])
                if (!ra.count(c.row_id)) printf(" %lld", (long long)c.row_id);
        }
        printf(" | d:");
        for (size_t j = 0; j < single.size() && j < 4; ++j)
            printf(" %.4f", single[j].dist);
        printf(" /");
        for (size_t j = 0; j < out[i].size() && j < 4; ++j)
            printf(" %.4f", out[i][j].dist);
        printf("\n");
    }
    {
        bool same = seq[3].size() == out[3].size();
        for (size_t j = 0; same && j < seq[3].size(); ++j)
            same &= seq[3][j].row_id == out[3][j].row_id;
        printf("seq-singles q3 == batch q3: %d\n", same);
    }
    return 0;
}
