// plane_pred_selftest.cpp — plane+predicate composition check on a real
// filtered tree: (1) brute-force filtered GT (predicate-respecting top-k
// from the base), (2) recall of plane-routed vs legacy-routed filtered
// searches, (3) plane never returns rows the predicate excludes.
#include "../src/tree/filter_data_io.hpp"
#include "../src/tree/ivf_tree_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace sextant;
using tree::IVFTreeIndex;

static std::vector<float> load_fbin(const char* path, uint32_t& n,
                                    uint32_t& d) {
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&d), 4);
    std::vector<float> v(static_cast<size_t>(n) * d);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * 4);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s tree base.fbin query.fbin fdat cat_lo cat_hi [nq]\n",
            argv[0]);
        return 1;
    }
    const uint32_t lo = std::atoi(argv[5]);
    const uint32_t hi = std::atoi(argv[6]);
    const uint32_t nq = argc > 7 ? std::atoi(argv[7]) : 200;

    uint32_t nb, db, nq_, dq;
    auto base = load_fbin(argv[2], nb, db);
    auto query = load_fbin(argv[3], nq_, dq);
    // fdat: synthetic int32 col "category": format from gen_filter_data
    // — parse via the engine? Simplest: regenerate the mask here from
    // the fdat int32 column (row i value stored in column section).
    // .fdat layout (filter_data_io): header + columns; we only need the
    // int32 column named category. Read via engine API instead:
    // FilterColumnData load — use the shared loader.
    auto idx = IVFTreeIndex::open(argv[1]);
    auto fdat = tree::read_filter_data(argv[4]);
    std::vector<int32_t> cat(nb, 0);
    for (auto& col : fdat.cols) {
        std::fprintf(stderr, "col type=%u fixed=%zu name=%s\n",
                     static_cast<unsigned>(col.type),
                     col.fixed_data.size(),
                     fdat.schema.columns.size() ? "?" : "?");
        if (col.type == ColumnType::Int32 && !col.fixed_data.empty())
            std::memcpy(cat.data(), col.fixed_data.data(),
                        std::min<size_t>(col.fixed_data.size(),
                                         static_cast<size_t>(nb) * 4));
    }
    std::fprintf(stderr, "cat[0..4]=%d %d %d %d %d\n", cat[0], cat[1],
                 cat[2], cat[3], cat[4]);

    Predicate p;
    p.column = "category";
    p.op = PredicateOp::Between;
    p.value = static_cast<double>(lo);
    p.value2 = static_cast<double>(hi);

    double rec_plane = 0, rec_legacy = 0;
    uint32_t violations = 0, q = 0;
    for (uint32_t qi = 0; qi < std::min(nq, nq_); ++qi) {
        const float* qv = &query[static_cast<size_t>(qi) * db];
        // Brute-force filtered top-10 (IP; cohere is normalized).
        std::vector<std::pair<float, uint32_t>> cand;
        for (uint32_t i = 0; i < nb; ++i)
            if (cat[i] >= static_cast<int32_t>(lo) &&
                cat[i] <= static_cast<int32_t>(hi)) {
                float ip = 0;
                for (uint32_t d = 0; d < db; ++d)
                    ip += qv[d] * base[static_cast<size_t>(i) * db + d];
                cand.push_back({ip, i});
            }
        if (cand.size() < 10) continue;
        std::partial_sort(cand.begin(), cand.begin() + 10, cand.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<uint32_t> gt10;
        for (int i = 0; i < 10; ++i) gt10.push_back(cand[i].second);

        SearchConfig cfg;
        cfg.k = 10;
        cfg.probe_fraction = 0.1f;
        cfg.predicates = {p};
        cfg.use_plane = true;
        auto rp = idx->search(qv, 10, cfg);
        cfg.use_plane = false;
        auto rl = idx->search(qv, 10, cfg);
        if (qi < 3)
            std::fprintf(stderr, "q%u: cand=%zu plane=%zu legacy=%zu\n",
                         qi, cand.size(), rp.size(), rl.size());
        if (rl.empty()) continue;
        ++q;
        uint32_t hp = 0, hl = 0;
        for (const auto& c : rp) {
            if (cat[c.row_id] < static_cast<int32_t>(lo) ||
                cat[c.row_id] > static_cast<int32_t>(hi))
                ++violations;
            for (uint32_t g : gt10) if (c.row_id == (int64_t)g) ++hp;
        }
        for (const auto& c : rl)
            for (uint32_t g : gt10) if (c.row_id == (int64_t)g) ++hl;
        rec_plane += static_cast<double>(hp) / 10;
        rec_legacy += static_cast<double>(hl) / 10;
    }
    std::printf("q=%u plane=%.4f legacy=%.4f predicate-violations=%u\n",
                q, rec_plane / q, rec_legacy / q, violations);
    return 0;
}
