// Probe: fused scalar_arith_dist1 (the ARM scalar fallback path) vs exact
// decode l2sq, on a scalar_shape tree. Builds nothing — reuses mini10k.
#include "tree/ivf_tree_index.hpp"
#include "tree/coders/coder_util.hpp"
#include "fbin_source.hpp"
#include "util/fp16.hpp"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace sextant;
using namespace sextant::tree;

int main() {
    setenv("SEXTANT_LOG_LEVEL", "error", 0);
    const std::string base = SEXTANT_SOURCE_DIR "/datasets/mini10k_base.fbin";
    const std::string tree = "/tmp/probe_shape.tree";
    std::remove(tree.c_str()); setenv("SEXTANT_LOG_LEVEL", "error", 0);

    IVFTreeIndex::BuildConfig cfg;
    cfg.params.metric = MetricKind::L2Sq;
    cfg.params.quantizer_type = "scalar_shape";
    cfg.k_root = 8;
    cfg.leaf_capacity = 2000;
    cfg.num_threads = 2;
    {
        FbinSource s(base);
        IVFTreeIndex::build_streaming_pca(s, tree, cfg);
    }
    auto idx = IVFTreeIndex::open(tree);
    printf("tree open, dim=%u\n", idx->dim());

    // Load query vectors from the fbin (first rows as queries).
    FILE* f = fopen(base.c_str(), "rb");
    uint32_t n, dim;
    if (fread(&n, 4, 1, f) != 1 || fread(&dim, 4, 1, f) != 1) return 1;
    std::vector<float> data(static_cast<size_t>(n) * dim);
    fseek(f, 8, SEEK_SET);
    if (fread(data.data(), 4, static_cast<size_t>(n) * dim, f) !=
        static_cast<size_t>(n) * dim)
        return 1;
    fclose(f);
    printf("n=%u dim=%u\n", n, dim);

    double worst_rel = 0, mean_abs = 0;
    uint32_t count = 0;
    for (uint32_t q = 0; q < 5; ++q) {
        const float* query = &data[q * dim];
        SearchConfig sc;
        sc.k = 10; sc.n_probe = 8; sc.fastscan_W = 1000;
        sc.rerank = true;  // res[i].dist = fused rerank distance
        std::vector<std::pair<const uint8_t*, uint32_t>> locs;
        auto res = idx->search(query, 10, sc, &locs);
        for (uint32_t i = 0; i < res.size(); ++i) {
            // exact: decode stored vector, l2sq.
            std::vector<float> dec(dim);
            if (!idx->fetch_vector(locs[i].first, locs[i].second, dec.data())) continue;
            float exact = 0;
            for (uint32_t d = 0; d < dim; ++d) {
                const float diff = query[d] - dec[d];
                exact += diff * diff;
            }
            const float fused = res[i].dist;
            const double rel = std::fabs(fused - exact) /
                (exact > 1e-9 ? exact : 1e-9);
            if (rel > worst_rel) worst_rel = rel;
            mean_abs += std::fabs(fused - exact);
            ++count;
            if (i < 3 && q == 0)
                printf("q%u r%u: fused=%.4f exact=%.4f rel=%.5f\n",
                       q, i, fused, exact, rel);
        }
    }
    printf("count=%u worst_rel=%.6f mean_abs=%.4f\n",
           count, worst_rel, count ? mean_abs / count : 0);
    return 0;
}
