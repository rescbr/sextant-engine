// Spike: GT-dispersion vs routing-score flatness (F3 adaptive-plane study).
//
// Question: can a CHEAP, rank-time signal computed BEFORE any probing —
// the shape of the query's score distribution over root centroids and
// leaf centroids — detect scattered-GT queries (the IVF blind spot,
// SP geocoder no_city tier)?
//
// Per query, projects into PCA routing space exactly like search Phase A
// (dot(proj_k, q) - mean_proj_k), scores all root / leaf PCA centroids,
// and emits scale-free peakedness metrics:
//   gap12   — (s1 - s2) / std(s)      top margin, z-normalized
//   nent    — normalized entropy of softmax(z): 1 = flat, 0 = one-hot
//   mass8   — softmax mass of the top-8 centroids (root: k_root may be
//             small; leaf view is the finer signal on shallow trees)
//
// Output: CSV  qid,root_gap12,root_nent,root_mass8,leaf_gap12,leaf_nent,
//              leaf_mass8  — join with per-query GT dispersion / recall
//              in Python (see gen_dispersion_corpus.py meta.npz).
//
// Usage: spike_dispersion <index.tree> <query.fbin>

#include "../src/tree/ivf_tree_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sextant;
using tree::IVFTreeIndex;

namespace {

struct Fbin {
    uint32_t n = 0, d = 0;
    std::vector<float> v;
};

Fbin read_fbin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open " + path);
    Fbin b;
    f.read(reinterpret_cast<char*>(&b.n), 4);
    f.read(reinterpret_cast<char*>(&b.d), 4);
    b.v.resize(size_t(b.n) * b.d);
    f.read(reinterpret_cast<char*>(b.v.data()),
           std::streamsize(b.v.size() * 4));
    if (!f) throw std::runtime_error("short read " + path);
    return b;
}

struct Flat {
    double gap12 = 0, nent = 0, mass8 = 0;
};

// scores: higher = closer. Softmax on z-scores (scale-free).
Flat flatness(std::vector<float> s) {
    Flat out;
    const size_t n = s.size();
    if (n < 2) return out;
    double mu = 0;
    for (float x : s) mu += x;
    mu /= double(n);
    double var = 0;
    for (float x : s) var += (x - mu) * (x - mu);
    const double sd = std::sqrt(var / double(n));
    for (float& x : s) x = static_cast<float>((x - mu) / (sd > 1e-9 ? sd : 1.0));
    std::vector<float> z = s;
    std::sort(z.begin(), z.end(), std::greater<float>());
    out.gap12 = z[0] - z[1];
    double mx = z[0];
    std::vector<double> p(n);
    double tot = 0;
    for (size_t i = 0; i < n; ++i) {
        p[i] = std::exp(double(s[i]) - mx);
        tot += p[i];
    }
    double h = 0, m8 = 0;
    for (size_t i = 0; i < n; ++i) {
        p[i] /= tot;
        if (p[i] > 0) h -= p[i] * std::log(p[i]);
    }
    std::vector<double> ps(p.begin(), p.end());
    std::sort(ps.begin(), ps.end(), std::greater<double>());
    for (size_t i = 0; i < std::min<size_t>(8, n); ++i) m8 += ps[i];
    out.nent = h / std::log(double(n));
    out.mass8 = m8;
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <index.tree> <query.fbin>\n", argv[0]);
        return 2;
    }
    auto idx = IVFTreeIndex::open(argv[1], 0, 1, 0, false);
    if (!idx) { std::fprintf(stderr, "open failed\n"); return 1; }
    const Fbin q = read_fbin(argv[2]);
    const uint32_t pd = idx->pca_dims_view();
    if (pd == 0) { std::fprintf(stderr, "no PCA routing blob\n"); return 1; }
    const uint32_t dim = q.d;
    const float* proj = idx->pca_proj_view();
    const float* mproj = idx->pca_proj_mean_view();
    const uint32_t nrc = idx->pca_root_centroid_count_view();
    const uint32_t nlc = idx->pca_leaf_centroid_count_view();
    const float* rc = idx->pca_root_centroids_view();
    const float* lc = idx->pca_leaf_centroids_view();

    std::printf("qid,root_gap12,root_nent,root_mass8,leaf_gap12,leaf_nent,leaf_mass8\n");
    std::vector<float> qp(pd), rs(nrc), ls(nlc);
    for (uint32_t i = 0; i < q.n; ++i) {
        const float* v = &q.v[size_t(i) * dim];
        for (uint32_t k = 0; k < pd; ++k)
            qp[k] = [&] {
                float acc = 0;
                for (uint32_t d = 0; d < dim; ++d) acc += proj[k * dim + d] * v[d];
                return acc - mproj[k];
            }();
        for (uint32_t c = 0; c < nrc; ++c) {
            float acc = 0;
            for (uint32_t k = 0; k < pd; ++k) acc += rc[c * pd + k] * qp[k];
            rs[c] = acc;
        }
        for (uint32_t c = 0; c < nlc; ++c) {
            float acc = 0;
            for (uint32_t k = 0; k < pd; ++k) acc += lc[c * pd + k] * qp[k];
            ls[c] = acc;
        }
        const Flat r = flatness(rs), l = flatness(ls);
        std::printf("%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    i, r.gap12, r.nent, r.mass8, l.gap12, l.nent, l.mass8);
    }
    return 0;
}
