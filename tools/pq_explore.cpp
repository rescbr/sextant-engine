// pq_explore — sweep PQ parameters and measure ranking recall.
//
// For each (m, bits) config:
//   1. Train PQ on a sample (using PqQuantizer::train, threaded).
//   2. Encode all sample vectors.
//   3. Compute brute-force true top-K neighbors for N random queries.
//   4. Compute PQ top-K' for those queries; measure recall.
//   5. PQ-to-true distance correlation (Pearson).
//   6. Report: code_bytes, table_MB, train_time, recall, correlation.
//
// Usage:
//   pq_explore --input <fbin> [--sample N] [--queries N] [--topk K]
//              [--recall-k K'] [--m-sweep 32,64,96,128]
//              [--bits-sweep 4,8] [--metric l2|ip]

#include "quant/pq_quantizer.hpp"
#include "sextant/error.hpp"
#include "sextant/types.hpp"

#include <cmdline/cmdline.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using sextant::Dim;
using sextant::MetricKind;
using sextant::PqQuantizer;
using Clock = std::chrono::steady_clock;

template <typename T>
double elapsed_sec(T start, T end) {
    return std::chrono::duration<double>(end - start).count();
}

// ---------------------------------------------------------------------------
// .fbin reader: [uint32 n][uint32 dim][n*dim float32 vectors].
// ---------------------------------------------------------------------------

struct FbinData {
    uint32_t n = 0;
    uint32_t dim = 0;
    std::vector<float> data;  // n * dim, row-major
};

bool read_fbin(const std::string& path, FbinData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&out.n), sizeof(out.n));
    f.read(reinterpret_cast<char*>(&out.dim), sizeof(out.dim));
    if (!f.good() || out.n == 0 || out.dim == 0) return false;
    out.data.resize(static_cast<size_t>(out.n) * out.dim);
    f.read(reinterpret_cast<char*>(out.data.data()),
           static_cast<std::streamsize>(out.data.size() * sizeof(float)));
    return f.good() || f.eof();
}

// ---------------------------------------------------------------------------
// CSV int list parser ("32,64,96" -> vector).
// ---------------------------------------------------------------------------

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try {
            out.push_back(std::stoi(tok));
        } catch (...) {
            std::fprintf(stderr, "pq_explore: bad integer in list: %s\n",
                         tok.c_str());
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Brute-force KNN via |a-b|^2 = |a|^2 + |b|^2 - 2 a.b decomposition.
// We only need the ranking (argpartition by L2Sq), so we compute the
// cross dot-products and norms once. Returns, for each query, the indices
// of the true top-K nearest (excluding self).
// ---------------------------------------------------------------------------

struct KnnResult {
    std::vector<uint32_t> ids;  // n_queries * k
};

/// Compute squared norms for all rows.
std::vector<float> compute_norms(const float* data, uint32_t n, uint32_t dim) {
    std::vector<float> norms(n, 0.0f);
    for (uint32_t i = 0; i < n; i++) {
        const float* v = data + static_cast<size_t>(i) * dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; d++) acc += double(v[d]) * v[d];
        norms[i] = float(acc);
    }
    return norms;
}

/// Compute true top-K for `n_queries` query rows against the pool.
/// `query_idx` are row indices into `data`. Excludes self.
KnnResult brute_force_knn(const float* data, uint32_t n, uint32_t dim,
                          const std::vector<uint32_t>& query_idx,
                          uint32_t topk) {
    const auto norms = compute_norms(data, n, dim);
    std::vector<float> dots(n);

    KnnResult res;
    res.ids.reserve(query_idx.size() * topk);

    for (const uint32_t qi : query_idx) {
        const float* q = data + static_cast<size_t>(qi) * dim;
        // Dot products q . x for all x.
        for (uint32_t i = 0; i < n; i++) {
            const float* x = data + static_cast<size_t>(i) * dim;
            double acc = 0.0;
            for (uint32_t d = 0; d < dim; d++) acc += double(q[d]) * x[d];
            dots[i] = float(acc);
        }
        // L2Sq(q, x) = |q|^2 + |x|^2 - 2 q.x. Since |q|^2 is constant across
        // the pool it doesn't affect ranking; use |x|^2 - 2 q.x.
        std::vector<std::pair<float, uint32_t>> dists;
        dists.reserve(n);
        for (uint32_t i = 0; i < n; i++) {
            if (i == qi) continue;
            const float d = norms[i] - 2.0f * dots[i];
            dists.emplace_back(d, i);
        }
        // Partial sort for top-K.
        if (topk < dists.size()) {
            std::nth_element(
                dists.begin(), dists.begin() + topk, dists.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            std::sort(dists.begin(), dists.begin() + topk,
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });
            for (uint32_t k = 0; k < topk; k++) res.ids.push_back(dists[k].second);
        } else {
            std::sort(dists.begin(), dists.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });
            for (uint32_t k = 0; k < topk && k < dists.size(); k++)
                res.ids.push_back(dists[k].second);
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Pearson correlation between PQ-estimated and true L2Sq distances for
// random pairs of vectors.
// ---------------------------------------------------------------------------

double pearson_correlation(const PqQuantizer& pq,
                           const std::vector<uint8_t>& codes,
                           const std::vector<float>& data, uint32_t n,
                           uint32_t dim, uint32_t n_pairs, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> u(0, n - 1);
    const uint32_t cs = pq.code_size();

    std::vector<double> true_d, pq_d;
    true_d.reserve(n_pairs);
    pq_d.reserve(n_pairs);

    for (uint32_t p = 0; p < n_pairs; p++) {
        const uint32_t i = u(rng);
        const uint32_t j = u(rng);
        if (i == j) continue;
        const float* vi = data.data() + static_cast<size_t>(i) * dim;
        const float* vj = data.data() + static_cast<size_t>(j) * dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; d++) {
            const float diff = vi[d] - vj[d];
            acc += double(diff) * diff;
        }
        true_d.push_back(acc);
        pq_d.push_back(pq.code_distance(codes.data() + size_t(i) * cs,
                                        codes.data() + size_t(j) * cs));
    }

    const size_t m = true_d.size();
    if (m < 2) return 0.0;
    double mt = 0, mp = 0;
    for (size_t k = 0; k < m; k++) { mt += true_d[k]; mp += pq_d[k]; }
    mt /= m; mp /= m;
    double cov = 0, vt = 0, vp = 0;
    for (size_t k = 0; k < m; k++) {
        const double dt = true_d[k] - mt;
        const double dp = pq_d[k] - mp;
        cov += dt * dp; vt += dt * dt; vp += dp * dp;
    }
    if (vt == 0 || vp == 0) return 0.0;
    return cov / std::sqrt(vt * vp);
}

// ---------------------------------------------------------------------------
// Run one PQ config: train, encode, recall, correlation.
// ---------------------------------------------------------------------------

struct ConfigResult {
    int m;
    int bits;
    uint32_t code_bytes;
    double table_mb;
    double train_sec;
    double recall;       // recall@recall_k of true top-(topk)
    double corr;
};

ConfigResult run_config(const std::vector<float>& sample, uint32_t n,
                        uint32_t dim, MetricKind metric, int m_val, int bits,
                        const KnnResult& truth, uint32_t topk,
                        uint32_t recall_k,
                        const std::vector<uint32_t>& query_idx) {
    ConfigResult r;
    r.m = m_val;
    r.bits = bits;

    PqQuantizer pq(metric, dim, static_cast<uint16_t>(m_val),
                   static_cast<uint8_t>(bits));

    const auto t0 = Clock::now();
    pq.train(sample.data(), n);
    const auto t1 = Clock::now();
    r.train_sec = elapsed_sec(t0, t1);

    const uint32_t cs = pq.code_size();
    r.code_bytes = cs;
    // Cross-distance table size: m * K * K * 4 bytes.
    const uint32_t K = 1u << bits;
    r.table_mb = double(m_val) * K * K * sizeof(float) / (1024.0 * 1024.0);

    // Encode all sample vectors.
    std::vector<uint8_t> codes(static_cast<size_t>(n) * cs);
    for (uint32_t i = 0; i < n; i++) {
        pq.encode(sample.data() + static_cast<size_t>(i) * dim,
                  codes.data() + static_cast<size_t>(i) * cs);
    }

    // PQ top-recall_k for each query via ADC LUT.
    std::vector<float> lut(pq.lut_size());
    const uint32_t nq = static_cast<uint32_t>(query_idx.size());

    uint64_t hits = 0;
    uint64_t total_truth = 0;
    for (uint32_t q = 0; q < nq; q++) {
        const uint32_t qi = query_idx[q];
        pq.preprocess_query(sample.data() + static_cast<size_t>(qi) * dim,
                            lut.data());

        // Rank all by PQ distance.
        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(n - 1);
        for (uint32_t i = 0; i < n; i++) {
            if (i == qi) continue;
            const float d =
                pq.lut_distance(codes.data() + static_cast<size_t>(i) * cs,
                                lut.data());
            ranked.emplace_back(d, i);
        }
        const uint32_t k_pick = std::min<uint32_t>(recall_k, ranked.size());
        std::nth_element(
            ranked.begin(), ranked.begin() + k_pick, ranked.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        std::unordered_set<uint32_t> pq_set;
        pq_set.reserve(k_pick);
        for (uint32_t k = 0; k < k_pick; k++) pq_set.insert(ranked[k].second);

        // True top-topk ids for this query.
        const uint32_t* truth_row = &truth.ids[static_cast<size_t>(q) * topk];
        for (uint32_t k = 0; k < topk; k++) {
            total_truth++;
            if (pq_set.count(truth_row[k])) hits++;
        }
    }
    r.recall = total_truth ? double(hits) / double(total_truth) : 0.0;

    r.corr = pearson_correlation(pq, codes, sample, n, dim, 5000, 0xC0FFEEULL);

    return r;
}

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition via cyclic Jacobi rotations.
// For small dense symmetric matrices (d × d, d ≤ ~32). Returns eigenvalues
// sorted descending. O(d³) per sweep, converges in ~5-10 sweeps.
// ---------------------------------------------------------------------------

void jacobi_eig_sym(std::vector<float> A, uint32_t d,
                    std::vector<float>& eigvals) {
    eigvals.assign(d, 0.0f);
    if (d == 1) { eigvals[0] = A[0]; return; }
    const uint32_t max_sweeps = 60;
    for (uint32_t sweep = 0; sweep < max_sweeps; sweep++) {
        double off = 0.0;
        for (uint32_t p = 0; p < d; p++)
            for (uint32_t q = p + 1; q < d; q++)
                off += double(A[p * d + q]) * A[p * d + q];
        if (off < 1e-20) break;
        for (uint32_t p = 0; p < d; p++) {
            for (uint32_t q = p + 1; q < d; q++) {
                double apq = A[p * d + q];
                if (std::fabs(apq) < 1e-18) continue;
                double app = A[p * d + p];
                double aqq = A[q * d + q];
                double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
                double c = std::cos(phi), s = std::sin(phi);
                for (uint32_t i = 0; i < d; i++) {
                    double aip = A[i * d + p];
                    double aiq = A[i * d + q];
                    A[i * d + p] = float(c * aip - s * aiq);
                    A[i * d + q] = float(s * aip + c * aiq);
                }
                for (uint32_t i = 0; i < d; i++) {
                    double api = A[p * d + i];
                    double aqi = A[q * d + i];
                    A[p * d + i] = float(c * api - s * aqi);
                    A[q * d + i] = float(s * api + c * aqi);
                }
            }
        }
    }
    for (uint32_t i = 0; i < d; i++) eigvals[i] = A[i * d + i];
    std::sort(eigvals.begin(), eigvals.end(), std::greater<float>());
}

// ---------------------------------------------------------------------------
// Per-segment diagnostic. For segment `s` of a trained quantizer:
//   - covariance eigenvalues (from the sample sub-vectors) → participation
//     ratio sum(λ)^2 / sum(λ^2) and condition λ1 / λ_min
//   - SSE under this segment's codebook (quantization error on the sample)
//   - assignment entropy (distribution of nearest-centroid picks)
// ---------------------------------------------------------------------------

struct SegDiag {
    double sse;
    double entropy_bits;  // entropy in bits, [0, log2(K)]
    double eig_participation;  // participation ratio, [1, sub_dim]
    double eig_condition;      // λ1 / λ_min (capped)
};

SegDiag diagnose_segment(const float* sub_vectors, uint32_t n,
                         uint32_t sub_dim, const float* slot_book,
                         uint32_t K) {
    SegDiag d{};
    // SSE + assignment counts.
    std::vector<uint64_t> counts(K, 0);
    double sse = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        const float* v = sub_vectors + static_cast<size_t>(i) * sub_dim;
        float best = std::numeric_limits<float>::infinity();
        uint32_t best_c = 0;
        for (uint32_t c = 0; c < K; c++) {
            const float* cen = slot_book + c * sub_dim;
            float dd = 0.0f;
            for (uint32_t z = 0; z < sub_dim; z++) {
                const float t = v[z] - cen[z];
                dd += t * t;
            }
            if (dd < best) { best = dd; best_c = c; }
        }
        sse += best;
        counts[best_c]++;
    }
    d.sse = sse / n;
    // Entropy.
    double H = 0.0;
    for (uint32_t c = 0; c < K; c++) {
        if (counts[c] == 0) continue;
        double p = double(counts[c]) / n;
        H -= p * std::log2(p);
    }
    d.entropy_bits = H;
    // Covariance eigenvalues (center the sub-vectors).
    std::vector<double> mean(sub_dim, 0.0);
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t z = 0; z < sub_dim; z++)
            mean[z] += sub_vectors[static_cast<size_t>(i) * sub_dim + z];
    for (uint32_t z = 0; z < sub_dim; z++) mean[z] /= n;
    std::vector<float> cov(static_cast<size_t>(sub_dim) * sub_dim, 0.0f);
    for (uint32_t i = 0; i < n; i++) {
        const float* v = sub_vectors + static_cast<size_t>(i) * sub_dim;
        for (uint32_t a = 0; a < sub_dim; a++) {
            const double da = v[a] - mean[a];
            for (uint32_t b = a; b < sub_dim; b++) {
                const double db = v[b] - mean[b];
                cov[a * sub_dim + b] += float(da * db);
            }
        }
    }
    const double inv = 1.0 / (n > 1 ? n - 1 : 1);
    for (uint32_t a = 0; a < sub_dim; a++)
        for (uint32_t b = a; b < sub_dim; b++) {
            cov[a * sub_dim + b] = float(cov[a * sub_dim + b] * inv);
            cov[b * sub_dim + a] = cov[a * sub_dim + b];  // symmetric
        }
    std::vector<float> eig;
    jacobi_eig_sym(cov, sub_dim, eig);
    double sum_eig = 0.0, sum_eig_sq = 0.0;
    for (double e : eig) {
        if (e < 0) e = 0;
        sum_eig += e;
        sum_eig_sq += e * e;
    }
    d.eig_participation = sum_eig_sq > 0 ? (sum_eig * sum_eig) / sum_eig_sq : 0.0;
    const double lmax = eig.front();
    double lmin = eig.back();
    if (lmin < 1e-12) lmin = 1e-12;
    d.eig_condition = lmax / lmin;
    return d;
}

/// Run the per-segment diagnostic at a fixed m. Trains both 4-bit and 8-bit
/// PQ, then for each segment prints: SSE_4/SSE_8 ratio, assignment entropy,
//  eigen participation ratio, eigen condition.
void run_diagnostic(const std::vector<float>& sample, uint32_t n, Dim dim,
                    MetricKind metric, int m_val) {
    const uint32_t sub_dim = dim / m_val;
    std::printf("\n=== Diagnostic: m=%d (sub_dim=%u), n=%u ===\n", m_val,
                sub_dim, n);

    // Gather per-segment sub-vectors once (shared across bit depths).
    std::vector<std::vector<float>> subvecs(m_val);
    for (int s = 0; s < m_val; s++) {
        subvecs[s].resize(static_cast<size_t>(n) * sub_dim);
        for (uint32_t i = 0; i < n; i++) {
            std::memcpy(subvecs[s].data() + static_cast<size_t>(i) * sub_dim,
                        sample.data() + static_cast<size_t>(i) * dim +
                            s * sub_dim,
                        sub_dim * sizeof(float));
        }
    }

    auto train_at_bits = [&](uint8_t bits) -> std::unique_ptr<PqQuantizer> {
        auto pq = std::make_unique<PqQuantizer>(metric, dim,
                                                static_cast<uint16_t>(m_val), bits);
        pq->train(sample.data(), n);
        return pq;
    };
    auto pq8 = train_at_bits(8);
    auto pq4 = train_at_bits(4);
    const uint32_t K8 = pq8->K(), K4 = pq4->K();

    std::printf("%-4s %-10s %-10s %-10s %-12s %-12s %-12s\n", "seg",
                "SSE_8", "SSE_4", "SSE4/8", "entropy_8b", "eig_part", "eig_cond");
    std::printf("%s\n", "--------------------------------------------------------"
                        "-------------------------------");

    std::vector<double> ratios;
    ratios.reserve(m_val);
    for (int s = 0; s < m_val; s++) {
        const float* sv = subvecs[s].data();
        const float* book8 = pq8->codebook() + static_cast<size_t>(s) * K8 * sub_dim;
        const float* book4 = pq4->codebook() + static_cast<size_t>(s) * K4 * sub_dim;
        const auto d8 = diagnose_segment(sv, n, sub_dim, book8, K8);
        const auto d4 = diagnose_segment(sv, n, sub_dim, book4, K4);
        const double ratio = d8.sse > 0 ? d4.sse / d8.sse : 0.0;
        ratios.push_back(ratio);
        std::printf("%-4d %-10.4g %-10.4g %-10.3f %-12.3f %-12.2f %-12.1f\n",
                    s, d8.sse, d4.sse, ratio, d8.entropy_bits,
                    d8.eig_participation, d8.eig_condition);
    }

    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    };
    std::printf("\n  SSE4/8  ratio: median=%.3f  min=%.3f  max=%.3f\n",
                med(ratios), *std::min_element(ratios.begin(), ratios.end()),
                *std::max_element(ratios.begin(), ratios.end()));
    std::printf("  (ratio≈1 → 4-bit loses little; ratio>>1 → 8-bit needed)\n");
}

// ---------------------------------------------------------------------------
// Per-partition experiment (--partitions K).
//
// K-means partitions the sample into K shards (on raw floats, approximating
// what the engine does on PQ codes). For each partition: train 4-bit and
// 8-bit PQ on its members, measure recall@K' against within-partition
// brute-force truth. The spread of (r8 - r4) across partitions tells us
// whether per-partition bits is worth it.
// ---------------------------------------------------------------------------

/// Lloyd's k-means on raw floats (shared utility for the experiment).
std::vector<uint32_t> kmeans_partition_floats(const float* data, uint32_t n,
                                              uint32_t dim, uint32_t K,
                                              uint32_t iters, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> centroids(static_cast<size_t>(K) * dim);
    // k-means++ init.
    std::uniform_int_distribution<uint32_t> first(0, n - 1);
    std::memcpy(centroids.data(), data + static_cast<size_t>(first(rng)) * dim,
                dim * sizeof(float));
    std::vector<double> min_d2(n, std::numeric_limits<double>::infinity());
    for (uint32_t k = 1; k < K; k++) {
        const float* prev = centroids.data() + static_cast<size_t>(k - 1) * dim;
        double sum = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            double d = 0.0;
            const float* v = data + static_cast<size_t>(i) * dim;
            for (uint32_t z = 0; z < dim; z++) d += double(v[z] - prev[z]) * (v[z] - prev[z]);
            if (d < min_d2[i]) min_d2[i] = d;
            sum += min_d2[i];
        }
        std::uniform_real_distribution<double> pick(0.0, sum);
        double r = pick(rng);
        uint32_t chosen = n - 1;
        for (uint32_t i = 0; i < n; i++) { r -= min_d2[i]; if (r <= 0) { chosen = i; break; } }
        std::memcpy(centroids.data() + static_cast<size_t>(k) * dim,
                    data + static_cast<size_t>(chosen) * dim, dim * sizeof(float));
    }
    // Lloyd iterations.
    std::vector<uint32_t> assign(n, 0);
    for (uint32_t iter = 0; iter < iters; iter++) {
        for (uint32_t i = 0; i < n; i++) {
            double best = std::numeric_limits<double>::infinity();
            uint32_t bk = 0;
            for (uint32_t k = 0; k < K; k++) {
                double d = 0.0;
                const float* v = data + static_cast<size_t>(i) * dim;
                const float* c = centroids.data() + static_cast<size_t>(k) * dim;
                for (uint32_t z = 0; z < dim; z++) d += double(v[z] - c[z]) * (v[z] - c[z]);
                if (d < best) { best = d; bk = k; }
            }
            assign[i] = bk;
        }
        std::vector<double> sumv(static_cast<size_t>(K) * dim, 0.0);
        std::vector<uint64_t> cnt(K, 0);
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t k = assign[i];
            cnt[k]++;
            const float* v = data + static_cast<size_t>(i) * dim;
            double* acc = sumv.data() + static_cast<size_t>(k) * dim;
            for (uint32_t z = 0; z < dim; z++) acc[z] += v[z];
        }
        for (uint32_t k = 0; k < K; k++) {
            if (cnt[k] > 0) {
                float* c = centroids.data() + static_cast<size_t>(k) * dim;
                double* acc = sumv.data() + static_cast<size_t>(k) * dim;
                for (uint32_t z = 0; z < dim; z++) c[z] = float(acc[z] / cnt[k]);
            }
        }
    }
    return assign;
}

/// Train PQ at given bits on a vector subset, measure recall@K' against the
/// supplied within-subset brute-force truth.
double probe_recall(const float* members, uint32_t n_member, Dim dim,
                    MetricKind metric, int m_val, uint8_t bits,
                    const KnnResult& truth, uint32_t topk, uint32_t recall_k,
                    const std::vector<uint32_t>& q_local) {
    if (n_member < 4) return 1.0;  // degenerate
    PqQuantizer pq(metric, dim, static_cast<uint16_t>(m_val), bits);
    // Ensure divisibility: if dim % m != 0, this throws — caller filters.
    pq.train(members, n_member);
    const uint32_t cs = pq.code_size();
    std::vector<uint8_t> codes(static_cast<size_t>(n_member) * cs);
    for (uint32_t i = 0; i < n_member; i++)
        pq.encode(members + static_cast<size_t>(i) * dim,
                  codes.data() + static_cast<size_t>(i) * cs);
    std::vector<float> lut(pq.lut_size());
    uint64_t hits = 0, total = 0;
    for (uint32_t qi = 0; qi < q_local.size(); qi++) {
        pq.preprocess_query(members + static_cast<size_t>(q_local[qi]) * dim,
                            lut.data());
        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(n_member - 1);
        for (uint32_t i = 0; i < n_member; i++) {
            if (i == q_local[qi]) continue;
            ranked.emplace_back(
                pq.lut_distance(codes.data() + static_cast<size_t>(i) * cs,
                                lut.data()), i);
        }
        uint32_t kp = std::min<uint32_t>(recall_k, ranked.size());
        std::nth_element(ranked.begin(), ranked.begin() + kp, ranked.end(),
                         [](const auto& a, const auto& b){ return a.first < b.first; });
        std::unordered_set<uint32_t> s;
        for (uint32_t k = 0; k < kp; k++) s.insert(ranked[k].second);
        const uint32_t* tr = &truth.ids[static_cast<size_t>(qi) * topk];
        for (uint32_t k = 0; k < topk; k++) { total++; if (s.count(tr[k])) hits++; }
    }
    return total ? double(hits) / total : 0.0;
}

void run_partitions(const std::vector<float>& sample, uint32_t n, Dim dim,
                    MetricKind metric, int m_val, int K, uint32_t topk,
                    uint32_t recall_k, uint32_t queries_per_part) {
    if (dim % static_cast<Dim>(m_val) != 0) {
        std::fprintf(stderr, "pq_explore: dim %u not divisible by m=%d\n", dim, m_val);
        return;
    }
    std::printf("\n=== Per-partition probe: K=%d, m=%d (sub_dim=%u), n=%u ===\n",
                K, m_val, dim / m_val, n);
    std::printf("Partitioning %u vectors into %d shards (k-means on floats)...\n",
                n, K);
    const auto assign = kmeans_partition_floats(sample.data(), n, dim,
                                                static_cast<uint32_t>(K),
                                                25, 0x5E4747ULL);
    // Bucket members by partition.
    std::vector<std::vector<uint32_t>> members(K);
    for (uint32_t i = 0; i < n; i++) members[assign[i]].push_back(i);

    std::printf("%-6s %-8s %-10s %-10s %-9s\n", "part", "n", "recall4", "recall8", "r8-r4");
    std::printf("%s\n", "-------------------------------------------------");

    std::vector<double> r4s, r8s, deltas;
    for (int k = 0; k < K; k++) {
        const auto& mem = members[k];
        const uint32_t nm = static_cast<uint32_t>(mem.size());
        if (nm < static_cast<uint32_t>(topk) + 4) {
            std::printf("%-6d %-8u (too few members)\n", k, nm);
            continue;
        }
        // Gather member vectors into a contiguous buffer.
        std::vector<float> mvecs(static_cast<size_t>(nm) * dim);
        for (uint32_t i = 0; i < nm; i++)
            std::memcpy(mvecs.data() + static_cast<size_t>(i) * dim,
                        sample.data() + static_cast<size_t>(mem[i]) * dim,
                        dim * sizeof(float));
        // Local query indices.
        std::mt19937_64 rng(0xABCDEFULL + k);
        std::uniform_int_distribution<uint32_t> u(0, nm - 1);
        std::vector<uint32_t> q_local;
        std::unordered_set<uint32_t> seen;
        const uint32_t nq = std::min<uint32_t>(queries_per_part, nm - 1);
        while (q_local.size() < nq) { uint32_t q = u(rng); if (seen.insert(q).second) q_local.push_back(q); }
        // Within-partition brute-force truth.
        const auto truth = brute_force_knn(mvecs.data(), nm, dim, q_local, topk);
        const double r4 = probe_recall(mvecs.data(), nm, dim, metric, m_val, 4,
                                       truth, topk, recall_k, q_local);
        const double r8 = probe_recall(mvecs.data(), nm, dim, metric, m_val, 8,
                                       truth, topk, recall_k, q_local);
        r4s.push_back(r4); r8s.push_back(r8); deltas.push_back(r8 - r4);
        std::printf("%-6d %-8u %-10.4f %-10.4f %-+9.4f\n", k, nm, r4, r8, r8 - r4);
    }

    auto stats = [](const std::vector<double>& v) {
        if (v.empty()) return std::array<double,3>{0.0,0.0,0.0};
        auto s = v; std::sort(s.begin(), s.end());
        return std::array<double,3>{s.front(), s[s.size()/2], s.back()};
    };
    auto d4 = stats(r4s), d8 = stats(r8s), dd = stats(deltas);
    std::printf("\n  recall4: min=%.4f med=%.4f max=%.4f\n", d4[0], d4[1], d4[2]);
    std::printf("  recall8: min=%.4f med=%.4f max=%.4f\n", d8[0], d8[1], d8[2]);
    std::printf("  r8-r4:   min=%.4f med=%.4f max=%.4f  (spread = max-min = %.4f)\n",
                dd[0], dd[1], dd[2], dd[2] - dd[0]);
    std::printf("  (small spread → global bits suffices; wide spread → per-partition bits justified)\n");
}

}  // namespace

int main(int argc, char** argv) {
    cmdline::parser p;
    p.add<std::string>("input", 'i', "input .fbin file", true);
    p.add<uint32_t>("sample", 's', "sample N vectors for PQ training/eval",
                    false, 20000);
    p.add<uint32_t>("queries", 'q', "number of random query vectors", false,
                    200);
    p.add<uint32_t>("topk", 'k', "true top-K for recall denominator", false,
                    10);
    p.add<uint32_t>("recall-k", 0,
                    "shortlist size K' for PQ ranking (recall@K')", false, 30);
    p.add<std::string>("m-sweep", 'm', "comma-separated m values", false,
                       "32,64,96,128");
    p.add<std::string>("bits-sweep", 'b', "comma-separated bits values", false,
                       "8");
    p.add<std::string>("metric", 0, "metric: l2 or ip", false, "l2");
    p.add("diagnose", 0, "per-segment diagnostic mode (SSE ratio, entropy, eigen)");
    p.add<int>("diag-m", 0, "m for diagnostic mode", false, 64);
    p.add<int>("partitions", 0, "per-partition 4-vs-8-bit probe mode (K shards)", false, 0);
    p.add<int>("part-m", 0, "m for per-partition mode", false, 64);
    p.parse_check(argc, argv);

    const std::string input = p.get<std::string>("input");
    const uint32_t sample_n = p.get<uint32_t>("sample");
    const uint32_t n_queries = p.get<uint32_t>("queries");
    const uint32_t topk = p.get<uint32_t>("topk");
    const uint32_t recall_k = p.get<uint32_t>("recall-k");
    const auto m_sweep = parse_int_list(p.get<std::string>("m-sweep"));
    const auto bits_sweep = parse_int_list(p.get<std::string>("bits-sweep"));
    const std::string metric_s = p.get<std::string>("metric");
    const MetricKind metric =
        (metric_s == "ip") ? MetricKind::InnerProduct : MetricKind::L2Sq;

    FbinData fbin;
    if (!read_fbin(input, fbin)) {
        std::fprintf(stderr, "pq_explore: cannot read %s\n", input.c_str());
        return 1;
    }
    std::printf("Loaded %s: %u vectors, dim=%u\n", input.c_str(), fbin.n,
                fbin.dim);
    const Dim dim = fbin.dim;

    if (topk > recall_k) {
        std::fprintf(stderr,
                     "pq_explore: --topk (%u) must be <= --recall-k (%u)\n",
                     topk, recall_k);
        return 1;
    }

    // Sample N vectors (deterministic for reproducibility).
    std::mt19937_64 rng(0x5E4747ULL);
    const uint32_t n = std::min<uint32_t>(sample_n, fbin.n);
    std::vector<uint32_t> sample_idx(fbin.n);
    std::iota(sample_idx.begin(), sample_idx.end(), 0u);
    std::shuffle(sample_idx.begin(), sample_idx.end(), rng);
    sample_idx.resize(n);

    std::vector<float> sample(static_cast<size_t>(n) * dim);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t src = sample_idx[i];
        std::memcpy(sample.data() + static_cast<size_t>(i) * dim,
                    fbin.data.data() + static_cast<size_t>(src) * dim,
                    dim * sizeof(float));
    }

    // Diagnostic mode: per-segment analysis, no sweep.
    if (p.exist("diagnose")) {
        const int diag_m = p.get<int>("diag-m");
        if (diag_m <= 0 || dim % static_cast<Dim>(diag_m) != 0) {
            std::fprintf(stderr,
                         "pq_explore: --diag-m %d invalid for dim %u\n",
                         diag_m, dim);
            return 1;
        }
        run_diagnostic(sample, n, dim, metric, diag_m);
        return 0;
    }

    // Per-partition probe mode.
    if (const int K = p.get<int>("partitions"); K > 0) {
        const int part_m = p.get<int>("part-m");
        run_partitions(sample, n, dim, metric, part_m, K, topk, recall_k,
                       std::min<uint32_t>(n_queries, 50u));
        return 0;
    }

    // Pick query indices (subset of the sample, so we have truth within pool).
    std::vector<uint32_t> query_idx;
    {
        std::uniform_int_distribution<uint32_t> u(0, n - 1);
        std::unordered_set<uint32_t> seen;
        while (query_idx.size() < n_queries) {
            const uint32_t q = u(rng);
            if (seen.insert(q).second) query_idx.push_back(q);
        }
    }

    std::printf("Computing brute-force true top-%u for %u queries over %u "
                "vectors...\n",
                topk, n_queries, n);
    const auto tb0 = Clock::now();
    const KnnResult truth = brute_force_knn(sample.data(), n, dim, query_idx, topk);
    const auto tb1 = Clock::now();
    std::printf("Brute-force KNN done (%.2fs)\n\n", elapsed_sec(tb0, tb1));

    // Header. Build the recall column label from the recall-k value.
    char recall_col[24];
    std::snprintf(recall_col, sizeof(recall_col), "recall@%u", recall_k);
    std::printf("%-5s %-5s %-8s %-9s %-9s %-13s %-10s\n", "m", "bits",
                "code", "table", "train", recall_col, "PQ-true");
    std::printf("%-5s %-5s %-8s %-9s %-9s %-13s %-10s\n", "", "",
                "bytes", "MB", "sec",
                ("of top-" + std::to_string(topk)).c_str(), "corr");
    std::printf("%s\n",
                "----------------------------------------------------------------"
                "-------------------");

    std::vector<ConfigResult> results;
    for (const int bits : bits_sweep) {
        for (const int m_val : m_sweep) {
            if (m_val <= 0 || dim % static_cast<Dim>(m_val) != 0) {
                std::printf(
                    "%-5d %-5d  (skipped: dim %u not divisible by m=%d)\n",
                    m_val, bits, dim, m_val);
                continue;
            }
            if (bits != 4 && bits != 8) {
                std::printf("%-5d %-5d  (skipped: bits must be 4 or 8)\n",
                            m_val, bits);
                continue;
            }
            try {
                const auto r = run_config(sample, n, dim, metric, m_val, bits,
                                          truth, topk, recall_k, query_idx);
                std::printf("%-5d %-5d %-8u %-9.2f %-9.2f %-13.4f %-10.4f\n",
                            r.m, r.bits, r.code_bytes, r.table_mb, r.train_sec,
                            r.recall, r.corr);
                results.push_back(r);
            } catch (const sextant::Error& e) {
                std::printf("%-5d %-5d  ERROR: %s\n", m_val, bits, e.what());
            }
        }
        std::printf("\n");
    }

    return 0;
}
