// ProductResidualQuantizer — additive-residual PQ for the IVF scan path.
//
// Implements progressive per-level k-means training, greedy residual encoding,
// and an m×K LUT whose entries sum across the M_sub levels of each split. The
// scan-path machinery (4-bit packing, FastScan kernel) is inherited from
// PqQuantizer unchanged.

#include "product_residual_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/error.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace sextant {

namespace {

inline uint32_t read_code(const uint8_t* code, uint8_t bits, uint32_t s) {
    if (bits == 8) {
        return static_cast<uint32_t>(code[s]);
    }
    const uint32_t byte_off = s / 2;
    const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
    return (static_cast<uint32_t>(code[byte_off]) >> shift) & 0x0Fu;
}

inline void write_code(uint8_t* code, uint8_t bits, uint32_t s, uint32_t cid) {
    if (bits == 8) {
        code[s] = static_cast<uint8_t>(cid & 0xFFu);
        return;
    }
    const uint32_t byte_off = s / 2;
    const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
    const uint8_t nibble = static_cast<uint8_t>(cid & 0x0Fu);
    code[byte_off] = static_cast<uint8_t>(
        (code[byte_off] & ~(0x0Fu << shift)) | (nibble << shift));
}

/// k-means++ seeding + Lloyd iterations on `n` points of dimension `dim`.
/// Writes K centroids to `centroids_out`. Self-contained (no external scratch).
void rq_kmeans(const float* data, uint64_t n, uint32_t dim, uint32_t k,
               uint64_t seed, uint32_t max_iters, float* centroids_out) {
    if (n == 0) {
        std::memset(centroids_out, 0, size_t(k) * dim * sizeof(float));
        return;
    }
    std::mt19937_64 rng(seed);

    if (n < k) {
        for (uint32_t c = 0; c < k; c++) {
            std::memcpy(centroids_out + c * dim, data + (c % n) * dim,
                        dim * sizeof(float));
        }
        return;
    }

    // k-means++ seeding.
    {
        std::vector<float> min_d2(n, std::numeric_limits<float>::infinity());
        std::uniform_int_distribution<uint64_t> first(0, n - 1);
        const uint64_t i0 = first(rng);
        std::memcpy(centroids_out, data + i0 * dim, dim * sizeof(float));
        for (uint32_t c = 1; c < k; c++) {
            const float* prev = centroids_out + (c - 1) * dim;
            double sum = 0.0;
            for (uint64_t i = 0; i < n; i++) {
                const float d = simd::l2sq_f32(data + i * dim, prev, dim);
                if (d < min_d2[i]) min_d2[i] = d;
                sum += min_d2[i];
            }
            if (sum <= 0.0) {
                std::uniform_int_distribution<uint64_t> u(0, n - 1);
                std::memcpy(centroids_out + c * dim, data + u(rng) * dim,
                            dim * sizeof(float));
                continue;
            }
            std::uniform_real_distribution<double> pick(0.0, sum);
            double r = pick(rng);
            uint64_t chosen = n - 1;
            for (uint64_t i = 0; i < n; i++) {
                r -= min_d2[i];
                if (r <= 0.0) { chosen = i; break; }
            }
            std::memcpy(centroids_out + c * dim, data + chosen * dim,
                        dim * sizeof(float));
        }
    }

    std::vector<uint32_t> assign(n, 0);
    std::vector<double> sum_vec(size_t(k) * dim, 0.0);
    std::vector<uint64_t> count(k, 0);
    std::vector<float> dists(k);

    for (uint32_t iter = 0; iter < max_iters; iter++) {
        uint64_t changed = 0;
        for (uint64_t i = 0; i < n; i++) {
            const float* x = data + i * dim;
            float best_d = std::numeric_limits<float>::infinity();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k; c++) {
                dists[c] = simd::l2sq_f32(x, centroids_out + c * dim, dim);
                if (dists[c] < best_d) { best_d = dists[c]; best_c = c; }
            }
            if (assign[i] != best_c) { assign[i] = best_c; changed++; }
        }

        std::fill(sum_vec.begin(), sum_vec.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);
        for (uint64_t i = 0; i < n; i++) {
            const uint32_t c = assign[i];
            count[c]++;
            const float* v = data + i * dim;
            double* acc = sum_vec.data() + size_t(c) * dim;
            for (uint32_t d = 0; d < dim; d++) acc[d] += v[d];
        }
        for (uint32_t c = 0; c < k; c++) {
            if (count[c] > 0) {
                const double inv = 1.0 / double(count[c]);
                float* cen = centroids_out + c * dim;
                double* acc = sum_vec.data() + size_t(c) * dim;
                for (uint32_t d = 0; d < dim; d++) cen[d] = float(acc[d] * inv);
            } else {
                float worst = -1.0f;
                uint64_t worst_i = 0;
                for (uint64_t i = 0; i < n; i++) {
                    const float* x = data + i * dim;
                    const float d = simd::l2sq_f32(
                        x, centroids_out + assign[i] * dim, dim);
                    if (d > worst) { worst = d; worst_i = i; }
                }
                std::memcpy(centroids_out + c * dim, data + worst_i * dim,
                            dim * sizeof(float));
            }
        }
        if (iter > 0 && changed * 100 < n) break;
    }
}

/// Find the nearest centroid to `sub` among K centroids in `book` (L2).
/// Returns the centroid id and writes its squared distance to `best_d_out`.
inline uint32_t nearest_centroid_l2(const float* book, const float* sub,
                                    uint32_t K, uint32_t dim,
                                    float* best_d_out) {
    uint32_t best = 0;
    float best_d = std::numeric_limits<float>::infinity();
    for (uint32_t c = 0; c < K; c++) {
        const float d = simd::l2sq_f32(sub, book + c * dim, dim);
        if (d < best_d) { best_d = d; best = c; }
    }
    if (best_d_out) *best_d_out = best_d;
    return best;
}

}  // namespace

ProductResidualQuantizer::ProductResidualQuantizer(
    MetricKind metric, Dim dim, uint16_t m, uint8_t bits, uint32_t nsplits,
    uint32_t beam_size, uint64_t seed,
    std::string encode_mode, uint32_t icm_iters,
    uint32_t ils_iters, uint32_t ils_perturb)
    : PqQuantizer(metric, dim, m, bits, seed),
      nsplits_(nsplits == 0 ? 1 : nsplits),
      M_sub_(nsplits == 0 ? 0 : m / nsplits),
      sub_dim_prq_(nsplits == 0 ? dim : dim / nsplits),
      beam_size_(beam_size),
      encode_mode_(std::move(encode_mode)),
      icm_iters_(icm_iters),
      ils_iters_(ils_iters),
      ils_perturb_(ils_perturb) {
    if (nsplits == 0) {
        throw Error(ErrorCode::InvalidParam, "PRQ: nsplits must be > 0");
    }
    if (m % nsplits_ != 0) {
        throw Error(ErrorCode::InvalidParam,
                    "PRQ: m must be divisible by nsplits");
    }
    if (dim % nsplits_ != 0) {
        throw Error(ErrorCode::InvalidParam,
                    "PRQ: dim must be divisible by nsplits");
    }
    if (bits != 4) {
        throw Error(ErrorCode::InvalidParam,
                    "PRQ: only bits=4 is supported (4-bit scan path)");
    }
    if (M_sub_ == 0) {
        throw Error(ErrorCode::InvalidParam, "PRQ: M_sub = m/nsplits must be > 0");
    }
    rq_codebooks_.assign(
        static_cast<size_t>(nsplits_) * M_sub_ * K_ * sub_dim_prq_, 0.0f);
}

// ---------------------------------------------------------------------------
// Training: progressive per-level k-means, per split.
// ---------------------------------------------------------------------------

void ProductResidualQuantizer::train(const float* samples, uint64_t n) {
    if (n == 0) {
        throw Error(ErrorCode::InvalidParam, "PRQ train: no samples provided");
    }
    rq_codebooks_.assign(
        static_cast<size_t>(nsplits_) * M_sub_ * K_ * sub_dim_prq_, 0.0f);

    const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t n_threads = std::min(hw, nsplits_);
    std::atomic<uint32_t> next_split{0};

    auto worker = [&]() {
        // Per-thread scratch, reused across the splits this worker handles.
        std::vector<float> sub_vecs(static_cast<size_t>(n) * sub_dim_prq_);
        std::vector<float> residual(static_cast<size_t>(n) * sub_dim_prq_);
        std::vector<float> level_book(static_cast<size_t>(K_) * sub_dim_prq_);
        std::vector<uint32_t> codes(static_cast<size_t>(n) * M_sub_);

        uint32_t s;
        while ((s = next_split.fetch_add(1, std::memory_order_relaxed)) <
               nsplits_) {
            const float* src = samples + static_cast<size_t>(s) * sub_dim_prq_;
            // Gather sub-vectors for split s.
            for (uint64_t i = 0; i < n; i++) {
                std::memcpy(sub_vecs.data() + i * sub_dim_prq_,
                            src + i * dim_,
                            sub_dim_prq_ * sizeof(float));
            }

            // residual starts as a copy of the sub-vectors.
            std::memcpy(residual.data(), sub_vecs.data(),
                        sub_vecs.size() * sizeof(float));

            float* split_books = rq_codebooks_.data() +
                                 static_cast<size_t>(s) * M_sub_ * K_ *
                                     sub_dim_prq_;

            for (uint32_t lev = 0; lev < M_sub_; lev++) {
                // k-means++ on the current residuals → this level's codebook.
                rq_kmeans(residual.data(), n, sub_dim_prq_, K_,
                          seed_ + s * 1000 + lev, /*max_iters=*/25,
                          level_book.data());
                std::memcpy(split_books + static_cast<size_t>(lev) * K_ *
                                                  sub_dim_prq_,
                            level_book.data(),
                            level_book.size() * sizeof(float));

                // Encode every residual with this level: find nearest centroid,
                // subtract it to form the next level's residual.
                for (uint64_t i = 0; i < n; i++) {
                    float* r = residual.data() + i * sub_dim_prq_;
                    const uint32_t cid = nearest_centroid_l2(
                        level_book.data(), r, K_, sub_dim_prq_, nullptr);
                    codes[i * M_sub_ + lev] = cid;
                    const float* cen = level_book.data() + cid * sub_dim_prq_;
                    for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                        r[d] -= cen[d];
                    }
                }
            }
        }
    };

    std::vector<std::thread> pool;
    for (uint32_t t = 0; t < n_threads; t++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    // Populate per-centroid ||c||² for the L2sq LUT decomposition.
    rq_centroid_sqnorms_.assign(
        static_cast<size_t>(nsplits_) * M_sub_ * K_, 0.0f);
    for (uint32_t s = 0; s < nsplits_; s++) {
        for (uint32_t lev = 0; lev < M_sub_; lev++) {
            const float* book = level_book_(s, lev);
            float* out = rq_centroid_sqnorms_.data() +
                         (static_cast<size_t>(s) * M_sub_ + lev) * K_;
            for (uint32_t c = 0; c < K_; c++) {
                out[c] = simd::dot_f32(book + c * sub_dim_prq_,
                                       book + c * sub_dim_prq_, sub_dim_prq_);
            }
        }
    }

    // Also mirror the codebooks into the base codebook_ so that PQ-derived
    // helpers (e.g. code_distance fallback, build_cross_distance_table) see a
    // sane m×K×sub_dim_pq-shaped buffer. PRQ segments map 1:1 to base segments
    // (m_ == nsplits*M_sub); the per-segment sub-dimension differs but each
    // base segment's block is only read by code that uses the PRQ overrides.
    // We leave codebook_ zero-filled — PRQ never reads it on the scan path.
}

// ---------------------------------------------------------------------------
// Encoding: greedy beam search (beam_size=1 for now).
// ---------------------------------------------------------------------------

void ProductResidualQuantizer::encode(const float* vec,
                                      uint8_t* code_out) const {
    std::memset(code_out, 0, code_size());

    if (encode_mode_ == "icm") {
        encode_icm_(vec, code_out);
        return;
    }

    if (beam_size_ <= 1) {
        std::vector<float> residual(sub_dim_prq_);
        for (uint32_t s = 0; s < nsplits_; s++) {
            const float* sub = vec + static_cast<size_t>(s) * sub_dim_prq_;
            std::memcpy(residual.data(), sub, sub_dim_prq_ * sizeof(float));
            for (uint32_t lev = 0; lev < M_sub_; lev++) {
                const float* book = level_book_(s, lev);
                const uint32_t cid = nearest_centroid_l2(
                    book, residual.data(), K_, sub_dim_prq_, nullptr);
                const uint32_t global_seg = s * M_sub_ + lev;
                write_code(code_out, bits_, global_seg, cid);
                const float* cen = book + cid * sub_dim_prq_;
                for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                    residual[d] -= cen[d];
                }
            }
        }
        return;
    }

    // Beam search encoding. For each split, maintain `beam_size_` hypotheses
    // (residual + partial codes + residual norm). At each level, expand each
    // hypothesis by all K centroids, compute the resulting residual norm, and
    // keep the best `beam_size_` candidates. This escapes the greedy local
    // minima that limit beam=1 quality.
    const uint32_t B = beam_size_;

    struct Hypothesis {
        std::vector<float> residual;
        std::vector<uint32_t> codes;
        float res_norm_sq;
    };

    struct Candidate {
        uint32_t hyp_idx;
        uint32_t cid;
        float new_res_norm_sq;
    };

    for (uint32_t s = 0; s < nsplits_; s++) {
        const float* sub = vec + static_cast<size_t>(s) * sub_dim_prq_;

        std::vector<Hypothesis> beam;
        beam.reserve(B);
        {
            Hypothesis h0;
            h0.residual.assign(sub, sub + sub_dim_prq_);
            h0.codes.reserve(M_sub_);
            h0.res_norm_sq = simd::dot_f32(h0.residual.data(),
                                           h0.residual.data(), sub_dim_prq_);
            beam.push_back(std::move(h0));
        }

        for (uint32_t lev = 0; lev < M_sub_; lev++) {
            const float* book = level_book_(s, lev);
            const uint32_t cur_beam = static_cast<uint32_t>(beam.size());

            std::vector<Candidate> cands;
            cands.reserve(static_cast<size_t>(cur_beam) * K_);
            std::vector<float> new_res(sub_dim_prq_);

            for (uint32_t bi = 0; bi < cur_beam; bi++) {
                const float* res = beam[bi].residual.data();
                for (uint32_t c = 0; c < K_; c++) {
                    const float* cen = book + c * sub_dim_prq_;
                    for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                        new_res[d] = res[d] - cen[d];
                    }
                    const float nrn = simd::dot_f32(new_res.data(),
                                                    new_res.data(),
                                                    sub_dim_prq_);
                    cands.push_back({bi, c, nrn});
                }
            }

            std::partial_sort(cands.begin(),
                              cands.begin() + std::min(B, static_cast<uint32_t>(cands.size())),
                              cands.end(),
                              [](const Candidate& a, const Candidate& b) {
                                  return a.new_res_norm_sq < b.new_res_norm_sq;
                              });

            const uint32_t n_keep = std::min(B, static_cast<uint32_t>(cands.size()));
            std::vector<Hypothesis> new_beam;
            new_beam.reserve(n_keep);
            for (uint32_t ki = 0; ki < n_keep; ki++) {
                const auto& cand = cands[ki];
                Hypothesis h;
                h.codes = beam[cand.hyp_idx].codes;
                h.codes.push_back(cand.cid);
                const float* res = beam[cand.hyp_idx].residual.data();
                const float* cen = book + cand.cid * sub_dim_prq_;
                h.residual.resize(sub_dim_prq_);
                for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                    h.residual[d] = res[d] - cen[d];
                }
                h.res_norm_sq = cand.new_res_norm_sq;
                new_beam.push_back(std::move(h));
            }
            beam = std::move(new_beam);
        }

        const auto& best = beam[0];
        for (uint32_t lev = 0; lev < M_sub_; lev++) {
            const uint32_t global_seg = s * M_sub_ + lev;
            write_code(code_out, bits_, global_seg, best.codes[lev]);
        }
    }
}

// ---------------------------------------------------------------------------
// ICM+ILS encoding (on-the-fly, no precomputed tables).
//
// ICM coordinate descent: for each codebook j, fix all others, find the
// centroid that minimizes the residual. The residual is computed on-the-fly
// from a running reconstruction vector maintained incrementally.
//
// ILS: perturb npert random codebook assignments, re-run ICM, accept
// per-sub-space only if total reconstruction error decreased.
//
// All per-vector state (reconstruction, codes) fits in L1/L2. Codebooks
// (16 KB per sub-space at K=16, sub_dim=32) stay L1-resident across the
// batch. No tables — compute is cheaper than RAM.
// ---------------------------------------------------------------------------

void ProductResidualQuantizer::encode_icm_(const float* vec,
                                            uint8_t* code_out) const {
    std::vector<float> reconstruction(sub_dim_prq_);
    std::vector<float> residual(sub_dim_prq_);
    std::vector<uint32_t> codes(M_sub_);
    std::vector<uint32_t> best_codes(M_sub_);
    std::mt19937 rng(seed_ ^ std::hash<const float*>{}(vec));

    float best_err = std::numeric_limits<float>::max();

    for (uint32_t s = 0; s < nsplits_; s++) {
        const float* sub = vec + static_cast<size_t>(s) * sub_dim_prq_;

        // --- Greedy init ---
        std::memcpy(residual.data(), sub, sub_dim_prq_ * sizeof(float));
        std::memset(reconstruction.data(), 0, sub_dim_prq_ * sizeof(float));
        for (uint32_t lev = 0; lev < M_sub_; lev++) {
            const float* book = level_book_(s, lev);
            const uint32_t cid = nearest_centroid_l2(
                book, residual.data(), K_, sub_dim_prq_, nullptr);
            codes[lev] = cid;
            const float* cen = book + cid * sub_dim_prq_;
            for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                residual[d] -= cen[d];
                reconstruction[d] += cen[d];
            }
        }

        float current_err = simd::dot_f32(residual.data(), residual.data(),
                                           sub_dim_prq_);
        best_err = current_err;
        best_codes = codes;

        // --- ILS loop ---
        for (uint32_t ils = 0; ils < ils_iters_; ils++) {
            // Perturb: replace npert random codebook assignments.
            if (ils > 0 && ils_perturb_ > 0 && M_sub_ > 1) {
                const uint32_t npert = std::min(ils_perturb_, M_sub_);
                for (uint32_t p = 0; p < npert; p++) {
                    const uint32_t lev = rng() % M_sub_;
                    const uint32_t old_cid = codes[lev];
                    const float* book = level_book_(s, lev);
                    const float* old_cen = book + old_cid * sub_dim_prq_;
                    const uint32_t new_cid = rng() % K_;
                    const float* new_cen = book + new_cid * sub_dim_prq_;
                    codes[lev] = new_cid;
                    for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                        reconstruction[d] += new_cen[d] - old_cen[d];
                    }
                }
            }

            // ICM sweeps.
            for (uint32_t iter = 0; iter < icm_iters_; iter++) {
                for (uint32_t lev = 0; lev < M_sub_; lev++) {
                    const float* book = level_book_(s, lev);
                    const uint32_t old_cid = codes[lev];
                    const float* old_cen = book + old_cid * sub_dim_prq_;

                    // residual = x_sub - reconstruction + old_cen
                    // (remove this codebook's contribution)
                    for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                        residual[d] = sub[d] - reconstruction[d] + old_cen[d];
                    }

                    const uint32_t new_cid = nearest_centroid_l2(
                        book, residual.data(), K_, sub_dim_prq_, nullptr);
                    if (new_cid != old_cid) {
                        const float* new_cen = book + new_cid * sub_dim_prq_;
                        codes[lev] = new_cid;
                        for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                            reconstruction[d] += new_cen[d] - old_cen[d];
                        }
                    }
                }
            }

            // Compute total error for this sub-space.
            for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                residual[d] = sub[d] - reconstruction[d];
            }
            const float err = simd::dot_f32(residual.data(), residual.data(),
                                             sub_dim_prq_);

            if (err < best_err) {
                best_err = err;
                best_codes = codes;
            } else {
                // Revert to best: rebuild reconstruction from best_codes.
                codes = best_codes;
                std::memset(reconstruction.data(), 0,
                            sub_dim_prq_ * sizeof(float));
                for (uint32_t lev = 0; lev < M_sub_; lev++) {
                    const float* cen = level_book_(s, lev) +
                                       codes[lev] * sub_dim_prq_;
                    for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                        reconstruction[d] += cen[d];
                    }
                }
            }
        }

        // Write best codes for this sub-space.
        for (uint32_t lev = 0; lev < M_sub_; lev++) {
            const uint32_t global_seg = s * M_sub_ + lev;
            write_code(code_out, bits_, global_seg, best_codes[lev]);
        }
    }
}

// ---------------------------------------------------------------------------
// Decode: sum the M_sub assigned centroids per sub-space (additive).
// ---------------------------------------------------------------------------

void ProductResidualQuantizer::decode_code(const uint8_t* code,
                                           float* out) const {
    std::memset(out, 0, dim_ * sizeof(float));
    for (uint32_t s = 0; s < nsplits_; s++) {
        float* sub_out = out + static_cast<size_t>(s) * sub_dim_prq_;
        for (uint32_t lev = 0; lev < M_sub_; lev++) {
            const uint32_t global_seg = s * M_sub_ + lev;
            const uint32_t cid = read_code(code, bits_, global_seg);
            const float* cen = level_book_(s, lev) + cid * sub_dim_prq_;
            for (uint32_t d = 0; d < sub_dim_prq_; d++) {
                sub_out[d] += cen[d];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LUT construction. m × K floats; segment = s * M_sub + lev.
// ---------------------------------------------------------------------------

void ProductResidualQuantizer::preprocess_query_as(
    MetricKind metric, const float* query, float* out) const {
    // For each split s, the M_sub levels share the same query sub-vector
    // q_sub (dim sub_dim_prq) but dot against distinct codebooks. The full
    // distance for the split is the SUM of per-level contributions, which the
    // scan kernel accumulates automatically by summing LUT[seg][cid] over all
    // segments — so each segment just contributes its own per-level value.
    for (uint32_t s = 0; s < nsplits_; s++) {
        const float* q_sub = query + static_cast<size_t>(s) * sub_dim_prq_;
        for (uint32_t lev = 0; lev < M_sub_; lev++) {
            const float* book = level_book_(s, lev);
            const uint32_t seg = s * M_sub_ + lev;
            float* row = out + seg * K_;
            switch (metric) {
            case MetricKind::InnerProduct:
                for (uint32_t c = 0; c < K_; c++) {
                    row[c] = -simd::dot_f32(q_sub, book + c * sub_dim_prq_,
                                            sub_dim_prq_);
                }
                break;
            case MetricKind::L2Sq:
            default: {
                // ||q-c||² = ||q||² - 2<q,c> + ||c||². ||c||² cached.
                const float q_sq = simd::dot_f32(q_sub, q_sub, sub_dim_prq_);
                const float* cnorms = rq_centroid_sqnorms_.data() +
                                      static_cast<size_t>(seg) * K_;
                for (uint32_t c = 0; c < K_; c++) {
                    const float dot = simd::dot_f32(
                        q_sub, book + c * sub_dim_prq_, sub_dim_prq_);
                    row[c] = q_sq - 2.0f * dot + cnorms[c];
                }
                break;
            }
            }
        }
    }
}

void ProductResidualQuantizer::preprocess_query(const float* query,
                                                float* out) const {
    preprocess_query_as(metric_, query, out);
}

// ---------------------------------------------------------------------------
// Serialization (PRQ-specific, magic-byte-tagged).
// ---------------------------------------------------------------------------

void ProductResidualQuantizer::serialize(std::vector<uint8_t>& out) const {
    // Layout (little-endian):
    //   [magic:u8 = 0xA5][metric:u8][m:u16 LE][bits:u8][dim:u32 LE]
    //   [nsplits:u32 LE][M_sub:u32 LE][beam_size:u32 LE]
    //   [codebooks:f32 × nsplits*M_sub*K*sub_dim_prq]
    const size_t header = 1 + 1 + 2 + 1 + 4 + 4 + 4 + 4;
    const size_t book_bytes = rq_codebooks_.size() * sizeof(float);
    out.resize(header + book_bytes);
    uint8_t* p = out.data();
    p[0] = kMagic;
    p[1] = static_cast<uint8_t>(metric_);
    uint16_t m = m_;
    std::memcpy(p + 2, &m, sizeof(m));
    p[4] = bits_;
    uint32_t d = dim_;
    std::memcpy(p + 5, &d, sizeof(d));
    uint32_t ns = nsplits_;
    std::memcpy(p + 9, &ns, sizeof(ns));
    uint32_t ms = M_sub_;
    std::memcpy(p + 13, &ms, sizeof(ms));
    uint32_t bs = beam_size_;
    std::memcpy(p + 17, &bs, sizeof(bs));
    if (book_bytes > 0) {
        std::memcpy(p + header, rq_codebooks_.data(), book_bytes);
    }
}

void ProductResidualQuantizer::deserialize(const uint8_t* in, size_t size) {
    const size_t header = 1 + 1 + 2 + 1 + 4 + 4 + 4 + 4;
    if (size < header) {
        throw Error(ErrorCode::CorruptIndex, "PRQ deserialize: blob too small");
    }
    if (in[0] != kMagic) {
        throw Error(ErrorCode::CorruptIndex,
                    "PRQ deserialize: bad magic byte");
    }
    metric_ = static_cast<MetricKind>(in[1]);
    uint16_t m;
    std::memcpy(&m, in + 2, sizeof(m));
    m_ = m;
    bits_ = in[4];
    uint32_t d;
    std::memcpy(&d, in + 5, sizeof(d));
    dim_ = d;
    uint32_t ns, ms, bs;
    std::memcpy(&ns, in + 9, sizeof(ns));
    std::memcpy(&ms, in + 13, sizeof(ms));
    std::memcpy(&bs, in + 17, sizeof(bs));

    if (m_ == 0 || ns == 0 || ms == 0 || d == 0 || m_ % ns != 0 ||
        d % ns != 0 || bits_ != 4) {
        throw Error(ErrorCode::CorruptIndex,
                    "PRQ deserialize: invalid header");
    }
    K_ = 1u << bits_;
    sub_dim_ = dim_ / m_;          // base-class sub_dim (unused by PRQ paths).
    nsplits_ = ns;
    M_sub_ = ms;
    sub_dim_prq_ = dim_ / ns;
    beam_size_ = bs;

    const size_t book_floats =
        static_cast<size_t>(nsplits_) * M_sub_ * K_ * sub_dim_prq_;
    const size_t book_bytes = book_floats * sizeof(float);
    if (size != header + book_bytes) {
        throw Error(ErrorCode::CorruptIndex,
                    "PRQ deserialize: size mismatch");
    }
    rq_codebooks_.assign(book_floats, 0.0f);
    std::memcpy(rq_codebooks_.data(), in + header, book_bytes);

    // Rebuild derived centroid norms.
    rq_centroid_sqnorms_.assign(
        static_cast<size_t>(nsplits_) * M_sub_ * K_, 0.0f);
    for (uint32_t s = 0; s < nsplits_; s++) {
        for (uint32_t lev = 0; lev < M_sub_; lev++) {
            const float* book = level_book_(s, lev);
            float* out = rq_centroid_sqnorms_.data() +
                         (static_cast<size_t>(s) * M_sub_ + lev) * K_;
            for (uint32_t c = 0; c < K_; c++) {
                out[c] = simd::dot_f32(book + c * sub_dim_prq_,
                                       book + c * sub_dim_prq_, sub_dim_prq_);
            }
        }
    }
}

}  // namespace sextant
