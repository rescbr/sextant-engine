// Shared PQ-probe and sampling helpers — extracted from engine.cpp.
// See probe.hpp for the design rationale (Builder + Estimator share these).

#include "probe.hpp"

#include "quant/pq_quantizer.hpp"
#include "sextant/error.hpp"

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sextant {

// ---------------------------------------------------------------------------
// compute_truth — brute-force true top-k for the probe queries.
// ---------------------------------------------------------------------------

ProbeTruth compute_truth(const float* pool, uint32_t pool_n, Dim dim,
                         const std::vector<uint32_t>& qidx,
                         uint32_t truth_k) {
    const auto t_norms_start = std::chrono::steady_clock::now();
    std::vector<float> norms(pool_n);
    for (uint32_t i = 0; i < pool_n; i++) {
        const float* v = pool + static_cast<size_t>(i) * dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; d++) acc += double(v[d]) * v[d];
        norms[i] = float(acc);
    }
    const double norms_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_norms_start).count();

    ProbeTruth truth;
    truth.ids.resize(qidx.size());
    truth.dists.resize(qidx.size());
    truth.band_counts.resize(qidx.size(), 0);
    truth.band30_counts.resize(qidx.size(), 0);

    // Parallelize across queries — each query's truth computation is fully
    // independent: it reads pool/norms/qidx (read-only) and writes only its
    // own disjoint truth.*[qi] slots. The per-query `ranked` vector is
    // stack-local. No shared mutable state → no locking needed.
    const uint32_t nthreads = std::thread::hardware_concurrency();
    const auto t_qloop_start = std::chrono::steady_clock::now();
    std::atomic<size_t> next_qi{0};
    auto worker = [&]() {
        while (true) {
            const size_t qi = next_qi.fetch_add(1, std::memory_order_relaxed);
            if (qi >= qidx.size()) break;
            const float* q = pool + static_cast<size_t>(qidx[qi]) * dim;
            double qn = 0.0;
            for (uint32_t d = 0; d < dim; d++) qn += double(q[d]) * q[d];
            std::vector<std::pair<float, uint32_t>> ranked;
            ranked.reserve(pool_n - 1);
            for (uint32_t i = 0; i < pool_n; i++) {
                if (i == qidx[qi]) continue;
                double dot = 0.0;
                const float* v = pool + static_cast<size_t>(i) * dim;
                for (uint32_t d = 0; d < dim; d++)
                    dot += double(q[d]) * v[d];
                ranked.emplace_back(float(norms[i] - 2.0 * dot + qn), i);
            }
            const size_t ksort = std::min<size_t>(
                std::max<uint32_t>(truth_k, kProbeRecallK), ranked.size());
            std::partial_sort(ranked.begin(), ranked.begin() + ksort,
                              ranked.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first < b.first;
                              });
            const size_t kk = std::min<size_t>(truth_k, ranked.size());
            for (size_t k = 0; k < kk; k++) {
                truth.ids[qi].push_back(ranked[k].second);
                truth.dists[qi].push_back(ranked[k].first);
            }
            const float band_radius = ranked[kk - 1].first;
            const float band30_radius = ranked[ksort - 1].first;
            uint32_t bc = 0, bc30 = 0;
            for (uint32_t i = 0; i < pool_n; i++) {
                if (i == qidx[qi]) continue;
                const float* v = pool + static_cast<size_t>(i) * dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < dim; d++)
                    dot += double(q[d]) * v[d];
                const float td = float(norms[i] - 2.0 * dot + qn);
                if (td <= band30_radius + 1e-6f) {
                    ++bc30;
                    if (td <= band_radius + 1e-6f) ++bc;
                }
            }
            truth.band_counts[qi] = bc;
            truth.band30_counts[qi] = bc30;
        }
    };
    std::vector<std::thread> thrpool;
    for (uint32_t t = 0; t < std::min(nthreads, uint32_t(qidx.size())); t++) {
        thrpool.emplace_back(worker);
    }
    for (auto& th : thrpool) th.join();

    const double qloop_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_qloop_start).count();
    spdlog::debug("[sextant] compute_truth: norms precompute {:.3f}s, "
                  "query loop {:.3f}s ({} threads, {} queries)",
                  norms_sec, qloop_sec, nthreads, qidx.size());
    return truth;
}

// ---------------------------------------------------------------------------
// probe_recall — measure PQ quality for one (m, bits) config.
// ---------------------------------------------------------------------------

ProbeScore probe_recall(const float* pool, uint32_t pool_n, Dim dim,
                        MetricKind metric, uint16_t pq_m, uint8_t bits,
                        const ProbeTruth& truth, const std::vector<uint32_t>& qidx) {
    PqQuantizer q(metric, dim, pq_m, bits);
    q.train(pool, pool_n);
    const uint32_t cs = q.code_size();
    std::vector<uint8_t> codes(static_cast<size_t>(pool_n) * cs);
    for (uint32_t i = 0; i < pool_n; i++)
        q.encode(pool + static_cast<size_t>(i) * dim,
                 codes.data() + static_cast<size_t>(i) * cs);
    std::vector<float> lut(q.lut_size());

    std::vector<double> ratios;
    ratios.reserve(qidx.size() * kProbeTopk);

    std::vector<float> norms(pool_n);
    for (uint32_t i = 0; i < pool_n; i++) {
        const float* v = pool + static_cast<size_t>(i) * dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; d++) acc += double(v[d]) * v[d];
        norms[i] = float(acc);
    }

    uint64_t band_hits = 0, band_total = 0;

    for (size_t qi = 0; qi < qidx.size(); qi++) {
        const float* qv = pool + static_cast<size_t>(qidx[qi]) * dim;
        double qn = 0.0;
        for (uint32_t d = 0; d < dim; d++) qn += double(qv[d]) * qv[d];

        q.preprocess_query(qv, lut.data());

        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(pool_n - 1);
        for (uint32_t i = 0; i < pool_n; i++) {
            if (i == qidx[qi]) continue;
            ranked.emplace_back(
                q.lut_distance(codes.data() + static_cast<size_t>(i) * cs,
                               lut.data()), i);
        }
        const uint32_t kp = std::min<uint32_t>(kProbeRecallK, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + kp, ranked.end(),
                          [](const auto& a, const auto& b){ return a.first < b.first; });

        for (size_t t = 0; t < truth.ids[qi].size(); t++) {
            const uint32_t tid = truth.ids[qi][t];
            const float true_d = truth.dists[qi][t];
            const float pq_d = q.lut_distance(codes.data() + static_cast<size_t>(tid) * cs,
                                              lut.data());
            if (true_d > 1e-9f) {
                ratios.push_back(std::fabs(1.0 -
                    static_cast<double>(pq_d) / static_cast<double>(true_d)));
            }
        }

        const float band_radius = truth.dists[qi].empty()
            ? std::numeric_limits<float>::infinity()
            : truth.dists[qi].back();
        bool any_in_band = false;
        for (uint32_t k = 0; k < kp; k++) {
            const uint32_t id = ranked[k].second;
            const float* v = pool + static_cast<size_t>(id) * dim;
            double dot = 0.0;
            for (uint32_t d = 0; d < dim; d++) dot += double(qv[d]) * v[d];
            const float td = float(norms[id] - 2.0 * dot + qn);
            if (td <= band_radius + 1e-6f) { any_in_band = true; break; }
        }
        for (size_t t = 0; t < truth.ids[qi].size(); t++) {
            band_total++;
            if (any_in_band) band_hits++;
        }
    }

    ProbeScore s;
    if (!ratios.empty()) {
        std::nth_element(ratios.begin(),
                         ratios.begin() + ratios.size() / 2,
                         ratios.end());
        s.distortion = ratios[ratios.size() / 2];
    }
    s.band_recall = band_total ? double(band_hits) / double(band_total) : 0.0;
    if (!truth.band_counts.empty()) {
        uint64_t tied = 0;
        for (uint32_t bc : truth.band_counts) {
            if (bc > kProbeTopk) ++tied;
        }
        s.tie_fraction = double(tied) / double(truth.band_counts.size());
    }
    if (!truth.band30_counts.empty()) {
        uint64_t tied30 = 0;
        for (uint32_t bc : truth.band30_counts) {
            if (bc > kProbeRecallK) ++tied30;
        }
        s.tie30_fraction = double(tied30) / double(truth.band30_counts.size());
    }
    return s;
}

// ---------------------------------------------------------------------------
// candidate_ms + probe_best_config — sweep (m, bits) and pick min-cost.
// ---------------------------------------------------------------------------

std::vector<uint16_t> candidate_ms(Dim dim) {
    std::vector<uint16_t> ms;
    for (uint32_t m = 4; m <= 256; m++) {
        if (dim % m == 0) {
            const uint32_t sd = dim / m;
            if (sd >= 1 && sd <= 12) ms.push_back(static_cast<uint16_t>(m));
        }
    }
    return ms;
}

ProbedConfig probe_best_config(const float* pool, uint32_t pool_n, Dim dim,
                                MetricKind metric,
                                const ProbeTruth& truth,
                                const std::vector<uint32_t>& qidx,
                                uint16_t fixed_m, uint8_t fixed_bits,
                                double max_distortion,
                                std::vector<ProbedConfig>& all_out,
                                std::string& reason_out) {
    std::vector<uint16_t> ms;
    if (fixed_m != 0) {
        ms.push_back(fixed_m);
    } else {
        ms = candidate_ms(dim);
    }
    std::vector<uint8_t> bit_vals;
    if (fixed_bits != 0) bit_vals.push_back(fixed_bits);
    else { bit_vals.push_back(4); bit_vals.push_back(8); }

    std::vector<ProbedConfig> all;
    for (const uint16_t m : ms) {
        if (dim % m != 0) continue;
        for (const uint8_t bits : bit_vals) {
            const ProbeScore s = probe_recall(pool, pool_n, dim, metric, m, bits, truth, qidx);
            const uint32_t cb = (static_cast<uint32_t>(m) * bits + 7) / 8;
            const uint32_t tb = pq_table_bytes(m, bits);
            all.push_back({m, bits, cb, tb, s.distortion, s.band_recall,
                           s.tie_fraction, s.tie30_fraction, 0.0});
            spdlog::info("[sextant] pass 1:   probe m={} bits={} code={}B table={}KB "
                         "distortion={:.4f} band_recall@{}={:.4f} ties@{}={:.2f} ties@{}={:.2f}",
                         m, bits, cb, tb / 1024, s.distortion, kProbeRecallK,
                         s.band_recall, kProbeTopk, s.tie_fraction,
                         kProbeRecallK, s.tie30_fraction);
        }
    }
    if (all.empty()) {
        all_out = all;
        return {ms.empty() ? uint16_t{0} : ms.front(), uint8_t{8}, 0, 0,
                0.0, 0.0, 0.0, 0.0};
    }

    for (auto& c : all) {
        c.cost = static_cast<double>(c.m);
    }

    std::vector<const ProbedConfig*> eligible;
    for (const auto& c : all) {
        if (c.distortion <= max_distortion) eligible.push_back(&c);
    }

    ProbedConfig best;
    std::string reason;
    if (eligible.empty()) {
        best = *std::min_element(all.begin(), all.end(),
            [](const auto& a, const auto& b){ return a.distortion < b.distortion; });
        reason = "above max_distortion; lowest distortion";
        spdlog::warn("[sextant] pass 1: no config meets max_distortion {:.3f}; "
                     "selected lowest-distortion (m={} bits={} distortion={:.4f})",
                     max_distortion, best.m, best.bits, best.distortion);
    } else if (eligible.size() == 1) {
        best = *eligible[0];
        reason = "only config within max_distortion";
    } else {
        const ProbedConfig* pick = eligible[0];
        for (const auto* c : eligible) {
            if (c->cost < pick->cost ||
                (c->cost == pick->cost && c->table_bytes < pick->table_bytes) ||
                (c->cost == pick->cost && c->table_bytes == pick->table_bytes &&
                 c->code_bytes < pick->code_bytes)) {
                pick = c;
            }
        }
        best = *pick;
        reason = "min cost (distortion ≤ bound)";
    }
    spdlog::info("[sextant] pass 1: selected m={} bits={} (code={}B table={}KB "
                 "distortion={:.4f} cost={:.0f}) [{}; max_distortion {:.3f}]",
                 best.m, best.bits, best.code_bytes, best.table_bytes / 1024,
                 best.distortion, best.cost, reason, max_distortion);
    all_out = std::move(all);
    reason_out = reason;
    return best;
}

// ---------------------------------------------------------------------------
// draw_random_sample — seek-based Vitter reservoir sampling.
// ---------------------------------------------------------------------------

SampleElemType infer_sample_elem_type(const std::string& path) {
    auto ends_with_ci = [&](const char* suf) {
        const size_t nn = path.size();
        const size_t mm = std::strlen(suf);
        if (nn < mm) return false;
        for (size_t i = 0; i < mm; i++) {
            char c = path[nn - mm + i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != suf[i]) return false;
        }
        return true;
    };
    if (ends_with_ci(".ibin")) return SampleElemType::Int8;
    if (ends_with_ci(".bbin") || ends_with_ci(".bvecs"))
        return SampleElemType::Uint8;
    return SampleElemType::Float32;
}

std::vector<float> draw_random_sample(const std::string& path,
                                       uint64_t n, Dim dim, uint64_t k) {
    if (k == 0) return {};
    if (k > n) {
        throw Error(ErrorCode::InvalidParam,
                    "draw_random_sample: k (" + std::to_string(k) +
                        ") > n (" + std::to_string(n) + ")");
    }

    const SampleElemType etype = infer_sample_elem_type(path);
    const uint32_t elem_size = (etype == SampleElemType::Float32) ? 4 : 1;
    const size_t vec_bytes = static_cast<size_t>(dim) * elem_size;

    // Partial Fisher-Yates with a sparse swap map: O(k) memory.
    std::vector<uint64_t> indices;
    indices.reserve(static_cast<size_t>(k));
    std::unordered_map<uint64_t, uint64_t> swap_map;
    swap_map.reserve(static_cast<size_t>(k) * 2);

    const auto val_at = [&](uint64_t i) -> uint64_t {
        const auto it = swap_map.find(i);
        return it == swap_map.end() ? i : it->second;
    };
    const auto set_at = [&](uint64_t i, uint64_t v) { swap_map[i] = v; };

    std::mt19937_64 rng(0xC0DE1234ULL);
    for (uint64_t i = 0; i < k; ++i) {
        const uint64_t span = n - i;
        const uint64_t j = i + (rng() % span);
        const uint64_t vi = val_at(i);
        const uint64_t vj = val_at(j);
        set_at(i, vj);
        if (j != i) set_at(j, vi);
        indices.push_back(vj);
    }
    std::sort(indices.begin(), indices.end());

    if (indices.size() != k) {
        throw Error(ErrorCode::InvalidParam,
                    "draw_random_sample: produced " +
                        std::to_string(indices.size()) + " indices, expected " +
                        std::to_string(k));
    }
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= n) {
            throw Error(ErrorCode::InvalidParam,
                        "draw_random_sample: index " +
                            std::to_string(indices[i]) + " >= n=" +
                            std::to_string(n));
        }
        if (i > 0 && indices[i] <= indices[i - 1]) {
            throw Error(ErrorCode::InvalidParam,
                        "draw_random_sample: indices not strictly increasing "
                        "at position " +
                            std::to_string(i));
        }
    }
    spdlog::debug("[sextant] seek-based sampling: k={} of N={}, k seeks",
                  k, n);

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw Error(ErrorCode::IoError,
                    "draw_random_sample: failed to open '" + path + "': " +
                        std::strerror(errno));
    }

    std::vector<float> out(static_cast<size_t>(k) * dim);
    std::vector<uint8_t> raw;
    if (etype != SampleElemType::Float32) {
        raw.resize(vec_bytes);
    }

    double io_sec = 0.0;
    size_t seek_count = 0;

    for (size_t i = 0; i < indices.size(); i++) {
        const uint64_t idx = indices[i];
        const off_t off = static_cast<off_t>(
            8 + idx * static_cast<uint64_t>(dim) * elem_size);
        float* dst = out.data() + static_cast<size_t>(i) * dim;

        const auto t0 = std::chrono::steady_clock::now();

        if (::lseek(fd, off, SEEK_SET) < 0) {
            ::close(fd);
            throw Error(ErrorCode::IoError,
                        "draw_random_sample: lseek failed on '" + path +
                            "': " + std::strerror(errno));
        }
        ++seek_count;

        if (etype == SampleElemType::Float32) {
            size_t got = 0;
            while (got < vec_bytes) {
                ssize_t r = ::read(fd,
                                   reinterpret_cast<uint8_t*>(dst) + got,
                                   vec_bytes - got);
                if (r <= 0) {
                    ::close(fd);
                    throw Error(ErrorCode::CorruptIndex,
                                "draw_random_sample: unexpected EOF on '" +
                                    path + "'");
                }
                got += static_cast<size_t>(r);
            }
        } else {
            size_t got = 0;
            while (got < vec_bytes) {
                ssize_t r = ::read(fd, raw.data() + got, vec_bytes - got);
                if (r <= 0) {
                    ::close(fd);
                    throw Error(ErrorCode::CorruptIndex,
                                "draw_random_sample: unexpected EOF on '" +
                                    path + "'");
                }
                got += static_cast<size_t>(r);
            }
            if (etype == SampleElemType::Int8) {
                for (uint32_t d = 0; d < dim; d++) {
                    dst[d] = static_cast<float>(
                        static_cast<int8_t>(raw[d]));
                }
            } else {  // Uint8
                for (uint32_t d = 0; d < dim; d++) {
                    dst[d] = static_cast<float>(raw[d]);
                }
            }
        }

        io_sec += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }

    ::close(fd);

    spdlog::debug("[sextant] seek-based sampling: read {} vectors in "
                  "{:.3f}s ({} seeks)",
                  k, io_sec, seek_count);

    return out;
}

}  // namespace sextant
