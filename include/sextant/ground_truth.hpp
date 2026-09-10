#pragma once

/// @file ground_truth.hpp
/// Canonical reader for sextant ground-truth kNN files (.gtmm).
///
/// Format (little-endian, produced by scripts/gen_ground_truth.py and
/// tools writers):
///   [magic "GTMM":4B][n:u32][k:u32][metric:u8]
///   then n query rows, each: [ids (k×u32)][dists (k×f32)]
///
/// GOTCHAS this class exists to kill (each has bitten a copy-pasted
/// reader at least once):
///   - Row stride: the FILE's k (e.g. 100) can exceed the graded k —
///     always advance by 2×k×4 bytes per row; never assume k=10.
///   - ids are u32 on disk but RowId (i64) in memory — widen explicitly.
///   - The magic must be validated (legacy headerless files exist).
///   - DISTS MAY BE ZERO-FILLED: some generators (e.g.
///     scripts/fix_cohere10m_gt.sh) write real ids but zeros for the
///     distances — recall only needs ids. Consumers that gate on true
///     distances must check has_dists() first; a zero-filled file reads
///     back as all-zero floats, which is indistinguishable from "the
///     k-th neighbor is at distance 0" if you don't check.

#include "sextant/types.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sextant {

class GroundTruth {
public:
    GroundTruth() = default;

    /// Load a .gtmm file. Throws std::runtime_error on I/O error, bad
    /// magic, or truncated rows.
    static GroundTruth load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open ground truth '" + path + "'");
        constexpr uint32_t kGtMagic = 0x4D4D5447u;  // "GTMM" LE
        uint32_t magic = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        GroundTruth gt;
        uint32_t n = 0, k = 0;
        f.read(reinterpret_cast<char*>(&n), 4);
        f.read(reinterpret_cast<char*>(&k), 4);
        char metric = 0;
        f.read(&metric, 1);
        if (!f || magic != kGtMagic || n == 0 || k == 0) {
            throw std::runtime_error(
                "ground truth '" + path +
                "' missing GTMM magic (legacy format removed — regenerate "
                "with scripts/gen_ground_truth.py)");
        }
        gt.n_ = n;
        gt.k_ = k;
        gt.metric_is_ip_ = (metric != 0);
        gt.ids_.resize(static_cast<size_t>(n) * k);
        gt.dists_.resize(static_cast<size_t>(n) * k);
        std::vector<uint32_t> row(k);
        for (uint32_t i = 0; i < n; ++i) {
            f.read(reinterpret_cast<char*>(row.data()),
                   static_cast<std::streamsize>(k) * 4);
            if (!f) throw std::runtime_error("ground truth '" + path + "' truncated (ids)");
            for (uint32_t j = 0; j < k; ++j)
                gt.ids_[static_cast<size_t>(i) * k + j] = row[j];
            f.read(reinterpret_cast<char*>(&gt.dists_[static_cast<size_t>(i) * k]),
                   static_cast<std::streamsize>(k) * 4);
            if (!f) throw std::runtime_error("ground truth '" + path + "' truncated (dists)");
        }
        gt.dists_valid_ = std::any_of(
            gt.dists_.begin(), gt.dists_.end(),
            [](float d) { return d != 0.0f; });
        return gt;
    }

    uint32_t n() const { return n_; }
    uint32_t k() const { return k_; }
    bool metric_is_ip() const { return metric_is_ip_; }
    /// True when the file carried real distances. False for ids-only
    /// GT (zero-filled dists) — do not use dist() for gating then.
    bool has_dists() const { return dists_valid_; }

    /// Row `q`'s id at rank `r` (0-based). Row ids are widened to RowId.
    RowId id(uint32_t q, uint32_t r) const {
        return ids_[static_cast<size_t>(q) * k_ + r];
    }
    /// Row `q`'s distance at rank `r`.
    float dist(uint32_t q, uint32_t r) const {
        return dists_[static_cast<size_t>(q) * k_ + r];
    }
    /// Flat access (row-major, n × k) for tools that index by
    /// (query × k + rank) directly.
    RowId id_flat(size_t i) const { return ids_[i]; }
    float dist_flat(size_t i) const { return dists_[i]; }
    /// The first `gk` ids of row `q` as a set (for recall scoring).
    /// `gk` is clamped to the file's k.
    std::vector<RowId> top(uint32_t q, uint32_t gk) const {
        gk = std::min(gk, k_);
        return std::vector<RowId>(
            ids_.begin() + static_cast<size_t>(q) * k_,
            ids_.begin() + static_cast<size_t>(q) * k_ + gk);
    }

private:
    uint32_t n_ = 0, k_ = 0;
    bool metric_is_ip_ = false;
    bool dists_valid_ = false;
    std::vector<RowId> ids_;     // n × k, row-major
    std::vector<float> dists_;   // n × k, row-major (zeros if !dists_valid_)
};

}  // namespace sextant
