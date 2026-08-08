#include "scalar_lloydmax_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <numeric>

namespace sextant {

ScalarLloydMaxQuantizer::ScalarLloydMaxQuantizer(MetricKind metric, Dim dim,
                                                   uint8_t bits)
    : metric_(metric), dim_(dim), bits_(bits), K_(1u << bits) {
    if (bits_ != 4 && bits_ != 8) {
        throw Error(ErrorCode::InvalidParam,
                    "ScalarLloydMaxQuantizer: bits must be 4 or 8");
    }
    levels_.resize(static_cast<size_t>(dim_) * K_);
    bounds_.resize(static_cast<size_t>(dim_) * (K_ - 1));
}

uint32_t ScalarLloydMaxQuantizer::code_size() const {
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(dim_) * bits_ + 7) / 8);
}

// ---------------------------------------------------------------------------
// 1D Lloyd-Max: find K optimal quantization levels for a 1D distribution.
// Returns sorted levels. Uses quantile init + optional restarts.
// ---------------------------------------------------------------------------
namespace {

/// One Lloyd-Max run from a given init. Returns final levels (sorted).
std::vector<float> lloyd_max_1d(const float* data, uint64_t n,
                                 uint32_t num_levels, uint32_t iters,
                                 const std::vector<float>& init) {
    std::vector<float> levels = init;
    std::sort(levels.begin(), levels.end());

    // Scratch: per-level sum and count for centroid update.
    std::vector<double> sums(num_levels, 0.0);
    std::vector<uint64_t> counts(num_levels, 0);

    for (uint32_t iter = 0; iter < iters; ++iter) {
        // Boundaries = midpoints between adjacent levels.
        std::vector<float> bounds(num_levels - 1);
        for (uint32_t i = 0; i < num_levels - 1; ++i)
            bounds[i] = 0.5f * (levels[i] + levels[i + 1]);

        // Assign + accumulate.
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (uint64_t i = 0; i < n; ++i) {
            // Binary search for the level index.
            uint32_t idx = static_cast<uint32_t>(
                std::upper_bound(bounds.begin(), bounds.end(), data[i]) -
                bounds.begin());
            sums[idx] += data[i];
            counts[idx]++;
        }

        // Update levels = mean of assigned data.
        bool converged = true;
        for (uint32_t k = 0; k < num_levels; ++k) {
            if (counts[k] > 0) {
                float new_level = static_cast<float>(sums[k] / counts[k]);
                if (std::fabs(new_level - levels[k]) > 1e-8f)
                    converged = false;
                levels[k] = new_level;
            }
        }
        if (converged) break;
    }

    std::sort(levels.begin(), levels.end());
    return levels;
}

/// Compute MSE of a quantizer on the data.
double quantizer_mse(const float* data, uint64_t n,
                     const std::vector<float>& levels) {
    uint32_t num_levels = static_cast<uint32_t>(levels.size());
    std::vector<float> bounds(num_levels - 1);
    for (uint32_t i = 0; i < num_levels - 1; ++i)
        bounds[i] = 0.5f * (levels[i] + levels[i + 1]);

    double total_sq_error = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t idx = static_cast<uint32_t>(
            std::upper_bound(bounds.begin(), bounds.end(), data[i]) -
            bounds.begin());
        float err = data[i] - levels[idx];
        total_sq_error += static_cast<double>(err) * err;
    }
    return total_sq_error / static_cast<double>(n);
}

}  // namespace

void ScalarLloydMaxQuantizer::train(const float* samples, uint64_t n,
                                     uint32_t n_restarts,
                                     uint32_t lloyd_iters) {
    if (n < K_) {
        throw Error(ErrorCode::InvalidParam,
                    "ScalarLloydMaxQuantizer::train: n=" +
                        std::to_string(n) + " < K=" + std::to_string(K_));
    }

    // Extract per-dimension data into a scratch buffer.
    std::vector<float> col(n);

    for (uint32_t d = 0; d < dim_; ++d) {
        for (uint64_t i = 0; i < n; ++i)
            col[i] = samples[i * dim_ + d];

        // Init 1: uniform quantiles (best default).
        std::vector<float> quantile_init(K_);
        for (uint32_t k = 0; k < K_; ++k) {
            double frac = static_cast<double>(k) / (K_ - 1);
            uint64_t idx = static_cast<uint64_t>(frac * (n - 1));
            quantile_init[k] = col[idx];
        }
        // Sort col for quantile computation (needed for proper quantiles).
        std::sort(col.begin(), col.end());

        auto best = lloyd_max_1d(col.data(), n, K_, lloyd_iters, quantile_init);
        double best_mse = quantizer_mse(col.data(), n, best);

        // Restart with perturbed inits.
        std::mt19937 rng(42 + d);
        std::uniform_real_distribution<float> jitter(-1e-3f, 1e-3f);
        for (uint32_t r = 1; r < n_restarts; ++r) {
            std::vector<float> init(K_);
            for (uint32_t k = 0; k < K_; ++k) {
                double frac = static_cast<double>(k + 0.5) / K_;
                uint64_t idx = static_cast<uint64_t>(frac * (n - 1));
                init[k] = col[idx] + jitter(rng) * std::max(1e-6f,
                    (col.back() - col.front()) * 0.01f);
            }
            auto cand = lloyd_max_1d(col.data(), n, K_, lloyd_iters, init);
            double mse = quantizer_mse(col.data(), n, cand);
            if (mse < best_mse) {
                best_mse = mse;
                best = std::move(cand);
            }
        }

        // Store levels for this dimension.
        std::memcpy(&levels_[static_cast<size_t>(d) * K_], best.data(),
                    K_ * sizeof(float));
    }

    compute_bounds_();
    compute_int8_levels_();
}

void ScalarLloydMaxQuantizer::compute_bounds_() {
    for (uint32_t d = 0; d < dim_; ++d) {
        const float* lv = &levels_[static_cast<size_t>(d) * K_];
        float* bd = &bounds_[static_cast<size_t>(d) * (K_ - 1)];
        for (uint32_t i = 0; i < K_ - 1; ++i)
            bd[i] = 0.5f * (lv[i] + lv[i + 1]);
    }
}

void ScalarLloydMaxQuantizer::compute_int8_levels_() {
    float level_max = 0.0f;
    for (float v : levels_)
        level_max = std::max(level_max, std::fabs(v));
    int8_scale_ = 127.0f / std::max(level_max, 1e-15f);
    levels_i8_.resize(levels_.size());
    for (size_t i = 0; i < levels_.size(); ++i)
        levels_i8_[i] = static_cast<int8_t>(
            std::lround(levels_[i] * int8_scale_));
}

// ---------------------------------------------------------------------------
// int8 query / decode-dot helpers
// ---------------------------------------------------------------------------

void ScalarLloydMaxQuantizer::build_query_i8(const float* query,
                                               int8_t* q_i8_out,
                                               float* scale_out) const {
    float qmax = 0.0f;
    for (uint32_t d = 0; d < dim_; ++d)
        qmax = std::max(qmax, std::fabs(query[d]));
    const float scale = 127.0f / std::max(qmax, 1e-15f);
    *scale_out = scale;
    for (uint32_t d = 0; d < dim_; ++d)
        q_i8_out[d] = static_cast<int8_t>(std::lround(query[d] * scale));
}

void ScalarLloydMaxQuantizer::decode_to_i8(const uint8_t* code,
                                             int8_t* out) const {
    const int8_t* li8 = levels_i8_.data();
    if (bits_ == 4) {
        for (uint32_t d = 0; d < dim_; d += 2) {
            const uint8_t byte = code[d / 2];
            out[d] = li8[static_cast<size_t>(d) * K_ + (byte & 0x0F)];
            if (d + 1 < dim_)
                out[d + 1] = li8[static_cast<size_t>(d + 1) * K_ +
                                 ((byte >> 4) & 0x0F)];
        }
    } else {
        for (uint32_t d = 0; d < dim_; ++d)
            out[d] = li8[static_cast<size_t>(d) * K_ + code[d]];
    }
}

// ---------------------------------------------------------------------------
// Encode / Decode
// ---------------------------------------------------------------------------

void ScalarLloydMaxQuantizer::encode(const float* vec, uint8_t* code_out) const {
    // For each dim, binary-search the nearest level, pack the index.
    if (bits_ == 4) {
        // 4-bit: pack 2 codes per byte.
        for (uint32_t d = 0; d < dim_; d += 2) {
            const float* bd = &bounds_[static_cast<size_t>(d) * (K_ - 1)];
            uint32_t idx0 = static_cast<uint32_t>(
                std::upper_bound(bd, bd + K_ - 1, vec[d]) - bd);
            uint8_t nib0 = static_cast<uint8_t>(idx0 & 0x0F);

            uint8_t nib1 = 0;
            if (d + 1 < dim_) {
                const float* bd1 = &bounds_[static_cast<size_t>(d + 1) * (K_ - 1)];
                uint32_t idx1 = static_cast<uint32_t>(
                    std::upper_bound(bd1, bd1 + K_ - 1, vec[d + 1]) - bd1);
                nib1 = static_cast<uint8_t>(idx1 & 0x0F);
            }
            code_out[d / 2] = nib0 | (nib1 << 4);
        }
    } else {
        // 8-bit: one byte per dim.
        for (uint32_t d = 0; d < dim_; ++d) {
            const float* bd = &bounds_[static_cast<size_t>(d) * (K_ - 1)];
            uint32_t idx = static_cast<uint32_t>(
                std::upper_bound(bd, bd + K_ - 1, vec[d]) - bd);
            code_out[d] = static_cast<uint8_t>(idx);
        }
    }
}

void ScalarLloydMaxQuantizer::decode(const uint8_t* code, float* vec_out) const {
    if (bits_ == 4) {
        for (uint32_t d = 0; d < dim_; d += 2) {
            uint8_t byte = code[d / 2];
            uint32_t idx0 = byte & 0x0F;
            vec_out[d] = levels_[static_cast<size_t>(d) * K_ + idx0];
            if (d + 1 < dim_) {
                uint32_t idx1 = (byte >> 4) & 0x0F;
                vec_out[d + 1] = levels_[static_cast<size_t>(d + 1) * K_ + idx1];
            }
        }
    } else {
        for (uint32_t d = 0; d < dim_; ++d) {
            vec_out[d] = levels_[static_cast<size_t>(d) * K_ + code[d]];
        }
    }
}

// ---------------------------------------------------------------------------
// LUT build + distance estimation
// ---------------------------------------------------------------------------

void ScalarLloydMaxQuantizer::build_float_lut(const float* query,
                                                float* lut_out) const {
    // For each dim d and level c:
    //   IP:  lut[d*K + c] = -query[d] * levels[d*K + c]  (negated → min-heap)
    //   L2sq: lut[d*K + c] = (query[d] - levels[d*K + c])^2
    if (metric_ == MetricKind::InnerProduct) {
        for (uint32_t d = 0; d < dim_; ++d) {
            const float q = query[d];
            const float* lv = &levels_[static_cast<size_t>(d) * K_];
            float* row = &lut_out[static_cast<size_t>(d) * K_];
            for (uint32_t c = 0; c < K_; ++c)
                row[c] = -q * lv[c];
        }
    } else {
        for (uint32_t d = 0; d < dim_; ++d) {
            const float q = query[d];
            const float* lv = &levels_[static_cast<size_t>(d) * K_];
            float* row = &lut_out[static_cast<size_t>(d) * K_];
            for (uint32_t c = 0; c < K_; ++c) {
                const float diff = q - lv[c];
                row[c] = diff * diff;
            }
        }
    }
}

void ScalarLloydMaxQuantizer::build_fastscan_lut4(const float* query,
                                                    uint8_t* lut4,
                                                    float* scale_out) const {
    if (bits_ != 4) {
        throw Error(ErrorCode::InvalidParam,
                    "build_fastscan_lut4 requires bits=4");
    }
    std::vector<float> lut_f32(static_cast<size_t>(dim_) * K_);
    std::vector<float> seg_min(dim_);
    build_float_lut(query, lut_f32.data());
    simd::quantize_lut_u8(lut_f32.data(), dim_, K_, lut4, scale_out,
                           seg_min.data());
}

void ScalarLloydMaxQuantizer::build_fastscan_lut(const float* query,
                                                   uint8_t* lut8,
                                                   float* scale_out,
                                                   float* offset_out) const {
    std::vector<float> lut_f32(static_cast<size_t>(dim_) * K_);
    std::vector<float> seg_min(dim_);
    build_float_lut(query, lut_f32.data());
    simd::quantize_lut_u8_scaled(lut_f32.data(), dim_, K_, lut8,
                                  scale_out, offset_out, seg_min.data());
}

float ScalarLloydMaxQuantizer::lut_distance(const uint8_t* code,
                                              const float* lut) const {
    float acc = 0.0f;
    if (bits_ == 4) {
        for (uint32_t d = 0; d < dim_; d += 2) {
            uint8_t byte = code[d / 2];
            acc += lut[static_cast<size_t>(d) * K_ + (byte & 0x0F)];
            if (d + 1 < dim_)
                acc += lut[static_cast<size_t>(d + 1) * K_ + ((byte >> 4) & 0x0F)];
        }
    } else {
        for (uint32_t d = 0; d < dim_; ++d)
            acc += lut[static_cast<size_t>(d) * K_ + code[d]];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Serialize / Deserialize
// ---------------------------------------------------------------------------

// Layout:
//   {magic:u32, metric:u8, bits:u8, dim:u32 LE, levels:float32[dim*K]}
static constexpr uint32_t kScalarLloydMaxMagic = 0x534C4D58;  // "SLMX"

void ScalarLloydMaxQuantizer::serialize(std::vector<uint8_t>& out) const {
    const size_t header = 4 + 1 + 1 + 4;
    const size_t levels_bytes = levels_.size() * sizeof(float);
    out.resize(header + levels_bytes);
    uint8_t* ptr = out.data();
    std::memcpy(ptr, &kScalarLloydMaxMagic, 4);
    ptr[4] = static_cast<uint8_t>(metric_);
    ptr[5] = bits_;
    std::memcpy(ptr + 6, &dim_, sizeof(dim_));
    std::memcpy(ptr + header, levels_.data(), levels_bytes);
}

void ScalarLloydMaxQuantizer::deserialize(const uint8_t* in, size_t size) {
    const size_t header = 4 + 1 + 1 + 4;
    if (size < header) {
        throw Error(ErrorCode::CorruptIndex,
                    "ScalarLloydMax deserialize: blob too small");
    }
    uint32_t magic;
    std::memcpy(&magic, in, 4);
    if (magic != kScalarLloydMaxMagic) {
        throw Error(ErrorCode::CorruptIndex,
                    "ScalarLloydMax deserialize: magic mismatch");
    }
    metric_ = static_cast<MetricKind>(in[4]);
    bits_ = in[5];
    std::memcpy(&dim_, in + 6, sizeof(dim_));
    K_ = 1u << bits_;
    if (bits_ != 4 && bits_ != 8) {
        throw Error(ErrorCode::CorruptIndex,
                    "ScalarLloydMax deserialize: invalid bits");
    }
    const size_t levels_floats = static_cast<size_t>(dim_) * K_;
    const size_t levels_bytes = levels_floats * sizeof(float);
    if (size < header + levels_bytes) {
        throw Error(ErrorCode::CorruptIndex,
                    "ScalarLloydMax deserialize: levels truncated");
    }
    levels_.resize(levels_floats);
    std::memcpy(levels_.data(), in + header, levels_bytes);
    compute_bounds_();
    compute_int8_levels_();
}

}  // namespace sextant
