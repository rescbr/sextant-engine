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

        // Reseed empty levels to the data point with the max quantization
        // error. A dead level kept at its stale position wastes code space
        // forever (classic k-means empty-cluster repair). Placing it at the
        // worst-served point directly attacks the largest error term.
        for (uint32_t k = 0; k < num_levels; ++k) {
            if (counts[k] > 0) continue;
            float worst_err = -1.0f;
            float worst_val = levels[k];
            for (uint64_t i = 0; i < n; ++i) {
                const uint32_t idx = static_cast<uint32_t>(
                    std::upper_bound(bounds.begin(), bounds.end(),
                                     data[i]) - bounds.begin());
                const float err = std::fabs(data[i] - levels[idx]);
                if (err > worst_err) {
                    worst_err = err;
                    worst_val = data[i];
                }
            }
            levels[k] = worst_val;
            // Re-sort so bounds stay consistent for the next iteration.
            std::sort(levels.begin(), levels.end());
            break;  // recompute bounds with the reseeded level
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

        // Sort col FIRST — quantile inits below require ordered data.
        std::sort(col.begin(), col.end());

        // Init 1: uniform quantiles (best default).
        std::vector<float> quantile_init(K_);
        for (uint32_t k = 0; k < K_; ++k) {
            double frac = static_cast<double>(k) / (K_ - 1);
            uint64_t idx = static_cast<uint64_t>(frac * (n - 1));
            quantile_init[k] = col[idx];
        }

        auto best = lloyd_max_1d(col.data(), n, K_, lloyd_iters, quantile_init);
        double best_mse = quantizer_mse(col.data(), n, best);

        // Restarts with DIVERSE inits: K uniformly-spaced random data points.
        // (The old jittered-quantile init perturbed by ±0.1% of range — every
        // restart converged to the same local minimum, wasting the restart.)
        // Sampling actual data points gives genuinely different basins,
        // matching the spirit of sklearn's n_init with random inits.
        std::mt19937 rng(42 + d);
        for (uint32_t r = 1; r < n_restarts; ++r) {
            std::vector<float> init(K_);
            // Spaced random positions: level k draws from the k-th slice of
            // the sorted data. Spacing keeps the init sorted-ish (diverse but
            // not degenerate) and covers the whole distribution.
            for (uint32_t k = 0; k < K_; ++k) {
                std::uniform_real_distribution<double> u(
                    static_cast<double>(k) / K_,
                    static_cast<double>(k + 1) / K_);
                uint64_t idx = static_cast<uint64_t>(u(rng) * (n - 1));
                init[k] = col[idx];
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
}

void ScalarLloydMaxQuantizer::train_uniform(const float* samples, uint64_t n) {
    if (n < 2) {
        throw Error(ErrorCode::InvalidParam,
                    "ScalarLloydMaxQuantizer::train_uniform: need n >= 2");
    }
    for (uint32_t d = 0; d < dim_; ++d) {
        float mn = samples[d], mx = samples[d];
        for (uint64_t i = 1; i < n; ++i) {
            const float x = samples[i * dim_ + d];
            mn = std::min(mn, x);
            mx = std::max(mx, x);
        }
        float* lv = &levels_[static_cast<size_t>(d) * K_];
        const float step = K_ > 1 ? (mx - mn) / (K_ - 1) : 0.f;
        for (uint32_t k = 0; k < K_; ++k) lv[k] = mn + step * k;
    }
    mode_ = 1;
    mode_ = 1;
    // steps_ for the arithmetic scan: f = identity, step = level spacing.
    steps_.assign(dim_, 0.f);
    for (uint32_t d = 0; d < dim_; ++d) {
        const float* lv = &levels_[static_cast<size_t>(d) * K_];
        steps_[d] = K_ > 1 ? lv[1] - lv[0] : 0.f;
    }
    compute_bounds_();
}

void ScalarLloydMaxQuantizer::train_shape(const float* samples, uint64_t n,
                                          uint32_t n_restarts,
                                          uint32_t lloyd_iters) {
    // Stage 1: full Lloyd-Max training (levels_ = per-dim MSE-optimal).
    train(samples, n, n_restarts, lloyd_iters);

    // Stage 2: alternating LS factorization  lm_d[k] ≈ lo_d + step_d·f[k].
    // Init f = linear; iterate (per-dim (lo,step) fit) <-> (f = mean_d
    // (lm_d[k]-lo_d)/step_d). Converges in <8 iterations (measured).
    std::vector<float> f(K_);
    for (uint32_t k = 0; k < K_; ++k) f[k] = static_cast<float>(k);
    std::vector<float> lo(dim_), st(dim_);
    for (uint32_t it = 0; it < 8; ++it) {
        for (uint32_t d = 0; d < dim_; ++d) {
            const float* lv = &levels_[static_cast<size_t>(d) * K_];
            double sf = 0, sf2 = 0, s1 = 0, sfx = 0;
            for (uint32_t k = 0; k < K_; ++k) {
                sf += f[k];
                sf2 += static_cast<double>(f[k]) * f[k];
                s1 += lv[k];
                sfx += static_cast<double>(f[k]) * lv[k];
            }
            const double denom = K_ * sf2 - sf * sf;
            const double step = denom != 0 ? (K_ * sfx - sf * s1) / denom : 0.0;
            st[d] = static_cast<float>(step);
            lo[d] = static_cast<float>((s1 - step * sf) / K_);
        }
        std::vector<double> fn(K_, 0.0);
        for (uint32_t d = 0; d < dim_; ++d) {
            const float* lv = &levels_[static_cast<size_t>(d) * K_];
            for (uint32_t k = 0; k < K_; ++k)
                fn[k] += (lv[k] - lo[d]) / (st[d] + 1e-30f);
        }
        for (uint32_t k = 0; k < K_; ++k) f[k] = static_cast<float>(fn[k] / dim_);
        for (uint32_t k = 1; k < K_; ++k)  // monotone (defensive)
            if (f[k] < f[k - 1]) f[k] = f[k - 1];
    }

    // Stage 3: rebuild levels_ from the factorization (the shape class is
    // the constraint — encode/decode/rerank see the shared-shape levels),
    // fill steps_ + the u8 TBL table.
    float fmin = f[0], fmax = f[0];
    for (uint32_t k = 0; k < K_; ++k) {
        fmin = std::min(fmin, f[k]);
        fmax = std::max(fmax, f[k]);
    }
    f_ = f;
    steps_.assign(dim_, 0.f);
    fu8_.assign(K_, 0);
    // Affine map f -> [0, 255] for the TBL kernel. The constant shift is
    // ranking-invariant (it adds shift·Σa_d to every dot); the 255 scale
    // likewise. Do NOT map signed f directly into u8 — it wraps.
    const float S = fmax > fmin ? 255.f / (fmax - fmin) : 1.f;
    for (uint32_t k = 0; k < K_; ++k)
        fu8_[k] = static_cast<uint8_t>(
            std::lround((f[k] - fmin) * S));
    for (uint32_t d = 0; d < dim_; ++d) {
        float* lv = &levels_[static_cast<size_t>(d) * K_];
        for (uint32_t k = 0; k < K_; ++k)
            lv[k] = lo[d] + st[d] * f[k];
        steps_[d] = st[d];
    }
    mode_ = 2;
    compute_bounds_();
}

void ScalarLloydMaxQuantizer::compute_bounds_() {
    for (uint32_t d = 0; d < dim_; ++d) {
        const float* lv = &levels_[static_cast<size_t>(d) * K_];
        float* bd = &bounds_[static_cast<size_t>(d) * (K_ - 1)];
        for (uint32_t i = 0; i < K_ - 1; ++i)
            bd[i] = 0.5f * (lv[i] + lv[i + 1]);
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
    // Trailing tail: [mode byte] (+ for mode 2: f[K] + steps[dim] floats).
    // Absent tail (pre-uniform blobs) => mode 0.
    size_t tail = 1;
    if (mode_ == 2) tail += (f_.size() + steps_.size()) * sizeof(float);
    out.resize(header + levels_bytes + tail);
    uint8_t* ptr = out.data();
    std::memcpy(ptr, &kScalarLloydMaxMagic, 4);
    ptr[4] = static_cast<uint8_t>(metric_);
    ptr[5] = bits_;
    std::memcpy(ptr + 6, &dim_, sizeof(dim_));
    std::memcpy(ptr + header, levels_.data(), levels_bytes);
    ptr[header + levels_bytes] = mode_;
    if (mode_ == 2) {
        uint8_t* t = ptr + header + levels_bytes + 1;
        std::memcpy(t, f_.data(), f_.size() * sizeof(float));
        t += f_.size() * sizeof(float);
        std::memcpy(t, steps_.data(), steps_.size() * sizeof(float));
    }
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
    const size_t tail_off = header + levels_bytes;
    if (size == tail_off) {
        mode_ = 0;  // pre-uniform blob
    } else {
        mode_ = in[tail_off];
        if (mode_ > 2)
            throw Error(ErrorCode::CorruptIndex,
                        "ScalarLloydMax deserialize: invalid mode");
        if (mode_ == 2) {
            const size_t need = tail_off + 1 +
                (static_cast<size_t>(K_) + dim_) * sizeof(float);
            if (size < need)
                throw Error(ErrorCode::CorruptIndex,
                            "ScalarLloydMax deserialize: shape tail truncated");
            const uint8_t* t = in + tail_off + 1;
            f_.resize(K_);
            std::memcpy(f_.data(), t, K_ * sizeof(float));
            t += K_ * sizeof(float);
            steps_.resize(dim_);
            std::memcpy(steps_.data(), t, dim_ * sizeof(float));
            float fmin = f_[0], fmax = f_[0];
            for (uint32_t k = 0; k < K_; ++k) {
                fmin = std::min(fmin, f_[k]);
                fmax = std::max(fmax, f_[k]);
            }
            const float S = fmax > fmin ? 255.f / (fmax - fmin) : 1.f;
            fu8_.assign(K_, 0);
            for (uint32_t k = 0; k < K_; ++k)
                fu8_[k] = static_cast<uint8_t>(
                    std::lround((f_[k] - fmin) * S));
        } else if (mode_ == 1) {
            steps_.assign(dim_, 0.f);
            for (uint32_t d = 0; d < dim_; ++d) {
                const float* lv = &levels_[static_cast<size_t>(d) * K_];
                steps_[d] = K_ > 1 ? lv[1] - lv[0] : 0.f;
            }
        }
    }
    compute_bounds_();
}

}  // namespace sextant
