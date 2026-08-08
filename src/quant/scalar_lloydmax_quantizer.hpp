#pragma once

/// @file scalar_lloydmax_quantizer.hpp
/// Per-dimension Lloyd-Max (optimal 1D scalar) quantization.
///
/// Each dimension gets its own K-level quantizer trained via 1D Lloyd-Max
/// (optimal scalar quantization). Codes are centroid-independent: they're
/// computed against global per-dim levels, so inserts/deletes don't
/// invalidate existing codes.
///
/// Structurally identical to PQ with sub_dim=1: m = dim subquantizers, each
/// with K = 2^bits levels. Uses the same FastScan infrastructure (lut4 format,
/// pq4_block32 kernel) as PQ4.

#include <sextant/types.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace sextant {

class ScalarLloydMaxQuantizer {
public:
    ScalarLloydMaxQuantizer(MetricKind metric, Dim dim, uint8_t bits = 4);

    /// Train per-dim levels from a sample (n × dim floats, row-major).
    /// Runs 1D Lloyd-Max per dimension with quantile init + restarts.
    void train(const float* samples, uint64_t n,
               uint32_t n_restarts = 3, uint32_t lloyd_iters = 30);

    /// Encode: for each dim, assign to nearest level → pack nibbles.
    /// Output: dim * bits / 8 bytes (code_size()).
    void encode(const float* vec, uint8_t* code_out) const;

    /// Decode: for each dim, lookup level value from packed nibbles.
    void decode(const uint8_t* code, float* vec_out) const;

    /// Build the float LUT for ADC distance estimation.
    /// lut_out: dim * K floats.
    /// For IP: lut[d*K + c] = -query[d] * levels[d*K + c]
    /// For L2sq: lut[d*K + c] = (query[d] - levels[d*K + c])^2
    void build_float_lut(const float* query, float* lut_out) const;

    /// Build the FastScan LUT for 4-bit codes (m × K bytes, uint8).
    /// Internally computes float LUT then calls simd::quantize_lut_u8.
    void build_fastscan_lut4(const float* query, uint8_t* lut4,
                             float* scale_out) const;

    /// Build the FastScan LUT for 8-bit codes (m × K bytes, uint8).
    void build_fastscan_lut(const float* query, uint8_t* lut8,
                            float* scale_out, float* offset_out) const;

    /// LUT distance from a float LUT (for rerank or testing).
    float lut_distance(const uint8_t* code, const float* lut) const;

    uint32_t fastscan_lut_bytes() const { return dim_ * K_; }
    uint32_t code_size() const;
    uint32_t K() const { return K_; }
    uint32_t m() const { return dim_; }
    MetricKind metric() const { return metric_; }
    Dim dim() const { return dim_; }
    uint8_t bits() const { return bits_; }

    const float* levels() const { return levels_.data(); }

    /// Build the int8 query for the decode-dot kernels.
    /// Scales query so its max-abs value maps to ±127.
    /// q_i8_out: dim int8 values. scale_out: the scale factor used.
    void build_query_i8(const float* query, int8_t* q_i8_out,
                        float* scale_out) const;

    /// Decode a packed code into int8 levels (pre-scaled).
    /// out: dim int8 values (one per dimension, looked up from levels_i8_).
    void decode_to_i8(const uint8_t* code, int8_t* out) const;

    /// Pre-scaled int8 levels table: dim × K int8 values.
    const int8_t* levels_i8() const { return levels_i8_.data(); }
    float int8_scale() const { return int8_scale_; }

    void serialize(std::vector<uint8_t>& out) const;
    void deserialize(const uint8_t* in, size_t size);

private:
    MetricKind metric_;
    Dim dim_;
    uint8_t bits_;
    uint32_t K_;

    /// Per-dim quantization levels: dim × K floats (sorted ascending per dim).
    std::vector<float> levels_;   // [dim * K]

    /// Pre-scaled int8 levels for the decode-dot kernels: dim × K int8 values.
    /// Populated by compute_int8_levels_() after train() / deserialize().
    std::vector<int8_t> levels_i8_;  // [dim * K]
    float int8_scale_ = 0.0f;

    /// Per-dim boundaries for fast assignment: dim × (K-1) floats.
    /// boundary[d*(K-1) + i] = midpoint between levels[d*K+i] and levels[d*K+i+1].
    std::vector<float> bounds_;   // [dim * (K-1)]

    /// Recompute bounds_ from levels_.
    void compute_bounds_();

    /// Recompute levels_i8_ + int8_scale_ from levels_.
    void compute_int8_levels_();
};

}  // namespace sextant
