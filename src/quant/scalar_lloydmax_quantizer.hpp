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
    /// Default restarts = 10 (matches sklearn k-means n_init=10). Each restart
    /// uses spaced-random data-point inits for basin diversity.
    void train(const float* samples, uint64_t n,
               uint32_t n_restarts = 10, uint32_t lloyd_iters = 30);

    /// Train per-dim EQUIDISTANT levels from the per-dim [min, max].
    /// At 8 bits this matches (slightly beats) Lloyd-Max recall — level
    /// placement is irrelevant once K is large — and it enables the
    /// arithmetic scan: dot = Σq_d·lo_d + Σ(q_d·step_d)·code_d, a pure MAC
    /// over sequential codes with no levels-table gather (measured 1.86x
    /// kernel speedup over the gather form at 4 bits, scalar code).
    void train_uniform(const float* samples, uint64_t n);

    /// Shared-shape training (see the arithmetic_scan docs). Lloyd-Max +
    /// alternating LS factorization; levels_ are rebuilt from the fitted
    /// (lo_d, step_d, f) so encode/decode/rerank are self-consistent.
    void train_shape(const float* samples, uint64_t n,
                     uint32_t n_restarts = 10, uint32_t lloyd_iters = 30);

    /// Shared-shape companding: level_d[k] = lo_d + step_d·f[k] with a single
    /// monotone f shared across dims. Train = Lloyd-Max, then an alternating
    /// least-squares factorization of the LM level table into per-dim
    /// (lo_d, step_d) + shared f. Recall matches Lloyd-Max on near-Gaussian
    /// marginals (arxiv) at the uniform kernel's speed: the scan stays
    /// dot = c0 + Σ (q_d·step_d)·f[code_d] — f is 16 bytes, one NEON TBL per
    /// 16 nibbles, no per-dim gather.

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

    /// True when trained via train_uniform (equidistant levels). Serialized
    /// with the quantizer so the scan can pick the arithmetic kernel.
    bool is_uniform() const { return mode_ == 1; }

    /// True when the arithmetic MAC scan applies (uniform or shared-shape):
    /// dot = c0 + Σ (q_d·step_d)·f[code_d] with f == identity for uniform.
    bool arithmetic_scan() const { return mode_ >= 1; }

    /// Shared-shape table f (K floats, monotone). Identity for uniform/LM.
    const float* shape() const { return f_.data(); }

    /// Per-dim scan scale (q_d·step_d precomputed per query). For uniform,
    /// step_d = level[d*K+1]-level[d*K].
    const float* steps() const { return steps_.data(); }

    /// f quantized to u8 for the NEON TBL kernel (scale folded into steps by
    /// the caller: fu8[k] = round(f[k]·S), multiply the accumulated dot by 1/S
    /// — ranking-invariant, so the scan can skip the rescale entirely).
    const uint8_t* shape_u8() const { return fu8_.data(); }

    void serialize(std::vector<uint8_t>& out) const;
    void deserialize(const uint8_t* in, size_t size);

private:
    MetricKind metric_;
    Dim dim_;
    uint8_t bits_;
    uint32_t K_;

    /// Per-dim quantization levels: dim × K floats (sorted ascending per dim).
    std::vector<float> levels_;   // [dim * K]

    /// Per-dim boundaries for fast assignment: dim × (K-1) floats.
    /// boundary[d*(K-1) + i] = midpoint between levels[d*K+i] and levels[d*K+i+1].
    std::vector<float> bounds_;   // [dim * (K-1)]

    /// Equidistant-levels mode (train_uniform). Levels stay in levels_
    /// (same layout), so encode/decode/serialize are shared.
    /// 0 = Lloyd-Max, 1 = uniform (linear f), 2 = shared-shape f.
    uint8_t mode_ = 0;

    /// Shared shape + per-dim scan scales (mode 2; steps_ also filled for
    /// mode 1). f_ is K floats; steps_ is dim floats.
    std::vector<float> f_;
    std::vector<float> steps_;
    std::vector<uint8_t> fu8_;  // f scaled to u8 for the TBL kernel

    /// Recompute bounds_ from levels_.
    void compute_bounds_();
};

}  // namespace sextant
