#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pq_quantizer.hpp"
#include "sextant/types.hpp"

namespace sextant {

class RaBitQQuantizer : public PqQuantizer {
public:
    RaBitQQuantizer(MetricKind metric, Dim dim, uint64_t seed = 0xC0DE1234ULL);

    static constexpr uint32_t kQueryBits = 4;
    static constexpr uint32_t kQueryMaxCode = 15;
    static constexpr float kConstEpsilon = 1.9f;
    static constexpr float kXuCbNormSqr = 0.25f;

    void train(const float* samples, uint64_t n) override;
    void encode(const float* vec, uint8_t* code_out) const override;
    void encode_with_centroid(const float* vec, const float* centroid,
                             uint8_t* code_out) const;

    void preprocess_query(const float* query, float* out) const override;
    void preprocess_query_as(MetricKind metric, const float* query,
                             float* out) const override;
    void preprocess_query_with_centroid(const float* query,
                                        const float* centroid, float* out) const;

    void build_fastscan_lut4_with_centroid(const float* query,
                                           const float* centroid,
                                           uint8_t* lut4,
                                           float* scale_out) const;

    /// Map a raw uint4 FastScan scan result back to the true float sign-dot.
    /// Requires build_fastscan_lut4_with_centroid to have been called for the
    /// current shard (sets lut_scale_ and seg_min_sum_).
    float dequantize_scan_result(float raw_scan_result) const {
        return (lut_scale_ > 0.0f)
                   ? raw_scan_result / lut_scale_ + seg_min_sum_
                   : raw_scan_result;
    }

    /// Finalize a raw FastScan result into an estimated L2sq distance. The
    /// `factors` pointer addresses the 2 per-vector floats stored in the
    /// `.factors` sidecar: factors[0] = dp_multiplier, factors[1] =
    /// or_minus_c_l2sqr. Requires build_fastscan_lut4_with_centroid to have
    /// been called for this shard (sets c34_ and qr_to_c_l2sqr_).
    float finalize_distance(float raw_scan_result, const float* factors) const;
    float finalize_distance_ip(float raw_scan_result, const float* factors) const;

    /// Per-vector distance error bound for selective rerank (Phase 3). Derived
    /// Per-query state for RaBitQ distance finalization. Stored per-thread
    /// (in IVFScanWorkerState) to avoid races on the shared quantizer.
    struct QueryState {
        float qr_to_c_l2sqr = 0;
        float c34 = 0;
        float lut_scale = 1;
        float seg_min_sum = 0;
    };

    /// Build LUT and populate query state. Thread-safe: writes only to `qs`.
    void build_lut4_with_state(const float* query, const float* centroid,
                               uint8_t* lut4, QueryState& qs) const;

    /// Dequantize + finalize using per-thread query state. Thread-safe.
    float dequant_and_finalize(uint32_t raw_uint4, const float* factors,
                               const QueryState& qs) const;

    /// Error bound using per-thread query state.
    float error_bound(const float* factors, const QueryState& qs) const;

    /// from the stored factors (dp_multiplier, or_minus_c_l2sqr) so it can be
    /// computed at search time without the original vector.
    float get_error_bound(const float* factors) const;

    void decode_code(const uint8_t* code, float* out) const override;

    void serialize(std::vector<uint8_t>& out) const override;
    void deserialize(const uint8_t* in, size_t size) override;

    static constexpr uint8_t kMagic = 0xB4;

    uint32_t padded_dim() const { return padded_dim_; }
    float get_padded_dim_sqrt() const { return padded_dim_sqrt_; }
    uint32_t factors_offset() const { return sign_code_bytes_; }
    uint32_t full_code_size() const { return sign_code_bytes_ + 2 * sizeof(float); }

    float get_query_factor(const std::string& name) const;

    void fwht_inplace(float* buf, size_t n) const;
    void rotate(const float* in, float* out) const;

private:
    uint32_t padded_dim_;
    float padded_dim_sqrt_;
    float dim_sqrt_;
    uint32_t sign_code_bytes_;
    std::vector<float> signs_;

    mutable float qr_to_c_l2sqr_ = 0.0f;
    mutable float c1_ = 0.0f;
    mutable float c2_ = 0.0f;
    mutable float c34_ = 0.0f;
    // FastScan dequantization params, set by build_fastscan_lut4_with_centroid.
    // The uint4 kernel stores (lut - seg_min) * A and sums over segments, so
    // the raw scan result is A*(sign_dot - seg_min_sum). dequantize_scan_result
    // divides by lut_scale_ and adds seg_min_sum_ to recover the true sign-dot.
    mutable float lut_scale_ = 1.0f;
    mutable float seg_min_sum_ = 0.0f;

    void generate_signs_();
};

}  // namespace sextant
