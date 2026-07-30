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

    float finalize_distance(float raw_scan_result, const uint8_t* code,
                            const float* centroid) const;
    float finalize_distance_ip(float raw_scan_result, const uint8_t* code,
                               const float* centroid) const;

    float get_error_bound(const uint8_t* code, const float* centroid) const;

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
    uint32_t sign_code_bytes_;
    std::vector<float> signs_;

    mutable float qr_to_c_l2sqr_ = 0.0f;
    mutable float c1_ = 0.0f;
    mutable float c2_ = 0.0f;
    mutable float c34_ = 0.0f;

    void generate_signs_();
};

}  // namespace sextant
