// PQ quantizer — stub implementation.
// Full port from duckdb-vector-index with NumKong adaptation dispatched separately.

#include "pq_quantizer.hpp"
#include "sextant/error.hpp"

#include <numkong/numkong.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace sextant {

namespace {

/// L2 squared distance via NumKong.
inline float l2sq_f32(const float* a, const float* b, uint32_t dim) {
    nk_f64_t result = 0;
    nk_sqeuclidean_f32(reinterpret_cast<const nk_f32_t*>(a),
                       reinterpret_cast<const nk_f32_t*>(b),
                       static_cast<nk_size_t>(dim), &result);
    return static_cast<float>(result);
}

/// Dot product via NumKong.
inline float dot_f32(const float* a, const float* b, uint32_t dim) {
    nk_f64_t result = 0;
    nk_dot_f32(reinterpret_cast<const nk_f32_t*>(a),
               reinterpret_cast<const nk_f32_t*>(b),
               static_cast<nk_size_t>(dim), &result);
    return static_cast<float>(result);
}

}  // namespace

PqQuantizer::PqQuantizer(MetricKind metric, Dim dim, uint8_t m, uint8_t bits,
                         uint64_t seed)
    : metric_(metric), dim_(dim), m_(m), bits_(bits), seed_(seed) {
    if (m_ == 0) {
        throw Error(ErrorCode::InvalidParam, "PQ 'm' must be > 0");
    }
    if (dim_ % static_cast<Dim>(m_) != 0) {
        throw Error(ErrorCode::InvalidParam, "PQ requires dim divisible by m");
    }
    if (bits_ != 4 && bits_ != 8) {
        throw Error(ErrorCode::InvalidParam, "PQ 'bits' must be 4 or 8");
    }

    K_ = 1u << bits_;
    sub_dim_ = dim_ / m_;
    codebook_.resize(static_cast<size_t>(m_) * K_ * sub_dim_, 0.0f);
}

uint32_t PqQuantizer::code_size() const {
    return (static_cast<uint32_t>(m_) * bits_ + 7) / 8;
}

uint32_t PqQuantizer::lut_size() const {
    return static_cast<uint32_t>(m_) * K_;
}

void PqQuantizer::train(const float* /*samples*/, uint64_t /*n*/) {
    // TODO: batch k-means++ port from duckdb-vector-index.
}

void PqQuantizer::build_cross_distance_table() {
    cross_distance_table_.resize(
        static_cast<size_t>(m_) * K_ * K_, 0.0f);
    // TODO: populate.
}

void PqQuantizer::encode(const float* /*vec*/, uint8_t* /*code_out*/) const {
    // TODO: port encode logic.
}

void PqQuantizer::preprocess_query(const float* /*query*/,
                                    float* /*out*/) const {
    // TODO: port PreprocessQuery.
}

float PqQuantizer::lut_distance(const uint8_t* /*code*/,
                                 const float* /*lut*/) const {
    return 0.0f;  // TODO
}

bool PqQuantizer::build_code_lut(const uint8_t* /*code*/,
                                  float* /*out*/) const {
    return false;  // TODO
}

float PqQuantizer::code_distance(const uint8_t* /*code_a*/,
                                  const uint8_t* /*code_b*/) const {
    return 0.0f;  // TODO
}

void PqQuantizer::serialize(std::vector<uint8_t>& /*out*/) const {
    // TODO
}

void PqQuantizer::deserialize(const uint8_t* /*in*/, size_t /*size*/) {
    // TODO
}

}  // namespace sextant
