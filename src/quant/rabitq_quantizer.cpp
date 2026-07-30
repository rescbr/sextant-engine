#include "rabitq_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include "sextant/error.hpp"
#include "simd_kernels.hpp"

namespace sextant {

namespace {

inline void write_code(uint8_t* code, uint8_t bits, uint32_t s, uint32_t cid) {
    const uint32_t byte_off = s / 2;
    const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
    const uint8_t nibble = static_cast<uint8_t>(cid & 0x0Fu);
    code[byte_off] = static_cast<uint8_t>(
        (code[byte_off] & ~(0x0Fu << shift)) | (nibble << shift));
}

inline void write_subset_sum_lut(float* out, float base, float v0, float v1,
                                 float v2, float v3) {
    out[0] = base;
    out[1] = base + v0;
    out[2] = base + v1;
    out[3] = base + v0 + v1;
    out[4] = base + v2;
    out[5] = base + v0 + v2;
    out[6] = base + v1 + v2;
    out[7] = base + v0 + v1 + v2;
    out[8] = base + v3;
    out[9] = base + v0 + v3;
    out[10] = base + v1 + v3;
    out[11] = base + v0 + v1 + v3;
    out[12] = base + v2 + v3;
    out[13] = base + v0 + v2 + v3;
    out[14] = base + v1 + v2 + v3;
    out[15] = base + v0 + v1 + v2 + v3;
}

inline void write_factor(uint8_t* code, uint32_t off, uint32_t idx, float v) {
    std::memcpy(code + off + idx * sizeof(float), &v, sizeof(float));
}

}  // namespace

RaBitQQuantizer::RaBitQQuantizer(MetricKind metric, Dim dim, uint64_t seed)
    : PqQuantizer(metric, dim, dim / 4, 4, seed) {
    if (dim == 0) {
        throw Error(ErrorCode::InvalidParam, "RaBitQ dim must be > 0");
    }
    if (dim % 4 != 0) {
        throw Error(ErrorCode::InvalidParam, "RaBitQ dim must be divisible by 4");
    }
    uint32_t p = 1;
    while (p < dim) p <<= 1;
    padded_dim_ = p;
    padded_dim_sqrt_ = std::sqrt(static_cast<float>(p));
    dim_sqrt_ = std::sqrt(static_cast<float>(dim_));
    sign_code_bytes_ = (m_ * bits_ + 7) / 8;
    generate_signs_();
}

void RaBitQQuantizer::generate_signs_() {
    signs_.assign(3 * padded_dim_, 0.0f);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed_));
    std::uniform_int_distribution<int> dist(0, 1);
    for (uint32_t r = 0; r < 3; r++) {
        for (uint32_t i = 0; i < padded_dim_; i++) {
            signs_[r * padded_dim_ + i] = (dist(rng) == 0) ? -1.0f : 1.0f;
        }
    }
}

void RaBitQQuantizer::fwht_inplace(float* buf, size_t n) const {
    for (size_t step = 1; step < n; step *= 2) {
        for (size_t i = 0; i < n; i += step * 2) {
            for (size_t j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j] = a + b;
                buf[j + step] = a - b;
            }
        }
    }
}

void RaBitQQuantizer::rotate(const float* in, float* out) const {
    const uint32_t p = padded_dim_;
    std::vector<float> buf(p, 0.0f);
    for (uint32_t i = 0; i < dim_; i++) buf[i] = in[i];
    for (uint32_t r = 0; r < 3; r++) {
        const float* s = signs_.data() + r * p;
        for (uint32_t i = 0; i < p; i++) buf[i] *= s[i];
        fwht_inplace(buf.data(), p);
    }
    const float norm = 1.0f / (static_cast<float>(p) * padded_dim_sqrt_);
    for (uint32_t i = 0; i < p; i++) out[i] = buf[i] * norm;
}

void RaBitQQuantizer::train(const float* samples, uint64_t n) {
    (void)samples;
    (void)n;
}

void RaBitQQuantizer::encode(const float* vec, uint8_t* code_out) const {
    std::vector<float> zero(dim_, 0.0f);
    encode_with_centroid(vec, zero.data(), code_out);
}

void RaBitQQuantizer::encode_with_centroid(const float* vec,
                                           const float* centroid,
                                           uint8_t* code_out) const {
    std::memset(code_out, 0, full_code_size());
    std::vector<float> residual(dim_);
    float norm_l2sqr = 0.0f;
    float or_l2sqr = 0.0f;
    float dp_oO = 0.0f;
    for (uint32_t i = 0; i < dim_; i++) {
        const float r = vec[i] - centroid[i];
        residual[i] = r;
        norm_l2sqr += r * r;
        or_l2sqr += vec[i] * vec[i];
        dp_oO += std::fabs(r);
    }
    std::vector<float> rotated(padded_dim_, 0.0f);
    rotate(residual.data(), rotated.data());
    for (uint32_t group = 0; group < m_; group++) {
        uint32_t code = 0;
        for (uint32_t b = 0; b < 4; b++) {
            if (rotated[group * 4 + b] > 0.0f) code |= (1u << b);
        }
        write_code(code_out, bits_, group, code);
    }
    const float dp_multiplier =
        (dp_oO > 0.0f) ? norm_l2sqr * dim_sqrt_ / dp_oO : 0.0f;
    const float or_minus_c_l2sqr =
        (metric_ == MetricKind::InnerProduct) ? (norm_l2sqr - or_l2sqr)
                                              : norm_l2sqr;
    write_factor(code_out, sign_code_bytes_, 0, dp_multiplier);
    write_factor(code_out, sign_code_bytes_, 1, or_minus_c_l2sqr);
}

void RaBitQQuantizer::preprocess_query(const float* query, float* out) const {
    std::vector<float> zero(dim_, 0.0f);
    preprocess_query_with_centroid(query, zero.data(), out);
}

void RaBitQQuantizer::preprocess_query_as(MetricKind metric, const float* query,
                                          float* out) const {
    (void)metric;
    preprocess_query(query, out);
}

void RaBitQQuantizer::preprocess_query_with_centroid(
    const float* query, const float* centroid, float* out) const {
    std::vector<float> rq(dim_);
    for (uint32_t i = 0; i < dim_; i++) {
        rq[i] = query[i] - centroid[i];
    }
    qr_to_c_l2sqr_ = simd::dot_f32(rq.data(), rq.data(), dim_);
    std::vector<float> rotated_q(padded_dim_, 0.0f);
    rotate(rq.data(), rotated_q.data());

    float v_min = rotated_q[0];
    float v_max = rotated_q[0];
    for (uint32_t i = 0; i < dim_; i++) {
        if (rotated_q[i] < v_min) v_min = rotated_q[i];
        if (rotated_q[i] > v_max) v_max = rotated_q[i];
    }
    const float span = v_max - v_min;
    const float delta = span / static_cast<float>(kQueryMaxCode);
    c1_ = 2.0f * delta / dim_sqrt_;
    c2_ = 2.0f * v_min / dim_sqrt_;
    std::vector<float> qq(dim_);
    float sum_qq = 0.0f;
    for (uint32_t i = 0; i < dim_; i++) {
        qq[i] = (span > 0.0f) ? std::round((rotated_q[i] - v_min) / delta) : 0.0f;
        sum_qq += qq[i];
    }
    c34_ = (delta * sum_qq + static_cast<float>(dim_) * v_min) / dim_sqrt_;

    for (uint32_t m = 0; m < m_; m++) {
        const uint32_t ds = m * 4;
        const float v0 = c1_ * qq[ds + 0] + c2_;
        const float v1 = c1_ * qq[ds + 1] + c2_;
        const float v2 = c1_ * qq[ds + 2] + c2_;
        const float v3 = c1_ * qq[ds + 3] + c2_;
        write_subset_sum_lut(out + m * 16, 0.0f, v0, v1, v2, v3);
    }
}

void RaBitQQuantizer::build_fastscan_lut4_with_centroid(
    const float* query, const float* centroid, uint8_t* lut4,
    float* scale_out) const {
    std::vector<float> lut_f32(static_cast<size_t>(m_) * 16);
    preprocess_query_with_centroid(query, centroid, lut_f32.data());
    simd::quantize_lut_u8(lut_f32.data(), m_, 16, lut4, scale_out);
    // Capture the per-segment-minimum sum and scale so finalize_distance can
    // dequantize the raw FastScan result back to the true float sign-dot.
    // quantize_lut_u4 uses A = 15/max_span and stores (val - seg_min)*A.
    float seg_min_sum = 0.0f;
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = lut_f32.data() + s * 16;
        float mn = row[0];
        for (uint32_t c = 1; c < 16; c++) {
            if (row[c] < mn) mn = row[c];
        }
        seg_min_sum += mn;
    }
    seg_min_sum_ = seg_min_sum;
    lut_scale_ = scale_out ? *scale_out : 1.0f;
}

void RaBitQQuantizer::build_lut4_with_state(
    const float* query, const float* centroid,
    uint8_t* lut4, QueryState& qs) const {
    // Compute residual
    std::vector<float> rq(dim_);
    for (uint32_t i = 0; i < dim_; i++) rq[i] = query[i] - centroid[i];
    qs.qr_to_c_l2sqr = simd::dot_f32(rq.data(), rq.data(), dim_);
    qs.qr_norm_l2sqr = simd::dot_f32(query, query, dim_);

    // Rotate
    std::vector<float> rotated_q(padded_dim_, 0.0f);
    rotate(rq.data(), rotated_q.data());

    // Scalar quantize query
    float v_min = rotated_q[0], v_max = rotated_q[0];
    for (uint32_t i = 0; i < dim_; i++) {
        if (rotated_q[i] < v_min) v_min = rotated_q[i];
        if (rotated_q[i] > v_max) v_max = rotated_q[i];
    }
    const float span = v_max - v_min;
    const float delta = span / static_cast<float>(kQueryMaxCode);
    const float c1 = 2.0f * delta / dim_sqrt_;
    const float c2 = 2.0f * v_min / dim_sqrt_;
    std::vector<float> qq(dim_);
    float sum_qq = 0.0f;
    for (uint32_t i = 0; i < dim_; i++) {
        qq[i] = (span > 0.0f) ? std::round((rotated_q[i] - v_min) / delta) : 0.0f;
        sum_qq += qq[i];
    }
    qs.c34 = (delta * sum_qq + static_cast<float>(dim_) * v_min) / dim_sqrt_;

    // Build float LUT
    std::vector<float> lut_f32(static_cast<size_t>(m_) * 16);
    for (uint32_t m = 0; m < m_; m++) {
        const uint32_t ds = m * 4;
        const float v0 = c1 * qq[ds] + c2;
        const float v1 = c1 * qq[ds+1] + c2;
        const float v2 = c1 * qq[ds+2] + c2;
        const float v3 = c1 * qq[ds+3] + c2;
        write_subset_sum_lut(lut_f32.data() + m * 16, 0.0f, v0, v1, v2, v3);
    }

    // Quantize to uint4 and capture dequant params
    float scale;
    simd::quantize_lut_u8(lut_f32.data(), m_, 16, lut4, &scale);
    qs.lut_scale = scale;

    float seg_min_sum = 0.0f;
    for (uint32_t s = 0; s < m_; s++) {
        const float* row = lut_f32.data() + s * 16;
        float mn = row[0];
        for (uint32_t c = 1; c < 16; c++) if (row[c] < mn) mn = row[c];
        seg_min_sum += mn;
    }
    qs.seg_min_sum = seg_min_sum;
}

float RaBitQQuantizer::dequant_and_finalize(uint32_t raw_uint4,
                                            const float* factors,
                                            const QueryState& qs) const {
    const float dp_multiplier = factors[0];
    const float or_minus_c_l2sqr = factors[1];
    // Dequantize: raw = Σ (val_s - seg_min_s) * A → Σ val_s = raw/A + seg_min_sum
    const float dequant = (qs.lut_scale > 0.0f)
        ? static_cast<float>(raw_uint4) / qs.lut_scale + qs.seg_min_sum
        : static_cast<float>(raw_uint4);
    // final_dot = dequant - c34 (matches FAISS: c1*dot_qo + c2*sum_q - c34)
    const float final_dot = dequant - qs.c34;
    const float est_ip = dp_multiplier * final_dot;
    // pre_dist estimates ‖o-q‖² = ‖o-c‖² + ‖q-c‖² - 2·dp_mul·final_dot
    const float pre_dist = or_minus_c_l2sqr + qs.qr_to_c_l2sqr - 2.0f * est_ip;
    if (metric_ == MetricKind::InnerProduct) {
        // FAISS: IP_dist = -0.5 * (pre_dist - ‖q‖²)
        // We want: smaller return = larger <q,o> = better
        return -0.5f * (pre_dist - qs.qr_norm_l2sqr);
    }
    return std::max(0.0f, pre_dist);
}

float RaBitQQuantizer::error_bound(const float* factors,
                                   const QueryState& qs) const {
    const float norm_l2sqr = std::fmax(factors[1], 0.0f);
    const float dp_multiplier = factors[0];
    if (dp_multiplier <= 0.0f || norm_l2sqr <= 0.0f) return 0.0f;
    const float norm_l2 = std::sqrt(norm_l2sqr);
    const float dp_oO = norm_l2sqr * dim_sqrt_ / dp_multiplier;
    const float ip_resi_xucb = 0.5f * dp_oO;
    const float ratio_sq =
        (norm_l2sqr * kXuCbNormSqr * static_cast<float>(dim_)) /
        (ip_resi_xucb * ip_resi_xucb);
    if (ratio_sq <= 1.0f || dim_ <= 1) return 0.0f;
    const float tmp_error = norm_l2 * kConstEpsilon *
        std::sqrt((ratio_sq - 1.0f) / static_cast<float>(dim_ - 1));
    const float g_error = std::sqrt(qs.qr_to_c_l2sqr);
    return tmp_error * g_error;
}

float RaBitQQuantizer::get_query_factor(const std::string& name) const {
    if (name == "qr_to_c_l2sqr") return qr_to_c_l2sqr_;
    if (name == "c1") return c1_;
    if (name == "c2") return c2_;
    if (name == "c34") return c34_;
    throw std::out_of_range("unknown query factor: " + name);
}

float RaBitQQuantizer::finalize_distance(float raw_scan_result,
                                         const float* factors) const {
    const float dp_multiplier = factors[0];
    const float or_minus_c_l2sqr = factors[1];
    // raw_scan_result = c1*dot_qo + c2*sum_q (accumulated over positive-sign
    // dims by the subset-sum LUT). final_dot = raw - c34, matching FAISS:
    // final_dot = query_fac.c1 * dot_qo + query_fac.c2 * sum_q - query_fac.c34
    const float final_dot = raw_scan_result - c34_;
    const float est_ip = dp_multiplier * final_dot;
    return or_minus_c_l2sqr + qr_to_c_l2sqr_ - 2.0f * est_ip;
}

float RaBitQQuantizer::finalize_distance_ip(float raw_scan_result,
                                            const float* factors) const {
    const float dp_multiplier = factors[0];
    const float final_dot = raw_scan_result - c34_;
    return dp_multiplier * final_dot;
}

float RaBitQQuantizer::get_error_bound(const float* factors) const {
    // Recover norm_l2sqr and dp_oO from the stored factors:
    //   factors[0] = dp_multiplier = sqrt(norm_l2sqr) * padded_dim_sqrt / dp_oO
    //   factors[1] = or_minus_c_l2sqr = norm_l2sqr (L2) or norm_l2sqr - or_l2sqr (IP)
    // For the bound we only need norm_l2sqr and dp_oO. For IP the stored
    // factors[1] is norm_l2sqr - or_l2sqr, but the error bound is driven by
    // the residual norm, which for IP builds is still norm_l2sqr. We
    // approximate by assuming factors[1] ≈ norm_l2sqr (exact for L2; the IP
    // case differs by the constant or_l2sqr, which only weakly affects the
    // bound ratio). This matches the RaBitQ theory where the bound is on the
    // residual-vector inner-product error.
    const float norm_l2sqr = std::fmax(factors[1], 0.0f);
    const float dp_multiplier = factors[0];
    if (dp_multiplier <= 0.0f || norm_l2sqr <= 0.0f) {
        return 0.0f;
    }
    const float norm_l2 = std::sqrt(norm_l2sqr);
    const float dp_oO = norm_l2sqr * dim_sqrt_ / dp_multiplier;
    const float ip_resi_xucb = 0.5f * dp_oO;
    const float ratio_sq =
        (norm_l2sqr * kXuCbNormSqr * static_cast<float>(dim_)) /
        (ip_resi_xucb * ip_resi_xucb);
    float tmp_error = 0.0f;
    if (ratio_sq > 1.0f && dim_ > 1) {
        tmp_error = norm_l2 * kConstEpsilon *
                    std::sqrt((ratio_sq - 1.0f) / static_cast<float>(dim_ - 1));
    }
    const float f_error =
        (metric_ == MetricKind::InnerProduct) ? tmp_error : 2.0f * tmp_error;
    const float g_error = std::sqrt(qr_to_c_l2sqr_);
    return f_error * g_error;
}

void RaBitQQuantizer::decode_code(const uint8_t* code, float* out) const {
    std::memset(out, 0, dim_ * sizeof(float));
    (void)code;
}

void RaBitQQuantizer::serialize(std::vector<uint8_t>& out) const {
    out.resize(1 + 1 + 4 + 8);
    uint8_t* p = out.data();
    p[0] = kMagic;
    p[1] = static_cast<uint8_t>(metric_);
    uint32_t d = dim_;
    std::memcpy(p + 2, &d, sizeof(d));
    std::memcpy(p + 6, &seed_, sizeof(seed_));
}

void RaBitQQuantizer::deserialize(const uint8_t* in, size_t size) {
    const size_t header = 1 + 1 + 4 + 8;
    if (size < header) {
        throw Error(ErrorCode::CorruptIndex, "RaBitQ deserialize: blob too small");
    }
    if (in[0] != kMagic) {
        throw Error(ErrorCode::CorruptIndex, "RaBitQ deserialize: bad magic byte");
    }
    metric_ = static_cast<MetricKind>(in[1]);
    uint32_t d;
    std::memcpy(&d, in + 2, sizeof(d));
    std::memcpy(&seed_, in + 6, sizeof(seed_));
    dim_ = d;
    m_ = static_cast<uint16_t>(dim_ / 4);
    K_ = 16;
    sub_dim_ = 4;
    uint32_t p = 1;
    while (p < dim_) p <<= 1;
    padded_dim_ = p;
    padded_dim_sqrt_ = std::sqrt(static_cast<float>(p));
    dim_sqrt_ = std::sqrt(static_cast<float>(dim_));
    sign_code_bytes_ = (m_ * bits_ + 7) / 8;
    generate_signs_();
}

}  // namespace sextant
