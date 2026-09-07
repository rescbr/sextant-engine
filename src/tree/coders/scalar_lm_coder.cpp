#include "scalar_lm_coder.hpp"

#include "coder_util.hpp"
#include "quant/scalar_lloydmax_quantizer.hpp"
#include "simd_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace sextant::tree {

using coders::lloyd_split_f32;
using coders::scalar_scan_leaf;
using coders::ScalarScanCtx;

struct ScalarLmCoder::Setup : public ScanSetup {
    // Per-query global-ruler transform (a_d = q_d·step_d, zero-padded to the
    // 16-dim kernel width) + additive constant c0 = Σ q_d·lo_d.
    std::vector<float> a_uni;
    float c0 = 0.f;
    std::vector<float> query_copy;
    // i8 SDOT operands (mode 1/2).
    std::vector<int8_t> a8, a8_lo;
    float i8_inv = 0.f;
    int i8_mode = 0;
};

ScalarLmCoder::ScalarLmCoder(LevelPolicy policy, const CoderParams& params)
    : policy_(policy), params_(params),
      quantizer_(std::make_unique<ScalarLloydMaxQuantizer>(
          params.metric, params.dim, params.pq_bits)) {}

ScalarLmCoder::~ScalarLmCoder() = default;

std::string ScalarLmCoder::family_name() const {
    switch (policy_) {
        case LevelPolicy::Uniform: return "scalar_uniform";
        case LevelPolicy::Shape: return "scalar_shape";
        default: return "scalar_lloydmax";
    }
}

LeafGeometry ScalarLmCoder::geometry(const TreeLeafHeader* h) const {
    const uint32_t cs = quantizer_->code_size();
    LeafGeometry g;
    g.codes_offset = leaf_codes_offset(h->summary_size);
    g.rowids_offset = scalar_rowids_offset(h->summary_size, h->count, cs,
                                           params_.has_ip_bias);
    g.filter_offset =
        g.rowids_offset + static_cast<uint64_t>(h->count) * sizeof(RowId);
    g.extent_bytes = g.rowids_offset +
        static_cast<uint64_t>(h->count) * sizeof(RowId);
    return g;
}

uint64_t ScalarLmCoder::extent_bytes(uint32_t count, uint32_t summary_size,
                                     uint64_t filter_cols_bytes) const {
    // Flat packed-nibble layout: [header][summary][codes][ip_biases?]
    // [row_ids][filters].
    const uint64_t codes_bytes =
        static_cast<uint64_t>(count) * quantizer_->code_size();
    const uint64_t bias_bytes = scalar_bias_bytes(count, params_.has_ip_bias);
    const uint64_t rowids_bytes =
        static_cast<uint64_t>(count) * sizeof(RowId);
    return leaf_codes_offset(summary_size) + codes_bytes + bias_bytes +
           rowids_bytes + filter_cols_bytes;
}

uint32_t ScalarLmCoder::code_size() const {
    return quantizer_->code_size();
}

void ScalarLmCoder::train(const float* sample, uint32_t n) {
    switch (policy_) {
        case LevelPolicy::Uniform: quantizer_->train_uniform(sample, n); break;
        case LevelPolicy::Shape: quantizer_->train_shape(sample, n); break;
        default: quantizer_->train(sample, n); break;
    }
}

bool ScalarLmCoder::serialize_global(std::vector<uint8_t>& out) const {
    quantizer_->serialize(out);
    return true;
}

bool ScalarLmCoder::deserialize_global(const uint8_t* data, uint64_t size) {
    quantizer_->deserialize(data, static_cast<size_t>(size));
    return true;
}

void ScalarLmCoder::encode(const float* vec, uint8_t* code_out,
                           float* ip_bias_out) {
    quantizer_->encode(vec, code_out);
    if (ip_bias_out) {
        // bias = ||x||/||x_hat||: cancels the per-vector reconstruction
        // norm shrinkage in the IP scan ordering.
        std::vector<float> dec(params_.dim);
        quantizer_->decode(code_out, dec.data());
        double sx = 0, sxh = 0;
        for (uint16_t d = 0; d < params_.dim; ++d) {
            sx += static_cast<double>(vec[d]) * vec[d];
            sxh += static_cast<double>(dec[d]) * dec[d];
        }
        const float nx = static_cast<float>(std::sqrt(sx));
        const float nxh = static_cast<float>(std::sqrt(std::max(sxh, 1e-30)));
        *ip_bias_out = float16_t(nx / nxh);
    }
}

uint64_t ScalarLmCoder::flush_leaf(const LeafFlushInput& in,
                                   uint8_t* leaf_out) {
    const uint32_t cs = quantizer_->code_size();
    uint8_t* codes_out = leaf_out + leaf_codes_offset(in.summary_size);
    std::memcpy(codes_out, in.codes,
                static_cast<size_t>(in.count) * cs);
    if (params_.has_ip_bias && in.ip_biases) {
        std::memcpy(codes_out + static_cast<uint64_t>(in.count) * cs,
                    in.ip_biases,
                    static_cast<size_t>(in.count) * sizeof(float16_t));
    }
    return scalar_rowids_offset(in.summary_size, in.count, cs,
                                params_.has_ip_bias);
}

std::unique_ptr<ScanSetup> ScalarLmCoder::scan_setup(const float* query) {
    auto s = std::make_unique<Setup>();
    s->query_copy.assign(query, query + params_.dim);
    const uint32_t dim = params_.dim;
    const bool slm_arith = quantizer_->arithmetic_scan();
    [[maybe_unused]] const bool slm_shaped = slm_arith && !quantizer_->is_uniform();
    if (slm_arith) {
        // Zero-pad a_uni to the 16-dim kernel width: tail dims read garbage
        // nibbles but multiply by 0. c0 = Σ q_d·lo_d — NOT droppable once the
        // per-vector IP bias multiplies the score.
        const uint32_t padded = (dim + 15) / 16 * 16;
        s->a_uni.assign(padded, 0.f);
        const float* steps = quantizer_->steps();
        const float* levels0 = quantizer_->levels();
        for (uint32_t d = 0; d < dim; ++d) {
            s->a_uni[d] = query[d] * steps[d];
            s->c0 += query[d] * levels0[d];
        }
    }
    // i8 scan-kernel selection: thread-local override, else env-resolved
    // default from CoderParams.scan_i8_mode.
    s->i8_mode = 0;
#if defined(__ARM_FEATURE_DOTPROD) || defined(SEXTANT_HAS_AVX512_SCAN)
    if (slm_arith && !slm_shaped) {
        const int ov = scan_detail::scan_i8_override();
        if (ov > 0) s->i8_mode = ov;
        else if (ov < 0 && params_.scan_i8_mode > 0)
            s->i8_mode = params_.scan_i8_mode;
    }
#endif
    // i8 operand prep: a8[d] = clamp(round(a_d · s)), s = 127/amax.
    if (s->i8_mode) {
        const uint32_t dim = params_.dim;
        const uint32_t padded8 = (dim + 15) / 16 * 16;
        s->a8.assign(padded8, 0);
        if (s->i8_mode >= 2) s->a8_lo.assign(padded8, 0);
        float amax = 1e-12f;
        for (uint32_t d = 0; d < dim; ++d)
            amax = std::max(amax, std::fabs(s->a_uni[d]));
        const float i8_scale = 127.0f / amax;
        s->i8_inv = 1.0f / i8_scale;
        for (uint32_t d = 0; d < dim; ++d) {
            float q = s->a_uni[d] * i8_scale;
            const int8_t hi = static_cast<int8_t>(q >= 0.f
                ? (q + 0.5f >= 127.f ? 127.f : q + 0.5f)
                : (q - 0.5f <= -127.f ? -127.f : q - 0.5f));
            s->a8[d] = hi;
            if (s->i8_mode >= 2) {
                const float r = q - static_cast<float>(hi);
                s->a8_lo[d] = static_cast<int8_t>(r >= 0.f
                    ? r * 127.f + 0.5f : r * 127.f - 0.5f);
            }
        }
    }
    return s;
}

void ScalarLmCoder::bind_leaf(ScanSetup&, const uint8_t*) const {}

void ScalarLmCoder::scan_leaf(const ScanSetup& setup, const uint8_t* leaf,
                              RawScanHeap& heap) {
    const auto& s = static_cast<const Setup&>(setup);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t count = lh->count;
    if (count == 0) return;

    ScalarScanCtx c;
    c.codes = leaf + leaf_codes_offset(lh->summary_size);
    c.count = count;
    c.cs = quantizer_->code_size();
    c.ip_bias = getenv("SEXTANT_NO_IP_BIAS")
        ? nullptr
        : (params_.metric == MetricKind::InnerProduct
               ? reinterpret_cast<const float16_t*>(
                     c.codes + static_cast<uint64_t>(count) * c.cs)
               : nullptr);
    c.a_uni = s.a_uni.data();
    c.c0 = s.c0;
    c.query = s.query_copy.data();
    c.dim = params_.dim;
    c.i8_mode = s.i8_mode;
    c.a8 = s.a8.data();
    c.a8_lo = s.a8_lo.data();
    c.i8_inv = s.i8_inv;
    c.slm_arith = quantizer_->arithmetic_scan();
    c.slm_shaped = c.slm_arith && !quantizer_->is_uniform();
    c.levels = quantizer_->levels();
    c.K = quantizer_->K();
    c.shape_u8 = quantizer_->shape_u8();
    c.shape_f32 = quantizer_->shape();
    scalar_scan_leaf(c, heap);
}

float ScalarLmCoder::rerank(const float* query, const uint8_t* leaf,
                            uint32_t local_idx, float* scratch_decoded) {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t slm_cs = quantizer_->code_size();
    const uint8_t* code_ptr = leaf + leaf_codes_offset(lh->summary_size) +
        static_cast<uint64_t>(local_idx) * slm_cs;
    if (params_.metric == MetricKind::InnerProduct) {
        // Dispatcher: SVE2 gather-dot on c4a, NEON decode-dot elsewhere.
        float dot = simd::scalar_dot_u4_sve2(
            query, quantizer_->levels(), code_ptr, params_.dim,
            quantizer_->K());
        // Apply the leaf's per-vector IP bias (same correction as the scan).
        if (lh->count > 0) {
            const float16_t* bias = reinterpret_cast<const float16_t*>(
                leaf + leaf_codes_offset(lh->summary_size) +
                static_cast<uint64_t>(lh->count) * slm_cs);
            dot *= static_cast<float>(bias[local_idx]);
        }
        return -dot;
    }
    std::vector<float> dec(params_.dim);
    quantizer_->decode(code_ptr, dec.data());
    if (scratch_decoded)
        std::memcpy(scratch_decoded, dec.data(), params_.dim * sizeof(float));
    return simd::l2sq_f32(query, dec.data(), params_.dim);
}

void ScalarLmCoder::decode_one(const uint8_t* leaf, uint32_t local_idx,
                               float* out) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    quantizer_->decode(
        leaf + leaf_codes_offset(lh->summary_size) +
            static_cast<uint64_t>(local_idx) * quantizer_->code_size(),
        out);
}

void ScalarLmCoder::extract_codes(const uint8_t* leaf, uint32_t count,
                                  uint8_t* codes) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    std::memcpy(codes, leaf + leaf_codes_offset(lh->summary_size),
                static_cast<size_t>(count) * quantizer_->code_size());
}

LeafCoder::SplitPlan ScalarLmCoder::plan_split(const uint8_t*,
                                               const uint8_t* codes,
                                               const float* vecs,
                                               uint32_t count, uint32_t) {
    (void)codes;  // vecs are pre-decoded by the caller (split_leaf_)
    SplitPlan plan;
    lloyd_split_f32(vecs, count, params_.dim, plan.group0, plan.group1,
                    plan.cent0_fp16, plan.cent1_fp16);
    return plan;
}

void ScalarLmCoder::encode_group(const GroupEncodeInput& in,
                                 uint8_t* leaf_out) {
    // Flat packed-nibble layout: codes, then IP biases (InnerProduct only),
    // then row_ids. No blocks.
    const uint32_t cs = quantizer_->code_size();
    uint8_t* ncb = leaf_out + leaf_codes_offset(in.summary_size);
    std::memcpy(ncb, in.src_codes, static_cast<size_t>(in.count) * cs);
    if (params_.has_ip_bias && in.src_biases) {
        float16_t* nbias = reinterpret_cast<float16_t*>(
            ncb + static_cast<uint64_t>(in.count) * cs);
        std::memcpy(nbias, in.src_biases,
                    static_cast<size_t>(in.count) * sizeof(float16_t));
    }
    RowId* nrid = reinterpret_cast<RowId*>(
        leaf_out + scalar_rowids_offset(in.summary_size, in.count, cs,
                                        params_.has_ip_bias));
    std::memcpy(nrid, in.row_ids, static_cast<size_t>(in.count) * sizeof(RowId));
}

void ScalarLmCoder::append_encode(const AppendInput& in) {
    const auto* olh = reinterpret_cast<const TreeLeafHeader*>(in.old_leaf);
    const uint32_t cs = quantizer_->code_size();
    uint8_t* ncb = in.new_leaf + leaf_codes_offset(olh->summary_size);
    std::memcpy(ncb, in.old_leaf + leaf_codes_offset(olh->summary_size),
                static_cast<size_t>(in.old_count) * cs);
    float16_t* nbias = nullptr;
    if (params_.has_ip_bias) {
        nbias = reinterpret_cast<float16_t*>(
            ncb + static_cast<uint64_t>(in.new_count) * cs);
        std::memcpy(nbias,
                    in.old_leaf + leaf_codes_offset(olh->summary_size) +
                        static_cast<uint64_t>(in.old_count) * cs,
                    static_cast<size_t>(in.old_count) * sizeof(float16_t));
    }
    std::vector<float> dec(params_.has_ip_bias ? params_.dim : 0);
    for (uint32_t ai = 0; ai < in.n_vecs; ++ai) {
        uint8_t* dst = ncb + static_cast<uint64_t>(in.old_count + ai) * cs;
        quantizer_->encode(in.vecs[ai], dst);
        if (params_.has_ip_bias) {
            quantizer_->decode(dst, dec.data());
            const float* xv = in.vecs[ai];
            double sx = 0, sxh = 0;
            for (uint16_t d = 0; d < params_.dim; ++d) {
                sx += static_cast<double>(xv[d]) * xv[d];
                sxh += static_cast<double>(dec[d]) * dec[d];
            }
            const float nx = static_cast<float>(std::sqrt(sx));
            const float nxh = static_cast<float>(
                std::sqrt(std::max(sxh, 1e-30)));
            nbias[in.old_count + ai] = float16_t(nx / nxh);
        }
    }
}

}  // namespace sextant::tree
