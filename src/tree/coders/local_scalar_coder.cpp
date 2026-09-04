#include "local_scalar_coder.hpp"

#include "coder_util.hpp"
#include "simd_kernels.hpp"
#include "util/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace sextant::tree {

using coders::extract_code_from_leaf;
using coders::lloyd_split_f32;
using coders::scalar_scan_leaf;
using coders::ScalarScanCtx;
using coders::write_code_to_fs_block;

// Per-(query,leaf) scan context. a_uni / lut / c0 are PER-LEAF
// mutable state — each parallel worker owns its own Setup (see the
// per-worker a_buf note in the old search()).
struct LocalScalarCoder::Setup : public ScanSetup {
    std::vector<float> a_uni;      // per-leaf a_d = q_d*step_d (scratch)
    float c0 = 0.f;
    std::vector<float> query_copy;
    coders::ScalarLut lut;         // per-leaf FastScan LUT (dim x 16)
    int i8_mode = 0;
};

LocalScalarCoder::LocalScalarCoder(const CoderParams& params)
    : params_(params) {}

// --- static kernels (verbatim moves) ---

void LocalScalarCoder::fit_local_scalar_levels(const float* vecs,
                                               uint32_t count, uint16_t dim,
                                               float16_t* lo,
                                               float16_t* steps) {
    std::vector<float> mean(dim, 0.f), var(dim, 0.f);
    for (uint32_t i = 0; i < count; ++i)
        for (uint16_t d = 0; d < dim; ++d)
            mean[d] += vecs[static_cast<size_t>(i) * dim + d];
    for (uint16_t d = 0; d < dim; ++d) mean[d] /= count;
    for (uint32_t i = 0; i < count; ++i)
        for (uint16_t d = 0; d < dim; ++d) {
            const float e = vecs[static_cast<size_t>(i) * dim + d] - mean[d];
            var[d] += e * e;
        }
    std::vector<float> mn(dim, 1e30f), mx(dim, -1e30f);
    for (uint32_t i = 0; i < count; ++i)
        for (uint16_t d = 0; d < dim; ++d) {
            const float v = vecs[static_cast<size_t>(i) * dim + d];
            mn[d] = std::min(mn[d], v);
            mx[d] = std::max(mx[d], v);
        }
    for (uint16_t d = 0; d < dim; ++d) {
        const float sd = std::sqrt(var[d] / count);
        // Pure σ-loaded ruler (mean ± 2.7σ). Do NOT expand to min/max.
        float lo_f = mean[d] - 2.7f * sd, hi_f = mean[d] + 2.7f * sd;
        if (hi_f - lo_f <= 0.f) { lo_f = mn[d]; hi_f = mx[d]; }
        const float step = std::max((hi_f - lo_f) / 15.0f, 1e-8f);
        lo[d] = float16_t(lo_f - 0.5f * step);
        steps[d] = float16_t(step);
    }
}

void LocalScalarCoder::encode_one(const float* x, uint16_t dim,
                                  const float16_t* lo, const float16_t* steps,
                                  uint8_t* code, float16_t* bias_out) {
    double sx = 0, sxh = 0;
    for (uint16_t d = 0; d < dim; ++d) {
        const float lo_d = static_cast<float>(lo[d]);
        const float st_d = static_cast<float>(steps[d]);
        int c = static_cast<int>((x[d] - lo_d) / st_d + 0.5f);
        c = std::clamp(c, 0, 15);
        const float r = lo_d + st_d * c;
        sx += static_cast<double>(x[d]) * x[d];
        sxh += static_cast<double>(r) * r;
        code[d / 2] = static_cast<uint8_t>(
            (d % 2 == 0)
                ? ((code[d / 2] & 0xF0u) | (c & 0x0Fu))
                : ((code[d / 2] & 0x0Fu) | ((c & 0x0Fu) << 4)));
    }
    if (bias_out) {
        const float nx = static_cast<float>(std::sqrt(sx));
        const float nxh = static_cast<float>(
            std::sqrt(std::max(sxh, 1e-30)));
        *bias_out = float16_t(nx / nxh);
    }
}

void LocalScalarCoder::decode_one_with_levels(const float16_t* lo,
                                              const float16_t* steps,
                                              const uint8_t* code,
                                              uint16_t dim, float* out) {
    for (uint16_t d = 0; d < dim; ++d) {
        const uint8_t byte = code[d / 2];
        const uint32_t c = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        out[d] = static_cast<float>(lo[d]) + static_cast<float>(steps[d]) * c;
    }
}

// --- layout ---

LeafGeometry LocalScalarCoder::geometry(const TreeLeafHeader* h) const {
    LeafGeometry g;
    g.codes_offset = lsc_codes_offset(h->summary_size, params_.dim);
    g.rowids_offset = lsc_rowids_offset(h->summary_size, params_.dim,
                                        h->count, params_.has_ip_bias);
    g.filter_offset =
        g.rowids_offset + static_cast<uint64_t>(h->count) * sizeof(RowId);
    g.extent_bytes = g.rowids_offset +
        static_cast<uint64_t>(h->count) * sizeof(RowId);
    return g;
}

uint64_t LocalScalarCoder::extent_bytes(uint32_t count, uint32_t summary_size,
                                        uint64_t filter_cols_bytes) const {
    // [header][summary][lo][steps][code blocks (FastScan, m4=dim)]
    // [ip_biases?][row_ids][filters]
    return lsc_codes_offset(summary_size, params_.dim) +
           static_cast<uint64_t>(scalar_n_blocks(count)) *
               scalar_block_bytes(params_.dim) +
           scalar_bias_bytes(count, params_.has_ip_bias) +
           static_cast<uint64_t>(count) * sizeof(RowId) + filter_cols_bytes;
}

uint32_t LocalScalarCoder::code_size() const {
    return (params_.dim + 1) / 2;
}

// --- build ---

void LocalScalarCoder::train(const float*, uint32_t) {
    // No global state: levels are fitted per leaf at flush time.
}

bool LocalScalarCoder::serialize_global(std::vector<uint8_t>&) const {
    return false;
}

bool LocalScalarCoder::deserialize_global(const uint8_t*, uint64_t) {
    return false;
}

void LocalScalarCoder::encode(const float*, uint8_t*, float*) {
    // Emission keeps the raw FP16 vectors; encoding happens per leaf at
    // flush time.
}

uint64_t LocalScalarCoder::flush_leaf(const LeafFlushInput& in,
                                      uint8_t* leaf_out) {
    const uint16_t dim = params_.dim;
    const uint32_t cs = code_size();
    // Fit per-leaf uniform levels from the raw FP16 vectors, then encode.
    std::vector<float16_t> lo(dim), steps(dim);
    {
        std::vector<float> f32vecs(static_cast<size_t>(in.count) * dim);
        for (size_t i = 0; i < f32vecs.size(); ++i)
            f32vecs[i] = static_cast<float>(in.fp16_vecs[i]);
        fit_local_scalar_levels(f32vecs.data(), in.count, dim, lo.data(),
                                steps.data());
    }
    std::vector<uint8_t> codes(static_cast<size_t>(in.count) * cs, 0);
    std::vector<float16_t> biases(
        params_.has_ip_bias ? in.count : 0);
    std::vector<float> x(dim);
    for (uint32_t i = 0; i < in.count; ++i) {
        const float16_t* fv = in.fp16_vecs + static_cast<size_t>(i) * dim;
        for (uint16_t d = 0; d < dim; ++d) x[d] = static_cast<float>(fv[d]);
        encode_one(x.data(), dim, lo.data(), steps.data(),
                   codes.data() + static_cast<size_t>(i) * cs,
                   params_.has_ip_bias ? &biases[i] : nullptr);
    }
    auto* lh = reinterpret_cast<TreeLeafHeader*>(leaf_out);
    lh->leaf_state = static_cast<uint8_t>(LeafState::CodedLocalScalar);
    std::memcpy(leaf_out + lsc_levels_offset(in.summary_size), lo.data(),
                dim * sizeof(float16_t));
    std::memcpy(leaf_out + lsc_levels_offset(in.summary_size) +
                    static_cast<uint64_t>(dim) * sizeof(float16_t),
                steps.data(), dim * sizeof(float16_t));
    uint8_t* codes_out = leaf_out + lsc_codes_offset(in.summary_size, dim);
    const uint32_t bb = scalar_block_bytes(dim);
    const uint32_t nb = scalar_n_blocks(in.count);
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t* blk = codes_out + static_cast<uint64_t>(b) * bb;
        std::memset(blk, 0, bb);
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t gi = b * 32 + j;
            if (gi >= in.count) break;
            write_code_to_fs_block(blk, j, dim, 4, cs,
                                   codes.data() + static_cast<size_t>(gi) * cs);
        }
    }
    if (params_.has_ip_bias) {
        std::memcpy(codes_out + static_cast<uint64_t>(nb) * bb,
                    biases.data(),
                    static_cast<size_t>(in.count) * sizeof(float16_t));
    }
    return lsc_rowids_offset(in.summary_size, dim, in.count,
                             params_.has_ip_bias);
}

// --- search ---

std::unique_ptr<ScanSetup> LocalScalarCoder::scan_setup(const float* query) {
    auto s = std::make_unique<Setup>();
    s->query_copy.assign(query, query + params_.dim);
    const uint32_t padded = (params_.dim + 15) / 16 * 16;
    s->a_uni.assign(padded, 0.f);
    s->i8_mode = 0;
#if defined(__ARM_FEATURE_DOTPROD)
    {
        // Default: dual-LUT FastScan; per-thread override, then the
        // env-resolved params default.
        s->i8_mode = 2;
        const int ov = scan_detail::scan_i8_override();
        if (ov > 0) s->i8_mode = ov;
        else if (ov == 0) s->i8_mode = 0;
        else if (params_.scan_i8_mode > 0) s->i8_mode = params_.scan_i8_mode;
    }
#endif
    return s;
}

void LocalScalarCoder::bind_leaf(ScanSetup& setup, const uint8_t* leaf) const {
    auto& s = static_cast<Setup&>(setup);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint16_t dim = params_.dim;
    const float* query = s.query_copy.data();
    const float16_t* lo16 = reinterpret_cast<const float16_t*>(
        leaf + lsc_levels_offset(lh->summary_size));
    const float16_t* st16 = lo16 + dim;
    s.c0 = 0.f;  // local sets its own per-leaf c0 (NOT the global one)
    for (uint32_t d = 0; d < dim; ++d) {
        s.a_uni[d] = query[d] * static_cast<float>(st16[d]);
        s.c0 += query[d] * static_cast<float>(lo16[d]);
    }
    if (s.i8_mode) {
        // Per-leaf LUT build (a_d depends on this leaf's step levels).
        // Cost: ~2 x dim x 16 u8 writes + 2 passes over dim terms; tiny next
        // to the dim x count table lookups it feeds.
        if (!coders::scalar_build_lut(s.a_uni.data(), dim, nullptr,
                                      s.i8_mode >= 2, s.lut))
            s.i8_mode = 0;  // dim > 65535: u16 accum headroom gone
    }
}

void LocalScalarCoder::scan_leaf(const ScanSetup& setup, const uint8_t* leaf,
                                 RawScanHeap& heap) {
    const auto& s = static_cast<const Setup&>(setup);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t count = lh->count;
    if (count == 0) return;

    ScalarScanCtx c;
    c.codes = leaf + lsc_codes_offset(lh->summary_size, params_.dim);
    c.count = count;
    c.block_bytes = scalar_block_bytes(params_.dim);
    c.ip_bias = getenv("SEXTANT_NO_IP_BIAS")
        ? nullptr
        : (params_.metric == MetricKind::InnerProduct
               ? reinterpret_cast<const float16_t*>(
                     c.codes + static_cast<uint64_t>(scalar_n_blocks(count)) *
                                   c.block_bytes)
               : nullptr);
    c.a_uni = s.a_uni.data();
    c.c0 = s.c0;
    c.query = s.query_copy.data();
    c.dim = params_.dim;
    c.i8_mode = s.i8_mode;
    c.lut_hi = s.lut.hi.data();
    c.lut_lo = s.lut.dual ? s.lut.lo.data() : nullptr;
    c.lut_inv_ah = s.lut.inv_ah;
    c.lut_inv_al = s.lut.inv_al;
    c.lut_lo_off = s.lut.lo_off;
    c.lut_b = s.lut.b;
    c.slm_arith = true;   // local levels are always arithmetic (uniform)
    c.slm_shaped = false;
    c.K = 16;
    scalar_scan_leaf(c, heap);
}

float LocalScalarCoder::rerank(const float* query, const uint8_t* leaf,
                               uint32_t local_idx, float* scratch_decoded) {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint16_t dim = params_.dim;
    const float16_t* lo16 = reinterpret_cast<const float16_t*>(
        leaf + lsc_levels_offset(lh->summary_size));
    const float16_t* st16 = lo16 + dim;
    const uint32_t bb = scalar_block_bytes(dim);
    const uint64_t codes_off = lsc_codes_offset(lh->summary_size, dim);
    std::vector<uint8_t> code_buf(code_size());
    extract_code_from_leaf(leaf, local_idx, lh->summary_size, dim, 4, 32, bb,
                           code_buf.data(), codes_off);
    // Per-vector IP bias, same correction as the scan.
    float16_t ls_bias = 1.f;
    if (params_.metric == MetricKind::InnerProduct && lh->count > 0) {
        const float16_t* biases = reinterpret_cast<const float16_t*>(
            leaf + codes_off +
            static_cast<uint64_t>(scalar_n_blocks(lh->count)) * bb);
        ls_bias = biases[local_idx];
    }
    std::vector<float> decoded(dim);
    decode_one_with_levels(lo16, st16, code_buf.data(), dim, decoded.data());
    // Fold the per-vector IP bias into the decoded vector so the generic dot
    // below carries the correction (IP only; bias == 1 otherwise).
    if (ls_bias != float16_t(1.f)) {
        const float b = static_cast<float>(ls_bias);
        for (uint16_t d = 0; d < dim; ++d) decoded[d] *= b;
    }
    if (scratch_decoded)
        std::memcpy(scratch_decoded, decoded.data(), dim * sizeof(float));
    return (params_.metric == MetricKind::InnerProduct)
        ? -simd::dot_f32(query, decoded.data(), dim)
        : simd::l2sq_f32(query, decoded.data(), dim);
}

// --- mutation ---

void LocalScalarCoder::decode_one(const uint8_t* leaf, uint32_t local_idx,
                                  float* out) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const float16_t* lo16 = reinterpret_cast<const float16_t*>(
        leaf + lsc_levels_offset(lh->summary_size));
    const float16_t* st16 = lo16 + params_.dim;
    const uint32_t dim = params_.dim;
    std::vector<uint8_t> code_buf(code_size());
    extract_code_from_leaf(leaf, local_idx, lh->summary_size, dim, 4, 32,
                           scalar_block_bytes(dim), code_buf.data(),
                           lsc_codes_offset(lh->summary_size, dim));
    decode_one_with_levels(lo16, st16, code_buf.data(), dim, out);
}

void LocalScalarCoder::extract_codes(const uint8_t* leaf, uint32_t count,
                                     uint8_t* codes) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t dim = params_.dim;
    const uint32_t cs = code_size();
    for (uint32_t i = 0; i < count; ++i)
        extract_code_from_leaf(leaf, i, lh->summary_size, dim, 4, 32,
                               scalar_block_bytes(dim),
                               codes + static_cast<size_t>(i) * cs,
                               lsc_codes_offset(lh->summary_size, dim));
}

LeafCoder::SplitPlan LocalScalarCoder::plan_split(const uint8_t*,
                                                  const uint8_t* codes,
                                                  const float* vecs,
                                                  uint32_t count, uint32_t) {
    (void)codes;
    SplitPlan plan;
    lloyd_split_f32(vecs, count, params_.dim, plan.group0, plan.group1,
                    plan.cent0_fp16, plan.cent1_fp16);
    return plan;
}

void LocalScalarCoder::encode_group(const GroupEncodeInput& in,
                                    uint8_t* leaf_out) {
    const uint16_t dim = params_.dim;
    const uint32_t cs = code_size();
    // Refit uniform levels per half, re-encode (the one re-fit point).
    std::vector<float16_t> lo(dim), st(dim);
    fit_local_scalar_levels(in.vecs, in.count, dim, lo.data(), st.data());
    const uint32_t bb = scalar_block_bytes(dim);
    const uint32_t nb = scalar_n_blocks(in.count);
    uint8_t* ncb = leaf_out + lsc_codes_offset(in.summary_size, dim);
    std::vector<uint8_t> code(cs, 0);
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;
        std::memset(blk, 0, bb);
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t gi = b * 32 + j;
            if (gi >= in.count) break;
            encode_one(in.vecs + static_cast<size_t>(gi) * dim, dim, lo.data(),
                       st.data(), code.data(), nullptr);
            write_code_to_fs_block(blk, j, dim, 4, cs, code.data());
        }
    }
    if (params_.has_ip_bias && in.src_biases) {
        float16_t* nbias = reinterpret_cast<float16_t*>(
            ncb + static_cast<uint64_t>(nb) * bb);
        std::memcpy(nbias, in.src_biases,
                    static_cast<size_t>(in.count) * sizeof(float16_t));
    }
    RowId* nrid = reinterpret_cast<RowId*>(
        leaf_out + lsc_rowids_offset(in.summary_size, dim, in.count,
                                     params_.has_ip_bias));
    std::memcpy(nrid, in.row_ids, static_cast<size_t>(in.count) * sizeof(RowId));
    auto* lh = reinterpret_cast<TreeLeafHeader*>(leaf_out);
    lh->leaf_state = static_cast<uint8_t>(LeafState::CodedLocalScalar);
    std::memcpy(leaf_out + lsc_levels_offset(in.summary_size), lo.data(),
                dim * sizeof(float16_t));
    std::memcpy(leaf_out + lsc_levels_offset(in.summary_size) +
                    static_cast<uint64_t>(dim) * sizeof(float16_t),
                st.data(), dim * sizeof(float16_t));
}

void LocalScalarCoder::append_encode(const AppendInput& in) {
    const auto* olh = reinterpret_cast<const TreeLeafHeader*>(in.old_leaf);
    const uint16_t dim = params_.dim;
    const uint32_t cs = code_size();
    const uint64_t codes_off = lsc_codes_offset(olh->summary_size, dim);
    // Preserve the leaf's levels (frozen — insert never refits).
    std::memcpy(in.new_leaf + lsc_levels_offset(olh->summary_size),
                in.old_leaf + lsc_levels_offset(olh->summary_size),
                codes_off - lsc_levels_offset(olh->summary_size));
    const float16_t* lo16 = reinterpret_cast<const float16_t*>(
        in.old_leaf + lsc_levels_offset(olh->summary_size));
    const float16_t* st16 = lo16 + dim;
    const uint32_t bb = scalar_block_bytes(dim);
    const uint32_t old_nb = scalar_n_blocks(in.old_count);
    uint8_t* ncb = in.new_leaf + codes_off;
    std::memcpy(ncb, in.old_leaf + codes_off,
                static_cast<size_t>(old_nb) * bb);
    float16_t* nbias = nullptr;
    if (params_.has_ip_bias) {
        const uint32_t nb = scalar_n_blocks(in.new_count);
        nbias = reinterpret_cast<float16_t*>(
            ncb + static_cast<uint64_t>(nb) * bb);
        std::memcpy(nbias,
                    in.old_leaf + codes_off +
                        static_cast<uint64_t>(old_nb) * bb,
                    static_cast<size_t>(in.old_count) * sizeof(float16_t));
    }
    std::vector<uint8_t> code(cs, 0);
    for (uint32_t ai = 0; ai < in.n_vecs; ++ai) {
        const uint32_t gi = in.old_count + ai;
        encode_one(in.vecs[ai], dim, lo16, st16, code.data(),
                   nbias ? &nbias[gi] : nullptr);
        write_code_to_fs_block(ncb + static_cast<uint64_t>(gi / 32) * bb,
                               gi % 32, dim, 4, cs, code.data());
    }
}

}  // namespace sextant::tree
