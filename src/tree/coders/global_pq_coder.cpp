#include "global_pq_coder.hpp"

#include "coder_util.hpp"
#include "engine/partition.hpp"
#include "quant/anisotropic_pq_quantizer.hpp"
#include "quant/product_residual_quantizer.hpp"
#include "simd_kernels.hpp"
#include "util/fp16.hpp"

#include <cstring>

namespace sextant::tree {

using coders::extract_code_from_leaf;
using coders::u32_min16;
using coders::u32_min32;
using coders::u32_mask_sentinel32;
using coders::write_code_to_fs_block;

// ---------------------------------------------------------------------------
// Setup: per-query LUTs + PQ-LUT-rerank tables.
// ---------------------------------------------------------------------------
struct GlobalPqCoder::Setup : public ScanSetup {
    std::vector<uint8_t> lut4, lut8;
    // LUT rerank tables (moved verbatim from search()): m × K partial dots
    // and codeword norms, plus the query norm².
    std::vector<float> rerank_dot, rerank_nrm;
    float query_sq = 0.f;
};

GlobalPqCoder::GlobalPqCoder(Kind kind, const CoderParams& params)
    : kind_(kind), params_(params) {
    const uint16_t dim = params.dim;
    const uint16_t m4 = params.m4;
    const uint8_t bits = params.pq_bits;
    if (kind == Kind::Prq) {
        quantizer_ = std::make_unique<ProductResidualQuantizer>(
            params.metric, dim, m4, bits, params.prq_nsplits,
            params.prq_beam_size, 42);
    } else if (kind == Kind::Anisotropic) {
        quantizer_ = std::make_unique<AnisotropicPqQuantizer>(
            params.metric, dim, m4, bits);
    } else {
        quantizer_ = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, bits, 42);
    }
}

GlobalPqCoder::~GlobalPqCoder() = default;

std::string GlobalPqCoder::family_name() const {
    switch (kind_) {
        case Kind::Prq: return "prq";
        case Kind::Anisotropic: return "anisotropic_pq";
        default: return "pq";
    }
}

MetricKind GlobalPqCoder::metric() const { return quantizer_->metric(); }
const PqQuantizer* GlobalPqCoder::pq_quantizer() const {
    return quantizer_.get();
}

LeafGeometry GlobalPqCoder::geometry(const TreeLeafHeader* h) const {
    const uint32_t cpb = (h->pq_bits == 4) ? 32 : 16;
    const uint32_t bb = h->m4 * 16;
    const uint64_t n_blocks =
        (h->count + cpb - 1) / cpb;
    LeafGeometry g;
    g.codes_offset = leaf_codes_offset(h->summary_size);
    g.rowids_offset = leaf_rowids_offset(h->summary_size, n_blocks, bb);
    g.filter_offset =
        g.rowids_offset + static_cast<uint64_t>(h->count) * sizeof(RowId);
    g.extent_bytes = leaf_extent_bytes(h->count, h->m4, h->pq_bits,
                                       h->summary_size, 0);
    return g;
}

uint64_t GlobalPqCoder::extent_bytes(uint32_t count, uint32_t summary_size,
                                     uint64_t filter_cols_bytes) const {
    return leaf_extent_bytes(count, params_.m4, params_.pq_bits, summary_size,
                             filter_cols_bytes);
}

uint32_t GlobalPqCoder::code_size() const { return quantizer_->code_size(); }

void GlobalPqCoder::train(const float* sample, uint32_t n) {
    quantizer_->train(sample, n);
}

bool GlobalPqCoder::serialize_global(std::vector<uint8_t>& out) const {
    quantizer_->serialize(out);
    return true;
}

bool GlobalPqCoder::deserialize_global(const uint8_t* data, uint64_t size) {
    quantizer_->deserialize(data, static_cast<size_t>(size));
    return true;
}

void GlobalPqCoder::encode(const float* vec, uint8_t* code_out,
                           float* ip_bias_out) {
    (void)ip_bias_out;  // global PQ leaves carry no per-vector bias
    quantizer_->encode(vec, code_out);
}

uint64_t GlobalPqCoder::flush_leaf(const LeafFlushInput& in,
                                   uint8_t* leaf_out) {
    const uint32_t cpb = (params_.pq_bits == 4) ? 32 : 16;
    const uint32_t bb = params_.m4 * 16;
    const uint32_t n_blocks = (in.count + cpb - 1) / cpb;
    const uint32_t code_size = quantizer_->code_size();
    uint8_t* codes_out = leaf_out + leaf_codes_offset(in.summary_size);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint32_t base = b * cpb;
        uint8_t* blk = codes_out + static_cast<uint64_t>(b) * bb;
        std::memset(blk, 0, bb);
        for (uint32_t j = 0; j < cpb; ++j) {
            const uint32_t gi = base + j;
            if (gi >= in.count) break;
            write_code_to_fs_block(blk, j, params_.m4, params_.pq_bits,
                                   code_size, in.codes + gi * code_size);
        }
    }
    return leaf_rowids_offset(in.summary_size, n_blocks, bb);
}

std::unique_ptr<ScanSetup> GlobalPqCoder::scan_setup(const float* query) {
    auto s = std::make_unique<Setup>();
    if (params_.pq_bits == 8) {
        s->lut8.resize(static_cast<size_t>(params_.m4) * 256);
        float scale = 0, offset = 0;
        quantizer_->build_fastscan_lut(query, s->lut8.data(), &scale, &offset);
    } else {
        s->lut4.resize(static_cast<size_t>(params_.m4) * 16);
        quantizer_->build_fastscan_lut4(query, s->lut4.data(), nullptr);
    }
    // Per-query LUT rerank tables (moved verbatim from search()).
    const uint32_t K = quantizer_->K();
    const uint32_t sub = quantizer_->sub_dim();
    const float* cb = quantizer_->codebook();
    const uint32_t m = params_.m4;
    s->rerank_dot.assign(static_cast<size_t>(m) * K, 0.f);
    s->rerank_nrm.assign(static_cast<size_t>(m) * K, 0.f);
    for (uint32_t d = 0; d < params_.dim; ++d)
        s->query_sq += query[d] * query[d];
    for (uint32_t sg = 0; sg < m; ++sg) {
        const float* qseg = query + static_cast<size_t>(sg) * sub;
        for (uint32_t c = 0; c < K; ++c) {
            const float* cw = cb + (static_cast<size_t>(sg) * K + c) * sub;
            float dsum = 0.f, nsum = 0.f;
            for (uint32_t j = 0; j < sub; ++j) {
                dsum += qseg[j] * cw[j];
                nsum += cw[j] * cw[j];
            }
            s->rerank_dot[static_cast<size_t>(sg) * K + c] = dsum;
            s->rerank_nrm[static_cast<size_t>(sg) * K + c] = nsum;
        }
    }
    return s;
}

void GlobalPqCoder::bind_leaf(ScanSetup&, const uint8_t*) const {}

void GlobalPqCoder::scan_leaf(const ScanSetup& setup, const uint8_t* leaf,
                              ScanSink& sink) {
    const auto& s = static_cast<const Setup&>(setup);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t count = lh->count;
    if (count == 0) return;

    const bool scan_8bit = (lh->pq_bits == 8);
    const uint32_t m = lh->m4;
    const uint32_t codes_per_block = scan_8bit ? 16 : 32;
    const uint32_t block_bytes = m * 16;
    const uint32_t n_blocks = (count + codes_per_block - 1) / codes_per_block;
    const uint8_t* codes = leaf + leaf_codes_offset(lh->summary_size);
    const uint8_t* lut_ptr = scan_8bit ? s.lut8.data() : s.lut4.data();

    // Valid-mask for the tail block (moved verbatim).
    const uint32_t full_blocks = count / codes_per_block;
    const uint32_t tail_count = count - full_blocks * codes_per_block;
    const uint32_t tail_mask = (tail_count == 0)
        ? (codes_per_block == 32 ? 0xFFFFFFFFu : 0x0000FFFFu)
        : (codes_per_block == 32
               ? (tail_count >= 32 ? 0xFFFFFFFFu : (1u << tail_count) - 1u)
               : (tail_count >= 16 ? 0xFFFFu : (1u << tail_count) - 1u));

    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = codes + static_cast<uint64_t>(b) * block_bytes;
        const uint32_t valid_mask = (b + 1 < n_blocks)
            ? (codes_per_block == 32 ? 0xFFFFFFFFu : 0xFFFFu)
            : tail_mask;

        if (scan_8bit) {
            uint32_t out[16];
            simd::fastscan_block16(blk, lut_ptr, m,
                                    static_cast<uint16_t>(valid_mask), out);
            const uint32_t base = b * 16;
            if (!sink.full()) {
                for (uint32_t j = 0; j < 16; ++j) {
                    if (out[j] == 0xFFFFFFFFu) continue;
                    sink.push(out[j], base + j);
                    if (sink.full()) break;
                }
                if (!sink.full()) continue;
            }
            const uint32_t block_min = u32_min16(out);
            if (block_min >= sink.front()) continue;
            for (uint32_t j = 0; j < 16; ++j) {
                if (out[j] == 0xFFFFFFFFu || out[j] >= sink.front()) continue;
                sink.replace(out[j], base + j);
            }
        } else {
            uint32_t out[32];
            simd::pq4_block32(blk, lut_ptr, m, out);
            if (valid_mask != 0xFFFFFFFFu)
                u32_mask_sentinel32(out, valid_mask);  // tail block
            const uint32_t base = b * 32;

            if (!sink.full()) {
                for (uint32_t j = 0; j < 32; ++j) {
                    if (out[j] == 0xFFFFFFFFu) continue;
                    sink.push(out[j], base + j);
                    if (sink.full()) break;
                }
                if (!sink.full()) continue;
            }
            if (u32_min32(out) >= sink.front()) continue;
            for (uint32_t j = 0; j < 32; ++j) {
                if (out[j] >= sink.front()) continue;
                sink.replace(out[j], base + j);
            }
        }
    }
}

float GlobalPqCoder::rerank(const float* query, const ScanSetup& setup,
                            const uint8_t* leaf, uint32_t local_idx,
                            float* scratch_decoded) {
    (void)query;
    (void)scratch_decoded;
    const auto& s = static_cast<const Setup&>(setup);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t codes_per_block = (lh->pq_bits == 8) ? 16 : 32;
    const uint32_t block_bytes = lh->m4 * 16;
    const uint32_t block = local_idx / codes_per_block;
    const uint32_t slot = local_idx % codes_per_block;
    const uint8_t* blk = leaf + leaf_codes_offset(lh->summary_size) +
        static_cast<uint64_t>(block) * block_bytes;
    const uint32_t m = lh->m4;
    const uint32_t K = quantizer_->K();
    float acc = 0.f, nacc = 0.f;
    if (lh->pq_bits == 8) {
        for (uint32_t sg = 0; sg < m; ++sg) {
            const uint32_t c = blk[sg * 16 + slot];
            acc += s.rerank_dot[static_cast<size_t>(sg) * K + c];
            nacc += s.rerank_nrm[static_cast<size_t>(sg) * K + c];
        }
    } else {
        const uint32_t byte_idx = slot % 16;
        const bool is_hi = slot >= 16;
        for (uint32_t sg = 0; sg < m; ++sg) {
            const uint8_t byte = blk[sg * 16 + byte_idx];
            const uint32_t c = is_hi ? (byte >> 4) : (byte & 0x0F);
            acc += s.rerank_dot[static_cast<size_t>(sg) * K + c];
            nacc += s.rerank_nrm[static_cast<size_t>(sg) * K + c];
        }
    }
    return (params_.metric == MetricKind::InnerProduct)
        ? -acc
        : (s.query_sq - 2.f * acc + nacc);
}

float GlobalPqCoder::rerank(const float* query, const uint8_t* leaf,
                            uint32_t local_idx, float* scratch_decoded) {
    // Setup-less fallback: extract + decode + exact distance (identical
    // value to the LUT path up to float summation order).
    std::vector<uint8_t> code(quantizer_->code_size());
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    extract_code_from_leaf(leaf, local_idx, lh->summary_size, lh->m4,
                           lh->pq_bits, (lh->pq_bits == 8) ? 16 : 32,
                           lh->m4 * 16, code.data());
    std::vector<float> dec(params_.dim);
    quantizer_->decode_code(code.data(), dec.data());
    if (scratch_decoded)
        std::memcpy(scratch_decoded, dec.data(), params_.dim * sizeof(float));
    return (params_.metric == MetricKind::InnerProduct)
        ? -simd::dot_f32(query, dec.data(), params_.dim)
        : simd::l2sq_f32(query, dec.data(), params_.dim);
}

void GlobalPqCoder::decode_one(const uint8_t* leaf, uint32_t local_idx,
                               float* out) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    std::vector<uint8_t> code(quantizer_->code_size());
    extract_code_from_leaf(leaf, local_idx, lh->summary_size, lh->m4,
                           lh->pq_bits, (lh->pq_bits == 8) ? 16 : 32,
                           lh->m4 * 16, code.data());
    quantizer_->decode_code(code.data(), out);
}

void GlobalPqCoder::extract_codes(const uint8_t* leaf, uint32_t count,
                                  uint8_t* codes) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t cpb = (lh->pq_bits == 8) ? 16 : 32;
    const uint32_t bb = lh->m4 * 16;
    const uint32_t code_size = quantizer_->code_size();
    for (uint32_t i = 0; i < count; ++i)
        extract_code_from_leaf(leaf, i, lh->summary_size, lh->m4, lh->pq_bits,
                               cpb, bb, codes + static_cast<size_t>(i) * code_size);
}

LeafCoder::SplitPlan GlobalPqCoder::plan_split(const uint8_t* /*leaf*/,
                                               const uint8_t* codes,
                                               const float* /*vecs*/,
                                               uint32_t count,
                                               uint32_t leaf_id) {
    SplitPlan plan;
    auto km = kmeans_pq(*quantizer_, codes, count, quantizer_->code_size(),
                        /*K=*/2, /*iterations=*/10, /*num_threads=*/1,
                        /*seed=*/0xDEADBEEFULL + leaf_id);
    for (uint32_t i = 0; i < count; ++i) {
        if (km.assign[i] == 0) plan.group0.push_back(i);
        else plan.group1.push_back(i);
    }
    auto decode_centroid_fp16 = [&](uint32_t k) -> std::vector<float16_t> {
        std::vector<float> f32(params_.dim);
        quantizer_->decode_code(km.centroids[k].data(), f32.data());
        std::vector<float16_t> fp16(params_.dim);
        cast_fp32_to_fp16(f32.data(), fp16.data(), params_.dim);
        return fp16;
    };
    plan.cent0_fp16 = decode_centroid_fp16(0);
    plan.cent1_fp16 = decode_centroid_fp16(1);
    return plan;
}

void GlobalPqCoder::encode_group(const GroupEncodeInput& in,
                                 uint8_t* leaf_out) {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_out);
    const uint32_t cpb = (lh->pq_bits == 4) ? 32 : 16;
    const uint32_t bb = lh->m4 * 16;
    const uint32_t code_size = quantizer_->code_size();
    const uint32_t new_nb = (in.count + cpb - 1) / cpb;
    uint8_t* ncb = leaf_out + leaf_codes_offset(lh->summary_size);
    for (uint32_t i = 0; i < in.count; ++i) {
        const uint32_t b = i / cpb;
        const uint32_t j = i % cpb;
        uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;
        write_code_to_fs_block(blk, j, lh->m4, lh->pq_bits, code_size,
                               in.src_codes + static_cast<size_t>(i) * code_size);
    }
    RowId* nrid = reinterpret_cast<RowId*>(
        leaf_out + leaf_rowids_offset(lh->summary_size, new_nb, bb));
    std::memcpy(nrid, in.row_ids, static_cast<size_t>(in.count) * sizeof(RowId));
}

void GlobalPqCoder::append_encode(const AppendInput& in) {
    const auto* olh = reinterpret_cast<const TreeLeafHeader*>(in.old_leaf);
    const uint32_t cpb = (olh->pq_bits == 4) ? 32 : 16;
    const uint32_t bb = olh->m4 * 16;
    const uint32_t old_nb = (in.old_count + cpb - 1) / cpb;
    const uint32_t code_size = quantizer_->code_size();
    uint8_t* ncb = in.new_leaf + leaf_codes_offset(olh->summary_size);
    std::memcpy(ncb, in.old_leaf + leaf_codes_offset(olh->summary_size),
                static_cast<size_t>(old_nb) * bb);
    for (uint32_t ai = 0; ai < in.n_vecs; ++ai) {
        const uint32_t gi = in.old_count + ai;
        const uint32_t b = gi / cpb;
        const uint32_t j = gi % cpb;
        uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;
        std::vector<uint8_t> code(code_size);
        quantizer_->encode(in.vecs[ai], code.data());
        write_code_to_fs_block(blk, j, olh->m4, olh->pq_bits, code_size,
                               code.data());
    }
}

void GlobalPqCoder::compact(const uint8_t* leaf_in, const uint32_t* keep_idx,
                            uint32_t keep_count, uint8_t* leaf_out) {
    const auto* olh = reinterpret_cast<const TreeLeafHeader*>(leaf_in);
    const uint32_t cpb = (olh->pq_bits == 4) ? 32 : 16;
    const uint32_t bb = olh->m4 * 16;
    const uint32_t m4 = olh->m4;
    const uint32_t summary_size = olh->summary_size;
    uint8_t* ncb = leaf_out + leaf_codes_offset(summary_size);
    for (uint32_t i = 0; i < keep_count; ++i) {
        const uint32_t src_slot = keep_idx[i];
        const uint32_t b = i / cpb;
        const uint32_t j = i % cpb;
        uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;
        const uint32_t old_b = src_slot / cpb;
        const uint32_t old_j = src_slot % cpb;
        const uint8_t* old_blk = leaf_in + leaf_codes_offset(summary_size)
            + static_cast<uint64_t>(old_b) * bb;
        if (olh->pq_bits == 8) {
            for (uint16_t s = 0; s < m4; ++s)
                blk[s * 16 + j] = old_blk[s * 16 + old_j];
        } else {
            const uint8_t old_byte_idx = old_j % 16;
            const bool old_hi = (old_j >= 16);
            const uint8_t new_byte_idx = j % 16;
            const bool new_hi = (j >= 16);
            for (uint16_t s = 0; s < m4; ++s) {
                const uint8_t nib = old_hi
                    ? static_cast<uint8_t>(old_blk[s * 16 + old_byte_idx] >> 4)
                    : static_cast<uint8_t>(old_blk[s * 16 + old_byte_idx] & 0x0F);
                if (new_hi) blk[s * 16 + new_byte_idx] |= static_cast<uint8_t>(nib << 4);
                else        blk[s * 16 + new_byte_idx] |= nib;
            }
        }
    }
}

}  // namespace sextant::tree
