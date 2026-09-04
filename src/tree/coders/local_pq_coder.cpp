#include "local_pq_coder.hpp"

#include "coder_util.hpp"
#include "quant/pq_quantizer.hpp"
#include "simd_kernels.hpp"

#include <cstring>

namespace sextant::tree {

using coders::extract_code_from_leaf;
using coders::lloyd_split_f32;
using coders::u32_min16;
using coders::u32_min32;
using coders::u32_mask_sentinel32;
using coders::write_code_to_fs_block;

namespace {
/// Cached LUT-builder wrapper + last-leaf pointer for the (per-query serial)
/// rerank path. Thread-local: search() runs on multiple query threads.
struct RerankCtx {
    std::unique_ptr<PqQuantizer> quant;
    const uint8_t* last_leaf = nullptr;
};
thread_local RerankCtx g_rerank_ctx;
}  // namespace

struct LocalPqCoder::Setup : public ScanSetup {
    std::vector<float> residual;
    std::vector<uint8_t> lut4, lut8;
    std::vector<float> query_copy;  // query for the residual transform
    std::unique_ptr<PqQuantizer> leaf_q;  // LUT builder (params fixed)
};

LocalPqCoder::LocalPqCoder(const CoderParams& params) : params_(params) {}
LocalPqCoder::~LocalPqCoder() = default;

LeafGeometry LocalPqCoder::geometry(const TreeLeafHeader* h) const {
    const uint32_t cpb = (h->pq_bits == 4) ? 32 : 16;
    const uint32_t bb = h->m4 * 16;
    const uint64_t n_blocks = (h->count + cpb - 1) / cpb;
    LeafGeometry g;
    g.codes_offset = local_codes_offset(h->summary_size, params_.dim, h->m4,
                                        h->pq_bits);
    g.rowids_offset = local_rowids_offset(h->summary_size, params_.dim, h->m4,
                                          h->pq_bits, n_blocks, bb);
    g.filter_offset =
        g.rowids_offset + static_cast<uint64_t>(h->count) * sizeof(RowId);
    g.extent_bytes = local_coded_extent_bytes(h->count, params_.dim, h->m4,
                                              h->pq_bits, h->summary_size, 0);
    return g;
}

uint64_t LocalPqCoder::extent_bytes(uint32_t count, uint32_t summary_size,
                                    uint64_t filter_cols_bytes) const {
    return local_coded_extent_bytes(count, params_.dim, params_.m4,
                                    params_.pq_bits, summary_size,
                                    filter_cols_bytes);
}

uint32_t LocalPqCoder::code_size() const {
    return static_cast<uint32_t>(
        (static_cast<uint32_t>(params_.m4) * params_.pq_bits + 7) / 8);
}

void LocalPqCoder::train(const float*, uint32_t) {
    // No global state: each leaf trains its own codebook at flush time.
}

bool LocalPqCoder::serialize_global(std::vector<uint8_t>&) const {
    return false;
}

bool LocalPqCoder::deserialize_global(const uint8_t*, uint64_t) {
    return false;
}

void LocalPqCoder::encode(const float*, uint8_t*, float*) {
    // Emission keeps the raw FP16 vectors; encoding happens per leaf at
    // flush time.
}

void LocalPqCoder::train_leaf_codebook(const float* vecs, uint32_t count,
                                       const float* centroid,
                                       uint8_t* codes) const {
    const uint16_t dim = params_.dim;
    const uint32_t cs = code_size();
    // For IP/cosine on normalized vectors, L2sq ranking == IP ranking; the
    // residual decomposition only holds for L2sq, so local codebooks always
    // train with L2sq regardless of the tree's metric.
    PqQuantizer local_q(MetricKind::L2Sq, dim, params_.m4, params_.pq_bits, 42);
    std::vector<float> residuals(static_cast<size_t>(count) * dim);
    for (uint32_t i = 0; i < count; ++i)
        for (uint16_t d = 0; d < dim; ++d)
            residuals[static_cast<size_t>(i) * dim + d] =
                vecs[static_cast<size_t>(i) * dim + d] - centroid[d];
    local_q.train(residuals.data(), count);
    for (uint32_t i = 0; i < count; ++i)
        local_q.encode(residuals.data() + static_cast<size_t>(i) * dim,
                       codes + static_cast<size_t>(i) * cs);
}

uint64_t LocalPqCoder::flush_leaf(const LeafFlushInput& in,
                                  uint8_t* leaf_out) {
    const uint16_t dim = params_.dim;
    const uint32_t cs = code_size();
    // Residuals from the raw FP16 vectors.
    std::vector<float> f32vecs(static_cast<size_t>(in.count) * dim);
    for (size_t i = 0; i < f32vecs.size(); ++i)
        f32vecs[i] = static_cast<float>(in.fp16_vecs[i]);
    std::vector<uint8_t> codes(static_cast<size_t>(in.count) * cs);
    PqQuantizer local_q(MetricKind::L2Sq, dim, params_.m4, params_.pq_bits, 42);
    {
        std::vector<float> residuals(static_cast<size_t>(in.count) * dim);
        for (uint32_t i = 0; i < in.count; ++i)
            for (uint16_t d = 0; d < dim; ++d)
                residuals[static_cast<size_t>(i) * dim + d] =
                    f32vecs[static_cast<size_t>(i) * dim + d] -
                    in.centroid_f32[d];
        local_q.train(residuals.data(), in.count);
        for (uint32_t i = 0; i < in.count; ++i)
            local_q.encode(residuals.data() + static_cast<size_t>(i) * dim,
                           codes.data() + static_cast<size_t>(i) * cs);
    }
    auto* lh = reinterpret_cast<TreeLeafHeader*>(leaf_out);
    lh->leaf_state = static_cast<uint8_t>(LeafState::CodedLocal);
    lh->centroid_offset =
        static_cast<uint32_t>(local_centroid_offset(in.summary_size));
    lh->codebook_offset =
        static_cast<uint32_t>(local_codebook_offset(in.summary_size, dim));
    std::memcpy(leaf_out + local_centroid_offset(in.summary_size),
                in.centroid_f32, dim * sizeof(float));
    const uint32_t K = local_q.K();
    const uint32_t sub_dim = local_q.sub_dim();
    const uint64_t cb_bytes =
        static_cast<uint64_t>(params_.m4) * K * sub_dim * sizeof(float);
    std::memcpy(leaf_out + local_codebook_offset(in.summary_size, dim),
                local_q.codebook(), cb_bytes);

    const uint32_t cpb = (params_.pq_bits == 4) ? 32 : 16;
    const uint32_t bb = params_.m4 * 16;
    const uint32_t n_blocks = (in.count + cpb - 1) / cpb;
    uint8_t* codes_out = leaf_out +
        local_codes_offset(in.summary_size, dim, params_.m4, params_.pq_bits);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint32_t base = b * cpb;
        uint8_t* blk = codes_out + static_cast<uint64_t>(b) * bb;
        std::memset(blk, 0, bb);
        for (uint32_t j = 0; j < cpb; ++j) {
            const uint32_t gi = base + j;
            if (gi >= in.count) break;
            write_code_to_fs_block(blk, j, params_.m4, params_.pq_bits, cs,
                                   codes.data() + gi * cs);
        }
    }
    return local_rowids_offset(in.summary_size, dim, params_.m4,
                               params_.pq_bits, n_blocks, bb);
}

std::unique_ptr<ScanSetup> LocalPqCoder::scan_setup(const float* query) {
    auto s = std::make_unique<Setup>();
    s->residual.assign(params_.dim, 0.f);
    s->query_copy.assign(query, query + params_.dim);
    if (params_.pq_bits == 8)
        s->lut8.assign(static_cast<size_t>(params_.m4) * 256, 0);
    else
        s->lut4.assign(static_cast<size_t>(params_.m4) * 16, 0);
    s->leaf_q = std::make_unique<PqQuantizer>(
        MetricKind::L2Sq, params_.dim, params_.m4, params_.pq_bits);
    return s;
}

void LocalPqCoder::bind_leaf(ScanSetup& setup, const uint8_t* leaf) const {
    auto& s = static_cast<Setup&>(setup);
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const float* centroid = reinterpret_cast<const float*>(
        leaf + lh->centroid_offset);
    const float* query = s.query_copy.data();
    for (uint32_t d = 0; d < params_.dim; ++d)
        s.residual[d] = query[d] - centroid[d];
    const float* codebook = reinterpret_cast<const float*>(
        leaf + lh->codebook_offset);
    s.leaf_q->set_codebook_data(codebook);
    if (params_.pq_bits == 8) {
        float lut_scale = 0, lut_offset = 0;
        s.leaf_q->build_fastscan_lut(s.residual.data(), s.lut8.data(),
                                     &lut_scale, &lut_offset);
    } else {
        s.leaf_q->build_fastscan_lut4(s.residual.data(), s.lut4.data(),
                                      nullptr);
    }
}

void LocalPqCoder::scan_leaf(const ScanSetup& setup, const uint8_t* leaf,
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
    const uint8_t* codes = leaf + local_codes_offset(lh->summary_size,
        params_.dim, lh->m4, lh->pq_bits);
    const uint8_t* lut_ptr = scan_8bit ? s.lut8.data() : s.lut4.data();

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
                u32_mask_sentinel32(out, valid_mask);
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

float LocalPqCoder::rerank(const float* query, const uint8_t* leaf,
                           uint32_t local_idx, float* scratch_decoded) {
    auto& ctx = g_rerank_ctx;
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    if (ctx.last_leaf != leaf || !ctx.quant) {
        ctx.last_leaf = leaf;
        if (!ctx.quant)
            ctx.quant = std::make_unique<PqQuantizer>(
                MetricKind::L2Sq, params_.dim, params_.m4, params_.pq_bits);
        ctx.quant->set_codebook_data(reinterpret_cast<const float*>(
            leaf + lh->codebook_offset));
    }
    const uint64_t codes_off = local_codes_offset(
        lh->summary_size, params_.dim, lh->m4, lh->pq_bits);
    const uint32_t cpb = (lh->pq_bits == 8) ? 16 : 32;
    uint8_t code[64];
    extract_code_from_leaf(leaf, local_idx, lh->summary_size, lh->m4,
                           lh->pq_bits, cpb, lh->m4 * 16, code, codes_off);
    std::vector<float> dec(params_.dim);
    ctx.quant->decode_code(code, dec.data());
    const float* centroid = reinterpret_cast<const float*>(
        leaf + lh->centroid_offset);
    for (uint32_t d = 0; d < params_.dim; ++d) dec[d] += centroid[d];
    if (scratch_decoded)
        std::memcpy(scratch_decoded, dec.data(), params_.dim * sizeof(float));
    return (params_.metric == MetricKind::InnerProduct)
        ? -simd::dot_f32(query, dec.data(), params_.dim)
        : simd::l2sq_f32(query, dec.data(), params_.dim);
}

void LocalPqCoder::decode_one(const uint8_t* leaf, uint32_t local_idx,
                              float* out) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t cpb = (lh->pq_bits == 8) ? 16 : 32;
    const uint32_t cs = code_size();
    const uint64_t codes_off = local_codes_offset(lh->summary_size,
        params_.dim, lh->m4, lh->pq_bits);
    const uint8_t* blk = leaf + codes_off +
        static_cast<uint64_t>(local_idx / cpb) * (lh->m4 * 16);
    const uint32_t slot = local_idx % cpb;
    std::vector<uint8_t> code(cs, 0);
    for (uint16_t sg = 0; sg < lh->m4; ++sg) {
        uint32_t cid;
        if (lh->pq_bits == 8) {
            cid = blk[sg * 16 + slot];
        } else {
            const uint32_t byte_idx = slot % 16;
            const bool is_hi = slot >= 16;
            const uint8_t byte = blk[sg * 16 + byte_idx];
            cid = is_hi ? (byte >> 4) : (byte & 0x0F);
        }
        const uint32_t byte_off = sg / 2;
        const uint8_t shift = static_cast<uint8_t>((sg % 2) * 4);
        code[byte_off] = static_cast<uint8_t>(
            ((sg % 2 == 0) ? (code[byte_off] & 0xF0u)
                           : (code[byte_off] & 0x0Fu)) |
            ((cid & 0x0Fu) << shift));
    }
    PqQuantizer leaf_q(MetricKind::L2Sq, params_.dim, params_.m4,
                       lh->pq_bits, 0);
    leaf_q.set_codebook_data(reinterpret_cast<const float*>(
        leaf + lh->codebook_offset));
    const float* centroid = reinterpret_cast<const float*>(
        leaf + lh->centroid_offset);
    leaf_q.decode_code(code.data(), out);
    for (uint32_t d = 0; d < params_.dim; ++d) out[d] += centroid[d];
}

void LocalPqCoder::extract_codes(const uint8_t* leaf, uint32_t count,
                                 uint8_t* codes) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
    const uint32_t cpb = (lh->pq_bits == 8) ? 16 : 32;
    const uint32_t cs = code_size();
    const uint64_t codes_off = local_codes_offset(lh->summary_size,
        params_.dim, lh->m4, lh->pq_bits);
    for (uint32_t i = 0; i < count; ++i)
        extract_code_from_leaf(leaf, i, lh->summary_size, lh->m4, lh->pq_bits,
                               cpb, lh->m4 * 16,
                               codes + static_cast<size_t>(i) * cs, codes_off);
}

LeafCoder::SplitPlan LocalPqCoder::plan_split(const uint8_t*,
                                              const uint8_t* codes,
                                              const float* vecs,
                                              uint32_t count, uint32_t) {
    (void)codes;
    SplitPlan plan;
    lloyd_split_f32(vecs, count, params_.dim, plan.group0, plan.group1,
                    plan.cent0_fp16, plan.cent1_fp16);
    return plan;
}

void LocalPqCoder::encode_group(const GroupEncodeInput& in,
                                uint8_t* leaf_out) {
    const uint16_t dim = params_.dim;
    const uint32_t cs = code_size();
    const uint32_t cpb = (params_.pq_bits == 4) ? 32 : 16;
    const uint32_t bb = params_.m4 * 16;
    auto* lh = reinterpret_cast<TreeLeafHeader*>(leaf_out);
    const uint32_t gnb = (in.count + cpb - 1) / cpb;

    // CodedLocal leaf with a RETRAINED per-half codebook: the split changes
    // the centroid, so residuals change — retrain + re-encode.
    std::vector<float> cent(dim, 0.f);
    for (uint32_t i = 0; i < in.count; ++i)
        for (uint32_t d = 0; d < dim; ++d)
            cent[d] += in.vecs[static_cast<size_t>(i) * dim + d];
    for (uint32_t d = 0; d < dim; ++d) cent[d] /= in.count;

    PqQuantizer local_q(MetricKind::L2Sq, dim, params_.m4, params_.pq_bits, 42);
    std::vector<float> residuals(static_cast<size_t>(in.count) * dim);
    for (uint32_t i = 0; i < in.count; ++i)
        for (uint32_t d = 0; d < dim; ++d)
            residuals[static_cast<size_t>(i) * dim + d] =
                in.vecs[static_cast<size_t>(i) * dim + d] - cent[d];
    local_q.train(residuals.data(), in.count);
    std::vector<uint8_t> gcodes(static_cast<size_t>(in.count) * cs);
    for (uint32_t i = 0; i < in.count; ++i)
        local_q.encode(residuals.data() + static_cast<size_t>(i) * dim,
                       gcodes.data() + static_cast<size_t>(i) * cs);

    lh->leaf_state = static_cast<uint8_t>(LeafState::CodedLocal);
    lh->centroid_offset =
        static_cast<uint32_t>(local_centroid_offset(in.summary_size));
    lh->codebook_offset =
        static_cast<uint32_t>(local_codebook_offset(in.summary_size, dim));
    std::memcpy(leaf_out + local_centroid_offset(in.summary_size),
                cent.data(), dim * sizeof(float));
    std::memcpy(leaf_out + local_codebook_offset(in.summary_size, dim),
                local_q.codebook(),
                static_cast<size_t>(params_.m4) * local_q.K() *
                    local_q.sub_dim() * sizeof(float));

    uint8_t* ncb = leaf_out +
        local_codes_offset(in.summary_size, dim, params_.m4, params_.pq_bits);
    for (uint32_t b = 0; b < gnb; ++b) {
        const uint32_t gbase = b * cpb;
        uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;
        std::memset(blk, 0, bb);
        for (uint32_t j = 0; j < cpb; ++j) {
            const uint32_t gi = gbase + j;
            if (gi >= in.count) break;
            write_code_to_fs_block(blk, j, params_.m4, params_.pq_bits, cs,
                                   gcodes.data() + gi * cs);
        }
    }
    RowId* nrid = reinterpret_cast<RowId*>(
        leaf_out + local_rowids_offset(in.summary_size, dim, params_.m4,
                                       params_.pq_bits, gnb, bb));
    std::memcpy(nrid, in.row_ids, static_cast<size_t>(in.count) * sizeof(RowId));
}

void LocalPqCoder::append_encode(const AppendInput& in) {
    const auto* olh = reinterpret_cast<const TreeLeafHeader*>(in.old_leaf);
    const uint16_t dim = params_.dim;
    const uint32_t cs = code_size();
    const uint32_t cpb = (olh->pq_bits == 4) ? 32 : 16;
    const uint32_t bb = olh->m4 * 16;
    const uint32_t old_nb = (in.old_count + cpb - 1) / cpb;
    const uint64_t codes_off =
        local_codes_offset(olh->summary_size, dim, olh->m4, olh->pq_bits);
    // Preserve centroid + codebook (frozen — insert never retrains).
    std::memcpy(in.new_leaf + local_centroid_offset(olh->summary_size),
                in.old_leaf + local_centroid_offset(olh->summary_size),
                codes_off - local_centroid_offset(olh->summary_size));
    uint8_t* ncb = in.new_leaf + codes_off;
    std::memcpy(ncb, in.old_leaf + codes_off, static_cast<size_t>(old_nb) * bb);

    PqQuantizer leaf_q(MetricKind::L2Sq, dim, olh->m4, olh->pq_bits, 0);
    leaf_q.set_codebook_data(reinterpret_cast<const float*>(
        in.old_leaf + olh->codebook_offset));
    const float* centroid = reinterpret_cast<const float*>(
        in.old_leaf + olh->centroid_offset);
    std::vector<float> residual(dim);
    std::vector<uint8_t> code(cs);
    for (uint32_t ai = 0; ai < in.n_vecs; ++ai) {
        const uint32_t gi = in.old_count + ai;
        const uint32_t b = gi / cpb;
        const uint32_t j = gi % cpb;
        uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;
        for (uint32_t d = 0; d < dim; ++d)
            residual[d] = in.vecs[ai][d] - centroid[d];
        leaf_q.encode(residual.data(), code.data());
        write_code_to_fs_block(blk, j, olh->m4, olh->pq_bits, cs, code.data());
    }
}

}  // namespace sextant::tree
