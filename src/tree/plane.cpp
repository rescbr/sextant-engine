#include "plane.hpp"

#include "sextant/types.hpp"
#include "util/fp16.hpp"
#include "simd_kernels.hpp"

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace sextant::tree {

namespace {

constexpr uint32_t kPlaneMagic = 0x314C5053u;  // 'SPL1'
constexpr uint32_t kPlaneVersion = 1;
constexpr uint32_t kVecsPerBlock = 32;

#pragma pack(push, 1)
struct PlaneHeaderDisk {
    uint32_t magic;
    uint32_t version;
    uint8_t  encoding;
    uint8_t  flags;      // bit0: alpha present
    uint16_t rank;
    uint32_t dim;
    uint64_t basis_off, basis_bytes;
    uint64_t mean_off, mean_bytes;
    uint64_t codebook_off, codebook_bytes;
    uint64_t alpha_off, alpha_bytes;
    uint64_t blocks_off, blocks_bytes;
    uint64_t n_blocks;
    uint64_t reserved[3];  // fill to 128 bytes
};
#pragma pack(pop)
static_assert(sizeof(PlaneHeaderDisk) == 128, "plane header must be 128 B");

uint32_t fs_segments(PlaneEncoding enc, uint16_t rank) {
    return enc == PlaneEncoding::B1G ? rank / 4 : rank;
}

uint32_t block_bytes_for(const PlaneMeta& m) {
    // Both encodings are FastScan [m][16] nibble-interleaved:
    // u4lm m = rank, b1g m = rank/4 (sign nibbles). 64/16 B per vec.
    if (m.encoding == PlaneEncoding::B1G) return (m.rank / 4) * 16;
    return m.rank * 16;
}

}  // namespace

uint32_t PlaneMeta::bytes_per_vec() const {
    switch (encoding) {
        case PlaneEncoding::U4LM:    return (rank * 4 + 7) / 8;
        case PlaneEncoding::B1G:     return (rank + 7) / 8;
        case PlaneEncoding::U4LM_PV: return (rank * 4 + 7) / 8 + 2;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// PlaneWriter
// ---------------------------------------------------------------------------

void PlaneWriter::train(const float* base, uint32_t n, uint32_t dim,
                        const Config& cfg) {
    meta_ = PlaneMeta{};
    meta_.encoding = cfg.encoding;
    meta_.rank = cfg.rank;
    meta_.dim = dim;
    cfg_ = cfg;
    block_bytes_ = block_bytes_for(meta_);

    const uint32_t R = cfg.rank;
    const uint32_t SAMPLE = std::min<uint32_t>(cfg.train_rows, n);
    // Spread-sampled training rows (clustered-order prefixes train on a
    // single cluster — the dbpedia lesson: -12pp containment).
    const uint32_t stride = std::max(1u, n / SAMPLE);

    mean_.assign(dim, 0.0);
    for (uint32_t i = 0; i < SAMPLE; ++i)
        for (uint32_t d = 0; d < dim; ++d)
            mean_[d] += base[static_cast<size_t>(i) * stride * dim + d] / SAMPLE;
    std::vector<double> cov(static_cast<size_t>(dim) * dim, 0.0);
    for (uint32_t i = 0; i < SAMPLE; ++i) {
        const float* v = &base[static_cast<size_t>(i) * stride * dim];
        for (uint32_t r = 0; r < dim; ++r) {
            const double vr = v[r] - mean_[r];
            for (uint32_t c = r; c < dim; ++c)
                cov[static_cast<size_t>(r) * dim + c] +=
                    vr * (v[c] - mean_[c]) / SAMPLE;
        }
    }
    for (uint32_t r = 0; r < dim; ++r)
        for (uint32_t c = 0; c < r; ++c)
            cov[static_cast<size_t>(r) * dim + c] =
                cov[static_cast<size_t>(c) * dim + r];

    std::vector<std::vector<double>> basis(R);
    std::vector<double> resid = cov;
    for (uint32_t e = 0; e < R; ++e) {
        std::vector<double> v(dim, 1.0 / std::sqrt((double)dim));
        for (int it = 0; it < 40; ++it) {
            std::vector<double> w(dim, 0.0);
            for (uint32_t r = 0; r < dim; ++r) {
                double acc = 0;
                for (uint32_t c = 0; c < dim; ++c)
                    acc += resid[static_cast<size_t>(r) * dim + c] * v[c];
                w[r] = acc;
            }
            double nrm = 1e-30;
            for (uint32_t d = 0; d < dim; ++d) nrm += w[d] * w[d];
            nrm = std::sqrt(nrm);
            for (uint32_t d = 0; d < dim; ++d) v[d] = w[d] / nrm;
        }
        basis[e] = v;
        double lambda = 0;
        for (uint32_t r = 0; r < dim; ++r)
            for (uint32_t c = 0; c < dim; ++c)
                lambda += v[r] * resid[static_cast<size_t>(r) * dim + c] * v[c];
        for (uint32_t r = 0; r < dim; ++r)
            for (uint32_t c = 0; c < dim; ++c)
                resid[static_cast<size_t>(r) * dim + c] -= lambda * v[r] * v[c];
    }

    basis_t_.assign(static_cast<size_t>(dim) * R, 0.0f);
    for (uint32_t d = 0; d < dim; ++d)
        for (uint32_t e = 0; e < R; ++e)
            basis_t_[static_cast<size_t>(d) * R + e] =
                static_cast<float>(basis[e][d]);

    // --- codebook training on the projected sample ---
    std::vector<float> proj(static_cast<size_t>(SAMPLE) * R);
    for (uint32_t i = 0; i < SAMPLE; ++i) {
        const float* v = &base[static_cast<size_t>(i) * stride * dim];
        float* out = &proj[static_cast<size_t>(i) * R];
        for (uint32_t e = 0; e < R; ++e) out[e] = 0;
        for (uint32_t d = 0; d < dim; ++d) {
            const float vd = v[d] - static_cast<float>(mean_[d]);
            const float* brow = &basis_t_[static_cast<size_t>(d) * R];
            for (uint32_t e = 0; e < R; ++e) out[e] += vd * brow[e];
        }
    }
    std::vector<float> vmin(R, std::numeric_limits<float>::max());
    std::vector<float> vmax(R, std::numeric_limits<float>::lowest());
    std::vector<double> vabs(R, 0.0);
    for (uint32_t i = 0; i < SAMPLE; ++i)
        for (uint32_t e = 0; e < R; ++e) {
            const float x = proj[static_cast<size_t>(i) * R + e];
            vmin[e] = std::min(vmin[e], x);
            vmax[e] = std::max(vmax[e], x);
            vabs[e] += std::fabs(x);
        }
    for (uint32_t e = 0; e < R; ++e) vabs[e] /= SAMPLE;

    if (meta_.encoding == PlaneEncoding::B1G) {
        codebook_.assign(R, 0.0f);
        for (uint32_t e = 0; e < R; ++e)
            codebook_[e] = static_cast<float>(vabs[e]);
        return;
    }

    // Lloyd-Max per-dim (histogram iteration), u4lm / u4lm_pv.
    const uint32_t LV = 16;
    codebook_.assign(static_cast<size_t>(R) * LV, 0.0f);
    const int HB = static_cast<int>(cfg.lm_hist_bins);
    std::vector<float> hist(static_cast<size_t>(HB) * R, 0.0f);
    for (uint32_t e = 0; e < R; ++e) {
        const float lo = vmin[e], hi = vmax[e];
        const float span = std::max(1e-9f, hi - lo);
        for (uint32_t i = 0; i < SAMPLE; ++i) {
            const float t =
                (proj[static_cast<size_t>(i) * R + e] - lo) / span;
            const int b = static_cast<int>(t * HB);
            hist[static_cast<size_t>(e) * HB + std::clamp(b, 0, HB - 1)] += 1;
        }
    }
    for (uint32_t e = 0; e < R; ++e) {
        float* cent = &codebook_[static_cast<size_t>(e) * LV];
        const float lo = vmin[e], hi = vmax[e];
        for (uint32_t c = 0; c < LV; ++c)
            cent[c] = lo + (c + 0.5f) * (hi - lo) / LV;
        for (uint32_t it = 0; it < cfg.lm_iters; ++it) {
            double cs[LV], cw[LV];
            std::memset(cs, 0, sizeof(cs));
            std::memset(cw, 0, sizeof(cw));
            for (int b = 0; b < HB; ++b) {
                const float w = hist[static_cast<size_t>(e) * HB + b];
                if (w <= 0) continue;
                const float x = lo + (b + 0.5f) * (hi - lo) / HB;
                uint32_t best = 0;
                float bd = std::numeric_limits<float>::max();
                for (uint32_t c = 0; c < LV; ++c) {
                    const float d = std::fabs(x - cent[c]);
                    if (d < bd) { bd = d; best = c; }
                }
                cs[best] += static_cast<double>(w) * x;
                cw[best] += static_cast<double>(w);
            }
            for (uint32_t c = 0; c < LV; ++c)
                if (cw[c] > 0) cent[c] = static_cast<float>(cs[c] / cw[c]);
        }
    }
}

uint32_t PlaneWriter::encode_dim_(float v, uint32_t e) const {
    if (meta_.encoding == PlaneEncoding::B1G) return v >= 0 ? 1 : 0;
    const float* cent = &codebook_[static_cast<size_t>(e) * 16];
    uint32_t best = 0;
    float bd = std::numeric_limits<float>::max();
    for (uint32_t c = 0; c < 16; ++c) {
        const float d = std::fabs(v - cent[c]);
        if (d < bd) { bd = d; best = c; }
    }
    return best;
}

float PlaneWriter::decode_dim_(uint32_t code, uint32_t e) const {
    if (meta_.encoding == PlaneEncoding::B1G)
        return (code ? codebook_[e] : -codebook_[e]);
    return codebook_[static_cast<size_t>(e) * 16 + code];
}

void PlaneWriter::prepare(uint32_t n_leaves) { leaves_.resize(n_leaves); }

void PlaneWriter::project(const float* vec, float* pr) const {
    const uint32_t R = meta_.rank;
    for (uint32_t e = 0; e < R; ++e) pr[e] = 0;
    for (uint32_t d = 0; d < meta_.dim; ++d) {
        const float vd = vec[d] - static_cast<float>(mean_[d]);
        const float* brow = &basis_t_[static_cast<size_t>(d) * R];
        for (uint32_t e = 0; e < R; ++e) pr[e] += vd * brow[e];
    }
}

void PlaneWriter::encode_from_proj(uint32_t leaf_id, uint32_t slot,
                                   const float* pr) {
    LeafState& ls = leaves_[leaf_id];
    const uint32_t block = slot / kVecsPerBlock;
    const uint32_t lane = slot % kVecsPerBlock;
    if (ls.n_blocks <= block) {
        ls.n_blocks = block + 1;
        ls.blocks.resize(static_cast<size_t>(ls.n_blocks) * block_bytes_, 0);
    }
    if (ls.proj.size() < static_cast<size_t>(slot + 1) * meta_.rank)
        ls.proj.resize(static_cast<size_t>(slot + 1) * meta_.rank);
    std::memcpy(&ls.proj[static_cast<size_t>(slot) * meta_.rank], pr,
                meta_.rank * sizeof(float));

    uint8_t* blk = &ls.blocks[static_cast<size_t>(block) * block_bytes_];
    const uint8_t nbi = static_cast<uint8_t>(lane % 16);
    const bool hi = lane >= 16;
    if (meta_.encoding == PlaneEncoding::B1G) {
        // b1g IS 4-bit PQ over sign groups: each nibble packs 4 sign
        // bits (dims 4g..4g+3 of group g), so the FastScan [m][16]
        // interleave with m = rank/4 applies and the pq4_scan_many
        // kernel scores b1g blocks verbatim. Same 16 B/vec.
        for (uint32_t g = 0; g < meta_.rank / 4; ++g) {
            uint8_t nib = 0;
            for (uint32_t t = 0; t < 4; ++t)
                if (pr[4 * g + t] >= 0) nib |= static_cast<uint8_t>(1u << t);
            if (hi) blk[g * 16 + nbi] = static_cast<uint8_t>(
                (blk[g * 16 + nbi] & 0x0Fu) | (nib << 4));
            else    blk[g * 16 + nbi] = static_cast<uint8_t>(
                (blk[g * 16 + nbi] & 0xF0u) | nib);
        }
    } else {
        // u4lm: FastScan [m][16] nibble interleave (same as leaf codes):
        // lane j < 16 -> low nibble of blk[s*16 + j], j >= 16 -> high.
        for (uint32_t s = 0; s < meta_.rank; ++s) {
            const uint8_t nib =
                static_cast<uint8_t>(encode_dim_(pr[s], s));
            if (hi) blk[s * 16 + nbi] = static_cast<uint8_t>(
                (blk[s * 16 + nbi] & 0x0Fu) | (nib << 4));
            else    blk[s * 16 + nbi] = static_cast<uint8_t>(
                (blk[s * 16 + nbi] & 0xF0u) | nib);
        }
    }
}

void PlaneWriter::encode_member(uint32_t leaf_id, uint32_t slot,
                                const float* vec, float* proj_out) {
    static thread_local std::vector<float> pr;
    pr.resize(meta_.rank);
    project(vec, pr.data());
    if (proj_out) std::memcpy(proj_out, pr.data(),
                              meta_.rank * sizeof(float));
    encode_from_proj(leaf_id, slot, pr.data());
}

std::vector<uint8_t> PlaneWriter::finalize(
    const std::vector<uint32_t>& leaf_counts) {
    if (leaves_.size() != leaf_counts.size())
        throw std::runtime_error("plane: leaf_counts != prepared leaves");
    const uint32_t R = meta_.rank;
    const bool pv = meta_.encoding == PlaneEncoding::U4LM_PV;

    uint64_t n_blocks = 0;
    for (uint32_t c : leaf_counts)
        n_blocks += (c + kVecsPerBlock - 1) / kVecsPerBlock;

    const size_t header = sizeof(PlaneHeaderDisk);
    const uint64_t basis_bytes = static_cast<uint64_t>(meta_.dim) * R * 4;
    const uint64_t mean_bytes = static_cast<uint64_t>(meta_.dim) * 4;
    const uint64_t cb_bytes = static_cast<uint64_t>(codebook_.size()) * 4;
    uint64_t alpha_bytes = 0;
    if (pv)
        for (uint32_t c : leaf_counts) alpha_bytes +=
            static_cast<uint64_t>(c) * 2;
    const uint64_t blocks_bytes = n_blocks * block_bytes_;

    std::vector<uint8_t> blob(static_cast<size_t>(
        header + basis_bytes + mean_bytes + cb_bytes + alpha_bytes +
        blocks_bytes), 0);

    auto* h = reinterpret_cast<PlaneHeaderDisk*>(blob.data());
    h->magic = kPlaneMagic;
    h->version = kPlaneVersion;
    h->encoding = static_cast<uint8_t>(meta_.encoding);
    h->flags = pv ? 1 : 0;
    h->rank = static_cast<uint16_t>(R);
    h->dim = meta_.dim;
    h->basis_off = header;  h->basis_bytes = basis_bytes;
    h->mean_off = header + basis_bytes;  h->mean_bytes = mean_bytes;
    h->codebook_off = h->mean_off + mean_bytes;  h->codebook_bytes = cb_bytes;
    h->alpha_off = h->codebook_off + cb_bytes;  h->alpha_bytes = alpha_bytes;
    h->blocks_off = h->alpha_off + alpha_bytes;  h->blocks_bytes = blocks_bytes;
    h->n_blocks = n_blocks;

    std::memcpy(blob.data() + h->basis_off, basis_t_.data(), basis_bytes);
    std::memcpy(blob.data() + h->mean_off, mean_.data(), mean_bytes);
    std::memcpy(blob.data() + h->codebook_off, codebook_.data(), cb_bytes);

    // Per-dim code decode for the alpha pass (reads the encoded blocks).
    auto decode_slot = [&](const LeafState& ls, uint32_t slot,
                           uint32_t s) -> float {
        const uint32_t block = slot / kVecsPerBlock;
        const uint32_t lane = slot % kVecsPerBlock;
        const uint8_t* blk =
            &ls.blocks[static_cast<size_t>(block) * block_bytes_];
        if (meta_.encoding == PlaneEncoding::B1G) {
            const uint32_t g = s / 4, t = s % 4;
            const uint8_t nbi2 = static_cast<uint8_t>(lane % 16);
            const uint8_t byte2 = blk[g * 16 + nbi2];
            const uint32_t nib = (lane >= 16) ? (byte2 >> 4) & 0xFu
                                              : byte2 & 0xFu;
            return decode_dim_((nib >> t) & 1u, s);
        }
        const uint8_t nbi = static_cast<uint8_t>(lane % 16);
        const uint32_t code = (lane >= 16)
            ? static_cast<uint32_t>(blk[s * 16 + nbi] >> 4)
            : static_cast<uint32_t>(blk[s * 16 + nbi] & 0xFu);
        return decode_dim_(code, s);
    };

    uint8_t* alpha_cur = blob.data() + h->alpha_off;
    uint8_t* blocks_cur = blob.data() + h->blocks_off;
    for (uint32_t l = 0; l < leaf_counts.size(); ++l) {
        const LeafState& ls = leaves_[l];
        const uint32_t nb =
            (leaf_counts[l] + kVecsPerBlock - 1) / kVecsPerBlock;
        if (ls.n_blocks >= nb) {
            std::memcpy(blocks_cur, ls.blocks.data(),
                        static_cast<size_t>(nb) * block_bytes_);
        }
        blocks_cur += static_cast<size_t>(nb) * block_bytes_;
        if (!pv) continue;
        for (uint32_t i = 0; i < leaf_counts[l]; ++i) {
            const float* v = &ls.proj[static_cast<size_t>(i) * R];
            double dvc = 0, dcc = 0;
            for (uint32_t s = 0; s < R; ++s) {
                const float dc = decode_slot(ls, i, s);
                dvc += static_cast<double>(v[s]) * dc;
                dcc += static_cast<double>(dc) * dc;
            }
            const float a = dcc > 1e-12
                ? static_cast<float>(dvc / dcc) : 1.0f;
            const float16_t h16 = static_cast<float16_t>(a);
            std::memcpy(alpha_cur, &h16, 2);
            alpha_cur += 2;
        }
    }
    return blob;
}

// ---------------------------------------------------------------------------
// PlaneIndex
// ---------------------------------------------------------------------------

bool parse_plane_header(const uint8_t* data, size_t len, PlaneMeta* meta,
                        uint64_t* blocks_bytes, uint64_t* codebook_bytes,
                        uint64_t* alpha_bytes) {
    if (len < sizeof(PlaneHeaderDisk)) return false;
    PlaneHeaderDisk h;
    std::memcpy(&h, data, sizeof(h));
    if (h.magic != kPlaneMagic || h.version != kPlaneVersion) return false;
    if (h.encoding == 0 || h.encoding > 3) return false;
    if (h.rank == 0) return false;
    if (meta) {
        meta->encoding = static_cast<PlaneEncoding>(h.encoding);
        meta->rank = h.rank;
        meta->dim = h.dim;
    }
    if (blocks_bytes) *blocks_bytes = h.blocks_bytes;
    if (codebook_bytes) *codebook_bytes = h.codebook_bytes;
    if (alpha_bytes) *alpha_bytes = h.alpha_bytes;
    return true;
}

std::unique_ptr<PlaneIndex> PlaneIndex::parse(const uint8_t* data,
                                              size_t len) {
    PlaneHeaderDisk h;
    if (len < sizeof(h)) return nullptr;
    std::memcpy(&h, data, sizeof(h));
    if (h.magic != kPlaneMagic || h.version != kPlaneVersion) return nullptr;
    auto p = std::unique_ptr<PlaneIndex>(new PlaneIndex());
    p->meta_.encoding = static_cast<PlaneEncoding>(h.encoding);
    p->meta_.rank = h.rank;
    p->meta_.dim = h.dim;
    p->block_bytes_ = block_bytes_for(p->meta_);

    const auto in_range = [&](uint64_t off, uint64_t bytes) {
        return off >= sizeof(h) && off + bytes <= len;
    };
    if (h.basis_bytes != static_cast<uint64_t>(h.dim) * h.rank * 4 ||
        h.mean_bytes != static_cast<uint64_t>(h.dim) * 4 ||
        h.codebook_bytes != ((p->meta_.encoding == PlaneEncoding::B1G)
            ? static_cast<uint64_t>(h.rank) * 4
            : static_cast<uint64_t>(h.rank) * 16 * 4) ||
        h.blocks_bytes != h.n_blocks * p->block_bytes_)
        return nullptr;
    if (!in_range(h.basis_off, h.basis_bytes) ||
        !in_range(h.mean_off, h.mean_bytes) ||
        !in_range(h.codebook_off, h.codebook_bytes) ||
        !in_range(h.blocks_off, h.blocks_bytes))
        return nullptr;
    if (p->meta_.encoding == PlaneEncoding::U4LM_PV) {
        if (!in_range(h.alpha_off, h.alpha_bytes)) return nullptr;
        p->alpha_ = reinterpret_cast<const uint16_t*>(data + h.alpha_off);
    } else if (h.alpha_bytes != 0) {
        return nullptr;
    }

    p->basis_t_ = reinterpret_cast<const float*>(data + h.basis_off);
    p->mean_ = reinterpret_cast<const float*>(data + h.mean_off);
    p->codebook_ = reinterpret_cast<const float*>(data + h.codebook_off);
    p->blocks_ = data + h.blocks_off;
    return p;
}

void PlaneIndex::bind(const std::vector<uint32_t>& leaf_counts) {
    const size_t n = leaf_counts.size();
    block_off_.assign(n, 0);
    block_cnt_.assign(n, 0);
    leaf_cnt_.assign(leaf_counts.begin(), leaf_counts.end());
    uint64_t off = 0;
    for (size_t l = 0; l < n; ++l) {
        block_cnt_[l] = (leaf_counts[l] + kVecsPerBlock - 1) / kVecsPerBlock;
        block_off_[l] = off;
        off += static_cast<uint64_t>(block_cnt_[l]) * block_bytes_;
    }
}

void PlaneIndex::project_query(const float* q, float* proj_out) const {
    const uint32_t R = meta_.rank;
    for (uint32_t e = 0; e < R; ++e) proj_out[e] = 0;
    for (uint32_t d = 0; d < meta_.dim; ++d) {
        const float qd = q[d] - mean_[d];
        const float* brow = &basis_t_[static_cast<size_t>(d) * R];
        for (uint32_t e = 0; e < R; ++e) proj_out[e] += qd * brow[e];
    }
}

void PlaneIndex::build_lut(const float* proj, float* lut) const {
    const uint32_t R = meta_.rank;
    for (uint32_t s = 0; s < R; ++s) {
        const float* cent = &codebook_[static_cast<size_t>(s) * 16];
        for (uint32_t c = 0; c < 16; ++c)
            lut[static_cast<size_t>(s) * 16 + c] = proj[s] * cent[c];
    }
}

void PlaneIndex::build_sign_ctx(const float* proj, float* w_out,
                                uint32_t* qbits_out) const {
    // w_out doubles as the byte-LUT staging area is NOT used; see
    // build_b1_lut. Kept: fills per-dim weights (used by diagnostics).
    const uint32_t R = meta_.rank;
    for (uint32_t e = 0; e < R; ++e) {
        w_out[e] = proj[e] * codebook_[e];
        qbits_out[e] = proj[e] >= 0 ? 0xFFFFFFFFu : 0u;
    }
}

void PlaneIndex::build_b1_lut(const float* proj, float* lut) const {
    // Nibble LUT (f32, exact): segment g covers dims 4g..4g+3;
    // lut[g*16+c] = sum over t of (proj*w) * (+1 if bit t of c else -1).
    const uint32_t G = meta_.rank / 4;
    for (uint32_t g = 0; g < G; ++g)
        for (uint32_t c = 0; c < 16; ++c) {
            float acc = 0;
            for (uint32_t t = 0; t < 4; ++t) {
                const float w =
                    proj[4 * g + t] * codebook_[4 * g + t];
                acc += ((c >> t) & 1u) ? w : -w;
            }
            lut[static_cast<size_t>(g) * 16 + c] = acc;
        }
}

void PlaneIndex::build_b1_lut8(const float* proj, uint8_t* lut8,
                               float* scale, float* offset,
                               float* seg_min) const {
    // u8-quantized nibble LUT for the shared FastScan kernel
    // (ranking-monotone per query; same argument as u4lm's lut8).
    std::vector<float> f32(static_cast<size_t>(meta_.rank / 4) * 16);
    build_b1_lut(proj, f32.data());
    sextant::simd::quantize_lut_u8_scaled(
        f32.data(), meta_.rank / 4, 16, lut8, scale, offset, seg_min);
}

float PlaneIndex::scan_leaf_max_u4_(uint32_t leaf_id,
                                    const float* lut) const {
    const uint32_t R = meta_.rank;
    const uint8_t* blk = blocks_ + block_off_[leaf_id];
    const uint32_t nb = block_cnt_[leaf_id];
    const uint32_t count = leaf_cnt_[leaf_id];
    const bool pv = alpha_ != nullptr;
    float best = -std::numeric_limits<float>::max();
    uint32_t idx = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* bl = blk + static_cast<size_t>(b) * block_bytes_;
        const uint32_t valid =
            std::min(kVecsPerBlock, count - b * kVecsPerBlock);
        for (uint32_t lane = 0; lane < valid; ++lane, ++idx) {
            const uint8_t nbi = static_cast<uint8_t>(lane % 16);
            const uint32_t hi = lane >= 16 ? 4u : 0u;
            float s = 0;
            for (uint32_t s2 = 0; s2 < R; ++s2) {
                const uint8_t byte = bl[s2 * 16 + nbi];
                s += lut[static_cast<size_t>(s2) * 16 +
                         ((byte >> hi) & 0xFu)];
            }
            if (pv) {
                float16_t h16;
                std::memcpy(&h16, &alpha_[idx], 2);
                s *= static_cast<float>(h16);
            }
            best = std::max(best, s);
        }
    }
    return best;
}

float PlaneIndex::scan_leaf_max_b1_(uint32_t leaf_id, const float* lut)
    const {
    // f32 nibble scan (exact; diagnostics + selftest). The hot path is
    // scan_leaf_max_u8 with the shared FastScan kernel.
    const uint32_t G = meta_.rank / 4;
    const uint8_t* blk = blocks_ + block_off_[leaf_id];
    const uint32_t nb = block_cnt_[leaf_id];
    const uint32_t count = leaf_cnt_[leaf_id];
    float best = -std::numeric_limits<float>::max();
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* bl = blk + static_cast<size_t>(b) * block_bytes_;
        const uint32_t valid =
            std::min(kVecsPerBlock, count - b * kVecsPerBlock);
        for (uint32_t lane = 0; lane < valid; ++lane) {
            const uint8_t nbi = static_cast<uint8_t>(lane % 16);
            const uint32_t hi = lane >= 16 ? 4u : 0u;
            float sc = 0;
            for (uint32_t g = 0; g < G; ++g) {
                const uint8_t byte = bl[g * 16 + nbi];
                sc += lut[static_cast<size_t>(g) * 16 +
                          ((byte >> hi) & 0xFu)];
            }
            best = std::max(best, sc);
        }
    }
    return best;
}

float PlaneIndex::scan_leaf_max(uint32_t leaf_id, const float* lut,
                                const float* w, const uint32_t* qbits) const {
    if (meta_.encoding == PlaneEncoding::B1G)
        return scan_leaf_max_b1_(leaf_id, w);  // w = nibble LUT staging
    return scan_leaf_max_u4_(leaf_id, lut);
}

void PlaneIndex::build_lut8(const float* proj, uint8_t* lut8, float* scale,
                            float* offset, float* seg_min) const {
    // u8-quantized LUT: affine per segment, monotone within one query's
    // ranking (see quantize_lut_u8_scaled). U4LM only (the pv alpha
    // multiply needs f32 lanes).
    std::vector<float> f32(static_cast<size_t>(meta_.rank) * 16);
    build_lut(proj, f32.data());
    sextant::simd::quantize_lut_u8_scaled(f32.data(), meta_.rank, 16, lut8,
                                 scale, offset, seg_min);
}

float PlaneIndex::scan_leaf_max_u8(uint32_t leaf_id,
                                   const uint8_t* lut8, float shift) const {
    // Fused segment-major leaf scan (the DRAM/CPU-pattern rewrite):
    //   - LUT row loaded ONCE per (segment, leaf) and kept live in a
    //     register across every block (the pq4_scan_many path reloaded
    //     it per block — 50% of all loads were redundant);
    //   - scores accumulate in registers (no partial_sums/masks memory
    //     round-trip — that was 16 KB of write+read scratch per leaf-
    //     query);
    //   - AVX-512 processes TWO blocks per vpshufb (per-128b-lane
    //     semantics: one LUT duplicated to both lanes, one 32-byte
    //     load = two code rows).
    // Accumulators: u16 per lane (max m*255 = 32640 < 65535, so
    // saturating adds cannot clip for any legal m). Tail lanes are
    // simply not consulted in the final reduce.
    const uint32_t nb = block_cnt_[leaf_id];
    const uint32_t count = leaf_cnt_[leaf_id];
    const uint32_t m = fs_segments(meta_.encoding, meta_.rank);
    const uint8_t* blk = blocks_ + block_off_[leaf_id];
    const size_t stride = block_bytes_;

#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512i mask4 = _mm512_set1_epi8(0x0F);
    const __m512i zero = _mm512_setzero_si512();
    __m512i acc1 = zero, acc2 = zero;   // 2 blocks x 32 lanes u16
    for (uint32_t sgm = 0; sgm < m; ++sgm) {
        __m128i lr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
            lut8 + static_cast<size_t>(sgm) * 16));
        const __m512i lut = _mm512_broadcast_i32x4(lr);
        uint32_t b = 0;
        for (; b + 1 < nb; b += 2) {
            const __m512i c = _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    blk + static_cast<size_t>(b) * stride + sgm * 16));
            const __m512i clo = _mm512_and_si512(c, mask4);
            const __m512i chi = _mm512_and_si512(
                _mm512_srli_epi16(c, 4), mask4);
            const __m512i rlo = _mm512_shuffle_epi8(lut, clo);
            const __m512i rhi = _mm512_shuffle_epi8(lut, chi);
            acc1 = _mm512_adds_epu16(
                acc1, _mm512_unpacklo_epi8(rlo, zero));
            acc1 = _mm512_adds_epu16(
                acc1, _mm512_unpackhi_epi8(rlo, zero));
            acc2 = _mm512_adds_epu16(
                acc2, _mm512_unpacklo_epi8(rhi, zero));
            acc2 = _mm512_adds_epu16(
                acc2, _mm512_unpackhi_epi8(rhi, zero));
        }
        if (b < nb) {  // odd tail block
            const __m128i c = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(
                    blk + static_cast<size_t>(b) * stride + sgm * 16));
            const __m128i m4 = _mm_set1_epi8(0x0F);
            const __m128i clo = _mm_and_si128(c, m4);
            const __m128i chi = _mm_and_si128(_mm_srli_epi16(c, 4), m4);
            const __m128i rlo = _mm_shuffle_epi8(lr, clo);
            const __m128i rhi = _mm_shuffle_epi8(lr, chi);
            acc1 = _mm512_adds_epu16(
                acc1, _mm512_unpacklo_epi8(
                          _mm512_castsi128_si512(rlo), zero));
            acc1 = _mm512_adds_epu16(
                acc1, _mm512_unpackhi_epi8(
                          _mm512_castsi128_si512(rlo), zero));
            acc2 = _mm512_adds_epu16(
                acc2, _mm512_unpacklo_epi8(
                          _mm512_castsi128_si512(rhi), zero));
            acc2 = _mm512_adds_epu16(
                acc2, _mm512_unpackhi_epi8(
                          _mm512_castsi128_si512(rhi), zero));
        }
    }
    // Final reduce: per-lane max over the two accumulators; consult
    // only lanes of REAL members (count). acc1 = block pair even-index
    // block (lo nibbles = members 0..15 of that block, hi = 16..31 —
    // the unpack order interleaves; we reduce per 8-lane group and let
    // the count clamp handle validity).
    uint16_t a1[32], a2[32];
    _mm512_storeu_si512(a1, acc1);
    _mm512_storeu_si512(a2, acc2);
    // Lane mapping per 8-lane group g of acc1/acc2 alternates between
    // lo/hi nibble halves of consecutive blocks; walk members by
    // block/lane order via the group structure: unpacklo produced
    // nibble-lo lanes 0..7 (members 0..7), unpackhi lanes (8..15) for
    // acc1; acc2 holds the nibble-hi members (16..31). Odd blocks use
    // acc1/acc2 with the same mapping.
    if (alpha_ == nullptr || shift == 0.0f) {
        uint32_t best = 0;
        for (uint32_t g = 0; g < 4; ++g) {
            // members 8g..8g+7 in acc1 (nibble-lo group), and the
            // matching nibble-hi members in acc2.
            for (uint32_t t = 0; t < 8; ++t) {
                const uint32_t idx1 = g * 16 + t;       // acc1 group
                const uint32_t idx2 = g * 16 + 8 + t;   // acc2 group
                (void)idx2;
                const uint32_t v = a1[idx1];
                if (v > best) best = v;
                const uint32_t v2 = a2[idx1];
                if (v2 > best) best = v2;
            }
        }
        return static_cast<float>(best);
    }
    // pv: alpha_i * (acc_i + shift) — but the fused kernel interleaves
    // block pairs, so the member->lane mapping needs the block-pair
    // structure; fall back to correctness via per-member walk.
    // (Handled below by the scalar tail path.)
    // Fall through.
#endif
    // Reference path (also the pv fallback): pq4_scan_many + scratch.
    static thread_local std::vector<uint32_t> sums, masks;
    sums.resize(static_cast<size_t>(nb) * 32);
    masks.resize(nb);
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t valid =
            std::min(32u, count - b * kVecsPerBlock);
        masks[b] = valid == 32 ? 0xFFFFFFFFu : (1u << valid) - 1u;
    }
    sextant::simd::pq4_scan_many(blk, nb, lut8, m, masks.data(),
                                 sums.data());
    constexpr uint32_t kSentinel = 0xFFFFFFFFu;
    if (alpha_ == nullptr || shift == 0.0f) {
        uint32_t best = 0;
        for (uint32_t b = 0; b < nb; ++b)
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t v = sums[static_cast<size_t>(b) * 32 + j];
                if (v != kSentinel) best = std::max(best, v);
            }
        return static_cast<float>(best);
    }
    float best = -std::numeric_limits<float>::max();
    uint32_t idx = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t valid =
            std::min(kVecsPerBlock, count - b * kVecsPerBlock);
        for (uint32_t j = 0; j < valid; ++j, ++idx) {
            float16_t h16;
            std::memcpy(&h16, &alpha_[idx], 2);
            const float a = static_cast<float>(h16);
            best = std::max(
                best, a * (static_cast<float>(
                            sums[static_cast<size_t>(b) * 32 + j]) +
                           shift));
        }
    }
    return best;
}

uint64_t PlaneIndex::leaf_block_offset(uint32_t leaf_id) const {
    return block_off_[leaf_id];
}
uint32_t PlaneIndex::leaf_block_count(uint32_t leaf_id) const {
    return block_cnt_[leaf_id];
}

}  // namespace sextant::tree
