#pragma once

/// @file plane.hpp
/// Stage-1 routing plane (plan D): a per-vector quantized PCA-128 plane,
/// leaf-aligned, used to rank leaves by max over members of query·proj.
///
/// Two encodings (measured in results/plane_quant_20260911/):
///   U4LM    64 B/vec — per-dim Lloyd-Max 4-bit centroids (128×16 f32
///                     codebook), nibble FastScan blocks ([m][16], 32
///                     vectors/block — same interleave as leaf codes, so
///                     the pq4_scan_many kernel scores plane blocks).
///   U4LM_PV U4LM + per-vector fp16 length renorm alpha (+2 B/vec).
///   B1G     16 B/vec — sign codes + per-dim abs-mean scale; bitplane
///                     blocks (rank × u32 sign bits per 32 vectors).
///
/// On-disk blob (one extent, superblock plane_page/plane_pages):
///   [PlaneHeaderDisk][basis_t f32 dim×rank][mean f32 dim]
///   [codebook][alpha?][blocks in leaf order]
/// Per-leaf block counts are DERIVED at open from the leaf table's stored
/// counts (blocks = ceil(count / 32) per leaf) — no indirection table.
///
/// Lifecycle: written by IVFTreeIndex::attach_plane() (post-build pass:
/// reopen members via the debug API, sequential base re-read, encode,
/// extent write, superblock re-commit). Plane-bearing indexes are
/// IMMUTABLE (mutable ops throw, like scalar_lloydmax) — v1 semantics.

#include "page_file.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sextant::tree {

enum class PlaneEncoding : uint8_t {
    U4LM = 1,    ///< 4-bit Lloyd-Max, 64 B/vec at rank 128
    B1G = 2,     ///< 1-bit sign + abs-mean scale, 16 B/vec
    U4LM_PV = 3, ///< U4LM + per-vector fp16 alpha (+2 B/vec)
};

/// Parsed plane header (disk struct lives in plane.cpp; packed).
struct PlaneMeta {
    PlaneEncoding encoding = PlaneEncoding::U4LM;
    uint16_t rank = 128;
    uint32_t dim = 0;
    bool has_alpha() const { return encoding == PlaneEncoding::U4LM_PV; }
    /// Bytes per vector (honest, including alpha).
    uint32_t bytes_per_vec() const;
};

/// Build/attach side: trains basis + codebooks from the base vectors,
/// encodes all leaves, and produces the serialized blob.
class PlaneWriter {
public:
    struct Config {
        PlaneEncoding encoding = PlaneEncoding::U4LM;
        uint16_t rank = 128;
        uint32_t train_rows = 20000;  ///< spread-sampled basis training
        uint32_t lm_hist_bins = 512;
        uint32_t lm_iters = 50;
    };

    /// Trains the PCA basis (power iteration + deflation on a
    /// SPREAD-sampled covariance — clustered-order bases trained on a
    /// prefix lose ~12pp containment) and the per-dim codebooks.
    /// `base` is n×dim row-major.
    void train(const float* base, uint32_t n, uint32_t dim, const Config& cfg);

    /// Sizes per-leaf encoding state (call once after train, before any
    /// encode_member).
    void prepare(uint32_t n_leaves);

    /// Projects one vector (rank floats into `pr`).
    void project(const float* vec, float* pr) const;

    /// Emits codes for an ALREADY-PROJECTED member (chunked parallel
    /// encode: phase 1 projects rows in parallel, phase 2 encodes
    /// leaf-sharded — no two threads ever touch one leaf).
    void encode_from_proj(uint32_t leaf_id, uint32_t slot,
                          const float* pr);

    /// Convenience: project + encode one member.
    void encode_member(uint32_t leaf_id, uint32_t slot, const float* vec,
                       float* proj_out = nullptr);

    /// Finalizes padded blocks, computes alpha if flagged, and serializes
    /// the blob. `leaf_counts` is the per-leaf stored-member count (leaf
    /// table order). The plane covers exactly ceil(count/32)*32 lanes per
    /// leaf; tail lanes are zero-coded and must be masked at scan time.
    std::vector<uint8_t> finalize(const std::vector<uint32_t>& leaf_counts);

    const PlaneMeta& meta() const { return meta_; }

private:
    PlaneMeta meta_{};
    Config cfg_{};
    std::vector<float> basis_t_;   // [dim][rank] f32 (transposed)
    std::vector<float> mean_;      // dim
    std::vector<float> codebook_;  // u4lm*: rank×16 centroids; b1g: rank scales
    // Pending per-leaf encoding state (build-side only).
    struct LeafState {
        std::vector<uint8_t> blocks;  // ceil(count/32) × block_bytes
        std::vector<float> proj;      // count × rank (alpha pass)
        uint32_t n_blocks = 0;
    };
    std::vector<LeafState> leaves_;
    uint32_t block_bytes_ = 0;

    uint32_t encode_dim_(float v, uint32_t e) const;
    float decode_dim_(uint32_t code, uint32_t e) const;
};

/// Search side: read-only view over an attached plane extent.
class PlaneIndex {
public:
    /// Parse a plane blob already read from the extent. Returns nullptr
    /// on magic/version mismatch (caller treats as corrupt).
    static std::unique_ptr<PlaneIndex> parse(const uint8_t* data, size_t len);

    const PlaneMeta& meta() const { return meta_; }

    /// Projects an original-space query to the plane (rank floats).
    void project_query(const float* q, float* proj_out) const;

    /// LUT for the u4lm ADC kernel: lut[s][c] = proj[s]·centroid[s][c],
    /// s < rank, c < 16 (row-major f32, rank*16 floats). b1g: unused.
    void build_lut(const float* proj, float* lut) const;
    /// b1g per-dim weights w_e = proj[e]·sign_scale[e] (rank floats) and
    /// replicated query sign bits per dim (rank u32s). u4lm: unused.
    void build_sign_ctx(const float* proj, float* w_out, uint32_t* qbits_out) const;

    /// b1g f32 nibble LUT (rank/4 x 16 floats) — exact scan path.
    void build_b1_lut(const float* proj, float* lut) const;

    /// b1g u8 nibble LUT (rank/4 x 16 bytes) for the shared FastScan
    /// kernel (ranking-monotone per query).
    void build_b1_lut8(const float* proj, uint8_t* lut8, float* scale,
                       float* offset, float* seg_min) const;

    /// u8-quantized LUT (affine-monotone per query) for the FastScan
    /// kernel path. U4LM only. lut8 = rank*16 bytes; scale/offset/
    /// seg_min are scratch (rank floats each / rank for seg_min).
    void build_lut8(const float* proj, uint8_t* lut8, float* scale,
                    float* offset, float* seg_min) const;

    /// FastScan-kernel leaf max (U4LM only). Returns the raw u32
    /// accumulator — affine-monotone in the true score, so rankings
    /// within one query are exact; cross-query comparisons are not.
    float scan_leaf_max_u8(uint32_t leaf_id, const uint8_t* lut8) const;

    /// Max ADC score over one leaf's plane blocks (masked tail lanes).
    /// `lut` from build_lut (u4lm*) or `w`/`qbits` from build_sign_ctx
    /// (b1g). Leaf offsets are computed at bind() time.
    float scan_leaf_max(uint32_t leaf_id, const float* lut,
                        const float* w, const uint32_t* qbits) const;

    /// Binds per-leaf block offsets from the leaf table's stored counts.
    /// Must be called once after parse(), before scanning.
    void bind(const std::vector<uint32_t>& leaf_counts);

    /// Diagnostics: block byte offset of a leaf (fsck cross-check).
    uint64_t leaf_block_offset(uint32_t leaf_id) const;
    /// Debug: raw block bytes for a leaf (selftest bit verification).
    const uint8_t* debug_block_ptr(uint32_t leaf_id) const {
        return blocks_ + block_off_[leaf_id];
    }
    uint32_t debug_block_bytes() const { return block_bytes_; }
    uint32_t leaf_block_count(uint32_t leaf_id) const;

private:
    PlaneMeta meta_{};
    const uint8_t* blocks_ = nullptr;  // into blob
    const uint16_t* alpha_ = nullptr;  // u4lm_pv: per real member, leaf-aligned
    const float* basis_t_ = nullptr;
    const float* mean_ = nullptr;
    const float* codebook_ = nullptr;
    std::vector<uint64_t> block_off_;  // per leaf
    std::vector<uint32_t> block_cnt_;
    std::vector<uint32_t> leaf_cnt_;   // real member counts (tail clamps)
    uint32_t block_bytes_ = 0;
    float scan_leaf_max_u4_(uint32_t leaf_id, const float* lut) const;
    float scan_leaf_max_b1_(uint32_t leaf_id, const float* blut) const;
};

/// Parse just the header (fsck sizing checks). Returns false on mismatch.
bool parse_plane_header(const uint8_t* data, size_t len, PlaneMeta* meta,
                        uint64_t* blocks_bytes = nullptr,
                        uint64_t* codebook_bytes = nullptr,
                        uint64_t* alpha_bytes = nullptr);

}  // namespace sextant::tree
