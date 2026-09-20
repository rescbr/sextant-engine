#pragma once

/// @file plane.hpp
/// Stage-1 routing plane (plan D): a per-vector quantized PCA-128 plane,
/// leaf-aligned, used to rank leaves by max over members of query·proj.
///
/// Two encodings (archived measurement):
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
#include "sextant/types.hpp"

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

    /// In-build emission support: rows are encoded under their CLUSTER id
    /// (stable during the streaming merge) via encode_staged; when a
    /// cluster's buffer flushes as global leaf `leaf_id`, transfer_leaf
    /// moves the accumulated blocks there and resets the cluster slot.
    /// O(1) swap. Cluster staging is separate from leaves_ so early
    /// global ids (< k_root) cannot collide with in-flight clusters.
    void encode_staged(uint32_t cluster, uint32_t slot, const float* pr);
    void transfer_leaf(uint32_t cluster, uint32_t leaf_id);

    /// Layout-v1 RAM bound: with the spill enabled, transfer_leaf appends
    /// each flushed leaf's blocks (+ pv alphas) to a temp file instead of
    /// retaining every leaf's blocks in RAM until finalize. The file is
    /// unlinked at open (POSIX unlink-while-open: it can never outlive the
    /// writer) and closed by finalize/the destructor.
    void enable_spill(const std::string& temp_path);

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

    /// Plane v2 (blocks-in-leaf layout): serializes the HEADER-ONLY blob
    /// (header + basis + mean + codebook; no blocks, no alpha). Per-leaf
    /// blocks are detached at flush time via detach_leaf_blocks instead.
    std::vector<uint8_t> finalize_header_only();

    /// Plane v2: detach cluster `c`'s staged rows for a leaf of `count`
    /// members — ceil(count/32) padded blocks, followed by count fp16
    /// alphas for u4lm_pv. Resets the cluster's staging. The staged slots
    /// are leaf-local (they reset at every flush), so at flush time the
    /// staging holds exactly this leaf's rows.
    std::vector<uint8_t> detach_leaf_blocks(uint32_t cluster,
                                            uint32_t count);

    ~PlaneWriter();

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
        std::vector<float16_t> alphas;  // count fp16 alphas (u4lm_pv only;
                                        // computed at encode time, 2 B/vec)
        uint32_t n_blocks = 0;
    };
    std::vector<LeafState> leaves_;
    std::vector<LeafState> stage_;    // in-build: per-cluster accumulation
    uint32_t block_bytes_ = 0;

    void encode_into_(LeafState& ls, uint32_t slot, const float* pr);

    uint32_t encode_dim_(float v, uint32_t e) const;
    float decode_dim_(uint32_t code, uint32_t e) const;

    // Leaf-block spill (layout v1 streaming build). Append-only temp file,
    // unlink-at-open, sequential records in increasing leaf_id order:
    //   [u32 leaf_id][u32 n_blocks][u32 n_alphas]
    //   [n_blocks × block_bytes blocks][n_alphas × 2 alpha bytes]
    std::string spill_path_;
    int spill_fd_ = -1;
    std::vector<uint8_t> spill_wbuf_;  // 1 MiB append buffer
    size_t spill_wpos_ = 0;            // next file byte for the buffer
    size_t spill_wused_ = 0;           // bytes pending in spill_wbuf_
    uint32_t spill_n_recs_ = 0;
    void spill_put(const void* p, size_t n);
    void spill_flush();
    void spill_close();
    // Fills the blob's blocks/alpha regions from the spill (one sequential
    // pass). `blocks_prefix`/`alpha_prefix` are per-leaf blob offsets.
    void finalize_from_spill(uint8_t* blob,
                             const std::vector<uint32_t>& leaf_counts,
                             uint64_t blocks_off, uint64_t alpha_off);
};

/// Search side: read-only view over an attached plane extent.
class PlaneIndex {
public:
    /// Parse a plane blob already read from the extent. Returns nullptr
    /// on magic/version mismatch (caller treats as corrupt).
    static std::unique_ptr<PlaneIndex> parse(const uint8_t* data, size_t len);

    const PlaneMeta& meta() const { return meta_; }

    /// v2 layout: per-leaf blocks live inside the leaf extents (suffix at
    /// TreeLeafHeader::plane_offset) — no blob blocks region, no blob
    /// alpha; scans must supply per-leaf pointers (scan_leaf_max_u8_at).
    bool v2_layout() const { return v2_; }

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

    /// FastScan-kernel leaf max (U4LM + U4LM_PV). Without alpha:
    /// returns the raw u32 accumulator (affine-monotone per query).
    /// With alpha (pv): per-member alpha * (acc + shift), where shift
    /// = scale*offset from build_lut8 UNDOES the per-segment min
    /// subtraction exactly (lut8 = A*(lut - min_s) => true =
    /// acc/A + B; alpha * (acc + A*B) ranks identically). shift must
    /// be 0 for non-pv (callers rank on raw accumulators).
    float scan_leaf_max_u8(uint32_t leaf_id, const uint8_t* lut8,
                           float shift = 0.0f) const;

    /// scan_leaf_max_u8 with the caller-supplied block pointer (e.g. rows
    /// pinned in the plane cache) instead of the mmap'd blob. leaf_id
    /// still supplies the block count and member count; `alpha_override`
    /// (v2: per-leaf alphas inside the leaf extent) replaces the blob's
    /// leaf-aligned alpha base when non-null.
    float scan_leaf_max_u8_at(uint32_t leaf_id, const uint8_t* blk,
                              const uint8_t* lut8,
                              float shift = 0.0f,
                              const uint16_t* alpha_override = nullptr) const;

    /// Query-tiled variant: Q queries against one leaf's blocks per
    /// pass — code loads amortize across the tile, LUT rows stay
    /// register-resident per (segment, tile). `luts[q]` = rank-dep
    /// u8 LUTs; out[q] = per-query max (alpha-aware when the plane
    /// carries pv). Callers with Q < 4 should use scan_leaf_max_u8.
    void scan_leaf_max_u8_q(uint32_t leaf_id, const uint8_t* const* luts,
                            uint32_t Q, const float* shifts,
                            float* out) const;

    /// Max ADC score over one leaf's plane blocks (masked tail lanes).
    /// `lut` from build_lut (u4lm*) or `w`/`qbits` from build_sign_ctx
    /// (b1g). Leaf offsets are computed at bind() time.
    float scan_leaf_max(uint32_t leaf_id, const float* lut,
                        const float* w, const uint32_t* qbits) const;

    /// Binds per-leaf block offsets from the leaf table's stored counts.
    /// Must be called once after parse(), before scanning.
    void bind(const std::vector<uint32_t>& leaf_counts);

    /// Byte offset of the blocks region within the plane blob (the blob
    /// starts with header/basis/mean/codebook). Callers converting a leaf's
    /// block offset into a FILE offset must add this to the extent's page
    /// base (the plane cache does exactly that).
    uint64_t blocks_offset_in_blob() const {
        return static_cast<uint64_t>(blocks_ - blob_);
    }

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
    const uint8_t* blob_ = nullptr;    // parse() input (offset base)
    const uint8_t* blocks_ = nullptr;  // into blob
    const uint16_t* alpha_ = nullptr;  // u4lm_pv: per real member, leaf-aligned
    const float* basis_t_ = nullptr;
    const float* mean_ = nullptr;
    const float* codebook_ = nullptr;
    std::vector<uint64_t> block_off_;  // per leaf
    std::vector<uint32_t> block_cnt_;
    std::vector<uint32_t> leaf_cnt_;   // real member counts (tail clamps)
    uint32_t block_bytes_ = 0;
    bool v2_ = false;  // blocks live in leaf extents, not in this blob
    float scan_leaf_max_u4_(uint32_t leaf_id, const float* lut) const;
    float scan_leaf_max_b1_(uint32_t leaf_id, const float* blut) const;
};

/// Parse just the header (fsck sizing checks). Returns false on mismatch.
bool parse_plane_header(const uint8_t* data, size_t len, PlaneMeta* meta,
                        uint64_t* blocks_bytes = nullptr,
                        uint64_t* codebook_bytes = nullptr,
                        uint64_t* alpha_bytes = nullptr,
                        uint8_t* flags = nullptr);

}  // namespace sextant::tree
