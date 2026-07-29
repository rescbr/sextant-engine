#pragma once

/// @file product_residual_quantizer.hpp
/// Product Residual Quantization (PRQ) — additive-residual variant of PQ.
///
/// **References:**
/// - Additive Quantization (AQ): Babenko, Slesarev, Chigorin, Lempitsky,
///   "Neural Codes for Image Retrieval," ECCV 2014. Introduced representing
///   x ≈ Σ_m c_m[code_m] as a sum of full-dimensional codewords.
/// - Residual Quantizer encoding: Martínez, Zholbussirov, Martínez,
///   "LSQ++: Lower running time and higher recall in multi-codebook
///   quantization," ECCV 2016. Greedy residual beam search with
///   progressive-dim k-means codebook training.
/// - Product decomposition + FastScan integration: FAISS
///   `IndexIVFProductResidualQuantizerFastScan` (Douze, Guzhva, Deng, Johnson,
///   Sivic, Jégou). Splits the vector into nsplits sub-spaces, runs RQ
///   per sub-space, packs 4-bit codes into the same FastScan block layout
///   as IVFPQFastScan.
///
/// PRQ splits the vector into `nsplits` contiguous sub-spaces of dimension
/// `sub_dim = dim / nsplits`. Within each sub-space, a Residual Quantizer
/// encodes the residual after the previous levels: level 0 quantizes the raw
/// sub-vector, level m quantizes the residual left after levels 0..m-1. The
/// reconstruction is the SUM of the M_sub assigned centroids (additive), not
/// the disjoint-block concatenation of standard PQ.
///
/// The total segment count is `m = nsplits × M_sub`, matching the PQ `m`
/// parameter so the same 4-bit FastScan scan path and `m × K` LUT layout apply.
/// The structural difference from PQ is in the LUT: multiple segments (the
/// M_sub levels of one split) dot against the SAME query sub-vector but
/// DIFFERENT codebooks, and their entries SUM into the per-sub-space distance.
///
/// Subclasses PqQuantizer to be a drop-in scan quantizer for the IVF scan
/// path. Overrides train/encode/preprocess_query/decode_code/serialize/
/// deserialize; the FastScan LUT build and the scan kernels are inherited
/// unchanged (they consume the m×K LUT + packed codes agnostically).
///
/// ## Empirical findings (Sphere IP, LID 20.8, 10M × 768, K=256)
///
/// PRQ breaks the 4-bit PQ recall ceiling, with the gain scaling monotonically
/// with nsplits (more splits = fewer residual levels per sub-space = less
/// cascading greedy encoding error):
///
///   nsplits=192 (M_sub=1) → degenerates to standard PQ (no additive structure)
///   nsplits=96  (M_sub=2) → recall 0.927 (greedy) / 0.936 (beam=5)
///   nsplits=48  (M_sub=4) → recall 0.919
///   nsplits=24  (M_sub=8) → recall 0.907
///   nsplits=12  (M_sub=16) → recall 0.877 (WORSE than PQ — too many levels)
///   PQ baseline              → recall 0.897
///
/// Recommended default: nsplits=96 (auto when sub_dim=8), beam_size=5.
/// At matched QPS (~290, np=64), PRQ s96 b1 delivers 0.905 vs PQ's 0.875.
/// The 8-bit PQ ceiling (0.999) remains unreachable at 4-bit; PRQ closes
/// ~43% of the gap. See docs/quantization_findings.md §PRQ for full analysis.

#include "pq_quantizer.hpp"
#include <sextant/types.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace sextant {

class ProductResidualQuantizer : public PqQuantizer {
public:
    /// Construct a PRQ.
    /// `m` is the total number of code segments = nsplits × M_sub.
    /// `bits` must be 4 (4-bit FastScan scan path; K=16).
    /// `nsplits` is the number of contiguous sub-spaces; must divide both
    ///   `dim` and `m` evenly.
    /// `beam_size` controls the greedy/beam encode search width (1 = pure
    ///   greedy; >1 = beam search). Ignored when encode_mode = "icm".
    /// `encode_mode` selects the encoding strategy: "greedy" (sequential
    ///   residual, fast), "beam" (beam search, moderate), or "icm"
    ///   (ICM+ILS coordinate descent, best quality, still L1/L2-resident).
    ProductResidualQuantizer(MetricKind metric, Dim dim, uint16_t m,
                             uint8_t bits, uint32_t nsplits,
                             uint32_t beam_size = 1,
                             uint64_t seed = 0xC0DE1234ULL,
                             std::string encode_mode = "greedy",
                             uint32_t icm_iters = 4,
                             uint32_t ils_iters = 4,
                             uint32_t ils_perturb = 4,
                             uint32_t lsq_train_iters = 0);

    void train(const float* samples, uint64_t n) override;
    void encode(const float* vec, uint8_t* code_out) const override;
    void preprocess_query(const float* query, float* out) const override;
    void preprocess_query_as(MetricKind metric, const float* query,
                             float* out) const override;
    void decode_code(const uint8_t* code, float* out) const override;
    void serialize(std::vector<uint8_t>& out) const override;
    void deserialize(const uint8_t* in, size_t size) override;

    uint32_t nsplits() const { return nsplits_; }
    uint32_t m_sub() const { return M_sub_; }
    uint32_t rq_sub_dim() const { return sub_dim_prq_; }
    const std::string& encode_mode() const { return encode_mode_; }

    /// PRQ codebook layout: [nsplits][M_sub][K][sub_dim_prq] row-major.
    /// For split s, level l, centroid c: ((s * M_sub + l) * K + c) * sub_dim_prq.
    const float* rq_codebooks() const { return rq_codebooks_.data(); }

    /// Magic byte marking a PRQ serialization blob (distinct from PQ).
    static constexpr uint8_t kMagic = 0xA5;

private:
    uint32_t nsplits_;
    uint32_t M_sub_;       ///< levels per split = m / nsplits.
    uint32_t sub_dim_prq_; ///< per-split sub-vector dimension = dim / nsplits.
    uint32_t beam_size_;

    std::string encode_mode_ = "greedy"; ///< "greedy", "beam", or "icm"
    uint32_t icm_iters_ = 4;   ///< ICM sweeps per ILS cycle
    uint32_t ils_iters_ = 4;   ///< ILS cycles (perturb + ICM + accept)
    uint32_t ils_perturb_ = 4; ///< codes to perturb per ILS cycle

    /// LSQ training iterations (alternating codebook update + ICM re-encode).
    /// 0 = use progressive k-means only (no LSQ refinement).
    uint32_t lsq_train_iters_ = 0;

    /// On-the-fly ICM+ILS encoding (no precomputed tables).
    /// See docs/quantization_findings.md §PRQ and the plan at
    /// ~/.local/state/maki/plans/amazing-striking-toad.md.
    void encode_icm_(const float* vec, uint8_t* code_out) const;

    /// LSQ codebook update: solve (BᵀB + ρI) Cᵀ = XBᵀ for one sub-space.
    /// `codes` is [n][M_sub], `samples` is [n][sub_dim]. Updates `books`
    /// in-place. Uses Gaussian elimination with partial pivoting (system is
    /// tiny: M_sub·K × M_sub·K, at most 128×128).
    void update_codebooks_lsq_(const float* samples, uint64_t n,
                               const uint32_t* codes, float* books) const;

    /// Additive codebooks: [nsplits][M_sub][K][sub_dim_prq].
    std::vector<float> rq_codebooks_;

    /// Per-centroid squared L2 norms for the L2sq LUT decomposition:
    /// [nsplits][M_sub][K]. ||q-c||² = ||q||² - 2<q,c> + ||c||².
    std::vector<float> rq_centroid_sqnorms_;

    inline const float* level_book_(uint32_t split, uint32_t level) const {
        const size_t idx =
            (static_cast<size_t>(split) * M_sub_ + level) * K_ * sub_dim_prq_;
        return rq_codebooks_.data() + idx;
    }
};

}  // namespace sextant
