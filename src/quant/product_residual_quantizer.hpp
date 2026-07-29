#pragma once

/// @file product_residual_quantizer.hpp
/// Product Residual Quantization (PRQ) — additive-residual variant of PQ.
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

#include "pq_quantizer.hpp"
#include <sextant/types.hpp>
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
    /// `beam_size` controls the greedy encode search width (1 = pure greedy;
    ///   >1 reserved for future beam-search encoding — currently behaves as 1).
    ProductResidualQuantizer(MetricKind metric, Dim dim, uint16_t m,
                             uint8_t bits, uint32_t nsplits,
                             uint32_t beam_size = 1,
                             uint64_t seed = 0xC0DE1234ULL);

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
