#pragma once

/// @file pq_quantizer.hpp
/// Product Quantization (PQ) for vector compression.
///
/// Evolved from Sextant's exploratory phase with simsimd → NumKong adaptation.

#include <sextant/types.hpp>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace sextant {

class PqQuantizer {
public:
    PqQuantizer(MetricKind metric, Dim dim, uint16_t m, uint8_t bits = 8,
                uint64_t seed = 0xC0DE1234ULL);

    /// Number of bytes per PQ code.
    uint32_t code_size() const;

    /// Number of PQ segments.
    uint16_t m() const { return m_; }

    /// Bits per segment (4 or 8).
    uint8_t bits() const { return bits_; }

    /// Train the codebook via batch k-means++ on a sample of vectors.
    /// `samples` is `n × dim` float32, row-major.
    void train(const float* samples, uint64_t n);

    /// Encode a single vector into `code_out` (must be code_size() bytes).
    void encode(const float* vec, uint8_t* code_out) const;

    /// Build the symmetric cross-distance table (code-to-code).
    /// Called after train(). Stored internally.
    void build_cross_distance_table();

    /// Preprocess a query vector into a LUT for PQ distance estimation.
    /// `out` must hold lut_size() floats.
    void preprocess_query(const float* query, float* out) const;

    /// Estimate distance from a query LUT to a PQ code.
    float lut_distance(const uint8_t* code, const float* lut) const;

    /// Size of the LUT (in floats) = m × K.
    uint32_t lut_size() const;

    /// Read-only codebook access for diagnostics / build-time heuristics.
    /// Layout: m segments × K centroids × sub_dim floats, row-major.
    const float* codebook() const { return codebook_.data(); }
    uint32_t K() const { return K_; }
    uint32_t sub_dim() const { return sub_dim_; }

    /// Build a LUT from a PQ code (for HDC build mode).
    bool build_code_lut(const uint8_t* code, float* out) const;

    /// Code-to-code distance via the cross-distance table.
    float code_distance(const uint8_t* code_a, const uint8_t* code_b) const;

    /// Batch code-to-code distance: fixed anchor vs 4 candidates.
    /// Writes 4 distances to `out`. Uses SIMD + interleaved loads for
    /// 2× throughput vs 4 individual code_distance calls. The anchor's
    /// per-segment centroid ids are pre-extracted to avoid redundant work.
    void code_distance_batch4(const uint8_t* anchor,
                              const uint8_t* code_b0,
                              const uint8_t* code_b1,
                              const uint8_t* code_b2,
                              const uint8_t* code_b3,
                              float* out) const;

    /// Batch LUT distance: fixed LUT vs 4 candidate codes.
    void lut_distance_batch4(const uint8_t* code_b0,
                             const uint8_t* code_b1,
                             const uint8_t* code_b2,
                             const uint8_t* code_b3,
                             const float* lut,
                             float* out) const;

    /// Serialize the quantizer state (codebook + params).
    void serialize(std::vector<uint8_t>& out) const;

    /// Deserialize from a buffer. Replaces current state.
    void deserialize(const uint8_t* in, size_t size);

private:
    MetricKind metric_;
    Dim dim_;
    uint16_t m_;
    uint8_t bits_;
    uint64_t seed_;

    uint32_t K_;           ///< 2^bits centroids per segment
    uint32_t sub_dim_;     ///< dim / m

    /// Codebook: m segments × K centroids × sub_dim floats.
    std::vector<float> codebook_;

    /// Cross-distance table: m × K × K floats (code-to-code).
    std::vector<float> cross_distance_table_;
};

}  // namespace sextant
