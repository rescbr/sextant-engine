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

    /// Virtual destructor: PqQuantizer is held via unique_ptr<PqQuantizer> and
    /// subclassed by AnisotropicPqQuantizer. Without this, deleting through the
    /// base pointer would be UB.
    virtual ~PqQuantizer() = default;

    /// Number of bytes per PQ code.
    uint32_t code_size() const;

    /// Configured distance metric. Drives `preprocess_query` (LUT semantics:
    /// L2sq distance vs negated IP) and the FP16/FP32 true-distance paths
    /// (routing, rerank) via the `dist_f16`/`dist_f32` dispatch helpers in
    /// `vamana_core.hpp`. For L2-normalized data both metrics are
    /// rank-equivalent on true distances; IP is cheaper per eval.
    MetricKind metric() const { return metric_; }

    /// Number of PQ segments.
    uint16_t m() const { return m_; }

    /// Bits per segment (4 or 8).
    uint8_t bits() const { return bits_; }

    /// Train the codebook via batch k-means++ on a sample of vectors.
    /// `samples` is `n × dim` float32, row-major. Virtual so AnisotropicPqQuantizer
    /// can override with ScaNN's anisotropic Lloyd's algorithm; the hot-path
    /// methods (lut_distance, batch4, code_distance) stay non-virtual —
    /// VamanaCore calls them monomorphically through PqQuantizer&.
    virtual void train(const float* samples, uint64_t n);

    /// Set the thread cap for the next train() call. 0 (default) means use
    /// std::thread::hardware_concurrency(). Call before train(). This prevents
    /// oversubscription when train() runs inside an already-parallel context
    /// (e.g. the build's encode_pool worker or the estimator's mini-build loop).
    void set_num_threads(uint32_t n) { num_threads_ = n; }

    /// Enable/disable covariance-based anisotropic codebook training.
    ///
    /// When enabled, each subspace's dimensions are scaled by √w_d before
    /// k-means, where w_d = eigenvalue_d / mean(eigenvalues) of the subspace
    /// covariance (clamped to [0.1, 10]). High-variance dims get w>1 (more
    /// important), low-variance get w<1. This makes standard k-means minimize
    /// the anisotropic distance Σ_d w_d·(x_d-c_d)² implicitly — the k-means
    /// code itself is unchanged (scale-transform trick). After k-means the
    /// centroids are unscaled back to the original data space.
    ///
    /// λ > 0 enables it; λ = 0 disables (plain L2sq k-means). The λ value is
    /// ignored (covariance determines the weights); kept as a float for CLI
    /// compatibility. Must be called BEFORE train().
    void set_anisotropy(float lambda) {
        anisotropic_ = (lambda > 0.0f);
        anisotropy_lambda_ = lambda;
    }
    float anisotropy_lambda() const { return anisotropy_lambda_; }
    bool anisotropic() const { return anisotropic_; }

    /// Request OPQ (PCA rotation) training. Must be called BEFORE train():
    /// train() computes the d×d PCA rotation from the training-sample
    /// covariance and applies it to the sample before PQ k-means. The
    /// resulting rotation_ is then applied at encode/search time.
    void enable_opq() { opq_enabled_ = true; }
    bool opq_enabled() const { return opq_enabled_; }

    /// OPQ (Optimized Product Quantization) via PCA rotation.
    ///
    /// Sets the d×d rotation matrix R (row-major). When set, R is applied to
    /// every vector before PQ encoding and to every query before LUT
    /// construction, so PQ operates in the rotated (decorrelated) basis where
    /// dimensions are ordered by variance. This lets the PQ split align with
    /// the data's principal components, lowering reconstruction MSE. Must be
    /// called BEFORE train() (train() consumes rotation_ to rotate the
    /// training sample).
    void set_rotation(std::vector<float> r) {
        rotation_ = std::move(r);
        has_rotation_ = !rotation_.empty();
    }
    const float* rotation() const { return rotation_.data(); }
    bool has_rotation() const { return has_rotation_; }

    /// Encode a single vector into `code_out` (must be code_size() bytes).
    /// Virtual so ProductResidualQuantizer can override with greedy beam-search
    /// residual encoding; the PQ path stays monomorphic when held by value.
    virtual void encode(const float* vec, uint8_t* code_out) const;

    /// Build the symmetric cross-distance table (code-to-code).
    /// Called after train(). Stored internally.
    void build_cross_distance_table();

    /// Preprocess a query vector into a LUT for PQ distance estimation.
    /// `out` must hold lut_size() floats. Uses the configured `metric_`.
    /// Virtual: ProductResidualQuantizer builds a structurally different LUT
    /// (multiple codebook levels share one query sub-vector per split).
    virtual void preprocess_query(const float* query, float* out) const;

    /// Preprocess a query into a LUT under an explicit metric, regardless of the
    /// configured `metric_`. Used by `Estimator::estimate_config` to measure
    /// IP-ADC vs L2sq-ADC ranking agreement (the metric-recommendation signal):
    /// builds both LUTs on the same query and compares top-k overlap.
    /// L2sq: out[s*K+c] = ||q_sub - c||².  IP: out[s*K+c] = -<q_sub, c>.
    virtual void preprocess_query_as(MetricKind metric, const float* query, float* out) const;

    /// Build the uint8-quantized FastScan LUT for a query. Same per-segment
    /// distances as `preprocess_query` (under the configured metric_), but
    /// quantized to uint8 with a per-query-global (A, B) scale
    /// (`simd::quantize_lut_u8` — FAISS `NormTableScaler` approach).
    ///
    /// Caller buffers (must be sized before the call):
    ///   - `lut8`: m × K bytes (use `fastscan_lut_bytes()`).
    ///   - `scale`/`offset`: one float each (A and B).
    ///
    /// `dist_approx = (uint16_acc / A) + B`. For argmin over a single LUT the
    /// raw uint16 accumulator suffices; the inverse is only needed to compare
    /// distances across LUTs or to rerank thresholds.
    void build_fastscan_lut(const float* query,
                            uint8_t* lut8,
                            float* scale,
                            float* offset) const;

    /// Size of the uint8 FastScan LUT in bytes = m × K.
    uint32_t fastscan_lut_bytes() const {
        return static_cast<uint32_t>(m_) * K_;
    }

    /// Build the uint4-quantized FastScan LUT for a query (4-bit PQ path,
    /// Option A). Same per-segment distances as `preprocess_query` (under the
    /// configured metric_), but quantized to 4-bit values stored one-per-byte
    /// in `lut4` (m × K bytes; each byte's low nibble holds the value, high
    /// nibble is 0).
    ///
    /// Uses the validated `simd::quantize_lut_u4` scheme: per-segment min
    /// subtracted, one global scale A = 15/max_span, NO clamp on A. The 4-bit
    /// kernel's u32 extraction gives wide headroom; clamping crushed precision
    /// in the spike (recall 0.0006). See `simd::quantize_lut_u4` docs.
    ///
    /// Caller buffers (must be sized before the call):
    ///   - `lut4`: m × K bytes (use `fastscan_lut_bytes()`).
    ///   - `scale_out`: one float (A; only needed for cross-LUT comparisons,
    ///     which the scan path never makes — all shards in one query share one
    ///     LUT). May be nullptr to skip.
    ///
    /// Requires `bits_ == 4` (K_ == 16). Throws on misuse.
    void build_fastscan_lut4(const float* query,
                             uint8_t* lut4,
                             float* scale_out) const;

    /// Estimate distance from a query LUT to a PQ code.
    float lut_distance(const uint8_t* code, const float* lut) const;

    /// Size of the LUT (in floats) = m × K.
    uint32_t lut_size() const;

    /// Read-only codebook access for diagnostics / build-time heuristics.
    /// Layout: m segments × K centroids × sub_dim floats, row-major.
    const float* codebook() const { return codebook_.data(); }
    uint32_t K() const { return K_; }
    uint32_t sub_dim() const { return sub_dim_; }

    /// Build a LUT from a PQ code (for PQ-construct build mode).
    bool build_code_lut(const uint8_t* code, float* out) const;

    /// Code-to-code distance via the cross-distance table.
    float code_distance(const uint8_t* code_a, const uint8_t* code_b) const;

    /// Extract centroid id for slot `s` from a packed code (4-bit or 8-bit).
    /// Public so partition.cpp can do batch table lookups without going
    /// through code_distance one-at-a-time.
    static uint32_t read_code_public(const uint8_t* code, uint8_t bits, uint32_t s) {
        if (bits == 8) return static_cast<uint32_t>(code[s]);
        const uint32_t byte_off = s / 2;
        const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
        return static_cast<uint32_t>((code[byte_off] >> shift) & 0x0F);
    }

    /// Read-only access to the cross-distance table (m × K × K floats).
    /// Used by partition.cpp for batch assignment with pruning.
    const float* cross_distance_table() const { return cross_distance_table_.data(); }

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

    /// Decode a PQ code back to an approximate vector (sum of assigned
    /// centroids). Writes `dim` floats to `out`. When OPQ is enabled, the
    /// inverse (transpose) rotation is applied so the result is in the
    /// original input space. Used to reverse-map partition centroids (PQ
    /// codes) back to vector space for IVF routing.
    /// Virtual: ProductResidualQuantizer decodes by summing M_sub centroids
    /// per sub-space (additive residual reconstruction).
    virtual void decode_code(const uint8_t* code, float* out) const;

    /// Serialize the quantizer state (codebook + params).
    /// Virtual: ProductResidualQuantizer uses a distinct (magic-byte-tagged)
    /// format to distinguish its additive codebook layout from PQ.
    virtual void serialize(std::vector<uint8_t>& out) const;

    /// Deserialize from a buffer. Replaces current state.
    virtual void deserialize(const uint8_t* in, size_t size);

protected:
    // Members accessible to AnisotropicPqQuantizer (the subclass overriding
    // train()). Kept non-public so external code can't mutate the codebook
    // or bypass the training invariants.
    MetricKind metric_;
    Dim dim_;
    uint16_t m_;
    uint8_t bits_;
    uint64_t seed_;

    /// Thread count for train(). 0 = use hardware_concurrency() at train()
    /// time. Set via set_num_threads() before train() to cap parallelism
    /// (e.g. when train() runs inside an already-parallel build, spawning
    /// hardware_concurrency() workers per call causes oversubscription).
    uint32_t num_threads_ = 0;

    uint32_t K_;           ///< 2^bits centroids per segment
    uint32_t sub_dim_;     ///< dim / m

    /// Anisotropic (covariance-based scale-transform) codebook training. When
    /// true, each subspace's dims are scaled by √w_d (covariance eigenvalue
    /// weights) before k-means, then centroids are unscaled. Off by default.
    bool anisotropic_ = false;
    float anisotropy_lambda_ = 0.0f;

    /// OPQ rotation matrix (d × d, row-major). Identity-equivalent when empty
    /// (has_rotation_ == false). Applied to vectors before PQ encoding and to
    /// queries before LUT construction. Computed in train() from the PCA of
    /// the training-sample covariance when set_opq() was requested.
    std::vector<float> rotation_;  // rotation_[r*dim + c] = R[r][c]
    bool has_rotation_ = false;
    /// When true, train() computes the PCA rotation from the training sample
    /// and applies it before PQ k-means. Set via enable_opq(); off by default.
    bool opq_enabled_ = false;

    /// Codebook: m segments × K centroids × sub_dim floats.
    std::vector<float> codebook_;

    /// Per-centroid squared L2 norms: m × K floats. Computed at train() time
    /// for the fast L2sq path via decomposition (||q-c||² = ||q||² - 2<q,c> + ||c||²).
    /// Empty when not yet computed.
    std::vector<float> centroid_sqnorms_;

    /// Cross-distance table: m × K × K floats (code-to-code).
    std::vector<float> cross_distance_table_;

    /// Populate `centroid_sqnorms_` from the trained codebook. Called at the
    /// end of train(); cheap (m*K tiny norm computations, ~12K at m=96).
    /// Protected so AnisotropicPqQuantizer::train can call it after refining.
    void compute_centroid_sqnorms_();
};

}  // namespace sextant
