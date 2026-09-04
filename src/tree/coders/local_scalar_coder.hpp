#pragma once

/// @file local_scalar_coder.hpp
/// LeafCoder for the "local_scalar"" family (LeafState::CodedLocalScalar):
/// per-leaf uniform lo/steps rulers fitted at flush/split, flat packed
/// nibble codes + optional per-vector fp16 IP bias, per-leaf query
/// transform (a_d = q_d·step_d, c0 = Σ q_d·lo_d), dual-SDOT i8 scan kernel.

#include "../leaf_coder.hpp"

#include <vector>

namespace sextant::tree {

class LocalScalarCoder final : public LeafCoder {
public:
    explicit LocalScalarCoder(const CoderParams& params);
    ~LocalScalarCoder() override = default;

    LeafState leaf_state() const override {
        return LeafState::CodedLocalScalar;
    }
    CoderFamily family() const override { return CoderFamily::LocalScalar; }
    std::string family_name() const override { return "local_scalar"; }
    MetricKind metric() const override { return params_.metric; }
    bool has_global_state() const override { return false; }
    bool stores_raw_vectors_during_build() const override { return true; }

    LeafGeometry geometry(const TreeLeafHeader* h) const override;
    uint64_t extent_bytes(uint32_t count, uint32_t summary_size,
                          uint64_t filter_cols_bytes) const override;
    bool leaf_has_ip_bias() const override { return params_.has_ip_bias; }
    uint32_t code_size() const override;

    void train(const float* sample, uint32_t n) override;
    bool serialize_global(std::vector<uint8_t>& out) const override;
    bool deserialize_global(const uint8_t* data, uint64_t size) override;
    void encode(const float* vec, uint8_t* code_out,
                float* ip_bias_out) override;
    uint64_t flush_leaf(const LeafFlushInput& in, uint8_t* leaf_out) override;

    std::unique_ptr<ScanSetup> scan_setup(const float* query) override;
    bool per_leaf_setup() const override { return true; }
    void bind_leaf(ScanSetup& setup, const uint8_t* leaf) const override;
    void scan_leaf(const ScanSetup& setup, const uint8_t* leaf,
                   RawScanHeap& heap) override;
    float rerank(const float* query, const uint8_t* leaf, uint32_t local_idx,
                 float* scratch_decoded) override;

    SplitPlan plan_split(const uint8_t* leaf, const uint8_t* codes,
                         const float* vecs, uint32_t count,
                         uint32_t leaf_id) override;
    void decode_one(const uint8_t* leaf, uint32_t local_idx,
                    float* out) const override;
    void extract_codes(const uint8_t* leaf, uint32_t count,
                       uint8_t* codes) const override;
    void encode_group(const GroupEncodeInput& in, uint8_t* leaf_out) override;
    void append_encode(const AppendInput& in) override;

    // --- family-owned static kernels (the single copies) ---

    /// Fit per-leaf uniform levels (lo, step) from `count` interleaved
    /// vectors (moved verbatim from ivf_tree_index.cpp).
    static void fit_local_scalar_levels(const float* vecs, uint32_t count,
                                        uint16_t dim, float16_t* lo,
                                        float16_t* steps);

    /// THE single copy of the local_scalar encode loop (was duplicated at
    /// flush / split / insert). Encodes `x` against the leaf's frozen
    /// lo/steps rulers into a flat nibble code; optionally emits the IP
    /// bias ||x||/||x_hat||.
    static void encode_one(const float* x, uint16_t dim,
                           const float16_t* lo, const float16_t* steps,
                           uint8_t* code, float16_t* bias_out);

    /// Decode one flat nibble code with the leaf's rulers.
    static void decode_one_with_levels(const float16_t* lo,
                                       const float16_t* steps,
                                       const uint8_t* code, uint16_t dim,
                                       float* out);

private:
    struct Setup;
    CoderParams params_;
};

}  // namespace sextant::tree
