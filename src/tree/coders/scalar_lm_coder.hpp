#pragma once

/// @file scalar_lm_coder.hpp
/// LeafCoder for the global scalar families "scalar_lloydmax",
/// "scalar_uniform", "scalar_shape" (one class; the level policy only
/// changes training — leaf layouts are identical: flat packed nibbles +
/// optional per-vector fp16 IP bias region between codes and row_ids).

#include "../leaf_coder.hpp"

#include <memory>
#include <vector>

namespace sextant {
class ScalarLloydMaxQuantizer;
}

namespace sextant::tree {

class ScalarLmCoder final : public LeafCoder {
public:
    enum class LevelPolicy : uint8_t { LloydMax, Uniform, Shape };

    ScalarLmCoder(LevelPolicy policy, const CoderParams& params);
    ~ScalarLmCoder() override;

    LevelPolicy policy() const { return policy_; }

    LeafState leaf_state() const override { return LeafState::Coded; }
    CoderFamily family() const override { return CoderFamily::ScalarLm; }
    std::string family_name() const override;
    MetricKind metric() const override { return params_.metric; }

    LeafGeometry geometry(const TreeLeafHeader* h) const override;
    uint64_t extent_bytes(uint32_t count, uint32_t summary_size,
                          uint64_t filter_cols_bytes) const override;
    bool leaf_has_ip_bias() const override { return params_.has_ip_bias; }
    uint32_t code_size() const override;

    void train(const float* sample, uint32_t n) override;
    void set_train_threads(uint32_t t) override;
    bool serialize_global(std::vector<uint8_t>& out) const override;
    bool deserialize_global(const uint8_t* data, uint64_t size) override;
    void encode(const float* vec, uint8_t* code_out,
                float* ip_bias_out) override;
    uint64_t flush_leaf(const LeafFlushInput& in, uint8_t* leaf_out) override;

    std::unique_ptr<ScanSetup> scan_setup(const float* query) override;
    bool per_leaf_setup() const override { return false; }
    void bind_leaf(ScanSetup& setup, const uint8_t* leaf) const override;
    void scan_leaf(const ScanSetup& setup, const uint8_t* leaf,
                   RawScanHeap& heap) override;
    float rerank(const float* query, const uint8_t* leaf, uint32_t local_idx,
                 float* scratch_decoded) override;
    float rerank(const float* query, const ScanSetup& setup,
                 const uint8_t* leaf, uint32_t local_idx,
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

    const ScalarLloydMaxQuantizer& quantizer() const { return *quantizer_; }

private:
    struct Setup;
    LevelPolicy policy_;
    CoderParams params_;
    std::unique_ptr<ScalarLloydMaxQuantizer> quantizer_;
};

}  // namespace sextant::tree
