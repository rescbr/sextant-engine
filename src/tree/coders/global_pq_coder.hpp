#pragma once

/// @file global_pq_coder.hpp
/// LeafCoder for the global-codebook PQ families: "pq",
/// "anisotropic_pq"/"anisotropic-pq", and "prq". They differ only in
/// trainer/encoder selection; the FastScan leaf layout and search path are
/// identical.

#include "../leaf_coder.hpp"
#include "quant/pq_quantizer.hpp"

#include <memory>
#include <vector>

namespace sextant::tree {

class GlobalPqCoder final : public LeafCoder {
public:
    enum class Kind : uint8_t { Plain, Anisotropic, Prq };

    /// `params` must already carry the resolved m4 / pq_bits / metric and,
    /// for prq, the (clamped) nsplits + beam size.
    GlobalPqCoder(Kind kind, const CoderParams& params);
    ~GlobalPqCoder() override;

    Kind kind() const { return kind_; }

    // --- LeafCoder ---
    LeafState leaf_state() const override { return LeafState::Coded; }
    CoderFamily family() const override { return CoderFamily::GlobalPq; }
    std::string family_name() const override;
    MetricKind metric() const override;
    const PqQuantizer* pq_quantizer() const override;
    uint32_t prq_nsplits() const override {
        return kind_ == Kind::Prq ? params_.prq_nsplits : 0;
    }

    LeafGeometry geometry(const TreeLeafHeader* h) const override;
    uint64_t extent_bytes(uint32_t count, uint32_t summary_size,
                          uint64_t filter_cols_bytes) const override;
    uint32_t code_size() const override;

    void train(const float* sample, uint32_t n) override;
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
    float rerank(const float* query, const ScanSetup& setup,
                 const uint8_t* leaf, uint32_t local_idx,
                 float* scratch_decoded) override;
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
    void compact(const uint8_t* leaf_in, const uint32_t* keep_idx,
                 uint32_t keep_count, uint8_t* leaf_out) override;

private:
    struct Setup;
    Kind kind_;
    CoderParams params_;
    std::unique_ptr<PqQuantizer> quantizer_;
};

}  // namespace sextant::tree
