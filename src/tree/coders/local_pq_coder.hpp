#pragma once

/// @file local_pq_coder.hpp
/// LeafCoder for the "local_pq" family (LeafState::CodedLocal): per-leaf
/// FP32 centroid + per-leaf PQ codebook trained on residuals at flush time,
/// per-leaf LUT scan, residual encode on insert, retrain-per-half on split.

#include "../leaf_coder.hpp"

#include <memory>
#include <vector>

namespace sextant {
class PqQuantizer;
}

namespace sextant::tree {

class LocalPqCoder final : public LeafCoder {
public:
    explicit LocalPqCoder(const CoderParams& params);
    ~LocalPqCoder() override;

    LeafState leaf_state() const override { return LeafState::CodedLocal; }
    CoderFamily family() const override { return CoderFamily::LocalPq; }
    std::string family_name() const override { return "local_pq"; }
    MetricKind metric() const override { return params_.metric; }
    bool has_global_state() const override { return false; }
    bool stores_raw_vectors_during_build() const override { return true; }

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
    bool per_leaf_setup() const override { return true; }
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

    /// Train a fresh per-leaf codebook on `count` vectors' residuals about
    /// `centroid` and encode all of them into `codes` (count × code_size()).
    /// Shared by flush_leaf and encode_group (the two train sites).
    void train_leaf_codebook(const float* vecs, uint32_t count,
                             const float* centroid, uint8_t* codes) const;

private:
    struct Setup;
    /// Shared decode+distance core for both rerank entry points (ctx-cache
    /// and setup-bound codebook).
    float rerank_with(const float* query, const uint8_t* leaf,
                      uint32_t local_idx, float* scratch_decoded,
                      PqQuantizer& quant);
    CoderParams params_;
};

}  // namespace sextant::tree
