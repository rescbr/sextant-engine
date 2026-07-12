// Vamana core — stub implementation.
// Full port from duckdb-vector-index/src/algo/aisaq/aisaq_core.cpp dispatched separately.

#include "vamana_core.hpp"
#include "../quant/pq_quantizer.hpp"
#include "sextant/error.hpp"

#include <cstring>
#include <algorithm>
#include <thread>

namespace sextant {

// Node layout offsets (matching duckdb-vector-index aisaq_core.hpp:382-386).
// offset 0:   row_id          (8 bytes)
// offset 8:   internal_id     (4 bytes)
// offset 12:  neighbor_count  (2 bytes)
// offset 14:  inline_pq_count (2 bytes)
// offset 16:  neighbor_array  (R × 4 bytes)
// [optional: inline PQ codes for first inline_pq_count neighbors]
inline constexpr uint32_t kRowIdOffset = 0;
inline constexpr uint32_t kInternalIdOffset = 8;
inline constexpr uint32_t kNeighborCountOffset = 12;
inline constexpr uint32_t kInlinePqCountOffset = 14;
inline constexpr uint32_t kNeighborArrayOffset = 16;

void VamanaTLS::resize(uint32_t max_nodes) {
    visited_flags.resize(max_nodes, 0);
    prune_buffer.reserve(max_nodes);
    occlusion_set.reserve(max_nodes);
}

VamanaCore::VamanaCore(VamanaParams params, PqQuantizer& quantizer)
    : params_(params), quantizer_(quantizer) {
    code_size_ = static_cast<uint8_t>(quantizer_.code_size());
    node_size_ = static_node_size(params_.R, params_.inline_pq_count, code_size_);
    num_locks_ = std::max(256u, std::thread::hardware_concurrency() * 4);
    node_locks_ = std::unique_ptr<Mutex[]>(new Mutex[num_locks_]);
}

VamanaCore::~VamanaCore() {
    clear_build_buffers();
}

uint32_t VamanaCore::static_node_size(uint16_t R, uint16_t inline_pq_count,
                                       uint8_t code_size) {
    // (16 + R*4 + 7) & ~7, plus inline codes.
    uint32_t base = (kNeighborArrayOffset + static_cast<uint32_t>(R) * 4 + 7) & ~7u;
    uint32_t inline_bytes = static_cast<uint32_t>(inline_pq_count) * code_size;
    return base + inline_bytes;
}

void VamanaCore::prepare_for_build(uint32_t count) {
    count_ = count;
    // Allocate flat node buffer (all zeroed).
    // build_nodes_ is owned by the caller (Engine), not VamanaCore.
    // This just sets up entry_points_ capacity.
    entry_points_.reserve(params_.n_entry_points);
}

void VamanaCore::set_build_codes(const uint8_t* codes, uint32_t /*count*/) {
    build_codes_ = codes;
}

void VamanaCore::set_build_nodes(uint8_t* nodes) {
    build_nodes_ = nodes;
}

void VamanaCore::clear_build_buffers() {
    build_codes_ = nullptr;
    build_nodes_ = nullptr;
}

uint8_t* VamanaCore::node_ptr(uint32_t internal_id) {
    return build_nodes_ + static_cast<size_t>(internal_id) * node_size_;
}

const uint8_t* VamanaCore::node_ptr(uint32_t internal_id) const {
    return build_nodes_ + static_cast<size_t>(internal_id) * node_size_;
}

Mutex* VamanaCore::node_lock(uint32_t internal_id) {
    return &node_locks_[internal_id % num_locks_];
}
// --- Node accessors ---

RowId VamanaCore::get_row_id(const uint8_t* node) {
    RowId val;
    std::memcpy(&val, node + kRowIdOffset, sizeof(val));
    return val;
}

void VamanaCore::set_row_id(uint8_t* node, RowId val) {
    std::memcpy(node + kRowIdOffset, &val, sizeof(val));
}

uint32_t VamanaCore::get_internal_id(const uint8_t* node) {
    uint32_t val;
    std::memcpy(&val, node + kInternalIdOffset, sizeof(val));
    return val;
}

void VamanaCore::set_internal_id(uint8_t* node, uint32_t val) {
    std::memcpy(node + kInternalIdOffset, &val, sizeof(val));
}

uint16_t VamanaCore::get_neighbor_count(const uint8_t* node) {
    uint16_t val;
    std::memcpy(&val, node + kNeighborCountOffset, sizeof(val));
    return val;
}

void VamanaCore::set_neighbor_count(uint8_t* node, uint16_t val) {
    std::memcpy(node + kNeighborCountOffset, &val, sizeof(val));
}

uint16_t VamanaCore::get_inline_pq_count(const uint8_t* node) {
    uint16_t val;
    std::memcpy(&val, node + kInlinePqCountOffset, sizeof(val));
    return val;
}

void VamanaCore::set_inline_pq_count(uint8_t* node, uint16_t val) {
    std::memcpy(node + kInlinePqCountOffset, &val, sizeof(val));
}

uint32_t VamanaCore::get_neighbor(const uint8_t* node, uint32_t i) {
    uint32_t val;
    std::memcpy(&val, node + kNeighborArrayOffset + i * sizeof(uint32_t),
                sizeof(val));
    return val;
}

void VamanaCore::set_neighbor(uint8_t* node, uint32_t i, uint32_t val) {
    std::memcpy(node + kNeighborArrayOffset + i * sizeof(uint32_t), &val,
                sizeof(val));
}

// --- Algorithm stubs ---

void VamanaCore::insert_build_from_code(uint32_t /*internal_id*/, RowId /*row_id*/,
                                         VamanaTLS& /*tls*/) {
    // TODO: port InsertBuildFromCode from aisaq_core.cpp:732-754.
}

void VamanaCore::insert_build(uint32_t /*internal_id*/, RowId /*row_id*/,
                               const float* /*vec*/, VamanaTLS& /*tls*/) {
    // TODO: port InsertBuild from aisaq_core.cpp:701-730.
}

std::vector<Candidate> VamanaCore::beam_search(
    const float* /*query_lut*/, uint32_t /*L*/, uint32_t /*io_limit*/,
    VamanaTLS& /*tls*/,
    const std::vector<uint32_t>* /*forced_entry_points*/) const {
    return {};  // TODO: port BeamSearch from aisaq_core.cpp:163-283.
}

std::vector<Candidate> VamanaCore::robust_prune(
    std::vector<Candidate> /*candidates*/, uint16_t /*R*/, float /*alpha*/,
    VamanaTLS& /*tls*/, uint32_t /*max_occlusion_size*/) const {
    return {};  // TODO: port RobustPrune from aisaq_core.cpp:290-381.
}

void VamanaCore::connect_and_prune(uint32_t /*new_internal_id*/,
                                    const std::vector<Candidate>& /*selected*/,
                                    VamanaTLS& /*tls*/) {
    // TODO: port ConnectAndPrune from aisaq_core.cpp:417-498.
}

void VamanaCore::finalize_inline_codes() {
    // TODO: port FinalizeInlineCodes from aisaq_core.cpp:786-807.
}

void VamanaCore::compute_entry_points() {
    // TODO: port ComputeEntryPoints from aisaq_core.cpp:819-833.
}

std::vector<Candidate> VamanaCore::search(const float* query_lut, uint32_t k,
                                           uint32_t L_search,
                                           uint32_t io_limit) const {
    VamanaTLS tls;
    tls.resize(count_);
    auto candidates = beam_search(query_lut, L_search, io_limit, tls);
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    return candidates;
}

}  // namespace sextant
