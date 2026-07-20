// MemGraph — in-memory entry-point neighborhood cache.
//
// The hot path (pin_node/pin_code) is a single branch + direct-indexed array
// lookup into RAM-resident buffers. The BFS neighborhood is collected at
// construction by walking neighbor lists (offset 12 = count, offset 16 =
// neighbor array — matches the Vamana node layout).

#include "memgraph.hpp"

#include "sextant/error.hpp"
#include "storage/direct_io.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <vector>

namespace sextant {

namespace {

// Node layout offsets (must match src/algo/vamana_core.cpp).
inline constexpr uint32_t kNeighborCountOffset = 12;
inline constexpr uint32_t kNeighborArrayOffset = 16;

// Read the neighbor count from a node record (little-endian uint16 at off 12).
inline uint16_t node_neighbor_count(const uint8_t* node) {
    uint16_t v;
    std::memcpy(&v, node + kNeighborCountOffset, sizeof(v));
    return v;
}

// Read neighbor i from a node record (uint32 at off 16 + i*4).
inline uint32_t node_neighbor(const uint8_t* node, uint32_t i) {
    uint32_t v;
    std::memcpy(&v, node + kNeighborArrayOffset + i * sizeof(uint32_t),
                sizeof(v));
    return v;
}

// Read `count` bytes of payload starting at `offset` from a sidecar DirectFile,
// via an aligned staging buffer (O_DIRECT requirement). Mirrors
// engine_detail::read_exact but stays within the storage layer.
void read_at(DirectFile& f, uint8_t* dst, size_t count, uint64_t offset) {
    if (count == 0) return;
    const size_t aligned =
        (count + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
    void* stage = aligned_alloc(kDiskAlign, aligned);
    std::memset(stage, 0, aligned);
    f.pread_aligned(stage, aligned, offset);
    std::memcpy(dst, stage, count);
    aligned_free(stage);
}

// Read `count` bytes of payload starting just after the SidecarHeader.
void read_payload(DirectFile& f, uint8_t* dst, size_t count) {
    read_at(f, dst, count, sizeof(SidecarHeader));
}

}  // namespace

// ===========================================================================
// Construction
// ===========================================================================

MemGraph::MemGraph(const uint8_t* all_nodes, const uint8_t* all_codes,
                   uint32_t node_size, uint32_t code_size, uint32_t total_count,
                   const std::vector<uint32_t>& entry_points, uint32_t num_hops)
    : node_size_(node_size),
      code_size_(code_size),
      total_count_(total_count) {
    if (node_size_ == 0) {
        throw Error(ErrorCode::InvalidParam, "MemGraph: node_size must be > 0");
    }
    collect_neighborhood(all_nodes, total_count, entry_points, num_hops);
    materialize(all_nodes, all_codes);
}

MemGraph::MemGraph(const std::string& graph_path, const std::string& codes_path,
                   const std::string& vecs_path,
                   uint32_t node_size, uint32_t code_size, uint32_t total_count,
                   uint32_t dim,
                   const std::vector<uint32_t>& entry_points, uint32_t num_hops)
    : node_size_(node_size),
      code_size_(code_size),
      total_count_(total_count),
      dim_(dim) {
    if (node_size_ == 0) {
        throw Error(ErrorCode::InvalidParam, "MemGraph: node_size must be > 0");
    }
    if (total_count_ == 0) {
        membership_.clear();
        id_to_local_.clear();
        return;
    }

    // Load the .graph payload (after the SidecarHeader) into RAM, then BFS.
    // The payload is a flat run of total_count × node_size records; the file
    // may be padded to kDiskAlign at the tail, which we tolerate by reading
    // exactly total_count × node_size bytes via an aligned staging buffer.
    const size_t graph_payload = static_cast<size_t>(total_count_) * node_size_;
    std::vector<uint8_t> graph_buf(graph_payload);
    {
        DirectFile f(graph_path, /*create=*/false);
        read_payload(f, graph_buf.data(), graph_payload);
    }

    collect_neighborhood(graph_buf.data(), total_count_, entry_points, num_hops);

    // Load the codes payload and materialize. Codes are contiguous
    // (total_count × code_size) right after the header.
    const size_t codes_payload =
        static_cast<size_t>(total_count_) * code_size_;
    std::vector<uint8_t> codes_buf(codes_payload);
    {
        DirectFile f(codes_path, /*create=*/false);
        read_payload(f, codes_buf.data(), codes_payload);
    }

    materialize(graph_buf.data(), codes_buf.data());

    // Optionally load the `.ball` FP16 sidecar (ball-only vectors in the same
    // collected/local-index order produced by materialize). Enables the hybrid
    // FP16+PQ distance path in beam_search. Absent or mismatched → PQ-only.
    if (!vecs_path.empty()) {
        std::error_code ec;
        if (std::filesystem::exists(vecs_path, ec)) {
            DirectFile f(vecs_path, /*create=*/false);
            SidecarHeader h;
            read_at(f, reinterpret_cast<uint8_t*>(&h), sizeof(h), 0);
            const uint32_t ball_count = static_cast<uint32_t>(h.n_vectors);
            if (h.dim == dim_ && ball_count == cached_count_) {
                fp16_data_.resize(static_cast<size_t>(ball_count) * dim_);
                read_payload(f, reinterpret_cast<uint8_t*>(fp16_data_.data()),
                             static_cast<size_t>(ball_count) * dim_ *
                                 sizeof(float16_t));
                spdlog::info("[sextant] MemGraph: loaded {} FP16 vectors from {}",
                             ball_count, vecs_path);
            } else {
                spdlog::warn("[sextant] MemGraph: .ball dim={}/ball_count={} != "
                             "expected dim={}/cached_count={}, skipping FP16",
                             h.dim, ball_count, dim_, cached_count_);
            }
        }
    }
}

void MemGraph::collect_neighborhood(const uint8_t* nodes, uint32_t total_count,
                                    const std::vector<uint32_t>& entry_points,
                                    uint32_t num_hops) {
    collected_.clear();
    if (total_count == 0 || entry_points.empty()) {
        membership_.assign(total_count, false);
        id_to_local_.assign(total_count, std::numeric_limits<uint32_t>::max());
        return;
    }

    // Visited bitset (one byte per node — cheaper than std::vector<bool>
    // indirection and plenty fast for the index sizes we cache).
    std::vector<uint8_t> visited(total_count, 0);

    // BFS frontier carries (node_id, hop). Enqueue entry points at hop 0.
    struct Item {
        uint32_t id;
        uint32_t hop;
    };
    std::deque<Item> queue;

    for (uint32_t ep : entry_points) {
        if (ep < total_count && !visited[ep]) {
            visited[ep] = 1;
            queue.push_back({ep, 0});
        }
    }

    while (!queue.empty()) {
        const Item it = queue.front();
        queue.pop_front();
        collected_.push_back(it.id);

        if (it.hop >= num_hops) {
            continue;  // still record this node, but don't expand further
        }

        const uint8_t* node = nodes + static_cast<size_t>(it.id) * node_size_;
        const uint16_t ncount = node_neighbor_count(node);
        // Guard against a corrupt/truncated neighbor count.
        const uint16_t safe_count = std::min<uint32_t>(
            ncount, (node_size_ - kNeighborArrayOffset) / sizeof(uint32_t));
        for (uint16_t i = 0; i < safe_count; i++) {
            const uint32_t nb = node_neighbor(node, i);
            if (nb < total_count && !visited[nb]) {
                visited[nb] = 1;
                queue.push_back({nb, it.hop + 1});
            }
        }
    }

    // membership_/id_to_local_ are finalized in materialize(); pre-size here.
    membership_.assign(total_count, false);
}

void MemGraph::materialize(const uint8_t* nodes, const uint8_t* codes) {
    cached_count_ = static_cast<uint32_t>(collected_.size());

    node_data_.assign(static_cast<size_t>(cached_count_) * node_size_, 0);
    code_data_.assign(static_cast<size_t>(cached_count_) * code_size_, 0);
    id_map_.assign(cached_count_, 0);
    id_to_local_.assign(total_count_, std::numeric_limits<uint32_t>::max());
    membership_.assign(total_count_, false);

    for (uint32_t local = 0; local < cached_count_; local++) {
        const uint32_t id = collected_[local];
        id_map_[local] = id;
        id_to_local_[id] = local;
        membership_[id] = true;

        std::memcpy(node_data_.data() + static_cast<size_t>(local) * node_size_,
                    nodes + static_cast<size_t>(id) * node_size_, node_size_);
        std::memcpy(code_data_.data() + static_cast<size_t>(local) * code_size_,
                    codes + static_cast<size_t>(id) * code_size_, code_size_);
    }

    // Free the BFS scratch; collected_ is no longer needed after materialize.
    collected_.clear();
    collected_.shrink_to_fit();
}

const float16_t* MemGraph::fp16_ptr(uint32_t id) const {
    if (fp16_data_.empty() || dim_ == 0) return nullptr;
    const uint32_t local = (id < id_to_local_.size())
        ? id_to_local_[id]
        : std::numeric_limits<uint32_t>::max();
    if (local == std::numeric_limits<uint32_t>::max()) return nullptr;
    return fp16_data_.data() + static_cast<size_t>(local) * dim_;
}

// ===========================================================================
// NodeStore interface — single branch + array lookup on the hot path.
// ===========================================================================

PinResult MemGraph::pin_node(uint32_t id) {
    if (id < id_to_local_.size() &&
        id_to_local_[id] != std::numeric_limits<uint32_t>::max()) {
        return {node_data_.data() +
                    static_cast<size_t>(id_to_local_[id]) * node_size_,
                false};  // MemGraph hit — served from RAM
    }
    // Cache miss: delegate to backing store, propagate its from_ssd flag.
    return backing_ ? backing_->pin_node(id) : PinResult{};
}

PinResult MemGraph::pin_code(uint32_t id) {
    if (id < id_to_local_.size() &&
        id_to_local_[id] != std::numeric_limits<uint32_t>::max()) {
        return {code_data_.data() +
                    static_cast<size_t>(id_to_local_[id]) * code_size_,
                false};
    }
    return backing_ ? backing_->pin_code(id) : PinResult{};
}

}  // namespace sextant
