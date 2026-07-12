// Vamana core — algorithm implementation.
//
// Evolved from Sextant's exploratory phase:
//  - DuckDB dependencies stripped (stdlib + sextant::Error only)
//  - Per-node spinlocks replaced with the sextant::Mutex sharded lock pool
//  - LabelFilter machinery stripped (Phase 1 is label-less)
//  - Storage backend is flat-in-RAM during build (build_codes_ always active)
//
// Algorithmic reference: Suhas Jayaram Subramanya et al., "DiskANN: Fast
// Accurate Billion-point Nearest Neighbor Search on a Single Node", NeurIPS
// 2019 — Vamana variant (build with alpha relaxation).

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "sextant/error.hpp"
#include "sextant/sync.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>
#include <thread>

namespace sextant {

// Node layout offsets.
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

namespace {

// Min-heap frontier (pop closest first).
struct FrontierItem {
    float dist;
    uint32_t internal_id;
};
struct FrontierCmp {
    bool operator()(const FrontierItem& a, const FrontierItem& b) const {
        return a.dist > b.dist;
    }
};

// Max-heap working set W (pop farthest first).
struct WorkingItem {
    float dist;
    uint32_t internal_id;
};
struct WorkingCmp {
    bool operator()(const WorkingItem& a, const WorkingItem& b) const {
        return a.dist < b.dist;
    }
};

}  // namespace

void VamanaTLS::resize(uint32_t max_nodes) {
    visited_flags.resize(max_nodes, 0);
    prune_buffer.reserve(max_nodes);
    occlusion_set.reserve(max_nodes);
}

VamanaCore::VamanaCore(VamanaParams params, PqQuantizer& quantizer)
    : params_(params), quantizer_(quantizer) {
    if (params_.R == 0) {
        throw Error(ErrorCode::InvalidParam, "VamanaCore: R must be >= 1");
    }
    if (params_.L < params_.R) {
        params_.L = params_.R;
    }
    if (params_.alpha < 1.0f) {
        params_.alpha = 1.0f;
    }
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
    entry_points_.clear();
    entry_points_.reserve(std::max<uint16_t>(1, params_.n_entry_points));
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

// ===========================================================================
// beam_search — graph traversal with frontier + working set.
//
// Stripped: LabelFilter parameter, row_id_map, expanded_pool_out.
// The internal candidate type carries internal_id in Candidate::row_id; the
// caller resolves to the table row_id via get_row_id() if needed.
//
// Frontier = min-heap by distance (expand closest first). W = max-heap working
// set, capped at L (evict farthest). Terminates when the next frontier item is
// farther than the current L-th best.
// ===========================================================================

std::vector<Candidate> VamanaCore::beam_search(
    const float* query_lut, uint32_t L, uint32_t io_limit, VamanaTLS& tls,
    const std::vector<uint32_t>* forced_entry_points) const {
    std::vector<Candidate> out;
    if (count_ == 0 || L == 0) {
        return out;
    }
    L = std::max<uint32_t>(L, 1);

    // Bump the visit epoch. Each beam_search call uses a unique non-zero
    // token so visited_flags doubles as a generation marker — no O(n) clear
    // between searches.
    uint32_t vc = ++tls.visit_token;
    if (vc == 0) {
        // Wrapped around: clear the flags and restart at token 1 so we never
        // misread a stale flag as "visited".
        std::fill(tls.visited_flags.begin(), tls.visited_flags.end(), 0);
        vc = tls.visit_token = 1;
    }
    auto mark_visited = [&](uint32_t id) {
        if (id >= tls.visited_flags.size()) {
            tls.visited_flags.resize(
                std::max<size_t>(tls.visited_flags.size() * 2,
                                 static_cast<size_t>(id) + 1),
                0);
        }
        tls.visited_flags[id] = vc;
    };
    auto is_visited = [&](uint32_t id) {
        return id < tls.visited_flags.size() && tls.visited_flags[id] == vc;
    };

    auto dist_to = [&](uint32_t id) {
        if (store_) {
            PinResult pr = store_->pin_code(id);
            const float d = quantizer_.lut_distance(pr.data, query_lut);
            store_->unpin_code(id);
            return d;
        }
        return quantizer_.lut_distance(
            build_codes_ + static_cast<size_t>(id) * code_size_, query_lut);
    };

    std::priority_queue<FrontierItem, std::vector<FrontierItem>, FrontierCmp>
        frontier;
    std::priority_queue<WorkingItem, std::vector<WorkingItem>, WorkingCmp> W;

    uint32_t io_count = 0;

    // -----------------------------------------------------------------------
    // DynamicWidth (PipeANN OSDI 2025, OctopusANN VLDB 2026): two-phase beam
    // search. Start with a small width during the approach phase (rapid
    // navigation toward the query region — large L wastes I/O on far-away
    // nodes) and widen to the full L once the search converges (near the
    // query region — most visited nodes are genuine candidates).
    //
    // Only active on the paged (SSD) search path. When the graph is flat in
    // RAM or fully cached in MemGraph, there's no I/O waste to reduce.
    //
    // Convergence heuristic: if the best distance hasn't improved by more
    // than 1% for 5 consecutive pops, we've entered the converge phase.
    // -----------------------------------------------------------------------
    constexpr uint32_t kConvergePatience = 5;     // pops w/o improvement
    constexpr float kConvergeImprovRatio = 0.99f; // <1% improvement
    // DynamicWidth (PipeANN/OctopusANN): controls the I/O pipeline width —
    // small during approach phase, large during converge phase. This requires
    // async I/O (the Pipeline). With synchronous reads, there is no pipeline
    // width to adjust, so we use the fixed L throughout.
    //
    // TODO: activate when the async I/O pipeline is implemented.
    const bool use_dynamic_width = false;
    uint32_t L_current = use_dynamic_width
        ? std::min(L, std::max(static_cast<uint32_t>(params_.R), L / 4))
        : L;
    bool converged = !use_dynamic_width;
    uint32_t no_improvement_count = 0;
    float prev_best_dist = std::numeric_limits<float>::max();

    // Seed entry points.
    if (forced_entry_points && !forced_entry_points->empty()) {
        for (uint32_t ep_id : *forced_entry_points) {
            if (is_visited(ep_id)) {
                continue;
            }
            mark_visited(ep_id);
            io_count++;
            const float d = dist_to(ep_id);
            frontier.push({d, ep_id});
            W.push({d, ep_id});
            if (W.size() > L_current) {
                W.pop();
            }
        }
    } else {
        // Pick the entry point closest to the query among entry_points_, or
        // fall back to node 0 if none are registered (during early build).
        uint32_t entry_internal = 0;
        if (!entry_points_.empty()) {
            entry_internal = entry_points_[0];
            float best_d = dist_to(entry_internal);
            for (size_t i = 1; i < entry_points_.size(); i++) {
                const float d = dist_to(entry_points_[i]);
                if (d < best_d) {
                    best_d = d;
                    entry_internal = entry_points_[i];
                }
            }
        }
        if (!is_visited(entry_internal)) {
            mark_visited(entry_internal);
            io_count++;
            const float entry_dist = dist_to(entry_internal);
            frontier.push({entry_dist, entry_internal});
            W.push({entry_dist, entry_internal});
        }
    }

    if (frontier.empty()) {
        return out;
    }

    while (!frontier.empty()) {
        const auto best = frontier.top();
        frontier.pop();

        // DynamicWidth: track convergence and widen the beam when it stalls.
        if (!converged) {
            if (best.dist < prev_best_dist * kConvergeImprovRatio) {
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
            }
            prev_best_dist = best.dist;
            if (no_improvement_count >= kConvergePatience) {
                converged = true;
                L_current = L;
            }
        }

        if (W.size() >= L_current && best.dist > W.top().dist) {
            break;
        }

        // -----------------------------------------------------------------
        // Pin the best candidate's node. Capture whether this pin caused
        // I/O (cache miss). If from_ssd, PageSearch is worthwhile — the
        // block read was expensive, so amortize it by scanning co-located
        // nodes. If from_ssd=false (MemGraph hit or LRU hit), skip — the
        // data is already in RAM and extra distance computations are
        // pure overhead.
        // -----------------------------------------------------------------
        uint16_t n;
        uint32_t neighbors_buf[1024];
        const uint32_t* nb_ptr;
        bool node_from_ssd = false;

        if (store_) {
            PinResult pr = store_->pin_node(best.internal_id);
            node_from_ssd = pr.from_ssd;
            n = get_neighbor_count(pr.data);
            nb_ptr = (n <= 1024) ? neighbors_buf : nullptr;
            if (nb_ptr) {
                std::memcpy(neighbors_buf,
                            pr.data + kNeighborArrayOffset,
                            n * sizeof(uint32_t));
            }
            store_->unpin_node(best.internal_id);
        } else {
            const uint8_t* node = node_ptr(best.internal_id);
            n = get_neighbor_count(node);
            if (n <= 1024) {
                std::memcpy(neighbors_buf,
                            node + kNeighborArrayOffset,
                            n * sizeof(uint32_t));
                nb_ptr = neighbors_buf;
            } else {
                nb_ptr = nullptr;
            }
        }

        // -----------------------------------------------------------------
        // PageSearch: only when this node caused an SSD read (from_ssd=true).
        // Scan R co-located nodes in the same block — after PageShuffle, they
        // include the target's nearest graph neighbors.
        // -----------------------------------------------------------------
        if (node_from_ssd) {
            const uint32_t nodes_per_block =
                std::max(1u, kBlockSize / node_size_);
            const uint32_t block_first =
                (best.internal_id / nodes_per_block) * nodes_per_block;
            const uint32_t block_last =
                std::min(block_first + nodes_per_block, count_);
            const uint32_t max_pagescan =
                std::min(nodes_per_block, static_cast<uint32_t>(params_.R));
            const uint32_t scan_last =
                std::min(block_last, block_first + max_pagescan);
            for (uint32_t bid = block_first; bid < scan_last; bid++) {
                if (is_visited(bid)) {
                    continue;
                }
                mark_visited(bid);

                if (io_limit > 0 && io_count >= io_limit) {
                    continue;
                }
                io_count++;
                const float d = dist_to(bid);
                if (W.size() < L_current || d < W.top().dist) {
                    frontier.push({d, bid});
                    W.push({d, bid});
                    if (W.size() > L_current) {
                        W.pop();
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // Expand explicit neighbors of `best`.
        // -----------------------------------------------------------------

        for (uint16_t i = 0; i < n; i++) {
            const uint32_t nb_internal =
                nb_ptr ? nb_ptr[i]
                       : (store_
                              ? get_neighbor(store_->pin_node(
                                                 best.internal_id).data,
                                             i)
                              : get_neighbor(
                                   node_ptr(best.internal_id), i));
            if (store_ && !nb_ptr) {
                store_->unpin_node(best.internal_id);
            }
            if (is_visited(nb_internal)) {
                continue;
            }
            mark_visited(nb_internal);

            if (io_limit > 0 && io_count >= io_limit) {
                continue;  // out of I/O budget: mark visited, skip distance.
            }
            io_count++;
            const float d = dist_to(nb_internal);
            if (W.size() < L_current || d < W.top().dist) {
                frontier.push({d, nb_internal});
                W.push({d, nb_internal});
                if (W.size() > L_current) {
                    W.pop();
                }
            }
        }
    }

    out.reserve(W.size());
    while (!W.empty()) {
        const auto& t = W.top();
        // internal_id stashed in Candidate::row_id; resolved by the caller.
        out.push_back({static_cast<RowId>(t.internal_id), t.dist});
        W.pop();
    }
    std::reverse(out.begin(), out.end());  // ascending distance
    return out;
}

// ===========================================================================
// robust_prune — occlusion-filtered neighbor selection.
//
// Stripped: the query_lut parameter (unused; occlusion uses candidate-candidate
// distances). The flat build buffer is always active, so we index build_codes_
// directly (no per-prune gather).
//
// Occlusion rule (DiskANN): a candidate pp is pruned if there exists an
// already-selected neighbor p such that alpha * d(p, pp) <= d(query, pp).
// The per-anchor LUT is rebuilt once per outer iteration for L2-cache-
// resident distance gathers.
// ===========================================================================

std::vector<Candidate> VamanaCore::robust_prune(
    std::vector<Candidate> candidates, uint16_t R, float alpha,
    VamanaTLS& tls, uint32_t max_occlusion_size) const {
    (void)tls;  // prune_buffer/occlusion_set reserved for a future gather path.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.dist < b.dist;
              });
    // Cap the candidate pool (mirrors DiskANN's maxc cap).
    if (max_occlusion_size > 0 && candidates.size() > max_occlusion_size) {
        candidates.resize(max_occlusion_size);
    }
    const size_t n = candidates.size();
    if (n == 0) {
        return {};
    }

    auto code_at = [&](size_t idx) {
        return build_codes_ +
               static_cast<size_t>(candidates[idx].row_id) * code_size_;
    };

    // Per-anchor LUT scratch (local; rebuilt per outer iteration so the inner
    // loop gathers from an L2-resident m×K buffer instead of striding through
    // the cache-cold cross-distance table).
    const uint32_t lut_sz = quantizer_.lut_size();
    std::vector<float> anchor_lut;
    bool anchor_lut_ok = lut_sz > 0;
    if (anchor_lut_ok) {
        anchor_lut.resize(lut_sz);
    }

    std::vector<bool> removed(n, false);
    std::vector<Candidate> kept;
    kept.reserve(std::min(n, static_cast<size_t>(R)));

    for (size_t p_idx = 0; p_idx < n && kept.size() < static_cast<size_t>(R);
         p_idx++) {
        if (removed[p_idx]) {
            continue;
        }
        kept.push_back(candidates[p_idx]);

        bool lut_ready = false;
        if (anchor_lut_ok) {
            lut_ready =
                quantizer_.build_code_lut(code_at(p_idx), anchor_lut.data());
        }

        for (size_t pp_idx = p_idx + 1; pp_idx < n; pp_idx++) {
            if (removed[pp_idx]) {
                continue;
            }
            float d_pp;
            if (lut_ready) {
                d_pp = quantizer_.lut_distance(code_at(pp_idx),
                                               anchor_lut.data());
            } else {
                d_pp = quantizer_.code_distance(code_at(p_idx),
                                                code_at(pp_idx));
            }
            if (alpha * d_pp <= candidates[pp_idx].dist) {
                removed[pp_idx] = true;
            }
        }
    }
    return kept;
}

// ===========================================================================
// connect_and_prune — wire reciprocal edges with re-pruning.
//
// Forward edges on the new node are written without a lock (each
// new_internal_id is exclusive to one build task). Reciprocal edges each
// acquire the target's sharded lock individually — at most one lock held at
// a time per task, so deadlock is impossible regardless of acquisition order.
//
// On overflow (neighbor count would exceed R), the target's neighbor list is
// re-pruned with robust_prune under the lock.
// ===========================================================================

void VamanaCore::connect_and_prune(uint32_t new_internal_id,
                                    const std::vector<Candidate>& selected,
                                    VamanaTLS& tls) {
    // Forward edges — race-free (disjoint internal_id per task).
    // Write neighbor slots BEFORE publishing the count so a concurrent
    // beam_search reader never observes a non-zero count with uninitialized
    // slots (which would traverse to node 0).
    {
        uint8_t* node = node_ptr(new_internal_id);
        const uint16_t cnt =
            static_cast<uint16_t>(std::min<size_t>(selected.size(), params_.R));
        for (uint16_t i = 0; i < cnt; i++) {
            set_neighbor(node, i,
                         static_cast<uint32_t>(selected[i].row_id));
        }
        set_neighbor_count(node, cnt);
    }

    const uint8_t* new_code =
        build_codes_ + static_cast<size_t>(new_internal_id) * code_size_;

    // Reciprocal edges. Each target's lock is acquired and released in
    // isolation — no nested lock acquisition anywhere in this loop.
    for (const auto& s : selected) {
        const uint32_t s_internal = static_cast<uint32_t>(s.row_id);

        ScopedWriteLock guard(*node_lock(s_internal));

        uint8_t* s_node = node_ptr(s_internal);
        uint16_t cnt = get_neighbor_count(s_node);
        if (cnt < params_.R) {
            // Common case: room left. Append the back-edge.
            set_neighbor(s_node, cnt, new_internal_id);
            set_neighbor_count(s_node, static_cast<uint16_t>(cnt + 1));
            continue;
        }

        // Overflow: re-prune s's neighbor list. Candidate pool =
        // [new_internal, s's current neighbors], distances measured from s.
        std::vector<Candidate> cand;
        cand.reserve(static_cast<size_t>(cnt) + 1);
        const uint8_t* s_code =
            build_codes_ + static_cast<size_t>(s_internal) * code_size_;
        cand.push_back({static_cast<RowId>(new_internal_id),
                        quantizer_.code_distance(s_code, new_code)});
        for (uint16_t i = 0; i < cnt; i++) {
            const uint32_t nb = get_neighbor(s_node, i);
            const uint8_t* nb_code =
                build_codes_ + static_cast<size_t>(nb) * code_size_;
            cand.push_back({static_cast<RowId>(nb),
                            quantizer_.code_distance(s_code, nb_code)});
        }
        auto kept = robust_prune(std::move(cand), params_.R, params_.alpha, tls,
                                 params_.max_occlusion);
        // Write neighbors before publishing count.
        for (size_t i = 0; i < kept.size(); i++) {
            set_neighbor(s_node, static_cast<uint16_t>(i),
                         static_cast<uint32_t>(kept[i].row_id));
        }
        set_neighbor_count(s_node, static_cast<uint16_t>(kept.size()));
    }
}

// ===========================================================================
// insert_build_from_code — SDC build path (LUT from own PQ code).
// SDC build mode: build the query LUT from the node's own PQ code.
// ===========================================================================

void VamanaCore::insert_build_from_code(uint32_t internal_id, RowId row_id,
                                         VamanaTLS& tls) {
    if (!build_nodes_ || !build_codes_) {
        throw Error(ErrorCode::InvalidParam,
                    "VamanaCore::insert_build_from_code: build buffers not set");
    }

    // Zero the fixed header + neighbor array. The inline PQ region, if any,
    // is filled later by finalize_inline_codes().
    uint8_t* node = node_ptr(internal_id);
    std::memset(node, 0,
                kNeighborArrayOffset +
                    static_cast<size_t>(params_.R) * sizeof(uint32_t));
    set_row_id(node, row_id);
    set_internal_id(node, internal_id);
    set_neighbor_count(node, 0);
    set_inline_pq_count(node, params_.inline_pq_count);

    if (internal_id >= tls.visited_flags.size()) {
        tls.visited_flags.resize(
            std::max<size_t>(tls.visited_flags.size() * 2,
                             static_cast<size_t>(internal_id) + 1),
            0);
    }

    // First-insert claim of the entry point. The parallel build coordinator
    // MUST serialize the first insert across all tasks (run it once before
    // spawning tasks). Subsequent calls are safe on disjoint id ranges.
    if (count_ == 0 || entry_points_.empty()) {
        entry_points_.clear();
        entry_points_.push_back(internal_id);
        count_ = std::max(count_, internal_id + 1);
        return;
    }

    // Build the query LUT from the node's OWN PQ code (symmetric distance):
    // build_code_lut extracts the code's row from the cross-distance table,
    // giving a LUT identical in layout to preprocess_query's output.
    const uint32_t lut_sz = quantizer_.lut_size();
    std::vector<float> lut(lut_sz > 0 ? lut_sz : 1, 0.0f);
    if (lut_sz > 0) {
        const bool ok = quantizer_.build_code_lut(
            build_codes_ + static_cast<size_t>(internal_id) * code_size_,
            lut.data());
        if (!ok) {
            // Quantizer doesn't support SDC LUTs: fall back to zeros. The
            // build continues (degraded topology); the engine logs this.
        }
    }

    // InsertBuild tail: beam_search → robust_prune → connect_and_prune.
    const uint32_t L_build =
        params_.L_build > 0 ? params_.L_build : params_.L;
    auto selected_candidates =
        beam_search(lut.data(), L_build, 0 /* io_limit=0 → unlimited */, tls);
    auto selected = robust_prune(std::move(selected_candidates), params_.R,
                                 params_.alpha, tls, params_.max_occlusion);
    connect_and_prune(internal_id, selected, tls);
    count_ = std::max(count_, internal_id + 1);
}

// ===========================================================================
// insert_build — ADC build path (LUT from raw vector).
// ADC build mode: build the query LUT from the raw vector.
// ===========================================================================

void VamanaCore::insert_build(uint32_t internal_id, RowId row_id,
                               const float* vec, VamanaTLS& tls) {
    if (!build_nodes_) {
        throw Error(ErrorCode::InvalidParam,
                    "VamanaCore::insert_build: build buffer not set");
    }

    uint8_t* node = node_ptr(internal_id);
    std::memset(node, 0,
                kNeighborArrayOffset +
                    static_cast<size_t>(params_.R) * sizeof(uint32_t));
    set_row_id(node, row_id);
    set_internal_id(node, internal_id);
    set_neighbor_count(node, 0);
    set_inline_pq_count(node, params_.inline_pq_count);

    if (internal_id >= tls.visited_flags.size()) {
        tls.visited_flags.resize(
            std::max<size_t>(tls.visited_flags.size() * 2,
                             static_cast<size_t>(internal_id) + 1),
            0);
    }

    if (count_ == 0 || entry_points_.empty()) {
        entry_points_.clear();
        entry_points_.push_back(internal_id);
        count_ = std::max(count_, internal_id + 1);
        return;
    }

    // Build a LUT from the raw vector for distance estimates during construct.
    const uint32_t lut_sz = quantizer_.lut_size();
    std::vector<float> lut(lut_sz > 0 ? lut_sz : 1, 0.0f);
    if (lut_sz > 0) {
        quantizer_.preprocess_query(vec, lut.data());
    }

    // InsertBuild tail: beam_search → robust_prune → connect_and_prune.
    const uint32_t L_build =
        params_.L_build > 0 ? params_.L_build : params_.L;
    auto selected_candidates =
        beam_search(lut.data(), L_build, 0 /* io_limit=0 → unlimited */, tls);
    auto selected = robust_prune(std::move(selected_candidates), params_.R,
                                 params_.alpha, tls, params_.max_occlusion);
    connect_and_prune(internal_id, selected, tls);
    count_ = std::max(count_, internal_id + 1);
}

// ===========================================================================
// finalize_inline_codes — copy neighbor PQ codes into inline region.
// Serial sweep; parallelism is orchestrated at the engine level if desired.
//
// After all nodes are built, copy each neighbor's PQ code into the inline
// region of the node so search can read neighbor codes without a separate
// codes fetch for the first inline_pq_count neighbors.
// ===========================================================================

void VamanaCore::finalize_inline_codes() {
    if (params_.inline_pq_count == 0 || count_ == 0) {
        return;
    }
    const uint32_t neighbor_region_end =
        kNeighborArrayOffset +
        static_cast<uint32_t>(params_.R) * sizeof(uint32_t);
    // The inline region starts at the 8-byte-aligned boundary after the
    // neighbor array (matches static_node_size's base padding).
    const uint32_t inline_region_off = (neighbor_region_end + 7u) & ~7u;

    for (uint32_t id = 0; id < count_; id++) {
        uint8_t* node = node_ptr(id);
        const uint16_t n = get_neighbor_count(node);
        const uint16_t nin =
            std::min<uint16_t>(n, params_.inline_pq_count);
        for (uint16_t i = 0; i < nin; i++) {
            const uint32_t nb = get_neighbor(node, i);
            std::memcpy(node + inline_region_off +
                            static_cast<size_t>(i) * code_size_,
                        build_codes_ + static_cast<size_t>(nb) * code_size_,
                        code_size_);
        }
    }
}

// ===========================================================================
// compute_entry_points — select entry points via evenly-spaced node IDs.
//
// Selects up to n_entry_points evenly-spread nodes. The original caches each
// entry point's PQ code inline; here beam_search reads codes from the flat
// build buffer directly, so we only need the internal_ids.
// ===========================================================================

void VamanaCore::compute_entry_points() {
    entry_points_.clear();
    if (count_ == 0) {
        return;
    }
    const uint16_t want =
        std::min<uint32_t>(params_.n_entry_points, count_);
    entry_points_.reserve(want);
    for (uint16_t i = 0; i < want; i++) {
        // Evenly-spaced internal_ids across the dataset.
        const uint32_t id =
            (static_cast<uint32_t>(i) * count_) / static_cast<uint32_t>(want);
        entry_points_.push_back(id);
    }
}

// ===========================================================================
// search — top-k retrieval. LabelFilter stripped.
//
// The internal candidates from beam_search carry internal_id in row_id; we
// resolve each to its table row_id here.
// ===========================================================================

std::vector<Candidate> VamanaCore::search(const float* query_lut, uint32_t k,
                                           uint32_t L_search,
                                           uint32_t io_limit) const {
    if (count_ == 0 || k == 0) {
        return {};
    }
    if (L_search == 0) {
        L_search = params_.L;
    }
    if (L_search < k) {
        L_search = k;
    }

    VamanaTLS tls;
    tls.resize(count_);

    auto cands = beam_search(query_lut, L_search, io_limit, tls);
    if (cands.size() > k) {
        cands.resize(k);
    }
    // Resolve internal_id → row_id.
    for (auto& c : cands) {
        const uint32_t iid = static_cast<uint32_t>(c.row_id);
        if (iid < count_) {
            if (store_) {
                PinResult pr = store_->pin_node(iid);
                const RowId rid = get_row_id(pr.data);
                store_->unpin_node(iid);
                c.row_id = rid;
            } else {
                c.row_id = get_row_id(node_ptr(iid));
            }
        }
    }
    return cands;
}

}  // namespace sextant
