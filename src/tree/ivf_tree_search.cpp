#include "ivf_tree_index.hpp"

#include "engine/manifest_io.hpp"
#include "engine/probe.hpp"
#include "engine/partition.hpp"
#include "util/fp16.hpp"
#include "simd_kernels.hpp"
#include "tree/filter_column_write.hpp"  // filter column write path
#include "tree/filter_column_read.hpp"   // filter column read path (mutable ops)
#include "tree/filter_scan.hpp"         // filter predicate evaluation
#include "tree/coders/coder_factory.hpp"
#include "tree/coders/global_pq_coder.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"
#include "sextant/engine_trace.hpp"
#include "sextant/vector_source.hpp"
#include "sextant/filter_column_data.hpp"
#include "sextant/phase_timer.hpp"

#include <spdlog/spdlog.h>

#include <ctpl/ctpl_stl_tls.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <numeric>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>


namespace sextant::tree {

// ===========================================================================
// Search
// ===========================================================================

namespace {

/// A max-heap entry for the FastScan candidate selection. Carries enough
/// state to decode the candidate's PQ code back to FP32 for reranking:
/// `leaf_ptr` (the mmap base of the leaf extent) and `local_idx` (the
/// vector's index within that leaf). `pq_dist` is the PQ-approximate
/// uint32 ADC distance used to order the heap during the scan.
/// Scan-heap entry: the minimum state needed to maintain the top-W max-heap
/// — 12 bytes vs the previous 32. Sifting moves 2.7x less data, which the
/// pq4 profiles showed as 16% of scan CPU (weak ranking -> many replaces).
/// row_id/leaf_ptr are re-derived AFTER the scan for the <=W survivors
/// (materialized into HeapEntryFull) — one load each, negligible.
/// Post-scan form: everything the extraction/rerank/filter paths need.
/// Produced from HeapEntry by a <=W-entry materialization pass.
struct HeapEntryFull {
    uint32_t pq_dist;
    int64_t  row_id;
    const uint8_t* leaf_ptr;   // scan/rerank pointer (mmap or cache buffer)
    const uint8_t* mmap_ptr;   // always the mmap base of the extent — for
                               // pointers that outlive the search (payload
                               // locs): cache pins are released at exit
    uint32_t local_idx;        // vector index within the leaf
};

}  // namespace
// ===========================================================================
// SearchScratch — per-thread reusable arena for search().
//
// search() allocates ~15-20 vectors per query (LUTs, routing frontier, heap,
// rerank buffers, results). Under query-level parallelism (CLI std::async
// workers) those allocations hammer malloc from every thread. This arena is
// thread_local: buffers are clear()ed per call and keep their capacity, so a
// warm thread performs zero heap allocations in the steady state. Retained
// memory is bounded by W_max × sizeof(HeapEntry) plus LUTs (a few MB/thread).
//
// The within-query parallel scan path (search_threads > 1) gives each worker
// a private heap from `worker_heaps`: resized to the worker count per call,
// cleared (not destroyed) so capacity persists across queries.
// ===========================================================================
namespace {

/// Result candidate carrying its payload location (was local to search()).
struct ResultWithLoc {
    Candidate cand;
    const uint8_t* leaf_ptr;
    uint32_t local_idx;
};

}  // namespace (anon, re-opened below)

void set_scan_i8_override(int v) { scan_detail::set_override(v); }

namespace {


}  // namespace

struct SearchScratch {
    // query prep
    std::vector<float16_t> query_fp16;
    std::vector<float> query_pca;
    // predicate column resolution
    std::vector<uint32_t> pred_col_indices, geo_lng_col_indices;
    // routing
    std::vector<std::pair<float, uint32_t>> root_dists, child_dists;
    std::vector<const uint8_t*> root_summaries;
    std::vector<ProbeEntry> frontier, next_frontier;
    std::vector<uint32_t> root_idx, next_root_idx;
    std::vector<LeafCandidate> candidates, pruned_candidates;
    // routing side channels (filled by route_query_, read by the caller):
    // predicate selectivity, accumulated internal-node bytes, resolved gap
    // pruning threshold, and whether feedback probing is active for this
    // config (the feedback loop in search() reuses root_dists + gap).
    float selectivity = 1.0f;
    uint64_t route_node_bytes = 0;
    float route_gap = 0.0f;
    bool route_feedback = false;
    // scan + predicate filtering of the heap
    std::vector<HeapEntry> heap;              // scan-time (12B entries)
    std::vector<HeapEntryFull> heap_full;      // materialized post-scan
    std::vector<HeapEntryFull> filtered_heap;
    std::vector<std::vector<HeapEntry>> worker_heaps;
    std::vector<ColumnView> filter_cols;
    // results
    std::vector<Candidate> results;
    std::vector<ResultWithLoc> results_loc;
    // sweep scratch
    struct SweepEntry { uint32_t pq_dist; RowId row_id; float dist; };
    std::vector<SweepEntry> sweep_entries;
    std::vector<uint32_t> sweep_order;
    std::vector<Candidate> sweep_work;
    std::vector<std::pair<uint32_t, uint32_t>> sweep_plan;  // (W, out index)
    // LeafExtentCache pins: every pin taken during the query lives here
    // until QueryGuard exit (scan + materialization + rerank share the
    // buffers; unpinning mid-query would force refills).
    std::vector<LeafExtentCache::Handle> pins;
    std::vector<std::vector<LeafExtentCache::Handle>> worker_pins;
    // Cache-mode fill stage: candidate indices in page-sorted order + one
    // resolved pointer per candidate (shared by scan + materialization).
    std::vector<uint32_t> leaf_order;
    std::vector<const uint8_t*> leaf_ptrs;
};

bool IVFTreeIndex::resolve_pred_columns_(
        const SearchConfig& config,
        std::vector<uint32_t>& pred_col_indices,
        std::vector<uint32_t>& geo_lng_col_indices) const {
    pred_col_indices.clear();
    geo_lng_col_indices.clear();
    if (config.predicates.empty()) return true;
    pred_col_indices.reserve(config.predicates.size());
    geo_lng_col_indices.resize(config.predicates.size(), UINT32_MAX);
    for (uint32_t pi = 0; pi < config.predicates.size(); ++pi) {
        const auto& pred = config.predicates[pi];
        const auto* col = manifest_.schema.find(pred.column);
        if (!col) return false;  // Predicate references unknown column.
        pred_col_indices.push_back(
            static_cast<uint32_t>(col - manifest_.schema.columns.data()));
        // Geo predicates need a second column (longitude).
        if (pred.op == PredicateOp::GeoRadius ||
            pred.op == PredicateOp::GeoBox) {
            if (pred.geo_lng_column.empty()) return false;
            const auto* lng_col = manifest_.schema.find(pred.geo_lng_column);
            if (!lng_col) return false;
            geo_lng_col_indices[pi] = static_cast<uint32_t>(
                lng_col - manifest_.schema.columns.data());
        }
    }
    return true;
}

void IVFTreeIndex::expand_probe_(const ProbeEntry& e, bool use_pca_leaves,
                                 uint32_t root_child_for_pca, float gap,
                                 uint32_t n_probe_ln,
                                 const SearchConfig& config,
                                 const std::vector<uint32_t>& pred_col_indices,
                                 const std::vector<uint32_t>& geo_lng_col_indices,
                                 SearchScratch& scratch,
                                 std::vector<ProbeEntry>& out,
                                 uint64_t& node_bytes) const {
    const bool has_predicates = !config.predicates.empty();
    const uint32_t cesize =
        child_entry_size(manifest_.dim, manifest_.summary_size);
    const uint8_t* node_ptr = mmap_base_ +
        static_cast<uint64_t>(e.page) * kPageSize;
    const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
    node_bytes += static_cast<uint64_t>(e.pages) * kPageSize;
    const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);

    auto& child_dists = scratch.child_dists;
    child_dists.clear();
    child_dists.reserve(nh->n_children);
    // Summary offset within each child entry (after the inline FP16
    // centroid). Used for subtree-level pruning during routing.
    const uint32_t child_summary_off =
        sizeof(ChildEntry) + manifest_.dim * sizeof(float16_t);
    for (uint32_t j = 0; j < nh->n_children; ++j) {
        // Summary-aware routing: skip children whose filter summary rules
        // out all matches for the predicates (prunes whole subtrees during
        // descent, not just leaves after descent). Conservative — never
        // produces false negatives.
        if (has_predicates && manifest_.summary_size > 0) {
            const uint8_t* child_summary = p + child_summary_off;
            if (!summary_may_match(child_summary, manifest_.summary_size,
                                   manifest_.schema, config.predicates,
                                   pred_col_indices, geo_lng_col_indices)) {
                p += cesize;
                continue;  // PRUNED: subtree can't contain matches
            }
        }
        float d;
        if (use_pca_leaves && root_child_for_pca < pca_leaf_base_.size() &&
            pca_leaf_base_[root_child_for_pca] != UINT64_MAX) {
            const uint32_t gid = static_cast<uint32_t>(
                pca_leaf_base_[root_child_for_pca]) + j;
            const float* lc = &pca_leaf_centroids_[gid * pca_dims_];
            d = 0.0f;
            for (uint32_t kk = 0; kk < pca_dims_; ++kk) {
                const float diff = scratch.query_pca[kk] - lc[kk];
                d += diff * diff;
            }
        } else {
            const float16_t* cent = reinterpret_cast<const float16_t*>(
                p + sizeof(ChildEntry));
            d = simd::dist_f16(coder_->metric(), scratch.query_fp16.data(),
                               cent, manifest_.dim);
        }
        child_dists.emplace_back(d, j);
        p += cesize;
    }
    std::sort(child_dists.begin(), child_dists.end());

    // n_probe_ln is bounded by child_dists.size(), not nh->n_children:
    // summary-aware pruning above may have removed children that can't
    // match the predicates, so child_dists can be smaller than
    // n_children. Using n_children here causes an out-of-bounds access
    // under heavy filtering.
    const uint32_t n_probe_ln_eff = std::min(
        static_cast<uint64_t>(n_probe_ln),
        static_cast<uint64_t>(child_dists.size()));
    const uint8_t* p2 = node_ptr + sizeof(TreeNodeHeader);
    for (uint32_t j = 0; j < n_probe_ln_eff; ++j) {
        if (gap > 0 && j > 0 &&
            child_dists[j].first > child_dists[j - 1].first * gap) break;
        const uint32_t idx = child_dists[j].second;
        const auto* ce = reinterpret_cast<const ChildEntry*>(
            p2 + idx * cesize);
        if (ce->child_page == kInvalidPage) continue;  // empty child
        // Leaf children store a leaf_id (index into leaf_table_);
        // internal children store a physical page directly.
        PageId rpage = ce->child_page;
        uint64_t rpages = ce->child_pages;
        if (ce->is_leaf) {
            rpage = leaf_table_[ce->child_page].page;
            rpages = leaf_table_[ce->child_page].pages;
        }
        const float16_t* cent = reinterpret_cast<const float16_t*>(
            reinterpret_cast<const uint8_t*>(ce) + sizeof(ChildEntry));
        out.push_back({child_dists[j].first, rpage,
                       rpages, ce->is_leaf, cent});
    }
}

IVFTreeIndex::RouteStatus IVFTreeIndex::route_query_(
        const float* query, const SearchConfig& config,
        SearchScratch& scratch,
        std::vector<LeafCandidate>& candidates) const {
    candidates.clear();

    // Cast query to FP16 for routing (non-PCA path + leaf centroid reads).
    auto& query_fp16 = scratch.query_fp16;
    query_fp16.resize(manifest_.dim);
    cast_fp32_to_fp16(query, query_fp16.data(), manifest_.dim);

    // --- Project query to PCA space if PCA routing is enabled ---
    auto& query_pca = scratch.query_pca;
    query_pca.clear();
    if (pca_dims_ > 0) {
        query_pca.resize(pca_dims_);
        for (uint32_t k = 0; k < pca_dims_; ++k) {
            query_pca[k] = simd::dot_f32(&pca_proj_[k * manifest_.dim],
                                          query, manifest_.dim)
                           - pca_mean_proj_[k];
        }
    }

    // --- Resolve predicate column indices (once per query) ---
    if (!resolve_pred_columns_(config, scratch.pred_col_indices,
                               scratch.geo_lng_col_indices)) {
        return RouteStatus::Empty;  // unknown predicate column
    }
    const bool has_predicates = !config.predicates.empty();

    // --- Descent configuration ---
    // Adaptive gap resolution (SearchConfig::adaptive_probe_gap):
    //   <0 = off (disable early-exit entirely)
    //    0 = auto (use the value baked into the manifest at build time)
    //   >0 = explicit override
    // NOTE: gap pruning is a QPS/recall trade knob. On noise-dominated
    // embeddings (e.g. Cohere), centroid distances are nearly uniform, so
    // even a modest gap (1.5) prunes probing to a few leaves and silently
    // destroys recall. Disabled by default; see ResolvedParams.
    float gap = manifest_.adaptive_probe_gap;
    if (config.adaptive_probe_gap < 0) gap = 0.0f;
    else if (config.adaptive_probe_gap > 0) gap = config.adaptive_probe_gap;
    scratch.route_gap = gap;

    // Probe-fraction routing (leaf-coverage contract). Precedence:
    // explicit n_probe (absolute, expert) > probe_fraction (call or
    // manifest) > legacy manifest counts.
    float probe_frac = config.probe_fraction;
    if (probe_frac <= 0.0f && config.n_probe == 0) {
        probe_frac = manifest_.probe_fraction;
    }
    const bool fraction_routing = probe_frac > 0.0f && config.n_probe == 0;

    // Scan-feedback probing (FeedbackProbe): root children probed in
    // routing order one subtree block at a time, stopping on scan feedback
    // instead of a fixed fraction. Not coalescible — search_batch rejects
    // it; search() runs the feedback loop itself (reusing root_dists).
    const bool feedback_active =
        config.feedback.mode != FeedbackProbe::Mode::Off
        && config.n_probe == 0 && !has_predicates
        && manifest_.depth <= 2;
    scratch.route_feedback = feedback_active;

    const uint32_t n_probe_ln_cfg = (fraction_routing || feedback_active)
        ? UINT32_MAX  // probe ALL leaves of each selected root child
        : (config.n_probe_ln > 0
               ? config.n_probe_ln
               : (manifest_.n_probe_ln > 0 ? manifest_.n_probe_ln : 4));

    // --- Phase D: compute selectivity BEFORE routing ---
    float selectivity = 1.0f;
    if (has_predicates) {
        if (!card_table_.empty()) {
            selectivity = card_table_.selectivity_combined(
                manifest_.schema, config.predicates, scratch.pred_col_indices,
                scratch.geo_lng_col_indices);
        } else {
            // Fallback: root-summary-based subtree overlap estimation.
            selectivity = 1.0f;
            auto& root_summaries = scratch.root_summaries;
            root_summaries.clear();
            root_summaries.resize(root_children_.size());
            const uint32_t dim16 = manifest_.dim;
            for (uint32_t c = 0; c < root_children_.size(); ++c) {
                root_summaries[c] = reinterpret_cast<const uint8_t*>(
                    root_children_[c].centroid) + dim16 * sizeof(float16_t);
            }
            for (uint32_t p = 0; p < config.predicates.size(); ++p) {
                const auto& pred = config.predicates[p];
                const uint32_t col_idx = scratch.pred_col_indices[p];
                const auto& col = manifest_.schema.columns[col_idx];
                float s = 1.0f;
                if (col.type == ColumnType::Int32 || col.type == ColumnType::Int64 ||
                    col.type == ColumnType::Float) {
                    s = estimate_numeric_selectivity(
                        root_summaries.data(),
                        static_cast<uint32_t>(root_children_.size()),
                        manifest_.summary_size, manifest_.schema,
                        pred, col_idx);
                }
                selectivity *= std::max(s, 0.0001f);
            }
            selectivity = std::min(selectivity, 1.0f);
        }
    }
    scratch.selectivity = selectivity;

    // Brute-force PQ-decode fallback for extreme low selectivity (<1%).
    // Triggered before routing — no point routing when we'll scan all
    // matching leaves anyway.
    if (has_predicates && selectivity > 0.0f && selectivity < 0.01f) {
        return RouteStatus::FallbackFiltered;
    }

    // --- Level 0: root children ---
    const uint32_t k_root = root_header_->n_children;
    uint32_t n_probe_l0 = config.n_probe > 0
        ? config.n_probe
        : manifest_.n_probe_l0;
    n_probe_l0 = std::min(n_probe_l0, k_root);

    // MUST_ENTER filter-directed routing (§3.8): at low selectivity (≤20%),
    // the matching vectors concentrate in a few subtrees that may be FAR from
    // the query centroid. Normal centroid-distance ranking would skip them.
    // Instead, probe ALL root children whose summary indicates they CAN contain
    // matches. The summary pruning in the scoring loop below already removes
    // children that can't match; here we widen n_probe_l0 to include every
    // surviving child when selectivity is low.
    const bool filter_directed = has_predicates && selectivity <= 0.20f;

    // Score root children, sort, select top-n_probe_l0 with gap pruning.
    auto& root_dists = scratch.root_dists;
    root_dists.clear();
    root_dists.reserve(k_root);
    // Summary offset within each root child entry (after the inline FP16
    // centroid). root_children_[c].centroid points at the centroid, which
    // sits at child-entry offset sizeof(ChildEntry); the summary follows.
    const uint32_t root_child_summary_bytes =
        manifest_.dim * sizeof(float16_t);
    for (uint32_t c = 0; c < k_root; ++c) {
        // Summary-aware routing at the root: prune whole root subtrees whose
        // filter summary rules out all predicate matches.
        if (has_predicates && manifest_.summary_size > 0) {
            const uint8_t* child_summary =
                reinterpret_cast<const uint8_t*>(root_children_[c].centroid)
                + root_child_summary_bytes;
            if (!summary_may_match(child_summary, manifest_.summary_size,
                                   manifest_.schema, config.predicates,
                                   scratch.pred_col_indices,
                                   scratch.geo_lng_col_indices)) {
                continue;  // PRUNED: root subtree can't contain matches
            }
        }
        float d;
        if (pca_dims_ > 0) {
            // PCA-space L2sq to root centroid.
            const float* rcc = &pca_root_centroids_[c * pca_dims_];
            d = 0.0f;
            for (uint32_t kk = 0; kk < pca_dims_; ++kk) {
                const float diff = query_pca[kk] - rcc[kk];
                d += diff * diff;
            }
        } else {
            d = simd::dist_f16(coder_->metric(), query_fp16.data(),
                               root_children_[c].centroid, manifest_.dim);
        }
        root_dists.emplace_back(d, c);
    }
    std::sort(root_dists.begin(), root_dists.end());

    auto& frontier = scratch.frontier;
    auto& root_idx = scratch.root_idx;  // root-child index per frontier entry
    frontier.clear();
    root_idx.clear();
    // When filter-directed, probe ALL summary-matching children (no
    // n_probe_l0 cap, no gap pruning). Otherwise: fraction routing cuts by
    // cumulative subtree extent; legacy path takes top-n_probe_l0 with gap
    // pruning.
    const uint32_t effective_probe = filter_directed
        ? static_cast<uint32_t>(root_dists.size())
        : n_probe_l0;
    frontier.reserve(effective_probe);
    if (fraction_routing && !filter_directed) {
        // Leaf-coverage cut: walk children nearest-first, keep selecting
        // until their cumulative subtree extent (pages) reaches
        // probe_frac of the scored total. Pages ≈ vectors at fixed
        // bytes/vector, so the fraction measures actual scan budget
        // regardless of how unbalanced the children are — and no gap
        // pruning: an early gap break would void the coverage contract.
        uint64_t total_pages = 0;
        for (const auto& rd : root_dists)
            total_pages += root_children_[rd.second].pages;
        const uint64_t budget = static_cast<uint64_t>(
            probe_frac * static_cast<double>(total_pages));
        uint64_t cum = 0;
        for (size_t i = 0; i < root_dists.size(); ++i) {
            const uint32_t c = root_dists[i].second;
            const auto& rc = root_children_[c];
            frontier.push_back({root_dists[i].first, rc.page, rc.pages,
                                rc.is_leaf, rc.centroid});
            root_idx.push_back(c);
            cum += rc.pages;
            if (cum >= budget) break;  // >=1 child always selected
        }
    } else {
        for (uint32_t i = 0; i < effective_probe && i < root_dists.size(); ++i) {
            if (!filter_directed && gap > 0 && i > 0 &&
                root_dists[i].first > root_dists[i - 1].first * gap) break;
            const uint32_t c = root_dists[i].second;
            const auto& rc = root_children_[c];
            frontier.push_back({root_dists[i].first, rc.page, rc.pages,
                                rc.is_leaf, rc.centroid});
            root_idx.push_back(c);
        }
    }

    // --- Descend through internal levels ---
    // depth=1 → no descent (frontier is already leaves).
    // depth=2 → one expansion (root children → leaves).
    // depth=3 → two expansions (root → L1 → L2, then L2 → leaves).
    const bool pca_depth2 = (pca_dims_ > 0 && manifest_.depth == 2);
    for (uint16_t level = 1; level < manifest_.depth; ++level) {
        auto& next_frontier = scratch.next_frontier;
        auto& next_root_idx = scratch.next_root_idx;  // only depth=2
        next_frontier.clear();
        next_root_idx.clear();
        // Capacity hint only — clamp the multiply: probe-all callers pass
        // n_probe_ln = UINT32_MAX, and frontier.size() × UINT32_MAX
        // overflows into a multi-TB reserve (bad_alloc). The real per-node
        // clamp lives at the expansion loop (min against each node's child
        // count).
        next_frontier.reserve(std::min<uint64_t>(
            frontier.size() * std::max(1u, n_probe_ln_cfg), 1u << 20));

        for (uint32_t fi = 0; fi < frontier.size(); ++fi) {
            const auto& e = frontier[fi];
            if (e.is_leaf) {
                // Already a leaf — carry through.
                next_frontier.push_back(e);
                if (pca_depth2) next_root_idx.push_back(root_idx[fi]);
                continue;
            }
            if (e.page == kInvalidPage) continue;  // empty internal node
            // At the root→L1 step of a depth=2 PCA tree, use PCA leaf
            // centroids; elsewhere route by FP16 inline centroids.
            const bool use_pca_leaves = pca_depth2 && (level == 1);
            const uint32_t rc_for_pca = use_pca_leaves ? root_idx[fi] : 0;
            const size_t before = next_frontier.size();
            expand_probe_(e, use_pca_leaves, rc_for_pca, gap, n_probe_ln_cfg,
                          config, scratch.pred_col_indices,
                          scratch.geo_lng_col_indices, scratch,
                          next_frontier, scratch.route_node_bytes);
            // Propagate root-child index to children (only needed for the
            // depth=2 PCA path, which terminates at this level).
            if (pca_depth2 && level == 1) {
                for (size_t j = before; j < next_frontier.size(); ++j)
                    next_root_idx.push_back(root_idx[fi]);
            }
        }
        if (next_frontier.empty()) break;
        frontier = std::move(next_frontier);
        root_idx = std::move(next_root_idx);
    }

    // --- Collect leaf candidates from the final frontier ---
    for (const auto& e : frontier) {
        if (e.is_leaf && e.page != kInvalidPage) {
            candidates.push_back({e.page, e.pages, e.dist, e.centroid});
        }
    }
    if (candidates.empty()) {
        return RouteStatus::Empty;
    }

    // --- Phase D: summary-based leaf pruning ---
    // Before scanning, drop any candidate leaf whose summary rules out all
    // matches for the predicates (numeric range miss or bloom negative).
    // This skips entire leaves, saving the FastScan cost at low
    // selectivity. Pins are taken through scratch.pins (released by the
    // caller's guard).
    if (has_predicates) {
        auto& pruned = scratch.pruned_candidates;
        pruned.clear();
        pruned.reserve(candidates.size());
        for (const auto& cand : candidates) {
            if (cand.page == kInvalidPage) continue;
            LeafExtentCache::Handle h;
            const uint8_t* leaf_ptr =
                pin_leaf_(cand.page, static_cast<uint32_t>(cand.pages), h);
            if (h.entry) scratch.pins.push_back(h);
            const uint8_t* summary = leaf_ptr + leaf_filter_offset();
            if (summary_may_match(
                    summary, manifest_.summary_size, manifest_.schema,
                    config.predicates, scratch.pred_col_indices,
                    scratch.geo_lng_col_indices)) {
                pruned.push_back(cand);
            }
        }
        candidates = std::move(pruned);
        if (candidates.empty()) {
            return RouteStatus::Empty;
        }
    }

    return RouteStatus::Ok;
}

std::vector<Candidate> IVFTreeIndex::search(const float* query, uint32_t k,
                                             const SearchConfig& config,
    std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs,
    const std::vector<uint32_t>* sweep_Ws,
    std::vector<std::vector<Candidate>>* sweep_out,
    std::vector<PageId>* visited_leaf_pages) const {
    // Per-query observability (SearchStats, config.hpp): wall/leaves/bytes
    // recorded on EVERY exit path via the guard destructor. Fields are
    // assigned (not accumulated) once the candidate set is final below.
    struct QueryGuard {
        IVFTreeIndex const* idx;
        std::chrono::steady_clock::time_point t0;
        uint64_t leaves = 0, bytes = 0, reranked = 0;
        // Routing observability: descent wall (query start → candidate set
        // final) and internal-node bytes touched during the descent.
        uint64_t routing_ns = 0, node_bytes = 0;
        std::chrono::steady_clock::time_point t_route_end{};
        bool route_done = false;
        // Set once the search scratch is acquired; released on every exit.
        std::vector<LeafExtentCache::Handle>* pins = nullptr;
        ~QueryGuard() {
            if (pins) idx->release_leaf_pins_(*pins);
            const uint64_t r_ns =
                route_done
                    ? static_cast<uint64_t>(
                          std::chrono::duration<double>(t_route_end - t0)
                              .count() *
                          1e9)
                    : 0;
            idx->search_stats_.on_query(
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count(),
                leaves, bytes, reranked, r_ns, node_bytes);
        }
    } qguard{this, std::chrono::steady_clock::now()};

    // Thread-local arena: buffers keep their capacity across calls on this
    // thread (see SearchScratch above). CLI std::async workers and test
    // threads each get their own instance.
    static thread_local SearchScratch scratch;
    qguard.pins = &scratch.pins;
    const bool sweep = sweep_Ws != nullptr && sweep_out != nullptr
        && payload_locs == nullptr && !sweep_Ws->empty();
    if (sweep) {
        sweep_out->clear();
        sweep_out->resize(sweep_Ws->size());
    }
    // Per-query scan context (quantized query / LUTs / scalar transform).
    // Local families re-bind it per leaf inside the scan loop.
    auto scan_setup = coder_->scan_setup(query);

    // --- Route: shared descent (route_query_, also used by search_batch) ---
    const MetricKind metric = coder_->metric();
    {
        // Plane stage-1. Predicates compose by summary-pruning the
        // plane-selected candidates (same conservative filter the
        // descent applies). Feedback probing is descent-based and not
        // meaningful under plane routing.
        const bool plane_active = plane_ != nullptr && config.use_plane;
        RouteStatus rstatus;
        if (plane_active) {
            scratch.route_feedback = false;
            scratch.selectivity = 1.0f;
            scratch.route_gap = 0.0f;
            scratch.root_dists.clear();
            rstatus = RouteStatus::Ok;
            // Plane routing normally skips the descent-side selectivity
            // logic — but the extreme-low-selectivity brute-force fallback
            // (<1%, e.g. an Eq on an untracked/identifier-like column)
            // must still trigger: a W-shortlist scan would almost never
            // surface a handful of matching rows out of millions.
            if (!config.predicates.empty() &&
                resolve_pred_columns_(config, scratch.pred_col_indices,
                                      scratch.geo_lng_col_indices)) {
                scratch.selectivity = card_table_.empty()
                    ? 1.0f
                    : card_table_.selectivity_combined(
                          manifest_.schema, config.predicates,
                          scratch.pred_col_indices,
                          scratch.geo_lng_col_indices);
                if (scratch.selectivity > 0.0f &&
                    scratch.selectivity < 0.01f) {
                    rstatus = RouteStatus::FallbackFiltered;
                }
            }
            if (rstatus != RouteStatus::FallbackFiltered)
                plane_route_(query, config, scratch.candidates);
            if (!config.predicates.empty() &&
                rstatus == RouteStatus::Ok &&
                !scratch.candidates.empty()) {
                (void)resolve_pred_columns_(config,
                                            scratch.pred_col_indices,
                                            scratch.geo_lng_col_indices);
                auto& pruned = scratch.pruned_candidates;
                pruned.clear();
                pruned.reserve(scratch.candidates.size());
                for (const auto& cand : scratch.candidates) {
                    if (cand.page == kInvalidPage) continue;
                    LeafExtentCache::Handle h;
                    const uint8_t* leaf_ptr =
                        pin_leaf_(cand.page,
                                  static_cast<uint32_t>(cand.pages), h);
                    if (h.entry) scratch.pins.push_back(h);
                    const uint8_t* summary = leaf_ptr + leaf_filter_offset();
                    if (summary_may_match(
                            summary, manifest_.summary_size,
                            manifest_.schema, config.predicates,
                            scratch.pred_col_indices,
                            scratch.geo_lng_col_indices))
                        pruned.push_back(cand);
                }
                scratch.candidates = std::move(pruned);
            }
            // (FallbackFiltered survives even though candidates is empty —
            // the brute-force path below does its own full-leaf walk.)
            if (scratch.candidates.empty() &&
                rstatus != RouteStatus::FallbackFiltered)
                rstatus = RouteStatus::Empty;
        } else {
            rstatus = route_query_(query, config, scratch, scratch.candidates);
        }
        qguard.t_route_end = std::chrono::steady_clock::now();
        qguard.route_done = true;
        qguard.node_bytes += scratch.route_node_bytes;
        scratch.route_node_bytes = 0;
        if (rstatus == RouteStatus::Empty) return {};
        if (rstatus == RouteStatus::FallbackFiltered) {
            // Brute-force PQ-decode fallback for extreme low selectivity
            // (<1%) — per-query, not coalescible.
            return search_brute_force_filtered(
                query, k, config, scratch.pred_col_indices,
                scratch.geo_lng_col_indices, payload_locs);
        }
    }
    auto& candidates = scratch.candidates;
    const bool has_predicates = !config.predicates.empty();
    const bool feedback_active = scratch.route_feedback;
    const float selectivity = scratch.selectivity;
    const float gap = scratch.route_gap;
    auto& pred_col_indices = scratch.pred_col_indices;
    auto& geo_lng_col_indices = scratch.geo_lng_col_indices;
    auto& root_dists = scratch.root_dists;

    // Routing diagnostics: report the physical first-page of every leaf that
    // will be scanned (post predicate pruning). The harness maps these back
    // to leaf contents via debug_leaf_info()/debug_leaf_row_ids().
    if (visited_leaf_pages) {
        visited_leaf_pages->clear();
        visited_leaf_pages->reserve(candidates.size());
        for (const auto& c : candidates) visited_leaf_pages->push_back(c.page);
    }

    // --- Prefetch leaf extents ---
    // On Linux, posix_fadvise triggers async NVMe prefetch. On macOS it's
    // a no-op (the unified buffer cache handles read-ahead for sequential
    // mmap access). With the leaf cache on we pread each missing extent as
    // one large sequential read anyway — prefetching the whole candidate
    // set upfront would just fill the page cache (charged to our cgroup /
    // polluting a budgeted host) ahead of the copies we are about to make.
    if (!leaf_cache_) {
        for (const auto& c : candidates) {
#ifdef __linux__
            ::posix_fadvise(fd_, static_cast<off_t>(c.page) * kPageSize,
                            static_cast<off_t>(c.pages) * kPageSize,
                            POSIX_FADV_WILLNEED);
#else
            (void)c;  // macOS: no-op
#endif
        }
    }

    // --- Leaf cache fill planning (page-sorted + contiguous runs) ---
    // Leaf extents emitted by the build are physically contiguous (measured
    // cohere-10m: all 2864 leaves form ONE contiguous 4.3GiB run), and probe
    // semantics are subtree-whole — so candidates from one subtree are
    // page-adjacent. Filling in page order lets us (a) merge adjacent
    // extents into single preads (fewer syscalls, near-streaming I/O) via
    // alias-keyed run entries, and (b) issue reads in offset order for the
    // NVMe queue. Fills are NOT done upfront: an all-at-once fill stage
    // phase-separates I/O from compute and, when query threads run similar
    // candidate sets (zipf), phase-locks them into lockstep — measured
    // −14% QPS at f=0.01. Instead the scan loop fills one run AHEAD of the
    // run it scans, keeping cross-thread overlap. Results are order-
    // independent (heap entries carry leaf_slot; ties break on it).
    if (leaf_cache_) {
        auto& ord = scratch.leaf_order;
        ord.clear();
        ord.reserve(candidates.size());
        for (uint32_t ci = 0; ci < candidates.size(); ++ci)
            if (candidates[ci].page != kInvalidPage) ord.push_back(ci);
        std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
            return candidates[a].page < candidates[b].page;
        });
        auto& ptrs = scratch.leaf_ptrs;
        ptrs.clear();
        ptrs.resize(candidates.size(), nullptr);
    }
    // Run-fill helper shared by the scan loop (serial path) and the
    // upfront stage (parallel path). Merges ONLY when every member is
    // currently uncached — a merged fill over partially-resident members
    // would duplicate their bytes and double-claim their keys.
    auto fill_run = [&](size_t i, size_t j,
                        std::vector<PageId>& alias_buf) {
        if (i == j) return;
        const auto& ord = scratch.leaf_order;
        for (size_t k = i; k <= j; ++k) {
            if (leaf_cache_->contains(candidates[ord[k]].page)) return;
        }
        alias_buf.clear();
        for (size_t k = i + 1; k <= j; ++k)
            alias_buf.push_back(candidates[ord[k]].page);
        const PageId start = candidates[ord[i]].page;
        const uint32_t run_pages = static_cast<uint32_t>(
            (candidates[ord[j]].page + candidates[ord[j]].pages) - start);
        LeafExtentCache::Handle h;
        (void)leaf_cache_->pin(
            start, run_pages, h,
            mmap_base_ + static_cast<uint64_t>(start) * kPageSize,
            nullptr, nullptr, alias_buf.data(),
            static_cast<uint32_t>(alias_buf.size()));
        if (h.entry) scratch.pins.push_back(h);
    };
    auto run_end = [&](size_t i) {
        // End position (inclusive) of the contiguous run starting at i,
        // capped at kMaxRunBytes.
        constexpr uint64_t kMaxRunBytes = 2ull << 20;
        const auto& ord = scratch.leaf_order;
        size_t j = i;
        uint64_t run_bytes = candidates[ord[i]].pages * kPageSize;
        while (j + 1 < ord.size() &&
               candidates[ord[j]].page + candidates[ord[j]].pages ==
                   candidates[ord[j + 1]].page &&
               run_bytes + candidates[ord[j + 1]].pages * kPageSize <=
                   kMaxRunBytes) {
            ++j;
            run_bytes += candidates[ord[j]].pages * kPageSize;
        }
        return j;
    };

    // --- Scan leaves ---
    // Adaptive W driven by predicate selectivity (computed above, before routing).
    uint32_t W = std::max(config.fastscan_W > 0 ? config.fastscan_W : 1000u, k);
    if (sweep) {
        // One scan at W_max serves every W in the sweep (prefix cut below).
        W = std::max(W, *std::max_element(sweep_Ws->begin(), sweep_Ws->end()));
    }
    if (has_predicates) {
        // Adaptive W: the heap collects top-W by PQ distance WITHOUT predicate
        // filtering (deferred). We need W wide enough that the true matching
        // neighbors make it into the heap despite PQ noise. The non-filtered
        // W=300 already captures 99% of true neighbors. Filtering only removes
        // candidates that happen to not match the predicate — it doesn't change
        // which candidates are nearest to the query. So a modest overscan (2x)
        // suffices: W = max(300, k * 2 / selectivity) ensures enough survivors.
        // Summary pruning already skips non-matching leaves, so wider W costs
        // more heap operations (compute), not more I/O.
        constexpr float kOverscan = 2.0f;
        const uint32_t adaptive_w = selectivity > 0.001f
            ? static_cast<uint32_t>(static_cast<float>(k) / selectivity * kOverscan)
            : k * 200u;  // extreme low selectivity — brute-force triggers before this
        W = std::max(W, adaptive_w);
        W = std::max(W, k * 10u);  // floor
    }

    // Max-heap of (pq_dist, row_id, leaf_ptr, local_idx). The leaf_ptr /
    // local_idx are carried so the rerank step can decode each candidate's
    // PQ code back to FP32 without a row_id -> code lookup.
    auto& sheap = scratch.heap;
    // Sentinel prefill: the bounded set must be the W smallest by
    // heap_entry_less regardless of scan order (see heap_init).
    heap_init(sheap, W);

    if (feedback_active) {
        // --- Feedback probing: incremental subtree-block scan loop ---
        // Walk root children in routing (PCA/FP16 distance) order. For each
        // child: expand its leaves, prefetch the contiguous extent, scan
        // into the shared bounded heap, evaluate the stop rule on SCAN
        // FEEDBACK (top-k composition / kth-key improvement). `candidates`
        // is rebuilt in probe order so downstream leaf_slot addressing is
        // unchanged. Serial per-query (the CLI parallelizes across
        // queries); scan-feedback semantics require sequential blocks.
        auto& fb_candidates = scratch.candidates;
        fb_candidates.clear();
        uint64_t total_pages_fb = 0;
        for (const auto& rd : root_dists)
            total_pages_fb += root_children_[rd.second].pages;
        const uint32_t k_eff = std::min(k, W);
        const bool per_leaf = coder_->per_leaf_setup();
        static std::atomic<uint32_t> fb_query_seq{0};
        const uint32_t qid = fb_query_seq.fetch_add(1) + 1;
        const bool pca_leaves_fb = (pca_dims_ > 0 && manifest_.depth == 2);
        uint64_t cum_pages = 0;
        uint32_t n_blocks = 0, stall_run = 0, kth_run = 0;
        uint32_t prev_kth = UINT32_MAX;
        std::vector<ProbeEntry> fb_blk;
        std::vector<HeapEntry> fb_ord;
        for (const auto& rd : root_dists) {
            const uint32_t c = rd.second;
            const auto& rc = root_children_[c];
            fb_blk.clear();
            if (rc.is_leaf) {
                fb_blk.push_back({rd.first, rc.page, rc.pages, rc.is_leaf,
                                  rc.centroid});
            } else {
                expand_probe_({rd.first, rc.page, rc.pages, rc.is_leaf,
                               rc.centroid},
                              pca_leaves_fb, c, gap, UINT32_MAX, config,
                              scratch.pred_col_indices,
                              scratch.geo_lng_col_indices, scratch,
                              fb_blk, qguard.node_bytes);
            }
            const uint32_t slot0 =
                static_cast<uint32_t>(fb_candidates.size());
            for (const auto& e : fb_blk) {
                if (e.is_leaf && e.page != kInvalidPage)
                    fb_candidates.push_back({e.page, e.pages, e.dist,
                                             e.centroid});
            }
            const uint32_t slot1 =
                static_cast<uint32_t>(fb_candidates.size());
            if (!qguard.route_done) {
                qguard.t_route_end = std::chrono::steady_clock::now();
                qguard.route_done = true;
            }
            // Prefetch this block's extents before scanning it (cold-
            // storage semantics: one fadvise per subtree, like the batch
            // path does for its whole selection).
            for (uint32_t j = slot0; j < slot1; ++j) {
#ifdef __linux__
                ::posix_fadvise(fd_,
                                static_cast<off_t>(fb_candidates[j].page) *
                                    kPageSize,
                                static_cast<off_t>(fb_candidates[j].pages) *
                                    kPageSize,
                                POSIX_FADV_WILLNEED);
#endif
            }
            for (uint32_t j = slot0; j < slot1; ++j) {
                LeafExtentCache::Handle h;
                const uint8_t* leaf_ptr =
                    pin_leaf_(fb_candidates[j].page,
                              static_cast<uint32_t>(fb_candidates[j].pages), h);
                if (h.entry) scratch.pins.push_back(h);
                if (per_leaf) coder_->bind_leaf(*scan_setup, leaf_ptr);
                RawScanHeap heap{&sheap, W, j};
                coder_->scan_leaf(*scan_setup, leaf_ptr, heap);
            }
            cum_pages += rc.pages;
            ++n_blocks;

            // Feedback statistics over the bounded heap: kth-best scan key
            // and how many current top-k entries came from this block.
            // (Strip sentinel slots first — they'd pollute kth/gained.)
            heap_compact(sheap);
            uint32_t kth = UINT32_MAX, gained = 0;
            if (sheap.size() >= k_eff) {
                fb_ord.assign(sheap.begin(), sheap.end());
                std::nth_element(fb_ord.begin(), fb_ord.begin() + k_eff - 1,
                                 fb_ord.end(), heap_entry_less);
                kth = fb_ord[k_eff - 1].pq_dist;
                for (uint32_t j = 0; j < k_eff; ++j)
                    if (fb_ord[j].leaf_slot >= slot0 &&
                        fb_ord[j].leaf_slot < slot1)
                        ++gained;
            }

            if (config.trace) {
                std::string ids;
                for (uint32_t j = 0; j < k_eff && j < fb_ord.size(); ++j) {
                    const auto& e = fb_ord[j];
                    const LeafCandidate& lc = fb_candidates[e.leaf_slot];
                    // The scan pins above are still held — a fresh pin would
                    // hit, but a plain mmap read is cheaper and equivalent
                    // for this diagnostic-only path.
                    const uint8_t* leaf_ptr = mmap_base_ +
                        static_cast<uint64_t>(lc.page) * kPageSize;
                    const auto* lh =
                        reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
                    const RowId* rids = reinterpret_cast<const RowId*>(
                        leaf_ptr + coder_->geometry(lh).rowids_offset);
                    if (j) ids += ',';
                    ids += std::to_string(rids[e.local_idx]);
                }
                config.trace->record_fmt(
                    "FB q={} b={} child={} pages={} cum={} tot={} kth={} "
                    "g={} topk={}",
                    qid, n_blocks, c, rc.pages, cum_pages, total_pages_fb,
                    kth, gained, ids);
            }

            // --- Stop rules ---
            bool stop = false;
            switch (config.feedback.mode) {
            case FeedbackProbe::Mode::Fixed:
                if (cum_pages >= static_cast<uint64_t>(
                        config.feedback.fixed_fraction *
                        static_cast<double>(total_pages_fb))) {
                    stop = true;
                }
                break;
            case FeedbackProbe::Mode::Stall:
                if (n_blocks >= config.feedback.min_blocks) {
                    if (gained == 0) {
                        if (++stall_run >= config.feedback.m) stop = true;
                    } else {
                        stall_run = 0;
                    }
                }
                break;
            case FeedbackProbe::Mode::Kth:
                if (n_blocks >= config.feedback.min_blocks) {
                    if (kth >= prev_kth) {
                        if (++kth_run >= config.feedback.m) stop = true;
                    } else {
                        kth_run = 0;
                    }
                }
                break;
            default:
                break;
            }
            prev_kth = kth;
            if (stop) break;
        }
    } else {
    // --- Scan leaves via the family coder ---
    // Within-query leaf-parallel scan. When search_threads <= 1 (default) or
    // fewer than 2 candidate leaves, run the original serial loop into the
    // shared heap. When enabled, shard the candidates across T workers, each
    // scanning into its own private heap with its own per-leaf setup state
    // (the scalar a_d buffers and local LUTs are PER-LEAF mutable state —
    // sharing them across workers was a data race), then merge the T heaps
    // into `heap` keeping the top-W by pq_dist.
    {
        const bool per_leaf = coder_->per_leaf_setup();
        const bool parallel_scan = config.search_threads > 1
            && candidates.size() >= 2;
        // With the leaf cache, the fill stage pre-resolved every pointer
        // (page-sorted, run-merged) into scratch.leaf_ptrs and scans follow
        // the same sorted order (I/O already done; order only affects heap
        // slot addressing, which is fixed per candidate either way).
        const auto& ord = scratch.leaf_order;
        const bool use_ord = leaf_cache_ != nullptr;
        const uint32_t n_scan = use_ord ? static_cast<uint32_t>(ord.size())
                                        : static_cast<uint32_t>(candidates.size());
        if (!parallel_scan) {
            if (use_ord) {
                // Run-pipelined cache path: fill the NEXT contiguous run
                // (merged pread) while scanning the current one; per-leaf
                // pins inside the loop hit the already-filled entries.
                // Restores cross-thread I/O/compute overlap (no upfront
                // fill stage — see fill planning comment above).
                std::vector<PageId> alias_buf;
                size_t next = 0;
                size_t filled = SIZE_MAX;  // run [0, run_end] already filled
                size_t pos = 0;
                fill_run(0, run_end(0), alias_buf);
                filled = 0;
                while (pos < n_scan) {
                    const size_t rend = run_end(pos);
                    if (rend != filled) {
                        // pos starts a new run (previous finished): fill
                        // THIS run's successors ahead is handled below; we
                        // fill the run we are about to scan only if it was
                        // not already prefilled as a predecessor's "next".
                        fill_run(pos, rend, alias_buf);
                        filled = rend;
                    }
                    for (size_t q = pos; q <= rend; ++q) {
                        const uint32_t ci = ord[q];
                        LeafExtentCache::Handle h;
                        const uint8_t* leaf_ptr =
                            pin_leaf_(candidates[ci].page,
                                      static_cast<uint32_t>(
                                          candidates[ci].pages),
                                      h);
                        if (h.entry) scratch.pins.push_back(h);
                        scratch.leaf_ptrs[ci] = leaf_ptr;
                        if (per_leaf)
                            coder_->bind_leaf(*scan_setup, leaf_ptr);
                        RawScanHeap heap{&sheap, W, ci};
                        coder_->scan_leaf(*scan_setup, leaf_ptr, heap);
                    }
                    // Prefill the next run so its read overlaps this loop's
                    // remaining compute and other threads' scans.
                    if (rend + 1 < n_scan) {
                        const size_t nrend = run_end(rend + 1);
                        fill_run(rend + 1, nrend, alias_buf);
                        filled = nrend;
                    }
                    pos = rend + 1;
                    (void)next;
                }
            } else {
                for (uint32_t ci = 0; ci < candidates.size(); ++ci) {
                    if (candidates[ci].page == kInvalidPage) continue;
                    LeafExtentCache::Handle h;
                    const uint8_t* leaf_ptr =
                        pin_leaf_(candidates[ci].page,
                                  static_cast<uint32_t>(candidates[ci].pages),
                                  h);
                    if (h.entry) scratch.pins.push_back(h);
                    if (per_leaf) coder_->bind_leaf(*scan_setup, leaf_ptr);
                    RawScanHeap heap{&sheap, W, ci};
                    coder_->scan_leaf(*scan_setup, leaf_ptr, heap);
                }
            }
        } else {
            const uint32_t T = std::min(config.search_threads,
                                        static_cast<uint32_t>(candidates.size()));
            auto& th = scratch.worker_heaps;
            if (th.size() < T) th.resize(T);
            for (auto& my : th) my.clear();
            // No-cache parallel scan: leaf_ptrs is only pre-sized on the
            // cache path (fill stage above); size it here or the worker
            // write below is out of bounds (crashed with search_threads>1
            // and cache off — caught by scripts/join_sim's probe pass).
            if (!use_ord) scratch.leaf_ptrs.assign(candidates.size(), nullptr);
            // Workers must fill the CALLING thread's leaf_ptrs — `scratch`
            // is thread_local, so referencing it inside the async worker
            // resolves to the worker's own (empty) instance.
            auto& out_ptrs = scratch.leaf_ptrs;
            // Per-worker setups: local families own mutable per-leaf state
            // (a_d transform, LUTs) in the setup, so each worker builds its
            // own from the same query (identical arithmetic). Global
            // families share one read-only setup.
            std::vector<std::future<void>> futs;
            futs.reserve(T);
            auto& wpins = scratch.worker_pins;
            if (wpins.size() < T) wpins.resize(T);
            for (auto& v : wpins) v.clear();
            const uint32_t per = (n_scan + T - 1) / T;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t start = t * per;
                const uint32_t end = std::min(start + per, n_scan);
                if (start >= end) break;
                futs.push_back(std::async(std::launch::async,
                    [&](uint32_t s, uint32_t e, uint32_t ti) {
                        auto& my = th[ti];
                        heap_init(my, W);
                        std::unique_ptr<ScanSetup> own_setup;
                        ScanSetup* use = scan_setup.get();
                        if (per_leaf) {
                            own_setup = coder_->scan_setup(query);
                            use = own_setup.get();
                        }
                        for (uint32_t pos = s; pos < e; ++pos) {
                            const uint32_t c = use_ord ? ord[pos] : pos;
                            if (candidates[c].page == kInvalidPage) continue;
                            // Workers pin per leaf (in page-sorted order for
                            // the cache path — no prefill stage; see fill
                            // planning comment).
                            LeafExtentCache::Handle h;
                            const uint8_t* leaf_ptr =
                                pin_leaf_(candidates[c].page,
                                          static_cast<uint32_t>(
                                              candidates[c].pages),
                                          h);
                            if (h.entry) wpins[ti].push_back(h);
                            out_ptrs[c] = leaf_ptr;
                            if (per_leaf) coder_->bind_leaf(*use, leaf_ptr);
                            RawScanHeap heap{&my, W, c};
                            coder_->scan_leaf(*use, leaf_ptr, heap);
                        }
                    }, start, end, t));
            }
            for (auto& f : futs) f.get();
            // Adopt worker pins (refcounts held) so they live until guard exit.
            for (auto& v : wpins)
                scratch.pins.insert(scratch.pins.end(), v.begin(), v.end());

            // Deterministic merge: collect every entry from all per-thread
            // heaps, then keep the top-W by (pq_dist asc, tie: leaf_slot,
            // local_idx) — bit-stable regardless of how candidates were
            // sharded. Downstream rerank iterates `heap` linearly, so a
            // flat sorted vector is exactly what it wants.
            size_t total = 0;
            for (const auto& my : th) total += my.size();
            sheap.clear();
            sheap.reserve(total);
            for (auto& my : th) {
                sheap.insert(sheap.end(),
                            std::make_move_iterator(my.begin()),
                            std::make_move_iterator(my.end()));
            }
            if (sheap.size() > W) {
                std::sort(sheap.begin(), sheap.end(),
                          [](const HeapEntry& a, const HeapEntry& b) {
                              if (a.pq_dist != b.pq_dist)
                                  return a.pq_dist < b.pq_dist;
                              if (a.leaf_slot != b.leaf_slot)
                                  return a.leaf_slot < b.leaf_slot;
                              return a.local_idx < b.local_idx;
                          });
                sheap.resize(W);
            }
        }
    }
    }  // end feedback_active else-branch

    // Drop unfilled sentinel slots before materialization/rerank.
    sheap.erase(std::remove_if(sheap.begin(), sheap.end(),
                               heap_entry_is_sentinel),
                sheap.end());

    // Final I/O accounting for this query (both probe paths converge here;
    // `candidates` is final). bytes_touched = leaf pages × page size —
    // the bytes-touched currency. `reranked` = decoded shortlist size.
    qguard.leaves = candidates.size();
    qguard.bytes = 0;
    for (const auto& c : candidates)
        qguard.bytes += static_cast<uint64_t>(c.pages) * kPageSize;
    qguard.reranked = sheap.size();

    // Plain path (no payload locs, no W sweep): the shared per-query
    // finalize pipeline — also used verbatim by search_batch().
    if (payload_locs == nullptr && !sweep) {
        finalize_query_(query, k, config, scratch, candidates,
                        scratch.leaf_ptrs, sheap, *scan_setup,
                        scratch.results);
        return scratch.results;
    }

    // --- Materialize full entries for the extraction/rerank paths ---
    // Resolve row_id + leaf_ptr from (leaf_slot, local_idx) for the <=W
    // survivors. Layout mirrors the scan-side row_ids placement.
    {
        auto& hf = scratch.heap_full;
        hf.clear();
        hf.reserve(sheap.size());
        // Pin each candidate leaf ONCE (heap entries reference the same
        // <=n_leaves extents — per-entry pinning would double-count cache
        // accesses and thrash the LRU). With the cache on, the fill stage
        // already resolved every pointer (and holds the pins) — reuse.
        auto& ptrs = scratch.leaf_ptrs;
        if (!leaf_cache_) {
            ptrs.clear();
            ptrs.resize(candidates.size(), nullptr);
            for (uint32_t ci = 0; ci < candidates.size(); ++ci) {
                if (candidates[ci].page == kInvalidPage) continue;
                LeafExtentCache::Handle h;
                ptrs[ci] = pin_leaf_(candidates[ci].page,
                                     static_cast<uint32_t>(candidates[ci].pages),
                                     h);
                if (h.entry) scratch.pins.push_back(h);
            }
        }
        for (const auto& e : sheap) {
            const uint32_t ci = e.leaf_slot;
            const LeafCandidate& c = candidates[ci];
            const uint8_t* leaf_ptr = ptrs[ci];
            const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
            const RowId* rids = reinterpret_cast<const RowId*>(
                leaf_ptr + coder_->geometry(lh).rowids_offset);
            hf.push_back({e.pq_dist, rids[e.local_idx], leaf_ptr,
                          mmap_base_ + static_cast<uint64_t>(c.page) * kPageSize,
                          e.local_idx});
        }
    }
    auto& heap = scratch.heap_full;

    // --- Extract top-k from the heap ---
    auto& results = scratch.results;
    results.clear();
    results.reserve(heap.size());
    auto& sweep_entries = scratch.sweep_entries;
    sweep_entries.clear();

    // Phase E: when payload locations are requested, carry (leaf_ptr, local_idx)
    // alongside each result through dedup/sort/truncate. The heap entries'
    // payload locations are always MMAP pointers (stable for the index's
    // life — cache pins would dangle after search() returns).
    const bool want_locs = (payload_locs != nullptr);
    auto& results_loc = scratch.results_loc;
    results_loc.clear();
    if (want_locs) results_loc.reserve(heap.size());

    // --- Phase D: filter the heap survivors by predicate ---
    // Predicates are evaluated AFTER heap selection, not during the scan.
    // This is the "fastest code is code that doesn't run" principle: we only
    // evaluate predicates on the W heap survivors (not on every scanned candidate).
    // The heap is large enough (W = k/selectivity * overscan) to contain enough
    // matching candidates even at low selectivity.
    //
    // Optimization: group heap entries by leaf_ptr so we parse each leaf's
    // filter columns only once (not once per entry). Multiple heap entries
    // from the same leaf share the same column layout.
    if (has_predicates && !heap.empty()) {
        // Sort by leaf_ptr so entries from the same leaf are contiguous.
        std::sort(heap.begin(), heap.end(),
                  [](const HeapEntryFull& a, const HeapEntryFull& b) {
                      return a.leaf_ptr < b.leaf_ptr;
                  });
        auto& filtered = scratch.filtered_heap;
        filtered.clear();
        filtered.reserve(heap.size());
        const uint8_t* cur_leaf = nullptr;
        auto& cols = scratch.filter_cols;
        cols.clear();
        for (const auto& entry : heap) {
            if (entry.leaf_ptr != cur_leaf) {
                cur_leaf = entry.leaf_ptr;
                const auto* lh = reinterpret_cast<const TreeLeafHeader*>(cur_leaf);
                auto layout = LeafFilterLayout::from_geometry(
                    cur_leaf, coder_->geometry(lh));
                cols = parse_filter_columns(layout.filter_base, lh->count,
                                             manifest_.schema);
            }
            if (eval_all_predicates(cols, manifest_.schema, entry.local_idx,
                                     config.predicates, pred_col_indices, geo_lng_col_indices))
                filtered.push_back(entry);
        }
        heap = std::move(filtered);
    }

    if (config.rerank && !heap.empty()) {
        // Rerank: refine each of the W candidates' distances through the
        // family coder (decode / LUT). Caller-provided exact rerank against
        // the original f32 corpus takes PRECEDENCE over the coder's own
        // rerank. Per-query serial (W is small; the CLI parallelizes
        // across queries).
        for (const auto& entry : heap) {
            float exact_dist;
            if (config.exact_rerank_base) {
                // Exact rerank against the caller's original vectors: no
                // decode, no extract, no quantization ranking error, no IP
                // bias (the true vector has no reconstruction shrinkage).
                const float* v = config.exact_rerank_base +
                    static_cast<size_t>(entry.row_id) * manifest_.dim;
                exact_dist =
                    (metric == MetricKind::InnerProduct)
                        ? -simd::dot_f32(query, v, manifest_.dim)
                        : simd::l2sq_f32(query, v, manifest_.dim);
            } else {
                exact_dist = coder_->rerank(query, *scan_setup,
                                            entry.leaf_ptr, entry.local_idx,
                                            nullptr);
            }
            if (sweep) {
                sweep_entries.push_back({entry.pq_dist, entry.row_id,
                                          exact_dist});
            }
            if (want_locs) {
                results_loc.push_back({{entry.row_id, exact_dist},
                                        entry.mmap_ptr, entry.local_idx});
            } else {
                results.push_back({entry.row_id, exact_dist});
            }
        }
    } else {
        // No rerank: use the raw PQ-approximate uint32 distances.
        for (const auto& entry : heap) {
            if (sweep) {
                sweep_entries.push_back(
                    {entry.pq_dist, entry.row_id,
                     static_cast<float>(entry.pq_dist)});
            }
            if (want_locs) {
                results_loc.push_back({{entry.row_id,
                                         static_cast<float>(entry.pq_dist)},
                                        entry.mmap_ptr, entry.local_idx});
            } else {
                results.push_back({entry.row_id,
                                   static_cast<float>(entry.pq_dist)});
            }
        }
    }

    // --- Sweep: per-W prefix cuts over the (filtered, reranked) heap ---
    // recall@k for shortlist W is exactly "top-k by exact distance among the
    // top-W entries by scan distance", so one scan at W_max serves every
    // W <= W_max: take the first W entries in scan order (pq_dist asc, ties
    // by row_id for determinism), then apply the SAME dedup-by-row_id (keep
    // min dist) + top-k-by-exact-dist selection as the normal path below.
    //
    // CRITICAL ordering: prefix-cut FIRST, then dedup — replicas of a row can
    // straddle the W boundary, and a standalone search with that W only sees
    // the in-prefix replicas. Predicates already filtered the heap above;
    // filtering doesn't reorder, so it commutes with the prefix cut.
    if (sweep) {
        const auto& entries = scratch.sweep_entries;
        auto& order = scratch.sweep_order;
        order.resize(entries.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) {
                      if (entries[a].pq_dist != entries[b].pq_dist)
                          return entries[a].pq_dist < entries[b].pq_dist;
                      return entries[a].row_id < entries[b].row_id;
                  });
        // Ascending W so each cut is a prefix of the previous one.
        auto& plan = scratch.sweep_plan;
        plan.clear();
        for (uint32_t i = 0; i < sweep_Ws->size(); ++i)
            plan.emplace_back((*sweep_Ws)[i], i);
        std::sort(plan.begin(), plan.end());
        for (const auto& step : plan) {
            auto& work = scratch.sweep_work;
            work.clear();
            const size_t cnt = std::min<size_t>(step.first, order.size());
            for (size_t j = 0; j < cnt; ++j) {
                const auto& e = entries[order[j]];
                work.push_back({e.row_id, e.dist});
            }
            std::sort(work.begin(), work.end(),
                      [](const Candidate& a, const Candidate& b) {
                          if (a.row_id != b.row_id) return a.row_id < b.row_id;
                          return a.dist < b.dist;
                      });
            auto wlast = std::unique(work.begin(), work.end(),
                                     [](const Candidate& a, const Candidate& b) {
                                         return a.row_id == b.row_id;
                                     });
            work.erase(wlast, work.end());
            if (work.size() > k) {
                std::nth_element(work.begin(), work.begin() + k, work.end(),
                                 [](const Candidate& a, const Candidate& b) {
                                     return a.dist < b.dist;
                                 });
                work.resize(k);
            }
            std::sort(work.begin(), work.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.dist < b.dist;
                      });
            (*sweep_out)[step.second] = work;
        }
    }

    if (!want_locs) {
        // Dedup by row_id (closure may replicate vectors across leaves → same
        // row_id appears with different distances → keep the min). Sort by
        // row_id first so duplicates are adjacent, with min-dist tiebreak.
        std::sort(results.begin(), results.end(),
                  [](const Candidate& a, const Candidate& b) {
                      if (a.row_id != b.row_id) return a.row_id < b.row_id;
                      return a.dist < b.dist;
                  });
        auto last = std::unique(results.begin(), results.end(),
                                [](const Candidate& a, const Candidate& b) {
                                    return a.row_id == b.row_id;
                                });
        results.erase(last, results.end());
        // Adaptive shortlist cut: keep up to W (not just k) and truncate at
        // the first distance gap past k. Clustered queries cut at ~k, noisy
        // queries keep the deep list — the caller's rerank bandwidth follows
        // the returned length. Off (0) or without rerank: plain top-k.
        //
        // Signal: the gap d[w]-d[k-1] against the top-k region's OWN typical
        // gap, g = (d[k-1]-d[0])/(k-1). τ is therefore a dimensionless
        // multiplier of the local score scale and one calibration transfers
        // across metrics (IP distances are negated dots ≈ -1, so scaling by
        // |d[k-1]| as in v1 compressed the signal and forced per-metric τ).
        size_t keep = k;
        if (config.adaptive_w_gap > 0 && config.rerank &&
            results.size() > k) {
            std::sort(results.begin(), results.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.dist < b.dist;
                      });
            const float dk = results[k - 1].dist;
            const float g = (dk - results[0].dist) / static_cast<float>(k - 1);
            const float scale = std::max(g, 1e-12f);
            keep = results.size();
            for (size_t w = k; w < results.size(); ++w) {
                if (results[w].dist - dk >
                    config.adaptive_w_gap * scale) {
                    keep = w;
                    break;
                }
            }
        }
        if (results.size() > keep) {
            std::nth_element(results.begin(), results.begin() + keep,
                             results.end(),
                             [](const Candidate& a, const Candidate& b) {
                                 return a.dist < b.dist;
                             });
            results.resize(keep);
        }
        std::sort(results.begin(), results.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.dist < b.dist;
                  });
        return results;
    }

    // --- Payload-location path: same dedup/sort/truncate, carrying locs ---
    // Dedup by row_id (keep min-dist; the row_id sort makes duplicates adjacent).
    std::sort(results_loc.begin(), results_loc.end(),
              [](const ResultWithLoc& a, const ResultWithLoc& b) {
                  if (a.cand.row_id != b.cand.row_id)
                      return a.cand.row_id < b.cand.row_id;
                  return a.cand.dist < b.cand.dist;
              });
    auto last_loc = std::unique(results_loc.begin(), results_loc.end(),
                                [](const ResultWithLoc& a, const ResultWithLoc& b) {
                                    return a.cand.row_id == b.cand.row_id;
                                });
    results_loc.erase(last_loc, results_loc.end());
    if (results_loc.size() > k) {
        std::nth_element(results_loc.begin(), results_loc.begin() + k,
                         results_loc.end(),
                         [](const ResultWithLoc& a, const ResultWithLoc& b) {
                             return a.cand.dist < b.cand.dist;
                         });
        results_loc.resize(k);
    }
    std::sort(results_loc.begin(), results_loc.end(),
              [](const ResultWithLoc& a, const ResultWithLoc& b) {
                  return a.cand.dist < b.cand.dist;
              });
    results.clear();
    results.reserve(results_loc.size());
    payload_locs->clear();
    payload_locs->reserve(results_loc.size());
    for (auto& rl : results_loc) {
        payload_locs->push_back({rl.leaf_ptr, rl.local_idx});
        results.push_back(rl.cand);
    }
    return results;
}

std::vector<Candidate> IVFTreeIndex::search(const float* query, uint32_t k,
        const SearchConfig& config,
        std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs) const {
    return search(query, k, config, payload_locs, nullptr, nullptr);
}

namespace {

/// Full pread (loops on short reads). Returns false on I/O error.
bool pread_full(int fd, void* buf, size_t n, uint64_t off) {
    auto* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
        const ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
        if (r <= 0) return false;
        p += r;
        off += static_cast<uint64_t>(r);
        n -= static_cast<size_t>(r);
    }
    return true;
}

/// Harvest-pool entry for the sweep path: leaf buffers die at the next
/// leaf (pread) or get unpinned (hot-set), so everything finalize needs
/// from the leaf — row_id, coder-rerank distance, predicate verdict — is
/// resolved at harvest time, right after the (query, leaf) scan.
struct PoolEntry {
    uint32_t pq_dist;
    uint32_t leaf_slot;
    uint32_t local_idx;
    int64_t row_id;
    float dist;        // exact distance when reranked, else unused
    bool reranked;     // coder rerank computed at harvest time
    bool pred_ok;      // harvest-time predicate verdict (true when no
                       // predicates)
};

/// Total order on pool entries — IDENTICAL to the scan heap's
/// (pq_dist, leaf_slot, local_idx) order. The pool is a W-bounded max-
/// heap under this order, and its input stream contains every entry of
/// the partial's FINAL heap (an entry in the final heap was never
/// evicted, so it was present at the end of its own leaf's scan, i.e.
/// harvested). Since fewer than W entries beat any final entry among
/// ALL pushes, fewer beat it among the pool's (smaller) input — so the
/// W-best pool provably contains every final entry. The pool size is
/// exactly bounded (≤ W per query per partial): no cap heuristics, no
/// overflow fallback path.
inline bool pool_entry_less(const PoolEntry& a, const PoolEntry& b) {
    if (a.pq_dist != b.pq_dist) return a.pq_dist < b.pq_dist;
    if (a.leaf_slot != b.leaf_slot) return a.leaf_slot < b.leaf_slot;
    return a.local_idx < b.local_idx;
}
inline bool pool_entry_is_sentinel(const PoolEntry& e) {
    return e.pq_dist == 0xFFFFFFFFu;
}
inline void pool_init(std::vector<PoolEntry>& p, uint32_t w) {
    p.clear();
    p.assign(w, {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, -1, 0.0f, false,
                 false});
}
/// Would a scan-heap entry displace the pool's worst slot? (Cheap
/// pq-dist-first compare; called per harvested candidate.)
inline bool pool_should_replace(const std::vector<PoolEntry>& p,
                                uint32_t pq_dist, uint32_t leaf_slot,
                                uint32_t local_idx) {
    const PoolEntry& f = p.front();
    if (pq_dist != f.pq_dist) return pq_dist < f.pq_dist;
    if (leaf_slot != f.leaf_slot) return leaf_slot < f.leaf_slot;
    return local_idx < f.local_idx;
}
/// Sift-down replacement of the pool's worst slot.
inline void pool_replace(std::vector<PoolEntry>& p, PoolEntry pe) {
    p[0] = pe;
    uint32_t pos = 0;
    const uint32_t n = static_cast<uint32_t>(p.size());
    while (true) {
        const uint32_t left = 2 * pos + 1;
        const uint32_t right = 2 * pos + 2;
        uint32_t largest = pos;
        if (left < n && pool_entry_less(p[largest], p[left]))
            largest = left;
        if (right < n && pool_entry_less(p[largest], p[right]))
            largest = right;
        if (largest == pos) break;
        std::swap(p[pos], p[largest]);
        pos = largest;
    }
}

/// Dedup by row_id (closure may replicate vectors across leaves → same
/// row_id appears with different distances → keep the min), adaptive
/// shortlist cut, top-k, distance sort. Shared tail of every per-query
/// finalize variant.
void cut_topk_results_(std::vector<Candidate>& results, uint32_t k,
                       const SearchConfig& config) {
    std::sort(results.begin(), results.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.row_id != b.row_id) return a.row_id < b.row_id;
                  return a.dist < b.dist;
              });
    auto last = std::unique(results.begin(), results.end(),
                            [](const Candidate& a, const Candidate& b) {
                                return a.row_id == b.row_id;
                            });
    results.erase(last, results.end());
    // Adaptive shortlist cut: keep up to W (not just k) and truncate at
    // the first distance gap past k. Clustered queries cut at ~k, noisy
    // queries keep the deep list. Off (0) or without rerank: plain top-k.
    // Signal: the gap d[w]-d[k-1] against the top-k region's own typical
    // gap (dimensionless, metric-transferable; see search()).
    size_t keep = k;
    if (config.adaptive_w_gap > 0 && config.rerank &&
        results.size() > k) {
        std::sort(results.begin(), results.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.dist < b.dist;
                  });
        const float dk = results[k - 1].dist;
        const float g = (dk - results[0].dist) / static_cast<float>(k - 1);
        const float scale = std::max(g, 1e-12f);
        keep = results.size();
        for (size_t w = k; w < results.size(); ++w) {
            if (results[w].dist - dk > config.adaptive_w_gap * scale) {
                keep = w;
                break;
            }
        }
    }
    if (results.size() > keep) {
        std::nth_element(results.begin(), results.begin() + keep,
                         results.end(),
                         [](const Candidate& a, const Candidate& b) {
                             return a.dist < b.dist;
                         });
        results.resize(keep);
    }
    std::sort(results.begin(), results.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.dist < b.dist;
              });
}

/// Pool-based finalize for the pread sweep path: look up each merged-
/// heap survivor's row_id (and, when the coder rerank ran eagerly at
/// harvest, its exact distance) in the query's pool. exact_rerank_base
/// reranks lazily here through the row_id — no leaf pointer needed.
void finalize_pool_query_(const float* query, uint32_t k,
                          MetricKind metric, uint32_t dim,
                          const SearchConfig& config,
                          std::vector<PoolEntry>& pool,
                          const std::vector<HeapEntry>& heap,
                          std::vector<Candidate>& results) {
    std::sort(pool.begin(), pool.end(),
              [](const PoolEntry& a, const PoolEntry& b) {
                  if (a.leaf_slot != b.leaf_slot)
                      return a.leaf_slot < b.leaf_slot;
                  return a.local_idx < b.local_idx;
              });
    results.clear();
    results.reserve(heap.size());
    for (const auto& e : heap) {
        const auto it = std::lower_bound(
            pool.begin(), pool.end(), e,
            [](const PoolEntry& p, const HeapEntry& h) {
                if (p.leaf_slot != h.leaf_slot)
                    return p.leaf_slot < h.leaf_slot;
                return p.local_idx < h.local_idx;
            });
        if (it == pool.end() || it->leaf_slot != e.leaf_slot ||
            it->local_idx != e.local_idx) {
            continue;  // defensive: containment is proven, this is unreachable
        }
        if (!it->pred_ok) continue;  // harvest-time predicate verdict
        float dist;
        if (it->reranked) {
            dist = it->dist;
        } else if (config.rerank && config.exact_rerank_base) {
            const float* v = config.exact_rerank_base +
                             static_cast<size_t>(it->row_id) * dim;
            dist = (metric == MetricKind::InnerProduct)
                       ? -simd::dot_f32(query, v, dim)
                       : simd::l2sq_f32(query, v, dim);
        } else {
            dist = static_cast<float>(e.pq_dist);
        }
        results.push_back({it->row_id, dist});
    }
    cut_topk_results_(results, k, config);
}

}  // namespace

void IVFTreeIndex::finalize_query_(const float* query, uint32_t k,
                                   const SearchConfig& config,
                                   SearchScratch& scratch,
                                   std::vector<LeafCandidate>& candidates,
                                   std::vector<const uint8_t*>& ptrs,
                                   std::vector<HeapEntry>& sheap,
                                   ScanSetup& scan_setup,
                                   std::vector<Candidate>& results) const {    const bool has_predicates = !config.predicates.empty();
    auto& pred_col_indices = scratch.pred_col_indices;
    auto& geo_lng_col_indices = scratch.geo_lng_col_indices;
    const MetricKind metric = coder_->metric();

    // --- Materialize full entries for the extraction/rerank paths ---
    // Resolve row_id + leaf_ptr from (leaf_slot, local_idx) for the <=W
    // survivors. Layout mirrors the scan-side row_ids placement.
    {
        auto& hf = scratch.heap_full;
        hf.clear();
        hf.reserve(sheap.size());
        // Pin each candidate leaf ONCE (heap entries reference the same
        // <=n_leaves extents — per-entry pinning would double-count cache
        // accesses and thrash the LRU). With the cache on, the fill stage
        // (query-major) or the sweep (batch) already resolved every
        // pointer (and holds the pins) — reuse.
        if (!leaf_cache_) {
            ptrs.clear();
            ptrs.resize(candidates.size(), nullptr);
            for (uint32_t ci = 0; ci < candidates.size(); ++ci) {
                if (candidates[ci].page == kInvalidPage) continue;
                LeafExtentCache::Handle h;
                ptrs[ci] = pin_leaf_(candidates[ci].page,
                                     static_cast<uint32_t>(candidates[ci].pages),
                                     h);
                if (h.entry) scratch.pins.push_back(h);
            }
        }
        for (const auto& e : sheap) {
            const uint32_t ci = e.leaf_slot;
            const LeafCandidate& c = candidates[ci];
            const uint8_t* leaf_ptr = ptrs[ci];
            const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
            const RowId* rids = reinterpret_cast<const RowId*>(
                leaf_ptr + coder_->geometry(lh).rowids_offset);
            hf.push_back({e.pq_dist, rids[e.local_idx], leaf_ptr,
                          mmap_base_ + static_cast<uint64_t>(c.page) * kPageSize,
                          e.local_idx});
        }
    }
    auto& heap = scratch.heap_full;

    // --- Extract top-k from the heap ---
    results.clear();
    results.reserve(heap.size());

    // --- Phase D: filter the heap survivors by predicate ---
    // Predicates are evaluated AFTER heap selection, not during the scan.
    // This is the "fastest code is code that doesn't run" principle: we only
    // evaluate predicates on the W heap survivors (not on every scanned candidate).
    // The heap is large enough (W = k/selectivity * overscan) to contain enough
    // matching candidates even at low selectivity.
    //
    // Optimization: group heap entries by leaf_ptr so we parse each leaf's
    // filter columns only once (not once per entry). Multiple heap entries
    // from the same leaf share the same column layout.
    if (has_predicates && !heap.empty()) {
        // Sort by leaf_ptr so entries from the same leaf are contiguous.
        std::sort(heap.begin(), heap.end(),
                  [](const HeapEntryFull& a, const HeapEntryFull& b) {
                      return a.leaf_ptr < b.leaf_ptr;
                  });
        auto& filtered = scratch.filtered_heap;
        filtered.clear();
        filtered.reserve(heap.size());
        const uint8_t* cur_leaf = nullptr;
        auto& cols = scratch.filter_cols;
        cols.clear();
        for (const auto& entry : heap) {
            if (entry.leaf_ptr != cur_leaf) {
                cur_leaf = entry.leaf_ptr;
                const auto* lh = reinterpret_cast<const TreeLeafHeader*>(cur_leaf);
                auto layout = LeafFilterLayout::from_geometry(
                    cur_leaf, coder_->geometry(lh));
                cols = parse_filter_columns(layout.filter_base, lh->count,
                                             manifest_.schema);
            }
            if (eval_all_predicates(cols, manifest_.schema, entry.local_idx,
                                     config.predicates, pred_col_indices, geo_lng_col_indices))
                filtered.push_back(entry);
        }
        heap = std::move(filtered);
    }

    if (config.rerank && !heap.empty()) {
        // Rerank: refine each of the W candidates' distances through the
        // family coder (decode / LUT). Caller-provided exact rerank against
        // the original f32 corpus takes PRECEDENCE over the coder's own
        // rerank. Per-query serial (W is small; batch parallelizes across
        // queries).
        for (const auto& entry : heap) {
            float exact_dist;
            if (config.exact_rerank_base) {
                // Exact rerank against the caller's original vectors: no
                // decode, no extract, no quantization ranking error, no IP
                // bias (the true vector has no reconstruction shrinkage).
                const float* v = config.exact_rerank_base +
                    static_cast<size_t>(entry.row_id) * manifest_.dim;
                exact_dist =
                    (metric == MetricKind::InnerProduct)
                        ? -simd::dot_f32(query, v, manifest_.dim)
                        : simd::l2sq_f32(query, v, manifest_.dim);
            } else {
                exact_dist = coder_->rerank(query, scan_setup,
                                            entry.leaf_ptr, entry.local_idx,
                                            nullptr);
            }
            results.push_back({entry.row_id, exact_dist});
        }
    } else {
        // No rerank: use the raw PQ-approximate uint32 distances.
        for (const auto& entry : heap) {
            results.push_back({entry.row_id,
                               static_cast<float>(entry.pq_dist)});
        }
    }

    // Shared dedup / adaptive cut / top-k tail (see cut_topk_results_).
    cut_topk_results_(results, k, config);
}


void IVFTreeIndex::search_batch(
        const float* queries, uint32_t nq, uint32_t k,
        const SearchConfig& config,
        std::vector<std::vector<Candidate>>& results,
        const std::vector<std::vector<Predicate>>* per_query_predicates,
        const std::vector<float>* per_query_probe_fraction)
        const {
    const auto t0 = std::chrono::steady_clock::now();
    results.clear();
    results.resize(nq);
    if (nq == 0) return;
    if (config.feedback.mode != FeedbackProbe::Mode::Off) {
        throw Error(ErrorCode::InvalidParam,
                    "search_batch: feedback probing adapts the probe set "
                    "per query and cannot be coalesced");
    }
    const uint32_t T = std::max(1u, config.search_threads);
    const bool per_leaf = coder_->per_leaf_setup();
    const uint32_t dim = manifest_.dim;

    // --- Per-query state (route output + sweep accumulation) ---
    // Predicates: per-query overrides when provided (empty vector = no
    // predicates for that query), else the base config's. Probe
    // fraction: per-query override (>0; overrides n_probe too) — a
    // RECALL knob, priced in bytes: deeper queries probe more leaves,
    // which merge into the same unique-leaf sweep (shared leaves read
    // once regardless of which query depth requested them). Everything
    // downstream (routing selectivity/summary pruning, harvest-time
    // evaluation, pool finalize) uses the query's own settings.
    struct QueryState {
        const float* query = nullptr;
        std::vector<LeafCandidate> candidates;
        std::unique_ptr<ScanSetup> setup;   // adopted from a sweep partial
        std::vector<HeapEntry> heap;        // merged, bounded W
        std::vector<PoolEntry> pool;        // harvested row-ids (bounded W)
        const std::vector<Predicate>* predicates = nullptr;
        std::vector<uint32_t> pred_col_indices, geo_lng_col_indices;
        float probe_fraction = 0.0f;       // 0 = base config's
        uint32_t W = 0;
        bool fallback = false;              // brute-force filtered path
        bool plane_done = false;            // candidates from Phase 0 sweep
        uint64_t routing_ns = 0;
    };
    auto qs = std::make_unique<QueryState[]>(nq);
    for (uint32_t i = 0; i < nq; ++i) {
        qs[i].query = queries + static_cast<size_t>(i) * dim;
        qs[i].predicates =
            (per_query_predicates && !(*per_query_predicates)[i].empty())
                ? &(*per_query_predicates)[i]
                : &config.predicates;
        if (per_query_probe_fraction)
            qs[i].probe_fraction = (*per_query_probe_fraction)[i];
    }
    const bool batch_has_predicates = [&qs, nq]() {
        for (uint32_t i = 0; i < nq; ++i)
            if (!qs[i].predicates->empty()) return true;
        return false;
    }();

    // --- Phase 0: batched plane stage-1 (deployment shape) ---
    // Predicate-free queries get their candidates from ONE leaf-major
    // plane sweep — the plane is read once per batch instead of once
    // per query. Queries WITH predicates keep the legacy descent (v1).
    const bool plane_active = plane_ != nullptr && config.use_plane;
    if (plane_active) {
        std::vector<uint32_t> plane_q;      // indexes into qs
        for (uint32_t i = 0; i < nq; ++i) plane_q.push_back(i);
        if (!plane_q.empty()) {
            std::vector<const float*> qptr;
            qptr.reserve(plane_q.size());
            for (uint32_t i : plane_q) qptr.push_back(qs[i].query);
            // plane_route_batch_ wants a contiguous query array; the
            // queries are already contiguous rows when no per-query
            // skipping happened. Build a compact copy (dim rows apart).
            std::vector<float> compact(static_cast<size_t>(
                plane_q.size()) * manifest_.dim);
            for (size_t j = 0; j < plane_q.size(); ++j)
                std::memcpy(&compact[j * manifest_.dim],
                            qs[plane_q[j]].query,
                            manifest_.dim * sizeof(float));
            std::vector<std::vector<LeafCandidate>> pc;
            // Compact the per-query budgets alongside the queries (the
            // fraction vector is indexed by original query position).
            std::vector<float> pf;
            const std::vector<float>* pfp = nullptr;
            if (per_query_probe_fraction) {
                pf.reserve(plane_q.size());
                for (uint32_t j : plane_q)
                    pf.push_back((*per_query_probe_fraction)[j]);
                pfp = &pf;
            }
            const auto tp0 = std::chrono::steady_clock::now();
            plane_route_batch_(compact.data(),
                               static_cast<uint32_t>(plane_q.size()),
                               config, pc, pfp);
            const uint64_t pns = static_cast<uint64_t>(
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - tp0).count() * 1e9);
            for (size_t j = 0; j < plane_q.size(); ++j) {
                auto& s = qs[plane_q[j]];
                s.candidates = std::move(pc[j]);
                s.plane_done = true;
                s.routing_ns = pns / (plane_q.size() + 1);
            }
        }
    }

    // --- Phase 1: route every query (parallel over queries) ---
    {
        std::atomic<uint32_t> next{0};
        auto route_worker = [&]() {
            static thread_local SearchScratch rscratch;
            for (;;) {
                const uint32_t i =
                    next.fetch_add(1, std::memory_order_relaxed);
                if (i >= nq) break;
                auto& s = qs[i];
                // Per-query effective config: base + this query's
                // predicates. (Stack copy; predicates are small.)
                SearchConfig qcfg = config;
                qcfg.predicates = *s.predicates;
                if (s.probe_fraction > 0.0f) {
                    qcfg.probe_fraction = s.probe_fraction;
                    qcfg.n_probe = 0;  // fraction routing requires it
                }
                if (s.plane_done) {
                    // Candidates + routing_ns came from the Phase 0
                    // batched plane sweep. Same W policy as the Ok path
                    // (plane queries are predicate-free here).
                    s.W = std::max(
                        config.fastscan_W > 0 ? config.fastscan_W : 1000u,
                        k);
                    continue;  // releaseLeafPins no-op (no pins taken)
                }
                const auto tr0 = std::chrono::steady_clock::now();
                const RouteStatus st =
                    route_query_(s.query, qcfg, rscratch, s.candidates);
                s.routing_ns = static_cast<uint64_t>(
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tr0).count() * 1e9);
                if (st == RouteStatus::FallbackFiltered) {
                    s.fallback = true;
                } else if (st == RouteStatus::Ok) {
                    s.pred_col_indices = rscratch.pred_col_indices;
                    s.geo_lng_col_indices = rscratch.geo_lng_col_indices;
                    // W: verbatim from search() (no sweep_Ws in batch).
                    uint32_t W = std::max(
                        config.fastscan_W > 0 ? config.fastscan_W : 1000u, k);
                    if (!qcfg.predicates.empty()) {
                        // Adaptive W: see search() — the heap collects
                        // top-W by PQ distance without predicate filtering;
                        // 2x overscan over k/selectivity suffices.
                        constexpr float kOverscan = 2.0f;
                        const uint32_t adaptive_w =
                            rscratch.selectivity > 0.001f
                                ? static_cast<uint32_t>(
                                      static_cast<float>(k) /
                                      rscratch.selectivity * kOverscan)
                                : k * 200u;
                        W = std::max(W, adaptive_w);
                        W = std::max(W, k * 10u);  // floor
                    }
                    s.W = W;
                }
                // Route-phase pins (predicate summary pruning) covered
                // only summary reads — release per query; the sweep
                // unpins each leaf after its fanout scans.
                release_leaf_pins_(rscratch.pins);
            }
        };
        if (T == 1) {
            route_worker();
        } else {
            std::vector<std::future<void>> futs;
            for (uint32_t t = 0; t < T; ++t)
                futs.push_back(
                    std::async(std::launch::async, route_worker));
            for (auto& f : futs) f.get();
        }
    }

    // --- Phase 2: invert probe sets into leaf-major refs ---
    // One (page, qidx, slot) triple per probed leaf occurrence, sorted by
    // page: the unique-leaf sweep list and each leaf's fanout (the queries
    // that probed it, with their candidate slots) are contiguous slices.
    struct ProbeRef {
        PageId page;
        uint32_t qidx;
        uint32_t slot;
    };
    std::vector<ProbeRef> refs;
    {
        size_t total = 0;
        for (uint32_t i = 0; i < nq; ++i)
            total += qs[i].candidates.size();
        refs.reserve(total);
        for (uint32_t i = 0; i < nq; ++i) {
            const auto& cands = qs[i].candidates;
            for (uint32_t slot = 0; slot < cands.size(); ++slot) {
                if (cands[slot].page == kInvalidPage) continue;
                refs.push_back({cands[slot].page, i, slot});
            }
        }
        std::sort(refs.begin(), refs.end(),
                  [](const ProbeRef& a, const ProbeRef& b) {
                      if (a.page != b.page) return a.page < b.page;
                      if (a.qidx != b.qidx) return a.qidx < b.qidx;
                      return a.slot < b.slot;
                  });
    }
    struct UniqueLeaf {
        PageId page;
        uint32_t pages;
        uint32_t ref_begin;  // slice [ref_begin, next.ref_begin) of refs
    };
    std::vector<UniqueLeaf> uleaves;
    uleaves.reserve(refs.size());
    for (size_t i = 0; i < refs.size(); ++i) {
        if (uleaves.empty() || refs[i].page != uleaves.back().page) {
            const auto& c =
                qs[refs[i].qidx].candidates[refs[i].slot];
            uleaves.push_back({refs[i].page,
                               static_cast<uint32_t>(c.pages),
                               static_cast<uint32_t>(i)});
        }
    }
    const uint32_t n_unique =
        static_cast<uint32_t>(uleaves.size());

    // --- Read layer: ONE path with two backends ---
    // No-cache: direct preads into a per-thread buffer (page-ordered
    // sequential streams; the ZFS mmap fault path measured 2.4x slower
    // cold). Hot-set (cache on): pin, scan+harvest, UNPIN — pins held
    // longer disable eviction (measured: the "bounded" cache silently
    // held the full working set). Both backends harvest everything
    // finalize needs from the leaf into W-bounded pools (row_id, rerank
    // distance, predicate verdict) — no mmap pointers anywhere in the
    // batch path. exact_rerank_base reranks lazily through the row_id.
    const bool pread_sweep = !leaf_cache_;
    uint32_t max_extent_pages = 0;
    if (pread_sweep) {
        for (const auto& ul : uleaves)
            max_extent_pages = std::max(max_extent_pages, ul.pages);
    }
    const bool eager_rerank =
        config.rerank && !config.exact_rerank_base;

    // --- Phase 3: sweep unique leaves in page order (parallel chunks) ---
    // Threads claim contiguous chunks of the page-ordered unique-leaf
    // list. A query's probe set is subtree-contiguous, so it intersects
    // few chunks; each thread keeps its OWN per-query partial (setup +
    // bounded heap + bounded pool) — a query's setup/heap are never
    // touched by two threads at once. Partials merge deterministically
    // below.
    struct Partial {
        uint32_t qidx;
        std::unique_ptr<ScanSetup> setup;
        std::vector<HeapEntry> heap;
        std::vector<PoolEntry> pool;
    };
    std::vector<Partial> partials;
    {
        std::mutex adopt_mu;
        std::atomic<uint32_t> next_chunk{0};
        constexpr uint32_t kChunk = 32;  // page-adjacent leaves per claim
        auto sweep_worker = [&]() {
            std::vector<Partial> my_partials;
            std::unordered_map<uint32_t, size_t> pmap;  // qidx → partial
            // Pread scratch: one buffer per thread (the scan + harvest of
            // leaf L complete before the next leaf's pread reuses it).
            std::vector<uint8_t> pread_buf(
                pread_sweep
                    ? static_cast<size_t>(max_extent_pages) * kPageSize
                    : 0);
            // Per-worker scan+harvest wall (excludes preads): feeds the
            // batch.scan_ns overlap-decomposition metric.
            uint64_t my_scan_ns = 0;
            // Filter columns of the CURRENT leaf, parsed once per leaf
            // and shared by every query in its fanout (column layout is
            // query-independent; predicates are per query).
            std::vector<ColumnView> leaf_cols;
            for (;;) {
                const uint32_t start =
                    next_chunk.fetch_add(kChunk, std::memory_order_relaxed);
                if (start >= n_unique) break;
                const uint32_t end = std::min(start + kChunk, n_unique);
                if (end < n_unique) {
                    // Read-ahead the NEXT chunk while scanning this one:
                    // each thread's blocking preads serialize against its
                    // scans (measured: read wall ~23% of per-thread time
                    // while aggregate disk demand sits far below the
                    // device). WILLNEED gives the kernel the whole chunk
                    // scan (~hundreds of ms) to prefetch the next ~tens
                    // of MB — the per-thread pread then hits warm pages.
                    // No threads, no rings; the engine already relies on
                    // fadvise prefetch on the cache fill path.
                    const uint32_t nend =
                        std::min(end + kChunk, n_unique);
                    if (pread_sweep) {
                        const auto& nb = uleaves[end];
                        const auto& ne = uleaves[nend - 1];
                        ::posix_fadvise(
                            fd_,
                            static_cast<off_t>(nb.page) * kPageSize,
                            static_cast<off_t>(
                                ne.page + ne.pages - nb.page) * kPageSize,
                            POSIX_FADV_WILLNEED);
                    } else {
                        // Cache backend: hint only NON-RESIDENT extents —
                        // resident leaves would waste disk reads (the
                        // probe is a cheap shard-lock lookup per leaf).
                        for (uint32_t li2 = end; li2 < nend; ++li2) {
                            const auto& nl = uleaves[li2];
                            if (leaf_cache_->contains(nl.page))
                                continue;
                            ::posix_fadvise(
                                fd_,
                                static_cast<off_t>(nl.page) * kPageSize,
                                static_cast<off_t>(nl.pages) * kPageSize,
                                POSIX_FADV_WILLNEED);
                        }
                    }
                }
                for (uint32_t li = start; li < end; ++li) {
                    const auto& ul = uleaves[li];
                    LeafExtentCache::Handle h;
                    const uint8_t* leaf_ptr;
                    if (pread_sweep) {
                        // Direct pread: page-ordered sequential streams,
                        // no mmap faults, no page-cache pollution.
                        // Read wall is accumulated for the effective-
                        // bandwidth / uncoalesced-capacity metrics.
                        const size_t bytes =
                            static_cast<size_t>(ul.pages) * kPageSize;
                        const auto tr0 = std::chrono::steady_clock::now();
                        const bool ok =
                            pread_full(fd_, pread_buf.data(), bytes,
                                       static_cast<uint64_t>(ul.page) *
                                           kPageSize);
                        batch_stats_.on_read(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - tr0)
                                .count());
                        if (!ok) {
                            continue;  // I/O error: leaf contributes
                                       // nothing (mmap would SIGBUS)
                        }
                        leaf_ptr = pread_buf.data();
                    } else {
                        // Cache backend: fill preads block the worker the
                        // same way — feed batch read_ns on MISS only (the
                        // hit path is lock+LRU, not a read).
                        const auto tr0 = std::chrono::steady_clock::now();
                        bool hit = true;
                        leaf_ptr = pin_leaf_(ul.page, ul.pages, h, &hit);
                        if (!hit) {
                            batch_stats_.on_read(
                                std::chrono::duration_cast<
                                    std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - tr0)
                                    .count());
                        }
                    }
                    // Scan+harvest wall for this leaf (read excluded).
                    const auto ts0 = std::chrono::steady_clock::now();
                    // Parse the leaf's filter columns once (predicates
                    // evaluate per query against this shared view).
                    if (batch_has_predicates) {
                        const auto* lh = reinterpret_cast<
                            const TreeLeafHeader*>(leaf_ptr);
                        auto layout = LeafFilterLayout::from_geometry(
                            leaf_ptr, coder_->geometry(lh));
                        leaf_cols = parse_filter_columns(
                            layout.filter_base, lh->count,
                            manifest_.schema);
                    }
                    const uint32_t ref_end =
                        li + 1 < n_unique ? uleaves[li + 1].ref_begin
                                          : static_cast<uint32_t>(refs.size());
                    // Resolve/ensure this leaf's Partials FIRST, by index —
                    // my_partials.push_back may reallocate, so pointers
                    // collected now would dangle. bind_leaf is per
                    // (query,leaf): setups are per-query objects, so each
                    // query's setup gets bound once per leaf, exactly as
                    // the per-ref loop did.
                    const uint32_t nrefs = ref_end - ul.ref_begin;
                    static thread_local std::vector<uint32_t> pidx;
                    pidx.resize(nrefs);
                    for (uint32_t k = 0; k < nrefs; ++k) {
                        const auto& fr = refs[ul.ref_begin + k];
                        QueryState& s = qs[fr.qidx];
                        Partial* p;
                        auto it = pmap.find(fr.qidx);
                        if (it == pmap.end()) {
                            Partial np;
                            np.qidx = fr.qidx;
                            np.setup = coder_->scan_setup(s.query);
                            heap_init(np.heap, s.W);
                            pool_init(np.pool, s.W);
                            my_partials.push_back(std::move(np));
                            p = &my_partials.back();
                            pmap.emplace(fr.qidx, my_partials.size() - 1);
                        } else {
                            p = &my_partials[it->second];
                        }
                        pidx[k] = static_cast<uint32_t>(
                            p - my_partials.data());
                        if (per_leaf) coder_->bind_leaf(*p->setup, leaf_ptr);
                    }
                    // Harvest: resolve everything finalize needs from
                    // THIS leaf while the buffer is valid. Entries are
                    // admitted to the W-bounded pool only if they beat
                    // its worst slot (cheap pq-dist compare first;
                    // row-id/rerank/predicate work only on admission).
                    auto harvest = [&](QueryState& s, Partial* p,
                                       const ProbeRef& fr) {
                        const auto* lh = reinterpret_cast<
                            const TreeLeafHeader*>(leaf_ptr);
                        const RowId* rids =
                            reinterpret_cast<const RowId*>(
                                leaf_ptr +
                                coder_->geometry(lh).rowids_offset);
                        const bool has_preds = !s.predicates->empty();
                        for (const auto& e : p->heap) {
                            if (e.leaf_slot != fr.slot) continue;
                            if (!pool_should_replace(
                                    p->pool, e.pq_dist, e.leaf_slot,
                                    e.local_idx))
                                continue;
                            PoolEntry pe;
                            pe.pq_dist = e.pq_dist;
                            pe.leaf_slot = e.leaf_slot;
                            pe.local_idx = e.local_idx;
                            pe.row_id = rids[e.local_idx];
                            pe.dist = 0.0f;
                            pe.reranked = false;
                            pe.pred_ok = true;
                            if (eager_rerank) {
                                pe.dist = coder_->rerank(
                                    s.query, *p->setup, leaf_ptr,
                                    e.local_idx, nullptr);
                                pe.reranked = true;
                            }
                            if (has_preds) {
                                pe.pred_ok = eval_all_predicates(
                                    leaf_cols, manifest_.schema,
                                    e.local_idx, *s.predicates,
                                    s.pred_col_indices,
                                    s.geo_lng_col_indices);
                            }
                            pool_replace(p->pool, std::move(pe));
                        }
                    };
                    // Scan: group this leaf's refs into batches of 4 when
                    // the coder shares the row decode across queries (one
                    // pass over the code rows per group instead of per
                    // ref); otherwise per-query as before.
                    if (coder_->supports_batch_scan()) {
                        for (uint32_t k = 0; k < nrefs;) {
                            const uint32_t bn = std::min(4u, nrefs - k);
                            const ScanSetup* st[4];
                            RawScanHeap rhs[4];
                            RawScanHeap* rhs_p[4];
                            for (uint32_t b = 0; b < bn; ++b) {
                                const auto& fr =
                                    refs[ul.ref_begin + k + b];
                                QueryState& s = qs[fr.qidx];
                                Partial* p = &my_partials[pidx[k + b]];
                                st[b] = p->setup.get();
                                rhs[b] = RawScanHeap{&p->heap, s.W,
                                                     fr.slot};
                                rhs_p[b] = &rhs[b];
                            }
                            coder_->scan_leaf_batch(st, leaf_ptr, rhs_p, bn);
                            for (uint32_t b = 0; b < bn; ++b) {
                                const auto& fr =
                                    refs[ul.ref_begin + k + b];
                                harvest(qs[fr.qidx],
                                        &my_partials[pidx[k + b]], fr);
                            }
                            k += bn;
                        }
                    } else {
                        for (uint32_t k = 0; k < nrefs; ++k) {
                            const auto& fr = refs[ul.ref_begin + k];
                            QueryState& s = qs[fr.qidx];
                            Partial* p = &my_partials[pidx[k]];
                            RawScanHeap rh{&p->heap, s.W, fr.slot};
                            coder_->scan_leaf(*p->setup, leaf_ptr, rh);
                            harvest(s, p, fr);
                        }
                    }
                    // Hot-set: the leaf's harvest is complete — nothing
                    // references the buffer. Unpin so the entry can
                    // evict (pins held longer disable eviction).
                    if (h.entry) leaf_cache_->unpin(h);
                    my_scan_ns +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - ts0)
                            .count();
                }
            }
            std::lock_guard<std::mutex> lk(adopt_mu);
            batch_stats_.on_scan(my_scan_ns);
            partials.insert(partials.end(),
                            std::make_move_iterator(my_partials.begin()),
                            std::make_move_iterator(my_partials.end()));
        };
        const auto t_sweep0 = std::chrono::steady_clock::now();
        {
            // Single leaf-stream invariant: at most one Phase-3 sweep in
            // flight process-wide (see scan_stream_mu_). Concurrent
            // windows' routing overlaps this; their sweeps queue here.
            std::lock_guard<std::mutex> stream_lk(scan_stream_mu_);
            if (T == 1) {
                sweep_worker();
            } else {
                std::vector<std::future<void>> futs;
                for (uint32_t t = 0; t < T; ++t)
                    futs.push_back(
                        std::async(std::launch::async, sweep_worker));
                for (auto& f : futs) f.get();
            }
        }
        batch_stats_.on_sweep(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_sweep0)
                .count(),
            T);
    }

    // --- Merge per-query partials (deterministic) ---
    // Heaps: concat, strip sentinel slots, keep the top-W by
    // (pq_dist asc, tie: leaf_slot, local_idx) — bit-stable regardless of
    // how leaves were sharded. Pools: concat, strip sentinels (sorted for
    // lookup below).
    {
        std::vector<std::pair<uint32_t, size_t>> by_q;  // (qidx, partial)
        by_q.reserve(partials.size());
        for (size_t pi = 0; pi < partials.size(); ++pi)
            by_q.push_back({partials[pi].qidx, pi});
        std::sort(by_q.begin(), by_q.end());
        size_t i = 0;
        while (i < by_q.size()) {
            size_t j = i;
            while (j < by_q.size() && by_q[j].first == by_q[i].first) ++j;
            QueryState& s = qs[by_q[i].first];
            if (!s.setup) s.setup = std::move(partials[by_q[i].second].setup);
            size_t total = 0, pool_total = 0;
            for (size_t l = i; l < j; ++l) {
                total += partials[by_q[l].second].heap.size();
                pool_total += partials[by_q[l].second].pool.size();
            }
            s.heap.reserve(total);
            s.pool.reserve(pool_total);
            for (size_t l = i; l < j; ++l) {
                auto& ph = partials[by_q[l].second].heap;
                s.heap.insert(s.heap.end(),
                              std::make_move_iterator(ph.begin()),
                              std::make_move_iterator(ph.end()));
                auto& pp = partials[by_q[l].second].pool;
                s.pool.insert(s.pool.end(),
                              std::make_move_iterator(pp.begin()),
                              std::make_move_iterator(pp.end()));
            }
            s.heap.erase(std::remove_if(s.heap.begin(), s.heap.end(),
                                        heap_entry_is_sentinel),
                         s.heap.end());
            s.pool.erase(std::remove_if(s.pool.begin(), s.pool.end(),
                                        pool_entry_is_sentinel),
                         s.pool.end());
            if (s.heap.size() > s.W) {
                std::sort(s.heap.begin(), s.heap.end(),
                          [](const HeapEntry& a, const HeapEntry& b) {
                              if (a.pq_dist != b.pq_dist)
                                  return a.pq_dist < b.pq_dist;
                              if (a.leaf_slot != b.leaf_slot)
                                  return a.leaf_slot < b.leaf_slot;
                              return a.local_idx < b.local_idx;
                          });
                s.heap.resize(s.W);
            }
            i = j;
        }
    }

    // --- Phase 4: per-query finalize (parallel over queries) ---
    // Single stable path: pool lookup (row_id + rerank distance +
    // predicate verdict). FallbackFiltered queries (extreme selectivity)
    // run the per-query brute-force path — that is a routing decision,
    // not a batch read layer.
    {
        std::atomic<uint32_t> next{0};
        std::atomic<uint64_t> fallback_count{0};
        auto finalize_worker = [&]() {
            for (;;) {
                const uint32_t i =
                    next.fetch_add(1, std::memory_order_relaxed);
                if (i >= nq) break;
                auto& s = qs[i];
                if (s.fallback) {
                    results[i] = search(s.query, k, config);
                    fallback_count.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (s.candidates.empty() || s.heap.empty() || !s.setup) {
                    continue;  // empty results
                }
                const auto tf0 = std::chrono::steady_clock::now();
                finalize_pool_query_(s.query, k, coder_->metric(), dim,
                                     config, s.pool, s.heap, results[i]);
                const double fw = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - tf0).count();
                uint64_t bytes = 0;
                for (const auto& c : s.candidates)
                    bytes += static_cast<uint64_t>(c.pages) * kPageSize;
                search_stats_.on_query(
                    fw + static_cast<double>(s.routing_ns) / 1e9,
                    s.candidates.size(), bytes, s.heap.size(),
                    s.routing_ns);
            }
        };
        if (T == 1) {
            finalize_worker();
        } else {
            std::vector<std::future<void>> futs;
            for (uint32_t t = 0; t < T; ++t)
                futs.push_back(
                    std::async(std::launch::async, finalize_worker));
            for (auto& f : futs) f.get();
        }
        (void)fallback_count;
    }

    uint64_t unique_bytes = 0;
    for (const auto& ul : uleaves)
        unique_bytes += static_cast<uint64_t>(ul.pages) * kPageSize;
    batch_stats_.on_batch(
        nq, n_unique, refs.size(), unique_bytes, 0,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count());
}
// Brute-force PQ-decode fallback for extreme low selectivity (<1%).
// ===========================================================================

std::vector<Candidate> IVFTreeIndex::search_brute_force_filtered(
    const float* query, uint32_t k, const SearchConfig& config,
    const std::vector<uint32_t>& pred_col_indices,
    const std::vector<uint32_t>& geo_lng_col_indices,
    std::vector<std::pair<const uint8_t*, uint32_t>>* /*payload_locs*/) const {

    // The brute-force body below is coder-generic (per-row rerank decode +
    // filter evaluation), but was originally implemented and validated only
    // for the global-PQ family; keep a debug log for the others.
    if (coder_->family() != CoderFamily::GlobalPq) {
        spdlog::debug("[sextant] brute-force filtered search on {} "
                      "(originally validated for global pq only)",
                      coder_->family_name());
    }
    const uint32_t summary_size = manifest_.summary_size;

    // Walk the tree to collect ALL leaf pages. We need a full tree walk,
    // not just routed candidates — at extreme low selectivity, the matching
    // leaves may be in subtrees that routing wouldn't visit.
    struct LeafInfo { PageId page; uint64_t pages; };
    std::vector<LeafInfo> all_leaves;

    const uint32_t cesize = child_entry_size(manifest_.dim, summary_size);

    // Simple DFS from root.
    std::vector<PageId> node_stack;
    node_stack.push_back(superblock_.root_node_page());

    while (!node_stack.empty()) {
        const PageId page = node_stack.back();
        node_stack.pop_back();
        if (page == kInvalidPage) continue;
        const uint8_t* node_ptr = mmap_base_ +
            static_cast<uint64_t>(page) * kPageSize;
        const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
        const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);
        for (uint32_t j = 0; j < nh->n_children; ++j) {
            const auto* ce = reinterpret_cast<const ChildEntry*>(p);
            if (ce->child_page == kInvalidPage) { p += cesize; continue; }
            if (ce->is_leaf) {
                all_leaves.push_back({leaf_table_[ce->child_page].page,
                                      leaf_table_[ce->child_page].pages});
            } else {
                node_stack.push_back(ce->child_page);
            }
            p += cesize;
        }
    }

    // Scan each leaf: check summary, then scan filter columns for exact matches.
    std::vector<Candidate> results;

    for (const auto& li : all_leaves) {
        const uint8_t* leaf_ptr = mmap_base_ +
            static_cast<uint64_t>(li.page) * kPageSize;
        const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
        const uint32_t count = lh->count;
        if (count == 0) continue;

        // Summary check: skip leaves that can't match.
        const uint8_t* summary = leaf_ptr + leaf_filter_offset();
        if (!summary_may_match(summary, summary_size, manifest_.schema,
                                config.predicates, pred_col_indices, geo_lng_col_indices))
            continue;

        // Parse filter columns for this leaf.
        auto layout = LeafFilterLayout::from_geometry(
            leaf_ptr, coder_->geometry(lh));
        auto col_views = parse_filter_columns(layout.filter_base, count,
                                                manifest_.schema);

        // Scan all rows: find exact matches.
        for (uint32_t i = 0; i < count; ++i) {
            if (!eval_all_predicates(col_views, manifest_.schema, i,
                                       config.predicates, pred_col_indices, geo_lng_col_indices))
                continue;

            // Exact match — decode the row and compute the exact distance.
            const float dist = coder_->rerank(query, leaf_ptr, i, nullptr);
            results.push_back({layout.row_ids[i], dist});
        }
    }

    // Sort by distance, return top-k.
    if (results.size() > k) {
        std::nth_element(results.begin(), results.begin() + k, results.end(),
                          [](const Candidate& a, const Candidate& b) {
                              return a.dist < b.dist;
                          });
        results.resize(k);
    }
    std::sort(results.begin(), results.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.dist < b.dist;
              });
    return results;
}

}  // namespace sextant::tree
