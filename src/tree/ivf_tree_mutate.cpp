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
#include "tree/tree_manifest.hpp"   // manifest_to_toml (n_vectors rewrite)
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
// Dynamic insert/delete (single-writer, multi-reader)
//
// Design (architecture plan §3.11):
// - Insert: route → leaf_id → grow extent + append code/row_id/filter/payload
//   → eager summary update → update leaf table → commit superblock.
// - Delete: route → leaf → swap-remove (last slot → deleted slot) → count--
//   → mark summary_dirty → commit.
// - Split: when count > 2×leaf_capacity, k-means split into two leaves.
//
// The mmap is read-only (PROT_READ). All writes go through file_ (RW fd).
// After commit, remap_() refreshes the mmap so search sees the new data.
// ===========================================================================

namespace {

/// Locate a leaf_id in the root children for depth=1 trees.
/// Returns the child index whose child_page == leaf_id.
/// For depth=1, root children are leaves directly; child_page is the leaf_id.
[[maybe_unused]]
inline int32_t find_root_child_for_leaf(const TreeNodeHeader* rh,
                                         uint16_t dim, uint32_t summary_size,
                                         uint32_t leaf_id) {
    const uint32_t cesize = child_entry_size(dim, summary_size);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(rh) +
                       sizeof(TreeNodeHeader);
    for (uint32_t i = 0; i < rh->n_children; ++i) {
        const auto* ce = reinterpret_cast<const ChildEntry*>(p);
        if (ce->is_leaf && ce->child_page == leaf_id) return static_cast<int32_t>(i);
        p += cesize;
    }
    return -1;
}

/// Find the (node_page, child_index) that points to a given leaf_id in a
/// depth=2 tree. The parent is an L2 internal node (root child).
/// Returns {parent_page, parent_pages, child_slot} or {kInvalidPage,0,-1}.
struct ParentLoc {
    PageId parent_page;
    uint32_t parent_pages;
    int32_t child_slot;  // index within the parent node
};

[[maybe_unused]]
inline ParentLoc find_parent_of_leaf(const uint8_t* mmap_base,
                                      PageId rc_page, uint64_t rc_pages,
                                      uint16_t is_leaf,
                                      uint16_t dim, uint32_t summary_size,
                                      uint32_t leaf_id) {
    if (is_leaf || rc_page == kInvalidPage)
        return {kInvalidPage, 0, -1};
    const uint8_t* node_ptr = mmap_base + rc_page * kPageSize;
    const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
    const uint32_t cesize = child_entry_size(dim, summary_size);
    const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);
    for (uint32_t j = 0; j < nh->n_children; ++j) {
        const auto* ce = reinterpret_cast<const ChildEntry*>(p);
        if (ce->is_leaf && ce->child_page == leaf_id)
            return {rc_page, static_cast<uint32_t>(rc_pages),
                    static_cast<int32_t>(j)};
        p += cesize;
    }
    return {kInvalidPage, 0, -1};
}

}  // namespace

// --- Route a vector to its nearest leaf_id ---

uint32_t IVFTreeIndex::route_to_leaf_id_depth2_pca_(const float* query,
                                                      const float* query_pca) const {
    // Find nearest root child (PCA space).
    uint32_t best_root = 0;
    float best_d = std::numeric_limits<float>::max();
    const uint32_t k_root = root_header_->n_children;
    const uint32_t pd = pca_dims_;
    for (uint32_t c = 0; c < k_root; ++c) {
        const float* rcc = &pca_root_centroids_[c * pd];
        float d = 0.0f;
        for (uint32_t kk = 0; kk < pd; ++kk) {
            const float diff = query_pca[kk] - rcc[kk];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best_root = c; }
    }
    // Descend into the root child's L2 node, find nearest leaf (PCA space).
    const auto& rc = root_children_[best_root];
    if (rc.is_leaf) return static_cast<uint32_t>(rc.page);  // depth=1 edge

    const uint8_t* node_ptr = mmap_base_ + rc.page * kPageSize;
    const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
    const uint32_t cesize = child_entry_size(manifest_.dim, manifest_.summary_size);

    uint32_t best_leaf = 0;
    float best_ld = std::numeric_limits<float>::max();
    const uint64_t base = pca_leaf_base_[best_root];
    const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);
    for (uint32_t j = 0; j < nh->n_children; ++j) {
        const auto* ce = reinterpret_cast<const ChildEntry*>(p);
        if (!ce->is_leaf || ce->child_page == kInvalidPage) { p += cesize; continue; }
        const uint32_t gid = static_cast<uint32_t>(base) + j;
        const float* lc = &pca_leaf_centroids_[gid * pd];
        float d = 0.0f;
        for (uint32_t kk = 0; kk < pd; ++kk) {
            const float diff = query_pca[kk] - lc[kk];
            d += diff * diff;
        }
        if (d < best_ld) { best_ld = d; best_leaf = static_cast<uint32_t>(ce->child_page); }
        p += cesize;
    }
    return best_leaf;
}

uint32_t IVFTreeIndex::route_to_leaf_id_(const float* query) const {
    const uint16_t dim = manifest_.dim;
    const uint32_t summary_size = manifest_.summary_size;
    const MetricKind metric = coder_->metric();

    // PCA projection if applicable.
    std::vector<float> query_pca;
    const float* pca_q = nullptr;
    if (pca_dims_ > 0) {
        query_pca.resize(pca_dims_);
        for (uint32_t k = 0; k < pca_dims_; ++k)
            query_pca[k] = simd::dot_f32(&pca_proj_[k * dim], query, dim)
                          - pca_mean_proj_[k];
        pca_q = query_pca.data();

        // Depth=2 PCA uses the specialized path.
        if (manifest_.depth == 2)
            return route_to_leaf_id_depth2_pca_(query, pca_q);
    }

    // Generic FP16 routing (depth=1 or depth≥3 without PCA leaf centroids).
    std::vector<float16_t> query_fp16(dim);
    cast_fp32_to_fp16(query, query_fp16.data(), dim);

    // Find nearest root child.
    const uint32_t k_root = root_header_->n_children;
    uint32_t best_root = 0;
    float best_rd = std::numeric_limits<float>::max();
    for (uint32_t c = 0; c < k_root; ++c) {
        float d;
        if (pca_dims_ > 0) {
            const float* rcc = &pca_root_centroids_[c * pca_dims_];
            d = 0.0f;
            for (uint32_t kk = 0; kk < pca_dims_; ++kk) {
                const float diff = pca_q[kk] - rcc[kk];
                d += diff * diff;
            }
        } else {
            d = simd::dist_f16(metric, query_fp16.data(),
                               root_children_[c].centroid, dim);
        }
        if (d < best_rd) { best_rd = d; best_root = c; }
    }

    const auto& rc = root_children_[best_root];
    if (rc.is_leaf) {
        // For depth=1, root children are leaves. The leaf_id is the child_page
        // field in the root node's child entry (before leaf table resolution).
        // root_children_ stores the RESOLVED physical page, so we read the raw
        // leaf_id from the mmap'd root node.
        const uint32_t cesize = child_entry_size(dim, summary_size);
        const auto* ce = reinterpret_cast<const ChildEntry*>(
            reinterpret_cast<const uint8_t*>(root_header_) +
            sizeof(TreeNodeHeader) + best_root * cesize);
        return static_cast<uint32_t>(ce->child_page);
    }

    // Descend through internal levels (depth≥2 FP16 or depth≥3).
    PageId cur_page = rc.page;
    const uint32_t cesize = child_entry_size(dim, summary_size);

    for (uint16_t level = 1; level < manifest_.depth; ++level) {
        const uint8_t* node_ptr = mmap_base_ + cur_page * kPageSize;
        const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
        const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);
        uint32_t best_j = 0;
        float best_d = std::numeric_limits<float>::max();
        for (uint32_t j = 0; j < nh->n_children; ++j) {
            const auto* ce = reinterpret_cast<const ChildEntry*>(p);
            if (ce->child_page == kInvalidPage) { p += cesize; continue; }
            const float16_t* cent = reinterpret_cast<const float16_t*>(
                p + sizeof(ChildEntry));
            const float d = simd::dist_f16(metric, query_fp16.data(), cent, dim);
            if (d < best_d) { best_d = d; best_j = j; }
            p += cesize;
        }
        const auto* best_ce = reinterpret_cast<const ChildEntry*>(
            node_ptr + sizeof(TreeNodeHeader) + best_j * cesize);
        if (best_ce->is_leaf) return static_cast<uint32_t>(best_ce->child_page);
        cur_page = best_ce->child_page;
    }
    // Should not reach here (depth exhausted without hitting a leaf).
    throw Error(ErrorCode::CorruptIndex,
                "route_to_leaf_id_: tree descent did not reach a leaf");
}

// --- Leaf read/write helpers ---

std::vector<uint8_t> IVFTreeIndex::read_leaf_(uint32_t leaf_id) const {
    if (leaf_id >= leaf_table_.size())
        throw Error(ErrorCode::CorruptIndex,
                    "read_leaf_: leaf_id " + std::to_string(leaf_id) +
                    " out of range (table size " +
                    std::to_string(leaf_table_.size()) + ")");
    const auto& e = leaf_table_[leaf_id];
    std::vector<uint8_t> buf(static_cast<size_t>(e.pages) * kPageSize);
    file_.read_pages(e.page, e.pages, buf.data());
    return buf;
}

void IVFTreeIndex::write_leaf_(uint32_t leaf_id, std::vector<uint8_t>& buf,
                                uint32_t new_pages, PageAllocator& alloc) {
    const auto& old_entry = leaf_table_[leaf_id];
    const uint32_t old_pages = old_entry.pages;

    if (new_pages <= old_pages) {
        // In-place write (shrink or same size).
        file_.write_pages(old_entry.page, new_pages, buf.data());
        leaf_table_[leaf_id].pages = new_pages;
        // If we shrank, free the tail pages.
        if (new_pages < old_pages) {
            alloc.free_extent(file_, old_entry.page + new_pages,
                              old_pages - new_pages);
        }
    } else {
        // Need to grow: allocate a new extent, write, free old, update table.
        const PageId new_page = alloc.alloc_extent(file_, new_pages);
        file_.write_pages(new_page, new_pages, buf.data());
        alloc.free_extent(file_, old_entry.page, old_pages);
        leaf_table_[leaf_id].page = new_page;
        leaf_table_[leaf_id].pages = new_pages;
    }
}

// --- Remap the read-only mmap after mutations ---

void IVFTreeIndex::remap_() {
    // Cached leaf extents are wholesale stale after any mutation commit —
    // pages may have been rewritten or relocated.
    if (leaf_cache_) leaf_cache_->invalidate_all();
    if (mmap_base_) {
        ::munmap(const_cast<uint8_t*>(mmap_base_), mmap_size_);
        mmap_base_ = nullptr;
    }
    // Reload the superblock (commit_seq, pointers may have changed).
    superblock_.load(file_);
    // Reload the leaf table from disk (page locations may have changed).
    if (superblock_.leaf_table_page() != kInvalidPage &&
        superblock_.leaf_table_pages() > 0) {
        const auto ltp = superblock_.leaf_table_page();
        const auto ltpg = superblock_.leaf_table_pages();
        std::vector<uint8_t> blob(static_cast<size_t>(ltpg) * kPageSize);
        file_.read_pages(ltp, ltpg, blob.data());
        uint64_t n_entries = 0;
        std::memcpy(&n_entries, blob.data(), 8);
        leaf_table_.resize(n_entries);
        std::memcpy(leaf_table_.data(), blob.data() + 8,
                    n_entries * sizeof(LeafTableEntry));
    }
    // Update the manifest's n_leaves (may have changed).
    manifest_.n_leaves = static_cast<uint32_t>(superblock_.n_leaves());

    mmap_size_ = file_.num_pages() * kPageSize;
    if (mmap_size_ == 0) return;
    void* addr = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED,
                        file_.fd(), 0);
    if (addr == MAP_FAILED) {
        throw Error(ErrorCode::IoError,
                    "IVFTreeIndex::remap_: mmap failed: " +
                        std::string(std::strerror(errno)));
    }
    mmap_base_ = static_cast<const uint8_t*>(addr);
    // Re-parse root node (pointers are into the old mmap).
    root_header_ = nullptr;
    root_children_.clear();
    load_root_from_mmap();
    // Rebuild PCA leaf centroids from the current on-disk tree structure.
    // This is critical: after a split, new leaves have no entry in the
    // build-time PCA blob. We re-derive PCA centroids by projecting the
    // inline FP16 leaf centroids through pca_proj_, so routing stays accurate.
    rebuild_pca_leaf_centroids_();
}

void IVFTreeIndex::rebuild_pca_leaf_centroids_() {
    pca_leaf_centroids_.clear();
    pca_leaf_base_.clear();
    if (pca_dims_ == 0 || manifest_.depth != 2) return;

    const uint16_t dim = manifest_.dim;
    const uint32_t summary_size = manifest_.summary_size;
    const uint32_t cesize = child_entry_size(dim, summary_size);

    pca_leaf_base_.assign(root_children_.size(), UINT64_MAX);
    uint32_t global_id = 0;
    std::vector<float> centroid_fp32(dim);
    std::vector<float> centroid_pca(pca_dims_);

    for (uint32_t c = 0; c < root_children_.size(); ++c) {
        const auto& rc = root_children_[c];
        if (rc.is_leaf || rc.page == kInvalidPage) continue;
        const uint8_t* node_ptr = mmap_base_ +
            static_cast<uint64_t>(rc.page) * kPageSize;
        const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
        pca_leaf_base_[c] = global_id;
        const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);
        for (uint32_t j = 0; j < nh->n_children; ++j) {
            const auto* ce = reinterpret_cast<const ChildEntry*>(p);
            if (!ce->is_leaf) { p += cesize; continue; }
            // Read the inline FP16 centroid → FP32.
            const float16_t* cent_fp16 = reinterpret_cast<const float16_t*>(
                p + sizeof(ChildEntry));
            for (uint16_t d = 0; d < dim; ++d)
                centroid_fp32[d] = static_cast<float>(cent_fp16[d]);
            // Project through PCA: proj[k] = dot(rotation[k], centroid) - mean_proj[k].
            for (uint32_t k = 0; k < pca_dims_; ++k)
                centroid_pca[k] = simd::dot_f32(&pca_proj_[k * dim],
                                                 centroid_fp32.data(), dim)
                                  - pca_mean_proj_[k];
            pca_leaf_centroids_.insert(pca_leaf_centroids_.end(),
                                       centroid_pca.begin(), centroid_pca.end());
            ++global_id;
            p += cesize;
        }
    }
}

void IVFTreeIndex::commit_mutable_(PageAllocator& alloc) {
    // Rewrite the leaf table blob.
    const PageId old_lt_page = superblock_.leaf_table_page();
    const uint32_t old_lt_pages = superblock_.leaf_table_pages();
    const uint64_t n_entries = leaf_table_.size();
    const uint64_t blob_bytes = 8 + n_entries * sizeof(LeafTableEntry);
    const uint32_t lt_npg = static_cast<uint32_t>(
        (blob_bytes + kPageSize - 1) / kPageSize);
    const PageId lt_page = alloc.alloc_extent(file_, lt_npg);
    {
        std::vector<uint8_t> b(lt_npg * kPageSize, 0);
        std::memcpy(b.data(), &n_entries, 8);
        for (uint64_t i = 0; i < n_entries; ++i) {
            LeafTableEntry e{leaf_table_[i].page, leaf_table_[i].pages};
            std::memcpy(b.data() + 8 + i * sizeof(LeafTableEntry),
                        &e, sizeof(e));
        }
        file_.write_pages(lt_page, lt_npg, b.data());
    }
    if (old_lt_page != kInvalidPage)
        alloc.free_extent(file_, old_lt_page, old_lt_pages);

    // Rewrite the manifest blob when the logical row count changed
    // (insert_batch / delete_batch update manifest_.n_vectors in memory;
    // the on-disk TOML must follow so a reopen sees the new count).
    // Must run BEFORE alloc.flush_bitmap — a post-flush allocation would
    // never reach the on-disk bitmap and a later commit would hand the
    // same pages out twice (silent leaf-table corruption).
    if (manifest_.n_vectors != n_vectors_disk_) {
        std::string cfg_toml = manifest_to_toml(manifest_);
        const uint32_t cfg_npg = static_cast<uint32_t>(
            (cfg_toml.size() + kPageSize - 1) / kPageSize);
        const PageId cfg_page = alloc.alloc_extent(file_, cfg_npg);
        {
            std::vector<uint8_t> b(cfg_npg * kPageSize, 0);
            std::memcpy(b.data(), cfg_toml.data(), cfg_toml.size());
            file_.write_pages(cfg_page, cfg_npg, b.data());
        }
        const PageId old_cfg = superblock_.config_page();
        const uint32_t old_cfg_pg = superblock_.config_pages();
        if (old_cfg != kInvalidPage)
            alloc.free_extent(file_, old_cfg, old_cfg_pg);
        superblock_.set_config(cfg_page, cfg_npg);
        n_vectors_disk_ = manifest_.n_vectors;
    }

    alloc.flush_bitmap(file_);
    superblock_.set_leaf_table(lt_page, lt_npg);
    superblock_.set_n_pages(file_.num_pages());
    superblock_.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    superblock_.set_n_leaves(leaf_table_.size());
    superblock_.commit(file_);
    file_.sync();
    alloc.clear_free_list();
    remap_();
}

// ===========================================================================
// split_leaf_: k-means(K=2) split of an oversized leaf.
//
// Extracts PQ codes from the leaf's FastScan layout, runs kmeans_pq(K=2),
// writes two new leaf extents from the two shards, decodes the medoid
// centroids → FP16 for the parent node, updates the parent node's child
// list, grows the leaf table, frees the old leaf extent.
//
// No closure (disjoint partition). The k-means medoids become the new
// leaf centroids.
// ===========================================================================

uint32_t IVFTreeIndex::split_leaf_(uint32_t leaf_id, PageAllocator& alloc) {
    const uint16_t dim = manifest_.dim;
    const uint32_t summary_size = manifest_.summary_size;
    const uint32_t code_size = coder_->code_size();
    const bool has_ip_bias = coder_->leaf_has_ip_bias();
    const CoderFamily fam = coder_->family();

    // 1. Read the leaf.
    std::vector<uint8_t> old_buf = read_leaf_(leaf_id);
    const auto* old_lh = reinterpret_cast<const TreeLeafHeader*>(old_buf.data());
    const uint32_t count = static_cast<uint32_t>(old_lh->count);
    const LeafGeometry old_geo = coder_->geometry(old_lh);

    // 2. Family-owned decode-out: compact codes (global PQ / scalar flat
    //    copy) and/or decoded f32 vectors (scalar / local families).
    std::vector<uint8_t> codes;
    if (fam == CoderFamily::GlobalPq || fam == CoderFamily::ScalarLm) {
        codes.resize(static_cast<size_t>(count) * code_size);
        coder_->extract_codes(old_buf.data(), count, codes.data());
    }
    std::vector<float> vecs;
    if (fam != CoderFamily::GlobalPq) {
        vecs.resize(static_cast<size_t>(count) * dim);
        for (uint32_t i = 0; i < count; ++i)
            coder_->decode_one(old_buf.data(), i,
                               vecs.data() + static_cast<size_t>(i) * dim);
    }
    // Scalar + InnerProduct: per-vector IP biases follow the codes.
    std::vector<float16_t> old_biases;
    if (has_ip_bias) {
        const float16_t* src = reinterpret_cast<const float16_t*>(
            old_buf.data() + old_geo.codes_offset +
            static_cast<uint64_t>(count) * code_size);
        old_biases.assign(src, src + count);
    }

    // 3. k-means(K=2) fork is family-owned: global PQ runs kmeans_pq on the
    //    codes; scalar/local run a small f32 Lloyd loop on decoded vectors.
    LeafCoder::SplitPlan plan = coder_->plan_split(
        old_buf.data(), codes.data(), vecs.data(), count, leaf_id);

    // Edge case: if one group is empty (all vectors assigned to one
    // centroid), split the assignment deterministically by index parity.
    if (plan.group0.empty() || plan.group1.empty()) {
        plan.group0.clear();
        plan.group1.clear();
        for (uint32_t i = 0; i < count; ++i) {
            if (i < count / 2) plan.group0.push_back(i);
            else plan.group1.push_back(i);
        }
    }

    // 5. Read old row_ids + filter column data (if present).
    const RowId* old_rids = reinterpret_cast<const RowId*>(
        old_buf.data() + old_geo.rowids_offset);
    const bool has_filter = manifest_.schema.n_filter_columns() > 0;
    std::vector<ColumnData> old_filter_cols;
    if (has_filter) {
        read_filter_columns(old_buf.data() + old_geo.filter_offset, count,
                            manifest_.schema, old_filter_cols);
    }

    // 7. Helper lambda to write a new leaf from a group of vector indices.
    auto write_leaf_from_group = [&](const std::vector<uint32_t>& group,
                                      const std::vector<float16_t>& centroid)
        -> LeafTableEntry {
        const uint32_t gc = static_cast<uint32_t>(group.size());

        // Compute filter_cols_bytes for the new leaf.
        uint64_t fcb = 0;
        std::vector<ColumnData> group_filter_cols;
        if (has_filter) {
            group_filter_cols = select_filter_rows(old_filter_cols,
                                                   manifest_.schema, group);
            fcb = filter_columns_bytes(gc, manifest_.schema, group_filter_cols);
        }

        const uint32_t npg = static_cast<uint32_t>(
            (coder_->extent_bytes(gc, summary_size, fcb) + kPageSize - 1) /
            kPageSize);
        std::vector<uint8_t> nb(static_cast<size_t>(npg) * kPageSize, 0);

        // Header + summary.
        std::memcpy(nb.data(), old_buf.data(),
                    sizeof(TreeLeafHeader) + summary_size);
        auto* nlh = reinterpret_cast<TreeLeafHeader*>(nb.data());
        nlh->count = gc;
        nlh->extent_pages = npg;

        // Gather the grouped inputs for the family re-encode.
        LeafCoder::GroupEncodeInput gin;
        gin.count = gc;
        gin.summary_size = summary_size;
        std::vector<float> gvecs;
        if (!vecs.empty()) {
            gvecs.resize(static_cast<size_t>(gc) * dim);
            for (uint32_t i = 0; i < gc; ++i)
                std::copy_n(vecs.data() + static_cast<size_t>(group[i]) * dim,
                            dim, gvecs.begin() + static_cast<size_t>(i) * dim);
            gin.vecs = gvecs.data();
        }
        std::vector<uint8_t> gcodes;
        if (!codes.empty()) {
            gcodes.resize(static_cast<size_t>(gc) * code_size);
            for (uint32_t i = 0; i < gc; ++i)
                std::memcpy(gcodes.data() + static_cast<size_t>(i) * code_size,
                            codes.data() +
                                static_cast<size_t>(group[i]) * code_size,
                            code_size);
            gin.src_codes = gcodes.data();
        }
        std::vector<float16_t> gbiases;
        if (!old_biases.empty()) {
            gbiases.resize(gc);
            for (uint32_t i = 0; i < gc; ++i) gbiases[i] = old_biases[group[i]];
            gin.src_biases = gbiases.data();
        }
        std::vector<RowId> grids(gc);
        for (uint32_t i = 0; i < gc; ++i) grids[i] = old_rids[group[i]];
        gin.row_ids = grids.data();

        // Family-owned re-encode: per-leaf state + codes (+ biases) +
        // row_ids. Local families refit levels / retrain the codebook here;
        // global families re-interleave; scalar_lm flat-copies.
        coder_->encode_group(gin, nb.data());

        // Filter column data (after row_ids) at the coder's geometry offset.
        const LeafGeometry new_geo = coder_->geometry(nlh);
        if (has_filter && fcb > 0) {
            nlh->filter_columns_offset = new_geo.filter_offset;
            write_filter_columns(nb.data() + new_geo.filter_offset, gc,
                                 manifest_.schema, group_filter_cols);
        } else {
            nlh->filter_columns_offset = 0;
        }
        nlh->header_crc = header_crc(nlh, offsetof(TreeLeafHeader, header_crc));

        (void)centroid;  // routing centroids are applied by the caller
        const PageId page = alloc.alloc_extent(file_, npg);
        file_.write_pages(page, npg, nb.data());
        return {page, npg};
    };

    // 8. Write the two new leaves.
    auto entry0 = write_leaf_from_group(plan.group0, plan.cent0_fp16);
    auto entry1 = write_leaf_from_group(plan.group1, plan.cent1_fp16);

    // 9. Free the old leaf extent.
    alloc.free_extent(file_, leaf_table_[leaf_id].page, leaf_table_[leaf_id].pages);

    // 10. Update the leaf table: entry[leaf_id] = leaf0, new entry = leaf1.
    uint32_t new_leaf_id = static_cast<uint32_t>(leaf_table_.size());
    leaf_table_[leaf_id] = entry0;
    leaf_table_.push_back(entry1);

    // 11. Update the parent node: replace the old child entry with leaf0,
    //     add a new child entry for leaf1.
    add_child_to_parent_(leaf_id, new_leaf_id, plan.cent0_fp16,
                          plan.cent1_fp16,
                         entry1.pages, alloc);

    spdlog::info("[sextant] split_leaf_: leaf {} (count={}) → leaf {} (count={}) "
                 "+ leaf {} (count={})",
                 leaf_id, count, leaf_id, plan.group0.size(), new_leaf_id,
                  plan.group1.size());

    return new_leaf_id;
}

// ===========================================================================
// add_child_to_parent_: update the parent node after a leaf split.
//
// For depth=1: the parent is the root node. The old child entry (pointing
// to leaf_id) is updated with the new centroid/pages, and a new child entry
// is appended for new_leaf_id.
//
// For depth≥2: the parent is the L2 internal node that contains leaf_id.
// We find it by scanning root children → L2 nodes, looking for the child
// entry whose child_page == leaf_id.
//
// The parent node may need to grow (new n_children → larger extent).
// ===========================================================================

void IVFTreeIndex::add_child_to_parent_(uint32_t leaf_id, uint32_t new_leaf_id,
                                          const std::vector<float16_t>& cent0_fp16,
                                          const std::vector<float16_t>& cent1_fp16,
                                          uint32_t new_leaf_pages,
                                          PageAllocator& alloc) {
    const uint16_t dim = manifest_.dim;
    const uint32_t summary_size = manifest_.summary_size;
    const uint32_t cesize = child_entry_size(dim, summary_size);

    // --- Find the parent node and the child slot pointing to leaf_id ---
    //
    // depth=1: parent is the root.
    // depth=2: parent is an L2 node (root child). We scan root children.
    // depth=3: parent is an L2 node under an L1 node. We descend root → L1 → L2.
    //
    // We also track the grandparent (the node pointing to the parent) so we
    // can fix up the pointer if the parent node relocates (grows beyond its
    // current page extent).

    PageId parent_page = kInvalidPage;
    uint32_t parent_pages = 0;
    uint32_t child_slot = UINT32_MAX;

    // Grandparent info: the node whose child entry points to the parent.
    // For depth=1: no grandparent (parent is root, root never relocates).
    // For depth=2: grandparent is the root.
    // For depth=3: grandparent is the L1 node; we also track the root child
    // slot pointing to the L1 node so we can re-read it if the L1 relocates.
    PageId grandparent_page = kInvalidPage;
    uint32_t grandparent_pages = 0;
    uint32_t grandparent_slot = UINT32_MAX;  // child slot in grandparent → parent

    // Node scan helper: read via the FILE, not mmap_base_. Splits in this
    // insert_batch may already have relocated nodes (and truncated/rewritten
    // the file); the build-time mmap is stale past the original EOF and old
    // pages may have been freed and rewritten — walking it reads garbage
    // headers (measured: depth-3 split cascades segfaulted on stale L1/L2).
    std::vector<uint8_t> node_buf_a, node_buf_b, node_buf_c;
    auto read_node = [&](PageId page, uint32_t pages,
                         std::vector<uint8_t>& buf) -> const uint8_t* {
        buf.resize(static_cast<size_t>(pages) * kPageSize);
        file_.read_pages(page, pages, buf.data());
        return buf.data();
    };

    if (manifest_.depth == 1) {
        parent_page = superblock_.root_node_page();
        parent_pages = superblock_.root_node_pages();
        const uint8_t* root_ptr = read_node(parent_page, parent_pages,
                                             node_buf_a);
        const auto* rh = reinterpret_cast<const TreeNodeHeader*>(root_ptr);
        const uint8_t* p = root_ptr + sizeof(TreeNodeHeader);
        for (uint32_t i = 0; i < rh->n_children; ++i) {
            const auto* ce = reinterpret_cast<const ChildEntry*>(p);
            if (ce->is_leaf && ce->child_page == leaf_id) {
                child_slot = i;
                break;
            }
            p += cesize;
        }
    } else if (manifest_.depth == 2) {
        // Root → L2 nodes → leaves. Grandparent is the root.
        // NOTE: the root is re-read from FILE here, NOT from root_children_:
        // earlier splits in the same insert_batch may have relocated L2
        // nodes (and the root's child entries with them), and the
        // in-memory root_children_ only refreshes at commit/remap_. A
        // stale scan read freed-and-reused pages, failed to find the
        // parent, and ORPHANED the split's new leaf — allocated in the
        // bitmap and leaf table but unreachable from the tree (measured:
        // ~2.7k dead pages per 100k-vector churned insert).
        grandparent_page = superblock_.root_node_page();
        grandparent_pages = superblock_.root_node_pages();
        const uint8_t* root_ptr = read_node(grandparent_page,
                                            grandparent_pages, node_buf_c);
        const auto* rh = reinterpret_cast<const TreeNodeHeader*>(root_ptr);
        const uint8_t* rp = root_ptr + sizeof(TreeNodeHeader);
        for (uint32_t c = 0; c < rh->n_children; ++c, rp += cesize) {
            const auto* rce = reinterpret_cast<const ChildEntry*>(rp);
            if (rce->is_leaf || rce->child_page == kInvalidPage) continue;
            const uint8_t* node_ptr = read_node(rce->child_page,
                                                rce->child_pages,
                                                node_buf_a);
            const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
            const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);
            const uint32_t n_ch = nh->n_children;
            for (uint32_t j = 0; j < n_ch; ++j) {
                const auto* ce = reinterpret_cast<const ChildEntry*>(p);
                if (ce->is_leaf && ce->child_page == leaf_id) {
                    parent_page = rce->child_page;
                    parent_pages = static_cast<uint32_t>(rce->child_pages);
                    child_slot = j;
                    grandparent_slot = c;  // root child c → this L2 node
                    break;
                }
                p += cesize;
            }
            if (parent_page != kInvalidPage) break;
        }
    } else {
        // depth=3: Root → L1 nodes → L2 nodes → leaves.
        // Scan root children (L1 nodes), descend into each L1 to find the L2
        // that contains leaf_id.
        for (uint32_t c = 0; c < root_children_.size(); ++c) {
            const auto& rc = root_children_[c];
            if (rc.is_leaf || rc.page == kInvalidPage) continue;
            const uint8_t* l1_ptr = read_node(rc.page, rc.pages, node_buf_a);
            const auto* l1h = reinterpret_cast<const TreeNodeHeader*>(l1_ptr);
            const uint8_t* lp = l1_ptr + sizeof(TreeNodeHeader);
            const uint32_t l1_n = l1h->n_children;
            for (uint32_t j = 0; j < l1_n; ++j) {
                const auto* l1ce = reinterpret_cast<const ChildEntry*>(lp);
                if (l1ce->is_leaf || l1ce->child_page == kInvalidPage) {
                    lp += cesize;
                    continue;
                }
                // l1ce->child_page is an L2 node. Scan its children for leaf_id.
                const uint8_t* l2_ptr = read_node(
                    l1ce->child_page, l1ce->child_pages, node_buf_b);
                const auto* l2h =
                    reinterpret_cast<const TreeNodeHeader*>(l2_ptr);
                const uint8_t* p2 = l2_ptr + sizeof(TreeNodeHeader);
                const uint32_t l2_n = l2h->n_children;
                for (uint32_t k = 0; k < l2_n; ++k) {
                    const auto* ce = reinterpret_cast<const ChildEntry*>(p2);
                    if (ce->is_leaf && ce->child_page == leaf_id) {
                        parent_page = l1ce->child_page;
                        parent_pages = static_cast<uint32_t>(l1ce->child_pages);
                        child_slot = k;
                        // Grandparent is the L1 node.
                        grandparent_page = rc.page;
                        grandparent_pages = static_cast<uint32_t>(rc.pages);
                        grandparent_slot = j;  // L1 child j → this L2 node
                        break;
                    }
                    p2 += cesize;
                }
                if (parent_page != kInvalidPage) break;
                lp += cesize;
            }
            if (parent_page != kInvalidPage) break;
        }
    }

    if (parent_page == kInvalidPage || child_slot == UINT32_MAX) {
        spdlog::error("[sextant] add_child_to_parent_: could not find parent "
                      "for leaf_id={}", leaf_id);
        return;
    }

    // --- Read the parent node, update child entry, append new child ---
    std::vector<uint8_t> buf(static_cast<size_t>(parent_pages) * kPageSize);
    file_.read_pages(parent_page, parent_pages, buf.data());
    auto* nh = reinterpret_cast<TreeNodeHeader*>(buf.data());
    const uint32_t old_n_children = static_cast<uint32_t>(nh->n_children);
    const uint32_t new_n_children = old_n_children + 1;

    // Update the existing child entry (leaf_id) with new pages + post-split centroid.
    {
        uint8_t* p = buf.data() + sizeof(TreeNodeHeader) + child_slot * cesize;
        auto* ce = reinterpret_cast<ChildEntry*>(p);
        ce->child_pages = leaf_table_[leaf_id].pages;
        float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
        std::memcpy(cent, cent0_fp16.data(), dim * sizeof(float16_t));
    }

    const uint32_t new_npg = node_extent_pages(dim, new_n_children, summary_size);

    std::vector<uint8_t> nb(static_cast<size_t>(new_npg) * kPageSize, 0);
    std::memcpy(nb.data(), buf.data(),
                sizeof(TreeNodeHeader) + old_n_children * cesize);

    // Append the new child entry for new_leaf_id.
    {
        uint8_t* p = nb.data() + sizeof(TreeNodeHeader) +
                      static_cast<uint64_t>(old_n_children) * cesize;
        auto* ce = reinterpret_cast<ChildEntry*>(p);
        ce->child_page = new_leaf_id;
        ce->child_pages = new_leaf_pages;
        ce->is_leaf = 1;
        float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
        std::memcpy(cent, cent1_fp16.data(), dim * sizeof(float16_t));
    }

    auto* new_nh = reinterpret_cast<TreeNodeHeader*>(nb.data());
    new_nh->n_children = new_n_children;
    new_nh->extent_pages = new_npg;
    new_nh->header_crc = header_crc(new_nh, offsetof(TreeNodeHeader, header_crc));

    // --- Write the parent node. If it grew beyond its extent, relocate ---
    if (new_npg <= parent_pages) {
        // Fits in-place.
        file_.write_pages(parent_page, new_npg, nb.data());
    } else {
        // Parent needs a new (larger) extent. Allocate, write, free old.
        const PageId new_page = alloc.alloc_extent(file_, new_npg);
        file_.write_pages(new_page, new_npg, nb.data());
        alloc.free_extent(file_, parent_page, parent_pages);

        // Fix up the grandparent's child pointer to the new page.
        if (manifest_.depth == 1) {
            // Parent is the root itself.
            superblock_.set_root(new_page, new_npg);
        } else {
            // Update the grandparent node's child entry: change child_page
            // from parent_page → new_page, child_pages → new_npg.
            update_internal_child_(grandparent_page, grandparent_pages,
                                   grandparent_slot, new_page, new_npg, alloc);
        }
    }
}

// ===========================================================================
// update_internal_child_: update one child entry in an internal node.
//
// Used by add_child_to_parent_ when the parent node relocates (grows beyond
// its current extent). Reads the grandparent node, updates the child_page and
// child_pages fields of the specified slot, and writes it back. If the
// grandparent itself needs to grow (extremely unlikely from a single pointer
// width change — child_pages is the same width), it relocates too.
//
// For depth=2: grandparent is the root node.
// For depth=3: grandparent is the L1 node.
// ===========================================================================

void IVFTreeIndex::update_internal_child_(PageId node_page, uint32_t node_pages,
                                            uint32_t slot,
                                            PageId new_child_page,
                                            uint32_t new_child_pages,
                                            PageAllocator& alloc) {
    const uint16_t dim = manifest_.dim;
    const uint32_t summary_size = manifest_.summary_size;
    const uint32_t cesize = child_entry_size(dim, summary_size);

    std::vector<uint8_t> buf(static_cast<size_t>(node_pages) * kPageSize);
    file_.read_pages(node_page, node_pages, buf.data());

    uint8_t* p = buf.data() + sizeof(TreeNodeHeader) + slot * cesize;
    auto* ce = reinterpret_cast<ChildEntry*>(p);
    ce->child_page = new_child_page;
    ce->child_pages = new_child_pages;

    // Recompute header CRC (n_children unchanged, but CRC covers the child area
    // only when summary_size > 0; safe to always recompute).
    auto* nh = reinterpret_cast<TreeNodeHeader*>(buf.data());
    nh->header_crc = header_crc(nh, offsetof(TreeNodeHeader, header_crc));

    file_.write_pages(node_page, node_pages, buf.data());
}

// ===========================================================================
// insert_batch
// ===========================================================================

void IVFTreeIndex::insert_batch(const std::vector<InsertPoint>& points) {
    if (points.empty()) return;
    if (plane_) {
        throw Error(ErrorCode::NotImplemented,
            "insert_batch not supported while a routing plane is attached "
            "(v1 planes are immutable)");
    }
    const auto t0 = std::chrono::steady_clock::now();

    const uint32_t summary_size = manifest_.summary_size;

    const bool has_filter = manifest_.schema.n_filter_columns() > 0;
    const bool has_payload = manifest_.schema.has_payload;

    // Validate filter columns in each point.
    if (has_filter) {
        for (const auto& p : points) {
            if (p.filter_values.size() != manifest_.schema.columns.size()) {
                throw Error(ErrorCode::InvalidParam,
                    "insert_batch: filter_values size mismatch (got " +
                    std::to_string(p.filter_values.size()) + ", expected " +
                    std::to_string(manifest_.schema.columns.size()) + ")");
            }
        }
    }

    // 1. Route each point to its target leaf_id.
    std::vector<uint32_t> leaf_ids(points.size());
    for (size_t i = 0; i < points.size(); ++i)
        leaf_ids[i] = route_to_leaf_id_(points[i].vector);

    // 2. Group points by leaf_id.
    std::unordered_map<uint32_t, std::vector<uint32_t>> by_leaf;
    by_leaf.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i)
        by_leaf[leaf_ids[i]].push_back(static_cast<uint32_t>(i));

    // 3. Initialize the page allocator from the superblock.
    PageAllocator alloc;
    alloc.load(file_, superblock_.alloc_bitmap_page(),
               superblock_.alloc_bitmap_pages(),
               superblock_.n_pages(),
               superblock_.free_list_head(),
               superblock_.n_free_pages());

    // 4. Process each leaf: read, grow, append, write.
    for (const auto& [leaf_id, indices] : by_leaf) {
        std::vector<uint8_t> buf = read_leaf_(leaf_id);
        auto* lh = reinterpret_cast<TreeLeafHeader*>(buf.data());
        const uint32_t old_count = static_cast<uint32_t>(lh->count);
        const uint32_t add_count = static_cast<uint32_t>(indices.size());
        const uint32_t new_count = old_count + add_count;

        // Check if we need filter column bytes. Read old filter_cols_bytes.
        const uint64_t old_filter_bytes = lh->filter_columns_offset != 0
            ? (static_cast<uint64_t>(lh->extent_pages) * kPageSize -
               coder_->extent_bytes(old_count, summary_size, 0))
            : 0;

        // Compute new extent size.
        // We need the new filter_cols_bytes. For simplicity, recompute the
        // full leaf: we rebuild filter columns by appending the new values.
        // Old filter data + new filter data.
        uint64_t new_filter_bytes = old_filter_bytes;
        if (has_filter) {
            for (uint32_t idx : indices) {
                // Compute per-vector filter bytes for this point.
                const auto& fv = points[idx].filter_values;
                for (uint32_t c = 0; c < manifest_.schema.columns.size(); ++c) {
                    const auto& col = manifest_.schema.columns[c];
                    switch (col.type) {
                        case ColumnType::Int32:
                        case ColumnType::Int64:
                        case ColumnType::Float:
                        case ColumnType::Bool:
                            new_filter_bytes += column_type_width(col.type);
                            break;
                        case ColumnType::String: {
                            const auto& fc = fv[c];
                            new_filter_bytes += 4 + 2 + 4;  // offset+length+hash
                            new_filter_bytes += fc.str_lengths[0];
                            break;
                        }
                        case ColumnType::Set: {
                            const auto& fc = fv[c];
                            const uint8_t ec = fc.set_counts[0];
                            new_filter_bytes += 1 + 4;  // count+offset
                            new_filter_bytes += static_cast<uint64_t>(ec) * 4;  // hashes
                            const uint32_t off = fc.set_offsets[0];
                            for (uint8_t e = 0; e < ec; ++e) {
                                new_filter_bytes += 2u + fc.set_elem_lengths[off + e];
                            }
                            break;
                        }
                    }
                }
            }
        }

        const uint32_t new_npg = static_cast<uint32_t>(
            (coder_->extent_bytes(new_count, summary_size, new_filter_bytes) +
             kPageSize - 1) / kPageSize);

        // Allocate the new buffer.
        std::vector<uint8_t> nb(static_cast<size_t>(new_npg) * kPageSize, 0);

        // Copy header + summary.
        std::memcpy(nb.data(), buf.data(), sizeof(TreeLeafHeader) + summary_size);

        const LeafGeometry old_geo = coder_->geometry(lh);

        // Family-owned append: copy the old code region + per-leaf state,
        // then encode + append the incoming vectors (frozen levels /
        // codebook). Row_ids, filters, payload and extent stay here.
        {
            std::vector<const float*> vec_ptrs;
            vec_ptrs.reserve(indices.size());
            for (uint32_t idx : indices)
                vec_ptrs.push_back(points[idx].vector);
            LeafCoder::AppendInput ai;
            ai.old_leaf = buf.data();
            ai.old_count = old_count;
            ai.new_count = new_count;
            ai.vecs = vec_ptrs.data();
            ai.n_vecs = static_cast<uint32_t>(indices.size());
            ai.new_leaf = nb.data();
            coder_->append_encode(ai);
        }

        // --- Copy + append row_ids (geometry-derived offsets) ---
        {
            // Set the new count first so geometry() resolves the new offsets.
            auto* nlh0 = reinterpret_cast<TreeLeafHeader*>(nb.data());
            nlh0->count = new_count;
            const LeafGeometry new_geo = coder_->geometry(nlh0);
            RowId* nrid = reinterpret_cast<RowId*>(
                nb.data() + new_geo.rowids_offset);
            std::memcpy(nrid,
                        buf.data() + old_geo.rowids_offset,
                        old_count * sizeof(RowId));
            for (uint32_t ai = 0; ai < indices.size(); ++ai)
                nrid[old_count + ai] = points[indices[ai]].row_id;
        }

        // --- Rebuild filter column data ---
        if (has_filter) {
            auto* nlh_f = reinterpret_cast<TreeLeafHeader*>(nb.data());
            const uint64_t filter_off = coder_->geometry(nlh_f).filter_offset;

            // Read existing filter data from the old buffer (if any).
            std::vector<ColumnData> all_cols;
            if (old_count > 0 && old_filter_bytes > 0) {
                const uint64_t old_filter_off = old_geo.filter_offset;
                read_filter_columns(buf.data() + old_filter_off, old_count,
                                    manifest_.schema, all_cols);
            } else {
                all_cols.assign(manifest_.schema.columns.size(), ColumnData{});
                for (uint32_t c = 0; c < manifest_.schema.columns.size(); ++c)
                    all_cols[c].type = manifest_.schema.columns[c].type;
            }

            // Append new filter values from the insert points.
            std::vector<uint32_t> new_indices(indices.size());
            std::iota(new_indices.begin(), new_indices.end(), 0);
            // Build a temporary ColumnData vector from the insert points.
            std::vector<ColumnData> new_cols(manifest_.schema.columns.size());
            for (uint32_t c = 0; c < manifest_.schema.columns.size(); ++c) {
                const auto& col = manifest_.schema.columns[c];
                new_cols[c].type = col.type;
                for (uint32_t ai = 0; ai < indices.size(); ++ai) {
                    const auto& pfc = points[indices[ai]].filter_values[c];
                    switch (col.type) {
                        case ColumnType::Int32:
                        case ColumnType::Int64:
                        case ColumnType::Float:
                        case ColumnType::Bool: {
                            const uint8_t w = column_type_width(col.type);
                            const size_t pos = new_cols[c].fixed_data.size();
                            new_cols[c].fixed_data.resize(pos + w);
                            std::memcpy(new_cols[c].fixed_data.data() + pos,
                                        pfc.fixed_data.data(), w);
                            break;
                        }
                        case ColumnType::String: {
                            const uint16_t len = pfc.str_lengths[0];
                            new_cols[c].str_offsets.push_back(
                                static_cast<uint32_t>(new_cols[c].str_data.size()));
                            new_cols[c].str_lengths.push_back(len);
                            const char* s = pfc.str_data.data() + pfc.str_offsets[0];
                            new_cols[c].str_data.insert(new_cols[c].str_data.end(),
                                                        s, s + len);
                            break;
                        }
                        case ColumnType::Set: {
                            const uint8_t ec = pfc.set_counts[0];
                            new_cols[c].set_counts.push_back(ec);
                            new_cols[c].set_offsets.push_back(
                                static_cast<uint32_t>(new_cols[c].set_elem_lengths.size()));
                            const uint32_t off = pfc.set_offsets[0];
                            for (uint8_t e = 0; e < ec; ++e)
                                new_cols[c].set_elem_lengths.push_back(pfc.set_elem_lengths[off + e]);
                            uint32_t src_byte_off = 0;
                            for (uint32_t k = 0; k < off; ++k)
                                src_byte_off += pfc.set_elem_lengths[k];
                            for (uint8_t e = 0; e < ec; ++e) {
                                const uint16_t elen = pfc.set_elem_lengths[off + e];
                                const char* edata = pfc.set_elem_data.data() + src_byte_off;
                                new_cols[c].set_elem_data.insert(
                                    new_cols[c].set_elem_data.end(), edata, edata + elen);
                                src_byte_off += elen;
                            }
                            break;
                        }
                    }
                }
            }
            append_filter_rows(all_cols, new_cols, manifest_.schema, new_indices);

            write_filter_columns(nb.data() + filter_off, new_count,
                                 manifest_.schema, all_cols);
        }

        // --- Handle payload ---
        if (has_payload) {
            // Read old payload extent, build new one with appended payloads.
            const PageId old_pp = lh->payload_extent_page;
            const uint32_t old_ppg = lh->payload_extent_pages;

            // Read old payload.
            std::vector<uint32_t> old_offsets(old_count);
            std::vector<uint32_t> old_lengths(old_count);
            std::vector<uint8_t> old_pdata;
            if (old_pp != kInvalidPage && old_ppg > 0 && old_count > 0) {
                std::vector<uint8_t> pbuf(static_cast<size_t>(old_ppg) * kPageSize);
                file_.read_pages(old_pp, old_ppg, pbuf.data());
                const uint32_t* offs = reinterpret_cast<const uint32_t*>(pbuf.data());
                const uint32_t* lens = offs + old_count;
                const uint8_t* pdata = reinterpret_cast<const uint8_t*>(lens + old_count);
                for (uint32_t i = 0; i < old_count; ++i) {
                    old_offsets[i] = offs[i];
                    old_lengths[i] = lens[i];
                }
                uint32_t total_old = 0;
                for (uint32_t i = 0; i < old_count; ++i) total_old += old_lengths[i];
                old_pdata.assign(pdata, pdata + total_old);
            }

            // Build new payload extent.
            uint64_t new_pdata_size = old_pdata.size();
            std::vector<uint32_t> new_lengths = old_lengths;
            std::vector<std::string_view> new_pld;
            for (uint32_t ai = 0; ai < indices.size(); ++ai) {
                new_pld.push_back(points[indices[ai]].payload);
                new_lengths.push_back(static_cast<uint32_t>(new_pld.back().size()));
                new_pdata_size += new_pld.back().size();
            }

            const uint64_t payload_bytes =
                static_cast<uint64_t>(new_count) * 4 +  // offsets
                static_cast<uint64_t>(new_count) * 4 +  // lengths
                new_pdata_size;
            const uint32_t new_ppg = static_cast<uint32_t>(
                (payload_bytes + kPageSize - 1) / kPageSize);

            if (new_count > 0) {
                const PageId new_pp = alloc.alloc_extent(file_, new_ppg);
                std::vector<uint8_t> pbuf(static_cast<size_t>(new_ppg) * kPageSize, 0);
                uint32_t* offs = reinterpret_cast<uint32_t*>(pbuf.data());
                uint32_t* lens = offs + new_count;
                uint8_t* pdata = reinterpret_cast<uint8_t*>(lens + new_count);
                uint32_t acc = 0;
                // Old data.
                for (uint32_t i = 0; i < old_count; ++i) {
                    offs[i] = acc;
                    lens[i] = old_lengths[i];
                    std::memcpy(pdata + acc, old_pdata.data() + old_offsets[i],
                                old_lengths[i]);
                    acc += old_lengths[i];
                }
                // New data.
                for (uint32_t ai = 0; ai < indices.size(); ++ai) {
                    const uint32_t gi = old_count + ai;
                    offs[gi] = acc;
                    lens[gi] = new_lengths[gi];
                    std::memcpy(pdata + acc, new_pld[ai].data(), new_pld[ai].size());
                    acc += static_cast<uint32_t>(new_pld[ai].size());
                }
                file_.write_pages(new_pp, new_ppg, pbuf.data());

                auto* nlh = reinterpret_cast<TreeLeafHeader*>(nb.data());
                nlh->payload_extent_page = new_pp;
                nlh->payload_extent_pages = new_ppg;

                if (old_pp != kInvalidPage)
                    alloc.free_extent(file_, old_pp, old_ppg);
            } else {
                auto* nlh = reinterpret_cast<TreeLeafHeader*>(nb.data());
                nlh->payload_extent_page = kInvalidPage;
                nlh->payload_extent_pages = 0;
            }
        }

        // --- Update leaf header ---
        auto* nlh = reinterpret_cast<TreeLeafHeader*>(nb.data());
        nlh->count = new_count;
        nlh->extent_pages = new_npg;
        nlh->filter_columns_offset = (has_filter && new_filter_bytes > 0)
            ? coder_->geometry(nlh).filter_offset : 0;
        nlh->header_crc = header_crc(nlh, offsetof(TreeLeafHeader, header_crc));

        // --- Eager summary update (min/max widen + bloom OR) ---
        if (summary_size > 0 && has_filter) {
            // For each new point, update the summary: widen min/max, set bloom bits.
            uint8_t* summary = nb.data() + leaf_filter_offset();
            // Build a single-point ColumnData for merge.
            // Actually, use merge_filter_summary: build a summary for the new
            // points and OR it in.
            std::vector<ColumnData> new_cols(manifest_.schema.columns.size());
            for (uint32_t c = 0; c < manifest_.schema.columns.size(); ++c) {
                const auto& col = manifest_.schema.columns[c];
                new_cols[c].type = col.type;
                for (uint32_t ai = 0; ai < indices.size(); ++ai) {
                    const auto& pfc = points[indices[ai]].filter_values[c];
                    switch (col.type) {
                        case ColumnType::Int32: case ColumnType::Int64:
                        case ColumnType::Float: case ColumnType::Bool: {
                            const uint8_t w = column_type_width(col.type);
                            const size_t pos = new_cols[c].fixed_data.size();
                            new_cols[c].fixed_data.resize(pos + w);
                            std::memcpy(new_cols[c].fixed_data.data() + pos,
                                        pfc.fixed_data.data(), w);
                            break;
                        }
                        case ColumnType::String: {
                            const uint16_t len = pfc.str_lengths[0];
                            new_cols[c].str_offsets.push_back(
                                static_cast<uint32_t>(new_cols[c].str_data.size()));
                            new_cols[c].str_lengths.push_back(len);
                            const char* s = pfc.str_data.data() + pfc.str_offsets[0];
                            new_cols[c].str_data.insert(new_cols[c].str_data.end(),
                                                        s, s + len);
                            break;
                        }
                        case ColumnType::Set: {
                            const uint8_t ec = pfc.set_counts[0];
                            new_cols[c].set_counts.push_back(ec);
                            new_cols[c].set_offsets.push_back(
                                static_cast<uint32_t>(new_cols[c].set_elem_lengths.size()));
                            const uint32_t off = pfc.set_offsets[0];
                            for (uint8_t e = 0; e < ec; ++e)
                                new_cols[c].set_elem_lengths.push_back(pfc.set_elem_lengths[off + e]);
                            uint32_t src_byte_off = 0;
                            for (uint32_t k = 0; k < off; ++k)
                                src_byte_off += pfc.set_elem_lengths[k];
                            for (uint8_t e = 0; e < ec; ++e) {
                                const uint16_t elen = pfc.set_elem_lengths[off + e];
                                const char* edata = pfc.set_elem_data.data() + src_byte_off;
                                new_cols[c].set_elem_data.insert(
                                    new_cols[c].set_elem_data.end(), edata, edata + elen);
                                src_byte_off += elen;
                            }
                            break;
                        }
                    }
                }
            }
            // Build summary for just the new points, then merge.
            std::vector<uint8_t> new_summary(summary_size, 0);
            write_filter_summary(new_summary.data(), summary_size,
                                 manifest_.schema,
                                 static_cast<uint32_t>(indices.size()), new_cols);
            merge_filter_summary(summary, new_summary.data(), summary_size);
        }

        write_leaf_(leaf_id, nb, new_npg, alloc);
    }

    // Logical row count grows by the inserted rows (regardless of any
    // splits). commit_mutable_ persists it to the manifest blob.
    manifest_.n_vectors += points.size();

    // 5. Commit: rewrite leaf table blob, flush bitmap, commit superblock.
    commit_mutable_(alloc);

    // 6. Split oversized leaves (count > 2×leaf_capacity).
    // Scan all leaves for oversized ones and split them. Each split adds a
    // new leaf to the table, so we re-scan until no more splits are needed.
    // The split modifies the parent node and leaf table; we commit after
    // each batch of splits.
    {
        const uint32_t leaf_cap = manifest_.leaf_capacity;
        bool any_split = false;
        bool did_split = true;
        while (did_split) {
            did_split = false;
            for (uint32_t lid = 0; lid < leaf_table_.size(); ++lid) {
                const auto& e = leaf_table_[lid];
                if (e.page == kInvalidPage) continue;
                // Read just the header to check count.
                std::vector<uint8_t> hdr(kPageSize);
                file_.read_pages(e.page, 1, hdr.data());
                const auto* lh = reinterpret_cast<const TreeLeafHeader*>(hdr.data());
                if (lh->count > 2u * leaf_cap) {
                    split_leaf_(lid, alloc);
                    any_split = true;
                    did_split = true;
                }
            }
        }
        if (any_split) {
            commit_mutable_(alloc);
        }
    }

    const double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    spdlog::info("[sextant] insert_batch: {} points into {} leaves in {:.3f}s",
                 points.size(), by_leaf.size(), dt);
#ifdef SEXTANT_DEBUG_INSERT
    for (size_t i = 0; i < points.size(); ++i)
        spdlog::info("[sextant] insert: point {} row_id={} → leaf_id={}",
                     i, points[i].row_id, leaf_ids[i]);
#endif
}

// ===========================================================================
// delete_batch
// ===========================================================================

void IVFTreeIndex::delete_batch(const std::vector<RowId>& row_ids) {
    if (row_ids.empty()) return;
    if (plane_) {
        throw Error(ErrorCode::NotImplemented,
            "delete_batch not supported while a routing plane is attached "
            "(v1 planes are immutable)");
    }
    // Payload guard: compaction below rewrites codes/row_ids/filter columns
    // but never rewrites the payload offset/len arrays, so any delete on a
    // payload-bearing index would silently desync fetch_payload. Loudly
    // refuse instead (REAL BUG found in the v1 release audit — e2e's
    // mutation tree is payload-less, which is why this was never hit).
    if (manifest_.schema.has_payload) {
        throw Error(ErrorCode::NotImplemented,
            "delete_batch not supported on payload-bearing indexes "
            "(payload extent remapping not implemented)");
    }
    const auto t0 = std::chrono::steady_clock::now();

    if (coder_->family() != CoderFamily::GlobalPq) {
        throw Error(ErrorCode::NotImplemented,
            "delete_batch not supported for quantizer '" +
            coder_->family_name() + "' (requires global PQ/PRQ codebook)");
    }
    const uint32_t summary_size = manifest_.summary_size;

    // We don't know which leaf each row_id is in. Scan all leaves to build a
    // row_id → leaf_id + slot map. For large indexes this is expensive, but
    // delete is not on the hot path. A future optimization: maintain a
    // row_id → leaf_id index (updated on insert/split).
    //
    // For now: group row_ids by leaf by scanning leaf row_id arrays.
    std::unordered_map<RowId, std::pair<uint32_t, uint32_t>> row_to_leaf_slot;
    row_to_leaf_slot.reserve(row_ids.size());
    for (RowId rid : row_ids) row_to_leaf_slot[rid] = {UINT32_MAX, UINT32_MAX};

    // Scan leaves to find slots.
    // We read each leaf's row_ids array (sequential I/O, cheap).
    std::unordered_map<uint32_t, std::vector<uint32_t>> leaf_deletes;
    for (uint32_t lid = 0; lid < leaf_table_.size(); ++lid) {
        std::vector<uint8_t> buf = read_leaf_(lid);
        const auto* lh = reinterpret_cast<const TreeLeafHeader*>(buf.data());
        const uint32_t count = static_cast<uint32_t>(lh->count);
        if (count == 0) continue;
        const RowId* rids = reinterpret_cast<const RowId*>(
            buf.data() + coder_->geometry(lh).rowids_offset);
        for (uint32_t s = 0; s < count; ++s) {
            auto it = row_to_leaf_slot.find(rids[s]);
            if (it != row_to_leaf_slot.end()) {
                it->second = {lid, s};
                leaf_deletes[lid].push_back(s);
            }
        }
    }

    // Check for not-found row_ids.
    uint32_t not_found = 0;
    for (RowId rid : row_ids)
        if (row_to_leaf_slot[rid].first == UINT32_MAX) ++not_found;
    if (not_found > 0) {
        spdlog::warn("[sextant] delete_batch: {} of {} row_ids not found",
                     not_found, row_ids.size());
    }

    if (leaf_deletes.empty()) return;

    // Logical row count: decrement by the rows actually deleted (row_ids
    // minus the not-found ones). commit_mutable_ persists it.
    manifest_.n_vectors -= (row_ids.size() - not_found);

    PageAllocator alloc;
    alloc.load(file_, superblock_.alloc_bitmap_page(),
               superblock_.alloc_bitmap_pages(),
               superblock_.n_pages(),
               superblock_.free_list_head(),
               superblock_.n_free_pages());

    // Process each leaf with deletions: swap-remove.
    for (const auto& [leaf_id, slots] : leaf_deletes) {
        std::vector<uint8_t> buf = read_leaf_(leaf_id);
        auto* lh = reinterpret_cast<TreeLeafHeader*>(buf.data());
        uint32_t count = static_cast<uint32_t>(lh->count);

        // Build a set of slots to delete for O(1) lookup.
        std::unordered_set<uint32_t> del_set(slots.begin(), slots.end());

        // Swap-remove: for each deleted slot (in reverse order), move the last
        // live slot into it. We process from the end, compacting.
        // Strategy: build a list of surviving slot indices, then rebuild the
        // leaf with only survivors. This is simpler than in-place swap-remove
        // for multiple deletions and handles code block compaction correctly.
        std::vector<uint32_t> survivors;
        survivors.reserve(count - del_set.size());
        for (uint32_t s = 0; s < count; ++s)
            if (!del_set.count(s)) survivors.push_back(s);
        const uint32_t new_count = static_cast<uint32_t>(survivors.size());
        if (new_count == count) continue;  // nothing to delete (shouldn't happen)

        const LeafGeometry old_geo = coder_->geometry(
            reinterpret_cast<const TreeLeafHeader*>(buf.data()));

        // Read old filter_cols_bytes for extent size computation.
        const uint64_t old_filter_bytes = lh->filter_columns_offset != 0
            ? (static_cast<uint64_t>(lh->extent_pages) * kPageSize
               - coder_->extent_bytes(count, summary_size, 0))
            : 0;

        const uint32_t new_npg = static_cast<uint32_t>(
            (coder_->extent_bytes(new_count, summary_size, old_filter_bytes) +
             kPageSize - 1) / kPageSize);
        std::vector<uint8_t> nb_buf(static_cast<size_t>(new_npg) * kPageSize, 0);

        // Copy header + summary, then set the new count so geometry()
        // resolves the new-layout offsets below.
        std::memcpy(nb_buf.data(), buf.data(),
                    sizeof(TreeLeafHeader) + summary_size);
        auto* nlh = reinterpret_cast<TreeLeafHeader*>(nb_buf.data());
        nlh->count = new_count;

        // --- Rebuild code blocks (family-owned nibble moves) ---
        coder_->compact(buf.data(), survivors.data(), new_count, nb_buf.data());

        // --- Rebuild row_ids ---
        const LeafGeometry new_geo = coder_->geometry(nlh);
        RowId* nrid = reinterpret_cast<RowId*>(
            nb_buf.data() + new_geo.rowids_offset);
        for (uint32_t i = 0; i < new_count; ++i)
            nrid[i] = *reinterpret_cast<const RowId*>(
                buf.data() + old_geo.rowids_offset +
                survivors[i] * sizeof(RowId));

        // --- Compact filter column data (remove deleted rows) ---
        if (old_filter_bytes > 0) {
            // Read old filter data, select survivors, write back.
            std::vector<ColumnData> old_cols;
            read_filter_columns(buf.data() + old_geo.filter_offset, count,
                                manifest_.schema, old_cols);
            auto new_cols = select_filter_rows(old_cols, manifest_.schema,
                                               survivors);
            write_filter_columns(nb_buf.data() + new_geo.filter_offset,
                                 new_count, manifest_.schema, new_cols);
        }

        // --- Update header ---
        nlh->extent_pages = new_npg;
        nlh->summary_dirty = 1;  // summary is now stale (deleted rows)
        nlh->header_crc = header_crc(nlh, offsetof(TreeLeafHeader, header_crc));

        write_leaf_(leaf_id, nb_buf, new_npg, alloc);
    }

    commit_mutable_(alloc);

    const double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    spdlog::info("[sextant] delete_batch: {} row_ids from {} leaves "
                 "({} not found) in {:.3f}s",
                 row_ids.size(), leaf_deletes.size(), not_found, dt);
}

uint64_t IVFTreeIndex::live_count() const {
    // Internal mutation diagnostics (reads every leaf header). For the
    // cheap logical row count use manifest_.n_vectors / n_vectors().
    uint64_t total = 0;
    for (uint32_t lid = 0; lid < leaf_table_.size(); ++lid) {
        const auto& e = leaf_table_[lid];
        if (e.page == kInvalidPage) continue;
        std::vector<uint8_t> buf(static_cast<size_t>(e.pages) * kPageSize);
        file_.read_pages(e.page, e.pages, buf.data());
        const auto* lh = reinterpret_cast<const TreeLeafHeader*>(buf.data());
        total += lh->count;
    }
    return total;
}

// ===========================================================================
// vacuum(): repair stale filter summaries on dirty leaves.
//
// After delete_batch marks leaves as summary_dirty (swap-remove leaves the
// summary out of date), vacuum recomputes their summaries from the surviving
// filter column data, restoring pruning effectiveness. Optionally rebuilds the
// global cardinality table by re-scanning every leaf's filter columns.
//
// Uses the same PageAllocator + commit_mutable_ + remap_ pattern as
// delete_batch, committing in batches of `batch_size` to bound RAM.
// ===========================================================================

IVFTreeIndex::VacuumResult IVFTreeIndex::vacuum(const VacuumConfig& config) {
    VacuumResult res;
    if (plane_) {
        throw Error(ErrorCode::NotImplemented,
            "vacuum not supported while a routing plane is attached "
            "(v1 planes are immutable)");
    }
    const auto t0 = std::chrono::steady_clock::now();

    const uint32_t summary_size = manifest_.summary_size;
    const bool has_filter = manifest_.schema.n_filter_columns() > 0;

    PageAllocator alloc;
    alloc.load(file_, superblock_.alloc_bitmap_page(),
               superblock_.alloc_bitmap_pages(),
               superblock_.n_pages(),
               superblock_.free_list_head(),
               superblock_.n_free_pages());

    const uint32_t batch_size = config.batch_size > 0 ? config.batch_size : 1;
    uint32_t batch_count = 0;

    // --- Pass 1: repair summaries on dirty leaves ---
    // A leaf is dirty when summary_dirty == 1. Deletes already physically
    // compacted the leaf (swap-remove), so all remaining entries are live and
    // the summary can be rebuilt from scratch from the surviving filter columns.
    for (uint32_t lid = 0; lid < leaf_table_.size(); ++lid) {
        ++res.leaves_scanned;
        if (leaf_table_[lid].page == kInvalidPage) continue;

        std::vector<uint8_t> buf = read_leaf_(lid);
        auto* lh = reinterpret_cast<TreeLeafHeader*>(buf.data());
        if (lh->summary_dirty != 1) continue;
        // No summary to repair (e.g. no filter columns); just clear the flag.
        if (summary_size == 0 || !has_filter || lh->count == 0) {
            lh->summary_dirty = 0;
            lh->next_dirty = kInvalidPage;
            lh->header_crc = header_crc(
                lh, offsetof(TreeLeafHeader, header_crc));
            write_leaf_(lid, buf, static_cast<uint32_t>(lh->extent_pages), alloc);
        } else {
            const uint32_t count = static_cast<uint32_t>(lh->count);
            auto layout = LeafFilterLayout::from_geometry(
                buf.data(), coder_->geometry(lh));
            std::vector<ColumnData> cols;
            read_filter_columns(layout.filter_base, count, manifest_.schema, cols);
            // Recompute the summary from scratch (min/max + blooms).
            write_filter_summary(buf.data() + leaf_filter_offset(),
                                 summary_size, manifest_.schema, count, cols);
            lh->summary_dirty = 0;
            lh->next_dirty = kInvalidPage;
            lh->header_crc = header_crc(
                lh, offsetof(TreeLeafHeader, header_crc));
            write_leaf_(lid, buf, static_cast<uint32_t>(lh->extent_pages), alloc);
        }

        ++res.summaries_repaired;
        if (++batch_count >= batch_size) {
            commit_mutable_(alloc);
            batch_count = 0;
            alloc.clear_free_list();
        }
    }
    if (batch_count > 0) {
        commit_mutable_(alloc);
        alloc.clear_free_list();
    }

    // --- Pass 2 (optional): rebuild the cardinality table ---
    if (config.rebuild_cardinality && has_filter) {
        // First pass: count live vectors so the table can size its histograms.
        uint64_t total_live = 0;
        for (uint32_t lid = 0; lid < leaf_table_.size(); ++lid) {
            if (leaf_table_[lid].page == kInvalidPage) continue;
            std::vector<uint8_t> buf = read_leaf_(lid);
            const auto* lh = reinterpret_cast<const TreeLeafHeader*>(buf.data());
            total_live += lh->count;
        }

        CardinalityTable new_card;
        new_card.init(manifest_.schema, total_live);
        // Preserve the built blob's per-column tracking policy; columns the
        // old blob never knew default to Off.
        new_card.copy_modes_from(card_table_);

        for (uint32_t lid = 0; lid < leaf_table_.size(); ++lid) {
            if (leaf_table_[lid].page == kInvalidPage) continue;
            std::vector<uint8_t> buf = read_leaf_(lid);
            const auto* lh = reinterpret_cast<const TreeLeafHeader*>(buf.data());
            const uint32_t count = static_cast<uint32_t>(lh->count);
            if (count == 0) continue;
            auto layout = LeafFilterLayout::from_geometry(
                buf.data(), coder_->geometry(lh));
            std::vector<ColumnData> cols;
            read_filter_columns(layout.filter_base, count, manifest_.schema, cols);

            for (uint32_t i = 0; i < count; ++i) {
                for (uint32_t c = 0; c < manifest_.schema.columns.size(); ++c) {
                    const auto& col = manifest_.schema.columns[c];
                    const auto& fc = cols[c];
                    switch (col.type) {
                        case ColumnType::String:
                            new_card.add_string(c,
                                std::string_view(fc.str_data.data() +
                                                    fc.str_offsets[i],
                                                  fc.str_lengths[i]));
                            break;
                        case ColumnType::Set:
                            new_card.add_set(c, fc.set_counts.data(),
                                             fc.set_offsets.data(),
                                             fc.set_elem_lengths.data(),
                                             fc.set_elem_data.data(), i);
                            break;
                        case ColumnType::Int32: {
                            int32_t v;
                            std::memcpy(&v, fc.fixed_data.data() +
                                              static_cast<size_t>(i) * 4, 4);
                            new_card.add_numeric(c, static_cast<double>(v));
                            break;
                        }
                        case ColumnType::Int64: {
                            int64_t v;
                            std::memcpy(&v, fc.fixed_data.data() +
                                              static_cast<size_t>(i) * 8, 8);
                            new_card.add_numeric(c, static_cast<double>(v));
                            break;
                        }
                        case ColumnType::Float: {
                            float v;
                            std::memcpy(&v, fc.fixed_data.data() +
                                              static_cast<size_t>(i) * 4, 4);
                            new_card.add_numeric(c, static_cast<double>(v));
                            break;
                        }
                        default:
                            break;
                    }
                    ++res.cardinality_entries_rebuilt;
                }
            }
        }

        // Serial-add path (no shards): decide pending `auto` columns now.
        new_card.evaluate_auto();

        // Serialize + write the new cardinality blob, then commit. The old blob
        // is freed as part of commit_mutable_ (which rewrites the leaf table +
        // flushes the bitmap). We allocate the new blob first, set the
        // superblock pointer, then commit.
        auto card_blob = new_card.serialize();
        const uint64_t card_bytes = card_blob.size();
        PageId card_page = kInvalidPage;
        uint32_t card_npg = 0;
        if (card_bytes > 0) {
            card_npg = static_cast<uint32_t>(
                (card_bytes + kPageSize - 1) / kPageSize);
            card_page = alloc.alloc_extent(file_, card_npg);
            std::vector<uint8_t> b(static_cast<size_t>(card_npg) * kPageSize, 0);
            std::memcpy(b.data(), card_blob.data(), card_bytes);
            file_.write_pages(card_page, card_npg, b.data());
        }
        // Free the old cardinality blob (if any) before swapping the pointer.
        const PageId old_card_page = superblock_.cardinality_page();
        const uint32_t old_card_pages = superblock_.cardinality_pages();
        if (old_card_page != kInvalidPage && old_card_pages > 0)
            alloc.free_extent(file_, old_card_page, old_card_pages);
        superblock_.set_cardinality(card_page, card_npg);
        // In-memory: adopt the rebuilt table immediately.
        card_table_ = std::move(new_card);

        commit_mutable_(alloc);
        alloc.clear_free_list();
    }

    res.elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    spdlog::info("[sextant] vacuum: scanned {} leaves, repaired {} summaries"
                 "{}, {:.3f}s",
                 res.leaves_scanned, res.summaries_repaired,
                 config.rebuild_cardinality
                     ? (", rebuilt " + std::to_string(res.cardinality_entries_rebuilt)
                        + " cardinality entries")
                     : std::string(),
                 res.elapsed_sec);
    return res;
}

// ===========================================================================
// defrag(): compact fragmented leaf extents into contiguous pages.
//
// Re-allocates each leaf extent freshly (alloc_extent returns the lowest
// available contiguous range) and frees the old one, so repeated churn from
// delete/split leaves the file with a compact set of live extents. Optionally
// truncates trailing free pages to shrink the file. Best-effort: internal
// fragmentation (free pages between live extents) cannot be reclaimed without a
// full file rewrite.
//
// Same PageAllocator + commit_mutable_ + remap_ pattern as delete_batch.
// ===========================================================================

IVFTreeIndex::DefragResult IVFTreeIndex::defrag(const DefragConfig& config) {
    DefragResult res;
    if (plane_) {
        throw Error(ErrorCode::NotImplemented,
            "defrag not supported while a routing plane is attached "
            "(v1 planes are immutable)");
    }
    const auto t0 = std::chrono::steady_clock::now();

    PageAllocator alloc;
    alloc.load(file_, superblock_.alloc_bitmap_page(),
               superblock_.alloc_bitmap_pages(),
               superblock_.n_pages(),
               superblock_.free_list_head(),
               superblock_.n_free_pages());

    const uint32_t batch_size = config.batch_size > 0 ? config.batch_size : 1;
    uint32_t batch_count = 0;

    const uint64_t pages_before = file_.num_pages();

    // Re-allocate each leaf extent freshly, freeing the old one. alloc_extent
    // reuses the lowest free pages, so after all leaves are relocated the live
    // extents are packed toward the front of the file.
    for (uint32_t lid = 0; lid < leaf_table_.size(); ++lid) {
        const auto& entry = leaf_table_[lid];
        if (entry.page == kInvalidPage) continue;
        const uint32_t pages = static_cast<uint32_t>(entry.pages);
        if (pages == 0) continue;

        std::vector<uint8_t> buf = read_leaf_(lid);
        const PageId new_page = alloc.alloc_extent(file_, pages);
        file_.write_pages(new_page, pages, buf.data());
        alloc.free_extent(file_, entry.page, pages);
        leaf_table_[lid] = {new_page, pages};
        ++res.leaves_relocated;

        if (++batch_count >= batch_size) {
            commit_mutable_(alloc);
            batch_count = 0;
            alloc.clear_free_list();
        }
    }
    if (batch_count > 0) {
        commit_mutable_(alloc);
        alloc.clear_free_list();
    }

    // --- Optional file shrink ---
    // Find the highest page referenced by any live extent (leaf table + all
    // superblock-tracked blobs) and truncate trailing pages beyond it.
    if (config.shrink_file) {
        PageId max_end = 0;
        auto consider = [&](PageId page, uint64_t pages) {
            if (page != kInvalidPage && pages > 0)
                max_end = std::max(max_end, page + pages);
        };
        for (const auto& e : leaf_table_) consider(e.page, e.pages);
        consider(superblock_.alloc_bitmap_page(),
                 superblock_.alloc_bitmap_pages());
        consider(superblock_.root_node_page(), superblock_.root_node_pages());
        consider(superblock_.codebook_page(), superblock_.codebook_pages());
        consider(superblock_.config_page(), superblock_.config_pages());
        consider(superblock_.pca_page(), superblock_.pca_pages());
        consider(superblock_.cardinality_page(),
                 superblock_.cardinality_pages());
        consider(superblock_.leaf_table_page(),
                 superblock_.leaf_table_pages());

        const uint64_t new_n_pages = static_cast<uint64_t>(max_end);
        const uint64_t old_n_pages = file_.num_pages();
        if (new_n_pages < old_n_pages) {
            file_.truncate(new_n_pages);
            superblock_.set_n_pages(new_n_pages);
            // Flush the (now authoritative) superblock + bitmap state. We don't
            // touch the free list: the reclaimed trailing pages were already
            // free (no live extent referenced them), so they aren't on it.
            alloc.flush_bitmap(file_);
            superblock_.commit(file_);
            file_.sync();
            remap_();
            res.pages_reclaimed = old_n_pages - new_n_pages;
        }
    }

    res.elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const uint64_t pages_after = file_.num_pages();
    spdlog::info("[sextant] defrag: relocated {} leaves, file {} → {} pages"
                 " (reclaimed {}) in {:.3f}s",
                 res.leaves_relocated, pages_before, pages_after,
                 res.pages_reclaimed, res.elapsed_sec);
    return res;
}

}  // namespace sextant::tree
