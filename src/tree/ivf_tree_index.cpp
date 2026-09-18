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
// Destructor + close
// ===========================================================================

IVFTreeIndex::~IVFTreeIndex() {
    close();
}

void IVFTreeIndex::close() {
    fd_ = -1;
    if (mmap_base_) {
        ::munmap(const_cast<uint8_t*>(mmap_base_), mmap_size_);
        mmap_base_ = nullptr;
    }
    // PageFile destructor closes the fd.
}

// ===========================================================================
// Open
// ===========================================================================

std::unique_ptr<IVFTreeIndex> IVFTreeIndex::open(const std::string& path,
                                              uint64_t leaf_cache_bytes,
                                              uint32_t cache_window_pct,
                                              uint64_t plane_cache_bytes,
                                              bool writable) {
    auto idx = std::unique_ptr<IVFTreeIndex>(new IVFTreeIndex());
    // Fail fast on a missing path in every mode: read-only opens would
    // otherwise surface a raw ENOENT (fine, but we want the path in the
    // message), and writable opens must never silently create an empty
    // tree where the caller expected an existing one.
    if (!std::filesystem::exists(path)) {
        throw Error(ErrorCode::InvalidParam,
                    "IVFTreeIndex::open: no such file: " + path);
    }
    idx->path_ = path;
    idx->file_ = PageFile(path, writable ? PageFileMode::ReadWrite
                                         : PageFileMode::ReadOnly);
    // fd for the search path's posix_fadvise prefetch calls (open-issued
    // read-ahead). Historically this was never wired — every fadvise64
    // silently returned EBADF, so all pre-2026-09 cold-search numbers ran
    // with NO kernel prefetch (pure 4KB demand paging). Found via strace
    // while investigating the cache-path I/O granularity.
    idx->fd_ = idx->file_.fd();

    // Load superblock.
    idx->superblock_.load(idx->file_);

    // Read the TOML config blob.
    if (idx->superblock_.config_page() == kInvalidPage) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFTreeIndex::open: no config blob in superblock");
    }
    const uint32_t cfg_pages = idx->superblock_.config_pages();
    std::string cfg_buf(static_cast<size_t>(cfg_pages) * kPageSize, '\0');
    idx->file_.read_pages(idx->superblock_.config_page(), cfg_pages, cfg_buf.data());
    // The config blob is page-padded with null bytes; trim to actual TOML.
    const size_t nul = cfg_buf.find('\0');
    if (nul != std::string::npos) cfg_buf.resize(nul);
    idx->manifest_ = manifest_from_toml(cfg_buf);
    idx->n_vectors_disk_ = idx->manifest_.n_vectors;

    // Construct the per-family leaf coder (the only quantizer_type
    // interpretation besides the build resolve). Local families carry no
    // global blob; their state lives in the leaves.
    {
        CoderParams cp;
        cp.dim = idx->manifest_.dim;
        cp.m4 = idx->manifest_.m4;
        cp.pq_bits = idx->manifest_.scan_pq_bits;
        cp.metric = static_cast<MetricKind>(idx->manifest_.metric);
        cp.prq_nsplits = idx->manifest_.prq_nsplits;
        idx->coder_ = make_leaf_coder(idx->manifest_.quantizer_type, cp,
                                      /*for_open=*/true);
    }

    // Load the global state blob (codebook / scalar ruler) when present.
    if (idx->coder_->has_global_state()) {
        const auto cb_page = idx->superblock_.codebook_page();
        const auto cb_pages = idx->superblock_.codebook_pages();
        if (cb_page == kInvalidPage || cb_pages == 0) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFTreeIndex::open: no codebook in superblock");
        }
        std::vector<uint8_t> blob(static_cast<size_t>(cb_pages) * kPageSize);
        idx->file_.read_pages(cb_page, cb_pages, blob.data());

        // The blob is prefixed with its serialized size (u64), then
        // page-padded. Read the size and pass only the real bytes.
        uint64_t qblob_size = 0;
        std::memcpy(&qblob_size, blob.data(), sizeof(qblob_size));
        idx->coder_->deserialize_global(blob.data() + sizeof(qblob_size),
                                        static_cast<size_t>(qblob_size));
    }

    // Load PCA routing data if present.
    if (idx->manifest_.pca_dims > 0 && idx->superblock_.pca_page() != kInvalidPage) {
        const auto pp = idx->superblock_.pca_page();
        const auto npg = idx->superblock_.pca_pages();
        std::vector<uint8_t> blob(static_cast<size_t>(npg) * kPageSize);
        idx->file_.read_pages(pp, npg, blob.data());
        const float* f = reinterpret_cast<const float*>(blob.data());
        const uint32_t pd = idx->manifest_.pca_dims;
        const uint32_t d = idx->manifest_.dim;
        const uint32_t nl = idx->manifest_.n_leaves;

        idx->pca_dims_ = pd;
        size_t off = 0;
        // Projection matrix (pd × d).
        idx->pca_proj_.assign(f + off, f + off + pd * d);
        off += pd * d;
        // Mean projection (pd).
        idx->pca_mean_proj_.assign(f + off, f + off + pd);
        off += pd;
        // Root centroids in PCA space (kr × pd).
        // For depth>=3: the root has k_l1 children (super-cluster centroids).
        // For depth<=2: the root has k_root children.
        const uint32_t kr = (idx->manifest_.depth >= 3 && idx->manifest_.k_l1 > 0)
            ? idx->manifest_.k_l1 : idx->manifest_.k_root;
        idx->pca_root_centroids_.assign(f + off, f + off + kr * pd);
        off += kr * pd;
        // Leaf centroids in PCA space (nl × pd).
        // For depth>=3 this array is unused (deeper routing is FP16), but it
        // is still present in the blob layout; read it to keep off correct.
        idx->pca_leaf_centroids_.assign(f + off, f + off + nl * pd);

        spdlog::info("[sextant] IVFTreeIndex: PCA routing enabled ({} dims)",
                     pd);
    }

    // Load the cardinality table (Phase D) if present. deserialize() reads
    // n_vectors_ from the blob header, so no separate reconstruction is needed.
    if (idx->superblock_.cardinality_page() != kInvalidPage &&
        idx->superblock_.cardinality_pages() > 0) {
        const auto cp = idx->superblock_.cardinality_page();
        const auto cpg = idx->superblock_.cardinality_pages();
        std::vector<uint8_t> blob(static_cast<size_t>(cpg) * kPageSize);
        idx->file_.read_pages(cp, cpg, blob.data());
        idx->card_table_.deserialize(blob.data(),
            static_cast<size_t>(cpg) * kPageSize);
    }

    // Load the leaf extent table (indirection for mutable leaf extents).
    // Mandatory: every tree built by the current build path writes it.
    if (idx->superblock_.leaf_table_page() == kInvalidPage ||
        idx->superblock_.leaf_table_pages() == 0) {
        throw Error(ErrorCode::CorruptIndex,
            "tree index is missing the leaf extent table");
    }
    {
        const auto ltp = idx->superblock_.leaf_table_page();
        const auto ltpg = idx->superblock_.leaf_table_pages();
        std::vector<uint8_t> blob(static_cast<size_t>(ltpg) * kPageSize);
        idx->file_.read_pages(ltp, ltpg, blob.data());
        // First 8 bytes = number of entries.
        uint64_t n_entries = 0;
        std::memcpy(&n_entries, blob.data(), 8);
        idx->leaf_table_.resize(n_entries);
        std::memcpy(idx->leaf_table_.data(), blob.data() + 8,
                    n_entries * sizeof(LeafTableEntry));
    }

    // mmap the file read-only for search.
    idx->mmap_size_ = idx->file_.num_pages() * kPageSize;
    void* addr = ::mmap(nullptr, idx->mmap_size_, PROT_READ, MAP_SHARED,
                        idx->file_.fd(), 0);
    if (addr == MAP_FAILED) {
        throw Error(ErrorCode::IoError,
                    "IVFTreeIndex::open: mmap failed on '" + path + "': " +
                        std::strerror(errno));
    }
    idx->mmap_base_ = static_cast<const uint8_t*>(addr);

    // Load the routing plane if present (optional extent, "0 pages =
    // absent"). Blocks point straight into the mmap — zero copies.
    if (idx->superblock_.plane_page() != kInvalidPage &&
        idx->superblock_.plane_pages() > 0) {
        const uint8_t* pb = idx->mmap_base_ +
            static_cast<uint64_t>(idx->superblock_.plane_page()) * kPageSize;
        const size_t plen = static_cast<size_t>(
            idx->superblock_.plane_pages()) * kPageSize;
        idx->plane_ = PlaneIndex::parse(pb, plen);
        if (!idx->plane_) {
            throw Error(ErrorCode::CorruptIndex,
                "tree index has a corrupt/unsupported routing plane");
        }
        idx->leaf_plane_offset_.reserve(idx->leaf_table_.size());
        std::vector<uint32_t> counts;
        counts.reserve(idx->leaf_table_.size());
        for (const auto& e : idx->leaf_table_) {
            uint32_t count = 0;
            uint32_t poff = 0;
            if (e.page != kInvalidPage) {
                const auto* lh =
                    reinterpret_cast<const TreeLeafHeader*>(
                        idx->mmap_base_ +
                        static_cast<uint64_t>(e.page) * kPageSize);
                count = static_cast<uint32_t>(lh->count);
                poff = lh->plane_offset;
            }
            counts.push_back(count);
            idx->leaf_plane_offset_.push_back(poff);
        }
        idx->plane_->bind(counts);
        spdlog::info("[sextant] IVFTreeIndex: routing plane attached "
                     "(enc={} rank={} {} B/vec)",
                     static_cast<unsigned>(idx->plane_->meta().encoding),
                     idx->plane_->meta().rank,
                     idx->plane_->meta().bytes_per_vec());
    }

    // Parse the root node from the mmap.
    idx->load_root_from_mmap();

    // Optional engine-owned leaf cache (see open() doc). Sized at open so
    // the DRAM budget is an explicit, reportable number.
    if (leaf_cache_bytes > 0) {
        constexpr uint32_t kLeafCacheShards = 16;
        idx->leaf_cache_ = std::make_unique<LeafExtentCache>(
            leaf_cache_bytes, kLeafCacheShards, idx->file_.fd(),
            cache_window_pct);
        idx->leaf_cache_->set_expected_entries(idx->manifest_.n_leaves);
        spdlog::info("[sextant] IVFTreeIndex: leaf cache {} bytes ({} shards)",
                     leaf_cache_bytes, kLeafCacheShards);
    }

    // Independent cache for plane row blocks (routing tier). Keyed by
    // the plane extent's page ids — disjoint from leaf extent pages, so
    // it never competes with the leaf budget. Same W-TinyLFU policy.
    // v2 layouts need no plane cache: the rows live inside the leaf
    // extents, so the leaf cache already covers them.
    if (plane_cache_bytes > 0 && idx->plane_ && !idx->plane_->v2_layout()) {
        constexpr uint32_t kPlaneCacheShards = 16;
        idx->plane_cache_ = std::make_unique<LeafExtentCache>(
            plane_cache_bytes, kPlaneCacheShards, idx->file_.fd(),
            cache_window_pct);
        idx->plane_cache_->set_expected_entries(idx->manifest_.n_leaves);
        spdlog::info("[sextant] IVFTreeIndex: plane cache {} bytes",
                     plane_cache_bytes);
    }

    // If PCA routing: build leaf_id mapping for depth=2 trees.
    // leaves were written during build). At search time, we need to map from
    // (root_child, local_child) → global_leaf_id.
    if (idx->pca_dims_ > 0 && idx->manifest_.depth == 2) {
        // Build pca_leaf_base_ indexed by root child ID (not sequential).
        // Empty children get a sentinel (UINT32_MAX).
        idx->pca_leaf_base_.assign(idx->root_children_.size(), UINT64_MAX);
        uint32_t global_id = 0;
        for (uint32_t c = 0; c < idx->root_children_.size(); ++c) {
            const auto& rc = idx->root_children_[c];
            if (rc.is_leaf || rc.page == kInvalidPage) continue;
            const uint8_t* node_ptr = idx->mmap_base_ +
                static_cast<uint64_t>(rc.page) * kPageSize;
            const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
            idx->pca_leaf_base_[c] = global_id;
            global_id += nh->n_children;
        }
    }

    spdlog::info("[sextant] IVFTreeIndex: opened '{}' (depth={}, k_root={}, "
                 "n_leaves={}, dim={}, m4={}, quantizer={})",
                 path, idx->manifest_.depth, idx->manifest_.k_root,
                 idx->manifest_.n_leaves, idx->manifest_.dim, idx->manifest_.m4,
                 idx->manifest_.quantizer_type);

    return idx;
}

void IVFTreeIndex::load_root_from_mmap() {
    const auto root_page = superblock_.root_node_page();
    if (root_page == kInvalidPage) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFTreeIndex: superblock has no root node");
    }
    const uint8_t* root_ptr = mmap_base_ +
        static_cast<uint64_t>(root_page) * kPageSize;
    root_header_ = reinterpret_cast<const TreeNodeHeader*>(root_ptr);
    if (root_header_->magic != kTreeNodeMagic) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFTreeIndex: root node magic mismatch");
    }

    const uint16_t dim = manifest_.dim;
    const uint32_t summary_size = manifest_.summary_size;
    const uint32_t cesize = child_entry_size(dim, summary_size);
    root_children_.resize(root_header_->n_children);
    const uint8_t* p = root_ptr + sizeof(TreeNodeHeader);
    for (uint32_t i = 0; i < root_header_->n_children; ++i) {
        const auto* ce = reinterpret_cast<const ChildEntry*>(p);
        // is_leaf children store a leaf_id (index into leaf_table_), not a
        // direct page pointer. Resolve it to the physical page here so the
        // search path is unchanged.
        PageId leaf_id = ce->child_page;
        PageId page = leaf_id;
        uint64_t pages = ce->child_pages;
        if (ce->is_leaf) {
            page = leaf_table_[leaf_id].page;
            pages = leaf_table_[leaf_id].pages;
        }
        root_children_[i].page = page;
        root_children_[i].pages = pages;
        root_children_[i].is_leaf = ce->is_leaf;
        root_children_[i].centroid = reinterpret_cast<const float16_t*>(
            p + sizeof(ChildEntry));
        p += cesize;
    }
}

const uint8_t* IVFTreeIndex::pin_leaf_(PageId page, uint32_t pages,
                                       LeafExtentCache::Handle& handle,
                                       bool* was_hit) const {
    handle.entry = nullptr;
    const uint8_t* mmap_ptr =
        mmap_base_ + static_cast<uint64_t>(page) * kPageSize;
    if (!leaf_cache_) {
        if (was_hit) *was_hit = false;
        return mmap_ptr;
    }
    bool hit = false;
    uint64_t filled = 0;
    const uint8_t* p = leaf_cache_->pin(page, pages, handle, mmap_ptr,
                                        &hit, &filled);
    search_stats_.on_cache_op(hit, filled);
    if (was_hit) *was_hit = hit;
    return p;
}

void IVFTreeIndex::release_leaf_pins_(
    std::vector<LeafExtentCache::Handle>& pins) const {
    if (leaf_cache_) {
        for (auto& h : pins) leaf_cache_->unpin(h);
    }
    pins.clear();
}

void IVFTreeIndex::plane_cache_hitmiss(uint64_t& hits,
                                       uint64_t& misses) const {
    hits = misses = 0;
    if (!plane_cache_) return;
    const auto st = plane_cache_->stats();
    hits = st.hits;
    misses = st.misses;
}

std::vector<IVFTreeIndex::DebugLeafInfo> IVFTreeIndex::debug_leaf_info()
    const {
    std::vector<DebugLeafInfo> out;
    out.reserve(leaf_table_.size());
    for (const auto& e : leaf_table_) {
        uint32_t count = 0;
        if (e.page != kInvalidPage) {
            const auto* lh = reinterpret_cast<const TreeLeafHeader*>(
                mmap_base_ + static_cast<uint64_t>(e.page) * kPageSize);
            count = static_cast<uint32_t>(lh->count);
        }
        out.push_back({e.page, e.pages, count});
    }
    return out;
}

std::vector<RowId> IVFTreeIndex::debug_leaf_row_ids(uint32_t leaf_id) const {
    if (leaf_id >= leaf_table_.size()) return {};
    const auto& e = leaf_table_[leaf_id];
    if (e.page == kInvalidPage) return {};
    const uint8_t* leaf_ptr = mmap_base_ +
        static_cast<uint64_t>(e.page) * kPageSize;
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
    const uint64_t count = lh->count;
    if (count == 0) return {};

    // row_ids sit after the codes region; the offsets are family-owned.
    const uint8_t* rids = leaf_ptr + coder_->geometry(lh).rowids_offset;
    std::vector<RowId> out(count);
    for (uint64_t i = 0; i < count; ++i) out[i] = load_rowid(rids, i);
    return out;
}

// ===========================================================================
// Payload fetch (Phase E) — O(1) read from the leaf's payload extent.
// ===========================================================================

std::string_view IVFTreeIndex::fetch_payload(const uint8_t* leaf_ptr,
                                              uint32_t slot) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
    if (lh->payload_extent_page == kInvalidPage || lh->payload_extent_pages == 0)
        return {};  // no payload extent for this leaf
    if (slot >= lh->count)
        return {};  // out of range

    const uint8_t* payload_base = mmap_base_ +
        static_cast<uint64_t>(lh->payload_extent_page) * kPageSize;
    const uint32_t* offsets = reinterpret_cast<const uint32_t*>(payload_base);
    const uint32_t* lengths = offsets + lh->count;
    const uint8_t* data = reinterpret_cast<const uint8_t*>(lengths + lh->count);

    const uint32_t off = offsets[slot];
    const uint32_t len = lengths[slot];
    return std::string_view(reinterpret_cast<const char*>(data + off), len);
}

bool IVFTreeIndex::fetch_vector(const uint8_t* leaf_ptr, uint32_t slot,
                                float* out) const {
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
    if (slot >= lh->count) return false;
    coder_->decode_one(leaf_ptr, slot, out);
    return true;
}

}  // namespace sextant::tree
