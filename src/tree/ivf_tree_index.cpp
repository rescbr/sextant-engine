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
// Helper functions
// ===========================================================================

/// Renormalize `vec` (dim floats) to unit length. No-op if already zero.
/// Uses SIMD for the norm computation (simd::dot_f32).
inline void renormalize_unit(float* vec, uint16_t dim) {
    const float norm_sq = simd::dot_f32(vec, vec, dim);
    if (norm_sq > 0.0f) {
        const float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (uint16_t d = 0; d < dim; ++d) vec[d] *= inv_norm;
    }
}

/// Compute centroid from accumulator `sum` (dim doubles) with `count` vectors.
/// For InnerProduct metric, renormalizes to unit length (spherical k-means).
/// For L2Sq metric, computes arithmetic mean (no renormalization).
/// Writes result to `out` (dim floats).
inline void compute_centroid_spherical(const double* sum, uint64_t count,
                                       uint16_t dim, MetricKind metric,
                                       float* out) {
    const double inv = 1.0 / static_cast<double>(count);
    for (uint16_t d = 0; d < dim; ++d) {
        out[d] = static_cast<float>(sum[d] * inv);
    }
    if (metric == MetricKind::InnerProduct) {
        renormalize_unit(out, dim);
    }
}

/// Working data for building a root child entry (build-time only).
struct RootChildData {
    std::vector<float16_t> centroid;
    PageId page = kInvalidPage;
    uint32_t pages = 0;
    uint16_t is_leaf = 0;
};

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
                                              uint32_t cache_window_pct) {
    auto idx = std::unique_ptr<IVFTreeIndex>(new IVFTreeIndex());
    idx->path_ = path;
    idx->file_ = PageFile(path);
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
        std::vector<uint32_t> counts;
        counts.reserve(idx->leaf_table_.size());
        for (const auto& e : idx->leaf_table_) {
            uint32_t count = 0;
            if (e.page != kInvalidPage) {
                const auto* lh =
                    reinterpret_cast<const TreeLeafHeader*>(
                        idx->mmap_base_ +
                        static_cast<uint64_t>(e.page) * kPageSize);
                count = static_cast<uint32_t>(lh->count);
            }
            counts.push_back(count);
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
// build_streaming_pca context + helpers
//
// build_streaming_pca was a 1401-line monolith. It is now factored into a
// plain TreeBuildContext data struct (holding all cross-phase state) and five
// free helper functions that operate on it. The struct lives in the anonymous
// namespace so it does not pollute the class header with build-only internals.
// ===========================================================================
namespace {

/// Per-leaf metadata captured at flush time and consumed by the tree-write
/// phase (page, page count, and the leaf centroid in original FP16 space).
struct LeafMeta {
    PageId page;
    uint32_t pages;
    std::vector<float16_t> centroid;  // dim FP16 values
};

/// All shared state for the streaming PCA build, threaded through the helper
/// phases. Deliberately a plain data struct (no methods); the helpers below
/// mutate it directly. A single constructor binds the three reference members
/// (which cannot be reassigned) and leaves every other field at its default.
struct TreeBuildContext {
    // --- Inputs ---
    VectorSource& source;
    const std::string& output_path;
    const IVFTreeIndex::BuildConfig& cfg;

    TreeBuildContext(VectorSource& src, const std::string& op,
                     const IVFTreeIndex::BuildConfig& c)
        : source(src), output_path(op), cfg(c), metrics(&src, nullptr) {}

    // --- Instrumentation: per-phase wall/CPU/RSS/source records ---
    metrics::MetricsCollector metrics;

    // --- Header / schema ---
    uint64_t n = 0;
    Dim dim = 0;
    uint32_t summary_size = 0;

    // --- Resolved build params ---
    uint16_t m4 = 0;
    uint8_t scan_bits = 0;
    uint32_t leaf_cap = 0;
    uint32_t k_root = 0;
    uint16_t depth = 2;
    uint32_t k_l1 = 0;
    uint32_t pca_dims = 0;   // working width: dim when pca_disabled
    bool pca_disabled = false;  // cfg requested 0 → full-dim fp16 routing
    MetricKind metric = MetricKind::L2Sq;
    bool has_filter = false;
    bool has_payload = false;
    uint32_t n_schema_cols = 0;

    // --- Vector source (replaces the FILE* f handle). The source is owned
    //     by the caller; the context holds a reference. reset()/next() drive
    //     all passes (train sample, Lloyd, emission). ---

    // --- Training sample (reused by Lloyd convergence checks) ---
    uint32_t train_n = 0;
    std::vector<float> sample;      // train_n × dim (raw FP32)
    std::vector<float> sample_pca;  // train_n × pca_dims

    // --- PCA state ---
    std::vector<double> mean;       // dim
    std::vector<float> mean_proj;   // pca_dims
    std::vector<float> rotation;    // pca_dims × dim (top-k rows of full R)
    float closure_epsilon = 0.0f;   // global (fallback)

    // --- Phase I: per-cluster d_eff and per-cluster closure epsilon ---
    // d_eff_c = mean gap to 2nd-nearest centroid for vectors in cluster c.
    // Larger gap = lower intrinsic dimensionality = easier to partition.
    // closure_eps_c = closure_multiplier × d_eff_c (per-cluster closure).
    std::vector<float> cluster_d_eff;       // k_root: mean gap per cluster
    std::vector<float> cluster_closure_eps; // k_root: per-cluster closure epsilon

    // --- Centroids (PCA space) ---
    std::vector<std::vector<float>> root_centroids_pca;     // k_root × pca_dims
    std::vector<uint32_t> super_group;                       // depth-3: fine_c → group
    std::vector<std::vector<float>> super_centroids_pca;    // depth-3: k_l1 × pca_dims

    // --- Quantizer ---
    // Single per-family coder: owns the quantizer + all family logic.
    std::unique_ptr<LeafCoder> coder;
    // Scalar + InnerProduct: leaves carry a per-vector fp16 IP bias
    // (||x||/||x_hat||) between codes and row_ids.
    bool has_ip_bias = false;

    // --- FP16 root centroids (original space, for tree storage) ---
    std::vector<std::vector<float16_t>> root_centroids_fp16;  // k_root × dim

    // --- Emission outputs ---
    std::vector<LeafMeta> leaf_metas;
    std::vector<std::vector<uint32_t>> root_to_leaves;  // cluster → leaf indices
    std::vector<std::vector<uint8_t>> leaf_summaries;
    uint32_t n_leaves_total = 0;

    // --- Cardinality table (Phase D) ---
    CardinalityTable card_table;
};

// ---------------------------------------------------------------------------
// Phase 1: read the source header (count/dim) and resolve all build params
// (m4, scan_bits, leaf_cap, k_root, depth, pca_dims). The source is kept open
// on ctx.source for the subsequent phases.
// ---------------------------------------------------------------------------
void resolve_build_params(TreeBuildContext& ctx) {
    const auto& cfg = ctx.cfg;

    ctx.n = ctx.source.count();
    ctx.dim = ctx.source.dim();
    if (ctx.n == 0 || ctx.dim == 0) {
        throw Error(ErrorCode::InvalidParam, "build_streaming_pca: empty");
    }
    ctx.summary_size = cfg.filter_schema.summary_size();

    spdlog::info("[sextant] build_streaming_pca: N={} dim={} → '{}'",
                 ctx.n, ctx.dim, ctx.output_path);

    // --- Phase D: initialize the global cardinality table for selectivity
    // estimation. Populated during the emission pass and serialized after all
    // tree nodes are written. No-op when there are no filter columns.
    if (!cfg.filter_schema.empty()) {
        ctx.card_table.init(cfg.filter_schema, ctx.n);
    }

    const auto& params = cfg.params;
    ctx.m4 = params.pq4_m > 0 ? params.pq4_m
                              : static_cast<uint16_t>(ctx.dim / 4);
    ctx.scan_bits = params.scan_pq_bits;
    ctx.leaf_cap = cfg.leaf_capacity > 0 ? cfg.leaf_capacity : 5000;

    // n_leaves estimate: vectors / leaf_capacity.
    const uint64_t n_leaves_est = std::max<uint64_t>(1, ctx.n / ctx.leaf_cap);

    uint32_t k_root = cfg.k_root;
    if (k_root == 0) {
        // K_root = round_pow2(n_leaves / 2).
        // Each root child holds ~2 leaves on average. With n_probe_ln typically
        // 4-8, this ensures probing is efficient (min(n_probe_ln, 2) = 2 leaves
        // per child). Higher k_root = fewer codes scanned per probe = higher QPS.
        // round_pow2 picks the nearest power of two for cache-aligned child extents.
        const uint64_t target = std::max<uint64_t>(1, n_leaves_est / 2);
        uint32_t p2 = 1;
        while (p2 * 2 <= target) p2 *= 2;
        if (p2 < (1u << 30) && (target - p2) > (p2 * 2 - target))
            p2 *= 2;  // next power of two is closer
        k_root = std::clamp(p2, 4u, 131072u);
    }
    ctx.k_root = k_root;

    // --- Depth selection ---
    // k_root is the TOTAL number of fine-grained (leaf-group) clusters. When
    // it exceeds k_root_max_depth2, a depth-2 root would be too large to
    // route efficiently, so we add an intermediate level (depth-3): the root
    // gets k_l1 children (L1 nodes), each L1 node groups k_root/k_l1 fine
    // centroids (L2 nodes → leaves). k_root stays as the fine-cluster count.
    const uint32_t k_root_max_depth2 = cfg.k_root_max_depth2;
    if (k_root > k_root_max_depth2) {
        ctx.depth = 3;
        const uint32_t target_l1_children = 256;
        uint32_t target = std::max(16u, k_root / target_l1_children);
        // round to nearest power of two (same scheme as k_root above).
        uint32_t p2 = 1;
        while (p2 * 2 <= target) p2 *= 2;
        if (p2 < (1u << 30) && (target - p2) > (p2 * 2 - target)) p2 *= 2;
        ctx.k_l1 = std::clamp(p2, 16u, 512u);
        spdlog::info("[sextant] build_streaming_pca: depth-3 (k_root={} > {}): "
                     "k_l1={} root branching", k_root, k_root_max_depth2, ctx.k_l1);
    }

    // PCA dimensions: project to this many components for routing.
    // Default: 32 (captures meaningful variance without being too large
    // for k-means to find structure). For d_eff≈2 data, even 8-16 PCs suffice.
    // pca_dims == 0 is HONORED: disables PCA routing — the build's
    // k-means then runs in FULL dimension via an identity rotation
    // (distance-preserving, so assignments equal raw-space k-means) and
    // the manifest records 0, keeping search on the full-dim fp16
    // centroid-summaries path. Previously 0 silently coerced to 32,
    // which made PCA impossible to turn off from the CLI; honoring it
    // WITHOUT the full-dim working space collapsed k-means to 0 dims
    // (15/16 empty root clusters, Depth3Split regression).
    ctx.pca_disabled = cfg.pca_dims == 0;
    ctx.pca_dims = ctx.pca_disabled
        ? ctx.dim
        : std::min(ctx.dim, cfg.pca_dims);
    ctx.metric = params.metric;

    // Construct the per-family leaf coder (the ONLY quantizer_type
    // interpretation besides open()). Scalar families normalize m4 = dim
    // and validate pq_bits inside the factory.
    CoderParams cp;
    cp.dim = static_cast<uint16_t>(ctx.dim);
    cp.m4 = ctx.m4;
    cp.pq_bits = ctx.scan_bits;
    cp.metric = params.metric;
    cp.prq_nsplits = params.prq_nsplits;
    cp.prq_beam_size = params.prq_beam_size;
    cp.prq_encode_mode = params.prq_encode_mode;
    cp.prq_icm_iters = params.prq_icm_iters;
    cp.prq_ils_iters = params.prq_ils_iters;
    cp.prq_ils_perturb = params.prq_ils_perturb;
    cp.prq_lsq_train_iters = params.prq_lsq_train_iters;
    ctx.coder = make_leaf_coder(params.quantizer_type, cp,
                                /*for_open=*/false);
    ctx.m4 = cp.m4;  // factory may normalize (scalar families)
    ctx.has_ip_bias = cp.has_ip_bias;
}

// ---------------------------------------------------------------------------
// Phase 2: sample vectors, train the PQ quantizer, compute the PCA rotation
// (mean, projection, variance explained), run k-means in PCA space for the
// root centroids, and compute the closure epsilon. The trained quantizer is
// stored on ctx.quantizer.
// ---------------------------------------------------------------------------
void train_quantizer_and_pca(TreeBuildContext& ctx) {
    const auto& cfg = ctx.cfg;
    const auto n = ctx.n;
    const Dim dim = ctx.dim;
    const uint32_t pca_dims = ctx.pca_dims;
    const uint32_t k_root = ctx.k_root;

    // --- 1. Sample + train quantizer ---
    const auto t_sample = std::chrono::steady_clock::now();
    auto m_sample = ctx.metrics.start("sample");
    // RANDOM sample, not a sequential prefix: embeddings arrive ordered
    // (topic/time-clustered fbins, streamed inserts), and a prefix-trained
    // global ruler is fitted to the wrong distribution — measured
    // dbpedia-933K scalar_uniform decoded ceiling 0.950 (20K prefix) vs
    // 0.956 (full-corpus stats). Algorithm R reservoir over next() ONLY:
    // the engine never opens source files directly (the tree is its only
    // owned artifact; seek-based file sampling would break the VectorSource
    // layering and the streaming contract). NOTE: this is a pre-pass over
    // the replayable source; the cleaner form — reservoir collected
    // incidentally during the partition pass, train at first flush — is
    // deferred to the LeafCoder build restructure (flushes interleave with
    // streaming today, and the global quantizer must exist before the
    // first encode).
    uint32_t train_n = std::min<uint64_t>(20'000, n);
    std::vector<float> sample;
    {
        ctx.source.reset();
        Chunk chunk;
        std::vector<float> reservoir(size_t(train_n) * dim);
        uint64_t seen = 0;
        std::mt19937_64 rng(42);
        while (ctx.source.next(chunk)) {
            for (uint32_t i = 0; i < chunk.count; ++i) {
                const float* v = chunk.vectors + size_t(i) * dim;
                if (seen < train_n) {
                    std::memcpy(&reservoir[size_t(seen) * dim], v,
                                size_t(dim) * sizeof(float));
                } else {
                    const uint64_t j = std::uniform_int_distribution<uint64_t>(
                        0, seen)(rng);
                    if (j < train_n)
                        std::memcpy(&reservoir[size_t(j) * dim], v,
                                    size_t(dim) * sizeof(float));
                }
                ++seen;
            }
        }
        sample = std::move(reservoir);
        ctx.source.reset();
    }

    const bool coder_has_global = ctx.coder->has_global_state();
    // Pin global training to the build's thread budget (scalar families
    // parallelize per-dim here; other families ignore it).
    ctx.coder->set_train_threads(
        ctx.cfg.num_threads ? ctx.cfg.num_threads
                            : std::thread::hardware_concurrency());
    ctx.coder->train(sample.data(), train_n);
    if (coder_has_global) {
        spdlog::info("[sextant] build_streaming_pca: trained {} in {:.2f}s",
                     ctx.coder->family_name(),
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_sample).count());
    } else {
        spdlog::info("[sextant] build_streaming_pca: {} mode, skipping "
                     "global quantizer training",
                     ctx.coder->family_name());
    }
    ctx.metrics.stop(m_sample);

    // --- 2. Compute PCA (reuse existing compute_pca_rotation_public) ---
    const auto t_pca = std::chrono::steady_clock::now();
    auto m_pca = ctx.metrics.start("pca");

    // compute_pca_rotation_public returns the full dim×dim rotation R,
    // with rows sorted by descending eigenvalue. We take the top pca_dims rows.
    // NOTE: this modifies `sample` (centers it). We re-read the sample for
    // PCA projection below using the rotation + original sample (re-read from
    // the uncentered copy we saved above? Actually compute_pca_rotation_public
    // takes samples and centers internally; it doesn't modify the input).
    // Wait — compute_pca_rotation_public does NOT modify `sample` (it reads
    // const float*). But we already centered sample at line ~1867 above in
    // the old code. Since we removed that, sample is still the raw FP32 sample.
    // Good — the function handles centering internally.
    std::vector<double> eigvals;
    std::vector<float> rotation;
    if (ctx.pca_disabled) {
        // Identity rotation: k-means works in centered full-dim space
        // (equivalent to raw-space k-means); no eigendecomposition, no
        // pca blob to store. mean_proj[k] = mean[k] falls out below.
        rotation.assign(size_t(dim) * dim, 0.0f);
        for (uint32_t d = 0; d < dim; ++d) rotation[d * dim + d] = 1.0f;
        spdlog::info("[sextant] build_streaming_pca: PCA disabled — "
                     "full-dim ({}) centroid routing", dim);
    } else {
        rotation = compute_pca_rotation_public(
            sample.data(), train_n, dim, &eigvals);
    }

    if (rotation.empty()) {
        throw Error(ErrorCode::InvalidParam,
                    "build_streaming_pca: degenerate covariance (no PCA)");
    }

    // Extract top-k rows: proj_rows[k][d] = rotation[k * dim + d].
    // These are the top-k eigenvectors (rows of R, sorted by eigenvalue).
    // Precompute mean_proj[k] = dot(proj_row_k, mean) so we can project
    // uncentered vectors: proj[k] = dot(proj_row_k, vec) - mean_proj[k].
    std::vector<double> mean(dim, 0.0);
    for (uint32_t i = 0; i < train_n; ++i)
        for (uint16_t d = 0; d < dim; ++d)
            mean[d] += sample[i * dim + d];
    for (uint16_t d = 0; d < dim; ++d) mean[d] /= train_n;

    std::vector<float> mean_proj(pca_dims);
    for (uint32_t k = 0; k < pca_dims; ++k) {
        double acc = 0.0;
        for (uint16_t d = 0; d < dim; ++d)
            acc += rotation[k * dim + d] * mean[d];
        mean_proj[k] = static_cast<float>(acc);
    }

    double var_explained = 0.0, var_total = 0.0;
    if (!ctx.pca_disabled) {
        for (uint16_t i = 0; i < dim; ++i) var_total += eigvals[i];
        for (uint32_t k = 0; k < pca_dims; ++k) var_explained += eigvals[k];
        spdlog::info("[sextant] build_streaming_pca: PCA {}→{} dims, "
                     "variance explained: {:.1f}% in {:.2f}s",
                     dim, pca_dims,
                     100.0 * var_explained / std::max(var_total, 1.0),
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_pca).count());
    }
    ctx.metrics.stop(m_pca);

    // Project the sample for k-means (parallelized across threads).
    // Each vector: pca_dims dot products of length dim.
    // With train_n=20k, dim=768, pca_dims=32 this is ~490M MACs —
    // serial it takes ~2s, parallel across all cores it's <0.3s.
    std::vector<float> sample_pca(static_cast<size_t>(train_n) * pca_dims);
    {
        const uint32_t proj_hw = cfg.num_threads > 0
            ? cfg.num_threads
            : std::max(1u, std::thread::hardware_concurrency());
        const uint32_t n_threads = std::min(proj_hw, train_n);
        std::vector<std::future<void>> futs;
        const uint32_t per = (train_n + n_threads - 1) / n_threads;
        for (uint32_t t = 0; t < n_threads; ++t) {
            const uint32_t start = t * per;
            const uint32_t end = std::min(start + per, train_n);
            if (start >= end) break;
            futs.push_back(std::async(std::launch::async,
                [&](uint32_t s, uint32_t e) {
                    for (uint32_t i = s; i < e; ++i) {
                        const float* xi = &sample[i * dim];
                        float* pi = &sample_pca[i * pca_dims];
                        for (uint32_t k = 0; k < pca_dims; ++k) {
                            pi[k] = simd::dot_f32(
                                &rotation[k * dim], xi, dim) - mean_proj[k];
                        }
                    }
                }, start, end));
        }
        for (auto& fut : futs) fut.get();
    }

    // --- 3. K-means in PCA space (root centroids) ---
    const auto t_kmeans = std::chrono::steady_clock::now();
    auto m_kmeans = ctx.metrics.start("kmeans");
    std::vector<std::vector<float>> root_centroids_pca(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        const uint32_t src = (c * train_n) / k_root;
        root_centroids_pca[c].assign(
            &sample_pca[src * pca_dims], &sample_pca[(src + 1) * pca_dims]);
    }

    std::vector<uint32_t> prev_assign(train_n, UINT32_MAX);
    for (uint32_t iter = 0; iter < 10; ++iter) {
        std::vector<std::vector<uint32_t>> assigns(k_root);
        uint32_t n_changed = 0;
        for (uint32_t i = 0; i < train_n; ++i) {
            const float* vi = &sample_pca[i * pca_dims];
            float best_d = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k_root; ++c) {
                float d = 0.0f;
                for (uint32_t k = 0; k < pca_dims; ++k) {
                    const float diff = vi[k] - root_centroids_pca[c][k];
                    d += diff * diff;
                }
                if (d < best_d) { best_d = d; best_c = c; }
            }
            assigns[best_c].push_back(i);
            if (iter > 0 && best_c != prev_assign[i]) ++n_changed;
            prev_assign[i] = best_c;
        }
        for (uint32_t c = 0; c < k_root; ++c) {
            if (assigns[c].empty()) {
                uint32_t src = (c * 7919 + 1) % train_n;
                root_centroids_pca[c].assign(
                    &sample_pca[src * pca_dims],
                    &sample_pca[(src + 1) * pca_dims]);
                continue;
            }
            std::vector<double> sum(pca_dims, 0.0);
            for (uint32_t i : assigns[c])
                for (uint32_t k = 0; k < pca_dims; ++k)
                    sum[k] += sample_pca[i * pca_dims + k];
            const double inv = 1.0 / assigns[c].size();
            for (uint32_t k = 0; k < pca_dims; ++k)
                root_centroids_pca[c][k] = static_cast<float>(sum[k] * inv);
        }
        if (iter >= 2 && iter > 0 && n_changed > 0 &&
            static_cast<double>(n_changed) / train_n < 0.01) {
            spdlog::info("[sextant] build_streaming_pca: k-means converged at "
                         "iter {} ({:.2f}% changed)", iter + 1,
                         100.0 * n_changed / train_n);
            break;
        }
        if (iter > 0)
            spdlog::info("[sextant] build_streaming_pca: k-means iter {} "
                         "changed={} ({:.1f}%)", iter + 1, n_changed,
                         100.0 * n_changed / train_n);
    }
    spdlog::info("[sextant] build_streaming_pca: k-means done in {:.2f}s",
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_kmeans).count());
    ctx.metrics.stop(m_kmeans);

    // --- 4. Compute closure epsilon in PCA space ---
    float closure_epsilon = 0.0f;
    {
        const uint32_t s = std::min<uint32_t>(4096, train_n);
        double sum_gap = 0.0;
        for (uint32_t i = 0; i < s; ++i) {
            const float* vi = &sample_pca[i * pca_dims];
            float d1 = std::numeric_limits<float>::max();
            float d2 = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < k_root; ++c) {
                float d = 0.0f;
                for (uint32_t k = 0; k < pca_dims; ++k) {
                    const float diff = vi[k] - root_centroids_pca[c][k];
                    d += diff * diff;
                }
                if (d < d1) { d2 = d1; d1 = d; }
                else if (d < d2) d2 = d;
            }
            sum_gap += (d2 - d1);
        }
        closure_epsilon = static_cast<float>(sum_gap / s *
            (cfg.closure_multiplier > 0 ? cfg.closure_multiplier : 0.15f));
    }
    spdlog::info("[sextant] build_streaming_pca: closure_eps={:.4f}",
                 closure_epsilon);

    // Commit Phase-2 state to the context.
    ctx.train_n = train_n;
    ctx.sample = std::move(sample);
    ctx.sample_pca = std::move(sample_pca);
    ctx.mean = std::move(mean);
    ctx.mean_proj = std::move(mean_proj);
    ctx.rotation = std::move(rotation);
    ctx.closure_epsilon = closure_epsilon;
    ctx.root_centroids_pca = std::move(root_centroids_pca);
}

// ---------------------------------------------------------------------------
// Phase 3: multi-pass streaming Lloyd refinement of the root centroids, then
// (for depth-3) super-clustering of the fine centroids into k_l1 groups.
// Re-reads the vector source from the start each pass; only centroids +
// accumulators are kept in memory.
// ---------------------------------------------------------------------------
void run_lloyd_refinement(TreeBuildContext& ctx) {
    const auto& cfg = ctx.cfg;
    const auto n = ctx.n;
    const Dim dim = ctx.dim;
    const uint32_t pca_dims = ctx.pca_dims;
    const uint32_t k_root = ctx.k_root;
    const uint32_t train_n = ctx.train_n;
    const auto& rotation = ctx.rotation;
    const auto& mean_proj = ctx.mean_proj;
    auto& root_centroids_pca = ctx.root_centroids_pca;

    // --- 5. Multi-pass streaming Lloyd refinement ---
    // Instead of one greedy pass, do P Lloyd passes over the file:
    // Each pass: project → assign → accumulate per-cluster sums → update centroids.
    // This is global k-means, but streaming from disk. O(1) RAM (only centroids
    // + accumulators in memory). Converges to the same fixed point as in-RAM k-means.
    //
    // After refinement, do one final emission pass: assign + closure + encode +
    // write leaves.
    const auto t_lloyd = std::chrono::steady_clock::now();
    auto m_lloyd = ctx.metrics.start("lloyd");
    const uint32_t max_lloyd_passes = cfg.max_lloyd_passes > 0 ? cfg.max_lloyd_passes : 10;

    // Per-cluster accumulators (double for numerical stability).
    std::vector<std::vector<double>> cluster_sums(k_root, std::vector<double>(pca_dims, 0.0));
    std::vector<uint64_t> cluster_counts(k_root, 0);
    std::vector<uint32_t> prev_assignment;  // for change tracking (sampled)

    for (uint32_t pass = 0; pass < max_lloyd_passes; ++pass) {
        const auto pass_t0 = std::chrono::steady_clock::now();
        // Reset accumulators.
        for (uint32_t c = 0; c < k_root; ++c) {
            std::fill(cluster_sums[c].begin(), cluster_sums[c].end(), 0.0);
            cluster_counts[c] = 0;
        }

        // Stream all vectors: project + assign + accumulate (parallel).
        ctx.source.reset();
        uint64_t vectors_done = 0;

        // Per-thread accumulators (avoid false sharing: pad to cache line).
        const uint32_t hw = cfg.num_threads > 0
            ? cfg.num_threads
            : std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::vector<double>> t_sums(
            hw, std::vector<double>(k_root * pca_dims, 0.0));
        std::vector<std::vector<uint64_t>> t_counts(hw, std::vector<uint64_t>(k_root, 0));

        // Centroid norms are constant per pass: compute once, share
        // across threads and chunks (was per-thread per-chunk).
        // dist = |proj|² - 2·proj·centroid + |centroid|²; |proj|² is
        // constant across centroids (skipped), so
        // argmin dist = argmin(cent_norms[c] - 2·proj·centroid).
        std::vector<float> cent_norms(k_root);
        for (uint32_t c = 0; c < k_root; ++c)
            cent_norms[c] = simd::dot_f32(
                root_centroids_pca[c].data(),
                root_centroids_pca[c].data(), pca_dims);

        while (vectors_done < n) {
            Chunk chunk;
            if (!ctx.source.next(chunk)) break;
            const uint32_t take = chunk.count;
            const float* vec_buf = chunk.vectors;  // valid until next next()

            // Parallel assign + accumulate.
            std::vector<std::future<void>> futs;
            const uint32_t n_threads = std::min(hw, take);
            const uint32_t per = (take + n_threads - 1) / n_threads;
            for (uint32_t t = 0; t < n_threads; ++t) {
                const uint32_t start = t * per;
                const uint32_t end = std::min(start + per, take);
                if (start >= end) break;
                futs.push_back(std::async(std::launch::async,
                    [&](uint32_t tid, uint32_t s, uint32_t e) {
                        double* sums = t_sums[tid].data();
                        uint64_t* counts = t_counts[tid].data();
                        // Hoisted scratch (one allocation per chunk, not
                        // per vector — the per-vector vector<> cost ~2.5x
                        // on this loop at 10M scale).
                        std::vector<float> proj(pca_dims);
                        for (uint32_t i = s; i < e; ++i) {
                            const float* xi = &vec_buf[i * dim];
                            for (uint32_t k = 0; k < pca_dims; ++k)
                                proj[k] = simd::dot_f32(
                                    &rotation[k * dim], xi, dim) - mean_proj[k];
                            float best_d = std::numeric_limits<float>::max();
                            uint32_t best_c = 0;
                            for (uint32_t c = 0; c < k_root; ++c) {
                                const float dot = simd::dot_f32(
                                    proj.data(), root_centroids_pca[c].data(), pca_dims);
                                const float d = cent_norms[c] - 2.0f * dot;
                                if (d < best_d) { best_d = d; best_c = c; }
                            }
                            for (uint32_t k = 0; k < pca_dims; ++k)
                                sums[best_c * pca_dims + k] += proj[k];
                            ++counts[best_c];
                        }
                    }, t, start, end));
            }
            for (auto& fut : futs) fut.get();
            vectors_done += take;
        }

        // Reduce per-thread accumulators into cluster_sums/cluster_counts.
        for (uint32_t c = 0; c < k_root; ++c) {
            std::fill(cluster_sums[c].begin(), cluster_sums[c].end(), 0.0);
            cluster_counts[c] = 0;
            for (uint32_t t = 0; t < hw; ++t) {
                for (uint32_t k = 0; k < pca_dims; ++k)
                    cluster_sums[c][k] += t_sums[t][c * pca_dims + k];
                cluster_counts[c] += t_counts[t][c];
            }
        }

        // Update centroids: mean of assigned vectors.
        uint32_t n_empty = 0;
        for (uint32_t c = 0; c < k_root; ++c) {
            if (cluster_counts[c] == 0) {
                ++n_empty;
                // Reseed from the largest cluster's centroid + small perturbation.
                uint32_t largest = 0;
                for (uint32_t cc = 1; cc < k_root; ++cc)
                    if (cluster_counts[cc] > cluster_counts[largest]) largest = cc;
                std::copy(root_centroids_pca[largest].begin(),
                          root_centroids_pca[largest].end(),
                          root_centroids_pca[c].begin());
                continue;
            }
            const double inv = 1.0 / cluster_counts[c];
            for (uint32_t k = 0; k < pca_dims; ++k)
                root_centroids_pca[c][k] = static_cast<float>(
                    cluster_sums[c][k] * inv);
        }

        const double pass_secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pass_t0).count();
        spdlog::info("[sextant] build_streaming_pca: Lloyd pass {} done "
                     "({:.1f}s, {} empty clusters)", pass + 1, pass_secs, n_empty);

        // Early exit: check convergence by sampling.
        // Project a small sample and check how many changed cluster.
        if (pass >= 1) {
            uint32_t sample_sz = std::min<uint32_t>(4096, train_n);
            uint32_t n_changed = 0;
            for (uint32_t i = 0; i < sample_sz; ++i) {
                const float* vi = &ctx.sample_pca[i * pca_dims];
                float best_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < k_root; ++c) {
                    float d = 0.0f;
                    for (uint32_t k = 0; k < pca_dims; ++k) {
                        const float diff = vi[k] - root_centroids_pca[c][k];
                        d += diff * diff;
                    }
                    if (d < best_d) { best_d = d; best_c = c; }
                }
                if (pass == 1) prev_assignment.push_back(best_c);
                else if (best_c != prev_assignment[i % prev_assignment.size()])
                    ++n_changed;
            }
            if (pass >= 2 && prev_assignment.size() > 0) {
                float change_rate = static_cast<float>(n_changed) / sample_sz;
                spdlog::info("[sextant] build_streaming_pca: Lloyd change rate "
                             "{:.2f}%", 100.0f * change_rate);
                if (change_rate < 0.01f) {
                    spdlog::info("[sextant] build_streaming_pca: Lloyd converged "
                                 "at pass {}", pass + 1);
                    break;
                }
                // Update prev_assignment for next comparison.
                for (uint32_t i = 0; i < sample_sz; ++i) {
                    const float* vi = &ctx.sample_pca[i * pca_dims];
                    float best_d = std::numeric_limits<float>::max();
                    uint32_t best_c = 0;
                    for (uint32_t c = 0; c < k_root; ++c) {
                        float d = 0.0f;
                        for (uint32_t k = 0; k < pca_dims; ++k) {
                            const float diff = vi[k] - root_centroids_pca[c][k];
                            d += diff * diff;
                        }
                        if (d < best_d) { best_d = d; best_c = c; }
                    }
                    prev_assignment[i] = best_c;
                }
            }
        }
    }
    spdlog::info("[sextant] build_streaming_pca: Lloyd refinement done in {:.2f}s",
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_lloyd).count());
    ctx.metrics.stop(m_lloyd);

    // --- Phase I: measure per-cluster d_eff (mean gap to 2nd-nearest) ---
    // Uses the training sample (already in PCA space). For each sample vector,
    // find the nearest and 2nd-nearest centroid, compute the gap (d2 - d1),
    // and accumulate per-cluster. The cluster assignment is by nearest centroid.
    // d_eff_c = mean gap for cluster c. Larger gap = lower intrinsic dim.
    {
        const uint32_t s = std::min<uint32_t>(8192, train_n);
        std::vector<double> sum_gap(k_root, 0.0);
        std::vector<uint64_t> gap_count(k_root, 0);
        for (uint32_t i = 0; i < s; ++i) {
            const float* vi = &ctx.sample_pca[i * pca_dims];
            float d1 = std::numeric_limits<float>::max();
            float d2 = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k_root; ++c) {
                float d = 0.0f;
                for (uint32_t k = 0; k < pca_dims; ++k) {
                    const float diff = vi[k] - root_centroids_pca[c][k];
                    d += diff * diff;
                }
                if (d < d1) { d2 = d1; d1 = d; best_c = c; }
                else if (d < d2) d2 = d;
            }
            sum_gap[best_c] += (d2 - d1);
            ++gap_count[best_c];
        }
        ctx.cluster_d_eff.resize(k_root);
        ctx.cluster_closure_eps.resize(k_root);
        const float cmult = cfg.closure_multiplier > 0
            ? cfg.closure_multiplier : 0.15f;
        for (uint32_t c = 0; c < k_root; ++c) {
            if (gap_count[c] > 0) {
                ctx.cluster_d_eff[c] = static_cast<float>(
                    sum_gap[c] / gap_count[c]);
            } else {
                ctx.cluster_d_eff[c] = ctx.closure_epsilon / cmult;
            }
            ctx.cluster_closure_eps[c] = ctx.cluster_d_eff[c] * cmult;
        }
        // Log summary stats.
        float min_eff = ctx.cluster_d_eff[0], max_eff = ctx.cluster_d_eff[0];
        double sum_eff = 0;
        for (uint32_t c = 0; c < k_root; ++c) {
            min_eff = std::min(min_eff, ctx.cluster_d_eff[c]);
            max_eff = std::max(max_eff, ctx.cluster_d_eff[c]);
            sum_eff += ctx.cluster_d_eff[c];
        }
        spdlog::info("[sextant] build_streaming_pca: per-cluster d_eff: "
                     "min={:.2f} max={:.2f} mean={:.2f} (global eps={:.4f})",
                     min_eff, max_eff, sum_eff / k_root, ctx.closure_epsilon);
    }

    // --- 5b. Depth-3 super-clustering ---
    // When depth==3, the k_root fine-grained centroids are grouped into k_l1
    // super-clusters (k-means in PCA space on the centroids themselves). Each
    // fine centroid c is assigned to one super-group; the super-group becomes
    // an L1 node whose children are the L2 nodes (fine centroids) in that
    // group. The root's PCA centroids (stored in the blob) are these k_l1
    // super-cluster centroids.
    if (ctx.depth == 3) {
        const uint32_t k_l1 = ctx.k_l1;
        const auto t_super = std::chrono::steady_clock::now();
        auto m_super = ctx.metrics.start("super");
        std::vector<std::vector<float>> super_centroids_pca(k_l1, std::vector<float>(pca_dims, 0.0f));
        // Initialize: k-means++-like even pick from the fine centroids.
        for (uint32_t g = 0; g < k_l1; ++g) {
            const uint32_t src = (g * k_root) / k_l1;
            super_centroids_pca[g].assign(
                root_centroids_pca[src].begin(),
                root_centroids_pca[src].end());
        }
        std::vector<uint32_t> super_group(k_root, 0);
        for (uint32_t iter = 0; iter < 10; ++iter) {
            // Assign each fine centroid to nearest super-cluster.
            std::vector<std::vector<double>> sums(
                k_l1, std::vector<double>(pca_dims, 0.0));
            std::vector<uint64_t> counts(k_l1, 0);
            uint32_t n_changed = 0;
            for (uint32_t c = 0; c < k_root; ++c) {
                const float* fc = root_centroids_pca[c].data();
                float best_d = std::numeric_limits<float>::max();
                uint32_t best_g = 0;
                for (uint32_t g = 0; g < k_l1; ++g) {
                    float d = 0.0f;
                    for (uint32_t k = 0; k < pca_dims; ++k) {
                        const float diff = fc[k] - super_centroids_pca[g][k];
                        d += diff * diff;
                    }
                    if (d < best_d) { best_d = d; best_g = g; }
                }
                if (iter > 0 && best_g != super_group[c]) ++n_changed;
                super_group[c] = best_g;
                for (uint32_t k = 0; k < pca_dims; ++k)
                    sums[best_g][k] += fc[k];
                ++counts[best_g];
            }
            // Update super-centroids.
            for (uint32_t g = 0; g < k_l1; ++g) {
                if (counts[g] == 0) {
                    // Reseed from a random fine centroid.
                    uint32_t src = (g * 7919 + 1) % k_root;
                    super_centroids_pca[g].assign(
                        root_centroids_pca[src].begin(),
                        root_centroids_pca[src].end());
                    continue;
                }
                const double inv = 1.0 / counts[g];
                for (uint32_t k = 0; k < pca_dims; ++k)
                    super_centroids_pca[g][k] =
                        static_cast<float>(sums[g][k] * inv);
            }
            if (iter >= 2 &&
                static_cast<double>(n_changed) / k_root < 0.01) {
                spdlog::info("[sextant] build_streaming_pca: super-k-means "
                             "converged at iter {}", iter + 1);
                break;
            }
        }
        spdlog::info("[sextant] build_streaming_pca: depth-3 super-clustering "
                     "({} fine → {} groups) in {:.2f}s", k_root, k_l1,
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_super).count());
        ctx.metrics.stop(m_super);
        ctx.super_group = std::move(super_group);
        ctx.super_centroids_pca = std::move(super_centroids_pca);
    }
}

// ---------------------------------------------------------------------------
// Phase 4: the streaming emission pass. Initializes per-cluster leaf buffers,
// defines the append_filter_row + flush_buffer lambdas, projects root centroids
// back to FP16 original space, streams all vectors (project → route → encode →
// buffer → flush on overflow), populates the cardinality table, and final-
// flushes all buffers. Writes leaf extents directly to the file via the
// allocator. Stores leaf_metas / root_to_leaves /
// leaf_summaries / n_leaves_total on the context.
// ---------------------------------------------------------------------------
void run_emission_pass(TreeBuildContext& ctx, PageFile& file, PageAllocator& alloc) {
    const auto& cfg = ctx.cfg;
    const auto n = ctx.n;
    const Dim dim = ctx.dim;
    const uint32_t pca_dims = ctx.pca_dims;
    const uint32_t k_root = ctx.k_root;
    const uint8_t scan_bits = ctx.scan_bits;
    const uint16_t m4 = ctx.m4;
    const uint32_t leaf_cap = ctx.leaf_cap;
    const uint32_t summary_size = ctx.summary_size;
    const MetricKind metric = ctx.metric;
    const auto& rotation = ctx.rotation;
    const auto& mean_proj = ctx.mean_proj;
    const auto& mean = ctx.mean;
    const float closure_epsilon = ctx.closure_epsilon;
    const auto& root_centroids_pca = ctx.root_centroids_pca;
    // Phase I: per-cluster closure epsilon. Falls back to global if not
    // populated (e.g. if d_eff measurement was skipped).
    const auto& per_cluster_eps = ctx.cluster_closure_eps;
    // --- 6. Emission pass: assign + encode + write leaves ---
    const auto t_stream = std::chrono::steady_clock::now();
    auto m_stream = ctx.metrics.start("stream");
    const uint32_t code_size = ctx.coder->code_size();
    // Local families keep the raw FP16 vectors in the buffers; their
    // per-leaf state is fitted at flush time.
    const bool keep_raw_vecs = ctx.coder->stores_raw_vectors_during_build();
    // has_filter is driven by the schema, not the sidecar data. When the source
    // provides filter columns per-chunk (e.g. ParquetSource), cfg.filter_column_data
    // is empty but the schema still declares the columns.
    const bool has_filter = cfg.filter_schema.n_filter_columns() > 0;
    const uint32_t n_schema_cols =
        static_cast<uint32_t>(cfg.filter_schema.columns.size());
    const bool has_payload =
        cfg.filter_schema.has_payload && cfg.payload_data && cfg.payload_offsets;
    ctx.has_filter = has_filter;
    ctx.has_payload = has_payload;
    ctx.n_schema_cols = n_schema_cols;

    struct LeafBuffer {
        std::vector<uint8_t> codes;
        std::vector<RowId> row_ids;
        // Per-vector IP bias (||x||/||x_hat||, fp16) for scalar + InnerProduct.
        std::vector<float16_t> ip_biases;
        // Incremental centroid accumulator (replaces fp16_vecs). A running
        // FP32-sum (stored as double for stability) + count per buffer.
        // At flush time, divide sum by count to get the centroid. This keeps
        // the per-buffer footprint at O(leaf_cap × (code_size + 8) + dim×8)
        // instead of O(leaf_cap × dim × 2).
        std::vector<double> centroid_sum;
        uint32_t centroid_count = 0;
        // Filter column data for the rows in this buffer (Phase C).
        std::vector<ColumnData> filter_cols;
        // Payload info for the rows in this buffer (Phase E).
        std::vector<uint32_t> payload_lens;
        std::vector<const uint8_t*> payload_ptrs;
        // For local_pq: raw FP16 vectors (for per-leaf codebook training at flush).
        // Empty when not local_pq.
        std::vector<float16_t> fp16_vecs;
    };
    std::vector<LeafBuffer> buffers(k_root);
    for (auto& b : buffers) b.centroid_sum.assign(dim, 0.0);

    if (has_filter) {
        for (auto& b : buffers) {
            b.filter_cols.resize(n_schema_cols);
            for (uint32_t c = 0; c < n_schema_cols; ++c)
                b.filter_cols[c].type = cfg.filter_schema.columns[c].type;
        }
    }

    // Per-chunk filter data state. Updated at the top of each chunk iteration
    // in the streaming loop below; read by append_filter_row and the
    // cardinality/payload dual-path logic.
    bool chunk_has_filter = false;
    const void* const* chunk_fcols = nullptr;

    /// Append the filter column values for a row into a leaf's per-column
    /// ColumnData (Phase C). `local_idx` is the row index within the current
    /// chunk; `row_id` is the global row id. When the chunk provides per-chunk
    /// filter data (chunk_has_filter), values are read from chunk_fcols by
    /// local_idx; otherwise the global cfg.filter_column_data fallback is used
    /// (indexed by the global row_id).
    auto append_filter_row = [&](std::vector<ColumnData>& dst, RowId row_id, uint32_t local_idx) {
        const uint32_t r = static_cast<uint32_t>(row_id);
        for (uint32_t c = 0; c < n_schema_cols; ++c) {
            auto& d = dst[c];
            switch (d.type) {
                case ColumnType::Int32:
                case ColumnType::Int64:
                case ColumnType::Float:
                case ColumnType::Bool: {
                    const uint8_t w = column_type_width(d.type);
                    const uint8_t* sp;
                    if (chunk_has_filter) {
                        sp = static_cast<const uint8_t*>(chunk_fcols[c]) +
                             static_cast<size_t>(local_idx) * w;
                    } else {
                        const auto& src = cfg.filter_column_data[c];
                        sp = src.fixed_data.data() + static_cast<size_t>(r) * w;
                    }
                    d.fixed_data.insert(d.fixed_data.end(), sp, sp + w);
                    break;
                }
                case ColumnType::String: {
                    // Validation happens in MemSourceBuilder (before uint16 truncation).
                    const FilterStringColumn* sc;
                    uint32_t off; uint16_t len;
                    if (chunk_has_filter) {
                        sc = static_cast<const FilterStringColumn*>(chunk_fcols[c]);
                        off = sc->offsets[local_idx]; len = sc->lengths[local_idx];
                    } else {
                        const auto& src = cfg.filter_column_data[c];
                        off = src.str_offsets[r]; len = src.str_lengths[r];
                        sc = nullptr;  // not used for data ptr
                    }
                    d.str_offsets.push_back(static_cast<uint32_t>(d.str_data.size()));
                    d.str_lengths.push_back(len);
                    if (chunk_has_filter) {
                        d.str_data.insert(d.str_data.end(),
                                          sc->data + off, sc->data + off + len);
                    } else {
                        const auto& src = cfg.filter_column_data[c];
                        d.str_data.insert(d.str_data.end(),
                                          src.str_data.data() + off,
                                          src.str_data.data() + off + len);
                    }
                    break;
                }
                case ColumnType::Set: {
                    // Validation happens in MemSourceBuilder (before uint8/uint16 truncation).
                    const FilterSetColumn* fsc;
                    uint8_t ec; uint32_t off; const uint16_t* elem_lens; const char* elem_data;
                    if (chunk_has_filter) {
                        fsc = static_cast<const FilterSetColumn*>(chunk_fcols[c]);
                        ec = fsc->counts[local_idx];
                        off = fsc->offsets[local_idx];
                        elem_lens = fsc->element_lengths;
                        elem_data = fsc->element_data;
                    } else {
                        const auto& src = cfg.filter_column_data[c];
                        fsc = nullptr;
                        ec = src.set_counts[r];
                        off = src.set_offsets[r];
                        elem_lens = src.set_elem_lengths.data();
                        elem_data = src.set_elem_data.data();
                    }
                    d.set_counts.push_back(ec);
                    d.set_offsets.push_back(static_cast<uint32_t>(d.set_elem_lengths.size()));
                    // Cumulative byte offset into elem_data for element `off`.
                    uint32_t byte_off = 0;
                    for (uint32_t k = 0; k < off; ++k)
                        byte_off += elem_lens[k];
                    for (uint32_t e = 0; e < ec; ++e) {
                        const uint16_t elen = elem_lens[off + e];
                        d.set_elem_lengths.push_back(elen);
                        d.set_elem_data.insert(d.set_elem_data.end(),
                            elem_data + byte_off,
                            elem_data + byte_off + elen);
                        byte_off += elen;
                    }
                    break;
                }
            }
        }
    };

    // Per-root-cluster leaf ordering (global leaf index per cluster). Built
    // during emission and consumed by the internal-node write phase.
    ctx.root_to_leaves.resize(k_root);
    std::vector<uint32_t> n_leaves_in_cluster(k_root, 0);
    std::vector<LeafMeta>& leaf_metas = ctx.leaf_metas;  // grows as leaves are flushed
    std::vector<std::vector<uint8_t>>& leaf_summaries = ctx.leaf_summaries;

    const uint32_t cpb = (scan_bits == 4) ? 32 : 16;
    const uint32_t bb = m4 * 16;

    auto flush_buffer = [&](uint32_t c) {
        auto& buf = buffers[c];
        if (buf.row_ids.empty()) return;
        const uint32_t count = static_cast<uint32_t>(buf.row_ids.size());

        // Compute the leaf centroid from the incremental accumulator.
        std::vector<float> centroid_f32(dim);
        compute_centroid_spherical(buf.centroid_sum.data(), buf.centroid_count,
                                   dim, metric, centroid_f32.data());
        std::vector<float16_t> leaf_centroid(dim);
        for (uint16_t d = 0; d < dim; ++d)
            leaf_centroid[d] = static_cast<float16_t>(centroid_f32[d]);

        // Phase C: filter column region size for this leaf.
        const uint64_t fcb = has_filter
            ? filter_columns_bytes(count, cfg.filter_schema, buf.filter_cols) : 0;

        // Build-time validation warnings (architecture plan §3.5.2).
        const uint32_t n_blocks_chk = (count + cpb - 1) / cpb;
        const uint64_t pq_code_bytes =
            static_cast<uint64_t>(n_blocks_chk) * bb;
        if (has_filter && pq_code_bytes > 0 && fcb > 10ull * pq_code_bytes) {
            spdlog::warn("[sextant] build_streaming_pca: leaf {} filter column "
                         "data ({}B) > 10× PQ codes ({}B)",
                         leaf_metas.size(), fcb, pq_code_bytes);
        }

        const uint32_t npg = static_cast<uint32_t>(
            (ctx.coder->extent_bytes(count, summary_size, fcb) +
             kPageSize - 1) / kPageSize);
        if (npg > 512) {
            spdlog::warn("[sextant] build_streaming_pca: leaf {} extent = {} "
                         "pages (> 512)", leaf_metas.size(), npg);
        }
        const PageId page = alloc.alloc_extent(file, npg);

        std::vector<uint8_t> obuf(static_cast<size_t>(npg) * kPageSize, 0);
        auto* lh = reinterpret_cast<TreeLeafHeader*>(obuf.data());
        lh->magic = kTreeLeafMagic;
        lh->count = count; lh->tombstone_count = 0; lh->m4 = m4;
        lh->pq_bits = scan_bits;
        lh->block_bytes = bb; lh->codes_per_block = cpb; lh->extent_pages = npg;
        lh->summary_size = summary_size;
        lh->n_filter_columns = cfg.filter_schema.n_filter_columns();

        // Phase E: allocate + build the payload extent for this leaf.
        PageId payload_page = kInvalidPage;
        uint32_t payload_npg = 0;
        std::vector<uint8_t> pbuf;
        if (has_payload && !buf.payload_lens.empty()) {
            uint64_t total_data = 0;
            for (uint32_t i = 0; i < count; ++i)
                total_data += buf.payload_lens[i];
            const uint64_t payload_bytes =
                static_cast<uint64_t>(count) * 4 +  // offsets
                static_cast<uint64_t>(count) * 4 +  // lengths
                total_data;                          // data
            payload_npg = static_cast<uint32_t>(
                (payload_bytes + kPageSize - 1) / kPageSize);
            payload_page = alloc.alloc_extent(file, payload_npg);

            pbuf.assign(static_cast<size_t>(payload_npg) * kPageSize, 0);
            uint32_t* offsets = reinterpret_cast<uint32_t*>(pbuf.data());
            uint32_t* lengths = offsets + count;
            uint8_t* pdata = reinterpret_cast<uint8_t*>(lengths + count);
            uint32_t acc = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t plen = buf.payload_lens[i];
                offsets[i] = acc;
                lengths[i] = plen;
                std::memcpy(pdata + acc, buf.payload_ptrs[i], plen);
                acc += plen;
            }
        }
        lh->payload_extent_page = payload_page;
        lh->payload_extent_pages = payload_npg;
        lh->summary_dirty = 0;
        lh->next_dirty = kInvalidPage;

        // Phase C: per-leaf filter summary (min/max + blooms).
        std::vector<uint8_t> leaf_summary;
        if (summary_size > 0) {
            write_filter_summary(obuf.data() + leaf_filter_offset(),
                                 summary_size, cfg.filter_schema, count,
                                 buf.filter_cols);
            leaf_summary.resize(summary_size);
            std::memcpy(leaf_summary.data(),
                        obuf.data() + leaf_filter_offset(), summary_size);
        }

        // Family-owned leaf emission: per-leaf state (levels / codebook /
        // centroid) + code region (+ IP biases). Returns the row_ids offset.
        LeafCoder::LeafFlushInput fin;
        fin.count = count;
        fin.fp16_vecs = buf.fp16_vecs.data();
        fin.centroid_f32 = centroid_f32.data();
        fin.codes = buf.codes.data();
        fin.ip_biases = ctx.has_ip_bias ? buf.ip_biases.data() : nullptr;
        fin.summary_size = summary_size;
        const uint64_t rowids_off = ctx.coder->flush_leaf(fin, obuf.data());

        RowId* rids = reinterpret_cast<RowId*>(obuf.data() + rowids_off);
        std::memcpy(rids, buf.row_ids.data(), count * sizeof(RowId));

        // Phase C: filter column data region (after row_ids). This must
        // match LeafFilterLayout::compute's filter_base offset.
        if (fcb > 0) {
            const uint64_t fc_off =
                rowids_off + static_cast<uint64_t>(count) * sizeof(RowId);
            lh->filter_columns_offset = fc_off;
            const uint64_t written = write_filter_columns(
                obuf.data() + fc_off, count, cfg.filter_schema, buf.filter_cols);
            if (written != fcb) {
                throw Error(ErrorCode::CorruptIndex,
                    "build_streaming_pca: filter column bytes mismatch "
                    "(wrote " + std::to_string(written) + ", expected " +
                    std::to_string(fcb) + ")");
            }
        } else {
            lh->filter_columns_offset = 0;
        }

        lh->header_crc = header_crc(lh, offsetof(TreeLeafHeader, header_crc));
        file.write_pages(page, npg, obuf.data());

        // Phase E: write the payload extent.
        if (payload_page != kInvalidPage && payload_npg > 0)
            file.write_pages(payload_page, payload_npg, pbuf.data());

        // Record metadata + cache the filter summary for bottom-up propagation.
        // The global leaf index is the current leaf_metas size (flush order).
        // root_to_leaves[c] stores the global indices of cluster c's leaves;
        // because leaves flush in arrival order (not cluster order), these
        // indices are NOT contiguous — the internal-node write phase must use
        // them directly rather than assuming cluster-contiguous layout.
        ctx.root_to_leaves[c].push_back(
            static_cast<uint32_t>(leaf_metas.size()));
        n_leaves_in_cluster[c]++;

        leaf_metas.push_back(LeafMeta{page, npg, std::move(leaf_centroid)});
        leaf_summaries.push_back(std::move(leaf_summary));

        // Reset the buffer for the next leaf in this cluster.
        buf.codes.clear();
        buf.row_ids.clear();
        buf.ip_biases.clear();
        buf.fp16_vecs.clear();
        std::fill(buf.centroid_sum.begin(), buf.centroid_sum.end(), 0.0);
        buf.centroid_count = 0;
        buf.payload_lens.clear();
        buf.payload_ptrs.clear();
        if (has_filter) {
            buf.filter_cols.assign(n_schema_cols, ColumnData{});
            for (uint32_t cc = 0; cc < n_schema_cols; ++cc)
                buf.filter_cols[cc].type = cfg.filter_schema.columns[cc].type;
        }
    };

    // Precompute FP16 root centroids for the search path (the tree stores
    // root centroids in original FP16 space, not PCA space — we project back).
    // Actually, the root centroids are in PCA space. For search, we need
    // centroids in original FP16 space. We'll store the PCA-space centroids
    // in the tree's leaf centroids (level-1 children) and use FP16 original-
    // space centroids for the root. For routing at search time, we project
    // the query to PCA space and compute distances there.
    // 
    // BUT: the search path currently routes by FP16 distance in original space.
    // To use PCA routing at search time, we'd need to modify the search path.
    // For now, let's project the root centroids BACK to original space for
    // storage. The search will route by FP16 distance in original space
    // (which is what the tree format supports).
    //
    // This is a limitation: the tree stores FP16 original-space centroids.
    // PCA routing at search time requires storing the projection matrix in
    // the tree and modifying the search path. For this prototype, we use
    // PCA for BUILD-TIME routing (better partition) and original-space FP16
    // for SEARCH-TIME routing (the existing path). The partition quality
    // improvement should still help recall even with original-space search
    // routing, because the leaf membership is better.

    // Project root centroids back to original FP16 space (add mean back).
    // original[d] = mean[d] + Σ_k pca_centroid[k] × rotation[k][d]
    ctx.root_centroids_fp16.resize(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        ctx.root_centroids_fp16[c].resize(dim);
        for (uint16_t d = 0; d < dim; ++d) {
            double val = mean[d];
            for (uint32_t k = 0; k < pca_dims; ++k)
                val += root_centroids_pca[c][k] * rotation[k * dim + d];
            ctx.root_centroids_fp16[c][d] = static_cast<float16_t>(val);
        }
    }

    ctx.source.reset();
    {
        std::vector<float16_t> fp16_buf;
        std::vector<float> pca_buf;

        // Precompute centroid norms for SIMD distance (dot-product decomposition).
        std::vector<float> cent_norms(k_root);
        for (uint32_t c = 0; c < k_root; ++c)
            cent_norms[c] = simd::dot_f32(root_centroids_pca[c].data(),
                                           root_centroids_pca[c].data(), pca_dims);

        const uint32_t hw = cfg.num_threads > 0
            ? cfg.num_threads
            : std::max(1u, std::thread::hardware_concurrency());

        // Per-thread sharded cluster buffer. Accumulates the encoded code (or
        // raw FP16 for local_pq), row id, the raw FP16 vector (source for the
        // incremental centroid during the serial merge), payload info, and the
        // chunk-local row index needed to replay the filter-column append.
        struct ThreadLeafBuffer {
            std::vector<uint8_t> codes;        // non-local_pq
            std::vector<float16_t> ip_biases;  // scalar + InnerProduct
            std::vector<float16_t> fp16_vecs;  // always (centroid source)
            std::vector<RowId> row_ids;
            std::vector<uint32_t> local_indices;
            std::vector<uint32_t> payload_lens;
            std::vector<const uint8_t*> payload_ptrs;
            void clear() {
                codes.clear(); ip_biases.clear(); fp16_vecs.clear();
                row_ids.clear(); local_indices.clear(); payload_lens.clear();
                payload_ptrs.clear();
            }
        };
        // Hoisted outside the chunk loop so capacity is reused across chunks.
        std::vector<std::vector<ThreadLeafBuffer>> thread_buffers(hw);
        std::vector<CardinalityTable> thread_cards(hw);
        for (uint32_t t = 0; t < hw; ++t) {
            thread_buffers[t].resize(k_root);
            if (has_filter)
                thread_cards[t].init(cfg.filter_schema, ctx.n);
        }

        uint64_t offset = 0;
        uint64_t prev_million = 0;
        while (true) {
            Chunk chunk;
            if (!ctx.source.next(chunk)) break;
            const uint32_t take = chunk.count;
            const float* vec_buf = chunk.vectors;  // NO copy
            const RowId* chunk_row_ids = chunk.row_ids;  // source-assigned ids
            // Capture whether this chunk carries per-chunk filter/payload data
            // (e.g. from a ParquetSource). The lambdas below read these.
            chunk_has_filter = (chunk.filter_columns != nullptr);
            chunk_fcols = chunk.filter_columns;
            const bool chunk_has_payload = (chunk.payload_data != nullptr);
            const uint8_t* chunk_pdata = chunk.payload_data;
            const uint32_t* chunk_poffsets = chunk.payload_offsets;
            if (fp16_buf.size() < static_cast<size_t>(take) * dim)
                fp16_buf.resize(static_cast<size_t>(take) * dim);
            cast_fp32_to_fp16(vec_buf, fp16_buf.data(),
                              static_cast<size_t>(take) * dim);

            // Parallel: project + route + encode. Store (target_centroids, code)
            // per vector for serial append below.
            // Per-vector result: the nearest centroid (+closure matches) and code.
            // Format: for each vector, a list of target centroid IDs + the code.
            std::vector<std::vector<uint8_t>> chunk_codes(take);
            std::vector<std::vector<uint32_t>> chunk_targets(take);
            std::vector<float16_t> chunk_biases(
                ctx.has_ip_bias ? take : 0, float16_t(1.0f));

            std::vector<std::future<void>> futs;
            const uint32_t n_threads = std::min(hw, take);
            const uint32_t per = (take + n_threads - 1) / n_threads;
            for (uint32_t t = 0; t < n_threads; ++t) {
                const uint32_t start = t * per;
                const uint32_t end = std::min(start + per, take);
                if (start >= end) break;
                futs.push_back(std::async(std::launch::async,
                    [&](uint32_t s, uint32_t e) {
                        std::vector<float> proj(pca_dims);
                        for (uint32_t i = s; i < e; ++i) {
                            const float* xi = &vec_buf[i * dim];
                            // Project to PCA space.
                            for (uint32_t k = 0; k < pca_dims; ++k)
                                proj[k] = simd::dot_f32(
                                    &rotation[k * dim], xi, dim) - mean_proj[k];
                            // Route: find nearest + closure matches.
                            float min_d = std::numeric_limits<float>::max();
                            for (uint32_t c = 0; c < k_root; ++c) {
                                const float dot = simd::dot_f32(
                                    proj.data(), root_centroids_pca[c].data(), pca_dims);
                                const float d = cent_norms[c] - 2.0f * dot;
                                if (d < min_d) min_d = d;
                            }
                            // Encode scan code (skipped for local families:
                            // raw vectors are kept and encoded per-leaf at
                            // flush).
                            if (!keep_raw_vecs) {
                                std::vector<uint8_t> code(code_size);
                                float bias = 0.f;
                                ctx.coder->encode(xi, code.data(),
                                                  ctx.has_ip_bias ? &bias
                                                                  : nullptr);
                                chunk_codes[i].assign(code.begin(),
                                                      code.end());
                                if (ctx.has_ip_bias)
                                    chunk_biases[i] = float16_t(bias);
                            }
                            // Find closure targets using per-cluster epsilon.
                            // Phase I: each cluster c has its own closure eps
                            // derived from its local d_eff. A vector is
                            // replicated into c if dist(v,c) is within
                            // cluster_closure_eps[c] of the nearest distance.
                            for (uint32_t c = 0; c < k_root; ++c) {
                                const float dot = simd::dot_f32(
                                    proj.data(), root_centroids_pca[c].data(), pca_dims);
                                const float d = cent_norms[c] - 2.0f * dot;
                                const float eps = (c < per_cluster_eps.size())
                                    ? per_cluster_eps[c] : closure_epsilon;
                                if (std::fabs(d - min_d) <= eps)
                                    chunk_targets[i].push_back(c);
                            }
                        }
                    }, start, end));
            }
            for (auto& fut : futs) fut.get();

            // Parallel append into per-thread sharded buffers + per-thread
            // cardinality tables, followed by a serial merge-and-flush
            // reduction. The serial merge replays each vector into the shared
            // buffers in the original global order (thread 0's contiguous
            // shard, then thread 1's, ...) and flushes at the exact same
            // leaf_cap overflow points the serial loop would, so every leaf
            // ends up with identical contents (same vectors + centroid).
            //
            // thread_buffers / thread_cards are hoisted outside the chunk
            // loop; clear the active shards for this chunk.
            for (uint32_t t = 0; t < n_threads; ++t) {
                for (auto& tb : thread_buffers[t]) tb.clear();
                if (has_filter) thread_cards[t].clear_stats();
            }

            {
                std::vector<std::future<void>> afuts;
                for (uint32_t t = 0; t < n_threads; ++t) {
                    const uint32_t start = t * per;
                    const uint32_t end = std::min(start + per, take);
                    if (start >= end) break;
                    afuts.push_back(std::async(std::launch::async,
                        [&](uint32_t s, uint32_t e, uint32_t tid) {
                            auto& tbufs = thread_buffers[tid];
                            CardinalityTable* tcard = has_filter
                                ? &thread_cards[tid] : nullptr;
                            for (uint32_t i = s; i < e; ++i) {
                                const float16_t* fvec = &fp16_buf[i * dim];
                                const RowId rid = chunk_row_ids[i];
                                // Phase D: record this row's filter column
                                // values in the per-thread cardinality table
                                // (once per row, not per closure target).
                                if (tcard) {
                                    const uint32_t r =
                                        static_cast<uint32_t>(rid);
                                    for (uint32_t c = 0;
                                         c < cfg.filter_schema.columns.size();
                                         ++c) {
                                        const auto& col =
                                            cfg.filter_schema.columns[c];
                                        if (col.type == ColumnType::String) {
                                            std::string_view sv;
                                            if (chunk_has_filter) {
                                                const auto* sc =
                                                    static_cast<const FilterStringColumn*>(
                                                        chunk_fcols[c]);
                                                sv = std::string_view(
                                                    sc->data + sc->offsets[i],
                                                    sc->lengths[i]);
                                            } else {
                                                const auto& src =
                                                    cfg.filter_column_data[c];
                                                sv = std::string_view(
                                                    src.str_data.data()
                                                        + src.str_offsets[r],
                                                    src.str_lengths[r]);
                                            }
                                            tcard->add_string(c, sv);
                                        } else if (col.type == ColumnType::Set) {
                                            if (chunk_has_filter) {
                                                const auto* fsc =
                                                    static_cast<const FilterSetColumn*>(
                                                        chunk_fcols[c]);
                                                tcard->add_set(c,
                                                    fsc->counts, fsc->offsets,
                                                    fsc->element_lengths,
                                                    fsc->element_data, i);
                                            } else {
                                                const auto& src =
                                                    cfg.filter_column_data[c];
                                                tcard->add_set(c,
                                                    src.set_counts.data(),
                                                    src.set_offsets.data(),
                                                    src.set_elem_lengths.data(),
                                                    src.set_elem_data.data(), r);
                                            }
                                        } else if (col.type == ColumnType::Int32) {
                                            int32_t v;
                                            if (chunk_has_filter) {
                                                std::memcpy(&v,
                                                    static_cast<const uint8_t*>(
                                                        chunk_fcols[c])
                                                        + static_cast<size_t>(i)*4, 4);
                                            } else {
                                                const auto& src =
                                                    cfg.filter_column_data[c];
                                                std::memcpy(&v,
                                                    src.fixed_data.data() + r*4, 4);
                                            }
                                            tcard->add_numeric(c, static_cast<double>(v));
                                        } else if (col.type == ColumnType::Int64) {
                                            int64_t v;
                                            if (chunk_has_filter) {
                                                std::memcpy(&v,
                                                    static_cast<const uint8_t*>(
                                                        chunk_fcols[c])
                                                        + static_cast<size_t>(i)*8, 8);
                                            } else {
                                                const auto& src =
                                                    cfg.filter_column_data[c];
                                                std::memcpy(&v,
                                                    src.fixed_data.data() + r*8, 8);
                                            }
                                            tcard->add_numeric(c, static_cast<double>(v));
                                        } else if (col.type == ColumnType::Float) {
                                            float v;
                                            if (chunk_has_filter) {
                                                std::memcpy(&v,
                                                    static_cast<const uint8_t*>(
                                                        chunk_fcols[c])
                                                        + static_cast<size_t>(i)*4, 4);
                                            } else {
                                                const auto& src =
                                                    cfg.filter_column_data[c];
                                                std::memcpy(&v,
                                                    src.fixed_data.data() + r*4, 4);
                                            }
                                            tcard->add_numeric(c, static_cast<double>(v));
                                        }
                                    }
                                }
                                // Append to per-thread cluster buffers (no
                                // flush — flushing stays serial in the merge).
                                for (uint32_t c : chunk_targets[i]) {
                                    auto& tb = tbufs[c];
                                    if (!keep_raw_vecs) {
                                        tb.codes.insert(tb.codes.end(),
                                                        chunk_codes[i].begin(),
                                                        chunk_codes[i].end());
                                    }
                                    if (ctx.has_ip_bias)
                                        tb.ip_biases.push_back(chunk_biases[i]);
                                    // Always store the raw FP16 vector: it is
                                    // the source for the incremental centroid
                                    // accumulated during the serial merge.
                                    tb.fp16_vecs.insert(tb.fp16_vecs.end(),
                                                        fvec, fvec + dim);
                                    tb.row_ids.push_back(rid);
                                    tb.local_indices.push_back(i);
                                    if (has_payload) {
                                        uint32_t plen; const uint8_t* pdata;
                                        if (chunk_has_payload) {
                                            plen = chunk_poffsets[i + 1]
                                                 - chunk_poffsets[i];
                                            pdata = chunk_pdata + chunk_poffsets[i];
                                        } else {
                                            const uint32_t r =
                                                static_cast<uint32_t>(rid);
                                            plen = cfg.payload_offsets[r + 1]
                                                 - cfg.payload_offsets[r];
                                            pdata = cfg.payload_data
                                                  + cfg.payload_offsets[r];
                                        }
                                        tb.payload_lens.push_back(plen);
                                        tb.payload_ptrs.push_back(pdata);
                                    }
                                }
                            }
                        }, start, end, t));
                }
                for (auto& fut : afuts) fut.get();
            }

            // Serial merge + flush: for each cluster, replay the per-thread
            // buffers into the shared buffers in global order and flush at
            // leaf_cap overflow. Because sharding is contiguous, iterating
            // thread 0..n_threads-1 reproduces the original global vector
            // order, so every leaf gets identical contents. flush_buffer
            // (disk + shared metadata) stays serial here.
            for (uint32_t c = 0; c < k_root; ++c) {
                for (uint32_t t = 0; t < n_threads; ++t) {
                    const auto& tb = thread_buffers[t][c];
                    const uint32_t m = static_cast<uint32_t>(tb.row_ids.size());
                    if (m == 0) continue;
                    for (uint32_t j = 0; j < m; ++j) {
                        auto& buf = buffers[c];
                        if (keep_raw_vecs) {
                            // local families: keep raw FP16 for
                            // per-leaf fitting at flush time.
                            buf.fp16_vecs.insert(buf.fp16_vecs.end(),
                                tb.fp16_vecs.data() + static_cast<size_t>(j) * dim,
                                tb.fp16_vecs.data() + static_cast<size_t>(j + 1) * dim);
                        } else {
                            buf.codes.insert(buf.codes.end(),
                                tb.codes.data() + static_cast<size_t>(j) * code_size,
                                tb.codes.data() + static_cast<size_t>(j + 1) * code_size);
                        }
                        if (ctx.has_ip_bias)
                            buf.ip_biases.push_back(tb.ip_biases[j]);
                        buf.row_ids.push_back(tb.row_ids[j]);
                        // Incremental centroid accumulator (from raw FP16).
                        const float16_t* fv = tb.fp16_vecs.data()
                            + static_cast<size_t>(j) * dim;
                        for (uint16_t d = 0; d < dim; ++d)
                            buf.centroid_sum[d] += static_cast<float>(fv[d]);
                        ++buf.centroid_count;
                        if (has_filter)
                            append_filter_row(buf.filter_cols, tb.row_ids[j],
                                              tb.local_indices[j]);
                        if (has_payload) {
                            buf.payload_lens.push_back(tb.payload_lens[j]);
                            buf.payload_ptrs.push_back(tb.payload_ptrs[j]);
                        }
                        if (buf.row_ids.size() >= leaf_cap)
                            flush_buffer(c);
                    }
                }
            }
            // Fold the per-thread cardinality tables into the global table.
            if (has_filter) {
                for (uint32_t t = 0; t < n_threads; ++t)
                    ctx.card_table.merge_from(thread_cards[t]);
            }
            offset += take;
            const uint64_t million = offset / 1'000'000;
            if (million != prev_million) {
                prev_million = million;
                uint64_t flushed = 0;
                for (uint32_t c = 0; c < k_root; ++c)
                    flushed += n_leaves_in_cluster[c];
                spdlog::info("[sextant] build_streaming_pca: {}M/{}M streamed, "
                             "{} leaves", million, n / 1'000'000,
                             flushed);
            }
        }
    }
    for (uint32_t c = 0; c < k_root; ++c) flush_buffer(c);

    ctx.n_leaves_total = static_cast<uint32_t>(leaf_metas.size());
    // Depth-1 short-circuit: when the actual leaf count fits within k_root,
    // the root's children point directly to leaves (no internal L2 nodes).
    // This is determined late because leaves are flushed dynamically during
    // the streaming emission pass, so n_leaves is only known now.
    if (ctx.n_leaves_total <= k_root) {
        ctx.depth = 1;
    }
    spdlog::info("[sextant] build_streaming_pca: streamed {} vectors, {} leaves "
                 "in {:.2f}s", n, ctx.n_leaves_total,
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_stream).count());
    ctx.metrics.stop(m_stream);
}

// ---------------------------------------------------------------------------
// Phase 5: write the tree structure above the leaves. Writes the per-fine-
// centroid L2 nodes (depth >= 2), the L1 nodes (depth-3), the root node, and
// the codebook / PCA / cardinality / config blobs, then commits the superblock.
// Returns the BuildResult.
// ---------------------------------------------------------------------------
BuildResult write_tree_structure(TreeBuildContext& ctx, PageFile& file,
                                 PageAllocator& alloc) {
    const auto& cfg = ctx.cfg;
    const auto& params = cfg.params;
    const Dim dim = ctx.dim;
    const uint16_t m4 = ctx.m4;
    const uint8_t scan_bits = ctx.scan_bits;
    const uint32_t leaf_cap = ctx.leaf_cap;
    const uint32_t k_root = ctx.k_root;
    const uint32_t k_l1 = ctx.k_l1;
    const uint16_t depth = ctx.depth;
    const uint32_t pca_dims = ctx.pca_dims;
    const bool pca_disabled = ctx.pca_disabled;
    const uint32_t summary_size = ctx.summary_size;
    const uint32_t n_leaves_total = ctx.n_leaves_total;
    const auto& mean = ctx.mean;
    const auto& rotation = ctx.rotation;
    const auto& mean_proj = ctx.mean_proj;
    const auto& root_centroids_pca = ctx.root_centroids_pca;
    const auto& super_group = ctx.super_group;
    const auto& super_centroids_pca = ctx.super_centroids_pca;
    const auto& root_centroids_fp16 = ctx.root_centroids_fp16;
    auto& leaf_metas = ctx.leaf_metas;
    const auto& root_to_leaves = ctx.root_to_leaves;
    const auto& leaf_summaries = ctx.leaf_summaries;
    auto& card_table = ctx.card_table;

    // --- 7. Write tree file: internal nodes, root, codebook, config, blobs ---
    // All leaf extents (+ payloads) were written directly to `file` at flush
    // time. The PageFile + PageAllocator are already initialized. This phase
    // only writes the tree structure above the leaves.
    // root_to_leaves[c][j] already holds the global leaf_metas index of cluster
    // c's j-th leaf (assigned at flush time, in flush/arrival order).
    const auto t_write = std::chrono::steady_clock::now();
    auto m_write = ctx.metrics.start("write");

    // Write the per-fine-centroid internal nodes (depth=2: these are L1 nodes
    // that the root points to; depth=3: these are L2 nodes grouped under L1).
    // Skipped for depth=1: root children point directly to leaves.
    struct L2PageInfo { PageId page; uint32_t pages; };
    std::vector<L2PageInfo> l2_pages(k_root, {kInvalidPage, 0});
    // Per-L2-node summary (OR of all child leaf summaries) for propagation
    // to L1/root. Indexed by fine centroid id c.
    std::vector<std::vector<uint8_t>> l2_summaries(k_root);

    if (depth >= 2) {
    for (uint32_t c = 0; c < k_root; ++c) {
        const auto& leaves = root_to_leaves[c];
        if (leaves.empty()) continue;  // empty group → no node
        const uint32_t nch = static_cast<uint32_t>(leaves.size());
        const uint32_t npg = node_extent_pages(dim, nch, summary_size);
        const PageId page = alloc.alloc_extent(file, npg);
        std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);
        auto* nh = reinterpret_cast<TreeNodeHeader*>(buf.data());
        nh->magic = kTreeNodeMagic;
        nh->n_children = nch; nh->extent_pages = npg; nh->dim = dim;
        nh->magic_pad = 0;
        const uint32_t cesize = child_entry_size(dim, summary_size);
        // The summary region sits after the inline FP16 centroid.
        const uint32_t summary_off = sizeof(ChildEntry) + dim * sizeof(float16_t);
        uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
        for (uint32_t j = 0; j < nch; ++j) {
            const uint32_t li = leaves[j];
            auto* ce = reinterpret_cast<ChildEntry*>(p);
            // Store leaf_id (index into leaf_table_) — not the raw page.
            // The leaf table resolves leaf_id → physical page at search time.
            ce->child_page = li;
            ce->child_pages = leaf_metas[li].pages;
            ce->is_leaf = 1;
            float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
            std::memcpy(cent, leaf_metas[li].centroid.data(), dim*sizeof(float16_t));
            // Propagate the child leaf's summary into the child entry.
            if (summary_size > 0 && li < leaf_summaries.size() &&
                !leaf_summaries[li].empty()) {
                std::memcpy(p + summary_off, leaf_summaries[li].data(),
                            summary_size);
            }
            p += cesize;
        }
        // Compute this L2 node's own summary (OR of all child leaf summaries)
        // for propagation to L1/root.
        if (summary_size > 0) {
            l2_summaries[c].resize(summary_size);
            init_empty_summary(l2_summaries[c].data(), summary_size,
                               cfg.filter_schema);
            for (uint32_t j = 0; j < nch; ++j) {
                const uint32_t li = leaves[j];
                if (li < leaf_summaries.size() && !leaf_summaries[li].empty())
                    merge_filter_summary(l2_summaries[c].data(),
                                         leaf_summaries[li].data(),
                                         summary_size);
            }
        }
        nh->header_crc = header_crc(nh, offsetof(TreeNodeHeader, header_crc));
        file.write_pages(page, npg, buf.data());
        l2_pages[c] = {page, npg};
    }
    }  // depth >= 2

    // Build the root's child list. For depth=2 the root points directly to the
    // per-fine-centroid nodes (L1). For depth=3 we insert an extra level: each
    // super-group becomes an L1 node whose children are the L2 nodes in that
    // group, and the root points to the k_l1 L1 nodes.
    std::vector<RootChildData> root_child_data;
    // Per-L1-node summary (OR of all child L2 summaries), indexed by group g.
    // Only populated for depth=3; consumed by the root write loop.
    std::vector<std::vector<uint8_t>> l1_summaries;
    if (depth == 1) {
        // Root children point directly to leaves. Each root cluster maps to at
        // most one leaf (n_leaves <= k_root). Reuse l2_summaries (indexed by
        // fine centroid c) to carry each leaf's summary into the root write.
        root_child_data.resize(k_root);
        if (summary_size > 0) l2_summaries.resize(k_root);
        for (uint32_t c = 0; c < k_root; ++c) {
            root_child_data[c].centroid = root_centroids_fp16[c];
            if (root_to_leaves[c].empty()) {
                root_child_data[c].is_leaf = 1;
                root_child_data[c].page = kInvalidPage;
                root_child_data[c].pages = 0;
                continue;
            }
            // Point to the (single) leaf in this cluster.
            // Store leaf_id — the leaf table resolves it at search time.
            const uint32_t leaf_idx = root_to_leaves[c][0];
            root_child_data[c].is_leaf = 1;
            root_child_data[c].page = leaf_idx;
            root_child_data[c].pages = leaf_metas[leaf_idx].pages;
            if (summary_size > 0 && leaf_idx < leaf_summaries.size() &&
                !leaf_summaries[leaf_idx].empty()) {
                l2_summaries[c].assign(leaf_summaries[leaf_idx].begin(),
                                       leaf_summaries[leaf_idx].end());
            }
        }
    } else if (depth == 3) {
        // Group fine centroids (L2 nodes) by super-group, preserving ascending
        // fine-centroid id within each group.
        std::vector<std::vector<uint32_t>> groups(k_l1);
        for (uint32_t c = 0; c < k_root; ++c) groups[super_group[c]].push_back(c);

        // Project each super-centroid back to FP16 original space (inline root
        // child centroid; routing at the root uses PCA, but the tree stores an
        // FP16 centroid in the ChildEntry).
        auto project_pca_to_fp16 = [&](const std::vector<float>& pca,
                                       std::vector<float16_t>& out) {
            out.resize(dim);
            for (uint16_t d = 0; d < dim; ++d) {
                double val = mean[d];
                for (uint32_t k = 0; k < pca_dims; ++k)
                    val += pca[k] * rotation[k * dim + d];
                out[d] = static_cast<float16_t>(val);
            }
        };

        root_child_data.resize(k_l1);
        if (summary_size > 0) l1_summaries.resize(k_l1);
        const uint32_t l1_summary_off =
            sizeof(ChildEntry) + dim * sizeof(float16_t);
        for (uint32_t g = 0; g < k_l1; ++g) {
            project_pca_to_fp16(super_centroids_pca[g],
                                root_child_data[g].centroid);
            // Collect the non-empty L2 nodes in this group.
            std::vector<uint32_t> members;
            for (uint32_t c : groups[g]) {
                if (l2_pages[c].page != kInvalidPage) members.push_back(c);
            }
            if (members.empty()) {
                root_child_data[g].is_leaf = 0;
                root_child_data[g].page = kInvalidPage;
                root_child_data[g].pages = 0;
                continue;
            }
            const uint32_t nch = static_cast<uint32_t>(members.size());
            const uint32_t npg = node_extent_pages(dim, nch, summary_size);
            const PageId page = alloc.alloc_extent(file, npg);
            std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);
            auto* nh = reinterpret_cast<TreeNodeHeader*>(buf.data());
            nh->magic = kTreeNodeMagic;
            nh->n_children = nch; nh->extent_pages = npg; nh->dim = dim;
            nh->magic_pad = 0;
            const uint32_t cesize = child_entry_size(dim, summary_size);
            uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
            for (uint32_t c : members) {
                auto* ce = reinterpret_cast<ChildEntry*>(p);
                ce->child_page = l2_pages[c].page;
                ce->child_pages = l2_pages[c].pages;
                ce->is_leaf = 0;  // L2 node → internal
                float16_t* cent = reinterpret_cast<float16_t*>(
                    p + sizeof(ChildEntry));
                std::memcpy(cent, root_centroids_fp16[c].data(),
                            dim * sizeof(float16_t));
                // Propagate the child L2 node's summary into the child entry.
                if (summary_size > 0 && c < l2_summaries.size() &&
                    !l2_summaries[c].empty()) {
                    std::memcpy(p + l1_summary_off, l2_summaries[c].data(),
                                summary_size);
                }
                p += cesize;
            }
            // Compute this L1 node's own summary (OR of all child L2
            // summaries) for propagation to the root.
            if (summary_size > 0) {
                l1_summaries[g].resize(summary_size);
                init_empty_summary(l1_summaries[g].data(), summary_size,
                                   cfg.filter_schema);
                for (uint32_t c : members) {
                    if (c < l2_summaries.size() && !l2_summaries[c].empty())
                        merge_filter_summary(l1_summaries[g].data(),
                                             l2_summaries[c].data(),
                                             summary_size);
                }
            }
            nh->header_crc = header_crc(nh, offsetof(TreeNodeHeader, header_crc));
            file.write_pages(page, npg, buf.data());
            root_child_data[g].is_leaf = 0;
            root_child_data[g].page = page;
            root_child_data[g].pages = npg;
        }
    } else {
        // depth=2: root points directly to the per-fine-centroid nodes.
        root_child_data.resize(k_root);
        for (uint32_t c = 0; c < k_root; ++c) {
            root_child_data[c].centroid = root_centroids_fp16[c];
            root_child_data[c].is_leaf = 0;
            root_child_data[c].page = l2_pages[c].page;
            root_child_data[c].pages = l2_pages[c].pages;
        }
    }

    // Root node: depth=3 → k_l1 children; depth=2 → k_root children.
    const uint32_t root_n_children = static_cast<uint32_t>(root_child_data.size());
    const uint32_t root_npg = node_extent_pages(dim, root_n_children, summary_size);
    const PageId root_page = alloc.alloc_extent(file, root_npg);
    {
        std::vector<uint8_t> buf(static_cast<size_t>(root_npg) * kPageSize, 0);
        auto* rh = reinterpret_cast<TreeNodeHeader*>(buf.data());
        rh->magic = kTreeNodeMagic;
        rh->n_children = root_n_children; rh->extent_pages = root_npg;
        rh->dim = dim; rh->magic_pad = 0;
        const uint32_t cesize = child_entry_size(dim, summary_size);
        const uint32_t root_summary_off =
            sizeof(ChildEntry) + dim * sizeof(float16_t);
        uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
        for (uint32_t c = 0; c < root_n_children; ++c) {
            auto* ce = reinterpret_cast<ChildEntry*>(p);
            ce->child_page = root_child_data[c].page;
            ce->child_pages = root_child_data[c].pages;
            ce->is_leaf = root_child_data[c].is_leaf;
            float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
            std::memcpy(cent, root_child_data[c].centroid.data(),
                        dim * sizeof(float16_t));
            // Propagate the child subtree's summary into the root child entry.
            // depth=3 → child is an L1 node (summary indexed by group c).
            // depth=2 → child is an L2 node (summary indexed by fine centroid c).
            if (summary_size > 0) {
                const auto* src = (depth == 3)
                    ? ((c < l1_summaries.size() && !l1_summaries[c].empty())
                          ? l1_summaries[c].data() : nullptr)
                    : ((c < l2_summaries.size() && !l2_summaries[c].empty())
                          ? l2_summaries[c].data() : nullptr);
                if (src) std::memcpy(p + root_summary_off, src, summary_size);
            }
            p += cesize;
        }
        rh->header_crc = header_crc(rh, offsetof(TreeNodeHeader, header_crc));
        file.write_pages(root_page, root_npg, buf.data());
    }

    // Serialize the global codebook. local_pq has no global codebook (each
    // leaf carries its own); skip the extent entirely. scalar_lloydmax stores
    // its levels table here (same size-prefixed format).
    PageId cb_page = kInvalidPage;
    uint32_t cb_npg = 0;
    if (ctx.coder->has_global_state()) {
        std::vector<uint8_t> qblob;
        ctx.coder->serialize_global(qblob);
        const uint64_t qbsz = qblob.size();
        std::vector<uint8_t> qbs(sizeof(qbsz) + qblob.size());
        std::memcpy(qbs.data(), &qbsz, sizeof(qbsz));
        std::memcpy(qbs.data() + sizeof(qbsz), qblob.data(), qblob.size());
        cb_npg = static_cast<uint32_t>((qbs.size()+kPageSize-1)/kPageSize);
        cb_page = alloc.alloc_extent(file, cb_npg);
        std::vector<uint8_t> b(cb_npg*kPageSize, 0);
        std::memcpy(b.data(), qbs.data(), qbs.size());
        file.write_pages(cb_page, cb_npg, b.data());
    }

    TreeManifest manifest;
    manifest.dim = dim; manifest.m4 = m4; manifest.scan_pq_bits = scan_bits;
    manifest.quantizer_type = params.quantizer_type;
    manifest.prq_nsplits = ctx.coder->prq_nsplits();
    manifest.metric = static_cast<uint8_t>(params.metric);
    manifest.depth = depth; manifest.k_root = k_root; manifest.k_l1 = k_l1;
    manifest.leaf_capacity = leaf_cap; manifest.n_leaves = n_leaves_total;
    manifest.n_probe_l0 = cfg.n_probe_l0 > 0 ? cfg.n_probe_l0
        : static_cast<uint32_t>(std::max(1.0, 2.0*std::sqrt(double(k_root))));
    // n_probe_ln default: cover ALL leaves of a probed root child (max
    // per-child leaf count), not a hardcoded 4. A stale ln below the
    // real leaves-per-child silently truncates probing and masquerades
    // as a routing regression — measured 17pp containment loss on a
    // 9888-leaf cohere-10M tree (ln=8 vs ~10 leaves/child). Experts can
    // still cap explicitly via BuildConfig; the fraction path is
    // unaffected (it already probes all leaves of selected children).
    {
        uint32_t max_leaves_per_child = 1;
        for (uint32_t c = 0; c < root_to_leaves.size(); ++c)
            max_leaves_per_child = std::max(max_leaves_per_child,
                static_cast<uint32_t>(root_to_leaves[c].size()));
        manifest.n_probe_ln = cfg.n_probe_ln > 0 ? cfg.n_probe_ln
                                                 : max_leaves_per_child;
    }
    // Corpus-fraction probe budget: the scale-stable default. New trees
    // persist 0.5 (measured ~0.99 recall@10 across 100K→933K at half the
    // flat-scan cost); legacy count fields remain for expert overrides.
    manifest.probe_fraction = cfg.probe_fraction > 0.0f ? cfg.probe_fraction : 0.5f;
    manifest.adaptive_probe_gap = cfg.adaptive_probe_gap;
    manifest.median_lid = cfg.median_lid;
    manifest.pca_dims = ctx.pca_disabled ? 0 : pca_dims;  // search-side PCA
    manifest.balance_factor = params.partition_balance_factor;
    manifest.schema = cfg.filter_schema;
    manifest.summary_size = summary_size;

    // --- Write PCA routing blob ---
    // Layout: [pca_dims:u32][proj:pca_dims×dim f32][mean_proj:pca_dims f32]
    //         [root_centroids:k_root×pca_dims f32]
    //         [leaf_centroids:n_leaves×pca_dims f32]
    PageId pca_page = kInvalidPage;
    uint32_t pca_npg = 0;
    if (pca_dims > 0 && !pca_disabled) {
        std::vector<float> pca_blob;
        // Projection matrix (pca_dims × dim).
        for (uint32_t k = 0; k < pca_dims; ++k)
            for (uint16_t d = 0; d < dim; ++d)
                pca_blob.push_back(rotation[k * dim + d]);
        // Mean projection (pca_dims).
        for (uint32_t k = 0; k < pca_dims; ++k)
            pca_blob.push_back(mean_proj[k]);
        // Root centroids in PCA space.
        // depth>=3: k_l1 super-cluster centroids (root branching = k_l1).
        // depth<=2: k_root fine centroids (root branching = k_root).
        const uint32_t n_root_cents = (depth >= 3) ? k_l1 : k_root;
        for (uint32_t c = 0; c < n_root_cents; ++c) {
            const auto& src = (depth >= 3) ? super_centroids_pca[c]
                                           : root_centroids_pca[c];
            for (uint32_t k = 0; k < pca_dims; ++k)
                pca_blob.push_back(src[k]);
        }
        // Leaf centroids in PCA space (n_leaves × pca_dims).
        for (uint32_t l = 0; l < n_leaves_total; ++l) {
            // Project the leaf's original-space FP16 centroid to PCA space.
            for (uint32_t k = 0; k < pca_dims; ++k) {
                double acc = 0.0;
                for (uint16_t d = 0; d < dim; ++d)
                    acc += rotation[k * dim + d] *
                           static_cast<float>(leaf_metas[l].centroid[d]);
                pca_blob.push_back(static_cast<float>(acc - mean_proj[k]));
            }
        }

        const size_t blob_bytes = pca_blob.size() * sizeof(float);
        pca_npg = static_cast<uint32_t>((blob_bytes + kPageSize - 1) / kPageSize);
        pca_page = alloc.alloc_extent(file, pca_npg);
        std::vector<uint8_t> b(pca_npg * kPageSize, 0);
        std::memcpy(b.data(), pca_blob.data(), blob_bytes);
        file.write_pages(pca_page, pca_npg, b.data());
    }

    // --- Write the cardinality table blob (Phase D) ---
    PageId card_page = kInvalidPage;
    uint32_t card_npg = 0;
    if (!cfg.filter_schema.empty() && !card_table.empty()) {
        auto card_blob = card_table.serialize();
        const uint64_t card_bytes = card_blob.size();
        card_npg = static_cast<uint32_t>((card_bytes + kPageSize - 1) / kPageSize);
        if (card_npg > 0) {
            card_page = alloc.alloc_extent(file, card_npg);
            std::vector<uint8_t> b(card_npg * kPageSize, 0);
            std::memcpy(b.data(), card_blob.data(), card_bytes);
            file.write_pages(card_page, card_npg, b.data());
        }
    }

    // --- Write the leaf extent table blob ---
    // Layout: [n_entries:u64][n_entries × LeafTableEntry]
    // Every tree gets a leaf table (even without filter columns) — it's the
    // indirection layer for leaf closure and dynamic insert/delete.
    PageId lt_page = kInvalidPage;
    uint32_t lt_npg = 0;
    {
        const uint64_t n_entries = leaf_metas.size();
        std::vector<uint8_t> lt_blob;
        lt_blob.resize(8 + n_entries * sizeof(LeafTableEntry), 0);
        std::memcpy(lt_blob.data(), &n_entries, 8);
        for (uint64_t i = 0; i < n_entries; ++i) {
            LeafTableEntry e{leaf_metas[i].page, leaf_metas[i].pages};
            std::memcpy(lt_blob.data() + 8 + i * sizeof(LeafTableEntry),
                        &e, sizeof(e));
        }
        lt_npg = static_cast<uint32_t>(
            (lt_blob.size() + kPageSize - 1) / kPageSize);
        lt_page = alloc.alloc_extent(file, lt_npg);
        std::vector<uint8_t> b(lt_npg * kPageSize, 0);
        std::memcpy(b.data(), lt_blob.data(), lt_blob.size());
        file.write_pages(lt_page, lt_npg, b.data());
    }

    std::string cfg_toml = manifest_to_toml(manifest);
    const uint32_t cfg_npg = static_cast<uint32_t>((cfg_toml.size()+kPageSize-1)/kPageSize);
    const PageId cfg_page = alloc.alloc_extent(file, cfg_npg);
    { std::vector<uint8_t> b(cfg_npg*kPageSize, 0); std::memcpy(b.data(),cfg_toml.data(),cfg_toml.size());
      file.write_pages(cfg_page, cfg_npg, b.data()); }

    alloc.flush_bitmap(file);
    Superblock sb;
    sb.init_fresh(alloc.bitmap_page(), alloc.bitmap_pages());
    sb.set_root(root_page, root_npg);
    sb.set_depth(depth);
    sb.set_n_leaves(n_leaves_total);
    sb.set_n_pages(file.num_pages());
    sb.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    sb.set_bitmap(alloc.bitmap_page(), alloc.bitmap_pages());
    sb.set_codebook(cb_page, cb_npg);
    sb.set_config(cfg_page, cfg_npg);
    sb.set_pca(pca_page, pca_npg);
    sb.set_cardinality(card_page, card_npg);
    sb.set_leaf_table(lt_page, lt_npg);
    sb.commit(file);
    file.sync();

    BuildResult result;
    result.index_path = ctx.output_path;
    result.n_vectors = ctx.n;
    result.dim = dim;
    result.pq_m = m4;
    result.pq_bits = scan_bits;
    result.build_time_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_write).count();
    ctx.metrics.stop(m_write);
    return result;
}

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
        // Plane stage-1 (predicate-free queries only in v1 — see
        // plane_route_). Feedback probing is descent-based and not
        // meaningful under plane routing.
        const bool plane_active = plane_ != nullptr && config.use_plane &&
                                  config.predicates.empty();
        RouteStatus rstatus;
        if (plane_active) {
            scratch.route_feedback = false;
            scratch.selectivity = 1.0f;
            scratch.route_gap = 0.0f;
            scratch.root_dists.clear();
            rstatus = RouteStatus::Ok;
            plane_route_(query, config, scratch.candidates);
            if (scratch.candidates.empty()) rstatus = RouteStatus::Empty;
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
        for (uint32_t i = 0; i < nq; ++i)
            if (qs[i].predicates->empty()) plane_q.push_back(i);
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
            const auto tp0 = std::chrono::steady_clock::now();
            plane_route_batch_(compact.data(),
                               static_cast<uint32_t>(plane_q.size()),
                               config, pc);
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
                    for (uint32_t ri = ul.ref_begin; ri < ref_end; ++ri) {
                        const auto& fr = refs[ri];
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
                        if (per_leaf) coder_->bind_leaf(*p->setup, leaf_ptr);
                        RawScanHeap rh{&p->heap, s.W, fr.slot};
                        coder_->scan_leaf(*p->setup, leaf_ptr, rh);
                        // Harvest: resolve everything finalize needs from
                        // THIS leaf while the buffer is valid. Entries are
                        // admitted to the W-bounded pool only if they beat
                        // its worst slot (cheap pq-dist compare first;
                        // row-id/rerank/predicate work only on admission).
                        {
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
        if (T == 1) {
            sweep_worker();
        } else {
            std::vector<std::future<void>> futs;
            for (uint32_t t = 0; t < T; ++t)
                futs.push_back(
                    std::async(std::launch::async, sweep_worker));
            for (auto& f : futs) f.get();
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
// ===========================================================================
// Routing diagnostics — loss-decomposition harness support.
// ===========================================================================

// ===========================================================================
// Routing plane attach (plane.hpp). Post-build pass: trains on a
// spread-sampled covariance, encodes members in ROW order (sequential
// base reads — leaf-order member passes are random-access page-fault
// storms at 10M scale), writes the extent, re-commits the superblock.
// ===========================================================================
// ===========================================================================
// Plane stage-1 routing (see declaration for the contract).
// ===========================================================================
void IVFTreeIndex::plane_route_(const float* query,
                                const SearchConfig& config,
                                std::vector<LeafCandidate>& candidates)
    const {
    std::vector<std::vector<LeafCandidate>> out(1);
    plane_route_batch_(query, 1, config, out);
    candidates = std::move(out[0]);
}

// ===========================================================================
// Batched plane stage-1 (the deployment shape): ONE leaf-major sweep over
// the plane per batch — each plane block is read from the page cache /
// NAND exactly once and scored against EVERY query in the batch while
// hot. Single-query search() pays the full plane read per query and is
// DRAM-bound (~50 GB/s ceiling measured); the batch amortizes the read
// across `nq` queries and turns stage-1 compute-bound.
// ===========================================================================
void IVFTreeIndex::plane_route_batch_(
        const float* queries, uint32_t nq, const SearchConfig& config,
        std::vector<std::vector<LeafCandidate>>& out) const {
    const uint32_t R = plane_->meta().rank;
    const uint32_t n_leaves = static_cast<uint32_t>(leaf_table_.size());
    const PlaneEncoding enc = plane_->meta().encoding;
    const bool is_b1g = enc == PlaneEncoding::B1G;
    const bool fastscan8 = enc == PlaneEncoding::U4LM;

    // Per-query projections + LUTs.
    std::vector<float> proj(static_cast<size_t>(nq) * R);
    std::vector<uint8_t> lut8;
    std::vector<float> lut;
    std::vector<float> blut;  // b1g byte-LUTs (16 KB/query)
    std::vector<float> seg_min(R);
    std::vector<float> so_scratch(R);
    if (fastscan8)
        lut8.resize(static_cast<size_t>(nq) * R * 16);
    if (!fastscan8 && !is_b1g)
        lut.resize(static_cast<size_t>(nq) * R * 16);
    if (is_b1g)
        blut.resize(static_cast<size_t>(nq) * (R / 8) * 256);
    for (uint32_t qi = 0; qi < nq; ++qi) {
        float* pr = &proj[static_cast<size_t>(qi) * R];
        plane_->project_query(queries + static_cast<size_t>(qi) *
                                manifest_.dim, pr);
        if (is_b1g)
            plane_->build_b1_lut(
                pr, &blut[static_cast<size_t>(qi) * (R / 8) * 256]);
        else if (fastscan8)
            plane_->build_lut8(
                pr, &lut8[static_cast<size_t>(qi) * R * 16],
                so_scratch.data(), so_scratch.data(), seg_min.data());
        else
            plane_->build_lut(pr, &lut[static_cast<size_t>(qi) * R * 16]);
    }

    // Leaf-major sweep: score EVERY query against each leaf's blocks
    // while they are cache-hot (the block read is the batch-shared part).
    // scores layout [l * nq + qi] — LEAF-major: each thread's writes for
    // a leaf are one contiguous run (query-major puts adjacent leaves in
    // the same cache line across threads = false-sharing ping-pong,
    // measured 6x slower). u8 accumulators stay raw (ranking-monotone
    // per query).
    // FLOAT scores: b1g/u4lm_pv scores are signed (sums of +/-w); a
    // uint32 cast wraps negatives to ~4e9 and they win every sort (the
    // u8 raw accumulators are positive-monotone, cast is lossless for
    // ranking).
    std::vector<float> scores(static_cast<size_t>(n_leaves) * nq, 0);
    // Threading: batch mode parallelizes the sweep over leaf chunks with
    // the engine's std::async worker pattern (the engine does NOT build
    // with OpenMP — pragmas are silent no-ops on x86, measured: the
    // batched sweep ran single-threaded until this was noticed). For
    // nq == 1 the sweep stays SERIAL: single-query callers (CLI
    // tree-search) parallelize across queries themselves, and a full
    // team per query would oversubscribe.
    const uint32_t T = nq > 1
        ? std::max(1u, config.search_threads > 0
                           ? config.search_threads
                           : std::thread::hardware_concurrency())
        : 1u;
    auto sweep_worker = [&](uint64_t l0, uint64_t l1) {
        for (uint64_t l = l0; l < l1; ++l) {
            if (leaf_table_[static_cast<uint32_t>(l)].page == kInvalidPage)
                continue;
            float* row = &scores[static_cast<size_t>(l) * nq];
            for (uint32_t qi = 0; qi < nq; ++qi)
                row[qi] =
                is_b1g
                    ? plane_->scan_leaf_max(
                          static_cast<uint32_t>(l), nullptr,
                          &blut[static_cast<size_t>(qi) * (R / 8) * 256],
                          nullptr)
                    : (fastscan8
                           ? plane_->scan_leaf_max_u8(
                                 static_cast<uint32_t>(l),
                                 &lut8[static_cast<size_t>(qi) * R * 16])
                           : plane_->scan_leaf_max(
                                 static_cast<uint32_t>(l),
                                 &lut[static_cast<size_t>(qi) * R * 16],
                                 nullptr, nullptr));
        }
    };
    {
        const uint64_t per = (n_leaves + T - 1) / T;
        std::vector<std::future<void>> futs;
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t l0 = static_cast<uint64_t>(t) * per;
            const uint64_t l1 = std::min<uint64_t>(
                n_leaves, l0 + per);
            if (l0 >= l1) break;
            if (t == T - 1 || l1 == n_leaves) {
                sweep_worker(l0, l1);
                break;
            }
            futs.push_back(std::async(std::launch::async, sweep_worker,
                                      l0, l1));
        }
        for (auto& f : futs) f.get();
    }

    // Per-query selection under the page-weighted fraction budget.
    uint64_t total = 0;
    for (const auto& e : leaf_table_)
        if (e.page != kInvalidPage) total += e.pages;
    out.resize(nq);
    for (uint32_t qi = 0; qi < nq; ++qi) {
        const float* sc = &scores[qi];  // column of the [l][nq] grid
        std::vector<uint32_t> order(n_leaves);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) {
                      return sc[static_cast<size_t>(a) * nq] >
                             sc[static_cast<size_t>(b) * nq];
                  });
        float f = config.probe_fraction;
        if (f <= 0.0f) f = manifest_.probe_fraction;
        if (f <= 0.0f) f = 0.5f;
        auto& candidates = out[static_cast<size_t>(qi)];
        candidates.reserve(n_leaves);
        uint64_t cum = 0;
        for (uint32_t l : order) {
            const auto& e = leaf_table_[l];
            if (e.page == kInvalidPage) continue;
            candidates.push_back({e.page, e.pages,
                                  sc[static_cast<size_t>(l) * nq],
                                  nullptr});
            cum += e.pages;
            if (static_cast<double>(cum) >=
                static_cast<double>(f) * static_cast<double>(total) - 0.5)
                break;
        }
    }
}

void IVFTreeIndex::attach_plane(const float* base, uint32_t n, uint32_t dim,
                                PlaneEncoding enc, uint16_t rank,
                                uint32_t train_rows) {
    if (plane_) {
        throw Error(ErrorCode::InvalidParam,
            "attach_plane: index already carries a routing plane");
    }
    if (dim != manifest_.dim) {
        throw Error(ErrorCode::InvalidParam,
            "attach_plane: base dim does not match index dim");
    }
    // Fail-loud precondition (F3/F4 class): the bitmap must cover the
    // whole file before we allocate anything. Trees built before the
    // bitmap upper-bound fix have bitmaps that cover only a prefix —
    // the allocator would hand out LIVE tree pages for the plane extent
    // (measured: 185K pages of leaves overwritten on cohere-10m). Run
    // `sextant fsck --repair` on such trees first.
    if (static_cast<uint64_t>(superblock_.alloc_bitmap_pages()) *
            static_cast<uint64_t>(kPageSize) * 8 <
        superblock_.n_pages()) {
        throw Error(ErrorCode::CorruptIndex,
            "attach_plane: allocation bitmap covers only a prefix of the "
            "file (under-provisioned at build time) — run "
            "`sextant fsck --repair` on this index first");
    }

    const auto t0 = std::chrono::steady_clock::now();
    PlaneWriter::Config pcfg;
    pcfg.encoding = enc;
    pcfg.rank = rank;
    pcfg.train_rows = train_rows;
    PlaneWriter writer;
    writer.train(base, n, dim, pcfg);
    writer.prepare(static_cast<uint32_t>(leaf_table_.size()));
    spdlog::info("[sextant] plane: basis+codebooks trained in {:.1}s",
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count());

    // Members per leaf + row -> (leaf, slot) destinations. Leaf-ordered
    // reads of the row_ids are random-access: warm the file sequentially
    // with a background WILLNEED sweep while building the dest map.
    {
        const int pfd = fd_;
        struct stat pst;
        if (::fstat(pfd, &pst) == 0) {
            const auto psize = static_cast<uint64_t>(pst.st_size);
            std::thread([pfd, psize] {
                const uint64_t kChunk = 256u << 20;
                for (uint64_t off = 0; off < psize; off += kChunk) {
                    ::posix_fadvise(pfd, static_cast<off_t>(off),
                                    static_cast<off_t>(kChunk),
                                    POSIX_FADV_WILLNEED);
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(50));
                }
            }).detach();
        }
    }
    std::vector<uint32_t> leaf_counts(leaf_table_.size(), 0);
    std::unordered_map<int64_t, std::vector<std::pair<uint32_t, uint32_t>>>
        dest;  // row -> [(leaf, slot)]
    dest.reserve(static_cast<size_t>(n) / 3 * 4 + 1);
    for (uint32_t l = 0; l < leaf_table_.size(); ++l) {
        const auto mem = debug_leaf_row_ids(l);
        leaf_counts[l] = static_cast<uint32_t>(mem.size());
        for (uint32_t slot = 0; slot < mem.size(); ++slot)
            dest[static_cast<int64_t>(mem[slot])].emplace_back(l, slot);
    }

    // Row-order sequential encode pass (OMP over row blocks).
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int64_t r0 = 0; r0 < static_cast<int64_t>(n); ++r0) {
            auto it = dest.find(static_cast<int64_t>(r0));
            if (it == dest.end() || it->second.empty()) continue;
            const float* v = &base[static_cast<size_t>(r0) * dim];
            for (const auto& ls : it->second)
                writer.encode_member(ls.first, ls.second, v, nullptr);
        }
    }
    spdlog::info("[sextant] plane: encoded {} rows in {:.1}s", n,
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count());

    auto blob = writer.finalize(leaf_counts);

    // The bitmap region is FIXED at format time; a plane extent can
    // outgrow its coverage (10M-scale plane = ~180K pages). When it
    // would, relocate the bitmap: append a larger one at EOF, free the
    // old region's pages, update the superblock pointers.
    const uint32_t npg0 = static_cast<uint32_t>(
        (blob.size() + kPageSize - 1) / kPageSize);
    const uint64_t kBitsPerBitmapPage =
        static_cast<uint64_t>(kPageSize) * 8;
    {
        const uint64_t covered =
            static_cast<uint64_t>(superblock_.alloc_bitmap_pages()) *
            kBitsPerBitmapPage;
        const uint64_t est_total = superblock_.n_pages() + npg0 + 64;
        if (est_total > covered) {
            const uint32_t new_nbm = static_cast<uint32_t>(
                est_total / kBitsPerBitmapPage + 2);
            const PageId old_bm_page = superblock_.alloc_bitmap_page();
            const uint32_t old_nbm = superblock_.alloc_bitmap_pages();
            const PageId new_bm_page = file_.num_pages();
            std::vector<uint8_t> nbm(
                static_cast<size_t>(new_nbm) * kPageSize, 0);
            std::vector<uint8_t> obm(
                static_cast<size_t>(old_nbm) * kPageSize, 0);
            file_.read_pages(old_bm_page, old_nbm, obm.data());
            const size_t copy_bits = std::min<uint64_t>(
                superblock_.n_pages(),
                static_cast<uint64_t>(old_nbm) * kBitsPerBitmapPage);
            std::memcpy(nbm.data(), obm.data(), (copy_bits + 7) / 8);
            for (uint64_t p = new_bm_page;
                 p < new_bm_page + new_nbm; ++p) {
                nbm[p / 8] |= static_cast<uint8_t>(1u << (p % 8));
            }
            file_.truncate(new_bm_page + new_nbm);
            file_.write_pages(new_bm_page, new_nbm, nbm.data());
            const PageId old_flh = superblock_.free_list_head();
            const uint64_t old_nfree = superblock_.n_free_pages();
            superblock_.set_bitmap(new_bm_page, new_nbm);
            superblock_.set_n_pages(file_.num_pages());
            // Reload the allocator over the new geometry, then push the
            // old bitmap pages onto the free list (maintains the
            // persisted chain + bitmap bits).
            PageAllocator alloc2;
            alloc2.load(file_, new_bm_page, new_nbm,
                        file_.num_pages(), old_flh, old_nfree);
            alloc2.free_extent(file_, old_bm_page, old_nbm);
            alloc2.flush_bitmap(file_);
            superblock_.set_free_list(alloc2.free_list_head(),
                                      alloc2.n_free_pages());
            superblock_.commit(file_);
            spdlog::info("[sextant] plane: bitmap relocated {} -> {} pages "
                         "(@page {})", old_nbm, new_nbm, new_bm_page);
        }
    }

    // Allocate the extent, write, re-commit the superblock.
    PageAllocator alloc;
    alloc.load(file_, superblock_.alloc_bitmap_page(),
               superblock_.alloc_bitmap_pages(), superblock_.n_pages(),
               superblock_.free_list_head(), superblock_.n_free_pages());
    const uint32_t npg = static_cast<uint32_t>(
        (blob.size() + kPageSize - 1) / kPageSize);
    const PageId ppage = alloc.alloc_extent(file_, npg);
    std::vector<uint8_t> padded(static_cast<size_t>(npg) * kPageSize, 0);
    std::memcpy(padded.data(), blob.data(), blob.size());
    file_.write_pages(ppage, npg, padded.data());
    alloc.flush_bitmap(file_);

    superblock_.set_plane(ppage, npg);
    superblock_.set_n_pages(file_.num_pages());
    superblock_.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    superblock_.commit(file_);
    file_.sync();

    // Reload in-memory state: re-mmap (the plane pages may extend the
    // file) and parse + bind.
    ::munmap(const_cast<uint8_t*>(mmap_base_), mmap_size_);
    mmap_size_ = file_.num_pages() * kPageSize;
    void* addr = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED,
                        file_.fd(), 0);
    if (addr == MAP_FAILED) {
        throw Error(ErrorCode::IoError,
            "attach_plane: re-mmap failed: " +
                std::string(std::strerror(errno)));
    }
    mmap_base_ = static_cast<const uint8_t*>(addr);
    plane_ = PlaneIndex::parse(mmap_base_ + static_cast<uint64_t>(ppage) *
                              kPageSize,
                              static_cast<size_t>(npg) * kPageSize);
    if (!plane_) {
        throw Error(ErrorCode::CorruptIndex,
            "attach_plane: written plane failed to parse");
    }
    plane_->bind(leaf_counts);
    spdlog::info("[sextant] plane: attached ({} pages, {} B/vec) in {:.1}s",
                 npg, plane_->meta().bytes_per_vec(),
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count());
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
    const RowId* rids = reinterpret_cast<const RowId*>(
        leaf_ptr + coder_->geometry(lh).rowids_offset);
    return std::vector<RowId>(rids, rids + count);
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

// ===========================================================================
// PCA-preconditioned streaming build.
//
// Projects vectors onto top-k principal components before routing. On
// high-LID data (Sphere-IP d_eff≈2), the raw 768D space is near-isotropic —
// all distances look the same. PCA exposes the low-dimensional manifold, so
// k-means in PCA space converges cleanly and routing works.
//
// PCA computation: covariance matrix (dim×dim) from the 20k sample, eigendecompose
// via the existing compute_pca_rotation_public (Jacobi eigendecomposition).
// Projection: SIMD dot product per PC (simd::dot_f32), parallelized across
// threads during the streaming phase.
// ===========================================================================
// PCA-preconditioned streaming build.
//
// Projects vectors onto top-k principal components before routing. On
// high-LID data (Sphere-IP d_eff≈2), the raw 768D space is near-isotropic —
// all distances look the same. PCA exposes the low-dimensional manifold, so
// k-means in PCA space converges cleanly and routing works.
//
// PCA computation: covariance matrix (dim×dim) from the 20k sample, eigendecompose
// via the existing compute_pca_rotation_public (Jacobi eigendecomposition).
// Projection: SIMD dot product per PC (simd::dot_f32), parallelized across
// threads during the streaming phase.
// ============================================================================
//
// The build is split into five phases over a shared TreeBuildContext (see the
// anonymous namespace above): resolve params -> train quantizer + PCA -> Lloyd
// refinement -> emission pass -> tree write. This function is a thin orchestrator.

BuildResult IVFTreeIndex::build_streaming_pca(VectorSource& source,
                                                 const std::string& output_path,
                                                 const BuildConfig& cfg) {
    const auto t0 = std::chrono::steady_clock::now();

    // Metrics: default = human log line; cfg.metrics_sink (e.g. JSONL file)
    // fans out alongside. Source label from the path extension (closed
    // vocabulary for the metrics wire format).
    metrics::LogMetricsSink metrics_log_sink;
    metrics::MultiMetricsSink metrics_sink;
    metrics_sink.add(&metrics_log_sink);
    if (cfg.metrics_sink) metrics_sink.add(cfg.metrics_sink);
    const std::string src_label =
        source.path().size() >= 8 &&
                source.path().compare(source.path().size() - 8, 8, ".parquet") == 0
            ? "parquet"
            : "fbin";

    TreeBuildContext ctx(source, output_path, cfg);
    ctx.metrics = metrics::MetricsCollector(&source, &metrics_sink, src_label);
    auto m_total = ctx.metrics.start("total");

    resolve_build_params(ctx);
    train_quantizer_and_pca(ctx);
    run_lloyd_refinement(ctx);

    // The PageFile + PageAllocator are initialized before the emission pass so
    // leaves can be allocated + written at flush time. The file starts with
    // just the bitmap (page 2); leaves are allocated sequentially.
    //
    // The bitmap region is FIXED at format time (it cannot grow in place —
    // everything after it shifts). One bitmap page addresses 32768 file
    // pages (128 MiB); the old hard-coded bitmap_pages=1 silently marked
    // every page past 128 MiB FREE on disk (audit F3/F4: fsck orphan
    // storms on large pristine trees, insert/delete handing out live
    // leaf pages). Size it from a generous upper bound on the final page
    // count: worst-case fp16-per-vec payload + per-leaf centroid/codebook
    // overhead at the smallest legal leaf (leaf_cap/2 after a split).
    const PageId bitmap_page = 2;
    const uint64_t kPagesPerBitmapPage =
        static_cast<uint64_t>(kPageSize) * 8;
    const uint64_t vec_bytes =
        ctx.n * (2ull * ctx.dim + 24);  // fp16 vec + rowid/ip-bias/slack
    const uint64_t n_leaves_est =
        ctx.n / std::max<uint64_t>(1, ctx.leaf_cap / 2) + 2;
    const uint64_t leaf_overhead =
        n_leaves_est * (4ull * ctx.dim * 17 + 2 * kPageSize);
    const uint64_t upper_pages =
        (vec_bytes + leaf_overhead) / kPageSize + 64;
    const uint32_t bitmap_pages = static_cast<uint32_t>(
        upper_pages / kPagesPerBitmapPage + 2);
    PageFile file(output_path);
    PageAllocator alloc;
    file.truncate(bitmap_page + bitmap_pages);
    alloc.init(file, bitmap_page, bitmap_pages);

    run_emission_pass(ctx, file, alloc);
    BuildResult result = write_tree_structure(ctx, file, alloc);

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    spdlog::info("[sextant] build_streaming_pca complete: N={} depth={} k_root={} "
                 "k_l1={} n_leaves={} in {:.2f}s (file={} pages)",
                 ctx.n, ctx.depth, ctx.k_root, ctx.k_l1, ctx.n_leaves_total, secs,
                 file.num_pages());
    result.build_time_sec = secs;
    // Instrumentation totals + records (per-phase records were emitted as
    // they completed; "total" wraps the whole build, phases excluded from
    // the sum to avoid double counting).
    const metrics::PhaseMetrics sums = ctx.metrics.totals("total");
    result.cpu_time_sec = sums.cpu_seconds;
    result.source_wait_sec = sums.source_wait_seconds;
    result.bytes_read = sums.bytes_read;
    result.peak_rss_bytes = metrics::MetricsCollector::peak_rss_bytes();
    ctx.metrics.stop(m_total);
    result.phases = ctx.metrics.records();
    return result;
}

// ===========================================================================
// Brute-force PQ-decode fallback for extreme low selectivity (<1%).
// ===========================================================================

std::vector<Candidate> IVFTreeIndex::search_brute_force_filtered(
    const float* query, uint32_t k, const SearchConfig& config,
    const std::vector<uint32_t>& pred_col_indices,
    const std::vector<uint32_t>& geo_lng_col_indices,
    std::vector<std::pair<const uint8_t*, uint32_t>>* /*payload_locs*/) const {

    // Brute-force filtered search for the local / scalar families is not
    // yet implemented. This path is only triggered at extreme low
    // selectivity (<1%) with filter columns.
    if (coder_->family() != CoderFamily::GlobalPq) {
        spdlog::warn("[sextant] brute-force filtered search not yet supported "
                     "for {}; returning empty results",
                     coder_->family_name());
        return {};
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

            // Exact match — decode PQ code and compute distance.
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
