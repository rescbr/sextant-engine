#include "ivf_tree_index.hpp"

#include "engine/manifest_io.hpp"
#include "engine/partition.hpp"
#include "quant/pq_quantizer.hpp"
#include "quant/product_residual_quantizer.hpp"
#include "util/fp16.hpp"
#include "simd_kernels.hpp"
#include "tree/filter_column_write.hpp"  // filter column write path
#include "tree/filter_column_read.hpp"   // filter column read path (mutable ops)
#include "tree/filter_scan.hpp"         // filter predicate evaluation
#include "sextant/error.hpp"
#include "sextant/logging.hpp"
#include "sextant/vector_source.hpp"
#include "sextant/filter_column_data.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
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
    if (mmap_base_) {
        ::munmap(const_cast<uint8_t*>(mmap_base_), mmap_size_);
        mmap_base_ = nullptr;
    }
    // PageFile destructor closes the fd.
}

// ===========================================================================
// Open
// ===========================================================================

std::unique_ptr<IVFTreeIndex> IVFTreeIndex::open(const std::string& path) {
    auto idx = std::unique_ptr<IVFTreeIndex>(new IVFTreeIndex());
    idx->path_ = path;
    idx->file_ = PageFile(path);

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

    // Load the codebook from the codebook extent.
    {
        const auto cb_page = idx->superblock_.codebook_page();
        const auto cb_pages = idx->superblock_.codebook_pages();
        if (cb_page == kInvalidPage || cb_pages == 0) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFTreeIndex::open: no codebook in superblock");
        }
        std::vector<uint8_t> blob(static_cast<size_t>(cb_pages) * kPageSize);
        idx->file_.read_pages(cb_page, cb_pages, blob.data());

        // The codebook blob is prefixed with its serialized size (u64),
        // then page-padded. Read the size and pass only the real bytes.
        uint64_t qblob_size = 0;
        std::memcpy(&qblob_size, blob.data(), sizeof(qblob_size));
        const uint8_t* qblob_data = blob.data() + sizeof(qblob_size);

        // Deserialize the codebook. Construct the right quantizer type.
        const auto& m = idx->manifest_;
        if (m.quantizer_type == "prq") {
            idx->quantizer_ = std::make_unique<ProductResidualQuantizer>(
                MetricKind::L2Sq, m.dim, m.m4, m.scan_pq_bits,
                m.prq_nsplits, /*beam_size=*/1);
        } else {
            idx->quantizer_ = std::make_unique<PqQuantizer>(
                MetricKind::L2Sq, m.dim, m.m4, m.scan_pq_bits);
        }
        idx->quantizer_->deserialize(qblob_data, static_cast<size_t>(qblob_size));
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
    if (idx->superblock_.leaf_table_page() != kInvalidPage &&
        idx->superblock_.leaf_table_pages() > 0) {
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

    // Parse the root node from the mmap.
    idx->load_root_from_mmap();

    // If PCA routing: build leaf_id mapping for depth=2 trees.
    // The pca_leaf_centroids_ array is indexed by global leaf ID (the order
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
        // When the leaf table is present, is_leaf children store a leaf_id
        // (index into leaf_table_) instead of a direct page pointer. Resolve
        // it to the physical page here so the search path is unchanged.
        PageId leaf_id = ce->child_page;
        PageId page = leaf_id;
        uint64_t pages = ce->child_pages;
        if (ce->is_leaf && !leaf_table_.empty() &&
            leaf_id < leaf_table_.size()) {
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

// ===========================================================================
// Search
// ===========================================================================

namespace {

/// A max-heap entry for the FastScan candidate selection. Carries enough
/// state to decode the candidate's PQ code back to FP32 for reranking:
/// `leaf_ptr` (the mmap base of the leaf extent) and `local_idx` (the
/// vector's index within that leaf). `pq_dist` is the PQ-approximate
/// uint32 ADC distance used to order the heap during the scan.
struct HeapEntry {
    uint32_t pq_dist;
    int64_t  row_id;
    const uint8_t* leaf_ptr;   // mmap base of the leaf extent
    uint32_t local_idx;        // vector index within the leaf
};

/// Extract a standard packed PQ code from a leaf's FastScan block layout.
/// Re-packs the interleaved FastScan blocks back into the compact code
/// representation that `decode_code` consumes (lo nibble = even segment,
/// hi nibble = odd segment for 4-bit; one byte per segment for 8-bit).
/// `out` must hold at least `code_size` bytes.
inline void extract_code_from_leaf(const uint8_t* leaf_ptr, uint32_t local_idx,
                                   uint32_t summary_size, uint16_t m4,
                                   uint8_t pq_bits, uint32_t codes_per_block,
                                   uint32_t block_bytes, uint8_t* out) {
    const uint32_t block = local_idx / codes_per_block;
    const uint32_t slot  = local_idx % codes_per_block;
    const uint8_t* codes_base = leaf_ptr + leaf_codes_offset(summary_size);
    const uint8_t* blk = codes_base + static_cast<uint64_t>(block) * block_bytes;

    if (pq_bits == 8) {
        // 16 vectors/block; segment s code at blk[s * 16 + slot].
        const uint32_t byte_idx = slot;
        for (uint16_t s = 0; s < m4; ++s) out[s] = blk[s * 16 + byte_idx];
    } else {
        // 32 vectors/block; slot < 16 uses lo nibble, slot >= 16 uses hi.
        const uint32_t byte_idx = slot % 16;
        const bool is_hi = (slot >= 16);
        for (uint16_t s = 0; s < m4; ++s) {
            const uint8_t nib = is_hi
                ? static_cast<uint8_t>(blk[s * 16 + byte_idx] >> 4)
                : static_cast<uint8_t>(blk[s * 16 + byte_idx] & 0x0F);
            const uint32_t byte_off = s / 2;
            const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
            out[byte_off] = static_cast<uint8_t>(
                ((s % 2 == 0) ? (out[byte_off] & 0xF0u) : (out[byte_off] & 0x0Fu))
                | (nib << shift));
        }
    }
}

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
        : source(src), output_path(op), cfg(c) {}

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
    uint32_t pca_dims = 0;
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
    std::unique_ptr<PqQuantizer> quantizer;

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
    ctx.pca_dims = std::min(ctx.dim, cfg.pca_dims > 0 ? cfg.pca_dims : 32u);
    ctx.metric = params.metric;
}

// ---------------------------------------------------------------------------
// Phase 2: sample vectors, train the PQ quantizer, compute the PCA rotation
// (mean, projection, variance explained), run k-means in PCA space for the
// root centroids, and compute the closure epsilon. The trained quantizer is
// stored on ctx.quantizer.
// ---------------------------------------------------------------------------
void train_quantizer_and_pca(TreeBuildContext& ctx) {
    const auto& cfg = ctx.cfg;
    const auto& params = cfg.params;
    const auto n = ctx.n;
    const Dim dim = ctx.dim;
    const uint32_t pca_dims = ctx.pca_dims;
    const uint32_t k_root = ctx.k_root;

    // --- 1. Sample + train quantizer ---
    const auto t_sample = std::chrono::steady_clock::now();
    uint32_t train_n = std::min<uint64_t>(20'000, n);
    ctx.source.reset();
    Chunk chunk;
    uint32_t collected = 0;
    std::vector<float> sample_buf;
    while (collected < train_n && ctx.source.next(chunk)) {
        const uint32_t take = std::min<uint32_t>(chunk.count, train_n - collected);
        sample_buf.insert(sample_buf.end(), chunk.vectors,
                          chunk.vectors + static_cast<size_t>(take) * dim);
        collected += take;
    }
    train_n = collected;  // actual count
    std::vector<float> sample = std::move(sample_buf);

    if (params.quantizer_type == "prq") {
        const uint32_t nsplits = (params.prq_nsplits > 0)
            ? params.prq_nsplits : static_cast<uint32_t>(dim) / 8;
        ctx.quantizer = std::make_unique<ProductResidualQuantizer>(
            params.metric, dim, ctx.m4, ctx.scan_bits, nsplits,
            params.prq_beam_size, 42);
    } else {
        ctx.quantizer = std::make_unique<PqQuantizer>(
            params.metric, dim, ctx.m4, ctx.scan_bits, 42);
    }
    ctx.quantizer->train(sample.data(), train_n);
    spdlog::info("[sextant] build_streaming_pca: trained {} in {:.2f}s",
                 params.quantizer_type,
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_sample).count());

    // --- 2. Compute PCA (reuse existing compute_pca_rotation_public) ---
    const auto t_pca = std::chrono::steady_clock::now();

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
    std::vector<float> rotation = compute_pca_rotation_public(
        sample.data(), train_n, dim, &eigvals);

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
    for (uint16_t i = 0; i < dim; ++i) var_total += eigvals[i];
    for (uint32_t k = 0; k < pca_dims; ++k) var_explained += eigvals[k];
    spdlog::info("[sextant] build_streaming_pca: PCA {}→{} dims, "
                 "variance explained: {:.1f}% in {:.2f}s",
                 dim, pca_dims, 100.0 * var_explained / std::max(var_total, 1.0),
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_pca).count());

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
                        // Precompute centroid norms (constant per pass).
                        // dist = |proj|² - 2·proj·centroid + |centroid|²
                        // |proj|² is constant across centroids (skip).
                        // |centroid|² is precomputed once.
                        // argmin dist = argmin(-2·proj·centroid + |centroid|²)
                        std::vector<float> cent_norms(k_root);
                        for (uint32_t c = 0; c < k_root; ++c) {
                            cent_norms[c] = simd::dot_f32(
                                root_centroids_pca[c].data(),
                                root_centroids_pca[c].data(), pca_dims);
                        }
                        for (uint32_t i = s; i < e; ++i) {
                            const float* xi = &vec_buf[i * dim];
                            std::vector<float> proj(pca_dims);
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
    const uint32_t code_size = ctx.quantizer->code_size();
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

        const uint32_t npg = leaf_extent_pages(count, m4, scan_bits,
                                               summary_size, fcb);
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

        const uint32_t n_blocks = (count + cpb - 1) / cpb;
        uint8_t* codes_out = obuf.data() + leaf_codes_offset(summary_size);
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t base = b * cpb;
            uint8_t* blk = codes_out + static_cast<uint64_t>(b) * bb;
            std::memset(blk, 0, bb);
            for (uint32_t j = 0; j < cpb; ++j) {
                const uint32_t gi = base + j;
                if (gi >= count) break;
                const uint8_t* code = buf.codes.data() + gi * code_size;
                if (scan_bits == 4) {
                    const uint8_t nbi = j % 16; const bool hi = (j >= 16);
                    for (uint16_t s = 0; s < m4; ++s) {
                        uint8_t nib = (s/2 < code_size)
                            ? ((s%2==0) ? (code[s/2]&0x0F) : (code[s/2]>>4)) : 0;
                        if (hi) blk[s*16+nbi] |= (nib<<4);
                        else    blk[s*16+nbi] |= nib;
                    }
                } else {
                    for (uint16_t s = 0; s < m4; ++s)
                        blk[s*16+j] = (s < code_size) ? code[s] : 0;
                }
            }
        }
        RowId* rids = reinterpret_cast<RowId*>(
            obuf.data() + leaf_rowids_offset(summary_size, n_blocks, bb));
        std::memcpy(rids, buf.row_ids.data(), count * sizeof(RowId));

        // Phase C: filter column data region (after row_ids). This must
        // match LeafFilterLayout::compute's filter_base offset.
        if (fcb > 0) {
            const uint64_t fc_off =
                leaf_rowids_offset(summary_size, n_blocks, bb) +
                static_cast<uint64_t>(count) * sizeof(RowId);
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

            std::vector<std::future<void>> futs;
            const uint32_t n_threads = std::min(hw, take);
            const uint32_t per = (take + n_threads - 1) / n_threads;
            for (uint32_t t = 0; t < n_threads; ++t) {
                const uint32_t start = t * per;
                const uint32_t end = std::min(start + per, take);
                if (start >= end) break;
                futs.push_back(std::async(std::launch::async,
                    [&](uint32_t s, uint32_t e) {
                        std::vector<uint8_t> code(code_size);
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
                            // Encode scan code.
                            ctx.quantizer->encode(xi, code.data());
                            chunk_codes[i].assign(code.begin(), code.end());
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

            // Serial: append to shared buffers + flush on overflow.
            for (uint32_t i = 0; i < take; ++i) {
                const float16_t* fvec = &fp16_buf[i * dim];
                const RowId rid = chunk_row_ids[i];
                // Phase D: record this row's filter column values in the global
                // cardinality table (once per row, not per closure target).
                // Dual-path: read from chunk_fcols by local idx when the chunk
                // provides filter data, else from cfg.filter_column_data by
                // global row id.
                if (!cfg.filter_schema.empty()) {
                    const uint32_t r = static_cast<uint32_t>(rid);
                    for (uint32_t c = 0; c < cfg.filter_schema.columns.size(); ++c) {
                        const auto& col = cfg.filter_schema.columns[c];
                        if (col.type == ColumnType::String) {
                            std::string_view sv;
                            if (chunk_has_filter) {
                                const auto* sc =
                                    static_cast<const FilterStringColumn*>(chunk_fcols[c]);
                                sv = std::string_view(sc->data + sc->offsets[i],
                                                      sc->lengths[i]);
                            } else {
                                const auto& src = cfg.filter_column_data[c];
                                sv = std::string_view(
                                    src.str_data.data() + src.str_offsets[r],
                                    src.str_lengths[r]);
                            }
                            ctx.card_table.add_string(c, sv);
                        } else if (col.type == ColumnType::Set) {
                            if (chunk_has_filter) {
                                const auto* fsc =
                                    static_cast<const FilterSetColumn*>(chunk_fcols[c]);
                                ctx.card_table.add_set(c,
                                    fsc->counts, fsc->offsets, fsc->element_lengths,
                                    fsc->element_data, i);
                            } else {
                                const auto& src = cfg.filter_column_data[c];
                                ctx.card_table.add_set(c, src.set_counts.data(),
                                                    src.set_offsets.data(),
                                                    src.set_elem_lengths.data(),
                                                    src.set_elem_data.data(), r);
                            }
                        } else if (col.type == ColumnType::Int32) {
                            int32_t v;
                            if (chunk_has_filter) {
                                std::memcpy(&v, static_cast<const uint8_t*>(chunk_fcols[c]) + static_cast<size_t>(i) * 4, 4);
                            } else {
                                const auto& src = cfg.filter_column_data[c];
                                std::memcpy(&v, src.fixed_data.data() + r * 4, 4);
                            }
                            ctx.card_table.add_numeric(c, static_cast<double>(v));
                        } else if (col.type == ColumnType::Int64) {
                            int64_t v;
                            if (chunk_has_filter) {
                                std::memcpy(&v, static_cast<const uint8_t*>(chunk_fcols[c]) + static_cast<size_t>(i) * 8, 8);
                            } else {
                                const auto& src = cfg.filter_column_data[c];
                                std::memcpy(&v, src.fixed_data.data() + r * 8, 8);
                            }
                            ctx.card_table.add_numeric(c, static_cast<double>(v));
                        } else if (col.type == ColumnType::Float) {
                            float v;
                            if (chunk_has_filter) {
                                std::memcpy(&v, static_cast<const uint8_t*>(chunk_fcols[c]) + static_cast<size_t>(i) * 4, 4);
                            } else {
                                const auto& src = cfg.filter_column_data[c];
                                std::memcpy(&v, src.fixed_data.data() + r * 4, 4);
                            }
                            ctx.card_table.add_numeric(c, static_cast<double>(v));
                        }
                    }
                }
                for (uint32_t c : chunk_targets[i]) {
                    auto& buf = buffers[c];
                    buf.codes.insert(buf.codes.end(),
                                     chunk_codes[i].begin(), chunk_codes[i].end());
                    buf.row_ids.push_back(rid);
                    // Incremental centroid accumulator (replaces fp16_vecs).
                    for (uint16_t d = 0; d < dim; ++d)
                        buf.centroid_sum[d] += static_cast<float>(fvec[d]);
                    ++buf.centroid_count;
                    if (has_filter) append_filter_row(buf.filter_cols, rid, i);
                    if (has_payload) {
                        uint32_t plen; const uint8_t* pdata;
                        if (chunk_has_payload) {
                            plen = chunk_poffsets[i + 1] - chunk_poffsets[i];
                            pdata = chunk_pdata + chunk_poffsets[i];
                        } else {
                            const uint32_t r = static_cast<uint32_t>(rid);
                            plen = cfg.payload_offsets[r + 1] - cfg.payload_offsets[r];
                            pdata = cfg.payload_data + cfg.payload_offsets[r];
                        }
                        buf.payload_lens.push_back(plen);
                        buf.payload_ptrs.push_back(pdata);
                    }
                    if (buf.row_ids.size() >= leaf_cap)
                        flush_buffer(c);
                }
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
    auto& quantizer = ctx.quantizer;
    auto& card_table = ctx.card_table;

    // --- 7. Write tree file: internal nodes, root, codebook, config, blobs ---
    // All leaf extents (+ payloads) were written directly to `file` at flush
    // time. The PageFile + PageAllocator are already initialized. This phase
    // only writes the tree structure above the leaves.
    // root_to_leaves[c][j] already holds the global leaf_metas index of cluster
    // c's j-th leaf (assigned at flush time, in flush/arrival order).
    const auto t_write = std::chrono::steady_clock::now();

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

    std::vector<uint8_t> qblob;
    quantizer->serialize(qblob);
    const uint64_t qbsz = qblob.size();
    std::vector<uint8_t> qbs(sizeof(qbsz) + qblob.size());
    std::memcpy(qbs.data(), &qbsz, sizeof(qbsz));
    std::memcpy(qbs.data() + sizeof(qbsz), qblob.data(), qblob.size());
    const uint32_t cb_npg = static_cast<uint32_t>((qbs.size()+kPageSize-1)/kPageSize);
    const PageId cb_page = alloc.alloc_extent(file, cb_npg);
    { std::vector<uint8_t> b(cb_npg*kPageSize, 0); std::memcpy(b.data(),qbs.data(),qbs.size());
      file.write_pages(cb_page, cb_npg, b.data()); }

    TreeManifest manifest;
    manifest.dim = dim; manifest.m4 = m4; manifest.scan_pq_bits = scan_bits;
    manifest.quantizer_type = params.quantizer_type;
    manifest.prq_nsplits = (params.quantizer_type == "prq")
        ? static_cast<uint32_t>(static_cast<ProductResidualQuantizer&>(*quantizer).nsplits()) : 0;
    manifest.depth = depth; manifest.k_root = k_root; manifest.k_l1 = k_l1;
    manifest.leaf_capacity = leaf_cap; manifest.n_leaves = n_leaves_total;
    manifest.n_probe_l0 = cfg.n_probe_l0 > 0 ? cfg.n_probe_l0
        : static_cast<uint32_t>(std::max(1.0, 2.0*std::sqrt(double(k_root))));
    manifest.n_probe_ln = cfg.n_probe_ln > 0 ? cfg.n_probe_ln : 4;
    manifest.adaptive_probe_gap = cfg.adaptive_probe_gap;
    manifest.median_lid = cfg.median_lid;
    manifest.pca_dims = pca_dims;  // enable PCA routing at search time
    manifest.balance_factor = params.partition_balance_factor;
    manifest.schema = cfg.filter_schema;
    manifest.summary_size = summary_size;

    // --- Write PCA routing blob ---
    // Layout: [pca_dims:u32][proj:pca_dims×dim f32][mean_proj:pca_dims f32]
    //         [root_centroids:k_root×pca_dims f32]
    //         [leaf_centroids:n_leaves×pca_dims f32]
    PageId pca_page = kInvalidPage;
    uint32_t pca_npg = 0;
    if (pca_dims > 0) {
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

    const PageId bitmap_page = 2;
    const uint32_t bitmap_pages = 1;
    alloc.flush_bitmap(file);
    Superblock sb;
    sb.init_fresh(bitmap_page, bitmap_pages);
    sb.set_root(root_page, root_npg);
    sb.set_depth(depth);
    sb.set_n_leaves(n_leaves_total);
    sb.set_n_pages(file.num_pages());
    sb.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    sb.set_bitmap(bitmap_page, bitmap_pages);
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
    return result;
}

}  // namespace

std::vector<Candidate> IVFTreeIndex::search(const float* query, uint32_t k,
                                             const SearchConfig& config,
    std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs) const {
    // Build the FastScan LUT once per query.
    const bool scan_8bit = (manifest_.scan_pq_bits == 8);
    const uint32_t m = manifest_.m4;
    const uint32_t codes_per_block = scan_8bit ? 16 : 32;
    const uint32_t block_bytes = m * 16;  // [m][16] for both 4-bit and 8-bit

    // LUT buffers.
    std::vector<uint8_t> lut4(scan_8bit ? 0 : m * 16);
    std::vector<uint8_t> lut8(scan_8bit ? m * 256 : 0);
    float lut_scale = 0, lut_offset = 0;

    if (scan_8bit) {
        quantizer_->build_fastscan_lut(query, lut8.data(), &lut_scale,
                                        &lut_offset);
    } else {
        quantizer_->build_fastscan_lut4(query, lut4.data(), nullptr);
    }

    // Cast query to FP16 for routing (used for non-PCA path + leaf centroid reads).
    std::vector<float16_t> query_fp16(manifest_.dim);
    cast_fp32_to_fp16(query, query_fp16.data(), manifest_.dim);
    const MetricKind metric = quantizer_->metric();

    // --- Project query to PCA space if PCA routing is enabled ---
    std::vector<float> query_pca;
    if (pca_dims_ > 0) {
        query_pca.resize(pca_dims_);
        for (uint32_t k = 0; k < pca_dims_; ++k) {
            query_pca[k] = simd::dot_f32(&pca_proj_[k * manifest_.dim],
                                          query, manifest_.dim)
                           - pca_mean_proj_[k];
        }
    }

    // --- Resolve predicate column indices (once per query) ---
    // Phase D: if predicates are present, map each predicate's column name to
    // its schema column index up-front. A predicate referencing an unknown
    // column yields no results. For geo predicates, also resolve the longitude
    // column name.
    std::vector<uint32_t> pred_col_indices;
    std::vector<uint32_t> geo_lng_col_indices;  // valid only for geo predicates
    if (!config.predicates.empty()) {
        pred_col_indices.reserve(config.predicates.size());
        geo_lng_col_indices.resize(config.predicates.size(), UINT32_MAX);
        for (uint32_t pi = 0; pi < config.predicates.size(); ++pi) {
            const auto& pred = config.predicates[pi];
            const auto* col = manifest_.schema.find(pred.column);
            if (!col) {
                return {};  // Predicate references unknown column.
            }
            pred_col_indices.push_back(
                static_cast<uint32_t>(col - manifest_.schema.columns.data()));
            // Geo predicates need a second column (longitude).
            if (pred.op == PredicateOp::GeoRadius ||
                pred.op == PredicateOp::GeoBox) {
                if (pred.geo_lng_column.empty()) {
                    return {};  // Misconfigured geo predicate.
                }
                const auto* lng_col = manifest_.schema.find(pred.geo_lng_column);
                if (!lng_col) {
                    return {};  // Longitude column not found.
                }
                geo_lng_col_indices[pi] = static_cast<uint32_t>(
                    lng_col - manifest_.schema.columns.data());
            }
        }
    }
    const bool has_predicates = !config.predicates.empty();

    // --- Route: iterative descent from root to leaves ---
    //
    // At each level we hold a frontier of (distance, child descriptor). The
    // frontier is expanded one level at a time: each internal node is read
    // from the mmap, its children are scored, and the top-n_probe_ln are kept.
    // Leaf frontier entries are carried through unchanged. After the final
    // descent, all surviving leaf entries become scan candidates.
    //
    // Level 0 (root) uses PCA-space distances when pca_dims_ > 0. For depth=2
    // with PCA enabled, the L1→leaf step also uses PCA leaf centroids. For
    // depth>=3, deeper levels (L1→L2, L2→leaves) route by FP16 distance to the
    // inline child centroids (the data is already well-partitioned there).
    struct ProbeEntry {
        float    dist;
        PageId   page;
        uint64_t pages;
        uint16_t is_leaf;
        const float16_t* centroid;  // inline FP16 centroid (points into mmap)
    };

    const uint32_t cesize = child_entry_size(manifest_.dim, manifest_.summary_size);
    const float gap = (config.adaptive_probe_gap > 0)
        ? config.adaptive_probe_gap
        : manifest_.adaptive_probe_gap;

    const uint32_t n_probe_ln_cfg = config.n_probe_ln > 0
        ? config.n_probe_ln
        : (manifest_.n_probe_ln > 0 ? manifest_.n_probe_ln : 4);

    // Lambda: read an internal node from mmap, score children, push top-n onto
    // `out`. `use_pca_leaves` selects PCA leaf-centroid lookup (depth=2 path).
    auto expand_internal = [&](const ProbeEntry& e, bool use_pca_leaves,
                               uint32_t root_child_for_pca,
                               std::vector<ProbeEntry>& out) {
        const uint8_t* node_ptr = mmap_base_ +
            static_cast<uint64_t>(e.page) * kPageSize;
        const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
        const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);

        std::vector<std::pair<float, uint32_t>> child_dists;
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
                    const float diff = query_pca[kk] - lc[kk];
                    d += diff * diff;
                }
            } else {
                const float16_t* cent = reinterpret_cast<const float16_t*>(
                    p + sizeof(ChildEntry));
                d = simd::dist_f16(metric, query_fp16.data(),
                                    cent, manifest_.dim);
            }
            child_dists.emplace_back(d, j);
            p += cesize;
        }
        std::sort(child_dists.begin(), child_dists.end());

        const uint32_t n_probe_ln = std::min(
            static_cast<uint64_t>(n_probe_ln_cfg), nh->n_children);
        const uint8_t* p2 = node_ptr + sizeof(TreeNodeHeader);
        for (uint32_t j = 0; j < n_probe_ln; ++j) {
            if (gap > 0 && j > 0 &&
                child_dists[j].first > child_dists[j - 1].first * gap) break;
            const uint32_t idx = child_dists[j].second;
            const auto* ce = reinterpret_cast<const ChildEntry*>(
                p2 + idx * cesize);
            if (ce->child_page == kInvalidPage) continue;  // empty child
            // Resolve leaf_id → physical page via leaf table.
            PageId leaf_id = ce->child_page;
            PageId rpage = leaf_id;
            uint64_t rpages = ce->child_pages;
            if (ce->is_leaf && !leaf_table_.empty() &&
                leaf_id < leaf_table_.size()) {
                rpage = leaf_table_[leaf_id].page;
                rpages = leaf_table_[leaf_id].pages;
            }
            const float16_t* cent = reinterpret_cast<const float16_t*>(
                reinterpret_cast<const uint8_t*>(ce) + sizeof(ChildEntry));
            out.push_back({child_dists[j].first, rpage,
                           rpages, ce->is_leaf, cent});
        }
    };

    // --- Phase D: compute selectivity BEFORE routing ---
    // selectivity = estimated fraction of rows passing all predicates.
    // Used for: (1) MUST_ENTER filter-directed routing at low selectivity,
    // (2) adaptive W, (3) brute-force fallback trigger at <1%.
    float selectivity = 1.0f;
    if (has_predicates) {
        if (!card_table_.empty()) {
            selectivity = card_table_.selectivity_combined(
                manifest_.schema, config.predicates, pred_col_indices, geo_lng_col_indices);
        } else {
            // Fallback: root-summary-based subtree overlap estimation.
            selectivity = 1.0f;
            std::vector<const uint8_t*> root_summaries(root_children_.size());
            const uint32_t dim16 = manifest_.dim;
            for (uint32_t c = 0; c < root_children_.size(); ++c) {
                root_summaries[c] = reinterpret_cast<const uint8_t*>(
                    root_children_[c].centroid) + dim16 * sizeof(float16_t);
            }
            for (uint32_t p = 0; p < config.predicates.size(); ++p) {
                const auto& pred = config.predicates[p];
                const uint32_t col_idx = pred_col_indices[p];
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

    // Brute-force PQ-decode fallback for extreme low selectivity (<1%).
    // Triggered before routing — no point routing when we'll scan all
    // matching leaves anyway.
    if (has_predicates && selectivity > 0.0f && selectivity < 0.01f) {
        return search_brute_force_filtered(query, k, config, pred_col_indices,
                                           geo_lng_col_indices, payload_locs);
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

    // Score root children, sort, select top-n_probe_l0 with gap pruning. We
    // track each entry's root-child index (needed later for the depth=2 PCA
    // leaf-centroid lookup).
    std::vector<std::pair<float, uint32_t>> root_dists;
    root_dists.reserve(k_root);
    // Summary offset within each root child entry (after the inline FP16
    // centroid). root_children_[c].centroid points at the centroid, which sits
    // at child-entry offset sizeof(ChildEntry); the summary follows it.
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
                                   pred_col_indices, geo_lng_col_indices)) {
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
            d = simd::dist_f16(metric, query_fp16.data(),
                               root_children_[c].centroid, manifest_.dim);
        }
        root_dists.emplace_back(d, c);
    }
    std::sort(root_dists.begin(), root_dists.end());

    std::vector<ProbeEntry> frontier;
    std::vector<uint32_t>   root_idx;  // root-child index per frontier entry
    // When filter-directed, probe ALL summary-matching children (no n_probe_l0
    // cap, no gap pruning). Otherwise, top-n_probe_l0 with gap pruning.
    const uint32_t effective_probe = filter_directed
        ? static_cast<uint32_t>(root_dists.size())
        : n_probe_l0;
    frontier.reserve(effective_probe);
    for (uint32_t i = 0; i < effective_probe && i < root_dists.size(); ++i) {
        if (!filter_directed && gap > 0 && i > 0 &&
            root_dists[i].first > root_dists[i - 1].first * gap) break;
        const uint32_t c = root_dists[i].second;
        const auto& rc = root_children_[c];
        frontier.push_back({root_dists[i].first, rc.page, rc.pages,
                            rc.is_leaf, rc.centroid});
        root_idx.push_back(c);
    }

    // --- Descend through internal levels ---
    // depth=1 → no descent (frontier is already leaves).
    // depth=2 → one expansion (root children → leaves).
    // depth=3 → two expansions (root → L1 → L2, then L2 → leaves).
    const bool pca_depth2 = (pca_dims_ > 0 && manifest_.depth == 2);
    for (uint16_t level = 1; level < manifest_.depth; ++level) {
        std::vector<ProbeEntry> next_frontier;
        std::vector<uint32_t>   next_root_idx;  // only meaningful for depth=2
        next_frontier.reserve(frontier.size() * std::max(1u, n_probe_ln_cfg));

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
            expand_internal(e, use_pca_leaves, rc_for_pca, next_frontier);
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
    std::vector<LeafCandidate> candidates;
    for (const auto& e : frontier) {
        if (e.is_leaf && e.page != kInvalidPage) {
            candidates.push_back({e.page, e.pages, e.dist, e.centroid});
        }
    }

    if (candidates.empty()) {
        return {};
    }

    // --- Phase D: summary-based leaf pruning ---
    // Before scanning, drop any candidate leaf whose summary rules out all
    // matches for the predicates (numeric range miss or bloom negative). This
    // skips entire leaves, saving the FastScan cost at low selectivity.
    if (has_predicates) {
        std::vector<LeafCandidate> pruned;
        pruned.reserve(candidates.size());
        for (const auto& cand : candidates) {
            if (cand.page == kInvalidPage) continue;
            const uint8_t* leaf_ptr = mmap_base_ +
                static_cast<uint64_t>(cand.page) * kPageSize;
            const uint8_t* summary = leaf_ptr + leaf_filter_offset();
            if (summary_may_match(summary, manifest_.summary_size,
                                  manifest_.schema, config.predicates,
                                  pred_col_indices, geo_lng_col_indices)) {
                pruned.push_back(cand);
            }
        }
        candidates = std::move(pruned);
        if (candidates.empty()) {
            return {};
        }
    }

    // --- Prefetch leaf extents ---
    // On Linux, posix_fadvise triggers async NVMe prefetch. On macOS it's
    // a no-op (the unified buffer cache handles read-ahead for sequential
    // mmap access).
    for (const auto& c : candidates) {
#ifdef __linux__
        ::posix_fadvise(fd_, static_cast<off_t>(c.page) * kPageSize,
                        static_cast<off_t>(c.pages) * kPageSize,
                        POSIX_FADV_WILLNEED);
#else
        (void)c;  // macOS: no-op
#endif
    }

    // --- Scan leaves ---
    // Adaptive W driven by predicate selectivity (computed above, before routing).
    uint32_t W = std::max(config.fastscan_W > 0 ? config.fastscan_W : 300u, k);
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
    // PQ code back to FP32 without a row_id → code lookup.
    std::vector<HeapEntry> heap;
    heap.reserve(W + 32);
    const auto heap_less = [](const HeapEntry& a, const HeapEntry& b) {
        return a.pq_dist < b.pq_dist;
    };

    auto heap_replace = [&](uint32_t new_d, int64_t new_id,
                            const uint8_t* leaf_ptr, uint32_t local_idx) {
        heap[0] = {new_d, new_id, leaf_ptr, local_idx};
        uint32_t pos = 0;
        const uint32_t n = heap.size();
        while (true) {
            const uint32_t left = 2 * pos + 1;
            const uint32_t right = 2 * pos + 2;
            uint32_t largest = pos;
            if (left < n && heap[left].pq_dist > heap[largest].pq_dist)
                largest = left;
            if (right < n && heap[right].pq_dist > heap[largest].pq_dist)
                largest = right;
            if (largest == pos) break;
            std::swap(heap[pos], heap[largest]);
            pos = largest;
        }
    };

    for (const auto& cand : candidates) {
        if (cand.page == kInvalidPage) continue;
        const uint8_t* leaf_ptr = mmap_base_ +
            static_cast<uint64_t>(cand.page) * kPageSize;
        const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
        const uint32_t count = lh->count;
        if (count == 0) continue;

        const uint32_t n_blocks = (count + codes_per_block - 1) / codes_per_block;
        const uint8_t* codes = leaf_ptr + leaf_codes_offset(manifest_.summary_size);
        const RowId* row_ids = reinterpret_cast<const RowId*>(
            leaf_ptr + leaf_rowids_offset(manifest_.summary_size,
                                          n_blocks, block_bytes));

        // Phase D: filter columns are evaluated AFTER heap selection, not during
        // scan. No col_views parsing needed here.

        // Compute the valid_mask for the tail block.
        const uint32_t full_blocks = count / codes_per_block;
        const uint32_t tail_count = count - full_blocks * codes_per_block;
        const uint32_t tail_mask = (tail_count == 0)
            ? (codes_per_block == 32 ? 0xFFFFFFFFu : 0x0000FFFFu)
            : (codes_per_block == 32
                   ? (tail_count >= 32 ? 0xFFFFFFFFu
                       : (1u << tail_count) - 1u)
                   : (tail_count >= 16 ? 0xFFFFu : (1u << tail_count) - 1u));

        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint8_t* blk = codes + static_cast<uint64_t>(b) * block_bytes;
            const uint32_t valid_mask = (b + 1 < n_blocks)
                ? (codes_per_block == 32 ? 0xFFFFFFFFu : 0xFFFFu)
                : tail_mask;

            if (scan_8bit) {
                uint32_t out[16];
                simd::fastscan_block16(blk, lut8.data(), m,
                                        static_cast<uint16_t>(valid_mask), out);
                const uint32_t base = b * 16;
                if (heap.size() < W) {
                    for (uint32_t j = 0; j < 16; ++j) {
                        if (out[j] == 0xFFFFFFFFu) continue;
                        heap.push_back({out[j], row_ids[base + j], leaf_ptr, base + j});
                        if (heap.size() == W) {
                            std::make_heap(heap.begin(), heap.end(), heap_less);
                            break;
                        }
                    }
                    if (heap.size() < W) continue;
                }
                const uint32_t front_d = heap[0].pq_dist;
                uint32_t block_min = 0xFFFFFFFFu;
                for (uint32_t j = 0; j < 16; ++j)
                    if (out[j] < block_min) block_min = out[j];
                if (block_min >= front_d) continue;
                for (uint32_t j = 0; j < 16; ++j) {
                    if (out[j] == 0xFFFFFFFFu || out[j] >= heap[0].pq_dist)
                        continue;
                    heap_replace(out[j], row_ids[base + j], leaf_ptr, base + j);
                }
            } else {
                uint32_t out[32];
                simd::pq4_block32(blk, lut4.data(), m, out);
                const uint32_t base = b * 32;

                if (heap.size() < W) {
                    for (uint32_t j = 0; j < 32; ++j) {
                        if (!((valid_mask >> j) & 1u)) continue;
                        heap.push_back({out[j], row_ids[base + j], leaf_ptr, base + j});
                        if (heap.size() == W) {
                            std::make_heap(heap.begin(), heap.end(), heap_less);
                            break;
                        }
                    }
                    if (heap.size() < W) continue;
                }

                const uint32_t front_d = heap[0].pq_dist;
                uint32_t block_min = 0xFFFFFFFFu;
                for (uint32_t j = 0; j < 32; ++j)
                    if ((valid_mask >> j) & 1u)
                        block_min = std::min(block_min, out[j]);
                if (block_min >= front_d) continue;

                for (uint32_t j = 0; j < 32; ++j) {
                    if (!((valid_mask >> j) & 1u)) continue;
                    if (out[j] >= heap[0].pq_dist) continue;
                    heap_replace(out[j], row_ids[base + j], leaf_ptr, base + j);
                }
            }
        }
    }

    // --- Extract top-k from the heap ---
    std::vector<Candidate> results;
    results.reserve(heap.size());

    // Phase E: when payload locations are requested, carry (leaf_ptr, local_idx)
    // alongside each result through dedup/sort/truncate. The heap entries'
    // payload locations are valid mmap pointers (stable for the index's life).
    struct ResultWithLoc {
        Candidate cand;
        const uint8_t* leaf_ptr;
        uint32_t local_idx;
    };
    const bool want_locs = (payload_locs != nullptr);
    std::vector<ResultWithLoc> results_loc;
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
                  [](const HeapEntry& a, const HeapEntry& b) {
                      return a.leaf_ptr < b.leaf_ptr;
                  });
        std::vector<HeapEntry> filtered;
        filtered.reserve(heap.size());
        const uint8_t* cur_leaf = nullptr;
        std::vector<ColumnView> cols;
        for (const auto& entry : heap) {
            if (entry.leaf_ptr != cur_leaf) {
                cur_leaf = entry.leaf_ptr;
                const auto* lh = reinterpret_cast<const TreeLeafHeader*>(cur_leaf);
                auto layout = LeafFilterLayout::compute(cur_leaf, manifest_.m4,
                    manifest_.scan_pq_bits, manifest_.summary_size);
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
        // Rerank: decode each of the W candidates' PQ codes back to FP32 and
        // compute the exact distance to the query. Re-sorting by exact
        // distance fixes the PQ-approximation error and substantially
        // improves recall@k. Per-query serial (W≈300 is cheap; the CLI
        // parallelizes across queries).
        const uint32_t code_sz = quantizer_->code_size();
        std::vector<uint8_t> code_buf(code_sz);
        std::vector<float> decoded_vec(manifest_.dim);

        for (const auto& entry : heap) {
            // extract_code_from_leaf fully writes every code byte for both
            // 4-bit (both nibbles of each byte) and 8-bit (one byte/segment)
            // layouts, so no prior zeroing of code_buf is needed.
            extract_code_from_leaf(entry.leaf_ptr, entry.local_idx,
                                   manifest_.summary_size, manifest_.m4,
                                   manifest_.scan_pq_bits, codes_per_block,
                                   block_bytes, code_buf.data());
            quantizer_->decode_code(code_buf.data(), decoded_vec.data());
            const float exact_dist =
                (metric == MetricKind::InnerProduct)
                    ? -simd::dot_f32(query, decoded_vec.data(), manifest_.dim)
                    :  simd::l2sq_f32(query, decoded_vec.data(), manifest_.dim);
            if (want_locs) {
                results_loc.push_back({{entry.row_id, exact_dist},
                                        entry.leaf_ptr, entry.local_idx});
            } else {
                results.push_back({entry.row_id, exact_dist});
            }
        }
    } else {
        // No rerank: use the raw PQ-approximate uint32 distances.
        for (const auto& entry : heap) {
            if (want_locs) {
                results_loc.push_back({{entry.row_id,
                                         static_cast<float>(entry.pq_dist)},
                                        entry.leaf_ptr, entry.local_idx});
            } else {
                results.push_back({entry.row_id,
                                   static_cast<float>(entry.pq_dist)});
            }
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

    TreeBuildContext ctx(source, output_path, cfg);

    resolve_build_params(ctx);
    train_quantizer_and_pca(ctx);
    run_lloyd_refinement(ctx);

    // The PageFile + PageAllocator are initialized before the emission pass so
    // leaves can be allocated + written at flush time. The file starts with
    // just the bitmap (page 2); leaves are allocated sequentially.
    const PageId bitmap_page = 2;
    const uint32_t bitmap_pages = 1;
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

    const MetricKind metric = quantizer_->metric();
    const uint32_t code_sz = quantizer_->code_size();
    const uint32_t summary_size = manifest_.summary_size;
    const uint16_t m4 = manifest_.m4;
    const uint8_t pq_bits = manifest_.scan_pq_bits;

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
                PageId leaf_id = ce->child_page;
                PageId lp = leaf_id;
                uint64_t lpg = ce->child_pages;
                if (!leaf_table_.empty() && leaf_id < leaf_table_.size()) {
                    lp = leaf_table_[leaf_id].page;
                    lpg = leaf_table_[leaf_id].pages;
                }
                all_leaves.push_back({lp, lpg});
            } else {
                node_stack.push_back(ce->child_page);
            }
            p += cesize;
        }
    }

    // Scan each leaf: check summary, then scan filter columns for exact matches.
    std::vector<Candidate> results;
    std::vector<uint8_t> code_buf(code_sz);
    std::vector<float> decoded_vec(manifest_.dim);

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
        auto layout = LeafFilterLayout::compute(leaf_ptr, m4, pq_bits,
                                                  summary_size);
        auto col_views = parse_filter_columns(layout.filter_base, count,
                                                manifest_.schema);

        // Scan all rows: find exact matches.
        for (uint32_t i = 0; i < count; ++i) {
            if (!eval_all_predicates(col_views, manifest_.schema, i,
                                       config.predicates, pred_col_indices, geo_lng_col_indices))
                continue;

            // Exact match — decode PQ code and compute distance.
            extract_code_from_leaf(leaf_ptr, i, summary_size, m4, pq_bits,
                                    layout.codes_per_block, layout.block_bytes,
                                    code_buf.data());
            quantizer_->decode_code(code_buf.data(), decoded_vec.data());
            const float dist = (metric == MetricKind::InnerProduct)
                ? -simd::dot_f32(query, decoded_vec.data(), manifest_.dim)
                :  simd::l2sq_f32(query, decoded_vec.data(), manifest_.dim);
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
    const MetricKind metric = quantizer_->metric();

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
    const uint16_t m4 = manifest_.m4;
    const uint8_t pq_bits = manifest_.scan_pq_bits;
    const uint32_t summary_size = manifest_.summary_size;
    const uint32_t cpb = (pq_bits == 4) ? 32 : 16;
    const uint32_t bb = m4 * 16;
    const uint32_t code_size = quantizer_->code_size();

    // 1. Read the leaf.
    std::vector<uint8_t> old_buf = read_leaf_(leaf_id);
    const auto* old_lh = reinterpret_cast<const TreeLeafHeader*>(old_buf.data());
    const uint32_t count = static_cast<uint32_t>(old_lh->count);
    const uint32_t old_nb = (count + cpb - 1) / cpb;

    // 2. Extract PQ codes from FastScan layout → compact code array.
    std::vector<uint8_t> codes(static_cast<size_t>(count) * code_size);
    for (uint32_t i = 0; i < count; ++i) {
        extract_code_from_leaf(old_buf.data(), i, summary_size, m4,
                               pq_bits, cpb, bb,
                               codes.data() + static_cast<size_t>(i) * code_size);
    }

    // 3. k-means(K=2) on the PQ codes.
    auto km = kmeans_pq(*quantizer_, codes.data(), count, code_size,
                        /*K=*/2, /*iterations=*/10, /*num_threads=*/1,
                        /*seed=*/0xDEADBEEFULL + leaf_id);

    // 4. Partition vectors into two groups based on the assignment.
    std::vector<uint32_t> group0, group1;
    group0.reserve(count / 2 + 1);
    group1.reserve(count / 2 + 1);
    for (uint32_t i = 0; i < count; ++i) {
        if (km.assign[i] == 0) group0.push_back(i);
        else group1.push_back(i);
    }

    // Edge case: if one group is empty (all vectors assigned to one centroid),
    // split the assignment deterministically by index parity.
    if (group0.empty() || group1.empty()) {
        group0.clear();
        group1.clear();
        for (uint32_t i = 0; i < count; ++i) {
            if (i < count / 2) group0.push_back(i);
            else group1.push_back(i);
        }
    }

    // 5. Read old row_ids.
    const RowId* old_rids = reinterpret_cast<const RowId*>(
        old_buf.data() + leaf_rowids_offset(summary_size, old_nb, bb));

    // 5b. Read old filter column data (if present).
    const bool has_filter = manifest_.schema.n_filter_columns() > 0;
    std::vector<ColumnData> old_filter_cols;
    if (has_filter) {
        const uint64_t old_filter_off =
            leaf_rowids_offset(summary_size, old_nb, bb) +
            static_cast<uint64_t>(count) * sizeof(RowId);
        read_filter_columns(old_buf.data() + old_filter_off, count,
                            manifest_.schema, old_filter_cols);
    }

    // 6. Decode medoid centroids → FP16.
    auto decode_centroid_fp16 = [&](uint32_t k) -> std::vector<float16_t> {
        std::vector<float> f32(dim);
        quantizer_->decode_code(km.centroids[k].data(), f32.data());
        std::vector<float16_t> fp16(dim);
        cast_fp32_to_fp16(f32.data(), fp16.data(), dim);
        return fp16;
    };
    auto cent0_fp16 = decode_centroid_fp16(0);
    auto cent1_fp16 = decode_centroid_fp16(1);

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

        const uint32_t npg = leaf_extent_pages(
            gc, m4, pq_bits, summary_size, fcb);
        std::vector<uint8_t> nb(static_cast<size_t>(npg) * kPageSize, 0);

        // Header + summary.
        std::memcpy(nb.data(), old_buf.data(),
                    sizeof(TreeLeafHeader) + summary_size);

        const uint32_t new_nb = (gc + cpb - 1) / cpb;

        // Code blocks: re-encode each vector's code into the new layout.
        uint8_t* ncb = nb.data() + leaf_codes_offset(summary_size);
        for (uint32_t i = 0; i < gc; ++i) {
            const uint32_t src = group[i];
            const uint8_t* code = codes.data() + static_cast<size_t>(src) * code_size;
            const uint32_t b = i / cpb;
            const uint32_t j = i % cpb;
            uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;
            if (pq_bits == 4) {
                const uint8_t nbi = j % 16;
                const bool hi = (j >= 16);
                for (uint16_t s = 0; s < m4; ++s) {
                    uint8_t nib = (s / 2 < code_size)
                        ? ((s % 2 == 0) ? (code[s / 2] & 0x0F)
                                          : (code[s / 2] >> 4)) : 0;
                    if (hi) blk[s * 16 + nbi] |= (nib << 4);
                    else    blk[s * 16 + nbi] |= nib;
                }
            } else {
                for (uint16_t s = 0; s < m4; ++s)
                    blk[s * 16 + j] = (s < code_size) ? code[s] : 0;
            }
        }

        // Row IDs.
        RowId* nrid = reinterpret_cast<RowId*>(
            nb.data() + leaf_rowids_offset(summary_size, new_nb, bb));
        for (uint32_t i = 0; i < gc; ++i)
            nrid[i] = old_rids[group[i]];

        // Write filter column data (if present).
        if (has_filter && fcb > 0) {
            const uint64_t filter_off =
                leaf_rowids_offset(summary_size, new_nb, bb) +
                static_cast<uint64_t>(gc) * sizeof(RowId);
            write_filter_columns(nb.data() + filter_off, gc,
                                 manifest_.schema, group_filter_cols);
        }

        // Update header.
        auto* lh = reinterpret_cast<TreeLeafHeader*>(nb.data());
        lh->count = gc;
        lh->extent_pages = npg;
        if (has_filter && fcb > 0) {
            lh->filter_columns_offset =
                leaf_rowids_offset(summary_size, new_nb, bb) +
                static_cast<uint64_t>(gc) * sizeof(RowId);
        } else {
            lh->filter_columns_offset = 0;
        }
        lh->header_crc = header_crc(lh, offsetof(TreeLeafHeader, header_crc));

        const PageId page = alloc.alloc_extent(file_, npg);
        file_.write_pages(page, npg, nb.data());
        return {page, npg};
    };

    // 8. Write the two new leaves.
    auto entry0 = write_leaf_from_group(group0, cent0_fp16);
    auto entry1 = write_leaf_from_group(group1, cent1_fp16);

    // 9. Free the old leaf extent.
    alloc.free_extent(file_, leaf_table_[leaf_id].page, leaf_table_[leaf_id].pages);

    // 10. Update the leaf table: entry[leaf_id] = leaf0, new entry = leaf1.
    uint32_t new_leaf_id = static_cast<uint32_t>(leaf_table_.size());
    leaf_table_[leaf_id] = entry0;
    leaf_table_.push_back(entry1);

    // 11. Update the parent node: replace the old child entry with leaf0,
    //     add a new child entry for leaf1.
    add_child_to_parent_(leaf_id, new_leaf_id, cent0_fp16, cent1_fp16,
                         entry1.pages, alloc);

    spdlog::info("[sextant] split_leaf_: leaf {} (count={}) → leaf {} (count={}) "
                 "+ leaf {} (count={})",
                 leaf_id, count, leaf_id, group0.size(), new_leaf_id, group1.size());

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

    if (manifest_.depth == 1) {
        parent_page = superblock_.root_node_page();
        parent_pages = superblock_.root_node_pages();
        const uint8_t* root_ptr = mmap_base_ + parent_page * kPageSize;
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
        grandparent_page = superblock_.root_node_page();
        grandparent_pages = superblock_.root_node_pages();
        for (uint32_t c = 0; c < root_children_.size(); ++c) {
            const auto& rc = root_children_[c];
            if (rc.is_leaf || rc.page == kInvalidPage) continue;
            const uint8_t* node_ptr = mmap_base_ + rc.page * kPageSize;
            const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
            const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);
            for (uint32_t j = 0; j < nh->n_children; ++j) {
                const auto* ce = reinterpret_cast<const ChildEntry*>(p);
                if (ce->is_leaf && ce->child_page == leaf_id) {
                    parent_page = rc.page;
                    parent_pages = static_cast<uint32_t>(rc.pages);
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
            const uint8_t* l1_ptr = mmap_base_ + rc.page * kPageSize;
            const auto* l1h = reinterpret_cast<const TreeNodeHeader*>(l1_ptr);
            const uint8_t* lp = l1_ptr + sizeof(TreeNodeHeader);
            for (uint32_t j = 0; j < l1h->n_children; ++j) {
                const auto* l1ce = reinterpret_cast<const ChildEntry*>(lp);
                if (l1ce->is_leaf || l1ce->child_page == kInvalidPage) {
                    lp += cesize;
                    continue;
                }
                // l1ce->child_page is an L2 node. Scan its children for leaf_id.
                const uint8_t* l2_ptr =
                    mmap_base_ + l1ce->child_page * kPageSize;
                const auto* l2h =
                    reinterpret_cast<const TreeNodeHeader*>(l2_ptr);
                const uint8_t* p2 = l2_ptr + sizeof(TreeNodeHeader);
                for (uint32_t k = 0; k < l2h->n_children; ++k) {
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
    const auto t0 = std::chrono::steady_clock::now();

    const uint16_t m4 = manifest_.m4;
    const uint8_t pq_bits = manifest_.scan_pq_bits;
    const uint32_t summary_size = manifest_.summary_size;
    const uint32_t cpb = (pq_bits == 4) ? 32 : 16;
    const uint32_t bb = m4 * 16;
    const uint32_t code_size = quantizer_->code_size();
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
            ? (static_cast<uint64_t>(lh->extent_pages) * kPageSize
               - leaf_extent_bytes(old_count, m4, pq_bits,
                                   summary_size, 0))
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

        const uint32_t new_npg = leaf_extent_pages(
            new_count, m4, pq_bits, summary_size, new_filter_bytes);

        // Allocate the new buffer.
        std::vector<uint8_t> nb(static_cast<size_t>(new_npg) * kPageSize, 0);

        // Copy header + summary.
        std::memcpy(nb.data(), buf.data(), sizeof(TreeLeafHeader) + summary_size);

        const uint32_t old_nb = (old_count + cpb - 1) / cpb;
        const uint32_t new_nb = (new_count + cpb - 1) / cpb;

        // --- Rebuild FastScan code blocks ---
        uint8_t* ncb = nb.data() + leaf_codes_offset(summary_size);
        std::memcpy(ncb, buf.data() + leaf_codes_offset(summary_size),
                    static_cast<size_t>(old_nb) * bb);

        // Encode + append new codes.
        for (uint32_t ai = 0; ai < indices.size(); ++ai) {
            const uint32_t gi = old_count + ai;
            const uint32_t b = gi / cpb;
            const uint32_t j = gi % cpb;
            uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;

            std::vector<uint8_t> code(code_size);
            quantizer_->encode(points[indices[ai]].vector, code.data());

            if (pq_bits == 4) {
                const uint8_t nbi = j % 16;
                const bool hi = (j >= 16);
                for (uint16_t s = 0; s < m4; ++s) {
                    uint8_t nib = (s / 2 < code_size)
                        ? ((s % 2 == 0) ? (code[s / 2] & 0x0F)
                                          : (code[s / 2] >> 4)) : 0;
                    if (hi) blk[s * 16 + nbi] |= (nib << 4);
                    else    blk[s * 16 + nbi] |= nib;
                }
            } else {
                for (uint16_t s = 0; s < m4; ++s)
                    blk[s * 16 + j] = (s < code_size) ? code[s] : 0;
            }
        }

        // --- Copy + append row_ids ---
        RowId* nrid = reinterpret_cast<RowId*>(
            nb.data() + leaf_rowids_offset(summary_size, new_nb, bb));
        std::memcpy(nrid,
                    buf.data() + leaf_rowids_offset(summary_size, old_nb, bb),
                    old_count * sizeof(RowId));
        for (uint32_t ai = 0; ai < indices.size(); ++ai)
            nrid[old_count + ai] = points[indices[ai]].row_id;

        // --- Rebuild filter column data ---
        if (has_filter) {
            const uint64_t filter_off =
                leaf_rowids_offset(summary_size, new_nb, bb) +
                static_cast<uint64_t>(new_count) * sizeof(RowId);

            // Read existing filter data from the old buffer (if any).
            std::vector<ColumnData> all_cols;
            if (old_count > 0 && old_filter_bytes > 0) {
                const uint64_t old_filter_off =
                    leaf_rowids_offset(summary_size, old_nb, bb) +
                    static_cast<uint64_t>(old_count) * sizeof(RowId);
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
            ? (leaf_rowids_offset(summary_size, new_nb, bb) +
               static_cast<uint64_t>(new_count) * sizeof(RowId))
            : 0;
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
    const auto t0 = std::chrono::steady_clock::now();

    const uint16_t m4 = manifest_.m4;
    const uint8_t pq_bits = manifest_.scan_pq_bits;
    const uint32_t summary_size = manifest_.summary_size;
    const uint32_t cpb = (pq_bits == 4) ? 32 : 16;
    const uint32_t bb = m4 * 16;
    const uint32_t code_size = quantizer_->code_size();

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
        const uint32_t nb = (count + cpb - 1) / cpb;
        const RowId* rids = reinterpret_cast<const RowId*>(
            buf.data() + leaf_rowids_offset(summary_size, nb, bb));
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
        const uint32_t nb = (count + cpb - 1) / cpb;

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

        // Read old filter_cols_bytes for extent size computation.
        const uint64_t old_filter_bytes = lh->filter_columns_offset != 0
            ? (static_cast<uint64_t>(lh->extent_pages) * kPageSize
               - leaf_extent_bytes(count, m4, pq_bits, summary_size, 0))
            : 0;

        const uint32_t new_npg = leaf_extent_pages(
            new_count, m4, pq_bits, summary_size, old_filter_bytes);
        std::vector<uint8_t> nb_buf(static_cast<size_t>(new_npg) * kPageSize, 0);

        // Copy header + summary.
        std::memcpy(nb_buf.data(), buf.data(),
                    sizeof(TreeLeafHeader) + summary_size);

        const uint32_t new_nb = (new_count + cpb - 1) / cpb;

        // --- Rebuild code blocks (extract surviving codes, re-pack) ---
        uint8_t* ncb = nb_buf.data() + leaf_codes_offset(summary_size);
        std::vector<uint8_t> code(code_size > 0 ? code_size : 1);
        for (uint32_t i = 0; i < new_count; ++i) {
            const uint32_t src_slot = survivors[i];
            const uint32_t b = i / cpb;
            const uint32_t j = i % cpb;
            uint8_t* blk = ncb + static_cast<uint64_t>(b) * bb;

            // Extract code from old layout.
            const uint32_t old_b = src_slot / cpb;
            const uint32_t old_j = src_slot % cpb;
            const uint8_t* old_blk = buf.data() + leaf_codes_offset(summary_size)
                                     + static_cast<uint64_t>(old_b) * bb;
            if (pq_bits == 8) {
                for (uint16_t s = 0; s < m4; ++s)
                    blk[s * 16 + j] = old_blk[s * 16 + old_j];
            } else {
                const uint8_t old_byte_idx = old_j % 16;
                const bool old_hi = (old_j >= 16);
                const uint8_t new_byte_idx = j % 16;
                const bool new_hi = (j >= 16);
                for (uint16_t s = 0; s < m4; ++s) {
                    const uint8_t nib = old_hi
                        ? static_cast<uint8_t>(old_blk[s * 16 + old_byte_idx] >> 4)
                        : static_cast<uint8_t>(old_blk[s * 16 + old_byte_idx] & 0x0F);
                    if (new_hi) blk[s * 16 + new_byte_idx] |= (nib << 4);
                    else        blk[s * 16 + new_byte_idx] |= nib;
                }
            }
        }

        // --- Rebuild row_ids ---
        RowId* nrid = reinterpret_cast<RowId*>(
            nb_buf.data() + leaf_rowids_offset(summary_size, new_nb, bb));
        for (uint32_t i = 0; i < new_count; ++i)
            nrid[i] = *reinterpret_cast<const RowId*>(
                buf.data() + leaf_rowids_offset(summary_size, nb, bb) +
                survivors[i] * sizeof(RowId));

        // --- Compact filter column data (remove deleted rows) ---
        if (old_filter_bytes > 0) {
            const uint64_t new_filter_off =
                leaf_rowids_offset(summary_size, new_nb, bb) +
                static_cast<uint64_t>(new_count) * sizeof(RowId);
            const uint64_t old_filter_off =
                leaf_rowids_offset(summary_size, nb, bb) +
                static_cast<uint64_t>(count) * sizeof(RowId);

            // Read old filter data, select survivors, write back.
            std::vector<ColumnData> old_cols;
            read_filter_columns(buf.data() + old_filter_off, count,
                                manifest_.schema, old_cols);
            auto new_cols = select_filter_rows(old_cols, manifest_.schema,
                                               survivors);
            write_filter_columns(nb_buf.data() + new_filter_off, new_count,
                                 manifest_.schema, new_cols);
        }

        // --- Update header ---
        auto* nlh = reinterpret_cast<TreeLeafHeader*>(nb_buf.data());
        nlh->count = new_count;
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

}  // namespace sextant::tree
