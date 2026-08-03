#pragma once

/// @file ivf_tree_index.hpp
/// Hierarchical IVF Tree index — the replacement for flat K-shard partitioning.
///
/// The tree partitions vectors top-down along the data manifold. A two-level
/// tree (Phase 1) has a root node pointing to K₁ level-1 nodes, each pointing
/// to leaves containing 5-10k vectors stored as FastScan PQ codes.
///
/// At search time:
/// 1. FP16 distance from query to all root centroids → sort → top n_probe_l0.
/// 2. For each probed root child: FP16 distance to its children → top n_probe_ln.
/// 3. Dispatch fadvise prefetch for all selected leaf extents.
/// 4. Scan leaves (FastScan), maintain top-W heap. Merge → rerank → return top-k.
///
/// The index file is a single mmap'd page-based file (see superblock.hpp).
/// Quantizer-agnostic: supports PQ (4/8-bit), PRQ (4-bit), RaBitQ (4-bit+factors).

#include "tree/page_file.hpp"
#include "tree/page_allocator.hpp"
#include "tree/superblock.hpp"
#include "tree/tree_manifest.hpp"
#include "tree/tree_nodes.hpp"
#include "tree/cardinality.hpp"  // CardinalityTable (Phase D selectivity estimation)
#include "engine/mem_source.hpp"  // MemColumnData (filter column write path, Phase C)
#include "sextant/config.hpp"
#include "sextant/types.hpp"
#include "sextant/schema.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sextant { class PqQuantizer; }

namespace sextant::tree {

/// A routed leaf candidate: the extent to scan + routing context.
struct LeafCandidate {
    PageId   page;           // leaf extent start page
    uint64_t pages;          // leaf extent length
    float    centroid_dist;  // distance from query to the leaf's parent centroid
    const float16_t* centroid;  // FP16 centroid of the leaf (for RaBitQ LUT rebuild)
};

/// The hierarchical IVF tree index.
///
/// Built via `build()` (bottom-up bulk build), searched via `search()`.
/// On open, the entire file is mmap'd read-only; routing reads internal nodes
/// and leaf headers from the mmap; leaf code scanning reads from the mmap too.
class IVFTreeIndex {
public:
    IVFTreeIndex() = default;
    ~IVFTreeIndex();

    IVFTreeIndex(const IVFTreeIndex&) = delete;
    IVFTreeIndex& operator=(const IVFTreeIndex&) = delete;

    // --- Build parameters ---
    struct BuildConfig {
        uint32_t k_root = 0;           // root branching factor (0 = auto)
        uint32_t leaf_capacity = 5000; // max vectors per leaf
        uint16_t n_probe_l0 = 0;       // probe count at level 0 (0 = auto)
        uint16_t n_probe_ln = 0;       // probe count at deeper levels (0 = auto)
        ResolvedParams params;         // PQ/PRQ/RaBitQ config
        float adaptive_probe_gap = 0.0f;
        float median_lid = 0.0f;
        uint32_t num_threads = 0;
        uint32_t pca_dims = 32;      // PCA dimensions for build_streaming_pca
        uint32_t max_lloyd_passes = 10;  // max streaming Lloyd refinement passes
        uint32_t k_root_max_depth2 = 2048;  // k_root above this forces depth-3
        float closure_multiplier = 0.15f;  // closure epsilon = multiplier × mean_gap
        Schema filter_schema;        // empty = no filter columns (today's behavior)

        /// Filter column data (Phase C). When non-empty, filter column values
        /// are written to leaf extents during build. Indexed by row_id
        /// (0..N-1). Must match filter_schema column count and types. Only
        /// build_streaming_pca consumes this; the other build paths ignore it.
        std::vector<MemColumnData> filter_column_data;

        /// Payload data (Phase E). When non-empty, per-vector opaque payload
        /// blobs are written to per-leaf payload extents. payload_offsets[i]
        /// ..payload_offsets[i+1] gives the byte range for row_id i.
        /// payload_offsets has N+1 entries. Must be non-empty when
        /// filter_schema.has_payload is true. Only build_streaming_pca consumes
        /// this; the other build paths ignore it.
        const uint8_t* payload_data = nullptr;
        const uint32_t* payload_offsets = nullptr;  // N+1 entries
    };

    /// Build a tree index from a flat fbin file.
    /// Writes the tree to `output_path` (a single file).
    static BuildResult build(const std::string& base_path,
                             const std::string& output_path,
                             const BuildConfig& cfg);

    /// Streaming build: sample k-means for root centroids, then stream all
    /// vectors through the tree (route by FP16 distance, append to leaf,
    /// split on overflow). O(1) RAM regardless of N. No global k-means.
    static BuildResult build_streaming(const std::string& base_path,
                                        const std::string& output_path,
                                        const BuildConfig& cfg);

    /// Two-phase streaming build: greedy stream (phase 1) + one refinement
    /// pass that re-assigns each vector to its nearest leaf centroid (phase 2).
    /// Combines streaming speed with k-means-quality assignment.
    static BuildResult build_streaming_refined(const std::string& base_path,
                                                const std::string& output_path,
                                                const BuildConfig& cfg);

    /// PCA-preconditioned streaming build: project vectors onto top-k PCs
    /// before routing. On high-LID data (d_eff≈2), this exposes the manifold
    /// structure so k-means converges. Scan codes stay in original space.
    static BuildResult build_streaming_pca(const std::string& base_path,
                                            const std::string& output_path,
                                            const BuildConfig& cfg);

    /// Open an existing tree index for searching. Mmaps the file.
    static std::unique_ptr<IVFTreeIndex> open(const std::string& path);

    // --- Search ---

    /// Search for the k nearest neighbors of `query` (dim floats).
    /// Returns up to k candidates sorted by ascending distance.
    ///
    /// If `payload_locs` is non-null, it is populated with (leaf_ptr, local_idx)
    /// for each returned result (same length/order as the returned vector),
    /// enabling O(1) payload fetch via fetch_payload(). The search path is
    /// unchanged otherwise — payload locations are only collected for the final
    /// top-k after scan + rerank complete.
    std::vector<Candidate> search(const float* query, uint32_t k,
                                  const SearchConfig& config,
        std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs
            = nullptr) const;

    /// Fetch the opaque payload blob for a result. O(1): reads the leaf's
    /// payload extent from the mmap, indexes by slot.
    /// `leaf_ptr` = mmap base of the leaf extent (from a search payload_loc).
    /// `slot` = vector index within the leaf.
    /// Returns the payload blob (non-owning view into mmap), or empty when
    /// the leaf has no payload extent.
    std::string_view fetch_payload(const uint8_t* leaf_ptr,
                                    uint32_t slot) const;

    // --- Accessors ---

    uint32_t dim() const { return manifest_.dim; }
    uint16_t m4() const { return manifest_.m4; }
    uint16_t depth() const { return manifest_.depth; }
    uint32_t k_root() const { return manifest_.k_root; }
    uint32_t n_leaves() const { return manifest_.n_leaves; }
    uint32_t n_probe_l0_default() const { return manifest_.n_probe_l0; }
    uint32_t n_probe_ln_default() const { return manifest_.n_probe_ln; }
    const std::string& quantizer_type() const { return manifest_.quantizer_type; }
    const PqQuantizer& quantizer() const { return *quantizer_; }

    /// Global cardinality table (Phase D). Empty when no filter columns were
    /// present at build time. Used for predicate selectivity estimation.
    const CardinalityTable& cardinality() const { return card_table_; }

private:
    // --- Open state ---
    std::string path_;
    int fd_ = -1;
    const uint8_t* mmap_base_ = nullptr;  // mmap'd file base (read-only)
    uint64_t mmap_size_ = 0;

    PageFile file_;
    Superblock superblock_;
    TreeManifest manifest_;
    CardinalityTable card_table_;  // per-value frequencies for selectivity (Phase D)
    std::unique_ptr<PqQuantizer> quantizer_;

    // Root node (parsed from mmap at open time; points into mmap_base_).
    const TreeNodeHeader* root_header_ = nullptr;
    // Flat array of root child centroids (FP16, k_root × dim).
    // Points into mmap_base_ at the root node's child entries.
    struct RootChild {
        PageId page;
        uint64_t pages;
        uint16_t is_leaf;
        const float16_t* centroid;  // points into mmap
    };
    std::vector<RootChild> root_children_;

    // --- PCA routing state (loaded from pca blob if pca_dims > 0) ---
    uint32_t pca_dims_ = 0;           // 0 = no PCA routing (use FP16)
    std::vector<float> pca_proj_;     // projection matrix: pca_dims × dim (row-major)
    std::vector<float> pca_mean_proj_; // mean projection: pca_dims (precomputed dot(proj_k, mean))
    std::vector<float> pca_root_centroids_; // k_root × pca_dims (root centroids in PCA space)
    // For depth=2: leaf centroids in PCA space, stored per level-1 child.
    // Flat array: all leaves' PCA centroids, indexed by leaf order.
    // The search accesses them via the mmap'd level-1 node structure.
    std::vector<float> pca_leaf_centroids_; // n_leaves_total × pca_dims
    std::vector<uint64_t> pca_leaf_base_;  // per root child: starting global leaf ID

    /// Parse the root node from the mmap.
    void load_root_from_mmap();

    /// Close mmap + fd.
    void close();

    /// RaBitQ-specific search (per-leaf LUT rebuild + factor finalization).
    std::vector<Candidate> search_rabitq(const float* query, uint32_t k,
                                          const SearchConfig& config,
        std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs
            = nullptr) const;

    /// Brute-force PQ-decode fallback for extreme low selectivity (<1%).
    /// Walks ALL leaves, checks summaries, scans filter columns for exact
    /// matches, PQ-decodes matching vectors, computes exact FP32 distance.
    /// Returns top-k by exact distance.
    std::vector<Candidate> search_brute_force_filtered(
        const float* query, uint32_t k, const SearchConfig& config,
        const std::vector<uint32_t>& pred_col_indices,
        std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs
            = nullptr) const;
};

}  // namespace sextant::tree
