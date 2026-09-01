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
/// Quantizer-agnostic: supports PQ (4/8-bit) and PRQ (4-bit).

#include "tree/page_file.hpp"
#include "tree/page_allocator.hpp"
#include "tree/superblock.hpp"
#include "tree/tree_manifest.hpp"
#include "tree/tree_nodes.hpp"
#include "tree/cardinality.hpp"  // CardinalityTable (Phase D selectivity estimation)
#include <sextant/column_data.hpp>
#include "sextant/config.hpp"
#include "sextant/types.hpp"
#include "sextant/schema.hpp"
#include <sextant/vector_source.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sextant { class PqQuantizer; }
namespace sextant { class ScalarLloydMaxQuantizer; }

namespace sextant::tree {

/// A routed leaf candidate: the extent to scan + routing context.
struct LeafCandidate {
    PageId   page;           // leaf extent start page
    uint64_t pages;          // leaf extent length
    float    centroid_dist;  // distance from query to the leaf's parent centroid
    const float16_t* centroid;  // FP16 centroid of the leaf
};

/// The hierarchical IVF tree index.
///
/// Built via `build_streaming_pca()` (streaming Lloyd build), searched via
/// `search()`. On open, the entire file is mmap'd read-only; routing reads
/// internal nodes and leaf headers from the mmap; leaf code scanning reads
/// from the mmap too.
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
        ResolvedParams params;         // PQ/PRQ config
        float adaptive_probe_gap = 0.0f;  // geometric gap pruning (0=off default;
                                          // see ResolvedParams note — 1.5 silently
                                          // destroyed recall on noise-heavy data)
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
        std::vector<ColumnData> filter_column_data;

        /// Payload data (Phase E). When non-empty, per-vector opaque payload
        /// blobs are written to per-leaf payload extents. payload_offsets[i]
        /// ..payload_offsets[i+1] gives the byte range for row_id i.
        /// payload_offsets has N+1 entries. Must be non-empty when
        /// filter_schema.has_payload is true. Only build_streaming_pca consumes
        /// this; the other build paths ignore it.
        const uint8_t* payload_data = nullptr;
        const uint32_t* payload_offsets = nullptr;  // N+1 entries
    };

    /// PCA-preconditioned streaming build: project vectors onto top-k PCs
    /// before routing. On high-LID data (d_eff≈2), this exposes the manifold
    /// structure so k-means converges. Scan codes stay in original space.
    /// Handles all depths: depth=1 (n_leaves ≤ k_root, root → leaves
    /// directly), depth=2 (root → L2 internal nodes → leaves), and depth=3
    /// (root → L1 → L2 → leaves when k_root exceeds k_root_max_depth2).
    static BuildResult build_streaming_pca(VectorSource& source,
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

    /// Sweep mode: evaluate multiple rerank shortlist widths W from ONE scan.
    /// When sweep_Ws is non-null, the search runs with W = max(*sweep_Ws) and
    /// fills sweep_out[i] with the result set (top-k, deduped) for sweep_Ws[i].
    /// Bit-exact equivalent to running search() separately with each W (each
    /// W result is a prefix cut of the W_max scan order, followed by the same
    /// dedup/top-k selection as the normal path).
    /// Ignored when payload_locs is non-null (payload locations require the
    /// single-W path); also ignored when either sweep argument is null or
    /// sweep_Ws is empty.
    std::vector<Candidate> search(const float* query, uint32_t k,
                                  const SearchConfig& config,
        std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs,
        const std::vector<uint32_t>* sweep_Ws,
        std::vector<std::vector<Candidate>>* sweep_out) const;

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
    uint64_t n_pages() const { return superblock_.n_pages(); }
    uint32_t n_probe_l0_default() const { return manifest_.n_probe_l0; }
    uint32_t n_probe_ln_default() const { return manifest_.n_probe_ln; }
    const std::string& quantizer_type() const { return manifest_.quantizer_type; }
    const PqQuantizer& quantizer() const { return *quantizer_; }

    /// Safe metric accessor: works even when quantizer_ is null (local_pq).
    /// Uses the manifest's metric field (stored for all trees).
    MetricKind metric() const {
        return static_cast<MetricKind>(manifest_.metric);
    }

    /// True when this index uses per-leaf codebooks (no global codebook).
    bool is_local_pq() const { return manifest_.quantizer_type == "local_pq"; }

    /// Global cardinality table (Phase D). Empty when no filter columns were
    /// present at build time. Used for predicate selectivity estimation.
    const CardinalityTable& cardinality() const { return card_table_; }

    // --- Dynamic insert/delete (single-writer, multi-reader) ---

    /// A single point to insert: vector + row_id + optional filter column
    /// values + optional payload. filter_values must match the index's schema
    /// (one ColumnData per column, each with exactly 1 row). payload may be
    /// empty.
    struct InsertPoint {
        const float* vector;
        RowId row_id;
        std::vector<ColumnData> filter_values;  // empty = no filter cols
        std::string_view payload;                   // empty = no payload
    };

    /// Insert a batch of points. Routes each to its nearest leaf, groups by
    /// leaf, grows each affected leaf once, appends codes/row_ids/filter
    /// values/payloads, updates summaries (eager), and commits. Leaves that
    /// exceed 2×leaf_capacity after insertion are split.
    ///
    /// Single-writer: no concurrent inserts/deletes. Readers see the old mmap
    /// until remap_after_commit() (called automatically).
    void insert_batch(const std::vector<InsertPoint>& points);

    /// Delete a batch of row_ids. Routes each to its leaf, finds the slot by
    /// scanning row_ids, swap-removes (last slot → deleted slot), decrements
    /// count, marks summary_dirty. Commits once at the end.
    void delete_batch(const std::vector<RowId>& row_ids);

    /// Vacuum result statistics.
    struct VacuumResult {
        uint64_t leaves_scanned = 0;
        uint64_t summaries_repaired = 0;
        uint64_t cardinality_entries_rebuilt = 0;
        double   elapsed_sec = 0;
    };

    /// Vacuum configuration.
    struct VacuumConfig {
        /// Rebuild the cardinality table by re-scanning all filter columns.
        /// Expensive (reads every leaf) but fixes selectivity drift from deletes.
        bool rebuild_cardinality = false;
        /// Number of leaves to process per commit batch.
        uint32_t batch_size = 256;
    };

    /// Repair stale filter summaries on dirty leaves. After delete_batch marks
    /// leaves as summary_dirty, vacuum recomputes their summaries from surviving
    /// filter column data, restoring pruning effectiveness. Optionally rebuilds
    /// the cardinality table.
    ///
    /// Single-writer: same contract as insert_batch / delete_batch.
    VacuumResult vacuum() { return vacuum(VacuumConfig{}); }
    VacuumResult vacuum(const VacuumConfig& config);

    /// Defrag result statistics.
    struct DefragResult {
        uint64_t leaves_relocated = 0;
        uint64_t pages_reclaimed = 0;
        double   elapsed_sec = 0;
    };

    /// Defrag configuration.
    struct DefragConfig {
        /// Attempt to shrink the file by truncating trailing free pages.
        bool shrink_file = true;
        /// Number of leaves to process per commit batch.
        uint32_t batch_size = 256;
    };

    /// Compact fragmented leaf extents into contiguous pages and optionally
    /// shrink the file. Reclaims space wasted by delete/split churn.
    ///
    /// Single-writer: same contract as insert_batch / delete_batch.
    DefragResult defrag() { return defrag(DefragConfig{}); }
    DefragResult defrag(const DefragConfig& config);

    /// Returns the number of live vectors across all leaves (sum of leaf
    /// counts). Requires a mutable open (reads leaf headers via file_).
    uint64_t live_count() const;

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
    std::unique_ptr<ScalarLloydMaxQuantizer> scalar_lm_quantizer_;

    // --- Leaf extent table (indirection: leaf_id → page+pages) ---
    // Loaded at open() from the leaf_table blob. Empty when the tree was
    // built without a leaf table (legacy format). When non-empty, ChildEntry
    // child_page for is_leaf=1 children is a leaf_id index into this table.
    std::vector<LeafTableEntry> leaf_table_;

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

    /// Brute-force PQ-decode fallback for extreme low selectivity (<1%).
    /// Walks ALL leaves, checks summaries, scans filter columns for exact
    /// matches, PQ-decodes matching vectors, computes exact FP32 distance.
    /// Returns top-k by exact distance.
    std::vector<Candidate> search_brute_force_filtered(
        const float* query, uint32_t k, const SearchConfig& config,
        const std::vector<uint32_t>& pred_col_indices,
        const std::vector<uint32_t>& geo_lng_col_indices = {},
        std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs
            = nullptr) const;

    // --- Mutable-state private helpers ---

    /// Route a single vector to its nearest leaf_id. Uses the same PCA/FP16
    /// routing as search, but n_probe=1 at every level (greedy descent).
    /// For depth=1: root → leaf directly. For depth≥2: root → L1 → leaf.
    /// Returns the leaf_id (index into leaf_table_).
    uint32_t route_to_leaf_id_(const float* query) const;

    /// Route a vector to its nearest leaf_id for depth=2 PCA trees.
    /// Separated because it uses pca_leaf_centroids_ directly.
    uint32_t route_to_leaf_id_depth2_pca_(const float* query,
                                           const float* query_pca) const;

    /// Read a leaf extent into a buffer. Returns the buffer (sized to pages).
    std::vector<uint8_t> read_leaf_(uint32_t leaf_id) const;

    /// Write a leaf buffer back to the file, updating the leaf table entry.
    /// If new_pages differs from the current entry, allocates a new extent
    /// and frees the old one.
    void write_leaf_(uint32_t leaf_id, std::vector<uint8_t>& buf,
                     uint32_t new_pages, PageAllocator& alloc);

    /// Remap the read-only mmap after a mutation commit. Search uses the mmap;
    /// after writes through file_, the mmap is stale until remapped.
    void remap_();

    /// Flush the leaf table blob + superblock commit. Called after any
    /// mutation that changes leaf_table_ or n_pages.
    void commit_mutable_(PageAllocator& alloc);

    /// Split a leaf that exceeds 2×leaf_capacity into two leaves via k-means
    /// (K=2) on PQ codes. Writes two new leaf extents, decodes medoid
    /// centroids → FP16 for the parent node, updates the parent node's child
    /// list (adds a sibling), grows the leaf table by one entry, frees the
    /// old leaf extent. Returns the new leaf_id of the sibling.
    /// Caller must commit_mutable_ after all splits.
    uint32_t split_leaf_(uint32_t leaf_id, PageAllocator& alloc);

    /// Update the parent node of a leaf to add a new child entry (the sibling
    /// from a split). For depth=1, the parent is the root node. For depth≥2,
    /// the parent is the L2 internal node containing the leaf. Reads the
    /// parent from disk, appends the child entry, grows the extent if needed,
    /// writes back. Returns the new (page, pages) of the parent.
    void add_child_to_parent_(uint32_t leaf_id, uint32_t new_leaf_id,
                               const std::vector<float16_t>& cent0_fp16,
                               const std::vector<float16_t>& cent1_fp16,
                               uint32_t new_leaf_pages,
                               PageAllocator& alloc);

    /// Update one child entry in an internal node (used when a child node
    /// relocates during split). Reads the node at node_page, updates the
    /// child_page/child_pages of the given slot, writes back.
    void update_internal_child_(PageId node_page, uint32_t node_pages,
                                 uint32_t slot,
                                 PageId new_child_page,
                                 uint32_t new_child_pages,
                                 PageAllocator& alloc);

    /// Rebuild pca_leaf_centroids_ and pca_leaf_base_ from the current on-disk
    /// tree structure. Called by remap_() after every structural change
    /// (insert, delete, split). Derives PCA centroids by projecting the inline
    /// FP16 leaf centroids through pca_proj_, so new leaves from splits get
    /// correct routing centroids without rewriting the PCA blob on disk.
    void rebuild_pca_leaf_centroids_();
};

}  // namespace sextant::tree
