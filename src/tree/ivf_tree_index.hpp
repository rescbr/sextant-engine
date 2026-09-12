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
#include "tree/plane.hpp"
#include "tree/superblock.hpp"
#include "tree/tree_manifest.hpp"
#include "tree/tree_nodes.hpp"
#include "tree/leaf_coder.hpp"
#include "tree/leaf_extent_cache.hpp"
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

struct SearchScratch;  // internal per-thread search buffers (ivf_tree_index.cpp)

/// Frontier entry during routing (shared by search / search_batch / the
/// feedback probe loop).
struct ProbeEntry {
    float    dist;
    PageId   page;
    uint64_t pages;
    uint16_t is_leaf;
    const float16_t* centroid;  // inline FP16 centroid (points into mmap)
};
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
/// Per-thread override for the i8 SDOT scan kernel (C ABI harnesses
/// serving mixed rows): >0 forces on, 0 forces off, -1 (default) = auto
/// (int8 kernel on AVX512/VNNI, float FMA otherwise). See the kernel
/// comment in search().
void set_scan_i8_override(int v);

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
        float probe_fraction = 0.0f;   // corpus-fraction probe budget to bake
                                       // into the manifest (0 = default 0.5)
        ResolvedParams params;         // PQ/PRQ config
        float adaptive_probe_gap = 0.0f;  // geometric gap pruning (0=off default;
                                          // see ResolvedParams note — 1.5 silently
                                          // destroyed recall on noise-heavy data)
        float median_lid = 0.0f;
        uint32_t num_threads = 0;
        uint32_t pca_dims = 32;      // PCA dimensions for build_streaming_pca
                                      // (0 = DISABLE PCA routing: full-dim
                                      //  fp16 centroid routing)
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

        /// Optional metrics sink (metrics.hpp). When null, phase records
        /// still accumulate into BuildResult::phases but emit only via the
        /// default log line created inside the build.
        metrics::MetricsSink* metrics_sink = nullptr;
    };

    // --- Search observability ---

    /// Aggregated search counters (queries, leaves probed, bytes touched,
    /// rerank count, LeafExtentCache hits/misses). Window semantics:
    /// accumulate across searches, take deltas via search_stats().
    /// snapshot_and_reset().
    const SearchStats& search_stats() const { return search_stats_; }

    /// Batch-path observability (search_batch only). Same delta contract
    /// as search_stats(): snapshot_and_reset() per window.
    const BatchStats& batch_stats() const { return batch_stats_; }

    /// Plane-cache hit/miss counters (cumulative since open; zeros when
    /// the plane cache is off). Lets benchmarks report leaf-tier and
    /// plane-tier hit rates separately.
    void plane_cache_hitmiss(uint64_t& hits, uint64_t& misses) const;

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
    ///
    /// `leaf_cache_bytes` (0 = off, the default and today's behavior) sizes
    /// an engine-owned LeafExtentCache for the search path: leaf extents are
    /// served from a DRAM-budgeted W-TinyLFU cache (pread on miss) instead of
    /// the mmap. This is the only BENCHMARK_RULES-reportable warm layer;
    /// internal/routing nodes always use the mmap (they are part of the
    /// routing DRAM budget). Mutations (insert/delete/split/vacuum/defrag)
    /// invalidate the cache wholesale via remap_().
    static std::unique_ptr<IVFTreeIndex> open(const std::string& path,
                                              uint64_t leaf_cache_bytes = 0,
                                              uint32_t cache_window_pct = 1,
                                              uint64_t plane_cache_bytes = 0);

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
    /// bit-exact equivalent to running search() separately with each W (each
    /// W result is a prefix cut of the W_max scan order, followed by the same
    /// dedup/top-k selection as the normal path).
    /// Ignored when payload_locs is non-null (payload locations require the
    /// single-W path); also ignored when either sweep argument is null or
    /// sweep_Ws is empty.
    std::vector<Candidate> search(const float* query, uint32_t k,
    const SearchConfig& config,
                                  std::vector<std::pair<const uint8_t*, uint32_t>>* payload_locs,
                                  const std::vector<uint32_t>* sweep_Ws,
                                  std::vector<std::vector<Candidate>>* sweep_out,
                                  std::vector<PageId>* visited_leaf_pages = nullptr) const;

    /// Subtree-major batch search (the 1B-scale path). Routes every query,
    /// inverts the probe sets into a leaf→queries fanout table, then sweeps
    /// each UNIQUE probed leaf exactly once in page order — every query that
    /// probed it is scanned against that single read. I/O is shared across
    /// the batch (measured 50x read amplification eliminated vs query-major
    /// at zero engine-owned DRAM); compute is not (each (leaf, query) pair
    /// still runs its own scan).
    ///
    /// `queries` holds nq row-major vectors of dim() floats. results[i]
    /// receives query i's top-k, produced by the SAME per-query
    /// selection semantics as search() — identical inputs yield the same
    /// candidates modulo pq_dist tie order under parallelism (the same
    /// contract as search()'s parallel scan path).
    ///
    /// Predicates: per-query overrides via `per_query_predicates` (entry
    /// i is query i's predicate list; an empty list = unfiltered query).
    /// Null = the base config's predicates apply to every query. Each
    /// query routes (selectivity, summary pruning, adaptive W) and
    /// evaluates (harvest-time column checks) against its OWN predicate
    /// set — mixed filters in one batch are fine.
    ///
    /// Read layer (one path, two backends): without a leaf cache the
    /// sweep reads each unique leaf with a direct page-ordered pread;
    /// with one it pins, scans, and unpins per leaf (cross-window hot
    /// set). Everything finalize needs from a leaf (row_id, rerank
    /// distance, predicate verdict) is harvested into W-bounded pools
    /// right after the (query, leaf) scan — provably containing every
    /// final-heap entry (pool and heap share the same total order), so
    /// there is no overflow path and no mmap dependency. Feedback
    /// probing is rejected (adaptive probe sets cannot be coalesced).
    /// Queries whose predicate selectivity triggers the brute-force
    /// filtered fallback are served by the per-query search() path.
    ///
    /// Concurrency: config.search_threads > 1 parallelizes both the route
    /// phase and the sweep (threads claim contiguous page-ordered chunks of
    /// the unique-leaf list; per-query heaps/setups are sharded per thread
    /// and merged deterministically).
    void search_batch(const float* queries, uint32_t nq, uint32_t k,
                                              const SearchConfig& config,
                                              std::vector<std::vector<Candidate>>& results,
                                              const std::vector<std::vector<Predicate>>*
                                                  per_query_predicates = nullptr,
                                              const std::vector<float>* per_query_probe_fraction =
                                                  nullptr) const;


    // --- Routing diagnostics (loss-decomposition harness) ---

    /// Physical location + stored vector count of every leaf, in leaf-table
    /// order. `page` is the first physical page of the leaf extent — the same
    /// value search() reports in `visited_leaf_pages` for scanned leaves.
    struct DebugLeafInfo { PageId page; uint64_t pages; uint32_t count; };
    std::vector<DebugLeafInfo> debug_leaf_info() const;

    /// Row ids stored in leaf `leaf_id` (reads the leaf extent's row_id
    /// array). With closure replication a row id can appear in several
    /// leaves. Returns empty for an out-of-range leaf_id.
    std::vector<RowId> debug_leaf_row_ids(uint32_t leaf_id) const;

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
    float probe_fraction_default() const { return manifest_.probe_fraction; }
    uint32_t pca_dims_default() const { return manifest_.pca_dims; }
    const std::string& quantizer_type() const { return manifest_.quantizer_type; }
    /// Global PQ quantizer view (global-codebook families only). Callers
    /// must not invoke this on local/scalar families.
    const PqQuantizer& quantizer() const { return *coder_->pq_quantizer(); }

    /// The per-family leaf coder owning all quantizer state.
    const LeafCoder& coder() const { return *coder_; }

    /// Safe metric accessor: works even when quantizer_ is null (local_pq).
    /// Uses the manifest's metric field (stored for all trees).
    MetricKind metric() const {
        return static_cast<MetricKind>(manifest_.metric);
    }

    /// True when this index uses per-leaf codebooks (no global codebook).
    bool is_local_pq() const {
        return coder_ && coder_->leaf_state() == LeafState::CodedLocal;
    }

    /// Global cardinality table (Phase D). Empty when no filter columns were
    /// present at build time. Used for predicate selectivity estimation.
    const CardinalityTable& cardinality() const { return card_table_; }

    // --- Stage-1 routing plane (plane.hpp) ---

    /// True when the index carries a plane extent (superblock plane_page).
    /// Plane-bearing indexes rank leaves by max query·proj over the
    /// per-vector quantized PCA plane instead of centroid distance.
    bool has_plane() const { return plane_ != nullptr; }
    const PlaneIndex* plane() const { return plane_.get(); }

    /// Attaches a routing plane to an ALREADY-BUILT, plane-less index:
    /// trains the PCA basis + per-dim codebooks on `base` (n×dim
    /// row-major, spread-sampled), encodes every stored member
    /// (sequential row-order pass over `base`), writes the plane extent
    /// and re-commits the superblock. The index is IMMUTABLE afterwards
    /// (mutable ops throw while a plane is attached) — v1 semantics.
    /// Reopen (or rely on the in-place reload) to search with the plane.
    void attach_plane(const float* base, uint32_t n, uint32_t dim,
                      PlaneEncoding enc, uint16_t rank = 128,
                      uint32_t train_rows = 20000);

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
    mutable SearchStats search_stats_;  // relaxed atomics; search() is const
    mutable BatchStats batch_stats_;     // relaxed atomics; search_batch() is const
    // Serializes Phase-3 leaf sweeps ACROSS concurrent search_batch calls
    // (window pipelining): the sweep's page-ordered pread stream must be
    // the only leaf I/O in flight — two concurrent sweeps interleave into
    // random I/O (measured: a depth-2 CLI pipeline failed to finish).
    // Uncontended for single-window callers. Phases 0-2 (routing) are
    // untouched and may overlap a sweep freely — they never read leaf
    // pages, only the (mmap'd) plane extent and centroids.
    mutable std::mutex scan_stream_mu_;
    int fd_ = -1;
    const uint8_t* mmap_base_ = nullptr;  // mmap'd file base (read-only)
    uint64_t mmap_size_ = 0;
    // Engine-owned leaf-extent cache (null = off: search reads leaves via
    // mmap_base_, exactly the pre-cache behavior). See open().
    std::unique_ptr<class LeafExtentCache> leaf_cache_;
    /// Independent W-TinyLFU cache for PLANE row blocks (routing tier):
    /// leaf-aligned, ~10x denser per byte than leaf extents at b1g — hot
    /// leaves' routing rows stay resident under skewed traffic without
    /// competing with the leaf-extent budget. Keyed by the plane extent's
    /// page ids (disjoint from leaf extent pages).
    std::unique_ptr<class LeafExtentCache> plane_cache_;

    PageFile file_;
    Superblock superblock_;
    TreeManifest manifest_;
    CardinalityTable card_table_;  // per-value frequencies for selectivity (Phase D)
    std::unique_ptr<PlaneIndex> plane_;  // stage-1 routing plane (optional)
    // Single per-family coder: owns the quantizer objects + all
    // family-specific leaf layout / scan / mutation logic.
    std::unique_ptr<LeafCoder> coder_;

    // --- Leaf extent table (indirection: leaf_id → page+pages) ---
    // Loaded at open() from the leaf_table blob. Mandatory: every tree the
    // current build path writes one; is_leaf ChildEntry child_page values
    // are leaf_id indexes into this table.
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

    /// Resolve a leaf extent pointer for the search path. With the cache
    /// off: plain mmap pointer, `handle` left null. With it on: pin the
    /// extent (hit/miss counted into search_stats_), caller must unpin via
    /// release_leaf_pins once the scan (and any leaf_ptr uses) are done.
    const uint8_t* pin_leaf_(PageId page, uint32_t pages,
                             LeafExtentCache::Handle& handle,
                             bool* was_hit = nullptr) const;

    /// Release every pin taken by one search() call (QueryGuard exit path).
    void release_leaf_pins_(std::vector<LeafExtentCache::Handle>& pins) const;

    // --- Shared route/scan/finalize decomposition (search + search_batch) ---

    /// Resolve predicate column names to schema indices once per query.
    /// Returns false when a predicate (or geo longitude) references an
    /// unknown column → the query yields no results.
    bool resolve_pred_columns_(const SearchConfig& config,
                               std::vector<uint32_t>& pred_col_indices,
                               std::vector<uint32_t>& geo_lng_col_indices) const;

    /// Outcome of routing one query.
    enum class RouteStatus : uint8_t {
        Ok,                ///< candidates populated
        Empty,             ///< no leaf candidates (or unknown predicate column)
        FallbackFiltered,  ///< selectivity < 1%: caller must use the
                           ///< per-query brute-force filtered path
    };

    /// Route one query from the root to its leaf candidate set: query prep
    /// (FP16/PCA projection), predicate column resolution + selectivity,
    /// root scoring (fraction / gap / filter-directed variants), level-by-
    /// level descent, summary-based leaf pruning. Fills
    /// scratch.{pred_col_indices, geo_lng_col_indices, selectivity,
    /// route_node_bytes, root_dists} as side effects (feedback probing
    /// reuses root_dists). This is the exact routing block of search(),
    /// extracted so search_batch() shares it verbatim.
    RouteStatus route_query_(const float* query, const SearchConfig& config,
                             SearchScratch& scratch,
                             std::vector<LeafCandidate>& candidates) const;

    /// Plane stage-1 routing: rank ALL leaves by max query·proj over the
    /// routing plane, then select in score order until the cumulative
    /// leaf extent reaches probe_fraction of the corpus (the plane makes
    /// f the direct stage-2 cut). Predicate queries fall back to legacy
    /// descent (plane + predicate composition lands with the batch
    /// sweep). Sets the scratch side channels to their no-filter
    /// defaults; feedback probing is not meaningful (no descent).
    void plane_route_(const float* query, const SearchConfig& config,
                      std::vector<LeafCandidate>& candidates) const;

    /// Batched plane stage-1 (deployment shape): ONE leaf-major plane
    /// sweep for the whole batch (blocks read once, scored against all
    /// queries while cache-hot), then per-query fraction selection.
    void plane_route_batch_(const float* queries, uint32_t nq,
                            const SearchConfig& config,
                            std::vector<std::vector<LeafCandidate>>& out)
        const;

    /// Expand one internal frontier entry: read the node, score children
    /// (PCA leaf centroids when use_pca_leaves, else inline FP16), keep the
    /// top-n_probe_ln with gap pruning, resolve leaf ids to physical pages.
    /// Summary-aware: children whose filter summary can't match the
    /// predicates are pruned. node_bytes accumulates internal-node I/O.
    void expand_probe_(const ProbeEntry& e, bool use_pca_leaves,
                       uint32_t root_child_for_pca, float gap,
                       uint32_t n_probe_ln, const SearchConfig& config,
                       const std::vector<uint32_t>& pred_col_indices,
                       const std::vector<uint32_t>& geo_lng_col_indices,
                       SearchScratch& scratch,
                       std::vector<ProbeEntry>& out,
                       uint64_t& node_bytes) const;

    /// Per-query post-scan pipeline shared by search() and search_batch():
    /// materialize heap survivors (row_id + leaf pointers), predicate-filter
    /// the heap, rerank (coder or exact_rerank_base), dedup by row_id,
    /// adaptive shortlist cut, top-k, distance sort. Plain path only —
    /// payload_locs / sweep_Ws variants remain inline in search().
    void finalize_query_(const float* query, uint32_t k,
                         const SearchConfig& config,
                         SearchScratch& scratch,
                         std::vector<LeafCandidate>& candidates,
                         std::vector<const uint8_t*>& leaf_ptrs,
                         std::vector<HeapEntry>& sheap,
                         ScanSetup& scan_setup,
                         std::vector<Candidate>& results) const;

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
