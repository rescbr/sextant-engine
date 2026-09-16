#pragma once

/// @file config.hpp
/// Build and search configuration structs.

#include <sextant/types.hpp>
#include <sextant/schema.hpp>
#include <sextant/metrics.hpp>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <string>
#include <vector>

namespace sextant {

class EngineTrace;  // engine_trace.hpp — caller-owned diagnostics sink

/// Scan-feedback probing (tree routing, experimental): instead of a fixed
/// probe fraction, root children are probed in routing (PCA-distance) order
/// one subtree block at a time, and probing STOPS on scan feedback — the
/// only untried member of the routing-stopping family (score-relative
/// thresholds were falsified 2026-09-06: no gain at matched mean fraction).
/// Per-query cost becomes adaptive: easy queries stop early, hard queries
/// keep reading — the honest generalization of probe_fraction's page-
/// weighted contract.
///
/// Semantics: each root child's ENTIRE subtree is scanned into the shared
/// top-W heap before the rule is evaluated (the block is the I/O unit —
/// contiguous extent, same as fraction routing selects). Rules:
///   Fixed — stop when cumulative subtree pages >= fixed_fraction × total
///           (the shipped probe_fraction behavior; the control arm).
///   Stall — stop after `m` consecutive blocks that contributed no entry
///           to the current top-k (k = search k).
///   Kth   — stop after `m` consecutive blocks with no improvement of the
///           k-th best scan key.
/// `min_blocks` are always probed before Stall/Kth may fire. Requires
/// n_probe == 0 and tree depth <= 2; silently ignored otherwise.
struct FeedbackProbe {
    enum class Mode : uint8_t { Off = 0, Fixed, Stall, Kth };
    Mode mode = Mode::Off;
    float fixed_fraction = 0.0f;   ///< Fixed: page-weighted budget [0,1]
    uint32_t m = 0;                ///< Stall/Kth: consecutive-block threshold
    uint32_t min_blocks = 1;       ///< Blocks probed before rules activate
};

/// Build configuration. Fields set to 0/default are auto-resolved.
/// Graph construction distance source (build-time only; search always
/// uses PQ LUTs regardless). Controls which representation the graph
/// builder's prune/construct distances are computed against.
/// Measured impact (docs/optimization_levers_and_attribution.md): FP16
/// build −0.5pp recall (FP16 precision < PQ LUT's FP32 accumulation);
/// FP32 build −0.6pp (train/serve skew — graph optimized for the wrong
/// metric). PQ-construct is the default and the RAM-lightest option;
/// FP32 additionally loads N×dim×4 bytes of raw vectors (K=1 path only).
enum class GraphBuildMetric : uint8_t {
    PqConstruct = 0,
    Fp16,
    Fp32,
};

struct BuildConfig {
    uint16_t R = 0;             ///< 0 = auto from N
    uint16_t L = 0;             ///< 0 = auto from R
    float alpha = 0.0f;         ///< Vamana prune threshold: alpha * d(p,pp) <= d(q,pp).
                                ///< 0 = auto (1.2).
    uint16_t pq_m = 0;          ///< 0 = auto (reservoir probe)
    uint8_t pq_bits = 8;        ///< 4, 8, or 0 (auto via reservoir probe)
    /// PQ max distortion for auto (m, bits) selection. Used only when pq_m or
    /// pq_bits is auto. 0 = use the built-in default (0.05).
    ///
    /// Distortion = median |1 - pq_estimated_distance / true_distance| over the
    /// true top-10 neighbors of 200 probe queries (measured on a 20K reservoir
    /// subset). 0.0 = PQ perfectly preserves neighbor distances; 0.15 = the
    /// median true-neighbor distance estimate is off by 15%. Always ≥0, lower is
    /// better; direction-agnostic (doesn't matter if PQ over- or under-estimates).
    /// This is a geometric, dataset-class-robust signal — unlike recall counts,
    /// which are fragile under near-duplicate clustering.
    ///
    /// Selection policy (distortion-bounded min cost):
    ///   1. Filter to configs with distortion ≤ max_distortion.
    ///   2. Among eligible configs, pick minimum cost, where
    ///      cost = m × residency_factor(table_bytes).
    ///        - `m` is the gather count per code_distance/lut_distance call —
    ///          the dominant cost factor in both build and search. Linear in m.
    ///        - residency_factor captures the effective per-gather latency from
    ///          table residency: L2 ≈ 1.0, L3 ≈ 1.5, RAM ≈ 1.9. These are
    ///          empirically calibrated (not raw cycle counts) because hardware
    ///          prefetching and temporal locality dilute the cache cliff. The
    ///          search path uses the PQ LUT (m×K×4 bytes, always L2-resident),
    ///          so residency affects only the build path (code_distance from
    ///          the m×K²×4 cross-distance table).
    ///
    /// Tuning:
    ///   0.05 (default): tight — mirrors the prior 0.96 recall floor's effective
    ///     strictness. SIFT-128 → m=32/8-bit (recall 0.997). Most 8-bit configs
    ///     at adequate m qualify; 4-bit only when sub_dim is small enough to
    ///     keep distortion low. Tighten further for quality-critical builds;
    ///     loosen (0.10–0.15) for high-dimensional data where coarser PQ is
    ///     acceptable.
    ///   LOWER (e.g. 0.05): forces higher fidelity PQ (higher m). Use when recall
    ///     is critical and you're willing to pay for more distance computations.
    ///   HIGHER (e.g. 0.30): admits aggressive low-m configs. Good for very
    ///     large N where throughput dominates.
    ///
    /// Note: the bound applies to the *probe* distortion (measured on a 20K
    /// reservoir subset), not end-to-end search recall. The elbow + bound pick
    /// a PQ config with good distance fidelity; the graph then navigates using
    /// those estimates. End-to-end recall depends on graph params (R, L) and
    /// dataset structure (clustering, dimensionality) in addition to PQ quality.
    float pq_max_distortion = 0.0f;
    uint64_t build_ram_budget = 0;  ///< 0 = auto (50% of physical RAM)
    MetricKind metric = MetricKind::L2Sq;
    uint32_t num_threads = 0;   ///< 0 = hardware_concurrency
    uint32_t max_occlusion = 0; ///< 0 = auto (max(L_build, R+1))

    /// Target proximity (in-band fraction) for auto-parameter estimation.
    /// 0 = use default (0.95). Measured at target_topk. Drives R selection.
    float proximity_target = 0.0f;

    /// Target recall@target_topk for auto-parameter estimation.
    /// 0 = use default (0.95). Workload-dependent: set BOTH proximity_target and
    /// recall_target to gate on both (the stricter one binds); set just one to
    /// gate on that alone. Neither → default recall@0.95.
    float recall_target = 0.0f;

    /// The k at which recall/proximity targets are measured and at which mini-build
    /// search quality is evaluated. Default 100 (VIBE / modern-retrieval convention).
    /// Mini-build measurements (alpha sweep, R validation, PQ verify) all run at
    /// this k. Note: proximity semantics depend on k — at k=100 the k-th NN
    /// distance is larger than at k=10, so proximity is looser at higher k.
    uint32_t target_topk = 100;

    /// Entry-point count (k-means centroids). 0 = default (16). Higher = better
    /// coverage for clustered datasets, more startup distance computations.
    uint16_t n_entry_points = 0;
    /// Multi-start: number of entry points to seed each search from (top-M
    /// closest to the query). 0 = default (4).
    uint16_t n_search_entry_points = 0;
    // NOTE: recall_target (defined above) also serves as the search early-exit
    // target. When set (e.g. 0.95), estimate_config picks the cheapest config
    // meeting it AND beam_search terminates early when convergence is detected.
    // No separate target_recall field needed.

    /// Closure-factor inputs for auto estimation. 0 = auto-estimated by
    /// estimate_config (closure_f_target defaults to 0.15; closure_d_eff
    /// estimated from LID). When non-zero, these lock the closure inputs.
    float closure_f_target = 0.0f;
    float closure_d_eff = 0.0f;

    /// PQ anisotropic codebook training (scale-transform k-means). When true,
    /// each subspace's dims are scaled by √(eigval/mean_eigval) of the subspace
    /// covariance before k-means; centroids are unscaled back afterward. Only
    /// affects codebook training — encoding/search/serialized format unchanged.
    /// One-time build-time cost.
    bool pq_anisotropy = false;

    /// Use ScaNN-style anisotropic PQ (AnisotropicPqQuantizer subclass). When
    /// true, Builder constructs AnisotropicPqQuantizer instead of PqQuantizer.
    /// The subclass overrides train() with ScaNN's anisotropic Lloyd's
    /// algorithm; all hot-path methods are inherited unchanged from PqQuantizer.
    /// See docs/plans/metric_per_tier_plan.md Phase 2.
    bool anisotropic_pq = false;

    /// Quantizer type for the IVF-scan codebook: "pq" (default, standard
    /// k-means), "anisotropic-pq" (ScaNN-style), or "prq" (Product Residual
    /// Quantization — additive-residual variant of PQ; see
    /// quant/product_residual_quantizer.hpp). Drives the 3-way dispatch in
    /// build_ivf_scan. "anisotropic-pq" sets anisotropic_pq=true for back-compat.
    std::string quantizer_type = "pq";

    /// PRQ nsplits (number of contiguous sub-spaces) for the IVF-scan codebook.
    /// 0 = auto (dim/32 → sub_dim=32 per the plan). Only meaningful with
    /// quantizer_type="prq". Must divide both `dim` and `m4`.
    uint32_t prq_nsplits = 0;

    /// PRQ beam size for encoding (1 = greedy, >1 = beam search).
    /// Higher = better quality, slower encode. Only meaningful with
    /// quantizer_type="prq".
    uint32_t prq_beam_size = 1;

    /// PRQ encoding mode: "greedy" (default), "beam", or "icm".
    /// ICM uses on-the-fly coordinate descent + ILS perturbation.
    std::string prq_encode_mode = "greedy";

    /// PRQ ICM iterations (sweeps per ILS cycle). Default 4.
    uint32_t prq_icm_iters = 4;

    /// PRQ ILS iterations (perturb + ICM + accept cycles). Default 4.
    uint32_t prq_ils_iters = 4;

    /// PRQ ILS perturbation count (codes to randomize per cycle). Default 4.
    uint32_t prq_ils_perturb = 4;

    /// LSQ training iterations (alternating codebook update + ICM re-encode).
    /// 0 = progressive k-means only (default). ~25 for full LSQ training.
    uint32_t prq_lsq_train_iters = 0;

    /// OPQ (PCA rotation). When true, a d×d PCA rotation is learned from the
    /// training-sample covariance and applied to vectors before PQ encoding
    /// and to queries before LUT construction, so PQ splits align with the
    /// principal components. The rotation matrix is serialized with the
    /// quantizer. Adds a one-time ~20s eigendecomposition at d=768.
    bool pq_opq = false;

    /// Partition count override. 0 = auto (resolved from build_ram_budget,
    /// plus a recall-driven floor when `sharded_graph` is true). >0 forces exactly
    /// this many partitions. Only meaningful for partitioned / IVF builds.
    uint32_t partition_count = 0;

    /// IVF-probe mode. When true, resolve_params applies a recall-driven
    /// partition-count floor of `clamp(sqrt(N)/8, 2, 256)` so the IVF path
    /// has enough shards for good routing even when RAM would allow K=1.
    /// See docs/ivf_probe_design.md "K selection". The recall floor is a
    /// lower bound — the RAM-driven K still wins when it's larger.
    bool sharded_graph = false;

    /// Use the merged-graph build path (Vamana per-shard) instead of the
    /// default IVF-list-scan + 4-bit PQ FastScan path. Inverts the older
    /// `--ivf` flag: IVF-scan is now the default for K≥1; `merged_graph=true`
    /// restores the prior graph-inside-shard behavior (`build_ivf`) for
    /// low-recall / high-QPS tier where graph traversal wins (recall < 0.88).
    /// See ~/.local/state/maki/plans/sharing-eternal-louse.md.
    bool merged_graph = false;

    /// Number of subquantizers for the 4-bit FastScan codebook. 0 = auto
    /// (dim/4 — 192 at dim=768, matching the validated spike). Only meaningful
    /// for the IVF-scan path (merged_graph=false).
    uint16_t pq4_m = 0;

    /// Bits per segment for the IVF-scan codebook. 4 (default, FastScan
    /// nibble-packed) or 8 (byte-per-code, Quicker-ADC shuffle kernel).
    /// 8-bit lifts the recall ceiling on high-LID datasets (e.g. Sphere
    /// LID 20.8 caps at ~0.57 with 4-bit; 8-bit breaks the ceiling) at
    /// 2× index size and ~½ scan throughput. The codebook is shared across
    /// shards; the routing PQ is unaffected (always 8-bit).
    uint8_t scan_pq_bits = 4;

    /// Partition size-balance strength (SPANN-style λ, arXiv:2111.08566).
    /// 0 = off (plain k-means + closure). >0 caps each shard's size at
    /// (n/K)·(1 + 1/balance_factor), spilling excess vectors from oversized
    /// closure clusters (primary membership always preserved). Bounds the
    /// worst-case disk read per shard at paged-billion-scale. Default 4
    /// (SPANN λ balancing — zero search-time cost, reduces shard skew).
    float partition_balance_factor = 4.0f;

    /// Absolute margin for SPANN-style boundary posting. -1 = auto
    /// (0.2 × mean NN distance). 0 = ratio-based (K-fragile, deprecated).
    /// See docs/closure_factor_derivation.md.
    float closure_epsilon = -1.0f;

    /// Graph construction distance source (see GraphBuildMetric).
    GraphBuildMetric graph_build_metric = GraphBuildMetric::PqConstruct;

    /// Optional metrics sink (metrics.hpp). When null, phase records still
    /// accumulate into BuildResult::phases and emit via the default log line.
    metrics::MetricsSink* metrics_sink = nullptr;

    /// Adaptive probe gap (0 = auto, derived from LID by the estimator).
    float adaptive_probe_gap = 0.0f;
    /// Median LID (0 = unmeasured; set by the estimator).
    float median_lid = 0.0f;
};

/// Result of a build operation.
struct BuildResult {
    std::string index_path;
    uint64_t n_vectors = 0;
    Dim dim = 0;
    double build_time_sec = 0;
    uint16_t R = 0;
    uint16_t L_build = 0;
    uint16_t pq_m = 0;
    uint8_t pq_bits = 8;

    // --- Instrumentation (see metrics.hpp; totals + per-phase records) ---
    /// Thread-sum CPU across the build (getrusage RUSAGE_SELF deltas).
    double cpu_time_sec = 0;
    /// Seconds the build spent blocked in VectorSource::next() (I/O waits).
    double source_wait_sec = 0;
    /// Bytes read through the VectorSource during the build.
    uint64_t bytes_read = 0;
    /// Process peak RSS over the build (ru_maxrss; monotone upper bound).
    uint64_t peak_rss_bytes = 0;
    /// Per-phase records (sample/pca/kmeans/lloyd/stream/write or graph
    /// passes), in emission order. Empty only for paths not yet wired.
    std::vector<metrics::PhaseMetrics> phases;
};

/// Aggregated search counters (window semantics). The engine accumulates
/// relaxed atomics per query; callers take windows via snapshot_and_reset()
/// and derive rates/means. CPU attribution is deliberately NOT here:
/// process-wide rusage under concurrent search misattributes other threads'
/// cycles — windows get CPU at the harness boundary (see phase_timer.hpp).
/// All fields except wall are pure counts; wall is summed per-query span.
struct SearchStats {
    /// Plain snapshot of a window (all deltas since the last snapshot).
    struct Snapshot {
        uint64_t queries = 0;
        uint64_t leaves_probed = 0;
        uint64_t bytes_touched = 0;   ///< leaf pages × page size (the currency)
        uint64_t rerank_count = 0;    ///< candidates decoded+reranked
        uint64_t cache_hits = 0;      ///< LeafExtentCache hits (0 when off)
        uint64_t cache_misses = 0;    ///< LeafExtentCache misses
        uint64_t cache_bytes_filled = 0;  ///< disk bytes pread into the cache
        uint64_t routing_ns = 0;         ///< per-query descent wall, summed
        uint64_t node_bytes_read = 0;     ///< internal-node extents touched
        double wall_seconds = 0;
    };

    void on_query(double wall_s, uint64_t leaves, uint64_t bytes,
                  uint64_t reranked, uint64_t routing_ns = 0,
                  uint64_t node_bytes = 0) const {
        queries_.fetch_add(1, std::memory_order_relaxed);
        wall_ns_.fetch_add(static_cast<uint64_t>(wall_s * 1e9),
                           std::memory_order_relaxed);
        leaves_probed_.fetch_add(leaves, std::memory_order_relaxed);
        bytes_touched_.fetch_add(bytes, std::memory_order_relaxed);
        rerank_count_.fetch_add(reranked, std::memory_order_relaxed);
        if (routing_ns)
            routing_ns_.fetch_add(routing_ns, std::memory_order_relaxed);
        if (node_bytes)
            node_bytes_.fetch_add(node_bytes, std::memory_order_relaxed);
    }

    /// Record one LeafExtentCache pin outcome (search path). Zero-cost when
    /// the cache is off (never called).
    void on_cache_op(bool hit, uint64_t bytes_filled) const {
        if (hit) cache_hits_.fetch_add(1, std::memory_order_relaxed);
        else cache_misses_.fetch_add(1, std::memory_order_relaxed);
        if (bytes_filled) cache_bytes_filled_.fetch_add(bytes_filled,
                                                        std::memory_order_relaxed);
    }

    Snapshot snapshot_and_reset() const {
        Snapshot s;
        s.queries = queries_.exchange(0, std::memory_order_relaxed);
        s.leaves_probed = leaves_probed_.exchange(0, std::memory_order_relaxed);
        s.bytes_touched = bytes_touched_.exchange(0, std::memory_order_relaxed);
        s.rerank_count = rerank_count_.exchange(0, std::memory_order_relaxed);
        s.cache_hits = cache_hits_.exchange(0, std::memory_order_relaxed);
        s.cache_misses = cache_misses_.exchange(0, std::memory_order_relaxed);
        s.cache_bytes_filled =
            cache_bytes_filled_.exchange(0, std::memory_order_relaxed);
        s.wall_seconds = static_cast<double>(
            wall_ns_.exchange(0, std::memory_order_relaxed)) / 1e9;
        s.routing_ns = routing_ns_.exchange(0, std::memory_order_relaxed);
        s.node_bytes_read =
            node_bytes_.exchange(0, std::memory_order_relaxed);
        return s;
    }

private:
    mutable std::atomic<uint64_t> queries_{0};
    mutable std::atomic<uint64_t> wall_ns_{0};
    mutable std::atomic<uint64_t> leaves_probed_{0};
    mutable std::atomic<uint64_t> bytes_touched_{0};
    mutable std::atomic<uint64_t> rerank_count_{0};
    mutable std::atomic<uint64_t> cache_hits_{0};
    mutable std::atomic<uint64_t> cache_misses_{0};
    mutable std::atomic<uint64_t> cache_bytes_filled_{0};
    mutable std::atomic<uint64_t> routing_ns_{0};
    mutable std::atomic<uint64_t> node_bytes_{0};
};

/// Subtree-major batch observability (IVFTreeIndex::search_batch).
/// Counters accumulate across batch calls; snapshot per metrics window.
/// The per-query SearchStats keep their query-major meaning (each query's
/// own probe set size and bytes), so query-major and batch runs remain
/// comparable; these counters describe the COALESCING.
struct BatchStats {
    struct Snapshot {
        uint64_t batches = 0;          ///< search_batch calls
        uint64_t queries = 0;          ///< queries across batches
        uint64_t leaves_unique = 0;    ///< unique leaves swept (page-deduped)
        uint64_t leaf_scans = 0;       ///< (leaf, query) scans executed
        uint64_t bytes_unique = 0;     ///< unique-leaf bytes read (the floor)
        uint64_t read_ns = 0;          ///< wall spent inside leaf preads
        uint64_t scan_ns = 0;  ///< per-thread scan+harvest wall (no reads)
        uint64_t sweep_wall_ns = 0;    ///< sweep phase wall span (summed)
        uint64_t sweep_threads = 0;    ///< sweep thread count (summed)
        uint64_t fallback_queries = 0; ///< queries served by per-query paths
        double wall_seconds = 0;
    };

    void on_batch(uint64_t queries, uint64_t leaves_unique,
                  uint64_t leaf_scans, uint64_t bytes_unique,
                  uint64_t fallback, double wall_s) const {
        batches_.fetch_add(1, std::memory_order_relaxed);
        queries_.fetch_add(queries, std::memory_order_relaxed);
        leaves_unique_.fetch_add(leaves_unique, std::memory_order_relaxed);
        leaf_scans_.fetch_add(leaf_scans, std::memory_order_relaxed);
        bytes_unique_.fetch_add(bytes_unique, std::memory_order_relaxed);
        fallback_queries_.fetch_add(fallback, std::memory_order_relaxed);
        wall_ns_.fetch_add(static_cast<uint64_t>(wall_s * 1e9),
                           std::memory_order_relaxed);
    }

    /// Accumulate actual leaf-read wall (sweep workers, pread backend).
    /// bytes_unique/read_ns is the true effective read bandwidth —
    /// batch wall would underestimate it under compute-bound windows.
    void on_read(uint64_t read_ns) const {
        read_ns_.fetch_add(read_ns, std::memory_order_relaxed);
    }

    /// Accumulate per-thread scan+harvest wall inside the sweep (excludes
    /// preads) and the sweep phase's wall-clock span (once per window).
    /// read_ns+scan_ns vs sweep_wall x sweep_threads = overlap efficiency.
    void on_scan(uint64_t scan_ns) const {
        scan_ns_.fetch_add(scan_ns, std::memory_order_relaxed);
    }
    void on_sweep(uint64_t wall_ns, uint32_t threads) const {
        sweep_wall_ns_.fetch_add(wall_ns, std::memory_order_relaxed);
        sweep_threads_.fetch_add(threads, std::memory_order_relaxed);
    }

    Snapshot snapshot_and_reset() const {
        Snapshot s;
        s.batches = batches_.exchange(0, std::memory_order_relaxed);
        s.queries = queries_.exchange(0, std::memory_order_relaxed);
        s.leaves_unique = leaves_unique_.exchange(0, std::memory_order_relaxed);
        s.leaf_scans = leaf_scans_.exchange(0, std::memory_order_relaxed);
        s.bytes_unique = bytes_unique_.exchange(0, std::memory_order_relaxed);
        s.read_ns = read_ns_.exchange(0, std::memory_order_relaxed);
        s.scan_ns = scan_ns_.exchange(0, std::memory_order_relaxed);
        s.sweep_wall_ns =
            sweep_wall_ns_.exchange(0, std::memory_order_relaxed);
        s.sweep_threads =
            sweep_threads_.exchange(0, std::memory_order_relaxed);
        s.fallback_queries =
            fallback_queries_.exchange(0, std::memory_order_relaxed);
        s.wall_seconds = static_cast<double>(
            wall_ns_.exchange(0, std::memory_order_relaxed)) / 1e9;
        return s;
    }

private:
    mutable std::atomic<uint64_t> batches_{0};
    mutable std::atomic<uint64_t> queries_{0};
    mutable std::atomic<uint64_t> leaves_unique_{0};
    mutable std::atomic<uint64_t> leaf_scans_{0};
    mutable std::atomic<uint64_t> bytes_unique_{0};
    mutable std::atomic<uint64_t> read_ns_{0};
    mutable std::atomic<uint64_t> scan_ns_{0};
    mutable std::atomic<uint64_t> sweep_wall_ns_{0};
    mutable std::atomic<uint64_t> sweep_threads_{0};
    mutable std::atomic<uint64_t> fallback_queries_{0};
    mutable std::atomic<uint64_t> wall_ns_{0};
};

/// Search configuration.
struct SearchConfig {
    uint32_t k = 10;
    uint32_t L_search = 200;
    uint32_t io_limit = 0;  ///< 0 = unlimited (visit as many nodes as L_search allows)

    /// IVF-probe: number of shards to probe per query. 0 = auto (max(1, K/4)).
    /// Ignored for single-shard (non-IVF) indexes.
    uint32_t n_probe = 0;

    /// Tree: leaf probe count per root child (depth=2). 0 = manifest default.
    uint32_t n_probe_ln = 0;

    /// Tree: probe budget as a FRACTION of the corpus (leaf-coverage
    /// contract). When >0 (and n_probe is 0), routing walks root children
    /// nearest-first and keeps selecting until their cumulative subtree
    /// extent reaches probe_fraction of the total, then probes ALL leaves
    /// of the selected children (n_probe_ln is bypassed — per-child caps
    /// silently cost 5-8pp recall@10 on unbalanced trees). Scale-stable
    /// where absolute counts are not: measured on dbpedia-1536 (100K→933K),
    /// f=0.25 → ~0.96 recall@10 at ~0.25x flat-scan latency, f=0.5 →
    /// ~0.99 at ~0.5x, holding across corpus sizes while a pinned np8
    /// decayed 0.9998 → 0.90. n_probe > 0 (expert absolute override) and
    /// exhaustive take precedence.
    float probe_fraction = 0.0f;

    /// Stage-1 routing plane (plane.hpp): when the index carries a plane
    /// extent, rank leaves by max query·proj over the per-vector plane
    /// and select the probe set by page-weighted probe_fraction (the
    /// plane composes with this contract directly — f becomes the
    /// stage-2 cut). False forces legacy centroid routing even when a
    /// plane is present (A/B knob). Ignored for plane-less indexes.
    bool use_plane = true;

    /// Two-stage plane routing: fraction of leaves (by PCA centroid
    /// distance, page-weighted) that survive to the plane sweep. The
    /// centroid filter is RAM-resident and free; the plane sweep (the
    /// expensive sequential stage) then runs on survivors only —
    /// expense-ordered composition: centroids -> plane -> leaf scan.
    /// 0 = off (sweep everything). Requires PCA routing data (depth-2
    /// trees); silently ignored otherwise. Default 0.25: measured winner
    /// in every encoding family (b1g/u4lm/u4lm_pv) on cohere10m at both
    /// probe fractions; recall-neutral, QPS-positive.
    float plane_pre_prune = 0.25f;

    /// IVF-probe merge oversampling: each probed shard is searched at
    /// k_local = k × merge_oversample, then results are merged/deduped and
    /// truncated to k. 1 = no oversampling. Ignored for single-shard indexes.
    uint32_t merge_oversample = 2;

    /// IVF-probe multi-probe ratio (Voronoi-boundary recall recovery). After
    /// picking the n_probe nearest centroids, extend the probe set to include
    /// ANY centroid whose distance ≤ ratio × d[n_probe-1] (the n_probe-th
    /// nearest centroid's distance). 1.0 = strict n_probe (off, back-compat).
    /// Values >1.0 catch true NNs that live in boundary shards routing missed.
    /// Increases recall at variable per-query cost (boundary queries probe more
    /// shards). See docs/ivf_routing_analysis.md. Measured (c4a K=21, np=1):
    /// ratio=1.05 recovers +11.8pp routing recall for +0.51 avg shards/query
    /// (large win at low n_probe; negligible at np≥4 where routing is already
    /// near-optimal). Harmless when the threshold isn't triggered.
    /// Default 1.05 (was 1.0); set to 1.0 to disable multi-probe.
    /// Ignored for single-shard indexes.
    float multiprobe_ratio = 1.05f;

    /// Search-time early-exit patience (post-convergence stall count before
    /// terminating beam_search). 0 = disabled (full L_search). UINT32_MAX =
    /// defer to the index's baked-in value (back-compat default). Any other
    /// value overrides at query time. Build-time beam_search is NEVER affected
    /// (it always runs to completion regardless of this or the index value).
    /// IVF uses this to pass a higher patience than the merged default.
    uint32_t early_exit_patience = 0xFFFFFFFFu;  // kDeferToParams

    /// IVF-list-scan rerank depth W (Option A). The scan path returns the
    /// top-W candidates by 4-bit PQ distance; the database layer does exact
    /// rerank + top-k outside sextant-engine. W replaces the `k ×
    /// merge_oversample` shortlist the graph path uses — directly sets the
    /// candidate-list size handed back to the caller. 0 = default (1000,
    /// the τ×W grid winner: +0.4–0.7pp recall over 300 at ~zero QPS cost,
    /// cohere-10M; W is per-shard and leaf-capacity-bound, scale-invariant).
    /// Ignored for graph-mode indexes.
    uint32_t fastscan_W = 0;

    /// Adaptive probe early-exit (B): after scanning each shard, check if the
    /// next shard's centroid distance is significantly farther than the current.
    /// If cent_dist[p+1] / cent_dist[p] > adaptive_probe_gap, stop scanning.
    /// This lets easy queries (tight neighborhood) stop early while hard queries
    /// (spread neighborhood) keep probing.
    ///   0 = auto (use the index's baked value from the manifest)
    ///  <0 = off (scan all n_probe)
    ///  >1 = explicit gap value
    float adaptive_probe_gap = 0.0f;

    /// Rerank top-W candidates by decoding PQ codes to FP32 and computing the
    /// exact distance to the query. The PQ-approximate distances used during
    /// the FastScan heap have non-trivial error (especially on high-LID data);
    /// PQ-decode rerank: decode each W≈300 heap survivor's PQ code back to FP32
    /// and re-sort by exact decoded distance. Benchmarking on sift1m/arxiv/
    /// msmarco/sphere showed <0.001 recall delta at all m values (2-192),
    /// because PQ ADC ranking already matches decoded-FP32 ranking at high m,
    /// and at low m the decoded vectors are too coarse to improve ranking.
    /// Disabled by default — the QPS penalty (10-30%) is not justified.
    /// A true rerank would use stored FP16 vectors, not PQ-decoded approximations.
    bool rerank = false;

    /// Exact-rerank base: row-major f32 vectors (n × dim, indexed by
    /// row_id), owned by the CALLER (e.g. an mmap of the original corpus).
    /// When set and rerank is on, rerank scores are computed against these
    /// ORIGINAL vectors instead of decoded quantized codes: removes the
    /// quantization-ranking error (dbpedia-933K local_scalar np64/W1000:
    /// engine-order 0.9480 = 4-bit ceiling → containment ceiling 1.0000,
    /// +5.2pp) and skips the per-entry code extract/decode work. No IP-bias
    /// correction applies (the true vector has no reconstruction shrinkage).
    /// null = decoded rerank (default; standalone operation, no corpus
    /// dependency). dim must equal the index dim (caller's contract).
    const float* exact_rerank_base = nullptr;

    /// Adaptive shortlist cut (rerank-bandwidth saver): after the scan heap
    /// is reranked and sorted, truncate the returned list at the first index
    /// w >= k whose distance gaps out from the k-th best:
    ///     d[w] - d[k-1] > adaptive_w_gap * max(|d[k-1]|, 1e-9)
    /// Clustered queries (clean score separation) cut just past k; noisy
    /// queries keep the full W. Saves downstream rerank I/O (at billion
    /// scale the caller re-scores each returned id against stored vectors —
    /// shortlist length IS the bandwidth). 0 = off (return top-k as
    /// before). Requires rerank (the signal is the reranked distance).
    float adaptive_w_gap = 0.0f;

    /// Filter predicates (Phase D). Empty = no filtering (today's behavior).
    std::vector<Predicate> predicates;

    /// Scan-feedback probing (see FeedbackProbe). Off by default — the
    /// shipped fixed-fraction contract is unchanged.
    FeedbackProbe feedback;

    /// Optional caller-owned engine trace sink (see engine_trace.hpp).
    /// When set, instrumented paths write diagnostic records (e.g. per-block
    /// feedback-probe traces consumed by `sextant trace`). null = off.
    EngineTrace* trace = nullptr;

    /// Whether to return opaque payload blobs with results (Phase E).
    bool with_payload = false;

    /// Within-query leaf-parallel scan thread count (scalar_lloydmax path).
    /// 0 = serial within-query scan (default); >0 = parallelize the leaf scan
    /// across this many threads for lower single-query latency. Orthogonal to
    /// query-level parallelism: when there are many queries, query-level
    /// parallelism already saturates cores, so leave this 0. This only helps
    /// the latency regime (few queries, high n-probe → many candidate leaves).
    /// The serial path (<=1) is byte-identical to the pre-option behavior.
    uint32_t search_threads = 0;
};

/// Adaptive parameters resolved from dataset/machine properties (Issue 37).
struct ResolvedParams {
    uint16_t R = 64;
    uint16_t L = 100;
    uint16_t L_build = 100;
    float alpha = 1.2f;
    uint16_t pq_m = 32;
    uint8_t pq_bits = 8;          ///< 0 = auto (resolved by reservoir probe in pass1)
    float pq_max_distortion = 0.0f;   ///< 0 = default (1.20); max acceptable PQ distortion
    uint32_t max_occlusion = 750;
    MetricKind metric = MetricKind::L2Sq;
    uint64_t build_ram_budget = 0;
    uint32_t num_threads = 0;
    uint32_t partition_count = 1; ///< Partition count (>1 → partitioned build)
    float closure_factor = 1.033f;  ///< Shard overlap radius ratio
    float closure_epsilon = 0.0f;  ///< Absolute margin for SPANN-style boundary posting. 0 = ratio-based.
    float adaptive_probe_gap = 0.0f;   ///< Geometric-gap early-exit. 0 = OFF
    /// (default). A nonzero value bakes QPS-over-recall probe pruning into
    /// the index. Derived from LID by the estimator for `autobuild` flows
    /// only when the LID measurement is trusted. DANGER: on noise-dominated
    /// embeddings (Cohere), even 1.5 prunes probing to a few leaves and
    /// silently destroys recall (measured 92.2% → 51.7% recall@10).
    float median_lid = 0.0f;           ///< Median LID measured at build time (0 = unmeasured).
    uint16_t n_entry_points = 16;   ///< K-means centroid count for entry-point selection
    uint16_t n_search_entry_points = 4;  ///< Multi-start: top-M entry points per query
    float target_recall = 0.0f;     ///< Recall target the index was built for (0 = unspecified). Drives search early-exit.
    uint32_t early_exit_patience = 0;  ///< Search early-exit: terminate after N stalled pops post-convergence (0 = disabled; set by resolve_params when recall_target is set)
    bool pq_anisotropy = false;  ///< PQ covariance-based anisotropic codebook training. Threaded into PqQuantizer::train.
    bool pq_opq = false;          ///< OPQ PCA rotation. Threaded into PqQuantizer::train.
    bool anisotropic_pq = false;  ///< Use AnisotropicPqQuantizer (ScaNN-style training).

    /// Quantizer type for the IVF-scan codebook ("pq" / "anisotropic-pq" / "prq").
    /// Persisted to the scan manifest; read back to reconstruct the right
    /// quantizer subclass at index-open.
    std::string quantizer_type = "pq";
    /// PRQ nsplits (sub-space count). 0 = auto (dim/32). Persisted to the manifest.
    uint32_t prq_nsplits = 0;
    uint32_t prq_beam_size = 1;
    std::string prq_encode_mode = "greedy";
    uint32_t prq_icm_iters = 4;
    uint32_t prq_ils_iters = 4;
    uint32_t prq_ils_perturb = 4;
    uint32_t prq_lsq_train_iters = 0;

    /// Merged-graph build path (vs the default IVF-list-scan + 4-bit PQ
    /// FastScan). Persisted so the search dispatcher can route correctly on
    /// reopen. See BuildConfig::merged_graph.
    bool merged_graph = false;

    /// 4-bit PQ subquantizer count for the IVF-scan path. 0 = auto (dim/4).
    /// Persisted; the searcher reads this to size its 4-bit LUT.
    uint16_t pq4_m = 0;

    /// Bits per segment for the IVF-scan codebook (4 or 8). Persisted; the
    /// searcher reads this to pick the FastScan kernel. See
    /// BuildConfig::scan_pq_bits.
    uint8_t scan_pq_bits = 4;

    /// Partition size-balance strength (SPANN λ). 0 = off. See
    /// BuildConfig::partition_balance_factor.
    float partition_balance_factor = 0.0f;
};

// ---------------------------------------------------------------------------
// ResolvedParams serialization (.meta sidecar, graph/flat path)
//
// HISTORY: this used to be a raw memcpy of the whole struct. ResolvedParams
// contains std::string members, so the bytes written to disk included live
// object pointers; reading them back over a constructed object produced
// dangling strings → free(): invalid pointer crashes (macOS, c4a VM, and
// Zen5/ZFS alike; SSO-short strings masked it much of the time). Replaced by
// explicit field-by-field serialization. Old .meta files (no SPRM magic)
// are rejected — regenerate any legacy graph indexes.
//
// MAINTENANCE RULE: adding a field here is a STOP-AND-RETHINK. This hand-
// rolled binary codec exists only because the block is embedded at the tail
// of the binary .meta payload (see Index::read / Builder::write_meta_file).
// The moment a second field gets added, migrate the params block to TOML
// (cpptoml is already a dependency; the .manifest files use it) — either as
// a length-prefixed text section in the payload or by folding params into
// the .manifest itself. TOML defaults missing keys, which removes the
// kParamsVersion bump dance this format requires per field.
// ---------------------------------------------------------------------------
inline constexpr uint32_t kParamsMagic = 0x5350524D;  // 'SPRM'
inline constexpr uint16_t kParamsVersion = 1;

inline void serialize_params(const ResolvedParams& p,
                             std::vector<uint8_t>& out) {
    auto put_u8 = [&](uint8_t v) { out.push_back(v); };
    auto put_u16 = [&](uint16_t v) {
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&v),
                   reinterpret_cast<uint8_t*>(&v) + 2);
    };
    auto put_u32 = [&](uint32_t v) {
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&v),
                   reinterpret_cast<uint8_t*>(&v) + 4);
    };
    auto put_u64 = [&](uint64_t v) {
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&v),
                   reinterpret_cast<uint8_t*>(&v) + 8);
    };
    auto put_f32 = [&](float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        put_u32(bits);
    };
    auto put_str = [&](const std::string& s) {
        put_u32(static_cast<uint32_t>(s.size()));
        out.insert(out.end(), s.begin(), s.end());
    };

    put_u32(kParamsMagic);
    put_u16(kParamsVersion);
    put_u16(p.R);                       put_u16(p.L);
    put_u16(p.L_build);                 put_f32(p.alpha);
    put_u16(p.pq_m);                    put_u8(p.pq_bits);
    put_f32(p.pq_max_distortion);       put_u32(p.max_occlusion);
    put_u8(static_cast<uint8_t>(p.metric));
    put_u64(p.build_ram_budget);        put_u32(p.num_threads);
    put_u32(p.partition_count);         put_f32(p.closure_factor);
    put_f32(p.closure_epsilon);         put_f32(p.adaptive_probe_gap);
    put_f32(p.median_lid);              put_u16(p.n_entry_points);
    put_u16(p.n_search_entry_points);   put_f32(p.target_recall);
    put_u32(p.early_exit_patience);     put_u8(p.pq_anisotropy ? 1 : 0);
    put_u8(p.pq_opq ? 1 : 0);           put_u8(p.anisotropic_pq ? 1 : 0);
    put_str(p.quantizer_type);
    put_u32(p.prq_nsplits);             put_u32(p.prq_beam_size);
    put_str(p.prq_encode_mode);
    put_u32(p.prq_icm_iters);           put_u32(p.prq_ils_iters);
    put_u32(p.prq_ils_perturb);         put_u32(p.prq_lsq_train_iters);
    put_u8(p.merged_graph ? 1 : 0);     put_u16(p.pq4_m);
    put_u8(p.scan_pq_bits);             put_f32(p.partition_balance_factor);
}

/// Returns false on truncated/garbled input (caller throws). `out` keeps its
/// defaults for any field the format does not cover yet.
inline bool deserialize_params(const uint8_t* data, size_t len,
                               ResolvedParams& out) {
    const uint8_t* p = data;
    const uint8_t* end = data + len;
    auto take = [&](void* dst, size_t n) -> bool {
        if (static_cast<size_t>(end - p) < n) return false;
        std::memcpy(dst, p, n);
        p += n;
        return true;
    };
    auto take_f32 = [&](float& v) -> bool {
        uint32_t bits;
        if (!take(&bits, 4)) return false;
        std::memcpy(&v, &bits, 4);
        return true;
    };
    auto take_str = [&](std::string& v) -> bool {
        uint32_t n;
        if (!take(&n, 4)) return false;
        if (static_cast<size_t>(end - p) < n) return false;
        v.assign(reinterpret_cast<const char*>(p), n);
        p += n;
        return true;
    };

    uint32_t magic;
    uint16_t version;
    if (!take(&magic, 4) || !take(&version, 2) || magic != kParamsMagic ||
        version != kParamsVersion) {
        return false;
    }
    uint8_t u8;
    bool ok = take(&out.R, 2) && take(&out.L, 2) && take(&out.L_build, 2) &&
              take_f32(out.alpha) && take(&out.pq_m, 2) && take(&u8, 1);
    if (!ok) return false;
    out.pq_bits = u8;
    if (!take_f32(out.pq_max_distortion) || !take(&out.max_occlusion, 4) ||
        !take(&u8, 1)) return false;
    out.metric = static_cast<MetricKind>(u8);
    if (!take(&out.build_ram_budget, 8) || !take(&out.num_threads, 4) ||
        !take(&out.partition_count, 4) || !take_f32(out.closure_factor) ||
        !take_f32(out.closure_epsilon) || !take_f32(out.adaptive_probe_gap) ||
        !take_f32(out.median_lid) || !take(&out.n_entry_points, 2) ||
        !take(&out.n_search_entry_points, 2) ||
        !take_f32(out.target_recall) ||
        !take(&out.early_exit_patience, 4)) return false;
    if (!take(&u8, 1)) return false;
    out.pq_anisotropy = u8 != 0;
    if (!take(&u8, 1)) return false;
    out.pq_opq = u8 != 0;
    if (!take(&u8, 1)) return false;
    out.anisotropic_pq = u8 != 0;
    if (!take_str(out.quantizer_type) || !take(&out.prq_nsplits, 4) ||
        !take(&out.prq_beam_size, 4) || !take_str(out.prq_encode_mode) ||
        !take(&out.prq_icm_iters, 4) || !take(&out.prq_ils_iters, 4) ||
        !take(&out.prq_ils_perturb, 4) || !take(&out.prq_lsq_train_iters, 4))
        return false;
    if (!take(&u8, 1)) return false;
    out.merged_graph = u8 != 0;
    if (!take(&out.pq4_m, 2) || !take(&u8, 1)) return false;
    out.scan_pq_bits = u8;
    return take_f32(out.partition_balance_factor);
}

/// Estimation diagnostics produced by `Engine::estimate_config` (and the
/// Estimator in the post-refactor world). Display-only — they do NOT influence
/// the build (the inputs that drove the resolved params are already in
/// `ResolvedParams`) and are NOT serialized to `.meta`. Kept in a separate
/// struct so the build-input `ResolvedParams` stays pure.
struct EstimationDiagnostics {
    double median_lid = 0.0;       ///< MLE LID from probe truth (0 = unmeasured)
    double avg_degree = 0.0;       ///< R̄ from mini-build (0 = unmeasured)
    double clustering_coeff = 0.0; ///< clustering coefficient (0 = unmeasured)
    double dead_end_frac = 0.0;    ///< fraction of nodes with degree ≤ 1

    /// L2 norm statistics over the sampled vectors. Detects L2-normalization
    /// (unit norms with tight spread). Normalization requires BOTH
    /// `norm_mean ≈ 1` (within ±5%) AND `norm_cv < 0.05`. SIFT-1M has
    /// tight norms (cv≈0.001) but mean≈508 — NOT normalized. This is the
    /// metric-recommendation signal: unit-normalized data → IP and L2sq
    /// rank-equivalent (IP cheaper per eval); non-normalized → L2sq required
    /// (IP misranks — SIFT-1M: 0.99 → 0.66 recall collapse).
    double norm_mean = 0.0;
    double norm_stddev = 0.0;
    double norm_min = 0.0;
    double norm_max = 0.0;
    double norm_cv = 0.0;          ///< 0 = unmeasured; <0.05 ≈ tight spread
};

/// Result of `Engine::estimate_config`: resolved build inputs + the measured
/// signals that informed them (for display / `--explain`).
struct EstimateResult {
    ResolvedParams params;
    EstimationDiagnostics diag;
};

/// Structural properties of a built graph, measured for parameter estimation.
/// (Historical: consumed by the flat/Vamana stack — removed at vamana-eol;
/// retained as inert metadata.)
struct GraphStats {
    double avg_degree = 0.0;       ///< mean post-prune degree (R̄)
    double clustering_coeff = 0.0; ///< sampled triangle fraction
    double dead_end_frac = 0.0;    ///< fraction of nodes with degree ≤ 1
    double median_lid = 0.0;       ///< MLE LID from k-NN distances
};

}  // namespace sextant
