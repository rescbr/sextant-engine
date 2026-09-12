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
// build_streaming_pca context + helpers
//
// build_streaming_pca was a 1401-line monolith. It is now factored into a
// plain TreeBuildContext data struct (holding all cross-phase state) and five
// free helper functions that operate on it. The struct lives in the anonymous
// namespace so it does not pollute the class header with build-only internals.
// ===========================================================================
namespace {

/// Per-worker scratch for the streaming build passes, carried in the
/// ctpl pool's TLS slots. Reused across chunks and passes — replaces the
/// per-chunk std::vector allocations that dominated build-time page-fault
/// overhead (measured: ~50% of cycles in kernel mm paths at 10M scale
/// when each 2048-vector chunk spawned fresh std::async threads).
struct StreamTLS {
    std::vector<float> proj;    // PCA projection scratch (pca_dims)
    std::vector<float> dists;   // per-centroid distances (k_root, emission)
};

/// Parallel nearest-centroid assignment over a row block of a sample
/// matrix (n_rows × pca_dims row-major) against per-cluster centroid
/// vectors. Each row task runs the EXACT scalar distance loop the old
/// serial code ran (same accumulation order, same tie handling), so the
/// winners are bit-identical; callers aggregate the per-row outputs
/// serially in row order to keep sums deterministic.
/// Outputs: best[i] = argmin cluster; gap[i] = (2nd-nearest − nearest)
/// squared distance (nullptr to skip — one extra compare per centroid).
void parallel_sample_assign(
    const float* rows, uint32_t n_rows, uint32_t pca_dims,
    const std::vector<std::vector<float>>& cents,
    uint32_t* best, float* gap,
    ctpl::thread_pool_tls<StreamTLS>& pool, uint32_t hw)
{
    const uint32_t k = static_cast<uint32_t>(cents.size());
    if (k == 0 || n_rows == 0) return;
    const uint32_t n_shards = std::min(hw, n_rows);
    const uint32_t per = (n_rows + n_shards - 1) / n_shards;
    std::vector<std::future<void>> futs;
    for (uint32_t t = 0; t < n_shards; ++t) {
        const uint32_t start = t * per;
        const uint32_t end = std::min(start + per, n_rows);
        if (start >= end) break;
        futs.push_back(pool.push(
            [&](size_t, StreamTLS&, uint32_t s, uint32_t e) {
                for (uint32_t i = s; i < e; ++i) {
                    const float* vi = rows + size_t(i) * pca_dims;
                    float d1 = std::numeric_limits<float>::max();
                    float d2 = d1;
                    uint32_t best_c = 0;
                    for (uint32_t c = 0; c < k; ++c) {
                        const float* cc = cents[c].data();
                        float d = 0.0f;
                        for (uint32_t dd = 0; dd < pca_dims; ++dd) {
                            const float diff = vi[dd] - cc[dd];
                            d += diff * diff;
                        }
                        if (d < d1) { d2 = d1; d1 = d; best_c = c; }
                        else if (d < d2) d2 = d;
                    }
                    best[i] = best_c;
                    if (gap) gap[i] = d2 - d1;
                }
            }, start, end));
    }
    for (auto& fut : futs) fut.get();
}

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

    // --- In-build routing plane (cfg.plane_attach): trained from the
    //     build's reservoir sample, encoded during the emission pass at
    //     the same points rows are appended to leaf buffers. ---
    PlaneWriter plane_writer;
    bool plane = false;
    std::vector<uint32_t> plane_leaf_counts;  // per flushed leaf, in order

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
    std::vector<float> plane_sample;  // v1-attach-equivalent stride sample
    {
        ctx.source.reset();
        Chunk chunk;
        std::vector<float> reservoir(size_t(train_n) * dim);
        // Stride sample for the routing plane, collected in the SAME pass:
        // rows 0, stride, 2*stride, ... — exactly the rows PlaneWriter::train
        // would pick from the full corpus (attach_plane). Makes the in-build
        // basis + codebooks byte-identical to the v1 post-hoc attach.
        const uint64_t plane_stride = std::max<uint64_t>(1, n / train_n);
        if (cfg.plane_attach)
            plane_sample.resize(size_t(train_n) * dim);
        uint64_t seen = 0;
        std::mt19937_64 rng(42);
        while (ctx.source.next(chunk)) {
            for (uint32_t i = 0; i < chunk.count; ++i) {
                const float* v = chunk.vectors + size_t(i) * dim;
                if (cfg.plane_attach && seen / plane_stride < train_n
                    && seen % plane_stride == 0)
                    std::memcpy(
                        &plane_sample[size_t(seen / plane_stride) * dim], v,
                        size_t(dim) * sizeof(float));
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
    // The pool persists through k-means and closure-epsilon (the sample
    // phases share it; one spawn set per build).
    std::vector<float> sample_pca(static_cast<size_t>(train_n) * pca_dims);
    const uint32_t hw = cfg.num_threads > 0
        ? cfg.num_threads
        : std::max(1u, std::thread::hardware_concurrency());
    ctpl::thread_pool_tls<StreamTLS> sample_pool(hw);
    {
        const uint32_t n_threads = std::min(hw, train_n);
        std::vector<std::future<void>> futs;
        const uint32_t per = (train_n + n_threads - 1) / n_threads;
        for (uint32_t t = 0; t < n_threads; ++t) {
            const uint32_t start = t * per;
            const uint32_t end = std::min(start + per, train_n);
            if (start >= end) break;
            futs.push_back(sample_pool.push(
                [&](size_t, StreamTLS&, uint32_t s, uint32_t e) {
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
    std::vector<uint32_t> best(train_n);
    for (uint32_t iter = 0; iter < 10; ++iter) {
        std::vector<std::vector<uint32_t>> assigns(k_root);
        uint32_t n_changed = 0;
        parallel_sample_assign(sample_pca.data(), train_n, pca_dims,
                                root_centroids_pca, best.data(), nullptr,
                                sample_pool, hw);
        for (uint32_t i = 0; i < train_n; ++i) {
            assigns[best[i]].push_back(i);
            if (iter > 0 && best[i] != prev_assign[i]) ++n_changed;
            prev_assign[i] = best[i];
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
        std::vector<uint32_t> best_gap_rows(s);
        std::vector<float> gap(s);
        parallel_sample_assign(sample_pca.data(), s, pca_dims,
                                root_centroids_pca, best_gap_rows.data(),
                                gap.data(), sample_pool, hw);
        double sum_gap = 0.0;
        for (uint32_t i = 0; i < s; ++i) sum_gap += gap[i];
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

    // In-build routing plane (phase A): train the basis + codebooks from
    // the SAME reservoir sample (uniform over the whole source — the
    // spread-sample property plane.hpp requires; prefix bases lose
    // containment). Zero extra source passes.
    if (cfg.plane_attach) {
        const auto tp = std::chrono::steady_clock::now();
        PlaneWriter::Config pcfg;
        pcfg.encoding = cfg.plane_enc;
        pcfg.rank = cfg.plane_rank;
        // train_rows = train_n => stride 1 over the pre-strided sample: the
        // SAME rows attach_plane would sample from the full corpus.
        pcfg.train_rows = train_n;
        ctx.plane_writer.train(plane_sample.data(), train_n, dim, pcfg);
        ctx.plane_writer.prepare(0);  // leaves grow on demand
        ctx.plane = true;
        spdlog::info("[sextant] plane: basis+codebooks trained in {:.2}s "
                     "(in-build, {} sample rows)",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - tp).count(),
                     pcfg.train_rows);
    }
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

    // Worker pool + shard accumulators for the whole refinement phase
    // (all passes, the convergence check, and d_eff below). One spawn set;
    // t_sums are zeroed at the top of each pass. Accumulators are indexed
    // by SHARD, so the reduction order — and the resulting centroids — is
    // independent of which worker runs which shard.
    const uint32_t hw = cfg.num_threads > 0
        ? cfg.num_threads
        : std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::vector<double>> t_sums(
        hw, std::vector<double>(k_root * pca_dims, 0.0));
    std::vector<std::vector<uint64_t>> t_counts(hw, std::vector<uint64_t>(k_root, 0));
    ctpl::thread_pool_tls<StreamTLS> pool(hw);

    for (uint32_t pass = 0; pass < max_lloyd_passes; ++pass) {
        const auto pass_t0 = std::chrono::steady_clock::now();
        // Reset accumulators.
        for (uint32_t c = 0; c < k_root; ++c) {
            std::fill(cluster_sums[c].begin(), cluster_sums[c].end(), 0.0);
            cluster_counts[c] = 0;
        }
        for (uint32_t t = 0; t < hw; ++t) {
            std::fill(t_sums[t].begin(), t_sums[t].end(), 0.0);
            std::fill(t_counts[t].begin(), t_counts[t].end(), 0);
        }

        // Stream all vectors: project + assign + accumulate (parallel).
        ctx.source.reset();
        uint64_t vectors_done = 0;

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

            // Parallel assign + accumulate. Tasks are indexed by POOL thread
            // id for the accumulators (t_sums/t_counts), so any task→thread
            // mapping is correct; shard boundaries stay as the work split.
            std::vector<std::future<void>> futs;
            const uint32_t n_shards = std::min(hw, take);
            const uint32_t per = (take + n_shards - 1) / n_shards;
            for (uint32_t t = 0; t < n_shards; ++t) {
                const uint32_t start = t * per;
                const uint32_t end = std::min(start + per, take);
                if (start >= end) break;
                futs.push_back(pool.push(
                    [&](size_t, StreamTLS& tls, uint32_t sh,
                        uint32_t s, uint32_t e) {
                        double* sums = t_sums[sh].data();
                        uint64_t* counts = t_counts[sh].data();
                        if (tls.proj.size() < pca_dims)
                            tls.proj.resize(pca_dims);
                        float* proj = tls.proj.data();
                        for (uint32_t i = s; i < e; ++i) {
                            const float* xi = &vec_buf[i * dim];
                            for (uint32_t k = 0; k < pca_dims; ++k)
                                proj[k] = simd::dot_f32(
                                    &rotation[k * dim], xi, dim) - mean_proj[k];
                            float best_d = std::numeric_limits<float>::max();
                            uint32_t best_c = 0;
                            for (uint32_t c = 0; c < k_root; ++c) {
                                const float dot = simd::dot_f32(
                                    proj, root_centroids_pca[c].data(), pca_dims);
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
        // Assign the sample once (parallel; identical winners to the old
        // serial loop) and reuse the result for both the change count and
        // the prev_assignment update (the old code computed it twice).
        if (pass >= 1) {
            const uint32_t sample_sz = std::min<uint32_t>(4096, train_n);
            std::vector<uint32_t> best(sample_sz);
            parallel_sample_assign(ctx.sample_pca.data(), sample_sz,
                                   pca_dims, root_centroids_pca,
                                   best.data(), nullptr, pool, hw);
            uint32_t n_changed = 0;
            for (uint32_t i = 0; i < sample_sz; ++i) {
                if (pass == 1) prev_assignment.push_back(best[i]);
                else if (best[i] != prev_assignment[i % prev_assignment.size()])
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
                for (uint32_t i = 0; i < sample_sz; ++i)
                    prev_assignment[i] = best[i];
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
        std::vector<uint32_t> best(s);
        std::vector<float> gap(s);
        parallel_sample_assign(ctx.sample_pca.data(), s, pca_dims,
                               root_centroids_pca, best.data(), gap.data(),
                               pool, hw);
        // Aggregate serially in row order — identical accumulation order
        // to the old serial loop.
        for (uint32_t i = 0; i < s; ++i) {
            sum_gap[best[i]] += gap[i];
            ++gap_count[best[i]];
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
    // Payload comes from either the fdat sidecar (cfg.payload_data/
    // payload_offsets, indexed by build row) or per-chunk source blobs
    // (chunk.payload_data — ParquetSource --payload-col). The schema flag
    // drives the extents; the per-chunk dual-path is resolved below.
    const bool has_payload = cfg.filter_schema.has_payload;
    const bool cfg_payload = cfg.payload_data && cfg.payload_offsets;
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
        // Payload info for the rows in this buffer (Phase E). Bytes are
        // COPIED into payload_store: source chunk payload buffers are
        // reused across chunks, so pointers would dangle at flush time.
        std::vector<uint32_t> payload_lens;
        std::vector<uint32_t> payload_offs;  // start offset into payload_store
        std::vector<uint8_t> payload_store;
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
                std::memcpy(pdata + acc,
                            buf.payload_store.data() + buf.payload_offs[i],
                            plen);
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
        if (ctx.plane) {
            const auto leaf_id = static_cast<uint32_t>(leaf_metas.size() - 1);
            ctx.plane_writer.transfer_leaf(c, leaf_id);
            ctx.plane_leaf_counts.push_back(count);
        }

        // Reset the buffer for the next leaf in this cluster.
        buf.codes.clear();
        buf.row_ids.clear();
        buf.ip_biases.clear();
        buf.fp16_vecs.clear();
        std::fill(buf.centroid_sum.begin(), buf.centroid_sum.end(), 0.0);
        buf.centroid_count = 0;
        buf.payload_lens.clear();
        buf.payload_offs.clear();
        buf.payload_store.clear();
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
            std::vector<uint32_t> payload_offs;
            std::vector<uint8_t> payload_store;
            void clear() {
                codes.clear(); ip_biases.clear(); fp16_vecs.clear();
                row_ids.clear(); local_indices.clear(); payload_lens.clear();
                payload_offs.clear(); payload_store.clear();
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
        // Persistent pool for the whole emission pass (see Lloyd note: fresh
        // std::async threads per chunk spent more cycles in the kernel mm
        // path than in the routing arithmetic itself).
        ctpl::thread_pool_tls<StreamTLS> pool(hw);
        // Hoisted per-chunk staging: flat code buffer (fixed code_size stride)
        // and a per-vector target list whose capacity persists across chunks.
        // The old per-chunk vector<vector> reallocation was a top page-fault
        // source; only the first chunk allocates.
        std::vector<uint8_t> codes_flat;
        std::vector<std::vector<uint32_t>> chunk_targets;
        // In-build plane projection scratch (rank floats per chunk row);
        // capacity hoisted like codes_flat. Encoded rows go to the plane
        // writer's per-leaf blocks at the serial merge (slot = leaf-local
        // index, same order the codes get flushed in).
        std::vector<float> plane_proj;
        const uint32_t plane_rank = ctx.plane ? ctx.plane_writer.meta().rank : 0;
        if (ctx.plane)
            plane_proj.resize(static_cast<size_t>(1u << 20) * plane_rank);
        while (true) {
            Chunk chunk;
            if (!ctx.source.next(chunk)) break;
            const uint32_t take = chunk.count;
            if (ctx.plane && plane_proj.size() <
                    static_cast<size_t>(take) * plane_rank)
                plane_proj.resize(static_cast<size_t>(take) *
                                  plane_rank);
            const float* vec_buf = chunk.vectors;  // NO copy
            const RowId* chunk_row_ids = chunk.row_ids;  // source-assigned ids
            // Capture whether this chunk carries per-chunk filter/payload data
            // (e.g. from a ParquetSource). The lambdas below read these.
            chunk_has_filter = (chunk.filter_columns != nullptr);
            chunk_fcols = chunk.filter_columns;
            const bool chunk_has_payload = (chunk.payload_data != nullptr);
            if (has_payload && !chunk_has_payload && !cfg_payload) {
                throw Error(ErrorCode::InvalidParam,
                            "run_emission_pass: schema declares payloads but "
                            "neither sidecar buffers nor chunk payloads are "
                            "present (missing --payload-col / fdat?)");
            }
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
            if (!keep_raw_vecs)
                codes_flat.resize(static_cast<size_t>(take) * code_size);
            if (chunk_targets.size() < take)
                chunk_targets.resize(take);
            else
                for (uint32_t i = 0; i < take; ++i) chunk_targets[i].clear();
            std::vector<float16_t> chunk_biases(
                ctx.has_ip_bias ? take : 0, float16_t(1.0f));

            std::vector<std::future<void>> futs;
            const uint32_t n_threads = std::min(hw, take);
            const uint32_t per = (take + n_threads - 1) / n_threads;
            for (uint32_t t = 0; t < n_threads; ++t) {
                const uint32_t start = t * per;
                const uint32_t end = std::min(start + per, take);
                if (start >= end) break;
                futs.push_back(pool.push(
                    [&](size_t, StreamTLS& tls, uint32_t s, uint32_t e) {
                        if (tls.proj.size() < pca_dims)
                            tls.proj.resize(pca_dims);
                        if (tls.dists.size() < k_root)
                            tls.dists.resize(k_root);
                        float* proj = tls.proj.data();
                        float* dists = tls.dists.data();
                        for (uint32_t i = s; i < e; ++i) {
                            const float* xi = &vec_buf[i * dim];
                            // Project to PCA space.
                            for (uint32_t k = 0; k < pca_dims; ++k)
                                proj[k] = simd::dot_f32(
                                    &rotation[k * dim], xi, dim) - mean_proj[k];
                            // Route: single pass computes every centroid
                            // distance once (the old form ran the k_root dot
                            // loop twice — once for the min, once for closure).
                            float min_d = std::numeric_limits<float>::max();
                            for (uint32_t c = 0; c < k_root; ++c) {
                                const float dot = simd::dot_f32(
                                    proj, root_centroids_pca[c].data(), pca_dims);
                                const float d = cent_norms[c] - 2.0f * dot;
                                dists[c] = d;
                                if (d < min_d) min_d = d;
                            }
                            // Encode scan code (skipped for local families:
                            // raw vectors are kept and encoded per-leaf at
                            // flush).
                            if (!keep_raw_vecs) {
                                float bias = 0.f;
                                ctx.coder->encode(xi,
                                    &codes_flat[static_cast<size_t>(i) * code_size],
                                    ctx.has_ip_bias ? &bias : nullptr);
                                if (ctx.has_ip_bias)
                                    chunk_biases[i] = float16_t(bias);
                            }
                            // In-build plane: project on the routing pass
                            // (read-only basis — thread-safe).
                            if (ctx.plane)
                                ctx.plane_writer.project(
                                    xi, &plane_proj[static_cast<size_t>(i) *
                                                      plane_rank]);
                            // Find closure targets using per-cluster epsilon.
                            // Phase I: each cluster c has its own closure eps
                            // derived from its local d_eff. A vector is
                            // replicated into c if dist(v,c) is within
                            // cluster_closure_eps[c] of the nearest distance.
                            for (uint32_t c = 0; c < k_root; ++c) {
                                const float d = dists[c];
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
                    // Shard-indexed buffers: the serial merge below replays
                    // shards in order 0..n_threads-1 to reproduce global
                    // vector order, so this task must write thread_buffers[t]
                    // regardless of which pool worker runs it.
                    afuts.push_back(pool.push(
                        [&](size_t, StreamTLS&, uint32_t s, uint32_t e, uint32_t tid) {
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
                                        const uint8_t* cd =
                                            &codes_flat[static_cast<size_t>(i) * code_size];
                                        tb.codes.insert(tb.codes.end(),
                                                        cd, cd + code_size);
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
                                        // Copy now — the source payload
                                        // buffer is reused next chunk.
                                        tb.payload_offs.push_back(
                                            static_cast<uint32_t>(
                                                tb.payload_store.size()));
                                        tb.payload_store.insert(
                                            tb.payload_store.end(), pdata,
                                            pdata + plen);
                                        tb.payload_lens.push_back(plen);
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
                        // In-build plane row: encoded under the stable
                        // CLUSTER id (slot = this row's leaf-local index);
                        // transfer_leaf moves the blocks to the global
                        // leaf id at flush. Same replication as v1 attach.
                        if (ctx.plane) {
                            ctx.plane_writer.encode_staged(
                                c,
                                static_cast<uint32_t>(buf.row_ids.size()),
                                &plane_proj[static_cast<size_t>(
                                    tb.local_indices[j]) * plane_rank]);
                        }
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
                            const uint32_t plen = tb.payload_lens[j];
                            buf.payload_offs.push_back(
                                static_cast<uint32_t>(
                                    buf.payload_store.size()));
                            buf.payload_store.insert(
                                buf.payload_store.end(),
                                tb.payload_store.data() + tb.payload_offs[j],
                                tb.payload_store.data() + tb.payload_offs[j]
                                    + plen);
                            buf.payload_lens.push_back(plen);
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

    // In-build plane tail: the writer holds trained basis + encoded
    // per-leaf blocks. Reopen the committed tree and append the plane
    // extent (shared with the post-hoc attach path).
    if (ctx.plane) {
        auto idx = IVFTreeIndex::open(output_path);
        idx->attach_plane_from(ctx.plane_writer, ctx.plane_leaf_counts);
    }

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

}  // namespace sextant::tree
