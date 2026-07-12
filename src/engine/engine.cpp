// Engine facade — build/open/search/insert/flush orchestration.
//
// Owns the PqQuantizer + VamanaCore + flat-in-RAM build buffers. The build
// pipeline is:
//   1. resolve_params (auto-defaults from N, dim)
//   2. allocate flat codes + nodes buffers
//   3. Pass 1: reservoir sample (256K) + PQ train
//   4. Pass 2: encode all vectors → codes_buffer_
//   5. Parallel SDC construct via CTPL (disjoint node ranges)
//   6. Finalize: compute_entry_points + finalize_inline_codes
//   7. Flush sidecars (.graph/.codes/.meta/.manifest)
//
// The search/load path is in search.cpp.

#include "build.hpp"
#include "fbin_source.hpp"
#include "partition.hpp"
#include "sidecar_io.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/direct_io.hpp"
#include "storage/sidecar_header.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sextant {

using engine_detail::fill_header;
using engine_detail::read_exact;
using engine_detail::write_padded;

// ---------------------------------------------------------------------------
// Node layout offsets (mirror vamana_core.cpp — kept private to the engine).
// ---------------------------------------------------------------------------
namespace {
inline constexpr uint32_t kNeighborArrayOffset = 16;

/// Reservoir sample target (Issue 19).
inline constexpr uint64_t kSampleTarget = 256'000;

/// Generate a shared index UUID (low + high) for a build, from a random device.
std::pair<uint64_t, uint64_t> make_uuid() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> dist;
    return {dist(rd), dist(rd)};
}

}  // namespace

// ===========================================================================
// Construction / destruction
// ===========================================================================

Engine::Engine() {
    // Default logger is spdlog's global; callers may init_logging() first.
}

Engine::~Engine() {
    if (codes_buffer_) aligned_free(codes_buffer_);
    if (nodes_buffer_) aligned_free(nodes_buffer_);
}

// ===========================================================================
// build()
// ===========================================================================

BuildResult Engine::build(VectorSource& source, const std::string& index_path,
                          const BuildConfig& config) {
    const auto t0 = std::chrono::steady_clock::now();

    count_ = source.count();
    dim_ = source.dim();
    if (count_ == 0) {
        throw Error(ErrorCode::InvalidParam, "Engine::build: source is empty");
    }
    if (dim_ == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::build: source has dim=0");
    }
    index_path_ = index_path;

    spdlog::info("[sextant] build: n={} dim={} → '{}'", count_, dim_,
                 index_path);

    // 1. Resolve parameters.
    auto params = resolve_params(count_, dim_, config);

    // ADC mode is not implemented in Phase 1 (SDC only).
    if (params.alpha == 1.5f) {
        spdlog::warn("[sextant] ADC build mode requested (alpha=1.5) but "
                     "Phase 1 is SDC-only; continuing with SDC construct.");
        params.alpha = 1.2f;
    }

    // Partitioned build (Step 11): when the monolithic graph would exceed the
    // build RAM budget (K>1), partition into K shards, build each, then merge.
    if (params.K > 1) {
        auto result = build_partitioned(source, index_path, params);
        const auto t1 = std::chrono::steady_clock::now();
        result.build_time_sec =
            std::chrono::duration<double>(t1 - t0).count() +
            result.build_time_sec;
        spdlog::info("[sextant] partitioned build complete (K={}) in {:.2f}s",
                     params.K, result.build_time_sec);
        return result;
    }

    // 2. Construct the quantizer + core with build-time params.
    //    Build ALWAYS uses inline_pq=0 flat nodes (Issue 26). The final
    //    inline_pq layout is materialized at flush time.
    quantizer_ = std::make_unique<PqQuantizer>(
        params.metric, dim_, params.pq_m, params.pq_bits);
    code_size_ = quantizer_->code_size();

    VamanaParams vparams;
    vparams.dim = dim_;
    vparams.R = params.R;
    vparams.L = params.L;
    vparams.L_build = params.L_build;
    vparams.alpha = params.alpha;
    vparams.inline_pq_count = 0;  // Issue 26: always 0 during build.
    vparams.n_entry_points = 16;
    vparams.max_occlusion = params.max_occlusion;
    core_ = std::make_unique<VamanaCore>(vparams, *quantizer_);

    // node_size for the flat BUILD buffer (inline_pq=0).
    node_size_ = VamanaCore::static_node_size(params.R, 0,
                                              static_cast<uint8_t>(code_size_));

    // Allocate flat buffers (aligned for potential direct-IO reuse).
    const size_t codes_bytes = static_cast<size_t>(count_) * code_size_;
    const size_t nodes_bytes = static_cast<size_t>(count_) * node_size_;
    codes_buffer_ = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, codes_bytes));
    nodes_buffer_ = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, nodes_bytes));
    std::memset(codes_buffer_, 0, codes_bytes);
    std::memset(nodes_buffer_, 0, nodes_bytes);
    if (!codes_buffer_ || !nodes_buffer_) {
        throw Error(ErrorCode::OutOfMemory, "Engine::build: buffer alloc failed");
    }
    spdlog::info("[sextant] allocating flat build buffers: codes={:.1f}MB "
                 "nodes={:.1f}MB total={:.1f}MB (per_vec={}B)",
                 codes_bytes / 1e6, nodes_bytes / 1e6,
                 (codes_bytes + nodes_bytes) / 1e6,
                 code_size_ + node_size_);

    core_->set_build_codes(codes_buffer_, static_cast<uint32_t>(count_));
    core_->set_build_nodes(nodes_buffer_);
    core_->prepare_for_build(static_cast<uint32_t>(count_));

    // 3-6. The pipeline passes.
    pass1_sample_and_train(source, params);
    pass2_encode(source, params);
    parallel_construct(params);
    finalize_and_flush(params);

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    spdlog::info("[sextant] build complete in {:.2f}s", secs);

    opened_ = true;

    BuildResult result;
    result.index_path = index_path;
    result.n_vectors = count_;
    result.dim = dim_;
    result.build_time_sec = secs;
    result.R = params.R;
    result.L_build = params.L_build;
    result.pq_m = params.pq_m;
    return result;
}

// ===========================================================================
// Pass 1: reservoir sample + PQ train
// ===========================================================================

void Engine::pass1_sample_and_train(VectorSource& source,
                                    const ResolvedParams& /*params*/) {
    spdlog::info("[sextant] pass 1: reservoir sample (target {} vectors)",
                 kSampleTarget);

    // Reservoir sampling (Algorithm R). We sample floats directly.
    const uint64_t sample_cap =
        std::min<uint64_t>(kSampleTarget, count_);
    if (sample_cap == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::pass1: cannot sample from empty source");
    }
    std::vector<float> reservoir(static_cast<size_t>(sample_cap) * dim_);

    source.reset();

    // Algorithm R: keep the first `sample_cap`, then replace index j (j<k)
    // with probability k/i for the i-th seen item.
    std::mt19937_64 rng(0xC0DE1234ULL);
    Chunk chunk{};
    uint64_t seen = 0;
    uint64_t filled = 0;
    while (source.next(chunk)) {
        for (uint32_t r = 0; r < chunk.count; r++) {
            const float* vec = chunk.vectors + static_cast<size_t>(r) * dim_;
            if (filled < sample_cap) {
                std::memcpy(reservoir.data() + filled * dim_, vec,
                            dim_ * sizeof(float));
                filled++;
            } else {
                std::uniform_int_distribution<uint64_t> dist(0, seen);
                const uint64_t j = dist(rng);
                if (j < sample_cap) {
                    std::memcpy(reservoir.data() + j * dim_, vec,
                                dim_ * sizeof(float));
                }
            }
            seen++;
        }
    }

    const uint64_t actual_sample = filled;
    spdlog::info("[sextant] pass 1: sampled {} / {} vectors", actual_sample,
                 seen);

    // Train the quantizer on the reservoir.
    spdlog::info("[sextant] pass 1: training PQ on {} samples", actual_sample);
    quantizer_->train(reservoir.data(), actual_sample);
    spdlog::info("[sextant] pass 1: PQ trained (code_size={})", code_size_);
}

// ===========================================================================
// Pass 2: encode all vectors → codes_buffer_
// ===========================================================================

void Engine::pass2_encode(VectorSource& source,
                          const ResolvedParams& /*params*/) {
    spdlog::info("[sextant] pass 2: encoding {} vectors", count_);

    source.reset();
    Chunk chunk{};
    uint64_t encoded = 0;
    while (source.next(chunk)) {
        for (uint32_t r = 0; r < chunk.count; r++) {
            const RowId rid = chunk.row_ids[r];
            if (rid < 0 || static_cast<uint64_t>(rid) >= count_) {
                throw Error(ErrorCode::InvalidParam,
                            "Engine::pass2: row_id out of range");
            }
            const float* vec = chunk.vectors + static_cast<size_t>(r) * dim_;
            quantizer_->encode(vec, codes_buffer_ +
                                       static_cast<size_t>(rid) * code_size_);
            encoded++;
        }
    }
    spdlog::info("[sextant] pass 2: encoded {} vectors", encoded);
}

// ===========================================================================
// Parallel construct (SDC, Issue 29 Mode C)
// ===========================================================================

void Engine::parallel_construct(const ResolvedParams& params) {
    const uint32_t n = static_cast<uint32_t>(count_);
    const uint32_t nthreads = params.num_threads > 0
                                  ? params.num_threads
                                  : std::thread::hardware_concurrency();
    spdlog::info("[sextant] construct: {} nodes across {} threads (SDC)", n,
                 nthreads);

    // The very first insert must be serialized before spawning tasks: it
    // claims the entry point (see VamanaCore::insert_build_from_code).
    {
        VamanaTLS tls;
        tls.resize(n);
        core_->insert_build_from_code(0, /*row_id=*/0, tls);
    }

    // Thread pool with per-thread VamanaTLS scratch.
    ctpl::thread_pool_tls<VamanaTLS> pool(
        nthreads,
        [n](size_t /*tid*/, std::shared_ptr<VamanaTLS>& tls) {
            tls = std::make_shared<VamanaTLS>();
            tls->resize(n);
        });

    // Split [1, n) into `nthreads` disjoint sub-ranges.
    const uint32_t lo = 1;
    const uint32_t hi = n;
    if (hi <= lo) {
        spdlog::info("[sextant] construct: only entry-point node (n=1)");
        return;
    }
    const uint32_t span = hi - lo;
    const uint32_t per = (span + nthreads - 1) / nthreads;

    std::vector<std::future<void>> futs;
    for (uint32_t t = 0; t < nthreads; t++) {
        const uint32_t t_lo = lo + t * per;
        const uint32_t t_hi = std::min(lo + (t + 1) * per, hi);
        if (t_lo >= t_hi) break;

        auto fut = pool.push(
            [this, t_lo, t_hi](size_t /*tid*/, VamanaTLS& tls) {
                VamanaCore& core = *core_;
                for (uint32_t id = t_lo; id < t_hi; id++) {
                    core.insert_build_from_code(id, static_cast<RowId>(id), tls);
                }
            });
        futs.push_back(std::move(fut));
    }

    // Wait for all tasks; rethrow the first exception encountered.
    for (auto& f : futs) {
        f.get();
    }

    spdlog::info("[sextant] construct: all {} nodes inserted", n);
}

// ===========================================================================
// Finalize + flush
// ===========================================================================

void Engine::finalize_and_flush(const ResolvedParams& params) {
    spdlog::info("[sextant] finalize: computing entry points");
    core_->compute_entry_points();

    // finalize_inline_codes operates on the build (inline_pq=0) layout, which
    // is a no-op. The final-layout inlining happens during flush below.
    // (Issue 26: build nodes are flat; flush reformats to final node_size.)
    core_->finalize_inline_codes();

    spdlog::info("[sextant] flush: writing sidecar files");
    flush_sidecars(params);
}

// ===========================================================================
// Partitioned build (Step 11)
//
// Phases:
//   1. Global PQ train + encode all vectors → codes_buffer_ (global codebook).
//   2. K-means on PQ codes → K shards with closure_factor overlap.
//   3. Per-shard build: each shard builds at R_shard = 2R/3, referencing a
//      contiguous copy of its members' PQ codes. Nodes store GLOBAL row_ids.
//   4. Merge: union neighbor lists per global node, remap shard-local IDs →
//      global IDs, truncate to R by SDC code distance.
//   5. Flush: standard flush_sidecars (merged graph + global codes).
// ===========================================================================

BuildResult Engine::build_partitioned(VectorSource& source,
                                       const std::string& index_path,
                                       const ResolvedParams& params) {
    using engine_detail::write_padded;
    using engine_detail::fill_header;
    using engine_detail::read_exact;

    spdlog::info("[sextant] partitioned build: N={} K={} closure_factor={:.4f}",
                 count_, params.K, params.closure_factor);

    // --- 1. Quantizer (global) ---
    quantizer_ = std::make_unique<PqQuantizer>(
        params.metric, dim_, params.pq_m, params.pq_bits);
    code_size_ = quantizer_->code_size();

    // node_size for the FINAL build buffer (inline_pq=0). Shards build at
    // R_shard, but the merged result lands in nodes_buffer_ at full R.
    node_size_ = VamanaCore::static_node_size(params.R, 0,
                                              static_cast<uint8_t>(code_size_));

    // Allocate global flat buffers.
    {
        const size_t codes_bytes = static_cast<size_t>(count_) * code_size_;
        const size_t nodes_bytes = static_cast<size_t>(count_) * node_size_;
        codes_buffer_ = static_cast<uint8_t*>(
            aligned_alloc(kDiskAlign, codes_bytes));
        nodes_buffer_ = static_cast<uint8_t*>(
            aligned_alloc(kDiskAlign, nodes_bytes));
        if (!codes_buffer_ || !nodes_buffer_) {
            throw Error(ErrorCode::OutOfMemory,
                        "build_partitioned: buffer alloc failed");
        }
        std::memset(codes_buffer_, 0, codes_bytes);
        std::memset(nodes_buffer_, 0, nodes_bytes);
    }

    // Train + encode (same passes as monolithic). The quantizer needs its
    // cross-distance table for SDC; PqQuantizer builds it during train().
    pass1_sample_and_train(source, params);
    pass2_encode(source, params);

    // --- 2. Partition via k-means on PQ codes ---
    const uint32_t n = static_cast<uint32_t>(count_);
    auto assignment = partition_codes(*quantizer_, codes_buffer_, n,
                                      code_size_, params.K,
                                      params.closure_factor);
    const uint32_t K = static_cast<uint32_t>(assignment.shards.size());

    // --- 3. Per-shard build ---
    // Each shard: contiguous codes buffer (copy of members' global codes),
    // VamanaCore at R_shard, parallel construct. Nodes store row_id = global ID.
    const uint16_t R_shard = static_cast<uint16_t>(
        std::max<uint32_t>(8u, (2u * params.R) / 3u));
    spdlog::info("[sextant] partitioned: R_shard={} (2R/3, R={})", R_shard,
                 params.R);

    const uint32_t shard_node_size = VamanaCore::static_node_size(
        R_shard, 0, static_cast<uint8_t>(code_size_));

    // Store each shard's node buffer + local→global map for the merge.
    std::vector<std::vector<uint8_t>> shard_node_bufs(K);
    std::vector<std::vector<uint32_t>> shard_local_to_global(K);

    // Build a single VamanaParams template; R/L per shard.
    VamanaParams vp_shard;
    vp_shard.dim = dim_;
    vp_shard.R = R_shard;
    vp_shard.L = params.L;
    vp_shard.L_build = params.L_build;
    vp_shard.alpha = params.alpha;
    vp_shard.inline_pq_count = 0;
    vp_shard.n_entry_points = 16;
    vp_shard.max_occlusion = params.max_occlusion;

    for (uint32_t k = 0; k < K; k++) {
        auto& members = assignment.shards[k];
        const uint32_t shard_n = static_cast<uint32_t>(members.size());
        if (shard_n == 0) {
            spdlog::warn("[sextant] shard {} empty, skipping", k);
            continue;
        }
        spdlog::info("[sextant] building shard {}/{} ({} vectors, RAM≈{:.1f}MB, "
                     "{} threads)",
                     k, K, shard_n,
                     (static_cast<double>(shard_n) *
                          (code_size_ + shard_node_size)) /
                         1e6,
                     params.num_threads);

        // Contiguous shard codes: local index i → members[i]'s global code.
        std::vector<uint8_t> shard_codes(
            static_cast<size_t>(shard_n) * code_size_, 0);
        shard_local_to_global[k].resize(shard_n);
        for (uint32_t i = 0; i < shard_n; i++) {
            const uint32_t gid = members[i];
            shard_local_to_global[k][i] = gid;
            std::memcpy(shard_codes.data() +
                            static_cast<size_t>(i) * code_size_,
                        codes_buffer_ + static_cast<size_t>(gid) * code_size_,
                        code_size_);
        }

        // Shard node buffer (aligned for direct-IO reuse).
        const size_t shard_nodes_bytes =
            static_cast<size_t>(shard_n) * shard_node_size;
        shard_node_bufs[k].resize(shard_nodes_bytes, 0);
        uint8_t* shard_nodes = shard_node_bufs[k].data();
        uint8_t* shard_codes_ptr = shard_codes.data();

        // Build a fresh VamanaCore for this shard.
        VamanaCore core(vp_shard, *quantizer_);
        core.set_build_codes(shard_codes_ptr, shard_n);
        core.set_build_nodes(shard_nodes);
        core.prepare_for_build(shard_n);

        // First insert must be serialized (entry-point claim).
        {
            VamanaTLS tls;
            tls.resize(shard_n);
            core.insert_build_from_code(0, /*row_id=*/members[0], tls);
        }

        // Parallel construct over local IDs [1, shard_n).
        const uint32_t nthreads = params.num_threads > 0
                                      ? params.num_threads
                                      : std::thread::hardware_concurrency();
        ctpl::thread_pool_tls<VamanaTLS> pool(
            std::max<uint32_t>(1, nthreads),
            [shard_n](size_t /*tid*/, std::shared_ptr<VamanaTLS>& tls) {
                tls = std::make_shared<VamanaTLS>();
                tls->resize(shard_n);
            });

        const uint32_t lo = 1;
        const uint32_t hi = shard_n;
        std::vector<std::future<void>> futs;
        if (hi > lo) {
            const uint32_t span = hi - lo;
            const uint32_t per =
                (span + std::max<uint32_t>(1, nthreads) - 1) /
                std::max<uint32_t>(1, nthreads);
            for (uint32_t t = 0; t < nthreads; t++) {
                const uint32_t t_lo = lo + t * per;
                const uint32_t t_hi = std::min(lo + (t + 1) * per, hi);
                if (t_lo >= t_hi) break;
                auto fut = pool.push(
                    [&core, &members, t_lo, t_hi](
                        size_t /*tid*/, VamanaTLS& tls) {
                        for (uint32_t id = t_lo; id < t_hi; id++) {
                            core.insert_build_from_code(
                                id, static_cast<RowId>(members[id]), tls);
                        }
                    });
                futs.push_back(std::move(fut));
            }
            for (auto& f : futs) f.get();
        }
        // Shard built; node buffer retained in shard_node_bufs[k].
    }

    // --- 4. Merge ---
    // For each global node, collect neighbor lists from all shards that
    // contain it (remapping shard-local IDs → global IDs), union, dedup,
    // truncate to R by SDC distance.
    // Merge is a three-phase streaming pass:
    //   (a) Gather each global node's candidates from its shard neighbor lists
    //       (shard-local IDs remapped to global IDs).
    //   (b) Add reciprocal edges: if A→B is a candidate, B→A becomes one too.
    //       This makes the merged graph undirected and guarantees connectivity
    //       (each shard's Vamana build is internally connected via the shared
    //       entry-point seed, and closure_factor overlap bridges shards).
    //   (c) For each node: dedup + truncate to R by SDC distance, write out.
    spdlog::info("[sextant] merging {} shards into global graph (R={})", K,
                 params.R);

    // (a) Gather candidates into per-node adjacency sets. We store one sorted,
    // deduped neighbor list per node. To bound memory we use vector-of-vectors
    // and reserve R_shard×K max.
    std::vector<std::vector<uint32_t>> adj(n);
    for (uint32_t k = 0; k < K; k++) {
        const auto& l2g = shard_local_to_global[k];
        const uint32_t shard_n = static_cast<uint32_t>(l2g.size());
        for (uint32_t lid = 0; lid < shard_n; lid++) {
            const uint32_t gid = l2g[lid];
            const uint8_t* snode =
                shard_node_bufs[k].data() +
                static_cast<size_t>(lid) * shard_node_size;
            const uint16_t ndeg = VamanaCore::get_neighbor_count(snode);
            for (uint16_t i = 0; i < ndeg; i++) {
                const uint32_t local_nb = VamanaCore::get_neighbor(snode, i);
                if (local_nb >= shard_n) continue;
                const uint32_t gnb = l2g[local_nb];
                if (gnb == gid) continue;  // no self-loops
                adj[gid].push_back(gnb);
            }
        }
    }

    // (b) Add reciprocal edges. For each A→B already gathered, ensure B also
    // lists A. We append to adj[B]; dedup happens in the truncate pass.
    for (uint32_t a = 0; a < n; a++) {
        for (uint32_t b : adj[a]) {
            adj[b].push_back(a);
        }
    }

    // (c) Dedup + truncate to R by SDC distance, write to nodes_buffer_.
    std::vector<uint32_t> seen(n, 0);
    uint32_t visit_token = 0;
    for (uint32_t gid = 0; gid < n; gid++) {
        uint8_t* out_node =
            nodes_buffer_ + static_cast<size_t>(gid) * node_size_;
        std::memset(out_node, 0,
                    kNeighborArrayOffset +
                        static_cast<size_t>(params.R) * sizeof(uint32_t));
        VamanaCore::set_row_id(out_node, static_cast<RowId>(gid));
        VamanaCore::set_internal_id(out_node, gid);
        VamanaCore::set_neighbor_count(out_node, 0);
        VamanaCore::set_inline_pq_count(out_node, 0);

        ++visit_token;
        std::vector<std::pair<float, uint32_t>> cands;
        cands.reserve(adj[gid].size());
        const uint8_t* my_code =
            codes_buffer_ + static_cast<size_t>(gid) * code_size_;
        for (uint32_t gnb : adj[gid]) {
            if (gnb == gid) continue;
            if (seen[gnb] == visit_token) continue;  // dedup
            seen[gnb] = visit_token;
            const uint8_t* nb_code =
                codes_buffer_ + static_cast<size_t>(gnb) * code_size_;
            const float d = quantizer_->code_distance(my_code, nb_code);
            cands.emplace_back(d, gnb);
        }
        // Free the adjacency now that we've consumed it.
        std::vector<uint32_t>().swap(adj[gid]);

        // Truncate to R: keep the R closest by SDC distance.
        if (cands.size() > params.R) {
            std::nth_element(
                cands.begin(), cands.begin() + params.R, cands.end(),
                [](const std::pair<float, uint32_t>& a,
                   const std::pair<float, uint32_t>& b) {
                    return a.first < b.first;
                });
            cands.resize(params.R);
        }
        std::sort(cands.begin(), cands.end(),
                  [](const std::pair<float, uint32_t>& a,
                     const std::pair<float, uint32_t>& b) {
                      return a.first < b.first;
                  });

        const uint16_t deg = static_cast<uint16_t>(cands.size());
        VamanaCore::set_neighbor_count(out_node, deg);
        for (uint16_t i = 0; i < deg; i++) {
            VamanaCore::set_neighbor(out_node, i, cands[i].second);
        }
    }

    // (d) Connectivity repair. The reciprocal edges make each node's local
    // neighborhood undirected, but the K shards may still form separate
    // connected components when closure_factor overlap is low on tightly-
    // clustered data. Union-Find detects components; we bridge each minor
    // component to the largest one with a single bidirectional edge between
    // the SDC-closest pair. This guarantees a single connected component.
    {
        std::vector<uint32_t> parent(n);
        for (uint32_t i = 0; i < n; i++) parent[i] = i;
        std::function<uint32_t(uint32_t)> find = [&](uint32_t x) -> uint32_t {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };
        auto uni = [&](uint32_t a, uint32_t b) {
            parent[find(a)] = find(b);
        };
        for (uint32_t a = 0; a < n; a++) {
            const uint8_t* node =
                nodes_buffer_ + static_cast<size_t>(a) * node_size_;
            const uint16_t deg = VamanaCore::get_neighbor_count(node);
            for (uint16_t i = 0; i < deg; i++) {
                uni(a, VamanaCore::get_neighbor(node, i));
            }
        }
        // Count components and their representative (smallest member).
        std::unordered_map<uint32_t, std::vector<uint32_t>> comp_map;
        for (uint32_t i = 0; i < n; i++) comp_map[find(i)].push_back(i);

        if (comp_map.size() > 1) {
            // Identify the largest component as the root.
            uint32_t root_rep = 0;
            size_t root_size = 0;
            for (auto& [rep, members] : comp_map) {
                if (members.size() > root_size) {
                    root_size = members.size();
                    root_rep = rep;
                }
            }
            spdlog::warn("[sextant] merge: {} components (largest={}); "
                         "bridging", comp_map.size(), root_size);

            // For each minor component, bridge to the root via the SDC-nearest
            // pair (greedy O(|comp| × |root|) is too costly for large roots;
            // we sample the root side to 1024 candidates).
            std::vector<uint32_t> root_sample;
            const auto& root_members = comp_map[root_rep];
            if (root_members.size() > 1024) {
                std::mt19937 rng(0xBAD5eed);
                std::uniform_int_distribution<size_t> d(
                    0, root_members.size() - 1);
                root_sample.reserve(1024);
                for (size_t s = 0; s < 1024; s++) {
                    root_sample.push_back(root_members[d(rng)]);
                }
            } else {
                root_sample = root_members;
            }
            for (auto& [rep, members] : comp_map) {
                if (rep == root_rep) continue;
                // Find nearest (comp_member, root_sample) pair by SDC.
                float best_d = std::numeric_limits<float>::max();
                uint32_t best_c = members[0];
                uint32_t best_r = root_sample[0];
                for (uint32_t c : members) {
                    const uint8_t* cc =
                        codes_buffer_ + static_cast<size_t>(c) * code_size_;
                    for (uint32_t r : root_sample) {
                        const uint8_t* rc =
                            codes_buffer_ +
                            static_cast<size_t>(r) * code_size_;
                        const float d = quantizer_->code_distance(cc, rc);
                        if (d < best_d) {
                            best_d = d;
                            best_c = c;
                            best_r = r;
                        }
                    }
                }
                // Add bidirectional edge best_c ↔ best_r. Append to each
                // node's neighbor list (both have room since R_shard*... but
                // final R may be full). We overwrite the last neighbor slot if
                // full to guarantee the bridge edge exists.
                auto append_edge = [&](uint32_t from, uint32_t to) {
                    uint8_t* node = nodes_buffer_ +
                                    static_cast<size_t>(from) * node_size_;
                    uint16_t deg = VamanaCore::get_neighbor_count(node);
                    uint16_t slot = deg;
                    // Check if 'to' already a neighbor.
                    for (uint16_t i = 0; i < deg; i++) {
                        if (VamanaCore::get_neighbor(node, i) == to) {
                            slot = 0xFFFF;  // already present
                            break;
                        }
                    }
                    if (slot == 0xFFFF) return;
                    if (deg < params.R) {
                        VamanaCore::set_neighbor(node, deg, to);
                        VamanaCore::set_neighbor_count(node, deg + 1);
                    } else {
                        // Replace the farthest neighbor (last slot, since list
                        // is sorted by distance ascending).
                        VamanaCore::set_neighbor(node, params.R - 1, to);
                    }
                };
                append_edge(best_c, best_r);
                append_edge(best_r, best_c);
            }
        }
    }
    // Debug: merged-graph degree histogram (post-repair).
    {
        uint64_t zero_deg = 0;
        uint64_t total_edges = 0;
        uint64_t max_deg = 0;
        for (uint32_t gid = 0; gid < n; gid++) {
            const uint8_t* node =
                nodes_buffer_ + static_cast<size_t>(gid) * node_size_;
            const uint16_t d = VamanaCore::get_neighbor_count(node);
            if (d == 0) zero_deg++;
            total_edges += d;
            if (d > max_deg) max_deg = d;
        }
        spdlog::info("[sextant] merge degrees: zero={}, avg={:.1f}, max={}",
                     zero_deg, static_cast<double>(total_edges) / n, max_deg);
    }

    spdlog::info("[sextant] merge complete; flushing sidecars");

    // Set up the master VamanaCore (full R) over the merged nodes_buffer_ for
    // entry-point computation and final-layout inlining during flush.
    VamanaParams vp_full;
    vp_full.dim = dim_;
    vp_full.R = params.R;
    vp_full.L = params.L;
    vp_full.L_build = params.L_build;
    vp_full.alpha = params.alpha;
    vp_full.inline_pq_count = 0;
    vp_full.n_entry_points = 16;
    vp_full.max_occlusion = params.max_occlusion;
    core_ = std::make_unique<VamanaCore>(vp_full, *quantizer_);
    core_->set_build_codes(codes_buffer_, n);
    core_->set_build_nodes(nodes_buffer_);
    core_->prepare_for_build(n);

    // --- 5. Flush (entry points + final-layout inlining + sidecars) ---
    core_->compute_entry_points();
    core_->finalize_inline_codes();
    flush_sidecars(params);

    opened_ = true;

    BuildResult result;
    result.index_path = index_path;
    result.n_vectors = count_;
    result.dim = dim_;
    result.R = params.R;
    result.L_build = params.L_build;
    result.pq_m = params.pq_m;
    return result;
}

// ===========================================================================
// Flush sidecars (Issue 32 — ring-buffered; here, sequential padded writes)
// ===========================================================================

void Engine::flush_sidecars(const ResolvedParams& params) {
    const auto uuid = make_uuid();
    const uint32_t n = static_cast<uint32_t>(count_);

    // Final node layout: node_size with the resolved inline_pq_count.
    const uint32_t final_node_size = VamanaCore::static_node_size(
        params.R, params.inline_pq_count, static_cast<uint8_t>(code_size_));
    const uint32_t build_node_size = node_size_;  // inline_pq=0 layout

    // ----- .codes -----
    {
        const std::string path = index_path_ + ".codes";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicCodes, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);
        const size_t codes_bytes = static_cast<size_t>(n) * code_size_;
        write_padded(f, codes_buffer_, codes_bytes, sizeof(h));
        f.sync();
        spdlog::info("[sextant] wrote {} ({} bytes)", path, codes_bytes);
    }

    // ----- .graph (reformat to final layout, inline neighbor PQ codes) -----
    {
        const std::string path = index_path_ + ".graph";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicGraph, count_, dim_, uuid);
        // Stash the final node_size in the checksum_algo field? No — keep
        // header clean; the reader recomputes final_node_size from params.
        write_padded(f, &h, sizeof(h), 0);

        // For each build node, produce a final-layout node in a staging
        // buffer and write it. We batch into a ring of aligned blocks to
        // amortize syscalls (Issue 32).
        const size_t block_cap = kBlockSize;  // 256KB
        const uint32_t per_block =
            std::max<uint32_t>(1, static_cast<uint32_t>(block_cap / final_node_size));
        const size_t buf_cap = static_cast<size_t>(per_block) * final_node_size;
        uint8_t* ring = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, buf_cap));

        const uint32_t neighbor_region_end =
            kNeighborArrayOffset + params.R * sizeof(uint32_t);
        const uint32_t inline_region_off =
            (neighbor_region_end + 7u) & ~7u;

        uint64_t write_off = sizeof(h);
        uint32_t in_block = 0;
        for (uint32_t id = 0; id < n; id++) {
            const uint8_t* src = nodes_buffer_ +
                                 static_cast<size_t>(id) * build_node_size;
            uint8_t* dst = ring + static_cast<size_t>(in_block) * final_node_size;

            // Copy the fixed header + neighbor array (identical in both layouts
            // since the build and final layouts share the same R).
            const uint32_t copy_len = (neighbor_region_end + 7u) & ~7u;
            std::memcpy(dst, src, copy_len);

            // Inline the first inline_pq_count neighbors' PQ codes.
            if (params.inline_pq_count > 0) {
                const uint16_t ndeg =
                    VamanaCore::get_neighbor_count(src);
                const uint16_t nin = std::min<uint16_t>(
                    ndeg, params.inline_pq_count);
                for (uint16_t i = 0; i < nin; i++) {
                    const uint32_t nb = VamanaCore::get_neighbor(src, i);
                    if (nb < n) {
                        std::memcpy(
                            dst + inline_region_off +
                                static_cast<size_t>(i) * code_size_,
                            codes_buffer_ + static_cast<size_t>(nb) * code_size_,
                            code_size_);
                    }
                }
            }

            in_block++;
            if (in_block >= per_block) {
                write_padded(f, ring,
                             static_cast<size_t>(in_block) * final_node_size,
                             write_off);
                write_off += static_cast<size_t>(in_block) * final_node_size;
                in_block = 0;
            }
        }
        if (in_block > 0) {
            write_padded(f, ring,
                         static_cast<size_t>(in_block) * final_node_size,
                         write_off);
        }
        aligned_free(ring);
        f.sync();
        spdlog::info("[sextant] wrote {} ({} nodes, {} bytes/node)", path, n,
                     final_node_size);
    }

    // ----- .meta (serialized quantizer + entry points + params) -----
    {
        const std::string path = index_path_ + ".meta";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicMeta, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);

        // Serialize the quantizer.
        std::vector<uint8_t> qblob;
        quantizer_->serialize(qblob);
        uint64_t qsize = qblob.size();
        // Payload layout: [u64 quantizer_size][quantizer_bytes]
        //                 [u16 entry_point_count][entry_point_count × u32]
        //                 [ResolvedParams POD block]
        std::vector<uint8_t> payload;
        payload.insert(payload.end(),
                       reinterpret_cast<uint8_t*>(&qsize),
                       reinterpret_cast<uint8_t*>(&qsize) + sizeof(qsize));
        payload.insert(payload.end(), qblob.begin(), qblob.end());

        const auto& eps = core_->entry_points();
        uint16_t ep_count = static_cast<uint16_t>(eps.size());
        payload.insert(payload.end(),
                       reinterpret_cast<uint8_t*>(&ep_count),
                       reinterpret_cast<uint8_t*>(&ep_count) + sizeof(ep_count));
        for (uint32_t ep : eps) {
            payload.insert(payload.end(),
                           reinterpret_cast<uint8_t*>(&ep),
                           reinterpret_cast<uint8_t*>(&ep) + sizeof(ep));
        }

        // Append the resolved params as a POD block so open() can rebuild the
        // VamanaCore with the same R/L/alpha/inline_pq/max_occlusion.
        ResolvedParams p = params;  // copy
        payload.insert(payload.end(),
                       reinterpret_cast<uint8_t*>(&p),
                       reinterpret_cast<uint8_t*>(&p) + sizeof(p));

        write_padded(f, payload.data(), payload.size(), sizeof(h));
        f.sync();
        spdlog::info("[sextant] wrote {} ({} bytes payload, {} entry points)",
                     path, payload.size(), eps.size());
    }

    // ----- .manifest (atomic commit — written LAST via temp + rename) -----
    {
        const std::string path = index_path_ + ".manifest";
        const std::string tmp = path + ".tmp";
        {
            DirectFile f(tmp, true);
            SidecarHeader h;
            fill_header(h, kMagicManifest, count_, dim_, uuid);
            write_padded(f, &h, sizeof(h), 0);
            // The manifest is the commit point. We record the four sidecar
            // basenames + a "ready" marker.
            std::string commit =
                std::string("ready\n") +
                std::to_string(count_) + "\n" +
                std::to_string(dim_) + "\n" +
                std::to_string(params.R) + "\n" +
                std::to_string(params.pq_m) + "\n";
            write_padded(f, commit.data(), commit.size(), sizeof(h));
            f.sync();
        }
        // Atomic rename: the manifest appears all at once, signaling a
        // complete, consistent build.
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            throw Error(ErrorCode::IoError,
                        "Engine::flush: manifest rename failed: " + ec.message());
        }
        spdlog::info("[sextant] wrote {} (commit point)", path);
    }
}

// ===========================================================================
// insert / flush
//
// Live insert = "build one node". We grow the flat codes/nodes buffers by one
// slot, encode the vector, then drive the Vamana insert flow (beam_search →
// robust_prune → connect_and_prune) via VamanaCore::insert_build.
//
// LIMITATION (Phase 1): the realloc strategy is O(N) per insert (copy the whole
// codes + nodes buffers). This is acceptable because live insert is NOT the hot
// path — batch build is. For high-throughput live ingest, a slab/arena growth
// strategy is deferred to a later phase.
// ===========================================================================

void Engine::insert(const float* vec, Dim dim, RowId row_id) {
    if (!opened_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: index not opened");
    }
    if (!quantizer_ || !core_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: quantizer/core not initialized");
    }
    if (vec == nullptr) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: null vector");
    }
    if (dim != dim_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: dim mismatch");
    }
    if (code_size_ == 0 || node_size_ == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: code/node size not initialized");
    }

    const uint32_t new_internal = static_cast<uint32_t>(count_);
    const uint64_t new_count = count_ + 1;

    // --- 1. Grow the codes buffer by one code (O(N) copy). ---
    {
        const size_t old_bytes = static_cast<size_t>(count_) * code_size_;
        const size_t new_bytes = static_cast<size_t>(new_count) * code_size_;
        const size_t alloc_bytes = (new_bytes + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
        uint8_t* nb =
            static_cast<uint8_t*>(aligned_alloc(kDiskAlign, alloc_bytes));
        if (!nb) {
            throw Error(ErrorCode::OutOfMemory,
                        "Engine::insert: codes realloc failed");
        }
        std::memcpy(nb, codes_buffer_, old_bytes);
        // Encode the new vector into the appended slot.
        quantizer_->encode(vec, nb + old_bytes);
        aligned_free(codes_buffer_);
        codes_buffer_ = nb;
    }

    // --- 2. Grow the nodes buffer by one node (O(N) copy). ---
    {
        const size_t old_bytes = static_cast<size_t>(count_) * node_size_;
        const size_t new_bytes = static_cast<size_t>(new_count) * node_size_;
        const size_t alloc_bytes = (new_bytes + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
        uint8_t* nb =
            static_cast<uint8_t*>(aligned_alloc(kDiskAlign, alloc_bytes));
        if (!nb) {
            throw Error(ErrorCode::OutOfMemory,
                        "Engine::insert: nodes realloc failed");
        }
        std::memcpy(nb, nodes_buffer_, old_bytes);
        std::memset(nb + old_bytes, 0, node_size_);
        aligned_free(nodes_buffer_);
        nodes_buffer_ = nb;
    }

    // --- 3. Publish the grown buffers + new count to the core. ---
    count_ = new_count;
    core_->set_build_codes(codes_buffer_, static_cast<uint32_t>(count_));
    core_->set_build_nodes(nodes_buffer_);

    // --- 4. Drive the Vamana insert flow (single-thread). ---
    //    insert_build handles the first-node case (becomes an entry point)
    //    and the general case (beam_search → robust_prune → connect_and_prune).
    //    It also bumps core_->count_ to internal_id + 1.
    VamanaTLS tls;
    tls.resize(static_cast<uint32_t>(count_));
    core_->insert_build(new_internal, row_id, vec, tls);

    spdlog::info("[sextant] insert: row_id={} internal_id={} (count now {})",
                 row_id, new_internal, count_);
}

void Engine::flush() {
    // After build(), all sidecars are already written synchronously.
    // flush() is meaningful after open() + insert(): it rewrites the sidecars
    // with the grown buffers (already in final layout, so no reformatting).
    if (!opened_ || !params_loaded_) {
        return;  // nothing pending
    }
    if (count_ == 0) {
        return;
    }
    if (!quantizer_ || !core_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::flush: quantizer/core not initialized");
    }

    const auto uuid = make_uuid();
    const ResolvedParams& params = loaded_params_;
    const uint32_t n = static_cast<uint32_t>(count_);

    // After open(), nodes_buffer_ is already in final layout (node_size_
    // includes inline_pq_count), so we write it verbatim — no reformat pass.
    spdlog::info("[sextant] flush: persisting {} vectors to '{}'", n,
                 index_path_);

    // ----- .codes -----
    {
        const std::string path = index_path_ + ".codes";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicCodes, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);
        const size_t codes_bytes = static_cast<size_t>(n) * code_size_;
        write_padded(f, codes_buffer_, codes_bytes, sizeof(h));
        f.sync();
    }

    // ----- .graph (verbatim — buffers already in final layout) -----
    {
        const std::string path = index_path_ + ".graph";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicGraph, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);
        const size_t nodes_bytes = static_cast<size_t>(n) * node_size_;
        write_padded(f, nodes_buffer_, nodes_bytes, sizeof(h));
        f.sync();
    }

    // ----- .meta (quantizer + entry points + params) -----
    {
        const std::string path = index_path_ + ".meta";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicMeta, count_, dim_, uuid);
        write_padded(f, &h, sizeof(h), 0);

        std::vector<uint8_t> qblob;
        quantizer_->serialize(qblob);
        uint64_t qsize = qblob.size();
        std::vector<uint8_t> payload;
        payload.insert(payload.end(),
                       reinterpret_cast<uint8_t*>(&qsize),
                       reinterpret_cast<uint8_t*>(&qsize) + sizeof(qsize));
        payload.insert(payload.end(), qblob.begin(), qblob.end());

        const auto& eps = core_->entry_points();
        uint16_t ep_count = static_cast<uint16_t>(eps.size());
        payload.insert(payload.end(),
                       reinterpret_cast<uint8_t*>(&ep_count),
                       reinterpret_cast<uint8_t*>(&ep_count) + sizeof(ep_count));
        for (uint32_t ep : eps) {
            payload.insert(payload.end(),
                           reinterpret_cast<uint8_t*>(&ep),
                           reinterpret_cast<uint8_t*>(&ep) + sizeof(ep));
        }
        ResolvedParams p = params;
        payload.insert(payload.end(),
                       reinterpret_cast<uint8_t*>(&p),
                       reinterpret_cast<uint8_t*>(&p) + sizeof(p));
        write_padded(f, payload.data(), payload.size(), sizeof(h));
        f.sync();
    }

    // ----- .manifest (atomic commit) -----
    {
        const std::string path = index_path_ + ".manifest";
        const std::string tmp = path + ".tmp";
        {
            DirectFile f(tmp, true);
            SidecarHeader h;
            fill_header(h, kMagicManifest, count_, dim_, uuid);
            write_padded(f, &h, sizeof(h), 0);
            std::string commit =
                std::string("ready\n") +
                std::to_string(count_) + "\n" +
                std::to_string(dim_) + "\n" +
                std::to_string(params.R) + "\n" +
                std::to_string(params.pq_m) + "\n";
            write_padded(f, commit.data(), commit.size(), sizeof(h));
            f.sync();
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            throw Error(ErrorCode::IoError,
                        "Engine::flush: manifest rename failed: " + ec.message());
        }
    }

    spdlog::info("[sextant] flush: sidecars rewritten (count={})", count_);
}

}  // namespace sextant
