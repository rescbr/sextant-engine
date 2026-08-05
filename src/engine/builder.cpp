// Builder — bulk-build + mutate an Index.
//
// Owns the two-pass streaming + parallel PQ-construct pipeline plus the
// post-build mutators (insert/flush). Methods moved here from engine.cpp;
// Engine keeps thin inline delegating wrappers in the header (deleted in
// Phase E along with Engine itself).
//
// Build pipeline:
//   1. resolve_params (auto-defaults from N, dim)  [caller]
//   2. allocate flat codes + nodes buffers         [build_partitioned]
//   3. Pass 1: reservoir sample (256K) + PQ train  [pass1_sample_and_train]
//   4. Pass 2: encode all vectors → codes buffer   [pass2_encode]
//   5. Parallel PQ-construct via CTPL             [parallel_construct]
//   6. Finalize: compute_entry_points              [snap_entry_points_]
//   7. Flush sidecars (.graph/.codes/.meta/.manifest)  [write_sidecars_]

#include "sextant/builder.hpp"

#include "partition.hpp"
#include "probe.hpp"
#include "manifest_io.hpp"
#include "sidecar_io.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "quant/anisotropic_pq_quantizer.hpp"
#include "quant/product_residual_quantizer.hpp"
#include "quant/rabitq_quantizer.hpp"
#include "storage/direct_io.hpp"
#include "storage/memgraph.hpp"      // complete type for Index's unique_ptr<MemGraph>
#include "storage/sidecar_header.hpp"
#include "util/fp16.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <cstdio>
#include <deque>
#include <sys/mman.h>   // madvise for buffer page placement
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sextant {

using engine_detail::fill_header;
using engine_detail::read_exact;
using engine_detail::write_padded;

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

/// PageShuffle — BFS reorder of build IDs → disk positions.
std::vector<uint32_t> compute_bfs_order(
    const uint8_t* nodes_buffer, uint32_t n, uint32_t build_node_size,
    const std::vector<uint32_t>& entry_points) {
    std::vector<uint32_t> bfs_order;
    bfs_order.reserve(n);
    std::vector<bool> visited(n, false);
    std::queue<uint32_t> queue;
    for (uint32_t ep : entry_points) {
        if (ep < n && !visited[ep]) {
            visited[ep] = true;
            queue.push(ep);
        }
    }
    while (!queue.empty()) {
        uint32_t node = queue.front();
        queue.pop();
        bfs_order.push_back(node);
        const uint8_t* np = nodes_buffer + static_cast<size_t>(node) * build_node_size;
        uint16_t ncount = VamanaCore::get_neighbor_count(np);
        for (uint16_t i = 0; i < ncount; i++) {
            uint32_t nb = VamanaCore::get_neighbor(np, i);
            if (nb < n && !visited[nb]) {
                visited[nb] = true;
                queue.push(nb);
            }
        }
    }
    for (uint32_t i = 0; i < n; i++) {
        if (!visited[i]) bfs_order.push_back(i);
    }
    return bfs_order;
}

}  // namespace

// ===========================================================================
// Builder ctor
// ===========================================================================

Builder::Builder(Index& index) : index_(index) {}

// ===========================================================================
// build()
// ===========================================================================

BuildResult Builder::build(VectorSource& source, const std::string& index_path,
                          const BuildConfig& config) {
    // Resolve params, then forward to the ResolvedParams overload which does
    // the count/dim/path setup, build_partitioned, and timing.
    index_.count = source.count();
    index_.dim = source.dim();
    return build(source, index_path, resolve_params(index_.count, index_.dim, config));
}

BuildResult Builder::build(VectorSource& source, const std::string& index_path,
                          const ResolvedParams& params) {
    const auto t0 = std::chrono::steady_clock::now();

    index_.count = source.count();
    index_.dim = source.dim();
    if (index_.count == 0) {
        throw Error(ErrorCode::InvalidParam, "Builder::build: source is empty");
    }
    if (index_.dim == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Builder::build: source has dim=0");
    }
    index_.path = index_path;

    spdlog::info("[sextant] build: n={} dim={} → '{}'", index_.count, index_.dim,
                 index_path);

    auto result = build_partitioned(source, index_path, params);
    const auto t1 = std::chrono::steady_clock::now();
    result.build_time_sec =
        std::chrono::duration<double>(t1 - t0).count() + result.build_time_sec;
    spdlog::info("[sextant] build complete (K={}) in {:.2f}s", params.partition_count,
                 result.build_time_sec);
    return result;
}

Builder::PqSelection Builder::probe_pq_config(const float* sample, uint64_t n,
                                            Dim dim, const ResolvedParams& params) {
    const uint16_t pq_m = params.pq_m;
    const uint8_t pq_bits = params.pq_bits;
    if (pq_m != 0 && pq_bits != 0) {
        // Both explicit — no probe needed.
        return {pq_m, pq_bits, {}, ""};
    }
    const uint32_t pool_n =
        static_cast<uint32_t>(std::min<uint64_t>(kProbePool, n));
    // Pick query indices (deterministic).
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_int_distribution<uint32_t> u(0, pool_n - 1);
    std::vector<uint32_t> qidx;
    std::unordered_set<uint32_t> seen;
    while (qidx.size() < std::min<uint32_t>(kProbeQueries, pool_n - 1)) {
        const uint32_t q = u(rng);
        if (seen.insert(q).second) qidx.push_back(q);
    }
    const auto truth = compute_truth(sample, pool_n, dim, qidx);
    std::vector<ProbedConfig> all_cfg;
    std::string reason;
    const auto best = probe_best_config(sample, pool_n, dim, params.metric,
                                        truth, qidx,
                                        /*fixed_m=*/pq_m,
                                        /*fixed_bits=*/pq_bits,
                                        /*max_distortion=*/params.pq_max_distortion,
                                        all_cfg, reason);
    uint16_t m = best.m;
    uint8_t bits = best.bits;
    if (m == 0) {
        uint32_t mm = dim / 4; if (mm < 4) mm = 4;
        while (mm > 1 && dim % mm != 0) mm--;
        m = static_cast<uint16_t>(mm);
    }
    if (bits == 0) bits = 8;
    // Copy probed rows for display.
    std::vector<Builder::ProbedRow> rows;
    rows.reserve(all_cfg.size());
    for (const auto& c : all_cfg) {
        rows.push_back({c.m, c.bits, c.code_bytes, c.table_bytes,
                        c.distortion, c.band_recall, c.tie_fraction,
                        c.tie30_fraction, c.cost});
    }
    return {m, bits, std::move(rows), std::move(reason)};
}

void Builder::pass1_sample_and_train(VectorSource& source,
                                    const ResolvedParams& params,
                                    bool compute_entry_points) {
    spdlog::info("[sextant] pass 1: reservoir sample (target {} vectors)",
                 kSampleTarget);

    // Reservoir sampling (Algorithm R). We sample floats directly.
    const uint64_t sample_cap =
        std::min<uint64_t>(kSampleTarget, index_.count);
    if (sample_cap == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Builder::pass1: cannot sample from empty source");
    }
    std::vector<float> reservoir(static_cast<size_t>(sample_cap) * index_.dim);
    uint64_t actual_sample = 0;
    uint64_t seen = 0;

    // --- Sampling strategy dispatch ---
    // When N >> sample (and the source is a seekable file), use seek-based
    // random sampling: O(sample) seeks instead of O(N) sequential I/O.
    //
    // Threshold choice (measurement-driven, not heuristic): on this machine
    // a sequential read sustains ~3.5 GB/s (~0.88us per 3KB vector) while a
    // seek+read costs ~26us. Seek-based total cost is ~k*26us regardless of N;
    // full-scan cost is ~N*0.88us. They cross over near N/k ~ 30. Below that,
    // the sequential scan is faster (cache/RAID-stripe prefetching dominates
    // scattered-seek overhead). Above it, seek-based wins, dramatically so as
    // N grows (full-scan reads 3TB at N=1B; seek-based still reads ~60MB).
    // Use N > 30*sample_cap as the cutoff.
    //
    // Only applies when source.path() is non-empty (a real on-disk file);
    // MemorySource and future non-file sources fall back to reservoir.
    const std::string src_path = source.path();
    constexpr uint64_t kSeekRatioThreshold = 30;
    const bool seek_eligible = !src_path.empty() &&
                               index_.count > kSeekRatioThreshold * sample_cap;

    if (seek_eligible) {
        spdlog::info("[sextant] pass 1: seek-based sampling ({} of {} vectors; "
                     "ratio {:.1f}:1, threshold {}:1)", sample_cap, index_.count,
                     static_cast<double>(index_.count) /
                         static_cast<double>(sample_cap),
                     kSeekRatioThreshold);
        reservoir = draw_random_sample(src_path, index_.count, index_.dim, sample_cap);
        actual_sample = sample_cap;
        seen = index_.count;
    } else {
        spdlog::info("[sextant] pass 1: reservoir sample (target {} vectors)",
                     kSampleTarget);
        source.reset();

    // Algorithm R: keep the first `sample_cap`, then replace index j (j<k)
    // with probability k/i for the i-th seen item.
    //
    // rng() % (seen+1) instead of std::uniform_int_distribution: the latter
    // uses general-purpose rejection sampling (~107M/s); raw modulo is ~5x
    // faster (~530M/s). At billion scale Phase B (the post-fill steady state)
    // is ~9s with the distribution vs ~2s with modulo. The modulo bias at
    // seen ~ 1e9 against the 2^64 range is ~1e-10 — far below the statistical
    // precision a 20K sample carries, and irrelevant for parameter estimation.
    // (Phase B does no memcpy in the steady state: replacement probability
    // drops to sample_cap/seen ~ 2e-5 at 1B, so nearly every iteration is
    // just RNG + compare.)
    //
    // Why not parallelize the sampling? Not because of reproducibility — a
    // fixed seed gives determinism regardless of iteration order, and nothing
    // here pins to a specific sample (the result feeds statistical aggregations).
    // The actual reasons:
    //   1. The workload is one sequential pass over the base file (large
    //      6MB chunked reads). On the GCP bench VM's 2-SSD RAID0, the kernel's
    //      md layer already parallelizes these reads across both devices under
    //      a single thread (read size >> stripe size), so app-level parallel
    //      reads wouldn't add bandwidth.
    //   2. Total I/O dwarfs compute at billion scale (1B × 3KB = 3TB; nothing
    //      in the sampling loop speeds that up). The only way to go faster is
    //      to NOT read the whole file — which is what seek-based sampling
    //      (above) does when N >> sample.
    //   3. The modulo swap above already captured the cheap serial CPU win.
    // Distributed reservoir sampling (per-shard Algorithm R + weighted merge)
    // is a solved algorithm if we ever shard the input across hosts.
    std::mt19937_64 rng(0xC0DE1234ULL);
    Chunk chunk{};
    uint64_t filled = 0;
    // Time I/O (source.next) and compute (the per-vector reservoir update)
    // separately: I/O scales with dataset size, compute is what we'd consider
    // parallelizing. Algorithm R is inherently serial (single ordered RNG
    // stream), so compute parallelism isn't applicable here — but measuring
    // confirms whether it would matter if it were.
    double io_sec = 0.0;
    double compute_sec = 0.0;
    while (true) {
        const auto t_io0 = std::chrono::steady_clock::now();
        const bool got = source.next(chunk);
        io_sec += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_io0).count();
        if (!got) break;
        const auto t_c0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < chunk.count; r++) {
            const float* vec = chunk.vectors + static_cast<size_t>(r) * index_.dim;
            if (filled < sample_cap) {
                std::memcpy(reservoir.data() + filled * index_.dim, vec,
                            index_.dim * sizeof(float));
                filled++;
            } else {
                const uint64_t j = rng() % (seen + 1);
                if (j < sample_cap) {
                    std::memcpy(reservoir.data() + j * index_.dim, vec,
                                index_.dim * sizeof(float));
                }
            }
            seen++;
        }
        compute_sec += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_c0).count();
    }
    actual_sample = filled;
    {
        const double total_sec = io_sec + compute_sec;
        const double vecs_per_s = (total_sec > 1e-9)
            ? static_cast<double>(seen) / total_sec : 0.0;
        spdlog::debug("[sextant] pass 1: reservoir sampling took {:.3f}s "
                      "(I/O {:.3f}s + compute {:.3f}s); {} vectors scanned "
                      "({:.0f} vecs/s)",
                      total_sec, io_sec, compute_sec, seen, vecs_per_s);
    }
    }  // end reservoir branch

    spdlog::info("[sextant] pass 1: sampled {} / {} vectors", actual_sample,
                 seen);

    // Build requires explicit pq_m and pq_bits. The probe-based auto-selection
    // belongs in `sextant analyze` (the advisory tool); build is a committed
    // path and should never spend 30-130s probing on the reservoir.
    const uint16_t pq_m = params.pq_m;
    const uint8_t pq_bits = params.pq_bits;
    if (pq_m == 0 || pq_bits == 0) {
        throw Error(
            ErrorCode::InvalidParam,
            "build requires explicit --pq-m and --pq-bits (got pq_m=" +
                std::to_string(pq_m) + ", pq_bits=" +
                std::to_string(pq_bits) +
                "). Pass them on the command line, e.g. `sextant build ... "
                "--pq-m 96 --pq-bits 8`, or run `sextant analyze` first to "
                "get a dataset-specific recommendation.");
    }

    // Construct + train the quantizer at the resolved params.
    if (params.anisotropic_pq) {
        index_.quantizer = std::make_unique<AnisotropicPqQuantizer>(
            params.metric, index_.dim, pq_m, pq_bits);
    } else {
        index_.quantizer = std::make_unique<PqQuantizer>(
            params.metric, index_.dim, pq_m, pq_bits);
    }
    if (params.pq_anisotropy) {
        index_.quantizer->set_anisotropy(1.0f);
    }
    if (params.pq_opq) {
        index_.quantizer->enable_opq();
    }
    index_.code_size = index_.quantizer->code_size();
    std::string aniso_note;
    if (params.pq_anisotropy) {
        aniso_note = std::string(", anisotropy=on (covariance-based)");
    }
    if (params.pq_opq) {
        aniso_note += ", opq=on (PCA rotation)";
    }
    spdlog::info("[sextant] pass 1: training PQ (m={}, bits={}{}) on {} samples",
                 pq_m, pq_bits, aniso_note, actual_sample);
    index_.quantizer->set_num_threads(params.num_threads);
    index_.quantizer->train(reservoir.data(), actual_sample);
    spdlog::info("[sextant] pass 1: PQ trained (code_size={})", index_.code_size);

    // Compute global k-means centroids for entry-point selection (EP-2).
    // Only needed for graph builds (snap_entry_points_ consumes these at
    // flush time). The scan path skips this entirely — it never writes
    // graph sidecars (.ball, entry points in .meta).
    if (compute_entry_points && actual_sample >= params.n_entry_points) {
        const uint32_t k = std::min<uint32_t>(params.n_entry_points,
                                              static_cast<uint32_t>(actual_sample));
        const Dim dim = index_.dim;
        entry_centroids_.resize(static_cast<size_t>(k) * dim);
        const uint32_t n_threads = std::max(1u, std::min(params.num_threads,
            static_cast<uint32_t>(actual_sample)));

        // k-means++ seeding (sequential centroid selection, parallel dist).
        std::mt19937_64 rng(0xC0DE1234ULL);
        entry_centroids_.assign(static_cast<size_t>(k) * dim, 0.0f);
        std::memcpy(entry_centroids_.data(), reservoir.data(),
                    static_cast<size_t>(dim) * sizeof(float));
        std::vector<float> min_d2(actual_sample,
                                  std::numeric_limits<float>::max());
        for (uint32_t c = 1; c < k; c++) {
            const float* prev = entry_centroids_.data() +
                                 static_cast<size_t>(c - 1) * dim;
            // Parallel: update min_d2[i] with dist(reservoir[i], prev).
            std::atomic<uint64_t> next_i{0};
            auto dist_worker = [&]() {
                uint64_t i;
                while ((i = next_i.fetch_add(1, std::memory_order_relaxed))
                       < actual_sample) {
                    const float d = simd::l2sq_f32(
                        reservoir.data() + i * dim, prev, dim);
                    if (d < min_d2[i]) min_d2[i] = d;
                }
            };
            std::vector<std::thread> pool;
            for (uint32_t t = 0; t < n_threads; t++) pool.emplace_back(dist_worker);
            for (auto& th : pool) th.join();

            std::discrete_distribution<uint64_t> dist(min_d2.begin(), min_d2.end());
            const uint64_t picked = dist(rng);
            std::memcpy(entry_centroids_.data() + static_cast<size_t>(c) * dim,
                        reservoir.data() + static_cast<size_t>(picked) * dim,
                        static_cast<size_t>(dim) * sizeof(float));
        }

        // Lloyd iterations (10). Assign is parallel; update is serial but
        // cheap (k × dim accumulators).
        std::vector<uint32_t> assign(actual_sample, 0);
        for (uint32_t iter = 0; iter < 10; iter++) {
            // Parallel assign.
            std::atomic<uint64_t> next_i{0};
            auto assign_worker = [&]() {
                uint64_t i;
                while ((i = next_i.fetch_add(1, std::memory_order_relaxed))
                       < actual_sample) {
                    const float* v = reservoir.data() + i * dim;
                    float best = std::numeric_limits<float>::infinity();
                    uint32_t best_c = 0;
                    for (uint32_t c = 0; c < k; c++) {
                        const float d = simd::l2sq_f32(
                            v, entry_centroids_.data() +
                               static_cast<size_t>(c) * dim, dim);
                        if (d < best) { best = d; best_c = c; }
                    }
                    assign[i] = best_c;
                }
            };
            std::vector<std::thread> pool;
            for (uint32_t t = 0; t < n_threads; t++) pool.emplace_back(assign_worker);
            for (auto& th : pool) th.join();

            // Serial update (k is small, ~16).
            std::vector<double> sum(static_cast<size_t>(k) * dim, 0.0);
            std::vector<uint64_t> cnt(k, 0);
            for (uint64_t i = 0; i < actual_sample; i++) {
                const float* v = reservoir.data() + static_cast<size_t>(i) * dim;
                double* s = sum.data() + static_cast<size_t>(assign[i]) * dim;
                for (uint32_t dd = 0; dd < dim; dd++) s[dd] += double(v[dd]);
                cnt[assign[i]]++;
            }
            for (uint32_t c = 0; c < k; c++) {
                float* cen = entry_centroids_.data() + static_cast<size_t>(c) * dim;
                if (cnt[c] > 0) {
                    const double* s = sum.data() + static_cast<size_t>(c) * dim;
                    for (uint32_t dd = 0; dd < dim; dd++)
                        cen[dd] = float(s[dd] / double(cnt[c]));
                }
            }
        }
        spdlog::info("[sextant] pass 1: computed {} FP32 k-means centroids "
                     "for entry-point selection", k);
    }
}

// ===========================================================================
// Pass 2: encode all vectors → index_.codes_buffer
// ===========================================================================

void Builder::pass2_encode(VectorSource& source,
                          const ResolvedParams& params) {
    const uint32_t nthreads = params.num_threads > 0
                                  ? params.num_threads
                                  : std::thread::hardware_concurrency();

    spdlog::info("[sextant] pass 2: encoding {} vectors ({} threads)", index_.count,
                 nthreads);

    // Streaming encode: pull vectors one chunk at a time and parallel-encode
    // each chunk directly into index_.codes_buffer at the vector's row_id slot.
    // This avoids materializing the full dataset into a transient flat buffer
    // (peak RAM is now chunk_size × dim × 4, not N × dim × 4).
    source.reset();
    Chunk chunk{};
    uint64_t encoded = 0;

    while (source.next(chunk)) {
        const uint32_t chunk_n = chunk.count;
        if (chunk_n == 0) continue;

        // Parallel encode with atomic-counter work-stealing (same pattern as
        // parallel_construct). Each thread encodes disjoint row_ids into the
        // appropriate slot of index_.codes_buffer. encode() is const (reads only the
        // codebook, writes only its disjoint output slot) → thread-safe.
        std::atomic<uint32_t> next_r{0};
        auto worker = [this, &chunk, chunk_n, &next_r]() {
            const PqQuantizer& q = *index_.quantizer;
            uint32_t r;
            while ((r = next_r.fetch_add(1, std::memory_order_relaxed))
                   < chunk_n) {
                const RowId rid = chunk.row_ids[r];
                if (rid < 0 || static_cast<uint64_t>(rid) >= index_.count) {
                    throw Error(ErrorCode::InvalidParam,
                                "Builder::pass2: row_id out of range");
                }
                const float* vec =
                    chunk.vectors + static_cast<size_t>(r) * index_.dim;
                q.encode(vec,
                         index_.codes_buffer + static_cast<size_t>(rid) * index_.code_size);
            }
        };

        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < std::min(nthreads, chunk_n); t++) {
            pool.emplace_back(worker);
        }
        for (auto& th : pool) th.join();

        encoded += chunk_n;
    }

    spdlog::info("[sextant] pass 2: encoded {} vectors", encoded);
}

// ===========================================================================
// Parallel construct (PQ-construct, Issue 29 Mode C)
// ===========================================================================

void Builder::parallel_construct(const ResolvedParams& params) {
    const uint32_t n = static_cast<uint32_t>(index_.count);
    const uint32_t nthreads = params.num_threads > 0
                                  ? params.num_threads
                                  : std::thread::hardware_concurrency();
    const uint32_t lut_sz = index_.quantizer ? index_.quantizer->lut_size() : 0;
    construct_into(*index_.core, n,
                   [](uint32_t id) { return static_cast<RowId>(id); },
                   lut_sz, nthreads, "construct");
}

// ===========================================================================
// construct_into — the canonical parallel construct loop.
// Chunked work-stealing + T5 dynamic L_build + progress logger.
// Shared by K==1 (parallel_construct → identity mapper) and K>1 (shard build
// → membership mapper). No special cases.
// ===========================================================================

void Builder::construct_into(VamanaCore& core, uint32_t count,
                            const std::function<RowId(uint32_t)>& row_id_at,
                            uint32_t lut_sz, uint32_t nthreads,
                            const char* label) {
    if (count == 0) return;
    nthreads = std::max(1u, nthreads);
    spdlog::info("[sextant] {}: {} nodes across {} threads (PQ-construct)", label, count,
                 nthreads);

    // The very first insert must be serialized before spawning tasks: it
    // claims the entry point (see VamanaCore::insert_build_from_code).
    {
        VamanaTLS tls;
        tls.resize(count);
        tls.resize_lut(lut_sz);
        core.insert_build_from_code(0, /*row_id=*/row_id_at(0), tls);
    }

    const uint32_t lo = 1;
    const uint32_t hi = count;
    if (hi <= lo) {
        spdlog::info("[sextant] {}: only entry-point node (n=1)", label);
        return;
    }

    // Thread pool with per-thread VamanaTLS scratch.
    ctpl::thread_pool_tls<VamanaTLS> pool(
        nthreads,
        [count, lut_sz](size_t /*tid*/, std::shared_ptr<VamanaTLS>& tls) {
            tls = std::make_shared<VamanaTLS>();
            tls->resize(count);
            tls->resize_lut(lut_sz);
        });

    // Dynamic work-stealing: threads pull node IDs from a shared atomic
    // counter. Chunked stealing (kChunk=64): each fetch_add grabs a chunk of
    // IDs, reducing atomic-counter contention by ~chunk_size×.
    constexpr uint32_t kChunk = 64;
    std::atomic<uint32_t> next_id{lo};
    core.set_build_progress(&next_id);
    auto worker = [&core, &row_id_at, hi,
                   &next_id](size_t /*tid*/, VamanaTLS& tls) {
        uint32_t chunk_lo;
        while ((chunk_lo = next_id.fetch_add(kChunk, std::memory_order_relaxed)) < hi) {
            const uint32_t chunk_hi = std::min(chunk_lo + kChunk, hi);
            for (uint32_t id = chunk_lo; id < chunk_hi; id++) {
                core.insert_build_from_code(id, row_id_at(id), tls);
            }
        }
    };

    std::vector<std::future<void>> futs;
    for (uint32_t t = 0; t < nthreads; t++) {
        futs.push_back(pool.push(worker));
    }

    // Progress logger: reads the atomic counter every 5s. Zero contention.
    std::thread logger([&]() {
        const auto t_start = std::chrono::steady_clock::now();
        auto t_last = t_start;
        uint32_t last_done = lo;
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const auto now = std::chrono::steady_clock::now();
            const uint32_t done = std::min(
                next_id.load(std::memory_order_relaxed), hi);
            const double elapsed = std::chrono::duration<double>(
                now - t_start).count();
            const double interval = std::chrono::duration<double>(
                now - t_last).count();
            const uint32_t processed = done - lo;
            const uint32_t interval_processed = done - last_done;
            const double rate = processed / elapsed;
            const uint32_t remaining = (hi - lo) - processed;
            const double eta = rate > 0 ? remaining / rate : 0;
            spdlog::info("[sextant] {}: {}/{} nodes ({:.0f}/s, ETA {:.0f}s)",
                         label, processed + 1, hi - lo,
                         interval_processed / interval, eta);
            if (done >= hi) break;
            last_done = done;
            t_last = now;
        }
    });

    for (auto& f : futs) {
        f.get();
    }
    logger.join();
    core.set_build_progress(nullptr);

    spdlog::info("[sextant] {}: all {} nodes inserted", label, count);
}

// ===========================================================================
// Partitioned build (Step 11)
//
// Phases:
//   1. Global PQ train + encode all vectors → index_.codes_buffer (global codebook).
//   2. K-means on PQ codes → K shards with closure_factor overlap.
//   3. Per-shard build: each shard builds at R_shard = 2R/3, referencing a
//      contiguous copy of its members' PQ codes. Nodes store GLOBAL row_ids.
//   4. Merge: union neighbor lists per global node, remap shard-local IDs →
//      global IDs, truncate to R by PQ code distance.
//   5. Flush: standard write_sidecars_ (merged graph + global codes).
// ===========================================================================

BuildResult Builder::build_partitioned(VectorSource& source,
                                       const std::string& index_path,
                                       const ResolvedParams& params) {
    using engine_detail::write_padded;
    using engine_detail::fill_header;
    using engine_detail::read_exact;

    // Partitioned build uses PQ-construct shard exclusively — RAM savings
    // matter more than marginal quality at large scale.

    spdlog::info("[sextant] build: N={} K={} closure_factor={:.4f}",
                 index_.count, params.partition_count, params.closure_factor);

    // --- 1. Quantizer (global): train via pass1 (resolves pq_bits if auto) ---
    // pass1 fills the reservoir, runs the global probe if pq_bits==0, then
    // constructs + trains the quantizer. After it, index_.code_size is valid.
    // prepare_codes also runs pass2_encode and loads raw_vecs_buffer (FP16).
    prepare_codes(source, params);

    // node_size for the build buffer (flat neighbor lists).
    // Shards build at R_shard, but the merged result lands in index_.nodes_buffer
    // at full R.
    index_.node_size = VamanaCore::static_node_size(params.R, index_.code_size);

    // Allocate the flat nodes buffer (codes + raw_vecs already populated by
    // prepare_codes). nodes_buffer is construct-only — the merged graph
    // lands here at full R, then gets flushed.
    {
        const size_t nodes_bytes = static_cast<size_t>(index_.count) * index_.node_size;
        AlignedBuf nodes(kDiskAlign, nodes_bytes);
        std::memset(nodes.get(), 0, nodes_bytes);
#ifdef __linux__
        // Same THP hint as the single-partition build path (see Builder::build).
        if (nodes_bytes > 0 && madvise(nodes.get(), nodes_bytes,
                                       MADV_HUGEPAGE) != 0) {
            spdlog::debug("[sextant] huge pages unavailable for nodes buffer "
                          "(partitioned), using standard pages");
        }
#endif
        index_.nodes_buffer = nodes.as<uint8_t>(); nodes.release();
    }

    const uint32_t n = static_cast<uint32_t>(index_.count);

    // Load raw vectors as FP16 for the FP16 prune. Used by BOTH K=1 (full
    // graph) and K>1 (per-shard copy). They enable the FP16 prune (exact FP16
    // L2sq occlusion check instead of PQ code_distance). Stored as FP16 (half
    // the RAM of FP32) and consumed directly by simd::l2sq_f16 — no per-call
    // conversion.
    {
        const size_t vecs_bytes =
            static_cast<size_t>(index_.count) * index_.dim * sizeof(float16_t);
        AlignedBuf vecs(kDiskAlign, vecs_bytes);
        spdlog::info("[sextant] loading raw vectors as FP16 ({:.1f}MB) for "
                      "PQ-construct (FP16 prune)",
                     vecs_bytes / 1e6);
        // FP32 build mode: also load FP32 vectors for exact build distances.
        // +N×dim×4 bytes RAM (e.g. +4.1GB at 1.34M). Gated by env var.
        static const bool fp32_build = []() {
            const char* e = std::getenv("SEXTANT_FP32_BUILD");
            return e && e[0] == '1';
        }();
        AlignedBuf fp32_vecs;
        if (fp32_build) {
            const size_t fp32_bytes =
                static_cast<size_t>(index_.count) * index_.dim * sizeof(float);
            fp32_vecs = AlignedBuf(kDiskAlign, fp32_bytes);
            spdlog::info("[sextant] FP32 build mode: loading raw FP32 vectors "
                         "({:.1f}MB) for exact build distances",
                         fp32_bytes / 1e6);
        }
        source.reset();
        Chunk chunk{};
        uint64_t loaded = 0;
        while (source.next(chunk)) {
            for (uint32_t r = 0; r < chunk.count; r++) {
                const RowId rid = chunk.row_ids[r];
                if (rid >= 0 && static_cast<uint64_t>(rid) < index_.count) {
                    float16_t* dst = vecs.as<float16_t>() +
                                  static_cast<size_t>(rid) * index_.dim;
                    const float* src = chunk.vectors +
                                       static_cast<size_t>(r) * index_.dim;
                    cast_fp32_to_fp16(src, dst, index_.dim);
                    if (fp32_build) {
                        std::memcpy(fp32_vecs.as<float>() +
                                        static_cast<size_t>(rid) * index_.dim,
                                    src, index_.dim * sizeof(float));
                    }
                    loaded++;
                }
            }
        }
        spdlog::info("[sextant] loaded {} raw vectors as FP16", loaded);
        index_.raw_vecs_buffer = vecs.as<float16_t>(); vecs.release();
        if (fp32_build) {
            index_.fp32_vecs_buffer = fp32_vecs.as<float>(); fp32_vecs.release();
        }
    }

    // =====================================================================
    // K==1 fast path: build the full graph directly into the global buffers.
    // This is the unified monolithic path — K=1 is a special case of the
    // partitioned path with a single identity shard. It reuses parallel_construct
    // (chunked work-stealing + T5 dynamic L_build progress logger) so the graph
    // is bit-identical to the former standalone monolithic build.
    // =====================================================================
    if (params.partition_count == 1) {
        // K==1 full-graph build core (flat build layout).
        VamanaParams vp_full =
            VamanaParams::from_resolved(params, index_.dim);

        index_.core = std::make_unique<VamanaCore>(vp_full, *index_.quantizer);
        index_.core->set_build_codes(index_.codes_buffer, n);
        index_.core->set_build_nodes(index_.nodes_buffer);
        index_.core->prepare_for_build(n);

        // FlatNodeStore over the flat buffers so beam_search (used in
        // insert_build_from_code) goes through the store interface.
        index_.flat_store = std::make_unique<FlatNodeStore>(
            index_.nodes_buffer, index_.codes_buffer, index_.node_size, index_.code_size);
        index_.core->set_store(index_.flat_store.get());

        index_.core->set_build_vecs(index_.raw_vecs_buffer);
        if (index_.fp32_vecs_buffer) {
            index_.core->set_build_fp32_vecs(index_.fp32_vecs_buffer);
        }
        VamanaCore::BuildVecLoan build_vec_loan(*index_.core);

        parallel_construct(params);
        // Raw vectors cleared by build_vec_loan destructor (K=1 no longer needs them).
    } else {
    // =====================================================================
    // K>1 partitioned path: partition → per-shard build (R_shard=2R/3) → merge.
    // =====================================================================

    // --- 2. Partition via k-means on PQ codes ---
    auto assignment = partition_codes(*index_.quantizer, index_.codes_buffer, n,
                                      index_.code_size, params.partition_count,
                                      params.closure_factor, /*iterations=*/10,
                                      params.num_threads,
                                      /*seed=*/0xC0DE1234ULL,
                                      /*balance_factor=*/0.0f,
                                      /*closure_epsilon=*/params.closure_epsilon);
    const uint32_t K = static_cast<uint32_t>(assignment.shards.size());

    // --- 3. Per-shard build ---
    // Each shard: contiguous codes buffer (copy of members' global codes),
    // VamanaCore at R_shard, parallel construct. Nodes store row_id = global ID.
    const uint16_t R_shard = static_cast<uint16_t>(
        std::max<uint32_t>(8u, (2u * params.R) / 3u));
    spdlog::info("[sextant] partitioned: R_shard={} (2R/3, R={})", R_shard,
                 params.R);

    const uint32_t shard_node_size =
        VamanaCore::static_node_size(R_shard, index_.code_size);

    // Store each shard's node buffer + local→global map for the merge.
    std::vector<std::vector<uint8_t>> shard_node_bufs(K);
    std::vector<std::vector<uint32_t>> shard_local_to_global(K);

    // Build a single VamanaParams template; R/L per shard.
    // R_shard is the per-shard degree (2R/3); flat build layout.
    VamanaParams vp_shard =
        VamanaParams::from_resolved(params, index_.dim, R_shard);

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
                          (index_.code_size + shard_node_size +
                           static_cast<size_t>(index_.dim) * sizeof(float16_t))) /
                         1e6,
                     params.num_threads);

        // Contiguous shard codes: local index i → members[i]'s global code.
        std::vector<uint8_t> shard_codes(
            static_cast<size_t>(shard_n) * index_.code_size, 0);
        // Contiguous shard FP16 vectors: same mapping, for the FP16 prune.
        std::vector<float16_t> shard_vecs(
            static_cast<size_t>(shard_n) * index_.dim, 0);
        shard_local_to_global[k].resize(shard_n);
        for (uint32_t i = 0; i < shard_n; i++) {
            const uint32_t gid = members[i];
            shard_local_to_global[k][i] = gid;
            std::memcpy(shard_codes.data() +
                            static_cast<size_t>(i) * index_.code_size,
                        index_.codes_buffer + static_cast<size_t>(gid) * index_.code_size,
                        index_.code_size);
            std::memcpy(shard_vecs.data() +
                            static_cast<size_t>(i) * index_.dim,
                        index_.raw_vecs_buffer + static_cast<size_t>(gid) * index_.dim,
                        index_.dim * sizeof(float16_t));
        }

        // Shard node buffer (aligned for direct-IO reuse).
        const size_t shard_nodes_bytes =
            static_cast<size_t>(shard_n) * shard_node_size;
        shard_node_bufs[k].resize(shard_nodes_bytes, 0);
        uint8_t* shard_nodes = shard_node_bufs[k].data();
        uint8_t* shard_codes_ptr = shard_codes.data();

        // Build a fresh VamanaCore for this shard.
        VamanaCore core(vp_shard, *index_.quantizer);
        core.set_build_codes(shard_codes_ptr, shard_n);
        core.set_build_nodes(shard_nodes);
        core.set_build_vecs(shard_vecs.data());
        core.prepare_for_build(shard_n);

        // Parallel construct via the canonical loop (chunked work-stealing,
        // T5 dynamic L_build, progress logger). Row IDs remapped to global.
        const uint32_t shard_lut_sz = index_.quantizer ? index_.quantizer->lut_size() : 0;
        const uint32_t nthreads = params.num_threads > 0
                                      ? params.num_threads
                                      : std::thread::hardware_concurrency();
        const std::string shard_label = "shard " + std::to_string(k) + "/" + std::to_string(K);
        construct_into(
            core, shard_n,
            [&members](uint32_t local_id) { return static_cast<RowId>(members[local_id]); },
            shard_lut_sz, nthreads, shard_label.c_str());
        // Shard built; node buffer retained in shard_node_bufs[k].
    }

    // --- 4. Merge ---
    // For each global node, collect neighbor lists from all shards that
    // contain it (remapping shard-local IDs → global IDs), union, dedup,
    // truncate to R by PQ distance.
    // Merge is a three-phase streaming pass:
    //   (a) Gather each global node's candidates from its shard neighbor lists
    //       (shard-local IDs remapped to global IDs).
    //   (b) Add reciprocal edges: if A→B is a candidate, B→A becomes one too.
    //       This makes the merged graph undirected and guarantees connectivity
    //       (each shard's Vamana build is internally connected via the shared
    //       entry-point seed, and closure_factor overlap bridges shards).
    //   (c) For each node: dedup + truncate to R by PQ distance, write out.
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

    // (c) Dedup + truncate to R by PQ distance, write to index_.nodes_buffer.
    std::vector<uint32_t> seen(n, 0);
    uint32_t visit_token = 0;
    for (uint32_t gid = 0; gid < n; gid++) {
        uint8_t* out_node =
            index_.nodes_buffer + static_cast<size_t>(gid) * index_.node_size;
        std::memset(out_node, 0,
                    kNeighborArrayOffset +
                        static_cast<size_t>(params.R) * sizeof(uint32_t));
        VamanaCore::set_row_id(out_node, static_cast<RowId>(gid));
        VamanaCore::set_internal_id(out_node, gid);
        VamanaCore::set_neighbor_count(out_node, 0);

        ++visit_token;
         std::vector<std::pair<float, uint32_t>> cands;
         cands.reserve(adj[gid].size());
         const float16_t* my_vec =
             index_.raw_vecs_buffer + static_cast<size_t>(gid) * index_.dim;
         for (uint32_t gnb : adj[gid]) {
             if (gnb == gid) continue;
             if (seen[gnb] == visit_token) continue;  // dedup
             seen[gnb] = visit_token;
             const float16_t* nb_vec =
                 index_.raw_vecs_buffer + static_cast<size_t>(gnb) * index_.dim;
             const float d = index_.raw_vecs_buffer
                 ?simd::l2sq_f16(my_vec, nb_vec, index_.dim)
                 : index_.quantizer->code_distance(
                       index_.codes_buffer + static_cast<size_t>(gid) * index_.code_size,
                       index_.codes_buffer + static_cast<size_t>(gnb) * index_.code_size);
             cands.emplace_back(d, gnb);
         }
        // Free the adjacency now that we've consumed it.
        std::vector<uint32_t>().swap(adj[gid]);

        // Truncate to R: keep the R closest by PQ distance.
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
    // the PQ-closest pair. This guarantees a single connected component.
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
                index_.nodes_buffer + static_cast<size_t>(a) * index_.node_size;
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

            // For each minor component, bridge to the root via the PQ-nearest
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
                 // Find nearest (comp_member, root_sample) pair by FP16 L2sq.
                 float best_d = std::numeric_limits<float>::max();
                 uint32_t best_c = members[0];
                 uint32_t best_r = root_sample[0];
                 for (uint32_t c : members) {
                     const float16_t* cv =
                         index_.raw_vecs_buffer + static_cast<size_t>(c) * index_.dim;
                     for (uint32_t r : root_sample) {
                         const float16_t* rv =
                             index_.raw_vecs_buffer + static_cast<size_t>(r) * index_.dim;
                         const float d = index_.raw_vecs_buffer
                             ?simd::l2sq_f16(cv, rv, index_.dim)
                             : index_.quantizer->code_distance(
                                 index_.codes_buffer + static_cast<size_t>(c) * index_.code_size,
                                 index_.codes_buffer + static_cast<size_t>(r) * index_.code_size);
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
                    uint8_t* node = index_.nodes_buffer +
                                    static_cast<size_t>(from) * index_.node_size;
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
                index_.nodes_buffer + static_cast<size_t>(gid) * index_.node_size;
            const uint16_t d = VamanaCore::get_neighbor_count(node);
            if (d == 0) zero_deg++;
            total_edges += d;
            if (d > max_deg) max_deg = d;
        }
        spdlog::info("[sextant] merge degrees: zero={}, avg={:.1f}, max={}",
                     zero_deg, static_cast<double>(total_edges) / n, max_deg);
    }

    spdlog::info("[sextant] merge complete; flushing sidecars");

    // Set up the master VamanaCore (full R) over the merged index_.nodes_buffer for
    // entry-point computation during flush. (K==1 already has index_.core set up
    // over the global buffers in the fast path above.)
    VamanaParams vp_full =
        VamanaParams::from_resolved(params, index_.dim);
    index_.core = std::make_unique<VamanaCore>(vp_full, *index_.quantizer);
    index_.core->set_build_codes(index_.codes_buffer, n);
    index_.core->set_build_nodes(index_.nodes_buffer);
    index_.core->prepare_for_build(n);
    }  // end K>1 partitioned branch

    // --- Flush (entry points + final-layout inlining + sidecars) ---
    // Shared by both K==1 (graph built directly) and K>1 (merged graph). For
    // K==1, index_.core is already wired to index_.nodes_buffer/index_.codes_buffer.
    snap_entry_points_(params);
    auto bfs = compute_bfs_reorder_(params);
    write_sidecars_(index_.path, bfs, params);

    // Post-flush: switch from flat build buffers to a PagedNodeStore so
    // post-build search is SSD-resident. The sidecars now hold the final
    // graph. This runs for ALL K — the partitioned path previously omitted
    // it (left flat buffers resident and no paged store), which was an
    // asymmetry vs the monolithic path.
    const uint32_t final_node_size =
        VamanaCore::static_node_size(params.R, index_.code_size);
    index_.node_size = final_node_size;
    index_.flat_store.reset();  // disconnect store before freeing buffers
    index_.core->set_store(nullptr);
    if (index_.codes_buffer) { aligned_free(index_.codes_buffer); index_.codes_buffer = nullptr; }
    if (index_.nodes_buffer) { aligned_free(index_.nodes_buffer); index_.nodes_buffer = nullptr; }
    if (index_.raw_vecs_buffer) { aligned_free(index_.raw_vecs_buffer); index_.raw_vecs_buffer = nullptr; }

    index_.paged_store = std::make_unique<PagedNodeStore>(
        index_.path + ".graph", index_.path + ".codes",
        final_node_size, index_.code_size,
        std::max(1u, std::thread::hardware_concurrency()),
        /*cache_size_bytes=*/64ull * 1024 * 1024);  // 64MB default cache
    index_.core->set_store(index_.paged_store.get());

    // (Builder doesn't track an opened_ flag; the Engine wrapper does.)

    BuildResult result;
    result.index_path = index_path;
    result.n_vectors = index_.count;
    result.dim = index_.dim;
    result.R = params.R;
    result.L_build = params.L_build;
    result.pq_m = index_.quantizer ? index_.quantizer->m() : params.pq_m;
    result.pq_bits = index_.quantizer ? index_.quantizer->bits() : params.pq_bits;
    return result;
}

// ===========================================================================
// compute_bfs_reorder_ — pure BFS reorder of build IDs → disk positions.
// write_sidecars_  — streams all four sidecars using the precomputed reorder.
// ===========================================================================

Builder::BfsReorder Builder::compute_bfs_reorder_(const ResolvedParams& params) const {
    const uint32_t n = static_cast<uint32_t>(index_.count);
    const uint32_t build_node_size = index_.node_size;  // flat build layout
    const auto& raw_entry_points = index_.core->entry_points();
    BfsReorder bfs;
    bfs.order = compute_bfs_order(index_.nodes_buffer, n, build_node_size, raw_entry_points);
    bfs.remap.resize(n);
    for (uint32_t new_pos = 0; new_pos < n; new_pos++) {
        bfs.remap[bfs.order[new_pos]] = new_pos;
    }
     return bfs;
 }

void Builder::snap_entry_points_(const ResolvedParams& params) {
    // Snap stored FP32 centroids to nearest data vectors (medoids) and set
    // them as entry points. Falls back to stride sampling if no centroids.
    if (entry_centroids_.empty() || !index_.raw_vecs_buffer || index_.count == 0) {
        index_.core->compute_entry_points();  // stride-sampled fallback
        return;
    }
    const uint32_t n = static_cast<uint32_t>(index_.count);
    const uint32_t k = static_cast<uint32_t>(entry_centroids_.size() / index_.dim);
    // Parallel medoid search: each centroid scans the full FP16 dataset
    // for its nearest data vector. O(k × N × dim) but embarrassingly
    // parallel across centroids.
    std::vector<uint32_t> medoid_ids(k);
    std::vector<std::thread> pool;
        auto worker = [&](uint32_t c) {
            const float* centroid = entry_centroids_.data() + static_cast<size_t>(c) * index_.dim;
            // Convert centroid to FP16 for simd::l2sq_f16 comparison.
            std::vector<float16_t> centroid_f16(index_.dim);
            cast_fp32_to_fp16(centroid, centroid_f16.data(), index_.dim);
        float best_d = std::numeric_limits<float>::infinity();
        uint32_t best_id = 0;
        for (uint32_t i = 0; i < n; i++) {
            const float d = simd::l2sq_f16(centroid_f16.data(),
                                     index_.raw_vecs_buffer + static_cast<size_t>(i) * index_.dim,
                                     index_.dim);
            if (d < best_d) { best_d = d; best_id = i; }
        }
        medoid_ids[c] = best_id;
    };
    for (uint32_t c = 0; c < k; c++) pool.emplace_back(worker, c);
    for (auto& t : pool) t.join();
    // Dedup (two centroids might snap to the same medoid).
    std::sort(medoid_ids.begin(), medoid_ids.end());
    medoid_ids.erase(std::unique(medoid_ids.begin(), medoid_ids.end()),
                     medoid_ids.end());
    // Top up with stride-sampled fallbacks if dedup reduced the count.
    while (medoid_ids.size() < k) {
        uint32_t id = static_cast<uint32_t>(medoid_ids.size() * n) / k;
        if (std::find(medoid_ids.begin(), medoid_ids.end(), id) == medoid_ids.end())
            medoid_ids.push_back(id);
        else break;
    }
    index_.core->set_entry_points(std::move(medoid_ids));
    spdlog::info("[sextant] entry points: {} k-means medoids (FP32 centroids, "
                 "FP16 snap)", index_.core->entry_points().size());
}
void Builder::write_sidecars_(const std::string& index_path,
                             const BfsReorder& bfs,
                             const ResolvedParams& params) {
    const auto uuid = make_uuid();
    const uint32_t n = static_cast<uint32_t>(index_.count);

    // Final node layout: same as build layout (flat neighbor lists, single
    // node_size for the whole index).
    const uint32_t final_node_size =
        VamanaCore::static_node_size(params.R, index_.code_size);
    const uint32_t build_node_size = index_.node_size;

    // ----- .codes (reordered to BFS order, STREAMED) -----
    {
        const std::string path = index_path + ".codes";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicCodes, index_.count, index_.dim, uuid);
        write_padded(f, &h, sizeof(h), 0);

        const size_t codes_bytes = static_cast<size_t>(n) * index_.code_size;
        // Stream the reorder through a block-aligned ring buffer (256KB) instead
        // of materializing the full N×code_size in RAM. At 1B×96B the old path
        // allocated 96GB transiently here.
        //
        // Chunk sizing: each FULL chunk must hold a kDiskAlign-multiple of bytes
        // so write_padded emits it verbatim (no interior zero-padding). Only the
        // final partial chunk is padded — matching the former single-write tail,
        // which is what keeps the on-disk layout byte-identical to the old path.
        const size_t block_cap = kBlockSize;  // 256KB
        const uint32_t align_step =
            kDiskAlign / std::gcd(kDiskAlign, index_.code_size);
        uint32_t codes_per_block =
            static_cast<uint32_t>(block_cap / index_.code_size);
        codes_per_block -= codes_per_block % align_step;  // round down
        codes_per_block = std::max<uint32_t>(codes_per_block, align_step);
        const size_t buf_cap = static_cast<size_t>(codes_per_block) * index_.code_size;
        AlignedBuf ring(kDiskAlign, buf_cap);

        uint64_t write_off = sizeof(h);
        uint32_t in_block = 0;
        for (uint32_t new_pos = 0; new_pos < n; new_pos++) {
            const uint32_t old_id = bfs.order[new_pos];
            std::memcpy(ring.as<uint8_t>() + static_cast<size_t>(in_block) * index_.code_size,
                        index_.codes_buffer + static_cast<size_t>(old_id) * index_.code_size,
                        index_.code_size);
            if (++in_block >= codes_per_block) {
                write_padded(f, ring.get(),
                             static_cast<size_t>(in_block) * index_.code_size, write_off);
                write_off += static_cast<size_t>(in_block) * index_.code_size;
                in_block = 0;
            }
        }
        if (in_block > 0) {
            write_padded(f, ring.get(),
                         static_cast<size_t>(in_block) * index_.code_size, write_off);
        }
        f.sync();
        spdlog::info("[sextant] wrote {} ({} bytes, BFS-reordered, streamed)",
                     path, codes_bytes);
    }

    // ----- .graph (reformat to final layout, inline neighbor PQ codes) -----
    {
        const std::string path = index_path + ".graph";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicGraph, index_.count, index_.dim, uuid);
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
        AlignedBuf ring(kDiskAlign, buf_cap);

        const uint32_t neighbor_region_end =
            kNeighborArrayOffset + params.R * sizeof(uint32_t);

        uint64_t write_off = sizeof(h);
        uint32_t in_block = 0;
        for (uint32_t new_pos = 0; new_pos < n; new_pos++) {
            // PageShuffle: emit nodes in BFS order. The node at disk position
            // new_pos is the build node whose old_id = bfs.order[new_pos].
            const uint32_t old_id = bfs.order[new_pos];
            const uint8_t* src = index_.nodes_buffer +
                                 static_cast<size_t>(old_id) * build_node_size;
            uint8_t* dst = ring.as<uint8_t>() + static_cast<size_t>(in_block) * final_node_size;

            // Copy the fixed header + neighbor array (identical in both layouts
            // since the build and final layouts share the same R).
            const uint32_t copy_len = (neighbor_region_end + 7u) & ~7u;
            std::memcpy(dst, src, copy_len);

            // PageShuffle: remap this node's internal_id and every neighbor ID
            // from build (old) IDs to BFS (new) IDs.
            VamanaCore::set_internal_id(dst, new_pos);
            const uint16_t ndeg = VamanaCore::get_neighbor_count(dst);
            for (uint16_t i = 0; i < ndeg; i++) {
                const uint32_t old_nb = VamanaCore::get_neighbor(dst, i);
                if (old_nb < n) {
                    VamanaCore::set_neighbor(dst, i, bfs.remap[old_nb]);
                }
            }


            in_block++;
            if (in_block >= per_block) {
                write_padded(f, ring.get(),
                             static_cast<size_t>(in_block) * final_node_size,
                             write_off);
                write_off += static_cast<size_t>(in_block) * final_node_size;
                in_block = 0;
            }
        }
        if (in_block > 0) {
            write_padded(f, ring.get(),
                         static_cast<size_t>(in_block) * final_node_size,
                         write_off);
        }
        f.sync();
        spdlog::info("[sextant] wrote {} ({} nodes, {} bytes/node)", path, n,
                     final_node_size);
    }

    // ----- .meta (serialized quantizer + entry points + params) -----
    // PageShuffle: remap entry points from build (old) IDs to BFS (new) IDs
    // so the search path starts at the correct disk positions.
    {
        const auto& eps = index_.core->entry_points();
        std::vector<uint32_t> remapped_eps;
        remapped_eps.reserve(eps.size());
        for (uint32_t ep : eps) {
            remapped_eps.push_back(ep < n ? bfs.remap[ep] : ep);
        }
        write_meta_file(params, remapped_eps, uuid);
    }

    // ----- .ball sidecar (FP16 vectors for the entry-point ball) — NOT WRITTEN -----
    // The FP16 ball tier was retired (it caused premature search convergence
    // and lower QPS — see results/p2.3-noball/). The .ball sidecar is no
    // longer loaded at search, so we don't write it. This saves I/O + disk
    // space at build time (~100MB at 1.34M for the 3-hop neighborhood × dim
    // × 2 bytes). The raw_vecs_buffer (all vectors as FP16 in RAM) is still
    // loaded for the build-time FP16 prune in robust_prune_into — separate.

    // ----- .manifest (atomic commit — written LAST via temp + rename) -----
    write_manifest_file(params, uuid);
}

// ===========================================================================
// write_meta_file / write_manifest_file — shared .meta and .manifest writers.
//
// Both write_sidecars_ (build path) and flush (post-insert path) emit the same
// .meta payload (serialized quantizer + entry points + ResolvedParams) and the
// same .manifest commit point. The only caller-specific detail is the entry-
// point vector: write_sidecars_ BFS-remaps build IDs to disk positions, while
// flush passes index_.core->entry_points() verbatim (buffers are already final).
// Callers prepare the entry-point vector and pass it in.
// ===========================================================================

void Builder::write_meta_file(const ResolvedParams& params,
                              const std::vector<uint32_t>& entry_points,
                              const std::pair<uint64_t, uint64_t>& uuid) {
    const std::string path = index_.path + ".meta";
    DirectFile f(path, true);
    SidecarHeader h;
    fill_header(h, kMagicMeta, index_.count, index_.dim, uuid);
    write_padded(f, &h, sizeof(h), 0);

    // Serialize the quantizer.
    std::vector<uint8_t> qblob;
    index_.quantizer->serialize(qblob);
    uint64_t qsize = qblob.size();
    // Payload layout: [u64 index_.quantizersize][index_.quantizerbytes]
    //                 [u16 entry_point_count][entry_point_count × u32]
    //                 [ResolvedParams POD block]
    std::vector<uint8_t> payload;
    payload.insert(payload.end(),
                   reinterpret_cast<uint8_t*>(&qsize),
                   reinterpret_cast<uint8_t*>(&qsize) + sizeof(qsize));
    payload.insert(payload.end(), qblob.begin(), qblob.end());

    uint16_t ep_count = static_cast<uint16_t>(entry_points.size());
    payload.insert(payload.end(),
                   reinterpret_cast<uint8_t*>(&ep_count),
                   reinterpret_cast<uint8_t*>(&ep_count) + sizeof(ep_count));
    for (uint32_t ep : entry_points) {
        payload.insert(payload.end(),
                       reinterpret_cast<uint8_t*>(&ep),
                       reinterpret_cast<uint8_t*>(&ep) + sizeof(ep));
    }

    // Append the resolved params as a POD block so open() can rebuild the
    // VamanaCore with the same R/L/alpha/max_occlusion.
    ResolvedParams p = params;  // copy
    payload.insert(payload.end(),
                   reinterpret_cast<uint8_t*>(&p),
                   reinterpret_cast<uint8_t*>(&p) + sizeof(p));

    write_padded(f, payload.data(), payload.size(), sizeof(h));
    f.sync();
    spdlog::info("[sextant] wrote {} ({} bytes payload, {} entry points)",
                 path, payload.size(), entry_points.size());
}

void Builder::write_manifest_file(const ResolvedParams& params,
                                  const std::pair<uint64_t, uint64_t>& uuid) {
    const std::string path = index_.path + ".manifest";
    GraphManifest m;
    m.n_vectors = index_.count;
    m.dim = index_.dim;
    m.R = params.R;
    m.pq_m = params.pq_m;
    write_manifest_atomic(path, graph_manifest_to_toml(m));
    spdlog::info("[sextant] wrote {} (commit point)", path);
}

// ===========================================================================
// insert / flush
//
// Live insert = "build one node". We grow the flat codes/nodes buffers by one
// slot, encode the vector, then drive the Vamana insert flow (beam_search →
// robust_prune → connect_and_prune) via VamanaCore::insert_build_from_code.
//
// PHASE 1: insert requires mutable flat buffers. After open() the engine is
// in SSD-resident (paged) mode; the first insert materializes the flat
// buffers from the sidecar files (O(N) I/O), then switches the core to
// FlatNodeStore. This is acceptable because insert is NOT the hot path.
// The realloc strategy is O(N) per insert (copy the whole buffers).
// ===========================================================================

void Builder::insert(const float* vec, Dim dim, RowId row_id) {
    if (!index_.quantizer || !index_.core) {
        throw Error(ErrorCode::InvalidParam,
                    "Builder::insert: index not built (quantizer/core missing)");
    }
    if (vec == nullptr) {
        throw Error(ErrorCode::InvalidParam,
                    "Builder::insert: null vector");
    }
    if (dim != index_.dim) {
        throw Error(ErrorCode::InvalidParam,
                    "Builder::insert: dim mismatch");
    }
    if (index_.code_size == 0 || index_.node_size == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "Builder::insert: code/node size not initialized");
    }

    // Phase 1: insert requires mutable flat buffers. If we're in paged
    // (SSD-resident) mode after open(), materialize the flat buffers from the
    // sidecar files on first insert. The FlatNodeStore is (re)created after the
    // reallocs below so it always points at the live buffers.
    // This is O(N) I/O but acceptable because insert is NOT the hot path.
    if (index_.paged_store && !index_.codes_buffer) {
        spdlog::info("[sextant] insert: materializing flat buffers from sidecars "
                     "for mutable insert");
        // Codes.
        {
            const std::string path = index_.path + ".codes";
            DirectFile f(path, false);
            const size_t codes_bytes =
                static_cast<size_t>(index_.count) * index_.code_size;
            AlignedBuf buf(kDiskAlign, codes_bytes);
            std::memset(buf.get(), 0, codes_bytes);
            read_exact(f, buf.get(), codes_bytes, sizeof(SidecarHeader));
            index_.codes_buffer = buf.as<uint8_t>();
            buf.release();
        }
        // Nodes.
        {
            const std::string path = index_.path + ".graph";
            DirectFile f(path, false);
            const size_t nodes_bytes =
                static_cast<size_t>(index_.count) * index_.node_size;
            AlignedBuf buf(kDiskAlign, nodes_bytes);
            std::memset(buf.get(), 0, nodes_bytes);
            read_exact(f, buf.get(), nodes_bytes, sizeof(SidecarHeader));
            index_.nodes_buffer = buf.as<uint8_t>();
            buf.release();
        }
        index_.paged_store.reset();
        index_.core->set_store(nullptr);  // cleared; recreated below
    }

    const uint32_t new_internal = static_cast<uint32_t>(index_.count);
    const uint64_t new_count = index_.count + 1;

    // --- 1. Grow the codes buffer by one code (O(N) copy). ---
    {
        const size_t old_bytes = static_cast<size_t>(index_.count) * index_.code_size;
        const size_t new_bytes = static_cast<size_t>(new_count) * index_.code_size;
        const size_t alloc_bytes = (new_bytes + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
        AlignedBuf nb(kDiskAlign, alloc_bytes);
        std::memcpy(nb.get(), index_.codes_buffer, old_bytes);
        // Encode the new vector into the appended slot.
        index_.quantizer->encode(vec, nb.as<uint8_t>() + old_bytes);
        aligned_free(index_.codes_buffer);
        index_.codes_buffer = nb.as<uint8_t>();
        nb.release();
    }

    // --- 2. Grow the nodes buffer by one node (O(N) copy). ---
    {
        const size_t old_bytes = static_cast<size_t>(index_.count) * index_.node_size;
        const size_t new_bytes = static_cast<size_t>(new_count) * index_.node_size;
        const size_t alloc_bytes = (new_bytes + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
        AlignedBuf nb(kDiskAlign, alloc_bytes);
        std::memcpy(nb.get(), index_.nodes_buffer, old_bytes);
        std::memset(nb.as<uint8_t>() + old_bytes, 0, index_.node_size);
        aligned_free(index_.nodes_buffer);
        index_.nodes_buffer = nb.as<uint8_t>();
        nb.release();
    }

    // --- 3. Publish the grown buffers + new count to the core. ---
    index_.count = new_count;
    index_.core->set_build_codes(index_.codes_buffer, static_cast<uint32_t>(index_.count));
    index_.core->set_build_nodes(index_.nodes_buffer);

    // (Re)create the FlatNodeStore over the (possibly realloc'd) live buffers
    // so beam_search inside insert_build_from_code reads current data.
    index_.flat_store = std::make_unique<FlatNodeStore>(
        index_.nodes_buffer, index_.codes_buffer,
        index_.node_size, index_.code_size);
    index_.core->set_store(index_.flat_store.get());

    // --- 3b. Grow index_.raw_vecs_buffer by one FP16 vector (O(N) copy). ---
    // Needed so build_vec_ptr(new_internal) works for the FP16 prune.
    {
        const size_t old_bytes = static_cast<size_t>(index_.count - 1) * index_.dim * sizeof(float16_t);
        const size_t new_bytes = static_cast<size_t>(index_.count) * index_.dim * sizeof(float16_t);
        const size_t alloc_bytes = (new_bytes + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
        AlignedBuf nb(kDiskAlign, alloc_bytes);
        if (index_.raw_vecs_buffer) {
            std::memcpy(nb.get(), index_.raw_vecs_buffer, old_bytes);
            aligned_free(index_.raw_vecs_buffer);
        }
        float16_t* dst = nb.as<float16_t>() + static_cast<size_t>(new_internal) * index_.dim;
        cast_fp32_to_fp16(vec, dst, index_.dim);
        index_.raw_vecs_buffer = nb.as<float16_t>();
        nb.release();
    }
    index_.core->set_build_vecs(index_.raw_vecs_buffer);

    // --- 4. Drive the Vamana insert flow (single-thread). ---
    //    insert_build_from_code handles the first-node case (becomes an entry
    //    point) and the general case (beam_search → robust_prune →
    //    connect_and_prune). It also bumps index_.core->index_.count to internal_id + 1.
    //    build_vecs_ is set above so the prune uses FP16 L2sq.
    VamanaTLS tls;
    tls.resize(static_cast<uint32_t>(index_.count));
    tls.resize_lut(index_.quantizer ? index_.quantizer->lut_size() : 0);
    index_.core->insert_build_from_code(new_internal, row_id, tls);

    spdlog::info("[sextant] insert: row_id={} internal_id={} (count now {})",
                 row_id, new_internal, index_.count);
}

void Builder::flush() {
    // After build(), all sidecars are already written synchronously.
    // flush() is meaningful after a build or after insert(): it rewrites the
    // sidecars with the (possibly grown) buffers. No-op if nothing to write.
    if (!index_.quantizer || !index_.core) {
        return;  // index not built — nothing to flush
    }
    if (index_.count == 0) {
        return;
    }
    if (!index_.quantizer || !index_.core) {
        throw Error(ErrorCode::InvalidParam,
                    "Builder::flush: quantizer/core not initialized");
    }

    const auto uuid = make_uuid();
    const ResolvedParams& params = index_.params;
    const uint32_t n = static_cast<uint32_t>(index_.count);

    // After open(), index_.nodes_buffer is already in final layout (index_.node_size
    // single node_size layout), so we write it verbatim — no reformat pass.
    spdlog::info("[sextant] flush: persisting {} vectors to '{}'", n,
                 index_.path);

    // ----- .codes -----
    {
        const std::string path = index_.path + ".codes";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicCodes, index_.count, index_.dim, uuid);
        write_padded(f, &h, sizeof(h), 0);
        const size_t codes_bytes = static_cast<size_t>(n) * index_.code_size;
        write_padded(f, index_.codes_buffer, codes_bytes, sizeof(h));
        f.sync();
    }

    // ----- .graph (verbatim — buffers already in final layout) -----
    {
        const std::string path = index_.path + ".graph";
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicGraph, index_.count, index_.dim, uuid);
        write_padded(f, &h, sizeof(h), 0);
        const size_t nodes_bytes = static_cast<size_t>(n) * index_.node_size;
        write_padded(f, index_.nodes_buffer, nodes_bytes, sizeof(h));
        f.sync();
    }

    // ----- .meta (quantizer + entry points + params) -----
    // Post-insert: entry points are already in final disk layout, so pass
    // them verbatim.
    write_meta_file(params, index_.core->entry_points(), uuid);

    // ----- .manifest (atomic commit) -----
    write_manifest_file(params, uuid);

    spdlog::info("[sextant] flush: sidecars rewritten (count={})", index_.count);
}

// =============================================================================
// prepare_codes — pass1 (reservoir + PQ train) + pass2 (encode all N into
// codes_buffer) + load FP16 raw_vecs_buffer. Leaves codes_buffer +
// raw_vecs_buffer populated for graph builds that need FP16 for
// prune/construct (set_build_vecs).
// =============================================================================
void Builder::prepare_codes(VectorSource& source, const ResolvedParams& params) {
    index_.count = source.count();
    index_.dim = source.dim();
    if (index_.count == 0) {
        throw Error(ErrorCode::InvalidParam, "prepare_codes: source is empty");
    }
    if (index_.dim == 0) {
        throw Error(ErrorCode::InvalidParam, "prepare_codes: source has dim=0");
    }
    // pass1 trains the quantizer and sets index_.code_size.
    // compute_entry_points=false: the merged-graph path computes entry points
    // later via snap_entry_points_ (stride-sampling fallback when
    // entry_centroids_ is empty), so the pass1 k-means is skipped here.
    pass1_sample_and_train(source, params, /*compute_entry_points=*/false);
    index_.node_size = VamanaCore::static_node_size(params.R, index_.code_size);

    // Allocate flat codes buffer.
    const size_t codes_bytes = static_cast<size_t>(index_.count) * index_.code_size;
    AlignedBuf codes(kDiskAlign, codes_bytes);
    std::memset(codes.get(), 0, codes_bytes);
    index_.codes_buffer = codes.as<uint8_t>();
    codes.release();
    spdlog::info("[sextant] prepare_codes: allocated {:.1f}MB codes",
                 codes_bytes / 1e6);

    pass2_encode(source, params);

    const size_t vecs_bytes =
        static_cast<size_t>(index_.count) * index_.dim * sizeof(float16_t);
    AlignedBuf vecs(kDiskAlign, vecs_bytes);
    index_.raw_vecs_buffer = vecs.as<float16_t>();
    vecs.release();
    spdlog::info("[sextant] prepare_codes: allocated {:.1f}MB FP16 raw_vecs_buffer",
                 vecs_bytes / 1e6);

    // Load raw vectors as FP16 for the FP16 prune (same as build_partitioned).
    source.reset();
    Chunk chunk{};
    uint64_t loaded = 0;
    while (source.next(chunk)) {
        for (uint32_t r = 0; r < chunk.count; r++) {
            const RowId rid = chunk.row_ids[r];
            if (rid >= 0 && static_cast<uint64_t>(rid) < index_.count) {
                float16_t* dst = index_.raw_vecs_buffer +
                                 static_cast<size_t>(rid) * index_.dim;
                const float* src = chunk.vectors +
                                   static_cast<size_t>(r) * index_.dim;
                cast_fp32_to_fp16(src, dst, index_.dim);
                loaded++;
            }
        }
    }
    spdlog::info("[sextant] prepare_codes: loaded {} FP16 vectors", loaded);
}

}  // namespace sextant
