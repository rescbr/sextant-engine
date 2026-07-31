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

#include "fbin_source.hpp"
#include "memory_source.hpp"
#include "partition.hpp"
#include "probe.hpp"
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

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <sys/mman.h>   // mmap for build_ivf_scan zero-copy source access
#include <sys/stat.h>   // fstat for mmap
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
    // graph sidecars (.epc, .ball, entry points in .meta).
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
// build_ivf — IVF-probe build (K independent shard Indexes + FP16 centroids).
//
// Pipeline (see docs/ivf_probe_design.md "Build pipeline"):
//   1. Global pass1 (quantizer train) + pass2 (encode all N) via prepare_codes.
//      The quantizer is trained ONCE and shared across every shard — per-shard
//      retraining would defeat the partitioned build.
//   2. partition_codes → PartitionAssignment (shards[] membership + centroids[]
//      PQ codes).
//   3. Per shard k: materialize shard-local codes/nodes/vecs, install a CLONE
//      of the global quantizer into a fresh shard Index, and delegate to
//      build_shard_into_ which wires a R_shard VamanaCore, constructs the graph,
//      and flushes standard sidecars to <index_path>.shards/shard_NNNN/.
//   4. centroids.bin: decode each of K centroid PQ codes → FP32 (decode_code)
//      → FP16 (cast_fp32_to_fp16), raw write (K × dim × float16_t, no header).
//   5. manifest: line-oriented text (ready / K / dim / n_probe_default /
//      closure_factor) — the IVFIndex::read commit point.
//
// The shard Index + Builder are destroyed after each flush; only sidecar files
// persist. No merge step (unlike build_partitioned). Each shard reopens via
// Index::read unchanged; the whole IVF index via IVFIndex::read.
// ===========================================================================
BuildResult Builder::build_ivf(VectorSource& source, const std::string& index_path,
                                const ResolvedParams& params,
                                uint32_t n_probe_default) {
    const auto t0 = std::chrono::steady_clock::now();

    // --- 1. Global quantizer (train once) + encode all N ---
    prepare_codes(source, params);

    const uint32_t n = static_cast<uint32_t>(index_.count);
    const uint32_t K = std::max<uint32_t>(2u, params.partition_count);
    const float closure_factor = params.closure_factor;

    spdlog::info("[sextant] build_ivf: N={} K={} closure_factor={:.4f} → '{}.shards'",
                 n, K, closure_factor, index_path);

    // --- 2. Partition via k-means on PQ codes (closure overlap) ---
    auto assignment = partition_codes(*index_.quantizer, index_.codes_buffer, n,
                                       index_.code_size, K, closure_factor,
                                       /*iterations=*/10, params.num_threads,
                                       /*seed=*/0xC0DE1234ULL,
                                       /*balance_factor=*/0.0f,
                                       /*closure_epsilon=*/params.closure_epsilon);
    if (assignment.shards.size() != K) {
        throw Error(ErrorCode::InvalidParam,
                    "build_ivf: partition returned K=" +
                        std::to_string(assignment.shards.size()) +
                        " (expected " + std::to_string(K) + ")");
    }

    // Per-shard degree: R_shard = 2R/3 (same as build_partitioned's K>1 path).
    // The reopened shard records R = R_shard in .meta so Index::read computes
    // node_size = static_node_size(R_shard, code_size).
    const uint16_t R_shard = static_cast<uint16_t>(
        std::max<uint32_t>(8u, (2u * params.R) / 3u));
    const uint32_t shard_node_size =
        VamanaCore::static_node_size(R_shard, index_.code_size);
    spdlog::info("[sextant] build_ivf: R_shard={} (2R/3, R={})", R_shard, params.R);

    // Shard params carry R = R_shard so write_sidecars_ emits the final layout
    // at R_shard (matching the build layout) and the serialized ResolvedParams
    // reopen the shard at the correct node_size.
    ResolvedParams shard_params = params;
    shard_params.R = R_shard;
    shard_params.partition_count = 1;  // each shard is itself a single partition
    shard_params.closure_factor = closure_factor;

    // Commit point: the .shards directory groups all shards + centroids + the
    // manifest. Create it up front; per-shard subdirs are created below.
    const std::string shards_dir = index_path + ".shards";
    std::error_code ec;
    std::filesystem::create_directories(shards_dir, ec);
    if (ec) {
        throw Error(ErrorCode::IoError,
                    "build_ivf: cannot create shards dir '" + shards_dir +
                        "': " + ec.message());
    }

    // --- 3. Per-shard construct + flush ---
    uint32_t shards_written = 0;
    for (uint32_t k = 0; k < K; k++) {
        auto& members = assignment.shards[k];
        const uint32_t shard_n = static_cast<uint32_t>(members.size());
        char dirbuf[32];
        std::snprintf(dirbuf, sizeof(dirbuf), "shard_%04u", k + 1);  // 1-indexed
        const std::string shard_prefix = shards_dir + "/" + dirbuf;
        if (shard_n == 0) {
            spdlog::warn("[sextant] build_ivf: shard {} empty, skipping (no "
                         "sidecars written)", k);
            continue;
        }

        // Fresh shard Index: a CLONE of the global quantizer (so the shard
        // Builder can serialize it independently in write_meta_file, and the
        // reopened shard reconstructs its own VamanaCore). Clone via
        // serialize+deserialize — the only supported deep-copy path.
        auto shard_index = std::make_unique<Index>();
        shard_index->dim = index_.dim;
        shard_index->count = shard_n;
        shard_index->code_size = index_.code_size;
        shard_index->node_size = shard_node_size;  // build layout = R_shard
        shard_index->params = shard_params;
        shard_index->path = shard_prefix;

        {
            std::vector<uint8_t> qblob;
            index_.quantizer->serialize(qblob);
            shard_index->quantizer = std::make_unique<PqQuantizer>(
                MetricKind::L2Sq, index_.dim, /*m=*/1, /*bits=*/8);  // placeholder
            shard_index->quantizer->deserialize(qblob.data(), qblob.size());
        }

        // Materialize shard-local buffers (local 0..shard_n-1). Each global
        // code/vec is memcpy'd once to its shard — O(N) total across shards.
        const size_t codes_bytes =
            static_cast<size_t>(shard_n) * index_.code_size;
        const size_t nodes_bytes =
            static_cast<size_t>(shard_n) * shard_node_size;
        const size_t vecs_bytes =
            static_cast<size_t>(shard_n) * index_.dim * sizeof(float16_t);
        AlignedBuf scodes(kDiskAlign, codes_bytes);
        AlignedBuf snodes(kDiskAlign, nodes_bytes);
        AlignedBuf svecs(kDiskAlign, vecs_bytes);
        std::memset(scodes.get(), 0, codes_bytes);
        std::memset(snodes.get(), 0, nodes_bytes);
        for (uint32_t i = 0; i < shard_n; i++) {
            const uint32_t gid = members[i];
            std::memcpy(scodes.as<uint8_t>() +
                            static_cast<size_t>(i) * index_.code_size,
                        index_.codes_buffer +
                            static_cast<size_t>(gid) * index_.code_size,
                        index_.code_size);
            std::memcpy(svecs.as<float16_t>() +
                            static_cast<size_t>(i) * index_.dim,
                        index_.raw_vecs_buffer +
                            static_cast<size_t>(gid) * index_.dim,
                        index_.dim * sizeof(float16_t));
        }
        // Hand ownership of the aligned buffers to the shard Index. They are
        // freed by ~Index (aligned_free). The fresh shard Builder + VamanaCore
        // read through these pointers during construct + flush.
        shard_index->codes_buffer = scodes.as<uint8_t>(); scodes.release();
        shard_index->nodes_buffer = snodes.as<uint8_t>(); snodes.release();
        shard_index->raw_vecs_buffer = svecs.as<float16_t>(); svecs.release();

        spdlog::info("[sextant] build_ivf: shard {}/{} ({} vectors, RAM≈{:.1f}MB)",
                     k + 1, K, shard_n,
                     (codes_bytes + nodes_bytes + vecs_bytes) / 1e6);

        build_shard_into_(*shard_index, shard_params, members, k, K, shard_prefix);
        // shard_index destroyed here — only the flushed sidecars persist.
        ++shards_written;
    }

    if (shards_written == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "build_ivf: every shard was empty (K=" +
                        std::to_string(K) + ", N=" + std::to_string(n) + ")");
    }

    // --- 4. centroids.bin: K × dim × float16_t (decoded from PQ centroid codes) ---
    // Each centroid PQ code is reverse-mapped to FP32 via the quantizer's
    // codebook (decode_code), then cast to FP16 for routing (symmetry with the
    // FP16 search tier; see design "Centroid format"). Raw write, no header.
    {
        const std::string path = shards_dir + "/centroids.bin";
        DirectFile f(path, true);
        const size_t vec_bytes = static_cast<size_t>(index_.dim) * sizeof(float16_t);
        // Block-aligned ring buffer to amortize syscalls (same pattern as
        // write_sidecars_'s .codes stream).
        const size_t block_cap = kBlockSize;  // 256KB
        const uint32_t vecs_per_block =
            std::max<uint32_t>(1u, static_cast<uint32_t>(block_cap / vec_bytes));
        const size_t buf_cap = static_cast<size_t>(vecs_per_block) * vec_bytes;
        AlignedBuf ring(kDiskAlign, buf_cap);

        std::vector<float> centroid_f32(index_.dim);
        std::vector<float16_t> centroid_f16(index_.dim);
        uint64_t write_off = 0;
        uint32_t in_block = 0;
        for (uint32_t k = 0; k < K; k++) {
            // An empty shard still has a centroid code from partition_codes;
            // decode it regardless so centroids.bin stays K-contiguous (the
            // manifest records K and IVFIndex::read expects K × dim).
            const auto& code = assignment.centroids[k];
            if (code.size() != index_.code_size) {
                throw Error(ErrorCode::InvalidParam,
                            "build_ivf: centroid " + std::to_string(k) +
                                " has wrong code size");
            }
            index_.quantizer->decode_code(code.data(), centroid_f32.data());
            cast_fp32_to_fp16(centroid_f32.data(), centroid_f16.data(), index_.dim);
            std::memcpy(ring.as<uint8_t>() +
                            static_cast<size_t>(in_block) * vec_bytes,
                        centroid_f16.data(), vec_bytes);
            if (++in_block >= vecs_per_block) {
                write_padded(f, ring.get(),
                             static_cast<size_t>(in_block) * vec_bytes, write_off);
                write_off += static_cast<size_t>(in_block) * vec_bytes;
                in_block = 0;
            }
        }
        if (in_block > 0) {
            write_padded(f, ring.get(),
                         static_cast<size_t>(in_block) * vec_bytes, write_off);
        }
        f.sync();
        spdlog::info("[sextant] wrote {} ({} centroids × dim={} FP16, {} bytes)",
                     path, K, index_.dim,
                     static_cast<uint64_t>(K) * vec_bytes);
    }

    // --- 5. manifest (line-oriented text; IVFIndex::read commit point) ---
    // Written LAST via temp + rename so its presence signals a complete build.
    {
        // np ∝ √K: the single best-validated scaling law (3 datasets, LID
        // 13-21, recall 0.55-0.99). K/4 over-probes at K>16.
        const uint32_t n_probe = n_probe_default > 0
                                     ? n_probe_default
                                     : std::max(1u, static_cast<uint32_t>(
                                           2.0f * std::sqrt(float(K))));
        const std::string path = shards_dir + "/manifest";
        const std::string tmp = path + ".tmp";
        {
            DirectFile f(tmp, true);
            std::string commit =
                std::string("ready\n") +
                std::to_string(K) + "\n" +
                std::to_string(index_.dim) + "\n" +
                std::to_string(n_probe) + "\n" +
                std::to_string(closure_factor) + "\n";
            write_padded(f, commit.data(), commit.size(), 0);
            f.sync();
        }
        std::error_code rec;
        std::filesystem::rename(tmp, path, rec);
        if (rec) {
            throw Error(ErrorCode::IoError,
                        "build_ivf: manifest rename failed: " + rec.message());
        }
        spdlog::info("[sextant] wrote {} (K={} n_probe_default={} closure_factor="
                     "{:.4f})", path, K, n_probe, closure_factor);
    }

    // The global build buffers (codes/raw_vecs on index_) are no longer needed;
    // free them so the Builder doesn't hold N-sized RAM after the build.
    if (index_.codes_buffer) { aligned_free(index_.codes_buffer); index_.codes_buffer = nullptr; }
    if (index_.raw_vecs_buffer) { aligned_free(index_.raw_vecs_buffer); index_.raw_vecs_buffer = nullptr; }
    // index_ never allocated nodes_buffer for IVF (no merged graph).

    const auto t1 = std::chrono::steady_clock::now();
    spdlog::info("[sextant] build_ivf complete (K={}, {}/{} shards written) "
                 "in {:.2f}s", K, shards_written, K,
                 std::chrono::duration<double>(t1 - t0).count());

    BuildResult result;
    result.index_path = shards_dir;
    result.n_vectors = index_.count;
    result.dim = index_.dim;
    result.R = params.R;
    result.L_build = params.L_build;
    result.pq_m = index_.quantizer ? index_.quantizer->m() : params.pq_m;
    result.pq_bits = index_.quantizer ? index_.quantizer->bits() : params.pq_bits;
    result.build_time_sec = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

// ===========================================================================
// build_ivf_scan — IVF-list-scan + 4-bit PQ FastScan (Option A, the DEFAULT
// build path when BuildConfig::merged_graph is false).
//
// Pipeline:
//   1. prepare_codes (existing): train the 8-bit routing PQ + encode all N
//      (used for partition_codes — k-means on PQ codes).
//   2. Train a SEPARATE 4-bit PQ codebook on a FP32 sample (the validated
//      m=192/bits=4 config from the spike). This codebook is shared across
//      all shards.
//   3. partition_codes → PartitionAssignment (shard membership + centroids).
//   4. For each shard: encode each member at 4-bit, pack into FastScan block
//      layout, write `.codes4` + `.rowids` sidecars.
//   5. centroids.bin: K × dim × float16_t (decoded from the partition's 8-bit
//      PQ centroid codes — same as build_ivf).
//   6. codebook4.bin: serialized 4-bit PqQuantizer (shared).
//   7. manifest (line-oriented text; IVFScanIndex::read commit point).
//
// NO graph, NO Vamana, NO BFS reorder, NO .epc. Each shard is a pure code
// container. See ~/.local/state/maki/plans/sharing-eternal-louse.md.
// ===========================================================================
BuildResult Builder::build_ivf_scan(VectorSource& source,
                                     const std::string& index_path,
                                     const ResolvedParams& params,
                                     uint32_t n_probe_default) {
    const auto t0 = std::chrono::steady_clock::now();

    // --- 1. Global 8-bit PQ (routing) + encode all N ---
    // prepare_routing_codes (not prepare_codes): the scan path mmaps the FP32
    // source directly for 4-bit encoding, so it does NOT need the N × dim × 2
    // FP16 raw_vecs_buffer (13.4 GB at 100M/768-dim) that prepare_codes
    // materializes for graph builds. Skipping it unblocks 100M+ builds on a
    // 30 GB VM.
    prepare_routing_codes(source, params);

    const uint32_t n = static_cast<uint32_t>(index_.count);
    const uint32_t K = std::max<uint32_t>(1u, params.partition_count);
    const Dim dim = index_.dim;
    const uint16_t m4 = params.pq4_m > 0 ? params.pq4_m
                                          : static_cast<uint16_t>(dim / 4);
    if (m4 == 0 || dim % m4 != 0) {
        throw Error(ErrorCode::InvalidParam,
                    "build_ivf_scan: pq4_m=" + std::to_string(m4) +
                        " invalid for dim=" + std::to_string(dim) +
                        " (must divide dim)");
    }

    spdlog::info("[sextant] build_ivf_scan: N={} K={} m4={} → '{}.shards'",
                 n, K, m4, index_path);

    // --- 2. Train the 4-bit codebook on a FP32 sample drawn from source ---
    // Re-reads source once (reset + sample). The 8-bit codebook trained on a
    // reservoir; we mirror that for 4-bit. Sample size matches the spike
    // (min(20k, N)) — PQ training is sample-size-insensitive past ~10k.
    const uint8_t scan_bits = params.scan_pq_bits;
    if (scan_bits != 4 && scan_bits != 8) {
        throw Error(ErrorCode::InvalidParam,
                    "build_ivf_scan: scan_pq_bits must be 4 or 8 (got " +
                        std::to_string(scan_bits) + ")");
    }
    // Construct the scan codebook. Dispatch on params.quantizer_type:
    //   "prq"            → ProductResidualQuantizer (additive-residual PQ;
    //                       4-bit only for now — FastScan nibble path).
    //   "anisotropic-pq" → AnisotropicPqQuantizer (ScaNN score-aware loss:
    //                       penalize parallel quantization error ~4× more than
    //                       orthogonal). Same code format, same FastScan kernel.
    //   "pq" (default)   → PqQuantizer (standard k-means).
    std::unique_ptr<PqQuantizer> qscan_owner;
    if (params.quantizer_type == "prq") {
        if (scan_bits != 4) {
            throw Error(ErrorCode::InvalidParam,
                        "build_ivf_scan: PRQ currently supports only 4-bit "
                        "FastScan (scan_pq_bits=4); got scan_pq_bits=" +
                            std::to_string(static_cast<unsigned>(scan_bits)));
        }
        // PRQ: nsplits defaults to dim/8 (sub_dim=8, M_sub=2 at m4=192).
        // Measured best: nsplits=96 at dim=768 (sub_dim=8). The old default
        // of dim/32 gave sub_dim=32 (M_sub=4) — strictly worse recall.
        const uint32_t nsplits = (params.prq_nsplits > 0)
            ? params.prq_nsplits
            : static_cast<uint32_t>(dim) / 8;
        if (nsplits == 0 || dim % nsplits != 0) {
            throw Error(ErrorCode::InvalidParam,
                        "build_ivf_scan: PRQ dim=" + std::to_string(dim) +
                            " not divisible by nsplits=" +
                            std::to_string(nsplits));
        }
        if (m4 % nsplits != 0) {
            throw Error(ErrorCode::InvalidParam,
                        "build_ivf_scan: PRQ m4=" + std::to_string(m4) +
                            " not divisible by nsplits=" +
                            std::to_string(nsplits));
        }
        qscan_owner = std::make_unique<ProductResidualQuantizer>(
            params.metric, dim, m4, scan_bits, nsplits,
            params.prq_beam_size, /*seed=*/42,
            params.prq_encode_mode, params.prq_icm_iters,
            params.prq_ils_iters, params.prq_ils_perturb,
            params.prq_lsq_train_iters);
    } else if (params.quantizer_type == "anisotropic-pq") {
        qscan_owner = std::make_unique<AnisotropicPqQuantizer>(
            params.metric, dim, m4, scan_bits,
            /*threshold=*/0.2f, /*seed=*/42);
    } else if (params.quantizer_type == "rabitq") {
        if (scan_bits != 4) {
            throw Error(ErrorCode::InvalidParam,
                        "build_ivf_scan: RaBitQ currently supports only 4-bit "
                        "FastScan (scan_pq_bits=4); got scan_pq_bits=" +
                            std::to_string(static_cast<unsigned>(scan_bits)));
        }
        // RaBitQ: per-vector 1-bit sign codes (nibble-packed, same FastScan
        // layout as PQ4) + 2 per-vector factors stored in a sidecar. m4 is
        // forced to dim/4 inside the constructor regardless of the passed m4.
        qscan_owner = std::make_unique<RaBitQQuantizer>(params.metric, dim, 42);
    } else {
        qscan_owner = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, scan_bits, /*seed=*/42);
    }
    PqQuantizer& qscan = *qscan_owner;
    {
        const uint32_t train_n = std::min<uint64_t>(20'000, n);
        std::vector<float> sample(static_cast<size_t>(train_n) * dim);
        // Prefer the first train_n contiguous vectors (matches the spike); if
        // source has a path, seek-sample; otherwise reset + next.
        source.reset();
        Chunk chunk;
        uint32_t filled = 0;
        while (filled < train_n && source.next(chunk)) {
            const uint32_t take = std::min<uint32_t>(
                train_n - filled, chunk.count);
            std::memcpy(sample.data() + static_cast<size_t>(filled) * dim,
                        chunk.vectors,
                        static_cast<size_t>(take) * dim * sizeof(float));
            filled += take;
        }
        if (filled == 0) {
            throw Error(ErrorCode::IoError,
                        "build_ivf_scan: source yielded no vectors for 4-bit "
                        "codebook training");
        }
        spdlog::info("[sextant] training {} (m={}, K=16) on {} samples",
                     params.quantizer_type == "prq" ? "PRQ"
                     : (params.quantizer_type == "anisotropic-pq" ? "4-bit anisotropic-PQ"
                      : (params.quantizer_type == "rabitq" ? "RaBitQ"
                                                           : "4-bit PQ")),
                     m4, filled);
        const auto ts = std::chrono::steady_clock::now();
        qscan.set_num_threads(params.num_threads);
        qscan.train(sample.data(), filled);
        const auto te = std::chrono::steady_clock::now();
        spdlog::info("[sextant]   4-bit codebook trained in {:.2f}s",
                     std::chrono::duration<double>(te - ts).count());
    }

    // --- 3. Partition via k-means on the 8-bit routing codes ---
    auto assignment = partition_codes(*index_.quantizer, index_.codes_buffer, n,
                                        index_.code_size, K, params.closure_factor,
                                        /*iterations=*/10, params.num_threads,
                                        /*seed=*/0xC0DE1234ULL,
                                        /*balance_factor=*/params.partition_balance_factor,
                                        /*closure_epsilon=*/params.closure_epsilon);
    if (assignment.shards.size() != K) {
        throw Error(ErrorCode::InvalidParam,
                    "build_ivf_scan: partition returned K=" +
                        std::to_string(assignment.shards.size()) +
                        " (expected " + std::to_string(K) + ")");
    }

    const std::string shards_dir = index_path + ".shards";
    std::error_code ec;
    std::filesystem::create_directories(shards_dir, ec);
    if (ec) {
        throw Error(ErrorCode::IoError,
                    "build_ivf_scan: cannot create shards dir '" + shards_dir +
                        "': " + ec.message());
    }

    // --- 4. Per-shard: encode 4-bit, pack FastScan blocks, write sidecars ---
    // Encode each member to 4-bit. The source vectors are needed for the
    // 4-bit PQ encoding. Instead of materializing all N FP32 vectors in RAM
    // (N × dim × 4 bytes = 4.1 GB at arxiv-nomic scale, 48 GB at 1B), we
    // mmap the source .fbin file for zero-copy access. The mmap'd pointer
    // replaces the all_vecs buffer — each member's vector is accessed
    // on-demand via its global ID offset.
    //
    // For sources that aren't file-backed (MemorySource), the source's own
    // chunk API is used instead. The mmap path is a specialization for
    // FbinSource (the common build/benchmark case).
    const float* vecs_base = nullptr;
    void* vecs_mmap = nullptr;
    uint64_t vecs_mmap_size = 0;
    // Try to get the source's file path for mmap. FbinSource returns its
    // path via the virtual VectorSource::path(); other sources return "".
    std::string source_path = source.path();
    if (!source_path.empty()) {
        int fd = ::open(source_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            struct stat st;
            if (::fstat(fd, &st) == 0 && st.st_size > 8) {
                vecs_mmap_size = st.st_size;
                vecs_mmap = ::mmap(nullptr, vecs_mmap_size, PROT_READ,
                                   MAP_SHARED, fd, 0);
                if (vecs_mmap != MAP_FAILED) {
                    // Skip the 8-byte fbin header (n + dim).
                    vecs_base = reinterpret_cast<const float*>(
                        static_cast<char*>(vecs_mmap) + 8);
                } else {
                    vecs_mmap = nullptr;
                }
            }
            ::close(fd);
        }
    }

    // Fallback: if mmap failed (non-file source), materialize all vectors.
    // This is the RAM-heavy path — only used by MemorySource or on mmap error.
    std::vector<float> all_vecs_fallback;
    if (!vecs_base) {
        spdlog::warn("[sextant] build_ivf_scan: source not mmap-able; "
                     "materializing all {} vectors in RAM", n);
        all_vecs_fallback.resize(static_cast<size_t>(n) * dim);
        source.reset();
        Chunk chunk;
        uint32_t filled = 0;
        while (filled < n && source.next(chunk)) {
            const uint32_t take = std::min<uint32_t>(n - filled, chunk.count);
            std::memcpy(all_vecs_fallback.data() +
                            static_cast<size_t>(filled) * dim,
                        chunk.vectors,
                        static_cast<size_t>(take) * dim * sizeof(float));
            filled += take;
        }
        if (filled != n) {
            throw Error(ErrorCode::IoError,
                        "build_ivf_scan: source yielded " +
                            std::to_string(filled) + " vectors (" +
                            std::to_string(n) + " expected)");
        }
        vecs_base = all_vecs_fallback.data();
    }

    const auto uuid = make_uuid();
    const uint32_t m = m4;
    const uint32_t block_bytes = static_cast<uint32_t>(m) * 16;
    const std::string codes_filename =
        (scan_bits == 4) ? "/.codes4" : "/.codes8";
    const uint32_t codes_per_block = (scan_bits == 4) ? 32 : 16;

    // --- Adaptive sub-shard threshold ---
    // When sub_shard_threshold == 0 (auto), compute a target sub-shard size
    // from the shard size distribution. The goal: average shards get ~4
    // sub-shards, fat shards get more, small shards stay flat (S=1).
    //
    // Heuristic: target = max(floor, mean_shard_size / 4). The floor
    // (2048) prevents over-splitting on datasets with naturally small
    // shards. At billion scale (mean_shard ~ 1M), this gives target=250k
    // → ~4 sub-shards per average shard, ~20+ for fat ones.
    //
    // Also auto-compute sub_shard_n_probe: probe_fraction = 50% of
    // sub-shards per shard. This is stored as a ratio in the manifest;
    // the search path computes actual sub_np = max(1, ceil(S * fraction))
    // per shard at query time.
    uint32_t effective_threshold = params.sub_shard_threshold;
    float sub_probe_fraction = 0.0f;
    if (effective_threshold == 0) {
        // Compute mean shard size (non-empty shards only).
        uint64_t total_members = 0;
        uint32_t non_empty = 0;
        uint32_t max_shard = 0;
        for (const auto& sh : assignment.shards) {
            if (!sh.empty()) {
                total_members += sh.size();
                non_empty++;
                max_shard = std::max(max_shard,
                    static_cast<uint32_t>(sh.size()));
            }
        }
        if (non_empty > 0 && total_members > 0) {
            const uint32_t mean_shard =
                static_cast<uint32_t>(total_members / non_empty);
            constexpr uint32_t kSubSizeFloor = 2048;
            effective_threshold = std::max(kSubSizeFloor, mean_shard / 4);
            // Only enable if at least one shard exceeds 2× the threshold
            // (otherwise sub-sharding adds overhead with no benefit).
            if (max_shard < 2 * effective_threshold) {
                effective_threshold = 0;  // disable
            }
            sub_probe_fraction = 0.5f;  // probe 50% of sub-shards
            spdlog::info("[sextant] build_ivf_scan: auto sub-shard threshold "
                         "= {} (mean_shard={}, max_shard={}, probe_frac={:.2f})",
                         effective_threshold, mean_shard, max_shard,
                         sub_probe_fraction);
        }
    } else {
        // Manual threshold: probe all sub-shards unless sub_np was set.
        // sub_probe_fraction=1.0 means "scan all" (backward compatible).
        sub_probe_fraction =
            (params.sub_shard_n_probe > 1) ? 1.0f : 1.0f;
    }

    // Shared accumulator for sub-shard centroids (FP16). Written by parallel
    // encode_shard tasks under sub_centroids_mutex.
    std::vector<float16_t> sub_centroids_accum;
    std::mutex sub_centroids_mutex;

    // --- Per-shard encode + write (PARALLEL across K) ---
    // Each shard is independent: its own member list, its own output
    // directory, no shared mutable state. Work-steal over K (one shard per
    // task). num_threads comes from resolve_params; clamped to K because
    // there's no benefit to more workers than shards. Mirrors the
    // partition_codes pool pattern (empty WorkerState — each task is
    // self-contained, no reusable per-thread buffers worth caching).
    const uint32_t encode_threads =
        std::max(1u, std::min(params.num_threads, K));

    struct ShardEncodeState {};
    ctpl::thread_pool_tls<ShardEncodeState> encode_pool(encode_threads);

    // Per-shard body. Returns 1 if the shard was written, 0 if it was
    // empty-skipped. Captures only const inputs + the shards_dir path.
    auto encode_shard = [&](uint32_t k) -> uint32_t {
        auto& members = assignment.shards[k];
        const uint32_t shard_n = static_cast<uint32_t>(members.size());
        char dirbuf[32];
        std::snprintf(dirbuf, sizeof(dirbuf), "shard_%04u", k + 1);
        const std::string shard_dir = shards_dir + "/" + dirbuf;
        if (shard_n == 0) {
            spdlog::warn("[sextant] build_ivf_scan: shard {} empty, skipping",
                         k + 1);
            return 0;
        }

        // --- Adaptive sub-shard decision ---
        // S = ceil(shard_n / effective_threshold). Shards that fit within
        // the threshold stay flat (S=1). No upper cap — at billion scale,
        // a shard may have millions of vectors and need many sub-shards.
        // Routing overhead (sub-centroid FP16 dists) is only 2.1% of cycles
        // (profiled), so more sub-shards = net win.
        uint32_t S = 1;
        if (effective_threshold > 0) {
            S = (shard_n + effective_threshold - 1) /
                effective_threshold;
            // Clamp to 1 (no split for small shards). S=0 would be a bug.
            S = std::max(1u, S);
        }
        const bool do_sub_shard = (S > 1);

        if (!do_sub_shard) {
            std::error_code mkrec;
            std::filesystem::create_directories(shard_dir, mkrec);
            if (mkrec) {
                throw Error(ErrorCode::IoError,
                            "build_ivf_scan: cannot create shard dir '" +
                                shard_dir + "': " + mkrec.message());
            }
            // Flat: encode all members to shard_dir (existing code path).
            // Fall through to the flat encode below.
        } else {
            // --- Sub-shard: local k-means(S) on scan codes ---
            spdlog::info("[sextant] build_ivf_scan: shard {} has {} vectors "
                         "(threshold={}), splitting into {} sub-shards",
                         k + 1, shard_n, effective_threshold, S);

            // Encode all members to scan codes first.
            std::vector<uint8_t> seg_codes(static_cast<size_t>(shard_n) * m);
            {
                std::vector<uint8_t> packed(qscan.code_size());
                for (uint32_t i = 0; i < shard_n; i++) {
                    const float* vec = vecs_base +
                        static_cast<size_t>(members[i]) * dim;
                    qscan.encode(vec, packed.data());
                    for (uint32_t s = 0; s < m; s++) {
                        seg_codes[static_cast<size_t>(i) * m + s] =
                            (scan_bits == 4)
                                ? static_cast<uint8_t>(
                                      (packed[s / 2] >> ((s % 2) * 4)) & 0xF)
                                : packed[s];
                    }
                }
            }

            // Local k-means(S) on the scan codes using the scan quantizer's
            // cross-distance table (if available; fallback to code_distance).
            const uint32_t code_sz = qscan.code_size();
            // Pack seg_codes into code_size bytes for partition_codes.
            std::vector<uint8_t> flat_codes(
                static_cast<size_t>(shard_n) * code_sz, 0);
            for (uint32_t i = 0; i < shard_n; i++) {
                for (uint32_t s = 0; s < m; s++) {
                    uint8_t val = seg_codes[static_cast<size_t>(i) * m + s];
                    if (scan_bits == 4) {
                        if (s % 2 == 0)
                            flat_codes[static_cast<size_t>(i) * code_sz + s / 2] = val;
                        else
                            flat_codes[static_cast<size_t>(i) * code_sz + s / 2] |= (val << 4);
                    } else {
                        flat_codes[static_cast<size_t>(i) * code_sz + s] = val;
                    }
                }
            }

            // Partition with closure overlap. Sub-shard boundary vectors
            // are replicated to multiple sub-shards to prevent recall loss
            // when sub_np < S. Use ratio-based closure (1.05) — the absolute
            // margin is harder to calibrate for the scan quantizer's distance
            // scale (4-bit PQ code distances are in different units than FP16).
            // Sub-shard k-means: run serially (num_threads=1). This runs inside
            // the encode_pool worker — spawning another pool would create nested
            // pools (8×8=64 threads on 8 cores → oversubscription). The sub-shard
            // data is small enough (~shard_n codes) that serial k-means is fast.
            auto sub_assignment = partition_codes(
                qscan, flat_codes.data(), shard_n, code_sz, S,
                /*closure=*/1.05f, /*iters=*/5, /*num_threads=*/1,
                /*seed=*/0xC0DE1234ULL + k,
                /*balance_factor=*/0.0f,
                /*closure_epsilon=*/0.0f);

            // Decode sub-centroids to FP16 for routing.
            std::vector<float16_t> sub_cents(S * dim);
            for (uint32_t s = 0; s < S; s++) {
                float centroid_f32[768];
                qscan.decode_code(sub_assignment.centroids[s].data(),
                                  centroid_f32);
                cast_fp32_to_fp16(centroid_f32, &sub_cents[size_t(s) * dim],
                                  dim);
            }

            // Write each sub-shard. Accumulate centroid offsets for shard.manifest.
            std::vector<uint32_t> sub_offsets;
            {
                std::lock_guard<std::mutex> lk(sub_centroids_mutex);
                for (uint32_t s = 0; s < S; s++) {
                    sub_offsets.push_back(
                        static_cast<uint32_t>(sub_centroids_accum.size()));
                    sub_centroids_accum.insert(sub_centroids_accum.end(),
                                               &sub_cents[size_t(s) * dim],
                                               &sub_cents[size_t(s) * dim] + dim);
                }
            }

            // Write sub-shard directories + files.
            std::error_code mkrec;
            std::filesystem::create_directories(shard_dir, mkrec);
            for (uint32_t s = 0; s < S; s++) {
                char sbuf[32];
                std::snprintf(sbuf, sizeof(sbuf), "sub_%04u", s);
                const std::string sub_dir = shard_dir + "/" + sbuf;
                std::filesystem::create_directories(sub_dir, mkrec);

                auto& sub_members = sub_assignment.shards[s];
                const uint32_t sub_n =
                    static_cast<uint32_t>(sub_members.size());
                if (sub_n == 0) continue;

                // Re-encode sub_members to this sub-shard's seg_codes.
                const uint32_t n_blocks =
                    (sub_n + codes_per_block - 1) / codes_per_block;
                std::vector<uint8_t> sub_seg(static_cast<size_t>(sub_n) * m);
                std::vector<uint8_t> packed(qscan.code_size());
                for (uint32_t i = 0; i < sub_n; i++) {
                    // sub_members[i] is an index into flat_codes (shard-local).
                    const uint32_t local_idx = sub_members[i];
                    const float* vec = vecs_base +
                        static_cast<size_t>(members[local_idx]) * dim;
                    qscan.encode(vec, packed.data());
                    for (uint32_t ss = 0; ss < m; ss++) {
                        sub_seg[static_cast<size_t>(i) * m + ss] =
                            (scan_bits == 4)
                                ? static_cast<uint8_t>(
                                      (packed[ss / 2] >> ((ss % 2) * 4)) & 0xF)
                                : packed[ss];
                    }
                }
                // Pack FastScan blocks.
                std::vector<uint8_t> sub_blocks(
                    static_cast<size_t>(n_blocks) * block_bytes, 0);
                for (uint32_t b = 0; b < n_blocks; b++) {
                    for (uint32_t ss = 0; ss < m; ss++) {
                        for (uint32_t kk = 0; kk < 16; kk++) {
                            if (scan_bits == 4) {
                                const uint32_t v0 = b * 32 + kk;
                                const uint32_t v1 = b * 32 + 16 + kk;
                                const uint8_t lo = (v0 < sub_n)
                                    ? sub_seg[static_cast<size_t>(v0) * m + ss] : 0;
                                const uint8_t hi = (v1 < sub_n)
                                    ? sub_seg[static_cast<size_t>(v1) * m + ss] : 0;
                                sub_blocks[((static_cast<size_t>(b) * m) + ss) * 16 + kk] =
                                    static_cast<uint8_t>((hi << 4) | lo);
                            } else {
                                const uint32_t v = b * 16 + kk;
                                sub_blocks[((static_cast<size_t>(b) * m) + ss) * 16 + kk] =
                                    (v < sub_n) ? sub_seg[static_cast<size_t>(v) * m + ss] : 0;
                            }
                        }
                    }
                }
                // Write .codes{4,8}.
                {
                    const uint64_t magic =
                        (scan_bits == 4) ? kMagicCodes4 : kMagicCodes8;
                    DirectFile f(sub_dir + codes_filename, true);
                    SidecarHeader h{};
                    fill_header(h, magic, sub_n, dim, uuid);
                    write_padded(f, &h, sizeof(h), 0);
                    write_padded(f, sub_blocks.data(), sub_blocks.size(),
                                 sizeof(h));
                    f.sync();
                }
                // Write .rowids (map to GLOBAL RowId via members[]).
                {
                    std::vector<RowId> rids(sub_n);
                    for (uint32_t i = 0; i < sub_n; i++)
                        rids[i] = static_cast<RowId>(members[sub_members[i]]);
                    DirectFile f(sub_dir + "/.rowids", true);
                    SidecarHeader h{};
                    fill_header(h, kMagicRowids, sub_n, dim, uuid);
                    write_padded(f, &h, sizeof(h), 0);
                    write_padded(f, rids.data(),
                                 static_cast<size_t>(sub_n) * sizeof(RowId),
                                 sizeof(h));
                    f.sync();
                }
                // Per-sub-shard manifest.
                {
                    const std::string mpath = sub_dir + "/.manifest";
                    const std::string tmp = mpath + ".tmp";
                    DirectFile f(tmp, true);
                    std::string commit = std::string("ready\n") +
                        std::to_string(sub_n) + "\n" +
                        std::to_string(dim) + "\n" +
                        std::to_string(m4) + "\n";
                    write_padded(f, commit.data(), commit.size(), 0);
                    f.sync();
                    std::error_code rec;
                    std::filesystem::rename(tmp, mpath, rec);
                }
                spdlog::info("[sextant] build_ivf_scan: shard {}/{} sub {}/{} "
                             "({} vectors)", k + 1, K, s, S, sub_n);
            }

            // Write shard.manifest (sub-centroid offsets).
            {
                const std::string mpath = shard_dir + "/shard.manifest";
                const std::string tmp = mpath + ".tmp";
                DirectFile f(tmp, true);
                std::string commit = "ready\n" + std::to_string(S) + "\n";
                for (uint32_t s = 0; s < S; s++)
                    commit += std::to_string(sub_offsets[s]) + "\n";
                write_padded(f, commit.data(), commit.size(), 0);
                f.sync();
                std::error_code rec;
                std::filesystem::rename(tmp, mpath, rec);
            }
            return 1;
        }

        // --- Flat shard path (existing code, unchanged) ---
        const uint32_t n_blocks = (shard_n + codes_per_block - 1) / codes_per_block;

        // Encode each member to m segment-codes (one byte per segment).
        // For 4-bit the byte value is 0..15 (one nibble); for 8-bit it's 0..255.
        std::vector<uint8_t> seg_codes(static_cast<size_t>(shard_n) * m);
        // RaBitQ-only: per-vector factors (dp_multiplier, or_minus_c_l2sqr),
        // 2 floats each, written to the `.factors` sidecar below.
        std::vector<float> factors;
        const bool is_rabitq = (params.quantizer_type == "rabitq");
        {
            // RaBitQ encodes relative to the shard centroid: its full code is
            // sign bytes (code_size()) + 2 factor floats (factors_offset()).
            // PQ/PRQ/anisotropic encode absolute vectors: code_size() only.
            const uint32_t packed_size =
                is_rabitq ? static_cast<RaBitQQuantizer&>(qscan).full_code_size()
                          : qscan.code_size();
            std::vector<uint8_t> packed(packed_size);
            // Decode the shard centroid to FP32 once (RaBitQ needs it). The
            // centroid is a PQ code of the 8-bit routing quantizer.
            std::vector<float> centroid_f32;
            if (is_rabitq) {
                centroid_f32.resize(dim);
                const auto& centroid_code = assignment.centroids[k];
                if (centroid_code.size() != index_.code_size) {
                    throw Error(ErrorCode::InvalidParam,
                                "build_ivf_scan: shard " +
                                    std::to_string(k + 1) +
                                    " centroid code size mismatch");
                }
                index_.quantizer->decode_code(centroid_code.data(),
                                              centroid_f32.data());
                factors.resize(static_cast<size_t>(shard_n) * 2);
            }
            const uint32_t factors_off =
                is_rabitq ? static_cast<RaBitQQuantizer&>(qscan).factors_offset()
                          : 0;
            for (uint32_t i = 0; i < shard_n; i++) {
                const uint32_t gid = members[i];
                const float* vec =
                    vecs_base + static_cast<size_t>(gid) * dim;
                if (is_rabitq) {
                    static_cast<RaBitQQuantizer&>(qscan).encode_with_centroid(
                        vec, centroid_f32.data(), packed.data());
                    // Stash the 2 factor floats (dp_multiplier, or_minus_c_l2sqr).
                    std::memcpy(&factors[static_cast<size_t>(i) * 2],
                                packed.data() + factors_off, 2 * sizeof(float));
                } else {
                    qscan.encode(vec, packed.data());
                }
                for (uint32_t s = 0; s < m; s++) {
                    if (scan_bits == 4) {
                        seg_codes[static_cast<size_t>(i) * m + s] =
                            static_cast<uint8_t>(
                                (packed[s / 2] >> ((s % 2) * 4)) & 0xF);
                    } else {
                        seg_codes[static_cast<size_t>(i) * m + s] = packed[s];
                    }
                }
            }
        }

        // Pack into FastScan block layout: blocks[b][s][16], segment-major.
        // Tail block zero-pads (padding lanes are masked at search time).
        //   4-bit: byte k of segment s packs low=vec(b*32+k), high=vec(b*32+16+k).
        //   8-bit: byte k of segment s is vec(b*16+k) directly (no packing).
        std::vector<uint8_t> blocks(static_cast<size_t>(n_blocks) * block_bytes,
                                    0);
        for (uint32_t b = 0; b < n_blocks; b++) {
            for (uint32_t s = 0; s < m; s++) {
                for (uint32_t kk = 0; kk < 16; kk++) {
                    if (scan_bits == 4) {
                        const uint32_t v0 = b * 32 + kk;        // low nibble
                        const uint32_t v1 = b * 32 + 16 + kk;   // high nibble
                        const uint8_t lo = (v0 < shard_n)
                            ? seg_codes[static_cast<size_t>(v0) * m + s] : 0;
                        const uint8_t hi = (v1 < shard_n)
                            ? seg_codes[static_cast<size_t>(v1) * m + s] : 0;
                        blocks[((static_cast<size_t>(b) * m) + s) * 16 + kk] =
                            static_cast<uint8_t>((hi << 4) | lo);
                    } else {
                        const uint32_t v = b * 16 + kk;
                        blocks[((static_cast<size_t>(b) * m) + s) * 16 + kk] =
                            (v < shard_n)
                                ? seg_codes[static_cast<size_t>(v) * m + s]
                                : 0;
                    }
                }
            }
        }

        // Write .codes{4,8} (SidecarHeader + block payload). The magic tells
        // the searcher which kernel + LUT builder to use.
        {
            const uint64_t magic = (scan_bits == 4) ? kMagicCodes4 : kMagicCodes8;
            const std::string path =
                shard_dir + ((scan_bits == 4) ? "/.codes4" : "/.codes8");
            DirectFile f(path, true);
            SidecarHeader h{};
            fill_header(h, magic, shard_n, dim, uuid);
            write_padded(f, &h, sizeof(h), 0);
            write_padded(f, blocks.data(), blocks.size(), sizeof(h));
            f.sync();
        }
        // Write .rowids (SidecarHeader + shard_n × int64). Map shard-local
        // idx → global RowId. The source's row_ids may carry DB-layer IDs;
        // we collected row_ids during the re-read above implicitly (source
        // emits them in order). If the source provides row_ids, use them;
        // otherwise fall back to global sequential IDs (members[i]).
        {
            std::vector<RowId> rids(shard_n);
            // We did not retain chunk.row_ids above; re-derive from members.
            // For FbinSource (the benchmark path), row_id == global index,
            // so members[i] IS the RowId. For DB-backed sources this would
            // need a row_id cache pass — flagged as a regroup point.
            for (uint32_t i = 0; i < shard_n; i++) {
                rids[i] = static_cast<RowId>(members[i]);
            }
            const std::string path = shard_dir + "/.rowids";
            DirectFile f(path, true);
            SidecarHeader h{};
            fill_header(h, kMagicRowids, shard_n, dim, uuid);
            write_padded(f, &h, sizeof(h), 0);
            write_padded(f, rids.data(),
                         static_cast<size_t>(shard_n) * sizeof(RowId),
                         sizeof(h));
            f.sync();
        }
        // RaBitQ-only: write `.factors` (SidecarHeader + shard_n × 2 floats).
        // The two factors (dp_multiplier, or_minus_c_l2sqr) are read back at
        // index-open into shard->factors and used to finalize scan distances.
        if (is_rabitq) {
            const std::string path = shard_dir + "/.factors";
            DirectFile f(path, true);
            SidecarHeader h{};
            fill_header(h, kMagicFactors, shard_n, dim, uuid);
            write_padded(f, &h, sizeof(h), 0);
            write_padded(f, factors.data(), factors.size() * sizeof(float),
                         sizeof(h));
            f.sync();
        }
        // Per-shard manifest (atomic commit).
        {
            const std::string path = shard_dir + "/.manifest";
            const std::string tmp = path + ".tmp";
            {
                DirectFile f(tmp, true);
                std::string commit =
                    std::string("ready\n") +
                    std::to_string(shard_n) + "\n" +
                    std::to_string(dim) + "\n" +
                    std::to_string(m4) + "\n";
                write_padded(f, commit.data(), commit.size(), 0);
                f.sync();
            }
            std::error_code rec;
            std::filesystem::rename(tmp, path, rec);
            if (rec) {
                throw Error(ErrorCode::IoError,
                            "build_ivf_scan: shard manifest rename failed: " +
                                rec.message());
            }
        }

        spdlog::info("[sextant] build_ivf_scan: shard {}/{} ({} vectors, "
                     "{} blocks, {:.1f}MB codes)", k + 1, K, shard_n,
                     n_blocks, blocks.size() / 1e6);
        return 1;
    };

    // Work-steal over K: one shard per task. Atomic next_id drives the
    // claim loop; the pool's own queue handles load balancing across the
    // (intentionally oversubscribed) K shards. spdlog is thread-safe;
    // DirectFile/PqQuantizer::encode are stateless on shared state (each
    // task writes its own shard directory).
    std::atomic<uint32_t> next_shard{0};
    std::atomic<uint32_t> shards_written{0};
    std::vector<std::future<void>> futs;
    futs.reserve(encode_threads);
    for (uint32_t t = 0; t < encode_threads; t++) {
        futs.push_back(encode_pool.push(
            [&encode_shard, K, &next_shard, &shards_written]
            (size_t /*id*/, ShardEncodeState& /*w*/) {
                while (true) {
                    const uint32_t k = next_shard.fetch_add(
                        1, std::memory_order_relaxed);
                    if (k >= K) break;
                    shards_written.fetch_add(
                        encode_shard(k), std::memory_order_relaxed);
                }
            }));
    }
    for (auto& f : futs) f.get();

    if (shards_written.load(std::memory_order_relaxed) == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "build_ivf_scan: every shard was empty (K=" +
                        std::to_string(K) + ", N=" + std::to_string(n) + ")");
    }

    // --- 5. centroids.bin: K × dim × float16_t (same as build_ivf) ---
    {
        const std::string path = shards_dir + "/centroids.bin";
        DirectFile f(path, true);
        const size_t vec_bytes = static_cast<size_t>(dim) * sizeof(float16_t);
        const size_t block_cap = kBlockSize;
        const uint32_t vecs_per_block = std::max<uint32_t>(
            1u, static_cast<uint32_t>(block_cap / vec_bytes));
        const size_t buf_cap = static_cast<size_t>(vecs_per_block) * vec_bytes;
        AlignedBuf ring(kDiskAlign, buf_cap);

        std::vector<float> centroid_f32(dim);
        std::vector<float16_t> centroid_f16(dim);
        uint64_t write_off = 0;
        uint32_t in_block = 0;
        for (uint32_t k = 0; k < K; k++) {
            const auto& code = assignment.centroids[k];
            if (code.size() != index_.code_size) {
                throw Error(ErrorCode::InvalidParam,
                            "build_ivf_scan: centroid " + std::to_string(k) +
                                " has wrong code size");
            }
            index_.quantizer->decode_code(code.data(), centroid_f32.data());
            cast_fp32_to_fp16(centroid_f32.data(), centroid_f16.data(), dim);
            std::memcpy(ring.as<uint8_t>() +
                            static_cast<size_t>(in_block) * vec_bytes,
                        centroid_f16.data(), vec_bytes);
            if (++in_block >= vecs_per_block) {
                write_padded(f, ring.get(),
                             static_cast<size_t>(in_block) * vec_bytes,
                             write_off);
                write_off += static_cast<size_t>(in_block) * vec_bytes;
                in_block = 0;
            }
        }
        if (in_block > 0) {
            write_padded(f, ring.get(),
                         static_cast<size_t>(in_block) * vec_bytes, write_off);
        }
        f.sync();
    }

    // --- 5b. subcentroids.bin: Σ sub-shards × dim × float16_t ---
    // Written only if any shards were sub-sharded (sub_centroids_accum non-empty).
    if (!sub_centroids_accum.empty()) {
        const std::string path = shards_dir + "/subcentroids.bin";
        DirectFile f(path, true);
        write_padded(f, sub_centroids_accum.data(),
                     sub_centroids_accum.size() * sizeof(float16_t), 0);
        f.sync();
        spdlog::info("[sextant] wrote {} ({} sub-centroids, {}KB)",
                     path, sub_centroids_accum.size() / dim,
                     sub_centroids_accum.size() * sizeof(float16_t) / 1024);
    }

    // --- 6. codebook{4,8}.bin: serialized scan PqQuantizer (shared) ---
    {
        const uint64_t magic = (params.quantizer_type == "prq")
            ? kMagicCodebookPRQ4
            : ((scan_bits == 4) ? kMagicCodebook4 : kMagicCodebook8);
        const std::string path = shards_dir +
            ((scan_bits == 4) ? "/codebook4.bin" : "/codebook8.bin");
        std::vector<uint8_t> blob;
        qscan.serialize(blob);
        DirectFile f(path, true);
        SidecarHeader h{};
        fill_header(h, magic, n, dim, uuid);
        write_padded(f, &h, sizeof(h), 0);
        const uint64_t qsize = blob.size();
        write_padded(f, &qsize, sizeof(qsize), sizeof(h));
        if (!blob.empty()) {
            write_padded(f, blob.data(), blob.size(),
                         sizeof(h) + sizeof(qsize));
        }
        f.sync();
    }

    // --- 7. manifest (line-oriented text; IVFScanIndex::read commit point) ---
    {
        // np ∝ √K: the single best-validated scaling law (3 datasets, LID
        // 13-21, recall 0.55-0.99). K/4 over-probes at K>16.
        const uint32_t n_probe = n_probe_default > 0
                                     ? n_probe_default
                                     : std::max(1u, static_cast<uint32_t>(
                                           2.0f * std::sqrt(float(K))));
        const std::string path = shards_dir + "/manifest";
        const std::string tmp = path + ".tmp";
        {
            DirectFile f(tmp, true);
            const uint32_t manifest_nsplits =
                (params.quantizer_type == "prq")
                    ? static_cast<ProductResidualQuantizer&>(qscan).nsplits()
                    : 0;
            // sub_probe_pct: integer percentage (0-100). At search time,
            // sub_np = max(1, ceil(n_sub * pct / 100)) per shard.
            const uint32_t sub_probe_pct = static_cast<uint32_t>(
                sub_probe_fraction * 100.0f + 0.5f);
            std::string commit =
                std::string("ready\n") +
                std::to_string(K) + "\n" +
                std::to_string(dim) + "\n" +
                std::to_string(n_probe) + "\n" +
                std::to_string(m4) + "\n" +
                std::to_string(static_cast<unsigned>(scan_bits)) + "\n" +
                params.quantizer_type + "\n" +
                std::to_string(manifest_nsplits) + "\n" +
                std::to_string(sub_probe_pct) + "\n";
            write_padded(f, commit.data(), commit.size(), 0);
            f.sync();
        }
        std::error_code rec;
        std::filesystem::rename(tmp, path, rec);
        if (rec) {
            throw Error(ErrorCode::IoError,
                        "build_ivf_scan: manifest rename failed: " +
                            rec.message());
        }
        spdlog::info("[sextant] wrote {} (K={} n_probe_default={} m4={})",
                     path, K, n_probe, m4);
    }

    // Free global build buffers + mmap.
    if (vecs_mmap) { ::munmap(vecs_mmap, vecs_mmap_size); }
    if (index_.codes_buffer) { aligned_free(index_.codes_buffer); index_.codes_buffer = nullptr; }
    if (index_.raw_vecs_buffer) { aligned_free(index_.raw_vecs_buffer); index_.raw_vecs_buffer = nullptr; }

    const auto t1 = std::chrono::steady_clock::now();
    spdlog::info("[sextant] build_ivf_scan complete (K={}, {}/{} shards, "
                 "m4={}) in {:.2f}s", K,
                 shards_written.load(std::memory_order_relaxed), K, m4,
                 std::chrono::duration<double>(t1 - t0).count());

    BuildResult result;
    result.index_path = shards_dir;
    result.n_vectors = n;
    result.dim = dim;
    result.R = params.R;
    result.L_build = params.L_build;
    result.pq_m = m4;
    result.pq_bits = 4;
    result.build_time_sec = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

// ===========================================================================
// build_shard_into_ — construct + flush ONE IVF shard via a fresh Builder.
//
// The shard Index arrives with: cloned quantizer, materialized shard-local
// codes/nodes/vecs buffers (local 0..shard_n-1; node row_id = global ID), and
// metadata (count/dim/code_size/node_size=R_shard/path) set. This helper:
//   - wires a fresh VamanaCore at R_shard over the shard buffers;
//   - binds a fresh Builder to the shard Index (reentrant — the top-level
//     Builder's index_ is untouched);
//   - runs the canonical construct loop (construct_into), then the unchanged
//     flush pipeline (snap_entry_points_ → compute_bfs_reorder_ →
//     write_sidecars_). The fresh Builder has empty entry_centroids_, so
//     snap_entry_points_ falls back to core.compute_entry_points().
// The quantizer is NEVER retrained here; `members` only remaps shard-local
// IDs → global row IDs during construct.
// ===========================================================================
void Builder::build_shard_into_(Index& shard_index,
                                 const ResolvedParams& shard_params,
                                 const std::vector<uint32_t>& members,
                                 uint32_t shard_k, uint32_t K,
                                 const std::string& shard_prefix) {
    const uint32_t shard_n = static_cast<uint32_t>(shard_index.count);
    const uint16_t R_shard = shard_params.R;

    // Fresh VamanaCore for this shard at R_shard, bound to the shard buffers.
    VamanaParams vp_shard =
        VamanaParams::from_resolved(shard_params, shard_index.dim, R_shard);
    shard_index.core = std::make_unique<VamanaCore>(vp_shard, *shard_index.quantizer);
    shard_index.core->set_build_codes(shard_index.codes_buffer, shard_n);
    shard_index.core->set_build_nodes(shard_index.nodes_buffer);
    shard_index.core->set_build_vecs(shard_index.raw_vecs_buffer);
    shard_index.core->prepare_for_build(shard_n);

    // FlatNodeStore over the shard buffers so beam_search (used inside
    // insert_build_from_code) goes through the store interface — same setup as
    // the K==1 path in build_partitioned.
    shard_index.flat_store = std::make_unique<FlatNodeStore>(
        shard_index.nodes_buffer, shard_index.codes_buffer,
        shard_index.node_size, shard_index.code_size);
    shard_index.core->set_store(shard_index.flat_store.get());

    // Fresh Builder bound to the shard Index. This is reentrant: the top-level
    // Builder (running build_ivf) keeps its own index_ untouched. The fresh
    // Builder's private flush helpers operate on shard_index exclusively.
    Builder shard_builder(shard_index);

    // Parallel construct via the canonical loop. Row IDs are remapped to the
    // GLOBAL IDs (the on-disk neighbor row_id must be the global vector ID so
    // IVF search results map back to the source dataset).
    const uint32_t shard_lut_sz = shard_index.quantizer->lut_size();
    const uint32_t nthreads = shard_params.num_threads > 0
                                  ? shard_params.num_threads
                                  : std::thread::hardware_concurrency();
    const std::string label =
        "build_ivf shard " + std::to_string(shard_k + 1) + "/" + std::to_string(K);
    {
        VamanaCore::BuildVecLoan build_vec_loan(*shard_index.core);
        shard_builder.construct_into(
            *shard_index.core, shard_n,
            [&members](uint32_t local_id) {
                return static_cast<RowId>(members[local_id]);
            },
            shard_lut_sz, nthreads, label.c_str());
        // raw_vecs_buffer loan released here — no longer needed post-construct.
    }

    // Flush pipeline. Entry-point selection is the IVF Workstream A1 path:
    // sub-clustered k-means medoids (k'=8 × M=1 = 8 entry points per shard),
    // replacing the stride-sampling fallback. This sets index_.sub_centroids /
    // sub_centroid_medoids / sub_medoids_per_cluster on shard_index so
    // write_sidecars_ can emit the `.epc` sidecar. (A3 validated that scaling
    // to M=4/32 entry points trades +0.4pp recall for −6% QPS locally — not
    // worth it when recall is already 0.98+. A2 validated that sub-cluster
    // indexing saves ~nothing vs the default multi-start, so it's off by
    // default; the .epc infra remains for future use.) Run on shard_builder
    // (bound to shard_index) — NOT on this top-level IVF Builder.
    shard_builder.compute_sub_cluster_entry_points_(shard_params,
                                                     /*n_sub=*/8,
                                                     /*medoids_per_cluster=*/1);
    auto bfs = shard_builder.compute_bfs_reorder_(shard_params);
    shard_builder.write_sidecars_(shard_prefix, bfs, shard_params);
    // flat_store / core / buffers freed when shard_index is destroyed by the
    // caller (build_ivf).
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

// ===========================================================================
// compute_sub_cluster_entry_points_ (IVF Workstream A1)
//
// For an IVF shard, run a small k-means (k' = n_sub) on the shard's PQ codes
// to find k' sub-clusters, then snap each sub-centroid to its medoid (the
// shard member nearest the sub-centroid by FP16 L2sq). The k' medoid LOCAL IDs
// become the shard's entry points, and the k' sub-centroids (decoded to FP16)
// are stored on index_.sub_centroids for A2's query-adaptive selection.
//
// This replaces the stride-sampling fallback that build_shard_into_ inherited
// (16 arbitrary evenly-spaced IDs with no awareness of shard structure). The
// sub-clustered medoids are spread across the shard's internal regions, so
// beam_search's approach phase is shorter → fewer distance evals → directly
// attacks the IVF work-multiplicity gap.
//
// The sub-centroids are PQ codes (k-means works in PQ-code space, same as
// partition_codes); they're decoded to FP32 then cast to FP16 for storage so
// A2 can compare them against the query's FP16 vector with simd::l2sq_f16
// (the same primitive used for routing).
//
// Degenerate cases (shard too small for k' sub-clusters): clamps k' to
// min(n_sub, shard_n) and, if that yields <2 sub-clusters, defers to
// snap_entry_points_ (stride sampling) and returns 0.
// ===========================================================================
uint32_t Builder::compute_sub_cluster_entry_points_(const ResolvedParams& params,
                                                     uint32_t n_sub,
                                                     uint32_t medoids_per_cluster) {
    const uint32_t shard_n = static_cast<uint32_t>(index_.count);
    if (shard_n == 0 || !index_.core || !index_.quantizer ||
        !index_.codes_buffer || !index_.raw_vecs_buffer || n_sub == 0) {
        // Stride-sampled fallback (mirrors snap_entry_points_). core may be
        // null only if count==0, in which case there's nothing to do.
        if (index_.core && shard_n > 0) {
            index_.core->compute_entry_points();
        }
        return 0;
    }

    // Clamp k' to the shard size. If fewer than 2 sub-clusters are possible,
    // the sub-clustering adds no value over stride sampling.
    uint32_t k_sub = std::min(n_sub, shard_n);
    if (k_sub < 2) {
        if (index_.core) index_.core->compute_entry_points();
        return 0;
    }

    const uint32_t code_size = index_.code_size;
    const Dim dim = index_.dim;
    PqQuantizer& q = *index_.quantizer;

    // --- k-means on PQ codes (k' = k_sub) ---
    // Initialize centroids as k' distinct random member codes (k-means++-ish
    // seeding isn't worth the cost at k'=8). Refine for a few iterations.
    std::mt19937_64 rng(0xA1A2A3A4ULL);
    std::vector<std::vector<uint8_t>> centroids(k_sub,
        std::vector<uint8_t>(code_size, 0));
    {
        std::vector<uint32_t> picks;
        picks.reserve(k_sub);
        std::uniform_int_distribution<uint32_t> dist(0, shard_n - 1);
        while (picks.size() < k_sub) {
            const uint32_t idx = dist(rng);
            if (std::find(picks.begin(), picks.end(), idx) == picks.end()) {
                picks.push_back(idx);
            }
        }
        for (uint32_t c = 0; c < k_sub; c++) {
            std::memcpy(centroids[c].data(),
                        index_.codes_buffer +
                            static_cast<size_t>(picks[c]) * code_size,
                        code_size);
        }
    }

    auto code_at = [&](uint32_t local_id) {
        return index_.codes_buffer + static_cast<size_t>(local_id) * code_size;
    };

    constexpr uint32_t kIters = 8;
    std::vector<uint32_t> assign(shard_n);
    std::vector<uint32_t> counts(k_sub);
    for (uint32_t iter = 0; iter < kIters; iter++) {
        // Assignment step: nearest centroid by PQ code distance.
        std::fill(counts.begin(), counts.end(), 0);
        for (uint32_t i = 0; i < shard_n; i++) {
            float best = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k_sub; c++) {
                const float d = q.code_distance(code_at(i), centroids[c].data());
                if (d < best) { best = d; best_c = c; }
            }
            assign[i] = best_c;
            counts[best_c]++;
        }
        // Update step: per-cluster PQ-code mean. PQ codes have no closed-form
        // mean, so recompute each centroid as the code minimizing total
        // within-cluster distance — i.e. the medoid. This is slower than a
        // continuous mean but k'×shard_n is small (seconds) and keeps the
        // centroid a valid code (snap-to-medoid below then becomes a no-op for
        // the centroid itself). For efficiency we approximate: pick the member
        // with the smallest sum of distances to its cluster (1-pass scan,
        // sub-sampled when the cluster is large).
        for (uint32_t c = 0; c < k_sub; c++) {
            if (counts[c] == 0) {
                // Reseed empty cluster from a random member.
                std::uniform_int_distribution<uint32_t> dist(0, shard_n - 1);
                std::memcpy(centroids[c].data(), code_at(dist(rng)), code_size);
                continue;
            }
            // Sub-sample the cluster for the medoid search when large: cap the
            // candidate evaluation at 256 members per cluster (at k'=8, that's
            // 2048 code_distance calls/iter — negligible).
            std::vector<uint32_t> members_c;
            members_c.reserve(std::min<uint32_t>(counts[c], 256u));
            const uint32_t stride =
                counts[c] > 256 ? (counts[c] + 255) / 256 : 1;
            for (uint32_t i = 0; i < shard_n && members_c.size() < 256; i++) {
                if (assign[i] == c && (i % stride == 0 || members_c.empty())) {
                    members_c.push_back(i);
                }
            }
            float best_sum = std::numeric_limits<float>::max();
            uint32_t best_medoid = members_c.empty() ? 0 : members_c[0];
            for (uint32_t m : members_c) {
                float sum = 0;
                for (uint32_t j : members_c) {
                    sum += q.code_distance(code_at(m), code_at(j));
                }
                if (sum < best_sum) { best_sum = sum; best_medoid = m; }
            }
            std::memcpy(centroids[c].data(), code_at(best_medoid), code_size);
        }
    }

    // --- Final assignment (in case the last update changed centroids) ---
    std::fill(counts.begin(), counts.end(), 0);
    for (uint32_t i = 0; i < shard_n; i++) {
        float best = std::numeric_limits<float>::max();
        uint32_t best_c = 0;
        for (uint32_t c = 0; c < k_sub; c++) {
            const float d = q.code_distance(code_at(i), centroids[c].data());
            if (d < best) { best = d; best_c = c; }
        }
        assign[i] = best_c;
        counts[best_c]++;
    }

    // --- Decode sub-centroids to FP16 and snap each to its top-M FP16 medoids ---
    // The k-means centroid is already a member code (medoid update above), but
    // to be safe we snap each decoded sub-centroid to its top-M nearest members
    // by FP16 L2sq (M = medoids_per_cluster). This gives k' × M entry points
    // spread across the shard's regions. A2/A3 indexes into them by sub-cluster
    // (k' FP16 distances) instead of scanning all k'×M, so E can grow without
    // search-time penalty.
    const uint32_t M = std::max(1u, medoids_per_cluster);
    index_.sub_centroids.resize(static_cast<size_t>(k_sub) * dim);
    index_.sub_centroid_medoids.assign(
        static_cast<size_t>(k_sub) * M, 0);
    index_.sub_medoids_per_cluster = M;

    // Parallel medoid snap: each sub-cluster scans its own members (the
    // assignment already partitioned them), so the work is balanced when the
    // clusters are balanced. Fall back to scanning all members if a cluster is
    // tiny (frontier safety). Thread c writes disjoint slices:
    //   sub_centroids[c*dim .. (c+1)*dim)
    //   sub_centroid_medoids[c*M .. (c+1)*M)
    std::vector<std::thread> pool;
    auto snap_one = [&](uint32_t c) {
        // Per-thread scratch (snap_one runs on worker threads concurrently).
        std::vector<float> cf32(dim);
        std::vector<float16_t> cf16(dim);
        q.decode_code(centroids[c].data(), cf32.data());
        cast_fp32_to_fp16(cf32.data(), cf16.data(), dim);
        // Write the sub-centroid FP16 into the shared buffer (disjoint slice).
        std::memcpy(index_.sub_centroids.data() + static_cast<size_t>(c) * dim,
                    cf16.data(), dim * sizeof(float16_t));
        // Find the top-M nearest members by FP16 L2sq within this cluster.
        // Use a max-heap of size M (pop the farthest when full) — O(N log M).
        const float16_t* vecs = index_.raw_vecs_buffer;
        struct Med { float d; uint32_t id; };
        auto med_cmp = [](const Med& a, const Med& b) { return a.d < b.d; };
        std::vector<Med> heap;  // max-heap by d (front = worst)
        heap.reserve(M + 1);
        const uint32_t cluster_count = counts[c];
        for (uint32_t i = 0; i < shard_n; i++) {
            if (cluster_count > 0 && assign[i] != c) continue;
            const float d = simd::l2sq_f16(cf16.data(),
                                     vecs + static_cast<size_t>(i) * dim, dim);
            if (heap.size() < M) {
                heap.push_back({d, i});
                std::push_heap(heap.begin(), heap.end(), med_cmp);
            } else if (d < heap.front().d) {
                std::pop_heap(heap.begin(), heap.end(), med_cmp);
                heap.back() = {d, i};
                std::push_heap(heap.begin(), heap.end(), med_cmp);
            }
        }
        // If the cluster had fewer than M members, heap is short — top up with
        // nearest members from ANY cluster so each sub-cluster contributes M
        // medoids (keeps the row-major layout uniform for A2).
        if (heap.size() < M) {
            for (uint32_t i = 0; i < shard_n && heap.size() < M; i++) {
                bool dup = false;
                for (const auto& m : heap) if (m.id == i) { dup = true; break; }
                if (dup) continue;
                const float d = simd::l2sq_f16(cf16.data(),
                                         vecs + static_cast<size_t>(i) * dim, dim);
                heap.push_back({d, i});
                std::push_heap(heap.begin(), heap.end(), med_cmp);
            }
        }
        // Sort ascending by distance so slot 0 is the nearest (primary medoid).
        std::sort(heap.begin(), heap.end(), med_cmp);
        uint32_t* out = index_.sub_centroid_medoids.data() +
                        static_cast<size_t>(c) * M;
        for (uint32_t s = 0; s < M && s < heap.size(); s++) {
            out[s] = heap[s].id;
        }
    };
    for (uint32_t c = 0; c < k_sub; c++) pool.emplace_back(snap_one, c);
    for (auto& t : pool) t.join();

    // Dedup medoids across sub-clusters (two sub-centroids may snap to the same
    // member). The deduped set is what the core uses for the default multi-
    // start path (A2, when active, picks one sub-cluster's slice instead).
    std::vector<uint32_t> medoids = index_.sub_centroid_medoids;
    std::sort(medoids.begin(), medoids.end());
    medoids.erase(std::unique(medoids.begin(), medoids.end()), medoids.end());
    // Top up with unused members if dedup reduced the count below a useful
    // floor (n_search_entry_points). This guards against degenerate shards
    // where many sub-clusters collapse to a few dense regions.
    const uint32_t ep_floor = std::min<uint32_t>(shard_n, 4u);
    if (medoids.size() < ep_floor) {
        for (uint32_t i = 0; i < shard_n && medoids.size() < ep_floor; i++) {
            if (std::find(medoids.begin(), medoids.end(), i) == medoids.end()) {
                medoids.push_back(i);
            }
        }
    }

    index_.core->set_entry_points(std::move(medoids));
    spdlog::info("[sextant] IVF shard entry points: {} sub-clustered medoids "
                 "(k'={}, M={}/{}, shard_n={}, {} unique entry points)",
                 k_sub * M, k_sub, M, medoids_per_cluster, shard_n,
                 index_.core->entry_points().size());
    return k_sub;
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

    // ----- .epc (IVF shard entry-point sub-centroids, A1/A2) -----
    // IVF-only: emitted when build_shard_into_ populated index_.sub_centroids
    // (the merged path leaves it empty, so no .epc is written for K==1). Stores
    // k' × dim FP16 sub-centroids followed by k' × u32 medoid disk-positions
    // (BFS-remapped from build-local IDs). IVFIndex::read loads this so A2 can
    // pick the closest sub-cluster per query and seed beam_search from its
    // medoid via forced_entry_points. Layout:
    //   [SidecarHeader][k_sub × dim × float16_t][k_sub × u32 disk-pos medoids]
    // The header's n_vectors field carries k_sub (the sub-cluster count).
    if (!index_.sub_centroids.empty()) {
        const std::string path = index_path + ".epc";
        const uint32_t k_sub = static_cast<uint32_t>(
            index_.sub_centroids.size() / index_.dim);
        const uint32_t M = index_.sub_medoids_per_cluster;
        const uint32_t n_medoids = static_cast<uint32_t>(
            index_.sub_centroid_medoids.size());
        DirectFile f(path, true);
        SidecarHeader h;
        fill_header(h, kMagicEpc, k_sub, index_.dim, uuid);
        write_padded(f, &h, sizeof(h), 0);

        const size_t vec_bytes = static_cast<size_t>(index_.dim) * sizeof(float16_t);
        const size_t centroid_bytes = static_cast<size_t>(k_sub) * vec_bytes;
        // Payload: [u16 M][k_sub × dim FP16 sub-centroids][k_sub × M u32 medoids]
        // (medoids are BFS-remapped from build-local IDs to disk positions,
        // mirroring .meta's entry-point remap above).
        std::vector<uint8_t> payload;
        payload.reserve(sizeof(uint16_t) + centroid_bytes +
                        n_medoids * sizeof(uint32_t));
        const uint16_t m16 = static_cast<uint16_t>(M);
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(&m16),
                       reinterpret_cast<const uint8_t*>(&m16) + sizeof(m16));
        payload.insert(payload.end(),
                       reinterpret_cast<const uint8_t*>(index_.sub_centroids.data()),
                       reinterpret_cast<const uint8_t*>(index_.sub_centroids.data())
                           + centroid_bytes);
        for (uint32_t i = 0; i < n_medoids; i++) {
            uint32_t disk_id = index_.sub_centroid_medoids[i];
            if (disk_id < n) disk_id = bfs.remap[disk_id];
            payload.insert(payload.end(),
                           reinterpret_cast<const uint8_t*>(&disk_id),
                           reinterpret_cast<const uint8_t*>(&disk_id) + sizeof(disk_id));
        }
        write_padded(f, payload.data(), payload.size(), sizeof(h));
        f.sync();
        spdlog::info("[sextant] wrote {} (k'={} M={} sub-centroids × dim={} FP16 "
                     "+ {} medoid IDs)", path, k_sub, M, index_.dim, n_medoids);
    }

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
    const std::string tmp = path + ".tmp";
    {
        DirectFile f(tmp, true);
        SidecarHeader h;
        fill_header(h, kMagicManifest, index_.count, index_.dim, uuid);
        write_padded(f, &h, sizeof(h), 0);
        // The manifest is the commit point. We record the four sidecar
        // basenames + a "ready" marker.
        std::string commit =
            std::string("ready\n") +
            std::to_string(index_.count) + "\n" +
            std::to_string(index_.dim) + "\n" +
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
                    "Builder::flush: manifest rename failed: " + ec.message());
    }
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
// prepare_routing_codes — pass1 (reservoir + PQ train) + pass2 (encode all N
// into codes_buffer). NO raw_vecs_buffer allocation. The IVF-list-scan path
// mmaps the FP32 source directly for 4-bit encoding; the 13.4 GB FP16 buffer
// (N × dim × 2 at 100M/768-dim) is pure waste there and blocks large builds
// on a 30 GB VM. Graph paths use prepare_codes (which calls this + adds FP16).
// =============================================================================
void Builder::prepare_routing_codes(VectorSource& source,
                                     const ResolvedParams& params) {
    index_.count = source.count();
    index_.dim = source.dim();
    if (index_.count == 0) {
        throw Error(ErrorCode::InvalidParam, "prepare_routing_codes: source is empty");
    }
    if (index_.dim == 0) {
        throw Error(ErrorCode::InvalidParam, "prepare_routing_codes: source has dim=0");
    }
    // pass1 trains the quantizer and sets index_.code_size.
    // compute_entry_points=false: the scan path never builds a graph, so
    // the ~14s serial entry-point k-means is dead work here.
    pass1_sample_and_train(source, params, /*compute_entry_points=*/false);
    index_.node_size = VamanaCore::static_node_size(params.R, index_.code_size);

    // Allocate flat codes buffer (no nodes_buffer — caller is not constructing
    // a graph; no raw_vecs_buffer — scan path mmaps FP32 source directly).
    const size_t codes_bytes = static_cast<size_t>(index_.count) * index_.code_size;
    AlignedBuf codes(kDiskAlign, codes_bytes);
    std::memset(codes.get(), 0, codes_bytes);
    index_.codes_buffer = codes.as<uint8_t>();
    codes.release();
    spdlog::info("[sextant] prepare_routing_codes: allocated {:.1f}MB codes "
                 "(no FP16 buffer — scan path mmaps source)",
                 codes_bytes / 1e6);

    pass2_encode(source, params);
}

// prepare_codes — prepare_routing_codes + load FP16 raw_vecs_buffer.
// Leaves codes_buffer + raw_vecs_buffer populated for graph builds that need
// FP16 for prune/construct (set_build_vecs). The scan path calls
// prepare_routing_codes directly to skip the FP16 materialization.
// =============================================================================
void Builder::prepare_codes(VectorSource& source, const ResolvedParams& params) {
    prepare_routing_codes(source, params);

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
