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

#include <chrono>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>
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
