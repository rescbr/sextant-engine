// Search pipeline — open() + search() for the Engine facade.
//
// Phase 1 strategy: after build, the flat nodes_buffer_ + codes_buffer_ stay
// resident. After open(), the .graph and .codes sidecars are read back into
// those same buffers. The VamanaCore reads from them transparently via
// node_ptr(). The LRU-paged search path is deferred to a later phase.

#include "search.hpp"
#include "sidecar_io.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/direct_io.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"
#include "storage/sidecar_header.hpp"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

namespace sextant {

using engine_detail::fill_header;
using engine_detail::read_exact;
using engine_detail::write_padded;

// ===========================================================================
// search()
// ===========================================================================

std::vector<Candidate> Engine::search(const float* query, uint32_t k,
                                      const SearchConfig& config) {
    if (!opened_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::search: index not opened");
    }
    if (!quantizer_ || !core_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::search: quantizer/core not initialized");
    }
    if (k == 0) {
        return {};
    }

    // Preprocess the query into an ADC LUT.
    const uint32_t lut_sz = quantizer_->lut_size();
    std::vector<float> lut(lut_sz > 0 ? lut_sz : 1, 0.0f);
    if (lut_sz > 0) {
        quantizer_->preprocess_query(query, lut.data());
    }

    // VamanaCore::search resolves internal_ids → row_ids for us.
    // Convert the query to FP16 for the hybrid FP16+PQ distance path
    // (MemGraph ball nodes use l2sq_f16; the rest use PQ lut_distance).
    std::vector<float16_t> query_fp16(dim_);
    for (uint32_t d = 0; d < dim_; d++) {
        query_fp16[d] = static_cast<float16_t>(query[d]);
    }
    auto results =
        core_->search(lut.data(), k, config.L_search, config.io_limit,
                      query_fp16.data());

    // Adaptive cache rebalance (paged mode only). Cheap relaxed-atomic add per
    // search; the rare resize is CAS-guarded so only one thread runs it.
    if (paged_store_ && cache_rebalance_enabled_) {
        maybe_rebalance_();
    }
    return results;
}

// ===========================================================================
// open()
// ===========================================================================

void Engine::open(const std::string& index_path) {
    index_path_ = index_path;

    // Validate the atomic commit point: the .manifest is written LAST via
    // temp+rename, so its presence means a build completed. If it's absent
    // but sidecar files linger, the index is corrupted (e.g. crashed build).
    std::error_code ec;
    const bool manifest_exists =
        std::filesystem::exists(index_path + ".manifest", ec);
    if (!manifest_exists) {
        std::vector<std::string> orphans;
        for (const char* suf : {".graph", ".codes", ".meta"}) {
            if (std::filesystem::exists(index_path + suf, ec)) {
                orphans.emplace_back(suf);
            }
        }
        if (!orphans.empty()) {
            std::string list = orphans[0];
            for (size_t i = 1; i < orphans.size(); ++i) {
                list += ", " + orphans[i];
            }
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: index '" + index_path +
                            "' is corrupted — manifest missing but "
                            "orphaned sidecar files found: " +
                            list);
        }
        throw Error(ErrorCode::CorruptIndex,
                    "Engine::open: index '" + index_path +
                        "' not found (no .manifest or sidecar files)");
    }

    // Release any previous buffers/stores.
    if (codes_buffer_) { aligned_free(codes_buffer_); codes_buffer_ = nullptr; }
    if (nodes_buffer_) { aligned_free(nodes_buffer_); nodes_buffer_ = nullptr; }
    flat_store_.reset();
    paged_store_.reset();
    memgraph_.reset();
    core_.reset();
    quantizer_.reset();
    params_loaded_ = false;

    load_sidecars();
    opened_ = true;
    spdlog::info("[sextant] opened index '{}' (n={} dim={})", index_path,
                 count_, dim_);
}

// ===========================================================================
// load_sidecars()
//
// Reads .meta (quantizer + entry points + params), then .graph and .codes into
// the flat buffers, and reconstructs the VamanaCore.
// ===========================================================================

void Engine::load_sidecars() {
    // --- .meta ---
    // Payload layout (written by write_sidecars_):
    //   [u64 quantizer_size][quantizer_bytes]
    //   [u16 entry_point_count][entry_point_count × u32]
    //   [ResolvedParams POD block]
    std::vector<ResolvedParams> params_holder(1);
    ResolvedParams& params = params_holder[0];
    std::vector<uint32_t> entry_points;

    {
        const std::string path = index_path_ + ".meta";
        DirectFile f(path, false);
        const uint64_t fsize = f.size();
        if (fsize < sizeof(SidecarHeader)) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .meta too small");
        }
        SidecarHeader h;
        read_exact(f, &h, sizeof(h), 0);
        if (h.magic != kMagicMeta) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .meta bad magic");
        }
        count_ = h.n_vectors;
        dim_ = h.dim;

        // Read the payload (everything after the header). The on-disk size was
        // padded up to kDiskAlign; we only need the leading payload bytes, but
        // we don't know that length a priori, so read a generous cap.
        const size_t payload_avail =
            static_cast<size_t>(fsize) - sizeof(SidecarHeader);
        std::vector<uint8_t> payload(payload_avail);
        read_exact(f, payload.data(), payload_avail, sizeof(h));

        const uint8_t* p = payload.data();
        const size_t avail = payload.size();

        uint64_t qsize = 0;
        if (avail < sizeof(qsize)) {
            throw Error(ErrorCode::CorruptIndex, "Engine::open: .meta truncated");
        }
        std::memcpy(&qsize, p, sizeof(qsize));
        p += sizeof(qsize);
        if (sizeof(qsize) + qsize > avail) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .meta quantizer size out of range");
        }

        // Deserialize the quantizer from the blob.
        quantizer_ = std::make_unique<PqQuantizer>(
            MetricKind::L2Sq, dim_, /*m=*/1, /*bits=*/8);  // placeholder ctor
        quantizer_->deserialize(p, static_cast<size_t>(qsize));
        p += qsize;
        code_size_ = quantizer_->code_size();

        // Entry points.
        uint16_t ep_count = 0;
        if (static_cast<size_t>(p - payload.data()) + sizeof(ep_count) > avail) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .meta entry-point count truncated");
        }
        std::memcpy(&ep_count, p, sizeof(ep_count));
        p += sizeof(ep_count);
        entry_points.reserve(ep_count);
        for (uint16_t i = 0; i < ep_count; i++) {
            uint32_t ep = 0;
            if (static_cast<size_t>(p - payload.data()) + sizeof(ep) > avail) {
                throw Error(ErrorCode::CorruptIndex,
                            "Engine::open: .meta entry points truncated");
            }
            std::memcpy(&ep, p, sizeof(ep));
            p += sizeof(ep);
            entry_points.push_back(ep);
        }

        // ResolvedParams POD block.
        if (static_cast<size_t>(p - payload.data()) + sizeof(params) > avail) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .meta params truncated");
        }
        std::memcpy(&params, p, sizeof(params));
    }

    // Remember the loaded params so flush() can persist post-insert state.
    loaded_params_ = params;
    params_loaded_ = true;

    // Verify the .codes and .graph sidecars are present and readable (validate
    // magic), but do NOT load them into flat RAM — the PagedNodeStore reads
    // blocks on demand through the LRU cache.
    {
        const std::string path = index_path_ + ".codes";
        DirectFile f(path, false);
        SidecarHeader h;
        read_exact(f, &h, sizeof(h), 0);
        if (h.magic != kMagicCodes) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .codes bad magic");
        }
    }
    {
        const std::string path = index_path_ + ".graph";
        DirectFile f(path, false);
        SidecarHeader h;
        read_exact(f, &h, sizeof(h), 0);
        if (h.magic != kMagicGraph) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .graph bad magic");
        }
    }

    // Reconstruct the final node size from the resolved params.
    node_size_ = VamanaCore::static_node_size(
        params.R, params.inline_pq_count,
        code_size_);

    // --- Reconstruct the VamanaCore with the loaded params ---
    // Search core uses the resolved inline_pq_count (nodes carry inline codes).
    VamanaParams vparams =
        VamanaParams::from_resolved(params, dim_, 0, params.inline_pq_count);
    core_ = std::make_unique<VamanaCore>(vparams, *quantizer_);
    core_->prepare_for_build(static_cast<uint32_t>(count_));

    // Install a PagedNodeStore over the sidecar files. The core reads nodes
    // and codes on demand through the LRU cache — idle RAM stays ~1MB (.meta)
    // regardless of index size.
    //
    // Cache size per Issue 5: min(graph_size × 0.50, physical_ram × 0.35).
    // The graph_size is n × node_size (final layout). physical_ram from sysconf.
    {
        const uint64_t graph_size =
            static_cast<uint64_t>(count_) * node_size_;
        // codes_size: n × code_size (the .codes sidecar payload). Previously
        // the budget considered only graph_size, under-provisioning the code
        // cache — now both files are accounted for.
        const uint64_t codes_size =
            static_cast<uint64_t>(count_) * code_size_;
        const uint64_t total_index_size = graph_size + codes_size;
        uint64_t phys_ram = 0;
#ifdef __APPLE__
        // sysctl hw.memsize
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        uint64_t memsize = 0;
        size_t len = sizeof(memsize);
        if (sysctl(mib, 2, &memsize, &len, nullptr, 0) == 0) {
            phys_ram = memsize;
        }
#else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            phys_ram = static_cast<uint64_t>(pages) * page_size;
        }
#endif
        uint64_t cache_bytes;
        if (cache_size_override_ > 0) {
            cache_bytes = cache_size_override_;
        } else if (phys_ram > 0) {
            const uint64_t ram_budget = phys_ram * 35 / 100;
            if (total_index_size <= ram_budget) {
                // Both files fit in the RAM budget — cache everything. No
                // reason to page when it all fits.
                cache_bytes = total_index_size;
            } else {
                // Index exceeds RAM budget — cache the working set fraction.
                cache_bytes = std::min(total_index_size / 2, ram_budget);
            }
        } else {
            cache_bytes = total_index_size;
        }
        // Clamp to a minimum of 16MB so tiny indices still have enough blocks.
        cache_bytes = std::max<uint64_t>(cache_bytes, 16ull * 1024 * 1024);

        spdlog::info("[sextant] search cache: {:.1f}MB (graph={:.1f}MB, "
                     "codes={:.1f}MB, phys_ram={:.1f}MB)",
                     cache_bytes / 1e6, graph_size / 1e6,
                     codes_size / 1e6, phys_ram / 1e6);

        paged_store_ = std::make_unique<PagedNodeStore>(
            index_path_ + ".graph", index_path_ + ".codes",
            node_size_, code_size_,
            std::max(1u, std::thread::hardware_concurrency()),
            cache_bytes);
    }

    // Build a MemGraph over the sidecar files: cache the entry-point BFS
    // neighborhood (default 3 hops) in RAM, delegate cold nodes to the
    // PagedNodeStore. The entry points were read from .meta above.
    {
        // Fall back to a deterministic single entry point if .meta had none.
        std::vector<uint32_t> eps = entry_points;
        if (eps.empty() && count_ > 0) {
            eps.push_back(0);
        }
        memgraph_ = std::make_unique<MemGraph>(
            index_path_ + ".graph", index_path_ + ".codes",
            index_path_ + ".vecs",   // FP16 ball sidecar (optional — PQ fallback)
            node_size_, code_size_,
            static_cast<uint32_t>(count_), dim_,
            eps, /*num_hops=*/3);
        memgraph_->set_backing(paged_store_.get());

        const uint64_t cached_bytes =
            static_cast<uint64_t>(memgraph_->cached_count()) *
            (node_size_ + code_size_);
        spdlog::info("[sextant] MemGraph: {} nodes cached ({:.1f}MB), 3 hops",
                     memgraph_->cached_count(), cached_bytes / 1e6);
    }
    core_->set_store(memgraph_.get());

    // Restore entry points from .meta if present; otherwise compute a
    // deterministic fallback set. `entry_points` is moved into the core here;
    // it must not be used after this point (its last prior use was the
    // MemGraph fallback copy above).
    if (!entry_points.empty()) {
        core_->set_entry_points(std::move(entry_points));
    } else {
        core_->compute_entry_points();
    }
}

uint64_t Engine::cache_graph_reads() const {
    return paged_store_ ? paged_store_->graph_reads() : 0;
}

uint64_t Engine::cache_code_reads() const {
    return paged_store_ ? paged_store_->code_reads() : 0;
}

Engine::AdmissionStats Engine::cache_admission_stats() const {
    if (!paged_store_) return {};
    const auto cs = paged_store_->cache_stats();
    return {
        cs.graph.hits_window + cs.code.hits_window,
        cs.graph.hits_probation + cs.code.hits_probation,
        cs.graph.hits_protected + cs.code.hits_protected,
        cs.graph.misses + cs.code.misses,
        cs.graph.evictions_admitted + cs.code.evictions_admitted,
        cs.graph.evictions_rejected + cs.code.evictions_rejected,
    };
}

void Engine::rebalance_caches() {
    if (paged_store_) {
        paged_store_->maybe_rebalance_caches();
    }
}

void Engine::maybe_rebalance_() {
    const uint64_t n = search_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n < rebalance_cadence_) return;
    // CAS: only one thread rebalances at a time. Losers bail (they'll retry
    // next cadence window; the counter is monotonic so no double-fire).
    bool expected = false;
    if (!rebalancing_.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel)) {
        return;
    }
    const double prev_frac = paged_store_->graph_cache_fraction();
    paged_store_->maybe_rebalance_caches();   // may call BlockCache::resize
    const double new_frac = paged_store_->graph_cache_fraction();
    rebalancing_.store(false, std::memory_order_release);
    // Adaptive cadence: if the split didn't move, back off (steady state).
    // If it moved, reset to initial (workload shifting — stay responsive).
    if (std::abs(new_frac - prev_frac) < 1e-6) {
        rebalance_cadence_ = std::min(kRebalanceCadenceMax,
                                      rebalance_cadence_ * 2);
    } else {
        rebalance_cadence_ = kRebalanceCadenceInitial;
    }
}

uint32_t Engine::memgraph_cached_count() const {
    return memgraph_ ? memgraph_->cached_count() : 0;
}

uint64_t Engine::tl_hits() const {
    return paged_store_ ? paged_store_->tl_hits() : 0;
}

uint64_t Engine::tl_misses() const {
    return paged_store_ ? paged_store_->tl_misses() : 0;
}

}  // namespace sextant
