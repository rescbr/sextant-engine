// Search pipeline — open() + search() for the Engine facade.
//
// Phase 1 strategy: after build, the flat index_->nodes_buffer + index_->codes_buffer stay
// resident. After open(), the .graph and .codes sidecars are read back into
// those same buffers. The VamanaCore reads from them transparently via
// node_ptr(). The LRU-paged search path is deferred to a later phase.

#include "search.hpp"
#include "sidecar_io.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"
#include "sextant/system.hpp"

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

namespace sextant {

using engine_detail::fill_header;
using engine_detail::read_exact;
using engine_detail::write_padded;

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

    // Construct a fresh Index. Any previous Index state is dropped (Engine::open
    // is destructive by design — see architecture audit §10.2).
    index_ = std::make_unique<Index>();
    index_->path = index_path;

    load_sidecars();
    searcher_ = std::make_unique<Searcher>(*index_);
    searcher_->set_cache_rebalance_enabled(cache_rebalance_enabled_);
    opened_ = true;
    spdlog::info("[sextant] opened index '{}' (n={} dim={})", index_path,
                 index_->count, index_->dim);
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
    //   [u64 index_->quantizersize][index_->quantizerbytes]
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
        index_->count = h.n_vectors;
        index_->dim = h.dim;

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
        index_->quantizer = std::make_unique<PqQuantizer>(
            MetricKind::L2Sq, index_->dim, /*m=*/1, /*bits=*/8);  // placeholder ctor
        index_->quantizer->deserialize(p, static_cast<size_t>(qsize));
        p += qsize;
        index_->code_size = index_->quantizer->code_size();

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
    index_->params = params;

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
    index_->node_size = VamanaCore::static_node_size(params.R, index_->code_size);

    // Reconstruct the search VamanaCore with the loaded params.
    VamanaParams vparams = VamanaParams::from_resolved(params, index_->dim);
    index_->core = std::make_unique<VamanaCore>(vparams, *index_->quantizer);
    index_->core->prepare_for_build(static_cast<uint32_t>(index_->count));

    // Install a PagedNodeStore over the sidecar files. The core reads nodes
    // and codes on demand through the LRU cache — idle RAM stays ~1MB (.meta)
    // regardless of index size.
    //
    // Cache size per Issue 5: min(graph_size × 0.50, physical_ram × 0.35).
    // The graph_size is n × node_size (final layout). physical_ram from sysconf.
    {
        const uint64_t graph_size =
            static_cast<uint64_t>(index_->count) * index_->node_size;
        // codes_size: n × code_size (the .codes sidecar payload). Previously
        // the budget considered only graph_size, under-provisioning the code
        // cache — now both files are accounted for.
        const uint64_t codes_size =
            static_cast<uint64_t>(index_->count) * index_->code_size;
        const uint64_t total_index_size = graph_size + codes_size;
        const uint64_t phys_ram = sextant::physical_ram_bytes();
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

        index_->paged_store = std::make_unique<PagedNodeStore>(
            index_path_ + ".graph", index_path_ + ".codes",
            index_->node_size, index_->code_size,
            std::max(1u, std::thread::hardware_concurrency()),
            cache_bytes);
    }

    // Build a MemGraph over the sidecar files: cache the entry-point BFS
    // neighborhood (default 3 hops) in RAM, delegate cold nodes to the
    // PagedNodeStore. The entry points were read from .meta above.
    {
        // Fall back to a deterministic single entry point if .meta had none.
        std::vector<uint32_t> eps = entry_points;
        if (eps.empty() && index_->count > 0) {
            eps.push_back(0);
        }
        index_->memgraph = std::make_unique<MemGraph>(
            index_path_ + ".graph", index_path_ + ".codes",
            index_path_ + ".ball",   // FP16 ball sidecar (optional — PQ fallback)
            index_->node_size, index_->code_size,
            static_cast<uint32_t>(index_->count), index_->dim,
            eps, /*num_hops=*/3);
        index_->memgraph->set_backing(index_->paged_store.get());

        const uint64_t cached_bytes =
            static_cast<uint64_t>(index_->memgraph->cached_count()) *
            (index_->node_size + index_->code_size);
        spdlog::info("[sextant] MemGraph: {} nodes cached ({:.1f}MB), 3 hops",
                     index_->memgraph->cached_count(), cached_bytes / 1e6);
    }
    index_->core->set_store(index_->memgraph.get());

    // Restore entry points from .meta if present; otherwise compute a
    // deterministic fallback set. `entry_points` is moved into the core here;
    // it must not be used after this point (its last prior use was the
    // MemGraph fallback copy above).
    if (!entry_points.empty()) {
        index_->core->set_entry_points(std::move(entry_points));
    } else {
        index_->core->compute_entry_points();
    }
}
}  // namespace sextant
