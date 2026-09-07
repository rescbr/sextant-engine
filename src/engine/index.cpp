#include "sextant/index.hpp"

#include "sidecar_io.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"
#include "sextant/system.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/direct_io.hpp"  // aligned_free
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"
#include "storage/sidecar_header.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

namespace sextant {

using engine_detail::read_exact;

Index::~Index() {
    if (codes_buffer) aligned_free(codes_buffer);
    if (nodes_buffer) aligned_free(nodes_buffer);
    if (raw_vecs_buffer) aligned_free(raw_vecs_buffer);
    if (fp32_vecs_buffer) aligned_free(fp32_vecs_buffer);
}

NodeStore* Index::top_store() const {
    if (memgraph) return memgraph.get();
    if (paged_store) return paged_store.get();
    return flat_store.get();
}

// ===========================================================================
// Index::read — load an index from sidecar files for searching.
// Replaces the former Engine::open / Engine::load_sidecars (Layer 2 Phase E).
// ===========================================================================
std::unique_ptr<Index> Index::read(const std::string& index_path,
                                    uint64_t cache_size_override) {
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
                        "Index::read: index '" + index_path +
                            "' is corrupted — manifest missing but "
                            "orphaned sidecar files found: " +
                            list);
        }
        throw Error(ErrorCode::CorruptIndex,
                    "Index::read: index '" + index_path +
                        "' not found (no .manifest or sidecar files)");
    }

    auto idx = std::make_unique<Index>();
    idx->path = index_path;

    // --- .meta ---
    // Payload layout (written by Builder::write_meta_file):
    //   [u64 quantizer_size][quantizer_bytes]
    //   [u16 entry_point_count][entry_point_count × u32]
    //   [ResolvedParams SPRM block (field-serialized, see config.hpp)]
    ResolvedParams params;
    std::vector<uint32_t> entry_points;
    {
        const std::string path = index_path + ".meta";
        DirectFile f(path, false);
        const uint64_t fsize = f.size();
        if (fsize < sizeof(SidecarHeader)) {
            throw Error(ErrorCode::CorruptIndex,
                        "Index::read: .meta too small");
        }
        SidecarHeader h;
        read_exact(f, &h, sizeof(h), 0);
        if (h.magic != kMagicMeta) {
            throw Error(ErrorCode::CorruptIndex,
                        "Index::read: .meta bad magic");
        }
        idx->count = h.n_vectors;
        idx->dim = h.dim;

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
            throw Error(ErrorCode::CorruptIndex, "Index::read: .meta truncated");
        }
        std::memcpy(&qsize, p, sizeof(qsize));
        p += sizeof(qsize);
        if (sizeof(qsize) + qsize > avail) {
            throw Error(ErrorCode::CorruptIndex,
                        "Index::read: .meta quantizer size out of range");
        }

        // Deserialize the quantizer from the blob.
        idx->quantizer = std::make_unique<PqQuantizer>(
            MetricKind::L2Sq, idx->dim, /*m=*/1, /*bits=*/8);  // placeholder ctor
        idx->quantizer->deserialize(p, static_cast<size_t>(qsize));
        p += qsize;
        idx->code_size = idx->quantizer->code_size();

        // Entry points.
        uint16_t ep_count = 0;
        if (static_cast<size_t>(p - payload.data()) + sizeof(ep_count) > avail) {
            throw Error(ErrorCode::CorruptIndex,
                        "Index::read: .meta entry-point count truncated");
        }
        std::memcpy(&ep_count, p, sizeof(ep_count));
        p += sizeof(ep_count);
        entry_points.reserve(ep_count);
        for (uint16_t i = 0; i < ep_count; i++) {
            uint32_t ep = 0;
            if (static_cast<size_t>(p - payload.data()) + sizeof(ep) > avail) {
                throw Error(ErrorCode::CorruptIndex,
                            "Index::read: .meta entry points truncated");
            }
            std::memcpy(&ep, p, sizeof(ep));
            p += sizeof(ep);
            entry_points.push_back(ep);
        }

        // ResolvedParams block (field-serialized; see config.hpp). Legacy
        // memcpy blobs are rejected by the magic check.
        if (!deserialize_params(p, avail - static_cast<size_t>(p - payload.data()),
                                params)) {
            throw Error(ErrorCode::CorruptIndex,
                        "Index::read: .meta params block corrupt or legacy "
                        "(pre-SPRM memcpy format — rebuild the index)");
        }
    }

    // Remember the loaded params so Builder::flush can persist post-insert state.
    idx->params = params;

    // Verify the .codes and .graph sidecars are present and readable (validate
    // magic), but do NOT load them into flat RAM — the PagedNodeStore reads
    // blocks on demand through the LRU cache.
    {
        const std::string path = index_path + ".codes";
        DirectFile f(path, false);
        SidecarHeader h;
        read_exact(f, &h, sizeof(h), 0);
        if (h.magic != kMagicCodes) {
            throw Error(ErrorCode::CorruptIndex,
                        "Index::read: .codes bad magic");
        }
    }
    {
        const std::string path = index_path + ".graph";
        DirectFile f(path, false);
        SidecarHeader h;
        read_exact(f, &h, sizeof(h), 0);
        if (h.magic != kMagicGraph) {
            throw Error(ErrorCode::CorruptIndex,
                        "Index::read: .graph bad magic");
        }
    }

    // Reconstruct the final node size from the resolved params.
    idx->node_size = VamanaCore::static_node_size(params.R, idx->code_size);

    // Reconstruct the search VamanaCore with the loaded params.
    VamanaParams vparams = VamanaParams::from_resolved(params, idx->dim);
    idx->core = std::make_unique<VamanaCore>(vparams, *idx->quantizer);
    idx->core->prepare_for_build(static_cast<uint32_t>(idx->count));

    // Install a PagedNodeStore over the sidecar files. The core reads nodes
    // and codes on demand through the LRU cache — idle RAM stays ~1MB (.meta)
    // regardless of index size.
    //
    // Cache size: min(graph_size × 0.50, physical_ram × 0.35).
    {
        const uint64_t graph_size =
            static_cast<uint64_t>(idx->count) * idx->node_size;
        // codes_size: n × code_size (the .codes sidecar payload).
        const uint64_t codes_size =
            static_cast<uint64_t>(idx->count) * idx->code_size;
        const uint64_t total_index_size = graph_size + codes_size;
        const uint64_t phys_ram = sextant::physical_ram_bytes();
        uint64_t cache_bytes;
        if (cache_size_override > 0) {
            cache_bytes = cache_size_override;
        } else if (phys_ram > 0) {
            const uint64_t ram_budget = phys_ram * 35 / 100;
            if (total_index_size <= ram_budget) {
                // Both files fit in the RAM budget — cache everything.
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

         idx->paged_store = std::make_unique<PagedNodeStore>(
             index_path + ".graph", index_path + ".codes",
             idx->node_size, idx->code_size,
             std::max(1u, std::thread::hardware_concurrency()),
             cache_bytes);
         idx->cache_size_bytes = cache_bytes;
     }

    // Build a MemGraph over the sidecar files: cache the entry-point BFS
    // neighborhood (default 3 hops) in RAM, delegate cold nodes to the
    // PagedNodeStore. The entry points were read from .meta above.
    {
        // Fall back to a deterministic single entry point if .meta had none.
        std::vector<uint32_t> eps = entry_points;
        if (eps.empty() && idx->count > 0) {
            eps.push_back(0);
        }
        idx->memgraph = std::make_unique<MemGraph>(
            index_path + ".graph", index_path + ".codes",
            idx->node_size, idx->code_size,
            static_cast<uint32_t>(idx->count),
            eps, /*num_hops=*/3);
        idx->memgraph->set_backing(idx->paged_store.get());

        const uint64_t cached_bytes =
            static_cast<uint64_t>(idx->memgraph->cached_count()) *
            (idx->node_size + idx->code_size);
        spdlog::info("[sextant] MemGraph: {} nodes cached ({:.1f}MB), 3 hops",
                     idx->memgraph->cached_count(), cached_bytes / 1e6);
    }
    idx->core->set_store(idx->memgraph.get());

    // Restore entry points from .meta if present; otherwise compute a
    // deterministic fallback set. Keep a copy on idx->entry_points so Searcher
    // can construct per-worker VamanaCores with the same entry points.
    if (entry_points.empty()) {
        idx->core->compute_entry_points();
        entry_points = idx->core->entry_points();
    }
    idx->entry_points = entry_points;
    idx->core->set_entry_points(std::move(entry_points));

    spdlog::info("[sextant] opened index '{}' (n={} dim={})", index_path,
                 idx->count, idx->dim);
    return idx;
}

}  // namespace sextant
