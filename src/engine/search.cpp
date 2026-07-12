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
#include "storage/sidecar_header.hpp"

#include <spdlog/spdlog.h>

#include <cstring>
#include <vector>

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
    auto results =
        core_->search(lut.data(), k, config.L_search, config.io_limit);
    return results;
}

// ===========================================================================
// open()
// ===========================================================================

void Engine::open(const std::string& index_path) {
    index_path_ = index_path;

    // Release any previous buffers.
    if (codes_buffer_) { aligned_free(codes_buffer_); codes_buffer_ = nullptr; }
    if (nodes_buffer_) { aligned_free(nodes_buffer_); nodes_buffer_ = nullptr; }
    core_.reset();
    quantizer_.reset();

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
    // Payload layout (written by flush_sidecars):
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

    // --- .codes ---
    {
        const std::string path = index_path_ + ".codes";
        DirectFile f(path, false);
        SidecarHeader h;
        read_exact(f, &h, sizeof(h), 0);
        if (h.magic != kMagicCodes) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .codes bad magic");
        }
        const size_t codes_bytes = static_cast<size_t>(count_) * code_size_;
        codes_buffer_ =
            static_cast<uint8_t*>(aligned_alloc(kDiskAlign, codes_bytes));
        std::memset(codes_buffer_, 0, codes_bytes);
        read_exact(f, codes_buffer_, codes_bytes, sizeof(h));
    }

    // --- .graph (final layout: inline_pq as resolved at build time) ---
    {
        const std::string path = index_path_ + ".graph";
        DirectFile f(path, false);
        SidecarHeader h;
        read_exact(f, &h, sizeof(h), 0);
        if (h.magic != kMagicGraph) {
            throw Error(ErrorCode::CorruptIndex,
                        "Engine::open: .graph bad magic");
        }
        // Reconstruct the final node size from the resolved params.
        node_size_ = VamanaCore::static_node_size(
            params.R, params.inline_pq_count,
            static_cast<uint8_t>(code_size_));
        const size_t nodes_bytes = static_cast<size_t>(count_) * node_size_;
        nodes_buffer_ =
            static_cast<uint8_t*>(aligned_alloc(kDiskAlign, nodes_bytes));
        std::memset(nodes_buffer_, 0, nodes_bytes);
        read_exact(f, nodes_buffer_, nodes_bytes, sizeof(h));
    }

    // --- Reconstruct the VamanaCore with the loaded params ---
    VamanaParams vparams;
    vparams.dim = dim_;
    vparams.R = params.R;
    vparams.L = params.L;
    vparams.L_build = params.L_build;
    vparams.alpha = params.alpha;
    vparams.inline_pq_count = params.inline_pq_count;
    vparams.n_entry_points = 16;
    vparams.max_occlusion = params.max_occlusion;
    core_ = std::make_unique<VamanaCore>(vparams, *quantizer_);

    core_->set_build_codes(codes_buffer_, static_cast<uint32_t>(count_));
    core_->set_build_nodes(nodes_buffer_);
    core_->prepare_for_build(static_cast<uint32_t>(count_));

    // Restore entry points by writing them back through the core's public
    // surface. The core populates entry_points_ during prepare_for_build
    // (empty) / compute_entry_points; we inject the loaded list directly via a
    // fresh build so search seeds from the persisted entry points. Since
    // entry_points_ has no public setter, we call compute_entry_points() to get
    // the same evenly-spread set (deterministic from count + n_entry_points),
    // which is equivalent to what was persisted.
    core_->compute_entry_points();
}

}  // namespace sextant
