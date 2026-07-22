#include "sextant/ivf_index.hpp"

#include "sidecar_io.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"
#include "storage/sidecar_header.hpp"
#include "util/fp16.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace sextant {

using engine_detail::read_exact;

namespace {

/// Parse the IVF manifest. We deliberately avoid a JSON dependency: the
/// manifest is a tiny line-oriented text file we control on both sides.
///   line 1: "ready"
///   line 2: K
///   line 3: dim
///   line 4: n_probe_default
///   line 5: closure_factor
struct IVFManifest {
    uint32_t K = 0;
    Dim dim = 0;
    uint32_t n_probe_default = 1;
    float closure_factor = 1.0f;
};

IVFManifest read_ivf_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFIndex::read: cannot open manifest '" + path + "'");
    }
    std::string ready_line;
    std::getline(f, ready_line);
    if (ready_line != "ready") {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFIndex::read: manifest not ready (first line: '" +
                        ready_line + "')");
    }
    IVFManifest m;
    std::string line;
    auto read_u32 = [&](uint32_t& out) {
        if (!std::getline(f, line)) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFIndex::read: manifest truncated");
        }
        out = static_cast<uint32_t>(std::stoul(line));
    };
    auto read_f32 = [&](float& out) {
        if (!std::getline(f, line)) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFIndex::read: manifest truncated");
        }
        out = std::stof(line);
    };
    read_u32(m.K);
    read_u32(m.dim);
    read_u32(m.n_probe_default);
    read_f32(m.closure_factor);
    if (m.K == 0 || m.dim == 0) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFIndex::read: manifest has K=0 or dim=0");
    }
    return m;
}

}  // namespace

IVFIndex::~IVFIndex() = default;

std::unique_ptr<IVFIndex> IVFIndex::read(const std::string& shards_dir,
                                         uint64_t cache_size_override) {
    // Validate the directory + manifest commit point.
    std::error_code ec;
    if (!std::filesystem::is_directory(shards_dir, ec)) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFIndex::read: '" + shards_dir +
                        "' is not a directory");
    }
    const std::string manifest_path = shards_dir + "/manifest";
    if (!std::filesystem::exists(manifest_path, ec)) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFIndex::read: manifest missing in '" + shards_dir + "'");
    }

    const auto manif = read_ivf_manifest(manifest_path);

    auto ivf = std::make_unique<IVFIndex>();
    ivf->path = shards_dir;
    ivf->K = manif.K;
    ivf->dim = manif.dim;
    ivf->n_probe_default = manif.n_probe_default;
    ivf->closure_factor = manif.closure_factor;

    // --- centroids.bin (K × dim × float16_t, raw, no header) ---
    {
        const std::string path = shards_dir + "/centroids.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFIndex::read: cannot open '" + path + "'");
        }
        const size_t expect =
            static_cast<size_t>(manif.K) * manif.dim * sizeof(float16_t);
        ivf->centroids.resize(static_cast<size_t>(manif.K) * manif.dim);
        f.read(reinterpret_cast<char*>(ivf->centroids.data()),
               static_cast<std::streamsize>(expect));
        if (!f || static_cast<size_t>(f.gcount()) != expect) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFIndex::read: centroids.bin short read");
        }
    }

    // --- Open each shard via Index::read ---
    // Shard paths are zero-padded to 4 digits: shard_0001 .. shard_K.
    ivf->shards.resize(manif.K);
    ivf->shard_sub_centroids.resize(manif.K);
    ivf->shard_sub_medoids.resize(manif.K);
    for (uint32_t k = 0; k < manif.K; k++) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "/shard_%04u", k + 1);
        const std::string shard_prefix = shards_dir + buf;
        // The shard may legitimately be absent if it was empty at build
        // time (k-means empty cluster that wasn't reseeded). Skip those.
        if (!std::filesystem::exists(shard_prefix + ".manifest", ec)) {
            spdlog::warn("[sextant] IVFIndex::read: shard {} missing, skipping",
                         k);
            continue;
        }
        ivf->shards[k] = Index::read(shard_prefix, cache_size_override);

        // --- Load the shard's .epc (entry-point sub-centroids, A1/A2/A3) ---
        // Optional sidecar: present when the shard was built with sub-clustered
        // entry points. Layout:
        //   [SidecarHeader][u16 M][k_sub × dim × float16_t][k_sub × M u32 disk-pos IDs]
        // When absent or unreadable, the shard falls back to its standard
        // entry points (multi-start selection in beam_search).
        const std::string epc_path = shard_prefix + ".epc";
        if (std::filesystem::exists(epc_path, ec)) {
            DirectFile f(epc_path, false);
            const uint64_t fsize = f.size();
            if (fsize >= sizeof(SidecarHeader)) {
                SidecarHeader h;
                read_exact(f, &h, sizeof(h), 0);
                if (h.magic == kMagicEpc && h.dim == manif.dim) {
                    const uint32_t k_sub = static_cast<uint32_t>(h.n_vectors);
                    const size_t vec_bytes =
                        static_cast<size_t>(manif.dim) * sizeof(float16_t);
                    const size_t centroid_bytes =
                        static_cast<size_t>(k_sub) * vec_bytes;
                    const size_t avail =
                        static_cast<size_t>(fsize) - sizeof(SidecarHeader);
                    if (avail >= sizeof(uint16_t) + centroid_bytes &&
                        k_sub > 0) {
                        // Read M, then the centroids + medoid block.
                        std::vector<uint8_t> raw(avail);
                        read_exact(f, raw.data(), avail, sizeof(h));
                        uint16_t m16 = 0;
                        std::memcpy(&m16, raw.data(), sizeof(m16));
                        const uint32_t M = std::max<uint32_t>(1u, m16);
                        const size_t medoid_bytes =
                            static_cast<size_t>(k_sub) * M * sizeof(uint32_t);
                        const size_t need =
                            sizeof(uint16_t) + centroid_bytes + medoid_bytes;
                        if (avail >= need) {
                            ivf->shard_sub_centroids[k].resize(
                                static_cast<size_t>(k_sub) * manif.dim);
                            std::memcpy(
                                ivf->shard_sub_centroids[k].data(),
                                raw.data() + sizeof(uint16_t), centroid_bytes);
                            ivf->shard_sub_medoids[k].resize(
                                static_cast<size_t>(k_sub) * M);
                            std::memcpy(
                                ivf->shard_sub_medoids[k].data(),
                                raw.data() + sizeof(uint16_t) + centroid_bytes,
                                medoid_bytes);
                        } else {
                            spdlog::warn("[sextant] IVFIndex::read: shard {} .epc "
                                         "truncated (M={}), ignoring", k, M);
                        }
                    } else {
                        spdlog::warn("[sextant] IVFIndex::read: shard {} .epc "
                                     "truncated, ignoring", k);
                    }
                } else if (h.magic == kMagicEpc) {
                    spdlog::warn("[sextant] IVFIndex::read: shard {} .epc dim "
                                 "mismatch ({} != {}), ignoring",
                                 k, h.dim, manif.dim);
                }
            }
        }
    }

    const uint32_t present =
        static_cast<uint32_t>(std::count_if(ivf->shards.begin(),
                                             ivf->shards.end(),
                                             [](const auto& s) { return !!s; }));
    spdlog::info("[sextant] opened IVF index '{}' (K={}, {} shards present, "
                 "dim={}, n_probe_default={})",
                 shards_dir, manif.K, present, manif.dim, manif.n_probe_default);

    return ivf;
}

}  // namespace sextant
