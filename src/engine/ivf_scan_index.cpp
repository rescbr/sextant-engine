#include "sextant/ivf_scan_index.hpp"

#include "engine/sidecar_io.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/code_stream.hpp"
#include "storage/sidecar_header.hpp"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sextant {

IVFScanIndex::~IVFScanIndex() = default;

namespace {

/// Read the line-oriented IVF-scan manifest. Format (one token per line):
///   ready
///   <K>
///   <dim>
///   <n_probe_default>
///   <m4>
/// Returns false if the file is missing or malformed.
bool read_scan_manifest(const std::string& path, uint32_t& K, Dim& dim,
                        uint32_t& n_probe_default, uint16_t& m4) {
    std::ifstream f(path);
    if (!f) return false;
    std::string tok;
    if (!std::getline(f, tok)) return false;
    if (tok != "ready") {
        spdlog::warn("[sextant] IVFScanIndex: manifest not ready (first line='{}')",
                     tok);
        return false;
    }
    auto read_line = [&](auto& out) -> bool {
        std::string line;
        if (!std::getline(f, line)) return false;
        std::istringstream iss(line);
        iss >> out;
        return !iss.fail();
    };
    if (!read_line(K)) return false;
    if (!read_line(dim)) return false;
    if (!read_line(n_probe_default)) return false;
    if (!read_line(m4)) return false;
    return true;
}

/// Read the shared 4-bit codebook (a serialized PqQuantizer blob prefixed by a
/// SidecarHeader). Reconstructs the quantizer in-place.
void read_codebook(const std::string& path, std::unique_ptr<PqQuantizer>& out,
                   Dim dim, uint16_t m4) {
    DirectFile f(path, /*create=*/false);
    SidecarHeader h{};
    engine_detail::read_exact(f, &h, sizeof(h), 0);
    if (h.magic != kMagicCodebook4) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFScanIndex: codebook magic mismatch on '" + path + "'");
    }
    // The blob layout after the header: [u64 qsize][quantizer bytes].
    uint64_t qsize = 0;
    engine_detail::read_exact(f, &qsize, sizeof(qsize), sizeof(SidecarHeader));
    std::vector<uint8_t> blob(static_cast<size_t>(qsize));
    if (qsize > 0) {
        engine_detail::read_exact(f, blob.data(), blob.size(),
                                  sizeof(SidecarHeader) + sizeof(qsize));
    }
    // Construct with the index's recorded dim/m4 (sanity vs the serialized
    // state); deserialize overwrites the placeholder training state.
    out = std::make_unique<PqQuantizer>(MetricKind::L2Sq, dim, m4, /*bits=*/4);
    out->deserialize(blob.data(), blob.size());
}

/// Read a shard's `.rowids` sidecar into `out`.
void read_rowids(const std::string& path, std::vector<RowId>& out) {
    DirectFile f(path, /*create=*/false);
    SidecarHeader h{};
    engine_detail::read_exact(f, &h, sizeof(h), 0);
    if (h.magic != kMagicRowids) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFScanIndex: rowids magic mismatch on '" + path + "'");
    }
    const uint32_t n = static_cast<uint32_t>(h.n_vectors);
    out.resize(n);
    if (n > 0) {
        engine_detail::read_exact(f, out.data(),
                                  static_cast<size_t>(n) * sizeof(RowId),
                                  sizeof(SidecarHeader));
    }
}

}  // namespace

std::unique_ptr<IVFScanIndex> IVFScanIndex::read(const std::string& shards_dir) {
    auto idx = std::unique_ptr<IVFScanIndex>(new IVFScanIndex());
    idx->path = shards_dir;

    const std::string manifest_path = shards_dir + "/manifest";
    uint32_t K = 0, n_probe_default = 1;
    Dim dim = 0;
    uint16_t m4 = 0;
    if (!read_scan_manifest(manifest_path, K, dim, n_probe_default, m4)) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFScanIndex: cannot read manifest '" + manifest_path + "'");
    }
    if (K == 0 || dim == 0 || m4 == 0) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFScanIndex: manifest has K/dim/m4 = 0");
    }
    idx->K = K;
    idx->dim = dim;
    idx->n_probe_default = n_probe_default > 0 ? n_probe_default : 1;
    idx->m4 = m4;

    // Load centroids (raw FP16, same layout as IVFIndex).
    {
        const std::string path = shards_dir + "/centroids.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFScanIndex: cannot open '" + path + "'");
        }
        idx->centroids.resize(static_cast<size_t>(K) * dim);
        f.read(reinterpret_cast<char*>(idx->centroids.data()),
               static_cast<std::streamsize>(idx->centroids.size() *
                                            sizeof(float16_t)));
        if (!f && !f.eof()) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFScanIndex: short read on '" + path + "'");
        }
    }

    // Load the shared 4-bit codebook.
    read_codebook(shards_dir + "/codebook4.bin", idx->quantizer, dim, m4);

    // Open per-shard CodeStreams + load rowids.
    idx->shards.resize(K);
    for (uint32_t k = 0; k < K; k++) {
        char dirbuf[32];
        std::snprintf(dirbuf, sizeof(dirbuf), "shard_%04u", k + 1);
        const std::string shard_dir = shards_dir + "/" + dirbuf;
        const std::string codes4_path = shard_dir + "/.codes4";
        const std::string rowids_path = shard_dir + "/.rowids";

        std::error_code ec;
        if (!std::filesystem::exists(codes4_path, ec)) {
            // Empty shard — skip (no entry created; router sees K but null).
            spdlog::debug("[sextant] IVFScanIndex: shard {} has no .codes4 (empty)",
                          k + 1);
            continue;
        }
        auto shard = std::make_unique<ScanShard>();
        shard->codes = std::make_unique<CodeStream>(codes4_path, m4);
        read_rowids(rowids_path, shard->row_ids);
        shard->count = static_cast<uint32_t>(shard->row_ids.size());
        if (shard->count != shard->codes->n_vectors()) {
            spdlog::warn("[sextant] IVFScanIndex: shard {} rowids count ({}) != "
                         "codes n_vectors ({}) — using the smaller",
                         k + 1, shard->count, shard->codes->n_vectors());
            shard->count = std::min(shard->count, shard->codes->n_vectors());
        }
        idx->shards[k] = std::move(shard);
    }

    spdlog::info("[sextant] IVFScanIndex: opened '{}' (K={}, dim={}, m4={}, "
                 "n_probe_default={})", shards_dir, K, dim, m4,
                 idx->n_probe_default);
    return idx;
}

}  // namespace sextant
