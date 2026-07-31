#include "sextant/ivf_scan_index.hpp"

#include "engine/sidecar_io.hpp"
#include "quant/pq_quantizer.hpp"
#include "quant/product_residual_quantizer.hpp"
#include "quant/rabitq_quantizer.hpp"
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
///   <scan_pq_bits>
///   <quantizer_type>
///   <prq_nsplits>
///   <sub_shard_probe_pct>
///   <adaptive_probe_gap>
///   <median_lid>
/// Returns false if the file is missing or malformed (all fields required).
bool read_scan_manifest(const std::string& path, uint32_t& K, Dim& dim,
                        uint32_t& n_probe_default, uint16_t& m4,
                        uint8_t& scan_pq_bits, std::string& quantizer_type,
                        uint32_t& prq_nsplits,
                        uint32_t& sub_shard_probe_pct,
                        float& adaptive_probe_gap,
                        float& median_lid) {
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
    uint16_t bits_raw = 4;
    if (!read_line(bits_raw)) return false;
    scan_pq_bits = (bits_raw == 8) ? 8 : 4;
    if (!read_line(quantizer_type)) return false;
    if (!read_line(prq_nsplits)) return false;
    if (!read_line(sub_shard_probe_pct)) return false;
    if (!read_line(adaptive_probe_gap)) return false;
    if (!read_line(median_lid)) return false;
    return true;
}

/// Read the shared scan codebook (a serialized PqQuantizer blob prefixed by a
/// SidecarHeader). Reconstructs the quantizer in-place. Dispatches on
/// `quantizer_type`: "prq" reconstructs a ProductResidualQuantizer (and accepts
/// the kMagicCodebookPRQ4 magic); otherwise a plain PqQuantizer. The sidecar
/// magic (kMagicCodebook4 vs kMagicCodebook8 vs kMagicCodebookPRQ4) selects the
/// quantizer/bits variant on disk.
void read_codebook(const std::string& path, std::unique_ptr<PqQuantizer>& out,
                   Dim dim, uint16_t m4, uint8_t scan_pq_bits,
                   const std::string& quantizer_type, uint32_t prq_nsplits) {
    DirectFile f(path, /*create=*/false);
    SidecarHeader h{};
    engine_detail::read_exact(f, &h, sizeof(h), 0);
    const uint64_t expected_magic =
        (quantizer_type == "prq") ? kMagicCodebookPRQ4
        : ((scan_pq_bits == 8) ? kMagicCodebook8 : kMagicCodebook4);
    if (h.magic != expected_magic) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFScanIndex: codebook magic mismatch on '" + path +
                        "' (quantizer_type=" + quantizer_type +
                        ", scan_pq_bits=" + std::to_string(scan_pq_bits) + ")");
    }
    // The blob layout after the header: [u64 qsize][quantizer bytes].
    uint64_t qsize = 0;
    engine_detail::read_exact(f, &qsize, sizeof(qsize), sizeof(SidecarHeader));
    std::vector<uint8_t> blob(static_cast<size_t>(qsize));
    if (qsize > 0) {
        engine_detail::read_exact(f, blob.data(), blob.size(),
                                  sizeof(SidecarHeader) + sizeof(qsize));
    }
    // Construct with the index's recorded dim/m4/bits (sanity vs the
    // serialized state); deserialize overwrites the placeholder training state.
    if (quantizer_type == "prq") {
        out = std::make_unique<ProductResidualQuantizer>(
            MetricKind::L2Sq, dim, m4, /*bits=*/scan_pq_bits, prq_nsplits,
            /*beam_size=*/1);
    } else if (quantizer_type == "rabitq") {
        // RaBitQ reconstructs its own code_size/rotation from the serialized
        // blob (seed + dim); m4/bits are derived (m4 = dim/4, bits = 4).
        out = std::make_unique<RaBitQQuantizer>(MetricKind::L2Sq, dim);
    } else {
        out = std::make_unique<PqQuantizer>(MetricKind::L2Sq, dim, m4,
                                            /*bits=*/scan_pq_bits);
    }
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

/// Read a shard's `.factors` sidecar (RaBitQ) into `out`: n × 2 floats
/// (dp_multiplier, or_minus_c_l2sqr), row-major, indexed by shard-local ID.
void read_factors(const std::string& path, std::vector<float>& out) {
    DirectFile f(path, /*create=*/false);
    SidecarHeader h{};
    engine_detail::read_exact(f, &h, sizeof(h), 0);
    if (h.magic != kMagicFactors) {
        throw Error(ErrorCode::CorruptIndex,
                     "IVFScanIndex: factors magic mismatch on '" + path + "'");
    }
    const uint32_t n = static_cast<uint32_t>(h.n_vectors);
    out.resize(static_cast<size_t>(n) * 2);
    if (n > 0) {
        engine_detail::read_exact(f, out.data(), out.size() * sizeof(float),
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
    uint8_t scan_pq_bits = 4;
    std::string quantizer_type = "pq";
    uint32_t prq_nsplits = 0;
    if (!read_scan_manifest(manifest_path, K, dim, n_probe_default, m4,
                            scan_pq_bits, quantizer_type, prq_nsplits,
                            idx->sub_shard_probe_pct,
                            idx->adaptive_probe_gap,
                            idx->median_lid)) {
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
    idx->scan_pq_bits = scan_pq_bits;
    idx->quantizer_type = quantizer_type;
    idx->prq_nsplits = prq_nsplits;

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

    // Load the shared scan codebook (4-bit or 8-bit per scan_pq_bits;
    // PRQ always writes codebook4.bin with kMagicCodebookPRQ4).
    {
        const std::string path = shards_dir +
            ((scan_pq_bits == 8) ? "/codebook8.bin" : "/codebook4.bin");
        read_codebook(path, idx->quantizer, dim, m4, scan_pq_bits,
                      quantizer_type, prq_nsplits);
    }

    // Open per-shard CodeStreams + load rowids.
    idx->shards.resize(K);
    const std::string codes_suffix =
        (scan_pq_bits == 8) ? "/.codes8" : "/.codes4";

    // Load sub-shard centroids if present (raw FP16, same as centroids.bin).
    {
        const std::string path = shards_dir + "/subcentroids.bin";
        std::ifstream f(path, std::ios::binary);
        if (f) {
            f.seekg(0, std::ios::end);
            const auto sz = f.tellg();
            f.seekg(0);
            if (sz > 0) {
                idx->sub_centroids.resize(sz / sizeof(float16_t));
                f.read(reinterpret_cast<char*>(idx->sub_centroids.data()), sz);
            }
        }
    }

    for (uint32_t k = 0; k < K; k++) {
        char dirbuf[32];
        std::snprintf(dirbuf, sizeof(dirbuf), "shard_%04u", k + 1);
        const std::string shard_dir = shards_dir + "/" + dirbuf;

        // Check for sub-shard layout: shard_NNNN/sub_0000/ directory.
        char subdir_buf[32];
        std::snprintf(subdir_buf, sizeof(subdir_buf), "sub_%04u", 0);
        const std::string sub0_dir = shard_dir + "/" + subdir_buf;
        const std::string sub0_codes = sub0_dir + codes_suffix;

        std::error_code ec;
        if (std::filesystem::exists(sub0_codes, ec)) {
            // --- Sub-shard layout ---
            auto shard = std::make_unique<ScanShard>();
            shard->has_sub_shards = true;
            for (uint32_t s = 0; ; s++) {
                char sbuf[32];
                std::snprintf(sbuf, sizeof(sbuf), "sub_%04u", s);
                const std::string sd = shard_dir + "/" + sbuf;
                const std::string cp = sd + codes_suffix;
                if (!std::filesystem::exists(cp, ec)) break;

                ScanSubShard ss;
                ss.codes = std::make_unique<CodeStream>(cp, m4, scan_pq_bits);
                read_rowids(sd + "/.rowids", ss.row_ids);
                ss.count = static_cast<uint32_t>(ss.row_ids.size());
                if (ss.count != ss.codes->n_vectors()) {
                    ss.count = std::min(ss.count, ss.codes->n_vectors());
                }
                shard->count += ss.count;
                shard->sub_shards.push_back(std::move(ss));
            }
            // Read sub-centroid offsets from shard.manifest.
            const std::string shard_manifest = shard_dir + "/shard.manifest";
            std::ifstream smf(shard_manifest);
            if (smf) {
                std::string tok;
                std::getline(smf, tok); // "ready"
                uint32_t n_subs = 0;
                std::getline(smf, tok);
                std::istringstream(tok) >> n_subs;
                for (uint32_t s = 0; s < n_subs; s++) {
                    uint32_t off = 0;
                    std::getline(smf, tok);
                    std::istringstream(tok) >> off;
                    shard->sub_centroid_offsets.push_back(off);
                }
            }
            idx->shards[k] = std::move(shard);
            continue;
        }

        // --- Flat shard layout (existing path) ---
        const std::string codes_path = shard_dir + codes_suffix;
        const std::string rowids_path = shard_dir + "/.rowids";

        if (!std::filesystem::exists(codes_path, ec)) {
            // Empty shard — skip (no entry created; router sees K but null).
            spdlog::debug("[sextant] IVFScanIndex: shard {} has no {} (empty)",
                          k + 1, codes_suffix);
            continue;
        }
        auto shard = std::make_unique<ScanShard>();
        shard->codes = std::make_unique<CodeStream>(codes_path, m4,
                                                     scan_pq_bits);
        read_rowids(rowids_path, shard->row_ids);
        shard->count = static_cast<uint32_t>(shard->row_ids.size());
        if (shard->count != shard->codes->n_vectors()) {
            spdlog::warn("[sextant] IVFScanIndex: shard {} rowids count ({}) != "
                         "codes n_vectors ({}) — using the smaller",
                         k + 1, shard->count, shard->codes->n_vectors());
            shard->count = std::min(shard->count, shard->codes->n_vectors());
        }
        // RaBitQ-only: load the per-vector factors sidecar used to finalize
        // scan distances during the heap walk.
        if (quantizer_type == "rabitq") {
            const std::string factors_path = shard_dir + "/.factors";
            read_factors(factors_path, shard->factors);
        }
        idx->shards[k] = std::move(shard);
    }

    spdlog::info("[sextant] IVFScanIndex: opened '{}' (K={}, dim={}, m4={}, "
                 "n_probe_default={}, quantizer={})",
                 shards_dir, K, dim, m4, idx->n_probe_default, quantizer_type);
    return idx;
}

}  // namespace sextant
