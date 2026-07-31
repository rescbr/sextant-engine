#pragma once

/// @file ivf_scan_index.hpp
/// IVFScanIndex — K shards of 4-bit FastScan codes + FP16 routing centroids
/// (Option A production path).
///
/// The default IVF build path (BuildConfig::merged_graph=false) writes one of
/// these instead of the graph-inside-shard `IVFIndex`. Each shard is a pure
/// container: a `.codes4` sidecar (FastScan block-packed 4-bit PQ codes) and a
/// `.rowids` sidecar (shard-local idx → RowId). The 4-bit codebook is shared
/// across all shards and stored once at `<prefix>.shards/codebook4.bin`.
///
/// Search: route query → `n_probe` nearest centroids, sequentially stream each
/// probed shard's `.codes4`, run `simd::pq4_scan_many`, take top-W by 4-bit
/// distance, return as Candidates. Exact rerank + top-k is the DB layer's job.
///
/// On-disk layout under `<prefix>.shards/`:
///   centroids.bin    — K × dim × float16_t (the routing centroids; same
///                      format as IVFIndex so routing code is shared)
///   codebook4.bin    — serialized 4-bit PqQuantizer (shared codebook)
///   manifest         — line-oriented text: ready/K/dim/n_probe_default/m4
///   shard_0001/      — per-shard sidecars:
///       .codes4      — FastScan block-packed codes (m × 16 bytes/block)
///       .rowids      — shard-local idx → RowId (n × int64)
///       .manifest    — atomic commit point
///   ...
///
/// See ~/.local/state/maki/plans/sharing-eternal-louse.md.

#include "quant/pq_quantizer.hpp"
#include "sextant/config.hpp"
#include "sextant/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sextant {

class CodeStream;

/// One sub-shard's scan-side data. Identical to ScanShard's fields but
/// without the sub-shard vector (no recursion). When a shard has no
/// sub-shards, its data lives directly in ScanShard (flat mode).
struct ScanSubShard {
    std::unique_ptr<CodeStream> codes;
    std::vector<RowId> row_ids;
    uint32_t count = 0;
};

/// One shard's scan-side data: a codes stream + the shard-local→RowId map.
struct ScanShard {
    /// Sequential pread streamer over `.codes4`.
    std::unique_ptr<CodeStream> codes;
    /// shard-local index (0..shard_n-1) → global RowId. Length = shard_n.
    /// Loaded once at index-open (the .rowids sidecar is small: 8 B/vec).
    std::vector<RowId> row_ids;
    /// RaBitQ per-vector factors (dp_multiplier, or_minus_c_l2sqr), row-major,
    /// length = shard_n × 2. Empty for non-RaBitQ indexes. Loaded from the
    /// `.factors` sidecar; used during the heap walk to finalize distances.
    std::vector<float> factors;
    /// Shard-local vector count (= row_ids.size()).
    uint32_t count = 0;

    // --- Sub-shard support (Phase 1) ---
    // When has_sub_shards=true, the flat codes/row_ids above are empty and
    // data lives in sub_shards. When false, the flat fields are used as today.
    std::vector<ScanSubShard> sub_shards;
    /// Offset of each sub-shard's FP16 centroid into IVFScanIndex::sub_centroids
    /// (in float16_t elements, not bytes). Empty when !has_sub_shards.
    std::vector<uint32_t> sub_centroid_offsets;
    bool has_sub_shards = false;
};

/// Owning container for an IVF-scan index: K ScanShards + shared 4-bit
/// codebook + FP16 routing centroids. The IVFScanSearcher holds a reference.
struct IVFScanIndex {
    /// K × dim FP16 routing centroids, row-major.
    std::vector<float16_t> centroids;

    Dim dim = 0;
    uint32_t K = 0;
    uint32_t n_probe_default = 1;
    uint16_t m4 = 0;  ///< Scan PQ subquantizer count (matches the codebook).
    uint8_t scan_pq_bits = 4;  ///< 4 (FastScan nibble) or 8 (byte-per-code).

    /// Quantizer type ("pq" / "anisotropic-pq" / "prq"). Read from the manifest;
    /// selects which PqQuantizer subclass reconstructs the codebook.
    std::string quantizer_type = "pq";
    /// PRQ nsplits (sub-space count), read from the manifest. 0 = not PRQ.
    uint32_t prq_nsplits = 0;

    /// Shared 4-bit codebook for all shards. Trained once at build time.
    std::unique_ptr<PqQuantizer> quantizer;

    /// Per-shard scan data. Null entries = empty shard (skipped at build).
    std::vector<std::unique_ptr<ScanShard>> shards;

    // --- Sub-shard support (Phase 1) ---
    /// Sub-shard centroids (FP16, same layout as `centroids`). Concatenated
    /// for all sub-shards across all shards. Indexed by
    /// ScanShard::sub_centroid_offsets. Empty when no shards have sub-shards.
    std::vector<float16_t> sub_centroids;
    /// How many sub-shards to probe per coarse shard during search.
    /// 0 or 1 = scan all sub-shards (= flat scan). >1 = two-level routing.
    uint32_t sub_shard_n_probe = 1;

    std::string path;

    IVFScanIndex() = default;
    ~IVFScanIndex();

    IVFScanIndex(const IVFScanIndex&) = delete;
    IVFScanIndex& operator=(const IVFScanIndex&) = delete;
    IVFScanIndex(IVFScanIndex&&) = delete;
    IVFScanIndex& operator=(IVFScanIndex&&) = delete;

    /// Open an IVF-scan index from a `<prefix>.shards/` directory.
    static std::unique_ptr<IVFScanIndex> read(const std::string& shards_dir);
};

}  // namespace sextant
