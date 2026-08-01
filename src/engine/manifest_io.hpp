#pragma once

/// @file manifest_io.hpp
/// TOML-based manifest serialization for all index types.
///
/// Replaces the former line-oriented text manifests ("ready\nK\ndim\n...") with
/// structured TOML. One central place for all manifest (de)serialization so the
/// format is consistent across graph, IVF-graph, and IVF-scan indexes.
///
/// Manifests are written as TOML text files (no SidecarHeader binary prefix).
/// The "ready" sentinel is replaced by successful TOML parsing — if the file
/// parses and has the required fields, it's a valid commit point.
///
/// Atomic commit: write to `<path>.tmp`, fdatasync, rename to `<path>`.

#include <cstdint>
#include <string>
#include <vector>

namespace sextant {

// ===========================================================================
// Graph index manifest (.manifest)
// ===========================================================================

struct GraphManifest {
    uint64_t n_vectors = 0;
    uint32_t dim = 0;
    uint16_t R = 0;
    uint16_t pq_m = 0;
};

std::string graph_manifest_to_toml(const GraphManifest& m);
GraphManifest graph_manifest_from_toml(const std::string& toml);

// ===========================================================================
// IVF graph manifest (shards_dir/manifest)
// ===========================================================================

struct IVFGraphManifest {
    uint32_t K = 0;
    uint32_t dim = 0;
    uint32_t n_probe_default = 1;
    float closure_factor = 1.0f;
};

std::string ivf_graph_manifest_to_toml(const IVFGraphManifest& m);
IVFGraphManifest ivf_graph_manifest_from_toml(const std::string& toml);

// ===========================================================================
// IVF scan manifest (shards_dir/manifest)
// ===========================================================================

struct IVFScanManifest {
    uint32_t K = 0;
    uint32_t dim = 0;
    uint32_t n_probe_default = 1;
    uint16_t m4 = 0;
    uint8_t  scan_pq_bits = 4;
    std::string quantizer_type = "pq";
    uint32_t prq_nsplits = 0;
    uint32_t sub_shard_probe_pct = 0;
    float adaptive_probe_gap = 0.0f;
    float median_lid = 0.0f;
};

std::string ivf_scan_manifest_to_toml(const IVFScanManifest& m);
IVFScanManifest ivf_scan_manifest_from_toml(const std::string& toml);

// ===========================================================================
// Per-shard manifest (scan: shard_dir/.manifest)
// ===========================================================================

struct ShardManifest {
    uint32_t count = 0;
    uint32_t dim = 0;
    uint16_t m4 = 0;
};

std::string shard_manifest_to_toml(const ShardManifest& m);
ShardManifest shard_manifest_from_toml(const std::string& toml);

// ===========================================================================
// Sub-shard centroid offsets (shard_dir/shard.manifest)
// ===========================================================================

/// TOML format:
///   ready = true
///   n_subs = 3
///   offsets = [10, 50, 120]
std::string shard_offsets_to_toml(uint32_t n_subs,
                                  const std::vector<uint32_t>& offsets);
/// Returns (n_subs, offsets). Throws on parse error.
std::pair<uint32_t, std::vector<uint32_t>>
shard_offsets_from_toml(const std::string& toml);

// ===========================================================================
// I/O helpers: atomic write + read
// ===========================================================================

/// Write `content` to `path` atomically: write to `path.tmp`, sync, rename.
void write_manifest_atomic(const std::string& path,
                           const std::string& content);

/// Read a file fully into a string. Throws if the file doesn't exist.
std::string read_file_to_string(const std::string& path);

}  // namespace sextant
