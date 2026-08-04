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
// I/O helpers: atomic write + read
// ===========================================================================

/// Write `content` to `path` atomically: write to `path.tmp`, sync, rename.
void write_manifest_atomic(const std::string& path,
                           const std::string& content);

/// Read a file fully into a string. Throws if the file doesn't exist.
std::string read_file_to_string(const std::string& path);

}  // namespace sextant
