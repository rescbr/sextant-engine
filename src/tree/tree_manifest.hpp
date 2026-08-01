#pragma once

/// @file tree_manifest.hpp
/// TOML manifest for the IVF tree, serialized via cpptoml.
///
/// The manifest stores build-time parameters and search-time routing config.
/// It's stored as a TOML blob inside the tree file (at the page range pointed
/// to by the superblock's config_page/config_pages).
///
/// The manifest is NOT the source of truth for structural pointers (root,
/// codebook, bitmap) — those are in the superblock. The manifest holds
/// parameters that the search path needs (dim, m4, n_probe, routing gaps, etc.)
/// and that the estimator/builder computed.

#include <cstdint>
#include <string>

namespace sextant::tree {

/// Tree manifest: all parameters needed to search the tree.
struct TreeManifest {
    // [index]
    uint32_t dim = 0;
    uint16_t m4 = 0;             // PQ subquantizers for scan codes
    uint8_t  scan_pq_bits = 4;   // bits per PQ code (4 or 8)
    std::string quantizer_type = "pq";
    uint32_t prq_nsplits = 0;

    // [tree]
    uint16_t depth = 0;
    uint32_t k_root = 0;          // branching factor at root
    uint32_t leaf_capacity = 0;   // max vectors per leaf
    uint32_t n_leaves = 0;
    uint32_t n_probe_l0 = 0;      // probe count at level 0
    uint32_t n_probe_ln = 0;      // probe count at deeper levels

    // [routing]
    float adaptive_probe_gap = 0.0f;  // 0 = disabled, >0 = gap threshold
    float median_lid = 0.0f;          // median local intrinsic dimensionality
    uint32_t pca_dims = 0;            // PCA routing dims (0 = no PCA routing)

    // [partition]
    float balance_factor = 4.0f;
    uint32_t sub_shard_probe_pct = 0;  // 0 = disabled
};

/// Serialize a TreeManifest to a TOML string.
std::string manifest_to_toml(const TreeManifest& m);

/// Parse a TOML string into a TreeManifest. Throws on parse error or
/// missing required fields.
TreeManifest manifest_from_toml(const std::string& toml);

}  // namespace sextant::tree
