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

#include <sextant/schema.hpp>

#include <cstdint>
#include <string>

namespace sextant::tree {

/// Tree manifest: all parameters needed to search the tree.
struct TreeManifest {
    // [meta]
    /// 32-char lowercase hex UUID (random, generated at build). Identifies
    /// the tree independent of its file path — the DuckDB extension stores
    /// it in the DuckDB-side index metadata and verifies it on every open
    /// (stale/wrong-file detection). Empty = pre-UUID tree (legacy).
    std::string uuid;

    // [index]
    uint32_t dim = 0;
    uint16_t m4 = 0;             // PQ subquantizers for scan codes
    uint8_t  scan_pq_bits = 4;   // bits per PQ code (4 or 8)
    std::string quantizer_type = "pq";
    uint32_t prq_nsplits = 0;
    uint8_t  metric = 0;           // MetricKind as uint8_t (0=L2Sq, 1=InnerProduct)

    // [tree]
    uint16_t depth = 0;
    uint32_t k_root = 0;          // branching factor at root
    uint32_t k_l1 = 0;            // depth-3: actual root branching (n L1 nodes).
                                  // 0 = depth <= 2 (root children = L1 or leaves).
    uint32_t leaf_capacity = 0;   // max vectors per leaf
    uint64_t n_vectors = 0;       // LOGICAL row count: input rows indexed
                                  // at build, pre-closure-replication
                                  // (mutations adjust it; leaves may hold
                                  // more rows after closure replication)
    uint32_t n_leaves = 0;
    uint32_t n_probe_l0 = 0;      // probe count at level 0
    uint32_t n_probe_ln = 0;      // probe count at deeper levels

    // [routing]
    float adaptive_probe_gap = 0.0f;  // 0 = disabled, >0 = gap threshold
    float median_lid = 0.0f;          // median local intrinsic dimensionality
    uint32_t pca_dims = 0;            // PCA routing dims (0 = no PCA routing)
    float probe_fraction = 0.0f;      // default probe budget as a corpus
                                     // fraction (0 = legacy count routing;
                                     // new builds persist 0.5)

    // [partition]
    float balance_factor = 4.0f;

    // [schema]  (Phase A; empty = no filter columns)
    Schema schema;                 // empty = no filter columns
    uint32_t summary_size = 0;     // schema-determined summary size (0 = none)
};

/// Serialize a TreeManifest to a TOML string.
std::string manifest_to_toml(const TreeManifest& m);

/// Generate a fresh random 32-char lowercase-hex tree UUID (v4-shaped).
std::string generate_tree_uuid();

/// Parse a TOML string into a TreeManifest. Throws on parse error or
/// missing required fields.
TreeManifest manifest_from_toml(const std::string& toml);

}  // namespace sextant::tree
