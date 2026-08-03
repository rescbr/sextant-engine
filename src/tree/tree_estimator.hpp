#pragma once

/// @file tree_estimator.hpp
/// Tree-specific parameter estimation.
///
/// Centralizes the tree build parameter formulas (k_root, leaf_capacity,
/// pca_dims, n_probe) that were previously scattered across build methods.
/// The tree estimator is a pure formula resolver — unlike the graph Estimator,
/// it does NOT build mini-indices or measure topology. The tree's parameter
/// space is small enough that closed-form heuristics suffice.

#include <sextant/types.hpp>
#include <cstdint>

namespace sextant::tree {

/// Resolved tree build parameters (output of resolve_tree_params).
struct TreeResolvedParams {
    uint32_t k_root = 0;           // fine-grained cluster count (L2 for depth-3)
    uint32_t k_l1 = 0;             // root branching factor for depth-3 (0 = depth <= 2)
    uint32_t leaf_capacity = 5000; // max vectors per leaf
    uint32_t pca_dims = 32;        // PCA routing dimensions (0 = no PCA)
    uint16_t depth = 2;            // tree depth (1, 2, or 3)
    uint32_t n_probe_l0 = 0;       // root-level probe count
    uint32_t n_probe_ln = 4;       // deeper-level probe count
};

/// Tree build parameter overrides (fields >0/non-default take precedence).
struct TreeBuildOverrides {
    uint32_t k_root = 0;
    uint32_t leaf_capacity = 0;
    uint32_t pca_dims = 0;
    uint32_t n_probe_l0 = 0;
    uint32_t n_probe_ln = 0;
    uint32_t k_root_max_depth2 = 512; // k_root threshold for depth-3
};

/// Resolve tree parameters from dataset properties + overrides.
/// Pure function — no I/O, no mini-build.
///
/// @param n_vectors  Total vector count.
/// @param dim        Vector dimensionality.
/// @param overrides  User-specified overrides (0 = auto).
TreeResolvedParams resolve_tree_params(uint64_t n_vectors, Dim dim,
                                        const TreeBuildOverrides& overrides = {});

}  // namespace sextant::tree
