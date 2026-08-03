#include "tree_estimator.hpp"

#include <algorithm>
#include <cmath>

namespace sextant::tree {

namespace {

/// Round to the nearest power of two.
uint32_t round_pow2(uint32_t x) {
    if (x <= 1) return 1;
    uint32_t lo = 1;
    while (lo * 2 <= x) lo *= 2;
    // Check if the next power of two is closer.
    if (lo < (1u << 30) && (x - lo) > (lo * 2 - x)) return lo * 2;
    return lo;
}

}  // namespace

TreeResolvedParams resolve_tree_params(uint64_t n_vectors, Dim dim,
                                        const TreeBuildOverrides& ov) {
    TreeResolvedParams r;

    // Leaf capacity.
    r.leaf_capacity = ov.leaf_capacity > 0 ? ov.leaf_capacity : 5000;

    // Estimate leaf count.
    const uint64_t n_leaves_est =
        std::max<uint64_t>(1, n_vectors / r.leaf_capacity);

    // K_root: round_pow2(n_leaves / 4).
    // Each root child holds ~4 leaves on average. round_pow2 picks the nearest
    // power of two for cache-aligned child extents.
    if (ov.k_root > 0) {
        r.k_root = ov.k_root;
    } else {
        const uint64_t target = std::max<uint64_t>(1, n_leaves_est / 4);
        r.k_root = std::clamp(round_pow2(static_cast<uint32_t>(
            std::min<uint64_t>(target, 1u << 20))), 4u, 65536u);
    }

    // PCA dims.
    r.pca_dims = ov.pca_dims > 0 ? ov.pca_dims : std::min(dim, 32u);

    // Depth + k_l1.
    if (r.k_root > ov.k_root_max_depth2) {
        r.depth = 3;
        const uint32_t target_l1_children = 256;
        r.k_l1 = std::clamp(
            round_pow2(std::max(16u, r.k_root / target_l1_children)),
            16u, 512u);
    } else {
        r.depth = (n_leaves_est <= r.k_root) ? 1 : 2;
        r.k_l1 = 0;
    }

    // N_probe_l0: max(1, 2 * sqrt(k_root_effective)).
    // k_root_effective is the actual root node child count.
    const uint32_t k_root_eff = (r.depth >= 3) ? r.k_l1 : r.k_root;
    r.n_probe_l0 = ov.n_probe_l0 > 0
        ? ov.n_probe_l0
        : static_cast<uint32_t>(std::max(1.0, 2.0 * std::sqrt(double(k_root_eff))));

    // N_probe_ln.
    r.n_probe_ln = ov.n_probe_ln > 0 ? ov.n_probe_ln : 4;

    return r;
}

}  // namespace sextant::tree
