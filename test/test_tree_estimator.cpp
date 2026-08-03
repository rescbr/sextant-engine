#include <gtest/gtest.h>

#include "tree/tree_estimator.hpp"

#include <cmath>

namespace sextant::tree {
namespace {

TEST(TreeEstimator, SmallDatasetDepth1) {
    // 5k vectors, leaf_cap=5000 → 1 leaf → depth=1.
    auto r = resolve_tree_params(5000, 128, {});
    EXPECT_EQ(r.leaf_capacity, 5000u);
    EXPECT_EQ(r.depth, 1u);  // 1 leaf <= k_root
    EXPECT_EQ(r.pca_dims, 32u);
}

TEST(TreeEstimator, MediumDatasetDepth2) {
    // 100k vectors, leaf_cap=5000 → 20 leaves.
    // k_root = round_pow2(20/4) = round_pow2(5) = 4.
    // 20 leaves > 4 k_root → depth=2.
    auto r = resolve_tree_params(100'000, 128, {});
    EXPECT_EQ(r.depth, 2u);
    EXPECT_GE(r.k_root, 4u);
    EXPECT_EQ(r.k_l1, 0u);  // no depth-3
    EXPECT_GT(r.n_probe_l0, 0u);
    EXPECT_EQ(r.n_probe_ln, 4u);
}

TEST(TreeEstimator, BillionScaleDepth3) {
    // 1B vectors, leaf_cap=5000 → 200k leaves.
    // k_root = round_pow2(200k/4) = round_pow2(50k) = 65536.
    // 65536 > 512 → depth=3, k_l1 = round_pow2(65536/256) = 256.
    auto r = resolve_tree_params(1'000'000'000ULL, 768, {});
    EXPECT_EQ(r.depth, 3u);
    EXPECT_EQ(r.k_root, 65536u);
    EXPECT_EQ(r.k_l1, 256u);
    EXPECT_GE(r.n_probe_l0, 16u);  // 2*sqrt(256) = 32
}

TEST(TreeEstimator, OverridesRespected) {
    TreeBuildOverrides ov;
    ov.k_root = 42;
    ov.leaf_capacity = 1000;
    ov.pca_dims = 16;
    ov.n_probe_ln = 8;
    auto r = resolve_tree_params(1'000'000, 128, ov);
    EXPECT_EQ(r.k_root, 42u);
    EXPECT_EQ(r.leaf_capacity, 1000u);
    EXPECT_EQ(r.pca_dims, 16u);
    EXPECT_EQ(r.n_probe_ln, 8u);
}

TEST(TreeEstimator, PcaDimsClampedToDim) {
    auto r = resolve_tree_params(100'000, 8, {});  // dim=8 < 32
    EXPECT_EQ(r.pca_dims, 8u);  // clamped to dim
}

}  // namespace
}  // namespace sextant::tree
