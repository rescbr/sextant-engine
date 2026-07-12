#include <gtest/gtest.h>
#include "algo/vamana_core.hpp"

namespace sextant {
namespace {

TEST(VamanaCore, NodeSize) {
    // R=64, inline_pq=0, code_size=32: (16 + 64*4 + 7) & ~7 = 272
    EXPECT_EQ(272u, VamanaCore::static_node_size(64, 0, 32));

    // R=64, inline_pq=64, code_size=32: 272 + 64*32 = 2320
    EXPECT_EQ(2320u, VamanaCore::static_node_size(64, 64, 32));
}

}  // namespace
}  // namespace sextant
