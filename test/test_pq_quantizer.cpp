#include <gtest/gtest.h>
#include "../src/quant/pq_quantizer.hpp"

namespace sextant {
namespace {

TEST(PqQuantizer, BasicConstruction) {
    PqQuantizer q(MetricKind::L2Sq, 128, 32, 8);
    EXPECT_EQ(32u, q.code_size());
    EXPECT_EQ(32u, q.m());
}

}  // namespace
}  // namespace sextant
