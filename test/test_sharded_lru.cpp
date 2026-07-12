#include <gtest/gtest.h>
#include "storage/sharded_lru.hpp"

namespace sextant {
namespace {

TEST(ShardedLRU, BasicInsertLookup) {
    ShardedLRUCache cache(4, 8, kBlockSize);
    std::vector<uint8_t> data(kBlockSize, 42);

    auto& shard = cache.shard(0);
    ScopedWriteLock lock(shard.mutex());
    shard.insert(0, data.data(), kBlockSize);
    uint8_t* ptr = shard.lookup(0);
    ASSERT_NE(nullptr, ptr);
    EXPECT_EQ(42, ptr[0]);
}

}  // namespace
}  // namespace sextant
