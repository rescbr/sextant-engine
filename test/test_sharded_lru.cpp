#include <gtest/gtest.h>
#include "storage/sharded_lru.hpp"

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

namespace sextant {
namespace {

// Helper: build a kBlockSize buffer tagged with a sentinel value.
static std::vector<uint8_t> make_block(uint8_t tag) {
    return std::vector<uint8_t>(kBlockSize, tag);
}

// ---------------------------------------------------------------------------
// Basic insert + lookup round-trip (existing sanity check).
// ---------------------------------------------------------------------------
TEST(ShardedLRU, BasicInsertLookup) {
    ShardedLRUCache cache(4, 8, kBlockSize);
    std::vector<uint8_t> data = make_block(42);

    auto& shard = cache.shard(0);
    ScopedWriteLock lock(shard.mutex());
    shard.insert(0, data.data(), kBlockSize);
    uint8_t* ptr = shard.lookup(0);
    ASSERT_NE(nullptr, ptr);
    EXPECT_EQ(42, ptr[0]);
}

// ---------------------------------------------------------------------------
// At-capacity hit-rate: 8 blocks fit in a shard of capacity 8, so all must be
// found (no eviction at exact capacity).
// ---------------------------------------------------------------------------
TEST(ShardedLRU, AtCapacityNoEviction) {
    ShardedLRUCache cache(2, 8, kBlockSize);

    // Route all inserts to a single shard by choosing indices that map there.
    // With 2 shards, even indices go to shard 0.
    auto& shard = cache.shard(0);
    ScopedWriteLock lock(shard.mutex());

    for (uint64_t i = 0; i < 8; ++i) {
        std::vector<uint8_t> data = make_block(static_cast<uint8_t>(i));
        shard.insert(i * 2, data.data(), kBlockSize);
    }

    // All 8 must still be resident.
    for (uint64_t i = 0; i < 8; ++i) {
        uint8_t* p = shard.lookup(i * 2);
        ASSERT_NE(nullptr, p) << "block " << i << " evicted at capacity";
        EXPECT_EQ(static_cast<uint8_t>(i), p[0]);
    }
}

// ---------------------------------------------------------------------------
// Eviction order: fill beyond capacity, verify the least-recently-used entry
// is evicted (strict LRU semantics).
//
// NOTE: lookup() moves the hit entry to MRU, so assertions that call lookup()
// perturb the recency order. We therefore structure the sequence so each
// verification reads a *different* key at most once, and reason about the
// final state in a single pass.
// ---------------------------------------------------------------------------
TEST(ShardedLRU, LruEvictionOrder) {
    // Capacity 3 in a single shard (2 shards; even indices -> shard 0).
    ShardedLRUCache cache(2, 3, kBlockSize);
    auto& shard = cache.shard(0);

    // Insert 0, 2, 4. Recency (MRU->LRU): 4, 2, 0. LRU = 0.
    {
        ScopedWriteLock lock(shard.mutex());
        for (uint64_t i : {0ull, 2ull, 4ull}) {
            std::vector<uint8_t> data = make_block(static_cast<uint8_t>(i));
            shard.insert(i, data.data(), kBlockSize);
        }
    }

    // Touch block 0 (MRU). Recency: 0, 4, 2. LRU = 2.
    {
        ScopedWriteLock lock(shard.mutex());
        ASSERT_NE(nullptr, shard.lookup(0));
    }

    // Insert 6 -> evicts LRU (block 2). Recency: 6, 0, 4. LRU = 4.
    {
        ScopedWriteLock lock(shard.mutex());
        std::vector<uint8_t> data = make_block(6);
        shard.insert(6, data.data(), kBlockSize);
    }
    // Verify block 2 is gone. Single lookup that only confirms absence.
    {
        ScopedWriteLock lock(shard.mutex());
        EXPECT_EQ(nullptr, shard.lookup(2)) << "LRU block 2 should be evicted";
    }

    // Insert 8 -> evicts current LRU (block 4). Recency: 8, 6, 0. LRU = 0.
    // (We touched nothing since insert 6, so 4 is still the tail.)
    {
        ScopedWriteLock lock(shard.mutex());
        std::vector<uint8_t> data = make_block(8);
        shard.insert(8, data.data(), kBlockSize);
    }

    // Final state: blocks {0, 6, 8} resident; {2, 4} evicted.
    // Check each exactly once (lookup moves to front, but we're done after).
    {
        ScopedWriteLock lock(shard.mutex());
        EXPECT_EQ(nullptr, shard.lookup(4)) << "LRU block 4 should be evicted";
        EXPECT_NE(nullptr, shard.lookup(0)) << "block 0 should be resident";
        EXPECT_NE(nullptr, shard.lookup(6)) << "block 6 should be resident";
        EXPECT_NE(nullptr, shard.lookup(8)) << "block 8 should be resident";
    }
}

// ---------------------------------------------------------------------------
// Insert over an existing key updates in place (no growth, data overwritten).
// ---------------------------------------------------------------------------
TEST(ShardedLRU, InsertUpdatesExisting) {
    ShardedLRUCache cache(2, 4, kBlockSize);
    auto& shard = cache.shard(0);

    {
        ScopedWriteLock lock(shard.mutex());
        std::vector<uint8_t> a = make_block(1);
        shard.insert(0, a.data(), kBlockSize);
        std::vector<uint8_t> b = make_block(2);
        shard.insert(0, b.data(), kBlockSize);  // overwrite
    }
    {
        ScopedWriteLock lock(shard.mutex());
        uint8_t* p = shard.lookup(0);
        ASSERT_NE(nullptr, p);
        EXPECT_EQ(2, p[0]);
    }
}

// ---------------------------------------------------------------------------
// Concurrent stress: many threads hammering insert/lookup across shards.
// Verifies no corruption / crashes under the write-locked access pattern.
// ---------------------------------------------------------------------------
TEST(ShardedLRU, ConcurrentStress) {
    const uint32_t kShards = 4;
    const uint32_t kCapPerShard = 16;
    const int kThreads = 8;
    const int kIters = 2000;

    ShardedLRUCache cache(kShards, kCapPerShard, kBlockSize);

    auto worker = [&cache]() {
        std::vector<uint8_t> data = make_block(0x5A);
        for (int i = 0; i < kIters; ++i) {
            // Cycle through many block indices so they spread across shards.
            uint64_t idx = static_cast<uint64_t>(i);
            LRUShard& s = cache.shard(idx);
            ScopedWriteLock lock(s.mutex());
            uint8_t* p = s.lookup(idx);
            if (p == nullptr) {
                p = s.insert(idx, data.data(), kBlockSize);
            }
            ASSERT_NE(nullptr, p);
            // Verify the sentinel survives (no torn write from another thread).
            EXPECT_EQ(0x5A, p[0]);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker);
    for (auto& th : threads) th.join();

    // If we reach here without crashing / failing an assertion, concurrency is
    // sound under the Phase 1 write-lock-everything contract.
    SUCCEED();
}

}  // namespace
}  // namespace sextant
