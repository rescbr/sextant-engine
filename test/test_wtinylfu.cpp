#include <gtest/gtest.h>
#include "storage/frequency_sketch.hpp"
#include "storage/block_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sextant {
namespace {

// Helper: build a kBlockSize buffer tagged with a sentinel value.
static std::vector<uint8_t> make_block(uint8_t tag) {
    return std::vector<uint8_t>(kBlockSize, tag);
}

// ===========================================================================
// FrequencySketch
// ===========================================================================

// ---------------------------------------------------------------------------
// An uninitialized sketch reports frequency 0 for everything.
// ---------------------------------------------------------------------------
TEST(FrequencySketch, UninitializedReportsZero) {
    FrequencySketch s;
    EXPECT_TRUE(s.is_not_initialized());
    EXPECT_EQ(0u, s.frequency(42));
    s.increment(42);  // no-op on uninitialized sketch
    EXPECT_EQ(0u, s.frequency(42));
}

// ---------------------------------------------------------------------------
// frequency() reflects the number of increments (exact for a handful of keys
// with no collisions and before aging kicks in).
// ---------------------------------------------------------------------------
TEST(FrequencySketch, TracksFrequency) {
    FrequencySketch s;
    s.ensure_capacity(256);

    for (int i = 0; i < 5; ++i) s.increment(100);
    for (int i = 0; i < 2; ++i) s.increment(200);
    s.increment(300);

    EXPECT_EQ(5u, s.frequency(100));
    EXPECT_EQ(2u, s.frequency(200));
    EXPECT_EQ(1u, s.frequency(300));
    EXPECT_EQ(0u, s.frequency(999));  // never seen
}

// ---------------------------------------------------------------------------
// Counters saturate at 15 (4-bit max).
// ---------------------------------------------------------------------------
TEST(FrequencySketch, SaturatesAt15) {
    FrequencySketch s;
    s.ensure_capacity(256);
    for (int i = 0; i < 100; ++i) s.increment(7);
    EXPECT_EQ(15u, s.frequency(7));
}

// ---------------------------------------------------------------------------
// ensure_capacity is idempotent and does not forget counts when not resizing.
// ---------------------------------------------------------------------------
TEST(FrequencySketch, EnsureCapacityIdempotent) {
    FrequencySketch s;
    s.ensure_capacity(256);
    for (int i = 0; i < 3; ++i) s.increment(1);
    ASSERT_EQ(3u, s.frequency(1));
    s.ensure_capacity(128);  // smaller: no-op
    EXPECT_EQ(3u, s.frequency(1));
}

// ---------------------------------------------------------------------------
// Distinct keys tracked independently: a frequent key has a strictly higher
// estimated frequency than a rare one. This is the property the admission
// filter relies on.
// ---------------------------------------------------------------------------
TEST(FrequencySketch, FrequencyOrderingPreserved) {
    FrequencySketch s;
    s.ensure_capacity(512);
    for (int i = 0; i < 10; ++i) s.increment(1);  // hot
    for (int i = 0; i < 2; ++i) s.increment(2);   // warm
    s.increment(3);                                // cold
    EXPECT_GT(s.frequency(1), s.frequency(2));
    EXPECT_GT(s.frequency(2), s.frequency(3));
}

// ===========================================================================
// W-TinyLFU shard: basic insert / lookup
// ===========================================================================

// ---------------------------------------------------------------------------
// Basic insert + lookup round-trip.
// ---------------------------------------------------------------------------
TEST(WTinyLFU, BasicInsertLookup) {
    BlockCache cache(4, 8, kBlockSize);
    std::vector<uint8_t> data = make_block(42);

    auto& shard = cache.shard(0);
    ScopedWriteLock lock(shard.mutex());
    shard.insert(0, data.data(), kBlockSize);
    uint8_t* ptr = shard.lookup_unlocked(0);
    ASSERT_NE(nullptr, ptr);
    EXPECT_EQ(42, ptr[0]);
}

// ---------------------------------------------------------------------------
// At-capacity: all blocks fit (no eviction at exact capacity).
// ---------------------------------------------------------------------------
TEST(WTinyLFU, AtCapacityAllResident) {
    BlockCache cache(2, 8, kBlockSize);
    auto& shard = cache.shard(0);
    ScopedWriteLock lock(shard.mutex());

    for (uint64_t i = 0; i < 8; ++i) {
        std::vector<uint8_t> data = make_block(static_cast<uint8_t>(i));
        shard.insert(i * 2, data.data(), kBlockSize);
    }
    for (uint64_t i = 0; i < 8; ++i) {
        uint8_t* p = shard.lookup_unlocked(i * 2);
        ASSERT_NE(nullptr, p) << "block " << i << " evicted at capacity";
        EXPECT_EQ(static_cast<uint8_t>(i), p[0]);
    }
}

// ---------------------------------------------------------------------------
// Insert over an existing key updates in place (data overwritten, no growth).
// ---------------------------------------------------------------------------
TEST(WTinyLFU, InsertUpdatesExisting) {
    BlockCache cache(2, 4, kBlockSize);
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
        uint8_t* p = shard.lookup_unlocked(0);
        ASSERT_NE(nullptr, p);
        EXPECT_EQ(2, p[0]);
    }
}

// ---------------------------------------------------------------------------
// capacity_ == 0: insert returns nullptr (shard holds nothing).
// ---------------------------------------------------------------------------
TEST(WTinyLFU, ZeroCapacityReturnsNull) {
    BlockCache cache(2, 0, kBlockSize);
    auto& shard = cache.shard(0);
    ScopedWriteLock lock(shard.mutex());
    std::vector<uint8_t> data = make_block(1);
    EXPECT_EQ(nullptr, shard.insert(0, data.data(), kBlockSize));
    EXPECT_EQ(nullptr, shard.lookup_unlocked(0));
}

// ===========================================================================
// W-TinyLFU: scan resistance
// ===========================================================================

// ---------------------------------------------------------------------------
// Scan resistance: a working set accessed multiple times survives a flood of
// one-time (scan) insertions. With plain LRU the scan would evict the working
// set; W-TinyLFU's admission filter keeps the hot blocks.
// ---------------------------------------------------------------------------
TEST(WTinyLFU, ScanResistance) {
    // Single shard (1 shard, all keys route there). Capacity 10:
    // window=2, protected=6, probation=2.
    BlockCache cache(1, 10, kBlockSize);
    auto& shard = cache.shard(0);

    // Prime a working set of 4 blocks, each accessed 3x (high frequency).
    for (int round = 0; round < 3; ++round) {
        ScopedWriteLock lock(shard.mutex());
        for (uint64_t k = 0; k < 4; ++k) {
            std::vector<uint8_t> data = make_block(static_cast<uint8_t>(k));
            shard.insert(k, data.data(), kBlockSize);
            shard.lookup_unlocked(k);
        }
    }

    // Flood with 20 one-time scan blocks.
    {
        ScopedWriteLock lock(shard.mutex());
        for (uint64_t k = 100; k < 120; ++k) {
            std::vector<uint8_t> data = make_block(static_cast<uint8_t>(k));
            shard.insert(k, data.data(), kBlockSize);
        }
    }

    // The entire working set must survive the scan.
    ScopedWriteLock lock(shard.mutex());
    for (uint64_t k = 0; k < 4; ++k) {
        EXPECT_NE(nullptr, shard.lookup_unlocked(k))
            << "working-set block " << k << " evicted by scan";
    }
}

// ===========================================================================
// W-TinyLFU: admission
// ===========================================================================

// ---------------------------------------------------------------------------
// Admission: a frequently-accessed block is retained over one-time blocks.
// Verify via stats() that admission decisions (admitted / rejected) actually
// occur under contention.
// ---------------------------------------------------------------------------
TEST(WTinyLFU, FrequentBlockAdmittedOverOneTime) {
    // Single shard, capacity 5: window=1, protected=3, probation=1.
    BlockCache cache(1, 5, kBlockSize);
    auto& shard = cache.shard(0);

    // Fill the cache with one-time blocks.
    {
        ScopedWriteLock lock(shard.mutex());
        for (uint64_t k = 0; k < 5; ++k) {
            std::vector<uint8_t> data = make_block(static_cast<uint8_t>(k));
            shard.insert(k, data.data(), kBlockSize);
        }
    }
    // Make block 0 hot.
    {
        ScopedWriteLock lock(shard.mutex());
        for (int n = 0; n < 5; ++n) {
            ASSERT_NE(nullptr, shard.lookup_unlocked(0));
        }
    }
    // Insert several one-time scan blocks; block 0 should survive.
    {
        ScopedWriteLock lock(shard.mutex());
        for (uint64_t k = 10; k < 14; ++k) {
            std::vector<uint8_t> data = make_block(static_cast<uint8_t>(k));
            shard.insert(k, data.data(), kBlockSize);
        }
    }
    {
        ScopedWriteLock lock(shard.mutex());
        EXPECT_NE(nullptr, shard.lookup_unlocked(0))
            << "hot block 0 should survive one-time inserts";
    }

    // The admission path must have run (some window overflow events).
    CacheStats s = cache.stats();
    EXPECT_GT(s.evictions_window, 0u)
        << "expected window overflow / admission activity";
}

// ---------------------------------------------------------------------------
// Capacity is never exceeded, even under heavy churn.
// ---------------------------------------------------------------------------
TEST(WTinyLFU, CapacityNeverExceeded) {
    const uint32_t kCap = 16;
    BlockCache cache(1, kCap, kBlockSize);
    auto& shard = cache.shard(0);

    // Insert far more blocks than capacity; randomly touch a subset to build
    // frequency. The resident count must never exceed kCap.
    ScopedWriteLock lock(shard.mutex());
    for (uint64_t k = 0; k < 200; ++k) {
        std::vector<uint8_t> data = make_block(static_cast<uint8_t>(k & 0xff));
        shard.insert(k, data.data(), kBlockSize);
        // Touch a "hot" range repeatedly.
        if (k % 7 == 0) {
            for (uint64_t h = 0; h < 8; ++h) shard.lookup_unlocked(h);
        }
    }

    // Count resident blocks by probing every inserted key once. Because lookup
    // perturbs order, do it after the churn loop: the invariant we check is that
    // no *new* insert silently grew the cache beyond capacity. We approximate by
    // counting how many of the last `kCap + 4` blocks are resident — it must be
    // <= kCap (a shard cannot hold more than its capacity).
    uint32_t resident_tail = 0;
    for (uint64_t k = 200 - (kCap + 4); k < 200; ++k) {
        if (shard.lookup_unlocked(k) != nullptr) ++resident_tail;
    }
    EXPECT_LE(resident_tail, kCap);
}

// ===========================================================================
// W-TinyLFU: adaptive (hill-climbing) window sizing
// ===========================================================================

// ---------------------------------------------------------------------------
// AdmissionFallback: early in the run, when probation holds only the
// candidate (no distinct victim), the candidate must be admitted (not
// rejected). This is the fix for the 0% admission rate root cause. We verify
// that after filling the cache, the main space (probation/protected) is
// non-empty — i.e. at least one block was admitted, not all rejected.
// ---------------------------------------------------------------------------
TEST(WTinyLFU, AdmissionFallback) {
    // Single shard, capacity 4. Initial window = 1 (1% of 4, clamped to >=1),
    // protected = 2, probation = 1.
    BlockCache cache(1, 4, kBlockSize);
    auto& shard = cache.shard(0);
    ScopedWriteLock lock(shard.mutex());

    // Insert exactly capacity blocks (0..3). Each insert beyond the window
    // quota must overflow into probation. With the fallback fix, these are
    // admitted rather than evicted.
    for (uint64_t k = 0; k < 4; ++k) {
        std::vector<uint8_t> data = make_block(static_cast<uint8_t>(k));
        shard.insert(k, data.data(), kBlockSize);
    }

    // The cache should be full and all 4 blocks resident: none were rejected
    // because there was no distinct victim to lose against.
    for (uint64_t k = 0; k < 4; ++k) {
        EXPECT_NE(nullptr, shard.lookup_unlocked(k))
            << "block " << k << " should be admitted at capacity (no victim)";
    }

    CacheStats s = cache.stats();
    // Window overflow must have occurred (capacity > window size).
    EXPECT_GT(s.evictions_window, 0u);
    // No rejections yet: the fallback path admits when there's no distinct
    // victim. (admitted + rejected may both be 0 if all overflows hit the
    // fallback branch; the key assertion is that blocks survived.)
    EXPECT_EQ(s.evictions_rejected, 0u)
        << "no rejections expected when probation had no distinct victim";
}

// ---------------------------------------------------------------------------
// AdaptiveConvergence: run a mixed workload (hot working set + scan) for many
// accesses across MULTIPLE shards. The cache-level hill-climber aggregates
// hit/miss samples across all shards, computes ONE window ratio, and applies
// it uniformly. The window should grow away from the initial 1% configuration,
// and every shard should end up with the same window capacity.
// ---------------------------------------------------------------------------
TEST(WTinyLFU, AdaptiveConvergence) {
    // 4 shards, capacity 100 each → total capacity 400.
    // Per-shard initial window = 1% = 1 block.
    // Cache sample_size = 10 * 400 = 4000 accesses per climb().
    const uint32_t kShards = 4;
    const uint32_t kCapPerShard = 100;
    BlockCache cache(kShards, kCapPerShard, kBlockSize);

    // Sanity: the constructor starts every shard's window at ~1% of capacity.
    uint32_t initial_window = cache.max_window();
    ASSERT_LE(initial_window, 2u) << "window should start near 1% of capacity";

    // Workload: a hot working set of 20 blocks accessed repeatedly, plus a
    // scan of many one-time blocks. The hot set and scans spread across all
    // shards via block_idx % kShards. This favors a larger window (recency) so
    // the hill-climber should grow the window above its initial value.
    std::vector<uint8_t> data = make_block(0xAB);
    const uint64_t kHotSet = 20;
    // Run well past several sample periods (sample_size = 4000). Each iteration
    // issues ~4 accesses (2 lookups + up to 2 inserts), so 6000 iterations
    // comfortably exceeds multiple sample windows.
    for (uint64_t i = 0; i < 6000; ++i) {
        // Hot access (cyclic over the working set).
        uint64_t hot = i % kHotSet;
        {
            CacheShard& s = cache.shard(hot);
            ScopedWriteLock lock(s.mutex());
            if (s.lookup_unlocked(hot) == nullptr) {
                s.insert(hot, data.data(), kBlockSize);
            }
        }
        // Interspersed scan: one-time blocks (distinct per iteration).
        uint64_t scan = 100000 + i;
        {
            CacheShard& s = cache.shard(scan);
            ScopedWriteLock lock(s.mutex());
            if (s.lookup_unlocked(scan) == nullptr) {
                s.insert(scan, data.data(), kBlockSize);
            }
        }
    }

    // The cache-level hill-climber must have adjusted the window at least once.
    // For a scan-heavy + hot-set workload the optimal window is larger than 1%,
    // so the window should have grown.
    uint32_t final_window = cache.max_window();
    EXPECT_NE(final_window, initial_window)
        << "hill-climber should have adapted the window from its initial value";
    EXPECT_GT(final_window, initial_window)
        << "expected the window to grow under a recency-favoring workload";

    // All shards must share the SAME window capacity (the whole point of the
    // per-cache refactor: one ratio applied uniformly).
    for (uint32_t s = 0; s < kShards; ++s) {
        EXPECT_EQ(cache.shard(s).max_window(), final_window)
            << "shard " << s << " window diverged from the cache-level ratio";
    }
}

}  // namespace
}  // namespace sextant
