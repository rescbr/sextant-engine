// Dedicated tests for the thread-local L1 block cache (TLBlockCache).
//
// These verify:
//   1. Unit: TLBlockCache insert/lookup hits, miss on unknown key, and FIFO
//      replacement after capacity+1 inserts.
//   2. Integration: repeated pin_node for the same block is served from L1
//      (graph_reads counter does not increment after the first miss), and
//      the tl_hits()/tl_misses() accessors reflect L1 activity.
//   3. Per-instance isolation: two distinct PagedNodeStore objects do NOT
//      share L1 entries (a block key in store A must not return store B's
//      data).

#include <gtest/gtest.h>
#include "engine/fbin_source.hpp"
#include "sextant/engine.hpp"
#include "sextant/types.hpp"
#include "storage/block_cache.hpp"
#include "storage/node_store.hpp"
#include "storage/tl_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace sextant {
namespace {

/// Helper: snapshot the global epoch so a test can restore it on exit. The
/// epoch is a process-global counter shared across all tests; bumping it in
/// one test (to exercise staleness) must not poison later tests.
struct EpochGuard {
    uint64_t saved;
    EpochGuard() : saved(g_global_epoch.load(std::memory_order_relaxed)) {}
    ~EpochGuard() {
        g_global_epoch.store(saved, std::memory_order_relaxed);
    }
};

/// Write n×dim random float vectors to a .fbin file. Returns the path.
static std::string write_random_fbin(const std::string& name, uint32_t n,
                                       uint32_t dim, uint64_t seed = 42) {
    std::string path =
        (std::filesystem::temp_directory_path() / name).string();
    FILE* fp = std::fopen(path.c_str(), "wb");
    EXPECT_NE(fp, nullptr);
    std::fwrite(&n, sizeof(n), 1, fp);
    std::fwrite(&dim, sizeof(dim), 1, fp);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> row(dim);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t d = 0; d < dim; d++) row[d] = dist(rng);
        std::fwrite(row.data(), sizeof(float), dim, fp);
    }
    std::fclose(fp);
    return path;
}

static void remove_sidecars(const std::string& base) {
    for (const char* suf : {".graph", ".codes", ".meta", ".manifest"}) {
        std::remove((base + suf).c_str());
    }
}

// ---------------------------------------------------------------------------
// Unit: basic insert + lookup hit (pointer-based, epoch-valid).
// ---------------------------------------------------------------------------
TEST(TLBlockCache, InsertAndLookupHit) {
    EpochGuard eg;
    TLBlockCache cache;
    // Backing storage the pointer references (the L1 does NOT copy it).
    std::array<uint8_t, kBlockSize> backing{};
    for (size_t i = 0; i < kBlockSize; i++)
        backing[i] = static_cast<uint8_t>(i & 0xFF);

    const uint64_t epoch = g_global_epoch.load(std::memory_order_relaxed);
    cache.insert(42, backing.data(), epoch);
    const uint8_t* hit = cache.lookup(42, epoch);
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(hit, backing.data());  // pointer identity — no copy
    EXPECT_EQ(std::memcmp(hit, backing.data(), kBlockSize), 0);
}

// ---------------------------------------------------------------------------
// Unit: lookup miss for an unknown key.
// ---------------------------------------------------------------------------
TEST(TLBlockCache, LookupMiss) {
    EpochGuard eg;
    TLBlockCache cache;
    const uint64_t epoch = g_global_epoch.load(std::memory_order_relaxed);
    EXPECT_EQ(cache.lookup(999, epoch), nullptr);

    // Insert one key; an unrelated key still misses.
    std::array<uint8_t, kBlockSize> backing{};
    cache.insert(1, backing.data(), epoch);
    EXPECT_EQ(cache.lookup(2, epoch), nullptr);
}

// ---------------------------------------------------------------------------
// Unit: FIFO replacement. After kCapacity+1 inserts, the oldest entry is
// evicted (miss) and the newest is still present (hit).
// ---------------------------------------------------------------------------
TEST(TLBlockCache, FIFOReplacement) {
    EpochGuard eg;
    TLBlockCache cache;
    std::array<uint8_t, kBlockSize> backing{};
    const uint64_t epoch = g_global_epoch.load(std::memory_order_relaxed);

    // Fill all slots with distinct keys 0..kCapacity-1.
    for (uint32_t i = 0; i < TLBlockCache::kCapacity; i++) {
        cache.insert(i, backing.data(), epoch);
    }
    // All should still be present.
    for (uint32_t i = 0; i < TLBlockCache::kCapacity; i++) {
        EXPECT_NE(cache.lookup(i, epoch), nullptr) << "key " << i << " should be cached";
    }

    // Insert one more — this evicts key 0 (FIFO: slot 0 was first).
    cache.insert(TLBlockCache::kCapacity, backing.data(), epoch);
    EXPECT_EQ(cache.lookup(0, epoch), nullptr)
        << "key 0 should have been evicted by FIFO replacement";
    EXPECT_NE(cache.lookup(TLBlockCache::kCapacity, epoch), nullptr);
}

// ---------------------------------------------------------------------------
// Unit: lookup is exact-match (no false positives from partial keys).
// ---------------------------------------------------------------------------
TEST(TLBlockCache, NoFalsePositiveOnAdjacentKeys) {
    EpochGuard eg;
    TLBlockCache cache;
    std::array<uint8_t, kBlockSize> backing{};
    const uint64_t epoch = g_global_epoch.load(std::memory_order_relaxed);
    cache.insert(100, backing.data(), epoch);
    // Adjacent keys must not spuriously match.
    EXPECT_EQ(cache.lookup(99, epoch), nullptr);
    EXPECT_EQ(cache.lookup(101, epoch), nullptr);
    EXPECT_NE(cache.lookup(100, epoch), nullptr);
}

// ---------------------------------------------------------------------------
// Unit: epoch staleness. An entry inserted at epoch E must return a hit while
// the global epoch is still E, and a miss (nullptr) after the global epoch
// advances (simulating an L2 eviction that would invalidate the pointer).
// ---------------------------------------------------------------------------
TEST(TLBlockCache, EpochStalenessInvalidatesPointer) {
    EpochGuard eg;
    TLBlockCache cache;
    std::array<uint8_t, kBlockSize> backing{};

    // Insert at the current epoch.
    g_global_epoch.store(1, std::memory_order_relaxed);
    cache.insert(7, backing.data(), /*epoch=*/1);

    // Same epoch → hit.
    EXPECT_NE(cache.lookup(7, /*current_epoch=*/1), nullptr);

    // Simulate an eviction: the global epoch advances. The L1 entry's stored
    // epoch (1) no longer matches the current epoch (2) → miss, even though
    // the key is present. This is the use-after-free guard.
    g_global_epoch.store(2, std::memory_order_relaxed);
    EXPECT_EQ(cache.lookup(7, /*current_epoch=*/2), nullptr)
        << "stale entry (epoch mismatch) must be treated as a miss";
}

// ---------------------------------------------------------------------------
// Integration: repeated pin_node for the same block is served from L1.
// After the first miss (which reads from disk + populates L1), subsequent
// pins of nodes in the same block must NOT increment graph_reads.
// ---------------------------------------------------------------------------
TEST(TLBlockCache, IntegrationL1ServesRepeatedPins) {
    const uint32_t n = 400;
    const uint32_t dim = 16;
    const std::string fbin =
        write_random_fbin("tl_cache_integ.fbin", n, dim);
    const std::string index_path =
        (std::filesystem::temp_directory_path() / "tl_cache_integ_idx").string();
    remove_sidecars(index_path);

    {
        Engine engine;
        FbinSource source(fbin);
        engine.build(source, index_path, BuildConfig{});
    }

    uint32_t node_size = 0;
    uint8_t code_size = 0;
    {
        std::error_code ec;
        const uint64_t gsize = std::filesystem::file_size(index_path + ".graph", ec);
        const uint64_t csize = std::filesystem::file_size(index_path + ".codes", ec);
        ASSERT_FALSE(ec);
        node_size = static_cast<uint32_t>((gsize - 64) / n);
        code_size = static_cast<uint8_t>((csize - 64) / n);
        ASSERT_GT(node_size, 0u);
    }

    // Tiny shared L2 (2 blocks) so that without L1, repeated pins of the same
    // block could still hit L2 — but the L1 should intercept FIRST.
    PagedNodeStore store(index_path + ".graph", index_path + ".codes",
                         node_size, code_size,
                         /*num_shards=*/2,
                         /*cache_size_bytes=*/2ull * 256 * 1024);

    // First pin of node 0 → L1 miss → L2 miss → disk read.
    PinResult pr1 = store.pin_node(0);
    ASSERT_NE(pr1.data, nullptr);
    store.unpin_node(0);
    const uint64_t reads_after_first = store.graph_reads();
    EXPECT_GE(reads_after_first, 1u);

    // Second pin of node 0 → L1 hit (no new read).
    PinResult pr2 = store.pin_node(0);
    ASSERT_NE(pr2.data, nullptr);
    store.unpin_node(0);
    EXPECT_EQ(store.graph_reads(), reads_after_first)
        << "repeated pin should be served from L1 without a new disk read";

    // The L1 counters should reflect exactly one miss and at least one hit.
    EXPECT_EQ(store.tl_misses(), 1u);
    EXPECT_GE(store.tl_hits(), 1u);

    remove_sidecars(index_path);
    std::remove(fbin.c_str());
}

// ---------------------------------------------------------------------------
// Per-instance isolation: two PagedNodeStore objects must not share L1
// entries. We build two small indices, pin node 0 in store A (L1 miss → read),
// then pin node 0 in store B. Store B must ALSO miss (its L1 is independent),
// proving no cross-instance leakage.
// ---------------------------------------------------------------------------
TEST(TLBlockCache, PerInstanceIsolation) {
    const uint32_t n = 400;
    const uint32_t dim = 16;

    // Two separate indices.
    const std::string fbin_a = write_random_fbin("tl_cache_iso_a.fbin", n, dim, 1);
    const std::string fbin_b = write_random_fbin("tl_cache_iso_b.fbin", n, dim, 2);
    const std::string idx_a =
        (std::filesystem::temp_directory_path() / "tl_cache_iso_a_idx").string();
    const std::string idx_b =
        (std::filesystem::temp_directory_path() / "tl_cache_iso_b_idx").string();
    remove_sidecars(idx_a);
    remove_sidecars(idx_b);

    {
        Engine ea; FbinSource sa(fbin_a); ea.build(sa, idx_a, BuildConfig{});
        Engine eb; FbinSource sb(fbin_b); eb.build(sb, idx_b, BuildConfig{});
    }

    uint32_t node_size_a = 0, node_size_b = 0;
    uint8_t code_size_a = 0, code_size_b = 0;
    {
        std::error_code ec;
        node_size_a = static_cast<uint32_t>(
            (std::filesystem::file_size(idx_a + ".graph", ec) - 64) / n);
        code_size_a = static_cast<uint8_t>(
            (std::filesystem::file_size(idx_a + ".codes", ec) - 64) / n);
        node_size_b = static_cast<uint32_t>(
            (std::filesystem::file_size(idx_b + ".graph", ec) - 64) / n);
        code_size_b = static_cast<uint8_t>(
            (std::filesystem::file_size(idx_b + ".codes", ec) - 64) / n);
        ASSERT_GT(node_size_a, 0u);
        ASSERT_GT(node_size_b, 0u);
    }

    PagedNodeStore store_a(idx_a + ".graph", idx_a + ".codes",
                           node_size_a, code_size_a, 2, 2ull * 256 * 1024);
    PagedNodeStore store_b(idx_b + ".graph", idx_b + ".codes",
                           node_size_b, code_size_b, 2, 2ull * 256 * 1024);

    // Pin node 0 in store A → its L1 misses, reads from disk.
    store_a.pin_node(0);
    store_a.unpin_node(0);
    EXPECT_EQ(store_a.graph_reads(), 1u);
    EXPECT_EQ(store_a.tl_misses(), 1u);
    EXPECT_EQ(store_a.tl_hits(), 0u);

    // Pin node 0 in store B → its L1 must ALSO miss (independent cache).
    store_b.pin_node(0);
    store_b.unpin_node(0);
    EXPECT_EQ(store_b.graph_reads(), 1u)
        << "store B must read from disk — its L1 is independent of store A's";
    EXPECT_EQ(store_b.tl_misses(), 1u);
    EXPECT_EQ(store_b.tl_hits(), 0u);

    // Store A's counters are unchanged by store B's activity.
    EXPECT_EQ(store_a.tl_hits(), 0u);
    EXPECT_EQ(store_a.tl_misses(), 1u);

    remove_sidecars(idx_a);
    remove_sidecars(idx_b);
    std::remove(fbin_a.c_str());
    std::remove(fbin_b.c_str());
}

}  // namespace
}  // namespace sextant
