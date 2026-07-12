#include <gtest/gtest.h>
#include "storage/direct_io.hpp"
#include "storage/block_allocator.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

namespace sextant {
namespace {

// ---------------------------------------------------------------------------
// DirectFile: write/read byte-exact round-trip
// ---------------------------------------------------------------------------
TEST(DirectFile, WriteRead) {
    std::string path = "test_direct_io_tmp.bin";
    constexpr size_t kSize = kBlockSize;
    uint8_t* wbuf = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, kSize));
    uint8_t* rbuf = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, kSize));

    for (size_t i = 0; i < kSize; ++i) wbuf[i] = static_cast<uint8_t>(i);

    {
        DirectFile f(path, true);
        f.pwrite_aligned(wbuf, kSize, 0);
        f.sync();
    }
    {
        DirectFile f(path, false);
        f.pread_aligned(rbuf, kSize, 0);
    }

    EXPECT_EQ(0, std::memcmp(wbuf, rbuf, kSize));

    aligned_free(wbuf);
    aligned_free(rbuf);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// DirectFile: aligned buffers are actually kDiskAlign-aligned.
// ---------------------------------------------------------------------------
TEST(DirectFile, AlignedBuffersAreAligned) {
    for (int i = 0; i < 16; ++i) {
        void* p = aligned_alloc(kDiskAlign, kBlockSize);
        ASSERT_NE(nullptr, p);
        EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(p) % kDiskAlign)
            << "allocation " << i << " not 4096-aligned";
        aligned_free(p);
    }
}

// ---------------------------------------------------------------------------
// DirectFile: multiple blocks at distinct offsets round-trip independently.
// ---------------------------------------------------------------------------
TEST(DirectFile, MultiBlockOffsets) {
    std::string path = "test_direct_io_multi.bin";
    constexpr uint32_t kNumBlocks = 4;
    std::vector<uint8_t*> wbufs, rbufs;
    for (uint32_t i = 0; i < kNumBlocks; ++i) {
        uint8_t* w = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, kBlockSize));
        uint8_t* r = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, kBlockSize));
        // Distinct fill per block so offsets can't mask a bug.
        std::memset(w, static_cast<int>(0xA0 + i), kBlockSize);
        std::memset(r, 0, kBlockSize);
        wbufs.push_back(w);
        rbufs.push_back(r);
    }

    {
        DirectFile f(path, true);
        for (uint32_t i = 0; i < kNumBlocks; ++i) {
            f.pwrite_aligned(wbufs[i], kBlockSize,
                             static_cast<uint64_t>(i) * kBlockSize);
        }
        f.sync();
    }
    {
        DirectFile f(path, false);
        EXPECT_EQ(static_cast<uint64_t>(kNumBlocks) * kBlockSize, f.size());
        for (uint32_t i = 0; i < kNumBlocks; ++i) {
            f.pread_aligned(rbufs[i], kBlockSize,
                            static_cast<uint64_t>(i) * kBlockSize);
        }
    }

    for (uint32_t i = 0; i < kNumBlocks; ++i) {
        EXPECT_EQ(0, std::memcmp(wbufs[i], rbufs[i], kBlockSize))
            << "block " << i << " mismatch";
    }

    for (uint32_t i = 0; i < kNumBlocks; ++i) {
        aligned_free(wbufs[i]);
        aligned_free(rbufs[i]);
    }
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// BlockAllocator: acquire returns aligned, non-null blocks up to capacity.
// ---------------------------------------------------------------------------
TEST(BlockAllocator, AcquireAlignedAndCapacity) {
    const uint32_t kNum = 8;
    BlockAllocator alloc(kNum);

    std::vector<uint8_t*> blocks;
    for (uint32_t i = 0; i < kNum; ++i) {
        uint8_t* b = alloc.acquire();
        ASSERT_NE(nullptr, b);
        EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(b) % kDiskAlign)
            << "block " << i << " not 4096-aligned";
        blocks.push_back(b);
    }
    // Pool exhausted.
    EXPECT_EQ(nullptr, alloc.acquire());
    EXPECT_EQ(0u, alloc.available());
    EXPECT_EQ(kNum, alloc.capacity());

    for (uint8_t* b : blocks) alloc.release(b);
}

// ---------------------------------------------------------------------------
// BlockAllocator: release then acquire reuses free-list (no leak/exhaustion).
// ---------------------------------------------------------------------------
TEST(BlockAllocator, ReleaseReusesFreeList) {
    const uint32_t kNum = 4;
    BlockAllocator alloc(kNum);

    // Acquire all, then release all.
    std::vector<uint8_t*> first_round;
    for (uint32_t i = 0; i < kNum; ++i) {
        first_round.push_back(alloc.acquire());
        ASSERT_NE(nullptr, first_round.back());
    }
    for (uint8_t* b : first_round) alloc.release(b);

    // Second round: must be able to acquire the full capacity again, and the
    // set of pointers returned must be exactly the same set (free-list reuse).
    std::vector<uint8_t*> second_round;
    for (uint32_t i = 0; i < kNum; ++i) {
        uint8_t* b = alloc.acquire();
        ASSERT_NE(nullptr, b);
        second_round.push_back(b);
    }
    EXPECT_EQ(nullptr, alloc.acquire());

    // Same set of addresses, order-independent.
    std::vector<uint8_t*> a = first_round, b = second_round;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    EXPECT_EQ(a, b);

    for (uint8_t* blk : second_round) alloc.release(blk);
    EXPECT_EQ(kNum, alloc.available());
}

// ---------------------------------------------------------------------------
// BlockAllocator: releasing nullptr is a safe no-op.
// ---------------------------------------------------------------------------
TEST(BlockAllocator, ReleaseNullptrIsNoop) {
    BlockAllocator alloc(2);
    alloc.release(nullptr);  // must not crash
    EXPECT_EQ(2u, alloc.available());
}

// ---------------------------------------------------------------------------
// DirectFile: file opens successfully on the platform (validates that the
// O_DIRECT / F_NOCACHE open path works at all).
// ---------------------------------------------------------------------------
TEST(DirectFile, OpenSucceeds) {
    std::string path = "test_direct_io_open.bin";
    {
        DirectFile f(path, true);  // create
        EXPECT_GE(f.fd(), 0);
    }
    {
        DirectFile f(path, false);  // open existing
        EXPECT_GE(f.fd(), 0);
        EXPECT_EQ(0u, f.size());
    }
    std::remove(path.c_str());
}

}  // namespace
}  // namespace sextant
