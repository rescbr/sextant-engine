#include <gtest/gtest.h>
#include "../src/storage/direct_io.hpp"
#include <cstdio>
#include <string>

namespace sextant {
namespace {

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

}  // namespace
}  // namespace sextant
