#include <gtest/gtest.h>
#include "engine/fbin_source.hpp"
#include "sextant/vector_source.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace sextant {
namespace {

/// Write an n×dim .fbin file with deterministic float values v(i,d) = i*dim+d.
static std::string write_fbin(const std::string& name, uint32_t n,
                              uint32_t dim) {
    std::string path = (std::filesystem::temp_directory_path() / name).string();
    FILE* fp = std::fopen(path.c_str(), "wb");
    EXPECT_NE(fp, nullptr);
    std::fwrite(&n, sizeof(n), 1, fp);
    std::fwrite(&dim, sizeof(dim), 1, fp);
    std::vector<float> row(dim);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            row[d] = static_cast<float>(i * dim + d);
        }
        std::fwrite(row.data(), sizeof(float), dim, fp);
    }
    std::fclose(fp);
    return path;
}

// ---------------------------------------------------------------------------
// dim() / count() from the header
// ---------------------------------------------------------------------------
TEST(FbinSource, HeaderFields) {
    const std::string path = write_fbin("fbin_hdr.fbin", 10, 8);
    FbinSource src(path);
    EXPECT_EQ(10u, src.count());
    EXPECT_EQ(8u, src.dim());
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// next() returns correct data, row_ids are 0-indexed sequential
// ---------------------------------------------------------------------------
TEST(FbinSource, NextReadsCorrectData) {
    const uint32_t n = 10;
    const uint32_t dim = 8;
    const std::string path = write_fbin("fbin_data.fbin", n, dim);
    FbinSource src(path, /*chunk_size=*/4);

    Chunk ch{};
    uint64_t seen = 0;
    while (src.next(ch)) {
        for (uint32_t r = 0; r < ch.count; r++) {
            const RowId rid = ch.row_ids[r];
            EXPECT_EQ(static_cast<RowId>(seen), rid);
            const float* vec = ch.vectors + static_cast<size_t>(r) * dim;
            for (uint32_t d = 0; d < dim; d++) {
                EXPECT_FLOAT_EQ(static_cast<float>(seen * dim + d), vec[d]);
            }
            seen++;
        }
    }
    EXPECT_EQ(static_cast<uint64_t>(n), seen);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// reset() allows re-reading from the start
// ---------------------------------------------------------------------------
TEST(FbinSource, ResetRereads) {
    const uint32_t n = 20;
    const uint32_t dim = 4;
    const std::string path = write_fbin("fbin_reset.fbin", n, dim);
    FbinSource src(path, /*chunk_size=*/7);

    // First pass.
    uint64_t seen1 = 0;
    Chunk ch{};
    while (src.next(ch)) {
        seen1 += ch.count;
    }
    EXPECT_EQ(static_cast<uint64_t>(n), seen1);

    // Reset and second pass should yield the same count + first row.
    src.reset();
    uint64_t seen2 = 0;
    ASSERT_TRUE(src.next(ch));
    // First row of the second pass should be row 0.
    EXPECT_EQ(0, ch.row_ids[0]);
    EXPECT_FLOAT_EQ(0.0f, ch.vectors[0]);
    seen2 += ch.count;
    while (src.next(ch)) {
        seen2 += ch.count;
    }
    EXPECT_EQ(static_cast<uint64_t>(n), seen2);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Chunk smaller than the file: many next() calls, last chunk is partial
// ---------------------------------------------------------------------------
TEST(FbinSource, PartialLastChunk) {
    const uint32_t n = 10;
    const uint32_t dim = 4;
    const std::string path = write_fbin("fbin_partial.fbin", n, dim);
    FbinSource src(path, /*chunk_size=*/3);  // 3 + 3 + 3 + 1

    Chunk ch{};
    std::vector<uint32_t> sizes;
    while (src.next(ch)) {
        sizes.push_back(ch.count);
    }
    ASSERT_EQ(4u, sizes.size());
    EXPECT_EQ(3u, sizes[0]);
    EXPECT_EQ(3u, sizes[1]);
    EXPECT_EQ(3u, sizes[2]);
    EXPECT_EQ(1u, sizes[3]);  // partial last chunk
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// .bbin (uint8) is cast to float on read (Issue 39: no normalization)
// ---------------------------------------------------------------------------
TEST(FbinSource, Uint8CastOnRead) {
    const uint32_t n = 5;
    const uint32_t dim = 4;
    std::string path =
        (std::filesystem::temp_directory_path() / "fbin_u8.bbin").string();
    FILE* fp = std::fopen(path.c_str(), "wb");
    EXPECT_NE(fp, nullptr);
    std::fwrite(&n, sizeof(n), 1, fp);
    std::fwrite(&dim, sizeof(dim), 1, fp);
    std::vector<uint8_t> bytes(n * dim);
    for (uint32_t i = 0; i < n * dim; i++) {
        bytes[i] = static_cast<uint8_t>(200 + i);  // > 127, tests unsigned cast
    }
    std::fwrite(bytes.data(), 1, bytes.size(), fp);
    std::fclose(fp);

    FbinSource src(path, /*chunk_size=*/n);
    Chunk ch{};
    ASSERT_TRUE(src.next(ch));
    ASSERT_EQ(n, ch.count);
    for (uint32_t i = 0; i < n * dim; i++) {
        EXPECT_FLOAT_EQ(static_cast<float>(bytes[i]),
                        ch.vectors[i])
            << "mismatch at element " << i;
    }
    // Specifically: 200 as float (not -56 from a signed cast).
    EXPECT_FLOAT_EQ(200.0f, ch.vectors[0]);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// .ibin (int8) is cast to float on read (Issue 39)
// ---------------------------------------------------------------------------
TEST(FbinSource, Int8CastOnRead) {
    const uint32_t n = 3;
    const uint32_t dim = 2;
    std::string path =
        (std::filesystem::temp_directory_path() / "fbin_i8.ibin").string();
    FILE* fp = std::fopen(path.c_str(), "wb");
    EXPECT_NE(fp, nullptr);
    std::fwrite(&n, sizeof(n), 1, fp);
    std::fwrite(&dim, sizeof(dim), 1, fp);
    int8_t vals[] = {-1, 100, -128, 127, 0, 42};
    std::fwrite(vals, 1, sizeof(vals), fp);
    std::fclose(fp);

    FbinSource src(path, /*chunk_size=*/n);
    Chunk ch{};
    ASSERT_TRUE(src.next(ch));
    ASSERT_EQ(n, ch.count);
    EXPECT_FLOAT_EQ(-1.0f, ch.vectors[0]);
    EXPECT_FLOAT_EQ(100.0f, ch.vectors[1]);
    EXPECT_FLOAT_EQ(-128.0f, ch.vectors[2]);
    EXPECT_FLOAT_EQ(127.0f, ch.vectors[3]);
    EXPECT_FLOAT_EQ(0.0f, ch.vectors[4]);
    EXPECT_FLOAT_EQ(42.0f, ch.vectors[5]);
    std::remove(path.c_str());
}

}  // namespace
}  // namespace sextant
