// Unit tests for ParquetSource — write a Parquet file via carquet's writer API,
// read it back via ParquetSource, verify vectors + filter columns round-trip.

#include <carquet/carquet.h>

#include "parquet_source.hpp"
#include <sextant/error.hpp>

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

/// RAII temp file: creates a unique path, deleted on destruction.
class TempFile {
public:
    TempFile(const std::string& suffix) {
        const auto tmpdir = std::filesystem::temp_directory_path();
        path_ = (tmpdir / ("sextant_test_XXXXXX" + suffix)).string();
        // Replace XXXXXX with PID-based unique suffix.
        auto pos = path_.find("XXXXXX");
        if (pos != std::string::npos) {
            path_.replace(pos, 6, std::to_string(getpid()));
        }
    }
    ~TempFile() { std::filesystem::remove(path_); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

/// Write a test Parquet file with:
///   - "embedding": FIXED_LEN_BYTE_ARRAY(dim × 4) — float vectors
///   - "category": BYTE_ARRAY (string)
///   - "year": INT32
/// Returns the path.
std::string write_test_parquet(const std::string& path, uint32_t n_rows,
                                uint32_t dim) {
    carquet_error_t err = {};

    // Create schema.
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) throw sextant::Error(sextant::ErrorCode::IoError, "schema create");

    auto check = [](carquet_status_t s, const char* what) {
        if (s != CARQUET_OK) {
            throw sextant::Error(sextant::ErrorCode::IoError,
                                 std::string(what) + ": " +
                                     carquet_status_string(s));
        }
    };

    // embedding: FIXED_LEN_BYTE_ARRAY of dim*4 bytes.
    check(carquet_schema_add_column(schema, "embedding",
                                    CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY, nullptr,
                                    CARQUET_REPETITION_REQUIRED,
                                    static_cast<int32_t>(dim * sizeof(float)), 0),
          "add embedding col");
    // category: optional string.
    check(carquet_schema_add_column(schema, "category", CARQUET_PHYSICAL_BYTE_ARRAY,
                                    nullptr, CARQUET_REPETITION_OPTIONAL, 0, 0),
          "add category col");
    // year: required int32.
    check(carquet_schema_add_column(schema, "year", CARQUET_PHYSICAL_INT32, nullptr,
                                    CARQUET_REPETITION_REQUIRED, 0, 0),
          "add year col");

    carquet_writer_options_t wopts;
    carquet_writer_options_init(&wopts);
    wopts.compression = CARQUET_COMPRESSION_ZSTD;

    carquet_writer_t* writer =
        carquet_writer_create(path.c_str(), schema, &wopts, &err);
    if (!writer) {
        carquet_schema_free(schema);
        throw sextant::Error(sextant::ErrorCode::IoError, "writer create");
    }

    // Generate vector data (n_rows × dim floats).
    std::vector<float> vecs(n_rows * dim);
    for (uint32_t i = 0; i < n_rows; ++i)
        for (uint32_t d = 0; d < dim; ++d)
            vecs[i * dim + d] = static_cast<float>(i * 100 + d);

    check(carquet_writer_write_batch(writer, 0, vecs.data(),
                                     static_cast<int64_t>(n_rows), nullptr, nullptr),
          "write embedding");

    // Write category column (BYTE_ARRAY: array of carquet_byte_array_t).
    std::vector<carquet_byte_array_t> cats(n_rows);
    std::vector<std::string> cat_strs(n_rows);
    // Only non-null values; use def_levels for optional column.
    // For simplicity, make every other row non-null.
    std::vector<int16_t> cat_defs(n_rows);
    uint32_t n_nonnull = 0;
    for (uint32_t i = 0; i < n_rows; ++i) {
        if (i % 3 != 0) {
            cat_strs[n_nonnull] = "cat" + std::to_string(i % 5);
            cats[n_nonnull].data =
                const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(
                    cat_strs[n_nonnull].data()));
            cats[n_nonnull].length =
                static_cast<int32_t>(cat_strs[n_nonnull].size());
            cat_defs[i] = 1;
            ++n_nonnull;
        } else {
            cat_defs[i] = 0;  // null
        }
    }
    check(carquet_writer_write_batch(writer, 1, cats.data(),
                                     static_cast<int64_t>(n_rows), cat_defs.data(),
                                     nullptr),
          "write category");

    // Write year column (INT32).
    std::vector<int32_t> years(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) years[i] = 2000 + static_cast<int32_t>(i);
    check(carquet_writer_write_batch(writer, 2, years.data(),
                                     static_cast<int64_t>(n_rows), nullptr, nullptr),
          "write year");

    carquet_status_t status = carquet_writer_close(writer);
    carquet_schema_free(schema);
    if (status != CARQUET_OK) {
        throw sextant::Error(sextant::ErrorCode::IoError,
                             "writer close failed: " +
                                 std::string(carquet_status_string(status)));
    }
    return path;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(ParquetSourceTest, OpensAndReadsBasicMetadata) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 100;
    constexpr uint32_t kDim = 16;
    write_test_parquet(tmp.path(), kN, kDim);

    sextant::ParquetSource src(tmp.path());
    EXPECT_EQ(src.dim(), kDim);
    EXPECT_EQ(src.count(), kN);
    EXPECT_FALSE(src.path().empty());

    // Schema should have 2 filter columns: category (String) + year (Int32).
    // embedding is the vector column, excluded from filter schema.
    auto schema = src.schema();
    ASSERT_EQ(schema.columns.size(), 2u);
    EXPECT_EQ(schema.columns[0].name, "category");
    EXPECT_EQ(schema.columns[0].type, sextant::ColumnType::String);
    EXPECT_EQ(schema.columns[1].name, "year");
    EXPECT_EQ(schema.columns[1].type, sextant::ColumnType::Int32);
}

TEST(ParquetSourceTest, ReadsAllVectorsInOnePass) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 50;
    constexpr uint32_t kDim = 8;
    write_test_parquet(tmp.path(), kN, kDim);

    sextant::ParquetSource src(tmp.path());
    src.reset();

    sextant::Chunk chunk;
    uint32_t total = 0;
    std::vector<sextant::RowId> seen_ids;
    while (src.next(chunk)) {
        ASSERT_NE(chunk.vectors, nullptr);
        ASSERT_NE(chunk.row_ids, nullptr);
        for (uint32_t i = 0; i < chunk.count; ++i) {
            // Verify vector data.
            const float* v = &chunk.vectors[i * kDim];
            for (uint32_t d = 0; d < kDim; ++d) {
                const uint32_t row = static_cast<uint32_t>(chunk.row_ids[i]);
                EXPECT_FLOAT_EQ(v[d], static_cast<float>(row * 100 + d))
                    << "mismatch at row " << row << " dim " << d;
            }
            seen_ids.push_back(chunk.row_ids[i]);
        }
        total += chunk.count;
    }
    EXPECT_EQ(total, kN);
    ASSERT_EQ(seen_ids.size(), kN);
    // Row IDs should be 0..kN-1.
    for (uint32_t i = 0; i < kN; ++i) {
        EXPECT_EQ(seen_ids[i], static_cast<sextant::RowId>(i));
    }
}

TEST(ParquetSourceTest, FilterColumnsRoundTrip) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 30;
    constexpr uint32_t kDim = 4;
    write_test_parquet(tmp.path(), kN, kDim);

    sextant::ParquetSource src(tmp.path());
    src.reset();

    sextant::Chunk chunk;
    uint32_t row_offset = 0;
    while (src.next(chunk)) {
        ASSERT_NE(chunk.filter_columns, nullptr);

        // Column 0 = "category" (String).
        const auto* sc = static_cast<const sextant::FilterStringColumn*>(
            chunk.filter_columns[0]);
        ASSERT_NE(sc, nullptr);
        ASSERT_NE(sc->offsets, nullptr);
        ASSERT_NE(sc->lengths, nullptr);

        // Column 1 = "year" (Int32).
        const auto* years = static_cast<const int32_t*>(chunk.filter_columns[1]);
        ASSERT_NE(years, nullptr);

        for (uint32_t i = 0; i < chunk.count; ++i) {
            const uint32_t global_row = row_offset + i;

            // Year check.
            EXPECT_EQ(years[i], static_cast<int32_t>(2000 + global_row))
                << "year mismatch at row " << global_row;

            // Category check: row is null if i % 3 == 0.
            if (global_row % 3 == 0) {
                EXPECT_EQ(sc->lengths[i], 0u) << "expected null string at row "
                                              << global_row;
            } else {
                std::string expected = "cat" + std::to_string(global_row % 5);
                std::string actual(sc->data + sc->offsets[i], sc->lengths[i]);
                EXPECT_EQ(actual, expected)
                    << "string mismatch at row " << global_row;
            }
        }
        row_offset += chunk.count;
    }
    EXPECT_EQ(row_offset, kN);
}

TEST(ParquetSourceTest, MultiplePassesViaReset) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 20;
    constexpr uint32_t kDim = 4;
    write_test_parquet(tmp.path(), kN, kDim);

    sextant::ParquetSource src(tmp.path());

    for (int pass = 0; pass < 3; ++pass) {
        src.reset();
        sextant::Chunk chunk;
        uint32_t total = 0;
        while (src.next(chunk)) {
            total += chunk.count;
        }
        EXPECT_EQ(total, kN) << "pass " << pass << " got wrong count";
    }
}

TEST(ParquetSourceTest, SmallBatchSizeChunksCorrectly) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 25;
    constexpr uint32_t kDim = 4;
    write_test_parquet(tmp.path(), kN, kDim);

    sextant::ParquetSourceConfig cfg;
    cfg.batch_size = 7;  // small batches to test chunking
    sextant::ParquetSource src(tmp.path(), cfg);
    src.reset();

    sextant::Chunk chunk;
    uint32_t total = 0;
    int n_chunks = 0;
    while (src.next(chunk)) {
        total += chunk.count;
        ++n_chunks;
    }
    EXPECT_EQ(total, kN);
    // With batch_size=7 and kN=25, expect ceil(25/7)=4 chunks.
    EXPECT_GE(n_chunks, 3);
}

TEST(ParquetSourceTest, ThrowsOnMissingVectorColumn) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 5;

    // Write a file WITHOUT an "embedding" column.
    carquet_error_t err = {};
    carquet_schema_t* schema = carquet_schema_create(&err);
    ASSERT_NE(schema, nullptr);
    ASSERT_EQ(carquet_schema_add_column(schema, "data", CARQUET_PHYSICAL_INT32,
                                        nullptr, CARQUET_REPETITION_REQUIRED, 0,
                                        0),
              CARQUET_OK);
    carquet_writer_options_t wopts;
    carquet_writer_options_init(&wopts);
    carquet_writer_t* w =
        carquet_writer_create(tmp.path().c_str(), schema, &wopts, &err);
    ASSERT_NE(w, nullptr);
    std::vector<int32_t> vals(kN, 42);
    ASSERT_EQ(carquet_writer_write_batch(w, 0, vals.data(), kN, nullptr, nullptr),
              CARQUET_OK);
    ASSERT_EQ(carquet_writer_close(w), CARQUET_OK);
    carquet_schema_free(schema);

    sextant::ParquetSourceConfig cfg;
    cfg.vector_col = "embedding";
    EXPECT_THROW(sextant::ParquetSource src(tmp.path(), cfg), sextant::Error);
}
