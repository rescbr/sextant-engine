// Unit tests for ParquetSource — write a Parquet file via carquet's writer API,
// read it back via ParquetSource, verify vectors + filter columns round-trip.

#include <carquet/carquet.h>

#include "parquet_source.hpp"
#include "parquet_glob_source.hpp"
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
// Helper: write a parquet with list<float> vector column
// ---------------------------------------------------------------------------

std::string write_list_float_parquet(const std::string& path, uint32_t n_rows,
                                      uint32_t dim) {
    carquet_error_t err = {};

    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) throw sextant::Error(sextant::ErrorCode::IoError, "schema create");

    // emb: list<float> (OPTIONAL container, OPTIONAL element)
    if (carquet_schema_add_list(schema, "emb", CARQUET_PHYSICAL_FLOAT,
                                nullptr, CARQUET_REPETITION_OPTIONAL, 0, 0) < 0) {
        carquet_schema_free(schema);
        throw sextant::Error(sextant::ErrorCode::IoError, "add list col failed");
    }

    carquet_writer_options_t wopts;
    carquet_writer_options_init(&wopts);
    wopts.compression = CARQUET_COMPRESSION_UNCOMPRESSED;

    carquet_writer_t* writer =
        carquet_writer_create(path.c_str(), schema, &wopts, &err);
    if (!writer) {
        carquet_schema_free(schema);
        throw sextant::Error(sextant::ErrorCode::IoError, "writer create");
    }

    // Generate vector data: each row is dim floats.
    std::vector<float> vecs(n_rows * dim);
    for (uint32_t i = 0; i < n_rows; ++i)
        for (uint32_t d = 0; d < dim; ++d)
            vecs[i * dim + d] = static_cast<float>(i * 100 + d);

    // Offsets: n_rows+1 entries, each list has exactly dim elements.
    std::vector<int32_t> offsets(n_rows + 1);
    for (uint32_t i = 0; i <= n_rows; ++i)
        offsets[i] = static_cast<int32_t>(i * dim);

    // The leaf column index is 0 (first and only leaf).
    carquet_status_t ws = carquet_writer_write_list_column(writer, 0,
            static_cast<int64_t>(n_rows), offsets.data(),
            nullptr,  // list_validity: all present
            vecs.data(), nullptr,  // values + value_validity: all present
            &err);
    if (ws != CARQUET_OK) {
        (void)carquet_writer_close(writer);
        carquet_schema_free(schema);
        throw sextant::Error(sextant::ErrorCode::IoError,
                             std::string("write list col: ") + carquet_status_string(ws));
    }

    carquet_status_t status = carquet_writer_close(writer);
    carquet_schema_free(schema);
    if (status != CARQUET_OK) {
        throw sextant::Error(sextant::ErrorCode::IoError,
                             "writer close failed");
    }
    return path;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Payload column round-trip
// ---------------------------------------------------------------------------

/// Write a parquet with a FLOAT vector col (dim>1 via FLBA) + "text" payload
/// col (BYTE_ARRAY, mixed lengths, nulls every 7th row, empty strings every
/// 13th) + "blob" FLBA payload col (8-byte stride).
std::string write_payload_parquet(const std::string& path, uint32_t n_rows,
                                  uint32_t dim, bool include_blob = true) {
    carquet_error_t err = {};
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) throw sextant::Error(sextant::ErrorCode::IoError, "schema create");

    auto check = [](carquet_status_t s, const char* what) {
        if (s != CARQUET_OK) {
            throw sextant::Error(sextant::ErrorCode::IoError,
                                 std::string(what) + ": " +
                                     carquet_status_string(s));
        }
    };

    check(carquet_schema_add_column(schema, "v",
                                    CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY, nullptr,
                                    CARQUET_REPETITION_REQUIRED,
                                    static_cast<int32_t>(dim * sizeof(float)), 0),
          "add v col");
    check(carquet_schema_add_column(schema, "text", CARQUET_PHYSICAL_BYTE_ARRAY,
                                    nullptr, CARQUET_REPETITION_OPTIONAL, 0, 0),
          "add text col");
    if (include_blob) {
        check(carquet_schema_add_column(schema, "blob",
                                        CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY, nullptr,
                                        CARQUET_REPETITION_OPTIONAL, 8, 0),
              "add blob col");
    }

    carquet_writer_options_t wopts;
    carquet_writer_options_init(&wopts);
    wopts.compression = CARQUET_COMPRESSION_ZSTD;
    carquet_writer_t* writer =
        carquet_writer_create(path.c_str(), schema, &wopts, &err);
    if (!writer) {
        carquet_schema_free(schema);
        throw sextant::Error(sextant::ErrorCode::IoError, "writer create");
    }

    std::vector<float> vecs(n_rows * dim);
    for (uint32_t i = 0; i < n_rows * dim; ++i) vecs[i] = static_cast<float>(i);
    check(carquet_writer_write_batch(writer, 0, vecs.data(),
                                     static_cast<int64_t>(n_rows), nullptr, nullptr),
          "write v");

    // text: "row-i" + i%23 'x's; null every 7th row; empty every 13th.
    // API contract: values are PACKED (non-null only), def_levels mark nulls.
    std::vector<std::string> texts;
    texts.reserve(n_rows);
    std::vector<carquet_byte_array_t> barr;
    barr.reserve(n_rows);
    std::vector<int16_t> defs(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        if (i % 7 == 0) {
            defs[i] = 0;
            continue;
        }
        texts.emplace_back("row-" + std::to_string(i) + std::string(i % 23, 'x'));
        if (i % 13 == 0) texts.back().clear();
        defs[i] = 1;
        carquet_byte_array_t ba;
        ba.data = const_cast<uint8_t*>(
            reinterpret_cast<const uint8_t*>(texts.back().data()));
        ba.length = static_cast<int32_t>(texts.back().size());
        barr.push_back(ba);
    }
    check(carquet_writer_write_batch(writer, 1, barr.data(),
                                     static_cast<int64_t>(n_rows), defs.data(),
                                     nullptr),
          "write text");

    if (!include_blob) {
        carquet_status_t status0 = carquet_writer_close(writer);
        carquet_schema_free(schema);
        if (status0 != CARQUET_OK) {
            throw sextant::Error(sextant::ErrorCode::IoError, "writer close failed");
        }
        return path;
    }

    // blob: 8-byte stride, value = row index little-endian; null every 11th.
    // API contract: values packed (non-null only), def_levels mark nulls.
    std::vector<uint8_t> blobs;
    blobs.reserve(static_cast<size_t>(n_rows) * 8);
    std::vector<int16_t> bdefs(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        if (i % 11 == 0) {
            bdefs[i] = 0;
            continue;
        }
        bdefs[i] = 1;
        uint64_t v = i;
        const auto* vb = reinterpret_cast<const uint8_t*>(&v);
        blobs.insert(blobs.end(), vb, vb + 8);
    }
    check(carquet_writer_write_batch(writer, 2, blobs.data(),
                                     static_cast<int64_t>(n_rows), bdefs.data(),
                                     nullptr),
          "write blob");

    carquet_status_t status = carquet_writer_close(writer);
    carquet_schema_free(schema);
    if (status != CARQUET_OK) {
        throw sextant::Error(sextant::ErrorCode::IoError, "writer close failed");
    }
    return path;
}

TEST(ParquetSourceTest, PayloadColumnRoundTrip) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 5000;  // multiple batches at batch_size 1024
    constexpr uint32_t kDim = 8;
    write_payload_parquet(tmp.path(), kN, kDim, /*include_blob=*/false);

    sextant::ParquetSourceConfig cfg;
    cfg.vector_col = "v";
    cfg.payload_col = "text";
    cfg.batch_size = 1024;
    sextant::ParquetSource src(tmp.path(), cfg);
    EXPECT_TRUE(src.schema().has_payload);
    // payload col must not appear as a filter column
    for (const auto& col : src.schema().columns)
        EXPECT_NE(col.name, "text");

    sextant::Chunk chunk;
    uint32_t total = 0;
    while (src.next(chunk)) {
        ASSERT_NE(chunk.payload_data, nullptr);
        ASSERT_NE(chunk.payload_offsets, nullptr);
        for (uint32_t i = 0; i < chunk.count; ++i) {
            const uint32_t row = total + i;
            const uint32_t begin = chunk.payload_offsets[i];
            const uint32_t end = chunk.payload_offsets[i + 1];
            std::string got(reinterpret_cast<const char*>(
                                chunk.payload_data + begin),
                            end - begin);
            std::string expected;
            if (row % 7 != 0) {
                expected = "row-" + std::to_string(row) +
                           std::string(row % 23, 'x');
                if (row % 13 == 0) expected.clear();
            }
            EXPECT_EQ(got, expected) << "row " << row;
        }
        total += chunk.count;
    }
    EXPECT_EQ(total, kN);
}

TEST(ParquetSourceTest, FixedLenPayloadRoundTrip) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 2000;
    constexpr uint32_t kDim = 8;
    write_payload_parquet(tmp.path(), kN, kDim);

    sextant::ParquetSourceConfig cfg;
    cfg.vector_col = "v";
    cfg.payload_col = "blob";
    cfg.batch_size = 777;
    sextant::ParquetSource src(tmp.path(), cfg);
    EXPECT_TRUE(src.schema().has_payload);

    sextant::Chunk chunk;
    uint32_t total = 0;
    while (src.next(chunk)) {
        for (uint32_t i = 0; i < chunk.count; ++i) {
            const uint32_t row = total + i;
            const uint32_t begin = chunk.payload_offsets[i];
            const uint32_t len = chunk.payload_offsets[i + 1] - begin;
            if (row % 11 == 0) {
                EXPECT_EQ(len, 0u) << "row " << row;
            } else {
                ASSERT_EQ(len, 8u) << "row " << row;
                uint64_t v = 0;
                std::memcpy(&v, chunk.payload_data + begin, 8);
                EXPECT_EQ(v, static_cast<uint64_t>(row)) << "row " << row;
            }
        }
        total += chunk.count;
    }
    EXPECT_EQ(total, kN);
}

TEST(ParquetSourceTest, PayloadOffByDefault) {
    TempFile tmp(".parquet");
    write_payload_parquet(tmp.path(), 100, 8, /*include_blob=*/false);
    sextant::ParquetSourceConfig cfg;
    cfg.vector_col = "v";
    sextant::ParquetSource src(tmp.path(), cfg);
    EXPECT_FALSE(src.schema().has_payload);
    sextant::Chunk chunk;
    ASSERT_TRUE(src.next(chunk));
    EXPECT_EQ(chunk.payload_data, nullptr);
    EXPECT_EQ(chunk.payload_offsets, nullptr);
    // text is a plain filter column when not the payload col
    bool found = false;
    for (const auto& col : src.schema().columns)
        if (col.name == "text") found = true;
    EXPECT_TRUE(found);
}

TEST(ParquetSourceTest, PayloadColValidation) {
    TempFile tmp(".parquet");
    write_payload_parquet(tmp.path(), 100, 8);

    sextant::ParquetSourceConfig cfg;
    cfg.vector_col = "v";
    cfg.payload_col = "nope";
    EXPECT_THROW(sextant::ParquetSource(tmp.path(), cfg), sextant::Error);

    // payload col == vector col is rejected
    sextant::ParquetSourceConfig dup = cfg;
    dup.payload_col = "v";
    EXPECT_THROW(sextant::ParquetSource(tmp.path(), dup), sextant::Error);

    // wrong physical type (INT32 filter col as payload) is rejected
    TempFile tmp2(".parquet");
    carquet_error_t err = {};
    carquet_schema_t* schema = carquet_schema_create(&err);
    (void)carquet_schema_add_column(schema, "v",
                                    CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY,
                                    nullptr, CARQUET_REPETITION_REQUIRED, 32, 0);
    (void)carquet_schema_add_column(schema, "n", CARQUET_PHYSICAL_INT32,
                                    nullptr, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_writer_options_t wopts;
    carquet_writer_options_init(&wopts);
    carquet_writer_t* writer =
        carquet_writer_create(tmp2.path().c_str(), schema, &wopts, &err);
    std::vector<float> vecs(100 * 8, 1.f);
    (void)carquet_writer_write_batch(writer, 0, vecs.data(), 100, nullptr,
                                     nullptr);
    std::vector<int32_t> nums(100, 7);
    (void)carquet_writer_write_batch(writer, 1, nums.data(), 100, nullptr,
                                     nullptr);
    (void)carquet_writer_close(writer);
    carquet_schema_free(schema);

    sextant::ParquetSourceConfig bad = cfg;
    bad.payload_col = "n";
    EXPECT_THROW(sextant::ParquetSource(tmp2.path(), bad), sextant::Error);
}

// ---------------------------------------------------------------------------
// Glob source payload forwarding
// ---------------------------------------------------------------------------

TEST(ParquetSourceTest, GlobSourceForwardsPayload) {
    TempFile shard0(".parquet");
    TempFile shard1(".parquet");
    constexpr uint32_t kN = 700;   // spans multiple batches at batch_size 256
    constexpr uint32_t kDim = 8;
    write_payload_parquet(shard0.path(), kN, kDim, /*include_blob=*/false);
    write_payload_parquet(shard1.path(), kN, kDim, /*include_blob=*/false);

    sextant::ParquetGlobSource::Config gcfg;
    gcfg.vector_col = "v";
    gcfg.payload_col = "text";
    gcfg.batch_size = 256;
    sextant::ParquetGlobSource src({shard0.path(), shard1.path()}, gcfg);
    EXPECT_EQ(src.count(), 2 * kN);
    EXPECT_TRUE(src.schema().has_payload);

    sextant::Chunk chunk;
    uint32_t total = 0;
    while (src.next(chunk)) {
        ASSERT_NE(chunk.payload_data, nullptr);
        for (uint32_t i = 0; i < chunk.count; ++i) {
            const uint32_t row = total + i;
            const uint32_t begin = chunk.payload_offsets[i];
            const uint32_t end = chunk.payload_offsets[i + 1];
            std::string got(reinterpret_cast<const char*>(
                                chunk.payload_data + begin),
                            end - begin);
            std::string expected;
            const uint32_t local = row % kN;
            if (local % 7 != 0) {
                expected = "row-" + std::to_string(local) +
                           std::string(local % 23, 'x');
                if (local % 13 == 0) expected.clear();
            }
            EXPECT_EQ(got, expected) << "global row " << row;
        }
        total += chunk.count;
    }
    EXPECT_EQ(total, 2 * kN);
}

TEST(ParquetSourceTest, ReadsListFloatVectors) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 100;
    constexpr uint32_t kDim = 16;
    write_list_float_parquet(tmp.path(), kN, kDim);

    sextant::ParquetSourceConfig cfg;
    cfg.vector_col = "emb";
    sextant::ParquetSource src(tmp.path(), cfg);
    EXPECT_EQ(src.dim(), kDim);
    EXPECT_EQ(src.count(), kN);

    // Read all vectors
    src.reset();
    sextant::Chunk chunk;
    uint32_t total = 0;
    while (src.next(chunk)) {
        for (uint32_t i = 0; i < chunk.count; ++i) {
            for (uint32_t d = 0; d < kDim; ++d) {
                float expected = static_cast<float>(
                    (total + i) * 100 + d);
                EXPECT_FLOAT_EQ(chunk.vectors[i * kDim + d], expected);
            }
        }
        total += chunk.count;
    }
    EXPECT_EQ(total, kN);
}

TEST(ParquetSourceTest, NormalizeFlagProducesUnitVectors) {
    TempFile tmp(".parquet");
    constexpr uint32_t kN = 50;
    constexpr uint32_t kDim = 8;
    write_list_float_parquet(tmp.path(), kN, kDim);

    sextant::ParquetSourceConfig cfg;
    cfg.vector_col = "emb";
    cfg.normalize = true;
    sextant::ParquetSource src(tmp.path(), cfg);

    src.reset();
    sextant::Chunk chunk;
    ASSERT_TRUE(src.next(chunk));
    ASSERT_GT(chunk.count, 0u);

    // Each vector should have norm ≈ 1.0
    for (uint32_t i = 0; i < chunk.count; ++i) {
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < kDim; ++d) {
            float v = chunk.vectors[i * kDim + d];
            norm_sq += v * v;
        }
        EXPECT_NEAR(norm_sq, 1.0f, 1e-4f);
    }
}

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
