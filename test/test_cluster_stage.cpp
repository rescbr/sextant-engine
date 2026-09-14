#include <gtest/gtest.h>

#include "tree/cluster_stage.hpp"

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace sextant;
using namespace sextant::tree;

// Serialize one staged record with the 4-byte rec_len prefix.
std::vector<uint8_t> make_record(RowId rid, const std::vector<uint8_t>& code,
                                 const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> body;
    auto push = [&](const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        body.insert(body.end(), b, b + n);
    };
    uint16_t flags = kStagedHasCode;
    if (!payload.empty()) flags |= kStagedHasPayload;
    uint64_t rid64 = static_cast<uint64_t>(rid);
    uint32_t clen = static_cast<uint32_t>(code.size());
    push(&rid64, 8); push(&flags, 2);
    push(&clen, 4); push(code.data(), clen);
    push(payload.data(), payload.size());
    std::vector<uint8_t> rec(4 + body.size());
    uint32_t len = static_cast<uint32_t>(body.size());
    std::memcpy(rec.data(), &len, 4);
    std::memcpy(rec.data() + 4, body.data(), body.size());
    return rec;
}

std::vector<uint8_t> make_filter_row(const Schema& schema,
                                     const std::vector<ColumnData>& row_vals,
                                     const std::vector<uint32_t>& str_lens,
                                     const std::vector<uint8_t>& set_counts) {
    // Build a serialized filter row from per-column raw values.
    // fixed_data holds exactly one row; String column uses str_data with
    // str_lens[c] bytes; Set column uses set_counts[c] copies of one element.
    std::vector<uint8_t> fr;
    auto put = [&](const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        fr.insert(fr.end(), b, b + n);
    };
    for (uint32_t c = 0; c < schema.n_filter_columns(); ++c) {
        const auto& v = row_vals[c];
        switch (schema.columns[c].type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool:
                put(v.fixed_data.data(), column_type_width(v.type));
                break;
            case ColumnType::String: {
                uint16_t len = static_cast<uint16_t>(str_lens[c]);
                put(&len, 2);
                put(v.str_data.data(), len);
                break;
            }
            case ColumnType::Set: {
                uint8_t ec = set_counts[c];
                put(&ec, 1);
                for (uint32_t e = 0; e < ec; ++e) {
                    uint16_t elen = static_cast<uint16_t>(v.set_elem_data.size());
                    put(&elen, 2);
                    put(v.set_elem_data.data(), elen);
                }
                break;
            }
        }
    }
    return fr;
}

TEST(ClusterStage, RoundTripRecords) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "cx_stage_test.stage").string();
    ClusterStage stage(path, 3);
    EXPECT_EQ(stage.count(0), 0u);

    std::vector<std::vector<uint8_t>> expected[3];
    for (uint32_t c = 0; c < 3; ++c) {
        for (int i = 0; i < 50; ++i) {
            std::vector<uint8_t> code(384);
            for (auto& b : code) b = static_cast<uint8_t>(i + c);
            std::vector<uint8_t> payload(static_cast<size_t>(i) * 997 + 1, 0xAB);
            auto rec = make_record(1000 + i * 3 + c, code, payload);
            expected[c].push_back(rec);
            stage.append(c, rec.data() + 4,
                         static_cast<uint32_t>(rec.size() - 4));
        }
    }
    for (uint32_t c = 0; c < 3; ++c) {
        EXPECT_EQ(stage.count(c), 50u);
        uint32_t i = 0;
        stage.for_each(c, [&](const uint8_t* rec, uint32_t len) {
            ASSERT_EQ(len, expected[c][i].size() - 4);
            EXPECT_EQ(0, std::memcmp(rec, expected[c][i].data() + 4, len));
            ++i;
        });
        EXPECT_EQ(i, 50u);
    }
    stage.drop(1);
    EXPECT_EQ(stage.count(1), 0u);
    stage.for_each(1, [](const uint8_t*, uint32_t) { FAIL(); });
    stage.finish();
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(ClusterStage, ParseAndFilterRow) {
    Schema schema;
    schema.columns = {
        {"i32", ColumnType::Int32},
        {"i64", ColumnType::Int64},
        {"f", ColumnType::Float},
        {"b", ColumnType::Bool},
        {"s", ColumnType::String},
        {"set", ColumnType::Set},
    };
    // Serialize values.
    std::vector<ColumnData> vals(6);
    for (uint32_t c = 0; c < 6; ++c) vals[c].type = schema.columns[c].type;
    int32_t i32 = -42; int64_t i64 = 1'000'000'000'000LL;
    float f = 3.5f; uint8_t b = 1;
    vals[0].fixed_data.resize(4); std::memcpy(vals[0].fixed_data.data(), &i32, 4);
    vals[1].fixed_data.resize(8); std::memcpy(vals[1].fixed_data.data(), &i64, 8);
    vals[2].fixed_data.resize(4); std::memcpy(vals[2].fixed_data.data(), &f, 4);
    vals[3].fixed_data.resize(1); vals[3].fixed_data[0] = b;
    vals[4].str_data.assign({'h', 'e', 'l', 'l', 'o'});
    vals[5].set_elem_data.assign({'x', 'y'});

    auto fr = make_filter_row(schema, vals, {0, 0, 0, 0, 5}, {0, 0, 0, 0, 0, 2});
    auto rec = make_record(7, {}, {});
    // Embed the filter row into a full record: rebuild with flags.
    {
        std::vector<uint8_t> body;
        auto push = [&](const void* p, size_t n) {
            const auto* q = static_cast<const uint8_t*>(p);
            body.insert(body.end(), q, q + n);
        };
        uint64_t rid = 7; uint16_t flags = kStagedHasFilter;
        uint32_t flen = static_cast<uint32_t>(fr.size());
        push(&rid, 8); push(&flags, 2); push(&flen, 4); push(fr.data(), flen);
        rec.assign(4 + body.size(), 0);
        uint32_t len = static_cast<uint32_t>(body.size());
        std::memcpy(rec.data(), &len, 4);
        std::memcpy(rec.data() + 4, body.data(), body.size());
    }

    StagedRow row;
    std::string err;
    ASSERT_TRUE(staged_parse(rec.data() + 4,
                             static_cast<uint32_t>(rec.size() - 4), row, err))
        << err;
    EXPECT_EQ(row.row_id, 7);
    ASSERT_NE(row.filter_row, nullptr);

    std::vector<ColumnData> dst(6);
    for (uint32_t c = 0; c < 6; ++c) dst[c].type = schema.columns[c].type;
    ASSERT_TRUE(staged_append_filter_row(row.filter_row, row.filter_row_len,
                                         schema, dst, err))
        << err;
    EXPECT_EQ(dst[0].fixed_data.size(), 4u);
    int32_t out32; std::memcpy(&out32, dst[0].fixed_data.data(), 4);
    EXPECT_EQ(out32, -42);
    int64_t out64; std::memcpy(&out64, dst[1].fixed_data.data(), 8);
    EXPECT_EQ(out64, 1'000'000'000'000LL);
    EXPECT_EQ(dst[3].fixed_data[0], 1);
    ASSERT_EQ(dst[4].str_lengths.size(), 1u);
    EXPECT_EQ(dst[4].str_lengths[0], 5u);
    EXPECT_EQ(std::string(dst[4].str_data.data(), 5), "hello");
    ASSERT_EQ(dst[5].set_counts.size(), 1u);
    EXPECT_EQ(dst[5].set_counts[0], 2);
    ASSERT_EQ(dst[5].set_elem_lengths.size(), 2u);
    EXPECT_EQ(dst[5].set_elem_lengths[0], 2u);
    EXPECT_EQ(std::string(dst[5].set_elem_data.data(), 2), "xy");

    // Malformed: truncated record.
    StagedRow bad;
    EXPECT_FALSE(staged_parse(rec.data() + 4, 3, bad, err));
}

// Tiny budget forcing spills: clusters must round-trip identically whether
// their records live in the RAM arena, got spilled to disk mid-stream, or
// were appended to disk after a spill. Also checks drop() across tiers.
TEST(ClusterStage, RamDiskHybridSpill) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "cx_stage_spill.stage")
            .string();
    constexpr uint32_t kClusters = 4;
    constexpr uint64_t kBudget = 64u << 10;  // 64 KiB -> forces spill
    ClusterStage stage(path, kClusters, kBudget);

    std::vector<std::vector<std::vector<uint8_t>>> expected(kClusters);
    // Append rounds cluster-major so arenas grow evenly; the budget is hit
    // quickly and victims (largest arenas) get spilled.
    for (int round = 0; round < 40; ++round) {
        for (uint32_t c = 0; c < kClusters; ++c) {
            std::vector<uint8_t> code(256);
            for (auto& b : code) b = static_cast<uint8_t>(round * 7 + c);
            std::vector<uint8_t> payload(600, static_cast<uint8_t>(round));
            auto rec = make_record(9000 + round * 11 + c, code, payload);
            expected[c].push_back(rec);
            stage.append(c, rec.data() + 4,
                         static_cast<uint32_t>(rec.size() - 4));
        }
    }
    for (uint32_t c = 0; c < kClusters; ++c) {
        ASSERT_EQ(stage.count(c), 40u);
        uint32_t i = 0;
        stage.for_each(c, [&](const uint8_t* rec, uint32_t len) {
            ASSERT_EQ(len, expected[c][i].size() - 4);
            EXPECT_EQ(0, std::memcmp(rec, expected[c][i].data() + 4, len));
            ++i;
        });
        EXPECT_EQ(i, 40u);
    }
    // drop() on both tiers, then continue appending: the dropped clusters
    // restart from zero records, order preserved.
    stage.drop(0);
    stage.drop(1);
    EXPECT_EQ(stage.count(0), 0u);
    EXPECT_EQ(stage.count(1), 0u);
    stage.for_each(0, [](const uint8_t*, uint32_t) { FAIL(); });
    for (uint32_t c : {0u, 1u}) {
        std::vector<uint8_t> code(128, static_cast<uint8_t>(0xC0 + c));
        auto rec = make_record(70'000 + c, code, {});
        expected[c].assign(1, rec);
        stage.append(c, rec.data() + 4,
                     static_cast<uint32_t>(rec.size() - 4));
        uint32_t i = 0;
        stage.for_each(c, [&](const uint8_t* rec, uint32_t len) {
            ASSERT_EQ(len, expected[c][i].size() - 4);
            EXPECT_EQ(0, std::memcmp(rec, expected[c][i].data() + 4, len));
            ++i;
        });
        EXPECT_EQ(i, 1u);
    }
    stage.finish();
    EXPECT_FALSE(std::filesystem::exists(path));
}

// Zero budget = pure disk staging (the pre-hybrid behavior).
TEST(ClusterStage, ZeroBudgetPureDisk) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "cx_stage_disk.stage")
            .string();
    ClusterStage stage(path, 2, 0);
    std::vector<uint8_t> rec = make_record(5, {1, 2, 3}, {4, 5});
    stage.append(0, rec.data() + 4, static_cast<uint32_t>(rec.size() - 4));
    EXPECT_EQ(stage.count(0), 1u);
    stage.for_each(0, [&](const uint8_t* r, uint32_t len) {
        ASSERT_EQ(len, rec.size() - 4);
        EXPECT_EQ(0, std::memcmp(r, rec.data() + 4, len));
    });
    stage.finish();
    EXPECT_FALSE(std::filesystem::exists(path));
}

}  // namespace
