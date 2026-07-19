// extract_duckdb_to_fbin — stream a DuckDB vector table to .fbin.
// Uses DuckDB's streaming Fetch() to write one chunk at a time (constant RAM).
//
// Build:
//   c++ -std=c++17 -O2 -o /tmp/extract_duckdb_to_fbin tools/extract_duckdb_to_fbin.cpp \
//       -I/opt/homebrew/include -L/opt/homebrew/lib -lduckdb -lpthread -ldl
//
// Usage:
//   extract_duckdb_to_fbin <input.duckdb> <output.fbin> [table] [vec_col] [where]

#include <duckdb.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <input.duckdb> <output.fbin> [table] [vec_col] [where]\n",
                     argv[0]);
        return 1;
    }
    const char* input = argv[1];
    const char* output = argv[2];
    const std::string table = (argc > 3) ? argv[3] : "base";
    const std::string vec_col = (argc > 4) ? argv[4] : "vec";
    const std::string where_sql = (argc > 5) ? ("WHERE " + std::string(argv[5])) : "";

    duckdb::DBConfig config;
    config.options.access_mode = duckdb::AccessMode::READ_ONLY;
    // Cap DuckDB's internal buffer pool so it doesn't pre-allocate 25% of RAM.
    // 256MB is enough for streaming chunks; the data goes to disk, not RAM.
    config.options.maximum_memory = 256 * 1024 * 1024;
    duckdb::DuckDB db(input, &config);
    duckdb::Connection con(db);

    // Count.
    auto res = con.Query("SELECT COUNT(*) FROM " + table + " " + where_sql);
    if (res->HasError()) { std::fprintf(stderr, "COUNT: %s\n", res->GetError().c_str()); return 1; }
    const uint64_t n = res->GetValue<uint64_t>(0, 0);
    if (n == 0) { std::fprintf(stderr, "0 rows\n"); return 1; }

    // Dim from the column type (FLOAT[dim]).
    auto sample_stream = con.SendQuery(
        "SELECT " + vec_col + " FROM " + table + " " + where_sql + " LIMIT 1");
    auto sample_chunk = sample_stream->Fetch();
    if (sample_chunk->size() == 0) { std::fprintf(stderr, "sample fail\n"); return 1; }
    const auto& col_type = sample_chunk->data[0].GetType();
    if (col_type.id() != duckdb::LogicalTypeId::ARRAY) {
        std::fprintf(stderr, "vec column is %s, expected ARRAY (FLOAT[dim])\n",
                     col_type.ToString().c_str());
        return 1;
    }
    const uint32_t dim = duckdb::ArrayType::GetSize(col_type);

    std::printf("Streaming %llu vectors (dim=%u) → %s\n",
                (unsigned long long)n, dim, output);

    // Stream.
    auto stream = con.SendQuery("SELECT " + vec_col + " FROM " + table + " " + where_sql);
    if (stream->HasError()) { std::fprintf(stderr, "%s\n", stream->GetError().c_str()); return 1; }

    std::FILE* f = std::fopen(output, "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", output); return 1; }
    uint32_t header[2] = {static_cast<uint32_t>(n), dim};
    std::fwrite(header, sizeof(header), 1, f);

    uint64_t written = 0;
    while (auto chunk = stream->Fetch()) {
        if (chunk->size() == 0) break;
        auto& vec = chunk->data[0];
        auto& child = duckdb::ArrayVector::GetEntry(vec);
        const float* data = duckdb::FlatVector::GetData<float>(child);
        const size_t count = static_cast<size_t>(chunk->size()) * dim;
        std::fwrite(data, sizeof(float), count, f);
        written += chunk->size();
        std::printf("  %llu/%llu (%.1f%%)\r", (unsigned long long)written,
                    (unsigned long long)n, written * 100.0 / n);
        std::fflush(stdout);
    }

    std::printf("\nDone: %llu vectors\n", (unsigned long long)written);
    std::fclose(f);
    return 0;
}
