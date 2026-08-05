// fbin_to_parquet — convert a .fbin vector file to Parquet with synthetic
// metadata columns, for benchmarking the ParquetSource build path.
//
// Usage:
//   fbin_to_parquet <input.fbin> <output.parquet> [--vector-col NAME]
//                   [--batch-size N] [--no-metadata]
//
// Output columns:
//   embedding  : FIXED_LEN_BYTE_ARRAY(dim × 4)   — the vectors
//   category   : BYTE_ARRAY (string)             — synthetic, 16 distinct values
//   year       : INT32                            — synthetic, 2015..2024
//   score      : FLOAT                            — synthetic, 0..1
//
// When --no-metadata is given, only the embedding column is written.

#include <carquet/carquet.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "fbin_to_parquet: %s\n", msg.c_str());
    std::exit(1);
}

void check(carquet_status_t s, const char* what) {
    if (s != CARQUET_OK) {
        die(std::string(what) + ": " + carquet_status_string(s));
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <input.fbin> <output.parquet> [--vector-col NAME] "
            "[--batch-size N] [--no-metadata]\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    const char* vector_col = "embedding";
    uint32_t batch_size = 50'000;
    bool with_metadata = true;
    bool uncompressed = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vector-col" && i + 1 < argc) {
            vector_col = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--no-metadata") {
            with_metadata = false;
        } else if (arg == "--uncompressed") {
            uncompressed = true;
        } else {
            die("unknown argument: " + arg);
        }
    }

    // --- Read .fbin header ---
    FILE* f = std::fopen(input_path, "rb");
    if (!f) die(std::string("cannot open ") + input_path);

    uint32_t header[2];
    if (std::fread(header, sizeof(uint32_t), 2, f) != 2)
        die("short header read");
    const uint32_t n = header[0];
    const uint32_t dim = header[1];
    if (n == 0 || dim == 0) die("zero n or dim");

    std::fprintf(stderr, "fbin_to_parquet: %s → %s  n=%u dim=%u  metadata=%d\n",
                 input_path, output_path, n, dim, with_metadata);

    // --- Create schema ---
    carquet_error_t err = {};
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) die("schema create");

    int32_t col = 0;
    check(carquet_schema_add_column(
              schema, vector_col, CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY,
              nullptr, CARQUET_REPETITION_REQUIRED,
              static_cast<int32_t>(dim * sizeof(float)), 0),
          "add vector column");
    const int32_t vec_col_idx = col++;

    int32_t cat_col = -1, year_col = -1, score_col = -1;
    if (with_metadata) {
        check(carquet_schema_add_column(schema, "category",
                                        CARQUET_PHYSICAL_BYTE_ARRAY, nullptr,
                                        CARQUET_REPETITION_REQUIRED, 0, 0),
              "add category");
        cat_col = col++;

        check(carquet_schema_add_column(schema, "year",
                                        CARQUET_PHYSICAL_INT32, nullptr,
                                        CARQUET_REPETITION_REQUIRED, 0, 0),
              "add year");
        year_col = col++;

        check(carquet_schema_add_column(schema, "score",
                                        CARQUET_PHYSICAL_FLOAT, nullptr,
                                        CARQUET_REPETITION_REQUIRED, 0, 0),
              "add score");
        score_col = col++;
    }

    // --- Create writer ---
    carquet_writer_options_t wopts;
    carquet_writer_options_init(&wopts);
    wopts.compression = uncompressed ? CARQUET_COMPRESSION_UNCOMPRESSED
                                      : CARQUET_COMPRESSION_ZSTD;
    wopts.compression_level = uncompressed ? 0 : 3;
    // Large row groups for better scan throughput.
    wopts.row_group_size = static_cast<int64_t>(n) * dim * sizeof(float) / 2;

    carquet_writer_t* writer =
        carquet_writer_create(output_path, schema, &wopts, &err);
    if (!writer) {
        carquet_schema_free(schema);
        die("writer create");
    }

    // --- Convert in batches ---
    std::vector<float> vec_buf(static_cast<size_t>(batch_size) * dim);

    // Metadata generation state.
    const char* categories[16] = {
        "cs.AI", "cs.CL", "cs.CV", "cs.DB", "cs.DC", "cs.DS", "cs.GL",
        "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
        "cs.MA", "cs.NI"};

    std::vector<carquet_byte_array_t> cat_buf;
    std::vector<std::string> cat_storage;
    std::vector<int32_t> year_buf;
    std::vector<float> score_buf;

    uint32_t written = 0;
    while (written < n) {
        const uint32_t take = std::min(batch_size, n - written);
        const size_t want = static_cast<size_t>(take) * dim;
        size_t got = 0;
        while (got < want) {
            size_t r = std::fread(vec_buf.data() + got, sizeof(float),
                                  want - got, f);
            if (r == 0) die("unexpected EOF");
            got += r;
        }

        // Write vector column.
        check(carquet_writer_write_batch(writer, vec_col_idx, vec_buf.data(),
                                         static_cast<int64_t>(take), nullptr,
                                         nullptr),
              "write vectors");

        if (with_metadata) {
            // Generate metadata for this batch.
            cat_storage.resize(take);
            cat_buf.resize(take);
            year_buf.resize(take);
            score_buf.resize(take);
            for (uint32_t i = 0; i < take; ++i) {
                const uint32_t row = written + i;
                cat_storage[i] = categories[row % 16];
                cat_buf[i].data = reinterpret_cast<uint8_t*>(
                    const_cast<char*>(cat_storage[i].data()));
                cat_buf[i].length = static_cast<int32_t>(cat_storage[i].size());
                year_buf[i] = 2015 + static_cast<int32_t>(row % 10);
                // Deterministic pseudo-random score in [0, 1).
                uint32_t hash = row * 2654435761u;
                score_buf[i] = static_cast<float>((hash >> 8) % 10000) / 10000.0f;
            }

            check(carquet_writer_write_batch(writer, cat_col, cat_buf.data(),
                                             static_cast<int64_t>(take), nullptr,
                                             nullptr),
                  "write category");
            check(carquet_writer_write_batch(writer, year_col, year_buf.data(),
                                             static_cast<int64_t>(take), nullptr,
                                             nullptr),
                  "write year");
            check(carquet_writer_write_batch(writer, score_col, score_buf.data(),
                                             static_cast<int64_t>(take), nullptr,
                                             nullptr),
                  "write score");
        }

        written += take;
        if (written % 100'000 < batch_size) {
            std::fprintf(stderr, "  %u/%u rows...\n", written, n);
        }
    }

    std::fclose(f);
    check(carquet_writer_close(writer), "writer close");
    carquet_schema_free(schema);

    std::fprintf(stderr, "fbin_to_parquet: done (%u vectors)\n", written);
    return 0;
}
