#pragma once

/// @file parquet_source.hpp
/// ParquetSource — VectorSource backed by carquet (pure-C Parquet reader).
///
/// Reads vectors + filter columns from a Parquet file. The vector column must
/// be FLOAT (1-dim), DOUBLE (1-dim), or FIXED_LEN_BYTE_ARRAY (N-dim).
/// All other columns become filter columns.
///
/// With use_mmap=true (default) + uncompressed + FIXED_LEN_BYTE_ARRAY vectors,
/// the batch reader returns zero-copy pointers directly into the mmap'd file.

#include <carquet/carquet.h>

#include <sextant/column_data.hpp>
#include <sextant/filter_column_data.hpp>
#include <sextant/schema.hpp>
#include <sextant/types.hpp>
#include <sextant/vector_source.hpp>

#include <memory>
#include <string>
#include <vector>

namespace sextant {

struct ParquetSourceConfig {
    std::string vector_col = "embedding";
    std::string payload_col;             // string/binary col streamed as opaque
                                        // per-row payload blobs ("" = off)
    int32_t batch_size = 8192;
    bool use_mmap = true;
    bool verify_checksums = false;  // off by default — CRC is pure waste for builds
    int32_t num_threads = 0;
    bool normalize = false;         // L2-normalize each vector row (for cosine/IP)
};

/// RAII wrappers for carquet C handles.
struct CarquetReaderDeleter {
    void operator()(carquet_reader_t* r) const { carquet_reader_close(r); }
};
struct CarquetBatchReaderDeleter {
    void operator()(carquet_batch_reader_t* r) const {
        carquet_batch_reader_free(r);
    }
};
struct CarquetBatchDeleter {
    void operator()(carquet_row_batch_t* b) const { carquet_row_batch_free(b); }
};
struct CarquetThreadPoolDeleter {
    void operator()(carquet_thread_pool_t* p) const {
        carquet_thread_pool_destroy(p);
    }
};

using CarquetReaderPtr =
    std::unique_ptr<carquet_reader_t, CarquetReaderDeleter>;
using CarquetBatchReaderPtr =
    std::unique_ptr<carquet_batch_reader_t, CarquetBatchReaderDeleter>;
using CarquetBatchPtr =
    std::unique_ptr<carquet_row_batch_t, CarquetBatchDeleter>;
using CarquetThreadPoolPtr =
    std::unique_ptr<carquet_thread_pool_t, CarquetThreadPoolDeleter>;

[[noreturn]] void throw_carquet_error(const carquet_error_t& err,
                                       const std::string& context);

ColumnType map_carquet_type(carquet_physical_type_t phys,
                            carquet_logical_type_id_t logical, bool is_repeated);

class ParquetSource : public VectorSource {
public:
    ParquetSource(const std::string& path, const ParquetSourceConfig& config = {});
    ~ParquetSource() override = default;

    Dim dim() const override { return dim_; }
    uint64_t count() const override { return count_; }
    std::string path() const override { return path_; }
    Schema schema() const override { return schema_; }

    void reset() override;
    bool next(Chunk& out) override;

private:
    void init_schema_();
    void detect_list_dim_();
    void materialize_batch_(carquet_row_batch_t* batch);

    std::string path_;
    ParquetSourceConfig config_;

    CarquetReaderPtr reader_;
    CarquetThreadPoolPtr thread_pool_;
    CarquetBatchReaderPtr batch_reader_;

    Dim dim_ = 0;
    uint64_t count_ = 0;
    int32_t num_file_cols_ = 0;
    uint64_t cursor_ = 0;
    bool vec_is_list_ = false;  // vector column is list<float> (repeated FLOAT)

    Schema schema_;
    int32_t vec_col_idx_ = -1;
    int32_t vec_type_len_ = 0;
    int32_t payload_col_idx_ = -1;   // file column index of the payload col
    int32_t payload_type_len_ = 0;   // FIXED_LEN_BYTE_ARRAY stride (0 for BYTE_ARRAY)
    std::vector<uint8_t> payload_data_;     // per-batch packed payload bytes
    std::vector<uint32_t> payload_offsets_;  // n+1 cumulative offsets
    std::vector<int32_t> filter_col_indices_;  // file col idx per filter col

    // When the batch reader returns a zero-copy view (mmap + uncompressed +
    // FIXED_LEN_BYTE_ARRAY + no nulls), vec_ptr_ aliases the mmap directly.
    // Otherwise we copy into vec_buf_ and vec_ptr_ aliases it.
    std::vector<float> vec_buf_;
    const float* vec_ptr_ = nullptr;
    std::vector<RowId> rowid_buf_;

    std::vector<std::vector<uint8_t>> fixed_data_;
    std::vector<FilterStringColumn> str_cols_;
    std::vector<std::vector<uint32_t>> str_offsets_;
    std::vector<std::vector<uint16_t>> str_lengths_;
    std::vector<std::vector<char>> str_data_;
    std::vector<FilterSetColumn> set_cols_;
    std::vector<std::vector<uint8_t>> set_counts_;
    std::vector<std::vector<uint32_t>> set_offsets_;
    std::vector<std::vector<uint16_t>> set_elem_lengths_;
    std::vector<std::vector<char>> set_elem_data_;

    std::vector<const void*> chunk_col_ptrs_;
};

}  // namespace sextant
