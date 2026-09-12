#pragma once

/// @file parquet_glob_source.hpp
/// ParquetGlobSource — multi-file parquet VectorSource with optional label join.
///
/// Wraps multiple ParquetSource instances (vector shards) and optionally reads
/// filter columns from a separate parquet file (label file) in lockstep.
///
/// Use case: Cohere 10M ships as 10 train-XX parquet shards (vectors only)
/// plus a separate scalar_labels.parquet (labels). This source chains the
/// shards and joins labels by row position (both are sorted by id == row_id).
///
/// For real-world datasets with embedded payload (vectors + filters in one
/// file), use ParquetSource directly — it auto-discovers filter columns.

#include "parquet_source.hpp"
#include <sextant/vector_source.hpp>
#include <memory>
#include <string>
#include <vector>

namespace sextant {

class ParquetGlobSource : public VectorSource {
public:
    struct Config {
        std::string vector_col = "emb";
        std::string payload_col;             // optional payload column (per shard)
        std::string label_file;               // optional: parquet with filter columns
        std::vector<std::string> label_cols;  // which columns from label_file (empty = all non-id)
        bool normalize = false;
        int32_t batch_size = 8192;
        int32_t num_threads = 0;
        bool use_mmap = true;
        bool vectors_only = false;  // project only vector col (filter data from sidecar)
    };

    ParquetGlobSource(const std::vector<std::string>& shard_paths,
                      const Config& config);
    ~ParquetGlobSource() override = default;

    Dim dim() const override;
    uint64_t count() const override;
    Schema schema() const override;
    void reset() override;
    bool next(Chunk& out) override;

private:
    void ensure_label_rows_(uint32_t need);
    void consume_labels_(uint32_t n, Chunk& out);

    Config config_;
    std::vector<std::unique_ptr<ParquetSource>> shards_;
    size_t cur_shard_ = 0;

    Dim dim_ = 0;
    uint64_t total_count_ = 0;
    Schema schema_;  // merged schema (label columns only, or shard filter cols + label cols)

    // Label reader (optional)
    std::unique_ptr<ParquetSource> label_reader_;
    uint32_t n_label_cols_ = 0;

    // Label buffer: accumulates label data across batch boundaries.
    // For each label column, we maintain accumulated string data.
    struct LabelStringBuf {
        std::vector<uint32_t> offsets;  // n+1 entries
        std::vector<uint16_t> lengths;  // n entries
        std::vector<char> data;
        size_t head = 0;  // rows consumed

        size_t available() const {
            return offsets.empty() ? 0 : offsets.size() - 1 - head;
        }

        // Append rows from a FilterStringColumn
        void append(const FilterStringColumn& col, uint32_t count);

        // Consume n rows, write into output FilterStringColumn
        void consume(uint32_t n, FilterStringColumn& out);

        void compact();

        void clear() {
            offsets.clear();
            lengths.clear();
            data.clear();
            head = 0;
        }
    };
    std::vector<LabelStringBuf> label_bufs_;

    // Output filter column pointers (reused per next() call)
    std::vector<const void*> chunk_filter_ptrs_;
    std::vector<FilterStringColumn> label_out_cols_;
};

}  // namespace sextant
