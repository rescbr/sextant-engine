#include "parquet_glob_source.hpp"

#include <sextant/error.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace sextant {

// ---------------------------------------------------------------------------
// LabelStringBuf
// ---------------------------------------------------------------------------

void ParquetGlobSource::LabelStringBuf::append(
    const FilterStringColumn& col, uint32_t count) {
    if (offsets.empty()) {
        offsets.push_back(0);
    }
    // The last offset is the running total
    uint32_t base = offsets.back();
    for (uint32_t i = 0; i < count; ++i) {
        uint16_t len = col.lengths[i];
        offsets.push_back(base + len);
        lengths.push_back(len);
        data.insert(data.end(), col.data + col.offsets[i],
                    col.data + col.offsets[i] + len);
        base += len;
    }
}

void ParquetGlobSource::LabelStringBuf::consume(
    uint32_t n, FilterStringColumn& out) {
    // Point output to our internal buffer starting at head
    out.offsets = offsets.data() + head;
    out.lengths = lengths.data() + head;
    out.data = data.data();
    // total_data_bytes is the full buffer (consumer indexes via offsets)
    out.total_data_bytes = static_cast<uint32_t>(data.size());
    head += n;
}

void ParquetGlobSource::LabelStringBuf::compact() {
    if (head == 0) return;
    // Remove consumed rows from the front
    uint32_t consumed = static_cast<uint32_t>(head);
    if (consumed >= offsets.size() - 1) {
        clear();
        offsets.push_back(0);
        return;
    }
    // Shift offsets, lengths, and data
    uint32_t data_offset = offsets[consumed];
    uint32_t remaining = static_cast<uint32_t>(offsets.size() - 1 - consumed);

    std::vector<uint32_t> new_offsets;
    std::vector<uint16_t> new_lengths;
    new_offsets.reserve(remaining + 1);
    new_lengths.reserve(remaining);

    for (uint32_t i = 0; i <= remaining; ++i) {
        new_offsets.push_back(offsets[consumed + i] - data_offset);
    }
    new_lengths.assign(lengths.begin() + consumed, lengths.end());

    // Shift data
    data.erase(data.begin(), data.begin() + data_offset);

    offsets = std::move(new_offsets);
    lengths = std::move(new_lengths);
    head = 0;
}

// ---------------------------------------------------------------------------
// ParquetGlobSource
// ---------------------------------------------------------------------------

ParquetGlobSource::ParquetGlobSource(
    const std::vector<std::string>& shard_paths, const Config& config)
    : config_(config) {

    if (shard_paths.empty()) {
        throw Error(ErrorCode::InvalidParam,
                    "ParquetGlobSource: no shard paths provided");
    }

    // Create vector shard sources
    ParquetSourceConfig pcfg;
    pcfg.vector_col = config_.vector_col;
    pcfg.normalize = config_.normalize;
    pcfg.batch_size = config_.batch_size;
    pcfg.num_threads = config_.num_threads;
    pcfg.use_mmap = config_.use_mmap;

    for (const auto& path : shard_paths) {
        auto src = std::make_unique<ParquetSource>(path, pcfg);
        if (shards_.empty()) {
            dim_ = src->dim();
        } else {
            if (src->dim() != dim_) {
                throw Error(ErrorCode::InvalidParam,
                            "ParquetGlobSource: dimension mismatch across shards");
            }
        }
        total_count_ += src->count();
        shards_.push_back(std::move(src));
    }

    // Set up label reader if specified
    if (!config_.label_file.empty()) {
        // Open the label file. Use a config that reads all columns
        // (auto-discover). The vector_col won't match, so all columns
        // become filter columns.
        ParquetSourceConfig lcfg;
        lcfg.vector_col = "__none__";  // no vector column — all cols are filters
        lcfg.batch_size = config_.batch_size;
        lcfg.num_threads = config_.num_threads;
        lcfg.use_mmap = config_.use_mmap;

        // We need to trick ParquetSource into treating all columns as filters.
        // Instead, open with a dummy vector_col that doesn't exist, catch the
        // error, and handle manually. Actually, ParquetSource requires a valid
        // vector column. Let's open with the first column as "vector" (which
        // we'll ignore) and use the rest as labels.
        //
        // Better: just open the label file normally and read filter columns.
        // We pick "id" as the vector col (it exists, is int64=1-dim) so the
        // source opens successfully, and the remaining columns are filters.
        lcfg.vector_col = "id";
        label_reader_ = std::make_unique<ParquetSource>(config_.label_file, lcfg);

        auto lschema = label_reader_->schema();
        n_label_cols_ = static_cast<uint32_t>(lschema.columns.size());

        // Filter out unwanted label columns
        if (!config_.label_cols.empty()) {
            Schema filtered;
            for (const auto& col : lschema.columns) {
                for (const auto& want : config_.label_cols) {
                    if (col.name == want) {
                        filtered.columns.push_back(col);
                        break;
                    }
                }
            }
            lschema = filtered;
            n_label_cols_ = static_cast<uint32_t>(lschema.columns.size());
        }

        // Merge label columns into our schema
        for (auto& col : lschema.columns) {
            schema_.columns.push_back(std::move(col));
        }

        label_bufs_.resize(n_label_cols_);
        label_out_cols_.resize(n_label_cols_);
    }

    // Also include filter columns from the vector shards (if any)
    if (n_label_cols_ == 0) {
        // No label file — use shard schema
        schema_ = shards_[0]->schema();
    }

    chunk_filter_ptrs_.resize(schema_.columns.size(), nullptr);

    spdlog::debug("ParquetGlobSource: {} shards, N={}, dim={}, label_cols={}",
                  shards_.size(), total_count_, dim_, n_label_cols_);
}

Dim ParquetGlobSource::dim() const { return dim_; }

uint64_t ParquetGlobSource::count() const { return total_count_; }

Schema ParquetGlobSource::schema() const { return schema_; }

void ParquetGlobSource::reset() {
    for (auto& s : shards_) s->reset();
    cur_shard_ = 0;
    if (label_reader_) {
        label_reader_->reset();
        for (auto& buf : label_bufs_) buf.clear();
    }
}

void ParquetGlobSource::ensure_label_rows_(uint32_t need) {
    if (!label_reader_) return;

    for (uint32_t li = 0; li < n_label_cols_; ++li) {
        // Compact previously consumed rows before appending new data
        label_bufs_[li].compact();
        while (label_bufs_[li].available() < need) {
            Chunk lc;
            if (!label_reader_->next(lc)) {
                throw Error(ErrorCode::InvalidParam,
                            "ParquetGlobSource: label file exhausted before vectors");
            }
            // Append label rows for each label column
            for (uint32_t ci = 0; ci < n_label_cols_; ++ci) {
                const auto* fcol = static_cast<const FilterStringColumn*>(
                    lc.filter_columns[ci]);
                label_bufs_[ci].append(*fcol, lc.count);
            }
        }
    }
}

void ParquetGlobSource::consume_labels_(uint32_t n, Chunk& out) {
    for (uint32_t li = 0; li < n_label_cols_; ++li) {
        label_bufs_[li].consume(n, label_out_cols_[li]);
        chunk_filter_ptrs_[li] = &label_out_cols_[li];
        // Don't compact here — the builder needs the data until the next next()
    }
    out.filter_columns = chunk_filter_ptrs_.data();
}

bool ParquetGlobSource::next(Chunk& out) {
    // Advance to next shard if current is exhausted
    while (cur_shard_ < shards_.size()) {
        if (shards_[cur_shard_]->next(out)) {
            // When vectors_only is set (filter data from sidecar), suppress
            // shard-local filter columns so the builder uses cfg.filter_column_data.
            if (config_.vectors_only) {
                out.filter_columns = nullptr;
            }
            // Got a vector batch — ensure matching labels
            if (label_reader_) {
                ensure_label_rows_(out.count);
                consume_labels_(out.count, out);
            }
            return true;
        }
        ++cur_shard_;
    }
    out.count = 0;
    return false;
}

}  // namespace sextant
