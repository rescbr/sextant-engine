#include "parquet_glob_source.hpp"

#include <sextant/error.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <future>

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

    // Probe each shard with a TEMPORARY ParquetSource (footer + schema work
    // only, no page reads) to validate dim/count, capture the merged schema
    // and the payload byte estimate — then CLOSE it. Keeping all shard
    // readers open pins one mmap per shard for the whole build: measured
    // 9.6 GB of resident page cache on the 60-shard CulturaX corpus.
    // Readers are opened lazily, one at a time, in next().
    ParquetSourceConfig pcfg;
    pcfg.vector_col = config_.vector_col;
    pcfg.payload_col = config_.payload_col;
    pcfg.normalize = config_.normalize;
    pcfg.batch_size = config_.batch_size;
    pcfg.num_threads = config_.num_threads;
    pcfg.use_mmap = config_.use_mmap;
    pcfg.vector_fp16 = config_.vector_fp16;

    // Probes are independent files — run them concurrently (serial probing
    // cost ~1.5-2 s per shard x 59 shards on CulturaX).
    struct ProbeResult {
        uint64_t count = 0;
        Dim dim = 0;
        uint64_t payload_bytes = 0;
        std::string error;
    };
    std::vector<std::future<ProbeResult>> futures;
    futures.reserve(shard_paths.size());
    for (const auto& path : shard_paths) {
        futures.push_back(std::async(std::launch::async, [&pcfg, &path]() {
            ProbeResult r;
            try {
                ParquetSource probe(path, pcfg);
                r.count = probe.count();
                r.dim = probe.dim();
                r.payload_bytes = probe.payload_total_bytes();
            } catch (const std::exception& e) {
                r.error = e.what();
            }
            return r;
        }));
    }
    const Schema first_schema = [&]() {
        ParquetSource probe(shard_paths.front(), pcfg);
        return probe.schema();
    }();
    shard_paths_ = shard_paths;
    for (size_t i = 0; i < futures.size(); ++i) {
        ProbeResult r = futures[i].get();
        if (!r.error.empty()) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetGlobSource: " + r.error);
        }
        if (i == 0) {
            dim_ = r.dim;
            schema_ = first_schema;
        } else if (r.dim != dim_) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetGlobSource: dimension mismatch across shards");
        }
        total_count_ += r.count;
        shard_base_.push_back(total_count_ - r.count);
        payload_total_bytes_ += r.payload_bytes;
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
        // No label file — use shard schema (captured by the constructor's
        // first probe).
    }

    chunk_filter_ptrs_.resize(schema_.columns.size(), nullptr);

    spdlog::debug("ParquetGlobSource: {} shards, N={}, dim={}, label_cols={}",
                  shard_paths_.size(), total_count_, dim_, n_label_cols_);
}

Dim ParquetGlobSource::dim() const { return dim_; }

uint64_t ParquetGlobSource::count() const { return total_count_; }

Schema ParquetGlobSource::schema() const { return schema_; }

uint64_t ParquetGlobSource::payload_total_bytes() const {
    // Sum of per-shard column-chunk metadata estimates, cached at
    // construction (each probe read footers only, no data pages).
    return payload_total_bytes_;
}

void ParquetGlobSource::reset() {
    // Close the open shard reader (drops its mmap) — reopened lazily.
    cur_src_.reset();
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
    // Advance to next shard if current is exhausted. One reader is open at
    // a time (lazily constructed); advancing closes the previous shard's
    // reader and its mmap.
    while (cur_shard_ < shard_paths_.size()) {
        if (!cur_src_) {
            ParquetSourceConfig pcfg;
            pcfg.vector_col = config_.vector_col;
            pcfg.payload_col = config_.payload_col;
            pcfg.normalize = config_.normalize;
            pcfg.batch_size = config_.batch_size;
            pcfg.num_threads = config_.num_threads;
            pcfg.use_mmap = config_.use_mmap;
    pcfg.vector_fp16 = config_.vector_fp16;
            cur_src_ = std::make_unique<ParquetSource>(
                shard_paths_[cur_shard_], pcfg);
        }
        if (cur_src_->next(out)) {
            // ParquetSource assigns row ids from a per-file cursor, so each
            // shard emits ids 0..n_shard-1. Re-base them to GLOBAL ids here:
            // the build treats source-assigned ids as corpus-wide row ids
            // (row ids index filter/payload sidecars and ground-truth
            // comparisons). Without this, every shard's rows collide into
            // the first shard's id space — searches return vectors with
            // correct codes but wrong (shard-local) row ids. The offset is
            // the shard's BASE (rows in all previous shards), a constant:
            // adding the running rows-served total would double-count the
            // current shard's own already-emitted prefix.
            if (out.row_ids) {
                rid_buf_.resize(out.count);
                const uint64_t base = shard_base_[cur_shard_];
                for (uint32_t i = 0; i < out.count; ++i)
                    rid_buf_[i] = static_cast<RowId>(
                        static_cast<uint64_t>(out.row_ids[i]) + base);
                out.row_ids = rid_buf_.data();
            }
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
        // Shard exhausted: close its reader (drops the mmap) and advance.
        cur_src_.reset();
        ++cur_shard_;
    }
    out.count = 0;
    return false;
}

bool ParquetGlobSource::parallel_for_each_chunk(
    uint32_t workers, const std::function<void(const Chunk&)>& fn) {
    if (shard_paths_.size() < 2 || workers < 2) return false;
    const uint32_t W = std::min<uint32_t>(workers,
                                          (uint32_t)shard_paths_.size());

    std::atomic<uint32_t> next_shard{0};
    std::vector<std::future<void>> futs;
    futs.reserve(W);
    for (uint32_t w = 0; w < W; ++w) {
        futs.push_back(std::async(std::launch::async, [&]() {
            ParquetSourceConfig pcfg;
            pcfg.vector_col = config_.vector_col;
            pcfg.payload_col = config_.payload_col;
            pcfg.normalize = config_.normalize;
            pcfg.batch_size = config_.batch_size;
            pcfg.num_threads = config_.num_threads;
            pcfg.use_mmap = config_.use_mmap;
    pcfg.vector_fp16 = config_.vector_fp16;
            for (;;) {
                const uint32_t s = next_shard.fetch_add(1);
                if (s >= shard_paths_.size()) break;
                ParquetSource src(shard_paths_[s], pcfg);
                src.set_vector_only(vector_only_);
                Chunk ch;
                while (src.next(ch))
                    if (ch.count) fn(ch);
            }
        }));
    }
    for (auto& f : futs) f.get();
    return true;
}

}  // namespace sextant
