#include "push_source.hpp"

#include "sextant/error.hpp"
#include "sextant/filter_column_data.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace sextant {

namespace {
constexpr uint32_t kChunkRows = 32768;
constexpr uint64_t kDefaultStagingBytes = 256ull << 20;

// Spill file chunk serialization. Each chunk:
//   [u32 count][vectors][row_ids]
//   [u64 cols_blob_len][cols_blob]     (present when schema non-empty)
//   [u64 payload_len][payload_offsets (count+1 × u32)][payload_data]
// The u64 lengths let vector-only passes seek past the blobs.

void w_u32(std::vector<uint8_t>& b, uint32_t v) {
    b.insert(b.end(), reinterpret_cast<const uint8_t*>(&v),
             reinterpret_cast<const uint8_t*>(&v) + 4);
}
void w_u64(std::vector<uint8_t>& b, uint64_t v) {
    b.insert(b.end(), reinterpret_cast<const uint8_t*>(&v),
             reinterpret_cast<const uint8_t*>(&v) + 8);
}

uint32_t r_u32(const uint8_t*& p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}
uint64_t r_u64(const uint8_t*& p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    p += 8;
    return v;
}
}  // namespace

PushStager::PushStager(uint32_t dim, Schema schema, bool has_payload,
                       std::string spill_path, uint64_t staging_bytes)
    : dim_(dim), schema_(std::move(schema)), has_payload_(has_payload),
      spill_path_(std::move(spill_path)),
      staging_bytes_(staging_bytes ? staging_bytes : kDefaultStagingBytes) {
    current_.cols.resize(schema_.columns.size());
    current_.payload_offsets.push_back(0);
}

PushStager::~PushStager() {
    if (spill_) {
        std::fclose(spill_);
        std::remove(spill_path_.c_str());
    }
}

void PushStager::append_vector(const float* v, RowId id) {
    current_.vectors.insert(current_.vectors.end(), v, v + dim_);
    current_.row_ids.push_back(id);
}

void PushStager::append_fixed(uint32_t col, const uint8_t* bytes, uint32_t width) {
    auto& c = current_.cols[col];
    c.fixed.insert(c.fixed.end(), bytes, bytes + width);
}

void PushStager::append_string(uint32_t col, const char* s, uint32_t len) {
    auto& c = current_.cols[col];
    c.str_offsets.push_back(static_cast<uint32_t>(c.str_data.size()));
    c.str_lengths.push_back(static_cast<uint16_t>(len));
    c.str_data.insert(c.str_data.end(), s, s + len);
}

void PushStager::append_set_row(uint32_t col, uint32_t n_elems, const char* const* elems,
                                const uint32_t* lens) {
    auto& c = current_.cols[col];
    c.set_counts.push_back(static_cast<uint8_t>(n_elems));
    c.set_offsets.push_back(static_cast<uint32_t>(c.set_elem_lengths.size()));
    for (uint32_t e = 0; e < n_elems; ++e) {
        c.set_elem_lengths.push_back(static_cast<uint16_t>(lens[e]));
        c.set_elem_data.insert(c.set_elem_data.end(), elems[e], elems[e] + lens[e]);
    }
}

void PushStager::append_payload(const uint8_t* data, uint32_t len) {
    current_.payload_data.insert(current_.payload_data.end(), data, data + len);
    payload_bytes_ += len;
}

void PushStager::end_row() {
    ++rows_;
    current_.count = static_cast<uint32_t>(current_.row_ids.size());
    current_.payload_offsets.push_back(static_cast<uint32_t>(current_.payload_data.size()));
    if (current_.count >= kChunkRows) {
        memory_chunks_.push_back(std::move(current_));
        current_ = StagedChunk{};
        current_.cols.resize(schema_.columns.size());
        current_.payload_offsets.push_back(0);
        maybe_spill();
    }
}

void PushStager::maybe_spill() {
    uint64_t staged = 0;
    for (const auto& c : memory_chunks_) {
        staged += c.vectors.size() * 4 + c.row_ids.size() * 8 +
                  c.payload_data.size();
        for (const auto& col : c.cols) {
            staged += col.fixed.size() + col.str_data.size() + col.set_elem_data.size();
        }
    }
    if (staged <= staging_bytes_) {
        return;
    }
    // Spill everything staged so far; RAM then holds only the accumulating
    // chunk.
    if (!spill_) {
        spill_ = std::fopen(spill_path_.c_str(), "wb+");
        if (!spill_) {
            throw Error(ErrorCode::IoError, "push staging: cannot create spill file " +
                                                spill_path_ + ": " + std::strerror(errno));
        }
    }
    for (const auto& c : memory_chunks_) {
        write_chunk(c);
        ++spilled_chunks_;
    }
    memory_chunks_.clear();
}

void PushStager::write_chunk(const StagedChunk& c) {
    std::vector<uint8_t> head;
    w_u32(head, c.count);
    if (std::fwrite(head.data(), 1, head.size(), spill_) != head.size() ||
        std::fwrite(c.vectors.data(), 4, c.vectors.size(), spill_) != c.vectors.size() ||
        std::fwrite(c.row_ids.data(), 8, c.row_ids.size(), spill_) != c.row_ids.size()) {
        throw Error(ErrorCode::IoError, "push staging: spill write failed");
    }

    // Filter columns blob.
    if (!schema_.columns.empty()) {
        std::vector<uint8_t> blob;
        for (const auto& col : c.cols) {
            w_u64(blob, col.fixed.size());
            blob.insert(blob.end(), col.fixed.begin(), col.fixed.end());
            w_u64(blob, col.str_data.size());
            blob.insert(blob.end(), col.str_data.begin(), col.str_data.end());
            w_u64(blob, col.str_offsets.size());
            for (uint32_t o : col.str_offsets) w_u32(blob, o);
            for (uint16_t l : col.str_lengths) {
                blob.insert(blob.end(), reinterpret_cast<const uint8_t*>(&l),
                            reinterpret_cast<const uint8_t*>(&l) + 2);
            }
            w_u64(blob, col.set_elem_data.size());
            blob.insert(blob.end(), col.set_elem_data.begin(), col.set_elem_data.end());
            w_u64(blob, col.set_elem_lengths.size());
            for (uint16_t l : col.set_elem_lengths) {
                blob.insert(blob.end(), reinterpret_cast<const uint8_t*>(&l),
                            reinterpret_cast<const uint8_t*>(&l) + 2);
            }
            w_u64(blob, col.set_counts.size());
            for (uint8_t v : col.set_counts) blob.push_back(v);
            w_u64(blob, col.set_offsets.size());
            for (uint32_t o : col.set_offsets) w_u32(blob, o);
        }
        std::vector<uint8_t> len;
        w_u64(len, blob.size());
        if (std::fwrite(len.data(), 1, len.size(), spill_) != len.size() ||
            std::fwrite(blob.data(), 1, blob.size(), spill_) != blob.size()) {
            throw Error(ErrorCode::IoError, "push staging: spill write failed");
        }
    }

    // Payload.
    {
        std::vector<uint8_t> len;
        w_u64(len, c.payload_data.size());
        if (std::fwrite(len.data(), 1, len.size(), spill_) != len.size()) {
            throw Error(ErrorCode::IoError, "push staging: spill write failed");
        }
        if (!c.payload_offsets.empty()) {
            if (std::fwrite(c.payload_offsets.data(), 4, c.payload_offsets.size(), spill_) !=
                c.payload_offsets.size()) {
                throw Error(ErrorCode::IoError, "push staging: spill write failed");
            }
        }
        if (!c.payload_data.empty() &&
            std::fwrite(c.payload_data.data(), 1, c.payload_data.size(), spill_) !=
                c.payload_data.size()) {
            throw Error(ErrorCode::IoError, "push staging: spill write failed");
        }
    }
}

void PushStager::read_chunk(StagedChunk& c, bool vector_only) {
    c = StagedChunk{};
    uint8_t head[4];
    if (std::fread(head, 1, 4, read_) != 4) {
        throw Error(ErrorCode::CorruptIndex, "push staging: spill truncated");
    }
    const uint8_t* hp = head;
    c.count = r_u32(hp);
    c.vectors.resize(static_cast<size_t>(c.count) * dim_);
    c.row_ids.resize(c.count);
    if (std::fread(c.vectors.data(), 4, c.vectors.size(), read_) != c.vectors.size() ||
        std::fread(c.row_ids.data(), 8, c.row_ids.size(), read_) != c.row_ids.size()) {
        throw Error(ErrorCode::CorruptIndex, "push staging: spill truncated");
    }

    if (!schema_.columns.empty()) {
        uint8_t lb[8];
        if (std::fread(lb, 1, 8, read_) != 8) {
            throw Error(ErrorCode::CorruptIndex, "push staging: spill truncated");
        }
        const uint8_t* lp = lb;
        const uint64_t blob_len = r_u64(lp);
        if (vector_only) {
            if (blob_len && fseeko(read_, static_cast<long>(blob_len), SEEK_CUR) != 0) {
                throw Error(ErrorCode::CorruptIndex, "push staging: spill seek failed");
            }
        } else {
            std::vector<uint8_t> blob(blob_len);
            if (std::fread(blob.data(), 1, blob_len, read_) != blob_len) {
                throw Error(ErrorCode::CorruptIndex, "push staging: spill truncated");
            }
            const uint8_t* p = blob.data();
            c.cols.resize(schema_.columns.size());
            for (auto& col : c.cols) {
                uint64_t n = r_u64(p);
                col.fixed.assign(p, p + n);
                p += n;
                n = r_u64(p);
                col.str_data.assign(reinterpret_cast<const char*>(p),
                                    reinterpret_cast<const char*>(p) + n);
                p += n;
                n = r_u64(p); // str_offsets count
                col.str_offsets.resize(n);
                for (uint64_t i = 0; i < n; ++i) col.str_offsets[i] = r_u32(p);
                col.str_lengths.resize(n);
                for (uint64_t i = 0; i < n; ++i) {
                    std::memcpy(&col.str_lengths[i], p, 2);
                    p += 2;
                }
                n = r_u64(p);
                col.set_elem_data.assign(reinterpret_cast<const char*>(p),
                                         reinterpret_cast<const char*>(p) + n);
                p += n;
                n = r_u64(p);
                col.set_elem_lengths.resize(n);
                for (uint64_t i = 0; i < n; ++i) {
                    std::memcpy(&col.set_elem_lengths[i], p, 2);
                    p += 2;
                }
                n = r_u64(p);
                col.set_counts.assign(p, p + n);
                p += n;
                n = r_u64(p);
                col.set_offsets.resize(n);
                for (uint64_t i = 0; i < n; ++i) col.set_offsets[i] = r_u32(p);
            }
        }
    }

    {
        uint8_t lb[8];
        if (std::fread(lb, 1, 8, read_) != 8) {
            throw Error(ErrorCode::CorruptIndex, "push staging: spill truncated");
        }
        const uint8_t* lp = lb;
        const uint64_t payload_len = r_u64(lp);
        if (vector_only) {
            // offsets (count+1 u32) + data
            const uint64_t skip = payload_len + uint64_t(c.count + 1) * 4;
            if (skip && fseeko(read_, static_cast<long>(skip), SEEK_CUR) != 0) {
                throw Error(ErrorCode::CorruptIndex, "push staging: spill seek failed");
            }
        } else {
            c.payload_offsets.resize(c.count + 1);
            if (std::fread(c.payload_offsets.data(), 4, c.payload_offsets.size(), read_) !=
                c.payload_offsets.size()) {
                throw Error(ErrorCode::CorruptIndex, "push staging: spill truncated");
            }
            c.payload_data.resize(payload_len);
            if (payload_len &&
                std::fread(c.payload_data.data(), 1, payload_len, read_) != payload_len) {
                throw Error(ErrorCode::CorruptIndex, "push staging: spill truncated");
            }
        }
    }
}

std::unique_ptr<StagedPushSource> PushStager::build_source() {
    // Flush the accumulating chunk.
    if (current_.count > 0) {
        memory_chunks_.push_back(std::move(current_));
        current_ = StagedChunk{};
    }
    maybe_spill();
    if (spill_) {
        std::fflush(spill_);
    }
    return std::make_unique<StagedPushSource>(*this);
}

//===--------------------------------------------------------------------===//
// StagedPushSource
//===--------------------------------------------------------------------===//

StagedPushSource::StagedPushSource(PushStager& stager) : stager_(stager) {
    fixed_views_.resize(stager_.schema_.columns.size());
    str_views_.resize(stager_.schema_.columns.size());
    set_views_.resize(stager_.schema_.columns.size());
}

void StagedPushSource::reset() {
    mem_idx_ = 0;
    if (stager_.spill_) {
        if (!stager_.read_) {
            // Reopen the spill file for reading (the write handle stays
            // open for the stager's lifetime; a separate read handle keeps
            // the two streams independent).
            stager_.read_ = std::fopen(stager_.spill_path_.c_str(), "rb");
            if (!stager_.read_) {
                throw Error(ErrorCode::IoError, "push staging: cannot reopen spill file");
            }
        }
        std::fseek(stager_.read_, 0, SEEK_SET);
    }
    remaining_spill_ = stager_.spilled_chunks_;
}

const StagedChunk* StagedPushSource::next_staged() {
    if (remaining_spill_ > 0) {
        --remaining_spill_;
        stager_.read_chunk(buf_, vector_only_);
        return &buf_;
    }
    if (mem_idx_ < stager_.memory_chunks_.size()) {
        return &stager_.memory_chunks_[mem_idx_++];
    }
    return nullptr;
}

bool StagedPushSource::next(Chunk& out) {
    const StagedChunk* c = next_staged();
    if (!c) {
        return false;
    }
    const auto& buf_ = *c;
    out.vectors = buf_.vectors.data();
    out.row_ids = buf_.row_ids.data();
    out.count = buf_.count;

    // Build filter views only when the chunk actually carries the filter
    // blob: vector-only reads (sample/PCA/Lloyd passes) skip it on the
    // spill tier, leaving cols empty — indexing cols here is OOB
    // (SIGSEGV, CulturaX-scale repro). Gate on the chunk's actual shape,
    // not the caller's flag.
    if (!stager_.schema_.columns.empty() &&
        buf_.cols.size() == stager_.schema_.columns.size()) {
        filter_ptrs_.clear();
        for (size_t ci = 0; ci < stager_.schema_.columns.size(); ++ci) {
            const auto& col = buf_.cols[ci];
            switch (stager_.schema_.columns[ci].type) {
                case ColumnType::Int32:
                case ColumnType::Int64:
                case ColumnType::Float:
                case ColumnType::Bool:
                    filter_ptrs_.push_back(col.fixed.data());
                    break;
                case ColumnType::String: {
                    auto& v = str_views_[ci];
                    v.offsets = col.str_offsets.data();
                    v.lengths = col.str_lengths.data();
                    v.data = col.str_data.data();
                    v.total_data_bytes = static_cast<uint32_t>(col.str_data.size());
                    filter_ptrs_.push_back(&v);
                    break;
                }
                case ColumnType::Set: {
                    auto& v = set_views_[ci];
                    v.counts = col.set_counts.data();
                    v.offsets = col.set_offsets.data();
                    v.element_lengths = col.set_elem_lengths.data();
                    v.element_data = col.set_elem_data.data();
                    v.total_elements = static_cast<uint32_t>(col.set_elem_lengths.size());
                    v.total_data_bytes = static_cast<uint32_t>(col.set_elem_data.size());
                    filter_ptrs_.push_back(&v);
                    break;
                }
            }
        }
        out.filter_columns = filter_ptrs_.data();
    } else {
        out.filter_columns = nullptr;
    }

    if (stager_.has_payload_) {
        out.payload_data = buf_.payload_data.data();
        out.payload_offsets = buf_.payload_offsets.data();
    } else {
        out.payload_data = nullptr;
        out.payload_offsets = nullptr;
    }
    return true;
}

}  // namespace sextant
