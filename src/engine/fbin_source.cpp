// FbinSource — .fbin / .ibin / .bbin / .bvecs reader.
//
// Issue 39: int8/uint8 values are cast to float on read with NO normalization
// and NO scaling — the raw integer value becomes the float value.

#include "fbin_source.hpp"
#include "sextant/error.hpp"

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <string>

namespace sextant {

FbinSource::ElemType FbinSource::infer_type(const std::string& path) {
    // Case-insensitive extension check.
    auto ends_with_ci = [&](const char* suf) {
        const size_t n = path.size();
        const size_t m = std::strlen(suf);
        if (n < m) return false;
        for (size_t i = 0; i < m; i++) {
            char c = path[n - m + i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != suf[i]) return false;
        }
        return true;
    };
    if (ends_with_ci(".ibin")) return ElemType::Int8;
    if (ends_with_ci(".bbin") || ends_with_ci(".bvecs")) return ElemType::Uint8;
    return ElemType::Float32;  // .fbin and anything else default to float32.
}

FbinSource::FbinSource(const std::string& path, uint32_t chunk_size)
    : path_(path), type_(infer_type(path)), chunk_size_(chunk_size) {
    if (chunk_size_ == 0) {
        chunk_size_ = kDefaultChunkSize;
    }

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw Error(ErrorCode::IoError,
                    "FbinSource: failed to open '" + path + "': " +
                        std::strerror(errno));
    }

    // Read the 8-byte header: [n:u32][dim:u32], little-endian.
    uint8_t hdr[8];
    ssize_t got = ::read(fd_, hdr, sizeof(hdr));
    if (got != static_cast<ssize_t>(sizeof(hdr))) {
        throw Error(ErrorCode::CorruptIndex,
                    "FbinSource: short header read on '" + path + "'");
    }
    uint32_t n = 0, d = 0;
    std::memcpy(&n, hdr, sizeof(n));
    std::memcpy(&d, hdr + 4, sizeof(d));
    if (n == 0 || d == 0) {
        throw Error(ErrorCode::CorruptIndex,
                    "FbinSource: zero n or dim in header of '" + path + "'");
    }
    count_ = n;
    dim_ = d;
    data_offset_ = 8;
    cursor_ = 0;

    // Pre-size the internal buffers for one chunk.
    vec_buf_.resize(static_cast<size_t>(chunk_size_) * dim_);
    rowid_buf_.resize(chunk_size_);
    for (uint32_t i = 0; i < chunk_size_; i++) {
        rowid_buf_[i] = static_cast<RowId>(i);
    }
    if (type_ != ElemType::Float32) {
        raw_buf_.resize(static_cast<size_t>(chunk_size_) * dim_);
    }

    spdlog::debug("FbinSource: opened '{}' n={} dim={} type={}", path, n, d,
                  type_ == ElemType::Float32  ? "f32"
                  : type_ == ElemType::Int8   ? "i8"
                                              : "u8");
}

FbinSource::~FbinSource() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void FbinSource::reset() {
    cursor_ = 0;
    if (::lseek(fd_, static_cast<off_t>(data_offset_), SEEK_SET) < 0) {
        throw Error(ErrorCode::IoError,
                    "FbinSource::reset: lseek failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
}

bool FbinSource::next(Chunk& out) {
    if (cursor_ >= count_) {
        out.count = 0;
        return false;
    }

    const uint32_t remaining =
        static_cast<uint32_t>(count_ - cursor_);
    const uint32_t this_chunk = std::min(chunk_size_, remaining);

    if (type_ == ElemType::Float32) {
        // Direct read into the float buffer.
        const size_t want_bytes =
            static_cast<size_t>(this_chunk) * dim_ * sizeof(float);
        size_t got = 0;
        while (got < want_bytes) {
            ssize_t r = ::read(fd_, vec_buf_.data() + got / sizeof(float),
                               want_bytes - got);
            if (r <= 0) {
                throw Error(ErrorCode::CorruptIndex,
                            "FbinSource::next: unexpected EOF on '" + path_ +
                                "'");
            }
            got += static_cast<size_t>(r);
        }
    } else {
        // Read raw bytes, cast element-by-element into the float buffer.
        const size_t want_bytes = static_cast<size_t>(this_chunk) * dim_;
        size_t got = 0;
        while (got < want_bytes) {
            ssize_t r = ::read(fd_, raw_buf_.data() + got, want_bytes - got);
            if (r <= 0) {
                throw Error(ErrorCode::CorruptIndex,
                            "FbinSource::next: unexpected EOF on '" + path_ +
                                "'");
            }
            got += static_cast<size_t>(r);
        }
        const size_t elems = static_cast<size_t>(this_chunk) * dim_;
        if (type_ == ElemType::Int8) {
            for (size_t i = 0; i < elems; i++) {
                vec_buf_[i] =
                    static_cast<float>(static_cast<int8_t>(raw_buf_[i]));
            }
        } else {  // Uint8
            for (size_t i = 0; i < elems; i++) {
                vec_buf_[i] = static_cast<float>(raw_buf_[i]);
            }
        }
    }

    // Shift row_ids so they track the global cursor. We reuse a fixed buffer
    // and rewrite the leading this_chunk entries.
    for (uint32_t i = 0; i < this_chunk; i++) {
        rowid_buf_[i] = static_cast<RowId>(cursor_ + i);
    }

    out.vectors = vec_buf_.data();
    out.row_ids = rowid_buf_.data();
    out.count = this_chunk;

    cursor_ += this_chunk;
    return true;
}

}  // namespace sextant
