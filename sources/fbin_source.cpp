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

    // Pre-size the internal buffers for one chunk (double-buffered).
    for (int b = 0; b < 2; ++b) {
        vec_buf_[b].resize(static_cast<size_t>(chunk_size_) * dim_);
        rowid_buf_[b].resize(chunk_size_);
    }
    if (type_ != ElemType::Float32) {
        for (int b = 0; b < 2; ++b)
            raw_buf_[b].resize(static_cast<size_t>(chunk_size_) * dim_);
    }

    spdlog::debug("FbinSource: opened '{}' n={} dim={} type={}", path, n, d,
                  type_ == ElemType::Float32  ? "f32"
                  : type_ == ElemType::Int8   ? "i8"
                                              : "u8");
}

FbinSource::~FbinSource() {
    if (pf_live_) {
        pf_thread_.join();
        pf_live_ = false;
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void FbinSource::reset() {
    if (pf_live_) {
        pf_thread_.join();
        pf_live_ = false;
    }
    have_pending_ = false;
    pf_err_.clear();
    cursor_ = 0;
}

void FbinSource::read_chunk(int buf, uint64_t at, uint32_t n) {
    const bool f32 = type_ == ElemType::Float32;
    const size_t want_bytes = f32
        ? static_cast<size_t>(n) * dim_ * sizeof(float)
        : static_cast<size_t>(n) * dim_;
    const uint64_t off = data_offset_ + at * dim_ * (f32 ? 4ULL : 1ULL);
    uint8_t* dst = f32
        ? reinterpret_cast<uint8_t*>(vec_buf_[buf].data())
        : raw_buf_[buf].data();
    size_t got = 0;
    while (got < want_bytes) {
        ssize_t r = ::pread(fd_, dst + got, want_bytes - got,
                            static_cast<off_t>(off + got));
        if (r <= 0) {
            throw Error(ErrorCode::CorruptIndex,
                        "FbinSource: unexpected EOF on '" + path_ + "'");
        }
        got += static_cast<size_t>(r);
    }
    bytes_read_.fetch_add(want_bytes, std::memory_order_relaxed);
    if (!f32) {
        const size_t elems = static_cast<size_t>(n) * dim_;
        if (type_ == ElemType::Int8) {
            for (size_t i = 0; i < elems; i++)
                vec_buf_[buf][i] = static_cast<float>(
                    static_cast<int8_t>(raw_buf_[buf][i]));
        } else {
            for (size_t i = 0; i < elems; i++)
                vec_buf_[buf][i] = static_cast<float>(raw_buf_[buf][i]);
        }
    }
}

bool FbinSource::next(Chunk& out) {
    // Collect any in-flight prefetch before handing out a buffer. This
    // join is the consumer-side I/O wait: if compute ran ahead of the
    // prefetch thread, the block here is exactly the worker-idle signal
    // the metrics attribute as source_wait_seconds (a finished prefetch
    // makes the join return in ~0 time and adds nothing).
    const auto wait_t0 = std::chrono::steady_clock::now();
    const bool was_live = pf_live_;
    if (pf_live_) {
        pf_thread_.join();
        pf_live_ = false;
        if (!pf_err_.empty())
            throw Error(ErrorCode::CorruptIndex, pf_err_);
    }
    if (was_live) {
        const double waited = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wait_t0).count();
        wait_seconds_.fetch_add(waited, std::memory_order_relaxed);
        if (waited > 0.0005)  // ~syscall noise floor: count real stalls only
            wait_count_.fetch_add(1, std::memory_order_relaxed);
    }

    int buf;
    uint32_t n;
    if (have_pending_) {
        have_pending_ = false;
        buf = pend_buf_;
        n = pend_count_;
    } else {
        if (cursor_ >= count_) { out.count = 0; return false; }
        n = std::min(chunk_size_, static_cast<uint32_t>(count_ - cursor_));
        buf = cur_ ^ 1;  // sync read into the buffer not in use
        read_chunk(buf, cursor_, n);
    }

    for (uint32_t i = 0; i < n; i++)
        rowid_buf_[buf][i] = static_cast<RowId>(cursor_ + i);
    out.vectors = vec_buf_[buf].data();
    out.row_ids = rowid_buf_[buf].data();
    out.count = n;
    cursor_ += n;
    cur_ = buf;

    // Kick off the following chunk's read in the background.
    if (cursor_ < count_) {
        const uint32_t nn = std::min(
            chunk_size_, static_cast<uint32_t>(count_ - cursor_));
        const int nb = buf ^ 1;
        pend_buf_ = nb;
        pend_count_ = nn;
        pf_live_ = true;
        const uint64_t at = cursor_;
        pf_thread_ = std::thread([this, nb, at, nn] {
            try {
                read_chunk(nb, at, nn);
            } catch (const std::exception& e) {
                pf_err_ = e.what();
            }
        });
        have_pending_ = true;
    }
    return true;
}

}  // namespace sextant
