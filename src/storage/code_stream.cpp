#include "storage/code_stream.hpp"

#include "engine/sidecar_io.hpp"

#include <sextant/error.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <sys/mman.h>
#include <utility>

namespace sextant {

CodeStream::CodeStream(const std::string& path, uint32_t m)
    : file_(path), m_(m) {
    if (m_ == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "CodeStream: m (segments) must be > 0");
    }
    block_bytes_ = m_ * 16;

    SidecarHeader h{};
    engine_detail::read_exact(file_, &h, sizeof(h), 0);
    if (h.magic != kMagicCodes4) {
        throw Error(ErrorCode::CorruptIndex,
                    "CodeStream: magic mismatch on '" + path +
                        "' (expected kMagicCodes4)");
    }
    if (h.dim == 0 || h.n_vectors == 0) {
        throw Error(ErrorCode::CorruptIndex,
                    "CodeStream: empty/corrupt header on '" + path + "'");
    }
    n_vectors_ = static_cast<uint32_t>(h.n_vectors);
    n_blocks_ = (n_vectors_ + 31) / 32;

    constexpr uint32_t kTargetChunkBytes = 4 * kBlockSize;
    chunk_blocks_ = std::max<uint32_t>(
        1u, kTargetChunkBytes / std::max<uint32_t>(1u, block_bytes_));
    chunk_blocks_ = std::min(chunk_blocks_, n_blocks_);

    file_size_ = file_.size();
    if (file_size_ > 0) {
        // MAP_SHARED + PROT_READ: the lightest-weight read-only file mapping.
        // No copy-on-write, no swap reservation (unlike MAP_PRIVATE which
        // reserves swap for potential CoW copies that never happen under
        // PROT_READ). The mapping is backed directly by the file's page cache.
        void* p = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED,
                         file_.fd(), 0);
        mapped_ = (p == MAP_FAILED) ? nullptr : static_cast<const uint8_t*>(p);
        // No madvise: each shard is small (~2MB at K=64), fits in L2/L3.
        // The default kernel policy (adaptive readahead) handles both the
        // cold first-pass (sequential within a shard) and warm steady-state
        // (resident pages). MADV_SEQUENTIAL was tried and caused page-cache
        // thrashing when competing with the base-data mmap (4GB+); the
        // default adaptive policy avoids that.
    }
}

CodeStream::~CodeStream() {
    if (mapped_) ::munmap((void*)mapped_, file_size_);
}

CodeStream::CodeStream(CodeStream&& other) noexcept
    : file_(std::move(other.file_)), m_(other.m_),
      block_bytes_(other.block_bytes_), n_blocks_(other.n_blocks_),
      n_vectors_(other.n_vectors_), chunk_blocks_(other.chunk_blocks_),
      mapped_(other.mapped_), file_size_(other.file_size_) {
    other.mapped_ = nullptr;
}

CodeStream& CodeStream::operator=(CodeStream&& other) noexcept {
    if (this != &other) {
        if (mapped_) ::munmap((void*)mapped_, file_size_);
        file_ = std::move(other.file_);
        m_ = other.m_;
        block_bytes_ = other.block_bytes_;
        n_blocks_ = other.n_blocks_;
        n_vectors_ = other.n_vectors_;
        chunk_blocks_ = other.chunk_blocks_;
        mapped_ = other.mapped_;
        file_size_ = other.file_size_;
        other.mapped_ = nullptr;
    }
    return *this;
}

bool CodeStream::scan(
    const std::function<void(uint32_t, const uint8_t*, uint32_t)>& on_block,
    uint8_t* staging) {
    if (n_blocks_ == 0) return true;
    const uint64_t data_off = sizeof(SidecarHeader);

    if (mapped_) {
        const uint8_t* base_ptr = mapped_ + data_off;
        for (uint32_t b = 0; b < n_blocks_; b++) {
            const uint8_t* blk = base_ptr + (size_t)b * block_bytes_;
            uint32_t mask = 0;
            const uint32_t lane_base = b * 32;
            const uint32_t lanes = std::min<uint32_t>(
                32u, n_vectors_ - std::min(lane_base, n_vectors_));
            for (uint32_t j = 0; j < lanes; j++) mask |= (1u << j);
            on_block(b, blk, mask);
        }
        return true;
    }

    uint32_t b = 0;
    while (b < n_blocks_) {
        const uint32_t blocks_this = std::min(chunk_blocks_, n_blocks_ - b);
        const size_t bytes_this = (size_t)blocks_this * block_bytes_;
        const uint64_t off = data_off + (uint64_t)b * block_bytes_;
        engine_detail::read_exact(file_, staging, bytes_this, off);
        for (uint32_t i = 0; i < blocks_this; i++) {
            const uint32_t gi = b + i;
            const uint8_t* blk = staging + (size_t)i * block_bytes_;
            uint32_t mask = 0;
            const uint32_t base = gi * 32;
            const uint32_t lanes = std::min<uint32_t>(32u,
                n_vectors_ - std::min(base, n_vectors_));
            for (uint32_t j = 0; j < lanes; j++) mask |= (1u << j);
            on_block(gi, blk, mask);
        }
        b += blocks_this;
    }
    return true;
}

}  // namespace sextant
