#include "storage/code_stream.hpp"

#include "engine/sidecar_io.hpp"

#include <sextant/error.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace sextant {

CodeStream::CodeStream(const std::string& path, uint32_t m)
    : file_(path, /*create=*/false), m_(m) {
    if (m_ == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "CodeStream: m (segments) must be > 0");
    }
    block_bytes_ = m_ * 16;

    // Read + validate the header.
    SidecarHeader h{};
    engine_detail::read_exact(file_, &h, sizeof(h), /*offset=*/0);
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

    // Chunk size: 4 × kBlockSize = 1 MB pre-read, divided by the per-block
    // size. Clamped to at least 1 block (tiny shards) and at most n_blocks_.
    constexpr uint32_t kTargetChunkBytes = 4 * kBlockSize;  // 1 MB
    chunk_blocks_ = std::max<uint32_t>(
        1u, kTargetChunkBytes / std::max<uint32_t>(1u, block_bytes_));
    chunk_blocks_ = std::min(chunk_blocks_, n_blocks_);
}

CodeStream::~CodeStream() = default;

CodeStream::CodeStream(CodeStream&&) noexcept = default;
CodeStream& CodeStream::operator=(CodeStream&&) noexcept = default;

bool CodeStream::scan(
    const std::function<void(uint32_t, const uint8_t*, uint32_t)>& on_block,
    uint8_t* staging) {
    if (n_blocks_ == 0) return true;

    // The on-disk block ordering is the order `Builder::build_ivf_scan` wrote
    // them: shard-local lane index 0..shard_n-1 in ascending order (block b's
    // lane j → vector b*32+j). `valid_mask` for block b has bit j set iff
    // b*32 + j < n_vectors_.
    const uint64_t data_off = sizeof(SidecarHeader);
    const size_t chunk_bytes = static_cast<size_t>(chunk_blocks_) * block_bytes_;

    uint32_t b = 0;
    while (b < n_blocks_) {
        const uint32_t blocks_this = std::min(chunk_blocks_, n_blocks_ - b);
        const size_t bytes_this =
            static_cast<size_t>(blocks_this) * block_bytes_;
        const uint64_t off = data_off + static_cast<uint64_t>(b) * block_bytes_;

        // pread_aligned requires count to be a multiple of kDiskAlign and the
        // buffer to be kDiskAlign-aligned. `staging` is expected to be aligned
        // (caller uses AlignedBuf). `bytes_this` is a multiple of block_bytes_
        // = m*16; we pad up to kDiskAlign via read_exact's staging buffer.
        engine_detail::read_exact(file_, staging, bytes_this, off);

        for (uint32_t i = 0; i < blocks_this; i++) {
            const uint32_t gi = b + i;  // global block index
            const uint8_t* blk =
                staging + static_cast<size_t>(i) * block_bytes_;
            // Compute the 32-bit valid_mask: bit j set iff gi*32 + j <
            // n_vectors_. Full block → 0xFFFFFFFF; tail block → only the real
            // bits.
            uint32_t mask = 0;
            const uint32_t base = gi * 32;
            const uint32_t lanes =
                std::min<uint32_t>(32u, n_vectors_ - std::min(base, n_vectors_));
            for (uint32_t j = 0; j < lanes; j++) mask |= (1u << j);
            on_block(gi, blk, mask);
        }
        b += blocks_this;
    }
    return true;
}

}  // namespace sextant
