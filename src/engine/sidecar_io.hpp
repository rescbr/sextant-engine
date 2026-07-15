#pragma once

/// @file sidecar_io.hpp
/// Internal helpers shared between engine.cpp (flush) and search.cpp (load):
/// padded aligned reads/writes around DirectFile, and SidecarHeader fill.

#include "storage/direct_io.hpp"
#include "storage/sidecar_header.hpp"

#include <chrono>
#include <cstring>
#include <utility>

namespace sextant {
namespace engine_detail {

/// Write `count` bytes from `buf` to a DirectFile at `offset`, rounding the
/// on-disk size up to kDiskAlign by zero-padding a staging buffer (the tail
/// padding is never read back — readers pass the exact payload size).
inline void write_padded(DirectFile& f, const void* buf, size_t count,
                         uint64_t offset) {
    if (count == 0) return;
    const size_t aligned =
        (count + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
    if (aligned == count) {
        f.pwrite_aligned(buf, count, offset);
        return;
    }
    void* stage = aligned_alloc(kDiskAlign, aligned);
    std::memset(stage, 0, aligned);
    std::memcpy(stage, buf, count);
    f.pwrite_aligned(stage, aligned, offset);
    aligned_free(stage);
}

/// Read exactly `count` bytes at `offset` into `buf` from a DirectFile, via an
/// aligned staging buffer. The caller must ensure [offset, offset+count) is
/// within the file (or tolerate trailing zeros from a padded write).
inline void read_exact(DirectFile& f, void* buf, size_t count, uint64_t offset) {
    if (count == 0) return;
    // pread_aligned handles offset alignment internally; we just need
    // an aligned staging buffer for the O_DIRECT count requirement.
    const size_t aligned =
        (count + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
    void* stage = aligned_alloc(kDiskAlign, aligned);
    std::memset(stage, 0, aligned);
    f.pread_aligned(stage, aligned, offset);
    std::memcpy(buf, stage, count);
    aligned_free(stage);
}

/// Fill a SidecarHeader with the common fields.
inline void fill_header(SidecarHeader& h, uint64_t magic, uint64_t n_vectors,
                        uint32_t dim,
                        const std::pair<uint64_t, uint64_t>& uuid) {
    std::memset(&h, 0, sizeof(h));
    h.magic = magic;
    h.format_version = kFormatVersion;
    h.header_size = sizeof(SidecarHeader);
    h.data_offset = sizeof(SidecarHeader);
    h.created_unix = static_cast<uint64_t>(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    h.index_uuid_low = uuid.first;
    h.index_uuid_high = uuid.second;
    h.n_vectors = n_vectors;
    h.dim = dim;
    h.checksum_algo = 0;
}

}  // namespace engine_detail
}  // namespace sextant
