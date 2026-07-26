#pragma once

/// @file buffered_io.hpp
/// Buffered file I/O for the scan path (lets the OS page cache manage
/// residency, as opposed to DirectFile's O_DIRECT bypass).
///
/// The scan path's access pattern is sequential whole-shard streams with
/// cross-query locality (queries routing to similar centroids re-probe the
/// same shards). That's exactly what the OS page cache is designed for:
/// hot shards stay cached automatically; cold shards stream from NVMe.
///
/// DirectFile (O_DIRECT) was the prior default — appropriate for the graph
/// path's random 4KB-per-hop reads (page-cache-thrashing) but NOT for the
/// scan path's sequential shard streams (page-cache-friendly). DiskANN
/// rejects buffered I/O because its access pattern is random; ours isn't.
///
/// On macOS, no special flags are needed (default is buffered). On Linux,
/// we open with plain O_RDONLY (no O_DIRECT) so the page cache is active.

#include <sextant/types.hpp>
#include <string>
#include <cstddef>
#include <cstdint>

namespace sextant {

/// Read-only buffered file. Open uses the kernel page cache (no O_DIRECT,
/// no F_NOCACHE). `read` is plain pread — buf/count/offset need NOT be
/// aligned (the kernel handles it). Throw on error.
class BufferedFile {
public:
    /// Open `path` for buffered reading. Throws on failure.
    explicit BufferedFile(const std::string& path);
    ~BufferedFile();

    BufferedFile(const BufferedFile&) = delete;
    BufferedFile& operator=(const BufferedFile&) = delete;
    BufferedFile(BufferedFile&&) noexcept;
    BufferedFile& operator=(BufferedFile&&) noexcept;

    /// Read up to `count` bytes at `offset` into `buf`. No alignment
    /// requirements on buf/count/offset — the kernel handles it via the
    /// page cache. Returns bytes read.
    size_t pread(void* buf, size_t count, uint64_t offset);

    /// File size in bytes.
    uint64_t size() const;

    int fd() const { return fd_; }

private:
    int fd_ = -1;
    std::string path_;
};

}  // namespace sextant
