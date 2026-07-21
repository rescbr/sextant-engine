#pragma once

/// @file direct_io.hpp
/// O_DIRECT / F_NOCACHE file I/O with aligned buffers.
///
/// DirectFile wraps a file descriptor opened for direct I/O:
/// - Linux: O_DIRECT, aligned buffers via posix_memalign.
/// - macOS: F_NOCACHE (advisory, dev/validation only).
///
/// All reads/writes use pread/pwrite. No buffered I/O leaks through.

#include <sextant/types.hpp>
#include <string>
#include <cstddef>
#include <cstdint>

namespace sextant {

class DirectFile {
public:
    /// Open a file for direct I/O. If `create` is true, the file is created
    /// (or truncated). Throws sextant::Error on failure.
    DirectFile(const std::string& path, bool create = false);
    ~DirectFile();

    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;
    DirectFile(DirectFile&&) noexcept;
    DirectFile& operator=(DirectFile&&) noexcept;

    /// Read `count` bytes at `offset` into an aligned buffer.
    /// `buf` must be aligned to kDiskAlign and be a multiple of kDiskAlign
    /// in size (on Linux O_DIRECT). Returns bytes read.
    size_t pread_aligned(void* buf, size_t count, uint64_t offset);

    /// Write `count` bytes at `offset` from an aligned buffer.
    size_t pwrite_aligned(const void* buf, size_t count, uint64_t offset);

    /// File size in bytes.
    uint64_t size() const;

    /// Flush to disk (fsync).
    void sync();

    int fd() const { return fd_; }

private:
    int fd_ = -1;
    std::string path_;
};

/// Allocate an aligned buffer (posix_memalign). Throws on OOM.
/// Use aligned_free to release.
void* aligned_alloc(size_t alignment, size_t size);
void aligned_free(void* ptr);

/// RAII owning wrapper around an aligned allocation. Frees on destruction;
/// move-only; supports release() to hand ownership to a raw pointer (e.g.
/// when transferring to a long-lived owner like `Index::codes_buffer`).
///
/// Use this instead of raw `aligned_alloc` + `aligned_free` locals: it closes
/// the leak-on-exception window between alloc and free, and documents ownership
/// at the type level.
class AlignedBuf {
public:
    AlignedBuf() noexcept = default;
    AlignedBuf(size_t alignment, size_t size)
        : ptr_(aligned_alloc(alignment, size)), size_(size) {}

    ~AlignedBuf() { reset(); }

    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    AlignedBuf(AlignedBuf&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_) { o.ptr_ = nullptr; o.size_ = 0; }
    AlignedBuf& operator=(AlignedBuf&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_; size_ = o.size_;
            o.ptr_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    void* get() noexcept { return ptr_; }
    const void* get() const noexcept { return ptr_; }
    template <typename T> T* as() noexcept { return static_cast<T*>(ptr_); }
    template <typename T> const T* as() const noexcept { return static_cast<const T*>(ptr_); }
    size_t size() const noexcept { return size_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    /// Release ownership. The caller is now responsible for `aligned_free`.
    void* release() noexcept {
        void* p = ptr_;
        ptr_ = nullptr; size_ = 0;
        return p;
    }

    void reset() noexcept {
        if (ptr_) { aligned_free(ptr_); ptr_ = nullptr; size_ = 0; }
    }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

}  // namespace sextant
