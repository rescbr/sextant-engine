#include "direct_io.hpp"

#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sextant {

#ifdef __linux__
static constexpr int kDirectFlags = O_DIRECT;
#elif defined(__APPLE__)
static constexpr int kDirectFlags = 0;  // F_NOCACHE applied after open
#else
static constexpr int kDirectFlags = 0;
#endif

DirectFile::DirectFile(const std::string& path, bool create)
    : path_(path) {
    int flags = kDirectFlags;
    if (create) {
        flags |= O_RDWR | O_CREAT | O_TRUNC;
    } else {
        flags |= O_RDONLY;
    }

    fd_ = ::open(path.c_str(), flags, 0644);
    if (fd_ < 0) {
        throw Error(ErrorCode::IoError,
                    "DirectFile: failed to open '" + path + "': " +
                        std::strerror(errno));
    }

#if defined(__APPLE__)
    // macOS: set F_NOCACHE to advise the kernel to evict cached pages.
    // This is advisory, not enforced like Linux O_DIRECT. Dev/validation only.
    int opt = 1;
    if (::fcntl(fd_, F_NOCACHE, &opt) != 0) {
        spdlog::warn("DirectFile: F_NOCACHE failed on '{}' (non-fatal on macOS)",
                     path);
    }
#endif

    spdlog::debug("DirectFile: opened '{}'", path);
}

DirectFile::~DirectFile() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

DirectFile::DirectFile(DirectFile&& other) noexcept
    : fd_(other.fd_), path_(std::move(other.path_)) {
    other.fd_ = -1;
}

DirectFile& DirectFile::operator=(DirectFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = other.fd_;
        path_ = std::move(other.path_);
        other.fd_ = -1;
    }
    return *this;
}

size_t DirectFile::pread_aligned(void* buf, size_t count, uint64_t offset) {
    if (count == 0) return 0;

    // O_DIRECT on Linux requires buf, count, AND offset to all be aligned
    // to the logical block size. Check all three; if any is unaligned, go
    // through a staging buffer.
    const uint64_t off_misalign = offset & (kDiskAlign - 1);
    const size_t count_misalign = count & (kDiskAlign - 1);
    const bool buf_aligned =
        (reinterpret_cast<uintptr_t>(buf) & (kDiskAlign - 1)) == 0;

    if (off_misalign == 0 && count_misalign == 0 && buf_aligned) {
        ssize_t n = ::pread(fd_, buf, count, static_cast<off_t>(offset));
        if (n < 0) {
            throw Error(ErrorCode::IoError,
                        "DirectFile::pread failed on '" + path_ + "': " +
                            std::strerror(errno));
        }
        return static_cast<size_t>(n);
    }

    // Slow path: align offset down, round count up, read via staging buffer.
    const uint64_t aligned_off = offset - off_misalign;
    const size_t total = count + off_misalign;
    const size_t aligned_count =
        (total + kDiskAlign - 1) & ~static_cast<size_t>(kDiskAlign - 1);
    AlignedBuf stage(kDiskAlign, aligned_count);
    std::memset(stage.get(), 0, aligned_count);
    ssize_t n = ::pread(fd_, stage.get(), aligned_count,
                        static_cast<off_t>(aligned_off));
    if (n < 0) {
        throw Error(ErrorCode::IoError,
                    "DirectFile::pread failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
    const size_t usable = (static_cast<size_t>(n) > off_misalign)
        ? std::min(static_cast<size_t>(static_cast<size_t>(n) - off_misalign), count) : 0;
    std::memcpy(buf, stage.as<uint8_t>() + off_misalign, usable);
    return usable;
}

size_t DirectFile::pwrite_aligned(const void* buf, size_t count, uint64_t offset) {
    if (count == 0) return 0;

    // Mirror of pread_aligned: O_DIRECT requires buf, count, AND offset
    // aligned. macOS never enforces this (F_NOCACHE is advisory), and the
    // VM test coverage never hit an unaligned write — ZFS on Linux rejects
    // it with EINVAL (caught on the 9950X workstation).
    const uint64_t off_misalign = offset & (kDiskAlign - 1);
    const size_t count_misalign = count & (kDiskAlign - 1);
    const bool buf_aligned =
        (reinterpret_cast<uintptr_t>(buf) & (kDiskAlign - 1)) == 0;

    if (off_misalign == 0 && count_misalign == 0 && buf_aligned) {
        ssize_t n = ::pwrite(fd_, buf, count, static_cast<off_t>(offset));
        if (n < 0) {
            throw Error(ErrorCode::IoError,
                        "DirectFile::pwrite failed on '" + path_ + "': " +
                            std::strerror(errno));
        }
        return static_cast<size_t>(n);
    }

    // Slow path: align the window down/up and read-modify-write through a
    // staging buffer so neighboring bytes in the head/tail blocks survive.
    const uint64_t aligned_off = offset - off_misalign;
    const uint64_t end = offset + count;
    const uint64_t aligned_end =
        (end + kDiskAlign - 1) & ~static_cast<uint64_t>(kDiskAlign - 1);
    const size_t aligned_count = static_cast<size_t>(aligned_end - aligned_off);
    // Loud by design: unaligned direct writes defeat O_DIRECT (extra read +
    // staging copy, RMW races with concurrent writers). The call sites that
    // land here are being migrated to aligned writes or BufferedFile; once
    // the list is empty this path becomes a hard error.
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        spdlog::warn("DirectFile::pwrite unaligned (off={}, count={}) on '{}' "
                     "- staging through {}B buffer; align the caller",
                     offset, count, path_, aligned_count);
    }
    AlignedBuf stage(kDiskAlign, aligned_count);
    // Preserve existing head/tail bytes; a short read is fine when the write
    // extends the file (bytes past old EOF are zero-filled in the stage).
    ssize_t n = ::pread(fd_, stage.get(), aligned_count,
                        static_cast<off_t>(aligned_off));
    if (n < 0) {
        throw Error(ErrorCode::IoError,
                    "DirectFile::pwrite staging read failed on '" + path_ +
                        "': " + std::strerror(errno));
    }
    std::memcpy(stage.as<uint8_t>() + off_misalign, buf, count);
    n = ::pwrite(fd_, stage.get(), aligned_count,
                 static_cast<off_t>(aligned_off));
    if (n < 0) {
        throw Error(ErrorCode::IoError,
                    "DirectFile::pwrite failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
    return count;
}

uint64_t DirectFile::size() const {
    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        throw Error(ErrorCode::IoError,
                    "DirectFile::size failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
    return static_cast<uint64_t>(st.st_size);
}

void DirectFile::sync() {
    if (::fsync(fd_) != 0) {
        throw Error(ErrorCode::IoError,
                    "DirectFile::sync failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
}

void* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, alignment, size) != 0 || ptr == nullptr) {
        throw Error(ErrorCode::OutOfMemory,
                    "aligned_alloc: posix_memalign failed for size " +
                        std::to_string(size));
    }
    return ptr;
}

void aligned_free(void* ptr) {
    ::free(ptr);
}

}  // namespace sextant
