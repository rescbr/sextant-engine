#include "direct_io.hpp"

#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
    ssize_t n = ::pread(fd_, buf, count, static_cast<off_t>(offset));
    if (n < 0) {
        throw Error(ErrorCode::IoError,
                    "DirectFile::pread failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
    return static_cast<size_t>(n);
}

size_t DirectFile::pwrite_aligned(const void* buf, size_t count, uint64_t offset) {
    ssize_t n = ::pwrite(fd_, buf, count, static_cast<off_t>(offset));
    if (n < 0) {
        throw Error(ErrorCode::IoError,
                    "DirectFile::pwrite failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
    return static_cast<size_t>(n);
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
