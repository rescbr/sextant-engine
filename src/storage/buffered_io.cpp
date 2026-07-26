#include "buffered_io.hpp"

#include "sextant/error.hpp"

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace sextant {

BufferedFile::BufferedFile(const std::string& path)
    : path_(path) {
    // Plain buffered open — no O_DIRECT, no F_NOCACHE. The kernel page cache
    // is active and will retain hot pages across reads.
    fd_ = ::open(path.c_str(), O_RDONLY, 0644);
    if (fd_ < 0) {
        throw Error(ErrorCode::IoError,
                    "BufferedFile: failed to open '" + path + "': " +
                        std::strerror(errno));
    }
    spdlog::debug("BufferedFile: opened '{}'", path);
}

BufferedFile::~BufferedFile() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

BufferedFile::BufferedFile(BufferedFile&& other) noexcept
    : fd_(other.fd_), path_(std::move(other.path_)) {
    other.fd_ = -1;
}

BufferedFile& BufferedFile::operator=(BufferedFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = other.fd_;
        path_ = std::move(other.path_);
        other.fd_ = -1;
    }
    return *this;
}

size_t BufferedFile::pread(void* buf, size_t count, uint64_t offset) {
    if (count == 0) return 0;
    // Plain pread; the kernel handles any alignment via the page cache.
    // Handle short reads (rare for regular files but defensive).
    size_t total = 0;
    while (total < count) {
        ssize_t n = ::pread(fd_, static_cast<char*>(buf) + total,
                             count - total,
                             static_cast<off_t>(offset + total));
        if (n < 0) {
            throw Error(ErrorCode::IoError,
                        "BufferedFile::pread failed on '" + path_ + "': " +
                            std::strerror(errno));
        }
        if (n == 0) break;  // EOF
        total += static_cast<size_t>(n);
    }
    return total;
}

uint64_t BufferedFile::size() const {
    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        throw Error(ErrorCode::IoError,
                    "BufferedFile::size failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
    return static_cast<uint64_t>(st.st_size);
}

}  // namespace sextant
