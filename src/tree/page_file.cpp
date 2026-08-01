#include "page_file.hpp"

#include "sextant/error.hpp"

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>

namespace sextant::tree {

PageFile::PageFile(const std::string& path)
    : path_(path) {
    // O_RDWR | O_CREAT: the tree file is read/write during build and vacuum.
    // No O_DIRECT — buffered I/O with explicit fdatasync at commit points.
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw Error(ErrorCode::IoError,
                    "PageFile: failed to open '" + path + "': " +
                        std::strerror(errno));
    }
    spdlog::debug("PageFile: opened '{}'", path);
}

PageFile::~PageFile() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

PageFile::PageFile(PageFile&& other) noexcept
    : fd_(other.fd_), path_(std::move(other.path_)) {
    other.fd_ = -1;
}

PageFile& PageFile::operator=(PageFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = other.fd_;
        path_ = std::move(other.path_);
        other.fd_ = -1;
    }
    return *this;
}

void PageFile::read_pages(PageId page, uint32_t n_pages, void* buf) const {
    if (n_pages == 0) return;
    const uint64_t offset = static_cast<uint64_t>(page) * kPageSize;
    const size_t count = static_cast<size_t>(n_pages) * kPageSize;

    size_t total = 0;
    while (total < count) {
        ssize_t n = ::pread(fd_, static_cast<char*>(buf) + total,
                            count - total,
                            static_cast<off_t>(offset + total));
        if (n < 0) {
            throw Error(ErrorCode::IoError,
                        "PageFile::read_pages failed on '" + path_ +
                            "' at page " + std::to_string(page) + ": " +
                            std::strerror(errno));
        }
        if (n == 0) {
            throw Error(ErrorCode::CorruptIndex,
                        "PageFile::read_pages: unexpected EOF on '" + path_ +
                            "' at page " + std::to_string(page) +
                            " (short by " + std::to_string(count - total) +
                            " bytes)");
        }
        total += static_cast<size_t>(n);
    }
}

void PageFile::write_pages(PageId page, uint32_t n_pages, const void* buf) {
    if (n_pages == 0) return;
    const uint64_t offset = static_cast<uint64_t>(page) * kPageSize;
    const size_t count = static_cast<size_t>(n_pages) * kPageSize;

    size_t total = 0;
    while (total < count) {
        ssize_t n = ::pwrite(fd_, static_cast<const char*>(buf) + total,
                             count - total,
                             static_cast<off_t>(offset + total));
        if (n < 0) {
            throw Error(ErrorCode::IoError,
                        "PageFile::write_pages failed on '" + path_ +
                            "' at page " + std::to_string(page) + ": " +
                            std::strerror(errno));
        }
        if (n == 0) {
            throw Error(ErrorCode::IoError,
                        "PageFile::write_pages: wrote 0 bytes on '" + path_ +
                            "' at page " + std::to_string(page));
        }
        total += static_cast<size_t>(n);
    }
}

void PageFile::sync() {
    // fdatasync on Linux (does not flush metadata unless size changes);
    // fsync on macOS (no fdatasync available).
#if defined(__APPLE__)
    if (::fsync(fd_) != 0) {
#else
    if (::fdatasync(fd_) != 0) {
#endif
        throw Error(ErrorCode::IoError,
                    "PageFile::sync failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
}

void PageFile::truncate(uint64_t n_pages) {
    const uint64_t new_size = n_pages * kPageSize;
    if (::ftruncate(fd_, static_cast<off_t>(new_size)) != 0) {
        throw Error(ErrorCode::IoError,
                    "PageFile::truncate failed on '" + path_ + "' to " +
                        std::to_string(n_pages) + " pages: " +
                        std::strerror(errno));
    }
}

uint64_t PageFile::num_pages() const {
    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        throw Error(ErrorCode::IoError,
                    "PageFile::num_pages (fstat) failed on '" + path_ + "': " +
                        std::strerror(errno));
    }
    // Round up to page granularity (a partial trailing page counts as one).
    return (static_cast<uint64_t>(st.st_size) + kPageSize - 1) / kPageSize;
}

}  // namespace sextant::tree
