#include "block_buffer_pool.hpp"
#include "direct_io.hpp"

#include <cstdlib>

namespace sextant {

BlockBufferPool::BlockBufferPool(uint32_t block_size)
    : block_size_(block_size) {}

BlockBufferPool::~BlockBufferPool() {
    // The CacheShard destructor releases every entry's buffer back to the
    // pool before the pool is destroyed, so free_list_ owns every outstanding
    // buffer at this point.
    for (uint8_t* buf : free_list_) {
        aligned_free(buf);
    }
}

uint8_t* BlockBufferPool::acquire() {
    {
        ScopedWriteLock lock(mu_);
        if (!free_list_.empty()) {
            uint8_t* buf = free_list_.back();
            free_list_.pop_back();
            return buf;
        }
    }
    // Pool empty: fall back to the system allocator. The buffer will be
    // recycled on the next release().
    return static_cast<uint8_t*>(aligned_alloc(kDiskAlign, block_size_));
}

void BlockBufferPool::release(uint8_t* buf) {
    if (buf == nullptr) return;
    ScopedWriteLock lock(mu_);
    free_list_.push_back(buf);
}

uint32_t BlockBufferPool::free_count() const {
    ScopedWriteLock lock(mu_);
    return static_cast<uint32_t>(free_list_.size());
}

}  // namespace sextant
