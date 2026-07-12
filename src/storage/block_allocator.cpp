#include "block_allocator.hpp"
#include "sextant/error.hpp"

#include <cstdlib>
#include <cstring>

namespace sextant {

BlockAllocator::BlockAllocator(uint32_t num_blocks)
    : num_blocks_(num_blocks) {
    // Each block: kBlockSize data + kDiskAlign padding for alignment.
    // We over-allocate and align the base.
    const size_t block_stride = kBlockSize;
    const size_t total = static_cast<size_t>(num_blocks) * block_stride + kDiskAlign;

    void* raw = nullptr;
    if (::posix_memalign(&raw, kDiskAlign, total) != 0 || raw == nullptr) {
        throw Error(ErrorCode::OutOfMemory, "BlockAllocator: OOM");
    }
    pool_ = static_cast<uint8_t*>(raw);
    std::memset(pool_, 0, total);
    next_free_ = pool_;
}

BlockAllocator::~BlockAllocator() {
    ::free(pool_);
}

uint8_t* BlockAllocator::acquire() {
    // Simple bump allocator for now. A free-list is added when the LRU
    // needs eviction-aware allocation.
    // TODO: implement free-list for reuse.
    return nullptr;  // stub
}

void BlockAllocator::release(uint8_t* /*block*/) {
    // TODO: implement free-list.
}

}  // namespace sextant
