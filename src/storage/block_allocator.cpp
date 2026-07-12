#include "block_allocator.hpp"
#include "sextant/error.hpp"

#include <cstdlib>
#include <cstring>

namespace sextant {

BlockAllocator::BlockAllocator(uint32_t num_blocks)
    : num_blocks_(num_blocks), pool_(nullptr) {
    if (num_blocks == 0) return;

    // Single contiguous, kDiskAlign-aligned allocation of
    // num_blocks × kBlockSize. Each block then sits on a kBlockSize boundary,
    // which is itself a multiple of kDiskAlign (4096), so every block is
    // kDiskAlign-aligned — a hard requirement for O_DIRECT / direct I/O.
    const size_t total = static_cast<size_t>(num_blocks) * kBlockSize;

    void* raw = nullptr;
    if (::posix_memalign(&raw, kDiskAlign, total) != 0 || raw == nullptr) {
        throw Error(ErrorCode::OutOfMemory,
                    "BlockAllocator: posix_memalign failed for " +
                        std::to_string(total) + " bytes");
    }
    pool_ = static_cast<uint8_t*>(raw);
    std::memset(pool_, 0, total);

    // Seed the free-list with every block pointer.
    free_list_.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        free_list_.push_back(pool_ + static_cast<size_t>(i) * kBlockSize);
    }
}

BlockAllocator::~BlockAllocator() {
    ::free(pool_);
}

uint8_t* BlockAllocator::acquire() {
    if (free_list_.empty()) return nullptr;
    uint8_t* block = free_list_.back();
    free_list_.pop_back();
    return block;
}

void BlockAllocator::release(uint8_t* block) {
    if (block == nullptr) return;
    free_list_.push_back(block);
}

}  // namespace sextant
