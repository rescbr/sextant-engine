#pragma once

/// @file block_allocator.hpp
/// Page-aligned block pool for direct I/O buffers.

#include <sextant/types.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sextant {

/// Allocates aligned blocks of kBlockSize (256 KB) for direct I/O.
/// Blocks are managed as a simple pool — allocated up front and reused via a
/// free-list (stack of pointers). Since every block is the same fixed size,
/// a free-list avoids fragmentation entirely.
class BlockAllocator {
public:
    explicit BlockAllocator(uint32_t num_blocks);
    ~BlockAllocator();

    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;

    /// Acquire a block. Returns a kDiskAlign-aligned, kBlockSize buffer,
    /// or nullptr if the pool is exhausted.
    uint8_t* acquire();

    /// Release a block back to the pool (free-list reuse).
    void release(uint8_t* block);

    uint32_t capacity() const { return num_blocks_; }

    /// Number of blocks currently available for acquire().
    uint32_t available() const { return static_cast<uint32_t>(free_list_.size()); }

private:
    uint32_t num_blocks_;
    uint8_t* pool_;  ///< Owned aligned backing allocation (freed in dtor).
    std::vector<uint8_t*> free_list_;  ///< Stack of free block pointers.
};

}  // namespace sextant
