#pragma once

/// @file block_allocator.hpp
/// Page-aligned block pool for direct I/O buffers.

#include <sextant/types.hpp>
#include <cstddef>
#include <cstdint>

namespace sextant {

/// Allocates aligned blocks of kBlockSize (256 KB) for direct I/O.
/// Blocks are managed as a simple pool — allocated up front and reused.
class BlockAllocator {
public:
    explicit BlockAllocator(uint32_t num_blocks);
    ~BlockAllocator();

    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;

    /// Acquire a block. Returns a kDiskAlign-aligned, kBlockSize buffer.
    uint8_t* acquire();

    /// Release a block back to the pool.
    void release(uint8_t* block);

    uint32_t capacity() const { return num_blocks_; }

private:
    uint32_t num_blocks_;
    uint8_t* pool_;
    uint8_t* next_free_;
};

}  // namespace sextant
