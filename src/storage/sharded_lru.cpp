#include "sharded_lru.hpp"
#include "direct_io.hpp"
#include "sextant/error.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

namespace sextant {

// --- LRUShard ----------------------------------------------------------------

LRUShard::LRUShard(uint32_t capacity_blocks)
    : capacity_(capacity_blocks) {
    map_.reserve(capacity_ + 1);
}

LRUShard::~LRUShard() {
    for (auto& [idx, entry] : map_) {
        aligned_free(entry.data);
    }
}

void LRUShard::move_to_front(LRUEntry* e) {
    if (e == head_) return;

    // Unlink
    if (e->prev) e->prev->next = e->next;
    if (e->next) e->next->prev = e->prev;
    if (e == tail_) tail_ = e->prev;

    // Insert at front
    e->prev = nullptr;
    e->next = head_;
    if (head_) head_->prev = e;
    head_ = e;
    if (!tail_) tail_ = e;
}

void LRUShard::evict_lru() {
    if (!tail_) return;

    LRUEntry* victim = tail_;
    // TODO: writeback if dirty (needs DirectFile reference).
    // For now, dirty blocks in search (read-only) cache are never dirty.

    // Unlink tail
    tail_ = victim->prev;
    if (tail_) tail_->next = nullptr;
    if (head_ == victim) head_ = nullptr;

    auto it = map_.find(victim->block_idx);
    if (it != map_.end()) {
        aligned_free(it->second.data);
        map_.erase(it);
    }
}

uint8_t* LRUShard::lookup(uint64_t block_idx) {
    auto it = map_.find(block_idx);
    if (it == map_.end()) return nullptr;
    move_to_front(&it->second);
    return it->second.data;
}

uint8_t* LRUShard::insert(uint64_t block_idx, const uint8_t* data,
                           uint32_t size) {
    // Check if already present (update)
    auto it = map_.find(block_idx);
    if (it != map_.end()) {
        std::memcpy(it->second.data, data, size);
        move_to_front(&it->second);
        return it->second.data;
    }

    // Evict if at capacity
    while (map_.size() >= capacity_) {
        evict_lru();
    }

    // Insert new entry
    uint8_t* buf = static_cast<uint8_t*>(aligned_alloc(kDiskAlign, size));
    std::memcpy(buf, data, size);

    auto [new_it, inserted] = map_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(block_idx),
        std::forward_as_tuple());
    LRUEntry& entry = new_it->second;
    entry.block_idx = block_idx;
    entry.data = buf;
    entry.dirty = false;
    entry.prev = nullptr;
    entry.next = nullptr;

    // Insert at front
    entry.next = head_;
    if (head_) head_->prev = &entry;
    head_ = &entry;
    if (!tail_) tail_ = &entry;

    return entry.data;
}

void LRUShard::mark_dirty(uint64_t block_idx) {
    auto it = map_.find(block_idx);
    if (it != map_.end()) {
        it->second.dirty = true;
    }
}

// --- ShardedLRUCache ---------------------------------------------------------

ShardedLRUCache::ShardedLRUCache(uint32_t num_shards,
                                   uint32_t blocks_per_shard,
                                   uint32_t block_size)
    : block_size_(block_size) {
    shards_.reserve(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        shards_.push_back(std::make_unique<LRUShard>(blocks_per_shard));
    }
}

ShardedLRUCache::~ShardedLRUCache() = default;

}  // namespace sextant
