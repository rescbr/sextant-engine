#pragma once

/// @file frequency_sketch.hpp
/// 4-bit CountMinSketch for W-TinyLFU admission control.
///
/// Ports Caffeine's FrequencySketch (Apache 2.0) to C++23. Tracks the historic
/// access frequency of block indices so the W-TinyLFU policy can decide whether
/// a window-overflow candidate deserves a slot in the main (SLRU) space.
///
/// Design (see Caffeine's FrequencySketch.java):
///   - `table` is an array of `std::atomic<uint64_t>`, each slot holding 16
///     4-bit counters (max 15).
///   - Four counters per item, all drawn from a single 64-byte block (one L1
///     cache line) for spatial locality.
///   - `spread()` is a 3-round multiplicative hash; `rehash()` selects counters.
///   - Aging: every `10 * maximum` increments, all counters are halved
///     (`>> 1 & RESET_MASK`), keeping the sketch fresh.
///
/// Thread-safety: fully atomic. `increment()` and `frequency()` are safe to
/// call concurrently from any thread. Caffeine uses the same design (Java's
/// AtomicLongArray + compareAndSwap). Previously the owning CacheShard guarded
/// the sketch with its write lock; atomicity lets CacheShard::lookup drop the
/// write lock on the read path.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>

namespace sextant {

class FrequencySketch {
public:
    FrequencySketch() = default;

    /// Initializes (or grows) the sketch to track up to `maximum_entries`
    /// distinct keys. Forgets all counts when resizing. No-op if already large
    /// enough. A minimum of 256 entries is enforced for accuracy.
    /// NOT thread-safe with respect to increment()/frequency() — call only
    /// at construction or under the owning shard's write lock (e.g. resize()).
    void ensure_capacity(uint32_t maximum_entries) {
        // Clamp to a sane maximum to avoid overflow in ceiling_power_of_two.
        uint32_t maximum = std::max(maximum_entries, MIN_SKETCH_SIZE);
        uint32_t new_size = ceiling_power_of_two(maximum);
        if (table_ && table_size_ >= maximum) {
            return;
        }

        sample_size_ = 10u * maximum;
        // unique_ptr<atomic[]> because std::vector<atomic> is not
        // MoveInsertable (matches BlockCache::shard_epochs_ pattern).
        table_ = std::make_unique<std::atomic<uint64_t>[]>(new_size);
        for (uint32_t i = 0; i < new_size; ++i) table_[i].store(0, std::memory_order_relaxed);
        table_size_ = new_size;
        block_mask_ = (new_size >> 3) - 1;
        size_.store(0, std::memory_order_relaxed);
    }

    /// Returns true if ensure_capacity() has not yet been called.
    bool is_not_initialized() const { return table_ == nullptr; }

    /// Number of additions between aging resets and hill-climbing samples
    /// (= 10 × tracked capacity, where tracked capacity is clamped to
    /// MIN_SKETCH_SIZE for accuracy on small caches).
    uint32_t sample_size() const { return sample_size_; }

    /// Estimated frequency of `key`, capped at 15. Returns 0 if uninitialized.
    uint32_t frequency(uint64_t key) const {
        if (is_not_initialized()) return 0;

        uint32_t frequency = 15;
        uint32_t block_hash = spread(static_cast<uint32_t>(key));
        uint32_t counter_hash = rehash(block_hash);
        uint32_t block = (block_hash & block_mask_) << 3;
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t h = counter_hash >> (i << 3);
            uint32_t index = (h >> 1) & 15;
            uint32_t slot = block + (h & 1) + (i << 1);
            uint32_t count = static_cast<uint32_t>(
                (table_[slot].load(std::memory_order_relaxed) >> (index << 2)) & 0xfULL);
            frequency = std::min(frequency, count);
        }
        return frequency;
    }

    /// Increments the frequency of `key` (each of its 4 counters), capping at
    /// 15. Triggers aging (`reset()`) after `sample_size_` successful additions.
    /// Thread-safe via atomic CAS in increment_at().
    void increment(uint64_t key) {
        if (is_not_initialized()) return;

        uint32_t block_hash = spread(static_cast<uint32_t>(key));
        uint32_t counter_hash = rehash(block_hash);
        uint32_t block = (block_hash & block_mask_) << 3;

        uint32_t h0 = counter_hash;
        uint32_t h1 = counter_hash >> 8;
        uint32_t h2 = counter_hash >> 16;
        uint32_t h3 = counter_hash >> 24;

        uint32_t index0 = (h0 >> 1) & 15;
        uint32_t index1 = (h1 >> 1) & 15;
        uint32_t index2 = (h2 >> 1) & 15;
        uint32_t index3 = (h3 >> 1) & 15;

        uint32_t slot0 = block + (h0 & 1);
        uint32_t slot1 = block + (h1 & 1) + 2;
        uint32_t slot2 = block + (h2 & 1) + 4;
        uint32_t slot3 = block + (h3 & 1) + 6;

        // Bitwise-or (not ||) is intentional: all four counters must be
        // inspected even when an earlier one already returned true, so that
        // saturated counters are still probed for the aging bookkeeping. The
        // int casts silence -Wbitwise-instead-of-logical while keeping the
        // non-short-circuit evaluation.
        bool added = static_cast<int>(increment_at(slot0, index0)) |
                     static_cast<int>(increment_at(slot1, index1)) |
                     static_cast<int>(increment_at(slot2, index2)) |
                     static_cast<int>(increment_at(slot3, index3));

        if (added) {
            // Lazy reset trigger: only one thread will see size_ == sample_size_
            // (the others will see sample_size_+k for some k≥1 and skip). That
            // thread runs reset(); the next incrementer re-triggers when size_
            // again reaches sample_size_ (modulo drift, which is harmless —
            // reset is statistical).
            uint32_t s = size_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (s == sample_size_) {
                reset();
            }
        }
    }

private:
    /// Smallest sketch size (256 slots) for acceptable accuracy.
    static constexpr uint32_t MIN_SKETCH_SIZE = 256;
    /// Mask that clears the MSB of each 4-bit counter (for halving).
    static constexpr uint64_t RESET_MASK = 0x7777777777777777ULL;
    /// One-bit selector mask across the 16 counters in a slot (for aging count).
    static constexpr uint64_t ONE_MASK = 0x1111111111111111ULL;

    /// Smallest power of two >= v. Assumes v > 0 and v <= 2^31.
    static uint32_t ceiling_power_of_two(uint32_t v) {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    /// 3-round multiplicative supplemental hash (defends against poor input).
    static uint32_t spread(uint32_t x) {
        x ^= x >> 17;
        x *= 0xed5ad4bbu;
        x ^= x >> 11;
        x *= 0xac4c1b51u;
        x ^= x >> 15;
        return x;
    }

    /// Second hash round for counter selection within a block.
    static uint32_t rehash(uint32_t x) {
        x *= 0x31848babu;
        x ^= x >> 14;
        return x;
    }

    /// Increments counter `j` (0..15) in table slot `i` if not already at 15.
    /// Returns true if the counter was incremented. Thread-safe via CAS retry
    /// (mirrors Caffeine's AtomicLongArray compareAndSwap approach).
    bool increment_at(uint32_t i, uint32_t j) {
        uint32_t offset = j << 2;
        uint64_t mask = 0xfULL << offset;
        uint64_t increment = 1ULL << offset;
        std::atomic<uint64_t>& slot = table_[i];
        uint64_t old = slot.load(std::memory_order_relaxed);
        while (true) {
            if ((old & mask) == mask) return false;  // saturated at 15
            uint64_t newv = old + increment;
            if (slot.compare_exchange_weak(old, newv,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                return true;
            }
            // CAS updated `old` with the current value; retry.
        }
    }

    /// Halves every counter (aging), adjusting the running size estimate.
    /// Called by the single thread that observes size_ == sample_size_. Other
    /// threads may race on individual slots but the halving is idempotent
    /// enough (worst case: a counter that was just incremented gets halved,
    /// losing half a count — negligible for a statistical sketch).
    void reset() {
        uint64_t count = 0;
        for (uint32_t i = 0; i < table_size_; ++i) {
            uint64_t slot = table_[i].load(std::memory_order_relaxed);
            count += popcount(slot & ONE_MASK);
            table_[i].store((slot >> 1) & RESET_MASK, std::memory_order_relaxed);
        }
        uint32_t cur = size_.load(std::memory_order_relaxed);
        size_.store((cur - static_cast<uint32_t>(count >> 2)) >> 1,
                    std::memory_order_relaxed);
    }

    static uint64_t popcount(uint64_t x) {
        // C++20 std::popcount, inlined manually to avoid <bit> header churn.
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
        return (x * 0x0101010101010101ULL) >> 56;
    }

    std::unique_ptr<std::atomic<uint64_t>[]> table_;
    uint32_t table_size_ = 0;
    uint32_t sample_size_ = 0;
    uint32_t block_mask_ = 0;
    std::atomic<uint32_t> size_{0};
};

}  // namespace sextant
