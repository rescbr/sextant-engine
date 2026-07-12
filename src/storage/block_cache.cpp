#include "block_cache.hpp"
#include "sextant/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

namespace sextant {

// --- CacheShard -------------------------------------------------------------

CacheShard::CacheShard(uint32_t capacity_blocks, uint32_t block_size,
                       BlockCache* cache)
    : cache_(cache),
      pool_(block_size),
      capacity_(capacity_blocks),
      max_window_(std::max(1u, capacity_ * 1 / 100)),
      max_protected_((capacity_ - max_window_.load()) * 80 / 100) {
    if (capacity_ > 0) {
        map_.reserve(capacity_ + 1);
        sketch_.ensure_capacity(capacity_);
    }
}

CacheShard::~CacheShard() {
    for (auto& [idx, entry] : map_) {
        pool_.release(entry.data);
        // Bump the global epoch: any L1 cache holding a pointer into this
        // shard's now-freed memory must treat it as stale. We bump once per
        // entry rather than once per shard to match free_entry() semantics.
        g_global_epoch.fetch_add(1, std::memory_order_relaxed);
    }
}

void CacheShard::free_entry(LRUEntry* e) {
    // Adjust the per-segment count before erasing (status tells us which list).
    switch (e->status) {
        case Status::WINDOW:    --size_window_;    break;
        case Status::PROTECTED: --size_protected_; break;
        case Status::PROBATION: /* derived count */ break;
    }
    pool_.release(e->data);
    e->data = nullptr;
    // Signal the thread-local L1 caches that an eviction occurred: any pointer
    // they hold into L2 memory MAY now be stale. Coarse-grained (invalidates
    // the entire L1) but cheap (relaxed atomic) and correct.
    g_global_epoch.fetch_add(1, std::memory_order_relaxed);
    auto it = map_.find(e->block_idx);
    if (it != map_.end()) {
        map_.erase(it);
    }
}

void CacheShard::on_window_hit(LRUEntry* e) {
    window_.move_to_mru(e);
}

void CacheShard::on_probation_hit(LRUEntry* e) {
    // Promote to protected (SLRU A2). If protected is over capacity, demote its
    // LRU back to probation.
    probation_.unlink(e);
    e->status = Status::PROTECTED;
    protected_.push_mru(e);
    ++size_protected_;
    if (size_protected_ > max_protected_) {
        LRUEntry* demote = protected_.lru();
        protected_.unlink(demote);
        demote->status = Status::PROBATION;
        probation_.push_mru(demote);
        --size_protected_;
    }
}

void CacheShard::on_protected_hit(LRUEntry* e) {
    protected_.move_to_mru(e);
}

// --- lazy adaptation to cache-level window changes ---------------------------
//
// The cache's hill-climber writes new atomic limits (max_window_ /
// max_protected_) without holding this shard's lock. maybe_adapt(), called
// under mu_ at the top of lookup()/insert(), reconciles the live entry lists
// with the current limits: demoting over-capacity protected entries back to
// probation, and spilling an oversized window through TinyLFU admission.

void CacheShard::maybe_adapt() {
    uint32_t cur_max_protected = max_protected_.load(std::memory_order_relaxed);
    uint32_t cur_max_window = max_window_.load(std::memory_order_relaxed);

    // Demote over-capacity protected entries back to probation (mirrors
    // Caffeine's demoteFromMain()).
    while (size_protected_ > cur_max_protected) {
        LRUEntry* e = protected_.lru();
        if (e == nullptr) break;
        protected_.unlink(e);
        e->status = Status::PROBATION;
        probation_.push_mru(e);
        --size_protected_;
    }

    // If the window shrank, spill its overflow into probation via admission.
    while (size_window_ > cur_max_window) {
        admit_one_from_window();
    }
}

void CacheShard::admit_one_from_window() {
    // Caffeine's evict(): if the window exceeds its quota, move its LRU
    // (candidate) into probation, then if the cache is over capacity run
    // TinyLFU admission between candidate and probation's LRU (victim).
    if (size_window_ <= max_window_) return;

    LRUEntry* candidate = window_.lru();
    window_.unlink(candidate);
    --size_window_;

    candidate->status = Status::PROBATION;
    probation_.push_mru(candidate);

    if (static_cast<uint32_t>(map_.size()) > capacity_) {
        LRUEntry* victim = probation_.lru();
        if (victim == nullptr || victim == candidate) {
            // No distinct victim to compare against: the probation space is
            // empty or holds only this candidate (the cache just reached
            // capacity). Admit the candidate so the main space fills up and
            // the sketch accumulates frequency data. Rejecting here is the
            // root cause of the 0% admission rate seen in profiling.
            ++evictions_window_;
            return;
        }
        uint32_t candidate_freq = sketch_.frequency(candidate->block_idx);
        uint32_t victim_freq = sketch_.frequency(victim->block_idx);
        bool admit;
        if (candidate_freq > victim_freq) {
            admit = true;
        } else if (candidate_freq >= 6 && (hc_rng() & 127u) == 0) {
            // Anti-hash-DoS / anti-starvation: on a tie (or near-tie) where
            // the candidate already has meaningful frequency, admit with small
            // probability (1/128) so distinct hot keys are not permanently
            // blocked by an unlucky sketch collision. Mirrors Caffeine's
            // random tie-break in admit().
            admit = true;
        } else {
            admit = false;
        }
        if (admit) {
            // Admit candidate (already in probation), evict the victim.
            probation_.unlink(victim);
            free_entry(victim);
            ++evictions_window_;
            ++evictions_admitted_;
        } else {
            // Reject candidate: free its data, keep the victim.
            probation_.unlink(candidate);
            free_entry(candidate);
            ++evictions_window_;
            ++evictions_rejected_;
        }
    } else {
        ++evictions_window_;
    }
}

uint8_t* CacheShard::lookup(uint64_t block_idx) {
    maybe_adapt();
    auto it = map_.find(block_idx);
    if (it == map_.end()) {
        sketch_.increment(block_idx);
        ++misses_;
        if (cache_) cache_->record_access(/*hit=*/false);
        return nullptr;
    }

    sketch_.increment(block_idx);
    LRUEntry* e = &it->second;
    switch (e->status) {
        case Status::WINDOW:
            on_window_hit(e);
            ++hits_window_;
            break;
        case Status::PROBATION:
            on_probation_hit(e);
            ++hits_probation_;
            break;
        case Status::PROTECTED:
            on_protected_hit(e);
            ++hits_protected_;
            break;
    }
    if (cache_) cache_->record_access(/*hit=*/true);
    return e->data;
}

uint8_t* CacheShard::insert(uint64_t block_idx, const uint8_t* data,
                           uint32_t size) {
    maybe_adapt();
    // Already present: update data, treat as a hit (promote per status).
    auto it = map_.find(block_idx);
    if (it != map_.end()) {
        std::memcpy(it->second.data, data, size);
        LRUEntry* e = &it->second;
        sketch_.increment(block_idx);
        switch (e->status) {
            case Status::WINDOW:
                on_window_hit(e);
                ++hits_window_;
                break;
            case Status::PROBATION:
                on_probation_hit(e);
                ++hits_probation_;
                break;
            case Status::PROTECTED:
                on_protected_hit(e);
                ++hits_protected_;
                break;
        }
        if (cache_) cache_->record_access(/*hit=*/true);
        return e->data;
    }

    // capacity_ == 0: shard holds nothing.
    if (capacity_ == 0) {
        if (cache_) cache_->record_access(/*hit=*/false);
        return nullptr;
    }

    // New entry: acquire a recycled buffer from this shard's own pool.
    uint8_t* buf = pool_.acquire();
    std::memcpy(buf, data, size);

    auto [new_it, inserted] = map_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(block_idx),
        std::forward_as_tuple());
    LRUEntry& entry = new_it->second;
    entry.block_idx = block_idx;
    entry.data = buf;
    entry.dirty = false;
    entry.status = Status::WINDOW;

    window_.push_mru(&entry);
    ++size_window_;
    sketch_.increment(block_idx);
    if (cache_) cache_->record_access(/*hit=*/false);

    // If the window exceeded its quota, run admission (may evict this entry if
    // it loses the frequency contest — that's the scan-resistance in action).
    admit_one_from_window();

    // The requested block may have been evicted. Return its pointer only if it
    // is still resident.
    auto after = map_.find(block_idx);
    if (after == map_.end()) return nullptr;
    return after->second.data;
}

void CacheShard::mark_dirty(uint64_t block_idx) {
    auto it = map_.find(block_idx);
    if (it != map_.end()) {
        it->second.dirty = true;
    }
}

// --- BlockCache ---------------------------------------------------------

BlockCache::BlockCache(uint32_t num_shards,
                                    uint32_t blocks_per_shard,
                                    uint32_t block_size)
    : block_size_(block_size),
      hc_per_shard_capacity_(blocks_per_shard),
      hc_capacity_(static_cast<uint64_t>(blocks_per_shard) * num_shards),
      hc_sample_size_(10u * hc_capacity_),
      hc_step_size_(std::max(2.0, hc_per_shard_capacity_ * 0.0625)) {
    shards_.reserve(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        shards_.push_back(
            std::make_unique<CacheShard>(blocks_per_shard, block_size, this));
    }
}

BlockCache::~BlockCache() = default;

CacheStats BlockCache::stats() const {
    CacheStats s;
    for (const auto& shard : shards_) {
        s.hits_window += shard->hits_window();
        s.hits_probation += shard->hits_probation();
        s.hits_protected += shard->hits_protected();
        s.misses += shard->misses();
        s.evictions_window += shard->evictions_window();
        s.evictions_admitted += shard->evictions_admitted();
        s.evictions_rejected += shard->evictions_rejected();
    }
    return s;
}

// --- cache-level hill-climbing (ported from Caffeine's BoundedLocalCache) ----
//
// The hill-climber now lives on the cache, aggregating hit/miss samples across
// ALL shards. This gives a far less noisy hit-rate signal than the per-shard
// version (each shard saw only 1/N of the access stream). Every
// hc_sample_size_ (= 10 × total capacity) accesses the hit rate is sampled and
// compared to the previous sample. If it improved, keep walking the window /
// main split in the same direction; if it worsened, reverse. The step decays
// each round so the optimizer homes in on a local optimum. A large hit-rate
// swing (>= 5%) resets the step, assuming the workload shifted.
//
// climb() does NOT acquire any shard locks: it only writes each shard's atomic
// max_window_ / max_protected_ limits. Each shard lazily reconciles its entry
// lists on its next lookup()/insert() via maybe_adapt() (under its own lock),
// so there is no cross-shard lock-acquisition deadlock risk.

void BlockCache::record_access(bool hit) {
    if (hit) {
        hc_hits_in_sample_.fetch_add(1, std::memory_order_relaxed);
    } else {
        hc_misses_in_sample_.fetch_add(1, std::memory_order_relaxed);
    }
    maybe_climb();
}

void BlockCache::maybe_climb() {
    uint64_t h = hc_hits_in_sample_.load(std::memory_order_relaxed);
    uint64_t m = hc_misses_in_sample_.load(std::memory_order_relaxed);
    if (h + m < hc_sample_size_) return;


    // Only one thread should run climb() at a time. Losers bail out; the next
    // sample period will re-trigger if needed.
    bool expected = false;
    if (!hc_climbing_.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel)) return;
    climb();
    hc_climbing_.store(false, std::memory_order_release);
}

void BlockCache::climb() {
    if (hc_capacity_ == 0) return;

    uint64_t h = hc_hits_in_sample_.exchange(0, std::memory_order_relaxed);
    uint64_t m = hc_misses_in_sample_.exchange(0, std::memory_order_relaxed);
    uint64_t total = h + m;
    if (total == 0) return;

    // --- determineAdjustment() ---
    double hit_rate = static_cast<double>(h) / static_cast<double>(total);
    double hit_rate_change = hit_rate - hc_prev_hit_rate_;
    double amount = (hit_rate_change >= 0.0) ? hc_step_size_ : -hc_step_size_;

    double next_step_size;
    if (std::abs(hit_rate_change) >= 0.05) {
        // Workload shifted: reset the step to its initial magnitude (keeping
        // the sign of `amount` so the optimizer continues in the improving
        // direction).
        double initial = std::max(hc_per_shard_capacity_ * 0.0625, 2.0);
        next_step_size = (amount >= 0.0) ? initial : -initial;
    } else {
        // Decay the step magnitude by 2% each round (regardless of direction).
        next_step_size = 0.98 * amount;
    }

    hc_prev_hit_rate_ = hit_rate;
    hc_step_size_ = next_step_size;

    // --- apply the adjustment: compute new uniform per-shard limits ---
    // max_window_ is a per-shard quantity, so the step and clamp are in
    // per-shard capacity units (hc_per_shard_capacity_), keeping the units
    // consistent. All shards receive the same window/protected limits.
    uint32_t cur_window = shards_[0]->max_window();
    int64_t new_window = static_cast<int64_t>(cur_window) +
                         static_cast<int64_t>(amount);
    int64_t cap = static_cast<int64_t>(hc_per_shard_capacity_);
    new_window = std::clamp(new_window, static_cast<int64_t>(1), cap - 2);
    uint32_t max_window = static_cast<uint32_t>(new_window);
    uint32_t max_protected = (hc_per_shard_capacity_ - max_window) * 80 / 100;

    for (auto& shard : shards_) {
        shard->max_window_.store(max_window, std::memory_order_relaxed);
        shard->max_protected_.store(max_protected, std::memory_order_relaxed);
    }
}

}  // namespace sextant
