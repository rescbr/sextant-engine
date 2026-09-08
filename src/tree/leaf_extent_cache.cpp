#include "leaf_extent_cache.hpp"

/// @file leaf_extent_cache.cpp — see header for the design notes.

#include "storage/frequency_sketch.hpp"

#include <sextant/error.hpp>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace sextant::tree {

// --- Private types (declared in the header, defined here) ---------------------

enum class LeafExtentCacheStatus : uint8_t { WINDOW, PROBATION, PROTECTED };

struct LeafExtentCache::Entry {
    PageId page = 0;
    uint32_t size = 0;          // bytes (= pages * kPageSize)
    LeafExtentCacheStatus status = LeafExtentCacheStatus::WINDOW;
    Entry* prev = nullptr;      // LRU list pointers (valid while linked)
    Entry* next = nullptr;
    std::atomic<uint32_t> refs{0};
    std::atomic<bool> retired{false};   // evicted / transient / invalidated
    std::atomic<bool> deleted{false};   // single-deleter claim flag
    uint8_t* data = nullptr;    // buffer from the owning shard's pool
    uint64_t buf_cap = 0;       // pool capacity of `data`
    struct Shard* owner = nullptr;
};

/// Sentinel-free intrusive lists: plain head/tail pointers (entries are
/// heap-stable, list membership guarded by the shard lock).
struct LeafExtentCache::List {
    Entry* head = nullptr;  // MRU
    Entry* tail = nullptr;  // LRU
    bool empty() const { return head == nullptr; }
    static void unlink(Entry* e, List& l);
    void push_mru(Entry* e);
    static void move_to_mru(Entry* e, List& l);
};

/// Buffer recycling: glibc services ~MB allocations via mmap/munmap, so an
/// alloc-per-miss policy page-faults the whole corpus per pass (measured:
/// +230s CPU on a 1000-query cold pass). Freed buffers are pooled per shard
/// (bounded by the shard's capacity) and reused by any later miss that fits
/// — same reasoning as the graph path's BlockBufferPool. All pool state is
/// guarded by mu.
struct LeafExtentCache::Shard {
    Mutex mu;
    std::unordered_map<PageId, Entry*> map;
    List window, probation, protected_;
    uint64_t bytes_window = 0;
    uint64_t bytes_probation = 0;
    uint64_t bytes_protected = 0;
    uint64_t max_window_bytes = 0;
    uint64_t max_main_bytes = 0;    // probation + protected budget
    uint64_t max_probation_bytes = 0;
    uint64_t max_protected_bytes = 0;
    FrequencySketch* sketch = nullptr;  // owned by the cache
    std::vector<std::pair<uint8_t*, uint64_t>> buf_pool;
    uint64_t buf_pool_bytes = 0;
    uint64_t capacity_bytes = 0;
    ~Shard();

    uint8_t* acquire_buf(uint64_t size);          // caller holds mu
    void release_buf(uint8_t* buf, uint64_t cap); // caller holds mu
};

// --- Entry lifetime ---------------------------------------------------------
//
// An Entry is owned by whoever transitions it to `retired` AND observes
// refs==0 — via maybe_delete()/maybe_delete_locked(), idempotent through the
// `deleted` CAS flag. Retiring happens under the shard lock (eviction /
// invalidation / transient creation); unpin() runs lock-free and re-checks
// `retired` after its decrement (acquire on the flag pairs with the releasing
// store under the lock), so the deleter is unique no matter how eviction and
// the last unpin interleave. Disposal returns the entry's buffer to its
// shard's pool under the shard lock: use maybe_delete_locked() when the
// caller already holds it, maybe_delete() otherwise.

void LeafExtentCache::maybe_delete_locked(Entry* e) {
    if (e->refs.load(std::memory_order_acquire) != 0) return;
    bool expected = false;
    if (e->deleted.compare_exchange_strong(expected, true,
                                           std::memory_order_acq_rel)) {
        e->owner->release_buf(e->data, e->buf_cap);
        delete e;
    }
}

void LeafExtentCache::maybe_delete(Entry* e) {
    if (e->refs.load(std::memory_order_acquire) != 0) return;
    bool expected = false;
    if (!e->deleted.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel)) {
        return;
    }
    ScopedWriteLock lock(e->owner->mu);
    e->owner->release_buf(e->data, e->buf_cap);
    delete e;
}

void LeafExtentCache::drop_filled(Entry* e) {
    e->retired.store(true, std::memory_order_release);
    maybe_delete_locked(e);
}

void LeafExtentCache::drop_entry(Entry* e) {
    e->retired.store(true, std::memory_order_release);
    maybe_delete_locked(e);  // all drop_entry call sites hold the shard lock
}

// --- Buffer pool (caller holds the shard lock) --------------------------------

uint8_t* LeafExtentCache::Shard::acquire_buf(uint64_t size) {
    // First-fit: buffers are near-uniform (leaf extents), so the first cap
    // >= size is almost always an exact match.
    for (size_t i = 0; i < buf_pool.size(); ++i) {
        if (buf_pool[i].second >= size) {
            uint8_t* buf = buf_pool[i].first;
            buf_pool_bytes -= buf_pool[i].second;
            buf_pool[i] = buf_pool.back();
            buf_pool.pop_back();
            return buf;
        }
    }
    return new uint8_t[size];
}

void LeafExtentCache::Shard::release_buf(uint8_t* buf, uint64_t cap) {
    // The pool shares the shard's capacity budget with LIVE entry data —
    // pooling up to `capacity_bytes` ON TOP of live data doubles the real
    // footprint (measured: anon-rss 4.18 GB with cache-mb=2048 → cgroup
    // OOM; kernel log CONSTRAINT_MEMCG, 2026-09-08). Only pool what fits
    // next to the live bytes.
    const uint64_t live = bytes_window + bytes_probation + bytes_protected;
    if (buf_pool_bytes + cap + live <= capacity_bytes) {
        buf_pool.emplace_back(buf, cap);
        buf_pool_bytes += cap;
    } else {
        delete[] buf;
    }
}

LeafExtentCache::Shard::~Shard() {
    // Engine shutdown: live entries (refs ignored — no searchers remain) and
    // pooled buffers all go back to the heap.
    for (auto& [page, e] : map) {
        (void)page;
        delete[] e->data;
        delete e;
    }
    for (auto& [buf, cap] : buf_pool) delete[] buf;
}

// --- Intrusive lists (all mutation under the shard lock) ---------------------

void LeafExtentCache::List::unlink(Entry* e, List& l) {
    if (l.head == e) l.head = e->next;
    if (l.tail == e) l.tail = e->prev;
    if (e->prev) e->prev->next = e->next;
    if (e->next) e->next->prev = e->prev;
    e->prev = e->next = nullptr;
}

void LeafExtentCache::List::push_mru(Entry* e) {
    e->prev = nullptr;
    e->next = head;
    if (head) head->prev = e;
    head = e;
    if (!tail) tail = e;
}

void LeafExtentCache::List::move_to_mru(Entry* e, List& l) {
    if (l.head == e) return;
    unlink(e, l);
    l.push_mru(e);
}

// --- Construction ------------------------------------------------------------

LeafExtentCache::LeafExtentCache(uint64_t capacity_bytes, uint32_t num_shards,
                                 int fd, uint32_t window_pct)
    : capacity_bytes_(capacity_bytes), fd_(fd) {
    if (num_shards == 0) num_shards = 1;
    if (window_pct == 0) window_pct = 1;
    if (window_pct > 90) window_pct = 90;
    shards_.reserve(num_shards);
    sketches_ = std::make_unique<FrequencySketch[]>(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        auto s = std::make_unique<Shard>();
        // Caffeine's default split: 1% window, main split 80/20
        // protected/probation, per shard.
        const uint64_t cap = capacity_bytes / num_shards;
        s->capacity_bytes = cap;
        s->max_window_bytes = cap * window_pct / 100;
        s->max_main_bytes = cap - s->max_window_bytes;
        s->max_protected_bytes = s->max_main_bytes * 80 / 100;
        s->max_probation_bytes = s->max_main_bytes - s->max_protected_bytes;
        s->sketch = &sketches_[i];
        shards_.push_back(std::move(s));
    }
}

LeafExtentCache::~LeafExtentCache() = default;

void LeafExtentCache::set_expected_entries(uint32_t n) {
    const uint32_t per = n / static_cast<uint32_t>(shards_.size()) + 1;
    for (auto& s : shards_) s->sketch->ensure_capacity(per);
}

LeafExtentCache::Shard& LeafExtentCache::shard_for(PageId page) {
    // Leaf start pages are multiples of extent sizes (hundreds of pages), so
    // the low bits carry little entropy — multiplicative-hash down.
    const uint64_t h = page * 0x9E3779B97F4A7C15ull;
    return *shards_[static_cast<size_t>(h >> 32) % shards_.size()];
}

// --- Fill path ----------------------------------------------------------------

LeafExtentCache::Entry* LeafExtentCache::new_entry(Shard& s, PageId page,
                                                   uint32_t pages) {
    const uint64_t size = entry_size(pages);
    auto* e = new Entry();
    e->page = page;
    e->size = static_cast<uint32_t>(size);
    e->owner = &s;
    {
        ScopedWriteLock lock(s.mu);
        e->data = s.acquire_buf(size);
    }
    e->buf_cap = size;  // pool buffers are reused at their original capacity
    const off_t off = static_cast<off_t>(page) * kPageSize;
    size_t done = 0;
    while (done < size) {
        const ssize_t r = ::pread(fd_, e->data + done, size - done,
                                  off + static_cast<off_t>(done));
        if (r < 0) {
            if (errno == EINTR) continue;
            {
                ScopedWriteLock lock(s.mu);
                s.release_buf(e->data, e->buf_cap);
            }
            delete e;
            throw Error(ErrorCode::IoError,
                        "LeafExtentCache: pread failed at page " +
                            std::to_string(page) + ": " + std::strerror(errno));
        }
        if (r == 0) {
            {
                ScopedWriteLock lock(s.mu);
                s.release_buf(e->data, e->buf_cap);
            }
            delete e;
            throw Error(ErrorCode::IoError,
                        "LeafExtentCache: short pread at page " +
                            std::to_string(page));
        }
        done += static_cast<size_t>(r);
    }
    return e;
}

// --- pin / unpin / contains ---------------------------------------------------

bool LeafExtentCache::contains(PageId page) {
    Shard& s = shard_for(page);
    ScopedReadLock lock(s.mu);
    return s.map.find(page) != s.map.end();
}

const uint8_t* LeafExtentCache::pin(PageId page, uint32_t pages, Handle& h,
                                    const uint8_t* fallback, bool* was_hit,
                                    uint64_t* filled_bytes) {
    h.entry = nullptr;
    if (was_hit) *was_hit = false;
    if (filled_bytes) *filled_bytes = 0;
    Shard& s = shard_for(page);

    // Fast path: hit (write lock — hits mutate LRU order + SLRU promotion).
    {
        ScopedWriteLock lock(s.mu);
        auto it = s.map.find(page);
        if (it != s.map.end()) {
            Entry* e = it->second;
            e->refs.fetch_add(1, std::memory_order_acq_rel);
            s.sketch->increment(page);
            hits_.fetch_add(1, std::memory_order_relaxed);
            if (was_hit) *was_hit = true;
            switch (e->status) {
            case LeafExtentCacheStatus::WINDOW:
                List::move_to_mru(e, s.window); break;
            case LeafExtentCacheStatus::PROBATION:
                List::unlink(e, s.probation);
                s.bytes_probation -= e->size;
                e->status = LeafExtentCacheStatus::PROTECTED;
                s.protected_.push_mru(e);
                s.bytes_protected += e->size;
                // Protected overflow demotes to probation MRU.
                while (s.bytes_protected > s.max_protected_bytes &&
                       !s.protected_.empty()) {
                    Entry* d = s.protected_.tail;
                    List::unlink(d, s.protected_);
                    s.bytes_protected -= d->size;
                    d->status = LeafExtentCacheStatus::PROBATION;
                    s.probation.push_mru(d);
                    s.bytes_probation += d->size;
                }
                // Probation overflow (from the demotion) evicts its LRU —
                // never a PINNED entry: evicting pinned data frees nothing,
                // creates an un-freeable transient, and the pileup OOMs
                // budgeted (partial-residency) runs. Skip to the next LRU;
                // if none is unpinned, leave the list oversized.
                while (s.bytes_probation > s.max_probation_bytes) {
                    Entry* v = s.probation.tail;
                    while (v != nullptr &&
                           v->refs.load(std::memory_order_relaxed) > 0) {
                        v = v->prev;
                    }
                    if (v == nullptr) break;
                    List::unlink(v, s.probation);
                    s.bytes_probation -= v->size;
                    s.map.erase(v->page);
                    evictions_.fetch_add(1, std::memory_order_relaxed);
                    drop_entry(v);
                }
                break;
            case LeafExtentCacheStatus::PROTECTED:
                List::move_to_mru(e, s.protected_);
                break;
            }
            h.entry = e;
            return e->data;
        }
    }

    // Miss: fill OUTSIDE the lock (pread latency must not serialize a shard).
    // NOTE: a fill that is later refused is NOT wasted I/O — it warmed the
    // page cache that the fallback (mmap) read will hit.
    misses_.fetch_add(1, std::memory_order_relaxed);
    Entry* e = new_entry(s, page, pages);
    bytes_filled_.fetch_add(e->size, std::memory_order_relaxed);
    if (filled_bytes) *filled_bytes = e->size;

    {
        ScopedWriteLock lock(s.mu);
        // Duplicate-insert race: another thread filled the same page first —
        // keep the incumbent, serve ourselves from the fallback (no private
        // transient).
        auto it = s.map.find(page);
        if (it != s.map.end()) {
            Entry* winner = it->second;
            winner->refs.fetch_add(1, std::memory_order_acq_rel);
            s.sketch->increment(page);
            h.entry = winner;
            drop_filled(e);  // our buffer never becomes visible
            return winner->data;
        }
        s.sketch->increment(page);

        // Too big to ever cache → serve from the fallback.
        if (e->size > s.max_window_bytes + s.max_main_bytes) {
            rejections_.fetch_add(1, std::memory_order_relaxed);
            h.entry = nullptr;
            drop_filled(e);
            return fallback;
        }

        // The caller holds a pin from here on: refs must be 1 BEFORE the
        // entry becomes reachable (map/lists), or a concurrent insert could
        // evict (and free) it between insertion and return.
        e->refs.store(1, std::memory_order_relaxed);
        h.entry = e;  // null handle = caller never unpins = permanent leak
        // Enter at window MRU, then run W-TinyLFU admission until the window
        // is back under budget. Candidate = window LRU, SKIPPING pinned
        // entries (rejecting an in-use entry would strand its buffer as a
        // transient until that query ends — the OOM mechanism). The loop
        // also never picks the just-inserted entry itself (`cand != e`).
        e->status = LeafExtentCacheStatus::WINDOW;
        s.window.push_mru(e);
        s.bytes_window += e->size;
        s.map.emplace(page, e);

        while (s.bytes_window > s.max_window_bytes) {
            Entry* cand = s.window.tail;
            while (cand != nullptr && cand != e &&
                   cand->refs.load(std::memory_order_relaxed) > 0) {
                cand = cand->prev;
            }
            if (cand == nullptr || cand == e) break;  // nothing movable
            List::unlink(cand, s.window);
            s.bytes_window -= cand->size;

            const bool main_has_room =
                s.bytes_probation + s.bytes_protected + cand->size <=
                s.max_main_bytes;
            if (main_has_room) {
                cand->status = LeafExtentCacheStatus::PROBATION;
                s.probation.push_mru(cand);
                s.bytes_probation += cand->size;
                continue;
            }
            // Main full: TinyLFU — candidate vs probation LRU on frequency.
            // STRICTLY greater (Caffeine semantics): on a tie the incumbent
            // stays (4-bit counters make ties dominate on a cold/aged
            // sketch; `>=` degenerated to FIFO churn). Victim = probation
            // LRU, SKIPPING pinned entries; none unpinned → reject candidate.
            Entry* victim = s.probation.tail;
            while (victim != nullptr &&
                   victim->refs.load(std::memory_order_relaxed) > 0) {
                victim = victim->prev;
            }
            if (victim != nullptr &&
                s.sketch->frequency(cand->page) >
                    s.sketch->frequency(victim->page)) {
                List::unlink(victim, s.probation);
                s.bytes_probation -= victim->size;
                s.map.erase(victim->page);
                evictions_.fetch_add(1, std::memory_order_relaxed);
                drop_entry(victim);
                cand->status = LeafExtentCacheStatus::PROBATION;
                s.probation.push_mru(cand);
                s.bytes_probation += cand->size;
            } else {
                // Rejected (unpinned, and never this caller's `e` — the
                // cand selection guarantees both): remove from the cache.
                // It frees immediately on drop (refs==0); no transient.
                s.map.erase(cand->page);
                rejections_.fetch_add(1, std::memory_order_relaxed);
                drop_entry(cand);
            }
        }
    }
    return e->data;
}

void LeafExtentCache::unpin(Handle& h) {
    Entry* e = static_cast<Entry*>(h.entry);
    h.entry = nullptr;
    if (!e) return;
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
        e->retired.load(std::memory_order_acquire)) {
        maybe_delete(e);
    }
}

void LeafExtentCache::invalidate_all() {
    for (auto& sp : shards_) {
        Shard& s = *sp;
        ScopedWriteLock lock(s.mu);
        for (auto& [page, e] : s.map) {
            (void)page;
            switch (e->status) {
            case LeafExtentCacheStatus::WINDOW:
                List::unlink(e, s.window); break;
            case LeafExtentCacheStatus::PROBATION:
                List::unlink(e, s.probation); break;
            case LeafExtentCacheStatus::PROTECTED:
                List::unlink(e, s.protected_); break;
            }
            drop_entry(e);
        }
        s.map.clear();
        s.bytes_window = s.bytes_probation = s.bytes_protected = 0;
    }
}

LeafExtentCache::Stats LeafExtentCache::stats() const {
    Stats st;
    st.hits = hits_.load(std::memory_order_relaxed);
    st.misses = misses_.load(std::memory_order_relaxed);
    st.bytes_filled = bytes_filled_.load(std::memory_order_relaxed);
    st.evictions = evictions_.load(std::memory_order_relaxed);
    st.rejections = rejections_.load(std::memory_order_relaxed);
    return st;
}

}  // namespace sextant::tree
