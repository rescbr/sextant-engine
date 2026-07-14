// NodeStore backends.
//
// FlatNodeStore is trivial pointer arithmetic. PagedNodeStore drives the LRU
// cache + DirectFile path: each pin is a shard-locked cache lookup with a
// disk read on miss. The block size is kBlockSize (256 KB); node and code
// records are packed contiguously within their respective files.

#include "node_store.hpp"

#include "sextant/error.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace sextant {

namespace {

struct alignas(kDiskAlign) AlignedStaging {
    static constexpr size_t kCapacity =
        kBlockSize * PagedNodeStore::kBlocksPerRead;
    uint8_t data[kCapacity];
};

thread_local AlignedStaging g_staging;

// Per-thread L1 caches, keyed by store instance ID (NOT by pointer). Using a
// monotonic ID avoids the address-reuse hazard: a store destroyed on thread A
// leaves a stale entry in thread A's map; if thread B later reuses that freed
// address for a new store, the new store's distinct ID prevents matching the
// stale entry.
thread_local std::unordered_map<uint64_t, TLBlockCache> g_tl_caches;

}  // namespace

// Monotonic instance-ID source for per-thread L1 keying.
std::atomic<uint64_t> PagedNodeStore::next_instance_id_{0};

// ===========================================================================
// FlatNodeStore
// ===========================================================================

FlatNodeStore::FlatNodeStore(uint8_t* nodes, const uint8_t* codes,
                              uint32_t node_size, uint32_t code_size)
    : nodes_(nodes),
      codes_(codes),
      node_size_(node_size),
      code_size_(code_size) {}

PinResult FlatNodeStore::pin_node(uint32_t id) {
    return {nodes_ + static_cast<size_t>(id) * node_size_, false};
}

PinResult FlatNodeStore::pin_code(uint32_t id) {
    return {codes_ + static_cast<size_t>(id) * code_size_, false};
}

uint8_t* FlatNodeStore::mutable_node(uint32_t id) {
    return nodes_ + static_cast<size_t>(id) * node_size_;
}

uint8_t* FlatNodeStore::mutable_code(uint32_t id) {
    return const_cast<uint8_t*>(codes_) +
           static_cast<size_t>(id) * code_size_;
}

// ===========================================================================
// PagedNodeStore
// ===========================================================================

PagedNodeStore::PagedNodeStore(const std::string& graph_path,
                               const std::string& codes_path,
                               uint32_t node_size, uint32_t code_size,
                               uint32_t num_shards,
                               uint64_t cache_size_bytes)
    : graph_file_(graph_path, /*create=*/false),
      codes_file_(codes_path, /*create=*/false),
      node_size_(node_size),
      code_size_(code_size),
      nodes_per_block_(std::max(1u, kBlockSize / std::max(1u, node_size))),
      codes_per_block_(std::max(1u, kBlockSize / std::max(1u, static_cast<uint32_t>(code_size)))),
      graph_block_size_(nodes_per_block_ * node_size),
      codes_block_size_(codes_per_block_ * static_cast<uint32_t>(code_size)),
      cache_(num_shards,
             std::max(1u, static_cast<uint32_t>(
                               cache_size_bytes / kBlockSize / num_shards)),
             kBlockSize),
      instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)) {}

PagedNodeStore::~PagedNodeStore() {
    // Drop this store's entry from the current thread's L1 map. Entries on
    // other threads' maps become stale but are harmless: they're keyed by
    // instance_id_, which is never reused, so they can never be matched by a
    // future store. They're freed when those threads exit.
    g_tl_caches.erase(instance_id_);
}

TLBlockCache& PagedNodeStore::tl_cache() const {
    return g_tl_caches[instance_id_];
}

PinResult PagedNodeStore::get_node_block(uint64_t block_idx) {
    return batched_read(graph_file_, graph_block_size_, block_idx,
                        /*key_base=*/block_idx, graph_reads_);
}

PinResult PagedNodeStore::get_code_block(uint64_t block_idx) {
    // Code-block keys carry the high bit so they never collide with graph
    // blocks in the shared cache.
    const uint64_t key = block_idx | kCodeKeyBit;
    return batched_read(codes_file_, codes_block_size_, block_idx,
                        /*key_base=*/key, code_reads_);
}

PinResult PagedNodeStore::batched_read(DirectFile& file, uint32_t block_size,
                                          uint64_t block_idx, uint64_t key_base,
                                          std::atomic<uint64_t>& counter) {
    // L1 check (thread-local, per-store, lock-free). The epoch is read once
    // here; if an eviction occurs during this call the epoch changes and the
    // L1 entry (if any) is treated as stale.
    TLBlockCache& l1 = tl_cache();
    const uint64_t epoch = cache_.shard_epoch(key_base);
    if (const uint8_t* hit = l1.lookup(key_base, epoch)) {
        ++l1.hits;
        tl_hits_.fetch_add(1, std::memory_order_relaxed);
        return {hit, false};  // L1 hit — no I/O, no L2 lock
    }
    ++l1.misses;
    tl_misses_.fetch_add(1, std::memory_order_relaxed);

    // Step 1–2: check the requested block's shard under its write lock.
    {
        CacheShard& shard = cache_.shard(key_base);
        ScopedWriteLock lock(shard.mutex());
        uint8_t* hit = shard.lookup(key_base);
        if (hit) {
            // L2 hit — cache the L2-owned pointer (NO COPY). Re-read the
            // epoch AFTER the L2 op to capture the latest value (the lookup
            // itself cannot evict, but a concurrent shard op on another
            // thread could have).
            const uint64_t post_epoch =
                cache_.shard_epoch(key_base);
            l1.insert(key_base, hit, post_epoch);
            return {hit, false};  // LRU hit — served from RAM
        }
        // Step 3: lock released here (scope exit) — never hold it across I/O.
    }

    // Step 4: clamp the read so we don't run past EOF.
    const uint64_t header = sizeof(SidecarHeader);
    const uint64_t payload = file.size() > header ? file.size() - header : 0;
    const uint64_t total_blocks = payload / block_size;
    const uint32_t n_to_read = static_cast<uint32_t>(
        std::min<uint64_t>(kBlocksPerRead,
                           total_blocks > block_idx ? total_blocks - block_idx
                                                    : 1));
    const uint32_t read_count = std::max(1u, n_to_read);

    // Step 5: one pread of the whole batch into the thread-local staging area.
    const uint64_t off = header + block_idx * block_size;
    file.pread_aligned(g_staging.data,
                       static_cast<size_t>(read_count) * block_size, off);

    // Step 6: one syscall → one counter tick, regardless of blocks fetched.
    counter.fetch_add(1, std::memory_order_relaxed);

    // Step 7: insert each fetched block into its own shard. Locks are held
    // briefly and one at a time (ascending i → no deadlock). Skip blocks
    // another thread already inserted meanwhile.
    for (uint32_t i = 0; i < read_count; ++i) {
        const uint64_t key_i = key_base + i;
        CacheShard& shard = cache_.shard(key_i);
        ScopedWriteLock lock(shard.mutex());
        if (shard.lookup(key_i) != nullptr) {
            continue;  // raced — another thread cached it first
        }
        shard.insert(key_i, g_staging.data + i * block_size, block_size);
    }

    // Step 8: re-acquire the requested block's shard and return its pointer.
    {
        CacheShard& shard = cache_.shard(key_base);
        ScopedWriteLock lock(shard.mutex());
        uint8_t* stored = shard.lookup(key_base);
        // stored is non-null: we just inserted it in the loop above (or a
        // racing thread did). Fallback to a single-block read guards against
        // a pathological eviction between insert and lookup.
        if (stored != nullptr) {
            // Cache the L2-OWNED pointer (`stored`), NOT g_staging.data (which
            // is a thread-local scratch buffer overwritten on the next miss).
            const uint64_t post_epoch =
                cache_.shard_epoch(key_base);
            l1.insert(key_base, stored, post_epoch);
            return {stored, true};
        }
        file.pread_aligned(g_staging.data, block_size, off);
        counter.fetch_add(1, std::memory_order_relaxed);
        stored = shard.insert(key_base, g_staging.data, block_size);
        if (stored != nullptr) {
            const uint64_t post_epoch =
                cache_.shard_epoch(key_base);
            l1.insert(key_base, stored, post_epoch);
        }
        return {stored, true};
    }
}

PinResult PagedNodeStore::pin_node(uint32_t id) {
    const uint64_t block_idx = id / nodes_per_block_;
    const uint32_t off_in_block = (id % nodes_per_block_) * node_size_;
    PinResult pr = get_node_block(block_idx);
    return {pr.data + off_in_block, pr.from_ssd};
}

void PagedNodeStore::unpin_node(uint32_t) {}

PinResult PagedNodeStore::pin_code(uint32_t id) {
    const uint64_t block_idx = id / codes_per_block_;
    const uint32_t off_in_block = (id % codes_per_block_) * code_size_;
    PinResult pr = get_code_block(block_idx);
    return {pr.data + off_in_block, pr.from_ssd};
}

void PagedNodeStore::unpin_code(uint32_t) {}

uint8_t* PagedNodeStore::mutable_node(uint32_t) {
    throw Error(ErrorCode::NotImplemented,
                "PagedNodeStore does not support mutable_node (build-only)");
}

uint8_t* PagedNodeStore::mutable_code(uint32_t) {
    throw Error(ErrorCode::NotImplemented,
                "PagedNodeStore does not support mutable_code (build-only)");
}

}  // namespace sextant
