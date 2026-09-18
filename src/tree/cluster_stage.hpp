#pragma once

/// @file cluster_stage.hpp
/// Per-cluster DISK staging for the streaming build emission pass.
///
/// The emission pass used to accumulate per-cluster LeafBuffer structs in RAM
/// (codes, raw fp16 vectors, ip biases, row ids, payload bytes, filter column
/// data). Because the in-flight set is the last partial leaf of EVERY cluster,
/// RAM was Θ(n/4 rows) — ~21% of the corpus. ClusterStage moves those row
/// payloads to a temp file; only small accumulators (centroid sums, row
/// counts) stay in memory.
///
/// Staged record layout (little-endian scalars, plain memcpy):
///   u32 rec_len   total length of everything after this field
///   u64 row_id
///   u16 flags     bit0: has code, bit1: has fp16 vec, bit2: has ip bias,
///                 bit3: has payload, bit4: has filter row
///   [code_size bytes]            when kStagedHasCode
///   [dim*2 bytes fp16 vector]    when kStagedHasFp16
///   [f16 ip bias]                when kStagedHasIpBias
///   [filter row]                 when kStagedHasFilter (see below)
///   [payload bytes]              when kStagedHasPayload (len implied by
///                                rec_len arithmetic)
///
/// Filter row format, per schema column in order:
///   Int32/Int64/Float/Bool: column_type_width raw bytes
///   String: u16 len + len bytes
///   Set:    u8 count, then per element u16 len + bytes

#include <sextant/column_data.hpp>
#include <sextant/error.hpp>
#include <sextant/schema.hpp>
#include <sextant/types.hpp>
#include "util/fp16.hpp"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace sextant::tree {

/// Staged-record flags.
enum StagedFlags : uint16_t {
    kStagedHasCode    = 1u << 0,
    kStagedHasFp16    = 1u << 1,
    kStagedHasIpBias  = 1u << 2,
    kStagedHasPayload = 1u << 3,
    kStagedHasFilter  = 1u << 4,
};

/// Scalar encode helpers (write at `p`, advance it).
inline void staged_put_u8(uint8_t*& p, uint8_t v) {
    *p++ = v;
}
inline void staged_put_u16(uint8_t*& p, uint16_t v) {
    std::memcpy(p, &v, 2); p += 2;
}
inline void staged_put_u32(uint8_t*& p, uint32_t v) {
    std::memcpy(p, &v, 4); p += 4;
}
inline void staged_put_u64(uint8_t*& p, uint64_t v) {
    std::memcpy(p, &v, 8); p += 8;
}

/// Decoded view of one staged record (points into the caller's buffer).
struct StagedRow {
    RowId row_id = 0;
    uint16_t flags = 0;
    const uint8_t* code = nullptr;      // code bytes or null
    uint32_t code_len = 0;
    const uint8_t* fp16_vec = nullptr;    // dim halves, BYTE-PACKED at an
                                           // arbitrary (possibly odd) record
                                           // offset — copy with memcpy only
    uint32_t fp16_vec_bytes = 0;
    bool has_ip_bias = false;
    float16_t ip_bias{1.0f};
    const uint8_t* filter_row = nullptr;  // serialized filter row or null
    uint32_t filter_row_len = 0;
    const uint8_t* payload = nullptr;   // raw payload bytes or null
    uint32_t payload_len = 0;
};

/// Parse a staged record (everything after the rec_len field). Returns false
/// and fills `err` if the record is malformed.
bool staged_parse(const uint8_t* rec, uint32_t rec_len, StagedRow& out,
                  std::string& err);

/// Append a serialized filter row (see file header for the format) into a
/// per-leaf column-major ColumnData vector, exactly mirroring what the old
/// in-RAM append path built. `dst` must be pre-sized to schema size with
/// column types set. Returns false on malformed data.
bool staged_append_filter_row(const uint8_t* fr, uint32_t fr_len,
                              const Schema& schema,
                              std::vector<ColumnData>& dst, std::string& err);

/// RAM-first hybrid staging for the emission pass. Serial-call only
/// (the emission pass's serial merge loop is the only caller). Records
/// buffer in per-cluster in-RAM arenas while the total arena footprint is
/// under `budget_bytes`; clusters are spilled (largest arena first, until
/// total ≤ 90% of budget) to a temp file and never return to RAM. A budget
/// of 0 disables the RAM tier (pure disk staging). The temp file is opened
/// lazily and unlinked by finish()/the destructor, so it never outlives the
/// build — including on exception.
class ClusterStage {
public:
    /// Write-buffer size for the disk tier. Multi-GB spill traffic in 4 KiB
    /// pwrites costs ~1 syscall / 4 KiB on ZFS; 1 MiB keeps flushes large.
    static constexpr uint32_t kWriteBuf = 1u << 20;
    /// Dedicated per-spilled-cluster write buffer and the cap on how many
    /// clusters get one (256 × 256 KiB = 64 MiB worst case).
    static constexpr uint32_t kCWriteBuf = 256u << 10;
    static constexpr uint32_t kMaxDedicated = 256;
    static constexpr uint64_t kDefaultBudget = 1ull << 30;  // 1 GiB

    /// `path` is the temp-file path (`<output_path>.stage`); `n_clusters`
    /// pre-sizes the per-cluster record lists. `budget_bytes` caps the total
    /// in-RAM arena footprint (0 = pure disk staging).
    ClusterStage(std::string path, uint32_t n_clusters,
                 uint64_t budget_bytes = kDefaultBudget);
    ~ClusterStage();
    ClusterStage(const ClusterStage&) = delete;
    ClusterStage& operator=(const ClusterStage&) = delete;

    /// Stage one record for a cluster. `rec[-4..rec)` must hold the record's
    /// u32 rec_len (the build loop serializes the record with that prefix).
    /// RAM tier: appended to the cluster's arena (spilling victim clusters to
    /// disk when the total exceeds the budget). Disk tier (spilled cluster or
    /// zero budget): buffered through a single page; larger records spill
    /// directly. No fsync — durability is irrelevant (the stage is discarded
    /// once leaves are flushed).
    void append(uint32_t cluster, const uint8_t* rec, uint32_t len);

    /// Invoke `fn(const uint8_t* rec, uint32_t len)` for every staged record
    /// of a cluster, in append order. A cluster lives entirely in one tier
    /// (spilling moves its whole arena at once), so tier choice is a single
    /// branch. Disk records for one cluster arrive nearly contiguously, so
    /// the preads stay effectively sequential.
    template <class F>
    void for_each(uint32_t cluster, F&& fn) {
        if (!spilled_[cluster]) {
            const auto& arena = arenas_[cluster];
            for (const auto& [off, len] : ram_recs_[cluster])
                fn(arena.data() + off + 4, len - 4);
            return;
        }
        // Pending buffered bytes must be visible to the preads below.
        flush_cbuf(cluster);
        flush_wbuf();
        const auto& list = recs_[cluster];
        // Records for one cluster arrive in long contiguous runs (the serial
        // merge appends cluster-major per chunk), so coalesce consecutive
        // records into large preads instead of one syscall per record. The
        // run cap keeps rbuf_ bounded.
        constexpr uint64_t kRunMax = 4u << 20;
        size_t i = 0;
        while (i < list.size()) {
            const uint64_t start = list[i].first - 4;  // incl. rec_len prefix
            uint64_t end = start + 4 + list[i].second;
            size_t j = i + 1;
            while (j < list.size() &&
                   list[j].first - 4 == end && end - start < kRunMax) {
                end += 4 + list[j].second;
                ++j;
            }
            const uint64_t run = end - start;
            if (rbuf_.size() < run) rbuf_.resize(static_cast<size_t>(run));
            uint64_t got = 0;
            while (got < run) {
                const ssize_t r = ::pread(fd_, rbuf_.data() + got, run - got,
                                          static_cast<off_t>(start + got));
                if (r <= 0)
                    throw Error(ErrorCode::IoError,
                                "ClusterStage: short pread at offset " +
                                    std::to_string(start));
                got += static_cast<uint64_t>(r);
            }
            for (size_t k = i; k < j; ++k) {
                const size_t off = static_cast<size_t>(list[k].first - start);
                fn(rbuf_.data() + off, list[k].second);
            }
            i = j;
        }
    }

    /// Forget a cluster's records: frees its RAM arena or clears its disk
    /// record list (file space is reclaimed by unlinking the temp file at the
    /// end of the build; mid-build reclamation is not attempted).
    void drop(uint32_t cluster);

    /// Number of staged records for a cluster (RAM + disk tiers, including
    /// records still in a pending disk write buffer).
    uint32_t count(uint32_t cluster) const {
        return static_cast<uint32_t>(recs_[cluster].size() +
                                     ram_recs_[cluster].size() +
                                     cpend_[cluster].size());
    }

    /// Flush + close + unlink the temp file. Idempotent; the destructor
    /// becomes a no-op afterwards. Call after the final flush loop succeeds.
    void finish();

private:
    void ensure_open();
    void flush_wbuf();
    /// Write a spilled cluster's pending dedicated write buffer to the file
    /// and convert its buffer-relative record list to absolute offsets.
    void flush_cbuf(uint32_t cluster);
    void append_disk(uint32_t cluster, const uint8_t* rec, uint32_t len);
    /// Write one cluster's whole arena to the disk tier and mark it spilled.
    void spill(uint32_t cluster);
    /// Spill victim clusters (largest arena first) while over budget.
    void maybe_spill();

    std::string path_;
    int fd_ = -1;
    uint64_t budget_ = kDefaultBudget;  // RAM arena cap; 0 = pure disk
    uint64_t ram_bytes_ = 0;            // total logical bytes across arenas
    uint64_t disk_bytes_ = 0;           // bytes written to the stage file
    uint64_t n_spills_ = 0;             // arenas moved RAM -> disk
    // RAM tier: one arena per cluster + per-record (arena offset, total
    // record length incl. the rec_len prefix). 12 B/row while in RAM.
    std::vector<std::vector<uint8_t>> arenas_;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> ram_recs_;
    std::vector<bool> spilled_;  // true once a cluster's records live on disk
    // Disk tier: per-cluster record list: (file byte offset, record length).
    // This is the only per-row RAM the disk tier keeps: 12 B/row staged. At
    // n/4 in-flight rows that is ~7 MB at 10M rows and ~750 MB at 1B rows —
    // acceptable for now; if it ever matters, switch to per-cluster page
    // runs + offsets-in-page.
    std::vector<std::vector<std::pair<uint64_t, uint32_t>>> recs_;
    // Disk tier write path. Records of different spilled clusters interleave
    // in one shared buffer, which fragments each cluster's file bytes into
    // ~chunk-sized islands and makes the flush-time preads small and
    // scattered (~100k syscall-bound reads on the full corpus). Each spilled
    // cluster therefore gets its OWN dedicated write buffer so its records
    // stay contiguous in the file and flush as one big pread. Capped at
    // kMaxDedicated buffers; beyond that clusters share wbuf_.
    std::vector<std::vector<uint8_t>> cbuf_;   // per spilled cluster, cap kCWriteBuf
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> cpend_;
    uint32_t n_dedicated_ = 0;                 // cbuf_s with capacity allocated
    std::vector<uint8_t> wbuf_;   // one write page
    uint64_t wbuf_off_ = 0;       // file offset where wbuf_[0] lives
    uint32_t wbuf_used_ = 0;
    std::vector<uint8_t> rbuf_;   // reusable read buffer
};

}  // namespace sextant::tree
