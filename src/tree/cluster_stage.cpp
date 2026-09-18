#include "cluster_stage.hpp"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdio>

namespace sextant::tree {

ClusterStage::ClusterStage(std::string path, uint32_t n_clusters,
                           uint64_t budget_bytes)
    : path_(std::move(path)), budget_(budget_bytes),
      arenas_(n_clusters), ram_recs_(n_clusters), spilled_(n_clusters, false),
      recs_(n_clusters), cbuf_(n_clusters), cpend_(n_clusters) {
    wbuf_.resize(kWriteBuf);
}

ClusterStage::~ClusterStage() {
    try {
        finish();
    } catch (const std::exception& e) {
        // Destructor must not throw (the temp file may already be gone).
        spdlog::warn("[sextant] ClusterStage: cleanup failed: {}", e.what());
    }
}

void ClusterStage::ensure_open() {
    if (fd_ >= 0) return;
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC,
                 0644);
    if (fd_ < 0)
        throw Error(ErrorCode::IoError,
                    "ClusterStage: cannot create stage file " + path_);
}

void ClusterStage::flush_wbuf() {
    if (wbuf_used_ == 0) return;
    uint32_t done = 0;
    while (done < wbuf_used_) {
        const ssize_t w = ::pwrite(fd_, wbuf_.data() + done,
                                   wbuf_used_ - done,
                                   static_cast<off_t>(wbuf_off_ + done));
        if (w <= 0)
            throw Error(ErrorCode::IoError,
                        "ClusterStage: short pwrite to " + path_);
        done += static_cast<uint32_t>(w);
    }
    disk_bytes_ += wbuf_used_;
    wbuf_off_ += wbuf_used_;
    wbuf_used_ = 0;
}

void ClusterStage::append(uint32_t cluster, const uint8_t* rec,
                          uint32_t len) {
    if (budget_ == 0 || spilled_[cluster]) {
        append_disk(cluster, rec, len);
        return;
    }
    auto& arena = arenas_[cluster];
    const uint32_t total = len + 4;
    const uint32_t off = static_cast<uint32_t>(arena.size());
    arena.insert(arena.end(), rec - 4, rec - 4 + total);
    ram_recs_[cluster].emplace_back(off, total);
    ram_bytes_ += total;
    maybe_spill();
}

void ClusterStage::maybe_spill() {
    if (ram_bytes_ <= budget_) return;
    // Spill until comfortably under budget so we do not thrash on every
    // append that straddles the line.
    const uint64_t target = budget_ - budget_ / 10;  // 90% of budget
    while (ram_bytes_ > target) {
        uint32_t victim = UINT32_MAX;
        size_t best = 0;
        for (uint32_t c = 0; c < arenas_.size(); ++c) {
            if (spilled_[c] || arenas_[c].empty()) continue;
            if (arenas_[c].size() > best) {
                best = arenas_[c].size();
                victim = c;
            }
        }
        if (victim == UINT32_MAX) break;  // nothing un-spilled left
        spill(victim);
    }
}

void ClusterStage::spill(uint32_t cluster) {
    ensure_open();
    flush_wbuf();  // arena bytes land at the current end of file
    auto& arena = arenas_[cluster];
    const uint64_t start = wbuf_off_;
    uint64_t done = 0;
    while (done < arena.size()) {
        const ssize_t w = ::pwrite(fd_, arena.data() + done,
                                   arena.size() - done,
                                   static_cast<off_t>(start + done));
        if (w <= 0)
            throw Error(ErrorCode::IoError,
                        "ClusterStage: short pwrite to " + path_);
        done += static_cast<uint64_t>(w);
    }
    wbuf_off_ += arena.size();
    auto& list = recs_[cluster];
    list.reserve(list.size() + ram_recs_[cluster].size());
    for (const auto& [off, len] : ram_recs_[cluster])
        list.emplace_back(start + off + 4, len - 4);  // skip rec_len prefix
    ram_bytes_ -= arena.size();
    ++n_spills_;
    std::vector<uint8_t> empty;
    arena.swap(empty);
    std::vector<std::pair<uint32_t, uint32_t>> empty_recs;
    ram_recs_[cluster].swap(empty_recs);
    spilled_[cluster] = true;
}

void ClusterStage::flush_cbuf(uint32_t cluster) {
    auto& b = cbuf_[cluster];
    if (b.empty()) return;
    flush_wbuf();  // keep all direct writes at the true end of file
    const uint64_t base = wbuf_off_;
    uint64_t done = 0;
    while (done < b.size()) {
        const ssize_t w = ::pwrite(fd_, b.data() + done, b.size() - done,
                                   static_cast<off_t>(base + done));
        if (w <= 0)
            throw Error(ErrorCode::IoError,
                        "ClusterStage: short pwrite to " + path_);
        done += static_cast<uint64_t>(w);
    }
    disk_bytes_ += b.size();
    wbuf_off_ += b.size();
    auto& list = recs_[cluster];
    list.reserve(list.size() + cpend_[cluster].size());
    for (const auto& [off, len] : cpend_[cluster])
        list.emplace_back(base + off + 4, len);  // skip rec_len prefix
    cpend_[cluster].clear();
    b.clear();  // keep capacity for the cluster's next fill cycle
}

void ClusterStage::append_disk(uint32_t cluster, const uint8_t* rec,
                               uint32_t len) {
    ensure_open();
    spilled_[cluster] = true;  // records now live on the disk path
    auto& b = cbuf_[cluster];
    const bool dedicated = b.capacity() >= kCWriteBuf ||
        (b.empty() && cpend_[cluster].empty() && n_dedicated_ < kMaxDedicated);
    const uint32_t total = len + 4;
    if (dedicated) {
        if (b.capacity() < kCWriteBuf) {
            b.reserve(kCWriteBuf);
            ++n_dedicated_;
        }
        if (b.size() + total > kCWriteBuf) flush_cbuf(cluster);
        const uint32_t off = static_cast<uint32_t>(b.size());
        b.insert(b.end(), rec - 4, rec - 4 + total);
        cpend_[cluster].emplace_back(off, len);
        return;
    }
    recs_[cluster].emplace_back(
        wbuf_off_ + wbuf_used_ + 4, len);  // skip the rec_len field on read
    const uint8_t* p = rec - 4;
    uint32_t rem = total;
    while (rem > 0) {
        const uint32_t space = kWriteBuf - wbuf_used_;
        const uint32_t take = rem < space ? rem : space;
        std::memcpy(wbuf_.data() + wbuf_used_, p, take);
        wbuf_used_ += take;
        p += take;
        rem -= take;
        if (wbuf_used_ == kWriteBuf) {
            flush_wbuf();
            // Records larger than the write buffer: spill aligned chunks
            // directly so the write side never needs a second buffer.
            while (rem >= kWriteBuf) {
                uint32_t done = 0;
                while (done < kWriteBuf) {
                    const ssize_t w = ::pwrite(fd_, p + done, kWriteBuf - done,
                                               static_cast<off_t>(wbuf_off_ + done));
                    if (w <= 0)
                        throw Error(ErrorCode::IoError,
                                    "ClusterStage: short pwrite to " + path_);
                    done += static_cast<uint32_t>(w);
                }
                disk_bytes_ += kWriteBuf;
                wbuf_off_ += kWriteBuf;
                p += kWriteBuf;
                rem -= kWriteBuf;
            }
        }
    }
}
void ClusterStage::drop(uint32_t cluster) {
    if (spilled_[cluster]) {
        recs_[cluster].clear();
        cpend_[cluster].clear();
        cbuf_[cluster].clear();  // keep capacity; cluster stays spilled
        return;
    }
    std::vector<uint8_t> empty;
    ram_bytes_ -= arenas_[cluster].size();
    arenas_[cluster].swap(empty);
    std::vector<std::pair<uint32_t, uint32_t>> empty_recs;
    ram_recs_[cluster].swap(empty_recs);
}

void ClusterStage::finish() {
    if (fd_ < 0) {
        // No spill ever happened; still release arena memory eagerly.
        for (auto& a : arenas_) {
            std::vector<uint8_t> empty;
            a.swap(empty);
        }
        return;
    }
    flush_wbuf();
    for (uint32_t c = 0; c < cbuf_.size(); ++c) flush_cbuf(c);
    spdlog::debug("[sextant] ClusterStage: {} MiB staged to disk, {} spills",
                  disk_bytes_ >> 20, n_spills_);
    ::close(fd_);
    fd_ = -1;
    ::unlink(path_.c_str());
    // Release arena and record-list memory eagerly (drop() only clears
    // logical state).
    for (auto& a : arenas_) {
        std::vector<uint8_t> empty;
        a.swap(empty);
    }
    for (auto& rl : ram_recs_) {
        std::vector<std::pair<uint32_t, uint32_t>> empty;
        rl.swap(empty);
    }
    for (auto& list : recs_) {
        std::vector<std::pair<uint64_t, uint32_t>> empty;
        list.swap(empty);
    }
}

namespace {

inline uint16_t get_u16(const uint8_t* p) {
    uint16_t v; std::memcpy(&v, p, 2); return v;
}
inline uint8_t get_u8(const uint8_t* p) { return *p; }

}  // namespace

bool staged_parse(const uint8_t* rec, uint32_t rec_len, StagedRow& out,
                  std::string& err) {
    out = StagedRow{};
    const uint8_t* p = rec;
    const uint8_t* end = rec + rec_len;
    if (rec_len < 10) {
        err = "record too short for header";
        return false;
    }
    std::memcpy(&out.row_id, p, 8); p += 8;
    out.flags = get_u16(p); p += 2;

    auto need = [&](uint32_t n) {
        if (static_cast<uint32_t>(end - p) < n) {
            err = "record truncated";
            return false;
        }
        return true;
    };

    if (out.flags & kStagedHasCode) {
        if (!need(4)) return false;
        uint32_t n; std::memcpy(&n, p, 4); p += 4;
        if (!need(n)) return false;
        out.code = p; out.code_len = n; p += n;
    }
    if (out.flags & kStagedHasFp16) {
        if (!need(4)) return false;
        uint32_t n; std::memcpy(&n, p, 4); p += 4;
        if (n % 2) { err = "fp16 vector has odd byte length"; return false; }
        if (!need(n)) return false;
        // Untyped: the record is byte-packed and this field can sit at an
        // ODD offset — a typed float16_t* here invites misaligned derefs.
        out.fp16_vec = p;
        out.fp16_vec_bytes = n; p += n;
    }
    if (out.flags & kStagedHasIpBias) {
        if (!need(2)) return false;
        out.has_ip_bias = true;
        std::memcpy(&out.ip_bias, p, 2); p += 2;
    }
    if (out.flags & kStagedHasFilter) {
        if (!need(4)) return false;
        uint32_t n; std::memcpy(&n, p, 4); p += 4;
        if (!need(n)) return false;
        out.filter_row = p; out.filter_row_len = n; p += n;
    }
    if (out.flags & kStagedHasPayload) {
        out.payload = p;
        out.payload_len = static_cast<uint32_t>(end - p);
        p = end;
    }
    if (p != end) {
        err = "record has trailing bytes";
        return false;
    }
    return true;
}

bool staged_append_filter_row(const uint8_t* fr, uint32_t fr_len,
                              const Schema& schema,
                              std::vector<ColumnData>& dst, std::string& err) {
    const uint8_t* p = fr;
    const uint8_t* end = fr + fr_len;
    const uint32_t n = schema.n_filter_columns();
    if (dst.size() < n) {
        err = "destination ColumnData smaller than schema";
        return false;
    }
    for (uint32_t c = 0; c < n; ++c) {
        auto& d = dst[c];
        switch (d.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool: {
                const uint8_t w = column_type_width(d.type);
                if (static_cast<uint32_t>(end - p) < w) {
                    err = "fixed-width value truncated";
                    return false;
                }
                d.fixed_data.insert(d.fixed_data.end(), p, p + w);
                p += w;
                break;
            }
            case ColumnType::String: {
                if (static_cast<uint32_t>(end - p) < 2) {
                    err = "string length truncated";
                    return false;
                }
                const uint16_t len = get_u16(p); p += 2;
                if (static_cast<uint32_t>(end - p) < len) {
                    err = "string bytes truncated";
                    return false;
                }
                d.str_offsets.push_back(
                    static_cast<uint32_t>(d.str_data.size()));
                d.str_lengths.push_back(len);
                d.str_data.insert(d.str_data.end(),
                                  reinterpret_cast<const char*>(p),
                                  reinterpret_cast<const char*>(p) + len);
                p += len;
                break;
            }
            case ColumnType::Set: {
                if (p >= end) {
                    err = "set count truncated";
                    return false;
                }
                const uint8_t ec = get_u8(p); ++p;
                d.set_counts.push_back(ec);
                d.set_offsets.push_back(
                    static_cast<uint32_t>(d.set_elem_lengths.size()));
                for (uint32_t e = 0; e < ec; ++e) {
                    if (static_cast<uint32_t>(end - p) < 2) {
                        err = "set element length truncated";
                        return false;
                    }
                    const uint16_t elen = get_u16(p); p += 2;
                    if (static_cast<uint32_t>(end - p) < elen) {
                        err = "set element bytes truncated";
                        return false;
                    }
                    d.set_elem_lengths.push_back(elen);
                    d.set_elem_data.insert(
                        d.set_elem_data.end(),
                        reinterpret_cast<const char*>(p),
                        reinterpret_cast<const char*>(p) + elen);
                    p += elen;
                }
                break;
            }
        }
    }
    if (p != end) {
        err = "filter row has trailing bytes";
        return false;
    }
    return true;
}

}  // namespace sextant::tree
