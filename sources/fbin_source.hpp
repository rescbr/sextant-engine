#pragma once

/// @file fbin_source.hpp
/// FbinSource: reads .fbin / .ibin / .bbin / .bvecs vector files.
///
/// .fbin layout (little-endian):
///   [uint32 n] [uint32 dim] [n × dim × float32 row-major]
/// .ibin: int8 values (cast to float on read, Issue 39 — NO normalization).
/// .bbin/.bvecs: uint8 values (cast to float on read).
///
/// Regular buffered open()/read() — input data is a sequential scan, not
/// direct-IO. The Engine drives the pull loop via next().

#include "sextant/vector_source.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace sextant {

class FbinSource : public VectorSource {
public:
    /// Chunk size in vectors (default 2048).
    static constexpr uint32_t kDefaultChunkSize = 2048;

    /// Open `path` and read the header. Infers element type from the extension.
    explicit FbinSource(const std::string& path,
                        uint32_t chunk_size = kDefaultChunkSize);
    ~FbinSource() override;

    FbinSource(const FbinSource&) = delete;
    FbinSource& operator=(const FbinSource&) = delete;

    Dim dim() const override { return dim_; }
    uint64_t count() const override { return count_; }
    void reset() override;
    bool next(Chunk& out) override;
    std::string path() const override { return path_; }

private:
    enum class ElemType { Float32, Int8, Uint8 };
    static ElemType infer_type(const std::string& path);

    std::string path_;
    int fd_ = -1;
    ElemType type_ = ElemType::Float32;
    uint32_t dim_ = 0;
    uint64_t count_ = 0;
    uint32_t chunk_size_ = 0;

    /// File offset where the vector data begins (after the 8-byte header).
    uint64_t data_offset_ = 8;

    /// Next vector index to hand out (prefetch reads run one chunk ahead).
    uint64_t cursor_ = 0;

    /// Internal chunk buffers, double-buffered: next() returns buffer
    /// `cur_` while a background thread preads the following chunk into
    /// the other one. The streaming build passes (Lloyd, emission) are
    /// otherwise I/O-serialized between chunks — measured ~2x more read
    /// time than compute time per chunk at 10M scale, idling all worker
    /// threads. The returned Chunk stays valid until two next() calls
    /// later (a superset of the old valid-until-next contract).
    std::vector<float> vec_buf_[2];
    std::vector<RowId> rowid_buf_[2];
    /// Byte buffers for int8/uint8 raw reads (cast into vec_buf_).
    std::vector<uint8_t> raw_buf_[2];
    int cur_ = 0;

    /// Prefetch state (all fields touched by next()/reset()/dtor only,
    /// except pf_err_ which the background thread writes before join).
    std::thread pf_thread_;
    bool pf_live_ = false;
    bool have_pending_ = false;
    int pend_buf_ = 0;
    uint32_t pend_count_ = 0;
    std::string pf_err_;

    /// Blocking read of `n` vectors at global index `at` into buffer
    /// `buf` (pread-based: safe to call from the prefetch thread).
    void read_chunk(int buf, uint64_t at, uint32_t n);

public:
    // --- I/O observability (relaxed atomics; monotone counters read by
    //     metrics::MetricsCollector at phase boundaries) ---
    double wait_seconds() const override {
        return wait_seconds_.load(std::memory_order_relaxed);
    }
    uint64_t wait_count() const override {
        return wait_count_.load(std::memory_order_relaxed);
    }
    uint64_t bytes_read() const override {
        return bytes_read_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<double> wait_seconds_{0};
    std::atomic<uint64_t> wait_count_{0};
    std::atomic<uint64_t> bytes_read_{0};
};

}  // namespace sextant