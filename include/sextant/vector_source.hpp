#pragma once

/// @file vector_source.hpp
/// Pull-based vector source interface.
///
/// The engine drives the build loop, pulling vectors from a VectorSource.
/// Phase 1 implements FbinSource; Phase 2 implements DuckDBChunkSource.

#include <sextant/types.hpp>
#include <sextant/schema.hpp>

namespace sextant {

/// A readable stream of vectors. The engine pulls from this.
class VectorSource {
public:
    virtual ~VectorSource() = default;

    /// Dimensionality. Called once at open.
    virtual Dim dim() const = 0;

    /// Total vector count, or 0 if unknown (streaming source).
    virtual uint64_t count() const = 0;

    /// Reset to the beginning for a new pass.
    virtual void reset() = 0;

    /// Read the next chunk. Returns false at EOF.
    virtual bool next(Chunk& out) = 0;

    /// Filesystem path backing this source, or empty if the source has no
    /// on-disk file (e.g. MemorySource, future DuckDB source).
    ///
    /// When non-empty, the engine can seek directly into the file for
    /// seek-based random sampling (O(k) I/O for a k-vector sample) instead of
    /// the O(N) sequential reservoir scan. When empty, callers fall back to
    /// Algorithm R reservoir sampling over next().
    virtual std::string path() const { return {}; }

    /// Filter schema for this source. Default: empty (no filter columns).
    /// Sources that provide filter data override this.
    virtual Schema schema() const { return {}; }

    // --- I/O observability (monotone across the source's lifetime; used
    //     by metrics::MetricsCollector for per-phase attribution). All
    //     default to 0; sources with real backing I/O override. ---
    /// Seconds consumers spent blocked in next() waiting for I/O (relaxed
    /// atomics inside implementers — safe under concurrent readers).
    virtual double wait_seconds() const { return 0; }
    /// Number of consumer-side blocking waits (waits long enough to matter).
    virtual uint64_t wait_count() const { return 0; }
    /// Bytes actually read from the backing store (raw, pre-cast).
    virtual uint64_t bytes_read() const { return 0; }
};

}  // namespace sextant
