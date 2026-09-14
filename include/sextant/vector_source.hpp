#pragma once

/// @file vector_source.hpp
/// Pull-based vector source interface.
///
/// The engine drives the build loop, pulling vectors from a VectorSource.
/// Phase 1 implements FbinSource; Phase 2 implements DuckDBChunkSource.

#include <sextant/types.hpp>
#include <sextant/schema.hpp>

#include <functional>

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

    /// Upper-bound estimate of the total decoded payload bytes this source
    /// will stream (0 = no payload column, or unknown). Used by the build to
    /// size the fixed allocation-bitmap region — the estimate must never
    /// undershoot. Parquet sources derive it from column-chunk metadata
    /// (total_uncompressed_size, no data pages read); dictionary-encoded
    /// chunks are boosted because the payload extents store raw per-row
    /// strings while the chunk may store dict + indices.
    virtual uint64_t payload_total_bytes() const { return 0; }

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

    /// Toggle vector-only streaming: subsequent chunks carry vectors (and
    /// row ids) but NO filter columns or payloads. Sources that decode
    /// columns (parquet) use this to project only the vector column,
    /// skipping filter/payload decompression entirely — training/Lloyd
    /// passes re-read the corpus many times and never touch those columns.
    /// Takes effect at the next reset().
    virtual void set_vector_only(bool) {}

    /// Stream the whole corpus through `fn`, with up to `workers` chunks
    /// being processed CONCURRENTLY on different threads. Chunk order is
    /// NOT preserved (and row ids may be shard-local); each chunk's
    /// buffers are valid only for the duration of the call. Returns false
    /// when unsupported — the caller falls back to reset()+next() passes.
    /// Intended for order-free training passes (Lloyd refinement): a
    /// serial-decoding source is a single-core bottleneck that this
    /// parallelizes across independent files/shards.
    virtual bool parallel_for_each_chunk(
        uint32_t, const std::function<void(const Chunk&)>&) {
        return false;
    }
};

}  // namespace sextant
