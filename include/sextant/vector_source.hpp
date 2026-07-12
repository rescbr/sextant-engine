#pragma once

/// @file vector_source.hpp
/// Pull-based vector source interface.
///
/// The engine drives the build loop, pulling vectors from a VectorSource.
/// Phase 1 implements FbinSource; Phase 2 implements DuckDBChunkSource.

#include <sextant/types.hpp>

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
};

}  // namespace sextant
