#pragma once

/// @file types.hpp
/// Public type aliases for the Sextant engine.
///
/// These types are C++17-compatible (consumable by the Phase 2 DuckDB extension).

#include <cstdint>
#include <string>
#include <vector>

namespace sextant {

/// A row identifier as seen by the consumer (DuckDB rowid or .fbin index).
using RowId = int64_t;

/// An internal graph node identifier (dense, 0-indexed within the index).
using InternalId = uint32_t;

/// Dimensionality of vectors.
using Dim = uint32_t;

/// Metric kind for distance computation.
enum class MetricKind : uint8_t {
    /// Squared Euclidean distance. The build metric always.
    L2Sq,
    /// Inner product.
    InnerProduct,
};

/// A candidate result from search: (row_id, approximate distance).
struct Candidate {
    RowId row_id;
    float dist;
};

/// A chunk of vectors pulled from a VectorSource.
struct Chunk {
    const float* vectors;   ///< count × dim, row-major
    const RowId* row_ids;   ///< count entries
    uint32_t count;
};

/// Block size for sidecar file I/O (256 KB).
///
/// This is the unit of caching and I/O for the paged search path. Each block
/// read brings in ~1260 nodes (at node_size=208B), enabling PageSearch to
/// compute distances for co-located neighbors at no extra I/O cost.
inline constexpr uint32_t kBlockSize = 256 * 1024;

/// Disk alignment for O_DIRECT / posix_memalign.
inline constexpr uint32_t kDiskAlign = 4096;

}  // namespace sextant
