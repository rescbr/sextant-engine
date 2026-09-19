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

/// Half-precision float for build-time distance computation (FP16 prune).
/// Maps to the compiler's native __fp16 (ARM NEON + x86 F16C). The meson build
/// verifies hardware FP16 support at configure time.
using float16_t = __fp16;

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

    /// Filter column data for this chunk (Phase C). One entry per schema
    /// column, in schema declaration order. nullptr when the source has no
    /// filter schema. Each entry points to a FilterColumnData structure
    /// appropriate for the column's type.
    ///
    /// For fixed-width types (int32/int64/float/bool): data points directly
    /// to a contiguous array of count × width bytes.
    ///
    /// For string/set types: data points to a FilterStringColumn /
    /// FilterSetColumn structure (defined in filter_column_data.hpp).
    const void* const* filter_columns = nullptr;

    /// Per-row NULL flags for the filter columns (1 = NULL), in schema
    /// order. May be NULL (= no NULLs in any column); individual entries
    /// may also be NULL (that column has no NULLs). Only columns declared
    /// nullable in the schema carry meaningful flags.
    const uint8_t* const* filter_nulls = nullptr;

    /// Opaque payload blobs (Phase E). Nullptr when no payload.
    /// payload_offsets[i] .. payload_offsets[i+1] gives the byte range for row i.
    const uint8_t* payload_data = nullptr;
    const uint32_t* payload_offsets = nullptr;  ///< count+1 entries
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
