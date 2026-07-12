#pragma once

/// @file sidecar_header.hpp
/// On-disk header for all sidecar files (.graph, .codes, .meta).
///
/// All fields are little-endian (x86 LE, aarch64 LE default).

#include <cstdint>

namespace sextant {

// Magic numbers for each file type (little-endian uint64).
inline constexpr uint64_t kMagicGraph = 0x484d5648'50415247ULL;  // "GRAPHHMV"
inline constexpr uint64_t kMagicCodes = 0x4f444f43'484d5643ULL;  // "CVMHCODO"
inline constexpr uint64_t kMagicMeta  = 0x484d564d'4154454dULL;  // "METAMVMH"
inline constexpr uint64_t kMagicManifest = 0x544e4946'53414d4eULL; // "NMASFNIT"

inline constexpr uint32_t kFormatVersion = 1;

#pragma pack(push, 1)
struct SidecarHeader {
    uint64_t magic;           ///< File-type magic
    uint32_t format_version;  ///< Bumped on layout changes
    uint32_t header_size;     ///< sizeof(this header)
    uint64_t data_offset;     ///< Byte offset where block data begins
    uint64_t created_unix;    ///< Creation timestamp
    uint64_t index_uuid_low;  ///< Shared across .graph/.codes/.meta
    uint64_t index_uuid_high;
    uint64_t n_vectors;       ///< Vector count at build time
    uint32_t dim;             ///< Vector dimensionality
    uint32_t checksum_algo;   ///< 0=none, 1=crc32c
};
#pragma pack(pop)

static_assert(sizeof(SidecarHeader) == 64, "SidecarHeader must be 64 bytes");

}  // namespace sextant
