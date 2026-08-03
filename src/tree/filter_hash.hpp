#pragma once

/// @file filter_hash.hpp
/// Deterministic 32-bit hash for filter column values (strings, set elements).
///
/// Uses rapidhash (64-bit) internally, truncated to 32-bit with a FIXED seed.
/// The fixed seed is critical: hashes are stored at build time and compared at
/// search time, across machines and architectures. A non-deterministic seed
/// would break cross-platform compatibility.
///
/// The 32-bit truncation is sufficient for filter columns — collision probability
/// is ~1/4B per pair, and the hash is always verified by exact string compare on
/// match (§3.3 of the architecture plan). The hash is a pre-filter, not the
/// final answer.

#include <cstdint>
#include <cstring>
#include <string_view>

/// Include rapidhash (C single-header, MIT).
/// We define RAPIDHASH_COMPACT to keep code size small (filter strings are short).
#define RAPIDHASH_COMPACT
#include "rapidhash/rapidhash.h"

namespace sextant {

/// Fixed seed for all filter column hashing. NEVER change this — it would
/// invalidate all existing indexes. The value is arbitrary; it just needs to
/// be constant and non-zero.
inline constexpr uint64_t kFilterHashSeed = 0x5E784A6E5Full;

/// Compute a deterministic 32-bit hash of a byte buffer.
/// Uses rapidhash with the fixed seed, truncates to uint32_t.
inline uint32_t filter_hash(const void* data, size_t len) {
    const uint64_t h = rapidhash_withSeed(data, len, kFilterHashSeed);
    // Mix the high and low 32 bits rather than just truncating — better
    // avalanche for the bloom filter bit-probing.
    return static_cast<uint32_t>(h ^ (h >> 32));
}

/// Compute a deterministic 32-bit hash of a string_view.
inline uint32_t filter_hash(std::string_view sv) {
    return filter_hash(sv.data(), sv.size());
}

/// Bloom filter bit-probe indices for a given hash.
/// Uses double hashing: probe i = (h1 + i * h2) % n_bits.
/// h1 = lower 32 bits, h2 = upper 32 bits (of the 64-bit rapidhash).
/// For k probes: probe_i = (h1 + i * h2) % n_bits, i = 0..k-1.
struct BloomProbe {
    uint32_t h1;
    uint32_t h2;  // must be odd (ensures full period modulo power-of-2 n_bits)
};

inline BloomProbe bloom_probe(uint32_t hash32) {
    // Split the 32-bit hash into two 16-bit halves, extend to 32-bit.
    // h2 must be odd for the double-hashing to cover all bits.
    return BloomProbe{
        hash32,
        (hash32 >> 16) | 1u  // upper 16 bits, forced odd
    };
}

/// Compute the k-th bloom probe bit index (0-indexed).
inline uint32_t bloom_bit(const BloomProbe& bp, uint32_t k, uint32_t n_bits) {
    return (bp.h1 + k * bp.h2) % n_bits;
}

/// Number of bloom hash probes (k). Shared by write (filter_column_write.hpp)
/// and read (filter_scan.hpp) paths. Must stay in sync.
inline constexpr uint32_t kBloomK = 3;

}  // namespace sextant
