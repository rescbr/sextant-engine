#pragma once

/// @file fbin_io.hpp
/// Shared .fbin I/O helpers for the Sextant CLI tools.
///
/// .fbin layout: [u32 n][u32 dim][n × dim × float32] (row-major).
/// Used by sextant_cli.cpp and benchmark.cpp (single-vector random access +
/// exact L2-squared distance for rerank). pq_explore.cpp bulk-loads whole
/// files and has its own reader.

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace sextant::fbin_io {

/// .fbin on-disk header (just n and dim; the rest is the float payload).
struct FbinHeader {
    uint32_t n = 0;
    uint32_t dim = 0;
};

/// Read the (n, dim) header from a .fbin file. Returns false on open/read error.
inline bool read_fbin_header(const std::string& path, FbinHeader& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&out.n), sizeof(out.n));
    f.read(reinterpret_cast<char*>(&out.dim), sizeof(out.dim));
    return f.good();
}

/// Read a single vector (row `idx`) from a .fbin file into `out` (dim floats).
/// Random-access: seeks to the row offset, no full-file load.
inline bool read_fbin_vector(const std::string& path, uint32_t dim, uint64_t idx,
                              std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    const uint64_t off = 8 + idx * static_cast<uint64_t>(dim) * sizeof(float);
    f.seekg(off);
    if (!f.good()) return false;
    out.resize(dim);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(dim * sizeof(float)));
    return f.good();
}

/// Exact L2-squared distance between two float vectors of length `dim`.
inline float l2sq_distance(const float* a, const float* b, uint32_t dim) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

}  // namespace sextant::fbin_io
