// compute_filtered_gt.cpp — Compute true filtered ground truth.
//
// Given a base .fbin (vectors), query .fbin, and a .fdat filter file,
// computes the exact top-k nearest neighbors among vectors matching
// a single int32 equality predicate. Writes a canonical GTMM file
// format (magic + n + k + row_ids).
//
// This is the "on-the-fly filtered GT" from task 4 — solves the
// GT-depth-too-shallow problem for selectivities ≥2%.
//
// Usage:
//   sextant_filtered_gt --base base.fbin --query query.fbin
//     --fdat filter.fdat --col-name year --col-val 2050
//     --topk 10 --out filtered.gtmm

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "fbin_io.hpp"

namespace {

struct FdatHeader {
    uint32_t magic;
    uint64_t n_rows;
    uint32_t n_cols;
};

struct FdatColumn {
    uint8_t type;
    std::string name;
    std::vector<uint8_t> fixed_data;  // for int32/int64/float/bool
    uint32_t width;
};

}  // namespace

int main(int argc, char* argv[]) {
    std::string base_path, query_path, fdat_path, col_name, out_path;
    int32_t col_val = 0;
    uint32_t topk = 10;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--base" && i + 1 < argc) base_path = argv[++i];
        else if (a == "--query" && i + 1 < argc) query_path = argv[++i];
        else if (a == "--fdat" && i + 1 < argc) fdat_path = argv[++i];
        else if (a == "--col-name" && i + 1 < argc) col_name = argv[++i];
        else if (a == "--col-val" && i + 1 < argc) col_val = std::atoi(argv[++i]);
        else if (a == "--topk" && i + 1 < argc) topk = std::atoi(argv[++i]);
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else {
            std::cerr << "unknown arg: " << a << "\n";
            return 1;
        }
    }

    if (base_path.empty() || query_path.empty() || fdat_path.empty() ||
        col_name.empty() || out_path.empty()) {
        std::cerr << "usage: sextant_filtered_gt --base X --query X --fdat X "
                     "--col-name X --col-val N --topk K --out X\n";
        return 1;
    }

    // --- Read fdat to find matching row_ids ---
    std::ifstream fdat(fdat_path, std::ios::binary);
    if (!fdat) { std::cerr << "cannot open " << fdat_path << "\n"; return 1; }

    FdatHeader fh;
    fdat.read(reinterpret_cast<char*>(&fh), sizeof(fh));
    if (fh.magic != 0x46444154) {
        std::cerr << "bad fdat magic\n"; return 1;
    }

    std::vector<FdatColumn> cols(fh.n_cols);
    uint32_t target_col = UINT32_MAX;
    for (uint32_t c = 0; c < fh.n_cols; ++c) {
        fdat.read(reinterpret_cast<char*>(&cols[c].type), 1);
        uint16_t name_len;
        fdat.read(reinterpret_cast<char*>(&name_len), 2);
        cols[c].name.resize(name_len);
        fdat.read(cols[c].name.data(), name_len);
        cols[c].width = (cols[c].type == 0) ? 4 : (cols[c].type == 1) ? 8 :
                        (cols[c].type == 2) ? 4 : (cols[c].type == 3) ? 1 : 0;
        if (cols[c].name == col_name && cols[c].type == 0) {  // int32
            target_col = c;
        }
    }
    if (target_col == UINT32_MAX) {
        std::cerr << "column '" << col_name
                  << "' (int32) not found in fdat\n"; return 1;
    }

    // Read fixed-width data for all columns (skip variable-length).
    for (uint32_t c = 0; c < fh.n_cols; ++c) {
        if (cols[c].width > 0) {
            cols[c].fixed_data.resize(static_cast<size_t>(fh.n_rows) * cols[c].width);
            fdat.read(reinterpret_cast<char*>(cols[c].fixed_data.data()),
                      cols[c].fixed_data.size());
        } else {
            // String (type=4) or set (type=5): skip.
            // For now, we only support int32 equality filters.
            std::cerr << "skipping variable-length column " << cols[c].name << "\n";
            // Read past the variable-length data.
            for (uint64_t r = 0; r < fh.n_rows; ++r) {
                if (cols[c].type == 4) {  // string
                    uint16_t len;
                    fdat.read(reinterpret_cast<char*>(&len), 2);
                    fdat.seekg(len, std::ios::cur);
                } else if (cols[c].type == 5) {  // set
                    uint8_t count;
                    fdat.read(reinterpret_cast<char*>(&count), 1);
                    for (uint8_t e = 0; e < count; ++e) {
                        uint16_t len;
                        fdat.read(reinterpret_cast<char*>(&len), 2);
                        fdat.seekg(len, std::ios::cur);
                    }
                }
            }
        }
    }

    // Find matching row_ids.
    std::vector<uint32_t> matching;
    matching.reserve(fh.n_rows / 10);  // estimate 10% selectivity
    const auto& col_data = cols[target_col].fixed_data;
    for (uint64_t r = 0; r < fh.n_rows; ++r) {
        int32_t val;
        std::memcpy(&val, col_data.data() + r * 4, 4);
        if (val == col_val) matching.push_back(static_cast<uint32_t>(r));
    }
    std::cerr << "filter: " << col_name << "==" << col_val << " → "
              << matching.size() << "/" << fh.n_rows
              << " (" << (100.0 * matching.size() / fh.n_rows) << "%)\n";

    if (matching.empty()) {
        std::cerr << "no matching vectors — nothing to do\n";
        return 1;
    }

    // --- Read base vectors (mmap for efficiency) ---
    FILE* bf = std::fopen(base_path.c_str(), "rb");
    if (!bf) { std::cerr << "cannot open " << base_path << "\n"; return 1; }
    uint32_t header[2];
    if (std::fread(header, 4, 2, bf) != 2) {
        std::cerr << "short read on base header\n";
        return 1;
    }
    const uint32_t dim = header[1];
    // Read matching vectors into RAM (only the ones we need).
    std::vector<float> match_vecs(matching.size() * dim);
    std::vector<float> buf(dim);
    for (size_t i = 0; i < matching.size(); ++i) {
        std::fseek(bf, 8 + static_cast<long>(matching[i]) * dim * 4, SEEK_SET);
        if (std::fread(match_vecs.data() + i * dim, 4, dim, bf) != dim) {
            std::cerr << "short read on base row " << matching[i] << "\n";
            return 1;
        }
    }
    std::fclose(bf);

    // --- Read queries ---
    FILE* qf = std::fopen(query_path.c_str(), "rb");
    if (!qf) { std::cerr << "cannot open " << query_path << "\n"; return 1; }
    uint32_t qheader[2];
    if (std::fread(qheader, 4, 2, qf) != 2) {
        std::cerr << "short read on query header\n";
        return 1;
    }
    const uint32_t nq = qheader[0];
    const uint32_t qdim = qheader[1];
    if (qdim != dim) {
        std::cerr << "dim mismatch: base=" << dim << " query=" << qdim << "\n";
        return 1;
    }
    std::vector<float> queries(static_cast<uint64_t>(nq) * dim);
    if (std::fread(queries.data(), 4, static_cast<size_t>(nq) * dim, qf)
        != static_cast<size_t>(nq) * dim) {
        std::cerr << "short read on queries\n";
        return 1;
    }
    std::fclose(qf);

    // --- Compute exact top-k for each query among matching vectors ---
    // Inner product (dot product) metric — matches sphere_ip.
    // For L2, negate the distance comparison.
    const uint32_t k = std::min(topk, static_cast<uint32_t>(matching.size()));
    std::vector<uint32_t> gt_results(static_cast<uint64_t>(nq) * k);

    for (uint32_t q = 0; q < nq; ++q) {
        const float* query = &queries[static_cast<uint64_t>(q) * dim];
        // Compute distances to all matching vectors.
        std::vector<std::pair<float, uint32_t>> dists(matching.size());
        for (size_t i = 0; i < matching.size(); ++i) {
            const float* vec = &match_vecs[i * dim];
            float dot = 0.0f;
            for (uint32_t d = 0; d < dim; ++d)
                dot += query[d] * vec[d];
            dists[i] = {-dot, matching[i]};  // negative for ascending sort
        }
        std::partial_sort(dists.begin(),
                          dists.begin() + std::min(k, static_cast<uint32_t>(dists.size())),
                          dists.end());
        for (uint32_t j = 0; j < k; ++j)
            gt_results[static_cast<uint64_t>(q) * k + j] = dists[j].second;
    }

    // --- Write GT file ---
    // Canonical GTMM (see include/sextant/ground_truth.hpp):
    // [magic][n][k][metric] then per query [ids k×u32][dists k×f32].
    // Dists are zero-filled (ids-only GT — recall does not need them).
    std::ofstream out(out_path, std::ios::binary);
    const uint32_t magic = 0x4D4D5447;  // "GTMM"
    const uint8_t metric = 0;  // IP
    out.write(reinterpret_cast<const char*>(&magic), 4);
    out.write(reinterpret_cast<const char*>(&nq), 4);
    out.write(reinterpret_cast<const char*>(&k), 4);
    out.write(reinterpret_cast<const char*>(&metric), 1);
    std::vector<float> zero_dists(k, 0.0f);
    for (uint32_t q = 0; q < nq; ++q) {
        out.write(reinterpret_cast<const char*>(
                      &gt_results[static_cast<size_t>(q) * k]),
                  static_cast<std::streamsize>(k) * 4);
        out.write(reinterpret_cast<const char*>(zero_dists.data()),
                  static_cast<std::streamsize>(k) * 4);
    }

    std::cerr << "wrote " << nq << " queries × " << k << " GT to " << out_path << "\n";
    return 0;
}
