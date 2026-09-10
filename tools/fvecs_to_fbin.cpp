// fvecs_to_fbin — convert .fvecs/.ivecs (Texmex format) to Sextant formats.
//
// .fvecs format (per vector):  [int32 dim][dim × float32 values]
// .ivecs format (per vector):  [int32 k][k × int32 values]
//
// .fbin format (global header): [uint32 n][uint32 dim][n × dim × float32]
// .gtmm format (ground truth):  [magic "GTMM"][uint32 n_queries][uint32 k]
//                               [uint8 metric (0=L2Sq)]
//                               per query: [k × uint32 ids][k × float32 dists]
//   With --base + --queries: distances are real L2-sq (enables proximity metrics).
//   Without: distances are zero-filled.
//
// Usage:
//   fvecs_to_fbin --input file.fvecs --output file.fbin
//   fvecs_to_fbin --input file.ivecs --output file.gtmm --gt

#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <cmdline/cmdline.h>

#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using sextant::Error;
using sextant::ErrorCode;

/// Read a whole file into a buffer. Throws sextant::Error on failure.
std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw Error(ErrorCode::IoError,
                    "cannot open input '" + path + "'");
    }
    const std::streamoff size = f.tellg();
    if (size <= 0) {
        throw Error(ErrorCode::IoError,
                    "input '" + path + "' is empty or unreadable");
    }
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    f.read(buf.data(), size);
    if (!f && !f.eof()) {
        throw Error(ErrorCode::IoError,
                    "short read on '" + path + "'");
    }
    return buf;
}

/// Convert .fvecs → .fbin.
/// Each record: [int32 dim][dim × float32]. We strip the per-vector dim prefix
/// and emit a global [uint32 n][uint32 dim] header + raw float32 payload.
void convert_fvecs(const std::string& input, const std::string& output) {
    auto buf = read_file(input);
    const char* p = buf.data();
    const char* end = buf.data() + buf.size();

    // Peek the first record to learn dim.
    if (end - p < static_cast<std::ptrdiff_t>(sizeof(int32_t))) {
        throw Error(ErrorCode::InvalidParam,
                    "fvecs file '" + input + "' too small for a record");
    }
    int32_t dim0 = 0;
    std::memcpy(&dim0, p, sizeof(int32_t));
    if (dim0 <= 0) {
        throw Error(ErrorCode::InvalidParam,
                    "fvecs file '" + input + "' has non-positive dim");
    }
    const uint32_t dim = static_cast<uint32_t>(dim0);
    const size_t record_bytes =
        sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(float);

    // Count records and sanity-check each one.
    uint32_t n = 0;
    const char* q = p;
    while (q + record_bytes <= end) {
        int32_t d = 0;
        std::memcpy(&d, q, sizeof(int32_t));
        if (static_cast<uint32_t>(d) != dim) {
            throw Error(ErrorCode::InvalidParam,
                        "fvecs file '" + input +
                            "' has inconsistent dim in record " +
                            std::to_string(n));
        }
        ++n;
        q += record_bytes;
    }
    if (n == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "fvecs file '" + input + "' contains no complete records");
    }
    spdlog::info("fvecs_to_fbin: {} vectors, dim {}", n, dim);

    // Write .fbin: header + raw floats (skip each 4-byte dim prefix).
    std::ofstream out(output, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw Error(ErrorCode::IoError,
                    "cannot open output '" + output + "'");
    }
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));

    const std::vector<char> zeros(sizeof(float) * dim, 0);  // unused
    (void)zeros;
    std::vector<float> row(dim);
    for (uint32_t i = 0; i < n; i++) {
        const char* rec = p + i * record_bytes + sizeof(int32_t);
        std::memcpy(row.data(), rec, static_cast<size_t>(dim) * sizeof(float));
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(dim * sizeof(float)));
    }
    if (!out) {
        throw Error(ErrorCode::IoError,
                    "write failed on '" + output + "'");
    }
    std::cout << "wrote " << n << " × " << dim << " → " << output << "\n";
}

/// Convert .ivecs → GTMM ground-truth file (canonical interleaved).
/// Each record: [int32 k][k × int32 neighbor IDs].
/// Output: canonical GTMM ([magic][n][k][metric] + per-query [ids][dists]).
///
/// If --base and --queries are provided, distances are computed as L2-squared
/// between each query vector and its k neighbor vectors in the base. Otherwise
/// distances are zero-filled (and proximity metrics will be unavailable).
void convert_ivecs(const std::string& input, const std::string& output,
                   const std::string& base_path = "",
                   const std::string& query_path = "") {
    auto buf = read_file(input);
    const char* p = buf.data();
    const char* end = buf.data() + buf.size();

    // Collect all records. k may vary per query in principle; we require a
    // uniform k for the GTMM format and take it from the first record.
    std::vector<std::vector<uint32_t>> records;
    uint32_t k = 0;
    const char* q = p;
    while (q + static_cast<std::ptrdiff_t>(sizeof(int32_t)) <= end) {
        int32_t kk = 0;
        std::memcpy(&kk, q, sizeof(int32_t));
        q += sizeof(int32_t);
        if (kk <= 0) {
            throw Error(ErrorCode::InvalidParam,
                        "ivecs file '" + input +
                            "' has non-positive k in record " +
                            std::to_string(records.size()));
        }
        const size_t want = static_cast<size_t>(kk) * sizeof(int32_t);
        if (q + static_cast<std::ptrdiff_t>(want) > end) {
            throw Error(ErrorCode::CorruptIndex,
                        "ivecs file '" + input +
                            "' truncated in record " +
                            std::to_string(records.size()));
        }
        if (records.empty()) {
            k = static_cast<uint32_t>(kk);
        } else if (static_cast<uint32_t>(kk) != k) {
            throw Error(ErrorCode::InvalidParam,
                        "ivecs file '" + input +
                            "' has inconsistent k in record " +
                            std::to_string(records.size()));
        }
        std::vector<uint32_t> ids(static_cast<size_t>(kk));
        std::memcpy(ids.data(), q, want);
        q += want;
        records.push_back(std::move(ids));
    }
    const uint32_t n = static_cast<uint32_t>(records.size());
    if (n == 0 || k == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "ivecs file '" + input + "' contains no records");
    }
    spdlog::info("fvecs_to_fbin: {} queries, k={}", n, k);

    // Optionally load base + query vectors to compute real distances.
    const bool compute_dists = !base_path.empty() && !query_path.empty();
    std::vector<float> base_vecs;
    std::vector<float> query_vecs;
    uint32_t base_n = 0, base_dim = 0, query_n = 0, query_dim = 0;
    if (compute_dists) {
        // Read base .fbin header + all vectors.
        {
            std::ifstream bf(base_path, std::ios::binary);
            if (!bf) throw Error(ErrorCode::IoError,
                                 "cannot open base '" + base_path + "'");
            bf.read(reinterpret_cast<char*>(&base_n), sizeof(uint32_t));
            bf.read(reinterpret_cast<char*>(&base_dim), sizeof(uint32_t));
            base_vecs.resize(static_cast<size_t>(base_n) * base_dim);
            bf.read(reinterpret_cast<char*>(base_vecs.data()),
                    static_cast<std::streamsize>(base_vecs.size() * sizeof(float)));
            if (!bf) throw Error(ErrorCode::IoError,
                                 "short read on base '" + base_path + "'");
        }
        // Read query .fbin header + all vectors.
        {
            std::ifstream qf(query_path, std::ios::binary);
            if (!qf) throw Error(ErrorCode::IoError,
                                 "cannot open queries '" + query_path + "'");
            qf.read(reinterpret_cast<char*>(&query_n), sizeof(uint32_t));
            qf.read(reinterpret_cast<char*>(&query_dim), sizeof(uint32_t));
            query_vecs.resize(static_cast<size_t>(query_n) * query_dim);
            qf.read(reinterpret_cast<char*>(query_vecs.data()),
                    static_cast<std::streamsize>(query_vecs.size() * sizeof(float)));
            if (!qf) throw Error(ErrorCode::IoError,
                                 "short read on queries '" + query_path + "'");
        }
        if (base_dim != query_dim) {
            throw Error(ErrorCode::InvalidParam,
                        "base dim " + std::to_string(base_dim) +
                        " != query dim " + std::to_string(query_dim));
        }
        if (query_n < n) {
            throw Error(ErrorCode::InvalidParam,
                        "query file has " + std::to_string(query_n) +
                        " vectors but GT has " + std::to_string(n) + " queries");
        }
        spdlog::info("fvecs_to_fbin: computing L2-sq distances (base_n={}, dim={})",
                     base_n, base_dim);
    }

    std::ofstream out(output, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw Error(ErrorCode::IoError,
                    "cannot open output '" + output + "'");
    }
    // GTMM format (the only GT layout the engine reads):
    // [magic "GTMM"][n:u32][k:u32][metric:u8=0 L2Sq]
    // per query: [k x u32 ids][k x f32 dists] (interleaved).
    constexpr uint32_t kGtMagic = 0x4D4D5447u;  // "GTMM" LE
    const uint8_t metric = 0;  // L2Sq
    out.write(reinterpret_cast<const char*>(&kGtMagic), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&k), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(uint8_t));
    // Distances: real L2-sq if base+queries provided, else zero-filled.
    // Written per query, interleaved after that query's ids.
    if (compute_dists) {
        const uint32_t dim = base_dim;
        std::vector<float> dists(k);
        for (uint32_t qi = 0; qi < n; qi++) {
            out.write(reinterpret_cast<const char*>(records[qi].data()),
                      static_cast<std::streamsize>(k * sizeof(uint32_t)));
            const float* qv = &query_vecs[static_cast<size_t>(qi) * dim];
            for (uint32_t ki = 0; ki < k; ki++) {
                const uint32_t nid = records[qi][ki];
                double acc = 0.0;
                if (nid < base_n) {
                    const float* bv = &base_vecs[static_cast<size_t>(nid) * dim];
                    for (uint32_t d = 0; d < dim; d++) {
                        const double diff = static_cast<double>(qv[d]) -
                                            static_cast<double>(bv[d]);
                        acc += diff * diff;
                    }
                }
                dists[ki] = static_cast<float>(acc);
            }
            out.write(reinterpret_cast<const char*>(dists.data()),
                      static_cast<std::streamsize>(k * sizeof(float)));
        }
        spdlog::info("fvecs_to_fbin: distances computed from base + queries");
    } else {
        spdlog::warn("fvecs_to_fbin: --base/--queries not provided; "
                     "distances zero-filled (proximity metrics will be unavailable)");
        std::vector<float> zeros(k, 0.0f);
        for (uint32_t i = 0; i < n; i++) {
            out.write(reinterpret_cast<const char*>(records[i].data()),
                      static_cast<std::streamsize>(k * sizeof(uint32_t)));
            out.write(reinterpret_cast<const char*>(zeros.data()),
                      static_cast<std::streamsize>(k * sizeof(float)));
        }
    }
    if (!out) {
        throw Error(ErrorCode::IoError,
                    "write failed on '" + output + "'");
    }
    std::cout << "wrote " << n << " × " << k << " → " << output << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    sextant::init_logging();

    cmdline::parser p;
    p.add<std::string>("input", 0, "Input .fvecs or .ivecs file", true);
    p.add<std::string>("output", 0, "Output .fbin or .gtmm file", true);
    p.add("gt", 0, "Ground-truth mode (.ivecs → .gtmm)");
    p.add<std::string>("base", 0, "Base .fbin (for --gt: compute real distances)", false, "");
    p.add<std::string>("queries", 0, "Query .fbin (for --gt: compute real distances)", false, "");
    p.parse_check(argc, argv);

    const std::string input = p.get<std::string>("input");
    const std::string output = p.get<std::string>("output");
    const bool gt_mode = p.exist("gt");
    const std::string base_path = p.get<std::string>("base");
    const std::string query_path = p.get<std::string>("queries");

    try {
        if (gt_mode) {
            convert_ivecs(input, output, base_path, query_path);
        } else {
            convert_fvecs(input, output);
        }
    } catch (const Error& e) {
        std::cerr << "fvecs_to_fbin: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "fvecs_to_fbin: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
