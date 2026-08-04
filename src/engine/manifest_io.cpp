#include "manifest_io.hpp"

#include "sextant/error.hpp"

#include <cpptoml.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

// For fdatasync / fsync
#ifdef __linux__
#include <unistd.h>
#else
#include <unistd.h>
#endif

namespace sextant {

// ===========================================================================
// Internal helpers
// ===========================================================================

namespace {

template <typename T>
T require_int(const cpptoml::table& tbl, const char* key) {
    auto val = tbl.get_as<int64_t>(key);
    if (!val) {
        throw Error(ErrorCode::CorruptIndex,
                    std::string("manifest: missing required field '") + key + "'");
    }
    return static_cast<T>(*val);
}

std::shared_ptr<cpptoml::table> parse_toml(const std::string& toml) {
    std::istringstream ss(toml);
    cpptoml::parser p(ss);
    return p.parse();
}

}  // namespace

// ===========================================================================
// Graph index manifest
// ===========================================================================

std::string graph_manifest_to_toml(const GraphManifest& m) {
    auto t = cpptoml::make_table();
    t->insert("ready", true);
    t->insert("n_vectors", static_cast<int64_t>(m.n_vectors));
    t->insert("dim", static_cast<int64_t>(m.dim));
    t->insert("R", static_cast<int64_t>(m.R));
    t->insert("pq_m", static_cast<int64_t>(m.pq_m));
    std::ostringstream ss;
    ss << *t;
    return ss.str();
}

GraphManifest graph_manifest_from_toml(const std::string& toml) {
    auto root = parse_toml(toml);
    GraphManifest m;
    m.n_vectors = require_int<uint64_t>(*root, "n_vectors");
    m.dim = require_int<uint32_t>(*root, "dim");
    m.R = require_int<uint16_t>(*root, "R");
    m.pq_m = require_int<uint16_t>(*root, "pq_m");
    return m;
}

// ===========================================================================
// I/O helpers
// ===========================================================================

void write_manifest_atomic(const std::string& path,
                           const std::string& content) {
    const std::string tmp = path + ".tmp";
    {
        // Write via ofstream (buffered). The manifest is small (a few hundred
        // bytes); no need for O_DIRECT or aligned buffers.
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            throw Error(ErrorCode::IoError,
                        "write_manifest_atomic: cannot open '" + tmp +
                            "' for writing: " + std::strerror(errno));
        }
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!f) {
            throw Error(ErrorCode::IoError,
                        "write_manifest_atomic: write failed on '" + tmp + "'");
        }
        f.flush();
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        throw Error(ErrorCode::IoError,
                    "write_manifest_atomic: rename '" + tmp + "' → '" + path +
                        "' failed: " + ec.message());
    }
}

std::string read_file_to_string(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw Error(ErrorCode::CorruptIndex,
                    "read_file_to_string: cannot open '" + path + "'");
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace sextant
