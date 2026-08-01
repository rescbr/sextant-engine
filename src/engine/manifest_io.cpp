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

template <typename T>
T require_double(const cpptoml::table& tbl, const char* key) {
    auto val = tbl.get_as<double>(key);
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
// IVF graph manifest
// ===========================================================================

std::string ivf_graph_manifest_to_toml(const IVFGraphManifest& m) {
    auto t = cpptoml::make_table();
    t->insert("ready", true);
    t->insert("K", static_cast<int64_t>(m.K));
    t->insert("dim", static_cast<int64_t>(m.dim));
    t->insert("n_probe_default", static_cast<int64_t>(m.n_probe_default));
    t->insert("closure_factor", static_cast<double>(m.closure_factor));
    std::ostringstream ss;
    ss << *t;
    return ss.str();
}

IVFGraphManifest ivf_graph_manifest_from_toml(const std::string& toml) {
    auto root = parse_toml(toml);
    IVFGraphManifest m;
    m.K = require_int<uint32_t>(*root, "K");
    m.dim = require_int<uint32_t>(*root, "dim");
    m.n_probe_default = require_int<uint32_t>(*root, "n_probe_default");
    m.closure_factor = static_cast<float>(require_double<double>(*root, "closure_factor"));
    return m;
}

// ===========================================================================
// IVF scan manifest
// ===========================================================================

std::string ivf_scan_manifest_to_toml(const IVFScanManifest& m) {
    auto t = cpptoml::make_table();
    t->insert("ready", true);
    t->insert("K", static_cast<int64_t>(m.K));
    t->insert("dim", static_cast<int64_t>(m.dim));
    t->insert("n_probe_default", static_cast<int64_t>(m.n_probe_default));
    t->insert("m4", static_cast<int64_t>(m.m4));
    t->insert("scan_pq_bits", static_cast<int64_t>(m.scan_pq_bits));
    t->insert("quantizer_type", m.quantizer_type);
    t->insert("prq_nsplits", static_cast<int64_t>(m.prq_nsplits));
    t->insert("sub_shard_probe_pct", static_cast<int64_t>(m.sub_shard_probe_pct));
    t->insert("adaptive_probe_gap", static_cast<double>(m.adaptive_probe_gap));
    t->insert("median_lid", static_cast<double>(m.median_lid));
    std::ostringstream ss;
    ss << *t;
    return ss.str();
}

IVFScanManifest ivf_scan_manifest_from_toml(const std::string& toml) {
    auto root = parse_toml(toml);
    IVFScanManifest m;
    m.K = require_int<uint32_t>(*root, "K");
    m.dim = require_int<uint32_t>(*root, "dim");
    m.n_probe_default = require_int<uint32_t>(*root, "n_probe_default");
    m.m4 = require_int<uint16_t>(*root, "m4");
    m.scan_pq_bits = static_cast<uint8_t>(require_int<int64_t>(*root, "scan_pq_bits"));
    if (auto qt = root->get_as<std::string>("quantizer_type")) {
        m.quantizer_type = *qt;
    }
    if (auto ns = root->get_as<int64_t>("prq_nsplits")) {
        m.prq_nsplits = static_cast<uint32_t>(*ns);
    }
    if (auto sp = root->get_as<int64_t>("sub_shard_probe_pct")) {
        m.sub_shard_probe_pct = static_cast<uint32_t>(*sp);
    }
    if (auto gap = root->get_as<double>("adaptive_probe_gap")) {
        m.adaptive_probe_gap = static_cast<float>(*gap);
    }
    if (auto lid = root->get_as<double>("median_lid")) {
        m.median_lid = static_cast<float>(*lid);
    }
    return m;
}

// ===========================================================================
// Per-shard manifest
// ===========================================================================

std::string shard_manifest_to_toml(const ShardManifest& m) {
    auto t = cpptoml::make_table();
    t->insert("ready", true);
    t->insert("count", static_cast<int64_t>(m.count));
    t->insert("dim", static_cast<int64_t>(m.dim));
    t->insert("m4", static_cast<int64_t>(m.m4));
    std::ostringstream ss;
    ss << *t;
    return ss.str();
}

ShardManifest shard_manifest_from_toml(const std::string& toml) {
    auto root = parse_toml(toml);
    ShardManifest m;
    m.count = require_int<uint32_t>(*root, "count");
    m.dim = require_int<uint32_t>(*root, "dim");
    m.m4 = require_int<uint16_t>(*root, "m4");
    return m;
}

// ===========================================================================
// Sub-shard centroid offsets
// ===========================================================================

std::string shard_offsets_to_toml(uint32_t n_subs,
                                  const std::vector<uint32_t>& offsets) {
    auto t = cpptoml::make_table();
    t->insert("ready", true);
    t->insert("n_subs", static_cast<int64_t>(n_subs));
    auto arr = cpptoml::make_array();
    for (uint32_t off : offsets) {
        arr->push_back(static_cast<int64_t>(off));
    }
    t->insert("offsets", arr);
    std::ostringstream ss;
    ss << *t;
    return ss.str();
}

std::pair<uint32_t, std::vector<uint32_t>>
shard_offsets_from_toml(const std::string& toml) {
    auto root = parse_toml(toml);
    uint32_t n_subs = require_int<uint32_t>(*root, "n_subs");
    std::vector<uint32_t> offsets;
    auto arr = root->get_array("offsets");
    if (arr) {
        for (const auto& v : *arr) {
            auto iv = v->as<int64_t>();
            if (iv) {
                offsets.push_back(static_cast<uint32_t>(iv->get()));
            }
        }
    }
    return {n_subs, offsets};
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
