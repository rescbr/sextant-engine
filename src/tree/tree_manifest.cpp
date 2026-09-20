#include "tree_manifest.hpp"

#include "sextant/error.hpp"

#include <cpptoml.h>

#include <chrono>
#include <cstddef>
#include <random>
#include <sstream>

namespace sextant::tree {

/// Manifest format version. The manifest declares this in [meta]; parsing
/// rejects any other value. Bump when the format changes incompatibly
/// (all pre-v1 trees were dev-only and are expected to be rebuilt).
constexpr int64_t kManifestFormatVersion = 1;

namespace {

const char* column_type_to_string(ColumnType t) {
    switch (t) {
        case ColumnType::Int32:  return "int32";
        case ColumnType::Int64:  return "int64";
        case ColumnType::Float:  return "float";
        case ColumnType::Bool:   return "bool";
        case ColumnType::String: return "string";
        case ColumnType::Set:    return "set";
    }
    return "int32";
}

ColumnType column_type_from_string(std::string_view s) {
    if (s == "int32")  return ColumnType::Int32;
    if (s == "int64")  return ColumnType::Int64;
    if (s == "float")  return ColumnType::Float;
    if (s == "bool")   return ColumnType::Bool;
    if (s == "string") return ColumnType::String;
    if (s == "set")    return ColumnType::Set;
    return ColumnType::Int32;
}

}  // namespace

std::string generate_tree_uuid() {
    // 128 random bits, v4-shaped (version/variant nibbles set). Two builds
    // must never share an identity, so no deterministic seeding.
    std::random_device rd;
    std::uniform_int_distribution<uint32_t> dist;
    uint32_t b[4] = {dist(rd), dist(rd), dist(rd), dist(rd)};
    b[1] = (b[1] & 0xFFFF'0FFFu) | 0x0000'4000u;  // version 4
    b[2] = (b[2] & 0x3FFF'FFFFu) | 0x8000'0000u;  // variant 10
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(32);
    for (uint32_t w : b)
        for (int shift = 28; shift >= 0; shift -= 4)
            out.push_back(hex[(w >> shift) & 0xFu]);
    return out;
}

std::string manifest_to_toml(const TreeManifest& m) {
    auto root = cpptoml::make_table();

    // [meta]  (uuid is mandatory — the format is pre-freeze, no legacy)
    if (m.uuid.size() != 32) {
        throw Error(ErrorCode::CorruptIndex,
                    "TreeManifest: uuid must be set (32 hex chars) before "
                    "serialization — builds must call generate_tree_uuid()");
    }
    auto meta = cpptoml::make_table();
    meta->insert("format_version", kManifestFormatVersion);
    meta->insert("uuid", m.uuid);
    root->insert("meta", meta);

    // [index]
    auto index = cpptoml::make_table();
    index->insert("dim", m.dim);
    index->insert("m4", m.m4);
    index->insert("scan_pq_bits", static_cast<uint64_t>(m.scan_pq_bits));
    index->insert("quantizer_type", m.quantizer_type);
    index->insert("prq_nsplits", m.prq_nsplits);
    index->insert("metric", static_cast<int64_t>(m.metric));
    index->insert("scalar_row_bias", m.scalar_row_bias);
    root->insert("index", index);

    // [tree]
    auto tree = cpptoml::make_table();
    tree->insert("depth", m.depth);
    tree->insert("k_root", m.k_root);
    tree->insert("k_l1", m.k_l1);
    tree->insert("leaf_capacity", m.leaf_capacity);
    tree->insert("n_vectors", m.n_vectors);
    tree->insert("n_leaves", m.n_leaves);
    tree->insert("n_probe_l0", m.n_probe_l0);
    tree->insert("n_probe_ln", m.n_probe_ln);
    root->insert("tree", tree);

    // [routing]
    auto routing = cpptoml::make_table();
    routing->insert("adaptive_probe_gap", static_cast<double>(m.adaptive_probe_gap));
    routing->insert("median_lid", static_cast<double>(m.median_lid));
    routing->insert("pca_dims", static_cast<int64_t>(m.pca_dims));
    if (m.probe_fraction > 0.0f) {
        routing->insert("probe_fraction", static_cast<double>(m.probe_fraction));
    }
    root->insert("routing", routing);

    // [partition]
    auto partition = cpptoml::make_table();
    partition->insert("balance_factor", static_cast<double>(m.balance_factor));
    root->insert("partition", partition);

    // [schema]  (only when non-empty)
    if (!m.schema.empty()) {
        auto schema_tbl = cpptoml::make_table();
        schema_tbl->insert("has_payload", m.schema.has_payload);
        schema_tbl->insert("summary_size", m.summary_size);
        auto cols = cpptoml::make_table_array();
        for (const auto& col : m.schema.columns) {
            auto ct = cpptoml::make_table();
            ct->insert("name", col.name);
            ct->insert("type", std::string(column_type_to_string(col.type)));
            if (col.nullable) {
                ct->insert("nullable", true);
            }
            cols->push_back(std::move(ct));
        }
        schema_tbl->insert("filter_columns", cols);
        root->insert("schema", schema_tbl);
    }

    std::ostringstream ss;
    ss << *root;
    return ss.str();
}

TreeManifest manifest_from_toml(const std::string& toml) {
    std::istringstream ss(toml);
    cpptoml::parser p(ss);
    auto root = p.parse();

    TreeManifest m;

    // [meta]  (uuid is mandatory — the format is pre-freeze, no legacy)
    auto meta = root->get_table("meta");
    if (!meta) {
        throw Error(ErrorCode::CorruptIndex, "TreeManifest: missing [meta] section (uuid required)");
    }
    auto fv = meta->get_as<int64_t>("format_version");
    if (!fv) {
        throw Error(ErrorCode::CorruptIndex,
                    "TreeManifest: missing required field format_version");
    }
    if (*fv != kManifestFormatVersion) {
        throw Error(ErrorCode::CorruptIndex,
                    "TreeManifest: format_version " + std::to_string(*fv) +
                    " unsupported (expected " +
                    std::to_string(kManifestFormatVersion) + ") — rebuild the index");
    }
    auto uuid = meta->get_as<std::string>("uuid");
    if (!uuid) {
        throw Error(ErrorCode::CorruptIndex, "TreeManifest: missing required field uuid");
    }
    m.uuid = *uuid;
    if (m.uuid.size() != 32 ||
        m.uuid.find_first_not_of("0123456789abcdef") != std::string::npos) {
        throw Error(ErrorCode::CorruptIndex,
                    "TreeManifest: uuid must be 32 lowercase hex chars, got '" + m.uuid + "'");
    }

    auto require_int = [&](const cpptoml::table& tbl,
                           const char* key) -> int64_t {
        auto val = tbl.get_as<int64_t>(key);
        if (!val) {
            throw Error(ErrorCode::CorruptIndex,
                        std::string("TreeManifest: missing required field ") + key);
        }
        return *val;
    };

    // [index]
    auto index = root->get_table("index");
    if (!index) {
        throw Error(ErrorCode::CorruptIndex, "TreeManifest: missing [index] section");
    }
    m.dim = static_cast<uint32_t>(require_int(*index, "dim"));
    m.m4 = static_cast<uint16_t>(require_int(*index, "m4"));
    m.scan_pq_bits = static_cast<uint8_t>(require_int(*index, "scan_pq_bits"));
    auto qt = index->get_as<std::string>("quantizer_type");
    if (!qt) {
        throw Error(ErrorCode::CorruptIndex,
                    "TreeManifest: missing required field quantizer_type");
    }
    m.quantizer_type = *qt;
    m.prq_nsplits = static_cast<uint32_t>(require_int(*index, "prq_nsplits"));
    m.metric = static_cast<uint8_t>(require_int(*index, "metric"));
    auto rb = index->get_as<bool>("scalar_row_bias");
    if (!rb) {
        throw Error(ErrorCode::CorruptIndex,
                    "TreeManifest: missing required field scalar_row_bias");
    }
    m.scalar_row_bias = *rb;

    // [tree]
    auto tree = root->get_table("tree");
    if (!tree) {
        throw Error(ErrorCode::CorruptIndex, "TreeManifest: missing [tree] section");
    }
    m.depth = static_cast<uint16_t>(require_int(*tree, "depth"));
    m.k_root = static_cast<uint32_t>(require_int(*tree, "k_root"));
    m.k_l1 = static_cast<uint32_t>(require_int(*tree, "k_l1"));
    m.leaf_capacity = static_cast<uint32_t>(require_int(*tree, "leaf_capacity"));
    m.n_vectors = static_cast<uint64_t>(require_int(*tree, "n_vectors"));
    m.n_leaves = static_cast<uint32_t>(require_int(*tree, "n_leaves"));
    m.n_probe_l0 = static_cast<uint32_t>(require_int(*tree, "n_probe_l0"));
    m.n_probe_ln = static_cast<uint32_t>(require_int(*tree, "n_probe_ln"));

    // [routing]  (probe_fraction stays optional: 0 = count routing is a
    // live setting, not a legacy read)
    auto routing = root->get_table("routing");
    if (!routing) {
        throw Error(ErrorCode::CorruptIndex,
                    "TreeManifest: missing [routing] section");
    }
    {
        auto gap = routing->get_as<double>("adaptive_probe_gap");
        if (!gap) {
            throw Error(ErrorCode::CorruptIndex,
                        "TreeManifest: missing required field adaptive_probe_gap");
        }
        m.adaptive_probe_gap = static_cast<float>(*gap);
        auto lid = routing->get_as<double>("median_lid");
        if (!lid) {
            throw Error(ErrorCode::CorruptIndex,
                        "TreeManifest: missing required field median_lid");
        }
        m.median_lid = static_cast<float>(*lid);
        m.pca_dims = static_cast<uint32_t>(require_int(*routing, "pca_dims"));
        if (auto pf = routing->get_as<double>("probe_fraction")) {
            m.probe_fraction = static_cast<float>(*pf);
        }
    }

    // [partition]
    auto partition = root->get_table("partition");
    if (!partition) {
        throw Error(ErrorCode::CorruptIndex,
                    "TreeManifest: missing [partition] section");
    }
    {
        auto bf = partition->get_as<double>("balance_factor");
        if (!bf) {
            throw Error(ErrorCode::CorruptIndex,
                        "TreeManifest: missing required field balance_factor");
        }
        m.balance_factor = static_cast<float>(*bf);
    }

    // [schema]  (optional; absent → empty schema, summary_size 0)
    if (auto schema_tbl = root->get_table("schema")) {
        if (auto hp = schema_tbl->get_as<bool>("has_payload")) {
            m.schema.has_payload = *hp;
        }
        if (auto ss = schema_tbl->get_as<int64_t>("summary_size")) {
            m.summary_size = static_cast<uint32_t>(*ss);
        }
        if (auto cols = schema_tbl->get_table_array("filter_columns")) {
            for (const auto& ct : *cols) {
                FilterColumn col;
                if (auto name = ct->get_as<std::string>("name")) {
                    col.name = *name;
                }
                if (auto type = ct->get_as<std::string>("type")) {
                    col.type = column_type_from_string(*type);
                }
                if (auto nullable = ct->get_as<bool>("nullable")) {
                    col.nullable = *nullable;
                }
                m.schema.columns.push_back(std::move(col));
            }
        }
    }

    return m;
}

}  // namespace sextant::tree
