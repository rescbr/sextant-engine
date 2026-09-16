#include "tree_manifest.hpp"

#include "sextant/error.hpp"

#include <cpptoml.h>

#include <sstream>

namespace sextant::tree {

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

std::string manifest_to_toml(const TreeManifest& m) {
    auto root = cpptoml::make_table();

    // [index]
    auto index = cpptoml::make_table();
    index->insert("dim", m.dim);
    index->insert("m4", m.m4);
    index->insert("scan_pq_bits", static_cast<uint64_t>(m.scan_pq_bits));
    index->insert("quantizer_type", m.quantizer_type);
    index->insert("prq_nsplits", m.prq_nsplits);
    index->insert("metric", static_cast<int64_t>(m.metric));
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
    if (auto qt = index->get_as<std::string>("quantizer_type")) {
        m.quantizer_type = *qt;
    }
    if (auto ns = index->get_as<int64_t>("prq_nsplits")) {
        m.prq_nsplits = static_cast<uint32_t>(*ns);
    }
    if (auto mt = index->get_as<int64_t>("metric")) {
        m.metric = static_cast<uint8_t>(*mt);
    }

    // [tree]
    auto tree = root->get_table("tree");
    if (!tree) {
        throw Error(ErrorCode::CorruptIndex, "TreeManifest: missing [tree] section");
    }
    m.depth = static_cast<uint16_t>(require_int(*tree, "depth"));
    m.k_root = static_cast<uint32_t>(require_int(*tree, "k_root"));
    // k_l1 is optional (depth-3 trees; older trees omit it → 0).
    if (auto kl1 = tree->get_as<int64_t>("k_l1")) {
        m.k_l1 = static_cast<uint32_t>(*kl1);
    }
    m.leaf_capacity = static_cast<uint32_t>(require_int(*tree, "leaf_capacity"));
    // n_vectors is optional (older manifests omit it → 0 = unknown).
    if (auto nv = tree->get_as<int64_t>("n_vectors")) {
        m.n_vectors = static_cast<uint64_t>(*nv);
    }
    m.n_leaves = static_cast<uint32_t>(require_int(*tree, "n_leaves"));
    m.n_probe_l0 = static_cast<uint32_t>(require_int(*tree, "n_probe_l0"));
    m.n_probe_ln = static_cast<uint32_t>(require_int(*tree, "n_probe_ln"));

    // [routing]
    auto routing = root->get_table("routing");
    if (routing) {
        if (auto gap = routing->get_as<double>("adaptive_probe_gap")) {
            m.adaptive_probe_gap = static_cast<float>(*gap);
        }
        if (auto lid = routing->get_as<double>("median_lid")) {
            m.median_lid = static_cast<float>(*lid);
        }
        if (auto pd = routing->get_as<int64_t>("pca_dims")) {
            m.pca_dims = static_cast<uint32_t>(*pd);
        }
        if (auto pf = routing->get_as<double>("probe_fraction")) {
            m.probe_fraction = static_cast<float>(*pf);
        }
    }

    // [partition]
    auto partition = root->get_table("partition");
    if (partition) {
        if (auto bf = partition->get_as<double>("balance_factor")) {
            m.balance_factor = static_cast<float>(*bf);
        }
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
                m.schema.columns.push_back(std::move(col));
            }
        }
    }

    return m;
}

}  // namespace sextant::tree
