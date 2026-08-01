#include "tree_manifest.hpp"

#include "sextant/error.hpp"

#include <cpptoml.h>

#include <sstream>

namespace sextant::tree {

std::string manifest_to_toml(const TreeManifest& m) {
    auto root = cpptoml::make_table();

    // [index]
    auto index = cpptoml::make_table();
    index->insert("dim", m.dim);
    index->insert("m4", m.m4);
    index->insert("scan_pq_bits", static_cast<uint64_t>(m.scan_pq_bits));
    index->insert("quantizer_type", m.quantizer_type);
    index->insert("prq_nsplits", m.prq_nsplits);
    root->insert("index", index);

    // [tree]
    auto tree = cpptoml::make_table();
    tree->insert("depth", m.depth);
    tree->insert("k_root", m.k_root);
    tree->insert("leaf_capacity", m.leaf_capacity);
    tree->insert("n_leaves", m.n_leaves);
    tree->insert("n_probe_l0", m.n_probe_l0);
    tree->insert("n_probe_ln", m.n_probe_ln);
    root->insert("tree", tree);

    // [routing]
    auto routing = cpptoml::make_table();
    routing->insert("adaptive_probe_gap", static_cast<double>(m.adaptive_probe_gap));
    routing->insert("median_lid", static_cast<double>(m.median_lid));
    routing->insert("pca_dims", static_cast<int64_t>(m.pca_dims));
    root->insert("routing", routing);

    // [partition]
    auto partition = cpptoml::make_table();
    partition->insert("balance_factor", static_cast<double>(m.balance_factor));
    partition->insert("sub_shard_probe_pct", m.sub_shard_probe_pct);
    root->insert("partition", partition);

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

    // [tree]
    auto tree = root->get_table("tree");
    if (!tree) {
        throw Error(ErrorCode::CorruptIndex, "TreeManifest: missing [tree] section");
    }
    m.depth = static_cast<uint16_t>(require_int(*tree, "depth"));
    m.k_root = static_cast<uint32_t>(require_int(*tree, "k_root"));
    m.leaf_capacity = static_cast<uint32_t>(require_int(*tree, "leaf_capacity"));
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
    }

    // [partition]
    auto partition = root->get_table("partition");
    if (partition) {
        if (auto bf = partition->get_as<double>("balance_factor")) {
            m.balance_factor = static_cast<float>(*bf);
        }
        if (auto ssp = partition->get_as<int64_t>("sub_shard_probe_pct")) {
            m.sub_shard_probe_pct = static_cast<uint32_t>(*ssp);
        }
    }

    return m;
}

}  // namespace sextant::tree
