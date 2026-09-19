#include "sextant/sextant_c.h"

/// @file sextant_c.cpp — see sextant_c.h for the ABI contract.

#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <string>

#include "fbin_source.hpp"
#include "mem_source.hpp"
#include "tree/ivf_tree_index.hpp"
#include "tree/scan_pool.hpp"

#include "push_source.hpp"
#include <sextant/metrics.hpp>

#include <sextant/column_data.hpp>
#include <sextant/schema.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <vector>

namespace {

/// Copy an exception message into the caller's error buffer.
void set_err(char* err, size_t err_len, const char* msg) {
    if (err && err_len > 0) {
        std::snprintf(err, err_len, "%s", msg);
    }
}

using sextant::tree::IVFTreeIndex;

/// Translate sextant_search_opts into the engine SearchConfig (shared by
/// sextant_search / _filtered / search2 / search_batch).
void fill_search_config(const sextant_search_opts* opts,
                         sextant::SearchConfig& cfg) {
    cfg.k = opts->k;
    if (opts->exhaustive) {
        // Probe-all: every root child, every internal child, no gap
        // pruning, no code budget. The clamp `min(n_probe, k_root)` makes
        // UINT32_MAX mean "all leaves" (see search()'s level-0/ln paths).
        cfg.n_probe = UINT32_MAX;
        cfg.n_probe_ln = UINT32_MAX;
        cfg.adaptive_probe_gap = -1.0f;  // <0 = off
    } else {
        cfg.n_probe = opts->n_probe;
        cfg.n_probe_ln = opts->n_probe_ln;
        cfg.probe_fraction = opts->probe_fraction;
        cfg.adaptive_probe_gap = opts->adaptive_probe_gap;
    }
    cfg.fastscan_W = opts->fastscan_W;
    cfg.rerank = opts->rerank != 0;
    cfg.exact_rerank_base = opts->exact_rerank_base;
    cfg.adaptive_w_gap = opts->adaptive_w_gap;
    cfg.search_threads = opts->search_threads;
    // The scan kernel mode is a process-wide setting with a per-thread
    // override the scan reads (set below).
    sextant::tree::set_scan_i8_override(opts->int8_scan);
}

/// Translate one C predicate into the engine Predicate. `who` prefixes
/// error messages. Returns 0 on success, negative on failure (msg set).
int translate_one(const sextant_predicate& p, sextant::Predicate& ep,
                  const char* who, char* err, size_t err_len) {
    if (!p.column) {
        set_err(err, err_len, "predicate: null column name");
        return -1;
    }
    ep.column = p.column;
    ep.value = p.value;
    ep.value2 = p.value2;
    if (p.str_value) ep.str_value = p.str_value;

    const auto append_values = [&](bool allow_empty) {
        if (p.n_values == 0) {
            if (allow_empty) return true;
            set_err(err, err_len,
                    "predicate: values list required (n_values == 0)");
            return false;
        }
        if (!p.values || !p.value_lengths) {
            set_err(err, err_len,
                    "predicate: values/value_lengths required with n_values");
            return false;
        }
        ep.values.reserve(ep.values.size() + p.n_values);
        for (uint32_t i = 0; i < p.n_values; ++i)
            ep.values.emplace_back(p.values[i], p.value_lengths[i]);
        return true;
    };

    switch (p.op) {
        case SEXTANT_PRED_EQ:      ep.op = sextant::PredicateOp::Eq; break;
        case SEXTANT_PRED_NEQ:     ep.op = sextant::PredicateOp::NotEq; break;
        case SEXTANT_PRED_LT:      ep.op = sextant::PredicateOp::Lt; break;
        case SEXTANT_PRED_LE:      ep.op = sextant::PredicateOp::Le; break;
        case SEXTANT_PRED_GT:      ep.op = sextant::PredicateOp::Gt; break;
        case SEXTANT_PRED_GE:      ep.op = sextant::PredicateOp::Ge; break;
        case SEXTANT_PRED_BETWEEN: ep.op = sextant::PredicateOp::Between; break;
        case SEXTANT_PRED_PREFIX:  ep.op = sextant::PredicateOp::Prefix; break;
        case SEXTANT_PRED_IN:
        case SEXTANT_PRED_NOT_IN:
            ep.op = (p.op == SEXTANT_PRED_IN) ? sextant::PredicateOp::In
                                              : sextant::PredicateOp::NotIn;
            // Membership is defined solely by the values list (the engine's
            // legacy numeric value..value4 comparand fast-path was removed;
            // scalar slots are not consulted for In/NotIn).
            if (!append_values(/*allow_empty=*/p.op == SEXTANT_PRED_NOT_IN))
                return -1;
            break;
        case SEXTANT_PRED_CONTAINS:
            // Engine convention (CLI): the sought element lives in str_value.
            if (!p.str_value) {
                set_err(err, err_len,
                        "predicate: CONTAINS requires str_value");
                return -1;
            }
            ep.op = sextant::PredicateOp::Contains;
            break;
        case SEXTANT_PRED_CONTAINS_ANY:
        case SEXTANT_PRED_CONTAINS_ALL:
            // Engine convention: str_value (optional) + values are folded
            // into one query set.
            if (p.n_values == 0 && !p.str_value) {
                set_err(err, err_len,
                        "predicate: CONTAINS_ANY/CONTAINS_ALL require "
                        "str_value or a non-empty values list");
                return -1;
            }
            ep.op = (p.op == SEXTANT_PRED_CONTAINS_ANY)
                        ? sextant::PredicateOp::ContainsAny
                        : sextant::PredicateOp::ContainsAll;
            if (!append_values(/*allow_empty=*/true)) return -1;
            break;
        case SEXTANT_PRED_GEO_RADIUS:
            if (!p.geo_lng_column) {
                set_err(err, err_len,
                        "predicate: GEO_RADIUS requires geo_lng_column");
                return -1;
            }
            ep.op = sextant::PredicateOp::GeoRadius;
            ep.geo_lng_column = p.geo_lng_column;
            ep.radius_km = p.radius_km;  // value/value2 = center lat/lng
            break;
        case SEXTANT_PRED_GEO_BOX:
            if (!p.geo_lng_column) {
                set_err(err, err_len,
                        "predicate: GEO_BOX requires geo_lng_column");
                return -1;
            }
            ep.op = sextant::PredicateOp::GeoBox;
            ep.geo_lng_column = p.geo_lng_column;
            ep.value3 = p.value3;  // max_lat
            ep.value4 = p.value4;  // max_lng
            break;
        case SEXTANT_PRED_IS_NULL:
            ep.op = sextant::PredicateOp::IsNull;
            break;
        case SEXTANT_PRED_IS_NOT_NULL:
            ep.op = sextant::PredicateOp::IsNotNull;
            break;
        default:
            set_err(err, err_len, "predicate: unknown op value");
            return -1;
    }
    (void)who;
    return 0;
}

/// Translate a C predicate conjunct list into engine Predicates (shared by
/// sextant_search_filtered / search2 / search_batch).
/// Returns 0 on success, negative on failure (message set).
int translate_predicates(const sextant_predicate* preds, uint32_t n_preds,
                          std::vector<sextant::Predicate>& out,
                          char* err, size_t err_len) {
    out.clear();
    out.reserve(n_preds);
    for (uint32_t i = 0; i < n_preds; ++i) {
        sextant::Predicate ep;
        if (int rc = translate_one(preds[i], ep, "predicate", err, err_len))
            return rc;
        out.push_back(std::move(ep));
    }
    return 0;
}

/// Common argument validation for the single-query search entry points.
bool validate_search_args(const IVFTreeIndex* idx, const float* query,
                           const sextant_search_opts* opts,
                           const uint64_t* out_row_ids, const float* out_dists,
                           char* err, size_t err_len, const char* who) {
    if (!idx || !query || !opts || !out_row_ids || !out_dists || opts->k == 0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s: null argument or k==0", who);
        set_err(err, err_len, buf);
        return false;
    }
    return true;
}

/// Streaming push-build state. Buffers vectors (via MemSourceBuilder),
/// filter columns (ColumnData, indexed by global row id) and payload blobs,
/// then feeds build_streaming_pca at finish — mirroring the FbinSource +
/// BuildConfig sidecar path used by the filter-column tests.
struct PushBuilder {
    uint32_t dim = 0;
    sextant_build_opts opts = {};
    sextant::Schema schema;  // declared filter columns
    bool has_payload = false;
    uint64_t rows = 0;
    // Bounded-memory staging (spills to a temp file on overflow); the
    // build streams the staged chunks like a parquet source.
    std::unique_ptr<sextant::PushStager> stager;
};

}  // namespace

extern "C" {

sextant_build_opts sextant_default_build_opts(void) {
    sextant_build_opts o{};
    // local_scalar is the product default: best recall-per-byte on
    // embedding corpora (~0.77 KB/vec at 4 bits), no global training
    // sample, and append/rebalance-safe (rulers re-fit per leaf at flush).
    o.quantizer = "local_scalar";
    o.pq_m = 0;
    o.k_root = 0;
    o.leaf_capacity = 0;
    o.pca_dims = 0;
    o.num_threads = 0;
    o.max_lloyd_passes = 0;
    o.metric = SEXTANT_METRIC_L2SQ;
    o.adaptive_probe_gap = 0.0f;
    o.plane_attach = 1;  // CLI serving parity: routing plane on
    o.plane_rank = 0;    // 0 = default rank 128 at finish
    return o;
}

int sextant_build_fbin(const char* fbin_path, const char* out_path,
                       const sextant_build_opts* opts,
                       char* err, size_t err_len) {
    if (!fbin_path || !out_path || !opts || !opts->quantizer) {
        set_err(err, err_len, "sextant_build_fbin: null argument");
        return -1;
    }
    try {
        sextant::FbinSource source(fbin_path);
        sextant::tree::IVFTreeIndex::BuildConfig cfg;
        cfg.k_root = opts->k_root;
        cfg.leaf_capacity = opts->leaf_capacity ? opts->leaf_capacity : 5000;
        cfg.pca_dims = opts->pca_dims ? opts->pca_dims : 32;
        cfg.num_threads = opts->num_threads;
        cfg.max_lloyd_passes = opts->max_lloyd_passes ? opts->max_lloyd_passes : 10;
        cfg.adaptive_probe_gap = opts->adaptive_probe_gap;
        cfg.params.quantizer_type = opts->quantizer;
        cfg.params.pq4_m = static_cast<uint16_t>(opts->pq_m);
        cfg.params.metric = opts->metric == SEXTANT_METRIC_IP
            ? sextant::MetricKind::InnerProduct
            : sextant::MetricKind::L2Sq;
        sextant::tree::IVFTreeIndex::build_streaming_pca(source, out_path, cfg);
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return -2;
    } catch (...) {
        set_err(err, err_len, "sextant_build_fbin: unknown exception");
        return -3;
    }
}

int sextant_build_mem(const float* vectors, uint32_t n, uint32_t dim,
                      const char* out_path,
                      const sextant_build_opts* opts,
                      char* err, size_t err_len) {
    if (!vectors || !out_path || !opts || !opts->quantizer || n == 0 || dim == 0) {
        set_err(err, err_len, "sextant_build_mem: null/zero argument");
        return -1;
    }
    try {
        sextant::MemSourceBuilder builder(dim);
        for (uint32_t i = 0; i < n; ++i) {
            builder.add_vector(vectors + static_cast<size_t>(i) * dim, i);
        }
        auto source = builder.build();
        sextant::tree::IVFTreeIndex::BuildConfig cfg;
        cfg.k_root = opts->k_root;
        cfg.leaf_capacity = opts->leaf_capacity ? opts->leaf_capacity : 5000;
        cfg.pca_dims = opts->pca_dims ? opts->pca_dims : 32;
        cfg.num_threads = opts->num_threads;
        cfg.max_lloyd_passes = opts->max_lloyd_passes ? opts->max_lloyd_passes : 10;
        cfg.adaptive_probe_gap = opts->adaptive_probe_gap;
        cfg.params.quantizer_type = opts->quantizer;
        cfg.params.pq4_m = static_cast<uint16_t>(opts->pq_m);
        cfg.params.metric = opts->metric == SEXTANT_METRIC_IP
            ? sextant::MetricKind::InnerProduct
            : sextant::MetricKind::L2Sq;
        sextant::tree::IVFTreeIndex::build_streaming_pca(*source, out_path, cfg);
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return -2;
    } catch (...) {
        set_err(err, err_len, "sextant_build_mem: unknown exception");
        return -3;
    }
}

void* sextant_open_index(const char* path, char* err, size_t err_len) {
    if (!path) {
        set_err(err, err_len, "sextant_open_index: null path");
        return nullptr;
    }
    try {
        return sextant::tree::IVFTreeIndex::open(path).release();
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return nullptr;
    } catch (...) {
        set_err(err, err_len, "sextant_open_index: unknown exception");
        return nullptr;
    }
}

void sextant_close_index(void* index) {
    delete reinterpret_cast<sextant::tree::IVFTreeIndex*>(index);
}

uint32_t sextant_index_dim(const void* index) {
    auto* idx = reinterpret_cast<const sextant::tree::IVFTreeIndex*>(index);
    return idx ? idx->dim() : 0;
}

int32_t sextant_index_uuid(const void* index, char* out, size_t out_len) {
    auto* idx = reinterpret_cast<const sextant::tree::IVFTreeIndex*>(index);
    if (!idx || !out) return -1;
    const std::string& uuid = idx->tree_uuid();
    if (uuid.empty()) return 0;  // pre-UUID legacy tree
    if (out_len < uuid.size() + 1) return -1;
    std::memcpy(out, uuid.c_str(), uuid.size() + 1);
    return static_cast<int32_t>(uuid.size());
}

uint64_t sextant_index_live_count(void* index) {
    auto* idx = reinterpret_cast<sextant::tree::IVFTreeIndex*>(index);
    return idx ? idx->live_count() : 0;
}

sextant_search_opts sextant_default_search_opts(void) {
    sextant_search_opts o{};
    o.k = 10;
    o.n_probe = 0;
    o.n_probe_ln = 0;
    o.fastscan_W = 0;
    o.adaptive_probe_gap = 0.0f;
    // Rerank by default, matching the CLI serving default: int8_scan auto
    // (on) + rerank off produces raw i8 scores whose ranking is lossy —
    // rerank-on is the quality contract callers expect from a default.
    o.rerank = 1;
    o.search_threads = 0;
    o.int8_scan = -1;  // -1 = auto (int8 on AVX512/VNNI); 0 here previously
                        // force-disabled the i8 kernel for default callers
    o.exhaustive = 0;
    o.probe_fraction = 0.0f;
    return o;
}

int32_t sextant_search(void* index, const float* query,
                       const sextant_search_opts* opts,
                       uint64_t* out_row_ids, float* out_dists,
                       uint32_t out_capacity, char* err, size_t err_len) {
    auto* idx = reinterpret_cast<IVFTreeIndex*>(index);
    if (!validate_search_args(idx, query, opts, out_row_ids, out_dists,
                             err, err_len, "sextant_search"))
        return -1;
    if (out_capacity < opts->k) {
        set_err(err, err_len, "sextant_search: out_capacity < k");
        return -1;
    }
    try {
        sextant::SearchConfig cfg;
        fill_search_config(opts, cfg);
        auto results = idx->search(query, opts->k, cfg);
        // Under the adaptive-W contract (adaptive_w_gap > 0) the engine may
        // return more than k ids. The caller's arrays are sized k, so clamp;
        // a harness wanting the full shortlist passes k = fastscan_W.
        const int32_t n = std::min<int32_t>(static_cast<int32_t>(results.size()),
                                           static_cast<int32_t>(out_capacity));
        for (int32_t i = 0; i < n; ++i) {
            out_row_ids[i] = static_cast<uint64_t>(results[i].row_id);
            out_dists[i] = results[i].dist;
        }
        return n;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return -2;
    } catch (...) {
        set_err(err, err_len, "sextant_search: unknown exception");
        return -3;
    }
}

int32_t sextant_search_filtered(void* index, const float* query,
                                const sextant_search_opts* opts,
                                const sextant_predicate* preds,
                                uint32_t n_preds,
                                uint64_t* out_row_ids, float* out_dists,
                                uint32_t out_capacity,
                                char* err, size_t err_len) {
    return sextant_search2(index, query, opts, preds, n_preds,
                          out_row_ids, out_dists, nullptr, nullptr,
                          out_capacity, err, err_len);
}

int32_t sextant_search2(void* index, const float* query,
                        const sextant_search_opts* opts,
                        const sextant_predicate* preds, uint32_t n_preds,
                        uint64_t* out_row_ids, float* out_dists,
                        void** out_leaf_ptrs, uint32_t* out_slots,
                        uint32_t out_capacity, char* err, size_t err_len) {
    auto* idx = reinterpret_cast<IVFTreeIndex*>(index);
    if (!validate_search_args(idx, query, opts, out_row_ids, out_dists,
                             err, err_len, "sextant_search2"))
        return -1;
    if (out_capacity < opts->k) {
        set_err(err, err_len, "sextant_search2: out_capacity < k");
        return -1;
    }
    try {
        sextant::SearchConfig cfg;
        fill_search_config(opts, cfg);
        if (n_preds) {
            if (int rc = translate_predicates(preds, n_preds,
                                             cfg.predicates, err, err_len))
                return rc;
        }
        std::vector<std::pair<const uint8_t*, uint32_t>> locs;
        std::vector<std::pair<const uint8_t*, uint32_t>>* locs_ptr =
            (out_leaf_ptrs && out_slots) ? &locs : nullptr;
        auto results = idx->search(query, opts->k, cfg, locs_ptr);
        const int32_t n = std::min<int32_t>(static_cast<int32_t>(results.size()),
                                           static_cast<int32_t>(out_capacity));
        for (int32_t i = 0; i < n; ++i) {
            out_row_ids[i] = static_cast<uint64_t>(results[i].row_id);
            out_dists[i] = results[i].dist;
            if (locs_ptr) {
                out_leaf_ptrs[i] = const_cast<void*>(
                    static_cast<const void*>(locs[i].first));
                out_slots[i] = locs[i].second;
            }
        }
        return n;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return -2;
    } catch (...) {
        set_err(err, err_len, "sextant_search2: unknown exception");
        return -3;
    }
}

int sextant_search_batch(void* index, const float* queries,
                         uint32_t n_queries,
                         const sextant_search_opts* opts,
                         const sextant_predicate* preds, uint32_t n_preds,
                         uint64_t* out_row_ids, float* out_dists,
                         uint32_t out_capacity_per_query,
                         uint32_t* out_n_per_query,
                         char* err, size_t err_len) {
    auto* idx = reinterpret_cast<IVFTreeIndex*>(index);
    if (!idx || !queries || !opts || !out_row_ids || !out_dists
        || !out_n_per_query || n_queries == 0 || opts->k == 0
        || out_capacity_per_query < opts->k) {
        set_err(err, err_len,
                "sextant_search_batch: null argument, n_queries==0 or k==0");
        return -1;
    }
    try {
        sextant::SearchConfig cfg;
        fill_search_config(opts, cfg);
        if (n_preds) {
            if (int rc = translate_predicates(preds, n_preds,
                                             cfg.predicates, err, err_len))
                return rc;
        }
        std::vector<std::vector<sextant::Candidate>> results;
        idx->search_batch(queries, n_queries, opts->k, cfg, results);
        for (uint32_t q = 0; q < n_queries; ++q) {
            const uint32_t n = std::min<uint32_t>(
                static_cast<uint32_t>(results[q].size()),
                out_capacity_per_query);
            uint64_t* rid = out_row_ids + static_cast<size_t>(q)
                           * out_capacity_per_query;
            float* ds = out_dists + static_cast<size_t>(q)
                       * out_capacity_per_query;
            for (uint32_t i = 0; i < n; ++i) {
                rid[i] = static_cast<uint64_t>(results[q][i].row_id);
                ds[i] = results[q][i].dist;
            }
            out_n_per_query[q] = n;
        }
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return -2;
    } catch (...) {
        set_err(err, err_len, "sextant_search_batch: unknown exception");
        return -3;
    }
}

uint32_t sextant_fetch_payload(const void* index, const void* leaf_ptr,
                               uint32_t slot, void* out_buf,
                               uint32_t buf_capacity) {
    auto* idx = reinterpret_cast<const IVFTreeIndex*>(index);
    if (!idx || !leaf_ptr) return 0;
    try {
        auto blob = idx->fetch_payload(
            static_cast<const uint8_t*>(leaf_ptr), slot);
        const uint32_t size = static_cast<uint32_t>(blob.size());
        if (out_buf && buf_capacity > 0)
            std::memcpy(out_buf, blob.data(),
                       std::min<size_t>(size, buf_capacity));
        return size;
    } catch (...) {
        return 0;  // never throw across the C boundary
    }
}

int sextant_fetch_vector(const void* index, const void* leaf_ptr,
                         uint32_t slot, float* out) {
    auto* idx = reinterpret_cast<const IVFTreeIndex*>(index);
    if (!idx || !leaf_ptr || !out) return -1;
    try {
        return idx->fetch_vector(static_cast<const uint8_t*>(leaf_ptr),
                                slot, out)
                   ? 0 : -2;
    } catch (const std::exception& e) {
        // No err buffer in this signature; the negative return is the
        // failure signal (kept minimal per the ABI contract).
        (void)e;
        return -3;
    } catch (...) {
        return -3;
    }
}

uint32_t sextant_scan_pool_set_threads(uint32_t n) {
    try {
        return sextant::tree::scan_pool_set_threads(n);
    } catch (...) {
        return sextant::tree::scan_pool_threads();
    }
}

uint32_t sextant_scan_pool_threads(void) {
    return sextant::tree::scan_pool_threads();
}

uint32_t sextant_index_code_size(const void* index) {
    auto* idx = reinterpret_cast<const IVFTreeIndex*>(index);
    if (!idx) return 0;
    try {
        return idx->coder().code_size();
    } catch (...) {
        return 0;
    }
}

int sextant_index_metric(const void* index) {
    if (!index) return -1;
    try {
        const auto& tree = *reinterpret_cast<const sextant::tree::IVFTreeIndex*>(index);
        return tree.metric() == sextant::MetricKind::InnerProduct
                   ? SEXTANT_METRIC_IP
                   : SEXTANT_METRIC_L2SQ;
    } catch (...) {
        return -2;
    }
}

uint32_t sextant_index_filter_col_count(const void* index) {
    if (!index) return 0;
    try {
        const auto& tree = *reinterpret_cast<const sextant::tree::IVFTreeIndex*>(index);
        return static_cast<uint32_t>(tree.filter_schema().columns.size());
    } catch (...) {
        return 0;
    }
}

int32_t sextant_index_filter_col(const void* index, uint32_t i,
                                 char* out_name, size_t name_len,
                                 int* out_type, int* out_nullable) {
    if (!index || !out_name || name_len == 0) return -1;
    try {
        const auto& tree = *reinterpret_cast<const sextant::tree::IVFTreeIndex*>(index);
        const auto& cols = tree.filter_schema().columns;
        if (i >= cols.size()) return -2;
        const auto& col = cols[i];
        if (col.name.size() + 1 > name_len) return -3;
        std::memcpy(out_name, col.name.c_str(), col.name.size() + 1);
        if (out_nullable) {
            *out_nullable = col.nullable ? 1 : 0;
        }
        if (out_type) {
            switch (col.type) {
                case sextant::ColumnType::Int32:  *out_type = SEXTANT_COL_INT32;  break;
                case sextant::ColumnType::Int64:  *out_type = SEXTANT_COL_INT64;  break;
                case sextant::ColumnType::Float:  *out_type = SEXTANT_COL_FLOAT;  break;
                case sextant::ColumnType::String: *out_type = SEXTANT_COL_STRING; break;
                case sextant::ColumnType::Set:    *out_type = SEXTANT_COL_SET;    break;
                case sextant::ColumnType::Bool:   *out_type = SEXTANT_COL_BOOL;   break;
                default: return -4;
            }
        }
        return 0;
    } catch (...) {
        return -5;
    }
}

uint64_t sextant_index_count(const void* index) {
    auto* idx = reinterpret_cast<const IVFTreeIndex*>(index);
    if (!idx) return 0;
    try {
        // Logical row count from the manifest — O(1), counts each input
        // row once (closure-replicated leaf slots are not counted) and
        // reflects the last committed build/mutation. 0 = legacy manifest
        // (pre-n_vectors); fall back to leaf-header sums / cardinality.
        const uint64_t n = idx->n_vectors();
        if (n > 0) return n;
        const uint64_t cn = idx->cardinality().n_vectors();
        if (cn > 0) return cn;
        uint64_t total = 0;
        for (const auto& l : idx->debug_leaf_info()) total += l.count;
        return total;
    } catch (...) {
        return 0;
    }
}

void* sextant_build_begin(const sextant_build_opts* opts, uint32_t dim,
                          const sextant_filter_col_def* cols, uint32_t n_cols,
                          int has_payload, char* err, size_t err_len) {
    if (!opts || !opts->quantizer || dim == 0
        || (!cols && n_cols > 0)) {
        set_err(err, err_len, "sextant_build_begin: null/zero argument");
        return nullptr;
    }
    try {
        auto* b = new PushBuilder(dim);
        b->opts = *opts;
        b->has_payload = has_payload != 0;
        // Temp spill location: TMPDIR (the final output path is unknown
        // until finish). mkstemp for uniqueness.
        std::string spill = (std::getenv("TMPDIR") ? std::getenv("TMPDIR")
                                                   : "/tmp");
        spill += "/sextant_push_stage_XXXXXX";
        std::vector<char> tmpl(spill.begin(), spill.end());
        tmpl.push_back('\0');
        const int fd = ::mkstemp(tmpl.data());
        if (fd < 0) {
            delete b;
            set_err(err, err_len, "sextant_build_begin: cannot create "
                                  "staging temp file");
            return nullptr;
        }
        ::close(fd);
        ::unlink(tmpl.data());  // PushStager creates/owns the real file
        spill = tmpl.data();
        for (uint32_t c = 0; c < n_cols; ++c) {
            if (!cols[c].name) {
                delete b;
                set_err(err, err_len,
                       "sextant_build_begin: filter column with null name");
                return nullptr;
            }
            sextant::ColumnType t;
            switch (cols[c].type) {
                case SEXTANT_COL_INT32:  t = sextant::ColumnType::Int32; break;
                case SEXTANT_COL_INT64:  t = sextant::ColumnType::Int64; break;
                case SEXTANT_COL_FLOAT:  t = sextant::ColumnType::Float; break;
                case SEXTANT_COL_STRING: t = sextant::ColumnType::String; break;
                case SEXTANT_COL_SET:    t = sextant::ColumnType::Set; break;
                case SEXTANT_COL_BOOL:   t = sextant::ColumnType::Bool; break;
                default:
                    delete b;
                    set_err(err, err_len,
                           "sextant_build_begin: unknown column type");
                    return nullptr;
            }
            b->schema.columns.push_back({cols[c].name, t,
                                          cols[c].nullable != 0});
        }
        b->schema.has_payload = b->has_payload;
        b->stager = std::make_unique<sextant::PushStager>(
            dim, b->schema, b->has_payload, std::move(spill),
            opts->staging_bytes);
        return b;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return nullptr;
    } catch (...) {
        set_err(err, err_len, "sextant_build_begin: unknown exception");
        return nullptr;
    }
}

int sextant_build_push(void* builder, const float* vectors, uint32_t n_rows,
                       const void* const* filter_values,
                       const uint64_t* payload_offsets,
                       const uint8_t* payload_data,
                       char* err, size_t err_len) {
    return sextant_build_push_validity(builder, vectors, n_rows,
                                       filter_values, nullptr,
                                       payload_offsets, payload_data,
                                       err, err_len);
}

int sextant_build_push_validity(void* builder, const float* vectors,
                                uint32_t n_rows,
                                const void* const* filter_values,
                                const uint8_t* const* filter_nulls,
                                const uint64_t* payload_offsets,
                                const uint8_t* payload_data,
                                char* err, size_t err_len) {
    auto* b = reinterpret_cast<PushBuilder*>(builder);
    if (!b || !vectors || n_rows == 0) {
        set_err(err, err_len, "sextant_build_push: null/zero argument");
        return -1;
    }
    if (b->has_payload && !payload_offsets) {
        set_err(err, err_len,
                "sextant_build_push: payload_offsets required (has_payload)");
        return -1;
    }
    try {
        auto& st = *b->stager;
        const uint32_t n_cols = static_cast<uint32_t>(b->schema.columns.size());
        for (uint32_t r = 0; r < n_rows; ++r) {
            st.append_vector(vectors + static_cast<size_t>(r) * b->dim,
                             static_cast<sextant::RowId>(b->rows + r));
            for (uint32_t c = 0; c < n_cols; ++c) {
                const void* fv = filter_values ? filter_values[c] : nullptr;
                const uint8_t* fn =
                    filter_nulls ? filter_nulls[c] : nullptr;
                if (fn && fn[r]) {
                    if (!b->schema.columns[c].nullable) {
                        set_err(err, err_len,
                                "sextant_build_push: NULL pushed for "
                                "non-nullable column");
                        return -1;
                    }
                    st.append_null(c);
                    continue;
                }
                const auto t = b->schema.columns[c].type;
                switch (t) {
                    case sextant::ColumnType::Int32: {
                        int32_t v = 0;
                        if (fv) v = static_cast<const int32_t*>(fv)[r];
                        st.append_fixed(c, reinterpret_cast<const uint8_t*>(&v), 4);
                        break;
                    }
                    case sextant::ColumnType::Int64: {
                        int64_t v = 0;
                        if (fv) v = static_cast<const int64_t*>(fv)[r];
                        st.append_fixed(c, reinterpret_cast<const uint8_t*>(&v), 8);
                        break;
                    }
                    case sextant::ColumnType::Float: {
                        float v = 0.0f;
                        if (fv) v = static_cast<const float*>(fv)[r];
                        st.append_fixed(c, reinterpret_cast<const uint8_t*>(&v), 4);
                        break;
                    }
                    case sextant::ColumnType::Bool: {
                        // 1-byte-per-row fixed column (uint8, 0/1).
                        uint8_t v = 0;
                        if (fv) v = static_cast<const uint8_t*>(fv)[r] != 0;
                        st.append_fixed(c, &v, 1);
                        break;
                    }
                    case sextant::ColumnType::String: {
                        const char* data = nullptr;
                        uint32_t len = 0;
                        if (fv) {
                            const auto* sv =
                                static_cast<const sextant_str_values*>(fv);
                            if (sv->data && sv->lengths) {
                                data = sv->data[r];
                                len = sv->lengths[r];
                            }
                        }
                        if (len > 65535) {
                            set_err(err, err_len,
                                    "sextant_build_push: string length "
                                    "exceeds 65535");
                            return -1;
                        }
                        st.append_string(c, data, len);
                        break;
                    }
                    case sextant::ColumnType::Set: {
                        uint32_t begin = 0, end = 0;
                        if (fv) {
                            const auto* sv =
                                static_cast<const sextant_set_values*>(fv);
                            // Row r's elements are offsets[r]..offsets[r+1];
                            // counts (optional) must agree with the run.
                            if (sv->offsets) {
                                begin = sv->offsets[r];
                                end = sv->offsets[r + 1];
                            } else if (sv->counts) {
                                begin = end;
                                for (uint32_t i = 0; i < r; ++i)
                                    end += sv->counts[i];
                            }
                            if (sv->counts &&
                                uint32_t(sv->counts[r]) != end - begin) {
                                set_err(err, err_len,
                                        "sextant_build_push: set count "
                                        "mismatches offset run");
                                return -1;
                            }
                        }
                        const uint32_t count = end - begin;
                        if (count > 255) {
                            set_err(err, err_len,
                                    "sextant_build_push: set element count "
                                    "exceeds 255");
                            return -1;
                        }
                        if (count) {
                            const auto* sv =
                                static_cast<const sextant_set_values*>(fv);
                            if (!sv->elem_data || !sv->elem_lengths) {
                                set_err(err, err_len,
                                        "sextant_build_push: set elements "
                                        "require elem_data/elem_lengths");
                                return -1;
                            }
                            for (uint32_t e = begin; e < end; ++e) {
                                if (sv->elem_lengths[e] > 65535) {
                                    set_err(err, err_len,
                                            "sextant_build_push: set element "
                                            "length exceeds 65535");
                                    return -1;
                                }
                            }
                            st.append_set_row(c, count, sv->elem_data + begin,
                                              sv->elem_lengths + begin);
                        } else {
                            st.append_set_row(c, 0, nullptr, nullptr);
                        }
                        break;
                    }
                }
            }
            if (b->has_payload) {
                const uint64_t p0 = payload_offsets[r];
                const uint64_t p1 = payload_offsets[r + 1];
                if (p1 < p0) {
                    set_err(err, err_len,
                            "sextant_build_push: payload offsets not "
                            "monotone");
                    return -1;
                }
                if (p1 - p0 > 0xFFFFFFFFull) {
                    set_err(err, err_len,
                            "sextant_build_push: payload blob exceeds 4GB");
                    return -1;
                }
                st.append_payload(payload_data + p0,
                                  static_cast<uint32_t>(p1 - p0));
            }
            st.end_row();
        }
        b->rows += n_rows;
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return -2;
    } catch (...) {
        set_err(err, err_len, "sextant_build_push: unknown exception");
        return -3;
    }
}

int sextant_build_finish(void* builder, const char* out_path,
                         char* err, size_t err_len) {
    auto* b = reinterpret_cast<PushBuilder*>(builder);
    if (!b || !out_path) {
        delete b;
        set_err(err, err_len, "sextant_build_finish: null argument");
        return -1;
    }
    try {
        auto source = b->stager->build_source();
        const sextant_build_opts& opts = b->opts;
        sextant::tree::IVFTreeIndex::BuildConfig cfg;
        cfg.k_root = opts.k_root;
        cfg.leaf_capacity = opts.leaf_capacity ? opts.leaf_capacity : 5000;
        cfg.pca_dims = opts.pca_dims ? opts.pca_dims : 32;
        cfg.num_threads = opts.num_threads;
        cfg.plane_attach = opts.plane_attach != 0;
        cfg.plane_rank = opts.plane_rank ? static_cast<uint16_t>(opts.plane_rank) : 128;
        cfg.max_lloyd_passes =
            opts.max_lloyd_passes ? opts.max_lloyd_passes : 10;
        cfg.adaptive_probe_gap = opts.adaptive_probe_gap;
        cfg.params.quantizer_type = opts.quantizer;
        cfg.params.pq4_m = static_cast<uint16_t>(opts.pq_m);
        cfg.params.metric = opts.metric == SEXTANT_METRIC_IP
            ? sextant::MetricKind::InnerProduct
            : sextant::MetricKind::L2Sq;
        // Filter columns + payload stream per-chunk from the staged source
        // (same path as the parquet sources) — bounded RAM end to end.
        cfg.filter_schema = source->schema();
        // Optional JSON-lines phase metrics (jsonl sink owns the FILE*;
        // must outlive the build call).
        std::unique_ptr<sextant::metrics::JsonlMetricsSink> metrics_json;
        if (opts.metrics_path && *opts.metrics_path) {
            metrics_json =
                std::make_unique<sextant::metrics::JsonlMetricsSink>(
                    opts.metrics_path);
            cfg.metrics_sink = metrics_json.get();
        }
        // Cardinality tracking spec (NULL/empty = all off, the historical
        // default). Grammar shared with the CLI --cardinality-col flag.
        if (opts.cardinality && *opts.cardinality) {
            cfg.cardinality_spec =
                sextant::tree::parse_cardinality_spec(opts.cardinality);
        }
        // Emission-pass cluster staging RAM budget (0 = BuildConfig
        // default). Mirrors the CLI --stage-budget-mb flag.
        if (opts.stage_budget_mb) {
            cfg.stage_budget_mb = opts.stage_budget_mb;
        }
        sextant::tree::IVFTreeIndex::build_streaming_pca(*source, out_path,
                                                         cfg);
        delete b;
        return 0;
    } catch (const std::exception& e) {
        delete b;
        set_err(err, err_len, e.what());
        return -2;
    } catch (...) {
        delete b;
        set_err(err, err_len, "sextant_build_finish: unknown exception");
        return -3;
    }
}

void sextant_build_abort(void* builder) {
    delete reinterpret_cast<PushBuilder*>(builder);
}

}  // extern "C"
