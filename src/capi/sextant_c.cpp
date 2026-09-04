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

namespace {

/// Copy an exception message into the caller's error buffer.
void set_err(char* err, size_t err_len, const char* msg) {
    if (err && err_len > 0) {
        std::snprintf(err, err_len, "%s", msg);
    }
}

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
    o.scan_code_budget = 0;
    o.rerank = 0;
    o.search_threads = 0;
    o.int8_scan = -1;  // env decides (SEXTANT_SCAN_I8); 0 here previously
                        // force-disabled the i8 kernel for default callers
    o.exhaustive = 0;
    return o;
}

int32_t sextant_search(void* index, const float* query,
                       const sextant_search_opts* opts,
                       uint64_t* out_row_ids, float* out_dists,
                       uint32_t out_capacity, char* err, size_t err_len) {
    auto* idx = reinterpret_cast<sextant::tree::IVFTreeIndex*>(index);
    if (!idx || !query || !opts || !out_row_ids || !out_dists || opts->k == 0
        || out_capacity < opts->k) {
        set_err(err, err_len, "sextant_search: null argument or k==0");
        return -1;
    }
    try {
        int scan_i8_override = -1;
        sextant::SearchConfig cfg;
        cfg.k = opts->k;
        if (opts->exhaustive) {
            // Probe-all: every root child, every internal child, no gap
            // pruning, no code budget. The clamp `min(n_probe, k_root)` makes
            // UINT32_MAX mean "all leaves" (see search()'s level-0/ln paths).
            cfg.n_probe = UINT32_MAX;
            cfg.n_probe_ln = UINT32_MAX;
            cfg.adaptive_probe_gap = -1.0f;  // <0 = off
            cfg.scan_code_budget = 0;
        } else {
            cfg.n_probe = opts->n_probe;
            cfg.n_probe_ln = opts->n_probe_ln;
            cfg.adaptive_probe_gap = opts->adaptive_probe_gap;
            cfg.scan_code_budget = opts->scan_code_budget;
        }
        cfg.fastscan_W = opts->fastscan_W;
        cfg.rerank = opts->rerank != 0;
        cfg.exact_rerank_base = opts->exact_rerank_base;
        cfg.adaptive_w_gap = opts->adaptive_w_gap;
        scan_i8_override = opts->int8_scan;
        cfg.search_threads = opts->search_threads;
        // The engine gates its i8 kernel on SEXTANT_SCAN_I8 (process-global
        // env). A per-call override uses a thread-local the scan reads.
        sextant::tree::set_scan_i8_override(scan_i8_override);
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

}  // extern "C"
