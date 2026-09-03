#pragma once

/// @file sextant_c.h
/// Minimal C ABI over the IVF tree index for external benchmark harnesses
/// (task 5: retrievalbench head-to-head). Build from an .fbin corpus, open,
/// search single queries. No filtering, no payloads — bench surface only.
///
/// All functions are thread-safe for concurrent sextant_search() on the same
/// index handle (search is const); build/open/close are single-threaded.
/// Errors: functions returning int report 0 on success and a negative value
/// on failure, writing a message into `err` when non-null.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Metric. 0 = squared L2, 1 = inner product (cosine after normalization).
#define SEXTANT_METRIC_L2SQ 0
#define SEXTANT_METRIC_IP 1

typedef struct sextant_build_opts {
    /// Quantizer family: "pq", "local_pq", "scalar_lloydmax",
    /// "scalar_uniform", "local_scalar".
    const char* quantizer;
    uint32_t pq_m;             ///< PQ subquantizers (0 = auto dim/4)
    uint32_t k_root;           ///< root branching (0 = auto)
    uint32_t leaf_capacity;    ///< max vectors per leaf (0 = 5000)
    uint32_t pca_dims;         ///< PCA routing dims (0 = 32)
    uint32_t num_threads;      ///< build threads (0 = hardware)
    uint32_t max_lloyd_passes; ///< 0 = 10
    int metric;                ///< SEXTANT_METRIC_*
    float adaptive_probe_gap;  ///< build-time bake (0 = off)
} sextant_build_opts;

/// Sensible defaults (pq, auto geometry, L2).
sextant_build_opts sextant_default_build_opts(void);

/// Build a tree from an .fbin file. Blocking; `num_threads` parallel.
/// Returns 0 on success. `err` (if non-null, `err_len` bytes) receives the
/// error message on failure.
int sextant_build_fbin(const char* fbin_path, const char* out_path,
                       const sextant_build_opts* opts,
                       char* err, size_t err_len);

/// Build from an in-memory row-major slice (n vectors of `dim` floats).
/// Row ids are 0..n-1. Same contract as sextant_build_fbin otherwise.
int sextant_build_mem(const float* vectors, uint32_t n, uint32_t dim,
                      const char* out_path,
                      const sextant_build_opts* opts,
                      char* err, size_t err_len);

typedef struct sextant_search_opts {
    uint32_t k;                ///< top-k
    uint32_t n_probe;          ///< root probe count (0 = index default,
                               ///<  UINT32_MAX = all)
    uint32_t n_probe_ln;       ///< per-root-child probe count (same semantics)
    uint32_t fastscan_W;       ///< shortlist width (0 = index default)
    float adaptive_probe_gap;  ///< <0 = off, 0 = index default, >1 = explicit
    uint32_t scan_code_budget; ///< 0 = unlimited
    int rerank;                ///< 0/1 (rerank top-W by decoded distance)
    float adaptive_w_gap;      ///< tau: post-rerank shortlist cut (0 = off).
                               ///< With it, search returns up to W>k ids whose
                               ///< reranked distance stays within gap of the
                               ///< k-th — the adaptive-W contract the bench
                               ///  grades recall over. Requires rerank.
    uint32_t search_threads;   ///< within-query leaf-parallel scan (0 = serial)
    int int8_scan;             ///< 1 = i8 SDOT kernel (uniform scalar codes).
                               ///< 3-4x faster, NOT score-bit-identical:
                               ///< raw-ranking recall drops (dbpedia 100K
                               ///< flat: -25pp; tau-rerank path: ~0). Use
                               ///< for rerank-serving rows only. -1 = env
                               ///< SEXTANT_SCAN_I8 decides. Default -1.
    int exhaustive;            ///< 1 overrides the above into probe-all:
                               ///< n_probe = n_probe_ln = UINT32_MAX,
                               ///< gap pruning off, budget unlimited.
    const float* exact_rerank_base; ///< NULL = decoded rerank (default);
                               ///< non-NULL = exact rerank against these
                               ///< row-major f32 vectors (n × dim, indexed
                               ///< by returned row_id; caller owns memory,
                               ///< e.g. an mmap of the corpus). Removes the
                               ///< quantization ranking error entirely.
                               ///< ABI: added at struct tail — old callers
                               ///< passing smaller structs read as NULL via
                               ///< designated-init zero fill.
} sextant_search_opts;

sextant_search_opts sextant_default_search_opts(void);

/// Open a built tree (mmap, read-only). NULL on failure.
void* sextant_open_index(const char* path, char* err, size_t err_len);
void sextant_close_index(void* index);

uint32_t sextant_index_dim(const void* index);
uint64_t sextant_index_live_count(void* index); ///< requires mutable open

/// Single-query search. Writes up to `k` (row_id, dist) pairs, ascending
/// distance. Returns the number of results written, or a negative error.
/// Caller-allocated `out_row_ids` / `out_dists` (k entries each).
int32_t sextant_search(void* index, const float* query,
                       const sextant_search_opts* opts,
                       uint64_t* out_row_ids, float* out_dists,
                       uint32_t out_capacity, char* err, size_t err_len);

#ifdef __cplusplus
}  // extern "C"
#endif
