#pragma once

/// @file sextant_c.h
/// C ABI over the IVF tree index. Originally the external benchmark-harness
/// surface (task 5: retrievalbench head-to-head: build from .fbin, open,
/// single-query search); extended for the DuckDB-ready v1 ABI: filtered
/// search, batch search, payload/vector fetch, and a streaming push build
/// with filter columns + payloads.
///
/// All functions are thread-safe for concurrent sextant_search() on the same
/// index handle (search is const); build/open/close are single-threaded.
/// Errors: functions returning int report 0 on success and a negative value
/// on failure, writing a message into `err` when non-null.
///
/// EXCEPTION-SAFETY CONTRACT: every entry point catches all exceptions
/// internally. No exception ever escapes the C boundary (DuckDB extension
/// TUs are -fno-exceptions). Failures surface as negative returns + `err`.

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
    /// In-build routing plane (CLI serving default). ABI tail-append:
    /// zero-filled by designated-init callers reads as off, matching the
    /// historical C API behavior; sextant_default_build_opts() sets it on.
    int plane_attach;          ///< 1 = attach B1G routing plane (default on)
    uint32_t plane_rank;       ///< plane rank (0 = 128)
    /// In-memory staging budget for push builds; staged chunks beyond it
    /// spill to a temp file and stream back at finish (bounded RAM end to
    /// end). 0 = 256 MiB default. ABI tail-append (zero = default).
    uint64_t staging_bytes;
    /// Write build phase metrics as JSON lines to this file (build.*
    /// field names, one record per phase, flushed per record so tail -F
    /// streams live). NULL/empty = no file. ABI tail-append (zero = off).
    const char* metrics_path;
    /// Cardinality tracking spec for filter columns, comma-separated:
    /// "name", "name=on|off|auto", or bare "auto" (default mode for all
    /// columns). NULL/empty = all off (historical default). Same grammar
    /// as the CLI --cardinality-col flag. ABI tail-append (zero = off).
    const char* cardinality;
    /// RAM budget (MiB) for the emission-pass cluster staging arenas
    /// (clusters spill to a temp file beyond it; 0 = BuildConfig default
    /// of 512). Mirrors the CLI --stage-budget-mb flag. ABI tail-append.
    uint64_t stage_budget_mb;
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
    int rerank;                ///< 0/1 (rerank top-W by decoded distance)
    float adaptive_w_gap;      ///< tau: post-rerank shortlist cut (0 = off,
                               ///< -1 = AUTO: tau per scan-code size
                               ///< (>=288B -> 2.5, else 5.0), CLI parity —
                               ///< the default from
                               ///< sextant_default_search_opts and the
                               ///< quality contract on near-dup corpora
                               ///< (measured 0.89 -> 0.99 recall@10 on a
                               ///< 23M geocoder corpus vs off)). With it,
                               ///< search returns up to W>k ids whose
                               ///< reranked distance stays within gap of the
                               ///< k-th — the adaptive-W contract the bench
                               ///< grades recall over. Requires rerank.
    uint32_t search_threads;   ///< within-query leaf-parallel scan (0 = serial)
    int int8_scan;             ///< 1 = i8 SDOT kernel (uniform scalar codes).
                               ///< 3-4x faster, NOT score-bit-identical:
                               ///< raw-ranking recall drops (dbpedia 100K
                               ///< flat: -25pp; tau-rerank path: ~0). Use
                               ///< for rerank-serving rows only. -1 = auto
                               ///< (int8 on AVX512/VNNI, float otherwise).
                               ///< Default -1.
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
    float probe_fraction;     ///< corpus-fraction probe budget (0 = index
                               ///<  default; 0.25/0.5 measured ~0.96/~0.99
                               ///<  recall@10 on dbpedia, scale-stable).
                               ///<  Ignored when n_probe > 0 or exhaustive.
                               ///<  ABI: added at struct tail.
    float plane_pre_prune;    ///< Two-stage plane routing: fraction of
                               ///<  leaves (PCA centroid distance,
                               ///<  page-weighted) surviving to the plane
                               ///<  sweep. 0 (incl. legacy zero-fill) =
                               ///<  engine default 0.25 (recall-neutral,
                               ///<  QPS-positive); <0 = off; >0 = explicit.
                               ///<  Ignored for plane-less trees.
                               ///<  ABI: added at struct tail.
} sextant_search_opts;

sextant_search_opts sextant_default_search_opts(void);

/// Open a built tree (mmap, read-only). NULL on failure.
/// `leaf_cache_bytes` / `plane_cache_bytes` size engine-owned W-TinyLFU
/// caches for the search path (0 = off, the default): leaf extents and
/// routing planes are served from a DRAM-budgeted cache (pread fill,
/// mmap fallback on admission rejection) instead of the mmap. Never
/// slower than mmap at any size; pays off when the hot leaf set fits
/// (repeated-query workloads).
void* sextant_open_index(const char* path, uint64_t leaf_cache_bytes,
                         uint64_t plane_cache_bytes, char* err, size_t err_len);
void sextant_close_index(void* index);

uint32_t sextant_index_dim(const void* index);
/// Copy the tree UUID (32 lowercase hex chars, mandatory in the format)
/// into `out` (NUL-terminated; pass out_len >= 33). Returns 32 on success,
/// -1 on bad arguments. Every openable tree has a UUID (manifests without
/// one fail at open).
int32_t sextant_index_uuid(const void* index, char* out, size_t out_len);
uint64_t sextant_index_live_count(void* index); ///< requires mutable open

/// Single-query search. Writes up to `k` (row_id, dist) pairs, ascending
/// distance. Returns the number of results written, or a negative error.
/// Caller-allocated `out_row_ids` / `out_dists` (k entries each).
int32_t sextant_search(void* index, const float* query,
                       const sextant_search_opts* opts,
                       uint64_t* out_row_ids, float* out_dists,
                       uint32_t out_capacity, char* err, size_t err_len);

// ===========================================================================
// DuckDB-ready v1 extension: predicates, filtered/batch search, payload and
// vector fetch, streaming push build. ABI: enum values below are FIXED
// forever; structs evolve by tail-append only.
// ===========================================================================

/// Predicate operators. Values are part of the ABI — never renumber.
/// All ops implemented. Per-op field usage (fields beyond those listed are
/// ignored; zero-init / designated-init zero-fills unused tail fields):
///
/// op              | column type | fields used
/// ----------------|-------------|------------------------------------------
/// EQ/NEQ          | any         | value (numeric) or str_value (string)
/// LT/LE/GT/GE     | numeric     | value
/// BETWEEN         | numeric     | value (low), value2 (high, inclusive)
/// PREFIX          | string      | str_value
/// IN / NOT_IN     | any         | values/value_lengths/n_values (strings;
///                 |             | numeric columns compare via the strings)
/// CONTAINS        | set         | str_value (the element)
/// CONTAINS_ANY    | set         | str_value (optional) + values (folds both)
/// CONTAINS_ALL    | set         | str_value (optional) + values (folds both)
/// GEO_RADIUS      | lat column  | geo_lng_column, value/value2 = center
///                 | (numeric)   | lat/lng, radius_km
/// GEO_BOX         | lat column  | geo_lng_column, value/value2/value3/
///                 | (numeric)   | value4 = min_lat/min_lng/max_lat/max_lng
///
/// Geo predicates name the latitude column in `column` and the longitude
/// column in `geo_lng_column` (both must be numeric filter columns).
enum {
    SEXTANT_PRED_EQ = 0,       ///< value == x (numeric) or str_value == s
    SEXTANT_PRED_NEQ = 1,      ///< value != x / str_value != s
    SEXTANT_PRED_LT = 2,       ///< x <  value           (numeric only)
    SEXTANT_PRED_LE = 3,       ///< x <= value
    SEXTANT_PRED_GT = 4,       ///< x >  value
    SEXTANT_PRED_GE = 5,       ///< x >= value
    SEXTANT_PRED_BETWEEN = 6,  ///< value <= x <= value2 (numeric only)
    SEXTANT_PRED_PREFIX = 7,   ///< str_value is a prefix of s (string only)
    SEXTANT_PRED_IN = 8,         ///< x in values set (n_values must be > 0)
    SEXTANT_PRED_NOT_IN = 9,     ///< x not in values set
    SEXTANT_PRED_CONTAINS = 10,        ///< set column contains str_value
    SEXTANT_PRED_CONTAINS_ANY = 11,    ///< set intersects str_value+values
    SEXTANT_PRED_CONTAINS_ALL = 12,    ///< str_value+values subset of set
    SEXTANT_PRED_GEO_RADIUS = 13,      ///< haversine((lat,lng),(value,value2))
                                      ///< <= radius_km
    SEXTANT_PRED_GEO_BOX = 14,         ///< lat/lng inside [min,max] box
    SEXTANT_PRED_IS_NULL = 15,         ///< column value is NULL (nullable
                                      ///< columns only)
    SEXTANT_PRED_IS_NOT_NULL = 16,     ///< column value is not NULL
};

typedef struct sextant_predicate {
    const char* column;    ///< filter column name (required, non-NULL)
    int op;                ///< SEXTANT_PRED_* (required)
    double value;          ///< numeric comparand; BETWEEN low bound;
                           ///< GEO_RADIUS center lat; GEO_BOX min_lat
    double value2;         ///< BETWEEN high bound (inclusive);
                           ///< GEO_RADIUS center lng; GEO_BOX min_lng
    const char* str_value; ///< string comparand (EQ/NEQ/PREFIX), set element
                           ///< (CONTAINS family); NULL = ""
    // --- tail-appended fields (ABI: old callers read them as zeroed) ---
    const char* const* values;    ///< IN/NOT_IN/CONTAINS_ANY/CONTAINS_ALL
                                  ///< list: n_values pointers to (not
                                  ///< NUL-terminated) data
    const uint32_t* value_lengths;///< n_values byte lengths
    uint32_t n_values;            ///< list length (0 = empty list)
    const char* geo_lng_column;   ///< GEO_RADIUS/GEO_BOX longitude column
                                  ///< name (required for geo ops)
    double radius_km;             ///< GEO_RADIUS radius in kilometers
    double value3;                ///< GEO_BOX max_lat
    double value4;                ///< GEO_BOX max_lng
} sextant_predicate;

/// Filter column types for sextant_build_begin. Values are ABI-fixed.
/// Note: the engine stores Float as IEEE binary32; predicate `value` doubles
/// are narrowed at evaluation.
enum {
    SEXTANT_COL_INT32 = 0,
    SEXTANT_COL_INT64 = 1,
    SEXTANT_COL_FLOAT = 2,  ///< engine-internal binary32
    SEXTANT_COL_STRING = 3,
    SEXTANT_COL_SET = 4,    ///< set of strings (via sextant_set_values)
    SEXTANT_COL_BOOL = 5,   ///< 1 byte per row (0/1); ABI tail-append
};

typedef struct sextant_filter_col_def {
    const char* name;  ///< column name (required, non-NULL)
    int type;          ///< SEXTANT_COL_*
    // --- tail-appended fields (ABI: old callers read them as zeroed) ---
    int nullable;      ///< 1 = store per-row NULL flags and evaluate
                       ///< predicates with SQL three-valued logic; 0
                       ///< (default/legacy) = no NULL support for this
                       ///< column
} sextant_filter_col_def;

/// Per-column string values for one sextant_build_push() call. The
/// filter_values entry for a String column points at one of these.
typedef struct sextant_str_values {
    const char* const* data;   ///< n_rows pointers to (not NUL-terminated) data
    const uint32_t* lengths;   ///< n_rows byte lengths (each <= 65535)
} sextant_str_values;

/// Per-column set values for one sextant_build_push() call. The
/// filter_values entry for a Set column points at one of these. Row r's
/// elements are the indices offsets[r] .. offsets[r+1] (exclusive); element
/// j's bytes live at elem_data[j] with byte length elem_lengths[j]
/// (each length <= 65535; at most 255 elements per row). A NULL entry (or
/// NULL offsets) fills the rows with empty sets.
typedef struct sextant_set_values {
    const uint32_t* counts;         ///< n_rows element counts (may be NULL)
    const uint32_t* offsets;        ///< n_rows+1 element-run bounds
    const char* const* elem_data;   ///< element pointers (indexed by run)
    const uint32_t* elem_lengths;   ///< element byte lengths
} sextant_set_values;

/// Single-query filtered search. Same contract as sextant_search(), plus a
/// conjunct list of `n_preds` predicates (all must hold). `preds` may be
/// NULL/0 for unfiltered. Returns the number of results written, or negative.
int32_t sextant_search_filtered(void* index, const float* query,
                                const sextant_search_opts* opts,
                                const sextant_predicate* preds,
                                uint32_t n_preds,
                                uint64_t* out_row_ids, float* out_dists,
                                uint32_t out_capacity,
                                char* err, size_t err_len);

/// Single-query filtered search with payload locations. In addition to the
/// sextant_search_filtered contract, writes the leaf location of every
/// result: out_leaf_ptrs[i] is an OPAQUE pointer into the index mmap,
/// out_slots[i] its slot; both are consumed by sextant_fetch_payload() /
/// sextant_fetch_vector(). Locations stay valid until sextant_close_index()
/// (read-only handles never mutate). Arrays are caller-allocated
/// (out_capacity entries); pass NULL for either to skip them.
int32_t sextant_search2(void* index, const float* query,
                        const sextant_search_opts* opts,
                        const sextant_predicate* preds, uint32_t n_preds,
                        uint64_t* out_row_ids, float* out_dists,
                        void** out_leaf_ptrs, uint32_t* out_slots,
                        uint32_t out_capacity, char* err, size_t err_len);

/// Batch search with shared predicates: all `n_queries` queries (row-major,
/// dim from the index) are searched under the same conjunct list (may be
/// NULL/0). Output is row-major with per-query capacity: query i's results
/// live at out_row_ids[i*out_capacity_per_query .. + out_n_per_query[i])
/// (same for out_dists). out_n_per_query must have n_queries entries.
/// Returns 0 on success, negative on failure.
int sextant_search_batch(void* index, const float* queries,
                         uint32_t n_queries,
                         const sextant_search_opts* opts,
                         const sextant_predicate* preds, uint32_t n_preds,
                         uint64_t* out_row_ids, float* out_dists,
                         uint32_t out_capacity_per_query,
                         uint32_t* out_n_per_query,
                         char* err, size_t err_len);

/// Fetch the payload blob for one search result. Returns the FULL blob size
/// in bytes and copies min(size, buf_capacity) bytes into out_buf. 0 = no
/// payload for this row (or null arguments). Call again with a larger buffer
/// when the returned size exceeds buf_capacity. leaf_ptr/slot come from
/// sextant_search2 output and stay valid until sextant_close_index().
uint32_t sextant_fetch_payload(const void* index, const void* leaf_ptr,
                               uint32_t slot, void* out_buf,
                               uint32_t buf_capacity);

/// Decode the stored vector for one search result into `out` (dim() floats,
/// caller-allocated). Stored codes are lossy (fp16/quantized): compare with
/// ~1e-2 relative tolerance. Returns 0 on success, negative on failure.
int sextant_fetch_vector(const void* index, const void* leaf_ptr,
                         uint32_t slot, float* out);

/// Logical row count: number of input rows indexed, each counted once
/// (closure-replicated leaf slots are NOT counted). O(1) — read from the
/// manifest; reflects the state at the last committed build/mutation.
/// Read-only-handle safe. Returns 0 only for legacy manifests built
/// before the logical count was persisted.
uint64_t sextant_index_count(const void* index);

// --- Scan worker pool -------------------------------------------------------

/// Resize the process-wide scan pool (shared by within-query parallel
/// scanning and batch sweeps; threads are warm and long-lived). n == 0
/// selects hardware_concurrency. Returns the new thread count. Embedders
/// running their own query-level parallelism (e.g. a DuckDB extension)
/// should size this to their thread budget.
uint32_t sextant_scan_pool_set_threads(uint32_t n);

/// Current scan pool thread count.
uint32_t sextant_scan_pool_threads(void);

/// Scan code size in bytes per vector (drives the adaptive-W tau default:
/// >= 288B -> 2.5, else 5.0 — same calibration the CLI ships).
uint32_t sextant_index_code_size(const void* index);

/// Distance metric of the tree: SEXTANT_METRIC_L2SQ or SEXTANT_METRIC_IP.
/// Read from the manifest; read-only-handle safe. Negative on error.
int sextant_index_metric(const void* index);

/// Number of declared filter columns in the index's schema (0 when the
/// tree was built without filter_cols). Read-only-handle safe.
uint32_t sextant_index_filter_col_count(const void* index);

/// Copy filter column i's name (NUL-terminated; pass name_len >= 128)
/// into out_name and its SEXTANT_COL_* type into out_type. out_nullable
/// (optional) receives 1 when the column carries per-row NULL flags.
/// Returns 0 on success, a negative value on bad arguments / out-of-range
/// index.
int32_t sextant_index_filter_col(const void* index, uint32_t i,
                                 char* out_name, size_t name_len,
                                 int* out_type, int* out_nullable);

// --- Streaming push build ------------------------------------------------

/// Begin a push build of `dim`-dimensional vectors. `cols` declares the
/// filter columns (may be NULL when n_cols == 0); `has_payload` enables
/// per-row payload blobs via sextant_build_push. Row ids are assigned 0..n-1
/// in push order. Returns an opaque builder handle, or NULL on failure.
void* sextant_build_begin(const sextant_build_opts* opts, uint32_t dim,
                          const sextant_filter_col_def* cols, uint32_t n_cols,
                          int has_payload, char* err, size_t err_len);

/// Push n_rows vectors (row-major, dim from sextant_build_begin). May be called
/// any number of times; row ids continue across calls. `filter_values`, when
/// non-NULL, holds one entry per declared column, each covering n_rows rows
/// in push order: for Int32/Int64/Float/Bool a pointer to a contiguous
/// array of that C type (Bool = 1 byte per row, 0/1); for String a pointer
/// to a sextant_str_values; for Set a pointer to a sextant_set_values. A NULL entry
/// (or NULL filter_values with declared columns) fills the rows with the
/// type's default (0 / empty string / empty set). Payload: payload_offsets has n_rows+1
/// entries delimiting n_rows blobs in payload_data (both required when the
/// build has payloads; pass NULL to give all rows empty payloads).
/// Returns 0 on success, negative on failure (builder stays usable).
int sextant_build_push(void* builder, const float* vectors, uint32_t n_rows,
                       const void* const* filter_values,
                       const uint64_t* payload_offsets,
                       const uint8_t* payload_data,
                       char* err, size_t err_len);

/// Like sextant_build_push, plus per-column NULL flags: filter_nulls[c]
/// (may be NULL per column, or NULL overall) holds one byte per row
/// (1 = NULL). A NULL row needs no meaningful value in filter_values.
/// NULLs are only allowed for columns declared nullable at
/// sextant_build_begin; pushing one for a non-nullable column is an error.
int sextant_build_push_validity(void* builder, const float* vectors,
                               uint32_t n_rows,
                               const void* const* filter_values,
                               const uint8_t* const* filter_nulls,
                               const uint64_t* payload_offsets,
                               const uint8_t* payload_data,
                               char* err, size_t err_len);

/// Finish: run the (blocking) build and write the index to out_path. Frees
/// the builder on BOTH success and failure. Returns 0 on success.
int sextant_build_finish(void* builder, const char* out_path,
                         char* err, size_t err_len);

/// Abort a push build and free the builder. No file is written.
void sextant_build_abort(void* builder);

// ===========================================================================
// Request scheduler: serving-layer windowing over search (v1 extension).
// Manufactures the batches that make cold batch-window throughput available
// to single-request arrival streams. ABI: structs evolve by tail-append
// only.
// ===========================================================================

/// Scheduler configuration. `base` supplies the search settings every
/// submitted query inherits (k, probe depth, rerank, int8_scan, ...);
/// per-submit arguments override k / predicates / probe depth / latency
/// tolerance. NULL = sextant_default_search_opts(). The struct is copied
/// at create time, except `base->exact_rerank_base`, which keeps the
/// sextant_search contract: caller owns that memory for the scheduler's
/// lifetime. Zero-valued knobs keep their documented defaults.
typedef struct sextant_scheduler_config {
    const sextant_search_opts* base;  ///< NULL = defaults
    uint64_t window_max_us;           ///< 0 = 100 ms (deadline bound)
    uint64_t idle_close_us;           ///< 0 = 200 us
    uint32_t max_window_queries;      ///< 0 = 4096
    uint32_t result_cache_entries;    ///< 0 = off
    uint32_t search_threads;          ///< 0 = 1
    uint32_t max_inflight_windows;    ///< 0 = 1, scheduler default 2
} sextant_scheduler_config_t;

/// Delta snapshot since the previous call (counters RESET on every
/// snapshot — a metrics loop owns these numbers after reading them).
/// Delay percentiles come from the submit→dispatch-delay samples of the
/// period (capped internally); p50/p99 are 0 when no query dispatched.
typedef struct sextant_scheduler_stats {
    uint64_t windows;        ///< windows dispatched
    uint64_t queries;        ///< queries served (incl. cache hits)
    uint64_t cache_hits;     ///< exact-repeat cache resolutions
    uint64_t sweep_ns;       ///< total search_batch wall time
    uint32_t peak_inflight;  ///< max concurrent sweeps observed
    // --- tail-appended fields (ABI: old callers read them as zeroed) ---
    uint64_t delay_p50_ns;   ///< submit→dispatch delay, median
    uint64_t delay_p99_ns;   ///< submit→dispatch delay, 99th pct
} sextant_scheduler_stats_t;

/// Create a scheduler on an open index handle. The index must outlive the
/// scheduler: call sextant_scheduler_stop() before sextant_close_index().
/// Starts the sweeper thread immediately. Returns NULL on failure (err set).
void* sextant_scheduler_create(const void* index,
                               const sextant_scheduler_config_t* cfg,
                               char* err, size_t err_len);

/// Submit one query (dim() floats, copied) and BLOCK until its window
/// has been swept and its results are ready — the deadline bound is the
/// latency contract, not the return time. Writes up to `k` (row_id, dist)
/// pairs, ascending distance; returns the number written, or a negative
/// error (in particular, submitting to a stopped scheduler fails without
/// blocking). `preds` may be NULL/0 for unfiltered. `max_delay_us` is the
/// queueing delay the request accepts (0 = the scheduler's window_max_us);
/// `probe_fraction` (0 = base) is the per-request recall-depth knob.
/// Thread-safe.
int32_t sextant_scheduler_submit(void* sched, const float* query,
                                 uint32_t k,
                                 const sextant_predicate* preds,
                                 uint32_t n_preds,
                                 uint64_t max_delay_us,
                                 float probe_fraction,
                                 uint64_t* out_row_ids, float* out_dists,
                                 uint32_t out_capacity,
                                 char* err, size_t err_len);

/// Snapshot stats into `out` (resets counters; see the struct docs).
/// Returns 0 on success, negative on bad arguments.
int sextant_scheduler_stats(void* sched, sextant_scheduler_stats_t* out);

/// Drain the queue, stop the sweeper, resolve pending futures, and FREE
/// the scheduler — the handle must not be used again (call
/// sextant_scheduler_create for a new one). NULL is a no-op. Submits
/// blocked in-flight when another thread stops resolve normally; submits
/// after stop fail without blocking.
void sextant_scheduler_stop(void* sched);

#ifdef __cplusplus
}  // extern "C"
#endif
