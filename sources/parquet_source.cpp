#include "parquet_source.hpp"

#include <sextant/error.hpp>
#include "../src/simd_kernels.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>

namespace sextant {

[[noreturn]] void throw_carquet_error(const carquet_error_t& err,
                                       const std::string& context) {
    std::string msg = context;
    msg += ": ";
    msg += carquet_status_string(err.code);
    if (err.message[0] != '\0') {
        msg += " — ";
        msg += err.message;
    }

    ErrorCode code = ErrorCode::IoError;
    switch (err.code) {
        case CARQUET_ERROR_OUT_OF_MEMORY:
            code = ErrorCode::OutOfMemory;
            break;
        case CARQUET_ERROR_INVALID_ARGUMENT:
        case CARQUET_ERROR_TYPE_MISMATCH:
        case CARQUET_ERROR_INVALID_SCHEMA:
            code = ErrorCode::InvalidParam;
            break;
        case CARQUET_ERROR_NOT_IMPLEMENTED:
            code = ErrorCode::NotImplemented;
            break;
        default:
            if (err.code >= CARQUET_ERROR_INVALID_MAGIC &&
                err.code <= CARQUET_ERROR_THRIFT_TRUNCATED)
                code = ErrorCode::CorruptIndex;
            else
                code = ErrorCode::IoError;
            break;
    }
    throw Error(code, msg);
}

ColumnType map_carquet_type(carquet_physical_type_t phys,
                            carquet_logical_type_id_t logical,
                            bool is_repeated) {
    if (is_repeated && phys == CARQUET_PHYSICAL_BYTE_ARRAY) {
        return ColumnType::Set;
    }
    switch (phys) {
        case CARQUET_PHYSICAL_INT32:  return ColumnType::Int32;
        case CARQUET_PHYSICAL_INT64:  return ColumnType::Int64;
        case CARQUET_PHYSICAL_FLOAT:  return ColumnType::Float;
        case CARQUET_PHYSICAL_BOOLEAN:return ColumnType::Bool;
        case CARQUET_PHYSICAL_BYTE_ARRAY: return ColumnType::String;
        default:
            throw Error(ErrorCode::InvalidParam,
                        "ParquetSource: unsupported physical type " +
                            std::to_string(static_cast<int>(phys)) +
                            " (logical=" +
                            std::to_string(static_cast<int>(logical)) + ")");
    }
}

ParquetSource::ParquetSource(const std::string& path,
                             const ParquetSourceConfig& config)
    : path_(path), config_(config) {
    carquet_error_t err = {};

    carquet_reader_options_t ropts;
    carquet_reader_options_init(&ropts);
    ropts.verify_checksums = config_.verify_checksums;
    ropts.use_mmap = config_.use_mmap;
    ropts.num_threads = config_.num_threads;

    reader_.reset(carquet_reader_open(path_.c_str(), &ropts, &err));
    if (!reader_) {
        throw_carquet_error(err, "ParquetSource: failed to open '" + path_ + "'");
    }

    count_ = static_cast<uint64_t>(carquet_reader_num_rows(reader_.get()));
    num_file_cols_ = carquet_reader_num_columns(reader_.get());

    init_schema_();

    if (vec_col_idx_ < 0) {
        throw Error(ErrorCode::InvalidParam,
                    "ParquetSource: vector column '" + config_.vector_col +
                        "' not found in '" + path_ + "'");
    }

    const carquet_schema_t* file_schema = carquet_reader_schema(reader_.get());
    carquet_physical_type_t vec_phys =
        carquet_schema_column_type(file_schema, vec_col_idx_);

    // Detect list<float> (repeated FLOAT): physical type is FLOAT but the
    // column has repetition level >= 1, meaning it's a list element.
    if (vec_phys == CARQUET_PHYSICAL_FLOAT) {
        int16_t max_rep = carquet_schema_max_rep_level(file_schema, vec_col_idx_);
        if (max_rep >= 1) {
            vec_is_list_ = true;
            // dim_ deferred to detect_list_dim_() — needs to read one batch.
        } else {
            dim_ = 1;
        }
    } else if (vec_phys == CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY) {
        const int32_t n_elem = carquet_schema_num_elements(file_schema);
        int32_t vec_elem_idx = -1;
        int32_t leaf_seen = 0;
        for (int32_t e = 1; e < n_elem; ++e) {
            const carquet_schema_node_t* node =
                carquet_schema_get_element(file_schema, e);
            if (carquet_schema_node_is_leaf(node)) {
                if (leaf_seen == vec_col_idx_) {
                    vec_elem_idx = e;
                    break;
                }
                ++leaf_seen;
            }
        }
        if (vec_elem_idx < 0) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetSource: cannot find element for vector column");
        }
        const carquet_schema_node_t* vec_node =
            carquet_schema_get_element(file_schema, vec_elem_idx);
        vec_type_len_ = carquet_schema_node_type_length(vec_node);
        if (vec_type_len_ % sizeof(float) != 0 || vec_type_len_ == 0) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetSource: vector column FIXED_LEN_BYTE_ARRAY "
                        "length " +
                            std::to_string(vec_type_len_) +
                            " is not a multiple of 4");
        }
        dim_ = static_cast<Dim>(vec_type_len_ / sizeof(float));
    } else if (vec_phys == CARQUET_PHYSICAL_DOUBLE) {
        dim_ = 1;
    } else if (vec_phys == CARQUET_PHYSICAL_INT64 ||
               vec_phys == CARQUET_PHYSICAL_INT32 ||
               vec_phys == CARQUET_PHYSICAL_BOOLEAN) {
        // Non-vector column used as placeholder (e.g., label-file reader where
        // all real columns are filters). dim_ = 0 means no vector data.
        dim_ = 0;
    } else {
        throw Error(ErrorCode::InvalidParam,
                    "ParquetSource: vector column must be FLOAT, DOUBLE, or "
                    "FIXED_LEN_BYTE_ARRAY, got physical type " +
                        std::to_string(static_cast<int>(vec_phys)));
    }

    if (count_ == 0) {
        throw Error(ErrorCode::InvalidParam,
                    "ParquetSource: file '" + path_ + "' has 0 rows");
    }

    if (vec_is_list_ && dim_ == 0) {
        detect_list_dim_();
    }

    spdlog::debug("ParquetSource: opened '{}' n={} dim={} vec_col={} "
                  "filter_cols={} list={} normalize={} payload_col={}",
                  path_, count_, dim_, vec_col_idx_,
                  filter_col_indices_.size(), vec_is_list_, config_.normalize,
                  payload_col_idx_);
}

void ParquetSource::init_schema_() {
    const carquet_schema_t* file_schema = carquet_reader_schema(reader_.get());
    const int32_t n = carquet_schema_num_columns(file_schema);

    vec_col_idx_ = carquet_schema_find_column(file_schema,
                                               config_.vector_col.c_str());

    // Resolve the payload column (optional). Must be a plain BYTE_ARRAY
    // (string/binary) or FIXED_LEN_BYTE_ARRAY leaf; it is excluded from the
    // filter columns and streamed as opaque per-row blobs via
    // Chunk::payload_data/payload_offsets.
    if (!config_.payload_col.empty()) {
        payload_col_idx_ = carquet_schema_find_column(
            file_schema, config_.payload_col.c_str());
        if (payload_col_idx_ < 0) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetSource: payload column '" +
                            config_.payload_col + "' not found in '" + path_ +
                            "'");
        }
        if (payload_col_idx_ == vec_col_idx_) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetSource: payload column '" +
                            config_.payload_col +
                            "' is the vector column");
        }
        const carquet_schema_node_t* pnode = nullptr;
        {
            // Map leaf column index -> schema element index (same dance as
            // the vector column path below).
            const int32_t n_elem =
                carquet_schema_num_elements(file_schema);
            int32_t leaf_seen = 0;
            for (int32_t e = 1; e < n_elem; ++e) {
                const carquet_schema_node_t* node =
                    carquet_schema_get_element(file_schema, e);
                if (carquet_schema_node_is_leaf(node)) {
                    if (leaf_seen == payload_col_idx_) {
                        pnode = node;
                        break;
                    }
                    ++leaf_seen;
                }
            }
            if (!pnode) {
                throw Error(ErrorCode::InvalidParam,
                            "ParquetSource: cannot find element for payload "
                            "column '" + config_.payload_col + "'");
            }
        }
        carquet_physical_type_t pphys =
            carquet_schema_column_type(file_schema, payload_col_idx_);
        if (pphys == CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY) {
            payload_type_len_ = carquet_schema_node_type_length(pnode);
            if (payload_type_len_ <= 0) {
                throw Error(ErrorCode::InvalidParam,
                            "ParquetSource: payload column '" +
                                config_.payload_col +
                                "' has invalid type length");
            }
        } else if (pphys != CARQUET_PHYSICAL_BYTE_ARRAY) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetSource: payload column '" +
                            config_.payload_col +
                            "' must be BYTE_ARRAY or FIXED_LEN_BYTE_ARRAY, "
                            "got physical type " +
                            std::to_string(static_cast<int>(pphys)));
        }
        schema_.has_payload = true;
    }

    // Upper-bound payload byte estimate for bitmap-region sizing. Derived
    // purely from column-chunk metadata (no data pages read): the summed
    // uncompressed size of the payload column across row groups. For
    // BYTE_ARRAY this includes 4-byte length prefixes and page overhead — a
    // slight overestimate of raw string bytes, i.e. the safe direction.
    //
    // Dictionary-encoded chunks are the exception: the chunk stores dict +
    // indices while the payload extents store raw per-row strings, so the
    // uncompressed size can undershoot badly on duplicate-heavy columns.
    // Boost those chunks 8x — over-provisioning is near-free (4 KiB of
    // bitmap addresses 128 MiB of file).
    if (payload_col_idx_ >= 0) {
        const int32_t n_rg = carquet_reader_num_row_groups(reader_.get());
        bool dict_warned = false;
        for (int32_t rg = 0; rg < n_rg; ++rg) {
            carquet_column_chunk_metadata_t md;
            carquet_status_t st = carquet_reader_column_chunk_metadata(
                reader_.get(), rg, payload_col_idx_, &md);
            if (st != CARQUET_OK) {
                carquet_error_t err = {};
                err.code = st;
                throw_carquet_error(
                    err, "ParquetSource: cannot read payload column chunk "
                         "metadata for '" +
                             path_ + "' row group " + std::to_string(rg));
            }
            bool dict = md.has_dictionary_page;
            for (int32_t e = 0; e < md.num_encodings && e < 4; ++e) {
                if (md.encodings[e] == CARQUET_ENCODING_PLAIN_DICTIONARY ||
                    md.encodings[e] == CARQUET_ENCODING_RLE_DICTIONARY) {
                    dict = true;
                }
            }
            if (dict) {
                payload_total_bytes_ +=
                    static_cast<uint64_t>(md.total_uncompressed_size) * 8;
                // Note: pyarrow and most writers dictionary-encode string
                // columns by default, so this is the COMMON case, not a
                // pathology — log once per file, not per row group.
                if (!dict_warned) {
                    dict_warned = true;
                    spdlog::warn(
                        "ParquetSource: payload column '{}' in '{}' is "
                        "dictionary-encoded — payload byte estimate boosted "
                        "8x for bitmap sizing (use --bitmap-pages to "
                        "override)",
                        config_.payload_col, path_);
                }
            } else {
                payload_total_bytes_ +=
                    static_cast<uint64_t>(md.total_uncompressed_size);
            }
        }
    }

    // For list<float> columns, find_column matches the leaf name (e.g. "item"),
    // not the parent field name (e.g. "emb"). Fall back to searching by
    // path prefix: a leaf whose path starts with the vector column name
    // and has repetition level >= 1.
    if (vec_col_idx_ < 0) {
        for (int32_t i = 0; i < n; ++i) {
            const char* path[8];
            int32_t depth = carquet_schema_column_path(
                file_schema, i, path, 8);
            if (depth > 0 && path[0] &&
                config_.vector_col == path[0]) {
                int16_t rep = carquet_schema_max_rep_level(file_schema, i);
                if (rep >= 1) {
                    vec_col_idx_ = i;
                    break;
                }
            }
        }
    }

    // Map leaf column index -> schema element node. carquet_schema_get_element
    // indexes ALL elements (root, groups, leaves), NOT leaf columns — calling
    // it with a column index misclassifies columns whenever a nested (list)
    // column shifts the element numbering (e.g. `language` resolved to the
    // emb.list group with max_rep=1 → typed Set → "read set column" errors).
    // Same leaf-walk the payload block above uses.
    std::vector<const carquet_schema_node_t*> leaf_nodes(
        static_cast<size_t>(n), nullptr);
    {
        const int32_t n_elem = carquet_schema_num_elements(file_schema);
        int32_t leaf_seen = 0;
        for (int32_t e = 1; e < n_elem; ++e) {
            const carquet_schema_node_t* node =
                carquet_schema_get_element(file_schema, e);
            if (carquet_schema_node_is_leaf(node)) {
                if (leaf_seen < n)
                    leaf_nodes[static_cast<size_t>(leaf_seen)] = node;
                ++leaf_seen;
            }
        }
    }

    for (int32_t i = 0; i < n; ++i) {
        if (i == vec_col_idx_) continue;
        if (i == payload_col_idx_) continue;

        const char* name = carquet_schema_column_name(file_schema, i);
        carquet_physical_type_t phys =
            carquet_schema_column_type(file_schema, i);
        const carquet_schema_node_t* node =
            leaf_nodes[static_cast<size_t>(i)];
        const carquet_logical_type_t* lt =
            node ? carquet_schema_node_logical_type(node) : nullptr;
        carquet_logical_type_id_t logical_id =
            lt ? lt->id : CARQUET_LOGICAL_UNKNOWN;
        int16_t max_rep = carquet_schema_node_max_rep_level(node);

        ColumnType ct = map_carquet_type(phys, logical_id, max_rep > 0);

        FilterColumn fc;
        fc.name = name ? name : ("col" + std::to_string(i));
        fc.type = ct;
        schema_.columns.push_back(std::move(fc));
        filter_col_indices_.push_back(i);
    }
}

void ParquetSource::detect_list_dim_() {
    reset();
    carquet_row_batch_t* raw_batch = nullptr;
    carquet_status_t status =
        carquet_batch_reader_next(batch_reader_.get(), &raw_batch);
    if (status != CARQUET_OK || !raw_batch) {
        carquet_error_t err = {};
        err.code = status;
        throw_carquet_error(err, "ParquetSource: cannot read batch to detect list dim");
    }

    {
        CarquetBatchPtr batch(raw_batch);
        const int32_t* offsets = nullptr;
        const void* values = nullptr;
        int64_t nlists = 0, nvals = 0;
        const uint8_t *vv = nullptr, *lv = nullptr;
        carquet_status_t s = carquet_row_batch_column_list(
            batch.get(), vec_col_idx_, &offsets, &nlists,
            &values, &vv, &nvals, &lv);
        if (s != CARQUET_OK || nlists == 0) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetSource: list column has no rows");
        }
        dim_ = static_cast<Dim>(offsets[1] - offsets[0]);
        if (dim_ == 0) {
            throw Error(ErrorCode::InvalidParam,
                        "ParquetSource: list column has zero-length vectors");
        }
    }
    // Release the batch (via CarquetBatchPtr destructor in the inner scope)
    // before resetting the batch reader.
    cursor_ = 0;
    batch_reader_.reset();
}

void ParquetSource::reset() {
    cursor_ = 0;
    batch_reader_.reset();
    chunk_col_ptrs_.clear();

    carquet_error_t err = {};
    carquet_batch_reader_config_t bcfg;
    carquet_batch_reader_config_init(&bcfg);
    bcfg.batch_size = config_.batch_size;
    bcfg.num_threads = config_.num_threads;
    bcfg.use_mmap = config_.use_mmap;
    // Vector-only streaming (training/Lloyd passes): project just the
    // vector column so carquet never decompresses filter/payload pages.
    const char* vec_name = config_.vector_col.c_str();
    if (vector_only_) {
        bcfg.column_names = &vec_name;
        bcfg.num_column_names = 1;
    }

    batch_reader_.reset(
        carquet_batch_reader_create(reader_.get(), &bcfg, &err));
    if (!batch_reader_) {
        throw_carquet_error(err, "ParquetSource::reset: batch reader create");
    }
}

bool ParquetSource::next(Chunk& out) {
    if (!batch_reader_) {
        reset();
    }

    carquet_row_batch_t* raw_batch = nullptr;
    carquet_status_t status =
        carquet_batch_reader_next(batch_reader_.get(), &raw_batch);

    if (status == CARQUET_ERROR_END_OF_DATA) {
        out.count = 0;
        return false;
    }
    if (status != CARQUET_OK || !raw_batch) {
        carquet_error_t err = {};
        err.code = status;
        throw_carquet_error(err, "ParquetSource::next: batch read");
    }

    CarquetBatchPtr batch(raw_batch);
    const int64_t n_rows = carquet_row_batch_num_rows(batch.get());
    if (n_rows == 0) {
        out.count = 0;
        return false;
    }

    const uint32_t n = static_cast<uint32_t>(n_rows);
    materialize_batch_(batch.get());

    out.vectors = vec_ptr_;
    out.row_ids = rowid_buf_.data();
    out.count = n;

    if (!chunk_col_ptrs_.empty()) {
        out.filter_columns = chunk_col_ptrs_.data();
    } else {
        out.filter_columns = nullptr;
    }
    if (payload_col_idx_ >= 0 && !vector_only_) {
        out.payload_data = payload_data_.data();
        out.payload_offsets = payload_offsets_.data();
    } else {
        out.payload_data = nullptr;
        out.payload_offsets = nullptr;
    }

    cursor_ += n;
    return true;
}

void ParquetSource::materialize_batch_(carquet_row_batch_t* batch) {
    const int64_t n_rows = carquet_row_batch_num_rows(batch);
    const uint32_t n = static_cast<uint32_t>(n_rows);

    // --- Vectors ---
    if (dim_ == 0) {
        // No vector column (label-only reader). Just set vec_ptr_ to null.
        vec_ptr_ = nullptr;
    } else if (vec_is_list_) {
        const int32_t* offsets = nullptr;
        const void* values = nullptr;
        int64_t nlists = 0, nvals = 0;
        const uint8_t *vv = nullptr, *lv = nullptr;
        carquet_status_t s = carquet_row_batch_column_list(
            batch, vec_col_idx_, &offsets, &nlists,
            &values, &vv, &nvals, &lv);
        if (s != CARQUET_OK) {
            carquet_error_t err = {};
            err.code = s;
            throw_carquet_error(err, "ParquetSource: read list vector column");
        }
        const float* fv = static_cast<const float*>(values);

        if (config_.normalize) {
            vec_buf_.resize(static_cast<size_t>(n) * dim_);
            for (uint32_t i = 0; i < n; ++i) {
                std::memcpy(&vec_buf_[i * dim_], fv + i * dim_,
                            static_cast<size_t>(dim_) * sizeof(float));
                simd::normalize_row_f32(&vec_buf_[i * dim_], dim_);
            }
            vec_ptr_ = vec_buf_.data();
        } else {
            // values is contiguous n×dim float — alias carquet's decoded buffer.
            vec_ptr_ = fv;
        }
    } else {
        const void* data = nullptr;
        const uint8_t* nulls = nullptr;
        int64_t count = 0;
        carquet_status_t s = carquet_row_batch_column(
            batch, vec_col_idx_, &data, &nulls, &count);
        if (s != CARQUET_OK) {
            carquet_error_t err = {};
            err.code = s;
            throw_carquet_error(err, "ParquetSource: read vector column");
        }

        const carquet_physical_type_t vec_phys =
            carquet_schema_column_type(
                carquet_reader_schema(reader_.get()), vec_col_idx_);

        const bool can_zero_copy =
            config_.use_mmap && nulls == nullptr &&
            (vec_phys == CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY ||
             vec_phys == CARQUET_PHYSICAL_FLOAT);

        if (can_zero_copy) {
            vec_ptr_ = static_cast<const float*>(data);
        } else if (dim_ == 1) {
            vec_buf_.resize(static_cast<size_t>(n));
            if (vec_phys == CARQUET_PHYSICAL_DOUBLE) {
                const auto* dptr = static_cast<const double*>(data);
                for (uint32_t i = 0; i < n; ++i)
                    vec_buf_[i] = static_cast<float>(dptr[i]);
            } else {
                std::memcpy(vec_buf_.data(), data, n * sizeof(float));
            }
            vec_ptr_ = vec_buf_.data();
            if (nulls) {
                for (uint32_t i = 0; i < n; ++i) {
                    if (!(nulls[i / 8] & (1 << (i % 8)))) {
                        vec_buf_[i] = 0.0f;
                    }
                }
            }
        } else {
            vec_buf_.resize(static_cast<size_t>(n) * dim_);
            std::memcpy(vec_buf_.data(), data,
                        static_cast<size_t>(n) * vec_type_len_);
            vec_ptr_ = vec_buf_.data();
            if (nulls) {
                for (uint32_t i = 0; i < n; ++i) {
                    if (!(nulls[i / 8] & (1 << (i % 8)))) {
                        std::fill_n(&vec_buf_[i * dim_], dim_, 0.0f);
                    }
                }
            }
        }
    }

    // --- Row IDs ---
    rowid_buf_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        rowid_buf_[i] = static_cast<RowId>(cursor_ + i);
    }

    // --- Filter columns ---
    // Skipped entirely in vector-only mode (the columns are not projected —
    // reading them would fail).
    const uint32_t n_filter = vector_only_
        ? 0u
        : static_cast<uint32_t>(filter_col_indices_.size());

    if (fixed_data_.size() < n_filter) fixed_data_.resize(n_filter);
    if (str_cols_.size() < n_filter) str_cols_.resize(n_filter);
    if (str_offsets_.size() < n_filter) str_offsets_.resize(n_filter);
    if (str_lengths_.size() < n_filter) str_lengths_.resize(n_filter);
    if (str_data_.size() < n_filter) str_data_.resize(n_filter);
    if (set_cols_.size() < n_filter) set_cols_.resize(n_filter);
    if (set_counts_.size() < n_filter) set_counts_.resize(n_filter);
    if (set_offsets_.size() < n_filter) set_offsets_.resize(n_filter);
    if (set_elem_lengths_.size() < n_filter) set_elem_lengths_.resize(n_filter);
    if (set_elem_data_.size() < n_filter) set_elem_data_.resize(n_filter);

    chunk_col_ptrs_.resize(n_filter, nullptr);

    for (uint32_t fi = 0; fi < n_filter; ++fi) {
        const int32_t col_idx = filter_col_indices_[fi];
        ColumnType ct = schema_.columns[fi].type;

        if (is_fixed_width(ct)) {
            const void* data = nullptr;
            const uint8_t* nulls = nullptr;
            int64_t count = 0;
            carquet_status_t s = carquet_row_batch_column(
                batch, col_idx, &data, &nulls, &count);
            if (s != CARQUET_OK) {
                carquet_error_t err = {};
                err.code = s;
                throw_carquet_error(err,
                                    "ParquetSource: read filter column " +
                                        std::to_string(col_idx));
            }

            uint8_t w = column_type_width(ct);
            auto& buf = fixed_data_[fi];
            buf.resize(static_cast<size_t>(n) * w);
            std::memcpy(buf.data(), data, static_cast<size_t>(n) * w);

            if (nulls) {
                for (uint32_t i = 0; i < n; ++i) {
                    if (!(nulls[i / 8] & (1 << (i % 8)))) {
                        std::memset(buf.data() + i * w, 0, w);
                    }
                }
            }

            chunk_col_ptrs_[fi] = buf.data();
        } else if (ct == ColumnType::String) {
            const void* data = nullptr;
            const uint8_t* nulls = nullptr;
            int64_t count = 0;
            carquet_status_t s = carquet_row_batch_column(
                batch, col_idx, &data, &nulls, &count);
            if (s != CARQUET_OK) {
                carquet_error_t err = {};
                err.code = s;
                throw_carquet_error(err,
                                    "ParquetSource: read string column " +
                                        std::to_string(col_idx));
            }

            const auto* barr =
                static_cast<const carquet_byte_array_t*>(data);

            auto& offsets = str_offsets_[fi];
            auto& lengths = str_lengths_[fi];
            auto& sdata = str_data_[fi];
            offsets.resize(n);
            lengths.resize(n);

            uint32_t total = 0;
            for (uint32_t i = 0; i < n; ++i) {
                offsets[i] = total;
                int32_t len = (nulls && !(nulls[i / 8] & (1 << (i % 8))))
                                  ? 0
                                  : barr[i].length;
                lengths[i] = static_cast<uint16_t>(len);
                total += static_cast<uint32_t>(len);
            }

            sdata.resize(total);
            for (uint32_t i = 0; i < n; ++i) {
                if (lengths[i] > 0) {
                    std::memcpy(sdata.data() + offsets[i],
                                barr[i].data, lengths[i]);
                }
            }

            auto& sc = str_cols_[fi];
            sc.offsets = offsets.data();
            sc.lengths = lengths.data();
            sc.data = sdata.data();
            sc.total_data_bytes = total;
            chunk_col_ptrs_[fi] = &sc;
        } else if (ct == ColumnType::Set) {
            const int32_t* list_offsets = nullptr;
            int64_t num_lists = 0;
            const void* values = nullptr;
            const uint8_t* value_validity = nullptr;
            int64_t num_values = 0;
            const uint8_t* list_validity = nullptr;

            carquet_status_t s = carquet_row_batch_column_list(
                batch, col_idx, &list_offsets, &num_lists, &values,
                &value_validity, &num_values, &list_validity);
            if (s != CARQUET_OK) {
                carquet_error_t err = {};
                err.code = s;
                throw_carquet_error(err,
                                    "ParquetSource: read set column " +
                                        std::to_string(col_idx));
            }

            const auto* barr =
                static_cast<const carquet_byte_array_t*>(values);

            auto& counts = set_counts_[fi];
            auto& soffsets = set_offsets_[fi];
            auto& elens = set_elem_lengths_[fi];
            auto& edata = set_elem_data_[fi];
            counts.resize(n);
            soffsets.resize(n);

            uint32_t total_elems = 0;
            uint32_t total_bytes = 0;
            for (uint32_t i = 0; i < n; ++i) {
                soffsets[i] = total_elems;
                int32_t begin = list_offsets[i];
                int32_t end = list_offsets[i + 1];
                uint8_t cnt = static_cast<uint8_t>(end - begin);
                counts[i] = cnt;
                total_elems += cnt;
                for (int32_t e = begin; e < end; ++e) {
                    total_bytes += static_cast<uint32_t>(barr[e].length);
                }
            }

            elens.resize(total_elems);
            edata.resize(total_bytes);

            uint32_t elem_idx = 0;
            uint32_t byte_off = 0;
            for (uint32_t i = 0; i < n; ++i) {
                int32_t begin = list_offsets[i];
                int32_t end = list_offsets[i + 1];
                for (int32_t e = begin; e < end; ++e) {
                    elens[elem_idx] = static_cast<uint16_t>(barr[e].length);
                    std::memcpy(edata.data() + byte_off, barr[e].data,
                                barr[e].length);
                    byte_off += barr[e].length;
                    ++elem_idx;
                }
            }

            auto& ssc = set_cols_[fi];
            ssc.counts = counts.data();
            ssc.offsets = soffsets.data();
            ssc.element_lengths = elens.data();
            ssc.element_data = edata.data();
            ssc.total_elements = total_elems;
            ssc.total_data_bytes = total_bytes;
            chunk_col_ptrs_[fi] = &ssc;
        }
    }

    // --- Payload column (optional) ---
    // Packed as [n+1 u32 offsets][bytes]; nulls map to empty blobs. Both
    // BYTE_ARRAY (per-row ptr+len) and FIXED_LEN_BYTE_ARRAY (contiguous
    // stride) are accepted.
    // Skipped in vector-only mode (the column is not projected).
    if (payload_col_idx_ >= 0 && !vector_only_) {
        const void* data = nullptr;
        const uint8_t* nulls = nullptr;
        int64_t count = 0;
        carquet_status_t s = carquet_row_batch_column(
            batch, payload_col_idx_, &data, &nulls, &count);
        if (s != CARQUET_OK) {
            carquet_error_t err = {};
            err.code = s;
            throw_carquet_error(err, "ParquetSource: read payload column " +
                                          std::to_string(payload_col_idx_));
        }

        payload_offsets_.resize(n + 1);
        auto is_null = [&](uint32_t i) {
            return nulls && !(nulls[i / 8] & (1 << (i % 8)));
        };

        if (payload_type_len_ > 0) {  // FIXED_LEN_BYTE_ARRAY
            const auto* rows = static_cast<const uint8_t*>(data);
            payload_data_.resize(
                static_cast<size_t>(n) * payload_type_len_);
            uint32_t off = 0;
            for (uint32_t i = 0; i < n; ++i) {
                payload_offsets_[i] = off;
                if (!is_null(i)) {
                    if (static_cast<uint64_t>(off) + payload_type_len_ >
                        UINT32_MAX) {
                        throw Error(ErrorCode::InvalidParam,
                                    "ParquetSource: payload column total "
                                    "exceeds u32 offsets");
                    }
                    std::memcpy(payload_data_.data() + off,
                                rows + static_cast<size_t>(i) * payload_type_len_,
                                static_cast<size_t>(payload_type_len_));
                    off += static_cast<uint32_t>(payload_type_len_);
                }
            }
            payload_offsets_[n] = off;
            payload_data_.resize(off);
        } else {  // BYTE_ARRAY
            const auto* barr =
                static_cast<const carquet_byte_array_t*>(data);
            uint64_t total = 0;
            for (uint32_t i = 0; i < n; ++i) {
                payload_offsets_[i] = static_cast<uint32_t>(total);
                total += is_null(i) ? 0
                                    : static_cast<uint64_t>(barr[i].length);
                if (total > UINT32_MAX) {
                    throw Error(ErrorCode::InvalidParam,
                                "ParquetSource: payload column batch total "
                                "exceeds u32 offsets");
                }
            }
            payload_offsets_[n] = static_cast<uint32_t>(total);
            payload_data_.resize(total);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t len = payload_offsets_[i + 1] - payload_offsets_[i];
                if (len > 0) {
                    std::memcpy(payload_data_.data() + payload_offsets_[i],
                                barr[i].data, len);
                }
            }
        }
    }
}

}  // namespace sextant
