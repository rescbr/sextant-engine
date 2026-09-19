#pragma once

/// @file push_source.hpp
/// Bounded-memory staging for the C ABI push build (sextant_build_begin /
/// _push / _finish).
///
/// Pushed rows (vectors + filter columns + payload blobs) accumulate in a
/// memory buffer up to `staging_bytes`; overflow chunks spill to a temp
/// file next to the final output. At finish, StagedPushSource streams the
/// staged chunks back as a VectorSource — the build then runs the same
/// bounded-RAM per-chunk path the parquet sources use (filter columns and
/// payloads ride each Chunk, not full-materialized BuildConfig sidecars).
///
/// Multi-pass support: reset() replays memory + re-reads the spill file
/// (parquet sources re-read their files the same way). set_vector_only()
/// skips filter/payload deserialization for training passes.

#include <sextant/filter_column_data.hpp>
#include <sextant/schema.hpp>
#include <sextant/types.hpp>
#include <sextant/vector_source.hpp>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace sextant {

/// One staged chunk in memory-serializable form.
struct StagedChunk {
    uint32_t count = 0;
    std::vector<float> vectors;   // count × dim
    std::vector<RowId> row_ids;   // count
    // Per filter column, schema order. Fixed-width: raw bytes
    // (count × width). String: offsets+lengths+data (see serialize).
    // Set: counts+offsets+elem_lengths+elem_data.
    struct Column {
        std::vector<uint8_t> fixed;
        std::vector<uint32_t> str_offsets;
        std::vector<uint16_t> str_lengths;
        std::vector<char> str_data;
        std::vector<uint8_t> set_counts;
        std::vector<uint32_t> set_offsets;
        std::vector<uint16_t> set_elem_lengths;
        std::vector<char> set_elem_data;
        /// Per-row NULL flags (1 = NULL). Only filled for nullable columns;
        /// end_row pads non-null rows with 0.
        std::vector<uint8_t> nulls;
    };
    std::vector<Column> cols;
    std::vector<uint8_t> payload_data;
    std::vector<uint32_t> payload_offsets; // count+1
};

/// Accumulates pushed rows with a memory budget, spilling to disk.
class PushStager {
public:
    /// `dim`/`schema`/`has_payload` mirror sextant_build_begin;
    /// `spill_path` is the temp file location; `staging_bytes` bounds
    /// in-memory staging (0 = a small default).
    PushStager(uint32_t dim, Schema schema, bool has_payload,
               std::string spill_path, uint64_t staging_bytes);
    ~PushStager();

    /// Row-append surface (the C ABI decodes its structs and calls these).
    /// append_vector starts a new row; column appends follow in schema
    /// order. For a nullable column, call EITHER append_null(col) OR the
    /// value append — never both (append_null stores placeholder bytes so
    /// the dense arrays stay row-aligned).
    void append_vector(const float* v, RowId id);
    void append_fixed(uint32_t col, const uint8_t* bytes, uint32_t width);
    void append_string(uint32_t col, const char* s, uint32_t len);
    void append_set_row(uint32_t col, uint32_t n_elems, const char* const* elems,
                        const uint32_t* lens);
    /// Mark the current row's column `col` NULL (nullable columns only).
    void append_null(uint32_t col);
    void append_payload(const uint8_t* data, uint32_t len);
    /// Close the current row (flushes the chunk when it is full).
    void end_row();

    uint64_t rows() const { return rows_; }
    uint64_t payload_bytes() const { return payload_bytes_; }

    /// Build a streaming source over the staged data. Can be called once;
    /// the source keeps the stager's spill file alive (the stager must
    /// outlive the source).
    std::unique_ptr<class StagedPushSource> build_source();

private:
    friend class StagedPushSource;
    friend class StagedPushSourceAccess;
    void maybe_spill();
    void write_chunk(const StagedChunk& c);
    void read_chunk(StagedChunk& c, bool vector_only);

    uint32_t dim_;
    Schema schema_;
    bool has_payload_;
    std::string spill_path_;
    uint64_t staging_bytes_;
    uint64_t rows_ = 0;
    uint64_t payload_bytes_ = 0;

    StagedChunk current_;        // accumulating chunk
    uint64_t staged_bytes_ = 0;  // current_ + spilled-not-yet-written bytes
    std::vector<StagedChunk> memory_chunks_; // chunks kept in RAM
    uint64_t spilled_chunks_ = 0;
    FILE* spill_ = nullptr; // write handle (also the read-back handle:
                            // reopened for reading by the source)
    FILE* read_ = nullptr;  // read handle for multi-pass replay
};

/// VectorSource over PushStager's staged data.
class StagedPushSource : public VectorSource {
public:
    StagedPushSource(PushStager& stager);

    Dim dim() const override { return stager_.dim_; }
    uint64_t count() const override { return stager_.rows_; }
    void reset() override;
    bool next(Chunk& out) override;
    Schema schema() const override { return stager_.schema_; }
    uint64_t payload_total_bytes() const override { return stager_.payload_bytes_; }
    void set_vector_only(bool on) override { vector_only_ = on; }
    std::string path() const override { return stager_.spilled_chunks_ ? stager_.spill_path_ : std::string(); }

private:
    const StagedChunk* next_staged();

    PushStager& stager_;
    bool vector_only_ = false;
    size_t mem_idx_ = 0;
    // Read-back state for the spill file (handle lives in the stager).
    uint64_t remaining_spill_ = 0;
    // Durable buffers for chunk deserialization (kept per source).
    StagedChunk buf_;
    // Durable Chunk-facing buffers: filter column views (fixed columns
    // point into buf_; string/set columns build FilterColumnData views).
    std::vector<std::vector<uint8_t>> fixed_views_;
    std::vector<FilterStringColumn> str_views_;
    std::vector<FilterSetColumn> set_views_;
    std::vector<const void*> filter_ptrs_;
    std::vector<const uint8_t*> null_ptrs_;
    std::vector<uint32_t> payload_offsets_view_;
};

}  // namespace sextant
