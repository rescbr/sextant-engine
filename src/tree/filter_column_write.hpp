#pragma once

/// @file filter_column_write.hpp
/// Build-time helpers for writing filter column data and per-leaf filter
/// summaries to leaf extents (Phase C).
///
/// These functions are called during the leaf write phase of build_streaming_pca.
/// They are NOT on the search hot path. The on-disk layouts they produce are
/// documented in the architecture plan §3.3 (string), §3.3.1 (set), §3.4
/// (summary), §3.5 (leaf layout).
///
/// Column data passed in is scoped to a single leaf's rows: filter_cols[c]
/// holds the values for rows 0..count-1 of THIS leaf (NOT the global row_ids).
/// The build path reindexes global filter column data into this per-leaf form
/// during the emission pass.

#include "sextant/schema.hpp"
#include <sextant/column_data.hpp>
#include "filter_hash.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace sextant::tree {

/// Distinct-value threshold for bloom inclusion (architecture plan §3.4).
inline constexpr uint32_t kBloomThreshold = 150;
// kBloomK is defined in filter_hash.hpp (shared by write + read paths).

/// Round a byte offset up to 4-byte alignment. Used between filter columns
/// so that uint32_t* / uint16_t* casts on the packed layout are always safe.
/// The filter region starts at an 8-aligned offset (leaf_codes_offset is
/// aligned), and every column boundary is rounded to 4.
inline uint64_t align4(uint64_t v) {
    return (v + 3) & ~uint64_t(3);
}

// ===========================================================================
// Size computation
// ===========================================================================

/// Compute the total bytes of filter column data for a leaf.
/// `count` = number of vectors in this leaf. `filter_cols` = per-column data
/// scoped to this leaf's rows. `schema` determines column types/order.
inline uint64_t filter_columns_bytes(uint32_t count, const Schema& schema,
                                     const std::vector<ColumnData>& filter_cols) {
    if (schema.columns.empty() || filter_cols.empty()) return 0;
    uint64_t total = 0;
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool: {
                total += static_cast<uint64_t>(count) * column_type_width(col.type);
                break;
            }
            case ColumnType::String: {
                // offsets (u32) + lengths (u16) + pad + hashes (u32) + packed data.
                // Padding after the u16 array keeps the u32 hashes 4-aligned.
                total += static_cast<uint64_t>(count) * (4 + 2);
                total = align4(total);
                total += static_cast<uint64_t>(count) * 4;  // hashes
                const auto& fc = filter_cols[c];
                // Sum of this leaf's string lengths.
                uint64_t str_bytes = 0;
                for (uint32_t i = 0; i < count; ++i)
                    str_bytes += fc.str_lengths[i];
                total += str_bytes;
                break;
            }
            case ColumnType::Set: {
                // counts (u8) + pad + offsets (u32) + hashes (u32 × n_elem) + data.
                // Padding after the u8 array keeps the u32 offsets 4-aligned.
                total += static_cast<uint64_t>(count) * 1;
                total = align4(total);
                total += static_cast<uint64_t>(count) * 4;  // offsets
                const auto& fc = filter_cols[c];
                uint64_t n_elem = 0;
                uint64_t data_bytes = 0;
                for (uint32_t i = 0; i < count; ++i) {
                    const uint8_t ec = fc.set_counts[i];
                    n_elem += ec;
                    const uint32_t off = fc.set_offsets[i];
                    for (uint32_t e = 0; e < ec; ++e)
                        data_bytes += 2u + fc.set_elem_lengths[off + e];
                }
                total += n_elem * 4;  // hashes
                // element data: per element [u16 length][bytes].
                total += data_bytes;
                break;
            }
        }
        // Nullable columns append a per-row validity byte (1 = NULL)
        // after their data section.
        if (col.nullable) {
            total += static_cast<uint64_t>(count);
        }
        // Align to 4 bytes after each column so the next column's typed
        // arrays (uint32_t*, uint16_t*) are naturally aligned.
        total = align4(total);
    }
    return total;
}

// ===========================================================================
// Filter column data writer
// ===========================================================================

/// Write filter column data into `buf` (must have at least
/// filter_columns_bytes(count, schema, filter_cols) bytes).
/// Writes each column's on-disk layout in schema order:
///   fixed-width columns, then string columns, then set columns.
/// Returns bytes written.
inline uint64_t write_filter_columns(uint8_t* buf, uint32_t count,
                                     const Schema& schema,
                                     const std::vector<ColumnData>& filter_cols) {
    if (schema.columns.empty() || filter_cols.empty()) return 0;
    uint8_t* p = buf;

    // Precompute cumulative byte offsets into each set column's element data.
    // set_elem_lengths[k] is the length of element k; element bytes are packed
    // contiguously in set_elem_data. We need element→byte-offset to read values.
    std::vector<std::vector<uint32_t>> set_byte_offs(filter_cols.size());
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        if (schema.columns[c].type != ColumnType::Set) continue;
        const auto& fc = filter_cols[c];
        auto& offs = set_byte_offs[c];
        offs.resize(fc.set_elem_lengths.size() + 1, 0);
        for (size_t k = 0; k < fc.set_elem_lengths.size(); ++k)
            offs[k + 1] = offs[k] + fc.set_elem_lengths[k];
    }

    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        const auto& fc = filter_cols[c];
        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool: {
                const uint8_t w = column_type_width(col.type);
                if (count > 0)
                    std::memcpy(p, fc.fixed_data.data(), static_cast<size_t>(count) * w);
                p += static_cast<uint64_t>(count) * w;
                // (No per-column align here: nullable validity bytes are
                // appended below BEFORE the single shared align, matching
                // filter_columns_bytes. The old align here would pad
                // between data and validity for odd-width Bool columns.)
                break;
            }
            case ColumnType::String: {
                // [offsets: count × u32][lengths: count × u16][pad]
                // [hashes: count × u32][data: packed bytes]
                auto* offsets = reinterpret_cast<uint32_t*>(p);
                p += static_cast<uint64_t>(count) * 4;
                auto* lengths = reinterpret_cast<uint16_t*>(p);
                p += static_cast<uint64_t>(count) * 2;
                // Align hashes to 4 bytes (u16 array may leave p 2-aligned).
                p = buf + align4(static_cast<uint64_t>(p - buf));
                auto* hashes = reinterpret_cast<uint32_t*>(p);
                p += static_cast<uint64_t>(count) * 4;
                uint32_t acc = 0;
                for (uint32_t i = 0; i < count; ++i) {
                    const uint16_t len = fc.str_lengths[i];
                    offsets[i] = acc;
                    lengths[i] = len;
                    const char* s = fc.str_data.data() + fc.str_offsets[i];
                    hashes[i] = filter_hash(std::string_view(s, len));
                    std::memcpy(p, s, len);
                    p += len;
                    acc += len;
                }
                break;
            }
            case ColumnType::Set: {
                // [counts: count × u8][pad][offsets: count × u32 (into hashes)]
                // [hashes: packed u32][data: [u16 len][bytes] per element]
                uint8_t* counts = p;
                p += static_cast<uint64_t>(count) * 1;
                // Align offsets to 4 bytes (u8 array may leave p misaligned).
                p = buf + align4(static_cast<uint64_t>(p - buf));
                auto* offsets = reinterpret_cast<uint32_t*>(p);
                p += static_cast<uint64_t>(count) * 4;
                // Total elements → determines hashes region size.
                uint32_t total_elem = 0;
                for (uint32_t i = 0; i < count; ++i)
                    total_elem += fc.set_counts[i];
                uint32_t* hashes = reinterpret_cast<uint32_t*>(p);
                p += static_cast<uint64_t>(total_elem) * 4;
                const auto& ebo = set_byte_offs[c];
                uint32_t elem_acc = 0;
                for (uint32_t i = 0; i < count; ++i) {
                    const uint8_t ec = fc.set_counts[i];
                    counts[i] = ec;
                    offsets[i] = elem_acc;
                    const uint32_t off = fc.set_offsets[i];
                    for (uint32_t e = 0; e < ec; ++e) {
                        const uint16_t elen = fc.set_elem_lengths[off + e];
                        const char* s = fc.set_elem_data.data() + ebo[off + e];
                        hashes[elem_acc + e] = filter_hash(std::string_view(s, elen));
                        // Data region: [u16 length][bytes].
                        std::memcpy(p, &elen, 2);
                        p += 2;
                        std::memcpy(p, s, elen);
                        p += elen;
                    }
                    elem_acc += ec;
                }
                break;
            }
        }
        // Nullable columns: per-row validity byte (1 = NULL) after the
        // data section. A null row's stored value bytes are arbitrary
        // (writers store zeros/empty) — evaluation never reads them.
        if (col.nullable) {
            if (count > 0) {
                if (!fc.null_mask.empty()) {
                    std::memcpy(p, fc.null_mask.data(), count);
                } else {
                    std::memset(p, 0, count);
                }
            }
            p += static_cast<uint64_t>(count);
        }
        // Align the write pointer to 4 bytes after each column, matching
        // filter_columns_bytes (which adds the same padding).
        p = buf + align4(static_cast<uint64_t>(p - buf));
    }
    return static_cast<uint64_t>(p - buf);
}

// ===========================================================================
// Filter summary writer
// ===========================================================================

namespace detail {

/// Read a numeric column value as a double (for min/max).
inline double read_numeric(const ColumnData& col, uint32_t i) {
    switch (col.type) {
        case ColumnType::Int32: {
            int32_t v;
            std::memcpy(&v, col.fixed_data.data() + static_cast<size_t>(i) * 4, 4);
            return static_cast<double>(v);
        }
        case ColumnType::Int64: {
            int64_t v;
            std::memcpy(&v, col.fixed_data.data() + static_cast<size_t>(i) * 8, 8);
            return static_cast<double>(v);
        }
        case ColumnType::Float: {
            float v;
            std::memcpy(&v, col.fixed_data.data() + static_cast<size_t>(i) * 4, 4);
            return static_cast<double>(v);
        }
        default:
            return 0.0;
    }
}

/// Collect distinct hash values in a string column.
inline std::vector<uint32_t> distinct_hashes_string(const ColumnData& col,
                                                    uint32_t count) {
    std::unordered_set<uint32_t> seen;
    seen.reserve(count);
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < count; ++i) {
        if (i < col.null_mask.size() && col.null_mask[i]) continue;  // NULL
        const uint16_t len = col.str_lengths[i];
        const char* s = col.str_data.data() + col.str_offsets[i];
        const uint32_t h = filter_hash(std::string_view(s, len));
        if (seen.insert(h).second) out.push_back(h);
    }
    return out;
}

/// Collect distinct hash values in a set column (across all rows + elements).
inline std::vector<uint32_t> distinct_hashes_set(const ColumnData& col,
                                                 uint32_t count) {
    // Build element byte offsets.
    std::vector<uint32_t> elem_byte_off(col.set_elem_lengths.size() + 1, 0);
    for (size_t k = 0; k < col.set_elem_lengths.size(); ++k)
        elem_byte_off[k + 1] = elem_byte_off[k] + col.set_elem_lengths[k];

    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < count; ++i) {
        if (i < col.null_mask.size() && col.null_mask[i]) continue;  // NULL
        const uint8_t ec = col.set_counts[i];
        const uint32_t off = col.set_offsets[i];
        for (uint32_t e = 0; e < ec; ++e) {
            const uint16_t elen = col.set_elem_lengths[off + e];
            const char* s = col.set_elem_data.data() + elem_byte_off[off + e];
            const uint32_t h = filter_hash(std::string_view(s, elen));
            if (seen.insert(h).second) out.push_back(h);
        }
    }
    return out;
}

}  // namespace detail

/// Write the filter summary into `buf` (must be summary_size bytes, already
/// zeroed). Computes min/max for numeric columns, blooms for string/set
/// columns (if distinct ≤ kBloomThreshold).
///
/// Summary layout (plan §3.4):
///   [n_numeric:u8]
///   [n_numeric × (col_id:u8, min:8B as double, max:8B as double)]
///   [n_string_bloom:u8]
///   [n_string_bloom × (col_id:u8, n_bits:u16, bloom:bloom_max_bytes)]
///   [n_set_bloom:u8]
///   [n_set_bloom × (col_id:u8, n_bits:u16, bloom:bloom_max_bytes)]
///
/// Numeric min/max are stored as 8-byte double for all numeric types
/// (int32/int64/float). This keeps the layout uniform and simple; the search
/// path compares as double. (int64 values up to 2^53 are exact as double.)
inline void write_filter_summary(uint8_t* buf, uint32_t summary_size,
                                 const Schema& schema, uint32_t count,
                                 const std::vector<ColumnData>& filter_cols) {
    if (summary_size == 0) return;
    // buf is pre-zeroed by the caller; we only write the fields we set.

    const uint32_t bloom_alloc = Schema::bloom_max_bytes(kBloomThreshold);

    // --- Numeric section ---
    uint8_t* p = buf;
    uint8_t* n_numeric_ptr = p;
    *n_numeric_ptr = 0;
    p += 1;
    uint8_t n_numeric = 0;
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        if (col.type != ColumnType::Int32 && col.type != ColumnType::Int64 &&
            col.type != ColumnType::Float)
            continue;
        double lo = std::numeric_limits<double>::max();
        double hi = std::numeric_limits<double>::lowest();
        const auto& nulls = filter_cols[c].null_mask;
        for (uint32_t i = 0; i < count; ++i) {
            if (i < nulls.size() && nulls[i]) continue;  // NULL: skip
            const double v = detail::read_numeric(filter_cols[c], i);
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        *p = static_cast<uint8_t>(c);  // col_id
        p += 1;
        std::memcpy(p, &lo, 8); p += 8;
        std::memcpy(p, &hi, 8); p += 8;
        ++n_numeric;
    }
    *n_numeric_ptr = n_numeric;

    // --- String bloom section ---
    uint8_t* n_string_ptr = p;
    *n_string_ptr = 0;
    p += 1;
    uint8_t n_string = 0;
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        if (col.type != ColumnType::String) continue;
        *p = static_cast<uint8_t>(c);  // col_id
        p += 1;
        const auto distinct = detail::distinct_hashes_string(filter_cols[c], count);
        if (!distinct.empty() && distinct.size() <= kBloomThreshold) {
            // n_bits = ceil(-distinct × ln(0.01) / ln(2)²), rounded up to the
            // allocated byte budget (bloom_alloc × 8). The allocated space is
            // fixed; the bloom uses up to that many bits.
            constexpr double kLn2 = 0.6931471805599453;
            constexpr double kNegLn001 = 4.605170185988091;
            double raw_bits = static_cast<double>(distinct.size()) * kNegLn001 /
                              (kLn2 * kLn2);
            uint32_t n_bits = static_cast<uint32_t>(std::ceil(raw_bits));
            // Round up to a multiple of 8 (byte-aligned) and cap at allocated.
            n_bits = (n_bits + 7u) & ~7u;
            const uint32_t max_bits = bloom_alloc * 8u;
            if (n_bits > max_bits) n_bits = max_bits;
            if (n_bits == 0) n_bits = 8;  // need at least 1 byte
            std::memcpy(p, &n_bits, 2); p += 2;
            uint8_t* bloom = p;
            // bloom bytes = n_bits / 8 (≤ bloom_alloc); rest stays zero.
            for (uint32_t h : distinct) {
                const BloomProbe bp = bloom_probe(h);
                for (uint32_t k = 0; k < kBloomK; ++k) {
                    const uint32_t bit = bloom_bit(bp, k, n_bits);
                    bloom[bit / 8u] |= (1u << (bit % 8u));
                }
            }
            p += bloom_alloc;
        } else {
            // distinct > threshold (or empty): n_bits = 0, bloom bytes zeroed.
            const uint16_t zero = 0;
            std::memcpy(p, &zero, 2); p += 2;
            p += bloom_alloc;  // already zero
        }
        ++n_string;
    }
    *n_string_ptr = n_string;

    // --- Set bloom section ---
    uint8_t* n_set_ptr = p;
    *n_set_ptr = 0;
    p += 1;
    uint8_t n_set = 0;
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        if (col.type != ColumnType::Set) continue;
        *p = static_cast<uint8_t>(c);  // col_id
        p += 1;
        const auto distinct = detail::distinct_hashes_set(filter_cols[c], count);
        if (!distinct.empty() && distinct.size() <= kBloomThreshold) {
            constexpr double kLn2 = 0.6931471805599453;
            constexpr double kNegLn001 = 4.605170185988091;
            double raw_bits = static_cast<double>(distinct.size()) * kNegLn001 /
                              (kLn2 * kLn2);
            uint32_t n_bits = static_cast<uint32_t>(std::ceil(raw_bits));
            n_bits = (n_bits + 7u) & ~7u;
            const uint32_t max_bits = bloom_alloc * 8u;
            if (n_bits > max_bits) n_bits = max_bits;
            if (n_bits == 0) n_bits = 8;
            std::memcpy(p, &n_bits, 2); p += 2;
            uint8_t* bloom = p;
            for (uint32_t h : distinct) {
                const BloomProbe bp = bloom_probe(h);
                for (uint32_t k = 0; k < kBloomK; ++k) {
                    const uint32_t bit = bloom_bit(bp, k, n_bits);
                    bloom[bit / 8u] |= (1u << (bit % 8u));
                }
            }
            p += bloom_alloc;
        } else {
            const uint16_t zero = 0;
            std::memcpy(p, &zero, 2); p += 2;
            p += bloom_alloc;
        }
        ++n_set;
    }
    *n_set_ptr = n_set;
    (void)summary_size;  // size is enforced by the caller's allocation
}

// ===========================================================================
// Summary merge (OR) — for propagating summaries up the tree
// ===========================================================================

/// Merge (OR) summary `src` into summary `dst`. Both must be summary_size bytes
/// and follow the same schema. Numeric: min of mins, max of maxs. Blooms:
/// bitwise OR (n_bits = max of the two). The merge is in-place into `dst`.
/// `src` and `dst` must NOT overlap.
inline void merge_filter_summary(uint8_t* dst, const uint8_t* src,
                                  uint32_t summary_size) {
    if (summary_size == 0) return;

    const uint8_t* sp = src;
    uint8_t* dp = dst;
    const uint32_t bloom_alloc = Schema::bloom_max_bytes(kBloomThreshold);

    // --- Numeric section ---
    uint8_t s_n_numeric = *sp++;
    uint8_t d_n_numeric = *dp++;  // advance dst past count byte (align with sp)
    // Both must have the same n_numeric (same schema). Assert-ish:
    // We merge by matching col_id. Since the schema is the same, the
    // numeric entries are in the same order with the same col_ids.
    // Simple approach: iterate dst's entries, find matching src entry by col_id.
    // For efficiency (both are in schema order), we assume they're aligned.
    // Build a col_id → offset map for src.
    // Actually, since both are written from the same schema in the same order,
    // the entries are identical in order. So entry i in src corresponds to
    // entry i in dst. We can merge positionally.
    for (uint8_t i = 0; i < d_n_numeric; ++i) {
        // dst entry: [col_id:u8][min:8B][max:8B]
        const uint8_t d_col = dp[0];  // col_id (dp points at this entry)
        // Find matching src entry. Since both are in schema order,
        // src entry i has the same col_id. But to be safe, search.
        const uint8_t* s_scan = sp;
        for (uint8_t j = 0; j < s_n_numeric; ++j) {
            const uint8_t s_col = s_scan[0];
            if (s_col == d_col) {
                // Merge: min of mins, max of maxs.
                double s_min, s_max, d_min, d_max;
                std::memcpy(&s_min, s_scan + 1, 8);
                std::memcpy(&s_max, s_scan + 9, 8);
                std::memcpy(&d_min, dp + 1, 8);
                std::memcpy(&d_max, dp + 9, 8);
                d_min = std::min(d_min, s_min);
                d_max = std::max(d_max, s_max);
                std::memcpy(dp + 1, &d_min, 8);
                std::memcpy(dp + 9, &d_max, 8);
                    break;
            }
            s_scan += 17;
        }
        dp += 17;  // advance past this dst entry
    }
    // sp already points past the numeric section (advanced per entry? no —
    // s_scan is a local; sp itself was only advanced past the count byte).
    // Advance sp past the numeric section to the n_string_bloom count byte.
    sp = src + 1 + static_cast<size_t>(s_n_numeric) * 17;
    // dp is at dst + 1 + d_n_numeric * 17 — the n_string_bloom byte.

    // --- String bloom section ---
    {
        uint8_t s_n = *sp++;
        uint8_t d_n = *dp;
        ++dp;  // skip n_string_bloom byte (keep d_n; both are same schema)
        for (uint8_t i = 0; i < d_n; ++i) {
            // dst entry: [col_id:u8][n_bits:u16][bloom:bloom_alloc]
            const uint8_t d_col = dp[0];
            uint16_t d_nbits;
            std::memcpy(&d_nbits, dp + 1, 2);

            // Find matching src entry by col_id.
            const uint8_t* s_scan = sp;
            for (uint8_t j = 0; j < s_n; ++j) {
                if (s_scan[0] == d_col) {
                    uint16_t s_nbits;
                    std::memcpy(&s_nbits, s_scan + 1, 2);
                    // Use the larger n_bits (covers more hash space).
                    uint16_t merged_nbits = std::max(d_nbits, s_nbits);
                    std::memcpy(dp + 1, &merged_nbits, 2);
                    // OR the bloom bytes (over bloom_alloc bytes).
                    // Use the merged nbits to determine how many bytes to OR.
                    // But since blooms may have different n_bits, OR all
                    // bloom_alloc bytes (the bits beyond n_bits are zero).
                    for (uint32_t b = 0; b < bloom_alloc; ++b)
                        dp[3 + b] |= s_scan[3 + b];
                    break;
                }
                s_scan += 3 + bloom_alloc;
            }
            dp += 3 + bloom_alloc;
        }
        // Advance sp past string section.
        sp = src + 1 + static_cast<size_t>(s_n_numeric) * 17
             + 1 + static_cast<size_t>(s_n) * (3 + bloom_alloc);
    }

    // --- Set bloom section ---
    {
        uint8_t s_n = *sp++;
        uint8_t d_n = *dp;
        ++dp;
        for (uint8_t i = 0; i < d_n; ++i) {
            const uint8_t d_col = dp[0];
            uint16_t d_nbits;
            std::memcpy(&d_nbits, dp + 1, 2);

            const uint8_t* s_scan = sp;
            for (uint8_t j = 0; j < s_n; ++j) {
                if (s_scan[0] == d_col) {
                    uint16_t s_nbits;
                    std::memcpy(&s_nbits, s_scan + 1, 2);
                    uint16_t merged_nbits = std::max(d_nbits, s_nbits);
                    std::memcpy(dp + 1, &merged_nbits, 2);
                    for (uint32_t b = 0; b < bloom_alloc; ++b)
                        dp[3 + b] |= s_scan[3 + b];
                    break;
                }
                s_scan += 3 + bloom_alloc;
            }
            dp += 3 + bloom_alloc;
        }
    }
}

/// Initialize a summary buffer to "empty" (no matches / widest range).
/// Numeric: min=+inf, max=-inf. Blooms: all zero.
/// This is the identity element for merge_filter_summary — merging into
/// this produces the other summary.
inline void init_empty_summary(uint8_t* buf, uint32_t summary_size,
                                const Schema& schema) {
    if (summary_size == 0) return;
    std::memset(buf, 0, summary_size);

    uint8_t* p = buf;
    // Count numeric columns.
    uint8_t n_numeric = 0;
    for (const auto& col : schema.columns) {
        if (col.type == ColumnType::Int32 || col.type == ColumnType::Int64 ||
            col.type == ColumnType::Float)
            ++n_numeric;
    }
    *p++ = n_numeric;
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        if (col.type != ColumnType::Int32 && col.type != ColumnType::Int64 &&
            col.type != ColumnType::Float)
            continue;
        *p++ = static_cast<uint8_t>(c);  // col_id
        double pos_inf = std::numeric_limits<double>::max();
        double neg_inf = std::numeric_limits<double>::lowest();
        std::memcpy(p, &pos_inf, 8); p += 8;  // min = +inf
        std::memcpy(p, &neg_inf, 8); p += 8;  // max = -inf
    }

    // String blooms: count + entries with n_bits=0.
    uint8_t n_string = 0;
    for (const auto& col : schema.columns)
        if (col.type == ColumnType::String) ++n_string;
    *p++ = n_string;
    const uint32_t bloom_alloc = Schema::bloom_max_bytes(kBloomThreshold);
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        if (schema.columns[c].type != ColumnType::String) continue;
        *p++ = static_cast<uint8_t>(c);  // col_id
        uint16_t zero = 0;
        std::memcpy(p, &zero, 2); p += 2;
        p += bloom_alloc;  // zeroed by memset
    }

    // Set blooms: same structure.
    uint8_t n_set = 0;
    for (const auto& col : schema.columns)
        if (col.type == ColumnType::Set) ++n_set;
    *p++ = n_set;
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        if (schema.columns[c].type != ColumnType::Set) continue;
        *p++ = static_cast<uint8_t>(c);
        uint16_t zero = 0;
        std::memcpy(p, &zero, 2); p += 2;
        p += bloom_alloc;
    }
}

}  // namespace sextant::tree
