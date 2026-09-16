#include "filter_scan.hpp"
#include "leaf_coder.hpp"  // LeafGeometry
#include "filter_column_write.hpp"  // kBloomThreshold

#include <algorithm>
#include <cmath>

namespace sextant::tree {

// ===========================================================================
// LeafFilterLayout
// ===========================================================================

LeafFilterLayout LeafFilterLayout::compute(const uint8_t* leaf_ptr, uint16_t m4,
                                            uint8_t pq_bits,
                                            uint32_t summary_size) {
    LeafFilterLayout l;
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
    l.count = static_cast<uint32_t>(lh->count);
    l.summary_size = summary_size;
    l.codes_per_block = (pq_bits == 4) ? 32 : 16;
    l.block_bytes = m4 * 16;
    l.n_blocks = (l.count + l.codes_per_block - 1) / l.codes_per_block;
    l.codes = leaf_ptr + leaf_codes_offset(summary_size);
    l.row_ids = reinterpret_cast<const RowId*>(
        leaf_ptr + leaf_rowids_offset(summary_size, l.n_blocks, l.block_bytes));
    l.filter_base = leaf_ptr + leaf_rowids_offset(summary_size, l.n_blocks,
                                                     l.block_bytes)
                     + static_cast<uint64_t>(l.count) * sizeof(RowId);
    return l;
}

LeafFilterLayout LeafFilterLayout::from_geometry(
        const uint8_t* leaf_ptr, const LeafGeometry& geo) {
    LeafFilterLayout l;
    const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
    l.count = static_cast<uint32_t>(lh->count);
    l.summary_size = lh->summary_size;
    l.codes_per_block = (lh->pq_bits == 4) ? 32 : 16;
    l.block_bytes = lh->m4 * 16;
    l.n_blocks = (l.count + l.codes_per_block - 1) / l.codes_per_block;
    l.codes = leaf_ptr + geo.codes_offset;
    l.row_ids = reinterpret_cast<const RowId*>(leaf_ptr + geo.rowids_offset);
    l.filter_base = leaf_ptr + geo.filter_offset;
    return l;
}

// ===========================================================================
// parse_filter_columns
// ===========================================================================

std::vector<ColumnView> parse_filter_columns(const uint8_t* filter_base,
                                              uint32_t count,
                                              const Schema& schema) {
    std::vector<ColumnView> views(schema.columns.size());
    const uint8_t* p = filter_base;

    // Single pass in SCHEMA ORDER, mirroring write_filter_columns /
    // filter_columns_bytes exactly: columns are laid out in the order they
    // appear in the schema, each followed by align4 padding. (The old
    // three-pass form — fixed, then strings, then sets — only agreed with
    // the writer when the schema happened to be type-grouped; a schema like
    // [String, String, String, Int64, ...] misaligned every parse.)
    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto& col = schema.columns[c];
        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float:
            case ColumnType::Bool: {
                views[c].type = col.type;
                views[c].fixed_width = column_type_width(col.type);
                views[c].fixed_base = p;
                p += static_cast<uint64_t>(count) * views[c].fixed_width;
                break;
            }
            case ColumnType::String: {
                views[c].type = ColumnType::String;
                views[c].str_offsets = reinterpret_cast<const uint32_t*>(p);
                p += static_cast<uint64_t>(count) * 4;
                views[c].str_lengths = reinterpret_cast<const uint16_t*>(p);
                p += static_cast<uint64_t>(count) * 2;
                // Align hashes to 4 bytes (u16 array may leave p 2-aligned).
                p = filter_base + align4(static_cast<uint64_t>(p - filter_base));
                views[c].str_hashes = reinterpret_cast<const uint32_t*>(p);
                p += static_cast<uint64_t>(count) * 4;
                views[c].str_data = reinterpret_cast<const char*>(p);
                // Advance past the packed string data: last string's offset
                // + length.
                if (count > 0) {
                    const uint32_t last_off = views[c].str_offsets[count - 1];
                    const uint16_t last_len = views[c].str_lengths[count - 1];
                    p += last_off + last_len;
                }
                break;
            }
            case ColumnType::Set: {
                views[c].type = ColumnType::Set;
                views[c].set_counts = p;
                p += static_cast<uint64_t>(count) * 1;
                // Align offsets to 4 bytes (u8 array may leave p misaligned).
                p = filter_base + align4(static_cast<uint64_t>(p - filter_base));
                views[c].set_offsets = reinterpret_cast<const uint32_t*>(p);
                p += static_cast<uint64_t>(count) * 4;
                // Total elements across all rows.
                uint32_t total_elem = 0;
                for (uint32_t i = 0; i < count; ++i)
                    total_elem += views[c].set_counts[i];
                views[c].set_hashes = reinterpret_cast<const uint32_t*>(p);
                p += static_cast<uint64_t>(total_elem) * 4;
                views[c].set_data = p;
                // Advance past the packed element data. Each element is
                // [u16 len][bytes] — walk the records to total the size.
                if (total_elem > 0) {
                    uint64_t data_bytes = 0;
                    for (uint32_t e = 0; e < total_elem; ++e) {
                        uint16_t elen;
                        std::memcpy(&elen, views[c].set_data + data_bytes, 2);
                        data_bytes += 2 + elen;
                    }
                    p += data_bytes;
                }
                break;
            }
        }
        // Align after EVERY column, matching the writer's per-column pad.
        p = filter_base + align4(static_cast<uint64_t>(p - filter_base));
    }

    return views;
}

// ===========================================================================
// Per-candidate predicate evaluation
// ===========================================================================

namespace {

/// Read a fixed-width value as double for comparison.
inline double read_fixed_as_double(const ColumnView& col, uint32_t idx) {
    const uint8_t* p = col.fixed_base + static_cast<size_t>(idx) * col.fixed_width;
    switch (col.type) {
        case ColumnType::Int32: {
            int32_t v;
            std::memcpy(&v, p, 4);
            return static_cast<double>(v);
        }
        case ColumnType::Int64: {
            int64_t v;
            std::memcpy(&v, p, 8);
            return static_cast<double>(v);
        }
        case ColumnType::Float: {
            float v;
            std::memcpy(&v, p, 4);
            return static_cast<double>(v);
        }
        case ColumnType::Bool:
            return static_cast<double>(p[0]);
        default:
            return 0.0;
    }
}

inline bool read_bool(const ColumnView& col, uint32_t idx) {
    return col.fixed_base[static_cast<size_t>(idx)] != 0;
}

inline std::string_view read_string(const ColumnView& col, uint32_t idx) {
    const uint32_t off = col.str_offsets[idx];
    const uint16_t len = col.str_lengths[idx];
    return std::string_view(col.str_data + off, len);
}

inline uint32_t read_string_hash(const ColumnView& col, uint32_t idx) {
    return col.str_hashes[idx];
}

/// Check if a set column contains an element with the given hash.
/// Uses the precomputed hashes array — SIMD-comparable (4/cycle on NEON).
/// For small element counts (typical: 3-8), a linear scan is optimal.
inline bool set_contains_hash(const ColumnView& col, uint32_t idx,
                               uint32_t query_hash) {
    const uint8_t ec = col.set_counts[idx];
    const uint32_t off = col.set_offsets[idx];
    const uint32_t* hashes = col.set_hashes + off;
    // Linear scan over element hashes. For ec ≤ 255 (uint8 max), this is
    // at most 255 comparisons — typically 3-8. SIMD benefits kick in at
    // ec ≥ 4 but the overhead of setting up SIMD for tiny counts is worse
    // than a scalar loop. The hash array is contiguous and cache-friendly.
    for (uint8_t e = 0; e < ec; ++e) {
        if (hashes[e] == query_hash) return true;
    }
    return false;
}

/// Verify a set CONTAINS by exact string match (after hash pre-filter).
/// Reads the element data to verify.
inline bool set_contains_string(const ColumnView& col, uint32_t idx,
                                 std::string_view query) {
    const uint8_t ec = col.set_counts[idx];
    const uint32_t off = col.set_offsets[idx];
    // Walk element data: [u16 len][bytes] per element, starting at set_data.
    const uint8_t* dp = col.set_data;
    // Skip to element `off`.
    for (uint32_t e = 0; e < off; ++e) {
        uint16_t elen;
        std::memcpy(&elen, dp, 2);
        dp += 2 + elen;
    }
    // Now check elements [off .. off+ec).
    for (uint8_t e = 0; e < ec; ++e) {
        uint16_t elen;
        std::memcpy(&elen, dp, 2);
        dp += 2;
        if (elen == query.size() &&
            std::memcmp(dp, query.data(), elen) == 0) {
            return true;
        }
        dp += elen;
    }
    return false;
}

}  // namespace

bool eval_predicate(const ColumnView& col, uint32_t idx, const Predicate& pred) {
    switch (col.type) {
        // --- Numeric types ---
        case ColumnType::Int32:
        case ColumnType::Int64:
        case ColumnType::Float: {
            const double val = read_fixed_as_double(col, idx);
            switch (pred.op) {
                case PredicateOp::Eq:       return val == pred.value;
                case PredicateOp::NotEq:    return val != pred.value;
                case PredicateOp::Lt:       return val <  pred.value;
                case PredicateOp::Le:       return val <= pred.value;
                case PredicateOp::Gt:       return val >  pred.value;
                case PredicateOp::Ge:       return val >= pred.value;
                case PredicateOp::Between:  return val >= pred.value && val <= pred.value2;
                case PredicateOp::In: {
                    // `values` is the operand. The legacy scalar fast-path
                    // (value..value4) consulted slots shared with Eq/Between/
                    // geo — callers building In via values left them zeroed,
                    // silently matching every row holding 0. Removed.
                    for (const auto& s : pred.values) {
                        try { if (val == std::stod(s)) return true; }
                        catch (...) { continue; }
                    }
                    return false;
                }
                case PredicateOp::NotIn: {
                    for (const auto& s : pred.values) {
                        try { if (val == std::stod(s)) return false; }
                        catch (...) { continue; }
                    }
                    return true;
                }
                default: return true;  // geo/set ops don't apply to numeric
            }
        }

        // --- Bool ---
        case ColumnType::Bool: {
            const bool val = read_bool(col, idx);
            switch (pred.op) {
                case PredicateOp::Eq:    return val == (pred.value != 0);
                case PredicateOp::NotEq: return val != (pred.value != 0);
                default: return true;
            }
        }

        // --- String ---
        case ColumnType::String: {
            switch (pred.op) {
                case PredicateOp::Eq: {
                    // Hash pre-filter + exact verify.
                    const uint32_t qhash = filter_hash(pred.str_value);
                    if (read_string_hash(col, idx) != qhash) return false;
                    return read_string(col, idx) == pred.str_value;
                }
                case PredicateOp::NotEq: {
                    const uint32_t qhash = filter_hash(pred.str_value);
                    if (read_string_hash(col, idx) == qhash) {
                        // Hash matches — verify to be sure.
                        if (read_string(col, idx) == pred.str_value) return false;
                    }
                    return true;
                }
                case PredicateOp::Prefix: {
                    auto sv = read_string(col, idx);
                    if (sv.size() < pred.str_value.size()) return false;
                    return std::memcmp(sv.data(), pred.str_value.data(),
                                       pred.str_value.size()) == 0;
                }
                case PredicateOp::In: {
                    const auto sv = read_string(col, idx);
                    const uint32_t qhash = filter_hash(sv);
                    (void)qhash;  // Could pre-compute query hashes, but In lists are small.
                    for (const auto& s : {pred.str_value}) {
                        if (sv == s) return true;
                    }
                    for (const auto& s : pred.values) {
                        if (sv == s) return true;
                    }
                    return false;
                }
                case PredicateOp::NotIn: {
                    const auto sv = read_string(col, idx);
                    for (const auto& s : {pred.str_value}) {
                        if (sv == s) return false;
                    }
                    for (const auto& s : pred.values) {
                        if (sv == s) return false;
                    }
                    return true;
                }
                default: return true;  // geo/set ops don't apply
            }
        }

        // --- Set ---
        case ColumnType::Set: {
            switch (pred.op) {
                case PredicateOp::Contains: {
                    const uint32_t qhash = filter_hash(pred.str_value);
                    if (!set_contains_hash(col, idx, qhash)) return false;
                    return set_contains_string(col, idx, pred.str_value);
                }
                case PredicateOp::ContainsAny: {
                    std::vector<std::string> query_vals;
                    if (!pred.str_value.empty()) query_vals.push_back(pred.str_value);
                    for (const auto& s : pred.values) query_vals.push_back(s);
                    for (const auto& qv : query_vals) {
                        const uint32_t qhash = filter_hash(qv);
                        if (set_contains_hash(col, idx, qhash) &&
                            set_contains_string(col, idx, qv))
                            return true;
                    }
                    return false;
                }
                case PredicateOp::ContainsAll: {
                    std::vector<std::string> query_vals;
                    if (!pred.str_value.empty()) query_vals.push_back(pred.str_value);
                    for (const auto& s : pred.values) query_vals.push_back(s);
                    for (const auto& qv : query_vals) {
                        const uint32_t qhash = filter_hash(qv);
                        if (!set_contains_hash(col, idx, qhash)) return false;
                        if (!set_contains_string(col, idx, qv)) return false;
                    }
                    return true;
                }
                default: return true;
            }
        }
    }
    return true;  // unreachable
}

// ===========================================================================
// Geo predicate evaluation (GeoBox, GeoRadius)
// ===========================================================================

bool eval_predicate_geo(const ColumnView& lat_col, const ColumnView& lng_col,
                         uint32_t idx, const Predicate& pred) {
    // Geo columns are stored as Float (latitude, longitude). Read both values.
    const double lat = read_fixed_as_double(lat_col, idx);
    const double lng = read_fixed_as_double(lng_col, idx);

    switch (pred.op) {
        case PredicateOp::GeoBox: {
            // value=min_lat, value2=min_lng, value3=max_lat, value4=max_lng.
            return lat >= pred.value  && lat <= pred.value3 &&
                   lng >= pred.value2 && lng <= pred.value4;
        }
        case PredicateOp::GeoRadius: {
            // value=center_lat, value2=center_lng, radius_km=radius.
            return haversine_km(pred.value, pred.value2, lat, lng) <= pred.radius_km;
        }
        default:
            return true;  // not a geo op
    }
}

// ===========================================================================
// Summary-based leaf pruning
// ===========================================================================

bool summary_may_match(const uint8_t* summary, uint32_t summary_size,
                        const Schema& schema,
                        const std::vector<Predicate>& preds,
                        const std::vector<uint32_t>& pred_col_indices,
                        const std::vector<uint32_t>& geo_lng_col_indices) {
    if (summary_size == 0 || preds.empty()) return true;

    const uint8_t* p = summary;

    // --- Parse numeric section ---
    const uint8_t n_numeric = *p++;
    // col_id → (min, max) map. col_id is the schema column index.
    // We store as a simple array indexed by col_id (max 256 columns).
    // For efficiency, we parse the summary once per leaf and check all
    // predicates against it.
    struct NumRange { double min, max; bool present; };
    NumRange num_ranges[256] = {};
    for (int i = 0; i < 256; ++i) num_ranges[i].present = false;
    for (uint8_t i = 0; i < n_numeric; ++i) {
        const uint8_t col_id = *p++;
        double mn, mx;
        std::memcpy(&mn, p, 8); p += 8;
        std::memcpy(&mx, p, 8); p += 8;
        num_ranges[col_id] = {mn, mx, true};
    }

    // --- Parse string bloom section ---
    struct BloomEntry { uint16_t n_bits; const uint8_t* bloom; };
    BloomEntry str_blooms[256] = {};
    const uint8_t n_string = *p++;
    const uint32_t bloom_alloc = Schema::bloom_max_bytes(kBloomThreshold);
    for (uint8_t i = 0; i < n_string; ++i) {
        const uint8_t col_id = *p++;
        uint16_t n_bits;
        std::memcpy(&n_bits, p, 2); p += 2;
        str_blooms[col_id] = {n_bits, p};
        p += bloom_alloc;
    }

    // --- Parse set bloom section ---
    BloomEntry set_blooms[256] = {};
    const uint8_t n_set = *p++;
    for (uint8_t i = 0; i < n_set; ++i) {
        const uint8_t col_id = *p++;
        uint16_t n_bits;
        std::memcpy(&n_bits, p, 2); p += 2;
        set_blooms[col_id] = {n_bits, p};
        p += bloom_alloc;
    }

    // --- Check each predicate against the summary ---
    for (uint32_t pi = 0; pi < preds.size(); ++pi) {
        const auto& pred = preds[pi];
        const uint32_t col_idx = pred_col_indices[pi];
        const auto& col = schema.columns[col_idx];

        // --- Geo predicates span two columns (lat + lng) ---
        if (pred.op == PredicateOp::GeoBox || pred.op == PredicateOp::GeoRadius) {
            double lat_lo, lat_hi, lng_lo, lng_hi;
            if (pred.op == PredicateOp::GeoBox) {
                // value=min_lat, value2=min_lng, value3=max_lat, value4=max_lng.
                lat_lo = pred.value;   lat_hi = pred.value3;
                lng_lo = pred.value2;  lng_hi = pred.value4;
            } else {
                // GeoRadius: derive a bounding box of the circle.
                // 1° latitude ≈ 111.0 km (varies slightly; this is a pre-filter,
                // so a slight over-estimate is safe/conservative).
                constexpr double kKmPerDegLat = 111.0;
                constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
                const double dlat = pred.radius_km / kKmPerDegLat;
                const double cos_lat = std::cos(pred.value * kDegToRad);
                // Guard against division by zero near the poles.
                const double dlng = (std::fabs(cos_lat) < 1e-6)
                    ? 180.0
                    : pred.radius_km / (kKmPerDegLat * cos_lat);
                lat_lo = pred.value  - dlat;  lat_hi = pred.value  + dlat;
                lng_lo = pred.value2 - dlng;  lng_hi = pred.value2 + dlng;
            }

            // Latitude column summary (col_idx).
            if (num_ranges[col_idx].present) {
                const auto& r = num_ranges[col_idx];
                if (lat_lo > r.max || lat_hi < r.min) return false;
            }
            // Longitude column summary (from geo_lng_col_indices).
            const uint32_t lng_col = (pi < geo_lng_col_indices.size())
                ? geo_lng_col_indices[pi] : UINT32_MAX;
            if (lng_col < 256 && num_ranges[lng_col].present) {
                const auto& r = num_ranges[lng_col];
                if (lng_lo > r.max || lng_hi < r.min) return false;
            }
            continue;  // geo handled; skip the single-column switch below.
        }

        switch (col.type) {
            case ColumnType::Int32:
            case ColumnType::Int64:
            case ColumnType::Float: {
                if (!num_ranges[col_idx].present) continue;  // no summary → can't prune
                const auto& r = num_ranges[col_idx];
                switch (pred.op) {
                    case PredicateOp::Eq:
                        // If val is outside [min, max], leaf can't contain it.
                        if (pred.value < r.min || pred.value > r.max) return false;
                        break;
                    case PredicateOp::Lt:
                        if (r.min >= pred.value) return false;
                        break;
                    case PredicateOp::Le:
                        if (r.min > pred.value) return false;
                        break;
                    case PredicateOp::Gt:
                        if (r.max <= pred.value) return false;
                        break;
                    case PredicateOp::Ge:
                        if (r.max < pred.value) return false;
                        break;
                    case PredicateOp::Between:
                        // [value, value2] must overlap [min, max].
                        if (pred.value > r.max || pred.value2 < r.min) return false;
                        break;
                    case PredicateOp::NotEq:
                    case PredicateOp::In:
                    case PredicateOp::NotIn:
                        // Conservative: can't prune these at summary level.
                        // (NotEq: even if val==min, other values exist. In: multiple values.
                        // NotIn: other values exist.)
                        break;
                    default: break;
                }
                break;
            }

            case ColumnType::String: {
                const auto& be = str_blooms[col_idx];
                switch (pred.op) {
                    case PredicateOp::Eq: {
                        const uint32_t qhash = filter_hash(pred.str_value);
                        if (!bloom_check(be.bloom, be.n_bits, qhash))
                            return false;
                        break;
                    }
                    case PredicateOp::In: {
                        // All query values must be bloom-absent to prune.
                        // If any value's bloom is positive, we can't prune.
                        bool any_may_match = false;
                        std::vector<std::string> vals;
                        if (!pred.str_value.empty()) vals.push_back(pred.str_value);
                        for (const auto& s : pred.values) vals.push_back(s);
                        for (const auto& s : vals) {
                            const uint32_t qhash = filter_hash(s);
                            if (bloom_check(be.bloom, be.n_bits, qhash)) {
                                any_may_match = true;
                                break;
                            }
                        }
                        if (!any_may_match && !vals.empty()) return false;
                        break;
                    }
                    case PredicateOp::Prefix:
                        // Can't prune prefix at bloom level (bloom is for exact values).
                        break;
                    default: break;  // NotEq, NotIn: conservative, can't prune
                }
                break;
            }

            case ColumnType::Set: {
                const auto& be = set_blooms[col_idx];
                switch (pred.op) {
                    case PredicateOp::Contains: {
                        const uint32_t qhash = filter_hash(pred.str_value);
                        if (!bloom_check(be.bloom, be.n_bits, qhash))
                            return false;
                        break;
                    }
                    case PredicateOp::ContainsAny: {
                        std::vector<std::string> vals;
                        if (!pred.str_value.empty()) vals.push_back(pred.str_value);
                        for (const auto& s : pred.values) vals.push_back(s);
                        bool any_may_match = false;
                        for (const auto& s : vals) {
                            if (bloom_check(be.bloom, be.n_bits, filter_hash(s))) {
                                any_may_match = true;
                                break;
                            }
                        }
                        if (!any_may_match && !vals.empty()) return false;
                        break;
                    }
                    case PredicateOp::ContainsAll: {
                        // ALL values must be bloom-positive for a match.
                        // If ANY is bloom-negative, prune.
                        std::vector<std::string> vals;
                        if (!pred.str_value.empty()) vals.push_back(pred.str_value);
                        for (const auto& s : pred.values) vals.push_back(s);
                        for (const auto& s : vals) {
                            if (!bloom_check(be.bloom, be.n_bits, filter_hash(s)))
                                return false;
                        }
                        break;
                    }
                    default: break;
                }
                break;
            }

            case ColumnType::Bool:
                // Bool has no summary entry (only 2 values). Can't prune.
                break;
        }
    }

    return true;  // conservative: leaf may contain a match
}

}  // namespace sextant::tree
