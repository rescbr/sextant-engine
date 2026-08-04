#include "cardinality.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sextant::tree {

void CardinalityTable::init(const Schema& schema, uint64_t n_vectors) {
    n_vectors_ = n_vectors;
    col_to_idx_.assign(schema.columns.size(), UINT32_MAX);
    columns_.clear();

    for (uint32_t c = 0; c < schema.columns.size(); ++c) {
        const auto type = schema.columns[c].type;
        if (type == ColumnType::String || type == ColumnType::Set ||
            type == ColumnType::Int32 || type == ColumnType::Int64 ||
            type == ColumnType::Float) {
            col_to_idx_[c] = static_cast<uint32_t>(columns_.size());
            auto& col = columns_.emplace_back();
            col.is_numeric = (type == ColumnType::Int32 || type == ColumnType::Int64 ||
                              type == ColumnType::Float);
        }
    }
}

void CardinalityTable::add_string(uint32_t col_idx, std::string_view val) {
    if (col_idx >= col_to_idx_.size()) return;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return;
    const uint32_t h = filter_hash(val);
    ++columns_[ci].freq[h];
}

void CardinalityTable::add_set(uint32_t col_idx,
                                const uint8_t* counts, const uint32_t* offsets,
                                const uint16_t* elem_lengths, const char* elem_data,
                                uint32_t row_idx) {
    if (col_idx >= col_to_idx_.size()) return;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return;

    const uint8_t ec = counts[row_idx];
    const uint32_t off = offsets[row_idx];

    uint32_t byte_off = 0;
    for (uint32_t k = 0; k < off; ++k)
        byte_off += elem_lengths[k];

    for (uint8_t e = 0; e < ec; ++e) {
        const uint16_t elen = elem_lengths[off + e];
        std::string_view sv(elem_data + byte_off, elen);
        const uint32_t h = filter_hash(sv);
        ++columns_[ci].freq[h];
        byte_off += elen;
    }
}

void CardinalityTable::add_numeric(uint32_t col_idx, double val) {
    if (col_idx >= col_to_idx_.size()) return;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return;
    ++columns_[ci].numeric_hist[val];
}

float CardinalityTable::selectivity_string(uint32_t col_idx,
                                             std::string_view val) const {
    if (col_idx >= col_to_idx_.size() || n_vectors_ == 0) return 0.0f;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return 0.0f;
    const uint32_t h = filter_hash(val);
    auto it = columns_[ci].freq.find(h);
    if (it == columns_[ci].freq.end()) return 0.0f;
    return static_cast<float>(it->second) / static_cast<float>(n_vectors_);
}

float CardinalityTable::selectivity_set(uint32_t col_idx,
                                          std::string_view val) const {
    return selectivity_string(col_idx, val);
}

float CardinalityTable::selectivity_numeric(uint32_t col_idx,
                                              const Predicate& pred) const {
    if (col_idx >= col_to_idx_.size() || n_vectors_ == 0) return 1.0f;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return 1.0f;

    const auto& hist = columns_[ci].numeric_hist;
    if (hist.empty()) return 1.0f;  // no data → conservative

    // Compute the count of values matching the predicate.
    uint64_t matching = 0;
    switch (pred.op) {
        case PredicateOp::Eq: {
            auto it = hist.find(pred.value);
            if (it != hist.end()) matching = it->second;
            break;
        }
        case PredicateOp::Gt: {
            for (auto it = hist.upper_bound(pred.value); it != hist.end(); ++it)
                matching += it->second;
            break;
        }
        case PredicateOp::Ge: {
            for (auto it = hist.lower_bound(pred.value); it != hist.end(); ++it)
                matching += it->second;
            break;
        }
        case PredicateOp::Lt: {
            for (auto it = hist.begin(); it != hist.end() && it->first < pred.value; ++it)
                matching += it->second;
            break;
        }
        case PredicateOp::Le: {
            for (auto it = hist.begin(); it != hist.end() && it->first <= pred.value; ++it)
                matching += it->second;
            break;
        }
        case PredicateOp::Between: {
            for (auto it = hist.lower_bound(pred.value);
                 it != hist.end() && it->first <= pred.value2; ++it)
                matching += it->second;
            break;
        }
        default:
            return 1.0f;  // NotEq, In, NotIn: conservative
    }

    return static_cast<float>(matching) / static_cast<float>(n_vectors_);
}

float CardinalityTable::selectivity_geo(uint32_t lat_col, uint32_t lng_col,
                                          const Predicate& pred) const {
    // Determine the query bounding box [lat_lo, lat_hi] × [lng_lo, lng_hi].
    double lat_lo, lat_hi, lng_lo, lng_hi;
    constexpr double kKmPerDegLat = 111.0;
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    if (pred.op == PredicateOp::GeoBox) {
        lat_lo = pred.value;   lat_hi = pred.value3;
        lng_lo = pred.value2;  lng_hi = pred.value4;
    } else {
        // GeoRadius: bounding box of the circle.
        const double dlat = pred.radius_km / kKmPerDegLat;
        const double cos_lat = std::cos(pred.value * kDegToRad);
        const double dlng = (std::fabs(cos_lat) < 1e-6)
            ? 180.0
            : pred.radius_km / (kKmPerDegLat * cos_lat);
        lat_lo = pred.value  - dlat;  lat_hi = pred.value  + dlat;
        lng_lo = pred.value2 - dlng;  lng_hi = pred.value2 + dlng;
    }

    // Helper: fraction of [lo, hi] overlapping the observed [col_min, col_max].
    auto overlap_frac = [&](uint32_t col_idx, double lo, double hi) -> float {
        if (col_idx >= col_to_idx_.size()) return 1.0f;
        const uint32_t ci = col_to_idx_[col_idx];
        if (ci == UINT32_MAX) return 1.0f;  // no data → conservative
        const auto& hist = columns_[ci].numeric_hist;
        if (hist.empty()) return 1.0f;
        const double col_min = hist.begin()->first;
        const double col_max = hist.rbegin()->first;
        const double span = col_max - col_min;
        if (span <= 0.0) {
            // Degenerate column (single value): match iff that value is in range.
            return (col_min >= lo && col_min <= hi) ? 1.0f : 0.0f;
        }
        const double lo_c = std::max(lo, col_min);
        const double hi_c = std::min(hi, col_max);
        if (hi_c < lo_c) return 0.0f;
        return static_cast<float>((hi_c - lo_c) / span);
    };

    const float f_lat = overlap_frac(lat_col, lat_lo, lat_hi);
    const float f_lng = overlap_frac(lng_col, lng_lo, lng_hi);
    return std::min(f_lat * f_lng, 1.0f);
}

float CardinalityTable::selectivity_combined(
    const Schema& schema,
    const std::vector<struct Predicate>& preds,
    const std::vector<uint32_t>& pred_col_indices,
    const std::vector<uint32_t>& geo_lng_col_indices) const {
    if (preds.empty()) return 1.0f;
    float sel = 1.0f;
    for (uint32_t p = 0; p < preds.size(); ++p) {
        const auto& pred = preds[p];
        const uint32_t col_idx = pred_col_indices[p];
        const auto& col = schema.columns[col_idx];

        float s = 1.0f;
        if (pred.op == PredicateOp::GeoBox || pred.op == PredicateOp::GeoRadius) {
            const uint32_t lng_col = (p < geo_lng_col_indices.size())
                ? geo_lng_col_indices[p] : UINT32_MAX;
            s = selectivity_geo(col_idx, lng_col, pred);
        } else if (col.type == ColumnType::Int32 || col.type == ColumnType::Int64 ||
            col.type == ColumnType::Float) {
            s = selectivity_numeric(col_idx, pred);
        } else if (col.type == ColumnType::String) {
            if (pred.op == PredicateOp::Eq)
                s = selectivity_string(col_idx, pred.str_value);
            else if (pred.op == PredicateOp::In) {
                float sum = 0;
                if (!pred.str_value.empty())
                    sum += selectivity_string(col_idx, pred.str_value);
                for (const auto& v : pred.values)
                    sum += selectivity_string(col_idx, v);
                s = std::min(sum, 1.0f);
            }
        } else if (col.type == ColumnType::Set &&
                   pred.op == PredicateOp::Contains) {
            s = selectivity_set(col_idx, pred.str_value);
        }
        sel *= std::max(s, 0.0001f);
    }
    return std::min(sel, 1.0f);
}

std::vector<uint8_t> CardinalityTable::serialize() const {
    std::vector<uint8_t> buf;

    auto write_u32 = [&](uint32_t v) {
        buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
                   reinterpret_cast<uint8_t*>(&v) + 4);
    };
    auto write_u8 = [&](uint8_t v) {
        buf.push_back(v);
    };
    auto write_f64 = [&](double v) {
        buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
                   reinterpret_cast<uint8_t*>(&v) + 8);
    };

    write_u32(0);  // placeholder for n_vectors low bits
    // Write n_vectors as u64.
    buf.resize(buf.size() - 4);  // undo the u32
    buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&n_vectors_),
               reinterpret_cast<const uint8_t*>(&n_vectors_) + 8);

    // Count columns with data.
    uint32_t n_cols = 0;
    for (const auto& col : columns_)
        if (!col.freq.empty() || !col.numeric_hist.empty()) ++n_cols;

    write_u32(n_cols);
    for (uint32_t c = 0; c < col_to_idx_.size(); ++c) {
        const uint32_t ci = col_to_idx_[c];
        if (ci == UINT32_MAX) continue;
        const auto& col = columns_[ci];
        if (col.freq.empty() && col.numeric_hist.empty()) continue;
        write_u32(c);
        write_u8(col.is_numeric ? 1 : 0);
        if (col.is_numeric) {
            write_u32(static_cast<uint32_t>(col.numeric_hist.size()));
            for (const auto& [val, cnt] : col.numeric_hist) {
                write_f64(val);
                write_u32(cnt);
            }
        } else {
            write_u32(static_cast<uint32_t>(col.freq.size()));
            for (const auto& [hash, cnt] : col.freq) {
                write_u32(hash);
                write_u32(cnt);
            }
        }
    }

    return buf;
}

void CardinalityTable::deserialize(const uint8_t* data, size_t len) {
    columns_.clear();
    col_to_idx_.clear();
    n_vectors_ = 0;

    if (len < 12) return;
    size_t off = 0;

    auto read_u32 = [&]() -> uint32_t {
        uint32_t v = 0;
        std::memcpy(&v, data + off, 4);
        off += 4;
        return v;
    };
    auto read_u8 = [&]() -> uint8_t {
        uint8_t v = data[off++];
        return v;
    };
    auto read_f64 = [&]() -> double {
        double v = 0;
        std::memcpy(&v, data + off, 8);
        off += 8;
        return v;
    };

    std::memcpy(&n_vectors_, data, 8);
    off = 8;

    const uint32_t n_cols = read_u32();

    // Determine max col_id to size col_to_idx_.
    uint32_t max_col = 0;
    {
        size_t peek = off;
        for (uint32_t i = 0; i < n_cols; ++i) {
            uint32_t cid = 0;
            std::memcpy(&cid, data + peek, 4); peek += 4;
            uint8_t is_num = data[peek]; peek += 1;
            uint32_t n_entries = 0;
            std::memcpy(&n_entries, data + peek, 4); peek += 4;
            peek += static_cast<size_t>(n_entries) * (is_num ? 12 : 8);
            max_col = std::max(max_col, cid);
        }
    }

    col_to_idx_.assign(max_col + 1, UINT32_MAX);
    for (uint32_t i = 0; i < n_cols; ++i) {
        const uint32_t col_id = read_u32();
        const uint8_t is_numeric = read_u8();
        const uint32_t n_entries = read_u32();
        const uint32_t ci = static_cast<uint32_t>(columns_.size());
        columns_.emplace_back();
        col_to_idx_[col_id] = ci;
        auto& col = columns_[ci];
        col.is_numeric = is_numeric;
        if (is_numeric) {
            col.numeric_hist.clear();
            for (uint32_t j = 0; j < n_entries; ++j) {
                const double val = read_f64();
                const uint32_t cnt = read_u32();
                col.numeric_hist[val] = cnt;
            }
        } else {
            col.freq.reserve(n_entries);
            for (uint32_t j = 0; j < n_entries; ++j) {
                const uint32_t hash = read_u32();
                const uint32_t cnt = read_u32();
                col.freq[hash] = cnt;
            }
        }
    }
}

}  // namespace sextant::tree
