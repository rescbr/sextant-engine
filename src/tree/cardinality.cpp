#include "cardinality.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sextant::tree {

// ===========================================================================
// HyperLogLog distinct estimator (2^12 registers, 6 bits each, ~3 KB/column).
// Self-contained: standard beta/bias-corrected estimator (Flajolet et al.
// 2007) with linear counting for the sparse range.
// ===========================================================================

namespace {

constexpr size_t kHllBytes = (ColumnCardinality::kHllRegs * 6 + 7) / 8;  // 3072

inline uint8_t hll_get(const std::vector<uint8_t>& p, uint32_t i) {
    const uint64_t bit = static_cast<uint64_t>(i) * 6;
    const size_t byte = bit >> 3;
    const uint32_t shift = bit & 7;
    const uint32_t v = static_cast<uint32_t>(p[byte]) |
                       (static_cast<uint32_t>(p[byte + 1]) << 8);
    return static_cast<uint8_t>((v >> shift) & 63);
}

inline void hll_set(std::vector<uint8_t>& p, uint32_t i, uint8_t r) {
    const uint64_t bit = static_cast<uint64_t>(i) * 6;
    const size_t byte = bit >> 3;
    const uint32_t shift = bit & 7;
    const uint32_t mask = 63u << shift;
    uint32_t v = static_cast<uint32_t>(p[byte]) |
                 (static_cast<uint32_t>(p[byte + 1]) << 8);
    v = (v & ~mask) | (static_cast<uint32_t>(r) << shift);
    p[byte] = static_cast<uint8_t>(v);
    p[byte + 1] = static_cast<uint8_t>(v >> 8);
}

inline uint32_t sat_inc(uint32_t v) { return v == UINT32_MAX ? v : v + 1; }

}  // namespace

uint32_t CardinalityTable::hll_finalize(uint32_t h) {
    // splitmix32-style finalizer (MurmurHash3 fmix32): filter_hash already
    // avalanches reasonably, but the HLL index takes the TOP 12 bits and the
    // rank the LOW 20, so both halves need good mixing.
    h ^= h >> 16;
    h *= 0x7feb352dU;
    h ^= h >> 15;
    h *= 0x846ca68bU;
    h ^= h >> 16;
    return h;
}

void CardinalityTable::hll_add(std::vector<uint8_t>& hll, uint32_t finalized_hash) {
    if (hll.empty()) hll.assign(kHllBytes, 0);
    const uint32_t idx = finalized_hash >> ColumnCardinality::kHllPadding;
    const uint32_t rest = finalized_hash & ((1u << ColumnCardinality::kHllPadding) - 1);
    // rank = 1 + leading zeros of the (32-kHllBits) remaining bits.
    const uint8_t rank = rest == 0
        ? static_cast<uint8_t>(ColumnCardinality::kHllPadding + 1)
        : static_cast<uint8_t>(32 - static_cast<uint32_t>(__builtin_clz(rest)) -
                               ColumnCardinality::kHllBits + 1);
    if (rank > hll_get(hll, idx)) hll_set(hll, idx, rank);
}

double CardinalityTable::hll_estimate(const std::vector<uint8_t>& hll) {
    if (hll.empty()) return 0.0;
    constexpr uint32_t m = ColumnCardinality::kHllRegs;
    double sum = 0.0;
    uint32_t zeros = 0;
    for (uint32_t i = 0; i < m; ++i) {
        const uint8_t r = hll_get(hll, i);
        sum += std::ldexp(1.0, -r);
        if (r == 0) ++zeros;
    }
    const double alpha = m == 16 ? 0.673 : m == 32 ? 0.697 : m == 64 ? 0.709
        : 0.7213 / (1.0 + 1.079 / m);
    double e = alpha * static_cast<double>(m) * static_cast<double>(m) / sum;
    if (e <= 2.5 * m && zeros > 0)
        e = static_cast<double>(m) * std::log(static_cast<double>(m) / zeros);
    return std::max(e, 1.0);
}

double ColumnCardinality::distinct_estimate() const {
    if (!hll.empty()) return CardinalityTable::hll_estimate(hll);
    if (est_distinct_stored > 0) return static_cast<double>(est_distinct_stored);
    return static_cast<double>(freq.size());
}

// ===========================================================================
// Table
// ===========================================================================

void CardinalityTable::init(const Schema& schema, uint64_t n_vectors, uint32_t cap) {
    n_vectors_ = n_vectors;
    cap_ = cap;
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

void CardinalityTable::freq_add_(ColumnCardinality& col, uint32_t h) {
    auto it = col.freq.find(h);
    if (it != col.freq.end()) {
        it->second = sat_inc(it->second);
        return;
    }
    if (cap_ == 0 || col.freq.size() < cap_) {
        col.freq.emplace(h, 1u);
    } else {
        // Cap reached: existing heavy hitters keep exact counts; new values
        // are only reflected in the distinct estimate (HLL, fed by caller).
        col.overflowed = true;
    }
}

void CardinalityTable::add_string(uint32_t col_idx, std::string_view val) {
    if (col_idx >= col_to_idx_.size()) return;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return;
    auto& col = columns_[ci];
    const uint32_t h = filter_hash(val);
    hll_add(col.hll, hll_finalize(h));  // always: distinct estimate is uncapped
    freq_add_(col, h);
}

void CardinalityTable::add_set(uint32_t col_idx,
                                const uint8_t* counts, const uint32_t* offsets,
                                const uint16_t* elem_lengths, const char* elem_data,
                                uint32_t row_idx) {
    if (col_idx >= col_to_idx_.size()) return;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return;
    auto& col = columns_[ci];

    const uint8_t ec = counts[row_idx];
    const uint32_t off = offsets[row_idx];

    uint32_t byte_off = 0;
    for (uint32_t k = 0; k < off; ++k)
        byte_off += elem_lengths[k];

    for (uint8_t e = 0; e < ec; ++e) {
        const uint16_t elen = elem_lengths[off + e];
        std::string_view sv(elem_data + byte_off, elen);
        const uint32_t h = filter_hash(sv);
        hll_add(col.hll, hll_finalize(h));
        freq_add_(col, h);
        byte_off += elen;
    }
}

void CardinalityTable::bin_add_(ColumnCardinality& col, double val, uint32_t cnt) {
    if (col.bins.empty()) {
        col.bins.assign(kCardinalityNumBins, 0);
        col.bin_min = col.bin_max = val;
        col.frame_lo = col.frame_hi = val;
    }
    const bool extend = val < col.frame_lo || val > col.frame_hi;
    col.bin_min = std::min(col.bin_min, val);
    col.bin_max = std::max(col.bin_max, val);
    col.bin_total += cnt;
    if (extend) {
        // Grow the frame geometrically (double the span) so a monotone
        // stream triggers only O(log) rescales, and rebin the old counts at
        // their absolute value positions in the new frame.
        double lo = std::min(val, col.frame_lo);
        double hi = std::max(val, col.frame_hi);
        const double span = col.frame_hi - col.frame_lo;
        if (span > 0.0) {
            if (val > col.frame_hi) hi = std::max(hi, col.frame_lo + 2.0 * span);
            if (val < col.frame_lo) lo = std::min(lo, col.frame_hi - 2.0 * span);
        }
        rescale_bins_(col, lo, hi);
    }
    uint32_t idx = 0;
    if (col.frame_hi > col.frame_lo) {
        double f = (val - col.frame_lo) / (col.frame_hi - col.frame_lo);
        f = std::clamp(f, 0.0, 1.0);
        idx = static_cast<uint32_t>(f * kCardinalityNumBins);
        if (idx >= kCardinalityNumBins) idx = kCardinalityNumBins - 1;
    }
    uint32_t& b = col.bins[idx];
    b = (cnt > UINT32_MAX - b) ? UINT32_MAX : b + cnt;
}

void CardinalityTable::rescale_bins_(ColumnCardinality& col, double lo, double hi) {
    if (col.bins.empty() || !(hi > lo)) {
        col.frame_lo = lo;
        col.frame_hi = hi;
        return;
    }
    const double old_span = col.frame_hi - col.frame_lo;
    std::vector<uint32_t> nb(kCardinalityNumBins, 0);
    for (uint32_t i = 0; i < kCardinalityNumBins; ++i) {
        if (col.bins[i] == 0) continue;
        const double center = col.frame_lo +
            (static_cast<double>(i) + 0.5) / kCardinalityNumBins * old_span;
        double f = (center - lo) / (hi - lo);
        f = std::clamp(f, 0.0, 1.0);
        uint32_t idx = static_cast<uint32_t>(f * kCardinalityNumBins);
        if (idx >= kCardinalityNumBins) idx = kCardinalityNumBins - 1;
        uint32_t& b = nb[idx];
        b = (col.bins[i] > UINT32_MAX - b) ? UINT32_MAX : b + col.bins[i];
    }
    col.bins = std::move(nb);
    col.frame_lo = lo;
    col.frame_hi = hi;
}

void CardinalityTable::add_numeric(uint32_t col_idx, double val) {
    if (col_idx >= col_to_idx_.size()) return;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return;
    auto& col = columns_[ci];
    if (col.binned) {
        bin_add_(col, val, 1);
        return;
    }
    auto [it, inserted] = col.numeric_hist.try_emplace(val, 1u);
    if (!inserted) {
        it->second = sat_inc(it->second);
        return;
    }
    if (cap_ != 0 && col.numeric_hist.size() > cap_)
        numeric_to_binned_(col);
}

void CardinalityTable::numeric_to_binned_(ColumnCardinality& col) {
    if (col.numeric_hist.empty()) return;
    col.bins.assign(kCardinalityNumBins, 0);
    col.bin_min = col.numeric_hist.begin()->first;
    col.bin_max = col.numeric_hist.rbegin()->first;
    col.frame_lo = col.bin_min;
    col.frame_hi = col.bin_max;
    col.bin_total = 0;
    for (const auto& [val, cnt] : col.numeric_hist)
        bin_add_(col, val, cnt);
    col.numeric_hist.clear();
    col.binned = true;
    col.overflowed = true;
}

void CardinalityTable::merge_from(const CardinalityTable& other) {
    // The two tables share the same schema (col_to_idx_ layout), so column
    // indices line up. Merge by summing per-column frequency counters and
    // numeric histograms, honoring this table's cap:
    //   - freq: sum counts for present keys; skip absent keys when the
    //     destination map is full (marking overflow).
    //   - numeric: exact×exact merges sum counts (converting to bins if the
    //     destination overflows mid-merge); anything touching a binned side
    //     lands in bins.
    //   - HLL registers are unioned (per-register max) so the merged
    //     distinct estimate covers all values either side observed.
    const uint32_t n = static_cast<uint32_t>(columns_.size());
    const uint32_t no = static_cast<uint32_t>(other.columns_.size());
    const uint32_t m = std::min(n, no);
    for (uint32_t ci = 0; ci < m; ++ci) {
        auto& dst = columns_[ci];
        const auto& src = other.columns_[ci];

        if (src.overflowed) dst.overflowed = true;

        // Union HLLs (string/set columns).
        if (!src.hll.empty()) {
            if (dst.hll.empty()) {
                dst.hll = src.hll;
            } else {
                for (uint32_t r = 0; r < ColumnCardinality::kHllRegs; ++r) {
                    // Rebuild via packed accessors: compare per register.
                    const uint8_t a = hll_get(dst.hll, r);
                    const uint8_t b = hll_get(src.hll, r);
                    if (b > a) hll_set(dst.hll, r, b);
                }
            }
        }

        for (const auto& [hash, cnt] : src.freq) {
            auto it = dst.freq.find(hash);
            if (it != dst.freq.end()) {
                it->second = (cnt > UINT32_MAX - it->second) ? UINT32_MAX
                                                             : it->second + cnt;
            } else if (cap_ == 0 || dst.freq.size() < cap_) {
                dst.freq.emplace(hash, cnt);
            } else {
                dst.overflowed = true;
            }
        }

        if (src.binned && !dst.binned) numeric_to_binned_(dst);

        if (dst.binned) {
            if (src.binned) {
                // Approximate: fold src bins into dst's [min,max] frame.
                for (uint32_t b = 0; b < kCardinalityNumBins; ++b) {
                    if (src.bins[b] == 0) continue;
                    const double f = (static_cast<double>(b) + 0.5) /
                                     kCardinalityNumBins;
                    const double v = src.bin_min + f * (src.bin_max - src.bin_min);
                    bin_add_(dst, v, src.bins[b]);
                }
            } else {
                for (const auto& [val, cnt] : src.numeric_hist)
                    bin_add_(dst, val, cnt);
            }
            continue;
        }

        for (const auto& [val, cnt] : src.numeric_hist) {
            auto [it, inserted] = dst.numeric_hist.try_emplace(val, cnt);
            if (!inserted)
                it->second = (cnt > UINT32_MAX - it->second) ? UINT32_MAX
                                                             : it->second + cnt;
        }
        if (cap_ != 0 && dst.numeric_hist.size() > cap_)
            numeric_to_binned_(dst);
    }
}

void CardinalityTable::clear_stats() {
    for (auto& col : columns_) {
        col.freq.clear();
        col.numeric_hist.clear();
        col.bins.clear();
        col.binned = false;
        col.overflowed = false;
        col.bin_total = 0;
        col.est_distinct_stored = 0;
        col.hll.clear();
    }
}

bool CardinalityTable::any_overflowed() const {
    for (const auto& col : columns_)
        if (col.overflowed) return true;
    return false;
}

size_t CardinalityTable::total_entries() const {
    size_t total = 0;
    for (const auto& col : columns_)
        total += col.freq.size() + col.numeric_hist.size();
    return total;
}

float CardinalityTable::selectivity_string(uint32_t col_idx,
                                             std::string_view val) const {
    if (col_idx >= col_to_idx_.size() || n_vectors_ == 0) return 0.0f;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return 0.0f;
    const auto& col = columns_[ci];
    const uint32_t h = filter_hash(val);
    auto it = col.freq.find(h);
    if (it == col.freq.end()) {
        // Key not among the exact (capped) entries. If the column never
        // overflowed, the value is truly absent → 0 (unchanged semantics).
        // If it overflowed, the value may exist but was rejected by the cap
        // → fall back to the uniform 1/est_distinct estimate so rare values
        // don't spuriously look absent (which would force the brute-force
        // path for every one of them).
        if (!col.overflowed) return 0.0f;
        const double d = std::max(col.distinct_estimate(), 1.0);
        return static_cast<float>(std::min(1.0, 1.0 / d));
    }
    return static_cast<float>(it->second) / static_cast<float>(n_vectors_);
}

float CardinalityTable::selectivity_set(uint32_t col_idx,
                                          std::string_view val) const {
    return selectivity_string(col_idx, val);
}

float CardinalityTable::selectivity_binned_(const ColumnCardinality& col,
                                              const Predicate& pred) const {
    const uint64_t total = col.bin_total;
    if (total == 0 || col.bins.empty()) return 1.0f;
    const double span = col.frame_hi - col.frame_lo;

    // Estimated mass (in rows) strictly below x, interpolating within the
    // edge bin. Cutoffs use the exact observed min/max; bin positions use
    // the (possibly wider) binning frame.
    auto mass_lt = [&](double x) -> double {
        if (!(x > col.bin_min)) return 0.0;
        if (x >= col.bin_max) return static_cast<double>(total);
        double pos = span > 0.0 ? (x - col.frame_lo) / span * kCardinalityNumBins
                                : 0.0;
        pos = std::min(std::max(pos, 0.0),
                       static_cast<double>(kCardinalityNumBins - 1));
        const uint32_t bi = static_cast<uint32_t>(pos);
        double mass = 0.0;
        for (uint32_t j = 0; j < bi; ++j) mass += col.bins[j];
        return mass + col.bins[bi] * (pos - bi);
    };
    auto mass_le = [&](double x) -> double {
        return x >= col.bin_max ? static_cast<double>(total)
                                : mass_lt(std::nextafter(x, HUGE_VAL));
    };

    double matching = 0.0;
    switch (pred.op) {
        case PredicateOp::Eq:
            if (pred.value >= col.bin_min && pred.value <= col.bin_max)
                matching = static_cast<double>(total) / kCardinalityNumBins;
            break;
        case PredicateOp::Gt:
            matching = static_cast<double>(total) - mass_le(pred.value);
            break;
        case PredicateOp::Ge:
            matching = static_cast<double>(total) - mass_lt(pred.value);
            break;
        case PredicateOp::Lt:
            matching = mass_lt(pred.value);
            break;
        case PredicateOp::Le:
            matching = mass_le(pred.value);
            break;
        case PredicateOp::Between:
            matching = std::max(0.0, mass_le(pred.value2) - mass_lt(pred.value));
            break;
        default:
            return 1.0f;  // NotEq, In, NotIn: conservative
    }
    return static_cast<float>(std::clamp(matching /
        static_cast<double>(n_vectors_), 0.0, 1.0));
}

float CardinalityTable::selectivity_numeric(uint32_t col_idx,
                                              const Predicate& pred) const {
    if (col_idx >= col_to_idx_.size() || n_vectors_ == 0) return 1.0f;
    const uint32_t ci = col_to_idx_[col_idx];
    if (ci == UINT32_MAX) return 1.0f;

    const auto& col = columns_[ci];
    if (col.binned) return selectivity_binned_(col, pred);

    const auto& hist = col.numeric_hist;
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
        const auto& col = columns_[ci];
        double col_min, col_max;
        if (col.binned) {
            if (col.bins.empty()) return 1.0f;
            col_min = col.bin_min;
            col_max = col.bin_max;
        } else {
            const auto& hist = col.numeric_hist;
            if (hist.empty()) return 1.0f;
            col_min = hist.begin()->first;
            col_max = hist.rbegin()->first;
        }
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

// ===========================================================================
// Serialization
// ===========================================================================

namespace {
// Versioned-blob magic. Bit 31 set: a legacy blob's first 4 bytes are the
// low half of n_vectors (u64), which cannot have bit 31 set for realistic
// row counts, so the two formats are unambiguously distinguishable.
constexpr uint32_t kCardMagic = 0xC4DA0002u;
constexpr uint8_t kCardVersion = 2;
}  // namespace

std::vector<uint8_t> CardinalityTable::serialize() const {
    std::vector<uint8_t> buf;

    auto write_u32 = [&](uint32_t v) {
        buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
                   reinterpret_cast<uint8_t*>(&v) + 4);
    };
    auto write_u8 = [&](uint8_t v) {
        buf.push_back(v);
    };
    auto write_u64 = [&](uint64_t v) {
        buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
                   reinterpret_cast<uint8_t*>(&v) + 8);
    };
    auto write_f64 = [&](double v) {
        buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
                   reinterpret_cast<uint8_t*>(&v) + 8);
    };

    write_u32(kCardMagic);
    write_u8(kCardVersion);
    write_u64(n_vectors_);

    // Count columns with data.
    uint32_t n_cols = 0;
    for (const auto& col : columns_)
        if (!col.freq.empty() || !col.numeric_hist.empty() || col.binned) ++n_cols;

    write_u32(n_cols);
    for (uint32_t c = 0; c < col_to_idx_.size(); ++c) {
        const uint32_t ci = col_to_idx_[c];
        if (ci == UINT32_MAX) continue;
        const auto& col = columns_[ci];
        if (col.freq.empty() && col.numeric_hist.empty() && !col.binned) continue;
        write_u32(c);
        write_u8((col.is_numeric ? 1 : 0) | (col.binned ? 2 : 0) |
                 (col.overflowed ? 4 : 0));
        if (col.is_numeric) {
            if (col.binned) {
                write_f64(col.bin_min);
                write_f64(col.bin_max);
                write_f64(col.frame_lo);
                write_f64(col.frame_hi);
                write_u64(col.bin_total);
                for (uint32_t b = 0; b < kCardinalityNumBins; ++b)
                    write_u32(col.bins.empty() ? 0 : col.bins[b]);
            } else {
                write_u32(static_cast<uint32_t>(col.numeric_hist.size()));
                for (const auto& [val, cnt] : col.numeric_hist) {
                    write_f64(val);
                    write_u32(cnt);
                }
            }
        } else {
            write_u32(static_cast<uint32_t>(col.freq.size()));
            write_u64(static_cast<uint64_t>(col.distinct_estimate()));
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

    uint32_t first = 0;
    std::memcpy(&first, data, 4);
    const bool versioned = (first == kCardMagic);

    size_t off = 0;
    if (versioned) {
        off = 4;  // magic
        ++off;    // version (validated by the magic; only one version exists)
    }

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
    auto read_u64 = [&]() -> uint64_t {
        uint64_t v = 0;
        std::memcpy(&v, data + off, 8);
        off += 8;
        return v;
    };
    auto read_f64 = [&]() -> double {
        double v = 0;
        std::memcpy(&v, data + off, 8);
        off += 8;
        return v;
    };

    std::memcpy(&n_vectors_, data + off, 8);
    off += 8;
    if (off > len) { n_vectors_ = 0; return; }

    const uint32_t n_cols = read_u32();

    // Determine max col_id to size col_to_idx_.
    uint32_t max_col = 0;
    {
        size_t peek = off;
        for (uint32_t i = 0; i < n_cols; ++i) {
            uint32_t cid = 0;
            if (peek + 4 > len) return;
            std::memcpy(&cid, data + peek, 4); peek += 4;
            uint8_t flags = data[peek]; peek += 1;
            const bool is_num = flags & 1;
            const bool binned = versioned && (flags & 2);
            if (binned) {
                peek += 8 + 8 + 8 + 8 + 8;  // min/max, frame, bin_total
                peek += static_cast<size_t>(kCardinalityNumBins) * 4;
            } else {
                uint32_t n_entries = 0;
                if (peek + 4 > len) return;
                std::memcpy(&n_entries, data + peek, 4); peek += 4;
                if (versioned && !is_num) peek += 8;  // est_distinct
                peek += static_cast<size_t>(n_entries) * (is_num ? 12 : 8);
            }
            max_col = std::max(max_col, cid);
        }
    }

    col_to_idx_.assign(max_col + 1, UINT32_MAX);
    for (uint32_t i = 0; i < n_cols; ++i) {
        const uint32_t col_id = read_u32();
        const uint8_t flags = read_u8();
        const bool is_numeric = flags & 1;
        const bool binned = versioned && (flags & 2);
        const bool overflowed = versioned && (flags & 4);
        const uint32_t ci = static_cast<uint32_t>(columns_.size());
        columns_.emplace_back();
        col_to_idx_[col_id] = ci;
        auto& col = columns_[ci];
        col.is_numeric = is_numeric;
        col.overflowed = overflowed;
        if (binned) {
            col.binned = true;
            col.bin_min = read_f64();
            col.bin_max = read_f64();
            col.frame_lo = read_f64();
            col.frame_hi = read_f64();
            col.bin_total = read_u64();
            col.bins.resize(kCardinalityNumBins);
            for (uint32_t b = 0; b < kCardinalityNumBins; ++b)
                col.bins[b] = read_u32();
        } else if (is_numeric) {
            const uint32_t n_entries = read_u32();
            for (uint32_t j = 0; j < n_entries; ++j) {
                const double val = read_f64();
                const uint32_t cnt = read_u32();
                col.numeric_hist[val] = cnt;
            }
        } else {
            const uint32_t n_entries = read_u32();
            if (versioned) col.est_distinct_stored = read_u64();
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
