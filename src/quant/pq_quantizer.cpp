// PQ quantizer — full implementation.
// Ported from duckdb-vector-index/src/quant/pq/pq_quantizer.cpp with
// simsimd → NumKong adaptation. DuckDB dependencies stripped.

#include "pq_quantizer.hpp"
#include "sextant/error.hpp"

#include <numkong/numkong.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace sextant {

namespace {

/// L2 squared distance via NumKong.
inline float l2sq_f32(const float* a, const float* b, uint32_t dim) {
    nk_f64_t result = 0;
    nk_sqeuclidean_f32(reinterpret_cast<const nk_f32_t*>(a),
                       reinterpret_cast<const nk_f32_t*>(b),
                       static_cast<nk_size_t>(dim), &result);
    return static_cast<float>(result);
}

/// Dot product via NumKong.
inline float dot_f32(const float* a, const float* b, uint32_t dim) {
    nk_f64_t result = 0;
    nk_dot_f32(reinterpret_cast<const nk_f32_t*>(a),
               reinterpret_cast<const nk_f32_t*>(b),
               static_cast<nk_size_t>(dim), &result);
    return static_cast<float>(result);
}

/// Extract centroid id for slot `s` from a packed code.
inline uint32_t read_code(const uint8_t* code, uint8_t bits, uint32_t s) {
    if (bits == 8) {
        return static_cast<uint32_t>(code[s]);
    }
    // bits == 4 — two codes per byte, slot 0 = low nibble, slot 1 = high.
    const uint32_t byte_off = s / 2;
    const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
    return (static_cast<uint32_t>(code[byte_off]) >> shift) & 0x0Fu;
}

/// Write centroid id for slot `s` into a packed code.
inline void write_code(uint8_t* code, uint8_t bits, uint32_t s, uint32_t cid) {
    if (bits == 8) {
        code[s] = static_cast<uint8_t>(cid & 0xFFu);
        return;
    }
    const uint32_t byte_off = s / 2;
    const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
    const uint8_t nibble = static_cast<uint8_t>(cid & 0x0Fu);
    code[byte_off] = static_cast<uint8_t>(
        (code[byte_off] & ~(0x0Fu << shift)) | (nibble << shift));
}

/// Assign a float sub-vector to its nearest centroid in a codebook slot.
/// Always uses L2 for centroid assignment (better PQ approximation).
inline uint32_t nearest_centroid(const float* codebook, uint32_t s,
                                 const float* sub, uint32_t K,
                                 uint32_t sub_dim) {
    const float* slot_book = codebook + size_t(s) * K * sub_dim;
    uint32_t best = 0;
    float best_d = std::numeric_limits<float>::infinity();
    for (uint32_t c = 0; c < K; c++) {
        const float d = l2sq_f32(sub, slot_book + c * sub_dim, sub_dim);
        if (d < best_d) {
            best_d = d;
            best = c;
        }
    }
    return best;
}

/// k-means++ seeding: pick the first centroid uniformly at random, then each
/// subsequent one with probability proportional to D(x)^2 (distance to the
/// nearest already-chosen centroid).
void kmeans_pp_seed(const float* data, uint64_t n, uint32_t dim, uint32_t k,
                    std::mt19937_64& rng, float* centroids_out) {
    std::vector<float> min_d2(n, std::numeric_limits<float>::infinity());

    // First centroid: uniform.
    std::uniform_int_distribution<uint64_t> first(0, n - 1);
    const uint64_t i0 = first(rng);
    std::memcpy(centroids_out, data + i0 * dim, dim * sizeof(float));

    for (uint32_t c = 1; c < k; c++) {
        const float* prev = centroids_out + (c - 1) * dim;
        double sum = 0.0;
        for (uint64_t i = 0; i < n; i++) {
            const float d = l2sq_f32(data + i * dim, prev, dim);
            if (d < min_d2[i]) {
                min_d2[i] = d;
            }
            sum += min_d2[i];
        }

        if (sum <= 0.0) {
            // All remaining points coincide with an existing centroid —
            // fall back to a random sample to fill up to k.
            std::uniform_int_distribution<uint64_t> u(0, n - 1);
            std::memcpy(centroids_out + c * dim, data + u(rng) * dim,
                        dim * sizeof(float));
            continue;
        }

        std::uniform_real_distribution<double> pick(0.0, sum);
        double r = pick(rng);
        uint64_t chosen = n - 1;
        for (uint64_t i = 0; i < n; i++) {
            r -= min_d2[i];
            if (r <= 0.0) {
                chosen = i;
                break;
            }
        }
        std::memcpy(centroids_out + c * dim, data + chosen * dim,
                    dim * sizeof(float));
    }
}

/// k-means++ initialization + Lloyd iterations. Standard Lloyd's algorithm
/// with early stop at <1% reassignment. Output written to `centroids_out`
/// (k x dim floats, preallocated).
void kmeans_pp(const float* data, uint64_t n, uint32_t dim, uint32_t k,
               uint64_t seed, uint32_t max_iters, float* centroids_out) {
    if (k == 0 || dim == 0) {
        throw Error(ErrorCode::InvalidParam, "k-means: dim and k must be positive");
    }
    if (n == 0) {
        std::memset(centroids_out, 0, size_t(k) * dim * sizeof(float));
        return;
    }

    std::mt19937_64 rng(seed);

    if (n < k) {
        // Can't make more unique centroids than points — repeat samples.
        for (uint32_t c = 0; c < k; c++) {
            std::memcpy(centroids_out + c * dim,
                        data + (c % n) * dim, dim * sizeof(float));
        }
    } else {
        kmeans_pp_seed(data, n, dim, k, rng, centroids_out);
    }

    std::vector<uint32_t> assign(n, 0);
    std::vector<double> sum_vec(size_t(k) * dim, 0.0);
    std::vector<uint64_t> count(k, 0);

    for (uint32_t iter = 0; iter < max_iters; iter++) {
        // 1) Assign every sample to its nearest centroid.
        uint64_t changed = 0;
        for (uint64_t i = 0; i < n; i++) {
            float best = std::numeric_limits<float>::infinity();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k; c++) {
                const float d = l2sq_f32(data + i * dim,
                                         centroids_out + c * dim, dim);
                if (d < best) {
                    best = d;
                    best_c = c;
                }
            }
            if (assign[i] != best_c) {
                changed++;
                assign[i] = best_c;
            }
        }

        // 2) Recompute centroids as the mean of their assigned points.
        std::fill(sum_vec.begin(), sum_vec.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);
        for (uint64_t i = 0; i < n; i++) {
            const uint32_t c = assign[i];
            count[c]++;
            const float* v = data + i * dim;
            double* acc = sum_vec.data() + size_t(c) * dim;
            for (uint32_t d = 0; d < dim; d++) {
                acc[d] += v[d];
            }
        }
        for (uint32_t c = 0; c < k; c++) {
            if (count[c] > 0) {
                double inv = 1.0 / double(count[c]);
                float* cen = centroids_out + c * dim;
                double* acc = sum_vec.data() + size_t(c) * dim;
                for (uint32_t d = 0; d < dim; d++) {
                    cen[d] = float(acc[d] * inv);
                }
            } else {
                // Empty cluster: reseed from the point farthest from its
                // assigned centroid.
                float worst = -1.0f;
                uint64_t worst_i = 0;
                for (uint64_t i = 0; i < n; i++) {
                    const float d = l2sq_f32(
                        data + i * dim,
                        centroids_out + assign[i] * dim, dim);
                    if (d > worst) {
                        worst = d;
                        worst_i = i;
                    }
                }
                std::memcpy(centroids_out + c * dim,
                            data + worst_i * dim, dim * sizeof(float));
            }
        }

        // Early stop: <1% of points changed assignment.
        if (iter > 0 && changed * 100 < n) {
            break;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PqQuantizer::PqQuantizer(MetricKind metric, Dim dim, uint8_t m, uint8_t bits,
                         uint64_t seed)
    : metric_(metric), dim_(dim), m_(m), bits_(bits), seed_(seed) {
    if (m_ == 0) {
        throw Error(ErrorCode::InvalidParam, "PQ 'm' must be > 0");
    }
    if (dim_ % static_cast<Dim>(m_) != 0) {
        throw Error(ErrorCode::InvalidParam, "PQ requires dim divisible by m");
    }
    if (bits_ != 4 && bits_ != 8) {
        throw Error(ErrorCode::InvalidParam, "PQ 'bits' must be 4 or 8");
    }

    K_ = 1u << bits_;
    sub_dim_ = dim_ / m_;
    codebook_.resize(static_cast<size_t>(m_) * K_ * sub_dim_, 0.0f);
}

uint32_t PqQuantizer::code_size() const {
    return (static_cast<uint32_t>(m_) * bits_ + 7) / 8;
}

uint32_t PqQuantizer::lut_size() const {
    return static_cast<uint32_t>(m_) * K_;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

void PqQuantizer::train(const float* samples, uint64_t n) {
    if (n == 0) {
        throw Error(ErrorCode::InvalidParam, "PQ train: no samples provided");
    }
    codebook_.assign(static_cast<size_t>(m_) * K_ * sub_dim_, 0.0f);

    // Each segment's k-means++ is fully independent (disjoint codebook
    // region, disjoint sub_buffer). We train each segment sequentially.
    for (uint32_t s = 0; s < m_; s++) {
        // Extract the sub-vectors for this segment: contiguous memcpy of
        // sub_dim_ floats per input row.
        std::vector<float> sub_buffer(size_t(n) * sub_dim_);
        for (uint64_t i = 0; i < n; i++) {
            std::memcpy(sub_buffer.data() + i * sub_dim_,
                        samples + i * dim_ + s * sub_dim_,
                        sub_dim_ * sizeof(float));
        }
        const uint64_t slot_seed =
            seed_ ^ (0x9E3779B97F4A7C15ULL * (uint64_t(s) + 1));
        kmeans_pp(sub_buffer.data(), n, sub_dim_, K_, slot_seed,
                  /*max_iters=*/25,
                  codebook_.data() + size_t(s) * K_ * sub_dim_);
    }

    build_cross_distance_table();
}

void PqQuantizer::build_cross_distance_table() {
    cross_distance_table_.assign(static_cast<size_t>(m_) * K_ * K_, 0.0f);
    // Always use L2SQ for the cross-distance table, regardless of the
    // declared search metric. This mirrors the reference DiskANN approach
    // (Neyshabur-Srebro transform -> build with L2). For unit-normalized
    // data, L2 and IP rankings are identical, and PQ-L2 has better
    // approximation properties than PQ-IP.
    for (uint32_t s = 0; s < m_; s++) {
        const float* book = codebook_.data() + size_t(s) * K_ * sub_dim_;
        float* table = cross_distance_table_.data() + size_t(s) * K_ * K_;
        for (uint32_t a = 0; a < K_; a++) {
            const float* ca = book + a * sub_dim_;
            table[a * K_ + a] = 0.0f;  // L2 diagonal is always 0
            for (uint32_t b = a + 1; b < K_; b++) {
                const float* cb = book + b * sub_dim_;
                const float d = l2sq_f32(ca, cb, sub_dim_);
                table[a * K_ + b] = d;
                table[b * K_ + a] = d;  // symmetric
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

void PqQuantizer::encode(const float* vec, uint8_t* code_out) const {
    std::memset(code_out, 0, code_size());
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t cid =
            nearest_centroid(codebook_.data(), s, vec + s * sub_dim_,
                             K_, sub_dim_);
        write_code(code_out, bits_, s, cid);
    }
}

// ---------------------------------------------------------------------------
// Query preprocessing (ADC LUT)
// ---------------------------------------------------------------------------

void PqQuantizer::preprocess_query(const float* query, float* out) const {
    // ADC table: out[s * K + c] = d(query_sub_s, centroid[s][c]).
    //   L2SQ: L2 squared distance.
    //   IP:   -dot(query_sub, centroid).
    for (uint32_t s = 0; s < m_; s++) {
        const float* q_sub = query + s * sub_dim_;
        const float* slot_book =
            codebook_.data() + size_t(s) * K_ * sub_dim_;
        float* row = out + s * K_;
        for (uint32_t c = 0; c < K_; c++) {
            const float* cen = slot_book + c * sub_dim_;
            switch (metric_) {
            case MetricKind::L2Sq:
                row[c] = l2sq_f32(q_sub, cen, sub_dim_);
                break;
            case MetricKind::InnerProduct:
                row[c] = -dot_f32(q_sub, cen, sub_dim_);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LUT distance
// ---------------------------------------------------------------------------

float PqQuantizer::lut_distance(const uint8_t* code, const float* lut) const {
    float acc = 0.0f;
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t cid = read_code(code, bits_, s);
        acc += lut[s * K_ + cid];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// SDC code LUT (build mode)
// ---------------------------------------------------------------------------

bool PqQuantizer::build_code_lut(const uint8_t* code, float* out) const {
    if (cross_distance_table_.empty()) {
        return false;
    }
    // For each segment, the anchor's centroid ca_s indexes a contiguous
    // K-float row: table[s*K*K + ca_s*K + 0..K-1]. Copy that row into the LUT.
    const float* table_base = cross_distance_table_.data();
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t ca = read_code(code, bits_, s);
        const float* src = table_base + size_t(s) * K_ * K_ + ca * K_;
        float* dst = out + s * K_;
        std::memcpy(dst, src, K_ * sizeof(float));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Code-to-code distance via cross-distance table
// ---------------------------------------------------------------------------

float PqQuantizer::code_distance(const uint8_t* code_a,
                                 const uint8_t* code_b) const {
    if (cross_distance_table_.empty()) {
        // Fallback: compute from codebook directly (shouldn't happen after
        // train() or deserialize()).
        float acc = 0.0f;
        for (uint32_t s = 0; s < m_; s++) {
            const uint32_t ca = read_code(code_a, bits_, s);
            const uint32_t cb = read_code(code_b, bits_, s);
            const float* book =
                codebook_.data() + size_t(s) * K_ * sub_dim_;
            const float* va = book + ca * sub_dim_;
            const float* vb = book + cb * sub_dim_;
            switch (metric_) {
            case MetricKind::L2Sq:
                acc += l2sq_f32(va, vb, sub_dim_);
                break;
            case MetricKind::InnerProduct:
                acc += -dot_f32(va, vb, sub_dim_);
                break;
            }
        }
        return acc;
    }

    const float* table_base = cross_distance_table_.data();
    float acc = 0.0f;
    for (uint32_t s = 0; s < m_; s++) {
        const uint32_t ca = read_code(code_a, bits_, s);
        const uint32_t cb = read_code(code_b, bits_, s);
        acc += table_base[size_t(s) * K_ * K_ + ca * K_ + cb];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

void PqQuantizer::serialize(std::vector<uint8_t>& out) const {
    // Layout: {metric:u8, m:u8, bits:u8, dim:u32, codebook:float32[m*K*sub_dim]}
    const size_t header = sizeof(uint8_t) * 3 + sizeof(uint32_t);
    const size_t book_bytes = codebook_.size() * sizeof(float);
    out.resize(header + book_bytes);
    uint8_t* ptr = out.data();
    ptr[0] = static_cast<uint8_t>(metric_);
    ptr[1] = m_;
    ptr[2] = bits_;
    uint32_t d = dim_;
    std::memcpy(ptr + 3, &d, sizeof(d));
    if (book_bytes > 0) {
        std::memcpy(ptr + header, codebook_.data(), book_bytes);
    }
}

void PqQuantizer::deserialize(const uint8_t* in, size_t size) {
    const size_t header = sizeof(uint8_t) * 3 + sizeof(uint32_t);
    if (size < header) {
        throw Error(ErrorCode::CorruptIndex, "PQ deserialize: blob too small");
    }
    metric_ = static_cast<MetricKind>(in[0]);
    m_ = in[1];
    bits_ = in[2];
    uint32_t d;
    std::memcpy(&d, in + 3, sizeof(d));
    dim_ = d;
    if (m_ == 0 || dim_ % static_cast<Dim>(m_) != 0 ||
        (bits_ != 4 && bits_ != 8)) {
        throw Error(ErrorCode::CorruptIndex,
                    "PQ deserialize: invalid header");
    }
    K_ = 1u << bits_;
    sub_dim_ = dim_ / m_;
    const size_t book_floats = static_cast<size_t>(m_) * K_ * sub_dim_;
    const size_t book_bytes = book_floats * sizeof(float);
    if (size != header + book_bytes) {
        throw Error(ErrorCode::CorruptIndex, "PQ deserialize: size mismatch");
    }
    codebook_.assign(book_floats, 0.0f);
    if (book_bytes > 0) {
        std::memcpy(codebook_.data(), in + header, book_bytes);
    }
    build_cross_distance_table();
}

}  // namespace sextant
