#include <gtest/gtest.h>
#include "quant/pq_quantizer.hpp"
#include "sextant/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace sextant {
namespace {

// ---------------------------------------------------------------------------
// Deterministic PRNG for reproducible tests.
// ---------------------------------------------------------------------------
class Rng {
public:
    explicit Rng(uint64_t seed) : rng_(seed) {}
    float uniform(float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        return d(rng_);
    }
    uint64_t next() { return dist_(rng_); }
private:
    std::mt19937_64 rng_;
    std::uniform_int_distribution<uint64_t> dist_;
};

/// Generate n × dim synthetic vectors with `n_clusters` Gaussian clusters.
/// Points are drawn from cluster centers (spread across the space) with
/// small Gaussian noise, so k-means has structure to find.
std::vector<float> make_clustered_data(uint64_t n, uint32_t dim,
                                       uint32_t n_clusters, uint64_t seed) {
    Rng rng(seed);
    std::vector<float> centers(size_t(n_clusters) * dim);
    for (auto& v : centers) v = rng.uniform(-10.0f, 10.0f);

    std::vector<float> data(size_t(n) * dim);
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t c = static_cast<uint32_t>(rng.next() % n_clusters);
        for (uint32_t d = 0; d < dim; d++) {
            data[i * dim + d] = centers[c * dim + d] + noise(gen);
        }
    }
    return data;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(PqQuantizer, BasicConstruction) {
    PqQuantizer q(MetricKind::L2Sq, 128, 32, 8);
    EXPECT_EQ(32u, q.code_size());
    EXPECT_EQ(32u, q.m());
}

TEST(PqQuantizer, Construction4Bit) {
    // m=16 segments, 4 bits → 2 codes per byte → 8 bytes.
    PqQuantizer q(MetricKind::L2Sq, 128, 16, 4);
    EXPECT_EQ(8u, q.code_size());
    EXPECT_EQ(16u, q.m());
}

TEST(PqQuantizer, InvalidM) {
    EXPECT_THROW(PqQuantizer(MetricKind::L2Sq, 128, 0, 8), Error);
}

TEST(PqQuantizer, InvalidDimDivisibility) {
    // dim=100, m=32 → 100 % 32 != 0
    EXPECT_THROW(PqQuantizer(MetricKind::L2Sq, 100, 32, 8), Error);
}

TEST(PqQuantizer, InvalidBits) {
    EXPECT_THROW(PqQuantizer(MetricKind::L2Sq, 128, 16, 6), Error);
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

TEST(PqQuantizer, TrainProducesNontrivialCodebook) {
    const uint64_t n = 1000;
    const uint32_t dim = 128;
    const uint8_t m = 16;
    const uint8_t bits = 8;
    auto data = make_clustered_data(n, dim, /*n_clusters=*/50, /*seed=*/42);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/12345);
    q.train(data.data(), n);

    // After training, the codebook should not be all zeros.
    // The codebook is internal, but we can check via serialization round-trip
    // and encoding behavior. At minimum, encode should produce valid codes.
    std::vector<uint8_t> code(q.code_size(), 0xFF);
    q.encode(data.data(), code.data());
    // Code should not be all-zeros (highly unlikely for clustered data).
    bool any_nonzero = false;
    for (auto b : code) { if (b != 0) { any_nonzero = true; break; } }
    EXPECT_TRUE(any_nonzero);
}

TEST(PqQuantizer, TrainZeroSamplesThrows) {
    PqQuantizer q(MetricKind::L2Sq, 128, 16, 8);
    EXPECT_THROW(q.train(nullptr, 0), Error);
}

// ---------------------------------------------------------------------------
// Encode → reconstruct: quantization error should be bounded
// ---------------------------------------------------------------------------

TEST(PqQuantizer, EncodeRoundTripErrorBounded) {
    const uint64_t n = 500;
    const uint32_t dim = 64;
    const uint8_t m = 8;
    const uint8_t bits = 8;
    auto data = make_clustered_data(n, dim, /*n_clusters=*/30, /*seed=*/7);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/999);
    q.train(data.data(), n);

    // For each vector, encode it, then find the L2 distance between the
    // original and its reconstruction (sum of assigned centroids). The
    // quantization error should be small relative to the data spread.
    double total_recon_sqerr = 0.0;
    double total_data_sqnorm = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const float* vec = data.data() + i * dim;
        std::vector<uint8_t> code(q.code_size(), 0);
        q.encode(vec, code.data());

        // Reconstruct from code: for each segment, look up the centroid.
        // We can't access the codebook directly, so measure the encoding
        // via the ADC LUT: preprocess_query on the vector itself, then
        // lut_distance gives the sum of sub-distances to assigned centroids
        // = the total squared quantization error.
        std::vector<float> lut(q.lut_size(), 0.0f);
        q.preprocess_query(vec, lut.data());
        const float recon_sqerr = q.lut_distance(code.data(), lut.data());
        total_recon_sqerr += recon_sqerr;

        for (uint32_t d = 0; d < dim; d++) {
            total_data_sqnorm += double(vec[d]) * vec[d];
        }
    }

    // Average quantization error per vector should be a small fraction of
    // the average data squared norm. With tight clusters (stddev=0.5) and
    // 256 centroids per segment, the reconstruction error should be tiny.
    const double avg_recon_err = total_recon_sqerr / n;
    const double avg_data_sqnorm = total_data_sqnorm / n;
    // Reconstruction error should be much smaller than data norm.
    EXPECT_LT(avg_recon_err, avg_data_sqnorm * 0.5);
}

// ---------------------------------------------------------------------------
// Serialize → Deserialize: byte-exact codebook
// ---------------------------------------------------------------------------

TEST(PqQuantizer, SerializeDeserializeByteExact) {
    const uint64_t n = 200;
    const uint32_t dim = 64;
    const uint8_t m = 8;
    const uint8_t bits = 8;
    auto data = make_clustered_data(n, dim, /*n_clusters=*/20, /*seed=*/55);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/777);
    q.train(data.data(), n);

    std::vector<uint8_t> blob;
    q.serialize(blob);
    EXPECT_GT(blob.size(), 0u);

    PqQuantizer q2(MetricKind::L2Sq, dim, m, bits, /*seed=*/1);
    q2.deserialize(blob.data(), blob.size());

    // Encode the same vectors with both quantizers — results must match
    // (codebook is byte-exact after round-trip).
    for (uint64_t i = 0; i < 20; i++) {
        const float* vec = data.data() + (i * 37 % n) * dim;
        std::vector<uint8_t> code1(q.code_size(), 0);
        std::vector<uint8_t> code2(q2.code_size(), 0);
        q.encode(vec, code1.data());
        q2.encode(vec, code2.data());
        EXPECT_EQ(0, std::memcmp(code1.data(), code2.data(), q.code_size()))
            << "Code mismatch at vector " << i;
    }

    // code_distance should also match.
    std::vector<uint8_t> ca(q.code_size(), 0), cb(q.code_size(), 0);
    q.encode(data.data(), ca.data());
    q.encode(data.data() + dim, cb.data());
    EXPECT_FLOAT_EQ(q.code_distance(ca.data(), cb.data()),
                    q2.code_distance(ca.data(), cb.data()));
}

TEST(PqQuantizer, SerializeDeserialize4Bit) {
    const uint64_t n = 200;
    const uint32_t dim = 32;
    const uint8_t m = 4;
    const uint8_t bits = 4;
    auto data = make_clustered_data(n, dim, /*n_clusters=*/10, /*seed=*/33);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/888);
    q.train(data.data(), n);

    std::vector<uint8_t> blob;
    q.serialize(blob);

    PqQuantizer q2(MetricKind::L2Sq, dim, m, bits, /*seed=*/1);
    q2.deserialize(blob.data(), blob.size());

    std::vector<uint8_t> code1(q.code_size(), 0);
    std::vector<uint8_t> code2(q2.code_size(), 0);
    q.encode(data.data(), code1.data());
    q2.encode(data.data(), code2.data());
    EXPECT_EQ(0, std::memcmp(code1.data(), code2.data(), q.code_size()));
}

// ---------------------------------------------------------------------------
// LUT path: preprocess_query → lut_distance ≈ code_distance
// ---------------------------------------------------------------------------

TEST(PqQuantizer, LUTDistanceMatchesCodeDistance) {
    // For the SDC (symmetric) path:
    //   build_code_lut(code_a) → lut, then lut_distance(code_b, lut)
    //   should equal code_distance(code_a, code_b).
    const uint64_t n = 300;
    const uint32_t dim = 64;
    const uint8_t m = 8;
    const uint8_t bits = 8;
    auto data = make_clustered_data(n, dim, /*n_clusters=*/25, /*seed=*/11);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/222);
    q.train(data.data(), n);

    // Pick a few pairs and verify SDC LUT path matches direct code_distance.
    for (uint64_t pair = 0; pair < 20; pair++) {
        const float* va = data.data() + (pair * 13 % n) * dim;
        const float* vb = data.data() + (pair * 29 % n) * dim;
        std::vector<uint8_t> ca(q.code_size(), 0), cb(q.code_size(), 0);
        q.encode(va, ca.data());
        q.encode(vb, cb.data());

        const float direct = q.code_distance(ca.data(), cb.data());

        std::vector<float> lut(q.lut_size(), 0.0f);
        ASSERT_TRUE(q.build_code_lut(ca.data(), lut.data()));
        const float via_lut = q.lut_distance(cb.data(), lut.data());

        EXPECT_FLOAT_EQ(direct, via_lut)
            << "SDC LUT mismatch for pair " << pair;
    }
}

TEST(PqQuantizer, CodeDistanceSelfIsZero) {
    const uint64_t n = 100;
    const uint32_t dim = 32;
    const uint8_t m = 4;
    const uint8_t bits = 8;
    auto data = make_clustered_data(n, dim, /*n_clusters=*/10, /*seed=*/44);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/666);
    q.train(data.data(), n);

    for (uint64_t i = 0; i < 10; i++) {
        const float* vec = data.data() + i * dim;
        std::vector<uint8_t> code(q.code_size(), 0);
        q.encode(vec, code.data());
        EXPECT_FLOAT_EQ(0.0f, q.code_distance(code.data(), code.data()));
    }
}

TEST(PqQuantizer, CodeDistanceSymmetric) {
    const uint64_t n = 100;
    const uint32_t dim = 32;
    const uint8_t m = 4;
    const uint8_t bits = 8;
    auto data = make_clustered_data(n, dim, /*n_clusters=*/10, /*seed=*/44);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/666);
    q.train(data.data(), n);

    std::vector<uint8_t> ca(q.code_size(), 0), cb(q.code_size(), 0);
    q.encode(data.data(), ca.data());
    q.encode(data.data() + dim, cb.data());
    EXPECT_FLOAT_EQ(q.code_distance(ca.data(), cb.data()),
                    q.code_distance(cb.data(), ca.data()));
}

// ---------------------------------------------------------------------------
// ADC LUT: preprocess_query → lut_distance consistency
// ---------------------------------------------------------------------------

TEST(PqQuantizer, PreprocessQueryLUTShape) {
    const uint32_t dim = 64;
    const uint8_t m = 8;
    const uint8_t bits = 8;
    PqQuantizer q(MetricKind::L2Sq, dim, m, bits);
    auto data = make_clustered_data(100, dim, 10, 99);
    q.train(data.data(), 100);

    // lut_size should be m * K.
    EXPECT_EQ(q.lut_size(), uint32_t(m) * (1u << bits));

    std::vector<float> query(dim, 1.0f);
    std::vector<float> lut(q.lut_size(), 0.0f);
    q.preprocess_query(query.data(), lut.data());
    // All entries should be finite (no NaN/Inf from untrained codebook).
    for (float v : lut) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

}  // namespace
}  // namespace sextant
