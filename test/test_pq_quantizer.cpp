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
        // via the PQ LUT: preprocess_query on the vector itself, then
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
    // For the PQ (symmetric) path:
    //   build_code_lut(code_a) → lut, then lut_distance(code_b, lut)
    //   should equal code_distance(code_a, code_b).
    const uint64_t n = 300;
    const uint32_t dim = 64;
    const uint8_t m = 8;
    const uint8_t bits = 8;
    auto data = make_clustered_data(n, dim, /*n_clusters=*/25, /*seed=*/11);

    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, /*seed=*/222);
    q.train(data.data(), n);

    // Pick a few pairs and verify the LUT path matches direct code_distance.
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
            << "PQ LUT mismatch for pair " << pair;
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
// PQ LUT: preprocess_query → lut_distance consistency
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

// ---------------------------------------------------------------------------
// OPQ (PCA rotation)
// ---------------------------------------------------------------------------

TEST(PqQuantizer, OpqTrainComputesRotation) {
    // With OPQ enabled, train() must populate the rotation matrix.
    const uint32_t dim = 32;
    const uint8_t m = 4;
    const uint8_t bits = 8;
    PqQuantizer q(MetricKind::L2Sq, dim, m, bits);
    q.enable_opq();
    EXPECT_TRUE(q.opq_enabled());
    EXPECT_FALSE(q.has_rotation());  // before train

    auto data = make_clustered_data(300, dim, 8, 7);
    q.train(data.data(), 300);

    ASSERT_TRUE(q.has_rotation());
    const float* R = q.rotation();
    // R must be orthonormal: R R^T == I.
    for (uint32_t i = 0; i < dim; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            double acc = 0.0;
            for (uint32_t k = 0; k < dim; k++) {
                acc += double(R[i * dim + k]) * double(R[j * dim + k]);
            }
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(expected, acc, 1e-4)
                << "R R^T not identity at (" << i << "," << j << ")";
        }
    }
}

TEST(PqQuantizer, OpqDisabledNoRotation) {
    // Without OPQ, train() must leave rotation empty.
    const uint32_t dim = 32;
    const uint8_t m = 4;
    PqQuantizer q(MetricKind::L2Sq, dim, m, 8);
    auto data = make_clustered_data(200, dim, 6, 5);
    q.train(data.data(), 200);
    EXPECT_FALSE(q.has_rotation());
}

TEST(PqQuantizer, OpqSerializeDeserializeRoundTrip) {
    const uint32_t dim = 32;
    const uint8_t m = 4;
    const uint8_t bits = 8;
    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, 4242);
    q.enable_opq();
    auto data = make_clustered_data(250, dim, 8, 11);
    q.train(data.data(), 250);
    ASSERT_TRUE(q.has_rotation());

    std::vector<uint8_t> blob;
    q.serialize(blob);
    ASSERT_GT(blob.size(), 0u);

    PqQuantizer q2(MetricKind::L2Sq, dim, m, bits, 1);
    q2.deserialize(blob.data(), blob.size());
    ASSERT_TRUE(q2.has_rotation());

    // Rotation must be byte-exact after round-trip.
    const size_t rot_n = size_t(dim) * dim;
    EXPECT_EQ(0, std::memcmp(q.rotation(), q2.rotation(),
                             rot_n * sizeof(float)));

    // Encode + preprocess_query must produce identical results: same rotated
    // vector → same code, same LUT.
    for (uint64_t i = 0; i < 20; i++) {
        const float* vec = data.data() + (i * 13 % 250) * dim;
        std::vector<uint8_t> c1(q.code_size(), 0), c2(q.code_size(), 0);
        q.encode(vec, c1.data());
        q2.encode(vec, c2.data());
        EXPECT_EQ(0, std::memcmp(c1.data(), c2.data(), q.code_size()))
            << "code mismatch at " << i;

        std::vector<float> lut1(q.lut_size()), lut2(q.lut_size());
        q.preprocess_query(vec, lut1.data());
        q2.preprocess_query(vec, lut2.data());
        for (size_t e = 0; e < lut1.size(); e++) {
            EXPECT_NEAR(lut1[e], lut2[e], 1e-5f);
        }
    }
}

TEST(PqQuantizer, OpqEncodeAppliesRotation) {
    // Directly verify that encode() rotates: manually apply R, find nearest
    // centroids in the codebook, and confirm the resulting code matches
    // encode() on the original (un-rotated) vector.
    const uint32_t dim = 32;
    const uint8_t m = 4;
    const uint8_t bits = 8;
    const uint32_t sub_dim = dim / m;
    PqQuantizer q(MetricKind::L2Sq, dim, m, bits, 99);
    q.enable_opq();
    auto data = make_clustered_data(250, dim, 8, 11);
    q.train(data.data(), 250);
    ASSERT_TRUE(q.has_rotation());

    const float* R = q.rotation();
    for (uint64_t i = 0; i < 10; i++) {
        const float* vec = data.data() + (i * 31 % 250) * dim;
        std::vector<uint8_t> code_direct(q.code_size(), 0);
        q.encode(vec, code_direct.data());

        // Manually rotate: rotated[r] = sum_c R[r*dim+c] * vec[c].
        std::vector<float> rotated(dim);
        for (uint32_t r = 0; r < dim; r++) {
            double acc = 0.0;
            for (uint32_t c = 0; c < dim; c++) {
                acc += double(R[r * dim + c]) * double(vec[c]);
            }
            rotated[r] = float(acc);
        }
        // Nearest centroid per segment from the rotated vector + codebook.
        for (uint32_t s = 0; s < m; s++) {
            const float* sub = rotated.data() + s * sub_dim;
            const float* book = q.codebook() + size_t(s) * q.K() * sub_dim;
            uint32_t best = 0;
            float best_d = std::numeric_limits<float>::infinity();
            for (uint32_t c = 0; c < q.K(); c++) {
                const float* cen = book + c * sub_dim;
                float d = 0.0f;
                for (uint32_t d2 = 0; d2 < sub_dim; d2++) {
                    const float diff = sub[d2] - cen[d2];
                    d += diff * diff;
                }
                if (d < best_d) { best_d = d; best = c; }
            }
            // code_direct is 8-bit packed (one byte per segment).
            EXPECT_EQ(best, uint32_t(code_direct[s]))
                << "segment " << s << " vector " << i;
        }
    }
}

}  // namespace
}  // namespace sextant
