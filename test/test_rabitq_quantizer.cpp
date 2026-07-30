#include <gtest/gtest.h>
#include "quant/rabitq_quantizer.hpp"
#include "sextant/error.hpp"
#include "simd_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace sextant {
namespace {

std::vector<float> make_random_vectors(uint64_t n, uint32_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(size_t(n) * dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

TEST(RaBitQQuantizer, Construction) {
    RaBitQQuantizer q(MetricKind::L2Sq, 64, 12345);
    EXPECT_EQ(16u, q.m());
    EXPECT_EQ(16u, q.K());
    EXPECT_EQ(16u * 4u / 8u, q.code_size());
    EXPECT_EQ(64u, q.padded_dim());
}

TEST(RaBitQQuantizer, RejectsBadDim) {
    EXPECT_THROW(RaBitQQuantizer(MetricKind::L2Sq, 66), Error);
    EXPECT_THROW(RaBitQQuantizer(MetricKind::L2Sq, 0), Error);
}

TEST(RaBitQQuantizer, PaddedDimFor768) {
    RaBitQQuantizer q(MetricKind::L2Sq, 768);
    EXPECT_EQ(1024u, q.padded_dim());
}

TEST(RaBitQQuantizer, FwhtCorrectness) {
    RaBitQQuantizer q(MetricKind::L2Sq, 8);
    float buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    q.fwht_inplace(buf, 8);
    EXPECT_NEAR(36.0f, buf[0], 1e-4f);
    EXPECT_NEAR(-4.0f, buf[1], 1e-4f);
    EXPECT_NEAR(-8.0f, buf[2], 1e-4f);
    EXPECT_NEAR(0.0f, buf[3], 1e-4f);
}

TEST(RaBitQQuantizer, RotationOrthogonality) {
    const uint32_t dim = 32;
    RaBitQQuantizer q(MetricKind::L2Sq, dim, 42);
    std::mt19937 rng(7);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (int t = 0; t < 10; t++) {
        std::vector<float> x(dim);
        for (auto& v : x) v = d(rng);
        float norm_in = 0.0f;
        for (auto v : x) norm_in += v * v;
        std::vector<float> rot(q.padded_dim(), 0.0f);
        q.rotate(x.data(), rot.data());
        float norm_rot = 0.0f;
        for (auto v : rot) norm_rot += v * v;
        EXPECT_NEAR(norm_in, norm_rot, 1e-3f * norm_in);
    }
}

TEST(RaBitQQuantizer, SignEncodingMatchesRotatedResidual) {
    const uint32_t dim = 16;
    RaBitQQuantizer q(MetricKind::L2Sq, dim, 99);
    auto data = make_random_vectors(1, dim, 3);
    std::vector<float> centroid(dim, 0.0f);
    std::vector<uint8_t> code(q.full_code_size(), 0);
    q.encode_with_centroid(data.data(), centroid.data(), code.data());

    std::vector<float> residual(dim);
    for (uint32_t i = 0; i < dim; i++) residual[i] = data[i] - centroid[i];
    std::vector<float> rotated(q.padded_dim(), 0.0f);
    for (uint32_t i = 0; i < dim; i++) rotated[i] = residual[i];
    q.rotate(rotated.data(), rotated.data());

    for (uint32_t i = 0; i < dim; i++) {
        const uint32_t group = i / 4;
        const uint32_t bit = i % 4;
        const uint32_t byte_off = group / 2;
        const uint8_t shift = static_cast<uint8_t>((group % 2) * 4);
        const uint32_t nibble =
            (static_cast<uint32_t>(code[byte_off]) >> shift) & 0x0Fu;
        const uint32_t expected = (rotated[i] < 0.0f) ? 1u : 0u;
        EXPECT_EQ(expected, (nibble >> bit) & 1u) << "dim " << i;
    }
}

TEST(RaBitQQuantizer, LutSubsetSumCorrectness) {
    const uint32_t dim = 16;
    RaBitQQuantizer q(MetricKind::L2Sq, dim, 77);
    auto query = make_random_vectors(1, dim, 5);
    std::vector<float> centroid(dim, 0.0f);
    std::vector<float> lut(q.lut_size());
    q.preprocess_query_with_centroid(query.data(), centroid.data(), lut.data());

    std::vector<float> residual(dim), rotated(q.padded_dim(), 0.0f);
    for (uint32_t i = 0; i < dim; i++) residual[i] = query[i] - centroid[i];
    for (uint32_t i = 0; i < dim; i++) rotated[i] = residual[i];
    q.rotate(rotated.data(), rotated.data());

    float v_min = rotated[0], v_max = rotated[0];
    for (uint32_t i = 0; i < dim; i++) {
        v_min = std::min(v_min, rotated[i]);
        v_max = std::max(v_max, rotated[i]);
    }
    const float span = v_max - v_min;
    const float delta = span / 15.0f;
    const float c1 = 2.0f * delta / q.get_padded_dim_sqrt();
    const float c2 = 2.0f * v_min / q.get_padded_dim_sqrt();
    std::vector<float> qq(dim);
    for (uint32_t i = 0; i < dim; i++) {
        qq[i] = (span > 0.0f) ? std::round((rotated[i] - v_min) / delta) : 0.0f;
    }

    for (uint32_t m = 0; m < dim / 4; m++) {
        const uint32_t ds = m * 4;
        for (uint32_t code = 0; code < 16; code++) {
            float expected = 0.0f;
            if (code & 1) expected += c1 * qq[ds + 0] + c2;
            if (code & 2) expected += c1 * qq[ds + 1] + c2;
            if (code & 4) expected += c1 * qq[ds + 2] + c2;
            if (code & 8) expected += c1 * qq[ds + 3] + c2;
            EXPECT_NEAR(expected, lut[m * 16 + code], 1e-4f);
        }
    }
}

TEST(RaBitQQuantizer, FastScanLut4Shape) {
    const uint32_t dim = 32;
    RaBitQQuantizer q(MetricKind::L2Sq, dim);
    auto query = make_random_vectors(1, dim, 11);
    std::vector<float> centroid(dim, 0.0f);
    std::vector<uint8_t> lut4(q.fastscan_lut_bytes());
    float scale;
    q.build_fastscan_lut4_with_centroid(query.data(), centroid.data(),
                                        lut4.data(), &scale);
    EXPECT_EQ(q.m() * 16u, q.fastscan_lut_bytes());
    for (auto v : lut4) EXPECT_LE(v, 15);
}

TEST(RaBitQQuantizer, DistanceEstimationL2) {
    const uint32_t dim = 64;
    RaBitQQuantizer q(MetricKind::L2Sq, dim, 2024);
    const uint32_t n = 40;
    auto data = make_random_vectors(n, dim, 9);
    std::vector<float> centroid(dim, 0.0f);
    auto queries = make_random_vectors(5, dim, 31);

    std::vector<std::vector<uint8_t>> codes(n);
    for (uint32_t i = 0; i < n; i++) {
        codes[i].resize(q.full_code_size());
        q.encode_with_centroid(data.data() + i * dim, centroid.data(),
                               codes[i].data());
    }

    std::vector<float> lut(q.lut_size());
    for (uint32_t qi = 0; qi < 5; qi++) {
        const float* query = queries.data() + qi * dim;
        q.preprocess_query_with_centroid(query, centroid.data(), lut.data());
        double total_abs_err = 0.0;
        uint32_t count = 0;
        for (uint32_t i = 0; i < n; i++) {
            float raw = 0.0f;
            for (uint32_t s = 0; s < q.m(); s++) {
                const uint32_t byte_off = s / 2;
                const uint8_t shift = static_cast<uint8_t>((s % 2) * 4);
                const uint32_t cid =
                    (static_cast<uint32_t>(codes[i][byte_off]) >> shift) &
                    0x0Fu;
                raw += lut[s * 16 + cid];
            }
            const float est = q.finalize_distance(
                raw, codes[i].data(), centroid.data());
            const float true_d =
                simd::l2sq_f32(query, data.data() + i * dim, dim);
            total_abs_err += std::fabs(est - true_d);
            count++;
        }
        const double mae = total_abs_err / count;
        const double avg_true = [&]() {
            double s = 0.0;
            for (uint32_t i = 0; i < n; i++)
                s += simd::l2sq_f32(query, data.data() + i * dim, dim);
            return s / n;
        }();
        EXPECT_LT(mae, 0.5 * avg_true + 1.0);
    }
}

TEST(RaBitQQuantizer, SerializeDeserializeRoundTrip) {
    RaBitQQuantizer q(MetricKind::InnerProduct, 48, 0xDEADBEEFULL);
    std::vector<uint8_t> blob;
    q.serialize(blob);

    RaBitQQuantizer q2(MetricKind::L2Sq, 48, 1);
    q2.deserialize(blob.data(), blob.size());
    EXPECT_EQ(64u, q2.padded_dim());
    EXPECT_EQ(MetricKind::InnerProduct, q2.metric());

    auto data = make_random_vectors(2, 48, 4);
    std::vector<float> centroid(48, 0.0f);
    std::vector<uint8_t> c1(q.full_code_size()), c2(q2.full_code_size());
    q.encode_with_centroid(data.data(), centroid.data(), c1.data());
    q2.encode_with_centroid(data.data(), centroid.data(), c2.data());
    EXPECT_EQ(0, std::memcmp(c1.data(), c2.data(), q.full_code_size()));
}

TEST(RaBitQQuantizer, DeserializeRejectsBadMagic) {
    RaBitQQuantizer q(MetricKind::L2Sq, 16);
    std::vector<uint8_t> blob = {0x00, 0x00, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_THROW(q.deserialize(blob.data(), blob.size()), Error);
}

}  // namespace
}  // namespace sextant
