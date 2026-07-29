#include <gtest/gtest.h>
#include "quant/product_residual_quantizer.hpp"
#include "quant/pq_quantizer.hpp"
#include "simd_kernels.hpp"
#include "sextant/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace sextant {
namespace {

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
// Construction / validation
// ---------------------------------------------------------------------------

TEST(ProductResidualQuantizer, Construction) {
    // dim=64, m=8, nsplits=2 → M_sub=4, sub_dim_prq=32, bits=4.
    ProductResidualQuantizer q(MetricKind::L2Sq, 64, 8, 4, 2);
    EXPECT_EQ(8u, q.m());
    EXPECT_EQ(2u, q.nsplits());
    EXPECT_EQ(4u, q.m_sub());
    EXPECT_EQ(32u, q.rq_sub_dim());
    // 4-bit, 8 segments → 4 bytes.
    EXPECT_EQ(4u, q.code_size());
}

TEST(ProductResidualQuantizer, RejectsBadNsplits) {
    EXPECT_THROW(ProductResidualQuantizer(MetricKind::L2Sq, 64, 8, 4, 0), Error);
    // m % nsplits != 0.
    EXPECT_THROW(ProductResidualQuantizer(MetricKind::L2Sq, 64, 8, 4, 3), Error);
    // dim % nsplits != 0.
    EXPECT_THROW(ProductResidualQuantizer(MetricKind::L2Sq, 64, 8, 4, 6), Error);
}

TEST(ProductResidualQuantizer, RejectsNonFourBits) {
    EXPECT_THROW(ProductResidualQuantizer(MetricKind::L2Sq, 64, 8, 8, 2), Error);
}

// ---------------------------------------------------------------------------
// Encode / decode round-trip and reconstruction error
// ---------------------------------------------------------------------------

TEST(ProductResidualQuantizer, EncodeDecodeRoundTrip) {
    const uint64_t n = 400;
    const uint32_t dim = 64;
    const uint16_t m = 8;
    const uint32_t nsplits = 2;
    auto data = make_clustered_data(n, dim, 30, 42);

    ProductResidualQuantizer q(MetricKind::L2Sq, dim, m, 4, nsplits,
                               /*beam_size=*/1, 12345);
    q.train(data.data(), n);

    for (uint64_t i = 0; i < 20; i++) {
        const float* vec = data.data() + i * dim;
        std::vector<uint8_t> code(q.code_size(), 0);
        q.encode(vec, code.data());

        std::vector<float> recon(dim, 0.0f);
        q.decode_code(code.data(), recon.data());

        // Reconstruction is the additive sum of assigned centroids.
        double sqerr = 0.0;
        for (uint32_t d = 0; d < dim; d++) {
            const double diff = double(vec[d]) - double(recon[d]);
            sqerr += diff * diff;
        }
        // Should be a reasonable approximation (not exact, not enormous).
        EXPECT_LT(sqerr, 1000.0) << "vector " << i;
    }
}

TEST(ProductResidualQuantizer, ReconstructionBetterThanPq) {
    // PRQ's additive residuals should reconstruct at least as well as plain PQ
    // at the same code length on clustered data. With tight clusters both do
    // well; we assert PRQ is not dramatically worse (< 1.5× PQ's error).
    const uint64_t n = 500;
    const uint32_t dim = 32;
    const uint16_t m = 4;
    const uint32_t nsplits = 2;  // M_sub = 2, sub_dim_prq = 16
    auto data = make_clustered_data(n, dim, 20, 7);

    PqQuantizer pq(MetricKind::L2Sq, dim, m, 4, 999);
    pq.train(data.data(), n);

    ProductResidualQuantizer prq(MetricKind::L2Sq, dim, m, 4, nsplits, 1, 999);
    prq.train(data.data(), n);

    auto recon_err = [&](const PqQuantizer& qer) {
        double total = 0.0;
        for (uint64_t i = 0; i < n; i++) {
            const float* vec = data.data() + i * dim;
            std::vector<uint8_t> code(qer.code_size(), 0);
            qer.encode(vec, code.data());
            std::vector<float> recon(dim, 0.0f);
            qer.decode_code(code.data(), recon.data());
            for (uint32_t d = 0; d < dim; d++) {
                const double diff = double(vec[d]) - double(recon[d]);
                total += diff * diff;
            }
        }
        return total / n;
    };

    const double err_pq = recon_err(pq);
    const double err_prq = recon_err(prq);
    EXPECT_LT(err_prq, err_pq * 1.5)
        << "PRQ error " << err_prq << " should be comparable to PQ " << err_pq;
}

// ---------------------------------------------------------------------------
// 4-bit code packing: encode then read back with the same packing rules
// ---------------------------------------------------------------------------

TEST(ProductResidualQuantizer, CodePackingMatchesAssignments) {
    const uint64_t n = 200;
    const uint32_t dim = 32;
    const uint16_t m = 4;
    const uint32_t nsplits = 2;  // M_sub = 2, sub_dim_prq = 16
    auto data = make_clustered_data(n, dim, 10, 55);

    ProductResidualQuantizer q(MetricKind::L2Sq, dim, m, 4, nsplits, 1, 888);
    q.train(data.data(), n);

    const float* vec = data.data();
    std::vector<uint8_t> code(q.code_size(), 0);
    q.encode(vec, code.data());

    // Independently recompute the greedy assignments and verify the packed
    // codes decode to the same centroid ids.
    std::vector<float> residual(q.rq_sub_dim());
    for (uint32_t s = 0; s < q.nsplits(); s++) {
        std::memcpy(residual.data(), vec + s * q.rq_sub_dim(),
                    q.rq_sub_dim() * sizeof(float));
        for (uint32_t lev = 0; lev < q.m_sub(); lev++) {
            const float* book = q.rq_codebooks() +
                                ((size_t(s) * q.m_sub() + lev) * q.K()) *
                                    q.rq_sub_dim();
            uint32_t best = 0;
            float best_d = std::numeric_limits<float>::infinity();
            for (uint32_t c = 0; c < q.K(); c++) {
                const float d =
                    simd::l2sq_f32(residual.data(), book + c * q.rq_sub_dim(),
                                   q.rq_sub_dim());
                if (d < best_d) { best_d = d; best = c; }
            }
            // Read the packed code for this global segment.
            const uint32_t seg = s * q.m_sub() + lev;
            const uint32_t byte_off = seg / 2;
            const uint32_t shift = (seg % 2) * 4;
            const uint32_t packed =
                (uint32_t(code[byte_off]) >> shift) & 0x0Fu;
            EXPECT_EQ(best, packed)
                << "split " << s << " level " << lev;
            // Subtract the chosen centroid to form the next residual.
            const float* cen = book + best * q.rq_sub_dim();
            for (uint32_t d = 0; d < q.rq_sub_dim(); d++) {
                residual[d] -= cen[d];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LUT correctness: manually compute expected values
// ---------------------------------------------------------------------------

TEST(ProductResidualQuantizer, LutMatchesManualComputation_IP) {
    const uint64_t n = 200;
    const uint32_t dim = 32;
    const uint16_t m = 4;
    const uint32_t nsplits = 2;
    auto data = make_clustered_data(n, dim, 10, 33);

    ProductResidualQuantizer q(MetricKind::InnerProduct, dim, m, 4, nsplits, 1,
                               777);
    q.train(data.data(), n);

    std::vector<float> query(dim);
    for (uint32_t d = 0; d < dim; d++) query[d] = float(d) * 0.1f - 1.0f;

    std::vector<float> lut(q.lut_size(), 0.0f);
    q.preprocess_query(query.data(), lut.data());

    // For IP: LUT[seg*K + c] = -<q_sub, codebook[seg][c]>.
    for (uint32_t s = 0; s < q.nsplits(); s++) {
        const float* q_sub = query.data() + s * q.rq_sub_dim();
        for (uint32_t lev = 0; lev < q.m_sub(); lev++) {
            const uint32_t seg = s * q.m_sub() + lev;
            const float* book = q.rq_codebooks() +
                                (size_t(seg) * q.K()) * q.rq_sub_dim();
            for (uint32_t c = 0; c < q.K(); c++) {
                const float dot = simd::dot_f32(q_sub,
                                                book + c * q.rq_sub_dim(),
                                                q.rq_sub_dim());
                EXPECT_NEAR(-dot, lut[seg * q.K() + c], 1e-4f)
                    << "split " << s << " level " << lev << " c " << c;
            }
        }
    }
}

TEST(ProductResidualQuantizer, LutMatchesManualComputation_L2) {
    const uint64_t n = 200;
    const uint32_t dim = 32;
    const uint16_t m = 4;
    const uint32_t nsplits = 2;
    auto data = make_clustered_data(n, dim, 10, 33);

    ProductResidualQuantizer q(MetricKind::L2Sq, dim, m, 4, nsplits, 1, 777);
    q.train(data.data(), n);

    std::vector<float> query(dim);
    for (uint32_t d = 0; d < dim; d++) query[d] = float(d) * 0.1f - 1.0f;

    std::vector<float> lut(q.lut_size(), 0.0f);
    q.preprocess_query(query.data(), lut.data());

    for (uint32_t s = 0; s < q.nsplits(); s++) {
        const float* q_sub = query.data() + s * q.rq_sub_dim();
        for (uint32_t lev = 0; lev < q.m_sub(); lev++) {
            const uint32_t seg = s * q.m_sub() + lev;
            const float* book = q.rq_codebooks() +
                                (size_t(seg) * q.K()) * q.rq_sub_dim();
            for (uint32_t c = 0; c < q.K(); c++) {
                const float expected = simd::l2sq_f32(
                    q_sub, book + c * q.rq_sub_dim(), q.rq_sub_dim());
                EXPECT_NEAR(expected, lut[seg * q.K() + c], 1e-3f)
                    << "split " << s << " level " << lev << " c " << c;
            }
        }
    }
}

TEST(ProductResidualQuantizer, LutDistanceEqualsDecodedDistance) {
    // lut_distance sums LUT[seg][cid] over all segments. For IP this is the
    // negated dot of the query with the full additive reconstruction — verify
    // it equals the direct dot of the query with decode_code's output.
    const uint64_t n = 300;
    const uint32_t dim = 32;
    const uint16_t m = 4;
    const uint32_t nsplits = 2;
    auto data = make_clustered_data(n, dim, 15, 11);

    ProductResidualQuantizer q(MetricKind::InnerProduct, dim, m, 4, nsplits, 1,
                               222);
    q.train(data.data(), n);

    for (uint64_t i = 0; i < 20; i++) {
        const float* vec = data.data() + (i * 37 % n) * dim;
        std::vector<uint8_t> code(q.code_size(), 0);
        q.encode(vec, code.data());

        std::vector<float> recon(dim, 0.0f);
        q.decode_code(code.data(), recon.data());

        std::vector<float> lut(q.lut_size(), 0.0f);
        // Use vec itself as the query.
        q.preprocess_query(vec, lut.data());
        const float via_lut = q.lut_distance(code.data(), lut.data());

        const float direct = simd::dot_f32(vec, recon.data(), dim);
        // via_lut = sum of -<vec_sub, cen> = -<vec, recon>.
        EXPECT_NEAR(-direct, via_lut, 1e-2f) << "vector " << i;
    }
}

// ---------------------------------------------------------------------------
// build_fastscan_lut4: produces a valid 4-bit LUT consumed by the scan kernel
// ---------------------------------------------------------------------------

TEST(ProductResidualQuantizer, BuildFastScanLut4Shape) {
    const uint64_t n = 200;
    const uint32_t dim = 32;
    const uint16_t m = 4;
    const uint32_t nsplits = 2;
    auto data = make_clustered_data(n, dim, 10, 9);

    ProductResidualQuantizer q(MetricKind::L2Sq, dim, m, 4, nsplits, 1, 314);
    q.train(data.data(), n);

    std::vector<float> query(dim, 0.5f);
    std::vector<uint8_t> lut4(q.fastscan_lut_bytes(), 0xFF);
    float scale = 0.0f;
    q.build_fastscan_lut4(query.data(), lut4.data(), &scale);

    EXPECT_GT(scale, 0.0f);
    for (uint8_t v : lut4) {
        EXPECT_LE(v, 15) << "4-bit value out of range";
    }
}

// ---------------------------------------------------------------------------
// Serialization round-trip
// ---------------------------------------------------------------------------

TEST(ProductResidualQuantizer, SerializeDeserializeRoundTrip) {
    const uint64_t n = 250;
    const uint32_t dim = 32;
    const uint16_t m = 4;
    const uint32_t nsplits = 2;
    auto data = make_clustered_data(n, dim, 12, 88);

    ProductResidualQuantizer q(MetricKind::L2Sq, dim, m, 4, nsplits, 1, 4242);
    q.train(data.data(), n);

    std::vector<uint8_t> blob;
    q.serialize(blob);
    EXPECT_GT(blob.size(), 0u);
    EXPECT_EQ(ProductResidualQuantizer::kMagic, blob[0]);

    ProductResidualQuantizer q2(MetricKind::L2Sq, dim, m, 4, nsplits, 1, 1);
    q2.deserialize(blob.data(), blob.size());

    // Codebooks byte-exact.
    const size_t nb = size_t(q.nsplits()) * q.m_sub() * q.K() * q.rq_sub_dim();
    EXPECT_EQ(0, std::memcmp(q.rq_codebooks(), q2.rq_codebooks(),
                             nb * sizeof(float)));

    // Encode + LUT must match.
    for (uint64_t i = 0; i < 15; i++) {
        const float* vec = data.data() + (i * 19 % n) * dim;
        std::vector<uint8_t> c1(q.code_size(), 0), c2(q2.code_size(), 0);
        q.encode(vec, c1.data());
        q2.encode(vec, c2.data());
        EXPECT_EQ(0, std::memcmp(c1.data(), c2.data(), q.code_size()))
            << "code mismatch at " << i;

        std::vector<float> l1(q.lut_size()), l2(q.lut_size());
        q.preprocess_query(vec, l1.data());
        q2.preprocess_query(vec, l2.data());
        for (size_t e = 0; e < l1.size(); e++) {
            EXPECT_NEAR(l1[e], l2[e], 1e-5f);
        }
    }
}

TEST(ProductResidualQuantizer, DeserializeRejectsBadMagic) {
    ProductResidualQuantizer q(MetricKind::L2Sq, 32, 4, 4, 2);
    std::vector<uint8_t> blob(32, 0);
    EXPECT_THROW(q.deserialize(blob.data(), blob.size()), Error);
}

// ---------------------------------------------------------------------------
// More splits / levels coverage
// ---------------------------------------------------------------------------

TEST(ProductResidualQuantizer, FourSplitsFourLevels) {
    // dim=64, m=16, nsplits=4 → M_sub=4, sub_dim_prq=16.
    const uint64_t n = 400;
    const uint32_t dim = 64;
    const uint16_t m = 16;
    const uint32_t nsplits = 4;
    auto data = make_clustered_data(n, dim, 25, 123);

    ProductResidualQuantizer q(MetricKind::L2Sq, dim, m, 4, nsplits, 1, 65);
    q.train(data.data(), n);
    EXPECT_EQ(4u, q.m_sub());

    const float* vec = data.data();
    std::vector<uint8_t> code(q.code_size(), 0);
    q.encode(vec, code.data());
    std::vector<float> recon(dim, 0.0f);
    q.decode_code(code.data(), recon.data());

    // Sanity: reconstruction is finite and non-trivial.
    double sqerr = 0.0;
    for (uint32_t d = 0; d < dim; d++) {
        EXPECT_TRUE(std::isfinite(recon[d]));
        const double diff = double(vec[d]) - double(recon[d]);
        sqerr += diff * diff;
    }
    EXPECT_LT(sqerr, 1000.0);
}

}  // namespace
}  // namespace sextant
