// Verify ScalarLloydMaxQuantizer serialize/deserialize round-trips the levels
// table exactly, and that encode->store->decode via the round-tripped
// quantizer reproduces the original vector.
#include "../src/quant/scalar_lloydmax_quantizer.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace sextant;

int main() {
    const uint32_t dim = 768;
    const uint32_t n = 5000;
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = ((i * 2654435761u) % 1000) / 1000.0f - 0.5f;

    ScalarLloydMaxQuantizer a(MetricKind::L2Sq, dim, 4);
    a.train(data.data(), n, 2, 30);

    std::vector<uint8_t> blob;
    a.serialize(blob);
    std::printf("blob size: %zu (expect 10 + %zu)\n", blob.size(),
                static_cast<size_t>(dim) * 16 * 4);

    ScalarLloydMaxQuantizer b(MetricKind::L2Sq, dim, 4);
    b.deserialize(blob.data(), blob.size());

    // 1. levels identical?
    const float* la = a.levels();
    const float* lb = b.levels();
    bool same = std::memcmp(la, lb, static_cast<size_t>(dim) * 16 * 4) == 0;
    std::printf("levels identical: %s\n", same ? "YES" : "NO");
    if (!same) {
        for (uint32_t d = 0; d < 3; ++d) {
            std::printf("  d=%u: a=[%g %g %g] b=[%g %g %g]\n", d,
                        la[d*16], la[d*16+1], la[d*16+2],
                        lb[d*16], lb[d*16+1], lb[d*16+2]);
        }
    }

    // 2. encode with a, decode with b: reproduces input?
    const uint32_t cs = a.code_size();
    std::printf("code_size: %u\n", cs);
    std::vector<uint8_t> code(cs);
    std::vector<float> dec(dim);
    a.encode(data.data(), code.data());
    b.decode(code.data(), dec.data());
    double err = 0;
    for (uint32_t d = 0; d < dim; ++d)
        err += (data[d] - dec[d]) * (data[d] - dec[d]);
    std::printf("encode(a)/decode(b) roundtrip err: %g (expect ~1e-4)\n", err);

    // 3. also decode with a itself (sanity)
    a.decode(code.data(), dec.data());
    double err2 = 0;
    for (uint32_t d = 0; d < dim; ++d)
        err2 += (data[d] - dec[d]) * (data[d] - dec[d]);
    std::printf("encode(a)/decode(a) err: %g\n", err2);
    return 0;
}
