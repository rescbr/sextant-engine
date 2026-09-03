// Direct quantizer probe: train_uniform → encode → decode fidelity, and
// arith-scan decomposition (Σ (q·step)·c + c0) vs exact <q, x̂>.
#include <cstdio>
#include <cmath>
#include <vector>
#include "quant/scalar_lloydmax_quantizer.hpp"
#include "sextant/types.hpp"

int main() {
    const uint32_t dim = 64, n = 4096;
    std::vector<float> X(n * dim);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t d = 0; d < dim; ++d)
            X[i * dim + d] = std::sin(i * 0.021f + d * 0.7f) * (1.f + 0.1f * d);
    sextant::ScalarLloydMaxQuantizer q(sextant::MetricKind::InnerProduct, dim, 4);
    q.train_uniform(X.data(), n);
    // decode fidelity of vector 7
    const float* x = &X[7 * dim];
    uint8_t codes[dim];
    q.encode(x, codes);
    float xh[dim];
    q.decode(codes, xh);
    double err2 = 0, n2 = 0;
    for (uint32_t d = 0; d < dim; ++d) { double e = x[d] - xh[d]; err2 += e * e; n2 += (double)x[d] * x[d]; }
    printf("decode rel err: %.4f\n", std::sqrt(err2 / n2));
    // arith decomposition vs exact decode-dot, query = vector 3
    const float* qq = &X[3 * dim];
    q.encode(qq, codes);  // reuse codes var for the query? no — recompute for x7 below
    q.encode(x, codes);
    double exact = 0;
    for (uint32_t d = 0; d < dim; ++d) exact += (double)qq[d] * xh[d];
    // arith: Σ (q·step)·c + Σ q·lo
    const float* steps = q.steps();
    const float* levels = q.levels();
    double arith = 0, c0 = 0;
    for (uint32_t d = 0; d < dim; ++d) {
        const float lo = levels[d * 16];
        arith += (double)(qq[d] * steps[d]) * codes[d];
        c0 += (double)qq[d] * lo;
    }
    printf("exact <q,xh> = %.5f | arith+c0 = %.5f | arith only = %.5f\n",
           exact, arith + c0, arith);
    {const float* lv=&levels[0];
    printf("levels row0:"); for(int k=0;k<16;++k) printf(" %.3f", lv[k]);
    printf("\n");}
    printf("steps[0..3] = %f %f %f %f\n", steps[0], steps[1], steps[2], steps[3]);
    return 0;
}
