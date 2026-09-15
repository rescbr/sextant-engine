// Microbench: per-query scalar_i8_dots4 vs 4-query scalar_i8_dots4_q4.
#include "tree/coders/coder_util.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sextant::tree::coders;

static ScalarScanCtx make_ctx(uint32_t dim) {
    ScalarScanCtx c{};
    c.dim = dim;
    c.cs = (dim + 1) / 2;
    c.i8_inv = 1.0f / 127.0f;
    c.slm_shaped = 0;
    c.i8_mode = 1;
    int8_t* a8 = (int8_t*)aligned_alloc(64, dim + 32);
    for (uint32_t d = 0; d < dim; ++d) a8[d] = (int8_t)(rand() % 255 - 127);
    c.a8 = a8;
    return c;
}

int main() {
    const uint32_t dim = 768;
    const uint32_t rows = 5000;
    const uint32_t Q = 252;
    ScalarScanCtx cs4[4];
    for (uint32_t q = 0; q < 4; ++q) cs4[q] = make_ctx(dim);
    std::vector<uint8_t> codes((size_t)rows * (dim / 2 + 16));
    for (auto& b : codes) b = rand();
    const uint8_t* cp[4];
    double sink = 0;
    float dots[4][4];

    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t q = 0; q < Q; ++q) {
        const ScalarScanCtx& c = cs4[q & 3];
        float d1[4];
        for (uint32_t i = 0; i + 4 <= rows; i += 4) {
            for (uint32_t v = 0; v < 4; ++v)
                cp[v] = codes.data() + (size_t)(i + v) * (dim / 2 + 16);
            scalar_i8_dots4(c, cp, d1);
            sink += d1[0] + d1[3];
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    const double t_cur = std::chrono::duration<double>(t1 - t0).count();
    const ScalarScanCtx* c4[4] = {&cs4[0], &cs4[1], &cs4[2], &cs4[3]};
    t0 = std::chrono::steady_clock::now();
    for (uint32_t grp = 0; grp < Q / 4; ++grp) {
        for (uint32_t i = 0; i + 4 <= rows; i += 4) {
            for (uint32_t v = 0; v < 4; ++v)
                cp[v] = codes.data() + (size_t)(i + v) * (dim / 2 + 16);
            scalar_i8_dots4_q4(c4, cp, dots);
            sink += dots[0][0] + dots[3][3];
        }
    }
    t1 = std::chrono::steady_clock::now();
    const double t_bat = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "cur: %.3fs  q4: %.3fs  speedup: %.2fx (sink %.0f)\n",
            t_cur, t_bat, t_cur / t_bat, sink);
    return 0;
}
