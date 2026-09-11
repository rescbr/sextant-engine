// spike_plane_quant.cpp — bytes-vs-containment curve for a PCA-128
// stage-1 plane (plan D). Complements spike_routing_ceiling (closed):
// that spike mapped the frontier at fp16-256B / i8-128B / i8-64B; this
// one maps 16–96 B/vec with each family's BEST deployable scoring
// (asymmetric fp32 query + per-dim trained codebooks + per-vector
// length renorm), NOT one shared technique. Variants are comparable at
// the outcome level (honest bytes vs containment), not implementation.
//
// Metric: containment@gt_k vs page-weighted coverage fraction, plus the
// NOISE-INFLATION diagnostic (max_quantized - max_true per query/leaf)
// that falsified PCA-32: max-statistics over many members inflate with
// per-vector quantization noise, so recall-style prior-art claims do
// not transfer. Inflation mean/p95 explains each variant's containment.
//
// Variants (bytes honest, incl. per-vector scales):
//   fp16x128  256  anchor (deployable fp16 plane)
//   i8x128    128  anchor, global per-dim max-abs scale, int dot
//                  (reproduces oracle_ceiling_10m.log: 0.917/0.983/1.000)
//   i8x64      64  prefix-64, same int-dot scoring
//   saq96      96  DP bit allocation {0,1,2,4,8} over dims + uniform cb
//   u4x96      48  prefix-96 uniform 4-bit
//   saq48      48  DP allocation
//   u4x64      32  prefix-64 uniform 4-bit
//   u2x128     32  uniform 2-bit all dims
//   u2lm       32  Lloyd-Max 2-bit
//   b1g        16  sign x128, per-dim abs-mean scale (exact 16B)
//   saq16      16  DP allocation
//   u4x128     64  uniform 4-bit all dims
//   u4lm       64  Lloyd-Max 4-bit all dims
// Every LUT variant also reports a +pv column (per-vector length renorm
// alpha = <v,c>/<c,c>, fp16, +2 B/vec) — same scoring pass, two maxes.
//
// Encoding layout: dims grouped by bit width {1,2,4,8}; each group is
// a member-major packed stream per leaf. Uniform-width kernels unpack
// w codes per byte; 1-bit groups use a 256-entry per-byte LUT.
//
// Usage: spike_plane_quant <index.tree> <base.fbin> <query.fbin> <gt.gtmm> [nq]

#include "../src/tree/ivf_tree_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace sextant;
using tree::IVFTreeIndex;
using tree::kInvalidPage;

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<float> load_fbin(const char* path, uint32_t& n, uint32_t& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s failed\n", path); std::exit(1); }
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&d), 4);
    std::vector<float> v(static_cast<size_t>(n) * d);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * 4);
    return v;
}

static bool load_gtmm(const char* path, std::vector<int32_t>& ids, uint32_t& k) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = 0, n = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x4D4D5447) return false;
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&k), 4);
    char metric; f.read(&metric, 1);
    ids.resize(static_cast<size_t>(n) * k);
    std::vector<float> dists(k);
    for (uint32_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(&ids[static_cast<size_t>(i) * k]), k * 4);
        f.read(reinterpret_cast<char*>(dists.data()), k * 4);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Per-dim scalar quantizer. Uniform: lo/step from sample min/max.
// LM: Lloyd-Max centroids trained on a sample histogram. Sign: abs-mean.
// ---------------------------------------------------------------------------
struct DimQuant {
    float lo = 0, step = 1;
    std::vector<float> cent;       // non-empty => LM
    uint32_t bits = 0;             // 0 = skipped dim (SAQ), 1,2,4,8
    float sign_scale = 1;          // bits==1
    uint32_t levels() const { return 1u << bits; }
    float decode(uint32_t code) const {
        if (bits == 1) return (code ? sign_scale : -sign_scale);
        if (!cent.empty())
            return cent[std::min<size_t>(code, cent.size() - 1)];
        return lo + (code + 0.5f) * step;
    }
    uint32_t encode(float v) const {
        if (bits == 1) return v >= 0 ? 1 : 0;
        if (!cent.empty()) {
            uint32_t best = 0; float bd = std::numeric_limits<float>::max();
            for (size_t c = 0; c < cent.size(); ++c) {
                const float d = std::fabs(v - cent[c]);
                if (d < bd) { bd = d; best = static_cast<uint32_t>(c); }
            }
            return best;
        }
        const int L = static_cast<int>(levels());
        const int c = static_cast<int>(
            std::lround((v - lo) / step - 0.5f));
        return static_cast<uint32_t>(std::clamp(c, 0, L - 1));
    }
};

// Encoded plane: dims grouped by bit width, member-major packed streams.
struct EncPlane {
    struct Group {
        uint32_t width;
        std::vector<uint32_t> dims;            // dim ids, ascending
        uint32_t stride;                       // bytes per member
        std::vector<std::vector<uint8_t>> buf; // [leaf]
    };
    std::vector<Group> groups;
    std::vector<std::vector<float>> alpha;     // [leaf][member], pv only
    uint64_t bits_total = 0;                   // honest bit count / vec
};

struct Result {
    std::string name;
    double bytes_per_vec = 0;
    double cont[5] = {0, 0, 0, 0, 0};
    double infl_mean = 0, infl_p95 = 0;
    uint32_t q_scored = 0;
};

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s index.tree base.fbin query.fbin gt.gtmm [nq]\n",
            argv[0]);
        return 1;
    }
    const uint32_t nq = argc > 5 ? std::atoi(argv[5]) : 200;
    const uint32_t P = 128;  // plane rank (PCA-128)
    const double t0 = now_s();
    const double fracs[] = {0.02, 0.05, 0.10, 0.20, 0.30};
    constexpr size_t NF = 5;

    uint32_t nb, db, nqt, dq, gtk_file;
    int base_err = 0;
    const float* base = [](const char* path, uint32_t& n, uint32_t& d,
                           int& err) -> const float* {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) { err = 1; return nullptr; }
        struct stat st;
        if (::fstat(fd, &st) != 0 || st.st_size < 8) {
            ::close(fd); err = 1; return nullptr;
        }
        void* m = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) { err = 1; return nullptr; }
        const auto* hdr = static_cast<const uint32_t*>(m);
        n = hdr[0]; d = hdr[1];
        if (static_cast<uint64_t>(n) * d * 4 + 8 >
            static_cast<uint64_t>(st.st_size)) { err = 1; return nullptr; }
        return reinterpret_cast<const float*>(hdr + 2);
    }(argv[2], nb, db, base_err);
    if (base_err) { std::fprintf(stderr, "base mmap failed\n"); return 1; }
    {   // background sequential prefetch (member pass is random-access)
        const int pfd = ::open(argv[2], O_RDONLY);
        if (pfd >= 0) {
            struct stat pst;
            if (::fstat(pfd, &pst) == 0) {
                const auto psize = static_cast<uint64_t>(pst.st_size);
                std::thread([pfd, psize] {
                    const uint64_t kChunk = 256u << 20;
                    for (uint64_t off = 0; off < psize; off += kChunk) {
                        ::posix_fadvise(pfd, static_cast<off_t>(off),
                                        static_cast<off_t>(kChunk),
                                        POSIX_FADV_WILLNEED);
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(50));
                    }
                    ::close(pfd);
                }).detach();
            } else {
                ::close(pfd);
            }
        }
    }
    auto query = load_fbin(argv[3], nqt, dq);
    std::vector<int32_t> gt;
    if (!load_gtmm(argv[4], gt, gtk_file) || gtk_file < 10) {
        std::fprintf(stderr, "gt load failed / k<10\n"); return 1;
    }
    const uint32_t gt_k = 10;   // containment@10, matches prior runs
    const uint32_t q_used = std::min(nq, nqt);
    if (dq != db) { std::fprintf(stderr, "dim mismatch q/base\n"); return 1; }
    std::printf("base=%u dim=%u queries=%u(used %u) gt_k=%u plane=%u\n",
                nb, db, nqt, q_used, gt_k, P);

    auto index = IVFTreeIndex::open(argv[1]);
    const auto leaf_info = index->debug_leaf_info();
    const uint32_t L = static_cast<uint32_t>(leaf_info.size());

    std::vector<std::vector<RowId>> members(L);
    std::vector<uint64_t> weight(L, 0);
    uint64_t total_w = 0;
    std::unordered_map<int64_t, std::vector<uint32_t>> home;
    for (uint32_t l = 0; l < L; ++l) {
        if (leaf_info[l].page == kInvalidPage) continue;
        members[l] = index->debug_leaf_row_ids(l);
        weight[l] = leaf_info[l].pages;
        total_w += weight[l];
        for (RowId rid : members[l]) home[rid].push_back(l);
    }
    {
        uint64_t found = 0, tot = 0, oob = 0;
        for (uint32_t qi = 0; qi < q_used; ++qi)
            for (uint32_t g = 0; g < gt_k; ++g) {
                const int32_t rid = gt[static_cast<size_t>(qi) * gtk_file + g];
                ++tot;
                if (rid < 0 || rid >= static_cast<int32_t>(nb)) ++oob;
                else if (home.count(rid)) ++found;
            }
        std::printf("[sanity] GT ids found %llu/%llu (oob %llu)\n",
                    static_cast<unsigned long long>(found),
                    static_cast<unsigned long long>(tot),
                    static_cast<unsigned long long>(oob));
        if (found < tot / 2) {
            std::fprintf(stderr, "GT does not resolve — wrong tree/base?\n");
            return 1;
        }
    }
    std::printf("leaves=%u total_pages=%llu\n", L,
                static_cast<unsigned long long>(total_w));

    // --- PCA-128 basis: power iteration + deflation on 20K-row sample
    // covariance (same estimator as spike_routing_ceiling).
    std::vector<std::vector<double>> basis(P);
    std::vector<double> mean(db, 0.0);
    {
        const uint32_t SAMPLE = std::min<uint32_t>(20000, nb);
        // SPIKE_BASIS_SPREAD=1 samples rows evenly across the corpus —
        // clustered-order bases (dbpedia) train on a single cluster if
        // only the prefix is used.
        const bool spread = std::getenv("SPIKE_BASIS_SPREAD") != nullptr;
        const uint32_t stride =
            spread ? std::max(1u, nb / SAMPLE) : 1u;
        auto train_row = [&](uint32_t i) {
            return static_cast<size_t>(i) * stride * db;
        };
        for (uint32_t i = 0; i < SAMPLE; ++i)
            for (uint32_t d = 0; d < db; ++d)
                mean[d] += base[train_row(i) + d] / SAMPLE;
        std::vector<double> cov(static_cast<size_t>(db) * db, 0.0);
        for (uint32_t i = 0; i < SAMPLE; ++i) {
            const float* v = &base[train_row(i)];
            for (uint32_t r = 0; r < db; ++r) {
                const double vr = v[r] - mean[r];
                for (uint32_t c2 = r; c2 < db; ++c2)
                    cov[static_cast<size_t>(r) * db + c2] +=
                        vr * (v[c2] - mean[c2]) / SAMPLE;
            }
        }
        for (uint32_t r = 0; r < db; ++r)
            for (uint32_t c2 = 0; c2 < r; ++c2)
                cov[static_cast<size_t>(r) * db + c2] =
                    cov[static_cast<size_t>(c2) * db + r];
        std::vector<double> resid = cov;
        for (uint32_t e = 0; e < P; ++e) {
            std::vector<double> v(db, 1.0 / std::sqrt((double)db));
            for (int it = 0; it < 40; ++it) {
                std::vector<double> w(db, 0.0);
                for (uint32_t r = 0; r < db; ++r) {
                    double acc = 0;
                    for (uint32_t c2 = 0; c2 < db; ++c2)
                        acc += resid[static_cast<size_t>(r) * db + c2] * v[c2];
                    w[r] = acc;
                }
                double nrm = 1e-30;
                for (uint32_t d = 0; d < db; ++d) nrm += w[d] * w[d];
                nrm = std::sqrt(nrm);
                for (uint32_t d = 0; d < db; ++d) v[d] = w[d] / nrm;
            }
            basis[e] = v;
            double lambda = 0;
            for (uint32_t r = 0; r < db; ++r)
                for (uint32_t c2 = 0; c2 < db; ++c2)
                    lambda += v[r] *
                        resid[static_cast<size_t>(r) * db + c2] * v[c2];
            for (uint32_t r = 0; r < db; ++r)
                for (uint32_t c2 = 0; c2 < db; ++c2)
                    resid[static_cast<size_t>(r) * db + c2] -=
                        lambda * v[r] * v[c2];
        }
        std::fprintf(stderr, "[pca] basis ready (%.1fs)\n", now_s() - t0);
    }

    // --- fp32 plane, leaf-aligned: proj[l][i*P + e].
    // Row-order sequential pass (the 10M-scale lesson from the ceiling
    // spike: leaf-order member passes are random-access page-fault
    // storms). Basis transposed to f32 [d][e] so the inner loop over e
    // is contiguous and autovectorizes.
    std::vector<std::vector<float>> proj(L);
    {
        std::vector<float> mean_f(db);
        for (uint32_t d = 0; d < db; ++d) mean_f[d] = static_cast<float>(mean[d]);
        std::vector<float> basisT(static_cast<size_t>(db) * P);
        for (uint32_t d = 0; d < db; ++d)
            for (uint32_t e = 0; e < P; ++e)
                basisT[static_cast<size_t>(d) * P + e] =
                    static_cast<float>(basis[e][d]);
        // dest entries (leaf, slot) sorted by row: sequential base reads.
        std::vector<uint32_t> dest_leaf, dest_slot;
        {
            std::vector<uint32_t> fill(L, 0);
            size_t total = 0;
            for (uint32_t l = 0; l < L; ++l)
                total += members[l].size();
            dest_leaf.reserve(total);
            dest_slot.reserve(total);
            for (uint32_t r = 0; r < nb; ++r) {
                auto it = home.find(static_cast<int64_t>(r));
                if (it == home.end()) continue;
                for (uint32_t l : it->second) {
                    dest_leaf.push_back(l);
                    dest_slot.push_back(fill[l]++);
                }
            }
        }
        for (uint32_t l = 0; l < L; ++l) proj[l].resize(members[l].size() * P);
        const uint32_t* dl = dest_leaf.data();
        const uint32_t* ds = dest_slot.data();
        const size_t DN = dest_leaf.size();
        // dest entries were pushed in row order; store rows explicitly
        // (32-bit ids, ~40MB at 10M) so the parallel pass has row ids.
        std::vector<uint32_t> dest_row(DN);
        {
            size_t w = 0;
            for (uint32_t r = 0; r < nb; ++r) {
                auto it = home.find(static_cast<int64_t>(r));
                if (it == home.end()) continue;
                for (size_t k = 0; k < it->second.size(); ++k)
                    dest_row[w++] = r;
            }
        }
        const uint32_t* dr = dest_row.data();
#pragma omp parallel for schedule(static)
        for (int64_t b0 = 0; b0 < static_cast<int64_t>(DN); ++b0) {
            const float* v = &base[static_cast<size_t>(dr[b0]) * db];
            float* out = &proj[dl[b0]][static_cast<size_t>(ds[b0]) * P];
            float acc[P];
            for (uint32_t e = 0; e < P; ++e) acc[e] = 0;
            for (uint32_t d = 0; d < db; ++d) {
                const float vd = v[d] - mean_f[d];
                const float* brow = &basisT[static_cast<size_t>(d) * P];
                for (uint32_t e = 0; e < P; ++e) acc[e] += vd * brow[e];
            }
            for (uint32_t e = 0; e < P; ++e) out[e] = acc[e];
        }
    }
    std::fprintf(stderr, "[pca] fp32 plane ready (%.1fs)\n", now_s() - t0);

    std::vector<float> qproj(static_cast<size_t>(q_used) * P);
    for (uint32_t qi = 0; qi < q_used; ++qi) {
        const float* q = &query[static_cast<size_t>(qi) * db];
        for (uint32_t e = 0; e < P; ++e) {
            const auto& b = basis[e];
            double acc = 0;
            for (uint32_t d = 0; d < db; ++d) acc += (q[d] - mean[d]) * b[d];
            qproj[static_cast<size_t>(qi) * P + e] = static_cast<float>(acc);
        }
    }

    // --- codebook training sample: spread across leaves/members.
    std::vector<float> samp;
    {
        const size_t TARGET = 200000;
        for (uint32_t l = 0; l < L && samp.size() / P < TARGET;
             l += std::max(1u, L / 64)) {
            const auto& pm = proj[l];
            const size_t cnt = pm.size() / P;
            if (!cnt) continue;
            const size_t take = std::min(cnt, TARGET / 64 + 1);
            for (size_t i = 0; i < cnt && samp.size() / P < TARGET;
                 i += std::max<size_t>(1, cnt / take)) {
                for (uint32_t e = 0; e < P; ++e)
                    samp.push_back(pm[i * P + e]);
            }
        }
        std::fprintf(stderr, "[train] codebook sample=%zu rows (%.1fs)\n",
                     samp.size() / P, now_s() - t0);
    }
    const size_t train_rows = samp.size() / P;
    if (!train_rows) { std::fprintf(stderr, "empty train sample\n"); return 1; }

    std::vector<float> vmin(P, std::numeric_limits<float>::max());
    std::vector<float> vmax(P, std::numeric_limits<float>::lowest());
    std::vector<double> vmean_abs(P, 0.0), var(P, 0.0);
    for (size_t i = 0; i < train_rows; ++i)
        for (uint32_t e = 0; e < P; ++e) {
            const float v = samp[i * P + e];
            vmin[e] = std::min(vmin[e], v);
            vmax[e] = std::max(vmax[e], v);
            vmean_abs[e] += std::fabs(v);
        }
    for (uint32_t e = 0; e < P; ++e) vmean_abs[e] /= train_rows;
    for (size_t i = 0; i < train_rows; ++i)
        for (uint32_t e = 0; e < P; ++e) {
            const double d = samp[i * P + e];
            var[e] += d * d / train_rows;  // plane is mean-centered
        }

    auto train_uniform = [&](uint32_t bits, uint32_t dims) {
        std::vector<DimQuant> q(P);
        for (uint32_t e = 0; e < P; ++e) {
            q[e].bits = e < dims ? bits : 0;
            if (e < dims) {
                q[e].lo = vmin[e];
                q[e].step = (vmax[e] - vmin[e]) / q[e].levels();
            }
        }
        return q;
    };
    auto train_lm = [&](uint32_t bits) {
        std::vector<DimQuant> q(P);
        const uint32_t LV = 1u << bits;
        constexpr int HB = 512;
        std::vector<float> hist(static_cast<size_t>(HB) * P, 0.0f);
        for (uint32_t e = 0; e < P; ++e) {
            const float lo = vmin[e], hi = vmax[e];
            for (size_t i = 0; i < train_rows; ++i) {
                const float t = (samp[i * P + e] - lo) /
                                std::max(1e-9f, hi - lo);
                const int b = static_cast<int>(t * HB);
                hist[static_cast<size_t>(e) * HB +
                     std::clamp(b, 0, HB - 1)] += 1.0f;
            }
        }
#pragma omp parallel for schedule(dynamic)
        for (int e = 0; e < static_cast<int>(P); ++e) {
            auto& dq = q[static_cast<uint32_t>(e)];
            dq.bits = bits;
            dq.cent.assign(LV, 0.0f);
            const float lo = vmin[static_cast<uint32_t>(e)];
            const float hi = vmax[static_cast<uint32_t>(e)];
            for (uint32_t c = 0; c < LV; ++c)
                dq.cent[c] = lo + (c + 0.5f) * (hi - lo) / LV;
            for (int it = 0; it < 50; ++it) {
                std::vector<double> cs(LV, 0.0), cw(LV, 0.0);
                for (int b = 0; b < HB; ++b) {
                    const float w =
                        hist[static_cast<size_t>(e) * HB + b];
                    if (w <= 0) continue;
                    const float x = lo + (b + 0.5f) * (hi - lo) / HB;
                    uint32_t best = 0;
                    float bd = std::numeric_limits<float>::max();
                    for (uint32_t c = 0; c < LV; ++c) {
                        const float d = std::fabs(x - dq.cent[c]);
                        if (d < bd) { bd = d; best = c; }
                    }
                    cs[best] += w * x; cw[best] += w;
                }
                for (uint32_t c = 0; c < LV; ++c)
                    if (cw[c] > 0)
                        dq.cent[c] = static_cast<float>(cs[c] / cw[c]);
            }
        }
        return q;
    };
    auto train_sign = [&]() {
        std::vector<DimQuant> q(P);
        for (uint32_t e = 0; e < P; ++e) {
            q[e].bits = 1;
            q[e].sign_scale = static_cast<float>(vmean_abs[e]);
        }
        return q;
    };
    // SAQ: DP segmented allocation, bits in {0,1,2,4,8}, segments of >=8
    // contiguous dims (deployable progressive-eval granularity),
    // distortion var_e * D(b) with Gaussian Lloyd-Max ratios.
    auto train_saq = [&](uint32_t budget_bytes) {
        const uint32_t BOPTS[] = {0, 1, 2, 4, 8};
        constexpr double DBOPT[5] = {1.0, 0.3634, 0.1175, 0.0149, 0.00022};
        const int B = static_cast<int>(budget_bytes);
        const double INF = 1e300;
        std::vector<std::vector<double>> dp(
            P + 1, std::vector<double>(B + 1, INF));
        std::vector<std::vector<uint8_t>> clen(P + 1,
            std::vector<uint8_t>(B + 1, 0));
        std::vector<std::vector<uint8_t>> cbits(P + 1,
            std::vector<uint8_t>(B + 1, 0));
        for (int b = 0; b <= B; ++b) dp[P][b] = 0;
        for (int i = static_cast<int>(P) - 1; i >= 0; --i)
            for (int b = 0; b <= B; ++b) {
                double best = INF;
                for (int len = 8; i + len <= static_cast<int>(P); len += 8)
                    for (int k = 0; k < 5; ++k) {
                        const int bytes = (len * BOPTS[k] + 7) / 8;
                        if (bytes > b) continue;
                        double d = 0;
                        for (int e = i; e < i + len; ++e)
                            d += var[static_cast<uint32_t>(e)] * DBOPT[k];
                        const double cand = d + dp[i + len][b - bytes];
                        if (cand < best) {
                            best = cand;
                            clen[i][b] = static_cast<uint8_t>(len);
                            cbits[i][b] = static_cast<uint8_t>(k);
                        }
                    }
                dp[i][b] = best;
            }
        std::vector<DimQuant> q(P);
        {
            int i = 0, b = B;
            while (i < static_cast<int>(P)) {
                int len = clen[i][b];
                int k = cbits[i][b];
                if (len == 0) {  // tail shorter than 8 dims
                    for (int kk = 4; kk >= 0; --kk) {
                        const int bytes =
                            ((static_cast<int>(P) - i) * BOPTS[kk] + 7) / 8;
                        if (bytes <= b) { k = kk; break; }
                    }
                    len = static_cast<int>(P) - i;
                }
                for (int e = i; e < i + len; ++e) {
                    auto& dq = q[static_cast<uint32_t>(e)];
                    dq.bits = BOPTS[k];
                    if (dq.bits >= 2) {
                        dq.lo = vmin[static_cast<uint32_t>(e)];
                        dq.step = (vmax[static_cast<uint32_t>(e)] -
                                   vmin[static_cast<uint32_t>(e)]) /
                                  dq.levels();
                    } else if (dq.bits == 1) {
                        dq.sign_scale =
                            static_cast<float>(vmean_abs[static_cast<uint32_t>(e)]);
                    }
                }
                b -= (len * BOPTS[k] + 7) / 8;
                i += len;
            }
        }
        return q;
    };

    // --- GT leaf sets per query (per neighbor, as the engine routes).
    std::vector<std::vector<uint32_t>> gt_sets(q_used);
    std::vector<std::vector<uint32_t>> gt_offs(q_used);
    std::vector<std::vector<char>> gt_mark(q_used);
    std::vector<uint32_t> q_list;
    for (uint32_t qi = 0; qi < q_used; ++qi) {
        std::vector<uint32_t>& sets = gt_sets[qi];
        std::vector<uint32_t>& offs = gt_offs[qi] = {0};
        for (uint32_t g = 0; g < gt_k; ++g) {
            const int32_t rid = gt[static_cast<size_t>(qi) * gtk_file + g];
            auto it = home.find(rid);
            if (it != home.end())
                for (uint32_t l : it->second) sets.push_back(l);
            offs.push_back(static_cast<uint32_t>(sets.size()));
        }
        if (sets.empty()) continue;
        gt_mark[qi].assign(L, 0);
        for (uint32_t l : sets) gt_mark[qi][l] = 1;
        q_list.push_back(qi);
    }
    std::fprintf(stderr, "[gt] %zu/%u queries scoreable\n",
                 q_list.size(), q_used);

    // ----- truth: fp32 plane max per (q, l) -----
    std::vector<float> max_true(static_cast<size_t>(q_used) * L,
        -std::numeric_limits<float>::max());
#pragma omp parallel for schedule(dynamic)
    for (int64_t qii = 0; qii < static_cast<int64_t>(q_list.size()); ++qii) {
        const uint32_t qi = q_list[static_cast<size_t>(qii)];
        const float* qp = &qproj[static_cast<size_t>(qi) * P];
        float* mt = &max_true[static_cast<size_t>(qi) * L];
        for (uint32_t l = 0; l < L; ++l) {
            const auto& pm = proj[l];
            const size_t cnt = pm.size() / P;
            float best = -std::numeric_limits<float>::max();
            for (size_t i = 0; i < cnt; ++i) {
                float ip = 0;
                for (uint32_t e = 0; e < P; ++e)
                    ip += qp[e] * pm[i * P + e];
                best = std::max(best, ip);
            }
            mt[l] = best;
        }
    }
    std::fprintf(stderr, "[truth] fp32 max pass done (%.1fs)\n", now_s() - t0);

    // containment walk (same contract as the ceiling spike)
    auto walk = [&](const std::vector<float>& leaf_max, uint32_t qi,
                    Result& r) {
        const auto& sets = gt_sets[qi];
        const auto& offs = gt_offs[qi];
        const auto& is_gt = gt_mark[qi];
        const uint32_t n_nb = static_cast<uint32_t>(offs.size() - 1);
        std::vector<std::pair<float, uint32_t>> so(L);
        for (uint32_t l = 0; l < L; ++l) so[l] = {-leaf_max[l], l};
        std::sort(so.begin(), so.end());
        std::vector<uint8_t> done(n_nb, 0);
        uint64_t cum = 0, covered = 0;
        size_t fi = 0;
        for (uint32_t pos = 0; pos < L; ++pos) {
            const uint32_t l = so[pos].second;
            cum += weight[l];
            if (is_gt[l])
                for (uint32_t i = 0; i < n_nb; ++i) {
                    if (done[i]) continue;
                    for (uint32_t j = offs[i]; j < offs[i + 1]; ++j)
                        if (sets[j] == l) { done[i] = 1; ++covered; break; }
                }
            while (fi < NF &&
                   static_cast<double>(cum) >= fracs[fi] * total_w - 1e-9) {
                r.cont[fi] += static_cast<double>(covered) / n_nb;
                ++fi;
            }
        }
        while (fi < NF) { r.cont[fi] += 1.0; ++fi; }
        ++r.q_scored;
    };

    // ----- encode / score a DimQuant set -----
    auto encode = [&](const std::vector<DimQuant>& q, bool with_pv) {
        EncPlane pl;
        for (uint32_t w : {1u, 2u, 4u, 8u}) {
            EncPlane::Group g;
            g.width = w;
            for (uint32_t e = 0; e < P; ++e)
                if (q[e].bits == w) g.dims.push_back(e);
            if (g.dims.empty()) continue;
            g.stride = static_cast<uint32_t>(
                (static_cast<uint64_t>(g.dims.size()) * w + 7) / 8);
            g.buf.resize(L);
            pl.bits_total +=
                static_cast<uint64_t>(g.dims.size()) * w;
            pl.groups.push_back(std::move(g));
        }
        if (with_pv) pl.alpha.resize(L);
#pragma omp parallel for schedule(dynamic)
        for (int64_t l = 0; l < static_cast<int64_t>(L); ++l) {
            const auto& pm = proj[static_cast<uint32_t>(l)];
            const size_t cnt = pm.size() / P;
            if (!cnt) continue;
            std::vector<float> al;
            if (with_pv) al.assign(cnt, 1.0f);
            for (auto& g : pl.groups) {
                auto& buf = g.buf[static_cast<uint32_t>(l)];
                buf.assign(cnt * g.stride, 0);
            }
            for (size_t i = 0; i < cnt; ++i) {
                double dot_vc = 0, dot_cc = 0;
                for (auto& g : pl.groups) {
                    uint8_t* out =
                        &g.buf[static_cast<uint32_t>(l)][i * g.stride];
                    for (size_t di = 0; di < g.dims.size(); ++di) {
                        const uint32_t e = g.dims[di];
                        const float v = pm[i * P + e];
                        const uint32_t c = q[e].encode(v);
                        const uint64_t bo =
                            static_cast<uint64_t>(di) * g.width;
                        for (uint32_t b = 0; b < g.width; ++b)
                            if (c & (1u << b))
                                out[static_cast<size_t>((bo + b) >> 3)] |=
                                    static_cast<uint8_t>(1u << ((bo + b) & 7));
                        if (with_pv) {
                            const float dc = q[e].decode(c);
                            dot_vc += static_cast<double>(v) * dc;
                            dot_cc += static_cast<double>(dc) * dc;
                        }
                    }
                }
                if (with_pv && dot_cc > 1e-12)
                    al[i] = static_cast<float>(dot_vc / dot_cc);
            }
            if (with_pv) pl.alpha[static_cast<uint32_t>(l)] = std::move(al);
        }
        return pl;
    };

    std::vector<Result> results;
    auto run_variant = [&](const char* name, std::vector<DimQuant> q,
                           bool pv) {
        const double ts = now_s();
        EncPlane pl = encode(q, pv);
        Result r_plain, r_pv;
        r_plain.name = name;
        r_plain.bytes_per_vec =
            static_cast<double>((pl.bits_total + 7) / 8);
        r_pv.name = std::string(name) + "+pv";
        r_pv.bytes_per_vec =
            static_cast<double>((pl.bits_total + 7) / 8) + 2;
        std::vector<double> d_plain, d_pv;
        // per-thread LUT staging
#pragma omp parallel
        {
            std::vector<float> lut(static_cast<size_t>(P) * 256);
            std::vector<float> blut;  // 1-bit group: [bytepos*256+byte]
            std::vector<float> lmax(L), lmax_pv(L);
            std::vector<double> td_plain, td_pv;
#pragma omp for schedule(dynamic)
            for (int64_t qii = 0;
                 qii < static_cast<int64_t>(q_list.size()); ++qii) {
                const uint32_t qi = q_list[static_cast<size_t>(qii)];
                const float* qp = &qproj[static_cast<size_t>(qi) * P];
                for (auto& g : pl.groups)
                    for (size_t di = 0; di < g.dims.size(); ++di) {
                        const uint32_t e = g.dims[di];
                        for (uint32_t c = 0; c < q[e].levels(); ++c)
                            lut[static_cast<size_t>(e) * 256 + c] =
                                qp[e] * q[e].decode(c);
                    }
                for (auto& g : pl.groups) {
                    if (g.width != 1) continue;
                    const uint32_t nb = g.stride;
                    blut.assign(static_cast<size_t>(nb) * 256, 0.0f);
                    for (uint32_t bp = 0; bp < nb; ++bp)
                        for (int bv = 0; bv < 256; ++bv) {
                            float s = 0;
                            for (uint32_t t = 0; t < 8; ++t) {
                                const size_t di =
                                    static_cast<size_t>(bp) * 8 + t;
                                if (di >= g.dims.size()) break;
                                const uint32_t e = g.dims[di];
                                s += lut[static_cast<size_t>(e) * 256 +
                                         ((bv >> t) & 1)];
                            }
                            blut[static_cast<size_t>(bp) * 256 + bv] = s;
                        }
                }
                for (uint32_t l = 0; l < L; ++l) {
                    float best = -std::numeric_limits<float>::max();
                    float best_pv = best;
                    const bool has_alpha = !pl.alpha.empty();
                    const float* al =
                        has_alpha && !pl.alpha[l].empty()
                            ? pl.alpha[l].data() : nullptr;
                    size_t cnt = 0;
                    for (auto& g : pl.groups) {
                        const size_t c =
                            g.buf[l].size() /
                            std::max<uint32_t>(1, g.stride);
                        cnt = std::max(cnt, c);
                    }
                    for (size_t i = 0; i < cnt; ++i) {
                        float s = 0;
                        for (auto& g : pl.groups) {
                            const uint8_t* buf =
                                &g.buf[l][i * g.stride];
                            if (g.width == 1) {
                                for (uint32_t bp = 0; bp < g.stride; ++bp)
                                    s += blut[static_cast<size_t>(bp) * 256 +
                                              buf[bp]];
                            } else if (g.width == 8) {
                                for (size_t di = 0; di < g.dims.size(); ++di)
                                    s += lut[
                                        static_cast<size_t>(g.dims[di]) * 256 +
                                        buf[di]];
                            } else if (g.width == 4) {
                                // pack order: dim di at bit offset 4*di,
                                // so dims[2bp] lands in the LOW nibble
                                for (size_t bp = 0; bp < g.stride; ++bp) {
                                    const uint8_t byte = buf[bp];
                                    s += lut[static_cast<size_t>(g.dims[2 * bp]) * 256 +
                                             (byte & 0xF)];
                                    s += lut[static_cast<size_t>(g.dims[2 * bp + 1]) * 256 +
                                             ((byte >> 4) & 0xF)];
                                }
                            } else {  // width 2
                                for (size_t bp = 0; bp < g.stride; ++bp) {
                                    const uint8_t byte = buf[bp];
                                    for (uint32_t t = 0; t < 4; ++t)
                                        s += lut[
                                            static_cast<size_t>(
                                                g.dims[4 * bp + t]) * 256 +
                                            ((byte >> (2 * t)) & 3)];
                                }
                            }
                        }
                        best = std::max(best, s);
                        if (al) best_pv = std::max(best_pv, s * al[i]);
                    }
                    lmax[l] = best;
                    lmax_pv[l] = best_pv;
                    const float mt =
                        max_true[static_cast<size_t>(qi) * L + l];
                    if (mt > -std::numeric_limits<float>::max() / 2)
                        td_plain.push_back(best - mt);
                    if (al && mt > -std::numeric_limits<float>::max() / 2)
                        td_pv.push_back(best_pv - mt);
                }
#pragma omp critical
                {
                    walk(lmax, qi, r_plain);
                    if (pv) walk(lmax_pv, qi, r_pv);
                }
                // keep per-thread deltas
                if (td_plain.size() > (1u << 20)) {
#pragma omp critical
                    {
                        d_plain.insert(d_plain.end(), td_plain.begin(),
                                       td_plain.end());
                        td_plain.clear();
                    }
                }
                if (td_pv.size() > (1u << 20)) {
#pragma omp critical
                    {
                        d_pv.insert(d_pv.end(), td_pv.begin(), td_pv.end());
                        td_pv.clear();
                    }
                }
            }
#pragma omp critical
            {
                d_plain.insert(d_plain.end(), td_plain.begin(), td_plain.end());
                d_pv.insert(d_pv.end(), td_pv.begin(), td_pv.end());
            }
        }
        auto finalize = [](Result& r, std::vector<double>& d) {
            if (d.empty()) return;
            std::sort(d.begin(), d.end());
            double s = 0;
            for (double x : d) s += x;
            r.infl_mean = s / d.size();
            r.infl_p95 = d[static_cast<size_t>(
                0.95 * static_cast<double>(d.size() - 1))];
        };
        finalize(r_plain, d_plain);
        finalize(r_pv, d_pv);
        const double bplain = r_plain.bytes_per_vec;
        const double bpv = r_pv.bytes_per_vec;
        results.push_back(std::move(r_plain));
        if (pv) results.push_back(std::move(r_pv));
        std::fprintf(stderr, "[var] %-10s %4.0f B/vec (+pv %.0f) in %.1fs\n",
                     name, bplain, pv ? bpv : 0.0, now_s() - ts);
    };

    run_variant("u4x128", train_uniform(4, 128), true);
    run_variant("u4lm", train_lm(4), true);
    run_variant("u4x96", train_uniform(4, 96), true);
    run_variant("u4x64", train_uniform(4, 64), true);
    run_variant("u2x128", train_uniform(2, 128), true);
    run_variant("u2lm", train_lm(2), true);
    run_variant("b1g", train_sign(), true);
    run_variant("saq16", train_saq(16), true);
    run_variant("saq48", train_saq(48), true);
    run_variant("saq96", train_saq(96), true);

    // ----- fp16x128 anchor: fp16 plane, fp32 query -----
    {
        const double ts = now_s();
        Result r;
        r.name = "fp16x128";
        r.bytes_per_vec = 256;
        std::vector<double> d;
#pragma omp parallel
        {
            std::vector<float> lmax(L);
            std::vector<double> td;
#pragma omp for schedule(dynamic)
            for (int64_t qii = 0;
                 qii < static_cast<int64_t>(q_list.size()); ++qii) {
                const uint32_t qi = q_list[static_cast<size_t>(qii)];
                const float* qp = &qproj[static_cast<size_t>(qi) * P];
                float qph[P];
                for (uint32_t e = 0; e < P; ++e)
                    qph[e] = static_cast<float>(
                        static_cast<_Float16>(qp[e]));
                for (uint32_t l = 0; l < L; ++l) {
                    const auto& pm = proj[l];
                    const size_t cnt = pm.size() / P;
                    float best = -std::numeric_limits<float>::max();
                    for (size_t i = 0; i < cnt; ++i) {
                        float ip = 0;
                        for (uint32_t e = 0; e < P; ++e)
                            ip += qph[e] * static_cast<float>(
                                static_cast<_Float16>(pm[i * P + e]));
                        best = std::max(best, ip);
                    }
                    lmax[l] = best;
                    const float mt =
                        max_true[static_cast<size_t>(qi) * L + l];
                    if (mt > -std::numeric_limits<float>::max() / 2)
                        td.push_back(best - mt);
                }
#pragma omp critical
                walk(lmax, qi, r);
            }
#pragma omp critical
            d.insert(d.end(), td.begin(), td.end());
        }
        if (!d.empty()) {
            std::sort(d.begin(), d.end());
            double s = 0;
            for (double x : d) s += x;
            r.infl_mean = s / d.size();
            r.infl_p95 = d[static_cast<size_t>(
                0.95 * static_cast<double>(d.size() - 1))];
        }
        results.push_back(std::move(r));
        std::fprintf(stderr, "[var] fp16x128 in %.1fs\n", now_s() - ts);
    }

    // ----- i8 anchors (integer dot, replicates the ceiling spike) -----
    {
        std::vector<float> i8_scale(P);
        for (uint32_t e = 0; e < P; ++e)
            i8_scale[e] = 127.0f / std::max(1e-9f,
                std::max(std::abs(vmin[e]), std::abs(vmax[e])));
        for (uint32_t dims : {128u, 64u}) {
            const double ts = now_s();
            std::vector<std::vector<int8_t>> plane(L);
#pragma omp parallel for schedule(dynamic)
            for (int64_t l = 0; l < static_cast<int64_t>(L); ++l) {
                const auto& pm = proj[static_cast<uint32_t>(l)];
                const size_t cnt = pm.size() / P;
                if (!cnt) continue;
                auto& st = plane[static_cast<uint32_t>(l)];
                st.resize(cnt * dims);
                for (size_t i = 0; i < cnt; ++i)
                    for (uint32_t e = 0; e < dims; ++e) {
                        const int v = static_cast<int>(std::lround(
                            pm[i * P + e] * i8_scale[e]));
                        st[i * dims + e] = static_cast<int8_t>(
                            std::clamp(v, -127, 127));
                    }
            }
            Result r;
            r.name = dims == 128 ? "i8x128" : "i8x64";
            r.bytes_per_vec = dims;
#pragma omp parallel
            {
                std::vector<float> lmax(L);
#pragma omp for schedule(dynamic)
                for (int64_t qii = 0;
                     qii < static_cast<int64_t>(q_list.size()); ++qii) {
                    const uint32_t qi = q_list[static_cast<size_t>(qii)];
                    std::vector<int8_t> qpi(dims);
                    for (uint32_t e = 0; e < dims; ++e) {
                        const int v = static_cast<int>(std::lround(
                            qproj[static_cast<size_t>(qi) * P + e] *
                            i8_scale[e]));
                        qpi[e] = static_cast<int8_t>(
                            std::clamp(v, -127, 127));
                    }
                    for (uint32_t l = 0; l < L; ++l) {
                        const auto& pm = plane[l];
                        const size_t cnt = pm.size() / dims;
                        int32_t best = -0x7fffffff;
                        for (size_t i = 0; i < cnt; ++i) {
                            int32_t acc = 0;
                            for (uint32_t e = 0; e < dims; ++e)
                                acc += qpi[e] * pm[i * dims + e];
                            best = std::max(best, acc);
                        }
                        lmax[l] = static_cast<float>(best);
                    }
#pragma omp critical
                    walk(lmax, qi, r);
                }
            }
            results.push_back(std::move(r));
            std::fprintf(stderr, "[var] i8x%u in %.1fs\n", dims,
                         now_s() - ts);
        }
    }

    // ----- report -----
    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) {
                  return a.bytes_per_vec < b.bytes_per_vec;
              });
    std::printf("\n== plane quantization: containment@%u vs page-weighted "
                "coverage, %u queries, %u leaves ==\n",
                gt_k, q_used, L);
    std::printf("%-10s %5s  %-7s %-7s %-7s %-7s %-7s  %10s %10s\n",
                "variant", "B/vec", "c@.02", "c@.05", "c@.10", "c@.20",
                "c@.30", "infl_mean", "infl_p95");
    for (const auto& r : results) {
        std::printf("%-10s %5.0f", r.name.c_str(), r.bytes_per_vec);
        for (size_t fi = 0; fi < NF; ++fi)
            std::printf(" %.4f", r.q_scored
                ? r.cont[fi] / r.q_scored : 0.0);
        std::printf("  %10.4f %10.4f  (q=%u)\n",
                    r.infl_mean, r.infl_p95, r.q_scored);
    }
    std::printf("\ntotal %.1fs\n", now_s() - t0);
    return 0;
}
