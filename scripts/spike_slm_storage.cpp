// Probe: verify that stored scalar_lloydmax codes decode back to their
// row_id's original vector. Isolates code storage + levels round-trip from
// the scan math. Reads the tree file directly (superblock → leaf table →
// leaves), decodes codes with the deserialized quantizer, and measures the
// distance between decode(stored_code) and base[row_id].
//
// Usage: spike_slm_storage <tree> <base.fbin> [n_check]

#include "../src/tree/superblock.hpp"
#include "../src/tree/tree_nodes.hpp"
#include "../src/tree/tree_manifest.hpp"
#include "../src/quant/scalar_lloydmax_quantizer.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <random>
#include <string>

using namespace sextant;
using namespace sextant::tree;

static std::vector<float> load_fbin(const char* path, uint32_t& n, uint32_t& d) {
    std::ifstream f(path, std::ios::binary);
    uint32_t hdr[2];
    f.read(reinterpret_cast<char*>(hdr), 8);
    n = hdr[0]; d = hdr[1];
    std::vector<float> v(static_cast<size_t>(n) * d);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(v.size()) * 4);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tree> <base.fbin> [n_check]\n",
                     argv[0]);
        return 1;
    }
    const uint32_t n_check = argc > 3 ? std::atoi(argv[3]) : 500;

    // Read the whole tree file.
    std::ifstream tf(argv[1], std::ios::binary);
    tf.seekg(0, std::ios::end);
    const size_t fsize = tf.tellg();
    tf.seekg(0);
    std::vector<uint8_t> filebuf(fsize);
    tf.read(reinterpret_cast<char*>(filebuf.data()),
            static_cast<std::streamsize>(fsize));

    // Superblock: active copy at page 0, shadow at page 1 — pick whichever
    // has valid magic (highest commit_seq), like Superblock::load.
    SuperblockDisk sb_a{}, sb_b{};
    std::memcpy(&sb_a, filebuf.data(), sizeof(sb_a));
    std::memcpy(&sb_b, filebuf.data() + kPageSize, sizeof(sb_b));
    SuperblockDisk sb;
    if (sb_a.magic == kSuperblockMagic && sb_b.magic == kSuperblockMagic)
        sb = (sb_a.commit_seq >= sb_b.commit_seq) ? sb_a : sb_b;
    else if (sb_a.magic == kSuperblockMagic) sb = sb_a;
    else if (sb_b.magic == kSuperblockMagic) sb = sb_b;
    else { std::fprintf(stderr, "bad superblock magic\n"); return 1; }

    // Quantizer blob.
    const uint8_t* cb = filebuf.data() +
        static_cast<size_t>(sb.codebook_page) * kPageSize;
    uint64_t qbsz = 0;
    std::memcpy(&qbsz, cb, 8);
    ScalarLloydMaxQuantizer quant(MetricKind::L2Sq, 768, 4);
    quant.deserialize(cb + 8, static_cast<size_t>(qbsz));

    // Manifest (TOML config blob) → summary_size.
    const char* cfg = reinterpret_cast<const char*>(
        filebuf.data() + static_cast<size_t>(sb.config_page) * kPageSize);
    const TreeManifest manifest = manifest_from_toml(
        std::string(cfg, strnlen(cfg,
            static_cast<size_t>(sb.config_pages) * kPageSize)));
    std::printf("manifest: dim=%u m4=%u bits=%u summary_size=%u depth=%u\n",
                manifest.dim, manifest.m4, manifest.scan_pq_bits,
                manifest.summary_size, manifest.depth);

    // Leaf table.
    const uint8_t* lt = filebuf.data() +
        static_cast<size_t>(sb.leaf_table_page) * kPageSize;
    uint64_t n_leaves = 0;
    std::memcpy(&n_leaves, lt, 8);
    std::vector<LeafTableEntry> leaves(n_leaves);
    std::memcpy(leaves.data(), lt + 8,
                static_cast<size_t>(n_leaves) * sizeof(LeafTableEntry));
    std::printf("leaves: %llu\n",
                static_cast<unsigned long long>(n_leaves));

    // Base vectors.
    uint32_t nb, dim;
    auto base = load_fbin(argv[2], nb, dim);
    std::printf("base: %u x %u\n", nb, dim);

    // For sampled leaves/entries: decode(stored code) vs base[row_id].
    const uint32_t cs = quant.code_size();
    const uint32_t K = quant.K();
    (void)K;
    std::vector<float> dec(dim);
    double err_sum = 0, err_max = 0;
    uint32_t n_done = 0, n_missing = 0;
    std::mt19937 rng(1);
    for (uint32_t li = 0; li < n_leaves && n_done < n_check; ++li) {
        const uint8_t* leaf = filebuf.data() +
            static_cast<size_t>(leaves[li].page) * kPageSize;
        const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
        const uint32_t count = lh->count;
        const uint8_t* codes = leaf +
            leaf_codes_offset(manifest.summary_size);
        const RowId* row_ids = reinterpret_cast<const RowId*>(
            codes + static_cast<uint64_t>(count) * cs);
        for (uint32_t j = 0; j < count && n_done < n_check;
             j += std::max(1u, count / 16), ++n_done) {
            const RowId rid = row_ids[j];
            if (rid < 0 || static_cast<uint32_t>(rid) >= nb) {
                ++n_missing;
                continue;
            }
            quant.decode(codes + static_cast<uint64_t>(j) * cs, dec.data());
            const float* orig = base.data() +
                static_cast<size_t>(rid) * dim;
            double e = 0;
            for (uint32_t d = 0; d < dim; ++d) {
                const double diff = orig[d] - dec[d];
                e += diff * diff;
            }
            err_sum += e;
            err_max = std::max(err_max, e);
        }
    }
    std::printf("checked=%u missing_rid=%u\n", n_done, n_missing);
    std::printf("decode-vs-base err: mean=%.6f max=%.6f (expect mean~0.004)\n",
                err_sum / std::max(1u, n_done), err_max);

    // Membership check: are specific row_ids present in ANY leaf?
    // (args: row ids to check, comma-separated, via argv[4])
    if (argc > 4) {
        std::vector<RowId> want;
        {
            std::string s = argv[4];
            size_t pos = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                want.push_back(std::atoll(s.substr(0, pos).c_str()));
                s.erase(0, pos + 1);
            }
            want.push_back(std::atoll(s.c_str()));
        }
        std::vector<int> found(want.size(), 0);
        std::vector<double> want_err(want.size(), -1.0);
        for (uint32_t li = 0; li < n_leaves; ++li) {
            const uint8_t* leaf = filebuf.data() +
                static_cast<size_t>(leaves[li].page) * kPageSize;
            const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf);
            const uint32_t count = lh->count;
            const uint8_t* codes = leaf +
                leaf_codes_offset(manifest.summary_size);
            const RowId* row_ids = reinterpret_cast<const RowId*>(
                codes + static_cast<uint64_t>(count) * cs);
            for (uint32_t j = 0; j < count; ++j) {
                for (size_t w = 0; w < want.size(); ++w) {
                    if (row_ids[j] != want[w]) continue;
                    ++found[w];
                    quant.decode(codes + static_cast<uint64_t>(j) * cs,
                                 dec.data());
                    const float* orig = base.data() +
                        static_cast<size_t>(want[w]) * dim;
                    double e = 0;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const double diff = orig[d] - dec[d];
                        e += diff * diff;
                    }
                    want_err[w] = e;
                }
            }
        }
        for (size_t w = 0; w < want.size(); ++w)
            std::printf("row %lld: in %d leaves, decode err %.6f\n",
                        static_cast<long long>(want[w]), found[w],
                        want_err[w]);
    }
    return 0;
}
