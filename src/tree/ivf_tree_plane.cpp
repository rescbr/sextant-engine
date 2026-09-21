#include "ivf_tree_index.hpp"

#include "engine/manifest_io.hpp"
#include "engine/probe.hpp"
#include "engine/partition.hpp"
#include "util/fp16.hpp"
#include "simd_kernels.hpp"
#include "tree/filter_column_write.hpp"  // filter column write path
#include "tree/filter_column_read.hpp"   // filter column read path (mutable ops)
#include "tree/filter_scan.hpp"         // filter predicate evaluation
#include "tree/coders/coder_factory.hpp"
#include "tree/coders/global_pq_coder.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"
#include "sextant/engine_trace.hpp"
#include "sextant/vector_source.hpp"
#include "sextant/filter_column_data.hpp"
#include "sextant/phase_timer.hpp"

#include <spdlog/spdlog.h>

#include <ctpl/ctpl_stl_tls.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <numeric>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>


namespace sextant::tree {

// ===========================================================================
// Routing diagnostics — loss-decomposition harness support.
// ===========================================================================

// ===========================================================================
// Routing plane attach (plane.hpp). Post-build pass: trains on a
// spread-sampled covariance, encodes members in ROW order (sequential
// base reads — leaf-order member passes are random-access page-fault
// storms at 10M scale), writes the extent, re-commits the superblock.
// ===========================================================================
// ===========================================================================
// Plane stage-1 routing (see declaration for the contract).
// ===========================================================================
void IVFTreeIndex::plane_route_(const float* query,
                                const SearchConfig& config,
                                std::vector<LeafCandidate>& candidates)
    const {
    std::vector<std::vector<LeafCandidate>> out(1);
    plane_route_batch_(query, 1, config, out);
    candidates = std::move(out[0]);
}

// ===========================================================================
// Batched plane stage-1 (the deployment shape): ONE leaf-major sweep over
// the plane per batch — each plane block is read from the page cache /
// NAND exactly once and scored against EVERY query in the batch while
// hot. Single-query search() pays the full plane read per query and is
// DRAM-bound (~50 GB/s ceiling measured); the batch amortizes the read
// across `nq` queries and turns stage-1 compute-bound.
// ===========================================================================
void IVFTreeIndex::plane_route_batch_(
        const float* queries, uint32_t nq, const SearchConfig& config,
        std::vector<std::vector<LeafCandidate>>& out,
        const std::vector<float>* per_query_fraction) const {
    const uint32_t R = plane_->meta().rank;
    const uint32_t n_leaves = static_cast<uint32_t>(leaf_table_.size());
    const PlaneEncoding enc = plane_->meta().encoding;
    const bool is_b1g = enc == PlaneEncoding::B1G;
    const bool fastscan8 = enc == PlaneEncoding::U4LM;
    const bool pv8 = enc == PlaneEncoding::U4LM_PV;

    // Worker count for every parallelizable phase below (LUT build,
    // survivor masks, leaf sweep, selection). nq == 1 stays serial —
    // single-query callers parallelize across queries themselves.
    const uint32_t T = nq > 1
        ? std::max(1u, config.search_threads > 0
                           ? config.search_threads
                           : std::thread::hardware_concurrency())
        : 1u;
    std::vector<uint8_t> lut8;
    std::vector<float> lut;
    std::vector<float> shifts(nq, 0.0f);  // pv: scale*offset per query
    // ALL nibble encodings ride the shared FastScan u8 kernel; b1g
    // segments are rank/4 (sign nibbles), u4lm/u4lm_pv rank. pv needs
    // the affine shift (scale*offset) to undo the per-segment min
    // subtraction before the per-member alpha multiply.
    const uint32_t SEG = is_b1g ? R / 4 : R;
    if (fastscan8 || is_b1g || pv8)
        lut8.resize(static_cast<size_t>(nq) * SEG * 16);
    if (!fastscan8 && !is_b1g && !pv8)
        lut.resize(static_cast<size_t>(nq) * R * 16);
    {
        // Per-query projections + LUTs, query-parallel (each worker owns
        // its quantizer scratch — the so_*/seg_min outputs are per-query
        // intermediates, only shifts[] survives the phase).
        auto lut_worker = [&](uint32_t q0, uint32_t q1) {
            std::vector<float> pr(R);
            std::vector<float> seg_min(R);
            std::vector<float> so_scale(1), so_offset(1);
            for (uint32_t qi = q0; qi < q1; ++qi) {
                plane_->project_query(queries + static_cast<size_t>(qi) *
                                                manifest_.dim, pr.data());
                if (is_b1g)
                    plane_->build_b1_lut8(
                        pr.data(), &lut8[static_cast<size_t>(qi) * SEG * 16],
                        so_scale.data(), so_offset.data(), seg_min.data());
                else if (fastscan8)
                    plane_->build_lut8(
                        pr.data(), &lut8[static_cast<size_t>(qi) * R * 16],
                        so_scale.data(), so_offset.data(), seg_min.data());
                else if (pv8) {
                    plane_->build_lut8(
                        pr.data(), &lut8[static_cast<size_t>(qi) * R * 16],
                        so_scale.data(), so_offset.data(), seg_min.data());
                    shifts[qi] = so_scale[0] * so_offset[0];
                } else
                    plane_->build_lut(
                        pr.data(), &lut[static_cast<size_t>(qi) * R * 16]);
            }
        };
        std::vector<std::future<void>> futs;
        const uint32_t per = (nq + T - 1) / T;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t q0 = t * per;
            const uint32_t q1 = std::min(nq, q0 + per);
            if (q0 >= q1) break;
            if (t == T - 1 || q1 == nq) {
                lut_worker(q0, q1);
                break;
            }
            futs.push_back(std::async(std::launch::async, lut_worker,
                                      q0, q1));
        }
        for (auto& f : futs) f.get();
    }

    // Two-stage: per-query centroid survivor masks (RAM, ~free). The
    // sweep scores only surviving leaves; non-survivors keep -inf and
    // are never selected. Page-weighted survivor budget like probe f.
    std::vector<float> survive;  // [qi][l] 1.0 / 0.0 (f32 for simplicity)
    if (config.plane_pre_prune > 0.0f && pca_dims_ > 0 &&
        manifest_.depth == 2 &&
        // Exhaustive (capi opts->exhaustive → n_probe = UINT32_MAX) means
        // probe-ALL: the survivor pre-prune must not drop leaves either.
        config.n_probe != UINT32_MAX &&
        pca_leaf_centroids_.size() ==
            static_cast<size_t>(n_leaves) * pca_dims_) {
        std::vector<float> qp(pca_dims_);
        survive.assign(static_cast<size_t>(nq) * n_leaves, 0.0f);
        uint64_t total = 0;
        for (const auto& e : leaf_table_)
            if (e.page != kInvalidPage) total += e.pages;
        // Query-parallel: each worker fills its query rows of survive.
        auto mask_worker = [&](uint32_t q0, uint32_t q1) {
            std::vector<float> qp(pca_dims_);
            for (uint32_t qi = q0; qi < q1; ++qi) {
                for (uint32_t k = 0; k < pca_dims_; ++k) {
                    float acc = 0;
                    const float* row = &pca_proj_[static_cast<size_t>(k) *
                                                  manifest_.dim];
                    for (uint32_t d = 0; d < manifest_.dim; ++d)
                        acc += row[d] *
                               queries[static_cast<size_t>(qi) *
                                           manifest_.dim + d];
                    qp[k] = acc - pca_mean_proj_[k];
                }
                std::vector<std::pair<float, uint32_t>> cd(n_leaves);
                for (uint32_t l = 0; l < n_leaves; ++l) {
                    const float* lc =
                        &pca_leaf_centroids_[static_cast<size_t>(l) *
                                             pca_dims_];
                    float d2 = 0;
                    for (uint32_t k = 0; k < pca_dims_; ++k) {
                        const float diff = qp[k] - lc[k];
                        d2 += diff * diff;
                    }
                    cd[l] = {d2, l};
                }
                std::sort(cd.begin(), cd.end());
                uint64_t cum = 0;
                float* row = &survive[static_cast<size_t>(qi) * n_leaves];
                for (const auto& [d2, l] : cd) {
                    if (leaf_table_[l].page == kInvalidPage) continue;
                    row[l] = 1.0f;
                    cum += leaf_table_[l].pages;
                    if (static_cast<double>(cum) >=
                        static_cast<double>(config.plane_pre_prune) *
                            static_cast<double>(total))
                        break;
                }
            }
        };
        std::vector<std::future<void>> mfuts;
        const uint32_t mper = (nq + T - 1) / T;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t q0 = t * mper;
            const uint32_t q1 = std::min(nq, q0 + mper);
            if (q0 >= q1) break;
            if (t == T - 1 || q1 == nq) {
                mask_worker(q0, q1);
                break;
            }
            mfuts.push_back(std::async(std::launch::async, mask_worker,
                                       q0, q1));
        }
        for (auto& f : mfuts) f.get();
    }

    // Leaf-major sweep: score EVERY query against each leaf's blocks
    // while they are cache-hot (the block read is the batch-shared part).
    // scores layout [l * nq + qi] — LEAF-major: each thread's writes for
    // a leaf are one contiguous run (query-major puts adjacent leaves in
    // the same cache line across threads = false-sharing ping-pong,
    // measured 6x slower). u8 accumulators stay raw (ranking-monotone
    // per query).
    // FLOAT scores: b1g/u4lm_pv scores are signed (sums of +/-w); a
    // uint32 cast wraps negatives to ~4e9 and they win every sort (the
    // u8 raw accumulators are positive-monotone, cast is lossless for
    // ranking).
    std::vector<float> scores(static_cast<size_t>(n_leaves) * nq, 0);
    // Threading: batch mode parallelizes the sweep over leaf chunks with
    // the engine's std::async worker pattern (the engine does NOT build
    // with OpenMP — pragmas are silent no-ops on x86, measured: the
    // batched sweep ran single-threaded until this was noticed).
    auto sweep_worker = [&](uint64_t l0, uint64_t l1) {
        // Per-query scan with survivor skip: a masked query costs one
        // compare + -inf store, never a block sweep. (A query-tiled
        // variant was tried; scanning all tile members when any survive
        // multiplies sweep work ~kQTile at prune budgets.)
        const bool pc = plane_cache_ != nullptr;
        const bool pv2 = plane_->v2_layout();
        const PageId plane_pg0 = superblock_.plane_page();
        // Plane cache active: pin each leaf's row blocks ONCE (fill on
        // miss preads them; fallback = the mmap'd blob pointer), scan all
        // surviving queries against the pinned rows, unpin. Under skewed
        // traffic the hot leaves' routing rows stay resident at ~16 B/vec
        // — 10x denser coverage per byte than leaf extents.
        // v2: this leaf's plane rows are a suffix of its own extent;
        // the offset comes from the open-time cache (the header page is
        // never faulted by the sweep). No plane cache tier in v2: the
        // leaf cache already covers these pages.
        const uint8_t* pinned = nullptr;
        LeafExtentCache::Handle ph;
        for (uint64_t l = l0; l < l1; ++l) {
            if (leaf_table_[static_cast<uint32_t>(l)].page == kInvalidPage)
                continue;
            float* row = &scores[static_cast<size_t>(l) * nq];
            const uint32_t leaf = static_cast<uint32_t>(l);
            const uint8_t* v2_blk = nullptr;
            const uint16_t* v2_alpha = nullptr;
            if (pv2) {
                v2_blk = mmap_base_ +
                    static_cast<uint64_t>(leaf_table_[leaf].page) *
                        kPageSize +
                    leaf_plane_offset_[leaf];
                if (pv8)
                    v2_alpha = reinterpret_cast<const uint16_t*>(
                        v2_blk +
                        static_cast<uint64_t>(
                            plane_->leaf_block_count(leaf)) *
                            plane_->debug_block_bytes());
            }
            if (pc) {
                // Pin only when at least one query survives this leaf —
                // a pin on an all-masked leaf would fill data nobody reads.
                bool any = false;
                if (survive.empty()) {
                    any = true;
                } else {
                    for (uint32_t qi = 0; qi < nq; ++qi) {
                        if (survive[static_cast<size_t>(qi) * n_leaves +
                                    l] != 0.0f) {
                            any = true;
                            break;
                        }
                    }
                }
                if (any) {
                    const uint64_t off = plane_->leaf_block_offset(leaf);
                    const uint32_t nb = plane_->leaf_block_count(leaf);
                    const uint64_t bytes =
                        uint64_t(nb) * plane_->debug_block_bytes();
                    const uint64_t fbyte = uint64_t(plane_pg0) * kPageSize +
                        plane_->blocks_offset_in_blob() + off;
                    const uint32_t pgoff = fbyte % kPageSize;
                    const PageId pg = fbyte / kPageSize;
                    const uint32_t pages = static_cast<uint32_t>(
                        (pgoff + bytes + kPageSize - 1) / kPageSize);
                    // pin() returns a PAGE-ALIGNED pointer; the rows start
                    // pgoff bytes into the first page (leaf extents are
                    // page-aligned, plane rows are not).
                    pinned = plane_cache_->pin(
                                 pg, pages, ph,
                                 mmap_base_ + fbyte)  // fallback: mmap rows
                             + pgoff;
                }
            }
            for (uint32_t qi = 0; qi < nq; ++qi) {
                if (!survive.empty() &&
                    survive[static_cast<size_t>(qi) * n_leaves + l] ==
                        0.0f) {
                    row[qi] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                row[qi] =
                    (is_b1g || fastscan8 || pv8)
                        ? (pv2
                               ? plane_->scan_leaf_max_u8_at(
                                    leaf, v2_blk,
                                    &lut8[static_cast<size_t>(qi) * SEG *
                                           16],
                                    pv8 ? shifts[qi] : 0.0f, v2_alpha)
                               : pinned
                                    ? plane_->scan_leaf_max_u8_at(
                                          leaf, pinned,
                                          &lut8[static_cast<size_t>(qi) *
                                                 SEG * 16],
                                          pv8 ? shifts[qi] : 0.0f)
                                    : plane_->scan_leaf_max_u8(
                                          leaf,
                                          &lut8[static_cast<size_t>(qi) *
                                                 SEG * 16],
                                          pv8 ? shifts[qi] : 0.0f))
                        : plane_->scan_leaf_max(
                              leaf,
                              &lut[static_cast<size_t>(qi) * R * 16],
                              nullptr, nullptr);
            }
            if (pinned) plane_cache_->unpin(ph);
        }
    };
    {
        const uint64_t per = (n_leaves + T - 1) / T;
        std::vector<std::future<void>> futs;
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t l0 = static_cast<uint64_t>(t) * per;
            const uint64_t l1 = std::min<uint64_t>(
                n_leaves, l0 + per);
            if (l0 >= l1) break;
            if (t == T - 1 || l1 == n_leaves) {
                sweep_worker(l0, l1);
                break;
            }
            futs.push_back(std::async(std::launch::async, sweep_worker,
                                      l0, l1));
        }
        for (auto& f : futs) f.get();
    }

    // Per-query selection under the page-weighted fraction budget.
    uint64_t total = 0;
    for (const auto& e : leaf_table_)
        if (e.page != kInvalidPage) total += e.pages;
    out.resize(nq);
    {
        // Query-parallel selection (each query's full leaf-score sort is
        // independent; a serial pass here starved the sweep workers).
        auto select_worker = [&](uint32_t q0, uint32_t q1) {
            std::vector<uint32_t> order(n_leaves);
            for (uint32_t qi = q0; qi < q1; ++qi) {
                const float* sc = &scores[qi];  // column of the [l][nq] grid
                std::iota(order.begin(), order.end(), 0u);
                std::sort(order.begin(), order.end(),
                          [&](uint32_t a, uint32_t b) {
                              return sc[static_cast<size_t>(a) * nq] >
                                     sc[static_cast<size_t>(b) * nq];
                          });
                // Per-query budget (caller-chosen, e.g. SLA tiers) falls
                // back to the shared config, then the manifest.
                // (Engine-ADAPTIVE per-query budgets are a measured
                // negative — scan-feedback closed 2026-09-07.)
                float f = per_query_fraction
                              ? (*per_query_fraction)[qi]
                              : 0.0f;
                if (f <= 0.0f) f = config.probe_fraction;
                if (f <= 0.0f) f = manifest_.probe_fraction;
                if (f <= 0.0f) f = 0.5f;
                // Exhaustive (n_probe = UINT32_MAX) = probe all leaves:
                // the fraction budget must not cap the candidate set, or
                // exact-rerank ground-truth parity fails (rows beyond the
                // plane's top-fraction are unreachable).
                if (config.n_probe == UINT32_MAX) f = 1.0f;
                auto& candidates = out[static_cast<size_t>(qi)];
                candidates.reserve(n_leaves);
                uint64_t cum = 0;
                for (uint32_t l : order) {
                    const auto& e = leaf_table_[l];
                    if (e.page == kInvalidPage) continue;
                    candidates.push_back({e.page, e.pages,
                                          sc[static_cast<size_t>(l) * nq],
                                          nullptr});
                    cum += e.pages;
                    if (static_cast<double>(cum) >=
                        static_cast<double>(f) * static_cast<double>(total)
                            - 0.5)
                        break;
                }
            }
        };
        std::vector<std::future<void>> futs;
        const uint32_t per = (nq + T - 1) / T;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t q0 = t * per;
            const uint32_t q1 = std::min(nq, q0 + per);
            if (q0 >= q1) break;
            if (t == T - 1 || q1 == nq) {
                select_worker(q0, q1);
                break;
            }
            futs.push_back(std::async(std::launch::async, select_worker,
                                      q0, q1));
        }
        for (auto& f : futs) f.get();
    }
}

void IVFTreeIndex::attach_plane(const float* base, uint32_t n, uint32_t dim,
                                PlaneEncoding enc, uint16_t rank,
                                uint32_t train_rows) {
    if (plane_) {
        throw Error(ErrorCode::InvalidParam,
            "attach_plane: index already carries a routing plane");
    }
    if (dim != manifest_.dim) {
        throw Error(ErrorCode::InvalidParam,
            "attach_plane: base dim does not match index dim");
    }
    // Fail-loud precondition (F3/F4 class): the bitmap must cover the
    // whole file before we allocate anything. Trees built before the
    // bitmap upper-bound fix have bitmaps that cover only a prefix —
    // the allocator would hand out LIVE tree pages for the plane extent
    // (measured: 185K pages of leaves overwritten on cohere-10m). Run
    // `sextant fsck --repair` on such trees first.
    if (static_cast<uint64_t>(superblock_.alloc_bitmap_pages()) *
            static_cast<uint64_t>(kPageSize) * 8 <
        superblock_.n_pages()) {
        throw Error(ErrorCode::CorruptIndex,
            "attach_plane: allocation bitmap covers only a prefix of the "
            "file (under-provisioned at build time) — run "
            "`sextant fsck --repair` on this index first");
    }

    const auto t0 = std::chrono::steady_clock::now();
    PlaneWriter::Config pcfg;
    pcfg.encoding = enc;
    pcfg.rank = rank;
    pcfg.train_rows = train_rows;
    PlaneWriter writer;
    writer.train(base, n, dim, pcfg);
    writer.prepare(static_cast<uint32_t>(leaf_table_.size()));
    spdlog::info("[sextant] plane: basis+codebooks trained in {:.1}s",
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count());

    // Members per leaf + row -> (leaf, slot) destinations. Leaf-ordered
    // reads of the row_ids are random-access: warm the file sequentially
    // with a background WILLNEED sweep while building the dest map.
    {
        const int pfd = fd_;
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
            }).detach();
        }
    }
    std::vector<uint32_t> leaf_counts(leaf_table_.size(), 0);
    std::unordered_map<int64_t, std::vector<std::pair<uint32_t, uint32_t>>>
        dest;  // row -> [(leaf, slot)]
    dest.reserve(static_cast<size_t>(n) / 3 * 4 + 1);
    for (uint32_t l = 0; l < leaf_table_.size(); ++l) {
        const auto mem = debug_leaf_row_ids(l);
        leaf_counts[l] = static_cast<uint32_t>(mem.size());
        for (uint32_t slot = 0; slot < mem.size(); ++slot)
            dest[static_cast<int64_t>(mem[slot])].emplace_back(l, slot);
    }

    // Chunked two-phase parallel encode (scales to 1B; serial was
    // ~60 s at 10M = ~100 min at 1B). Per chunk: (1) rows projected in
    // parallel into a BOUNDED buffer (base reads stay sequential);
    // (2) entries grouped by leaf and encoded LEAF-SHARDED — no two
    // threads touch one leaf, so the u4 nibble read-modify-writes
    // cannot race.
    const uint32_t T = std::max(
        1u, std::min<uint32_t>(std::thread::hardware_concurrency(), 64u));
    constexpr uint32_t kChunk = 1u << 21;  // 2M rows -> 1 GB proj buffer
    std::vector<float> proj_buf(static_cast<size_t>(kChunk) * rank);
    struct Ent { uint32_t leaf, slot, row_in_chunk; };
    for (uint32_t c0 = 0; c0 < n; c0 += kChunk) {
        const uint32_t cn = std::min(kChunk, n - c0);
        // Phase 1: parallel projection + entry collection.
        std::vector<std::vector<Ent>> ents(T);
        {
            std::vector<std::future<void>> futs;
            const uint32_t per = (cn + T - 1) / T;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t b0 = c0 + t * per;
                const uint32_t b1 = std::min(c0 + cn, b0 + per);
                if (b0 >= b1) break;
                if (t == T - 1 || b1 == c0 + cn) {
                    for (uint32_t r = b0; r < b1; ++r) {
                        auto it = dest.find(static_cast<int64_t>(r));
                        if (it == dest.end() || it->second.empty())
                            continue;
                        writer.project(
                            &base[static_cast<size_t>(r) * dim],
                            &proj_buf[static_cast<size_t>(r - c0) *
                                      rank]);
                        for (const auto& ls : it->second)
                            ents[t].push_back({ls.first, ls.second,
                                               r - c0});
                    }
                    break;
                }
                futs.push_back(std::async(std::launch::async, [&, b0, b1, t] {
                    for (uint32_t r = b0; r < b1; ++r) {
                        auto it = dest.find(static_cast<int64_t>(r));
                        if (it == dest.end() || it->second.empty())
                            continue;
                        writer.project(
                            &base[static_cast<size_t>(r) * dim],
                            &proj_buf[static_cast<size_t>(r - c0) *
                                      rank]);
                        for (const auto& ls : it->second)
                            ents[t].push_back({ls.first, ls.second,
                                               r - c0});
                    }
                }));
            }
            for (auto& f : futs) f.get();
        }
        // Phase 2: counting-sort by leaf, then leaf-sharded encode.
        std::vector<Ent> flat;
        {
            size_t tot = 0;
            for (auto& v : ents) tot += v.size();
            flat.reserve(tot);
            for (auto& v : ents)
                flat.insert(flat.end(), v.begin(), v.end());
        }
        std::vector<uint32_t> leaf_off(leaf_table_.size() + 1, 0);
        for (const auto& e : flat) ++leaf_off[e.leaf + 1];
        for (size_t i = 1; i < leaf_off.size(); ++i)
            leaf_off[i] += leaf_off[i - 1];
        std::vector<Ent> sorted(flat.size());
        {
            std::vector<uint32_t> cur(leaf_off.begin(),
                                      leaf_off.end() - 1);
            for (const auto& e : flat)
                sorted[cur[e.leaf]++] = e;
        }
        {
            const uint32_t n_leaf_total =
                static_cast<uint32_t>(leaf_table_.size());
            const uint32_t lper = (n_leaf_total + T - 1) / T;
            std::vector<std::future<void>> futs;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t l0 = t * lper;
                const uint32_t l1 = std::min(n_leaf_total, l0 + lper);
                if (l0 >= l1) break;
                auto shard = [&, l0, l1] {
                    for (uint32_t l = l0; l < l1; ++l)
                        for (uint32_t i = leaf_off[l]; i < leaf_off[l + 1];
                             ++i) {
                            const Ent& e = sorted[i];
                            writer.encode_from_proj(
                                e.leaf, e.slot,
                                &proj_buf[static_cast<size_t>(e.row_in_chunk) *
                                          rank]);
                        }
                };
                if (t == T - 1 || l1 == n_leaf_total) {
                    shard();
                    break;
                }
                futs.push_back(std::async(std::launch::async, shard));
            }
            for (auto& f : futs) f.get();
        }
    }
    spdlog::info("[sextant] plane: encoded {} rows in {:.1}s", n,
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count());

    attach_plane_from(writer, leaf_counts);
}

void IVFTreeIndex::attach_plane_from(PlaneWriter& writer,
                                     const std::vector<uint32_t>& leaf_counts) {
    auto blob = writer.finalize(leaf_counts);
    const auto t0 = std::chrono::steady_clock::now();

    // The bitmap region is FIXED at format time; a plane extent can
    // outgrow its coverage (10M-scale plane = ~180K pages). When it
    // would, relocate the bitmap: append a larger one at EOF, free the
    // old region's pages, update the superblock pointers.
    const uint32_t npg0 = static_cast<uint32_t>(
        (blob.size() + kPageSize - 1) / kPageSize);
    const uint64_t kBitsPerBitmapPage =
        static_cast<uint64_t>(kPageSize) * 8;
    {
        const uint64_t covered =
            static_cast<uint64_t>(superblock_.alloc_bitmap_pages()) *
            kBitsPerBitmapPage;
        const uint64_t est_total = superblock_.n_pages() + npg0 + 64;
        if (est_total > covered) {
            const uint32_t new_nbm = static_cast<uint32_t>(
                est_total / kBitsPerBitmapPage + 2);
            const PageId old_bm_page = superblock_.alloc_bitmap_page();
            const uint32_t old_nbm = superblock_.alloc_bitmap_pages();
            const PageId new_bm_page = file_.num_pages();
            std::vector<uint8_t> nbm(
                static_cast<size_t>(new_nbm) * kPageSize, 0);
            std::vector<uint8_t> obm(
                static_cast<size_t>(old_nbm) * kPageSize, 0);
            file_.read_pages(old_bm_page, old_nbm, obm.data());
            const size_t copy_bits = std::min<uint64_t>(
                superblock_.n_pages(),
                static_cast<uint64_t>(old_nbm) * kBitsPerBitmapPage);
            std::memcpy(nbm.data(), obm.data(), (copy_bits + 7) / 8);
            for (uint64_t p = new_bm_page;
                 p < new_bm_page + new_nbm; ++p) {
                nbm[p / 8] |= static_cast<uint8_t>(1u << (p % 8));
            }
            file_.truncate(new_bm_page + new_nbm);
            file_.write_pages(new_bm_page, new_nbm, nbm.data());
            const PageId old_flh = superblock_.free_list_head();
            const uint64_t old_nfree = superblock_.n_free_pages();
            superblock_.set_bitmap(new_bm_page, new_nbm);
            superblock_.set_n_pages(file_.num_pages());
            // Reload the allocator over the new geometry, then push the
            // old bitmap pages onto the free list (maintains the
            // persisted chain + bitmap bits).
            PageAllocator alloc2;
            alloc2.load(file_, new_bm_page, new_nbm,
                        file_.num_pages(), old_flh, old_nfree);
            alloc2.free_extent(file_, old_bm_page, old_nbm);
            alloc2.flush_bitmap(file_);
            superblock_.set_free_list(alloc2.free_list_head(),
                                      alloc2.n_free_pages());
            superblock_.commit(file_);
            spdlog::info("[sextant] plane: bitmap relocated {} -> {} pages "
                         "(@page {})", old_nbm, new_nbm, new_bm_page);
        }
    }

    // Allocate the extent, write, re-commit the superblock.
    PageAllocator alloc;
    alloc.load(file_, superblock_.alloc_bitmap_page(),
               superblock_.alloc_bitmap_pages(), superblock_.n_pages(),
               superblock_.free_list_head(), superblock_.n_free_pages());
    const uint32_t npg = static_cast<uint32_t>(
        (blob.size() + kPageSize - 1) / kPageSize);
    const PageId ppage = alloc.alloc_extent(file_, npg);
    std::vector<uint8_t> padded(static_cast<size_t>(npg) * kPageSize, 0);
    std::memcpy(padded.data(), blob.data(), blob.size());
    file_.write_pages(ppage, npg, padded.data());
    alloc.flush_bitmap(file_);

    superblock_.set_plane(ppage, npg);
    superblock_.set_n_pages(file_.num_pages());
    superblock_.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    superblock_.commit(file_);
    file_.sync();

    // Reload in-memory state: re-mmap (the plane pages may extend the
    // file) and parse + bind.
    ::munmap(const_cast<uint8_t*>(mmap_base_), mmap_size_);
    mmap_size_ = file_.num_pages() * kPageSize;
    void* addr = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED,
                        file_.fd(), 0);
    if (addr == MAP_FAILED) {
        throw Error(ErrorCode::IoError,
            "attach_plane: re-mmap failed: " +
                std::string(std::strerror(errno)));
    }
    mmap_base_ = static_cast<const uint8_t*>(addr);
    plane_ = PlaneIndex::parse(mmap_base_ + static_cast<uint64_t>(ppage) *
                              kPageSize,
                              static_cast<size_t>(npg) * kPageSize);
    if (!plane_) {
        throw Error(ErrorCode::CorruptIndex,
            "attach_plane: written plane failed to parse");
    }
    plane_->bind(leaf_counts);
    spdlog::info("[sextant] plane: attached ({} pages, {} B/vec) in {:.1}s",
                 npg, plane_->meta().bytes_per_vec(),
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count());
}

}  // namespace sextant::tree
