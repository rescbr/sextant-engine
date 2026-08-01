#include "ivf_tree_index.hpp"

#include "engine/manifest_io.hpp"
#include "engine/partition.hpp"
#include "quant/pq_quantizer.hpp"
#include "quant/product_residual_quantizer.hpp"
#include "quant/rabitq_quantizer.hpp"
#include "util/fp16.hpp"
#include "simd_kernels.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <future>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace sextant::tree {

/// Working data for building a root child entry (build-time only).
struct RootChildData {
    std::vector<float16_t> centroid;
    PageId page = kInvalidPage;
    uint32_t pages = 0;
    uint16_t is_leaf = 0;
};

// ===========================================================================
// Destructor + close
// ===========================================================================

IVFTreeIndex::~IVFTreeIndex() {
    close();
}

void IVFTreeIndex::close() {
    if (mmap_base_) {
        ::munmap(const_cast<uint8_t*>(mmap_base_), mmap_size_);
        mmap_base_ = nullptr;
    }
    // PageFile destructor closes the fd.
}

// ===========================================================================
// Open
// ===========================================================================

std::unique_ptr<IVFTreeIndex> IVFTreeIndex::open(const std::string& path) {
    auto idx = std::unique_ptr<IVFTreeIndex>(new IVFTreeIndex());
    idx->path_ = path;
    idx->file_ = PageFile(path);

    // Load superblock.
    idx->superblock_.load(idx->file_);

    // Read the TOML config blob.
    if (idx->superblock_.config_page() == kInvalidPage) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFTreeIndex::open: no config blob in superblock");
    }
    const uint32_t cfg_pages = idx->superblock_.config_pages();
    std::string cfg_buf(static_cast<size_t>(cfg_pages) * kPageSize, '\0');
    idx->file_.read_pages(idx->superblock_.config_page(), cfg_pages, cfg_buf.data());
    // The config blob is page-padded with null bytes; trim to actual TOML.
    const size_t nul = cfg_buf.find('\0');
    if (nul != std::string::npos) cfg_buf.resize(nul);
    idx->manifest_ = manifest_from_toml(cfg_buf);

    // Load the codebook from the codebook extent.
    {
        const auto cb_page = idx->superblock_.codebook_page();
        const auto cb_pages = idx->superblock_.codebook_pages();
        if (cb_page == kInvalidPage || cb_pages == 0) {
            throw Error(ErrorCode::CorruptIndex,
                        "IVFTreeIndex::open: no codebook in superblock");
        }
        std::vector<uint8_t> blob(static_cast<size_t>(cb_pages) * kPageSize);
        idx->file_.read_pages(cb_page, cb_pages, blob.data());

        // The codebook blob is prefixed with its serialized size (u64),
        // then page-padded. Read the size and pass only the real bytes.
        uint64_t qblob_size = 0;
        std::memcpy(&qblob_size, blob.data(), sizeof(qblob_size));
        const uint8_t* qblob_data = blob.data() + sizeof(qblob_size);

        // Deserialize the codebook. Construct the right quantizer type.
        const auto& m = idx->manifest_;
        if (m.quantizer_type == "prq") {
            idx->quantizer_ = std::make_unique<ProductResidualQuantizer>(
                MetricKind::L2Sq, m.dim, m.m4, m.scan_pq_bits,
                m.prq_nsplits, /*beam_size=*/1);
        } else if (m.quantizer_type == "rabitq") {
            idx->quantizer_ = std::make_unique<RaBitQQuantizer>(
                MetricKind::L2Sq, m.dim);
        } else {
            idx->quantizer_ = std::make_unique<PqQuantizer>(
                MetricKind::L2Sq, m.dim, m.m4, m.scan_pq_bits);
        }
        idx->quantizer_->deserialize(qblob_data, static_cast<size_t>(qblob_size));
    }

    // mmap the file read-only for search.
    idx->mmap_size_ = idx->file_.num_pages() * kPageSize;
    void* addr = ::mmap(nullptr, idx->mmap_size_, PROT_READ, MAP_SHARED,
                        idx->file_.fd(), 0);
    if (addr == MAP_FAILED) {
        throw Error(ErrorCode::IoError,
                    "IVFTreeIndex::open: mmap failed on '" + path + "': " +
                        std::strerror(errno));
    }
    idx->mmap_base_ = static_cast<const uint8_t*>(addr);

    // Parse the root node from the mmap.
    idx->load_root_from_mmap();

    spdlog::info("[sextant] IVFTreeIndex: opened '{}' (depth={}, k_root={}, "
                 "n_leaves={}, dim={}, m4={}, quantizer={})",
                 path, idx->manifest_.depth, idx->manifest_.k_root,
                 idx->manifest_.n_leaves, idx->manifest_.dim, idx->manifest_.m4,
                 idx->manifest_.quantizer_type);

    return idx;
}

void IVFTreeIndex::load_root_from_mmap() {
    const auto root_page = superblock_.root_node_page();
    if (root_page == kInvalidPage) {
        throw Error(ErrorCode::CorruptIndex,
                    "IVFTreeIndex: superblock has no root node");
    }
    const uint8_t* root_ptr = mmap_base_ +
        static_cast<uint64_t>(root_page) * kPageSize;
    root_header_ = reinterpret_cast<const TreeNodeHeader*>(root_ptr);

    const uint16_t dim = manifest_.dim;
    const uint32_t cesize = child_entry_size(dim);
    root_children_.resize(root_header_->n_children);
    const uint8_t* p = root_ptr + sizeof(TreeNodeHeader);
    for (uint32_t i = 0; i < root_header_->n_children; ++i) {
        const auto* ce = reinterpret_cast<const ChildEntry*>(p);
        root_children_[i].page = ce->child_page;
        root_children_[i].pages = ce->child_pages;
        root_children_[i].is_leaf = ce->is_leaf;
        root_children_[i].centroid = reinterpret_cast<const float16_t*>(
            p + sizeof(ChildEntry));
        p += cesize;
    }
}

// ===========================================================================
// Search
// ===========================================================================

std::vector<Candidate> IVFTreeIndex::search(const float* query, uint32_t k,
                                             const SearchConfig& config) const {
    // RaBitQ needs per-leaf LUT rebuild + factor finalization — separate path.
    if (manifest_.quantizer_type == "rabitq") {
        return search_rabitq(query, k, config);
    }

    // Build the FastScan LUT once per query.
    const bool scan_8bit = (manifest_.scan_pq_bits == 8);
    const uint32_t m = manifest_.m4;
    const uint32_t codes_per_block = scan_8bit ? 16 : 32;
    const uint32_t block_bytes = m * 16;  // [m][16] for both 4-bit and 8-bit
    const bool is_rabitq = (manifest_.quantizer_type == "rabitq");

    // LUT buffers.
    std::vector<uint8_t> lut4(scan_8bit ? 0 : m * 16);
    std::vector<uint8_t> lut8(scan_8bit ? m * 256 : 0);
    float lut_scale = 0, lut_offset = 0;

    if (scan_8bit) {
        quantizer_->build_fastscan_lut(query, lut8.data(), &lut_scale,
                                        &lut_offset);
    } else {
        quantizer_->build_fastscan_lut4(query, lut4.data(), nullptr);
    }

    // Cast query to FP16 for routing.
    std::vector<float16_t> query_fp16(manifest_.dim);
    cast_fp32_to_fp16(query, query_fp16.data(), manifest_.dim);
    const MetricKind metric = quantizer_->metric();

    // --- Route at level 0: FP16 dist to all root centroids ---
    const uint32_t k_root = root_header_->n_children;
    std::vector<std::pair<float, uint32_t>> root_dists;
    root_dists.reserve(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        const float d = simd::dist_f16(metric, query_fp16.data(),
                                        root_children_[c].centroid,
                                        manifest_.dim);
        root_dists.emplace_back(d, c);
    }
    std::sort(root_dists.begin(), root_dists.end());

    uint32_t n_probe_l0 = config.n_probe > 0
        ? config.n_probe
        : manifest_.n_probe_l0;
    n_probe_l0 = std::min(n_probe_l0, k_root);

    // --- Collect leaf candidates ---
    // For Phase 1 (2-level tree): root children are either leaves (depth=1)
    // or level-1 internal nodes (depth=2). For depth=2, we probe each
    // selected level-1 node's children.
    std::vector<LeafCandidate> candidates;
    const float gap = (config.adaptive_probe_gap > 0)
        ? config.adaptive_probe_gap
        : manifest_.adaptive_probe_gap;

    for (uint32_t i = 0; i < n_probe_l0; ++i) {
        const auto& [dist, child_idx] = root_dists[i];
        const auto& rc = root_children_[child_idx];

        // Adaptive gap pruning.
        if (gap > 0 && i > 0) {
            if (root_dists[i].first > root_dists[i - 1].first * gap) break;
        }

        if (rc.is_leaf) {
            // Depth=1: root child IS a leaf.
            if (rc.page == kInvalidPage) continue;  // empty group
            candidates.push_back({rc.page, rc.pages, dist, rc.centroid});
        } else {
            // Depth=2: root child is an internal node. Read its children
            // from the mmap and route.
            const uint8_t* node_ptr = mmap_base_ +
                static_cast<uint64_t>(rc.page) * kPageSize;
            const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
            const uint32_t cesize = child_entry_size(manifest_.dim);
            const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);

            std::vector<std::pair<float, uint32_t>> child_dists;
            child_dists.reserve(nh->n_children);
            for (uint32_t j = 0; j < nh->n_children; ++j) {
                const auto* ce = reinterpret_cast<const ChildEntry*>(p);
                const float16_t* cent = reinterpret_cast<const float16_t*>(
                    p + sizeof(ChildEntry));
                const float d = simd::dist_f16(metric, query_fp16.data(),
                                                cent, manifest_.dim);
                child_dists.emplace_back(d, j);
                p += cesize;
            }
            std::sort(child_dists.begin(), child_dists.end());

            uint32_t n_probe_ln = manifest_.n_probe_ln > 0
                ? manifest_.n_probe_ln
                : 4;
            n_probe_ln = std::min(n_probe_ln, nh->n_children);

            const uint8_t* p2 = node_ptr + sizeof(TreeNodeHeader);
            for (uint32_t j = 0; j < n_probe_ln; ++j) {
                if (gap > 0 && j > 0) {
                    if (child_dists[j].first >
                        child_dists[j - 1].first * gap) break;
                }
                const uint32_t idx = child_dists[j].second;
                const auto* ce = reinterpret_cast<const ChildEntry*>(
                    p2 + idx * cesize);
                if (ce->child_page == kInvalidPage) continue;  // empty leaf
                // The leaf's centroid is inline in the child entry.
                const float16_t* leaf_cent = reinterpret_cast<const float16_t*>(
                    reinterpret_cast<const uint8_t*>(ce) + sizeof(ChildEntry));
                candidates.push_back({ce->child_page, ce->child_pages,
                                      child_dists[j].first, leaf_cent});
            }
        }
    }

    if (candidates.empty()) {
        return {};
    }

    // --- Prefetch leaf extents ---
    // On Linux, posix_fadvise triggers async NVMe prefetch. On macOS it's
    // a no-op (the unified buffer cache handles read-ahead for sequential
    // mmap access).
    for (const auto& c : candidates) {
#ifdef __linux__
        ::posix_fadvise(fd_, static_cast<off_t>(c.page) * kPageSize,
                        static_cast<off_t>(c.pages) * kPageSize,
                        POSIX_FADV_WILLNEED);
#else
        (void)c;  // macOS: no-op
#endif
    }

    // --- Scan leaves ---
    const uint32_t W = std::max(config.fastscan_W > 0 ? config.fastscan_W : 300u, k);

    // Max-heap of (distance, global_row_id).
    std::vector<std::pair<uint32_t, int64_t>> heap;
    heap.reserve(W + 32);
    const auto heap_less = [](const auto& a, const auto& b) {
        return a.first < b.first;
    };

    auto heap_replace = [&](uint32_t new_d, int64_t new_id) {
        heap[0] = {new_d, new_id};
        uint32_t pos = 0;
        const uint32_t n = heap.size();
        while (true) {
            const uint32_t left = 2 * pos + 1;
            const uint32_t right = 2 * pos + 2;
            uint32_t largest = pos;
            if (left < n && heap[left].first > heap[largest].first)
                largest = left;
            if (right < n && heap[right].first > heap[largest].first)
                largest = right;
            if (largest == pos) break;
            std::swap(heap[pos], heap[largest]);
            pos = largest;
        }
    };

    for (const auto& cand : candidates) {
        if (cand.page == kInvalidPage) continue;
        const uint8_t* leaf_ptr = mmap_base_ +
            static_cast<uint64_t>(cand.page) * kPageSize;
        const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
        const uint32_t count = lh->count;
        if (count == 0) continue;

        const uint32_t n_blocks = (count + codes_per_block - 1) / codes_per_block;
        const uint8_t* codes = leaf_ptr + leaf_codes_offset();
        const RowId* row_ids = reinterpret_cast<const RowId*>(
            leaf_ptr + leaf_rowids_offset(n_blocks, block_bytes));
        const float* factors = is_rabitq
            ? reinterpret_cast<const float*>(
                leaf_ptr + leaf_factors_offset(n_blocks, block_bytes, count))
            : nullptr;

        // Compute the valid_mask for the tail block.
        const uint32_t full_blocks = count / codes_per_block;
        const uint32_t tail_count = count - full_blocks * codes_per_block;
        const uint32_t tail_mask = (tail_count == 0)
            ? (codes_per_block == 32 ? 0xFFFFFFFFu : 0x0000FFFFu)
            : (codes_per_block == 32
                   ? (tail_count >= 32 ? 0xFFFFFFFFu
                       : (1u << tail_count) - 1u)
                   : (tail_count >= 16 ? 0xFFFFu : (1u << tail_count) - 1u));

        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint8_t* blk = codes + static_cast<uint64_t>(b) * block_bytes;
            const uint32_t valid_mask = (b + 1 < n_blocks)
                ? (codes_per_block == 32 ? 0xFFFFFFFFu : 0xFFFFu)
                : tail_mask;

            if (scan_8bit) {
                uint32_t out[16];
                simd::fastscan_block16(blk, lut8.data(), m,
                                        static_cast<uint16_t>(valid_mask), out);
                const uint32_t base = b * 16;
                if (heap.size() < W) {
                    for (uint32_t j = 0; j < 16; ++j) {
                        if (out[j] == 0xFFFFFFFFu) continue;
                        heap.emplace_back(out[j], row_ids[base + j]);
                        if (heap.size() == W) {
                            std::make_heap(heap.begin(), heap.end(), heap_less);
                            break;
                        }
                    }
                    if (heap.size() < W) continue;
                }
                const uint32_t front_d = heap[0].first;
                uint32_t block_min = 0xFFFFFFFFu;
                for (uint32_t j = 0; j < 16; ++j)
                    if (out[j] < block_min) block_min = out[j];
                if (block_min >= front_d) continue;
                for (uint32_t j = 0; j < 16; ++j) {
                    if (out[j] == 0xFFFFFFFFu || out[j] >= heap[0].first)
                        continue;
                    heap_replace(out[j], row_ids[base + j]);
                }
            } else {
                uint32_t out[32];
                simd::pq4_block32(blk, lut4.data(), m, out);
                const uint32_t base = b * 32;

                if (heap.size() < W) {
                    for (uint32_t j = 0; j < 32; ++j) {
                        if (!((valid_mask >> j) & 1u)) continue;
                        heap.emplace_back(out[j], row_ids[base + j]);
                        if (heap.size() == W) {
                            std::make_heap(heap.begin(), heap.end(), heap_less);
                            break;
                        }
                    }
                    if (heap.size() < W) continue;
                }

                const uint32_t front_d = heap[0].first;
                uint32_t block_min = 0xFFFFFFFFu;
                for (uint32_t j = 0; j < 32; ++j)
                    if ((valid_mask >> j) & 1u)
                        block_min = std::min(block_min, out[j]);
                if (block_min >= front_d) continue;

                for (uint32_t j = 0; j < 32; ++j) {
                    if (!((valid_mask >> j) & 1u)) continue;
                    if (out[j] >= heap[0].first) continue;
                    heap_replace(out[j], row_ids[base + j]);
                }
            }
        }
    }

    // --- Extract top-k from the heap ---
    std::vector<Candidate> results;
    results.reserve(heap.size());
    for (const auto& [d, id] : heap) {
        results.push_back({id, static_cast<float>(d)});
    }
    // The heap is a max-heap; we want ascending order.
    // Dedup by row_id (closure may replicate vectors across leaves).
    std::sort(results.begin(), results.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.row_id != b.row_id) return a.row_id < b.row_id;
                  return a.dist < b.dist;
              });
    auto last = std::unique(results.begin(), results.end(),
                            [](const Candidate& a, const Candidate& b) {
                                return a.row_id == b.row_id;
                            });
    results.erase(last, results.end());
    if (results.size() > k) {
        std::nth_element(results.begin(), results.begin() + k, results.end(),
                         [](const Candidate& a, const Candidate& b) {
                             return a.dist < b.dist;
                         });
        results.resize(k);
    }
    std::sort(results.begin(), results.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.dist < b.dist;
              });
    return results;
}

// ===========================================================================
// RaBitQ search (per-leaf LUT rebuild + factor finalization)
// ===========================================================================

std::vector<Candidate> IVFTreeIndex::search_rabitq(const float* query,
                                                     uint32_t k,
                                                     const SearchConfig& config)
    const {
    auto& rabitq = static_cast<RaBitQQuantizer&>(*quantizer_);
    const uint32_t m = manifest_.m4;
    const uint32_t codes_per_block = 32;
    const uint32_t block_bytes = m * 16;

    std::vector<uint8_t> lut4(m * 16);
    std::vector<float16_t> query_fp16(manifest_.dim);
    cast_fp32_to_fp16(query, query_fp16.data(), manifest_.dim);
    const MetricKind metric = rabitq.metric();

    // RaBitQ query state (populated by build_lut4_with_state).
    // We need thread-local storage for rabitq's internal state (c34_,
    // qr_to_c_l2sqr_, rabitq_qs). Since search is const and single-threaded
    // for now, we use mutable scratch. The rabitq object stores these as
    // mutable members.

    // --- Route at level 0 ---
    const uint32_t k_root = root_header_->n_children;
    std::vector<std::pair<float, uint32_t>> root_dists;
    root_dists.reserve(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        const float d = simd::dist_f16(metric, query_fp16.data(),
                                        root_children_[c].centroid,
                                        manifest_.dim);
        root_dists.emplace_back(d, c);
    }
    std::sort(root_dists.begin(), root_dists.end());

    uint32_t n_probe_l0 = config.n_probe > 0
        ? config.n_probe
        : manifest_.n_probe_l0;
    n_probe_l0 = std::min(n_probe_l0, k_root);

    // --- Collect leaf candidates (same routing as PQ path) ---
    std::vector<LeafCandidate> candidates;
    const float gap = (config.adaptive_probe_gap > 0)
        ? config.adaptive_probe_gap
        : manifest_.adaptive_probe_gap;

    for (uint32_t i = 0; i < n_probe_l0; ++i) {
        const auto& [dist, child_idx] = root_dists[i];
        const auto& rc = root_children_[child_idx];

        if (gap > 0 && i > 0) {
            if (root_dists[i].first > root_dists[i - 1].first * gap) break;
        }

        if (rc.is_leaf) {
            if (rc.page == kInvalidPage) continue;
            candidates.push_back({rc.page, rc.pages, dist, rc.centroid});
        } else {
            const uint8_t* node_ptr = mmap_base_ +
                static_cast<uint64_t>(rc.page) * kPageSize;
            const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
            const uint32_t cesize = child_entry_size(manifest_.dim);
            const uint8_t* p = node_ptr + sizeof(TreeNodeHeader);

            std::vector<std::pair<float, uint32_t>> child_dists;
            child_dists.reserve(nh->n_children);
            for (uint32_t j = 0; j < nh->n_children; ++j) {
                const auto* ce = reinterpret_cast<const ChildEntry*>(p);
                const float16_t* cent = reinterpret_cast<const float16_t*>(
                    p + sizeof(ChildEntry));
                const float d = simd::dist_f16(metric, query_fp16.data(),
                                                cent, manifest_.dim);
                child_dists.emplace_back(d, j);
                p += cesize;
            }
            std::sort(child_dists.begin(), child_dists.end());

            uint32_t n_probe_ln = manifest_.n_probe_ln > 0
                ? manifest_.n_probe_ln
                : 4;
            n_probe_ln = std::min(n_probe_ln, nh->n_children);

            const uint8_t* p2 = node_ptr + sizeof(TreeNodeHeader);
            for (uint32_t j = 0; j < n_probe_ln; ++j) {
                if (gap > 0 && j > 0) {
                    if (child_dists[j].first >
                        child_dists[j - 1].first * gap) break;
                }
                const uint32_t idx = child_dists[j].second;
                const auto* ce = reinterpret_cast<const ChildEntry*>(
                    p2 + idx * cesize);
                if (ce->child_page == kInvalidPage) continue;
                const float16_t* leaf_cent = reinterpret_cast<const float16_t*>(
                    reinterpret_cast<const uint8_t*>(ce) + sizeof(ChildEntry));
                candidates.push_back({ce->child_page, ce->child_pages,
                                      child_dists[j].first, leaf_cent});
            }
        }
    }

    if (candidates.empty()) return {};

    // --- Per-leaf: rebuild LUT → scan → finalize → float heap ---
    const uint32_t W = std::max(config.fastscan_W > 0 ? config.fastscan_W : 300u, k);

    // Global float max-heap of (dist, row_id).
    std::vector<std::pair<float, int64_t>> heap;
    heap.reserve(W + 32);
    const auto heap_less = [](const auto& a, const auto& b) {
        return a.first < b.first;
    };
    auto heap_replace = [&](float new_d, int64_t new_id) {
        heap[0] = {new_d, new_id};
        uint32_t pos = 0;
        const uint32_t n = heap.size();
        while (true) {
            const uint32_t left = 2 * pos + 1;
            const uint32_t right = 2 * pos + 2;
            uint32_t largest = pos;
            if (left < n && heap[left].first > heap[largest].first)
                largest = left;
            if (right < n && heap[right].first > heap[largest].first)
                largest = right;
            if (largest == pos) break;
            std::swap(heap[pos], heap[largest]);
            pos = largest;
        }
    };

    std::vector<float> centroid_f32(manifest_.dim);
    RaBitQQuantizer::QueryState rabitq_qs;

    for (const auto& cand : candidates) {
        if (cand.page == kInvalidPage) continue;
        const uint8_t* leaf_ptr = mmap_base_ +
            static_cast<uint64_t>(cand.page) * kPageSize;
        const auto* lh = reinterpret_cast<const TreeLeafHeader*>(leaf_ptr);
        const uint32_t count = lh->count;
        if (count == 0) continue;

        const uint32_t n_blocks = (count + codes_per_block - 1) / codes_per_block;
        const uint8_t* codes = leaf_ptr + leaf_codes_offset();
        const RowId* row_ids = reinterpret_cast<const RowId*>(
            leaf_ptr + leaf_rowids_offset(n_blocks, block_bytes));
        const float* factors = reinterpret_cast<const float*>(
            leaf_ptr + leaf_factors_offset(n_blocks, block_bytes, count));

        // Convert the leaf centroid FP16 → FP32.
        for (uint32_t d = 0; d < manifest_.dim; ++d) {
            centroid_f32[d] = static_cast<float>(cand.centroid[d]);
        }

        // Rebuild the FastScan LUT for THIS leaf (encodes query-centroid
        // interaction). Also populates rabitq_qs and internal state.
        rabitq.build_lut4_with_state(query, centroid_f32.data(),
                                     lut4.data(), rabitq_qs);

        // Compute valid_mask for the tail block.
        const uint32_t full_blocks = count / codes_per_block;
        const uint32_t tail_count = count - full_blocks * codes_per_block;
        const uint32_t tail_mask = (tail_count == 0)
            ? 0xFFFFFFFFu
            : (tail_count >= 32 ? 0xFFFFFFFFu : (1u << tail_count) - 1u);

        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint8_t* blk = codes + static_cast<uint64_t>(b) * block_bytes;
            const uint32_t valid_mask = (b + 1 < n_blocks)
                ? 0xFFFFFFFFu
                : tail_mask;

            uint32_t out[32];
            simd::pq4_block32(blk, lut4.data(), m, out);
            const uint32_t base = b * 32;

            for (uint32_t j = 0; j < 32; ++j) {
                if (!((valid_mask >> j) & 1u)) continue;
                const uint32_t local_idx = base + j;
                if (local_idx >= count) continue;

                const float* fac = factors + static_cast<size_t>(local_idx) * 2;
                const float est = rabitq.dequant_and_finalize(out[j], fac,
                                                               rabitq_qs);

                if (heap.size() < W) {
                    heap.emplace_back(est, row_ids[local_idx]);
                    if (heap.size() == W) {
                        std::make_heap(heap.begin(), heap.end(), heap_less);
                    }
                    continue;
                }
                // Lower-bound admission with error bound.
                if (est >= heap[0].first) {
                    const float err = rabitq.error_bound(fac, rabitq_qs);
                    if (est - err >= heap[0].first) continue;
                }
                heap_replace(est, row_ids[local_idx]);
            }
        }
    }

    // --- Extract top-k (dedup by min distance) ---
    std::unordered_map<int64_t, float> best;
    best.reserve(heap.size());
    for (const auto& [d, id] : heap) {
        auto it = best.find(id);
        if (it == best.end() || d < it->second) best[id] = d;
    }

    std::vector<Candidate> results;
    results.reserve(best.size());
    for (const auto& [id, d] : best) {
        results.push_back({id, d});
    }
    if (results.size() > k) {
        std::nth_element(results.begin(), results.begin() + k, results.end(),
                         [](const Candidate& a, const Candidate& b) {
                             return a.dist < b.dist;
                         });
        results.resize(k);
    }
    std::sort(results.begin(), results.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.dist < b.dist;
              });
    return results;
}
// ===========================================================================

BuildResult IVFTreeIndex::build(const std::string& base_path,
                                 const std::string& output_path,
                                 const BuildConfig& cfg) {
    const auto t0 = std::chrono::steady_clock::now();

    // --- 1. Load the base vectors ---
    // Read the fbin header.
    FILE* f = std::fopen(base_path.c_str(), "rb");
    if (!f) {
        throw Error(ErrorCode::IoError,
                    "IVFTreeIndex::build: cannot open '" + base_path + "': " +
                        std::strerror(errno));
    }
    uint32_t header[2];  // [n, dim]
    if (std::fread(header, sizeof(uint32_t), 2, f) != 2) {
        std::fclose(f);
        throw Error(ErrorCode::IoError, "IVFTreeIndex::build: failed to read fbin header");
    }
    const uint64_t n = header[0];
    const Dim dim = header[1];
    if (n == 0 || dim == 0) {
        std::fclose(f);
        throw Error(ErrorCode::InvalidParam, "IVFTreeIndex::build: empty fbin");
    }

    spdlog::info("[sextant] build_tree: N={} dim={} → '{}'", n, dim, output_path);

    // --- 2. Resolve build parameters ---
    const auto& params = cfg.params;
    const uint16_t m4 = params.pq4_m > 0 ? params.pq4_m
                                              : static_cast<uint16_t>(dim / 4);
    const uint8_t scan_bits = params.scan_pq_bits;

    // Leaf / tree params.
    const uint32_t leaf_cap = cfg.leaf_capacity > 0 ? cfg.leaf_capacity : 5000;
    uint32_t k_root = cfg.k_root;
    if (k_root == 0) {
        // Auto: sqrt(N / leaf_cap) clamped [16, 256].
        k_root = static_cast<uint32_t>(std::sqrt(static_cast<double>(n) / leaf_cap));
        k_root = std::clamp(k_root, 16u, 256u);
    }
    const uint32_t n_leaves = std::max(1u, static_cast<uint32_t>(
        (n + leaf_cap - 1) / leaf_cap));

    spdlog::info("[sextant] build_tree: k_root={} n_leaves={} leaf_cap={} m4={}",
                 k_root, n_leaves, leaf_cap, m4);

    // --- 3. Train quantizer + encode all vectors ---
    std::unique_ptr<PqQuantizer> quantizer;
    uint8_t n_factors = 0;

    if (params.quantizer_type == "prq") {
        const uint32_t nsplits = (params.prq_nsplits > 0)
            ? params.prq_nsplits
            : static_cast<uint32_t>(dim) / 8;
        quantizer = std::make_unique<ProductResidualQuantizer>(
            params.metric, dim, m4, scan_bits, nsplits,
            params.prq_beam_size, /*seed=*/42);
    } else if (params.quantizer_type == "rabitq") {
        quantizer = std::make_unique<RaBitQQuantizer>(params.metric, dim, 42);
        n_factors = 2;
    } else {
        quantizer = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, scan_bits, /*seed=*/42);
    }

    // For RaBitQ: sign codes are centroid-relative and cannot be used for
    // partitioning. Train a separate plain PQ (8-bit) for routing/partitioning.
    // For PQ/PRQ: the scan quantizer IS the routing quantizer.
    std::unique_ptr<PqQuantizer> routing_quantizer;
    if (params.quantizer_type == "rabitq") {
        routing_quantizer = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, /*bits=*/8, /*seed=*/42);
    }

    // Train on a sample.
    {
        const uint32_t train_n = std::min<uint64_t>(20'000, n);
        std::vector<float> sample(static_cast<size_t>(train_n) * dim);
        std::fseek(f, 8, SEEK_SET);  // skip header
        const uint64_t to_read = static_cast<uint64_t>(train_n) * dim;
        if (std::fread(sample.data(), sizeof(float), to_read, f) != to_read) {
            std::fclose(f);
            throw Error(ErrorCode::IoError, "IVFTreeIndex::build: failed to read training sample");
        }
        spdlog::info("[sextant] build_tree: training {} on {} samples",
                     params.quantizer_type, train_n);
        quantizer->train(sample.data(), train_n);
        if (routing_quantizer) {
            spdlog::info("[sextant] build_tree: training routing PQ (8-bit) on {} samples",
                         train_n);
            routing_quantizer->train(sample.data(), train_n);
        }
    }

    // The quantizer used for partitioning and routing codes.
    // For RaBitQ: the routing PQ. For PQ/PRQ: the scan quantizer itself.
    PqQuantizer& routing_q = routing_quantizer ? *routing_quantizer : *quantizer;
    const uint32_t routing_code_size = routing_q.code_size();

    // Encode all vectors with the routing quantizer (for partitioning).
    // Also encode with the scan quantizer (for leaf codes) if different.
    const bool is_rabitq_build = (params.quantizer_type == "rabitq");
    const uint32_t encode_stride = is_rabitq_build
        ? static_cast<RaBitQQuantizer&>(*quantizer).full_code_size()
        : quantizer->code_size();
    const uint32_t code_size = routing_code_size;
    spdlog::info("[sextant] build_tree: encoding {} vectors (code_size={})",
                 n, code_size);

    // Encode all vectors. For RaBitQ, encode with the routing PQ (for
    // partitioning); scan codes are re-encoded per-leaf later. For PQ/PRQ,
    // the same codes serve both routing and scanning.
    std::vector<uint8_t> all_codes(static_cast<size_t>(n) * encode_stride);
    std::vector<RowId> all_rowids(n);
    for (uint64_t i = 0; i < n; ++i) all_rowids[i] = static_cast<RowId>(i);

    {
        const uint32_t hw = cfg.num_threads > 0
            ? cfg.num_threads
            : std::thread::hardware_concurrency();

        const uint32_t chunk_n = 100'000;
        std::vector<float> buf(static_cast<size_t>(chunk_n) * dim);

        std::fseek(f, 8, SEEK_SET);
        uint64_t offset = 0;
        while (offset < n) {
            const uint32_t take = static_cast<uint32_t>(
                std::min<uint64_t>(chunk_n, n - offset));
            const uint64_t to_read = static_cast<uint64_t>(take) * dim;
            if (std::fread(buf.data(), sizeof(float), to_read, f) != to_read) {
                std::fclose(f);
                throw Error(ErrorCode::IoError,
                            "IVFTreeIndex::build: failed to read chunk at offset " +
                                std::to_string(offset));
            }

            const uint32_t n_threads = std::min(hw, take);
            const uint32_t per = (take + n_threads - 1) / n_threads;
            std::vector<std::future<void>> futs;
            for (uint32_t t = 0; t < n_threads; ++t) {
                const uint32_t start = t * per;
                const uint32_t end = std::min(start + per, take);
                if (start >= end) break;
                futs.push_back(std::async(std::launch::async,
                    [&buf, dim, &all_codes, &routing_q, encode_stride, offset,
                     is_rabitq_build, &quantizer]
                    (uint32_t s, uint32_t e) {
                        for (uint32_t i = s; i < e; ++i) {
                            const float* vec = buf.data() +
                                static_cast<size_t>(i) * dim;
                            uint8_t* code_out = all_codes.data() +
                                static_cast<size_t>(offset + i) * encode_stride;
                            if (is_rabitq_build) {
                                // Encode with routing PQ for partitioning.
                                routing_q.encode(vec, code_out);
                            } else {
                                quantizer->encode(vec, code_out);
                            }
                        }
                    }, start, end));
            }
            for (auto& fut : futs) fut.get();
            offset += take;
        }
    }
    std::fclose(f);

    spdlog::info("[sextant] build_tree: encoded {} vectors", n);

    // --- 4. Leaf-granularity k-means ---
    // Partition all N vectors into n_leaves clusters using PQ code distances.
    spdlog::info("[sextant] build_tree: leaf k-means (K={})", n_leaves);
    auto pa = partition_codes(routing_q, all_codes.data(),
                              static_cast<uint32_t>(n), code_size,
                              n_leaves,
                              /*closure_factor=*/1.0f,  // no closure at leaf level
                              /*iterations=*/10,
                              /*num_threads=*/cfg.num_threads,
                              /*seed=*/42,
                              /*balance_factor=*/params.partition_balance_factor,
                              /*closure_epsilon=*/params.closure_epsilon);

    // --- 5. Compute FP16 centroids for each leaf ---
    // Decode the PQ centroid codes to FP16 vectors for routing.
    std::vector<std::vector<float16_t>> leaf_centroids_fp16(n_leaves);
    for (uint32_t l = 0; l < n_leaves; ++l) {
        leaf_centroids_fp16[l].resize(dim);
        if (l < pa.centroids.size()) {
            // Decode the PQ centroid code to a float vector, then cast to FP16.
            std::vector<float> centroid_f32(dim);
            routing_q.decode_code(pa.centroids[l].data(), centroid_f32.data());
            cast_fp32_to_fp16(centroid_f32.data(), leaf_centroids_fp16[l].data(), dim);
        }
    }

    // --- 6. Cluster leaf centroids into K₁ root groups ---
    // Level-1 k-means: group the n_leaves leaf centroids into k_root clusters.
    // This is a tiny k-means on n_leaves FP16 points — trivial.
    spdlog::info("[sextant] build_tree: root clustering (K₁={})", k_root);

    // Convert leaf centroids to float for the level-1 k-means.
    std::vector<float> leaf_centroids_f32(static_cast<size_t>(n_leaves) * dim);
    for (uint32_t l = 0; l < n_leaves; ++l) {
        for (uint16_t d = 0; d < dim; ++d) {
            leaf_centroids_f32[static_cast<size_t>(l) * dim + d] =
                leaf_centroids_fp16[l][d];
        }
    }

    // Assign each leaf to the nearest root centroid (simple k-means on floats).
    std::vector<std::vector<uint32_t>> root_groups(k_root);
    std::vector<float> root_centroids_f32(static_cast<size_t>(k_root) * dim);

    // Initialize root centroids by sampling leaf centroids evenly.
    for (uint32_t c = 0; c < k_root; ++c) {
        const uint32_t src = (c * n_leaves) / k_root;
        std::memcpy(&root_centroids_f32[c * dim],
                    &leaf_centroids_f32[src * dim], dim * sizeof(float));
    }

    // 10 iterations of k-means (on float32, using L2sq).
    for (uint32_t iter = 0; iter < 10; ++iter) {
        // Assign.
        root_groups.assign(k_root, {});
        for (uint32_t l = 0; l < n_leaves; ++l) {
            const float* lp = &leaf_centroids_f32[l * dim];
            float best_d = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k_root; ++c) {
                const float d = simd::l2sq_f32(lp, &root_centroids_f32[c * dim], dim);
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            root_groups[best_c].push_back(l);
        }
        // Update: mean of assigned leaf centroids.
        for (uint32_t c = 0; c < k_root; ++c) {
            if (root_groups[c].empty()) {
                // Reseed from a random leaf.
                const uint32_t src = (c * 7919 + 1) % n_leaves;
                std::memcpy(&root_centroids_f32[c * dim],
                            &leaf_centroids_f32[src * dim], dim * sizeof(float));
                continue;
            }
            std::vector<double> sum(dim, 0.0);
            for (uint32_t l : root_groups[c]) {
                for (uint16_t d = 0; d < dim; ++d) {
                    sum[d] += leaf_centroids_f32[l * dim + d];
                }
            }
            const double inv = 1.0 / root_groups[c].size();
            for (uint16_t d = 0; d < dim; ++d) {
                root_centroids_f32[c * dim + d] = static_cast<float>(sum[d] * inv);
            }
        }
    }

    // Cast root centroids to FP16.
    std::vector<std::vector<float16_t>> root_centroids_fp16(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        root_centroids_fp16[c].resize(dim);
        cast_fp32_to_fp16(&root_centroids_f32[c * dim],
                          root_centroids_fp16[c].data(), dim);
    }

    // --- 7. Write the tree file ---
    // Layout:
    //   Page 0-1: superblock (active + shadow)
    //   Page 2: allocation bitmap (1 page for now — covers 128GB)
    //   Page 3+: leaves (contiguous within root groups), then root node,
    //            then codebook blob, then config blob.

    const PageId bitmap_page = 2;
    const uint32_t bitmap_pages = 1;  // 32768 bits = 128GB at 4KB pages

    PageFile file(output_path);
    PageAllocator alloc;
    // Pre-size the file for the bitmap.
    file.truncate(bitmap_page + bitmap_pages);
    alloc.init(file, bitmap_page, bitmap_pages);

    // Determine tree depth.
    const uint16_t depth = (n_leaves <= k_root) ? 1 : 2;

    // --- Write leaves ---
    spdlog::info("[sextant] build_tree: writing {} leaves", n_leaves);

    const uint32_t cpb = (scan_bits == 4) ? 32 : 16;
    const uint32_t bb = m4 * 16;  // [m][16] for both 4-bit and 8-bit
    const bool is_rabitq = (params.quantizer_type == "rabitq");

    // For RaBitQ: re-open the base file to re-encode relative to each leaf
    // centroid. RaBitQ sign codes are residual (centroid-relative).
    FILE* base_f2 = nullptr;
    std::vector<float> leaf_centroid_f32;
    if (is_rabitq) {
        base_f2 = std::fopen(base_path.c_str(), "rb");
        if (!base_f2) {
            throw Error(ErrorCode::IoError,
                        "IVFTreeIndex::build: cannot reopen base for RaBitQ");
        }
        leaf_centroid_f32.resize(dim);
    }

    // Track each leaf's (page, pages) for the parent node.
    struct LeafInfo {
        PageId page;
        uint32_t pages;
    };
    std::vector<LeafInfo> leaf_infos(n_leaves);

    for (uint32_t l = 0; l < n_leaves; ++l) {
        const auto& members = pa.shards[l];
        const uint32_t count = static_cast<uint32_t>(members.size());

        if (count == 0) {
            leaf_infos[l] = {kInvalidPage, 0};
            continue;
        }

        const uint32_t npg = leaf_extent_pages(count, m4, scan_bits, n_factors);
        const PageId page = alloc.alloc_extent(file, npg);
        leaf_infos[l] = {page, npg};

        // Build the leaf buffer.
        std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);

        // Header.
        auto* lh = reinterpret_cast<TreeLeafHeader*>(buf.data());
        lh->count = count;
        lh->tombstone_count = 0;
        lh->m4 = m4;
        lh->pq_bits = scan_bits;
        lh->n_factors = n_factors;
        lh->block_bytes = bb;
        lh->codes_per_block = cpb;
        lh->extent_pages = npg;

        // FastScan code blocks.
        const uint32_t n_blocks = (count + cpb - 1) / cpb;
        uint8_t* codes_out = buf.data() + leaf_codes_offset();

        // For RaBitQ: decode the leaf centroid and re-encode each vector
        // relative to it. For PQ/PRQ: use the pre-encoded absolute codes.
        std::vector<uint8_t> rabitq_packed;
        std::vector<float> rabitq_factors;  // per-leaf, for RaBitQ
        if (is_rabitq) {
            // Decode the leaf centroid (PQ code → FP32).
            if (l < pa.centroids.size()) {
                routing_q.decode_code(pa.centroids[l].data(),
                                      leaf_centroid_f32.data());
            } else {
                std::fill(leaf_centroid_f32.begin(),
                          leaf_centroid_f32.end(), 0.0f);
            }
            rabitq_packed.resize(
                static_cast<RaBitQQuantizer&>(*quantizer).full_code_size());
            rabitq_factors.resize(static_cast<size_t>(count) * 2);
        }

        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t base = b * cpb;
            uint8_t* blk = codes_out + static_cast<uint64_t>(b) * bb;
            std::memset(blk, 0, bb);

            for (uint32_t j = 0; j < cpb; ++j) {
                const uint32_t gi = base + j;
                if (gi >= count) break;
                const uint32_t gid = members[gi];

                // Get the code bytes for this vector.
                const uint8_t* code;
                if (is_rabitq) {
                    // Read the raw vector and encode relative to centroid.
                    std::vector<float> vec(dim);
                    std::fseek(base_f2, 8 + static_cast<long>(gid) * dim * sizeof(float),
                               SEEK_SET);
                    std::fread(vec.data(), sizeof(float), dim, base_f2);
                    static_cast<RaBitQQuantizer&>(*quantizer).encode_with_centroid(
                        vec.data(), leaf_centroid_f32.data(), rabitq_packed.data());
                    code = rabitq_packed.data();
                    // Stash the 2 factors.
                    const uint32_t fac_off =
                        static_cast<RaBitQQuantizer&>(*quantizer).factors_offset();
                    std::memcpy(&rabitq_factors[gi * 2],
                                rabitq_packed.data() + fac_off, 2 * sizeof(float));
                } else {
                    code = all_codes.data() +
                        static_cast<size_t>(gid) * code_size;
                }

                if (scan_bits == 4) {
                    // 4-bit: FAISS perm0 layout. Byte k of segment s:
                    // lo nibble = vec[k], hi nibble = vec[16+k].
                    const uint8_t nibble_byte_idx = j % 16;
                    const bool is_hi = (j >= 16);
                    for (uint16_t s = 0; s < m4; ++s) {
                        uint8_t nib;
                        if (s / 2 < code_size) {
                            nib = (s % 2 == 0)
                                ? (code[s / 2] & 0x0F)
                                : (code[s / 2] >> 4);
                        } else {
                            nib = 0;
                        }
                        if (is_hi) {
                            blk[s * 16 + nibble_byte_idx] |= (nib << 4);
                        } else {
                            blk[s * 16 + nibble_byte_idx] |= nib;
                        }
                    }
                } else {
                    // 8-bit: one byte per segment per vector. No nibble packing.
                    // Block layout: [m][16], byte j of segment s = vec[j].
                    for (uint16_t s = 0; s < m4; ++s) {
                        blk[s * 16 + j] = (s < code_size) ? code[s] : 0;
                    }
                }
            }
        }

        // Row IDs.
        RowId* rids = reinterpret_cast<RowId*>(
            buf.data() + leaf_rowids_offset(n_blocks, bb));
        for (uint32_t i = 0; i < count; ++i) {
            rids[i] = static_cast<RowId>(members[i]);
        }

        // Factors (RaBitQ only).
        if (n_factors > 0) {
            float* fac = reinterpret_cast<float*>(
                buf.data() + leaf_factors_offset(n_blocks, bb, count));
            // rabitq_factors was populated during the per-vector re-encoding above.
            std::memcpy(fac, rabitq_factors.data(),
                        rabitq_factors.size() * sizeof(float));
        }

        file.write_pages(page, npg, buf.data());
    }

    if (base_f2) std::fclose(base_f2);

    // --- Write root node ---
    // For depth=1: root children are leaves.
    // For depth=2: root children are level-1 internal nodes (group of leaves).
    spdlog::info("[sextant] build_tree: writing root node (depth={})", depth);

    std::vector<RootChildData> root_child_data(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        root_child_data[c].centroid = root_centroids_fp16[c];
    }

    if (depth == 1) {
        // Root children = leaves. Assign each leaf to the nearest root centroid.
        // root_groups already has this mapping.
        for (uint32_t c = 0; c < k_root; ++c) {
            if (root_groups[c].empty()) {
                root_child_data[c].is_leaf = 1;
                root_child_data[c].page = kInvalidPage;
                root_child_data[c].pages = 0;
                continue;
            }
            // If exactly 1 leaf in this group, point directly to it.
            if (root_groups[c].size() == 1) {
                const uint32_t l = root_groups[c][0];
                root_child_data[c].is_leaf = 1;
                root_child_data[c].page = leaf_infos[l].page;
                root_child_data[c].pages = leaf_infos[l].pages;
            } else {
                // Multiple leaves in one group at depth=1 — shouldn't happen
                // if n_leaves <= k_root. Point to first leaf.
                const uint32_t l = root_groups[c][0];
                root_child_data[c].is_leaf = 1;
                root_child_data[c].page = leaf_infos[l].page;
                root_child_data[c].pages = leaf_infos[l].pages;
            }
        }
    } else {
        // Depth=2: write a level-1 internal node for each root group, then
        // the root node points to those.
        for (uint32_t c = 0; c < k_root; ++c) {
            const auto& group = root_groups[c];
            if (group.empty()) {
                root_child_data[c].is_leaf = 0;
                root_child_data[c].page = kInvalidPage;
                root_child_data[c].pages = 0;
                continue;
            }

            // Build the level-1 internal node.
            const uint32_t n_children = static_cast<uint32_t>(group.size());
            const uint32_t npg = node_extent_pages(dim, n_children);
            const PageId page = alloc.alloc_extent(file, npg);

            std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);
            auto* nh = reinterpret_cast<TreeNodeHeader*>(buf.data());
            nh->n_children = n_children;
            nh->extent_pages = npg;
            nh->dim = dim;
            nh->reserved = 0;

            const uint32_t cesize = child_entry_size(dim);
            uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
            for (uint32_t j = 0; j < n_children; ++j) {
                const uint32_t l = group[j];
                auto* ce = reinterpret_cast<ChildEntry*>(p);
                ce->child_page = leaf_infos[l].page;
                ce->child_pages = leaf_infos[l].pages;
                ce->is_leaf = 1;
                ce->reserved = 0;
                // Centroid.
                float16_t* cent = reinterpret_cast<float16_t*>(
                    p + sizeof(ChildEntry));
                std::memcpy(cent, leaf_centroids_fp16[l].data(),
                            dim * sizeof(float16_t));
                // Filter summary is zero (reserved).
                p += cesize;
            }

            file.write_pages(page, npg, buf.data());

            root_child_data[c].is_leaf = 0;
            root_child_data[c].page = page;
            root_child_data[c].pages = npg;
        }
    }

    // Write the root node itself.
    const uint32_t root_npg = node_extent_pages(dim, k_root);
    const PageId root_page = alloc.alloc_extent(file, root_npg);

    {
        std::vector<uint8_t> buf(static_cast<size_t>(root_npg) * kPageSize, 0);
        auto* rh = reinterpret_cast<TreeNodeHeader*>(buf.data());
        rh->n_children = k_root;
        rh->extent_pages = root_npg;
        rh->dim = dim;
        rh->reserved = 0;

        const uint32_t cesize = child_entry_size(dim);
        uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
        for (uint32_t c = 0; c < k_root; ++c) {
            auto* ce = reinterpret_cast<ChildEntry*>(p);
            ce->child_page = root_child_data[c].page;
            ce->child_pages = root_child_data[c].pages;
            ce->is_leaf = root_child_data[c].is_leaf;
            ce->reserved = 0;
            float16_t* cent = reinterpret_cast<float16_t*>(
                p + sizeof(ChildEntry));
            std::memcpy(cent, root_child_data[c].centroid.data(),
                        dim * sizeof(float16_t));
            p += cesize;
        }
        file.write_pages(root_page, root_npg, buf.data());
    }

    // --- Write codebook ---
    std::vector<uint8_t> qblob;
    quantizer->serialize(qblob);
    // Prefix with the serialized size so deserialization can trim page padding.
    const uint64_t qblob_size = qblob.size();
    std::vector<uint8_t> qblob_sized(sizeof(qblob_size) + qblob.size());
    std::memcpy(qblob_sized.data(), &qblob_size, sizeof(qblob_size));
    std::memcpy(qblob_sized.data() + sizeof(qblob_size), qblob.data(), qblob.size());
    const uint32_t cb_npg = static_cast<uint32_t>(
        (qblob_sized.size() + kPageSize - 1) / kPageSize);
    const PageId cb_page = alloc.alloc_extent(file, cb_npg);
    {
        std::vector<uint8_t> buf(static_cast<size_t>(cb_npg) * kPageSize, 0);
        std::memcpy(buf.data(), qblob_sized.data(), qblob_sized.size());
        file.write_pages(cb_page, cb_npg, buf.data());
    }

    // --- Write config blob (TOML manifest) ---
    TreeManifest manifest;
    manifest.dim = dim;
    manifest.m4 = m4;
    manifest.scan_pq_bits = scan_bits;
    manifest.quantizer_type = params.quantizer_type;
    manifest.prq_nsplits = (params.quantizer_type == "prq")
        ? static_cast<uint32_t>(static_cast<ProductResidualQuantizer&>(*quantizer).nsplits())
        : 0;
    manifest.depth = depth;
    manifest.k_root = k_root;
    manifest.leaf_capacity = leaf_cap;
    manifest.n_leaves = n_leaves;
    manifest.n_probe_l0 = cfg.n_probe_l0 > 0 ? cfg.n_probe_l0
        : static_cast<uint32_t>(std::max(1.0, 2.0 * std::sqrt(double(k_root))));
    manifest.n_probe_ln = cfg.n_probe_ln > 0 ? cfg.n_probe_ln : 4;
    manifest.adaptive_probe_gap = cfg.adaptive_probe_gap;
    manifest.median_lid = cfg.median_lid;
    manifest.balance_factor = params.partition_balance_factor;

    std::string cfg_toml = manifest_to_toml(manifest);
    const uint32_t cfg_npg = static_cast<uint32_t>(
        (cfg_toml.size() + kPageSize - 1) / kPageSize);
    const PageId cfg_page = alloc.alloc_extent(file, cfg_npg);
    {
        std::vector<uint8_t> buf(static_cast<size_t>(cfg_npg) * kPageSize, 0);
        std::memcpy(buf.data(), cfg_toml.data(), cfg_toml.size());
        file.write_pages(cfg_page, cfg_npg, buf.data());
    }

    // --- Flush bitmap ---
    alloc.flush_bitmap(file);

    // --- Write superblock ---
    Superblock sb;
    sb.init_fresh(bitmap_page, bitmap_pages);
    sb.set_root(root_page, root_npg);
    sb.set_depth(depth);
    sb.set_n_leaves(n_leaves);
    sb.set_n_pages(file.num_pages());
    sb.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    sb.set_bitmap(bitmap_page, bitmap_pages);
    sb.set_codebook(cb_page, cb_npg);
    sb.set_config(cfg_page, cfg_npg);
    sb.commit(file);
    file.sync();

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    spdlog::info("[sextant] build_tree complete: N={} k_root={} n_leaves={} "
                 "depth={} in {:.2f}s (file={} pages)",
                 n, k_root, n_leaves, depth, secs, file.num_pages());

    BuildResult result;
    result.index_path = output_path;
    result.n_vectors = n;
    result.dim = dim;
    result.R = params.R;
    result.L_build = params.L_build;
    result.pq_m = m4;
    result.pq_bits = scan_bits;
    result.build_time_sec = secs;
    return result;
}

}  // namespace sextant::tree
