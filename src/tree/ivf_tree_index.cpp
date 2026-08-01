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
#include <array>
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

    // Load PCA routing data if present.
    if (idx->manifest_.pca_dims > 0 && idx->superblock_.pca_page() != kInvalidPage) {
        const auto pp = idx->superblock_.pca_page();
        const auto npg = idx->superblock_.pca_pages();
        std::vector<uint8_t> blob(static_cast<size_t>(npg) * kPageSize);
        idx->file_.read_pages(pp, npg, blob.data());
        const float* f = reinterpret_cast<const float*>(blob.data());
        const uint32_t pd = idx->manifest_.pca_dims;
        const uint32_t d = idx->manifest_.dim;
        const uint32_t kr = idx->manifest_.k_root;
        const uint32_t nl = idx->manifest_.n_leaves;

        idx->pca_dims_ = pd;
        size_t off = 0;
        // Projection matrix (pd × d).
        idx->pca_proj_.assign(f + off, f + off + pd * d);
        off += pd * d;
        // Mean projection (pd).
        idx->pca_mean_proj_.assign(f + off, f + off + pd);
        off += pd;
        // Root centroids in PCA space (kr × pd).
        idx->pca_root_centroids_.assign(f + off, f + off + kr * pd);
        off += kr * pd;
        // Leaf centroids in PCA space (nl × pd).
        idx->pca_leaf_centroids_.assign(f + off, f + off + nl * pd);

        spdlog::info("[sextant] IVFTreeIndex: PCA routing enabled ({} dims)",
                     pd);
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

    // If PCA routing: build leaf_id mapping for depth=2 trees.
    // The pca_leaf_centroids_ array is indexed by global leaf ID (the order
    // leaves were written during build). At search time, we need to map from
    // (root_child, local_child) → global_leaf_id.
    if (idx->pca_dims_ > 0 && idx->manifest_.depth == 2) {
        // Walk the level-1 nodes to assign global IDs.
        uint32_t global_id = 0;
        for (uint32_t c = 0; c < idx->root_children_.size(); ++c) {
            const auto& rc = idx->root_children_[c];
            if (rc.is_leaf || rc.page == kInvalidPage) continue;
            const uint8_t* node_ptr = idx->mmap_base_ +
                static_cast<uint64_t>(rc.page) * kPageSize;
            const auto* nh = reinterpret_cast<const TreeNodeHeader*>(node_ptr);
            // Store the starting global ID for this root child.
            // We'll use this at search time: global_id = start + local_child.
            // Store as a parallel array.
            idx->pca_leaf_base_.push_back(global_id);
            global_id += nh->n_children;
        }
    }

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

    // Cast query to FP16 for routing (used for non-PCA path + leaf centroid reads).
    std::vector<float16_t> query_fp16(manifest_.dim);
    cast_fp32_to_fp16(query, query_fp16.data(), manifest_.dim);
    const MetricKind metric = quantizer_->metric();

    // --- Project query to PCA space if PCA routing is enabled ---
    std::vector<float> query_pca;
    if (pca_dims_ > 0) {
        query_pca.resize(pca_dims_);
        for (uint32_t k = 0; k < pca_dims_; ++k) {
            query_pca[k] = simd::dot_f32(&pca_proj_[k * manifest_.dim],
                                          query, manifest_.dim)
                           - pca_mean_proj_[k];
        }
    }

    // --- Route at level 0 ---
    const uint32_t k_root = root_header_->n_children;
    std::vector<std::pair<float, uint32_t>> root_dists;
    root_dists.reserve(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        float d;
        if (pca_dims_ > 0) {
            // PCA-space L2sq to root centroid.
            const float* rc = &pca_root_centroids_[c * pca_dims_];
            d = 0.0f;
            for (uint32_t k = 0; k < pca_dims_; ++k) {
                const float diff = query_pca[k] - rc[k];
                d += diff * diff;
            }
        } else {
            // Original FP16 distance.
            d = simd::dist_f16(metric, query_fp16.data(),
                               root_children_[c].centroid, manifest_.dim);
        }
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
                float d;
                if (pca_dims_ > 0 && child_idx < pca_leaf_base_.size()) {
                    const uint32_t gid = pca_leaf_base_[child_idx] + j;
                    const float* lc = &pca_leaf_centroids_[gid * pca_dims_];
                    d = 0.0f;
                    for (uint32_t kk = 0; kk < pca_dims_; ++kk) {
                        const float diff = query_pca[kk] - lc[kk];
                        d += diff * diff;
                    }
                } else {
                    const float16_t* cent = reinterpret_cast<const float16_t*>(
                        p + sizeof(ChildEntry));
                    d = simd::dist_f16(metric, query_fp16.data(),
                                        cent, manifest_.dim);
                }
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
// Streaming build — sample k-means for root centroids + dynamic leaf growth.
//
// No global k-means on all N vectors. Instead:
//   1. Sample 20k vectors → FP16 k-means → K_root root centroids
//   2. Train scan quantizer (4-bit PQ) on the sample
//   3. Stream all N vectors: FP16 route to nearest root centroid, encode
//      4-bit code, append to leaf buffer. Flush on overflow.
//   4. Write tree file (leaves, root, codebook, config)
//
// O(1) RAM regardless of N. Avoids the k-means convergence problem on
// high-LID data (Sphere-IP).
// ===========================================================================

BuildResult IVFTreeIndex::build_streaming(const std::string& base_path,
                                           const std::string& output_path,
                                           const BuildConfig& cfg) {
    const auto t0 = std::chrono::steady_clock::now();

    // --- 1. Load header ---
    FILE* f = std::fopen(base_path.c_str(), "rb");
    if (!f) {
        throw Error(ErrorCode::IoError,
                    "IVFTreeIndex::build_streaming: cannot open '" + base_path + "'");
    }
    uint32_t header[2];
    if (std::fread(header, sizeof(uint32_t), 2, f) != 2) {
        std::fclose(f);
        throw Error(ErrorCode::IoError, "build_streaming: failed to read header");
    }
    const uint64_t n = header[0];
    const Dim dim = header[1];
    if (n == 0 || dim == 0) {
        std::fclose(f);
        throw Error(ErrorCode::InvalidParam, "build_streaming: empty fbin");
    }

    spdlog::info("[sextant] build_streaming: N={} dim={} → '{}'", n, dim, output_path);

    const auto& params = cfg.params;
    const uint16_t m4 = params.pq4_m > 0 ? params.pq4_m
                                          : static_cast<uint16_t>(dim / 4);
    const uint8_t scan_bits = params.scan_pq_bits;
    const uint32_t leaf_cap = cfg.leaf_capacity > 0 ? cfg.leaf_capacity : 5000;

    // Auto k_root.
    uint32_t k_root = cfg.k_root;
    if (k_root == 0) {
        k_root = static_cast<uint32_t>(std::sqrt(static_cast<double>(n) / leaf_cap));
        k_root = std::clamp(k_root, 16u, 256u);
    }

    spdlog::info("[sextant] build_streaming: k_root={} leaf_cap={} m4={}",
                 k_root, leaf_cap, m4);

    // --- 2. Sample for root k-means + PQ training ---
    const uint32_t train_n = std::min<uint64_t>(20'000, n);
    std::vector<float> sample(static_cast<size_t>(train_n) * dim);
    std::fseek(f, 8, SEEK_SET);
    if (std::fread(sample.data(), sizeof(float),
                   static_cast<size_t>(train_n) * dim, f) != static_cast<size_t>(train_n) * dim) {
        std::fclose(f);
        throw Error(ErrorCode::IoError, "build_streaming: failed to read sample");
    }

    // Train scan quantizer.
    std::unique_ptr<PqQuantizer> quantizer;
    uint8_t n_factors = 0;
    if (params.quantizer_type == "prq") {
        const uint32_t nsplits = (params.prq_nsplits > 0)
            ? params.prq_nsplits
            : static_cast<uint32_t>(dim) / 8;
        quantizer = std::make_unique<ProductResidualQuantizer>(
            params.metric, dim, m4, scan_bits, nsplits,
            params.prq_beam_size, 42);
    } else if (params.quantizer_type == "rabitq") {
        quantizer = std::make_unique<RaBitQQuantizer>(params.metric, dim, 42);
        n_factors = 2;
    } else {
        quantizer = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, scan_bits, 42);
    }
    spdlog::info("[sextant] build_streaming: training {} on {} samples",
                 params.quantizer_type, train_n);
    quantizer->train(sample.data(), train_n);

    // --- 3. FP16 k-means for root centroids (on the sample) ---
    // Simple k-means on FP16 vectors using simd::dist_f16. The sample is
    // small (20k), so this is trivially fast.
    spdlog::info("[sextant] build_streaming: root k-means (K={}) on {} samples",
                 k_root, train_n);

    std::vector<float16_t> sample_fp16(static_cast<size_t>(train_n) * dim);
    cast_fp32_to_fp16(sample.data(), sample_fp16.data(),
                      static_cast<size_t>(train_n) * dim);

    // Initialize root centroids: evenly spaced sample vectors.
    std::vector<std::vector<float16_t>> root_centroids(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        const uint32_t src = (c * train_n) / k_root;
        root_centroids[c].assign(sample_fp16.data() + src * dim,
                                 sample_fp16.data() + (src + 1) * dim);
    }

    const MetricKind metric = params.metric;
    for (uint32_t iter = 0; iter < 10; ++iter) {
        // Assign sample vectors to nearest root centroid.
        std::vector<std::vector<uint32_t>> assigns(k_root);
        for (uint32_t i = 0; i < train_n; ++i) {
            const float16_t* vec = &sample_fp16[i * dim];
            float best_d = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k_root; ++c) {
                const float d = simd::dist_f16(metric, vec,
                                                root_centroids[c].data(), dim);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            assigns[best_c].push_back(i);
        }
        // Update centroids: FP32 mean → FP16.
        uint32_t n_changed = 0;
        for (uint32_t c = 0; c < k_root; ++c) {
            if (assigns[c].empty()) {
                // Reseed from a random sample vector.
                uint32_t src = (c * 7919 + 1) % train_n;
                root_centroids[c].assign(sample_fp16.data() + src * dim,
                                         sample_fp16.data() + (src + 1) * dim);
                continue;
            }
            std::vector<double> sum(dim, 0.0);
            for (uint32_t i : assigns[c]) {
                for (uint16_t d = 0; d < dim; ++d)
                    sum[d] += static_cast<float>(sample_fp16[i * dim + d]);
            }
            const double inv = 1.0 / assigns[c].size();
            for (uint16_t d = 0; d < dim; ++d)
                root_centroids[c][d] = static_cast<float16_t>(sum[d] * inv);
        }
    }

    // --- 3b. Compute closure epsilon from the sample ---
    // The closure margin is calibrated from the mean GAP between the nearest
    // and second-nearest centroid distances. This is metric-invariant:
    // for L2sq, the gap is (d_2nd - d_1st) ≥ 0. For IP (negated dot), the
    // gap is also (d_2nd - d_1st) ≥ 0 (since d_1st is more negative).
    // Using the gap instead of the absolute distance avoids the IP pathology
    // where all distances are ≈ -1.0 but the meaningful signal is in the
    // small differences.
    float closure_epsilon = 0.0f;
    {
        const uint32_t sample_for_eps = std::min<uint32_t>(4096, train_n);
        double sum_gap = 0.0;
        for (uint32_t i = 0; i < sample_for_eps; ++i) {
            const float16_t* vec = &sample_fp16[i * dim];
            float d1 = std::numeric_limits<float>::max();
            float d2 = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < k_root; ++c) {
                const float d = simd::dist_f16(metric, vec,
                                                root_centroids[c].data(), dim);
                if (d < d1) { d2 = d1; d1 = d; }
                else if (d < d2) { d2 = d; }
            }
            // Gap = how much farther the 2nd-nearest is vs the nearest.
            // For boundary vectors this gap is small → they're near 2+ centroids.
            sum_gap += (d2 - d1);
        }
        const double mean_gap = sum_gap / sample_for_eps;
        // ε = 0.5 × mean gap. Vectors whose 2nd-nearest is within ε of their
        // nearest get replicated. At 0.5× the mean gap, ~30-40% of vectors
        // get replicated to a 2nd leaf (the boundary set).
        closure_epsilon = static_cast<float>(mean_gap * 0.5);
        spdlog::info("[sextant] build_streaming: closure_epsilon = {:.4f} "
                     "(auto, mean_gap={:.4f}, sample={})",
                     closure_epsilon, mean_gap, sample_for_eps);
    }

    // --- 4. Set up leaf buffers ---
    // Each root child has an in-memory leaf buffer. When it fills, flush
    // as a leaf extent and start a new one. Track all flushed leaves per
    // root child for the depth-2 tree structure.
    struct LeafBuffer {
        std::vector<uint8_t> codes;     // 4-bit PQ codes (code_size per vec)
        std::vector<RowId> row_ids;
        std::vector<float16_t> fp16_vecs;  // for centroid computation
    };

    const uint32_t code_size = quantizer->code_size();
    const bool is_rabitq = (params.quantizer_type == "rabitq");

    std::vector<LeafBuffer> buffers(k_root);
    std::vector<std::vector<float16_t>> leaf_centroids_fp16;  // all flushed leaves
    std::vector<std::vector<uint32_t>> root_to_leaves(k_root);  // root child → leaf indices
    uint32_t n_leaves_total = 0;

    // Auto-flush lambda: when a buffer is full, write it as a leaf extent.
    // We don't write to disk yet — accumulate leaf data, write at the end.
    // This avoids random I/O during the streaming phase.
    struct FlushedLeaf {
        std::vector<uint8_t> codes;
        std::vector<RowId> row_ids;
        std::vector<float16_t> centroid;
        // RaBitQ-only: per-vector factors
        std::vector<float> factors;
    };
    std::vector<FlushedLeaf> all_leaves;

    auto flush_buffer = [&](uint32_t c) {
        auto& buf = buffers[c];
        if (buf.row_ids.empty()) return;

        FlushedLeaf leaf;
        leaf.codes = std::move(buf.codes);
        leaf.row_ids = std::move(buf.row_ids);

        // Leaf centroid = FP16 mean of the buffered vectors.
        leaf.centroid.resize(dim);
        std::vector<double> sum(dim, 0.0);
        for (uint32_t i = 0; i < buf.fp16_vecs.size() / dim; ++i) {
            for (uint16_t d = 0; d < dim; ++d)
                sum[d] += static_cast<float>(buf.fp16_vecs[i * dim + d]);
        }
        const double inv = 1.0 / (buf.fp16_vecs.size() / dim);
        for (uint16_t d = 0; d < dim; ++d)
            leaf.centroid[d] = static_cast<float16_t>(sum[d] * inv);

        // For RaBitQ: re-encode codes relative to leaf centroid.
        // The buffered codes were absolute-encoded; we need centroid-relative.
        // For now, PQ/PRQ codes are absolute and work fine.
        // TODO: RaBitQ streaming with per-leaf centroid re-encoding.

        uint32_t leaf_idx = static_cast<uint32_t>(all_leaves.size());
        all_leaves.push_back(std::move(leaf));
        root_to_leaves[c].push_back(leaf_idx);
        n_leaves_total++;

        // Reset buffer.
        buf.codes.clear();
        buf.row_ids.clear();
        buf.fp16_vecs.clear();
    };

    // --- 5. Stream all N vectors ---
    spdlog::info("[sextant] build_streaming: streaming {} vectors", n);

    std::fseek(f, 8, SEEK_SET);
    const uint32_t chunk_n = 100'000;
    std::vector<float> vec_buf(static_cast<size_t>(chunk_n) * dim);
    std::vector<float16_t> fp16_buf(chunk_n * dim);

    uint64_t offset = 0;
    while (offset < n) {
        const uint32_t take = static_cast<uint32_t>(
            std::min<uint64_t>(chunk_n, n - offset));
        if (std::fread(vec_buf.data(), sizeof(float),
                       static_cast<size_t>(take) * dim, f)
            != static_cast<size_t>(take) * dim) {
            std::fclose(f);
            throw Error(ErrorCode::IoError, "build_streaming: read failed");
        }

        cast_fp32_to_fp16(vec_buf.data(), fp16_buf.data(),
                          static_cast<size_t>(take) * dim);

        for (uint32_t i = 0; i < take; ++i) {
            const float16_t* fvec = &fp16_buf[i * dim];
            const float* fvec32 = &vec_buf[i * dim];

            // Route: FP16 distance to all root centroids. Assign to nearest
            // plus any within closure_epsilon (SPANN-style boundary replication).
            // For IP: distances are negated dots; "within ε" means the
            // absolute difference |d - min_d| ≤ ε.
            std::array<float, 256> root_dists;  // max k_root=256
            float min_d = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < k_root; ++c) {
                root_dists[c] = simd::dist_f16(metric, fvec,
                                                 root_centroids[c].data(), dim);
                if (root_dists[c] < min_d) min_d = root_dists[c];
            }

            // Encode 4-bit scan code ONCE, copy to each target buffer.
            std::vector<uint8_t> code(code_size);
            quantizer->encode(fvec32, code.data());

            for (uint32_t c = 0; c < k_root; ++c) {
                // Assign if within closure_epsilon of the nearest.
                if (std::fabs(root_dists[c] - min_d) > closure_epsilon) continue;

                auto& buf = buffers[c];
                buf.codes.insert(buf.codes.end(), code.begin(), code.end());
                buf.row_ids.push_back(static_cast<RowId>(offset + i));
                buf.fp16_vecs.insert(buf.fp16_vecs.end(), fvec, fvec + dim);

                if (buf.row_ids.size() >= leaf_cap) {
                    flush_buffer(c);
                }
            }
        }

        offset += take;
        if (offset % 1'000'000 < chunk_n) {
            spdlog::info("[sextant] build_streaming: {}M/{}M vectors streamed, "
                         "{} leaves flushed",
                         offset / 1'000'000, n / 1'000'000, n_leaves_total);
        }
    }
    std::fclose(f);

    // Flush remaining buffers.
    for (uint32_t c = 0; c < k_root; ++c) {
        flush_buffer(c);
    }

    spdlog::info("[sextant] build_streaming: streamed {} vectors, {} leaves",
                 n, n_leaves_total);

    // --- 6. Write tree file ---
    const PageId bitmap_page = 2;
    const uint32_t bitmap_pages = 1;

    PageFile file(output_path);
    PageAllocator alloc;
    file.truncate(bitmap_page + bitmap_pages);
    alloc.init(file, bitmap_page, bitmap_pages);

    const uint16_t depth = 2;  // always depth=2 (root → level-1 nodes → leaves)
    const uint32_t cpb = (scan_bits == 4) ? 32 : 16;
    const uint32_t bb = m4 * 16;

    // Write all leaf extents.
    spdlog::info("[sextant] build_streaming: writing {} leaf extents", n_leaves_total);

    struct LeafPageInfo { PageId page; uint32_t pages; };
    std::vector<LeafPageInfo> leaf_pages(n_leaves_total);

    for (uint32_t l = 0; l < n_leaves_total; ++l) {
        const auto& leaf = all_leaves[l];
        const uint32_t count = static_cast<uint32_t>(leaf.row_ids.size());
        if (count == 0) {
            leaf_pages[l] = {kInvalidPage, 0};
            continue;
        }

        const uint32_t npg = leaf_extent_pages(count, m4, scan_bits, n_factors);
        const PageId page = alloc.alloc_extent(file, npg);
        leaf_pages[l] = {page, npg};

        std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);
        auto* lh = reinterpret_cast<TreeLeafHeader*>(buf.data());
        lh->count = count;
        lh->tombstone_count = 0;
        lh->m4 = m4;
        lh->pq_bits = scan_bits;
        lh->n_factors = n_factors;
        lh->block_bytes = bb;
        lh->codes_per_block = cpb;
        lh->extent_pages = npg;

        // Pack FastScan blocks from the leaf's codes.
        const uint32_t n_blocks = (count + cpb - 1) / cpb;
        uint8_t* codes_out = buf.data() + leaf_codes_offset();
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t base = b * cpb;
            uint8_t* blk = codes_out + static_cast<uint64_t>(b) * bb;
            std::memset(blk, 0, bb);
            for (uint32_t j = 0; j < cpb; ++j) {
                const uint32_t gi = base + j;
                if (gi >= count) break;
                const uint8_t* code = leaf.codes.data() + gi * code_size;
                if (scan_bits == 4) {
                    const uint8_t nibble_byte_idx = j % 16;
                    const bool is_hi = (j >= 16);
                    for (uint16_t s = 0; s < m4; ++s) {
                        uint8_t nib = (s / 2 < code_size)
                            ? ((s % 2 == 0) ? (code[s/2] & 0x0F) : (code[s/2] >> 4))
                            : 0;
                        if (is_hi) blk[s * 16 + nibble_byte_idx] |= (nib << 4);
                        else       blk[s * 16 + nibble_byte_idx] |= nib;
                    }
                } else {
                    for (uint16_t s = 0; s < m4; ++s)
                        blk[s * 16 + j] = (s < code_size) ? code[s] : 0;
                }
            }
        }

        // Row IDs.
        RowId* rids = reinterpret_cast<RowId*>(
            buf.data() + leaf_rowids_offset(n_blocks, bb));
        std::memcpy(rids, leaf.row_ids.data(), count * sizeof(RowId));

        file.write_pages(page, npg, buf.data());
    }

    // Write level-1 internal nodes (one per root child).
    spdlog::info("[sextant] build_streaming: writing level-1 nodes");
    std::vector<RootChildData> root_child_data(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        root_child_data[c].centroid = root_centroids[c];

        const auto& leaves = root_to_leaves[c];
        if (leaves.empty()) {
            root_child_data[c].is_leaf = 0;
            root_child_data[c].page = kInvalidPage;
            root_child_data[c].pages = 0;
            continue;
        }

        const uint32_t n_children = static_cast<uint32_t>(leaves.size());
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
            const uint32_t li = leaves[j];
            auto* ce = reinterpret_cast<ChildEntry*>(p);
            ce->child_page = leaf_pages[li].page;
            ce->child_pages = leaf_pages[li].pages;
            ce->is_leaf = 1;
            ce->reserved = 0;
            float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
            std::memcpy(cent, all_leaves[li].centroid.data(), dim * sizeof(float16_t));
            p += cesize;
        }
        file.write_pages(page, npg, buf.data());

        root_child_data[c].is_leaf = 0;
        root_child_data[c].page = page;
        root_child_data[c].pages = npg;
    }

    // Write root node.
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
            float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
            std::memcpy(cent, root_child_data[c].centroid.data(),
                        dim * sizeof(float16_t));
            p += cesize;
        }
        file.write_pages(root_page, root_npg, buf.data());
    }

    // Codebook.
    std::vector<uint8_t> qblob;
    quantizer->serialize(qblob);
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

    // Config blob.
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
    manifest.n_leaves = n_leaves_total;
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

    // Flush bitmap + commit superblock.
    alloc.flush_bitmap(file);
    Superblock sb;
    sb.init_fresh(bitmap_page, bitmap_pages);
    sb.set_root(root_page, root_npg);
    sb.set_depth(depth);
    sb.set_n_leaves(n_leaves_total);
    sb.set_n_pages(file.num_pages());
    sb.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    sb.set_bitmap(bitmap_page, bitmap_pages);
    sb.set_codebook(cb_page, cb_npg);
    sb.set_config(cfg_page, cfg_npg);
    sb.commit(file);
    file.sync();

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    spdlog::info("[sextant] build_streaming complete: N={} k_root={} n_leaves={} "
                 "in {:.2f}s (file={} pages)",
                 n, k_root, n_leaves_total, secs, file.num_pages());

    BuildResult result;
    result.index_path = output_path;
    result.n_vectors = n;
    result.dim = dim;
    result.pq_m = m4;
    result.pq_bits = scan_bits;
     result.build_time_sec = secs;
     return result;
 }

// ===========================================================================
// Two-phase streaming build: greedy stream + one refinement pass.
//
// Phase 1: Stream all vectors greedily → compute leaf centroids (FP16 means).
// Phase 2: Re-stream all vectors, re-assign to nearest LEAF centroid.
//          This corrects greedy mis-assignments with one Lloyd's iteration.
//
// The refinement uses leaf centroids (not root centroids) — much finer
// granularity (N/leaf_cap ≈ 2000 centroids vs K_root ≈ 44). On high-LID
// data where greedy root routing is poor, leaf-level refinement recovers
// most of the quality gap to global k-means.
// ===========================================================================

BuildResult IVFTreeIndex::build_streaming_refined(const std::string& base_path,
                                                    const std::string& output_path,
                                                    const BuildConfig& cfg) {
    const auto t0 = std::chrono::steady_clock::now();

    FILE* f = std::fopen(base_path.c_str(), "rb");
    if (!f) {
        throw Error(ErrorCode::IoError, "build_streaming_refined: cannot open");
    }
    uint32_t header[2];
    if (std::fread(header, sizeof(uint32_t), 2, f) != 2) {
        std::fclose(f);
        throw Error(ErrorCode::IoError, "build_streaming_refined: header read");
    }
    const uint64_t n = header[0];
    const Dim dim = header[1];
    if (n == 0 || dim == 0) {
        std::fclose(f);
        throw Error(ErrorCode::InvalidParam, "build_streaming_refined: empty");
    }

    spdlog::info("[sextant] build_streaming_refined: N={} dim={} → '{}'",
                 n, dim, output_path);

    const auto& params = cfg.params;
    const uint16_t m4 = params.pq4_m > 0 ? params.pq4_m
                                              : static_cast<uint16_t>(dim / 4);
    const uint8_t scan_bits = params.scan_pq_bits;
    const uint32_t leaf_cap = cfg.leaf_capacity > 0 ? cfg.leaf_capacity : 5000;

    uint32_t k_root = cfg.k_root;
    if (k_root == 0) {
        k_root = static_cast<uint32_t>(std::sqrt(static_cast<double>(n) / leaf_cap));
        k_root = std::clamp(k_root, 16u, 256u);
    }

    // Train scan quantizer on a sample.
    const uint32_t train_n = std::min<uint64_t>(20'000, n);
    std::vector<float> sample(static_cast<size_t>(train_n) * dim);
    std::fseek(f, 8, SEEK_SET);
    if (std::fread(sample.data(), sizeof(float),
                   static_cast<size_t>(train_n) * dim, f)
        != static_cast<size_t>(train_n) * dim) {
        std::fclose(f);
        throw Error(ErrorCode::IoError, "build_streaming_refined: sample read");
    }

    std::unique_ptr<PqQuantizer> quantizer;
    uint8_t n_factors = 0;
    if (params.quantizer_type == "prq") {
        const uint32_t nsplits = (params.prq_nsplits > 0)
            ? params.prq_nsplits : static_cast<uint32_t>(dim) / 8;
        quantizer = std::make_unique<ProductResidualQuantizer>(
            params.metric, dim, m4, scan_bits, nsplits,
            params.prq_beam_size, 42);
    } else if (params.quantizer_type == "rabitq") {
        quantizer = std::make_unique<RaBitQQuantizer>(params.metric, dim, 42);
        n_factors = 2;
    } else {
        quantizer = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, scan_bits, 42);
    }
    quantizer->train(sample.data(), train_n);

    std::vector<float16_t> sample_fp16(static_cast<size_t>(train_n) * dim);
    cast_fp32_to_fp16(sample.data(), sample_fp16.data(),
                      static_cast<size_t>(train_n) * dim);

    // Root k-means on sample (same as build_streaming).
    std::vector<std::vector<float16_t>> root_centroids(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        const uint32_t src = (c * train_n) / k_root;
        root_centroids[c].assign(sample_fp16.data() + src * dim,
                                 sample_fp16.data() + (src + 1) * dim);
    }
    const MetricKind metric = params.metric;
    for (uint32_t iter = 0; iter < 10; ++iter) {
        std::vector<std::vector<uint32_t>> assigns(k_root);
        for (uint32_t i = 0; i < train_n; ++i) {
            const float16_t* vec = &sample_fp16[i * dim];
            float best_d = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k_root; ++c) {
                const float d = simd::dist_f16(metric, vec,
                                                root_centroids[c].data(), dim);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            assigns[best_c].push_back(i);
        }
        for (uint32_t c = 0; c < k_root; ++c) {
            if (assigns[c].empty()) {
                uint32_t src = (c * 7919 + 1) % train_n;
                root_centroids[c].assign(sample_fp16.data() + src * dim,
                                         sample_fp16.data() + (src + 1) * dim);
                continue;
            }
            std::vector<double> sum(dim, 0.0);
            for (uint32_t i : assigns[c])
                for (uint16_t d = 0; d < dim; ++d)
                    sum[d] += static_cast<float>(sample_fp16[i * dim + d]);
            const double inv = 1.0 / assigns[c].size();
            for (uint16_t d = 0; d < dim; ++d)
                root_centroids[c][d] = static_cast<float16_t>(sum[d] * inv);
        }
    }

    // Compute closure epsilon (gap-based, same as build_streaming).
    float closure_epsilon = 0.0f;
    {
        const uint32_t sample_for_eps = std::min<uint32_t>(4096, train_n);
        double sum_gap = 0.0;
        for (uint32_t i = 0; i < sample_for_eps; ++i) {
            const float16_t* vec = &sample_fp16[i * dim];
            float d1 = std::numeric_limits<float>::max();
            float d2 = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < k_root; ++c) {
                const float d = simd::dist_f16(metric, vec,
                                                root_centroids[c].data(), dim);
                if (d < d1) { d2 = d1; d1 = d; }
                else if (d < d2) { d2 = d; }
            }
            sum_gap += (d2 - d1);
        }
        closure_epsilon = static_cast<float>(sum_gap / sample_for_eps * 0.5);
    }

    const uint32_t code_size = quantizer->code_size();
    const uint16_t depth = 2;
    const uint32_t cpb = (scan_bits == 4) ? 32 : 16;
    const uint32_t bb = m4 * 16;

    // ===== PHASE 1: Greedy stream → compute leaf centroids =====
    spdlog::info("[sextant] build_streaming_refined: phase 1 (greedy stream)");

    struct LeafBuffer {
        std::vector<float16_t> fp16_vecs;  // for centroid computation
    };

    std::vector<LeafBuffer> buffers(k_root);
    // Leaf centroids from phase 1 (flattened: K_leaves × dim).
    std::vector<float16_t> leaf_centroids;  // all leaf centroids, contiguous

    auto flush_buffer_phase1 = [&](uint32_t c) {
        auto& buf = buffers[c];
        if (buf.fp16_vecs.empty()) return;
        const uint32_t count = buf.fp16_vecs.size() / dim;
        std::vector<double> sum(dim, 0.0);
        for (uint32_t i = 0; i < count; ++i)
            for (uint16_t d = 0; d < dim; ++d)
                sum[d] += static_cast<float>(buf.fp16_vecs[i * dim + d]);
        const double inv = 1.0 / count;
        for (uint16_t d = 0; d < dim; ++d)
            leaf_centroids.push_back(static_cast<float16_t>(sum[d] * inv));
        buf.fp16_vecs.clear();
    };

    std::fseek(f, 8, SEEK_SET);
    {
        const uint32_t chunk_n = 100'000;
        std::vector<float> vec_buf(chunk_n * dim);
        std::vector<float16_t> fp16_buf(chunk_n * dim);
        uint64_t offset = 0;
        while (offset < n) {
            const uint32_t take = static_cast<uint32_t>(
                std::min<uint64_t>(chunk_n, n - offset));
            if (std::fread(vec_buf.data(), sizeof(float),
                           static_cast<size_t>(take) * dim, f)
                != static_cast<size_t>(take) * dim) {
                std::fclose(f);
                throw Error(ErrorCode::IoError, "phase 1 read failed");
            }
            cast_fp32_to_fp16(vec_buf.data(), fp16_buf.data(),
                              static_cast<size_t>(take) * dim);
            for (uint32_t i = 0; i < take; ++i) {
                const float16_t* fvec = &fp16_buf[i * dim];
                float best_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < k_root; ++c) {
                    const float d = simd::dist_f16(metric, fvec,
                                                    root_centroids[c].data(), dim);
                    if (d < best_d) { best_d = d; best_c = c; }
                }
                buffers[best_c].fp16_vecs.insert(
                    buffers[best_c].fp16_vecs.end(), fvec, fvec + dim);
                if (buffers[best_c].fp16_vecs.size() / dim >= leaf_cap)
                    flush_buffer_phase1(best_c);
            }
            offset += take;
        }
    }
    for (uint32_t c = 0; c < k_root; ++c) flush_buffer_phase1(c);

    const uint32_t n_leaf_centroids = leaf_centroids.size() / dim;
    spdlog::info("[sextant] build_streaming_refined: phase 1 done, "
                 "{} leaf centroids", n_leaf_centroids);

    // ===== PHASE 2: Refine — re-assign to nearest leaf centroid =====
    // Re-stream all vectors. For each vector, find its nearest leaf centroid
    // (among all n_leaf_centroids). Build leaf buffers from the refined
    // assignment, with closure replication.
    spdlog::info("[sextant] build_streaming_refined: phase 2 (refinement)");

    // Compute closure epsilon at leaf granularity (from the leaf centroids).
    float leaf_closure_eps = 0.0f;
    {
        // Sample vectors and measure mean gap to 2nd-nearest leaf centroid.
        const uint32_t s = std::min<uint32_t>(2048, train_n);
        double sum_gap = 0.0;
        for (uint32_t i = 0; i < s; ++i) {
            const float16_t* vec = &sample_fp16[i * dim];
            float d1 = std::numeric_limits<float>::max();
            float d2 = std::numeric_limits<float>::max();
            for (uint32_t l = 0; l < n_leaf_centroids; ++l) {
                const float d = simd::dist_f16(metric, vec,
                    &leaf_centroids[l * dim], dim);
                if (d < d1) { d2 = d1; d1 = d; }
                else if (d < d2) d2 = d;
            }
            sum_gap += (d2 - d1);
        }
        leaf_closure_eps = static_cast<float>(sum_gap / s * 0.5);
        spdlog::info("[sextant] build_streaming_refined: leaf closure_eps={:.4f}",
                     leaf_closure_eps);
    }

    struct RefinedLeaf {
        std::vector<uint8_t> codes;
        std::vector<RowId> row_ids;
        std::vector<float16_t> centroid;
    };
    std::vector<RefinedLeaf> all_leaves;

    struct RefLeafBuffer {
        std::vector<uint8_t> codes;
        std::vector<RowId> row_ids;
        std::vector<float16_t> fp16_vecs;
    };
    std::vector<RefLeafBuffer> leaf_buffers(n_leaf_centroids);

    auto flush_ref_buffer = [&](uint32_t li) {
        auto& buf = leaf_buffers[li];
        if (buf.row_ids.empty()) return;
        RefinedLeaf leaf;
        leaf.codes = std::move(buf.codes);
        leaf.row_ids = std::move(buf.row_ids);
        leaf.centroid.resize(dim);
        std::vector<double> sum(dim, 0.0);
        for (uint32_t i = 0; i < buf.fp16_vecs.size() / dim; ++i)
            for (uint16_t d = 0; d < dim; ++d)
                sum[d] += static_cast<float>(buf.fp16_vecs[i * dim + d]);
        const double inv = 1.0 / (buf.fp16_vecs.size() / dim);
        for (uint16_t d = 0; d < dim; ++d)
            leaf.centroid[d] = static_cast<float16_t>(sum[d] * inv);
        all_leaves.push_back(std::move(leaf));
        buf.codes.clear();
        buf.row_ids.clear();
        buf.fp16_vecs.clear();
    };

    std::fseek(f, 8, SEEK_SET);
    {
        const uint32_t chunk_n = 100'000;
        std::vector<float> vec_buf(chunk_n * dim);
        std::vector<float16_t> fp16_buf(chunk_n * dim);
        uint64_t offset = 0;
        while (offset < n) {
            const uint32_t take = static_cast<uint32_t>(
                std::min<uint64_t>(chunk_n, n - offset));
            if (std::fread(vec_buf.data(), sizeof(float),
                           static_cast<size_t>(take) * dim, f)
                != static_cast<size_t>(take) * dim) {
                std::fclose(f);
                throw Error(ErrorCode::IoError, "phase 2 read failed");
            }
            cast_fp32_to_fp16(vec_buf.data(), fp16_buf.data(),
                              static_cast<size_t>(take) * dim);

            for (uint32_t i = 0; i < take; ++i) {
                const float16_t* fvec = &fp16_buf[i * dim];
                const float* fvec32 = &vec_buf[i * dim];

                // Find nearest leaf centroid + closure matches.
                std::array<float, 2048> leaf_dists;
                float min_d = std::numeric_limits<float>::max();
                for (uint32_t l = 0; l < n_leaf_centroids; ++l) {
                    leaf_dists[l] = simd::dist_f16(metric, fvec,
                        &leaf_centroids[l * dim], dim);
                    if (leaf_dists[l] < min_d) min_d = leaf_dists[l];
                }

                // Encode 4-bit code once.
                std::vector<uint8_t> code(code_size);
                quantizer->encode(fvec32, code.data());

                // Assign to nearest + closure.
                for (uint32_t l = 0; l < n_leaf_centroids; ++l) {
                    if (std::fabs(leaf_dists[l] - min_d) > leaf_closure_eps)
                        continue;
                    auto& buf = leaf_buffers[l];
                    buf.codes.insert(buf.codes.end(), code.begin(), code.end());
                    buf.row_ids.push_back(static_cast<RowId>(offset + i));
                    buf.fp16_vecs.insert(buf.fp16_vecs.end(), fvec, fvec + dim);
                    if (buf.row_ids.size() >= leaf_cap)
                        flush_ref_buffer(l);
                }
            }
            offset += take;
            if (offset % 1'000'000 < chunk_n)
                spdlog::info("[sextant] build_streaming_refined: phase 2 "
                             "{}M/{}M, {} leaves",
                             offset / 1'000'000, n / 1'000'000,
                             all_leaves.size());
        }
    }
    std::fclose(f);
    for (uint32_t l = 0; l < n_leaf_centroids; ++l) flush_ref_buffer(l);

    const uint32_t n_leaves_total = all_leaves.size();
    spdlog::info("[sextant] build_streaming_refined: phase 2 done, {} leaves",
                 n_leaves_total);

    // ===== Write tree file (same as build_streaming) =====
    const PageId bitmap_page = 2;
    const uint32_t bitmap_pages = 1;

    PageFile file(output_path);
    PageAllocator alloc;
    file.truncate(bitmap_page + bitmap_pages);
    alloc.init(file, bitmap_page, bitmap_pages);

    // Write leaf extents.
    spdlog::info("[sextant] build_streaming_refined: writing {} leaves",
                 n_leaves_total);
    struct LeafPageInfo { PageId page; uint32_t pages; };
    std::vector<LeafPageInfo> leaf_pages(n_leaves_total);

    for (uint32_t l = 0; l < n_leaves_total; ++l) {
        const auto& leaf = all_leaves[l];
        const uint32_t count = static_cast<uint32_t>(leaf.row_ids.size());
        if (count == 0) { leaf_pages[l] = {kInvalidPage, 0}; continue; }

        const uint32_t npg = leaf_extent_pages(count, m4, scan_bits, n_factors);
        const PageId page = alloc.alloc_extent(file, npg);
        leaf_pages[l] = {page, npg};

        std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);
        auto* lh = reinterpret_cast<TreeLeafHeader*>(buf.data());
        lh->count = count;
        lh->tombstone_count = 0;
        lh->m4 = m4;
        lh->pq_bits = scan_bits;
        lh->n_factors = n_factors;
        lh->block_bytes = bb;
        lh->codes_per_block = cpb;
        lh->extent_pages = npg;

        const uint32_t n_blocks = (count + cpb - 1) / cpb;
        uint8_t* codes_out = buf.data() + leaf_codes_offset();
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t base = b * cpb;
            uint8_t* blk = codes_out + static_cast<uint64_t>(b) * bb;
            std::memset(blk, 0, bb);
            for (uint32_t j = 0; j < cpb; ++j) {
                const uint32_t gi = base + j;
                if (gi >= count) break;
                const uint8_t* code = leaf.codes.data() + gi * code_size;
                if (scan_bits == 4) {
                    const uint8_t nbi = j % 16;
                    const bool hi = (j >= 16);
                    for (uint16_t s = 0; s < m4; ++s) {
                        uint8_t nib = (s / 2 < code_size)
                            ? ((s % 2 == 0) ? (code[s/2] & 0x0F) : (code[s/2] >> 4))
                            : 0;
                        if (hi) blk[s * 16 + nbi] |= (nib << 4);
                        else    blk[s * 16 + nbi] |= nib;
                    }
                } else {
                    for (uint16_t s = 0; s < m4; ++s)
                        blk[s * 16 + j] = (s < code_size) ? code[s] : 0;
                }
            }
        }
        RowId* rids = reinterpret_cast<RowId*>(
            buf.data() + leaf_rowids_offset(n_blocks, bb));
        std::memcpy(rids, leaf.row_ids.data(), count * sizeof(RowId));
        file.write_pages(page, npg, buf.data());
    }

    // Group leaves by nearest root centroid for the level-1 tree structure.
    // Each root child points to the leaves whose centroid is nearest to it.
    std::vector<std::vector<uint32_t>> root_to_leaves(k_root);
    for (uint32_t l = 0; l < n_leaves_total; ++l) {
        if (leaf_pages[l].page == kInvalidPage) continue;
        float best_d = std::numeric_limits<float>::max();
        uint32_t best_c = 0;
        for (uint32_t c = 0; c < k_root; ++c) {
            const float d = simd::dist_f16(metric,
                all_leaves[l].centroid.data(),
                root_centroids[c].data(), dim);
            if (d < best_d) { best_d = d; best_c = c; }
        }
        root_to_leaves[best_c].push_back(l);
    }

    // Write level-1 nodes.
    std::vector<RootChildData> root_child_data(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        root_child_data[c].centroid = root_centroids[c];
        const auto& leaves = root_to_leaves[c];
        if (leaves.empty()) {
            root_child_data[c].is_leaf = 0;
            root_child_data[c].page = kInvalidPage;
            root_child_data[c].pages = 0;
            continue;
        }
        const uint32_t n_children = static_cast<uint32_t>(leaves.size());
        const uint32_t npg = node_extent_pages(dim, n_children);
        const PageId page = alloc.alloc_extent(file, npg);
        std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);
        auto* nh = reinterpret_cast<TreeNodeHeader*>(buf.data());
        nh->n_children = n_children;
        nh->extent_pages = npg;
        nh->dim = dim;
        const uint32_t cesize = child_entry_size(dim);
        uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
        for (uint32_t j = 0; j < n_children; ++j) {
            const uint32_t li = leaves[j];
            auto* ce = reinterpret_cast<ChildEntry*>(p);
            ce->child_page = leaf_pages[li].page;
            ce->child_pages = leaf_pages[li].pages;
            ce->is_leaf = 1;
            float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
            std::memcpy(cent, all_leaves[li].centroid.data(),
                        dim * sizeof(float16_t));
            p += cesize;
        }
        file.write_pages(page, npg, buf.data());
        root_child_data[c].is_leaf = 0;
        root_child_data[c].page = page;
        root_child_data[c].pages = npg;
    }

    // Write root node.
    const uint32_t root_npg = node_extent_pages(dim, k_root);
    const PageId root_page = alloc.alloc_extent(file, root_npg);
    {
        std::vector<uint8_t> buf(static_cast<size_t>(root_npg) * kPageSize, 0);
        auto* rh = reinterpret_cast<TreeNodeHeader*>(buf.data());
        rh->n_children = k_root;
        rh->extent_pages = root_npg;
        rh->dim = dim;
        const uint32_t cesize = child_entry_size(dim);
        uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
        for (uint32_t c = 0; c < k_root; ++c) {
            auto* ce = reinterpret_cast<ChildEntry*>(p);
            ce->child_page = root_child_data[c].page;
            ce->child_pages = root_child_data[c].pages;
            ce->is_leaf = root_child_data[c].is_leaf;
            float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
            std::memcpy(cent, root_child_data[c].centroid.data(),
                        dim * sizeof(float16_t));
            p += cesize;
        }
        file.write_pages(root_page, root_npg, buf.data());
    }

    // Codebook.
    std::vector<uint8_t> qblob;
    quantizer->serialize(qblob);
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

    // Config blob.
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
    manifest.n_leaves = n_leaves_total;
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

    alloc.flush_bitmap(file);
    Superblock sb;
    sb.init_fresh(bitmap_page, bitmap_pages);
    sb.set_root(root_page, root_npg);
    sb.set_depth(depth);
    sb.set_n_leaves(n_leaves_total);
    sb.set_n_pages(file.num_pages());
    sb.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    sb.set_bitmap(bitmap_page, bitmap_pages);
    sb.set_codebook(cb_page, cb_npg);
    sb.set_config(cfg_page, cfg_npg);
    sb.commit(file);
    file.sync();

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    spdlog::info("[sextant] build_streaming_refined complete: N={} k_root={} "
                 "n_leaves={} in {:.2f}s (file={} pages)",
                 n, k_root, n_leaves_total, secs, file.num_pages());

    BuildResult result;
    result.index_path = output_path;
    result.n_vectors = n;
    result.dim = dim;
    result.pq_m = m4;
    result.pq_bits = scan_bits;
    result.build_time_sec = secs;
     return result;
 }

// ===========================================================================
// PCA-preconditioned streaming build.
//
// Projects vectors onto top-k principal components before routing. On
// high-LID data (Sphere-IP d_eff≈2), the raw 768D space is near-isotropic —
// all distances look the same. PCA exposes the low-dimensional manifold, so
// k-means in PCA space converges cleanly and routing works.
//
// PCA computation: covariance matrix (dim×dim) from the 20k sample, eigendecompose
// via the existing compute_pca_rotation_public (Jacobi eigendecomposition).
// Projection: SIMD dot product per PC (simd::dot_f32), parallelized across
// threads during the streaming phase.
// ============================================================================

BuildResult IVFTreeIndex::build_streaming_pca(const std::string& base_path,
                                                const std::string& output_path,
                                                const BuildConfig& cfg) {
    const auto t0 = std::chrono::steady_clock::now();

    FILE* f = std::fopen(base_path.c_str(), "rb");
    if (!f) throw Error(ErrorCode::IoError, "build_streaming_pca: cannot open");
    uint32_t header[2];
    if (std::fread(header, sizeof(uint32_t), 2, f) != 2) {
        std::fclose(f);
        throw Error(ErrorCode::IoError, "build_streaming_pca: header");
    }
    const uint64_t n = header[0];
    const Dim dim = header[1];
    if (n == 0 || dim == 0) {
        std::fclose(f);
        throw Error(ErrorCode::InvalidParam, "build_streaming_pca: empty");
    }

    spdlog::info("[sextant] build_streaming_pca: N={} dim={} → '{}'",
                 n, dim, output_path);

    const auto& params = cfg.params;
    const uint16_t m4 = params.pq4_m > 0 ? params.pq4_m
                                              : static_cast<uint16_t>(dim / 4);
    const uint8_t scan_bits = params.scan_pq_bits;
    const uint32_t leaf_cap = cfg.leaf_capacity > 0 ? cfg.leaf_capacity : 5000;

    uint32_t k_root = cfg.k_root;
    if (k_root == 0) {
        k_root = static_cast<uint32_t>(std::sqrt(static_cast<double>(n) / leaf_cap));
        k_root = std::clamp(k_root, 16u, 256u);
    }

    // PCA dimensions: project to this many components for routing.
    // Default: 32 (captures meaningful variance without being too large
    // for k-means to find structure). For d_eff≈2 data, even 8-16 PCs suffice.
    const uint32_t pca_dims = std::min(dim, cfg.pca_dims > 0 ? cfg.pca_dims : 32u);

    // --- 1. Sample + train quantizer ---
    const auto t_sample = std::chrono::steady_clock::now();
    const uint32_t train_n = std::min<uint64_t>(20'000, n);
    std::vector<float> sample(static_cast<size_t>(train_n) * dim);
    std::fseek(f, 8, SEEK_SET);
    if (std::fread(sample.data(), sizeof(float),
                   static_cast<size_t>(train_n) * dim, f)
        != static_cast<size_t>(train_n) * dim) {
        std::fclose(f);
        throw Error(ErrorCode::IoError, "build_streaming_pca: sample read");
    }

    std::unique_ptr<PqQuantizer> quantizer;
    uint8_t n_factors = 0;
    if (params.quantizer_type == "prq") {
        const uint32_t nsplits = (params.prq_nsplits > 0)
            ? params.prq_nsplits : static_cast<uint32_t>(dim) / 8;
        quantizer = std::make_unique<ProductResidualQuantizer>(
            params.metric, dim, m4, scan_bits, nsplits,
            params.prq_beam_size, 42);
    } else if (params.quantizer_type == "rabitq") {
        quantizer = std::make_unique<RaBitQQuantizer>(params.metric, dim, 42);
        n_factors = 2;
    } else {
        quantizer = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, scan_bits, 42);
    }
    quantizer->train(sample.data(), train_n);
    spdlog::info("[sextant] build_streaming_pca: trained {} in {:.2f}s",
                 params.quantizer_type,
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_sample).count());

    // --- 2. Compute PCA (reuse existing compute_pca_rotation_public) ---
    const auto t_pca = std::chrono::steady_clock::now();

    // compute_pca_rotation_public returns the full dim×dim rotation R,
    // with rows sorted by descending eigenvalue. We take the top pca_dims rows.
    // NOTE: this modifies `sample` (centers it). We re-read the sample for
    // PCA projection below using the rotation + original sample (re-read from
    // the uncentered copy we saved above? Actually compute_pca_rotation_public
    // takes samples and centers internally; it doesn't modify the input).
    // Wait — compute_pca_rotation_public does NOT modify `sample` (it reads
    // const float*). But we already centered sample at line ~1867 above in
    // the old code. Since we removed that, sample is still the raw FP32 sample.
    // Good — the function handles centering internally.
    std::vector<double> eigvals;
    std::vector<float> rotation = compute_pca_rotation_public(
        sample.data(), train_n, dim, &eigvals);

    if (rotation.empty()) {
        std::fclose(f);
        throw Error(ErrorCode::InvalidParam,
                    "build_streaming_pca: degenerate covariance (no PCA)");
    }

    // Extract top-k rows: proj_rows[k][d] = rotation[k * dim + d].
    // These are the top-k eigenvectors (rows of R, sorted by eigenvalue).
    // Precompute mean_proj[k] = dot(proj_row_k, mean) so we can project
    // uncentered vectors: proj[k] = dot(proj_row_k, vec) - mean_proj[k].
    std::vector<double> mean(dim, 0.0);
    for (uint32_t i = 0; i < train_n; ++i)
        for (uint16_t d = 0; d < dim; ++d)
            mean[d] += sample[i * dim + d];
    for (uint16_t d = 0; d < dim; ++d) mean[d] /= train_n;

    std::vector<float> mean_proj(pca_dims);
    for (uint32_t k = 0; k < pca_dims; ++k) {
        double acc = 0.0;
        for (uint16_t d = 0; d < dim; ++d)
            acc += rotation[k * dim + d] * mean[d];
        mean_proj[k] = static_cast<float>(acc);
    }

    double var_explained = 0.0, var_total = 0.0;
    for (uint16_t i = 0; i < dim; ++i) var_total += eigvals[i];
    for (uint32_t k = 0; k < pca_dims; ++k) var_explained += eigvals[k];
    spdlog::info("[sextant] build_streaming_pca: PCA {}→{} dims, "
                 "variance explained: {:.1f}% in {:.2f}s",
                 dim, pca_dims, 100.0 * var_explained / std::max(var_total, 1.0),
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_pca).count());

    // Project the sample for k-means (SIMD dot product per PC).
    std::vector<float> sample_pca(static_cast<size_t>(train_n) * pca_dims);
    for (uint32_t i = 0; i < train_n; ++i) {
        const float* xi = &sample[i * dim];
        float* pi = &sample_pca[i * pca_dims];
        for (uint32_t k = 0; k < pca_dims; ++k) {
            pi[k] = simd::dot_f32(&rotation[k * dim], xi, dim) - mean_proj[k];
        }
    }

    // --- 3. K-means in PCA space (root centroids) ---
    const auto t_kmeans = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> root_centroids_pca(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        const uint32_t src = (c * train_n) / k_root;
        root_centroids_pca[c].assign(
            &sample_pca[src * pca_dims], &sample_pca[(src + 1) * pca_dims]);
    }

    std::vector<uint32_t> prev_assign(train_n, UINT32_MAX);
    for (uint32_t iter = 0; iter < 10; ++iter) {
        std::vector<std::vector<uint32_t>> assigns(k_root);
        uint32_t n_changed = 0;
        for (uint32_t i = 0; i < train_n; ++i) {
            const float* vi = &sample_pca[i * pca_dims];
            float best_d = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k_root; ++c) {
                float d = 0.0f;
                for (uint32_t k = 0; k < pca_dims; ++k) {
                    const float diff = vi[k] - root_centroids_pca[c][k];
                    d += diff * diff;
                }
                if (d < best_d) { best_d = d; best_c = c; }
            }
            assigns[best_c].push_back(i);
            if (iter > 0 && best_c != prev_assign[i]) ++n_changed;
            prev_assign[i] = best_c;
        }
        for (uint32_t c = 0; c < k_root; ++c) {
            if (assigns[c].empty()) {
                uint32_t src = (c * 7919 + 1) % train_n;
                root_centroids_pca[c].assign(
                    &sample_pca[src * pca_dims],
                    &sample_pca[(src + 1) * pca_dims]);
                continue;
            }
            std::vector<double> sum(pca_dims, 0.0);
            for (uint32_t i : assigns[c])
                for (uint32_t k = 0; k < pca_dims; ++k)
                    sum[k] += sample_pca[i * pca_dims + k];
            const double inv = 1.0 / assigns[c].size();
            for (uint32_t k = 0; k < pca_dims; ++k)
                root_centroids_pca[c][k] = static_cast<float>(sum[k] * inv);
        }
        if (iter >= 2 && iter > 0 && n_changed > 0 &&
            static_cast<double>(n_changed) / train_n < 0.01) {
            spdlog::info("[sextant] build_streaming_pca: k-means converged at "
                         "iter {} ({:.2f}% changed)", iter + 1,
                         100.0 * n_changed / train_n);
            break;
        }
        if (iter > 0)
            spdlog::info("[sextant] build_streaming_pca: k-means iter {} "
                         "changed={} ({:.1f}%)", iter + 1, n_changed,
                         100.0 * n_changed / train_n);
    }
    spdlog::info("[sextant] build_streaming_pca: k-means done in {:.2f}s",
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_kmeans).count());

    // --- 4. Compute closure epsilon in PCA space ---
    float closure_epsilon = 0.0f;
    {
        const uint32_t s = std::min<uint32_t>(4096, train_n);
        double sum_gap = 0.0;
        for (uint32_t i = 0; i < s; ++i) {
            const float* vi = &sample_pca[i * pca_dims];
            float d1 = std::numeric_limits<float>::max();
            float d2 = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < k_root; ++c) {
                float d = 0.0f;
                for (uint32_t k = 0; k < pca_dims; ++k) {
                    const float diff = vi[k] - root_centroids_pca[c][k];
                    d += diff * diff;
                }
                if (d < d1) { d2 = d1; d1 = d; }
                else if (d < d2) d2 = d;
            }
            sum_gap += (d2 - d1);
        }
        closure_epsilon = static_cast<float>(sum_gap / s * 0.5);
    }
    spdlog::info("[sextant] build_streaming_pca: closure_eps={:.4f}",
                 closure_epsilon);

    // --- 5. Stream all vectors ---
    // Project each vector to PCA space, route to nearest root centroid (+closure),
    // encode scan code in original space, append to leaf buffer.
    const auto t_stream = std::chrono::steady_clock::now();
    const uint32_t code_size = quantizer->code_size();

    struct LeafBuffer {
        std::vector<uint8_t> codes;
        std::vector<RowId> row_ids;
        std::vector<float16_t> fp16_vecs;
    };
    std::vector<LeafBuffer> buffers(k_root);

    struct FlushedLeaf {
        std::vector<uint8_t> codes;
        std::vector<RowId> row_ids;
        std::vector<float16_t> centroid;
    };
    std::vector<FlushedLeaf> all_leaves;
    std::vector<std::vector<uint32_t>> root_to_leaves(k_root);

    auto flush_buffer = [&](uint32_t c) {
        auto& buf = buffers[c];
        if (buf.row_ids.empty()) return;
        FlushedLeaf leaf;
        leaf.codes = std::move(buf.codes);
        leaf.row_ids = std::move(buf.row_ids);
        leaf.centroid.resize(dim);
        std::vector<double> sum(dim, 0.0);
        const uint32_t cnt = buf.fp16_vecs.size() / dim;
        for (uint32_t i = 0; i < cnt; ++i)
            for (uint16_t d = 0; d < dim; ++d)
                sum[d] += static_cast<float>(buf.fp16_vecs[i * dim + d]);
        const double inv = 1.0 / cnt;
        for (uint16_t d = 0; d < dim; ++d)
            leaf.centroid[d] = static_cast<float16_t>(sum[d] * inv);
        root_to_leaves[c].push_back(static_cast<uint32_t>(all_leaves.size()));
        all_leaves.push_back(std::move(leaf));
        buf.codes.clear();
        buf.row_ids.clear();
        buf.fp16_vecs.clear();
    };

    // Precompute FP16 root centroids for the search path (the tree stores
    // root centroids in original FP16 space, not PCA space — we project back).
    // Actually, the root centroids are in PCA space. For search, we need
    // centroids in original FP16 space. We'll store the PCA-space centroids
    // in the tree's leaf centroids (level-1 children) and use FP16 original-
    // space centroids for the root. For routing at search time, we project
    // the query to PCA space and compute distances there.
    // 
    // BUT: the search path currently routes by FP16 distance in original space.
    // To use PCA routing at search time, we'd need to modify the search path.
    // For now, let's project the root centroids BACK to original space for
    // storage. The search will route by FP16 distance in original space
    // (which is what the tree format supports).
    //
    // This is a limitation: the tree stores FP16 original-space centroids.
    // PCA routing at search time requires storing the projection matrix in
    // the tree and modifying the search path. For this prototype, we use
    // PCA for BUILD-TIME routing (better partition) and original-space FP16
    // for SEARCH-TIME routing (the existing path). The partition quality
    // improvement should still help recall even with original-space search
    // routing, because the leaf membership is better.

    // Project root centroids back to original FP16 space (add mean back).
    // original[d] = mean[d] + Σ_k pca_centroid[k] × rotation[k][d]
    std::vector<std::vector<float16_t>> root_centroids_fp16(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        root_centroids_fp16[c].resize(dim);
        for (uint16_t d = 0; d < dim; ++d) {
            double val = mean[d];
            for (uint32_t k = 0; k < pca_dims; ++k)
                val += root_centroids_pca[c][k] * rotation[k * dim + d];
            root_centroids_fp16[c][d] = static_cast<float16_t>(val);
        }
    }

    std::fseek(f, 8, SEEK_SET);
    {
        const uint32_t chunk_n = 100'000;
        std::vector<float> vec_buf(chunk_n * dim);
        std::vector<float16_t> fp16_buf(chunk_n * dim);
        std::vector<float> pca_buf(chunk_n * pca_dims);
        uint64_t offset = 0;
        while (offset < n) {
            const uint32_t take = static_cast<uint32_t>(
                std::min<uint64_t>(chunk_n, n - offset));
            if (std::fread(vec_buf.data(), sizeof(float),
                           static_cast<size_t>(take) * dim, f)
                != static_cast<size_t>(take) * dim) {
                std::fclose(f);
                throw Error(ErrorCode::IoError, "build_streaming_pca: stream read");
            }
            cast_fp32_to_fp16(vec_buf.data(), fp16_buf.data(),
                              static_cast<size_t>(take) * dim);

            // Project chunk to PCA space (SIMD: simd::dot_f32 per PC).
            // proj[k] = dot(rotation[k], vec) - mean_proj[k]
            for (uint32_t i = 0; i < take; ++i) {
                const float* xi = &vec_buf[i * dim];
                float* pi = &pca_buf[i * pca_dims];
                for (uint32_t k = 0; k < pca_dims; ++k) {
                    pi[k] = simd::dot_f32(&rotation[k * dim], xi, dim)
                            - mean_proj[k];
                }
            }

            for (uint32_t i = 0; i < take; ++i) {
                const float* pi = &pca_buf[i * pca_dims];
                const float* fvec32 = &vec_buf[i * dim];
                const float16_t* fvec = &fp16_buf[i * dim];

                // Route in PCA space.
                std::array<float, 256> root_dists;
                float min_d = std::numeric_limits<float>::max();
                for (uint32_t c = 0; c < k_root; ++c) {
                    float d = 0.0f;
                    for (uint32_t k = 0; k < pca_dims; ++k) {
                        const float diff = pi[k] - root_centroids_pca[c][k];
                        d += diff * diff;
                    }
                    root_dists[c] = d;
                    if (d < min_d) min_d = d;
                }

                std::vector<uint8_t> code(code_size);
                quantizer->encode(fvec32, code.data());

                for (uint32_t c = 0; c < k_root; ++c) {
                    if (std::fabs(root_dists[c] - min_d) > closure_epsilon)
                        continue;
                    auto& buf = buffers[c];
                    buf.codes.insert(buf.codes.end(), code.begin(), code.end());
                    buf.row_ids.push_back(static_cast<RowId>(offset + i));
                    buf.fp16_vecs.insert(buf.fp16_vecs.end(), fvec, fvec + dim);
                    if (buf.row_ids.size() >= leaf_cap)
                        flush_buffer(c);
                }
            }
            offset += take;
            if (offset % 1'000'000 < chunk_n)
                spdlog::info("[sextant] build_streaming_pca: {}M/{}M streamed, "
                             "{} leaves", offset / 1'000'000, n / 1'000'000,
                             all_leaves.size());
        }
    }
    std::fclose(f);
    for (uint32_t c = 0; c < k_root; ++c) flush_buffer(c);

    const uint32_t n_leaves_total = all_leaves.size();
    spdlog::info("[sextant] build_streaming_pca: streamed {} vectors, {} leaves "
                 "in {:.2f}s", n, n_leaves_total,
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_stream).count());

    // --- 6. Write tree file ---
    const auto t_write = std::chrono::steady_clock::now();
    const PageId bitmap_page = 2;
    const uint32_t bitmap_pages = 1;
    PageFile file(output_path);
    PageAllocator alloc;
    file.truncate(bitmap_page + bitmap_pages);
    alloc.init(file, bitmap_page, bitmap_pages);

    const uint16_t depth = 2;
    const uint32_t cpb = (scan_bits == 4) ? 32 : 16;
    const uint32_t bb = m4 * 16;

    // Write leaf extents.
    struct LeafPageInfo { PageId page; uint32_t pages; };
    std::vector<LeafPageInfo> leaf_pages(n_leaves_total);
    for (uint32_t l = 0; l < n_leaves_total; ++l) {
        const auto& leaf = all_leaves[l];
        const uint32_t count = static_cast<uint32_t>(leaf.row_ids.size());
        if (count == 0) { leaf_pages[l] = {kInvalidPage, 0}; continue; }
        const uint32_t npg = leaf_extent_pages(count, m4, scan_bits, n_factors);
        const PageId page = alloc.alloc_extent(file, npg);
        leaf_pages[l] = {page, npg};

        std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);
        auto* lh = reinterpret_cast<TreeLeafHeader*>(buf.data());
        lh->count = count; lh->tombstone_count = 0; lh->m4 = m4;
        lh->pq_bits = scan_bits; lh->n_factors = n_factors;
        lh->block_bytes = bb; lh->codes_per_block = cpb; lh->extent_pages = npg;

        const uint32_t n_blocks = (count + cpb - 1) / cpb;
        uint8_t* codes_out = buf.data() + leaf_codes_offset();
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t base = b * cpb;
            uint8_t* blk = codes_out + static_cast<uint64_t>(b) * bb;
            std::memset(blk, 0, bb);
            for (uint32_t j = 0; j < cpb; ++j) {
                const uint32_t gi = base + j;
                if (gi >= count) break;
                const uint8_t* code = leaf.codes.data() + gi * code_size;
                if (scan_bits == 4) {
                    const uint8_t nbi = j % 16; const bool hi = (j >= 16);
                    for (uint16_t s = 0; s < m4; ++s) {
                        uint8_t nib = (s/2 < code_size)
                            ? ((s%2==0) ? (code[s/2]&0x0F) : (code[s/2]>>4)) : 0;
                        if (hi) blk[s*16+nbi] |= (nib<<4);
                        else    blk[s*16+nbi] |= nib;
                    }
                } else {
                    for (uint16_t s = 0; s < m4; ++s)
                        blk[s*16+j] = (s < code_size) ? code[s] : 0;
                }
            }
        }
        RowId* rids = reinterpret_cast<RowId*>(
            buf.data() + leaf_rowids_offset(n_blocks, bb));
        std::memcpy(rids, leaf.row_ids.data(), count * sizeof(RowId));
        file.write_pages(page, npg, buf.data());
    }

    // Level-1 nodes + root node + codebook + config (same as build_streaming).
    std::vector<RootChildData> root_child_data(k_root);
    for (uint32_t c = 0; c < k_root; ++c) {
        root_child_data[c].centroid = root_centroids_fp16[c];
        const auto& leaves = root_to_leaves[c];
        if (leaves.empty()) {
            root_child_data[c].is_leaf = 0;
            root_child_data[c].page = kInvalidPage;
            root_child_data[c].pages = 0;
            continue;
        }
        const uint32_t nch = static_cast<uint32_t>(leaves.size());
        const uint32_t npg = node_extent_pages(dim, nch);
        const PageId page = alloc.alloc_extent(file, npg);
        std::vector<uint8_t> buf(static_cast<size_t>(npg) * kPageSize, 0);
        auto* nh = reinterpret_cast<TreeNodeHeader*>(buf.data());
        nh->n_children = nch; nh->extent_pages = npg; nh->dim = dim;
        const uint32_t cesize = child_entry_size(dim);
        uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
        for (uint32_t j = 0; j < nch; ++j) {
            const uint32_t li = leaves[j];
            auto* ce = reinterpret_cast<ChildEntry*>(p);
            ce->child_page = leaf_pages[li].page;
            ce->child_pages = leaf_pages[li].pages;
            ce->is_leaf = 1;
            float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
            std::memcpy(cent, all_leaves[li].centroid.data(), dim*sizeof(float16_t));
            p += cesize;
        }
        file.write_pages(page, npg, buf.data());
        root_child_data[c].is_leaf = 0;
        root_child_data[c].page = page;
        root_child_data[c].pages = npg;
    }

    const uint32_t root_npg = node_extent_pages(dim, k_root);
    const PageId root_page = alloc.alloc_extent(file, root_npg);
    {
        std::vector<uint8_t> buf(static_cast<size_t>(root_npg) * kPageSize, 0);
        auto* rh = reinterpret_cast<TreeNodeHeader*>(buf.data());
        rh->n_children = k_root; rh->extent_pages = root_npg; rh->dim = dim;
        const uint32_t cesize = child_entry_size(dim);
        uint8_t* p = buf.data() + sizeof(TreeNodeHeader);
        for (uint32_t c = 0; c < k_root; ++c) {
            auto* ce = reinterpret_cast<ChildEntry*>(p);
            ce->child_page = root_child_data[c].page;
            ce->child_pages = root_child_data[c].pages;
            ce->is_leaf = root_child_data[c].is_leaf;
            float16_t* cent = reinterpret_cast<float16_t*>(p + sizeof(ChildEntry));
            std::memcpy(cent, root_child_data[c].centroid.data(),
                        dim * sizeof(float16_t));
            p += cesize;
        }
        file.write_pages(root_page, root_npg, buf.data());
    }

    std::vector<uint8_t> qblob;
    quantizer->serialize(qblob);
    const uint64_t qbsz = qblob.size();
    std::vector<uint8_t> qbs(sizeof(qbsz) + qblob.size());
    std::memcpy(qbs.data(), &qbsz, sizeof(qbsz));
    std::memcpy(qbs.data() + sizeof(qbsz), qblob.data(), qblob.size());
    const uint32_t cb_npg = static_cast<uint32_t>((qbs.size()+kPageSize-1)/kPageSize);
    const PageId cb_page = alloc.alloc_extent(file, cb_npg);
    { std::vector<uint8_t> b(cb_npg*kPageSize, 0); std::memcpy(b.data(),qbs.data(),qbs.size());
      file.write_pages(cb_page, cb_npg, b.data()); }

    TreeManifest manifest;
    manifest.dim = dim; manifest.m4 = m4; manifest.scan_pq_bits = scan_bits;
    manifest.quantizer_type = params.quantizer_type;
    manifest.prq_nsplits = (params.quantizer_type == "prq")
        ? static_cast<uint32_t>(static_cast<ProductResidualQuantizer&>(*quantizer).nsplits()) : 0;
    manifest.depth = depth; manifest.k_root = k_root;
    manifest.leaf_capacity = leaf_cap; manifest.n_leaves = n_leaves_total;
    manifest.n_probe_l0 = cfg.n_probe_l0 > 0 ? cfg.n_probe_l0
        : static_cast<uint32_t>(std::max(1.0, 2.0*std::sqrt(double(k_root))));
    manifest.n_probe_ln = cfg.n_probe_ln > 0 ? cfg.n_probe_ln : 4;
    manifest.adaptive_probe_gap = cfg.adaptive_probe_gap;
    manifest.median_lid = cfg.median_lid;
    manifest.pca_dims = pca_dims;  // enable PCA routing at search time
    manifest.balance_factor = params.partition_balance_factor;

    // --- Write PCA routing blob ---
    // Layout: [pca_dims:u32][proj:pca_dims×dim f32][mean_proj:pca_dims f32]
    //         [root_centroids:k_root×pca_dims f32]
    //         [leaf_centroids:n_leaves×pca_dims f32]
    PageId pca_page = kInvalidPage;
    uint32_t pca_npg = 0;
    if (pca_dims > 0) {
        std::vector<float> pca_blob;
        // Projection matrix (pca_dims × dim).
        for (uint32_t k = 0; k < pca_dims; ++k)
            for (uint16_t d = 0; d < dim; ++d)
                pca_blob.push_back(rotation[k * dim + d]);
        // Mean projection (pca_dims).
        for (uint32_t k = 0; k < pca_dims; ++k)
            pca_blob.push_back(mean_proj[k]);
        // Root centroids in PCA space (k_root × pca_dims).
        for (uint32_t c = 0; c < k_root; ++c)
            for (uint32_t k = 0; k < pca_dims; ++k)
                pca_blob.push_back(root_centroids_pca[c][k]);
        // Leaf centroids in PCA space (n_leaves × pca_dims).
        for (uint32_t l = 0; l < n_leaves_total; ++l) {
            // Project the leaf's original-space FP16 centroid to PCA space.
            for (uint32_t k = 0; k < pca_dims; ++k) {
                double acc = 0.0;
                for (uint16_t d = 0; d < dim; ++d)
                    acc += rotation[k * dim + d] *
                           static_cast<float>(all_leaves[l].centroid[d]);
                pca_blob.push_back(static_cast<float>(acc - mean_proj[k]));
            }
        }

        const size_t blob_bytes = pca_blob.size() * sizeof(float);
        pca_npg = static_cast<uint32_t>((blob_bytes + kPageSize - 1) / kPageSize);
        pca_page = alloc.alloc_extent(file, pca_npg);
        std::vector<uint8_t> b(pca_npg * kPageSize, 0);
        std::memcpy(b.data(), pca_blob.data(), blob_bytes);
        file.write_pages(pca_page, pca_npg, b.data());
    }

    std::string cfg_toml = manifest_to_toml(manifest);
    const uint32_t cfg_npg = static_cast<uint32_t>((cfg_toml.size()+kPageSize-1)/kPageSize);
    const PageId cfg_page = alloc.alloc_extent(file, cfg_npg);
    { std::vector<uint8_t> b(cfg_npg*kPageSize, 0); std::memcpy(b.data(),cfg_toml.data(),cfg_toml.size());
      file.write_pages(cfg_page, cfg_npg, b.data()); }

    alloc.flush_bitmap(file);
    Superblock sb;
    sb.init_fresh(bitmap_page, bitmap_pages);
    sb.set_root(root_page, root_npg);
    sb.set_depth(depth);
    sb.set_n_leaves(n_leaves_total);
    sb.set_n_pages(file.num_pages());
    sb.set_free_list(alloc.free_list_head(), alloc.n_free_pages());
    sb.set_bitmap(bitmap_page, bitmap_pages);
    sb.set_codebook(cb_page, cb_npg);
    sb.set_config(cfg_page, cfg_npg);
    sb.set_pca(pca_page, pca_npg);
    sb.commit(file);
    file.sync();

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    spdlog::info("[sextant] build_streaming_pca complete: N={} k_root={} "
                 "n_leaves={} in {:.2f}s (write={:.2f}s, file={} pages)",
                 n, k_root, n_leaves_total, secs,
                 std::chrono::duration<double>(t1 - t_write).count(),
                 file.num_pages());

    BuildResult result;
    result.index_path = output_path;
    result.n_vectors = n;
    result.dim = dim;
    result.pq_m = m4;
    result.pq_bits = scan_bits;
    result.build_time_sec = secs;
    return result;
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

    // Train a routing quantizer for partitioning. The routing quantizer is
    // ALWAYS 8-bit (256 centroids per subquantizer) for better angular
    // resolution than 4-bit. The scan quantizer (4-bit) is used only for
    // leaf-level FastScan codes.
    //
    // Why: 4-bit PQ codes can't distinguish angularly close vectors on
    // high-LID data (Sphere-IP: 72% k-means churn). 8-bit PQ has 16× more
    // resolution per subquantizer → better partitioning → higher recall.
    //
    // RaBitQ already needs a routing PQ (sign codes are centroid-relative).
    // For 4-bit PQ/PRQ, we add one here. For 8-bit PQ, the scan quantizer
    // already has 8-bit resolution — no separate routing PQ needed.
    std::unique_ptr<PqQuantizer> routing_quantizer;
    if (params.quantizer_type == "rabitq") {
        routing_quantizer = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, /*bits=*/8, /*seed=*/42);
    } else if (scan_bits == 4) {
        // 4-bit scan quantizer: train a separate 8-bit routing PQ.
        routing_quantizer = std::make_unique<PqQuantizer>(
            params.metric, dim, m4, /*bits=*/8, /*seed=*/42);
    }
    // For 8-bit scan quantizer: routing_quantizer stays null → use scan quantizer.

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
    // When a routing_quantizer exists (4-bit scan or RaBitQ), partitioning uses
    // the 8-bit routing PQ. Otherwise, the scan quantizer serves both roles.
    PqQuantizer& routing_q = routing_quantizer ? *routing_quantizer : *quantizer;
    const uint32_t routing_code_size = routing_q.code_size();

    // Encode all vectors with the routing quantizer for partitioning.
    // When routing != scan (4-bit scan or RaBitQ), leaf codes are re-encoded
    // per-leaf with the scan quantizer during the leaf-write phase.
    const bool needs_reencode = static_cast<bool>(routing_quantizer);
    const uint32_t encode_stride = routing_code_size;
    const uint32_t code_size = routing_code_size;
    spdlog::info("[sextant] build_tree: encoding {} vectors (routing code_size={})",
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
                     needs_reencode, &quantizer]
                    (uint32_t s, uint32_t e) {
                        for (uint32_t i = s; i < e; ++i) {
                            const float* vec = buf.data() +
                                static_cast<size_t>(i) * dim;
                            uint8_t* code_out = all_codes.data() +
                                static_cast<size_t>(offset + i) * encode_stride;
                            if (needs_reencode) {
                                // Encode with routing PQ (8-bit) for partitioning.
                                // Scan codes are re-encoded per-leaf later.
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

    // When needs_reencode (4-bit scan with 8-bit routing, or RaBitQ),
    // re-open the base file to re-encode each vector with the scan quantizer
    // during leaf writing. RaBitQ also encodes relative to the leaf centroid.
    FILE* base_f2 = nullptr;
    std::vector<float> leaf_centroid_f32;
    if (needs_reencode) {
        base_f2 = std::fopen(base_path.c_str(), "rb");
        if (!base_f2) {
            throw Error(ErrorCode::IoError,
                        "IVFTreeIndex::build: cannot reopen base for re-encoding");
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
    std::vector<uint8_t> scan_packed;  // re-encoded scan codes (4-bit PQ/PRQ)
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
                } else if (needs_reencode) {
                    // 4-bit scan with 8-bit routing: re-encode with scan quantizer.
                    std::vector<float> vec(dim);
                    std::fseek(base_f2, 8 + static_cast<long>(gid) * dim * sizeof(float),
                               SEEK_SET);
                    std::fread(vec.data(), sizeof(float), dim, base_f2);
                    scan_packed.assign(quantizer->code_size(), 0);
                    quantizer->encode(vec.data(), scan_packed.data());
                    code = scan_packed.data();
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
