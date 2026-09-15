#pragma once

/// @file leaf_coder.hpp
/// Per-family leaf coder abstraction.
///
/// ONE C++ class per quantizer family owns EVERYTHING family-specific:
/// leaf byte layout (offsets/extent sizes), build-time training + encoding,
/// per-leaf state regions (levels / codebook / centroid), the scan kernel,
/// the rerank branch, and the mutation paths (append on insert, refit on
/// split, compact on delete). ivf_tree_index.cpp holds a single
/// unique_ptr<LeafCoder> and contains NO quantizer_type strings — the only
/// place the family name exists is coder_factory (build + open).
///
/// Layout ownership: each coder computes its own region offsets from the
/// TreeLeafHeader (count / m4 / pq_bits / summary_size / leaf_state), using
/// the helpers in tree_nodes.hpp. This replaces the scattered 4-way offset
/// dispatch (search rowid materialization, filter layout, debug, split,
/// insert) and fixes the LeafFilterLayout hardcoded-global-PQ bug.
///
/// On-disk format is UNCHANGED by this refactor (commit 1 is
/// bit-identical); the FastScan 32-vec scalar block layout lands later
/// INSIDE local_scalar_coder / scalar_lm_coder only.

#include "tree/tree_nodes.hpp"
#include "sextant/types.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sextant {
class PqQuantizer;
class ScalarLloydMaxQuantizer;
}

namespace sextant::tree {

/// Per-thread i8-scan-kernel override (C ABI harnesses). >0 forces a mode,
/// 0 forces off, -1 (default) lets the env-resolved default decide.
/// Defined in coder_factory.cpp; set via set_scan_i8_override().
namespace scan_detail {
int scan_i8_override();
void set_override(int v);  // C ABI thread-local override
}  // namespace scan_detail
}  // namespace sextant::tree

namespace sextant::tree {

/// Quantizer family discriminator for the few remaining generic checks
/// (brute-force-filtered support, delete_batch support). NOT a dispatch
/// mechanism — the virtuals below are.
enum class CoderFamily : uint8_t {
    GlobalPq,     // pq / anisotropic_pq / prq
    LocalPq,      // local_pq
    ScalarLm,     // scalar_lloydmax / scalar_uniform / scalar_shape
    LocalScalar,  // local_scalar
};

/// Resolved layout of one leaf extent (derived from its header).
struct LeafGeometry {
    uint64_t codes_offset = 0;   // first code byte (or per-leaf state start)
    uint64_t rowids_offset = 0;  // row_ids array
    uint64_t filter_offset = 0;  // filter column data (== rowids end)
    uint64_t extent_bytes = 0;   // header..rowids end, EXCLUDING filter cols
};

/// One scan candidate in the bounded top-W heap. `dist_key` preserves
/// float score order as an order-preserving u32; `leaf_slot` indexes the
/// scan candidate list; `local_idx` is the vector's index within the leaf.
struct HeapEntry {
    uint32_t pq_dist;
    uint32_t leaf_slot;
    uint32_t local_idx;
};

/// Total order on scan-heap entries: (pq_dist, leaf_slot, local_idx).
/// The tiebreak makes the bounded top-W set DETERMINISTIC regardless of
/// scan order — required for bit-exact equivalence between query-major
/// search() and leaf-major search_batch() (both scan the same per-query
/// candidate list, so leaf_slot/local_idx agree across paths).
inline bool heap_entry_less(const HeapEntry& a, const HeapEntry& b) {
    if (a.pq_dist != b.pq_dist) return a.pq_dist < b.pq_dist;
    if (a.leaf_slot != b.leaf_slot) return a.leaf_slot < b.leaf_slot;
    return a.local_idx < b.local_idx;
}

/// Sift-down replacement of the heap root (worst entry by heap_entry_less).
inline void heap_replace(std::vector<HeapEntry>& h, uint32_t new_d,
                         uint32_t leaf_slot, uint32_t local_idx) {
    h[0] = {new_d, leaf_slot, local_idx};
    uint32_t pos = 0;
    const uint32_t n = static_cast<uint32_t>(h.size());
    while (true) {
        const uint32_t left = 2 * pos + 1;
        const uint32_t right = 2 * pos + 2;
        uint32_t largest = pos;
        if (left < n && heap_entry_less(h[largest], h[left]))
            largest = left;
        if (right < n && heap_entry_less(h[largest], h[right]))
            largest = right;
        if (largest == pos) break;
        std::swap(h[pos], h[largest]);
        pos = largest;
    }
}

/// The bounded top-W max-heap the scan kernels push into, passed as a plain
/// struct (NOT virtually — per-candidate calls sit in the hottest loop; a
/// virtual sink measurably regressed the arith kernel via register
/// pressure). The free functions below reproduce the original inline heap
/// logic exactly: push-back until W, make_heap at W, then sift-down
/// replace of the front.
struct RawScanHeap {
    std::vector<HeapEntry>* h = nullptr;
    uint32_t w = 0;
    uint32_t leaf_slot = 0;
};

inline bool heap_full(const RawScanHeap& s) {
    return s.h->size() >= s.w;
}
inline void heap_push(RawScanHeap& s, uint32_t dist_key, uint32_t local_idx) {
    s.h->push_back({dist_key, s.leaf_slot, local_idx});
    if (s.h->size() == s.w)
        std::make_heap(s.h->begin(), s.h->end(), heap_entry_less);
}
inline uint32_t heap_front(const RawScanHeap& s) {
    return s.h->front().pq_dist;
}
/// Sentinel entry marking "not yet filled": sorts worst in heap_entry_less
/// (max pq_dist) so real entries always displace it.
inline bool heap_entry_is_sentinel(const HeapEntry& e) {
    return e.pq_dist == 0xFFFFFFFFu;
}
/// Pre-fill a bounded top-W heap with sentinel entries. Every real entry
/// then takes the guarded replace path, so the final bounded set is
/// EXACTLY the W smallest by heap_entry_less — independent of arrival
/// order. Required for bit-exact equivalence between query-major
/// search() (candidate-order scans) and leaf-major search_batch()
/// (page-order sweeps); without it, the first W encountered entries are
/// kept un-evicted and the survivor set depends on scan order.
inline void heap_init(std::vector<HeapEntry>& h, uint32_t w) {
    h.clear();
    h.assign(w, {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu});
}
/// Drop sentinel entries and restore the heap invariant (feedback loop
/// reads heap_front between scans).
inline void heap_compact(std::vector<HeapEntry>& h) {
    h.erase(std::remove_if(h.begin(), h.end(), heap_entry_is_sentinel),
            h.end());
    std::make_heap(h.begin(), h.end(), heap_entry_less);
}
/// Should an entry (dist_key, local_idx) from THIS leaf displace the heap
/// front? Yes when strictly closer, or when tying pq_dist AND beating the
/// front on the total order's tiebreak. Kernels use this instead of a
/// bare `dist_key < heap_front()` so the bounded set is exactly the W
/// smallest by heap_entry_less (order-independent under ties).
inline bool heap_should_replace(const RawScanHeap& s, uint32_t dist_key,
                                uint32_t local_idx) {
    const HeapEntry& f = s.h->front();
    if (dist_key != f.pq_dist) return dist_key < f.pq_dist;
    if (s.leaf_slot != f.leaf_slot) return s.leaf_slot < f.leaf_slot;
    return local_idx < f.local_idx;
}
inline void heap_replace_top(RawScanHeap& s, uint32_t dist_key,
                             uint32_t local_idx) {
    heap_replace(*s.h, dist_key, s.leaf_slot, local_idx);
}

/// Order-preserving u32 encoding of a float score.
inline uint32_t f32_to_dist_key(float dist) {
    uint32_t bits;
    std::memcpy(&bits, &dist, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/// Per-query, per-leaf scan context produced by scan_setup(). Family-owned
/// (the concrete coder casts to its derived setup); holds whatever the
/// kernel needs — quantized query a_d / hi-lo split, LUTs, c0, bias ptr.
class ScanSetup {
public:
    virtual ~ScanSetup() = default;
};

/// Build/mutation scratch owned by the coder (e.g. leaf codebook trainer,
/// per-leaf buffers). One instance per index, guarded externally like the
/// quantizer objects it replaces.
class CoderBuildState {
public:
    virtual ~CoderBuildState() = default;
};

/// Immutable per-index coder configuration (from manifest + resolved params).
struct CoderParams {
    uint16_t dim = 0;
    uint16_t m4 = 0;
    uint8_t pq_bits = 4;
    MetricKind metric = MetricKind::L2Sq;
    bool has_ip_bias = false;    // scalar families + InnerProduct
    uint32_t prq_nsplits = 0;    // prq only
    uint8_t scan_i8_mode = 0;    // scalar scan kernel selection (0/1/2)
    uint32_t prq_beam_size = 1;  // prq only (build)
    // PRQ encode strategy (build): greedy / beam / icm + ICM/ILS/LSQ
    // counts. Threaded from ResolvedParams (CLI --prq-* flags).
    std::string prq_encode_mode = "greedy";
    uint32_t prq_icm_iters = 4;
    uint32_t prq_ils_iters = 4;
    uint32_t prq_ils_perturb = 4;
    uint32_t prq_lsq_train_iters = 0;
    /// summary_size is not needed by coders (it rides on the leaf header).
};

class LeafCoder {
public:
    virtual ~LeafCoder() = default;

    // --- identity ---
    virtual LeafState leaf_state() const = 0;
    virtual CoderFamily family() const = 0;
    virtual std::string family_name() const = 0;  // for logs/diagnostics ONLY
    virtual MetricKind metric() const = 0;
    /// True when the superblock carries a global state blob for this family.
    virtual bool has_global_state() const { return true; }
    /// Global PqQuantizer view (global families) or nullptr (local/scalar).
    virtual const PqQuantizer* pq_quantizer() const { return nullptr; }
    /// Effective prq nsplits (prq only; 0 otherwise) — recorded in the
    /// manifest at build time.
    virtual uint32_t prq_nsplits() const { return 0; }
    /// True when this family's leaves carry the per-vector fp16 IP bias
    /// region (scalar families under InnerProduct).
    virtual bool leaf_has_ip_bias() const { return false; }

    // --- layout (pure functions of the header) ---
    virtual LeafGeometry geometry(const TreeLeafHeader* h) const = 0;
    /// Extent bytes for a leaf with `count` live vectors (build/insert sizing).
    virtual uint64_t extent_bytes(uint32_t count, uint32_t summary_size,
                                  uint64_t filter_cols_bytes) const = 0;
    /// Bytes per packed code (flat families) or 0 (block families).
    virtual uint32_t code_size() const = 0;

    // --- build ---
    /// Train global state (no-op for local families; local state is fitted
    /// per leaf at flush/split time).
    virtual void train(const float* sample, uint32_t n) = 0;
    /// Thread budget for train() (no-op default: most families either
    /// train internally parallel or are local/serial-by-context). Callers
    /// pin it to the build's configured thread count.
    virtual void set_train_threads(uint32_t) {}
    /// Serialize global state (codebook/ruler blob) for the superblock;
    /// returns false when the family stores no global blob (local families).
    virtual bool serialize_global(std::vector<uint8_t>& out) const = 0;
    virtual bool deserialize_global(const uint8_t* data, uint64_t size) = 0;
    /// True for local families, whose build emission keeps the raw FP16
    /// vectors in the leaf buffers (per-leaf state is fitted at flush).
    virtual bool stores_raw_vectors_during_build() const { return false; }
    /// Encode one vector at build time. `code_out` is code_size() bytes
    /// (flat) or unused (block families interleave at flush). `ip_bias_out`
    /// is set when params.has_ip_bias.
    virtual void encode(const float* vec, uint8_t* code_out,
                        float* ip_bias_out) = 0;

    /// Inputs for flush-time leaf emission (build path). Local families fit
    /// their per-leaf state from `fp16_vecs` / `centroid_f32` here; flat and
    /// global families consume the pre-encoded `codes`.
    struct LeafFlushInput {
        uint32_t count = 0;
        const float16_t* fp16_vecs = nullptr;  // count × dim (always present)
        const float* centroid_f32 = nullptr;   // dim
        const uint8_t* codes = nullptr;        // count × code_size()
        const float16_t* ip_biases = nullptr;  // count (has_ip_bias) or null
        uint32_t summary_size = 0;
    };
    /// Write this coder's leaf regions (per-leaf state + codes + ip biases)
    /// into `leaf_out` (header + summary + payload header fields are already
    /// written; the coder sets leaf_state / centroid_offset / codebook_offset
    /// as needed). Returns the row_ids byte offset.
    virtual uint64_t flush_leaf(const LeafFlushInput& in,
                                uint8_t* leaf_out) = 0;

    // --- search ---
    /// Build the per-query scan context (quantize query, build LUTs).
    /// Called once per query for global families; per (query, leaf) for
    /// local families — the coder declares which via per_leaf_setup().
    virtual std::unique_ptr<ScanSetup> scan_setup(const float* query) = 0;
    virtual bool per_leaf_setup() const = 0;
    /// Refit a setup to a specific leaf (reads per-leaf levels/codebook).
    virtual void bind_leaf(ScanSetup& setup, const uint8_t* leaf) const = 0;
    /// Scan one leaf, pushing candidates into the sink. `leaf` points at
    /// the TreeLeafHeader. Must be thread-safe given per-worker setups.
    virtual void scan_leaf(const ScanSetup& setup, const uint8_t* leaf,
                           RawScanHeap& heap) = 0;
    /// True when scan_leaf_batch() exploits multi-query decoding (one
    /// pass over the leaf's code rows shared by up to 4 queries).
    virtual bool supports_batch_scan() const { return false; }
    /// Scan one leaf for up to 4 queries (`n` <= 4): `setups`/`heaps`
    /// are parallel per-query arrays, all already bound to this leaf
    /// (bind_leaf called per setup by the caller). The default loops
    /// scan_leaf per query — identical results, no decode sharing.
    virtual void scan_leaf_batch(const ScanSetup* const setups[4],
                                 const uint8_t* leaf,
                                 RawScanHeap* const heaps[4], uint32_t n) {
        for (uint32_t q = 0; q < n; ++q)
            scan_leaf(*setups[q], leaf, *heaps[q]);
    }
    /// Exact(ish) rerank of one candidate: returns the refined distance.
    /// When `scratch_decoded` != nullptr it receives the decoded vector
    /// (dim floats); families whose rerank never materializes a decode may
    /// leave it untouched.
    virtual float rerank(const float* query, const uint8_t* leaf,
                         uint32_t local_idx, float* scratch_decoded) = 0;
    /// Setup-aware rerank (search calls this form): coders with per-query
    /// rerank tables (global-PQ LUT rerank) consume their ScanSetup here.
    /// Default forwards to the setup-less overload.
    virtual float rerank(const float* query, const ScanSetup& setup,
                         const uint8_t* leaf, uint32_t local_idx,
                         float* scratch_decoded) {
        (void)setup;
        return rerank(query, leaf, local_idx, scratch_decoded);
    }

    // --- mutation ---
    /// Decode one vector's code to f32 (used by split's k-means and the
    /// post-split re-encode of local families).
    virtual void decode_one(const uint8_t* leaf, uint32_t local_idx,
                            float* out) const = 0;
    /// Extract all `count` compact codes from the leaf into `codes`
    /// (count × code_size()). Used by split / delete.
    virtual void extract_codes(const uint8_t* leaf, uint32_t count,
                               uint8_t* codes) const = 0;
    /// Split planning: partition the leaf's vectors into two groups and
    /// produce the two post-split FP16 routing centroids. Global PQ runs
    /// k-means in code space (uses `codes`); the other families run a small
    /// f32 Lloyd loop on the decoded `vecs`.
    struct SplitPlan {
        std::vector<uint32_t> group0, group1;
        std::vector<float16_t> cent0_fp16, cent1_fp16;
    };
    virtual SplitPlan plan_split(const uint8_t* leaf, const uint8_t* codes,
                                 const float* vecs, uint32_t count,
                                 uint32_t leaf_id) = 0;

    /// Inputs for post-split re-encode of one half.
    struct GroupEncodeInput {
        uint32_t count = 0;
        const float* vecs = nullptr;           // count × dim decoded f32
        const uint8_t* src_codes = nullptr;    // count × code_size (old leaf)
        const float16_t* src_biases = nullptr; // count (has_ip_bias) or null
        const RowId* row_ids = nullptr;        // count (already grouped)
        uint32_t summary_size = 0;
    };
    /// Write the new leaf's code region + per-leaf state + row_ids (the
    /// header with the new count, and the summary, are already in place).
    /// Local families refit levels / retrain the codebook here; global
    /// families re-interleave; scalar_lm flat-copies the old codes.
    virtual void encode_group(const GroupEncodeInput& in,
                              uint8_t* leaf_out) = 0;

    /// Inputs for insert_batch appends into a frozen leaf.
    struct AppendInput {
        const uint8_t* old_leaf = nullptr;  // old extent buffer
        uint32_t old_count = 0;
        uint32_t new_count = 0;
        const float* const* vecs = nullptr; // vectors to append, in order
        uint32_t n_vecs = 0;
        uint8_t* new_leaf = nullptr;        // new extent buffer (hdr+summary copied)
    };
    /// Copy the old leaf's code region + per-leaf state into `new_leaf` and
    /// append the encoded codes (+ ip biases) for the incoming vectors.
    /// Caller handles extent growth, row_ids, filter columns, payload.
    virtual void append_encode(const AppendInput& in) = 0;
    /// Compact after deletes: rewrite the code region for the surviving
    /// row order (default: unsupported → throws, matching today's guard).
    virtual void compact(const uint8_t* leaf_in, const uint32_t* keep_idx,
                         uint32_t keep_count, uint8_t* leaf_out) {
        (void)leaf_in; (void)keep_idx; (void)keep_count; (void)leaf_out;
        throw std::runtime_error("delete_batch unsupported for " +
                                 family_name());
    }
};

/// quantizer_type string → LeafCoder. The ONLY translation point of the
/// family strings (called from build resolve and open). Throws on unknown
/// (same message style as the old open()). `params` is IN/OUT: scalar
/// families normalize m4 to dim; the scan_i8 override (-1=auto, via
/// set_scan_i8_override) is
/// resolved into scan_i8_mode here once. `for_open` selects open-time
/// construction (deserialization-friendly defaults) vs build-time
/// (trainer selection from ResolvedParams).
std::unique_ptr<LeafCoder> make_leaf_coder(const std::string& quantizer_type,
                                           CoderParams& params, bool for_open);

}  // namespace sextant::tree
