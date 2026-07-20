// Engine facade — build/open/search/insert/flush orchestration.
//
// Owns the PqQuantizer + VamanaCore + flat-in-RAM build buffers. The build
// pipeline is:
//   1. resolve_params (auto-defaults from N, dim)
//   2. allocate flat codes + nodes buffers
//   3. Pass 1: reservoir sample (256K) + PQ train
//   4. Pass 2: encode all vectors → index_->codes_buffer
//   5. Parallel HDC construct via CTPL (disjoint node ranges)
//   6. Finalize: compute_entry_points
//   7. Flush sidecars (.graph/.codes/.meta/.manifest)
//
// The search/load path is in search.cpp.

#include "resolve_params.hpp"
#include "fbin_source.hpp"
#include "memory_source.hpp"
#include "partition.hpp"
#include "probe.hpp"
#include "sidecar_io.hpp"
#include "sextant/builder.hpp"
#include "sextant/engine.hpp"
#include "sextant/error.hpp"
#include "sextant/logging.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/direct_io.hpp"
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"
#include "storage/sidecar_header.hpp"

#include <ctpl/ctpl_stl_tls.h>

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#ifdef __linux__
#include <sys/mman.h>  // madvise(MADV_HUGEPAGE) for TLB-friendly build buffers
#endif
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace sextant {

using engine_detail::fill_header;
using engine_detail::read_exact;
using engine_detail::write_padded;

// ---------------------------------------------------------------------------
// VamanaParams::from_resolved — factory used by all build/search cores.
// Defined here (not in vamana_core.cpp) because it needs the full
// ResolvedParams definition from engine.hpp.
// ---------------------------------------------------------------------------
VamanaParams VamanaParams::from_resolved(const ResolvedParams& p, Dim dim,
                                           uint16_t R_override) {
    VamanaParams v;
    v.dim = dim;
    v.R = R_override ? R_override : p.R;
    v.L = p.L;
    v.L_build = p.L_build;
    v.alpha = p.alpha;
    v.n_entry_points = p.n_entry_points;
    v.n_search_entry_points = p.n_search_entry_points;
    v.early_exit_patience = p.early_exit_patience;
    v.max_occlusion = p.max_occlusion;
    return v;
}
// ===========================================================================
// Construction / destruction: defaults (header). Engine is a thin orchestrator
// over Index; the destructor is trivial because Index owns the heavy state.
// ===========================================================================

// ===========================================================================
// Build / insert / flush — Phase B delegators.
// The real work lives in Builder (src/engine/builder.cpp). Engine keeps these
// thin wrappers so existing callers (CLI, benchmark, tests) don't change yet.
// Phase E will delete Engine and callers will use Builder directly.
// ===========================================================================

BuildResult Engine::build(VectorSource& source, const std::string& index_path,
                          const BuildConfig& config) {
    if (!index_) index_ = std::make_unique<Index>();
    Builder b(*index_);
    auto result = b.build(source, index_path, config);
    searcher_ = std::make_unique<Searcher>(*index_);
    searcher_->set_cache_rebalance_enabled(cache_rebalance_enabled_);
    opened_ = true;
    return result;
}

BuildResult Engine::build(VectorSource& source, const std::string& index_path,
                          const ResolvedParams& params) {
    if (!index_) index_ = std::make_unique<Index>();
    Builder b(*index_);
    auto result = b.build(source, index_path, params);
    searcher_ = std::make_unique<Searcher>(*index_);
    searcher_->set_cache_rebalance_enabled(cache_rebalance_enabled_);
    opened_ = true;
    return result;
}

void Engine::insert(const float* vec, Dim dim, RowId row_id) {
    if (!opened_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::insert: index not opened");
    }
    Builder b(*index_);
    b.insert(vec, dim, row_id);
}

void Engine::flush() {
    if (!opened_ || !index_) return;
    Builder b(*index_);
    b.flush();
}

std::vector<Candidate> Engine::search(const float* query, uint32_t k,
                                       const SearchConfig& config) {
    if (!opened_ || !searcher_) {
        throw Error(ErrorCode::InvalidParam,
                    "Engine::search: index not opened");
    }
    return searcher_->search(query, k, config);
}

EstimateResult Engine::estimate_config(VectorSource& source,
                                        const BuildConfig& overrides) {
    // Phase D: thin delegator. The real work is in Estimator (src/engine/
    // estimator.cpp). Engine keeps this wrapper so existing callers (CLI,
    // analyze) don't change yet; Phase E deletes Engine.
    Estimator est;
    return est.estimate_config(source, overrides);
}

}  // namespace sextant
