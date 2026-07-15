// Build pipeline — parameter resolution (Issue 37).
// The two-pass streaming + parallel SDC construct lives in engine.cpp; this
// translation unit owns the auto-default resolution logic.

#include "build.hpp"
#include "sextant/error.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <unistd.h>

namespace sextant {

namespace {

/// Auto-degree R from dataset cardinality (Issue 37 / Issue 16).
uint16_t auto_R(uint64_t n) {
    if (n < 100'000ull) return 32;
    if (n < 1'000'000ull) return 48;
    if (n < 10'000'000ull) return 64;
    if (n < 100'000'000ull) return 96;
    return 128;
}

/// Auto-resolve the inline-PQ count preset (Issue 25). Phase 1 default is the
/// "balanced" preset; the override sentinel 0xFFFF means "auto".
uint16_t resolve_inline_pq(uint16_t override_val, uint64_t n_vectors,
                            uint16_t R, uint32_t code_size,
                            uint64_t cache_budget) {
    if (override_val != 0xFFFF) return override_val;

    // Cache-aware auto-resolution.
    //
    // inline_pq inflates each node by inline_pq × code_size bytes. This trades
    // search I/O (1 read/hop vs 2) for .graph file size. But it also shrinks
    // the effective cache: a 6× larger node means 6× fewer nodes per cache block.
    //
    // Key insight: once a block is cached, there's zero I/O cost difference
    // between 1 read/hop and 2 reads/hop. The inline benefit only matters for
    // actual SSD I/O (cache misses). So inline_pq should only be >0 when the
    // compact (inline_pq=0) graph exceeds the cache budget — i.e., when the
    // working set doesn't fit and every hop risks a real disk read.
    //
    // Resolution:
    //   1. Compute graph_size at inline_pq=0.
    //   2. If it fits in the cache budget → inline_pq=0 (compact, cache-friendly).
    //   3. If not, try inline_pq=R/2, then R, checking SSD space.
    const uint32_t base_node = ((16u + static_cast<uint32_t>(R) * 4u + 7u) & ~7u);
    const uint64_t compact_graph = n_vectors * base_node;

    if (compact_graph <= cache_budget) {
        return 0;  // compact graph fits cache — no inlining needed
    }

    // Graph doesn't fit cache. Inlining saves a read/hop at the cost of cache
    // pollution. Use R/2 as a middle ground (partial inline for closest neighbors).
    return static_cast<uint16_t>(std::min(static_cast<uint32_t>(R) / 2,
                                          static_cast<uint32_t>(R)));
}

/// Physical RAM in bytes. Platform-specific with a portable fallback.
uint64_t physical_ram_bytes() {
#if defined(__APPLE__)
    uint64_t membytes = 0;
    size_t len = sizeof(membytes);
    if (::sysctlbyname("hw.memsize", &membytes, &len, nullptr, 0) == 0 &&
        membytes > 0) {
        return membytes;
    }
#elif defined(__linux__)
    // /proc/meminfo is more reliable than sysconf on Linux, but sysconf is a
    // fine portable fallback. Try sysconf first (no file parse needed).
#endif
    long pages = ::sysconf(_SC_PHYS_PAGES);
    long page_size = ::sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    }
    return 0;
}

}  // namespace

ResolvedParams resolve_params(uint64_t n_vectors, Dim dim,
                              const BuildConfig& overrides) {
    ResolvedParams p;

    // --- R (degree) ---
    if (overrides.R != 0) {
        p.R = overrides.R;
        spdlog::info("[sextant] R (degree) = {} [override]", p.R);
    } else {
        p.R = auto_R(n_vectors);
        spdlog::info("[sextant] R (degree) = {} [auto]", p.R);
    }

    // --- L / L_build ---
    if (overrides.L != 0) {
        p.L = overrides.L;
    } else {
        // L_build = 2*R. This is the beam-search width during construct. The
        // occlusion loop is O(L_build × R), so L_build is the primary cost
        // driver. Empirically validated across three datasets:
        //   SIFT-1M (128-dim):     L=2R=128 → recall 0.9966
        //   GIST-100K (960-dim):   L=2R=96  → recall 0.9074 (PQ-capped)
        //   se_base-100K (768-dim): L=2R=96  → recall 0.9356
        // L=R was tested but collapses recall on high-dim embeddings (se_base
        // drops from 0.94 to 0.36) because beam_search can't navigate the
        // high-dim manifold with a narrow beam. The cost reduction comes from
        // the max_occlusion cap below, not from shrinking L_build.
        p.L = static_cast<uint16_t>(std::max<uint32_t>(2u * p.R, 100u));
    }
    p.L_build = p.L;
    spdlog::info("[sextant] L / L_build = {} [{}]", p.L,
                 overrides.L != 0 ? "override" : "auto");

    // --- alpha (prune threshold; default 1.2) ---
    if (overrides.alpha != 0.0f) {
        p.alpha = overrides.alpha;
    } else {
        p.alpha = 1.2f;
    }
    // --- build_mode (SDC default; ADC uses raw-vector construct) ---
    p.build_mode = overrides.build_mode;
    if (p.build_mode == BuildMode::ADC) {
        spdlog::info("[sextant] build_mode = ADC (raw-vector construct)");
    }
    spdlog::info("[sextant] alpha = {:.2f} [{}]", p.alpha,
                 overrides.alpha != 0.0f ? "override" : "auto");

    // --- pq_m ---
    if (overrides.pq_m != 0) {
        p.pq_m = overrides.pq_m;
    } else {
        p.pq_m = 0;  // auto: resolved by probing on the pass1 reservoir.
    }
    spdlog::info("[sextant] pq_m = {} [{}]", p.pq_m == 0 ? "auto" : std::to_string(p.pq_m),
                 overrides.pq_m != 0 ? "override" : "auto");

    // --- pq_bits ---
    // 0 = auto (resolved by global probe in pass1); 4 or 8 = explicit.
    if (overrides.pq_bits != 0 && overrides.pq_bits != 4 && overrides.pq_bits != 8) {
        throw Error(ErrorCode::InvalidParam,
                    "pq_bits must be 0 (auto), 4, or 8");
    }
    p.pq_bits = overrides.pq_bits;
    spdlog::info("[sextant] pq_bits = {} [{}]", p.pq_bits == 0 ? "auto" : std::to_string(p.pq_bits),
                 overrides.pq_bits != 0 ? "override" : "auto");

    // --- pq_max_distortion ---
    // Upper bound on acceptable PQ distortion (median |1 - pq_dist/true_dist|) for
    // auto (m, bits) selection. Only used when pq_m or pq_bits is auto. The
    // bound filters the candidate set; the elbow of the distortion-cost Pareto
    // frontier does the actual selection. See BuildConfig::pq_max_distortion
    // for the full rationale.
    p.pq_max_distortion = overrides.pq_max_distortion > 0.0f
                              ? overrides.pq_max_distortion
                              : 0.05f;
    if (overrides.pq_m == 0 || overrides.pq_bits == 0) {
        spdlog::info("[sextant] pq_max_distortion = {:.2f} [{}]",
                     p.pq_max_distortion,
                     overrides.pq_max_distortion > 0.0f ? "override" : "default");
    }

    // --- max_occlusion ---
    if (overrides.max_occlusion != 0) {
        p.max_occlusion = overrides.max_occlusion;
        spdlog::info("[sextant] max_occlusion = {} [override]", p.max_occlusion);
    } else {
        // Adaptive: cap the candidate pool at L_build. The old default
        // (max(750, 4*L_build+R)) never engaged since beam_search returns
        // ~L_build < 750 candidates. Capping at L_build means the occlusion
        // loop processes at most L_build² pairs instead of L_build × 750.
        p.max_occlusion = p.L_build;
        spdlog::info("[sextant] max_occlusion = {} [auto]", p.max_occlusion);
    }

    // --- inline_pq_count ---
    // Cache-aware resolution: inline_pq inflates nodes 6× but saves a read/hop.
    // Only inline when the compact graph doesn't fit the cache budget (Issue 36).
    {
        uint64_t phys_ram = physical_ram_bytes();
        uint64_t cache_budget = phys_ram > 0 ? phys_ram * 35 / 100 : 0;
        p.inline_pq_count = resolve_inline_pq(
            overrides.inline_pq_count, n_vectors, p.R, p.pq_m, cache_budget);
    }
    // Cap to R — you can't inline more neighbor codes than there are neighbors.
    if (p.inline_pq_count > p.R) {
        p.inline_pq_count = p.R;
    }
    spdlog::info("[sextant] inline_pq_count = {} [{}]", p.inline_pq_count,
                 overrides.inline_pq_count != 0xFFFF ? "override" : "auto");

    // --- metric ---
    p.metric = overrides.metric;

    // --- num_threads ---
    if (overrides.num_threads != 0) {
        p.num_threads = overrides.num_threads;
    } else {
        unsigned hw = std::thread::hardware_concurrency();
        p.num_threads = hw > 0 ? hw : 1;
    }
    spdlog::info("[sextant] num_threads = {} [{}]", p.num_threads,
                 overrides.num_threads != 0 ? "override" : "auto");

    // --- build_ram_budget (50% of physical RAM) ---
    if (overrides.build_ram_budget != 0) {
        p.build_ram_budget = overrides.build_ram_budget;
    } else {
        uint64_t ram = physical_ram_bytes();
        p.build_ram_budget = ram > 0 ? ram / 2 : 0;
    }
    spdlog::info("[sextant] build_ram_budget = {} bytes ({:.1f}MB) [{}] — "
                 "max RAM per shard, not total allocation",
                 p.build_ram_budget, p.build_ram_budget / 1e6,
                 overrides.build_ram_budget != 0 ? "override" : "auto");

    // --- closure_factor (Issue 24: ~15% replication) ---
    // c = (1 - f_target)^{-1/d_eff}, clamped to [1.0, 1.2].
    {
        constexpr float f_target = 0.15f;
        constexpr float d_eff = 5.0f;
        float c = std::pow(1.0f - f_target, -1.0f / d_eff);
        if (c < 1.0f) c = 1.0f;
        if (c > 1.2f) c = 1.2f;
        p.closure_factor = c;
        spdlog::info("[sextant] closure_factor = {:.4f} [auto]", p.closure_factor);
    }

    // --- K (partition count) ---
    // per_vec = code_size + node_size(R, inline_pq=0, code_size).
    // code_size = pq_m (for pq_bits=8, 1 byte per segment).
    {
        const uint32_t code_sz = static_cast<uint32_t>(p.pq_m);
        const uint32_t node_sz =
            ((16u + static_cast<uint32_t>(p.R) * 4u + 7u) & ~7u);
        const uint64_t per_vec = code_sz + node_sz;
        uint32_t k = 1;
        uint64_t max_per_partition = 0;
        if (per_vec > 0 && p.build_ram_budget > 0) {
            max_per_partition = p.build_ram_budget / per_vec;
            if (max_per_partition > 0) {
                k = static_cast<uint32_t>(
                    (n_vectors + max_per_partition - 1) / max_per_partition);
                if (k < 1) k = 1;
            }
        }
        p.K = k;
        const uint64_t monolithic_ram = n_vectors * per_vec;
        if (k == 1) {
            spdlog::info("[sextant] K (partitions) = 1 [monolithic, "
                         "flat RAM = {:.1f}MB ≤ budget {:.1f}MB]",
                         monolithic_ram / 1e6,
                         p.build_ram_budget / 1e6);
        } else {
            spdlog::info("[sextant] K (partitions) = {} [flat RAM {:.1f}MB > "
                         "budget {:.1f}MB → {} shards of ≤{} vectors each]",
                         p.K, monolithic_ram / 1e6,
                         p.build_ram_budget / 1e6, p.K, max_per_partition);
        }
    }

    return p;
}

}  // namespace sextant
