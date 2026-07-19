// Build pipeline — parameter resolution (Issue 37).
// The two-pass streaming + parallel HDC construct lives in engine.cpp; this
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
    // --- build_mode (HDC only; raw-vector construct removed) ---
    p.build_mode = overrides.build_mode;
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
        // max(L_build, R+1): guarantees the overflow pool's R+1 candidates are
        // never truncated by the occlusion cap. L_build is the beam width (the
        // usual binding constraint); R+1 is the correctness floor.
        p.max_occlusion = std::max<uint32_t>(p.L_build,
                                             static_cast<uint32_t>(p.R) + 1u);
        spdlog::info("[sextant] max_occlusion = {} [auto]", p.max_occlusion);
    }

    // --- inline_pq_count ---
    // Inline PQ is deprecated (two-cache search makes it redundant). The auto
    // sentinel (0xFFFF) is removed; default is 0 (compact). Users who explicitly
    // set >0 get a deprecation warning at flush time.
    p.inline_pq_count = overrides.inline_pq_count;
    // Defensive: an old caller/test may still pass the 0xFFFF auto sentinel.
    // Treat it as 0 (auto → compact).
    if (p.inline_pq_count == 0xFFFF) {
        p.inline_pq_count = 0;
    }
    // Cap to R — can't inline more neighbor codes than neighbors.
    if (p.inline_pq_count > p.R) {
        p.inline_pq_count = p.R;
    }
    spdlog::info("[sextant] inline_pq_count = {} [{}]", p.inline_pq_count,
                 overrides.inline_pq_count != 0 ? "override" : "default");

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
        const float f_target = overrides.closure_f_target > 0.0f
                                   ? overrides.closure_f_target
                                   : 0.15f;
        // d_eff: 0 in overrides → default 5.0 (estimate_config may override later
        // from measured LID; resolve_params is the fast no-sample fallback).
        const float d_eff = overrides.closure_d_eff > 0.0f
                                ? overrides.closure_d_eff
                                : 5.0f;
        float c = std::pow(1.0f - f_target, -1.0f / d_eff);
        if (c < 1.0f) c = 1.0f;
        if (c > 1.2f) c = 1.2f;
        p.closure_factor = c;
        spdlog::info("[sextant] closure_factor = {:.4f} [{}] (f_target={:.2f}, "
                     "d_eff={:.2f})", p.closure_factor,
                     overrides.closure_f_target > 0.0f || overrides.closure_d_eff > 0.0f
                         ? "override" : "auto",
                      f_target, d_eff);
    }

    // --- n_entry_points / n_search_entry_points / target_recall ---
    p.n_entry_points = overrides.n_entry_points > 0 ? overrides.n_entry_points : 16;
    p.n_search_entry_points = overrides.n_search_entry_points > 0
                                  ? overrides.n_search_entry_points : 4;
    p.target_recall = overrides.recall_target;
    // Enable search early-exit when a target_recall is set (default patience 5).
    p.early_exit_patience = (overrides.recall_target > 0.0f) ? 5 : 0;
    spdlog::info("[sextant] n_entry_points = {} [{}], n_search_entry_points = {} [{}], "
                 "target_recall = {} [{}], early_exit_patience = {}",
                 p.n_entry_points,
                 overrides.n_entry_points > 0 ? "override" : "default",
                 p.n_search_entry_points,
                 overrides.n_search_entry_points > 0 ? "override" : "default",
                 p.target_recall > 0 ? std::to_string(p.target_recall) : "off",
                 overrides.recall_target > 0.0f ? "override" : "default",
                 p.early_exit_patience);


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
