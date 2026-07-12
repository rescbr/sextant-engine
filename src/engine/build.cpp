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
uint16_t resolve_inline_pq(uint16_t override_val) {
    if (override_val != 0xFFFF) return override_val;
    // Balanced preset: inline half of a typical R's worth of neighbor codes.
    // The actual cap is applied later against R, so this is just a hint.
    return 32;
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
        p.L = static_cast<uint16_t>(std::max<uint32_t>(100, 2u * p.R));
    }
    p.L_build = p.L;
    spdlog::info("[sextant] L / L_build = {} [{}]", p.L,
                 overrides.L != 0 ? "override" : "auto");

    // --- alpha (SDC default 1.2) ---
    if (overrides.alpha != 0.0f) {
        p.alpha = overrides.alpha;
    } else {
        p.alpha = 1.2f;
    }
    if (p.alpha == 1.5f) {
        spdlog::warn("[sextant] alpha=1.5 requests ADC build mode, which is "
                     "not yet implemented in Phase 1 (SDC only). Falling back "
                     "to SDC construct.");
    }
    spdlog::info("[sextant] alpha = {:.2f} [{}]", p.alpha,
                 overrides.alpha != 0.0f ? "override" : "auto");

    // --- pq_m ---
    if (overrides.pq_m != 0) {
        p.pq_m = overrides.pq_m;
    } else {
        // clamp(dim/4, 4, 64); also require dim divisible by m (adjusted down).
        uint32_t m = dim / 4;
        if (m < 4) m = 4;
        if (m > 64) m = 64;
        // Ensure dim % m == 0: decrement until it divides.
        while (m > 1 && (dim % m) != 0) {
            m--;
        }
        if (m < 1) m = 1;
        p.pq_m = static_cast<uint8_t>(m);
    }
    spdlog::info("[sextant] pq_m = {} [{}]", p.pq_m,
                 overrides.pq_m != 0 ? "override" : "auto");

    // --- pq_bits ---
    p.pq_bits = overrides.pq_bits;
    spdlog::info("[sextant] pq_bits = {}", p.pq_bits);

    // --- max_occlusion ---
    p.max_occlusion =
        std::min<uint32_t>(4096, std::max<uint32_t>(750, 4u * p.L_build + p.R));
    spdlog::info("[sextant] max_occlusion = {} [auto]", p.max_occlusion);

    // --- inline_pq_count ---
    p.inline_pq_count = resolve_inline_pq(overrides.inline_pq_count);
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
