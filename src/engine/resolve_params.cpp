// resolve_params — auto-default parameter resolution (Issue 37).
// The two-pass streaming + parallel PQ-construct lives in engine.cpp; this
// translation unit owns the auto-default resolution logic.

#include "resolve_params.hpp"
#include "sextant/error.hpp"
#include "sextant/system.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <thread>

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

    // --- merged_graph / pq4_m (IVF-list-scan vs merged-graph) ---
    p.merged_graph = overrides.merged_graph;
    p.partition_balance_factor = overrides.partition_balance_factor;
    p.scan_pq_bits = (overrides.scan_pq_bits == 8) ? 8 : 4;  // sanitize: 4 or 8 only
    if (overrides.pq4_m != 0) {
        p.pq4_m = overrides.pq4_m;
    } else if (dim > 0) {
        // Default m for the 4-bit codebook: dim/4 (192 at dim=768 — matches the
        // validated spike). Kept a multiple of 4 so sub_dim = dim/m is exact.
        uint32_t m4 = dim / 4;
        if (m4 == 0) m4 = 1;
        p.pq4_m = static_cast<uint16_t>(m4);
    }
    spdlog::info("[sextant] merged_graph = {} ({} path); pq4_m = {}",
                 p.merged_graph, p.merged_graph ? "merged-graph" : "IVF-list-scan",
                 p.pq4_m == 0 ? std::string("auto") : std::to_string(p.pq4_m));

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

    // --- pq_anisotropy (covariance-based anisotropic codebook training) ---
    // Pure pass-through; defaults to off.
    p.pq_anisotropy = overrides.pq_anisotropy;
    if (p.pq_anisotropy) {
        spdlog::info("[sextant] pq_anisotropy = on [override] "
                     "(covariance-based anisotropic codebook training ENABLED)");
    }

    // --- pq_opq (OPQ PCA rotation) ---
    // Pure pass-through; defaults to off.
    p.pq_opq = overrides.pq_opq;
    if (p.pq_opq) {
        spdlog::info("[sextant] pq_opq = on [override] "
                     "(OPQ PCA rotation ENABLED)");
    }

    // --- anisotropic_pq (ScaNN-style anisotropic training) ---
    // Pure pass-through; defaults to off. When on, Builder constructs
    // AnisotropicPqQuantizer (subclass of PqQuantizer overriding train()).
    p.anisotropic_pq = overrides.anisotropic_pq;
    if (p.anisotropic_pq) {
        spdlog::info("[sextant] anisotropic_pq = on [override] "
                     "(ScaNN-style anisotropic training; see "
                     "docs/plans/metric_per_tier_plan.md Phase 2)");
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
        uint64_t ram = sextant::physical_ram_bytes();
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
    // Two drivers, take the max (then clamp to [1, K_max]):
    //   ram_driven_K   — from build_ram_budget / per_vec (existing logic).
    //                    Keeps each shard's flat buffers under the RAM budget.
    //   recall_driven_K — applied when sharded_graph is set OR merged_graph is false
    //                     (i.e. any IVF build — scan or graph-inside-shard).
    //                     - scan path (merged_graph=false): target ~200k
    //                       vectors/shard → K = clamp(N/200k, 16, 8192).
    //                       Larger K than graph because each shard is a pure
    //                       sequential stream (no graph to traverse) — smaller
    //                       shards mean less bytes streamed per probe.
    //                       arxiv-nomic 1.34M → K=8; 1B → K=8192 (clamped).
    //                     - graph path (merged_graph=true): target ~64k
    //                       vectors/shard → K = clamp(N/64k, 2, 256). Standard
    //                       DiskANN/FAISS shard size; bigger K degrades graph
    //                       routing recall.
    // An explicit partition_count override wins outright.
    // per_vec = code_size + node_size(R, code_size).
    // code_size = pq_m (for pq_bits=8, 1 byte per segment).
    {
        const bool is_scan = !overrides.merged_graph;
        const uint32_t kKMax = k_max_for_path(is_scan);
        const uint32_t code_sz = static_cast<uint32_t>(p.pq_m);
        const uint32_t node_sz =
            ((16u + static_cast<uint32_t>(p.R) * 4u + 7u) & ~7u);
        const uint64_t per_vec = code_sz + node_sz;

        if (overrides.partition_count > 0) {
            p.partition_count = std::min(overrides.partition_count, kKMax);
            spdlog::info("[sextant] K (partitions) = {} [override]", p.partition_count);
        } else {
            uint32_t ram_driven_k = 1;
            uint64_t max_per_partition = 0;
            if (per_vec > 0 && p.build_ram_budget > 0) {
                max_per_partition = p.build_ram_budget / per_vec;
                if (max_per_partition > 0) {
                    ram_driven_k = static_cast<uint32_t>(
                        (n_vectors + max_per_partition - 1) / max_per_partition);
                    if (ram_driven_k < 1) ram_driven_k = 1;
                }
            }

            const bool is_ivf = overrides.sharded_graph || is_scan;
            const uint32_t recall_driven_k =
                recall_driven_k_floor(n_vectors, is_scan, is_ivf);

            uint32_t k = std::max(ram_driven_k, recall_driven_k);
            k = std::min(k, kKMax);
            p.partition_count = k;

            const uint64_t monolithic_ram = n_vectors * per_vec;
            if (k == 1) {
                spdlog::info("[sextant] K (partitions) = 1 [monolithic, "
                             "flat RAM = {:.1f}MB ≤ budget {:.1f}MB]",
                             monolithic_ram / 1e6,
                             p.build_ram_budget / 1e6);
            } else if (ram_driven_k >= recall_driven_k) {
                spdlog::info("[sextant] K (partitions) = {} [flat RAM {:.1f}MB > "
                             "budget {:.1f}MB → {} shards of ≤{} vectors each]",
                             p.partition_count, monolithic_ram / 1e6,
                             p.build_ram_budget / 1e6, p.partition_count,
                             max_per_partition);
            } else {
                spdlog::info("[sextant] K (partitions) = {} [recall-driven {} "
                             "floor={}; RAM budget {:.1f}MB would allow K={}]",
                             p.partition_count,
                             is_scan ? "scan" : "graph",
                             recall_driven_k, p.build_ram_budget / 1e6,
                             ram_driven_k);
            }
        }
    }

    return p;
}

}  // namespace sextant
