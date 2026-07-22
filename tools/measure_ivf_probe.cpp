// measure_ivf_probe — A/B recall measurement for IVF-probe vs merged graph.
//
// Question: how much recall do we lose by NOT merging shards (searching each
// shard independently and routing by centroid distance), and can n_probe
// recover it?
//
// Method:
//   1. Build the baseline merged-graph index at K=1 (normal autobuild path).
//   2. Build K shards in-memory via Builder::build_shards_unmerged.
//   3. For each query:
//      a) Search the merged baseline → recall_baseline.
//      b) Route to n_probe nearest shards by LUT distance to centroids.
//      c) Search each probed shard independently.
//      d) Merge top-k candidates (dedup by row_id) → recall_ivf_probe.
//   4. Print comparison table over (K, n_probe).
//
// Usage:
//   measure_ivf_probe --input base.fbin --queries query.fbin \
//                     --ground-truth gt.gt --index-prefix /tmp/ivftest \
//                     --K 4,16,64 --n-probe 1,2,4 --L 400 --topk 100

#include <numkong/numkong.h>

#include "algo/vamana_core.hpp"
#include "engine/fbin_source.hpp"
#include "engine/partition.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/memgraph.hpp"  // Index holds unique_ptr<MemGraph>
#include "sextant/builder.hpp"
#include "sextant/config.hpp"
#include "sextant/index.hpp"
#include "sextant/searcher.hpp"
#include "sextant/types.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace sextant;

// ---------- fbin / gt readers (minimal — not worth a header) ----------

struct FbinHeader { uint32_t n; uint32_t dim; };

static bool read_fbin_header(const std::string& path, FbinHeader& h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    return f.good() && h.n > 0 && h.dim > 0;
}

static std::vector<float> read_fbin_vectors(const std::string& path, FbinHeader& h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    std::vector<float> v(static_cast<size_t>(h.n) * h.dim);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (!f) { std::cerr << "short read on " << path << "\n"; std::exit(1); }
    return v;
}

struct GroundTruth {
    uint32_t n;  // queries
    uint32_t k;  // gt-k
    std::vector<uint32_t> ids;  // n × k
};

static GroundTruth read_gt(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    GroundTruth gt;
    f.read(reinterpret_cast<char*>(&gt.n), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&gt.k), sizeof(uint32_t));
    gt.ids.resize(static_cast<size_t>(gt.n) * gt.k);
    f.read(reinterpret_cast<char*>(gt.ids.data()),
           static_cast<std::streamsize>(gt.ids.size() * sizeof(uint32_t)));
    if (!f) { std::cerr << "short read on " << path << "\n"; std::exit(1); }
    return gt;
}

// ---------- helpers ----------

static std::vector<uint32_t> parse_int_list(const char* s) {
    std::vector<uint32_t> out;
    const char* p = s;
    while (*p) {
        char* end;
        unsigned long v = std::strtoul(p, &end, 10);
        if (end == p) break;
        out.push_back(static_cast<uint32_t>(v));
        p = end;
        if (*p == ',') p++;
    }
    return out;
}

static float recall_at_k(const std::vector<uint32_t>& result_ids,
                         const GroundTruth& gt, uint32_t qi, uint32_t k) {
    const uint32_t gt_k = std::min(k, gt.k);
    std::vector<uint32_t> truth(gt.ids.begin() + static_cast<size_t>(qi) * gt.k,
                                 gt.ids.begin() + static_cast<size_t>(qi) * gt.k + gt_k);
    std::sort(truth.begin(), truth.end());
    uint32_t hits = 0;
    for (uint32_t i = 0; i < k && i < result_ids.size(); i++) {
        if (std::binary_search(truth.begin(), truth.end(), result_ids[i])) hits++;
    }
    return static_cast<float>(hits) / static_cast<float>(k);
}

// ---------- the measurement ----------

int main(int argc, char** argv) {
    std::string input, queries_path, gt_path, index_prefix;
    std::vector<uint32_t> K_values = {4, 16, 64};
    std::vector<uint32_t> probe_values = {1, 2, 4};
    uint32_t topk = 100;
    uint32_t L_search = 400;
    uint32_t R_override = 0;
    uint16_t pq_m = 0;
    uint8_t pq_bits = 0;
    std::string log_level = "warn";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--input") input = next();
        else if (a == "--queries") queries_path = next();
        else if (a == "--ground-truth") gt_path = next();
        else if (a == "--index-prefix") index_prefix = next();
        else if (a == "--K") K_values = parse_int_list(next().c_str());
        else if (a == "--n-probe") probe_values = parse_int_list(next().c_str());
        else if (a == "--topk") topk = std::stoul(next());
        else if (a == "--L") L_search = std::stoul(next());
        else if (a == "--R") R_override = std::stoul(next());
        else if (a == "--pq-m") pq_m = static_cast<uint16_t>(std::stoul(next()));
        else if (a == "--pq-bits") pq_bits = static_cast<uint8_t>(std::stoul(next()));
        else if (a == "--log-level") log_level = next();
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: measure_ivf_probe --input base.fbin --queries q.fbin "
                         "--ground-truth gt.gt --index-prefix /tmp/ivf\n";
            return 0;
        }
    }
    if (input.empty() || queries_path.empty() || gt_path.empty() || index_prefix.empty()) {
        std::cerr << "missing required args (use --help)\n";
        return 1;
    }

    spdlog::set_level(spdlog::level::from_str(log_level));

    std::cerr << "[step 1] loading queries + gt\n";
    // --- Load queries + GT ---
    FbinHeader qh;
    auto queries = read_fbin_vectors(queries_path, qh);
    GroundTruth gt = read_gt(gt_path);
    std::cerr << "[step 1] done. queries=" << qh.n << " dim=" << qh.dim
              << " gt.n=" << gt.n << " gt.k=" << gt.k << "\n";

    // --- Build baseline (K=1) + get quantizer/training for shards ---
    // We do one autobuild at K=1 to get the baseline index AND a trained
    // quantizer + encoded codes. Then we call build_shards_unmerged which
    // re-uses the codes (no re-encoding).
    ResolvedParams params;
    params.R = R_override > 0 ? R_override : 32;
    params.L = 100;
    params.L_build = 100;
    params.alpha = 1.2f;
    params.pq_m = pq_m > 0 ? pq_m : 96;
    params.pq_bits = pq_bits > 0 ? pq_bits : 8;
    params.max_occlusion = 750;
    params.metric = MetricKind::L2Sq;
    params.num_threads = std::thread::hardware_concurrency();
    params.build_ram_budget = 0;  // K=1 (we control K ourselves below)
    params.partition_count = 1;
    params.n_entry_points = 16;
    params.n_search_entry_points = 4;
    params.target_recall = 0.0f;
    params.early_exit_patience = 0;

    const std::string baseline_path = index_prefix + "_baseline";
    {
        Index build_index;
        Builder build_builder(build_index);
        FbinSource source(input, params.num_threads);
        std::cerr << "[step 2] building baseline (K=1) index...\n";
        auto t0 = std::chrono::steady_clock::now();
        build_builder.build(source, baseline_path, params);
        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "[step 2] built in "
                  << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()
                  << "s — re-opening via Index::read\n";
        // build_index is destroyed here; the baseline_index is opened fresh
        // from sidecars via the production path (same as sextant_bench).
    }
    auto baseline_index = Index::read(baseline_path);
    std::cerr << "[step 2] done — n=" << baseline_index->count
              << " R=" << params.R << "\n";

    // Sanity-check baseline recall via the merged graph.
    // We search the baseline directly (no rerank for speed; rerank doesn't
    // affect the relative comparison).
    std::cerr << "[step 3] constructing baseline searcher ("
              << params.num_threads << " threads)\n";
    Searcher baseline_searcher(*baseline_index, params.num_threads);
    SearchConfig scfg;
    scfg.k = topk;
    scfg.L_search = L_search;
    scfg.io_limit = 0;

    // Pre-build LUTs for all queries once (reused across measurements).
    const uint32_t lut_sz = baseline_index->quantizer->lut_size();
    std::cerr << "[step 4] precomputing " << qh.n << " query LUTs\n";
    std::vector<float> all_luts(static_cast<size_t>(qh.n) * lut_sz);
    for (uint32_t i = 0; i < qh.n; i++) {
        baseline_index->quantizer->preprocess_query(
            &queries[static_cast<size_t>(i) * qh.dim],
            &all_luts[static_cast<size_t>(i) * lut_sz]);
    }

    // --- Baseline recall (merged graph, normal search) ---
    std::cerr << "[step 5] searching baseline (" << qh.n << " queries)\n";
    float baseline_recall = 0.0f;
    for (uint32_t qi = 0; qi < qh.n; qi++) {
        auto cands = baseline_searcher.search(
            &queries[static_cast<size_t>(qi) * qh.dim], topk, scfg);
        std::vector<uint32_t> ids;
        ids.reserve(cands.size());
        for (auto& c : cands) ids.push_back(static_cast<uint32_t>(c.row_id));
        baseline_recall += recall_at_k(ids, gt, qi, topk);
    }
    baseline_recall /= static_cast<float>(qh.n);
    std::cout << "\n=== BASELINE (merged K=1) recall@" << topk << " = "
              << std::fixed << std::setprecision(4) << baseline_recall
              << " (L=" << L_search << ") ===\n\n";

    // --- For each K: build K shards unmerged, sweep n_probe ---
    // We reuse the baseline Index's trained quantizer + encoded codes.
    // build_shards_unmerged needs codes_buffer populated on an Index — we
    // re-run pass1+pass2 into a fresh Index (cheap; no construct).
    std::cerr << "[step 6] starting shard sweep over K={";
    for (size_t i = 0; i < K_values.size(); i++) {
        std::cerr << K_values[i];
        if (i + 1 < K_values.size()) std::cerr << ",";
    }
    std::cerr << "}\n";

    std::cout << "| K    | n_probe | recall@100 | vs baseline | shard_n (avg) |\n";
    std::cout << "|------|---------|------------|-------------|---------------|\n";

    for (uint32_t K : K_values) {
        std::cerr << "[step 6] K=" << K << ": preparing codes (pass1+pass2)\n";
        // Fresh Index just for codes/raw_vecs in RAM. Use prepare_codes to
        // skip the full construct+flush.
        ResolvedParams p1 = params;
        p1.partition_count = 1;
        Index tmp_index;
        Builder tmp_builder(tmp_index);
        FbinSource src2(input, params.num_threads);
        tmp_builder.prepare_codes(src2, p1);
        std::cerr << "[step 6] K=" << K << ": codes ready, building "
                  << K << " shards unmerged\n";

        auto shards = tmp_builder.build_shards_unmerged(p1, K);
        std::cerr << "[step 6] K=" << K << ": shards built\n";

        // Average shard size for reporting.
        double avg_shard_n = 0;
        for (auto& s : shards) avg_shard_n += s.shard_n;
        avg_shard_n /= static_cast<double>(shards.size());

        // Precompute centroid LUT distances for routing: for each query, we
        // need distance(query_lut, centroid_code). Use quantizer.code_distance
        // on the LUT — actually, we want LUT-vs-code which is lut_distance.
        // Simpler: preprocess each centroid to a LUT and use code_distance
        // between centroid codes — no, that gives code-code distance.
        // For routing we want query→centroid distance. The query arrives as
        // FP32; we have its LUT. centroid is a PQ code. So:
        //   d(query, centroid) ≈ lut_distance(centroid_code, query_lut)
        // which is exactly what search uses for candidates.

        // For each n_probe: run the measurement.
        for (uint32_t n_probe : probe_values) {
            if (n_probe > K) continue;
            float ivf_recall = 0.0f;

            // Per-shard TLS (reused across queries).
            std::vector<VamanaTLS> tls_per_shard(K);
            std::vector<std::vector<Candidate>> out_bufs(K);
            for (uint32_t k = 0; k < K; k++) {
                tls_per_shard[k].resize(shards[k].shard_n);
                tls_per_shard[k].resize_lut(lut_sz);
            }

            for (uint32_t qi = 0; qi < qh.n; qi++) {
                const float* query_lut = &all_luts[static_cast<size_t>(qi) * lut_sz];
                const float* query_fp32 =
                    &queries[static_cast<size_t>(qi) * qh.dim];

                // Route: compute distance to each centroid, pick n_probe nearest.
                std::vector<std::pair<float, uint32_t>> cent_dists(K);
                for (uint32_t k = 0; k < K; k++) {
                    if (shards[k].shard_n == 0) {
                        cent_dists[k] = {1e30f, k};
                        continue;
                    }
                    const float d = tmp_index.quantizer->lut_distance(
                        shards[k].centroid_code.data(), query_lut);
                    cent_dists[k] = {d, k};
                }
                std::nth_element(cent_dists.begin(),
                                  cent_dists.begin() + n_probe,
                                  cent_dists.end(),
                                  [](const auto& a, const auto& b) {
                                      return a.first < b.first;
                                  });

                // Search the n_probe nearest shards, collect candidates.
                std::vector<std::pair<float, uint32_t>> scored;  // (dist, global_id)
                scored.reserve(n_probe * topk * 2);
                for (uint32_t p = 0; p < n_probe; p++) {
                    const uint32_t k = cent_dists[p].second;
                    if (shards[k].shard_n == 0) continue;
                    BeamQuery bq;
                    bq.query_lut = query_lut;
                    bq.query_fp16 = nullptr;  // codes-only; no FP16 ball per shard
                    shards[k].core->beam_search_into(
                        out_bufs[k], bq, L_search, 0, tls_per_shard[k]);
                    // Remap local → global.
                    const auto& l2g = shards[k].local_to_global;
                    for (const auto& c : out_bufs[k]) {
                        if (c.row_id < 0 ||
                            static_cast<uint32_t>(c.row_id) >= l2g.size()) continue;
                        scored.emplace_back(c.dist, l2g[static_cast<uint32_t>(c.row_id)]);
                    }
                }
                // Dedup by global ID (closure_factor puts vectors in
                // multiple shards). Keep best distance per ID:
                //   sort by (id, dist asc) → unique keeps min-dist per id
                //   then re-sort by dist asc → take top-k.
                std::sort(scored.begin(), scored.end(),
                          [](const auto& a, const auto& b) {
                              if (a.second != b.second) return a.second < b.second;
                              return a.first < b.first;
                          });
                scored.erase(std::unique(scored.begin(), scored.end(),
                                          [](const auto& a, const auto& b) {
                                              return a.second == b.second;
                                          }),
                              scored.end());
                std::sort(scored.begin(), scored.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
                // Take top-k by distance.
                std::vector<uint32_t> result_ids;
                result_ids.reserve(topk);
                for (uint32_t i = 0; i < topk && i < scored.size(); i++) {
                    result_ids.push_back(scored[i].second);
                }
                ivf_recall += recall_at_k(result_ids, gt, qi, topk);
            }
            ivf_recall /= static_cast<float>(qh.n);

            const float delta = ivf_recall - baseline_recall;
            std::cout << "| " << K << " | " << n_probe << "       | "
                      << std::fixed << std::setprecision(4) << ivf_recall
                      << "    | " << std::setw(10) << std::setprecision(4)
                      << (delta >= 0 ? "+" : "") << delta
                      << "      | " << static_cast<int>(avg_shard_n)
                      << "           |\n";
        }
    }

    // Cleanup tmp files (best-effort).
    for (const std::string& suffix : {".graph", ".codes", ".meta", ".ball",
                                       ".manifest"}) {
        std::remove((index_prefix + "_baseline" + suffix).c_str());
        std::remove((index_prefix + "_codes_tmp" + suffix).c_str());
    }

    return 0;
}
