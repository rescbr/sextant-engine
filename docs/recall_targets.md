# ANN recall targets across workloads: a cited survey

## 1. ANN benchmark papers — standard reporting points (not prescriptions)

The ANN literature universally **reports** recall at fixed thresholds for algorithm comparison. **No paper prescribes these as application requirements** — they are community conventions for the QPS-vs-recall Pareto curve.

| Paper | Venue | Recall metric used | Role |
|---|---|---|---|
| **ANN-Benchmarks** (Aumüller, Bernhardsson, Faithfull) | *Information Systems*, 2020 | Reports full recall@k-QPS Pareto curve; the field's reference framework. Established the convention of plotting QPS at varying recall levels. | **Reporting convention only** |
| **DiskANN** (Subramanya et al.) | NeurIPS 2019 | **1-recall@1 = 95%+** as headline operating point on SIFT1B (<3ms, >5000 QPS). Reports 98.68% as best achievable. Also uses **5-recall@5 = 98%** as a graph-degree tuning target. | Headline operating point (engineering target, not application requirement) |
| **SPANN** (Chen et al.) | NeurIPS 2021 | **recall@1 = 90% and recall@10 = 90%** in ~1ms with 32GB memory; compares at **95% recall@1/recall@10** vs DiskANN. | Operating points for comparison |
| **PipeANN** (Guo & Lu) | OSDI 2025 | **0.9 recall** as the primary comparison point (latency/throughput at 0.9 recall). Also reports recall@10 = 0.99 in benchmarks. | Standard comparison point |
| **HNSW** (Malkov & Yashunin) | IEEE TPAMI 2020 | Controlled by `efSearch`; reports recall across a sweep. Does not fix a single target. | Algorithm parameterization |
| **ScaNN** (Guo et al.) | ICML 2020 | Reports recall@k vs QPS curves; no single threshold prescribed. | Reporting |
| **NeurIPS'21 Big-ANN Challenge** (Simhadri et al.) | 2022 (arXiv 2205.03763) | Uses **10-recall@10**; leaderboards report "recall achieved at minimum throughput 2K QPS" and "QPS at 90% recall." Baselines reach ~0.93–0.98 recall at fixed QPS. | Competition scoring |

**Key observation:** The de facto community thresholds that emerge repeatedly are **0.90, 0.95, and 0.99 recall@k**. A recent critical paper, **"ANN Search: Recall What Matters"** (arXiv 2606.04522, 2026), explicitly notes that "established ANN benchmarks rank algorithms by their throughput at fixed Recall thresholds" of **T ∈ {0.90, 0.95, 0.99}**, and that these have shaped over a decade of research — but argues they are **misaligned with downstream utility**.

## 2. RAG / retrieval pipelines — no hard recall@k requirement; quality is robust to low recall

### DPR (Karpukhin et al., EMNLP 2020)
- **Metric:** Top-k passage retrieval accuracy (recall@k), reported at **top-5, top-20, top-100**.
- **Reported:** DPR achieves **top-5 accuracy = 65.2%** (vs BM25 42.9%) on NQ; top-20 improvement of **9–19% absolute** over BM25.
- **Role:** **Reporting/comparison**, not a prescribed floor. The downstream reader consumes top-100 candidates. No threshold is declared "necessary."

### Atlas (Izacard et al., JMLR 2023)
- **Setup:** Retrieves **top-k passages (k up to ~10–20 for the reader)** from corpora up to 400M passages.
- **Role:** Reports downstream task accuracy (QA, fact-checking); does **not** set a recall@k quality bar. Performance is reported end-to-end.

### "ANN Search: Recall What Matters" (arXiv 2606.04522, 2026) — *the strongest evidence on the recall–downstream relationship*
This paper directly tests whether RAG answer quality degrades with recall, using controlled synthetic recall sweeps (**r ∈ [0.4, 1.0]** at k=10) across 5 datasets (HotpotQA, MS-MARCO, PubMedQA, SciFact, NFCorpus) with Llama 3.1:8b:

> **"Answer quality is effectively decoupled from recall across the full range r ∈ [0.4, 1.0]: BERTScore F1 varies by at most 1.2% on HotpotQA and under 1% on MS-MARCO and PubMedQA. Similarly, the LLM-graded score varies by under 1% on all three datasets."**

- On PubMedQA, the LLM grade varies by **0.07 points on a 10-point scale** across the entire recall sweep.
- **Recall drops 60% (1.0→0.4); downstream RAG quality drops <1–5%.**
- The inverse approximation ratio (1/Ratio@k) tracks true quality with **MAD < 2.6%**, vs Recall's **MAD ~30%**.
- **Role:** Empirical finding (not a prescription) that **high recall is not crucial for RAG downstream tasks**.

### eRAG (Salemi & Zamani, SIGIR 2024, arXiv 2404.13781)
- Proposes evaluating retrievers by correlation with downstream RAG performance rather than recall@k alone. Confirms recall@k is a weak proxy.

## 3. Recommendation systems — recall@k as evaluation metric, not a hard ANN floor

Recommendation systems measure **recall@k as a model-quality metric** (fraction of relevant items in top-k), but this is **retrieval-model recall, not ANN index recall**. The ANN layer is typically tuned to near-exact recall so it doesn't distort model evaluation.

| Source | Recall usage | Role |
|---|---|---|
| **YouTube Deep NN Recommendations** (Covington et al., RecSys 2016) | Candidate generation retrieves top-N from millions via ANN; evaluated by recall/precision@k on held-out. No published ANN-recall floor. | Model eval metric |
| **Mixed Negative Sampling / Two-Tower** (Yang et al., WWW 2020, Google) | Trains two-tower for retrieval; evaluates recall@k of the retrieval model. ANN index assumed accurate. | Model eval metric |
| **PinRec** (Pinterest, arXiv 2504.10507, 2025) | Reports **Recall@K** (whether next target item is in top-K) as an offline metric; A/B tests report engagement lifts (+0.55% time spent). No ANN-recall target stated. | Offline model metric |

**Finding:** No production recommendation paper found prescribes a specific ANN index recall threshold as necessary. The practice is: keep ANN recall high enough (typically **>0.95–0.99** per the systems literature) so retrieval-model metrics aren't contaminated, but this is engineering hygiene, not a cited requirement.

## 4. Entity matching / deduplication — high recall is operationally critical

This is the domain where **missed neighbors are explicitly costly**, and recall is treated as a hard quality bar (the blocking stage must not drop true matches).

| Source | Recall target | Role |
|---|---|---|
| **BlockingPy** (Beręsewicz & Strojny, *SoftwareX*, arXiv 2504.04266, 2025) | Reports **recall = 0.997** (precision 1.0) on census-cis linkage; designed to capture pairs that deterministic blocking **misses**. States user-defined rules "often resulted in imprecise population flow estimates due to missed matches." | **Near-1.0 recall is the operational requirement** — false negatives propagate to wrong population statistics. |
| **Christen (2012), *Data Matching*** (cited 375+×) | The canonical reference: blocking reduction ratio traded off against recall; missed true matches (false negatives) in blocking are unrecoverable downstream. | Frames the recall–cost tradeoff as the central design problem. |
| **Tilores / industry notes** | "Missing matches is costly — e.g. customer database deduplication, or identifying potential fraud where missing a case is worse." Suggests **F-score or recall-weighted** objectives. | Folklore/engineering, consistent with literature. |

**Finding:** Entity resolution is the one domain where literature explicitly frames **recall ≈ 1.0 (or as close as possible) as a near-requirement**, because false negatives at the blocking stage are irrecoverable. BlockingPy targets ~0.997.

## 5. Geospatial / location search — no specific ANN-recall cited requirement found

- Geospatial work traditionally uses **exact** spatial indices (R-trees, Voronoi diagrams: Kolahdouzan & Shahabi, 2004) where recall = 1.0 by construction.
- Recent hybrid work (e.g., **k-RANNS**, VLDB 2024; geo-tagged high-dimensional vectors) reports recall@10 but inherits the ANN convention (**≥0.9**) without application-specific justification.
- **Spanner ANN** (Google Cloud blog) advertises "high-recall" without a specific cited target.
- **Finding:** No peer-reviewed geospatial paper prescribes a specific ANN recall target tied to a location-application requirement. The default ANN convention (0.9–0.95) is inherited.

## 6. Recall@k vs. downstream task performance — where is the knee?

This is the most directly answered question, thanks to recent work:

### "ANN Search: Recall What Matters" (arXiv 2606.04522, 2026) — *best evidence*
- **Image classification (k=100, MNIST/Fashion-MNIST/CIFAR-10/SVHN):** Label Precision stays ≥ **0.943 (normalized)** even at **recall = 0.4**. The knee is effectively *below 0.4* — "downstream quality barely moves."
- **RAG (k=10, 5 datasets, Llama 3.1:8b):** BERTScore F1 varies **<1.2%**; LLM grade **<1%** as recall drops 1.0→0.4.
- **Cost implication:** Optimizing for recall@k = 0.95 costs **1.86×–9.36× more distance computations** than achieving the same 1/Ratio@k threshold; the gap **widens with k and intrinsic dimensionality**. At recall@100 ≥ 0.99, some algorithms (HNSW on Gist/ImageNet; Annoy on all datasets) are **unreachable**, while equivalent 1/Ratio is achievable.
- **Conclusion:** *"Configurations that Recall flags as 'inadequate' can still produce answers of identical quality as exact kNN search."* The recall–utility curve is **essentially flat** for classification and RAG down to recall ≈ 0.4.

### Iceberg (Chen et al., arXiv 2512.12980, 2025)
- Shows that an algorithm winning on recall@k is "often not the best algorithm based on the task-level metric (e.g. label recall, hit rate, matching score)."

### Robustness-δ@K (Wang et al., arXiv 2507.00379, 2025)
- Shows systems with the same *average* recall can produce noticeably different RAG answer quality because of long-tail (hard-query) behavior; proposes counting quality rather than averaging.

### DARTH (Chatzakis et al., *PACMMOD* 2025)
- Turns recall into a service-level objective via adaptive early termination at user-declared recall targets.

**Synthesis on the "knee":** For **RAG and classification**, multiple 2025–2026 papers show **no practical knee above recall ≈ 0.4–0.5** — downstream quality is flat. For **entity resolution**, the effective requirement is recall ≈ **0.99+** (missed matches are unrecoverable). For **recommendations and geospatial**, no published knee exists; convention uses 0.9–0.95.

## Summary table: prescribed vs. reported

| Workload | Typical recall@k in literature | Prescribed (hard) or Reported? | Strongest citation |
|---|---|---|---|
| ANN algorithm benchmarks | 0.90, 0.95, 0.99 | **Reported** (community convention) | Aumüller et al. 2020; NeurIPS'21 Challenge |
| DiskANN-style billion-scale | recall@1 ≥ 0.95 | **Engineering target** (headline operating point) | Subramanya et al. 2019 |
| RAG / open-domain QA | recall@10 measured; **downstream flat to 0.4** | **Reported**; evidence says high recall unnecessary | arXiv 2606.04522 (2026); Karpukhin et al. 2020 |
| Recommendations | recall@k of retrieval model; ANN kept ~0.95+ to avoid contamination | **Reported** (model metric) | Covington et al. 2016; Yang et al. 2020 |
| Entity resolution / dedup | **~0.99+ recall required** | **Near-prescribed** (FNs unrecoverable) | Beręsewicz & Strojny 2025; Christen 2012 |
| Geospatial | inherits 0.9–0.95 | **Reported** (no app-specific requirement) | k-RANNS, VLDB 2024 |

## Key caveats

1. **Almost no paper *prescribes* a recall target as necessary for an application.** The 0.90/0.95/0.99 thresholds are benchmark-reporting conventions, not derived requirements.
2. **The strongest evidence (2025–2026) shows recall@k systematically overstates the importance of exactness** for RAG and classification — downstream quality is robust to recall as low as 0.4.
3. **Entity resolution is the exception** where the literature explicitly demands near-1.0 recall because blocking false negatives are irrecoverable.
4. The emerging consensus (Iceberg, eRAG, "Recall What Matters," Robustness-δ@K) is that **1/Ratio@k, semantic recall, or task-level metrics** better predict downstream utility than raw recall@k.
