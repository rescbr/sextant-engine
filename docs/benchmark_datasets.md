# Modern ANN benchmark datasets for Sextant

Sextant reads `.fbin` (float32 LE, header `[uint32 n][uint32 dim][n×dim×float32]`).
Many datasets below are already in the compatible BigANN binary format; others
need conversion from HDF5/Parquet/fvecs (see conversion notes at end).

## 1. Modern transformer embeddings (768d–3072d)

These are the highest-priority additions for Sextant — they represent today's
RAG/search workloads that SIFT/GIST cannot.

### 1a. MS-MARCO Web Search (Microsoft) — top pick
- **Dim/Count**: 768d, **100.9M** document vectors; 9,374 queries
- **Model**: SimANS (Bing web relevance encoder)
- **Format**: SPTAG binary (`vectors.bin`, `metaidx.bin`) — needs conversion to `.fbin`
- **Ground truth**: Yes (`truth.txt`)
- **Size**: 289 GB on disk
- **URL**: https://github.com/microsoft/MS-MARCO-Web-Search (embeddings under "100M dataset" section)
- **Clustering**: Real web-search distribution — naturally clustered by topic

### 1b. DBPedia-Entities OpenAI Embeddings (Qdrant) — top pick, direct .fbin fit
- **Dim/Count**: **1536d AND 3072d**, 1M vectors each (parallel datasets)
- **Model**: OpenAI `text-embedding-3-large` (3072d) + `text-embedding-ada-002` (1536d)
- **Format**: Parquet on HuggingFace
- **Ground truth**: No (compute via brute-force); being contributed to BigANN framework
- **Size**: 9.55 GB (1536d), 24.8 GB (3072d)
- **URLs**:
  - 1536d: https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M
  - 3072d: https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M
- **Clustering**: Wikipedia entities — semantically clustered by topic domain

### 1c. Cohere Wikipedia Embeddings — 768d, huge scale
- **Dim/Count**: 768d, **35.2M** rows (English); also German (15M), Simple (486K)
- **Model**: Cohere `multilingual-22-12` / `embed-english-v2.0`
- **Format**: Parquet (each row has `emb` array of 768 floats)
- **Ground truth**: Pre-computed ground truth available for 50M subset (Qdrant benchmark)
- **Size**: ~100 GB (full English)
- **URL**: https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings
- **Qdrant 50M benchmark with GT**: https://github.com/qdrant/benchmark-cohere-wiki-50m

### 1d. Cohere BEIR Embeddings (18 datasets, pre-embedded)
- **Dim**: 1024d (Cohere `embed-english-v3.0`)
- **Count**: Varies — MS-MARCO (8.8M), NQ (2.68M), HotpotQA (5.2M), FEVER (5.4M), Climate-FEVER (5.4M), BioASQ (14.9M), etc.
- **Format**: HuggingFace Parquet
- **Ground truth**: Yes (qrels included — relevance labels)
- **URL**: https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3

### 1e. Cohere MS-MARCO v2.1 (TREC-RAG 2024)
- **Dim/Count**: 1024d, **113.5M** passages; 1,677 queries with top-1000 brute-force hits
- **Model**: Cohere Embed English v3
- **Format**: Parquet (`passages_parquet`, `passages_npy`)
- **Ground truth**: **Yes** — top-1000 from flat index for all 1677 queries
- **URL**: https://huggingface.co/datasets/CohereLabs/msmarco-v2.1-embed-english-v3

### 1f. VIBE Benchmark Datasets (2025) — best curated modern set
18 datasets, HDF5 format, with ground truth, covering modern embedding models:
- **In-distribution (768d+)**:
  - `arxiv-nomic-768-normalized` — 1.34M, Nomic text embedding
  - `ccnews-nomic-768-normalized` — 495K
  - `gooaq-distilroberta-768-normalized` — 1.47M
  - `simplewiki-openai-3072-normalized` — **260K, 3072d**, OpenAI
  - `codesearchnet-jina-768-cosine` — 1.37M, Jina
  - `yahoo-minilm-384-normalized` — 677K, MiniLM
- **Image (1024d–2048d)**:
  - `celeba-resnet-2048-cosine` — 202K, 2048d
  - `imagenet-clip-512-normalized` — 1.28M, 512d CLIP
  - `landmark-dino-768-cosine` — 761K, DINOv2
- **URL**: https://huggingface.co/datasets/vector-index-bench/vibe (individual `.hdf5` files)
- **Paper**: arXiv:2505.17810
- **Each includes**: train, test (queries), ground truth neighbors — ready for `.fbin` conversion

## 2. Large-scale datasets (100M–1B+)

All use the **BigANN `.fbin`/`.u8bin`/`.i8bin` format** — natively compatible or
trivially convertible for Sextant.

### 2a. SPACEV-1B (Microsoft Bing) — best billion-scale pick
- **Dim/Count**: 100d, **1.4B** document vectors; 29,316 queries + 94K query log
- **Model**: Microsoft SpaceV Superior (intent representation)
- **Format**: **`.i8bin`** (int8) — needs cast to float32 for Sextant
- **Ground truth**: **Yes** — 100-NN per query
- **Size**: ~140 GB
- **URLs**:
  - 100M subset (HuggingFace, easier): https://huggingface.co/datasets/unum-cloud/ann-spacev-100m
  - Full 1B: `s3://bigger-ann/spacev-1b/` (via `aws s3 cp --recursive --no-sign-request`)
  - Direct: https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/
- **Clustering**: Web search intent — strongly clustered by query intent

### 2b. Microsoft Turing-ANNS-1B
- **Dim/Count**: 100d, **1B** vectors; 100K queries
- **Model**: Turing AGI v5 (Transformer, Bing query intent)
- **Format**: **`.fbin`** (float32) — **directly Sextant-compatible!**
- **Ground truth**: Yes
- **Size**: ~400 GB
- **URL**: https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/

### 2c. Yandex DEEP-1B / DEEP-100M
- **Dim/Count**: 96d, **1B** vectors; 10K queries (also 10M/100M slices with pre-computed GT)
- **Model**: GoogLeNet last FC layer (image descriptors)
- **Format**: **`.fbin`** (float32) — **directly Sextant-compatible!**
- **Ground truth**: Yes — pre-computed for 1M/10M/100M/1B
- **Size**: ~360 GB (1B), 36 GB (100M)
- **URLs**:
  - 1B base: https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
  - Queries: https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
  - GT for subsets: https://github.com/matsui528/deep1b_gt (deep1M/10M/100M)
- **Clustering**: Image embeddings — moderately clustered by visual similarity

### 2d. BIGANN-1B (SIFT at billion scale)
- **Dim/Count**: 128d, **1B** vectors (uint8 SIFT descriptors); 10K queries
- **Format**: **`.u8bin`** — needs cast to float32 for Sextant
- **Ground truth**: Yes
- **Size**: ~128 GB
- **URL**: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
- **Clustering**: Non-clustered (random images)

### 2e. Yandex Text-to-Image-1B (OOD) — clustered, cross-modal
- **Dim/Count**: 200d, **1B** image vectors; 100K text queries + 50M learn queries
- **Model**: SE-ResNext-101 (images) + DSSM (text queries) — **different distributions**
- **Format**: **`.fbin`** (float32) — **directly Sextant-compatible!**
- **Ground truth**: Yes
- **Size**: ~800 GB
- **URL**: https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
- **Clustering**: **OOD by design** — base (images) and queries (text) in different distributions. Excellent stress test for clustered/OOD behavior.

### 2f. Facebook SimSearchNet++ 1B (SSN++) — copy detection, clustered
- **Dim/Count**: 256d, **1B** vectors (uint8); 100K queries
- **Model**: SimSearchNet++ (image copy detection)
- **Format**: **`.u8bin`** — needs cast to float32
- **Ground truth**: **Yes** (range search GT — AP metric)
- **Size**: ~256 GB
- **URL**: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/FB_ssnpp_database.u8bin
- **Clustering**: **Highly clustered** — near-duplicate image clusters (copy detection workload)

### 2g. MS-MARCO Web Search 10B variant
- **Dim/Count**: 768d, **10B** documents
- **URL**: See https://github.com/microsoft/MS-MARCO-Web-Search "10B dataset" section

## 3. High-dimensional clustered / real-world workloads

### 3a. DISC2021 (Image Similarity Challenge) — duplicate detection
- **Dim/Count**: 512d (SSCD) or 256d; **1M** reference + 50K query images
- **Model**: SSCD (Self-Supervised Copy Detection, ResNet50)
- **Ground truth**: Yes — 10K true copies among queries
- **URL**: https://ai.facebook.com/blog/detecting-manipulated-images-the-image-similarity-challenge-results-and-winners/
- **Clustering**: **Highly clustered** — near-duplicate clusters with varied manipulations (crop, blur, collage)

### 3b. Iceberg Benchmark (2025) — task-centric, clustered domains
7 datasets (1M–100M) with task-specific labels for end-to-end evaluation:

| Dataset | Count | Dim | Domain |
|---|---|---|---|
| ImageNet-DINOv2 | 1.28M | 768 | Image classification |
| ImageNet-EVA02 | 1.28M | 1024 | Image classification |
| ImageNet-ConvNeXt | 1.28M | 1536 | Image classification |
| Glint360K-IR101 | 17.1M | 512 | Face recognition (**clustered**) |
| Glint360K-ViT | 17.1M | 512 | Face recognition (**clustered**) |
| BookCorpus | 9.25M | 1024 | Text retrieval |
| Commerce | 99.1M | 48 | Recommendation (**clustered**) |

- **URL**: https://github.com/ZJU-DAILY/Iceberg
- **Format**: `.bin` files (build/search). Uses DBI clustering metrics to choose IP vs Euclidean.

### 3c. BigVectorBench Artificial Clustered Datasets — designed for clustering stress tests
- **Dim/Count**: 10M vectors, 128d or 768d, with controllable label skew
- Examples: `artificial-average-768d-6l-6a` (768d, 6 labels, 6% skew)
- **URL**: https://huggingface.co/datasets/AnnaZh/Bigvectorbench-artificial-datasets
- **Clustering**: **Explicitly controlled** — designed to test filtered/clustered ANN performance

### 3d. YFCC100M CLIP Embeddings (NeurIPS'23 Filtered Track)
- **Dim/Count**: 192d (quantized uint8), **10M** vectors; 100K queries
- **Model**: CLIP on Yahoo Flickr images
- **Format**: `.u8bin` + tag metadata
- **Ground truth**: Yes
- **URL**: Via `python create_dataset.py --dataset yfcc-10M` in big-ann-benchmarks repo
- **Clustering**: Image content clusters + metadata tags for filtering

## 4. LAION / CLIP multimodal datasets

### 4a. LAION-5B (CLIP ViT-L/14)
- **Dim/Count**: **768d**, **5.85B** image-text pairs
- **Model**: OpenAI CLIP ViT-L/14
- **Format**: `.npy` + Parquet (image + text embeddings separate)
- **100M subset**: https://clickhouse-datasets.s3.amazonaws.com/laion-5b/ (10 Parquet files × 10M rows)
- **URL**: https://clickhouse.com/docs/getting-started/example-datasets/laion-5b-dataset
- **Clustering**: Web image-text — clustered by visual/semantic content

### 4b. LAION-400M (CLIP ViT-B/32)
- **Dim/Count**: **512d**, **400M** pairs (image + text embeddings)
- **Model**: OpenAI CLIP ViT-B/32
- **Format**: `.npy` (float16 embeddings) + Parquet metadata
- **URL**: https://laion.ai/laion-400-open-dataset/ (410 files × ~1M rows)
- **Clustering**: CLIP-filtered web content — clustered by semantics

### 4c. VIBE LAION subset (ready for benchmarking)
- **Dim/Count**: 512d, 1M vectors; 1K queries
- **Format**: HDF5 with ground truth
- **URL**: https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/laion-clip-512-normalized.hdf5
- **Clustering**: Text-to-image OOD setting (text queries vs image corpus)

## 5. Recent benchmark suites & frameworks (2023–2026)

### 5a. Big-ANN Benchmarks (NeurIPS'21 + '23) — primary registry
- **Repo**: https://github.com/harsha-simhadri/big-ann-benchmarks
- **NeurIPS'23 tasks**: Filtered (YFCC-10M), OOD (T2I-10M), Sparse (MS-MARCO/SPLADE 8.8M), Streaming (MS-Turing 30M clustered)
- **Download all**: `for ds in yfcc-10M sparse-full text2image-10M msturing-30M-clustered; do python create_dataset.py --dataset $ds; done`
- All in BigANN `.fbin`/`.u8bin`/`.i8bin` format with ground truth

### 5b. ann-benchmarks (classic registry) — HDF5, all with GT
- **URL**: https://github.com/erikbern/ann-benchmarks
- **Available** (all HDF5, top-100 GT):
  - DEEP1B (96d, 9.99M) — 3.6 GB
  - SIFT (128d, 1M), GIST (960d, 1M)
  - GloVe (25/50/100/200d, 1.18M each)
  - NYTimes (256d, 290K), Fashion-MNIST (784d), MNIST (784d)
  - Last.fm (64d, 292K), COCO (512d)
- **Note**: These are older but well-standardized. HDF5 → `.fbin` conversion needed.

### 5c. VDBBench 1.0 (Milvus/Zilliz, July 2025) — modern embedding models
- **Repo**: https://github.com/zilliztech/VectorDBBench
- **Datasets** (Cohere/OpenAI, designed for RAG):

| Corpus | Model | Dim | Size |
|---|---|---|---|
| Wikipedia | Cohere V2 | 768 | 1M / 10M |
| BioASQ | Cohere V3 | 1024 | 1M / 10M |
| C4 | OpenAI | 1536 | 500K / 5M |
| MSMarco V2 | udever-bloom-1b1 | 1536 | 1M / 10M / 138M |

- **Focus**: Streaming ingestion, filtered search, concurrent workloads

### 5d. BigVectorBench (VLDB 2025)
- **Repo**: https://github.com/BenchCouncil/BigVectorBench
- **Datasets** (HDF5 on HuggingFace: `Patrickcode/BigVectorBench`):
  - `dbpedia-entities` OpenAI 1536d (990K) and 3072d (990K) — big queries
  - `amazon-384` (15.9M, all-MiniLM-L12-v2) — filtered, e-commerce
  - `app_reviews-384` (278K) — filtered
  - `ag_news-384`, `yahoo-384` — text classification
  - `webvid-512` (1M, CLIP) — multi-vector
  - `img-wikipedia-1024` (479K, ImageBind) — multi-modal
  - Artificial datasets with controllable label skew/clustering

### 5e. VIBE (2025) — most comprehensive algorithm comparison
- **Repo**: https://github.com/vector-index-bench/vibe
- 18 datasets (12 in-distribution + 6 OOD), HDF5 with GT
- Includes OOD text-to-image and LLM attention datasets
- Best for comparing Sextant against 21+ SOTA algorithms

### 5f. VecBench (ACM SIGMOD 2025) — filtered search focus
- **Focus**: Controllable filtered vector search, high-dim + large-scale
- Generates synthetic data preserving original similarity distribution
- Adjustable filter selectivity and correlation

### 5g. Qdrant vector-db-benchmark
- **Repo**: https://github.com/qdrant/vector-db-benchmark
- **Datasets**: GloVe, DBPedia-OpenAI (1536d), Cohere-Wiki-50M
- Engine-to-engine comparison framework

### 5h. TopK Bench (2025)
- **Repo**: https://github.com/topk-io/bench
- MS-MARCO + ModernBERT embeddings (768d), 100K/1M/10M sizes
- Pre-computed GT up to top-100
- **S3**: `s3://topk-bench/docs-{100k,1m,10m}.parquet`

## Format conversion guide (to Sextant `.fbin`)

| Source format | Conversion |
|---|---|
| **`.fbin`** (float32) | **None — native Sextant format** |
| **`.u8bin`/`.i8bin`** | Cast each value to float32; keep `[n][dim]` header |
| **`.fvecs`** | Skip 4-byte per-vector dim prefix; prepend `[n][dim]` header |
| **HDF5** (ann-benchmarks/VIBE) | Read `train` dataset → prepend `[n][dim]` header |
| **Parquet** (HuggingFace) | Extract embedding column to contiguous array → prepend header |
| **`.npy`** (LAION) | Load array → prepend `[n][dim]` header (check float16→float32) |

Ground truth conversion (to Sextant if needed): BigANN GT format is
`[n_queries][K][ids][distances]` — ann-benchmarks HDF5 `neighbors` array needs
this header prepended.

## Recommended priority for Sextant

Given Sextant already has SIFT-1M, GIST-1M, and SE-geocoding (768d clustered):

1. **Immediate (free, high-value, ≤1M)**: VIBE datasets (HDF5+GT) — covers
   384d/512d/768d/1024d/2048d/3072d with modern models
2. **Medium (1M–10M, modern embeddings)**: Qdrant DBPedia-OpenAI 1536d/3072d
   (Parquet); Cohere Wikipedia 768d (subset to 1M)
3. **Large-scale stress (100M)**: SPACEV-100M (HF subset), DEEP-100M
   (Yandex `.fbin`), MS-MARCO Web 100M
4. **Clustered/OOD**: Yandex T2I-1B subset, Facebook SSN++ (copy detection
   clustering), Iceberg Glint360K (face recognition clustering)
5. **Billion-scale**: SPACEV-1B, Turing-ANNS-1B, DEEP-1B (all
   `.fbin`/convertible)

**Natively Sextant-compatible (`.fbin` float32, no conversion)**: Turing-ANNS-1B,
DEEP-1B/100M, Yandex T2I-1B — these need only the header which is already the
BigANN standard.
