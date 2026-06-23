# Academic Paper Recommendation System

A content-based recommender for arXiv computer-science papers that compares two
scientific text-embedding models — **SPECTER2** and **SciBERT** — and serves
recommendations through an interactive Streamlit dashboard.

The project covers the full pipeline: data collection from the arXiv and
Semantic Scholar APIs, cleaning and automated annotation, embedding generation,
a cosine-similarity recommender with a validated category boost, retrieval
evaluation, and a multi-tab web frontend. It was built as a master's thesis
project.

> **Note on data:** the corpus, embedding matrices, and most generated results
> are large and are **not** committed to the repository (see `.gitignore`). The
> pipeline below regenerates them. Embedding generation needs a CUDA GPU.

---

## Features

- **Two embedding models, head-to-head** — SPECTER2 (`allenai/specter2_base` with
  the proximity adapter, CLS pooling) and SciBERT (`allenai/scibert_scivocab_uncased`,
  mean pooling), both producing 768-dim vectors over `title [SEP] abstract`.
- **Two query modes** — *title* (fuzzy-match a known paper and recommend similar
  work using its precomputed embedding) and *keywords* (embed free-text on the
  fly; SPECTER2 uses its `adhoc_query` adapter).
- **Validated ranking** — cosine similarity with a multiplicative same-category
  boost, `final = cosine × (1 + α · same_category)`, with `α = 0.03` chosen from
  an alpha sweep.
- **Honest evaluation** — known-item retrieval (MRR, Recall@K, NDCG@K), a
  temporal-split proxy experiment, and a SPECTER2+SciBERT late-fusion experiment,
  all tracked in Weights & Biases.
- **Interactive dashboard** — five Streamlit tabs: Browse, Recommendations,
  Analytics, Evaluation, and Project Info.

---

## The corpus

After collection and cleaning the dataset contains **49,992 papers** across four
arXiv categories, balanced across six years (2021–2026, ~8,333 papers/year).

| Category | Papers | Share |
|----------|-------:|------:|
| cs.AI — Artificial Intelligence | 22,876 | 45.8% |
| cs.LG — Machine Learning | 12,205 | 24.4% |
| cs.CV — Computer Vision | 7,840 | 15.7% |
| cs.CL — Natural Language Processing | 7,071 | 14.1% |

Cleaning retained 99.99% of collected rows (6 duplicate titles removed). 58.3%
of papers have at least one citation and 79.7% have reference counts, fetched
from Semantic Scholar. Full numbers are in `reports/preprocessing_report.txt`.

---

## How it works

**1. Data collection** (`src/data_collection.py`, `merge_raw_files.py`) — pulls
paper metadata from the arXiv API and enriches it with citation/reference counts
from the Semantic Scholar API, one CSV per year, then merges them.

**2. Validation & cleaning** (`src/data_validation.py`, `preprocessing.py`) —
quality report on the raw data, then de-duplication, encoding repair (ftfy),
whitespace normalization, and category/abstract-length filtering.

**3. Annotation** (`src/annotation.py`, `keyword_extraction_v2.py`) — *automated*
labelling: each category is mapped to a human-readable domain label, a keyword
dictionary tags sub-topics, and KeyBERT extracts up to five keyphrases per paper
from its title + abstract. (No manual labelling / Label Studio is used; the tags
drive filtering, analytics, and proxy-relevance only.)

**4. Embeddings** (`src/embeddings_specter2.py`, `embeddings_scibert.py`) —
generates the two 768-dim embedding sets on a GPU and saves them as `.npy`
matrices aligned to arXiv IDs.

**5. Recommendation** (`src/recommender.py`) — pre-normalizes embeddings so
cosine similarity is a dot product, applies the validated category boost in
title mode, and supports category/year filtering. Category, tags, and citation
count are returned as display metadata only — they don't affect ranking.

**6. Evaluation & experiments** (`src/evaluation.py`, `experiments*.py`) — see
below.

For deeper detail see `docs/architecture.md` and `docs/methodology.md`.

---

## Results

**Known-item retrieval** is the primary evaluation: each query in `Queries.xlsx`
was generated *from* a specific source paper, so there is exactly one relevant
document, and the metric is the rank at which the source paper is retrieved from
the full ~50k corpus. Four query phrasings (keyword / task / problem / natural)
of 479 queries each were tested. With one relevant document per query, Recall@K
equals Hit@K.

Averaged across the four query types:

| Model | MRR | Recall@5 | Recall@10 | Recall@20 | NDCG@10 |
|-------|----:|---------:|----------:|----------:|--------:|
| **SPECTER2** | **0.329** | 0.422 | **0.493** | 0.577 | 0.360 |
| SciBERT | 0.022 | 0.032 | 0.049 | 0.067 | 0.026 |

SPECTER2 outperforms SciBERT by roughly an order of magnitude on this task,
which is expected: SPECTER2 is trained on a citation objective for document-level
similarity, whereas SciBERT is a general scientific language model. SPECTER2 is
strongest on natural-language queries (Recall@10 ≈ 0.70, MRR ≈ 0.53).

(Precision@K is not highlighted because, with a single relevant document, it is
capped at 1/K and isn't an informative comparison metric here.)

The dashboard's **Evaluation** tab reproduces these numbers from the result files
and also shows the proxy experiment (temporal split: corpus ≤ 2025, queries from
2026; proxy relevance = same category + a shared KeyBERT tag) and a SPECTER2 +
SciBERT late-fusion sweep.

---

## Project structure

```
paper-recommender-system/
├── app/
│   ├── app.py                 # Streamlit dashboard (entry point)
│   ├── analytics.py           # Analytics-tab data logic (no Streamlit dep)
│   └── evaluation.py          # Evaluation-tab data loaders
├── src/
│   ├── data_collection.py     # arXiv + Semantic Scholar -> data/raw/papers_YYYY.csv
│   ├── merge_raw_files.py     # merge yearly files -> papers_raw.csv
│   ├── data_validation.py     # raw-data quality report
│   ├── preprocessing.py       # clean / dedupe / normalize -> papers_clean.csv
│   ├── annotation.py          # domain labels + keyword-dictionary tags
│   ├── keyword_extraction_v2.py   # KeyBERT keyphrases (title + abstract)
│   ├── embeddings_specter2.py # SPECTER2 embeddings (.npy)
│   ├── embeddings_scibert.py  # SciBERT embeddings (.npy)
│   ├── compute_projections.py # PCA / t-SNE + silhouette for the Analytics tab
│   ├── recommender.py         # core recommender (imported by the app)
│   ├── evaluation.py          # known-item retrieval evaluation
│   ├── experiments.py         # model × distance × k proxy grid
│   ├── experiments_with_alpha.py        # category-boost (alpha) sweep
│   ├── experiments_per_category.py      # per-category breakdown
│   ├── experiments_category_restricted.py  # within-category retrieval
│   └── experiments_hybrid.py  # SPECTER2 + SciBERT late-fusion
├── docs/                      # architecture.md, methodology.md, deployment.md
├── reports/                   # data-validation / preprocessing / annotation reports
├── notebooks/                 # exploratory_analysis.ipynb
├── Queries.xlsx               # evaluation queries (known-item ground truth)
├── Dockerfile, docker-compose.yml
├── *.yaml                     # Kubernetes manifests for the GPU cluster
├── requirements.txt
└── README.md

# generated locally, not in git: data/  models/  results/  wandb/
```

---

## Setup

Requires **Python 3.13**. A CUDA GPU is needed to (re)generate embeddings; the
dashboard itself runs on CPU once the embeddings exist.

```bash
git clone https://github.com/teejay222/paper-recommender-system.git
cd paper-recommender-system

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root for the Semantic Scholar API key (used
only by the data-collection step):

```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

---

## Usage

### Run the dashboard

```bash
streamlit run app/app.py        # run from the project root; serves on :8501
```

Or with Docker:

```bash
docker compose up --build       # serves on :8501
```

The app expects `data/processed/papers_keybert_final.csv` and the embedding
matrices under `models/embeddings/` to be present (regenerate them with the
pipeline below).

### Regenerate the data pipeline

Run from the project root, in order. Steps 1–4 build the corpus; embeddings
(step 5) require a GPU.

```bash
python src/data_collection.py        # 1. collect (set TARGET_YEAR per run)
python src/merge_raw_files.py        #    merge yearly CSVs
python src/data_validation.py        # 2. raw-data quality report
python src/preprocessing.py          # 3. clean -> papers_clean.csv
python src/annotation.py             # 4. automated annotation
python src/keyword_extraction_v2.py  #    KeyBERT tags -> papers_keybert_v2.csv
python src/embeddings_specter2.py    # 5. SPECTER2 embeddings  (GPU)
python src/embeddings_scibert.py     #    SciBERT embeddings   (GPU)
python src/compute_projections.py    # 6. projections for Analytics
python src/evaluation.py             # 7. known-item evaluation
python src/experiments.py            #    proxy experiments (logs to W&B)
```

The app reads `data/processed/papers_keybert_final.csv`, the finalized corpus
derived from `papers_keybert_v2.csv`.

---

## Tech stack

Python 3.13 · PyTorch · Hugging Face Transformers + Adapters · SPECTER2 · SciBERT
· KeyBERT · scikit-learn · pandas / NumPy · Streamlit · Plotly / Altair ·
Weights & Biases · Docker / Kubernetes.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
