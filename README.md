# Academic Paper Recommendation System

A recommendation system for arXiv computer science papers. You can search using a
paper you already know or simply describe what you're looking for, and the system
recommends the most similar papers. Behind it are two scientific embedding models,
SPECTER2 and SciBERT, and the whole thing is served through a Streamlit dashboard.

It runs end to end: pulling paper metadata from arXiv and Semantic Scholar,
cleaning and tagging it, generating embeddings, ranking by similarity, checking
how well it retrieves, and putting it all behind a web UI.

Note: The dataset, embedding files, and most generated results are too large to
include in the repository (see `.gitignore`). The pipeline below regenerates
everything. A CUDA GPU is required only for generating embeddings; once they are
created, the dashboard runs on CPU.

## Highlights

- Built a dataset of 49,992 arXiv papers enriched with Semantic Scholar metadata
- Compared SPECTER2 and SciBERT for scientific paper recommendation
- Implemented title-based and free-text retrieval
- Evaluated the system using known-item retrieval, temporal split experiments, and hybrid retrieval
- Interactive Streamlit dashboard for searching, browsing, analytics, and evaluation
- Docker and Kubernetes support

## What it does

Two ways to search:

- **By title** – type a paper title, it fuzzy-matches the closest one in the
  dataset and recommends similar work using that paper's embedding.
- **Free text** – type keywords or a plain question and it embeds that on the
  fly. For SPECTER2 this uses its `adhoc_query` adapter, which is built for
  exactly this kind of query.

Two models to compare:

- **SPECTER2** (`allenai/specter2_base` with the proximity adapter, CLS pooling).
  Used as the primary recommendation model.
- **SciBERT** (`allenai/scibert_scivocab_uncased`, mean pooling). Kept as a
  baseline to compare against.

Both turn a paper's `title` + `abstract` into a 768-dimensional vector.

Ranking is cosine similarity, with a small same-category boost in title mode:

```
final = cosine × (1 + α × same_category),  α = 0.03
```

I picked 0.03 from a sweep (see Results). Category, tags and citation count show
up in the results but they don't change the ranking.

The dashboard has five tabs:

- **Browse** – page through the whole dataset, filter and sort, no query needed.
- **Recommendations** – the actual title / free-text search.
- **Analytics** – dataset breakdown, citation spread, keyword trends, and 2-D
  maps of the embedding space.
- **Evaluation** – the retrieval numbers and the tuning experiments.
- **Project Info** – a plain-English summary of the whole thing.

## The dataset

After collection and preprocessing, the final dataset contains **49,992 papers**
across four arXiv categories, spread evenly over six years (2021 to 2026, roughly
8,333 papers a year).

| Category | Papers | Share |
|----------|-------:|------:|
| cs.AI – Artificial Intelligence | 22,876 | 45.8% |
| cs.LG – Machine Learning | 12,205 | 24.4% |
| cs.CV – Computer Vision | 7,840 | 15.7% |
| cs.CL – Natural Language Processing | 7,071 | 14.1% |

Cleaning kept 99.99% of what I collected (only 6 duplicate titles were dropped).
58.3% of papers have at least one citation and 79.7% have a reference count, both
pulled from Semantic Scholar. Full numbers are in
`reports/preprocessing_report.txt`.

## How the pipeline works

1. **Collect** (`src/data_collection.py`, `merge_raw_files.py`) – pull paper
   metadata from the arXiv API one year at a time, and enrich each paper with
   citation and reference counts from Semantic Scholar. Then merge the yearly
   CSVs into one file.
2. **Validate and clean** (`src/data_validation.py`, `preprocessing.py`) – a
   quality report on the raw pull, then de-duplication, encoding fixes (ftfy),
   whitespace cleanup, and dropping anything with the wrong category or a
   too-short abstract.
3. **Annotate** (`src/annotation.py`, `keyword_extraction_v2.py`) – each category
   gets a readable label, a keyword dictionary tags sub-topics, and KeyBERT pulls
   up to five key phrases per paper from its title and abstract. All automatic,
   no manual labelling. The tags feed the filtering, the analytics charts, and
   the proxy relevance used in the experiments.
4. **Embeddings** (`src/embeddings_specter2.py`, `embeddings_scibert.py`) –
   generate the two sets of 768-dim vectors on a GPU and save them as `.npy`
   files lined up with the arXiv IDs.
5. **Recommend** (`src/recommender.py`) – normalise the embeddings once so cosine
   is just a dot product, apply the category boost in title mode, and support
   filtering by category and year.
6. **Evaluate** (`src/evaluation.py`, `experiments*.py`) – known-item retrieval
   plus a few tuning experiments, tracked in Weights & Biases.

The app loads `data/processed/papers_keybert_final.csv` (the tagged dataset the
recommender reads) and the embedding files under `models/embeddings/`.

## Results

The main check is **known-item retrieval**. Every query in `Queries.xlsx` was
written from one specific source paper, so there's exactly one correct answer per
query, and the question is simply: how high up does that source paper come back
when you search the full ~50k dataset? I tested four phrasings of each query
(keyword, task, problem, natural), 479 queries per phrasing. With a single
relevant paper per query, Recall@K is the same as Hit@K.

Averaged over the four phrasings:

| Model | MRR | Recall@5 | Recall@10 | Recall@20 | NDCG@10 |
|-------|----:|---------:|----------:|----------:|--------:|
| **SPECTER2** | **0.329** | 0.422 | **0.493** | 0.577 | 0.360 |
| SciBERT | 0.022 | 0.032 | 0.049 | 0.067 | 0.026 |

SPECTER2 outperforms SciBERT by a wide margin, which is what you'd expect.
SPECTER2 is trained on citation signals for document-level similarity, while
SciBERT is a general scientific language model that wasn't built for this task.
SPECTER2 does best on natural-language queries (Recall@10 around 0.70, MRR around
0.53) and worst on short problem-style queries.

I left Precision@K out of the summary because with one relevant paper it maxes out
at 1/K, so it isn't a useful comparison metric here.

The **Evaluation** tab rebuilds these numbers from the saved result files and also
shows two extra things:

- A **proxy experiment** (dataset up to 2025, queries from 2026, a hit counted as
  same category plus a shared KeyBERT tag), used mainly for tuning.
- A **SPECTER2 + SciBERT blend**. It barely improves on SPECTER2 alone, so I kept
  the app on SPECTER2 only.

On the category boost: I swept α from 0 up to 0.5. Almost all the gain lands by
about 0.03 (global NDCG@10 on the proxy metric goes from 0.120 with no boost to
0.165, and it's basically flat past 0.03), so that's the value I used. That gain
is on the proxy metric, not the known-item numbers above.

## Project structure

```
paper-recommender-system/
├── app/
│   ├── app.py                 # Streamlit dashboard (run this)
│   ├── analytics.py           # data prep for the Analytics tab
│   └── evaluation.py          # loaders for the Evaluation tab
├── src/
│   ├── data_collection.py         # arXiv + Semantic Scholar -> data/raw/papers_YYYY.csv
│   ├── merge_raw_files.py         # merge the yearly files
│   ├── data_validation.py         # raw-data quality report
│   ├── preprocessing.py           # clean / dedupe / normalise -> papers_clean.csv
│   ├── annotation.py              # domain labels + dictionary tags
│   ├── keyword_extraction_v2.py   # KeyBERT key phrases
│   ├── embeddings_specter2.py     # SPECTER2 embeddings (.npy)
│   ├── embeddings_scibert.py      # SciBERT embeddings (.npy)
│   ├── compute_projections.py     # PCA / t-SNE + silhouette for the Analytics tab
│   ├── recommender.py             # the recommender the app imports
│   ├── evaluation.py              # known-item retrieval
│   └── experiments*.py            # proxy experiments, alpha sweep, per-category, hybrid
├── docs/                      # architecture / methodology / deployment notes
├── reports/                   # data-validation / preprocessing / annotation reports
├── notebooks/                 # exploratory_analysis.ipynb
├── Queries.xlsx               # evaluation queries (the known-item ground truth)
├── Dockerfile, docker-compose.yml
├── *.yaml                     # Kubernetes manifests for the GPU pod
├── requirements.txt
└── README.md

# generated locally, not in git: data/  models/  results/  wandb/
```

### Generated data files

As it runs, the pipeline writes a few files under `data/processed/`:

- `papers_clean.csv` – after cleaning
- `papers_annotated.csv` – after the dictionary-tag annotation
- `papers_keybert_v2.csv` – after KeyBERT tagging
- `papers_keybert_final.csv` – the tagged dataset the app and experiments load

Embeddings go to `models/embeddings/` (`specter2_embeddings.npy`,
`scibert_embeddings.npy`, and the matching arxiv_id files). Evaluation output
lands in `results/evaluation/`, projection and figure data in `results/figures/`,
and the Weights & Biases exports in `results/wandb_exports/`.

## Setup

Built and tested on **Python 3.12**. You need a CUDA GPU to regenerate the
embeddings; the dashboard runs on CPU once they exist.

```bash
git clone https://github.com/teejay222/paper-recommender-system.git
cd paper-recommender-system

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Data collection needs a Semantic Scholar API key. Put it in a `.env` file in the
project root:

```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

## Running it

Dashboard:

```bash
streamlit run app/app.py        # from the project root, serves on :8501
```

Docker:

```bash
docker compose up --build       # serves on :8501
```

The app expects `data/processed/papers_keybert_final.csv` and the embedding files
under `models/embeddings/`. If they aren't there yet, regenerate them below.

### Regenerating everything

Run from the project root, in order. Steps 1–4 build the dataset; embeddings
(step 5) need a GPU. I generated the embeddings on an A100 pod on a Kubernetes
cluster (the `*.yaml` manifests are for that).

```bash
python src/data_collection.py        # set TARGET_YEAR per run
python src/merge_raw_files.py
python src/data_validation.py
python src/preprocessing.py
python src/annotation.py
python src/keyword_extraction_v2.py
python src/embeddings_specter2.py    # GPU
python src/embeddings_scibert.py     # GPU
python src/compute_projections.py
python src/evaluation.py
python src/experiments.py            # logs to W&B
```

## Tech stack

Python • PyTorch • Hugging Face Transformers • Adapters • SPECTER2 • SciBERT • KeyBERT • scikit-learn • pandas • NumPy • Streamlit • Plotly • Altair • Docker • Kubernetes • Weights & Biases

## License

MIT. See [`LICENSE`](LICENSE).
