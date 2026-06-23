# Architecture

The system is organized as a modular recommendation pipeline.

## Components

- `data_collection.py`: fetches and normalizes paper metadata.
- `preprocessing.py`: cleans text and prepares model-ready fields.
- `annotation.py`: stores human relevance labels for evaluation.
- `embeddings.py`: generates dense vector representations.
- `recommender.py`: ranks papers by similarity.
- `evaluation.py`: computes recommendation metrics.
- `experiments.py`: coordinates repeatable experiments.
- `openreview_processing.py`: transforms OpenReview data into project records.
- `app/app.py`: provides the Streamlit interface.

## Data Flow

Raw data enters `data/raw/`, cleaned data is written to `data/processed/`, and labeled evaluation sets live in `data/evaluation/`. Model artifacts are saved in `models/`, while metrics and charts are exported to `results/`.
