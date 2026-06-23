# Paper Recommender System

A research-paper recommendation project with data collection, preprocessing, annotation, embedding generation, recommendation, evaluation, experiment tracking, and a Streamlit demo app.

## Repository Structure

```text
paper-recommender-system/
├── data/                 # Raw, processed, and evaluation datasets
├── notebooks/            # Exploratory notebooks
├── src/                  # Backend logic and ML pipeline modules
├── app/                  # Streamlit application
├── models/               # Embeddings and similarity model artifacts
├── results/              # Evaluation outputs, figures, and W&B exports
├── docs/                 # Architecture, methodology, and deployment notes
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app/app.py
```

## Docker

```bash
docker compose up --build
```

## Suggested Workflow

1. Collect paper metadata in `data/raw/`.
2. Clean and normalize records into `data/processed/`.
3. Generate embeddings and store artifacts in `models/embeddings/`.
4. Build similarity indexes or models in `models/similarity_models/`.
5. Run experiments and save evaluation outputs in `results/`.
6. Serve recommendations through the Streamlit app in `app/`.
