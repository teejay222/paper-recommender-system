"""
src/compute_projections.py — precompute 2-D embedding projections for the
Analytics tab.

t-SNE over 50k x 768 vectors is far too slow to run inside Streamlit, so this
script computes the projections ONCE (run it where the embeddings live, i.e.
the cluster) and writes small CSVs the app can load instantly.

For each model it produces, on a stratified sample of the corpus:
  - PCA(2) coordinates           -> pca_x, pca_y
  - t-SNE(2) coordinates         -> tsne_x, tsne_y   (PCA(50) -> t-SNE, standard)
  - a silhouette separation score (how well the 4 categories separate in the
    full embedding space, cosine metric) -> results/figures/embedding_separation.json

Embeddings are L2-normalised first, so the geometry matches the cosine space the
recommender actually ranks in.

Run from the PROJECT ROOT:
    python src/compute_projections.py

Requires: numpy, pandas, scikit-learn.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# --- Config (edit paths here if your layout differs) -----------------------
DATA_FILE       = Path("data/processed/papers_keybert_final.csv")
EMBEDDINGS_DIR  = Path("models/embeddings")
OUTPUT_DIR      = Path("results/figures")
MODELS          = ["specter2", "scibert"]

SAMPLE_SIZE     = 6000     # stratified sample used for the scatter + silhouette
RANDOM_STATE    = 42
PCA_PRETSNE_DIM = 50       # reduce to this before t-SNE (standard practice)
SIM_PAIRS       = 40000    # random pairs sampled for the similarity histogram
SIM_PER_GROUP   = 5000     # cap per same/different group (keeps the two balanced)


def _embedding_paths(model: str) -> tuple[Path, Path]:
    return (
        EMBEDDINGS_DIR / f"{model}_embeddings.npy",
        EMBEDDINGS_DIR / f"{model}_arxiv_ids.npy",
    )


def _stratified_sample(ids: np.ndarray, categories: np.ndarray,
                       n: int, seed: int) -> np.ndarray:
    """Returns row indices for a category-stratified sample of size ~n."""
    rng = np.random.default_rng(seed)
    n = min(n, len(ids))
    frac = n / len(ids)
    chosen: list[int] = []
    for cat in np.unique(categories):
        cat_idx = np.where(categories == cat)[0]
        take = max(1, int(round(len(cat_idx) * frac)))
        take = min(take, len(cat_idx))
        chosen.extend(rng.choice(cat_idx, size=take, replace=False).tolist())
    return np.array(sorted(chosen))


def _similarity_pairs(emb_norm: np.ndarray, categories: np.ndarray, model: str,
                      n_pairs: int, per_group: int, seed: int) -> pd.DataFrame:
    """Cosine similarity for random same-category vs different-category pairs.

    emb_norm must already be L2-normalised, so a dot product IS the cosine.
    Returns a long-format frame: model, comparison, cosine.
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(emb_norm), n_pairs)
    j = rng.integers(0, len(emb_norm), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    cos = np.sum(emb_norm[i] * emb_norm[j], axis=1)
    same = categories[i] == categories[j]
    same_cos = cos[same][:per_group]
    diff_cos = cos[~same][:per_group]
    rows = (
        [{"model": model, "comparison": "Same category", "cosine": float(c)}
         for c in same_cos]
        + [{"model": model, "comparison": "Different category", "cosine": float(c)}
           for c in diff_cos]
    )
    return pd.DataFrame(rows)


def process_model(model: str, id_to_cat: dict, id_to_title: dict):
    emb_path, id_path = _embedding_paths(model)
    if not emb_path.exists() or not id_path.exists():
        logger.warning("Skipping %s — embeddings not found at %s", model, emb_path)
        return float("nan"), None

    logger.info("[%s] loading embeddings", model)
    embeddings = np.load(emb_path)
    arxiv_ids  = np.load(id_path, allow_pickle=True).astype(str)
    categories = np.array([id_to_cat.get(a, "unknown") for a in arxiv_ids])

    # Keep only rows whose category is known (defensive).
    known = categories != "unknown"
    embeddings, arxiv_ids, categories = embeddings[known], arxiv_ids[known], categories[known]

    sample_idx = _stratified_sample(arxiv_ids, categories, SAMPLE_SIZE, RANDOM_STATE)
    emb_s = normalize(embeddings[sample_idx])          # L2 -> cosine geometry
    ids_s = arxiv_ids[sample_idx]
    cat_s = categories[sample_idx]
    logger.info("[%s] sampled %d points", model, len(ids_s))

    logger.info("[%s] PCA(2)", model)
    pca_xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(emb_s)

    logger.info("[%s] PCA(%d) -> t-SNE(2)  (slow)", model, PCA_PRETSNE_DIM)
    pre = PCA(n_components=min(PCA_PRETSNE_DIM, emb_s.shape[1]),
              random_state=RANDOM_STATE).fit_transform(emb_s)
    tsne_xy = TSNE(n_components=2, init="pca", random_state=RANDOM_STATE,
                   perplexity=30).fit_transform(pre)

    logger.info("[%s] silhouette (cosine, by category)", model)
    sil = float(silhouette_score(emb_s, cat_s, metric="cosine"))
    logger.info("[%s] silhouette = %.4f", model, sil)

    logger.info("[%s] sampling same/different-category similarity pairs", model)
    sim_df = _similarity_pairs(emb_s, cat_s, model, SIM_PAIRS, SIM_PER_GROUP,
                               RANDOM_STATE)

    out = pd.DataFrame({
        "arxiv_id": ids_s,
        "category": cat_s,
        "title": [id_to_title.get(a, "") for a in ids_s],
        "pca_x": pca_xy[:, 0], "pca_y": pca_xy[:, 1],
        "tsne_x": tsne_xy[:, 0], "tsne_y": tsne_xy[:, 1],
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"projections_{model}.csv"
    out.to_csv(out_path, index=False)
    logger.info("[%s] wrote %s (%d rows)", model, out_path, len(out))
    return sil, sim_df


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Corpus not found at {DATA_FILE}")

    meta = pd.read_csv(DATA_FILE, usecols=["arxiv_id", "category", "title"],
                       dtype={"arxiv_id": str})
    id_to_cat   = dict(zip(meta["arxiv_id"], meta["category"].astype(str)))
    id_to_title = dict(zip(meta["arxiv_id"], meta["title"].astype(str)))

    separation = {}
    sim_frames = []
    for model in MODELS:
        sil, sim_df = process_model(model, id_to_cat, id_to_title)
        separation[model] = sil
        if sim_df is not None:
            sim_frames.append(sim_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sep_path = OUTPUT_DIR / "embedding_separation.json"
    with open(sep_path, "w") as f:
        json.dump(separation, f, indent=2)
    logger.info("Wrote %s -> %s", sep_path, separation)

    if sim_frames:
        sim_path = OUTPUT_DIR / "similarity_pairs.csv"
        pd.concat(sim_frames, ignore_index=True).to_csv(sim_path, index=False)
        logger.info("Wrote %s (%d rows)", sim_path,
                    sum(len(s) for s in sim_frames))


if __name__ == "__main__":
    main()