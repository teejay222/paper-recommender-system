#!/usr/bin/env python3
"""Run the main proxy retrieval experiments."""

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Config

DATA_FILE    = Path("data/processed/papers_keybert_final.csv")
RESULTS_DIR  = Path("results/evaluation_results")
RESULTS_CSV  = RESULTS_DIR / "experiments.csv"
RESULTS_JSON = RESULTS_DIR / "experiments_full.json"

EMBEDDINGS_DIR = Path("models/embeddings")

MODEL_FILES = {
    "specter2": {
        "embeddings": EMBEDDINGS_DIR / "specter2_embeddings.npy",
        "arxiv_ids":  EMBEDDINGS_DIR / "specter2_arxiv_ids.npy",
    },
    "scibert": {
        "embeddings": EMBEDDINGS_DIR / "scibert_embeddings.npy",
        "arxiv_ids":  EMBEDDINGS_DIR / "scibert_arxiv_ids.npy",
    },
}

TOP_K_VALUES     = [5, 10, 20]
DISTANCE_METRICS = ["cosine", "euclidean"]
MODELS           = ["specter2", "scibert"]   # set to ["specter2"] if SciBERT missing

CORPUS_MAX_YEAR = 2025
QUERY_YEAR      = 2026

WANDB_PROJECT = "paper-recommender-system"
MAX_QUERIES_EVAL = None          # set to 500 for quick test
BATCH_SIZE       = 500           # for similarity matrix (memory control)


# Relevance definition (proxy)

def parse_tags(tag_string: str) -> set:
    if not isinstance(tag_string, str):
        return set()
    if tag_string.strip().lower() in ("", "nan", "none"):
        return set()
    return {t.strip() for t in tag_string.split(";") if t.strip()}


def is_relevant(query_row: pd.Series, candidate_row: pd.Series) -> bool:
    """Use category and shared tags as proxy relevance."""
    if query_row["category"] != candidate_row["category"]:
        return False
    q_tags = parse_tags(str(query_row.get("keybert_tags_v2", "")))
    c_tags = parse_tags(str(candidate_row.get("keybert_tags_v2", "")))
    return bool(q_tags and c_tags and (q_tags & c_tags))


# Similarity computation (vectorised, batch-wise)

def compute_similarities_batch(
    query_embeddings: np.ndarray,   # (Q, D)
    corpus_embeddings: np.ndarray,  # (N, D)
    metric: str,
) -> np.ndarray:
    """Returns similarity matrix (Q, N)."""
    if metric == "cosine":
        return query_embeddings @ corpus_embeddings.T
    elif metric == "euclidean":
        q_norm_sq = np.sum(query_embeddings ** 2, axis=1, keepdims=True)
        c_norm_sq = np.sum(corpus_embeddings ** 2, axis=1)
        dots = query_embeddings @ corpus_embeddings.T
        dist_sq = q_norm_sq + c_norm_sq - 2 * dots
        dist_sq = np.maximum(dist_sq, 0.0)
        distances = np.sqrt(dist_sq)
        return 1.0 / (1.0 + distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# Evaluation metrics

def precision_at_k(relevant_flags: list[bool], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(relevant_flags[:k]) / k


def recall_at_k(relevant_flags: list[bool], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(relevant_flags[:k]) / total_relevant


def ndcg_at_k(relevant_flags: list[bool], total_relevant: int, k: int) -> float:
    top_k = relevant_flags[:k]
    dcg = sum(1.0 / math.log2(i+2) for i, rel in enumerate(top_k) if rel)
    ideal_rel_count = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(i+2) for i in range(ideal_rel_count))
    return dcg / idcg if idcg > 0 else 0.0


# Evaluate one similarity matrix

def evaluate_all_k(
    sim_matrix: np.ndarray,           # (Q, N) similarity scores
    query_ids: list,
    corpus_ids: list,
    relevance_lookup: dict,
    k_values: list,
) -> dict:
    """Returns for each k: {'precision','recall','ndcg'}."""
    num_queries = sim_matrix.shape[0]
    max_k = max(k_values)
    # Ask for one extra row so self-matches can be skipped.
    buffer_k = min(max_k + 1, sim_matrix.shape[1])
    top_indices = np.argpartition(sim_matrix, -buffer_k, axis=1)[:, -buffer_k:]
    # Sort the buffered candidates.
    row_sorted = np.argsort(-sim_matrix[np.arange(num_queries)[:, None], top_indices], axis=1)
    top_indices_sorted = top_indices[np.arange(num_queries)[:, None], row_sorted]

    results = {k: {"precisions": [], "recalls": [], "ndcgs": []} for k in k_values}

    for q_idx, q_arxiv in enumerate(query_ids):
        total_relevant = len(relevance_lookup.get(q_arxiv, set()))
        # Skip self-matches while collecting candidates.
        cand_arxivs = []
        for idx in top_indices_sorted[q_idx]:
            cand_arxiv = corpus_ids[idx]
            if cand_arxiv == q_arxiv:
                continue
            cand_arxivs.append(cand_arxiv)
            if len(cand_arxivs) == max_k:
                break

        # Score each K.
        for k in k_values:
            relevant_flags = [
                cand in relevance_lookup.get(q_arxiv, set())
                for cand in cand_arxivs[:k]
            ]
            results[k]["precisions"].append(precision_at_k(relevant_flags, k))
            results[k]["recalls"].append(recall_at_k(relevant_flags, total_relevant, k))
            results[k]["ndcgs"].append(ndcg_at_k(relevant_flags, total_relevant, k))

    aggregated = {}
    for k in k_values:
        aggregated[k] = {
            "precision": float(np.mean(results[k]["precisions"])),
            "recall":    float(np.mean(results[k]["recalls"])),
            "ndcg":      float(np.mean(results[k]["ndcgs"])),
        }
    return aggregated


# Main

def run_experiments():
    logger.info("=" * 60)
    logger.info("Hyperparameter Experiments (Temporal Split + Proxy Relevance)")
    logger.info(f"Corpus: years ≤ {CORPUS_MAX_YEAR} | Queries: year = {QUERY_YEAR}")
    logger.info(f"Grid: {len(MODELS)} models x {len(DISTANCE_METRICS)} metrics x {len(TOP_K_VALUES)} k = "
                f"{len(MODELS)*len(DISTANCE_METRICS)*len(TOP_K_VALUES)} runs")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed_runs = set()
    existing_results = []
    if RESULTS_CSV.exists():
        existing_df = pd.read_csv(RESULTS_CSV)
        existing_results = existing_df.to_dict("records")
        for row in existing_results:
            completed_runs.add((row["model"], row["metric"], int(row["k"])))
        logger.info(f"Found {len(completed_runs)} already completed runs – will skip")

    # Load dataset
    df = pd.read_csv(DATA_FILE, dtype=str)
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    logger.info(f"Total papers: {len(df)}")

    corpus_df = df[df["publication_year"] <= CORPUS_MAX_YEAR].copy()
    query_df  = df[df["publication_year"] == QUERY_YEAR].copy()
    logger.info(f"Corpus (≤{CORPUS_MAX_YEAR}): {len(corpus_df)}")
    logger.info(f"Query ({QUERY_YEAR}): {len(query_df)}")
    if len(query_df) == 0:
        logger.error(f"No query papers for year {QUERY_YEAR}. Check data.")
        return

    # Build relevance lookup
    logger.info("Precomputing relevance lookup...")
    corpus_by_category = defaultdict(list)
    for _, row in corpus_df.iterrows():
        corpus_by_category[row["category"]].append(row)

    relevance_lookup = {}
    for _, q_row in query_df.iterrows():
        q_arxiv = str(q_row["arxiv_id"])
        q_cat = q_row["category"]
        q_tags = parse_tags(str(q_row.get("keybert_tags_v2", "")))
        if not q_tags:
            relevance_lookup[q_arxiv] = set()
            continue
        relevant_set = set()
        for c_row in corpus_by_category.get(q_cat, []):
            c_arxiv = str(c_row["arxiv_id"])
            if c_arxiv == q_arxiv:
                continue
            c_tags = parse_tags(str(c_row.get("keybert_tags_v2", "")))
            if c_tags and (q_tags & c_tags):
                relevant_set.add(c_arxiv)
        relevance_lookup[q_arxiv] = relevant_set
    logger.info("Relevance lookup built.")

    # Run each model
    all_results = list(existing_results)

    for model_name in MODELS:
        logger.info(f"--- Loading embeddings: {model_name} ---")
        paths = MODEL_FILES[model_name]
        if not paths["embeddings"].exists():
            logger.error(f"Embeddings not found: {paths['embeddings']}. Skipping {model_name}.")
            continue

        embeddings = np.load(paths["embeddings"]).astype(np.float32)
        arxiv_ids  = np.load(paths["arxiv_ids"])
        emb_id_to_idx = {str(aid): i for i, aid in enumerate(arxiv_ids)}

        # Get indices
        corpus_indices = []
        for arxiv_id in corpus_df["arxiv_id"].astype(str):
            idx = emb_id_to_idx.get(arxiv_id)
            if idx is not None:
                corpus_indices.append(idx)

        query_indices = []
        query_arxiv_ids = []
        for arxiv_id in query_df["arxiv_id"].astype(str):
            idx = emb_id_to_idx.get(arxiv_id)
            if idx is not None:
                query_indices.append(idx)
                query_arxiv_ids.append(arxiv_id)

        logger.info(f"  Corpus in embeddings: {len(corpus_indices)} / {len(corpus_df)}")
        logger.info(f"  Query in embeddings: {len(query_indices)} / {len(query_df)}")
        if len(query_indices) == 0:
            continue

        # Load embeddings
        corpus_emb_raw = embeddings[corpus_indices]
        query_emb_raw = embeddings[query_indices]

        if MAX_QUERIES_EVAL is not None and MAX_QUERIES_EVAL < len(query_arxiv_ids):
            query_emb_raw = query_emb_raw[:MAX_QUERIES_EVAL]
            query_arxiv_ids = query_arxiv_ids[:MAX_QUERIES_EVAL]

        # Pre‑normalise for cosine
        corpus_norms = np.linalg.norm(corpus_emb_raw, axis=1, keepdims=True)
        corpus_norms = np.clip(corpus_norms, 1e-10, None)
        corpus_emb_cosine = corpus_emb_raw / corpus_norms

        query_norms = np.linalg.norm(query_emb_raw, axis=1, keepdims=True)
        query_norms = np.clip(query_norms, 1e-10, None)
        query_emb_cosine = query_emb_raw / query_norms

        corpus_arxiv_ids = [arxiv_ids[i] for i in corpus_indices]

        for metric in DISTANCE_METRICS:
            # Skip metric blocks that are already done.
            if all((model_name, metric, k) in completed_runs for k in TOP_K_VALUES):
                logger.info(f"Skipping all runs for {model_name} | {metric} (already done)")
                continue

            logger.info(f"Computing similarity matrix for {model_name} / {metric} ...")
            t_sim = time.time()

            sim_matrix = np.zeros((len(query_arxiv_ids), len(corpus_arxiv_ids)), dtype=np.float32)
            if metric == "cosine":
                for start in range(0, len(query_arxiv_ids), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(query_arxiv_ids))
                    batch = query_emb_cosine[start:end]
                    sim_matrix[start:end] = compute_similarities_batch(batch, corpus_emb_cosine, "cosine")
            else:  # euclidean
                for start in range(0, len(query_arxiv_ids), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(query_arxiv_ids))
                    batch = query_emb_raw[start:end]
                    sim_matrix[start:end] = compute_similarities_batch(batch, corpus_emb_raw, "euclidean")

            logger.info(f"Similarity matrix computed in {time.time()-t_sim:.1f}s")

            results_per_k = evaluate_all_k(
                sim_matrix,
                query_arxiv_ids,
                corpus_arxiv_ids,
                relevance_lookup,
                TOP_K_VALUES,
            )

            # Log each K
            for k in TOP_K_VALUES:
                run_key = (model_name, metric, k)
                if run_key in completed_runs:
                    continue

                result = {
                    "model":           model_name,
                    "metric":          metric,
                    "k":               k,
                    "precision_at_k":  round(results_per_k[k]["precision"], 4),
                    "recall_at_k":     round(results_per_k[k]["recall"], 4),
                    "ndcg_at_k":       round(results_per_k[k]["ndcg"], 4),
                    "runtime_seconds": 0,
                }

                # W&B log
                wandb_run = wandb.init(
                    project=WANDB_PROJECT,
                    name=f"{model_name}_{metric}_k{k}",
                    config={
                        "model": model_name,
                        "distance_metric": metric,
                        "top_k": k,
                        "corpus_size": len(corpus_indices),
                        "query_size": len(query_arxiv_ids),
                        "relevance": "category + keybert_tag",
                        "corpus_years": f"≤{CORPUS_MAX_YEAR}",
                        "query_year": QUERY_YEAR,
                    },
                    reinit=True,
                )
                wandb.log({
                    f"precision_at_{k}": result["precision_at_k"],
                    f"recall_at_{k}":    result["recall_at_k"],
                    f"ndcg_at_{k}":      result["ndcg_at_k"],
                })
                wandb_run.finish()

                all_results.append(result)
                completed_runs.add(run_key)

                # Save after each K
                pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
                with open(RESULTS_JSON, "w") as f:
                    json.dump(all_results, f, indent=2)

    # Summary
    final_df = pd.DataFrame(all_results)
    if len(final_df) == 0:
        logger.error("No results generated.")
        return

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 60)
    final_df_sorted = final_df.sort_values("ndcg_at_k", ascending=False)
    print("\n" + final_df_sorted[[
        "model", "metric", "k", "precision_at_k", "recall_at_k", "ndcg_at_k"
    ]].to_string(index=False))

    best = final_df_sorted.iloc[0]
    logger.info(
        "\nBest configuration: model=%s | metric=%s | k=%d | NDCG@K=%.4f",
        best["model"], best["metric"], int(best["k"]), best["ndcg_at_k"]
    )
    logger.info(f"\nResults saved to:\n  {RESULTS_CSV}\n  {RESULTS_JSON}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_experiments()