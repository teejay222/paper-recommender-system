"""
src/experiments_category_restricted.py
---------------------------------------
Option C: Category-Restricted Retrieval Experiment.

Difference from experiments.py (full corpus):
  - For each query paper, the candidate pool is restricted to papers
    from the SAME category only instead of the entire corpus.
  - This mirrors how researchers actually search — a CV researcher
    looks within CV papers, not across all of CS.
  - Expected result: higher Precision, Recall, NDCG because ~75% of
    obvious non-relevant papers (different category) are removed.

Setup:
  - Model  : SPECTER2 only (winner from full-corpus experiment)
  - Metric : Cosine only
  - Top-K  : 5, 10, 20
  - Split  : corpus = papers <= 2025, queries = papers from 2026
  - Relevance: same category + at least 1 shared keybert tag

Output:
  results/evaluation_results/experiments_category_restricted.csv
  results/evaluation_results/experiments_category_restricted.json

W&B: each run logged with prefix "restricted_"
"""

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE   = Path("data/processed/papers_keybert_final.csv")
RESULTS_DIR = Path("results/evaluation_results")
RESULTS_CSV  = RESULTS_DIR / "experiments_category_restricted.csv"
RESULTS_JSON = RESULTS_DIR / "experiments_category_restricted.json"

EMBEDDINGS_FILE = Path("models/embeddings/specter2_embeddings.npy")
ARXIV_IDS_FILE  = Path("models/embeddings/specter2_arxiv_ids.npy")

TOP_K_VALUES  = [5, 10, 20]
CORPUS_MAX_YEAR = 2025
QUERY_YEAR      = 2026
WANDB_PROJECT   = "paper-recommender-system"

# Categories from the spec
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]


# ---------------------------------------------------------------------------
# Relevance (same as experiments.py — only retrieval pool changes)
# ---------------------------------------------------------------------------

def parse_tags(tag_string: str) -> set:
    """Parses semicolon-separated tag string into a set."""
    if not isinstance(tag_string, str):
        return set()
    if tag_string.strip().lower() in ("", "nan", "none"):
        return set()
    return {t.strip() for t in tag_string.split(";") if t.strip()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(relevant_flags: list, k: int) -> float:
    """Precision@K = relevant in top-K / K"""
    if k == 0:
        return 0.0
    return sum(relevant_flags[:k]) / k


def recall_at_k(relevant_flags: list, total_relevant: int, k: int) -> float:
    """Recall@K = relevant in top-K / total relevant in corpus"""
    if total_relevant == 0:
        return 0.0
    return sum(relevant_flags[:k]) / total_relevant


def ndcg_at_k(relevant_flags: list, total_relevant: int, k: int) -> float:
    """
    NDCG@K with correct IDCG.
    IDCG uses min(total_relevant, k) — not just relevant found in top-K.
    """
    top_k = relevant_flags[:k]
    dcg   = sum(
        1.0 / math.log2(i + 2)
        for i, rel in enumerate(top_k) if rel
    )
    ideal_count = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_category_restricted_experiment() -> None:
    """
    Runs the category-restricted retrieval experiment.

    For each query paper:
      1. Restrict candidate pool to same-category corpus papers only
      2. Compute cosine similarity within that pool
      3. Rank and evaluate at K = 5, 10, 20

    This removes cross-category noise and tests how well the system
    works when the user is searching within their own field.
    """
    logger.info("=" * 60)
    logger.info("Category-Restricted Retrieval Experiment")
    logger.info("Model : SPECTER2 | Metric: Cosine | K: %s", TOP_K_VALUES)
    logger.info("Corpus: years <= %d | Queries: year = %d", CORPUS_MAX_YEAR, QUERY_YEAR)
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Resume support ---
    completed_k = set()
    existing_results = []

    if RESULTS_CSV.exists():
        existing_df = pd.read_csv(RESULTS_CSV)
        existing_results = existing_df.to_dict("records")
        completed_k = {int(r["k"]) for r in existing_results}
        logger.info("Found %d completed K values — will skip: %s", len(completed_k), completed_k)

    remaining_k = [k for k in TOP_K_VALUES if k not in completed_k]
    if not remaining_k:
        logger.info("All K values already completed. Nothing to run.")
        return

    # --- Load data ---
    logger.info("Loading dataset: %s", DATA_FILE)
    df = pd.read_csv(DATA_FILE, dtype=str)
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    logger.info("Total papers: %d", len(df))

    corpus_df = df[df["publication_year"] <= CORPUS_MAX_YEAR].copy().reset_index(drop=True)
    query_df  = df[df["publication_year"] == QUERY_YEAR].copy().reset_index(drop=True)
    logger.info("Corpus (<= %d): %d papers", CORPUS_MAX_YEAR, len(corpus_df))
    logger.info("Queries (%d) : %d papers", QUERY_YEAR, len(query_df))

    if len(query_df) == 0:
        logger.error("No query papers found for year %d.", QUERY_YEAR)
        return

    # --- Load embeddings ---
    logger.info("Loading SPECTER2 embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    arxiv_ids  = np.load(ARXIV_IDS_FILE)
    emb_id_to_idx = {str(aid): i for i, aid in enumerate(arxiv_ids)}
    logger.info("Embeddings loaded: %s", embeddings.shape)

    # --- Build category-level corpus index ---
    # For each category, store:
    #   - list of arxiv_ids in that category (corpus only)
    #   - their embeddings stacked as one matrix
    #   - their DataFrame rows for relevance checking
    logger.info("Building per-category corpus index...")

    category_corpus = {}   # category -> {"arxiv_ids", "embeddings", "rows"}

    for cat in CATEGORIES:
        cat_df = corpus_df[corpus_df["category"] == cat].copy()

        # Get embedding indices for this category's papers
        cat_ids  = []
        cat_embs = []
        cat_rows = []

        for _, row in cat_df.iterrows():
            arxiv_id = str(row["arxiv_id"])
            idx = emb_id_to_idx.get(arxiv_id)
            if idx is not None:
                cat_ids.append(arxiv_id)
                cat_embs.append(embeddings[idx])
                cat_rows.append(row)

        if not cat_ids:
            logger.warning("No embeddings found for category %s", cat)
            continue

        # Stack into matrix and pre-normalize for cosine
        cat_emb_matrix = np.vstack(cat_embs).astype(np.float32)
        norms = np.linalg.norm(cat_emb_matrix, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        cat_emb_norm = cat_emb_matrix / norms

        category_corpus[cat] = {
            "arxiv_ids":      cat_ids,
            "embeddings_norm": cat_emb_norm,
            "rows":           cat_rows,
        }

        logger.info("  %s: %d corpus papers indexed", cat, len(cat_ids))

    # --- Precompute relevance lookup ---
    # For each query paper: set of corpus arxiv_ids that are relevant
    # (same category + at least 1 shared tag)
    logger.info("Precomputing relevance lookup...")

    relevance_lookup = {}

    for _, q_row in query_df.iterrows():
        q_arxiv = str(q_row["arxiv_id"])
        q_cat   = str(q_row["category"])
        q_tags  = parse_tags(str(q_row.get("keybert_tags_v2", "")))

        if not q_tags or q_cat not in category_corpus:
            relevance_lookup[q_arxiv] = set()
            continue

        relevant_set = set()
        for c_row in category_corpus[q_cat]["rows"]:
            c_arxiv = str(c_row["arxiv_id"])
            if c_arxiv == q_arxiv:
                continue
            c_tags = parse_tags(str(c_row.get("keybert_tags_v2", "")))
            if c_tags and (q_tags & c_tags):
                relevant_set.add(c_arxiv)

        relevance_lookup[q_arxiv] = relevant_set

    total_with_relevant = sum(1 for v in relevance_lookup.values() if v)
    logger.info(
        "Relevance lookup built. %d / %d queries have at least 1 relevant paper.",
        total_with_relevant, len(query_df)
    )

    # --- Evaluate per category ---
    # Collect scores per K across all queries
    scores_per_k = {k: {"precisions": [], "recalls": [], "ndcgs": []} for k in remaining_k}

    for cat in CATEGORIES:
        if cat not in category_corpus:
            continue

        cat_corpus    = category_corpus[cat]
        corpus_ids    = cat_corpus["arxiv_ids"]
        corpus_norm   = cat_corpus["embeddings_norm"]

        # Get query papers from this category
        cat_query_df = query_df[query_df["category"] == cat]
        if len(cat_query_df) == 0:
            logger.info("  %s: no query papers", cat)
            continue

        logger.info("  %s: %d queries | %d corpus papers", cat, len(cat_query_df), len(corpus_ids))

        # Stack query embeddings for this category
        q_ids  = []
        q_embs = []

        for _, q_row in cat_query_df.iterrows():
            q_arxiv = str(q_row["arxiv_id"])
            idx = emb_id_to_idx.get(q_arxiv)
            if idx is not None:
                q_ids.append(q_arxiv)
                q_embs.append(embeddings[idx])

        if not q_ids:
            continue

        q_emb_matrix = np.vstack(q_embs).astype(np.float32)
        q_norms      = np.linalg.norm(q_emb_matrix, axis=1, keepdims=True)
        q_norms      = np.clip(q_norms, 1e-10, None)
        q_emb_norm   = q_emb_matrix / q_norms

        # Compute similarity matrix: (num_queries, num_corpus)
        sim_matrix = q_emb_norm @ corpus_norm.T   # cosine similarity

        max_k = max(remaining_k)

        for q_pos, q_arxiv in enumerate(q_ids):
            sims         = sim_matrix[q_pos]
            total_rel    = len(relevance_lookup.get(q_arxiv, set()))
            relevant_set = relevance_lookup.get(q_arxiv, set())

            # Get top max_k+1 indices (buffer for self-match)
            buffer = min(max_k + 1, len(corpus_ids))
            top_indices = np.argpartition(sims, -buffer)[-buffer:]
            top_indices = top_indices[np.argsort(-sims[top_indices])]

            # Build ranked candidate list (skip self)
            candidates = []
            for idx in top_indices:
                cand_id = corpus_ids[idx]
                if cand_id == q_arxiv:
                    continue
                candidates.append(cand_id)
                if len(candidates) == max_k:
                    break

            # Score at each K
            for k in remaining_k:
                flags = [c in relevant_set for c in candidates[:k]]
                scores_per_k[k]["precisions"].append(precision_at_k(flags, k))
                scores_per_k[k]["recalls"].append(recall_at_k(flags, total_rel, k))
                scores_per_k[k]["ndcgs"].append(ndcg_at_k(flags, total_rel, k))

    # --- Aggregate and save results ---
    all_results = list(existing_results)

    for k in remaining_k:
        p = scores_per_k[k]["precisions"]
        r = scores_per_k[k]["recalls"]
        n = scores_per_k[k]["ndcgs"]

        if not p:
            logger.warning("No scores collected for k=%d", k)
            continue

        result = {
            "experiment":     "category_restricted",
            "model":          "specter2",
            "metric":         "cosine",
            "retrieval":      "category_restricted",
            "relevance":      "same_category_and_shared_tag",
            "k":              k,
            "precision_at_k": round(float(np.mean(p)), 4),
            "recall_at_k":    round(float(np.mean(r)), 4),
            "ndcg_at_k":      round(float(np.mean(n)), 4),
            "num_queries":    len(p),
        }

        # W&B logging
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"restricted_specter2_cosine_k{k}",
            config={
                "experiment":    "category_restricted",
                "model":         "specter2",
                "metric":        "cosine",
                "top_k":         k,
                "retrieval":     "category_restricted",
                "corpus_years":  f"<={CORPUS_MAX_YEAR}",
                "query_year":    QUERY_YEAR,
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
        logger.info(
            "k=%d | P@K=%.4f | R@K=%.4f | NDCG@K=%.4f",
            k, result["precision_at_k"], result["recall_at_k"], result["ndcg_at_k"]
        )

        # Save after each K
        pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
        with open(RESULTS_JSON, "w") as f:
            json.dump(all_results, f, indent=2)

    # --- Print summary ---
    final_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("CATEGORY-RESTRICTED RESULTS")
    print("=" * 60)
    print(final_df[[
        "k", "precision_at_k", "recall_at_k", "ndcg_at_k", "num_queries"
    ]].to_string(index=False))
    print("=" * 60)
    logger.info("Saved -> %s", RESULTS_CSV)
    logger.info("Saved -> %s", RESULTS_JSON)


if __name__ == "__main__":
    run_category_restricted_experiment()