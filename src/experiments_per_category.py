"""Break down proxy retrieval metrics by category."""

import json
import logging
import math
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
RESULTS_CSV  = RESULTS_DIR / "experiments_per_category.csv"
RESULTS_JSON = RESULTS_DIR / "experiments_per_category.json"

EMBEDDINGS_FILE = Path("models/embeddings/specter2_embeddings.npy")
ARXIV_IDS_FILE  = Path("models/embeddings/specter2_arxiv_ids.npy")

TOP_K_VALUES    = [5, 10, 20]
CORPUS_MAX_YEAR = 2025
QUERY_YEAR      = 2026
WANDB_PROJECT   = "paper-recommender-system"

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]

CATEGORY_LABELS = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Natural Language Processing",
    "cs.CV": "Computer Vision",
}


# Helpers

def parse_tags(tag_string: str) -> set:
    """Parse saved tag strings."""
    if not isinstance(tag_string, str):
        return set()
    if tag_string.strip().lower() in ("", "nan", "none"):
        return set()
    return {t.strip() for t in tag_string.split(";") if t.strip()}


def precision_at_k(relevant_flags: list, k: int) -> float:
    if k == 0:
        return 0.0
    return sum(relevant_flags[:k]) / k


def recall_at_k(relevant_flags: list, total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(relevant_flags[:k]) / total_relevant


def ndcg_at_k(relevant_flags: list, total_relevant: int, k: int) -> float:
    """NDCG@K with correct IDCG using min(total_relevant, k)."""
    top_k = relevant_flags[:k]
    dcg   = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(top_k) if rel)
    ideal_count = min(total_relevant, k)
    idcg  = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


# Main

def run_per_category_experiment() -> None:
    logger.info("=" * 60)
    logger.info("Per-Category Experiment")
    logger.info("Model: SPECTER2 | Metric: Cosine | K: %s", TOP_K_VALUES)
    logger.info("Retrieval: Full corpus | Queries: year=%d", QUERY_YEAR)
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed_keys  = set()
    existing_results = []

    if RESULTS_CSV.exists():
        existing_df = pd.read_csv(RESULTS_CSV)
        existing_results = existing_df.to_dict("records")
        for r in existing_results:
            completed_keys.add((r["category"], int(r["k"])))
        logger.info(
            "Found %d completed runs — will skip: %s",
            len(completed_keys), completed_keys
        )

    # Load dataset
    logger.info("Loading dataset: %s", DATA_FILE)
    df = pd.read_csv(DATA_FILE, dtype=str)
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    logger.info("Total papers: %d", len(df))

    corpus_df = df[df["publication_year"] <= CORPUS_MAX_YEAR].copy().reset_index(drop=True)
    query_df  = df[df["publication_year"] == QUERY_YEAR].copy().reset_index(drop=True)
    logger.info(
        "Corpus (<= %d): %d | Queries (%d): %d",
        CORPUS_MAX_YEAR, len(corpus_df), QUERY_YEAR, len(query_df)
    )

    if len(query_df) == 0:
        logger.error("No query papers found for year %d.", QUERY_YEAR)
        return

    # Load SPECTER2 embeddings
    logger.info("Loading SPECTER2 embeddings...")
    embeddings    = np.load(EMBEDDINGS_FILE).astype(np.float32)
    arxiv_ids     = np.load(ARXIV_IDS_FILE)
    emb_id_to_idx = {str(aid): i for i, aid in enumerate(arxiv_ids)}
    logger.info("Embeddings: %s", embeddings.shape)

    # Build full corpus embedding matrix (normalized for cosine)
    # Built once, reused for all categories
    logger.info("Building full corpus embedding matrix...")
    corpus_ids  = []
    corpus_embs = []

    for _, row in corpus_df.iterrows():
        arxiv_id = str(row["arxiv_id"])
        idx = emb_id_to_idx.get(arxiv_id)
        if idx is not None:
            corpus_ids.append(arxiv_id)
            corpus_embs.append(embeddings[idx])

    corpus_emb_matrix = np.vstack(corpus_embs).astype(np.float32)
    norms = np.linalg.norm(corpus_emb_matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    corpus_emb_norm = corpus_emb_matrix / norms

    logger.info("Full corpus matrix built: %s", corpus_emb_norm.shape)

    # Precompute corpus grouped by category ONCE
    # This avoids scanning all 41k corpus papers for every query
    # Each entry: (arxiv_id, parsed_tags_set)
    logger.info("Pre-grouping corpus by category and parsing tags...")
    corpus_by_cat = defaultdict(list)

    for c_id, c_row in zip(corpus_ids, [corpus_df.iloc[i] for i in range(len(corpus_df))]):
        c_cat  = str(c_row["category"])
        c_tags = parse_tags(str(c_row.get("keybert_tags_v2", "")))
        corpus_by_cat[c_cat].append((c_id, c_tags))

    for cat in CATEGORIES:
        logger.info("  %s: %d corpus papers", cat, len(corpus_by_cat[cat]))

    # Precompute ALL relevance lookups ONCE outside category loop
    # Instead of rebuilding per category, compute everything in one pass
    logger.info("Precomputing relevance lookup for all query papers...")
    relevance_lookup = {}

    # query_lookup dict replaces slow .loc scan
    query_lookup = {
        str(row["arxiv_id"]): row
        for _, row in query_df.iterrows()
    }

    for q_arxiv, q_row in query_lookup.items():
        q_cat  = str(q_row["category"])
        q_tags = parse_tags(str(q_row.get("keybert_tags_v2", "")))

        if not q_tags:
            relevance_lookup[q_arxiv] = set()
            continue

        # Only scan same-category corpus papers — not all 41k
        rel_set = {
            c_id
            for c_id, c_tags in corpus_by_cat.get(q_cat, [])
            if c_id != q_arxiv and c_tags and (q_tags & c_tags)
        }
        relevance_lookup[q_arxiv] = rel_set

    total_with_relevant = sum(1 for v in relevance_lookup.values() if v)
    logger.info(
        "Relevance lookup built. %d / %d queries have >= 1 relevant paper.",
        total_with_relevant, len(query_df)
    )

    # Evaluate categories
    all_results = list(existing_results)
    max_k       = max(TOP_K_VALUES)

    # buffer = max_k + 50 guarantees enough candidates after self-match removal
    CANDIDATE_BUFFER = max_k + 50

    for cat in CATEGORIES:
        cat_label    = CATEGORY_LABELS[cat]
        cat_query_df = query_df[query_df["category"] == cat]

        logger.info(
            "--- %s (%s) | %d queries ---",
            cat, cat_label, len(cat_query_df)
        )

        if len(cat_query_df) == 0:
            logger.warning("No query papers for %s — skipping", cat)
            continue

        # Build query embedding matrix for this category
        q_ids      = []
        q_embs     = []

        for _, q_row in cat_query_df.iterrows():
            q_arxiv = str(q_row["arxiv_id"])
            idx = emb_id_to_idx.get(q_arxiv)
            if idx is not None:
                q_ids.append(q_arxiv)
                q_embs.append(embeddings[idx])

        if not q_ids:
            logger.warning("No embeddings found for %s queries — skipping", cat)
            continue

        q_emb_matrix = np.vstack(q_embs).astype(np.float32)
        q_norms      = np.linalg.norm(q_emb_matrix, axis=1, keepdims=True)
        q_norms      = np.clip(q_norms, 1e-10, None)
        q_emb_norm   = q_emb_matrix / q_norms

        # Compute full corpus similarity matrix for this category's queries
        # Shape: (num_queries_in_category, num_corpus_papers)
        logger.info("  Computing similarity matrix...")
        sim_matrix = q_emb_norm @ corpus_emb_norm.T

        # Collect per-K scores for this category
        scores = {k: {"precisions": [], "recalls": [], "ndcgs": []} for k in TOP_K_VALUES}

        for q_pos, q_arxiv in enumerate(q_ids):
            sims         = sim_matrix[q_pos]
            relevant_set = relevance_lookup.get(q_arxiv, set())
            total_rel    = len(relevant_set)

            # Get top candidates with buffer to handle self-match removal
            buffer  = min(CANDIDATE_BUFFER, len(corpus_ids))
            top_idx = np.argpartition(sims, -buffer)[-buffer:]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

            # Build ranked list — skip self-match
            candidates = []
            for idx in top_idx:
                cand_id = corpus_ids[idx]
                if cand_id == q_arxiv:
                    continue
                candidates.append(cand_id)
                if len(candidates) == max_k:
                    break

            for k in TOP_K_VALUES:
                flags = [c in relevant_set for c in candidates[:k]]
                scores[k]["precisions"].append(precision_at_k(flags, k))
                scores[k]["recalls"].append(recall_at_k(flags, total_rel, k))
                scores[k]["ndcgs"].append(ndcg_at_k(flags, total_rel, k))

        # Use one W&B run per category and K.
        for k in TOP_K_VALUES:
            run_key = (cat, k)

            if run_key in completed_keys:
                logger.info("  Skipping %s k=%d (already done)", cat, k)
                continue

            p = scores[k]["precisions"]
            r = scores[k]["recalls"]
            n = scores[k]["ndcgs"]

            if not p:
                logger.warning("  No scores for %s k=%d", cat, k)
                continue

            result = {
                "experiment":     "per_category",
                "category":       cat,
                "category_label": cat_label,
                "model":          "specter2",
                "metric":         "cosine",
                "retrieval":      "full_corpus",
                "relevance":      "same_category_and_shared_tag",
                "k":              k,
                "precision_at_k": round(float(np.mean(p)), 4),
                "recall_at_k":    round(float(np.mean(r)), 4),
                "ndcg_at_k":      round(float(np.mean(n)), 4),
                "num_queries":    len(p),
            }

            # One W&B run per category+K
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                name=f"per_cat_{cat}_k{k}",
                config={
                    "experiment":    "per_category",
                    "category":      cat,
                    "category_label": cat_label,
                    "model":         "specter2",
                    "metric":        "cosine",
                    "top_k":         k,
                    "corpus_years":  f"<={CORPUS_MAX_YEAR}",
                    "query_year":    QUERY_YEAR,
                    "relevance":     "same_category_and_shared_tag",
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
            completed_keys.add(run_key)

            logger.info(
                "  %s k=%d | P=%.4f | R=%.4f | NDCG=%.4f | queries=%d",
                cat, k,
                result["precision_at_k"],
                result["recall_at_k"],
                result["ndcg_at_k"],
                result["num_queries"],
            )

            # Save after each result.
            pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
            with open(RESULTS_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

    # Final summary table
    final_df = pd.DataFrame(all_results)
    if final_df.empty:
        logger.warning("No results to display.")
        return

    print("\n" + "=" * 70)
    print("PER-CATEGORY RESULTS (SPECTER2 | Cosine | Full Corpus)")
    print("=" * 70)

    for k in TOP_K_VALUES:
        k_df = final_df[final_df["k"] == k].sort_values("ndcg_at_k", ascending=False)
        if k_df.empty:
            continue
        print(f"\nK = {k}")
        print("-" * 65)
        print(k_df[[
            "category", "category_label",
            "precision_at_k", "recall_at_k", "ndcg_at_k", "num_queries"
        ]].to_string(index=False))

    print("\n" + "=" * 70)
    logger.info("Results saved:")
    logger.info("  %s", RESULTS_CSV)
    logger.info("  %s", RESULTS_JSON)


if __name__ == "__main__":
    run_per_category_experiment()