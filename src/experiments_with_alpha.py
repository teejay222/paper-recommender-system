"""Sweep the category-boost weight."""

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
RESULTS_CSV  = RESULTS_DIR / "experiments_smalleralpha_sweep.csv"
RESULTS_JSON = RESULTS_DIR / "experiments_smalleralpha_sweep.json"

EMBEDDINGS_FILE = Path("models/embeddings/specter2_embeddings.npy")
ARXIV_IDS_FILE  = Path("models/embeddings/specter2_arxiv_ids.npy")

# Fine-grained alpha values.
ALPHA_VALUES    = [0.00, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
TOP_K_VALUES    = [5, 10, 20]
CORPUS_MAX_YEAR = 2025
QUERY_YEAR      = 2026
WANDB_PROJECT   = "paper-recommender-system"
CATEGORIES      = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]

CANDIDATE_BUFFER = max(TOP_K_VALUES) + 50


# Helpers
def parse_tags(tag_string: str) -> set:
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
    top_k = relevant_flags[:k]
    dcg   = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(top_k) if rel)
    ideal = min(total_relevant, k)
    idcg  = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# Main loop
def run_alpha_sweep() -> None:
    logger.info("=" * 60)
    logger.info("Alpha Sweep: Multiplicative Category Boost (Optimized)")
    logger.info("Alpha values : %s", ALPHA_VALUES)
    logger.info("Top-K values : %s", TOP_K_VALUES)
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from saved rows
    completed_alphas = set()
    existing_results = []
    
    # Expected rows per alpha = (4 categories + 1 global summary row) * 3 K-values = 15 rows
    EXPECTED_ROWS_PER_ALPHA = (len(CATEGORIES) + 1) * len(TOP_K_VALUES)

    if RESULTS_CSV.exists():
        existing_df = pd.read_csv(RESULTS_CSV)
        existing_results = existing_df.to_dict("records")
        
        alpha_counts = existing_df["alpha"].value_counts().to_dict()
        
        for a, count in alpha_counts.items():
            if count == EXPECTED_ROWS_PER_ALPHA:
                completed_alphas.add(float(a))
            else:
                logger.warning(
                    f"Alpha {a} has corrupted or partial tracking ({count}/{EXPECTED_ROWS_PER_ALPHA} rows). "
                    f"It will be forced to recalculate."
                )
                
        # Clean up in-memory array to discard split/partial run elements
        existing_results = [r for r in existing_results if float(r["alpha"]) in completed_alphas]
        logger.info("Verified completed alphas to safely skip: %s", completed_alphas)

    # Load datasets
    logger.info("Loading dataset: %s", DATA_FILE)
    df = pd.read_csv(DATA_FILE, dtype=str)
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")

    corpus_df = df[df["publication_year"] <= CORPUS_MAX_YEAR].copy().reset_index(drop=True)
    query_df  = df[df["publication_year"] == QUERY_YEAR].copy().reset_index(drop=True)

    # Load and Map Embeddings
    logger.info("Loading SPECTER2 embeddings...")
    embeddings    = np.load(EMBEDDINGS_FILE).astype(np.float32)
    arxiv_ids     = np.load(ARXIV_IDS_FILE)
    emb_id_to_idx = {str(aid): i for i, aid in enumerate(arxiv_ids)}

    # Secure & Aligned Corpus Extraction
    logger.info("Building verified corpus records mapping...")
    corpus_records = []
    for _, row in corpus_df.iterrows():
        arxiv_id = str(row["arxiv_id"])
        idx = emb_id_to_idx.get(arxiv_id)
        if idx is not None:
            corpus_records.append({
                "arxiv_id": arxiv_id,
                "category": str(row["category"]),
                "tags_str": str(row.get("keybert_tags_v2", "")),
                "embedding": embeddings[idx]
            })

    corpus_ids = [r["arxiv_id"] for r in corpus_records]
    corpus_cats_arr = np.array([r["category"] for r in corpus_records])
    
    corpus_emb_matrix = np.vstack([r["embedding"] for r in corpus_records]).astype(np.float32)
    norms = np.linalg.norm(corpus_emb_matrix, axis=1, keepdims=True)
    corpus_emb_norm = corpus_emb_matrix / np.clip(norms, 1e-10, None)
    logger.info("Aligned Corpus Matrix Built: %s", corpus_emb_norm.shape)

    # Secure & Aligned Query Extraction
    logger.info("Building verified query records mapping...")
    query_records = []
    for _, q_row in query_df.iterrows():
        q_arxiv = str(q_row["arxiv_id"])
        idx = emb_id_to_idx.get(q_arxiv)
        if idx is not None:
            query_records.append({
                "arxiv_id": q_arxiv,
                "category": str(q_row["category"]),
                "tags_str": str(q_row.get("keybert_tags_v2", "")),
                "embedding": embeddings[idx]
            })

    q_ids = [r["arxiv_id"] for r in query_records]
    q_cats = [r["category"] for r in query_records]
    q_cats_arr = np.array(q_cats)
    
    q_emb_matrix = np.vstack([r["embedding"] for r in query_records]).astype(np.float32)
    q_norms = np.linalg.norm(q_emb_matrix, axis=1, keepdims=True)
    q_emb_norm = q_emb_matrix / np.clip(q_norms, 1e-10, None)
    logger.info("Aligned Query Matrix Built: %s", q_emb_norm.shape)

    # Precompute Base Cosine Similarity
    logger.info("Precomputing un-boosted baseline cosine matrix...")
    base_sim_matrix = q_emb_norm @ corpus_emb_norm.T

    # Precompute Relevance Lookup Ground-Truth
    logger.info("Precomputing relevance dictionaries...")
    corpus_by_cat = defaultdict(list)
    for r in corpus_records:
        corpus_by_cat[r["category"]].append((r["arxiv_id"], parse_tags(r["tags_str"])))

    relevance_lookup = {}
    for r in query_records:
        q_arxiv = r["arxiv_id"]
        q_cat = r["category"]
        q_tags = parse_tags(r["tags_str"])

        if not q_tags:
            relevance_lookup[q_arxiv] = set()
            continue

        relevance_lookup[q_arxiv] = {
            c_id for c_id, c_tags in corpus_by_cat.get(q_cat, [])
            if c_id != q_arxiv and c_tags and (q_tags & c_tags)
        }

    # Precompute Category Boolean Array Masks
    category_masks = {cat: (corpus_cats_arr == cat).astype(np.float32) for cat in CATEGORIES}

    # Alpha Sweep Loop
    all_results = list(existing_results)
    max_k = max(TOP_K_VALUES)

    for alpha in ALPHA_VALUES:
        if alpha in completed_alphas:
            logger.info("Skipping alpha=%.2f (already executed cleanly)", alpha)
            continue

        logger.info("Evaluating Alpha Expansion = %.2f", alpha)
        boosted_sim = base_sim_matrix.copy()

        # Fully Vectorized Matrix Boosting (Slices entire categories using C-speed broadcasting)
        if alpha > 0.0:
            for cat in CATEGORIES:
                query_mask = (q_cats_arr == cat)
                mask = category_masks.get(cat)
                if mask is not None and np.any(query_mask):
                    boosted_sim[query_mask] *= (1.0 + alpha * mask)

        # Setup metric aggregation storage by Category & K
        scores = {
            cat: {k: {"p": [], "r": [], "n": []} for k in TOP_K_VALUES}
            for cat in CATEGORIES
        }

        for q_pos, q_arxiv in enumerate(q_ids):
            q_cat = q_cats[q_pos]
            if q_cat not in scores:
                continue

            sims = boosted_sim[q_pos]
            relevant_set = relevance_lookup.get(q_arxiv, set())
            total_rel = len(relevant_set)

            buffer = min(CANDIDATE_BUFFER, len(corpus_ids))
            top_idx = np.argpartition(sims, -buffer)[-buffer:]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

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
                scores[q_cat][k]["p"].append(precision_at_k(flags, k))
                scores[q_cat][k]["r"].append(recall_at_k(flags, total_rel, k))
                scores[q_cat][k]["n"].append(ndcg_at_k(flags, total_rel, k))

        # Consolidation and Multi-Level Metric Logging
        alpha_rows = []
        wandb_payload = {"alpha": alpha}

        for k in TOP_K_VALUES:
            global_p, global_r, global_n = [], [], []

            # Process Per-Category Metrics
            for cat in CATEGORIES:
                cat_p = scores[cat][k]["p"]
                cat_r = scores[cat][k]["r"]
                cat_n = scores[cat][k]["n"]

                global_p.extend(cat_p)
                global_r.extend(cat_r)
                global_n.extend(cat_n)

                if cat_p:
                    cp_mean = round(float(np.mean(cat_p)), 4)
                    cr_mean = round(float(np.mean(cat_r)), 4)
                    cn_mean = round(float(np.mean(cat_n)), 4)

                    alpha_rows.append({
                        "experiment": "alpha_sweep",
                        "model": "specter2",
                        "metric": "cosine",
                        "retrieval": "full_corpus_with_category_boost",
                        "category": cat,
                        "alpha": alpha,
                        "k": k,
                        "precision_at_k": cp_mean,
                        "recall_at_k": cr_mean,
                        "ndcg_at_k": cn_mean,
                        "num_queries": len(cat_p),
                    })
                    wandb_payload[f"{cat}/precision_at_{k}"] = cp_mean
                    wandb_payload[f"{cat}/recall_at_{k}"] = cr_mean
                    wandb_payload[f"{cat}/ndcg_at_{k}"] = cn_mean

            # Process Global Aggregated Metrics
            if global_p:
                gp_mean = round(float(np.mean(global_p)), 4)
                gr_mean = round(float(np.mean(global_r)), 4)
                gn_mean = round(float(np.mean(global_n)), 4)

                alpha_rows.append({
                    "experiment": "alpha_sweep",
                    "model": "specter2",
                    "metric": "cosine",
                    "retrieval": "full_corpus_with_category_boost",
                    "category": "global",
                    "alpha": alpha,
                    "k": k,
                    "precision_at_k": gp_mean,
                    "recall_at_k": gr_mean,
                    "ndcg_at_k": gn_mean,
                    "num_queries": len(global_p),
                })
                wandb_payload[f"global/precision_at_{k}"] = gp_mean
                wandb_payload[f"global/recall_at_{k}"] = gr_mean
                wandb_payload[f"global/ndcg_at_{k}"] = gn_mean

        logger.info("  Global Metrics at K=20: P=%.4f, NDCG=%.4f", 
                    wandb_payload["global/precision_at_20"], wandb_payload["global/ndcg_at_20"])

        # Init explicit W&B Run for this structural alpha value
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"alpha_sweep_a{str(alpha).replace('.', '')}",
            config={
                "experiment": "alpha_sweep_diagnostic",
                "model": "specter2",
                "alpha": alpha,
                "query_year": QUERY_YEAR,
            },
            reinit=True,
        )
        wandb.log(wandb_payload)
        wandb_run.finish()

        all_results.extend(alpha_rows)
        completed_alphas.add(alpha)

        # Atomic updates to disk
        pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
        with open(RESULTS_JSON, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print Terminal Summaries
    final_df = pd.DataFrame(all_results)
    print("\n" + "=" * 75)
    print("GLOBAL PERFORMANCE TRENDS")
    print("=" * 75)
    for k in TOP_K_VALUES:
        print(f"\n--- Metrics at K = {k} ---")
        gl_k = final_df[(final_df["k"] == k) & (final_df["category"] == "global")].sort_values("alpha")
        print(gl_k[["alpha", "precision_at_k", "recall_at_k", "ndcg_at_k"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("DIAGNOSTIC SNAPSHOT: CATEGORY DIVERGENCE AT K=20")
    print("=" * 75)
    cat_20 = final_df[(final_df["k"] == 20) & (final_df["category"] != "global")].sort_values(["category", "alpha"])
    for cat in CATEGORIES:
        print(f"\nDomain: {cat}")
        print(cat_20[cat_20["category"] == cat][["alpha", "precision_at_k", "ndcg_at_k"]].to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    run_alpha_sweep()