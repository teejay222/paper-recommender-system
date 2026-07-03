"""Run known-item retrieval checks for both models."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# CONFIG — adjust paths here if yours differ

# Compare both models on the known-item eval. Use ["specter2"] alone if needed.
MODELS = ["specter2", "scibert"]

MODEL_EMBEDDINGS = {
    "specter2": {
        "emb": Path("models/embeddings/specter2_embeddings.npy"),
        "ids": Path("models/embeddings/specter2_arxiv_ids.npy"),
    },
    "scibert": {
        "emb": Path("models/embeddings/scibert_embeddings.npy"),
        "ids": Path("models/embeddings/scibert_arxiv_ids.npy"),
    },
}

QUERIES_FILE = Path("Queries.xlsx")          # <- set to your path

# Smoke test: one type first. Full run: all four.
QUERY_TYPES = ["keyword_query", "task_query", "problem_query", "natural_query"]

K_VALUES    = [5, 10, 20]     # Hit/Precision/NDCG cut-offs
BATCH_SIZE  = 32              # query-embedding batch size
RESULTS_DIR = Path("results/evaluation")

# Weights & Biases (its own project, separate from the proxy experiments)
# No API key in code — relies on `wandb login`. Set USE_WANDB = False to skip.
USE_WANDB     = True
WANDB_PROJECT = "paper-recommender-known-item"
WANDB_GROUP   = "known_item_eval"


# Query encoders

def load_query_encoder(model_name):
    """Load the query encoder for one model."""
    import torch

    if model_name == "specter2":
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        logger.info("Loading SPECTER2 (adhoc_query adapter)...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model     = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        model.load_adapter(
            "allenai/specter2_adhoc_query", source="hf",
            load_as="adhoc_query", set_active=True,
        )
        pooling = "cls"

    elif model_name == "scibert":
        from transformers import AutoModel, AutoTokenizer
        logger.info("Loading SciBERT (no adapter, mean pooling)...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model     = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        pooling = "mean"

    else:
        raise ValueError(f"Unknown model '{model_name}' (use 'specter2' or 'scibert')")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info("%s encoder ready on %s (pooling=%s)", model_name, device, pooling)
    return tokenizer, model, device, pooling


def embed_queries(texts, tokenizer, model, device, pooling, batch_size=BATCH_SIZE):
    """Embed query strings and normalize them."""
    import torch

    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state        # (B, T, 768)

        if pooling == "cls":
            emb = hidden[:, 0, :]                              # CLS token
        else:                                                 # mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        out.append(emb.cpu().numpy())
        logger.info("  embedded %d / %d queries", min(start + batch_size, len(texts)), len(texts))

    emb = np.vstack(out).astype(np.float32)
    norms = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10, None)
    return emb / norms


# Evaluation

def evaluate_query_type(model_name, query_type, df, corpus_norm, id_to_pos,
                        tokenizer, model, device, pooling):
    """Evaluate query type."""
    logger.info("-" * 70)
    logger.info("[%s] evaluating query type: %s", model_name, query_type)

    rows = []
    for _, row in df.iterrows():
        src_id = str(row["arxiv_id"])
        query  = row[query_type]
        if src_id in id_to_pos and isinstance(query, str) and query.strip():
            rows.append((src_id, query.strip()))

    missing = len(df) - len(rows)
    if missing:
        logger.warning("  %d/%d queries skipped (source not in corpus or empty query)",
                       missing, len(df))

    queries = [q for _, q in rows]
    src_ids = [s for s, _ in rows]
    query_emb = embed_queries(queries, tokenizer, model, device, pooling)

    ranks = []
    for i, src_id in enumerate(src_ids):
        cosine    = corpus_norm @ query_emb[i]
        src_score = cosine[id_to_pos[src_id]]
        ranks.append(int(np.sum(cosine > src_score)) + 1)     # 1 = best
    ranks = np.array(ranks)

    summary = {"model": model_name, "query_type": query_type, "n_queries": len(ranks)}
    for k in K_VALUES:
        in_topk = ranks <= k
        # Hit@K == Recall@K here (exactly one relevant paper per query).
        summary[f"hit@{k}"]       = round(float(np.mean(in_topk)), 4)
        # Precision@K capped at 1/K (one relevant doc) = Hit@K / K.
        summary[f"precision@{k}"] = round(float(np.mean(in_topk) / k), 4)
        # NDCG@K with one relevant item: ideal DCG = 1, so NDCG = 1/log2(rank+1).
        dcg = np.where(in_topk, 1.0 / np.log2(ranks + 1.0), 0.0)
        summary[f"ndcg@{k}"]      = round(float(np.mean(dcg)), 4)
    summary["mrr"]         = round(float(np.mean(1.0 / ranks)), 4)
    summary["median_rank"] = int(np.median(ranks))
    summary["mean_rank"]   = round(float(np.mean(ranks)), 1)

    per_query = pd.DataFrame({"arxiv_id": src_ids, "query": queries, "rank": ranks})
    return summary, per_query


def log_summary_to_wandb(summary):
    """Logs one (model, query type) as a clean run in the known-item project."""
    import wandb
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{summary['model']}_{summary['query_type']}",
        group=WANDB_GROUP,
        job_type="known_item_eval",
        reinit=True,
        config={
            "model":        summary["model"],
            "k_values":     K_VALUES,
            "query_type":   summary["query_type"],
            "n_queries":    summary["n_queries"],
            "ground_truth": "known-item (source paper = one relevant doc)",
        },
    )
    wandb.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
    run.finish()


def main():
    df = pd.read_excel(QUERIES_FILE, dtype=str)
    logger.info("Loaded %d query rows", len(df))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name in MODELS:
        paths = MODEL_EMBEDDINGS[model_name]
        logger.info("=" * 70)
        logger.info("MODEL: %s — loading corpus embeddings", model_name)
        corpus = np.load(paths["emb"]).astype(np.float32)
        arxiv_ids = np.array([str(a) for a in np.load(paths["ids"])])
        norms = np.clip(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-10, None)
        corpus_norm = corpus / norms
        id_to_pos = {aid: i for i, aid in enumerate(arxiv_ids)}
        logger.info("  corpus: %s (%d papers)", corpus.shape, len(arxiv_ids))

        tokenizer, model, device, pooling = load_query_encoder(model_name)

        for qtype in QUERY_TYPES:
            if qtype not in df.columns:
                logger.error("Column '%s' not in queries — skipping.", qtype)
                continue
            summary, per_query = evaluate_query_type(
                model_name, qtype, df, corpus_norm, id_to_pos,
                tokenizer, model, device, pooling,
            )
            summaries.append(summary)
            out = RESULTS_DIR / f"known_item_{model_name}_{qtype}.csv"
            per_query.to_csv(out, index=False)
            logger.info("  per-query ranks saved to %s", out)

            if USE_WANDB:
                try:
                    log_summary_to_wandb(summary)
                    logger.info("  logged %s/%s to W&B project '%s'",
                                model_name, qtype, WANDB_PROJECT)
                except Exception as e:
                    logger.warning("  W&B logging failed for %s/%s (continuing): %s",
                                   model_name, qtype, e)

        # free GPU memory between models
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    results = pd.DataFrame(summaries)
    print("\n" + "=" * 70)
    print("KNOWN-ITEM RETRIEVAL  —  SPECTER2 vs SciBERT")
    print("=" * 70)
    print(results.to_string(index=False))
    print("=" * 70)
    print("Hit@K = source paper in top K  (= Recall@K here: one relevant doc)")
    print("Precision@K is capped at 1/K (one relevant doc) = Hit@K / K")
    print("NDCG@K = rank-weighted hit (1/log2(rank+1))   |   MRR = mean 1/rank")
    results.to_csv(RESULTS_DIR / "known_item_summary.csv", index=False)
    logger.info("Summary saved to %s", RESULTS_DIR / "known_item_summary.csv")


if __name__ == "__main__":
    main()