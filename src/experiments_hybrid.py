"""
src/experiments_hybrid.py
-------------------------
Hybrid SPECTER2 + SciBERT known-item evaluation via LATE FUSION (score
combination), reusing the exact known-item harness in src/evaluation.py.

Why late fusion (not vector concat / averaging):
  The two models live in different spaces (SPECTER2 uses the adhoc_query/
  proximity adapters with CLS pooling; SciBERT uses plain mean pooling), so
  averaging their dimensions is not meaningful and concatenation mixes scales.
  Instead we combine the *similarity scores* per query:

      score(paper) = w * cos_specter2(query, paper)
                   + (1 - w) * cos_scibert(query, paper)

  and sweep w. At w = 1.0 this reproduces the SPECTER2 known-item numbers; at
  w = 0.0 it reproduces SciBERT. Any 0 < w < 1 is the "hybrid".

  Honest expectation: SciBERT is near-random on this task, so blending it in is
  likely to only hurt. A flat/decreasing curve as w drops below 1.0 is itself
  the result — it shows SPECTER2 alone is the right choice.

Reuses from evaluation.py (no duplicated encoder logic):
  load_query_encoder, embed_queries, K_VALUES, QUERIES_FILE, QUERY_TYPES,
  MODEL_EMBEDDINGS, RESULTS_DIR

Output:
  results/evaluation/known_item_hybrid_sweep.csv
    columns: weight_specter2, query_type, n_queries,
             hit@5/10/20, precision@5/10/20, ndcg@5/10/20, mrr,
             median_rank, mean_rank

Run from the PROJECT ROOT (so the relative model/data paths resolve):
    python src/experiments_hybrid.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import (
    load_query_encoder, embed_queries,
    K_VALUES, QUERIES_FILE, QUERY_TYPES, MODEL_EMBEDDINGS, RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Weight on SPECTER2; (1 - w) goes to SciBERT. 1.0 == SPECTER2, 0.0 == SciBERT.
WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]


def _load_norm(emb_path, ids_path):
    """Load an embedding matrix + its arxiv_ids, L2-normalised."""
    emb = np.load(emb_path).astype(np.float32)
    ids = np.array([str(a) for a in np.load(ids_path)])
    norms = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10, None)
    return emb / norms, ids


def load_aligned_corpora():
    """Load both corpora and align them to a shared arxiv_id order.

    Returns (corpus_s, corpus_b, id_to_pos) where row i of each matrix is the
    same paper, and id_to_pos maps arxiv_id -> that shared row index.
    """
    s_emb, s_ids = _load_norm(MODEL_EMBEDDINGS["specter2"]["emb"],
                              MODEL_EMBEDDINGS["specter2"]["ids"])
    b_emb, b_ids = _load_norm(MODEL_EMBEDDINGS["scibert"]["emb"],
                              MODEL_EMBEDDINGS["scibert"]["ids"])
    pos_s = {a: i for i, a in enumerate(s_ids)}
    pos_b = {a: i for i, a in enumerate(b_ids)}

    common = [a for a in s_ids if a in pos_b]          # keep SPECTER2 order
    if len(common) != len(s_ids) or len(common) != len(b_ids):
        logger.warning("Corpora not identical: specter2=%d scibert=%d common=%d",
                       len(s_ids), len(b_ids), len(common))

    corpus_s = s_emb[[pos_s[a] for a in common]]
    corpus_b = b_emb[[pos_b[a] for a in common]]
    id_to_pos = {a: i for i, a in enumerate(common)}
    logger.info("Aligned corpora: %d papers (768-dim each)", len(common))
    return corpus_s, corpus_b, id_to_pos


def summarize(ranks):
    """Same metric definitions as evaluation.py (one relevant doc per query)."""
    ranks = np.asarray(ranks)
    out = {}
    for k in K_VALUES:
        in_topk = ranks <= k
        out[f"hit@{k}"]       = round(float(np.mean(in_topk)), 4)
        out[f"precision@{k}"] = round(float(np.mean(in_topk) / k), 4)
        dcg = np.where(in_topk, 1.0 / np.log2(ranks + 1.0), 0.0)
        out[f"ndcg@{k}"]      = round(float(np.mean(dcg)), 4)
    out["mrr"]         = round(float(np.mean(1.0 / ranks)), 4)
    out["median_rank"] = int(np.median(ranks))
    out["mean_rank"]   = round(float(np.mean(ranks)), 1)
    return out


def main():
    df = pd.read_excel(QUERIES_FILE, dtype=str)
    logger.info("Loaded %d query rows", len(df))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus_s, corpus_b, id_to_pos = load_aligned_corpora()

    # Load both query encoders once.
    tok_s, mdl_s, dev_s, pool_s = load_query_encoder("specter2")
    tok_b, mdl_b, dev_b, pool_b = load_query_encoder("scibert")

    rows = []
    for qtype in QUERY_TYPES:
        if qtype not in df.columns:
            logger.error("Column '%s' not in queries — skipping.", qtype)
            continue

        pairs = [
            (str(r["arxiv_id"]), r[qtype].strip())
            for _, r in df.iterrows()
            if str(r["arxiv_id"]) in id_to_pos
            and isinstance(r[qtype], str) and r[qtype].strip()
        ]
        if not pairs:
            logger.warning("No usable queries for %s — skipping.", qtype)
            continue
        src_ids = [s for s, _ in pairs]
        queries = [q for _, q in pairs]
        src_pos = np.array([id_to_pos[s] for s in src_ids])
        logger.info("[%s] %d queries", qtype, len(queries))

        # Embed the queries once per model, then reuse across all weights.
        q_s = embed_queries(queries, tok_s, mdl_s, dev_s, pool_s)
        q_b = embed_queries(queries, tok_b, mdl_b, dev_b, pool_b)

        # Full cosine matrices (queries x papers), computed once.
        cos_s = q_s @ corpus_s.T          # (nq, npapers)
        cos_b = q_b @ corpus_b.T

        for w in WEIGHTS:
            combined = w * cos_s + (1.0 - w) * cos_b
            src_scores = combined[np.arange(len(src_ids)), src_pos]
            # rank = (# papers scoring strictly higher) + 1
            ranks = (combined > src_scores[:, None]).sum(axis=1) + 1
            summary = summarize(ranks)
            summary.update({"weight_specter2": w, "query_type": qtype,
                            "n_queries": len(ranks)})
            rows.append(summary)
            logger.info("  w=%.2f  mrr=%.4f  hit@10=%.4f  median_rank=%d",
                        w, summary["mrr"], summary["hit@10"], summary["median_rank"])

    out = pd.DataFrame(rows)
    lead = ["weight_specter2", "query_type", "n_queries"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    dest = RESULTS_DIR / "known_item_hybrid_sweep.csv"
    out.to_csv(dest, index=False)

    print("\n" + "=" * 70)
    print("HYBRID (late fusion) KNOWN-ITEM  —  weight on SPECTER2 vs metrics")
    print("=" * 70)
    print(out.to_string(index=False))
    print("=" * 70)
    print("w=1.0 == SPECTER2 only, w=0.0 == SciBERT only. Compare against the")
    print("existing known_item_summary.csv to confirm w=1.0 matches SPECTER2.")
    logger.info("Saved %s", dest)


if __name__ == "__main__":
    main()
