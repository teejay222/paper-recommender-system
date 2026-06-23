"""
app/evaluation.py — data loaders + computations for the "Evaluation" tab.

Pure data logic (pandas only), no Streamlit — mirrors app/analytics.py: app.py
wraps these with caching and renders. This is DISTINCT from src/evaluation.py,
which is the script that *computes* the known-item results; this module only
*reads* the files that script (and the experiment exports) produce and derives
small summary frames from them.

Note: with exactly one relevant paper per known-item query, Recall@K and Hit@K
are the same quantity (the source paper is or isn't in the top K). The summary
files store the columns as "hit@K"; everything user-facing is labelled Recall@K.

Public functions:
    read_known_item(eval_dir)             -> DataFrame | None   summary frame
    read_known_item_queries(dir, m, q)    -> DataFrame | None   per-query ranks
    read_proxy(proxy_dir, name)           -> DataFrame | None   one proxy export
    read_proxy_sweep(proxy_dir)           -> DataFrame | None    the alpha sweep
    read_hybrid_sweep(eval_dir)           -> DataFrame | None    late-fusion sweep
    overall_metrics(summary_df)           -> DataFrame          per-model averages
    recall_curve(summary_df)              -> DataFrame          recall@{5,10,20}
    model_delta(overall_df)               -> dict | None        SPECTER2 vs SciBERT
    failure_picks(queries_df)             -> DataFrame | None   best/median/worst
    hybrid_overall(sweep_df)              -> DataFrame          per-weight averages
    hybrid_best(sweep_df)                 -> dict | None        best blend vs w=1.0
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_known_item(eval_dir):
    """Known-item summary (SPECTER2 vs SciBERT), or None if not generated yet."""
    p = Path(eval_dir) / "known_item_summary.csv"
    return pd.read_csv(p) if p.exists() else None


def read_known_item_queries(eval_dir, model: str, qtype: str):
    """Per-query ground-truth ranks for one model x query-type, or None.

    Reads known_item_<model>_<qtype>.csv (columns: arxiv_id, query, rank).
    """
    p = Path(eval_dir) / f"known_item_{model}_{qtype}.csv"
    return pd.read_csv(p, dtype={"arxiv_id": str}) if p.exists() else None


def read_proxy(proxy_dir, name: str):
    """One proxy-experiment export (experiments_<name>.csv), or None if absent."""
    p = Path(proxy_dir) / f"experiments_{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def read_proxy_sweep(proxy_dir):
    """The category-boost (alpha) sweep.

    Prefers the fine-grained small-alpha file (more resolution around the knee),
    falling back to the coarse sweep, or None if neither exists.
    """
    for name in ("smalleralpha_sweep", "alpha_sweep"):
        p = Path(proxy_dir) / f"experiments_{name}.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


def read_hybrid_sweep(eval_dir):
    """Hybrid late-fusion sweep (known_item_hybrid_sweep.csv), or None."""
    p = Path(eval_dir) / "known_item_hybrid_sweep.csv"
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Derived summaries (computed from the known-item summary frame)
# ---------------------------------------------------------------------------

def overall_metrics(summary_df):
    """Per-model metrics averaged across the query types.

    Query types carry equal query counts, so the simple mean equals the pooled
    (micro) average over all queries. hit@K columns are surfaced as recall@K.
    Returns columns: model, mrr, recall@5, recall@10, recall@20, ndcg@10
    (whichever are present), sorted best-MRR first.
    """
    wanted = ["mrr", "hit@5", "hit@10", "hit@20", "ndcg@10"]
    cols = [c for c in wanted if c in summary_df.columns]
    agg = summary_df.groupby("model", as_index=False)[cols].mean()
    agg = agg.rename(columns={"hit@5": "recall@5", "hit@10": "recall@10",
                              "hit@20": "recall@20"})
    if "mrr" in agg.columns:
        agg = agg.sort_values("mrr", ascending=False).reset_index(drop=True)
    return agg


def recall_curve(summary_df):
    """Long-form recall@{5,10,20} per model, averaged across query types.

    Returns columns: model, k, recall.
    """
    rows = []
    for model, g in summary_df.groupby("model"):
        for k in (5, 10, 20):
            col = f"hit@{k}"
            if col in g.columns:
                rows.append({"model": model, "k": k, "recall": float(g[col].mean())})
    return pd.DataFrame(rows)


def model_delta(overall_df, primary: str = "specter2", baseline: str = "scibert"):
    """How far the primary model leads the baseline on MRR and Recall@10.

    Returns {metric: {"primary": v, "baseline": v, "factor": v|None}} or None.
    """
    def _row(m):
        r = overall_df[overall_df["model"] == m]
        return None if r.empty else r.iloc[0]

    p, b = _row(primary), _row(baseline)
    if p is None or b is None:
        return None
    out = {}
    for col in ("mrr", "recall@10"):
        if col in overall_df.columns:
            pv, bv = float(p[col]), float(b[col])
            out[col] = {"primary": pv, "baseline": bv,
                        "factor": (pv / bv) if bv > 0 else None}
    return out or None


def failure_picks(queries_df):
    """Best / median / worst query by the source paper's rank.

    Returns a 3-row frame (label, query, rank), or None if input is unusable.
    """
    if queries_df is None or queries_df.empty or "rank" not in queries_df.columns:
        return None
    s = queries_df.sort_values("rank").reset_index(drop=True)
    best, worst, med = s.iloc[0], s.iloc[-1], s.iloc[len(s) // 2]
    return pd.DataFrame([
        {"label": "Best", "query": best["query"], "rank": int(best["rank"])},
        {"label": "Median", "query": med["query"], "rank": int(med["rank"])},
        {"label": "Worst", "query": worst["query"], "rank": int(worst["rank"])},
    ])


# ---------------------------------------------------------------------------
# Hybrid (late-fusion) weight sweep — optional experiment file
# ---------------------------------------------------------------------------

def hybrid_overall(sweep_df):
    """Per-weight metrics averaged across query types (equal n => pooled mean).

    Returns columns: weight_specter2, mrr, recall@10 (sorted weight ascending).
    """
    g = (sweep_df.groupby("weight_specter2")
                 .agg(mrr=("mrr", "mean"), recall10=("hit@10", "mean"))
                 .reset_index()
                 .rename(columns={"recall10": "recall@10"}))
    return g.sort_values("weight_specter2").reset_index(drop=True)


def hybrid_best(sweep_df):
    """Best overall weight by averaged MRR, vs the SPECTER2-only (w=1.0) baseline.

    Returns {"weight", "mrr", "baseline"} or None.
    """
    g = hybrid_overall(sweep_df)
    if g.empty:
        return None
    best = g.loc[g["mrr"].idxmax()]
    base = g.loc[g["weight_specter2"] == 1.0, "mrr"]
    return {"weight": float(best["weight_specter2"]),
            "mrr": float(best["mrr"]),
            "baseline": float(base.iloc[0]) if not base.empty else None}
