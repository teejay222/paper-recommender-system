"""Data helpers for the Evaluation tab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Readers

def read_known_item(eval_dir):
    """Known-item summary (SPECTER2 vs SciBERT), or None if not generated yet."""
    p = Path(eval_dir) / "known_item_summary.csv"
    return pd.read_csv(p) if p.exists() else None


def read_known_item_queries(eval_dir, model: str, qtype: str):
    """Read known item queries."""
    p = Path(eval_dir) / f"known_item_{model}_{qtype}.csv"
    return pd.read_csv(p, dtype={"arxiv_id": str}) if p.exists() else None


def read_proxy(proxy_dir, name: str):
    """One proxy-experiment export (experiments_<name>.csv), or None if absent."""
    p = Path(proxy_dir) / f"experiments_{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def read_proxy_sweep(proxy_dir):
    """Read proxy sweep."""
    for name in ("smalleralpha_sweep", "alpha_sweep"):
        p = Path(proxy_dir) / f"experiments_{name}.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


def read_hybrid_sweep(eval_dir):
    """Hybrid late-fusion sweep (known_item_hybrid_sweep.csv), or None."""
    p = Path(eval_dir) / "known_item_hybrid_sweep.csv"
    return pd.read_csv(p) if p.exists() else None


# Derived summaries (computed from the known-item summary frame)

def overall_metrics(summary_df):
    """Overall metrics."""
    wanted = ["mrr", "hit@5", "hit@10", "hit@20", "ndcg@10"]
    cols = [c for c in wanted if c in summary_df.columns]
    agg = summary_df.groupby("model", as_index=False)[cols].mean()
    agg = agg.rename(columns={"hit@5": "recall@5", "hit@10": "recall@10",
                              "hit@20": "recall@20"})
    if "mrr" in agg.columns:
        agg = agg.sort_values("mrr", ascending=False).reset_index(drop=True)
    return agg


def recall_curve(summary_df):
    """Prepare recall values for plotting."""
    rows = []
    for model, g in summary_df.groupby("model"):
        for k in (5, 10, 20):
            col = f"hit@{k}"
            if col in g.columns:
                rows.append({"model": model, "k": k, "recall": float(g[col].mean())})
    return pd.DataFrame(rows)


def model_delta(overall_df, primary: str = "specter2", baseline: str = "scibert"):
    """Model delta."""
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
    """Failure picks."""
    if queries_df is None or queries_df.empty or "rank" not in queries_df.columns:
        return None
    s = queries_df.sort_values("rank").reset_index(drop=True)
    best, worst, med = s.iloc[0], s.iloc[-1], s.iloc[len(s) // 2]
    return pd.DataFrame([
        {"label": "Best", "query": best["query"], "rank": int(best["rank"])},
        {"label": "Median", "query": med["query"], "rank": int(med["rank"])},
        {"label": "Worst", "query": worst["query"], "rank": int(worst["rank"])},
    ])


# Hybrid (late-fusion) weight sweep — optional experiment file

def hybrid_overall(sweep_df):
    """Hybrid overall."""
    g = (sweep_df.groupby("weight_specter2")
                 .agg(mrr=("mrr", "mean"), recall10=("hit@10", "mean"))
                 .reset_index()
                 .rename(columns={"recall10": "recall@10"}))
    return g.sort_values("weight_specter2").reset_index(drop=True)


def hybrid_best(sweep_df):
    """Hybrid best."""
    g = hybrid_overall(sweep_df)
    if g.empty:
        return None
    best = g.loc[g["mrr"].idxmax()]
    base = g.loc[g["weight_specter2"] == 1.0, "mrr"]
    return {"weight": float(best["weight_specter2"]),
            "mrr": float(best["mrr"]),
            "baseline": float(base.iloc[0]) if not base.empty else None}
