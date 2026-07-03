"""Data prep for the Analytics tab."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# A keyword must appear at least this often to enter the rising/declining lists.
# Adaptive: a small fraction of the corpus, with a hard floor so the trend lists
# don't disappear on smaller corpora (or hide everything on larger ones).
TAG_TREND_FLOOR = 20
TAG_TREND_RATIO = 0.002


def parse_tags(tags_str: str) -> list:
    """Parse saved tag strings."""
    s = (tags_str or "").strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [str(t).strip() for t in val if str(t).strip()]
        except Exception:                                   # noqa: BLE001
            pass
    parts = re.split(r"[,;|]", s)
    return [p.strip().strip("'\"") for p in parts if p.strip()]


def compute_analytics(df: pd.DataFrame) -> dict:
    """Build the frames used by the analytics tab."""
    years = sorted(int(y) for y in df["publication_year"].dropna().unique())

    headline = {
        "papers": len(df),
        "categories": int(df["category"].nunique()),
        "year_min": years[0], "year_max": years[-1],
        "pct_cited": round(float((df["citation_count"] > 0).mean()) * 100, 1),
        "mean_citations": round(float(df["citation_count"].mean()), 1),
        "max_citations": int(df["citation_count"].max()),
    }

    per_year = df.groupby("publication_year").size().reset_index(name="papers")
    per_cat = df["category"].value_counts().reset_index()
    per_cat.columns = ["category", "papers"]

    labels = ["0", "1-5", "6-20", "21-100", "100+"]
    buckets = pd.cut(df["citation_count"], bins=[-1, 0, 5, 20, 100, 10**18],
                     labels=labels)
    cit_buckets = buckets.value_counts().reindex(labels).reset_index()
    cit_buckets.columns = ["bucket", "papers"]

    top_cited = (df.nlargest(10, "citation_count")
                   [["title", "category", "publication_year",
                     "citation_count", "arxiv_id"]].reset_index(drop=True))
    top_cited["arxiv_url"] = "https://arxiv.org/abs/" + top_cited["arxiv_id"].astype(str)

    # Mean citations by category and by year. The mean is dragged up by a few
    # mega-cited papers (e.g. LoRA), so it is read as a rough signal, not a
    # typical value — the captions in the app say so.
    cit_by_cat = (df.groupby("category")["citation_count"].mean().round(1)
                    .reset_index(name="mean").sort_values("mean", ascending=False))
    cit_by_year = (df.groupby("publication_year")["citation_count"].mean().round(1)
                     .reset_index(name="mean"))

    # Keyword frequencies (overall and per year) from the KeyBERT tags.
    counts: Counter = Counter()
    tag_year: dict = defaultdict(Counter)
    for yr, raw in zip(df["publication_year"], df["keybert_tags_v2"]):
        for tag in parse_tags(raw):
            tag = tag.lower()
            counts[tag] += 1
            tag_year[tag][int(yr)] += 1
    top_tags = pd.DataFrame(counts.most_common(20), columns=["tag", "count"])

    first, last = years[0], years[-1]
    min_freq = max(TAG_TREND_FLOOR, round(len(df) * TAG_TREND_RATIO))
    cand = [t for t, n in counts.items() if n > min_freq]
    rising_tags = sorted(cand, key=lambda t: tag_year[t][last] - tag_year[t][first],
                         reverse=True)[:6]
    falling_tags = sorted(cand, key=lambda t: tag_year[t][last] - tag_year[t][first])[:6]

    def trend_frame(tags: list) -> pd.DataFrame:
        rows = [{"tag": t, "year": y, "count": tag_year[t][y]}
                for t in tags for y in years]
        return pd.DataFrame(rows)

    return {
        "headline": headline, "years": years,
        "per_year": per_year, "per_cat": per_cat,
        "cit_buckets": cit_buckets, "top_cited": top_cited,
        "cit_by_cat": cit_by_cat, "cit_by_year": cit_by_year,
        "top_tags": top_tags,
        "rising": trend_frame(rising_tags), "falling": trend_frame(falling_tags),
        "n_unique_tags": len(counts),
    }


def read_projections(figures_dir, model: str):
    """Loads precomputed 2-D projections for a model, or None if not generated."""
    p = Path(figures_dir) / f"projections_{model}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"arxiv_id": str})


def read_separation(figures_dir):
    """Loads the silhouette separation scores, or None if not generated yet."""
    p = Path(figures_dir) / "embedding_separation.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:                                       # noqa: BLE001
        return None


def read_similarity(figures_dir):
    """Loads same/different-category similarity pairs, or None if not generated."""
    p = Path(figures_dir) / "similarity_pairs.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)