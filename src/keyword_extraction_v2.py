#!/usr/bin/env python3
"""
src/keyword_extraction_v2.py
-----------------------------
Phase 3B (v2): KeyBERT tag extraction using Title + Abstract as input.

Differences from keyword_extraction.py (v1):
  - Input text: Title + Abstract combined (v1 used abstract only)
  - TOP_N: 8 candidates extracted, best 5 kept (v1 used TOP_N=4)
  - Score threshold: 0.20 (v1 used 0.35)

Input:  data/processed/papers_clean.csv
Output: data/processed/papers_keybert_v2.csv
"""

import logging
from pathlib import Path
import pandas as pd
from keybert import KeyBERT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/processed/papers_clean.csv")
OUTPUT_FILE = Path("data/processed/papers_keybert_v2.csv")

KEYPHRASE_NGRAM_RANGE = (1, 2)  # unigrams and bigrams
TOP_N_EXTRACT = 8               # candidates to extract per paper
SCORE_THRESHOLD = 0.20          # keep candidates with score >= this

def combine_title_abstract(title: str, abstract: str) -> str:
    """Combine title and abstract into a single input string for KeyBERT."""
    return f"Title: {str(title).strip()}. {str(abstract).strip()}"

def main():
    logger.info("=" * 60)
    logger.info("KeyBERT Tag Extraction v2 (Title + Abstract)")
    logger.info(f"Input : {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info(f"Config: TOP_N_EXTRACT={TOP_N_EXTRACT} | THRESHOLD={SCORE_THRESHOLD}")
    logger.info("=" * 60)

    if not INPUT_FILE.exists():
        logger.critical(f"Input file not found: {INPUT_FILE}")
        logger.critical("Run preprocessing.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} papers")

    # Build combined texts (title + abstract)
    texts = [
        combine_title_abstract(row["title"], row["abstract"])
        for _, row in df.iterrows()
    ]

    logger.info("Initializing KeyBERT model (all-MiniLM-L6-v2)...")
    kw_model = KeyBERT("all-MiniLM-L6-v2")

    logger.info("Extracting keywords (single batch)...")
    batch_results = kw_model.extract_keywords(
        texts,
        keyphrase_ngram_range=KEYPHRASE_NGRAM_RANGE,
        stop_words="english",
        top_n=TOP_N_EXTRACT
    )

    all_tags = []
    for keywords in batch_results:
        # Filter by threshold
        filtered = [(w, s) for w, s in keywords if s >= SCORE_THRESHOLD]
        # Keep top 5
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:5]
        # Format as hyphenated slugs
        tag_slugs = [w.strip().lower().replace(" ", "-") for w, _ in filtered]
        all_tags.append(";".join(tag_slugs))

    df["keybert_tags_v2"] = all_tags
    df["keybert_tag_count_v2"] = df["keybert_tags_v2"].apply(
        lambda t: len(t.split(";")) if t else 0
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved annotated dataset to {OUTPUT_FILE}")

    tagged = (df["keybert_tag_count_v2"] > 0).sum()
    avg = df["keybert_tag_count_v2"].mean()
    logger.info(f"Papers with >=1 tag: {tagged} / {len(df)} ({tagged/len(df)*100:.1f}%)")
    logger.info(f"Average tags per paper: {avg:.2f}")
    logger.info("KeyBERT extraction complete.")

if __name__ == "__main__":
    main()