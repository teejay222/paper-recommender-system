#!/usr/bin/env python3
"""Clean raw paper data before annotation and modeling."""

import logging
from pathlib import Path
import pandas as pd
import ftfy 

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Config
INPUT_FILE  = Path("data/raw/papers_raw.csv")
OUTPUT_FILE = Path("data/processed/papers_clean.csv")
REPORT_FILE = Path("reports/preprocessing_report.txt")

# Minimum text length for embedding models.
MIN_ABSTRACT_LENGTH = 50

CRITICAL_FIELDS = ["title", "abstract", "authors", "category"]
VALID_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "cs.CV"}


# Cleaning helpers

def remove_duplicate_arxiv_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates(subset=["arxiv_id"], keep="first")
    removed = before - len(df)
    logger.info("Duplicate arxiv_id records removed: %d", removed)
    return df, removed


def remove_duplicate_titles(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    
    # Use a lowercase helper column.
    df["_temp_title_lower"] = df["title"].astype(str).str.lower()
    
    # Drop duplicates.
    df = df.drop_duplicates(subset=["_temp_title_lower"], keep="first")
    
    # Remove the helper column.
    df = df.drop(columns=["_temp_title_lower"])
    
    removed = before - len(df)
    logger.info("Duplicate case-insensitive title records removed: %d", removed)
    return df, removed


def drop_missing_critical_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = {}
    before = len(df)

    for field in CRITICAL_FIELDS:
        field_before = len(df)
        df = df[df[field].notna() & (df[field].astype(str).str.strip() != "")]
        counts[field] = field_before - len(df)

    total_removed = before - len(df)
    logger.info("Rows dropped due to empty critical fields: %d", total_removed)
    return df, counts


def filter_invalid_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df[df["category"].isin(VALID_CATEGORIES)]
    removed = before - len(df)
    logger.info("Rows removed for invalid out-of-scope categories: %d", removed)
    return df, removed


def filter_short_abstracts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df[df["abstract"].astype(str).str.len() >= MIN_ABSTRACT_LENGTH]
    removed = before - len(df)
    logger.info("Rows removed for brief/corrupt abstracts (< %d chars): %d", MIN_ABSTRACT_LENGTH, removed)
    return df, removed


def normalize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text fields."""
    for col in ["title", "abstract", "authors"]:
        df[col] = (
            df[col]
            .astype(str)
            .apply(ftfy.fix_text)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    logger.info("Encoding artifacts fixed and whitespace normalized in title, abstract, authors")
    return df


def clean_citation_fields(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["citation_count", "reference_count"]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    logger.info("Metrics features cast successfully to Integer data types.")
    return df


def add_abstract_length(df: pd.DataFrame) -> pd.DataFrame:
    df["abstract_length"] = df["abstract"].str.len()
    logger.info("Feature columns added: abstract_length.")
    return df


def compute_citation_integrity(df: pd.DataFrame) -> dict:
    total = len(df)
    citation_nonzero  = (df["citation_count"] > 0).sum()
    reference_nonzero = (df["reference_count"] > 0).sum()

    return {
        "total_papers":              total,
        "citation_count_nonzero":    int(citation_nonzero),
        "citation_count_zero":       int(total - citation_nonzero),
        "citation_completeness_pct": round(citation_nonzero / total * 100, 2),
        "reference_count_nonzero":   int(reference_nonzero),
        "reference_count_zero":      int(total - reference_nonzero),
        "reference_completeness_pct": round(reference_nonzero / total * 100, 2),
    }


# Report builder

def build_report(
    rows_before:        int,
    rows_after:         int,
    dup_id_removed:     int,
    dup_title_removed:  int,
    missing_removed:    dict,
    invalid_cat_removed: int,
    short_abs_removed:  int,
    citation_stats:     dict,
    category_dist:      pd.Series,
    year_dist:          pd.Series,
    abstract_stats:     dict,
) -> str:
    lines = [
        "=" * 60,
        "PREPROCESSING QUALITY REPORT",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "\n--- Row Counts ---",
        f"Rows before cleaning : {rows_before:,}",
        f"Rows after cleaning  : {rows_after:,}",
        f"Total rows removed   : {rows_before - rows_after:,}",
        f"Retention rate       : {rows_after / rows_before * 100:.2f}%",
        "\n--- Duplicate Removal ---",
        f"Duplicate arxiv_id removed : {dup_id_removed:,}",
        f"Duplicate title removed    : {dup_title_removed:,}",
        "\n--- Missing Critical Fields Removed ---"
    ]
    for field, count in missing_removed.items():
        lines.append(f"  {field}: {count:,}")

    lines.extend([
        "\n--- Invalid Category Removal ---",
        f"Rows removed (not in target categories): {invalid_cat_removed:,}",
        "\n--- Short Abstract Removal ---",
        f"Minimum abstract threshold: {MIN_ABSTRACT_LENGTH} characters",
        f"Rows removed              : {short_abs_removed:,}",
        "\n--- Citation Metric Integrity ---",
        f"Papers with citations > 0 : {citation_stats['citation_count_nonzero']:,} ({citation_stats['citation_completeness_pct']}%)",
        f"Papers with citations = 0 : {citation_stats['citation_count_zero']:,}",
        f"Papers with references > 0: {citation_stats['reference_count_nonzero']:,} ({citation_stats['reference_completeness_pct']}%)",
        f"Papers with references = 0: {citation_stats['reference_count_zero']:,}",
        "Note: New papers are naturally expected to have 0 metrics; rows are not dropped.",
        "\n--- Target Category Distribution (Cleaned Dataset) ---"
    ])
    for cat, count in category_dist.items():
        lines.append(f"  {cat}: {count:,} ({count / rows_after * 100:.2f}%)")

    lines.append("\n--- Publication Year Distribution (Cleaned Dataset) ---")
    for year, count in year_dist.sort_index().items():
        lines.append(f"  {year}: {count:,} ({count / rows_after * 100:.2f}%)")

    lines.extend([
        "\n--- Abstract Character Length Distribution ---",
        f"  Min     : {abstract_stats['min']:,}",
        f"  Max     : {abstract_stats['max']:,}",
        f"  Mean    : {abstract_stats['mean']:.1f}",
        f"  Median  : {abstract_stats['median']:.1f}",
        "\n" + "=" * 60,
        "END OF REPORT",
        "=" * 60
    ])
    return "\n".join(lines)


# Run

def preprocess() -> None:
    logger.info("=" * 60)
    logger.info("Initializing Data Preprocessing Engine")
    logger.info("=" * 60)

    if not INPUT_FILE.exists():
        logger.critical("Missing raw data entry file: %s", INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE, dtype=str)
    rows_before = len(df)

    # Clean data
    df, dup_id_removed = remove_duplicate_arxiv_ids(df)
    df, dup_title_removed = remove_duplicate_titles(df)
    df, missing_removed = drop_missing_critical_fields(df)
    df, invalid_cat_removed = filter_invalid_categories(df)
    df, short_abs_removed = filter_short_abstracts(df)
    df = normalize_text_fields(df)
    df = clean_citation_fields(df)
    df = add_abstract_length(df)

    df = df.reset_index(drop=True)
    rows_after = len(df)

    # Summary stats
    citation_stats = compute_citation_integrity(df)
    category_dist  = df["category"].value_counts().sort_index()
    year_dist      = df["publication_year"].value_counts()

    abstract_stats = {
        "min":    int(df["abstract_length"].min()),
        "max":    int(df["abstract_length"].max()),
        "mean":   float(df["abstract_length"].mean()),
        "median": float(df["abstract_length"].median()),
    }

    report_text = build_report(
        rows_before, rows_after, dup_id_removed, dup_title_removed,
        missing_removed, invalid_cat_removed, short_abs_removed,
        citation_stats, category_dist, year_dist, abstract_stats
    )

    # Write report and cleaned data
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print("\n" + report_text + "\n")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    logger.info("Pristine processed target asset saved -> %s", OUTPUT_FILE)
    logger.info("Quality tracking logs recorded -> %s", REPORT_FILE)


if __name__ == "__main__":
    preprocess()