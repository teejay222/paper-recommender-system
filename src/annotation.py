"""
src/annotation.py
-----------------
Phase 3: Automated annotation.
Operates on data/processed/papers_cleaned.csv.

What this file does:
  1. Maps each arXiv category code to a human-readable label
       cs.AI -> Artificial Intelligence
       cs.LG -> Machine Learning
       cs.CL -> Natural Language Processing
       cs.CV -> Computer Vision

  2. Extracts keyword-based secondary tags from each paper's abstract
     using a local dictionary — each category has its own keyword list.
     Tags are matched by scanning the abstract for keyword presence.
     A paper gets only the tags whose keywords appear in its abstract.

  3. Saves annotated data to data/processed/papers_annotated.csv

What this file does NOT do:
  - No manual labeling
  - No Label Studio
  - No supervised training labels
  - Labels are used for: visualization, frontend filtering,
    evaluation topic consistency, and analytics only (per spec Section 7)

Output:
  data/processed/papers_annotated.csv

New columns added:
  - domain_label  : Human-readable category name (str)
  - tags          : Semicolon-separated keyword tags from abstract (str)
  - tag_count     : Number of tags matched (int)
"""

import logging
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_FILE  = Path("data/processed/papers_clean.csv")
OUTPUT_FILE = Path("data/processed/papers_annotated.csv")
REPORT_FILE = Path("reports/annotation_report.txt")


# ---------------------------------------------------------------------------
# Category -> human-readable label mapping
# Directly from spec Section 7
# ---------------------------------------------------------------------------

CATEGORY_LABEL_MAP = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Natural Language Processing",
    "cs.CV": "Computer Vision",
}


# ---------------------------------------------------------------------------
# Local keyword dictionary
# Each category has its own keyword list.
# Keywords are matched against the lowercase abstract text.
# A tag is assigned if the keyword appears anywhere in the abstract.
#
# Design decisions:
#   - Multi-word phrases are included (e.g. "reinforcement learning")
#     and are matched before single words to avoid partial overlaps
#   - Keywords are lowercase — matching is done on lowercased abstract
#   - Each keyword maps to a clean tag name (what gets stored in the CSV)
#   - Tags reflect research sub-topics within each category, useful for
#     filtering, clustering, and analytics in the frontend
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {

    "cs.AI": {
        # keyword to search in abstract : tag name to assign
        "large language model":       "large-language-models",
        "knowledge graph":            "knowledge-graphs",
        "reasoning":                  "reasoning",
        "planning":                   "planning",
        "multi-agent":                "multi-agent",
        "explainability":             "explainability",
        "interpretability":           "interpretability",
        "question answering":         "question-answering",
        "knowledge representation":   "knowledge-representation",
        "constraint satisfaction":    "constraint-satisfaction",
        "search algorithm":           "search-algorithms",
        "expert system":              "expert-systems",
        "autonomous agent":           "autonomous-agents",
        "llm":                        "large-language-models",
        "chatbot":                    "chatbots",
        "dialogue system":            "dialogue-systems",
        "common sense":               "commonsense-reasoning",
        "uncertainty":                "uncertainty",
        "bayesian":                   "bayesian-methods",
        "causal":                     "causal-inference",
    },

    "cs.LG": {
        "reinforcement learning":     "reinforcement-learning",
        "deep learning":              "deep-learning",
        "neural network":             "neural-networks",
        "graph neural":               "graph-neural-networks",
        "generative model":           "generative-models",
        "federated learning":         "federated-learning",
        "transfer learning":          "transfer-learning",
        "self-supervised":            "self-supervised-learning",
        "contrastive learning":       "contrastive-learning",
        "meta-learning":              "meta-learning",
        "few-shot":                   "few-shot-learning",
        "zero-shot":                  "zero-shot-learning",
        "optimization":               "optimization",
        "gradient":                   "gradient-methods",
        "generalization":             "generalization",
        "representation learning":    "representation-learning",
        "attention mechanism":        "attention-mechanisms",
        "transformer":                "transformers",
        "diffusion model":            "diffusion-models",
        "anomaly detection":          "anomaly-detection",
    },

    "cs.CL": {
        "transformer":                "transformers",
        "language model":             "language-models",
        "bert":                       "bert",
        "gpt":                        "gpt",
        "machine translation":        "machine-translation",
        "text classification":        "text-classification",
        "named entity":               "named-entity-recognition",
        "sentiment analysis":         "sentiment-analysis",
        "summarization":              "summarization",
        "question answering":         "question-answering",
        "information extraction":     "information-extraction",
        "dialogue":                   "dialogue-systems",
        "parsing":                    "parsing",
        "word embedding":             "word-embeddings",
        "semantic similarity":        "semantic-similarity",
        "coreference":                "coreference-resolution",
        "relation extraction":        "relation-extraction",
        "text generation":            "text-generation",
        "speech":                     "speech-processing",
        "multilingual":               "multilingual",
    },

    "cs.CV": {
        "object detection":           "object-detection",
        "image segmentation":         "image-segmentation",
        "image classification":       "image-classification",
        "convolutional":              "convolutional-networks",
        "generative adversarial":     "gans",
        "diffusion model":            "diffusion-models",
        "3d reconstruction":          "3d-reconstruction",
        "depth estimation":           "depth-estimation",
        "optical flow":               "optical-flow",
        "video understanding":        "video-understanding",
        "face recognition":           "face-recognition",
        "pose estimation":            "pose-estimation",
        "image generation":           "image-generation",
        "semantic segmentation":      "semantic-segmentation",
        "point cloud":                "point-clouds",
        "visual question":            "visual-question-answering",
        "multimodal":                 "multimodal",
        "vision transformer":         "vision-transformers",
        "self-supervised":            "self-supervised-learning",
        "medical image":              "medical-imaging",
    },
}


# ---------------------------------------------------------------------------
# Annotation functions
# ---------------------------------------------------------------------------

def assign_domain_label(category: str) -> str:
    """
    Maps an arXiv category code to a human-readable domain label.

    Args:
        category: arXiv category code e.g. "cs.CL"

    Returns:
        Human-readable label e.g. "Natural Language Processing"
        Returns empty string if category is not in the map.
    """
    return CATEGORY_LABEL_MAP.get(category, "")


def extract_tags(abstract: str, category: str) -> str:
    """
    Extracts keyword-based tags from a paper's abstract using the
    local keyword dictionary for the paper's category.

    Matching is case-insensitive. Multi-word phrases are checked
    before single words to avoid partial match issues.

    Args:
        abstract: Full abstract text of the paper
        category: arXiv category code e.g. "cs.LG"

    Returns:
        Semicolon-separated tag string e.g. "transformers;deep-learning"
        Returns empty string if no keywords match or category unknown.
    """
    if not isinstance(abstract, str) or not abstract.strip():
        return ""

    keyword_map = CATEGORY_KEYWORDS.get(category, {})
    if not keyword_map:
        return ""

    abstract_lower = abstract.lower()

    # Sort keywords by length descending so multi-word phrases are
    # checked before shorter single words
    sorted_keywords = sorted(keyword_map.keys(), key=len, reverse=True)

    matched_tags = []
    seen_tags    = set()   # prevent duplicate tags from synonym keywords

    for keyword in sorted_keywords:
        # Use word boundary matching to prevent substring false positives.
        # Example without this: "bert" would match inside "deliberate" or
        # "libertarian". With \b, only the exact word/phrase is matched.
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, abstract_lower, re.IGNORECASE):
            tag = keyword_map[keyword]
            if tag not in seen_tags:
                matched_tags.append(tag)
                seen_tags.add(tag)

    return ";".join(matched_tags)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain_label, tags, and tag_count columns to the DataFrame.

    Args:
        df: Cleaned papers DataFrame from papers_cleaned.csv

    Returns:
        Annotated DataFrame with three new columns added.
    """
    logger.info("Assigning domain labels...")
    df["domain_label"] = df["category"].apply(assign_domain_label)

    logger.info("Extracting keyword tags from abstracts...")
    df["tags"] = df.apply(
        lambda row: extract_tags(row["abstract"], row["category"]),
        axis=1
    )

    df["tag_count"] = df["tags"].apply(
        lambda t: len(t.split(";")) if t else 0
    )

    return df


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def build_annotation_report(df: pd.DataFrame) -> str:
    """
    Builds a text report summarizing the annotation results.

    Reports:
      - Domain label distribution
      - Tag coverage (how many papers got at least one tag)
      - Top tags per category
      - Papers with zero tags (may indicate abstract quality issues)

    Args:
        df: Fully annotated DataFrame

    Returns:
        Report as a multi-line string.
    """
    lines = []
    total = len(df)

    lines.append("=" * 60)
    lines.append("ANNOTATION REPORT")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # --- Domain label distribution ---
    lines.append("\n--- Domain Label Distribution ---")
    label_counts = df["domain_label"].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / total * 100
        lines.append(f"  {label:<35} {count:>6,}  ({pct:.2f}%)")

    # --- Tag coverage ---
    lines.append("\n--- Tag Coverage ---")
    papers_with_tags    = (df["tag_count"] > 0).sum()
    papers_without_tags = (df["tag_count"] == 0).sum()
    lines.append(f"  Papers with at least 1 tag : {papers_with_tags:,} ({papers_with_tags/total*100:.2f}%)")
    lines.append(f"  Papers with 0 tags         : {papers_without_tags:,} ({papers_without_tags/total*100:.2f}%)")
    lines.append(f"  Average tags per paper     : {df['tag_count'].mean():.2f}")
    lines.append(f"  Max tags on one paper      : {df['tag_count'].max()}")

    # --- Top tags per category ---
    lines.append("\n--- Top 10 Tags Per Category ---")
    for category, label in CATEGORY_LABEL_MAP.items():
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue

        # Flatten all tags across papers in this category into one list
        all_tags = []
        for tag_str in cat_df["tags"]:
            if tag_str:
                all_tags.extend(tag_str.split(";"))

        if not all_tags:
            lines.append(f"\n  {label}: no tags found")
            continue

        tag_series = pd.Series(all_tags).value_counts().head(10)
        lines.append(f"\n  {label} ({category}):")
        for tag, count in tag_series.items():
            pct = count / len(cat_df) * 100
            lines.append(f"    {tag:<40} {count:>5,}  ({pct:.1f}% of category papers)")

    # --- Papers with zero tags per category ---
    lines.append("\n--- Zero-Tag Papers Per Category ---")
    for category, label in CATEGORY_LABEL_MAP.items():
        cat_df    = df[df["category"] == category]
        zero_tags = (cat_df["tag_count"] == 0).sum()
        pct       = zero_tags / len(cat_df) * 100 if len(cat_df) > 0 else 0
        lines.append(f"  {label:<35} {zero_tags:>5,}  ({pct:.2f}%)")

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_annotation() -> None:
    """
    Runs the full annotation pipeline.

    Loads cleaned data, assigns domain labels, extracts keyword tags,
    saves annotated dataset and quality report.
    """
    logger.info("=" * 60)
    logger.info("Annotation started")
    logger.info("Input : %s", INPUT_FILE)
    logger.info("Output: %s", OUTPUT_FILE)
    logger.info("=" * 60)

    # --- Load ---
    logger.info("Loading cleaned data...")
    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded %d papers", len(df))

    # --- Annotate ---
    df = annotate(df)

    # --- Report ---
    report_text = build_annotation_report(df)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print("\n" + report_text + "\n")

    # --- Save ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Annotated dataset saved -> %s", OUTPUT_FILE)
    logger.info("Annotation report saved -> %s", REPORT_FILE)
    logger.info("Annotation complete. Total papers: %d", len(df))


if __name__ == "__main__":
    run_annotation()