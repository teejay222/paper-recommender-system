"""Label papers with broad domains and keyword tags."""

import logging
import re
from pathlib import Path

import pandas as pd

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Constants

INPUT_FILE  = Path("data/processed/papers_clean.csv")
OUTPUT_FILE = Path("data/processed/papers_annotated.csv")
REPORT_FILE = Path("reports/annotation_report.txt")


# Category labels
# Spec section 7

CATEGORY_LABEL_MAP = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Natural Language Processing",
    "cs.CV": "Computer Vision",
}


# Keyword list
# One list per category.
# Match against lowercase abstracts.
# Add the tag when the phrase appears.
# Notes:
# - Multi-word phrases are included (e.g. "reinforcement learning")
# This keeps short terms from winning too early.
# - Keywords are lowercase — matching is done on lowercased abstract
# - Each keyword maps to a clean tag name (what gets stored in the CSV)
# - Tags reflect research sub-topics within each category, useful for

CATEGORY_KEYWORDS = {

    "cs.AI": {
        # keyword in abstract : stored tag
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


# Annotation

def assign_domain_label(category: str) -> str:
    """Assign domain label."""
    return CATEGORY_LABEL_MAP.get(category, "")


def extract_tags(abstract: str, category: str) -> str:
    """Find dictionary tags in an abstract."""
    if not isinstance(abstract, str) or not abstract.strip():
        return ""

    keyword_map = CATEGORY_KEYWORDS.get(category, {})
    if not keyword_map:
        return ""

    abstract_lower = abstract.lower()

    # Check longer phrases first.
    sorted_keywords = sorted(keyword_map.keys(), key=len, reverse=True)

    matched_tags = []
    seen_tags    = set()   # prevent duplicate tags from synonym keywords

    for keyword in sorted_keywords:
        # Use word boundaries to avoid substring matches.
        # "bert" should not match inside another word.
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, abstract_lower, re.IGNORECASE):
            tag = keyword_map[keyword]
            if tag not in seen_tags:
                matched_tags.append(tag)
                seen_tags.add(tag)

    return ";".join(matched_tags)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate."""
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


# Report

def build_annotation_report(df: pd.DataFrame) -> str:
    """Build annotation report."""
    lines = []
    total = len(df)

    lines.append("=" * 60)
    lines.append("ANNOTATION REPORT")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Domain label distribution
    lines.append("\n--- Domain Label Distribution ---")
    label_counts = df["domain_label"].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / total * 100
        lines.append(f"  {label:<35} {count:>6,}  ({pct:.2f}%)")

    # Tag coverage
    lines.append("\n--- Tag Coverage ---")
    papers_with_tags    = (df["tag_count"] > 0).sum()
    papers_without_tags = (df["tag_count"] == 0).sum()
    lines.append(f"  Papers with at least 1 tag : {papers_with_tags:,} ({papers_with_tags/total*100:.2f}%)")
    lines.append(f"  Papers with 0 tags         : {papers_without_tags:,} ({papers_without_tags/total*100:.2f}%)")
    lines.append(f"  Average tags per paper     : {df['tag_count'].mean():.2f}")
    lines.append(f"  Max tags on one paper      : {df['tag_count'].max()}")

    # Top tags per category
    lines.append("\n--- Top 10 Tags Per Category ---")
    for category, label in CATEGORY_LABEL_MAP.items():
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue

        # Flatten tags for this category.
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

    # Papers with zero tags per category
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


# Main

def run_annotation() -> None:
    """Run annotation."""
    logger.info("=" * 60)
    logger.info("Annotation started")
    logger.info("Input : %s", INPUT_FILE)
    logger.info("Output: %s", OUTPUT_FILE)
    logger.info("=" * 60)

    # Load
    logger.info("Loading cleaned data...")
    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded %d papers", len(df))

    # Annotate
    df = annotate(df)

    # Report
    report_text = build_annotation_report(df)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print("\n" + report_text + "\n")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Annotated dataset saved -> %s", OUTPUT_FILE)
    logger.info("Annotation report saved -> %s", REPORT_FILE)
    logger.info("Annotation complete. Total papers: %d", len(df))


if __name__ == "__main__":
    run_annotation()