from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

FILES = [
    "papers_2021.csv",
    "papers_2022.csv",
    "papers_2023.csv",
    "papers_2024.csv",
    "papers_2025.csv",
    "papers_2026.csv"
]

all_dataframes = []

report_lines = []

report_lines.append("# Dataset Validation Report\n")

for file_name in FILES:

    file_path = RAW_DIR / file_name

    df = pd.read_csv(file_path)

    all_dataframes.append(df)

    total_records = len(df)

    unique_ids = df["arxiv_id"].nunique()

    duplicate_ids = total_records - unique_ids

    missing_title = df["title"].isna().sum()

    missing_abstract = df["abstract"].isna().sum()

    missing_authors = df["authors"].isna().sum()

    missing_category = df["category"].isna().sum()

    citation_zero = (df["citation_count"] == 0).sum()

    reference_zero = (df["reference_count"] == 0).sum()

    report_lines.append(f"\n## {file_name}")
    report_lines.append(f"Total Records: {total_records}")
    report_lines.append(f"Unique arXiv IDs: {unique_ids}")
    report_lines.append(f"Duplicate IDs: {duplicate_ids}")
    report_lines.append(f"Missing Titles: {missing_title}")
    report_lines.append(f"Missing Abstracts: {missing_abstract}")
    report_lines.append(f"Missing Authors: {missing_authors}")
    report_lines.append(f"Missing Categories: {missing_category}")
    report_lines.append(f"Citation Count = 0: {citation_zero}")
    report_lines.append(f"Reference Count = 0: {reference_zero}")

    report_lines.append("\nCategory Distribution:")

    category_counts = (
        df["category"]
        .value_counts()
        .sort_index()
    )

    for cat, count in category_counts.items():

        pct = (count / total_records) * 100

        report_lines.append(
            f"- {cat}: {count} ({pct:.2f}%)"
        )

combined_df = pd.concat(
    all_dataframes,
    ignore_index=True
)

total_records = len(combined_df)

unique_ids = combined_df["arxiv_id"].nunique()

duplicates = total_records - unique_ids

report_lines.append("\n# Combined Dataset")

report_lines.append(
    f"Total Records: {total_records}"
)

report_lines.append(
    f"Unique Records: {unique_ids}"
)

report_lines.append(
    f"Cross-Year Duplicates: {duplicates}"
)

report_path = REPORT_DIR / "data_validation_report.md"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(report_lines))

print("=" * 60)
print("Validation Complete")
print(f"Report saved to: {report_path}")
print("=" * 60)