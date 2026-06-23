from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

FILES = [
    "papers_2021.csv",
    "papers_2022.csv",
    "papers_2023.csv",
    "papers_2024.csv",
    "papers_2025.csv",
    "papers_2026.csv"
]

dfs = []

for file in FILES:

    path = RAW_DIR / file

    df = pd.read_csv(path)

    dfs.append(df)

merged_df = pd.concat(
    dfs,
    ignore_index=True
)

output_path = RAW_DIR / "papers_raw.csv"

merged_df.to_csv(
    output_path,
    index=False
)

print("=" * 60)
print(f"Rows: {len(merged_df)}")
print(f"Saved: {output_path}")
print("=" * 60)