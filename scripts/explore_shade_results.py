#!/usr/bin/env python3
"""Quick utilities for inspecting shade extraction parquet outputs."""

import argparse
from pathlib import Path

import pandas as pd


def summarize(path: Path, sample: int) -> None:
    df = pd.read_parquet(path)

    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows from {path}")
    print("\nColumns:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    shade_like = [c for c in df.columns if 'shade' in c or 'shadow' in c]

    print("\nUnique values:")
    if 'edge_uid' in df.columns:
        print(f"  edge_uid: {df['edge_uid'].nunique():,}")
    if 'binned_date' in df.columns:
        print(f"  binned_date: {df['binned_date'].nunique():,}")
    if 'hour_of_day' in df.columns:
        print(f"  hour_of_day: {df['hour_of_day'].nunique():,}")

    print("\nMissing counts (selected columns):")
    for col in shade_like:
        missing = df[col].isna().sum()
        pct = missing / total_rows * 100 if total_rows else 0
        print(f"  {col}: {missing:,} ({pct:.1f}%)")

    print("\nDescriptive stats (shade metrics):")
    stats_cols = [c for c in shade_like if pd.api.types.is_numeric_dtype(df[c])]
    if stats_cols:
        print(df[stats_cols].describe().transpose())
    else:
        print("  No numeric shade-like columns detected")

    if sample > 0:
        print(f"\nSample {sample} rows:")
        print(df.sample(n=min(sample, total_rows), random_state=42).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect parquet shade results")
    parser.add_argument("parquet", type=Path, help="Path to the parquet file to inspect")
    parser.add_argument("--sample", type=int, default=10, help="Number of rows to sample for display")
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"Error: {args.parquet} does not exist")
        return 1

    summarize(args.parquet, args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

