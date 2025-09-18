#!/usr/bin/env python3
"""Assemble per edge-time shade metrics into a flat Parquet table."""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_edge_time_index(points_path: Path) -> pd.DataFrame:
    """Return one row per (edge_uid, time, rounded_timestamp, binned_date)."""

    cols = ['edge_uid', 'time', 'rounded_timestamp', 'binned_date']
    df = pd.read_parquet(points_path, columns=cols)

    df = df.drop_duplicates(subset=cols).reset_index(drop=True)

    df = df.rename(columns={'time': 'timestamp_original',
                            'rounded_timestamp': 'timestamp_rounded'})

    df['hour_of_day'] = df['timestamp_rounded'].dt.strftime('%H%M')

    return df


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(metrics_path)

    if 'binned_date' in df.columns:
        df['binned_date'] = pd.to_datetime(df['binned_date'])

    if 'hour_of_day' not in df.columns and 'timestamp_rounded' in df.columns:
        df['hour_of_day'] = pd.to_datetime(df['timestamp_rounded']).dt.strftime('%H%M')

    metric_keys = ['edge_uid', 'binned_date', 'hour_of_day']
    if not set(metric_keys).issubset(df.columns):
        missing = ', '.join(sorted(set(metric_keys) - set(df.columns)))
        raise ValueError(f"Metrics parquet missing required columns: {missing}")

    df = df.drop_duplicates(subset=metric_keys)

    value_cols = [c for c in df.columns if c not in metric_keys]

    return df, value_cols


def load_edge_metadata(edges_path: Path) -> pd.DataFrame:
    gdf = gpd.read_file(edges_path)
    cols = [c for c in gdf.columns if c != 'geometry']
    return gdf[cols].drop_duplicates()


def build_table(points_path: Path, metrics_path: Path, edges_path: Path,
                output_path: Path) -> None:
    base_df = load_edge_time_index(points_path)
    metrics_df, metric_cols = load_metrics(metrics_path)

    merged = base_df.merge(metrics_df, how='left',
                           on=['edge_uid', 'binned_date', 'hour_of_day'],
                           suffixes=('', '_metric'))

    edge_meta = load_edge_metadata(edges_path)
    merged = merged.merge(edge_meta, how='left', on='edge_uid')

    if len(merged) != len(base_df):
        raise ValueError(f"Row count mismatch after merge: expected {len(base_df)}, got {len(merged)}")

    merged = merged.sort_values(['edge_uid', 'timestamp_rounded']).reset_index(drop=True)

    preferred_cols = ['edge_uid']
    for col in ['osmId', 'timestamp_original', 'timestamp_rounded', 'hour_of_day', 'binned_date']:
        if col in merged.columns and col not in preferred_cols:
            preferred_cols.append(col)
    metric_cols_ordered = [c for c in metric_cols if c in merged.columns]
    remaining_cols = [c for c in merged.columns if c not in preferred_cols + metric_cols_ordered]
    ordered_cols = preferred_cols + metric_cols_ordered + remaining_cols
    merged = merged[ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print(f"Wrote {len(merged):,} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create edge-time shade metrics table")
    parser.add_argument('--points', required=True, type=Path,
                        help='Parquet with point-level mapping (edge_uid, time, rounded_timestamp, binned_date)')
    parser.add_argument('--metrics', required=True, type=Path,
                        help='Parquet produced by edge shade extractor')
    parser.add_argument('--edges', required=True, type=Path,
                        help='GeoJSON with edge metadata (geometry ignored)')
    parser.add_argument('--output', required=True, type=Path,
                        help='Destination Parquet path (no geometry)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    build_table(args.points, args.metrics, args.edges, args.output)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
