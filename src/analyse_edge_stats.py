#!/usr/bin/env python3
"""
Quick analysis helper for edge-based shade outputs produced by
`aggregate_edge_shade_stats.py`.

It loads the GeoJSON, lets you filter by edges and dates, and writes:
  1) per-edge/day summary stats
  2) per-edge/day monotonicity report for shadow_fraction
  3) days with <24 hours coverage per edge
  4) per-edge correlation (combined_shade vs combined_shadow_fraction)
  5) per-edge time-series CSVs
Also prints a small console preview for a quick eyeball check.

Examples
--------
# Minimal: pick a few edges and days
python src/analyze_edge_stats.py \
  --input results/output/step6_final_result/cbdb17d4/edge_stats_full.geojson \
  --edge_uids 463865739,463866794 \
  --dates 2024-04-13,2024-04-15 \
  --outdir results/output/step6_final_result/cbdb17d4/qa

# Date range and sampling by edges
python src/analyze_edge_stats.py \
  --input results/output/step6_final_result/cbdb17d4/edge_stats_full.geojson \
  --start_date 2024-04-01 --end_date 2024-04-30 \
  --sample_n 3 --outdir results/output/step6_final_result/cbdb17d4/qa
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to edge_stats GeoJSON")
    p.add_argument("--edge_uids", default="", help="Comma-separated edge_uids to include")
    p.add_argument("--dates", default="", help="Comma-separated dates YYYY-MM-DD to include")
    p.add_argument("--start_date", default="", help="Inclusive start date YYYY-MM-DD")
    p.add_argument("--end_date", default="", help="Inclusive end date YYYY-MM-DD")
    p.add_argument("--sample_n", type=int, default=0, help="If >0, sample this many unique edges after filtering")
    p.add_argument("--outdir", default="qa_outputs", help="Directory to write CSV summaries")
    p.add_argument("--plot", action="store_true", help="If set, generate PNG time-series plots per edge")
    p.add_argument("--plot_n", type=int, default=6, help="Max number of edges to plot (after filtering)")
    p.add_argument("--dpi", type=int, default=120, help="DPI for saved PNG plots")
    return p.parse_args()


SHOW_COLS_BASE = [
    "edge_uid", "timestamp",
    "combined_shade", "combined_shadow_fraction",
    "combined_shade_buffer10", "combined_shadow_fraction_buffer10",
]


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _filter_dates(df: pd.DataFrame, dates_list, start_date, end_date) -> pd.DataFrame:
    if dates_list:
        days = [pd.to_datetime(d).date() for d in dates_list]
        return df[df["timestamp"].dt.date.isin(days)]
    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return df


def _monotonic_non_decreasing(series: pd.Series) -> bool:
    v = series.dropna().values
    return bool(np.all(v[1:] >= v[:-1])) if len(v) > 1 else True


def per_edge_day_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    agg = df.groupby(["edge_uid", "date"]).agg(
        hours=("timestamp", lambda s: s.dt.hour.nunique()),
        shade_mean=("combined_shade", "mean"),
        shade_median=("combined_shade", "median"),
        shade_min=("combined_shade", "min"),
        shade_max=("combined_shade", "max"),
        shade_b10_mean=("combined_shade_buffer10", "mean"),
        frac_mean=("combined_shadow_fraction", "mean"),
        frac_median=("combined_shadow_fraction", "median"),
        frac_min=("combined_shadow_fraction", "min"),
        frac_max=("combined_shadow_fraction", "max"),
        frac_b10_mean=("combined_shadow_fraction_buffer10", "mean"),
        corr_shade_frac=("combined_shade", lambda x: pd.Series(x).corr(df.loc[x.index, "combined_shadow_fraction"]))
    ).reset_index()
    return agg


def per_edge_day_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    rows = []
    for (edge, day), grp in df.sort_values("timestamp").groupby(["edge_uid", "date"], sort=False):
        monotonic = _monotonic_non_decreasing(grp["combined_shadow_fraction"]) if "combined_shadow_fraction" in grp else np.nan
        rows.append({
            "edge_uid": edge,
            "date": day,
            "hours": grp["timestamp"].dt.hour.nunique(),
            "monotonic_shadow_fraction": monotonic,
            "first_val_sf": float(grp["combined_shadow_fraction"].dropna().iloc[0]) if grp["combined_shadow_fraction"].notna().any() else np.nan,
            "last_val_sf": float(grp["combined_shadow_fraction"].dropna().iloc[-1]) if grp["combined_shadow_fraction"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def _plot_edge_timeseries(df: pd.DataFrame, edge_uid: str, outdir: Path, dpi: int = 120) -> Path:
    """Create a simple line plot of shade/fraction over time for a single edge.
    Saves a PNG and returns its path.
    """
    sub = df[df["edge_uid"].astype(str) == str(edge_uid)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    # Core series
    if "combined_shade" in sub.columns:
        ax.plot(sub["timestamp"], sub["combined_shade"], marker="o", linewidth=1.5, label="combined_shade")
    if "combined_shadow_fraction" in sub.columns:
        ax.plot(sub["timestamp"], sub["combined_shadow_fraction"], marker="o", linewidth=1.5, label="shadow_fraction")

    # Optional buffer10 overlays (dashed)
    if "combined_shade_buffer10" in sub.columns:
        ax.plot(sub["timestamp"], sub["combined_shade_buffer10"], linestyle="--", linewidth=1.0, label="combined_shade_b10")
    if "combined_shadow_fraction_buffer10" in sub.columns:
        ax.plot(sub["timestamp"], sub["combined_shadow_fraction_buffer10"], linestyle="--", linewidth=1.0, label="shadow_fraction_b10")

    ax.set_title(f"Edge {edge_uid} – time series")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.legend(loc="best", frameon=False)

    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    out_path = outdir / f"edge_{edge_uid}_timeseries.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.input}")
    gdf = gpd.read_file(args.input)
    print(f" rows={len(gdf):,} cols={len(gdf.columns)}")

    # Basic column checks (allow missing *_buffer10 or shadow_fraction columns)
    for c in ["edge_uid", "timestamp", "combined_shade"]:
        if c not in gdf.columns:
            raise SystemExit(f"Missing required column '{c}' in input GeoJSON")

    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    df = _ensure_timestamp(df)

    # Filter by edges
    if args.edge_uids:
        targets = [s.strip() for s in args.edge_uids.split(",") if s.strip()]
        df = df[df["edge_uid"].astype(str).isin(targets)]
        print(f"[filter] edges: kept {df['edge_uid'].nunique()} unique edges")

    # Filter by dates
    dates_list = [s.strip() for s in args.dates.split(",") if s.strip()]
    df = _filter_dates(df, dates_list, args.start_date, args.end_date)
    if len(dates_list) or args.start_date or args.end_date:
        print(f"[filter] rows after date filters: {len(df):,}")

    # Optional sampling by unique edges (after other filters)
    if args.sample_n and args.sample_n > 0:
        unique_edges = df["edge_uid"].unique()
        k = min(args.sample_n, len(unique_edges))
        rng = np.random.default_rng(53)
        sampled = set(rng.choice(unique_edges, size=k, replace=False))
        df = df[df["edge_uid"].isin(sampled)]
        print(f"[filter] sampled {k} edge(s)")

    if df.empty:
        print("[warn] no rows after filtering; exiting")
        return

    # 1) Per-edge/day summary stats
    day_sum = per_edge_day_summary(df)
    day_sum_csv = outdir / "per_edge_day_summary.csv"
    day_sum.to_csv(day_sum_csv, index=False)
    print(f"[out] per-edge/day summary → {day_sum_csv}")

    # 2) Monotonicity check of shadow_fraction per day per edge
    if "combined_shadow_fraction" in df.columns:
        mono = per_edge_day_monotonicity(df)
        mono_csv = outdir / "per_edge_day_shadow_fraction_monotonicity.csv"
        mono.to_csv(mono_csv, index=False)
        print(f"[out] monotonicity report → {mono_csv}")

    # 3) Coverage checks: days with <24 hours per edge
    df["date"] = df["timestamp"].dt.date
    coverage = df.groupby(["edge_uid", "date"]).agg(hours=("timestamp", lambda s: s.dt.hour.nunique())).reset_index()
    few_hours = coverage[coverage["hours"] < 24]
    few_hours_csv = outdir / "days_with_lt24_hours.csv"
    few_hours.to_csv(few_hours_csv, index=False)
    print(f"[out] days with <24 hours → {few_hours_csv}")

    # 4) Per-edge correlations across the whole (filtered) set
    if set(["combined_shade", "combined_shadow_fraction"]).issubset(df.columns):
        corr_rows = []
        for edge, grp in df.groupby("edge_uid"):
            valid = grp[["combined_shade", "combined_shadow_fraction"]].dropna()
            corr = valid["combined_shade"].corr(valid["combined_shadow_fraction"]) if len(valid) >= 3 else np.nan
            corr_rows.append({"edge_uid": edge, "corr_shade_vs_fraction": corr, "n": len(valid)})
        corr_df = pd.DataFrame(corr_rows)
        corr_csv = outdir / "per_edge_correlation.csv"
        corr_df.to_csv(corr_csv, index=False)
        print(f"[out] per-edge correlation → {corr_csv}")

    # 5) Console preview of a few rows
    show_cols = [c for c in SHOW_COLS_BASE if c in df.columns]
    print("\n===== SAMPLE ROWS (first 20) =====")
    print(df.sort_values(["edge_uid", "timestamp"]).head(20)[show_cols].to_string(index=False))

    # 6) Per-edge time series CSVs
    for edge, grp in df.groupby("edge_uid"):
        out_csv = outdir / f"edge_{edge}_timeseries.csv"
        grp.sort_values("timestamp")[show_cols].to_csv(out_csv, index=False)
    print(f"[out] per-edge timeseries CSVs → {outdir}/edge_*_timeseries.csv")

    # 7) Optional plotting
    if args.plot:
        edges_to_plot = list(df["edge_uid"].astype(str).unique())[: max(0, args.plot_n)]
        if not edges_to_plot:
            print("[plot] no edges to plot after filtering")
        else:
            print(f"[plot] generating plots for up to {len(edges_to_plot)} edge(s)…")
            for e in edges_to_plot:
                png = _plot_edge_timeseries(df, e, outdir, dpi=args.dpi)
                if png is not None:
                    print(f"[plot] saved {png}")


if __name__ == "__main__":
    main()