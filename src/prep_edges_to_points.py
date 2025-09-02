import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

def densify_lines_to_points(gdf_lines, spacing_m=10, edge_id_col="edge_uid"):
    # Work in metric CRS for distances
    gdf_m = gdf_lines.to_crs(3857)
    out_rows = []

    for _, row in gdf_m.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        length = geom.length
        if length == 0:
            continue

        edge_uid = row[edge_id_col]
        dists = np.arange(0, length + spacing_m, spacing_m)
        pts_m = [geom.interpolate(float(d)) for d in dists]
        pts = gpd.GeoSeries(pts_m, crs=3857).to_crs(4326)

        for i, p in enumerate(pts):
            out_rows.append({
                "edge_uid": edge_uid,
                "sample_idx": i,   
                "geometry": p
            })

    gdf_pts = gpd.GeoDataFrame(out_rows, geometry="geometry", crs="EPSG:4326")
    gdf_pts["sample_id"] = gdf_pts["edge_uid"].astype(str) + "_" + gdf_pts["sample_idx"].astype(str)
    gdf_pts["latitude"]  = gdf_pts.geometry.y
    gdf_pts["longitude"] = gdf_pts.geometry.x
    return gdf_pts[["sample_id", "edge_uid", "sample_idx", "latitude", "longitude", "geometry"]]

def load_hours_csvs(hour_csv_paths):
    # Expect columns: edge_uid, hour, osm_reference_id
    frames = []
    for p in hour_csv_paths:
        df = pd.read_csv(p, dtype={"edge_uid": str, "osm_reference_id": str})
        # Parse YYYY-MM-DDTHH -> keep as **naive local** timestamps (matches LST filenames)
        df["time"] = pd.to_datetime(df["hour"], format="%Y-%m-%dT%H", errors="coerce")
        frames.append(df[["edge_uid", "time", "osm_reference_id"]])
    hours = pd.concat(frames, ignore_index=True).dropna(subset=["time"])
    # normalize types
    hours["edge_uid"] = hours["edge_uid"].astype(str)
    hours["osm_reference_id"] = hours["osm_reference_id"].astype(str)

    # Debug: file-by-file counts and time spans
    print("\n[load_hours_csvs] Input CSV diagnostics:")
    for p in hour_csv_paths:
        df = pd.read_csv(p, dtype={"edge_uid": str, "osm_reference_id": str})
        df["time"] = pd.to_datetime(df["hour"], format="%Y-%m-%dT%H", errors="coerce")
        print(f"  {Path(p).name}: rows={len(df):,}, unique times={df['time'].nunique():,}, range=({df['time'].min()} -> {df['time'].max()})")

    print(f'Len hours df before drop_duplicates: {len(hours)}')
    return hours.drop_duplicates()

def analyze_hours(hours: pd.DataFrame, label="hours"):
    """Print rich diagnostics about hourly coverage.
    Expects columns: edge_uid (str), time (datetime64[ns]), osm_reference_id (str).
    """
    print("\n===== ANALYZE:", label, "=====")
    if hours.empty:
        print("[analyze_hours] Provided dataframe is EMPTY")
        return

    # Basic uniques
    n_rows = len(hours)
    n_times = hours["time"].nunique(dropna=True)
    n_edges = hours["edge_uid"].nunique(dropna=True)
    n_osm = hours["osm_reference_id"].nunique(dropna=True)
    tmin, tmax = hours["time"].min(), hours["time"].max()
    print(f"rows: {n_rows:,} | unique times: {n_times:,} | unique edges: {n_edges:,} | unique osm_reference_id: {n_osm:,}")
    print(f"time range: {tmin} -> {tmax}  (span = {tmax - tmin})")

    # Per-month coverage (assumes full-month expectation if any hours occur in that month)
    hours["month"] = hours["time"].dt.to_period("M")
    month_groups = hours.groupby("month", observed=True)
    print("\nPer-month coverage (actual vs expected hours = 24 * days_in_month):")
    for m, grp in month_groups:
        month_start = grp["time"].min().to_period("M").to_timestamp()
        days_in_month = month_start.days_in_month
        expected = 24 * days_in_month
        actual = grp['time'].nunique()
        pct = (actual / expected) * 100 if expected else 0
        print(f"  {m}: actual={actual:,} | expected={expected:,} | coverage={pct:5.1f}% | unique times={grp['time'].nunique():,}")

    # Per-day completeness: list dates with < 24 hours
    hours["date"] = hours["time"].dt.date
    per_day_counts = hours.groupby("date")["time"].nunique()
    incomplete_days = per_day_counts[per_day_counts < 24].sort_index()
    print(f"\nTotal days seen: {len(per_day_counts):,}; days with <24 hours: {len(incomplete_days):,}")
    if not incomplete_days.empty:
        print("Example of first 5 incomplete days and their hour counts:")
        print(incomplete_days.head(5).to_string())
        # For the first incomplete day, show which hours are missing
        first_bad_day = incomplete_days.index[0]
        day_mask = hours["date"] == first_bad_day
        present_hours = sorted(hours.loc[day_mask, "time"].dt.hour.unique())
        missing = [h for h in range(24) if h not in present_hours]
        print(f"Missing hour-of-day values for {first_bad_day}: {missing}")

    # Hour-of-day distribution (0..23)
    hod = hours["time"].dt.hour.value_counts().sort_index()
    print("\nHour-of-day distribution (counts across entire dataset):")
    print(hod.to_string())

    # Quick duplicate check (edge_uid, time)
    dupe_key = hours.duplicated(subset=["edge_uid", "time"], keep=False).sum()
    print(f"\nPotential duplicates on (edge_uid, time): {dupe_key:,}")

    # Show a small sample for manual inspection
    print("\nSample rows:")
    print(hours.sample(min(5, len(hours)), random_state=42).sort_values("time").to_string(index=False))
    print("===== END ANALYZE =====\n")

def main():
    # ---- EDIT THESE INPUTS ----
    edges_path = "../data/clean_data/strava/data_to_share/all_edges_hourly_2024-04-01-2024-04-30_ped_boston_filtered/53f3ef7e32738022bd45a2ed224bcbcf8cb63f07dc3d5bca7e9221dfe7fe4451-1738340934480.shp"            # LineStrings with edgeUID
    back_bay_poly = "../data/raw_data/bos/back_bay.json"            # Optional AOI
    hour_csvs = [
        "../data/clean_data/strava/data_to_share/all_edges_hourly_2024-04-01-2024-04-30_ped_boston_filtered/53f3ef7e32738022bd45a2ed224bcbcf8cb63f07dc3d5bca7e9221dfe7fe4451-1738340934480.csv",
        "../data/clean_data/strava/data_to_share/all_edges_hourly_2024-06-01-2024-06-30_ped_boston_filtered/2496bcdb5fd9d0ae571ecbe8bc16ecbbad2ebc9cab61ae94449698ef39e41cf4-1738340261837.csv",
        "../data/clean_data/strava/data_to_share/all_edges_hourly_2024-10-01-2024-10-31_ped_boston_filtered/5ea6147519dc3a820312867430fec9067a1eb42846d1ac03f5446f461b85c5b0-1738340998655.csv",
        "../data/clean_data/strava/data_to_share/all_edges_hourly_2024-12-01-2024-12-31_ped_boston_filtered/66feca840e0c280c5bd400436507918bbd27a19acd14e99d5bc00d0b4b144e9e-1738340241545.csv",
    ]
    spacing_m = 10
    prepared_out = Path("../data/clean_data/strava/back_bay_points_hours_all_months.parquet")

    # 1) Load edges, clip to Back Bay
    edges = gpd.read_file(edges_path)
    print(f"Loaded edges: {len(edges):,} features; CRS={edges.crs}")
    # Check for edgeUID column and rename it to edge_uid for consistency with CSV files
    if "edgeUID" not in edges.columns:
        raise ValueError("edges file must contain an 'edgeUID' column")
    edges = edges.rename(columns={"edgeUID": "edge_uid"})
    print("Edge columns:", list(edges.columns))
    
    poly = gpd.read_file(back_bay_poly)
    if poly.crs != edges.crs:
        poly = poly.to_crs(edges.crs)
    edges_aoi = gpd.clip(edges, poly.union_all())
    print(f"AOI polygon crs={poly.crs}; edges in AOI: {len(edges_aoi):,}")

    # Export AOI edges for further analysis
    edges_out = Path("../data/clean_data/strava/back_bay_edges_aoi.geojson")
    edges_aoi.to_file(edges_out, driver="GeoJSON")
    print(f"Exported AOI edges to {edges_out} ({len(edges_aoi):,} features)")

    # 2) Load hours (per edge)
    hours = load_hours_csvs(hour_csvs)
    print(f'Length hours df after dropping duplicated {len(hours)}')
    analyze_hours(hours, label="raw hours (dedup pending)")
    analyze_hours(hours, label="hours AFTER drop_duplicates")

    # 3) Keep only edges that appear in hours
    edges_aoi["edge_uid"] = edges_aoi["edge_uid"].astype(str)
    target_edges = edges_aoi.merge(hours[["edge_uid"]].drop_duplicates(), on="edge_uid", how="inner")
    print(f'After merging hours with AOI (Back Bay) edges: {len(target_edges)}')

    # Diagnostics & exports: AOI edges with vs without hours
    with_hours = target_edges.copy()
    without_hours = edges_aoi[~edges_aoi['edge_uid'].isin(with_hours['edge_uid'])].copy()

    print(f"AOI edges total: {len(edges_aoi):,}")
    print(f"AOI edges WITH hours: {len(with_hours):,}")
    print(f"AOI edges WITHOUT hours: {len(without_hours):,}")

    # Quick sample of missing edge_uids for QA
    if len(without_hours) > 0:
        print("Sample of AOI edges without hours (up to 10 edge_uids):")
        print(without_hours['edge_uid'].astype(str).head(10).to_list())

    # Export for inspection
    edges_with_hours_out = Path("../data/clean_data/strava/back_bay_edges_with_hours.geojson")
    edges_without_hours_out = Path("../data/clean_data/strava/back_bay_edges_no_hours.geojson")

    with_hours.to_file(edges_with_hours_out, driver="GeoJSON")
    print(f"Exported AOI edges WITH hours to {edges_with_hours_out} ({len(with_hours):,} features)")

    if len(without_hours) > 0:
        without_hours.to_file(edges_without_hours_out, driver="GeoJSON")
        print(f"Exported AOI edges WITHOUT hours to {edges_without_hours_out} ({len(without_hours):,} features)")
    else:
        print("No AOI edges without hours; skipping export.")

    # Build the expected (edge_uid, time) pairs for AOI edges
    aoi_edge_uids = set(target_edges['edge_uid'].astype(str).unique())
    expected_pairs = (
        hours.loc[hours['edge_uid'].astype(str).isin(aoi_edge_uids), ['edge_uid', 'time']]
             .drop_duplicates()
             .sort_values(['edge_uid', 'time'])
             .reset_index(drop=True)
    )
    print(f"Expected (edge_uid, time) pairs within AOI: {len(expected_pairs):,}")

    # How many hours per edge on average (post-AOI merge)?
    hrs_per_edge = hours.groupby('edge_uid')['time'].nunique()
    merged_counts = hrs_per_edge.reindex(target_edges['edge_uid']).dropna()
    if not merged_counts.empty:
        print(f"Per-edge unique hour count (sample): min={merged_counts.min()}, median={merged_counts.median()}, max={merged_counts.max()}")
    else:
        print("No per-edge hour stats available (empty merge).")

    # 4) Densify those edges to points
    pts = densify_lines_to_points(target_edges, spacing_m=spacing_m, edge_id_col="edge_uid")
    print(f'Points created: {len(pts)}')
    if len(target_edges) > 0:
        print(f"Avg points per edge: {len(pts) / len(target_edges):.2f}")

    # 5) Join points to their **own** hours (per edge), carrying osm_reference_id
    #    (small, per-edge cartesian product)
    pts["edge_uid"] = pts["edge_uid"].astype(str)
    hours["edge_uid"] = hours["edge_uid"].astype(str)
    df = pts.merge(hours, on="edge_uid", how="inner")
    print(f'Length of dataframe after merging hours to points: {len(df)}')

    # === VERIFY: all times per edge_uid in AOI are present in df ===
    observed_pairs = (
        df[['edge_uid', 'time']]
          .drop_duplicates()
          .sort_values(['edge_uid', 'time'])
          .reset_index(drop=True)
    )
    print(f"Observed unique (edge_uid, time) pairs in df: {len(observed_pairs):,}")

    # Missing pairs: expected but not found in df
    missing_pairs = expected_pairs.merge(observed_pairs, on=['edge_uid', 'time'], how='left', indicator=True)
    missing_pairs = missing_pairs[missing_pairs['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Extra pairs: in df but not expected (should be none, but good to see)
    extra_pairs = observed_pairs.merge(expected_pairs, on=['edge_uid', 'time'], how='left', indicator=True)
    extra_pairs = extra_pairs[extra_pairs['_merge'] == 'left_only'].drop(columns=['_merge'])

    print(f"Missing (edge_uid, time) pairs: {len(missing_pairs):,}")
    print(f"Extra (edge_uid, time) pairs:    {len(extra_pairs):,}")

    # Per-edge comparison of hour counts
    exp_counts = expected_pairs.groupby('edge_uid').size().rename('expected_hours')
    obs_counts = observed_pairs.groupby('edge_uid').size().rename('observed_hours')
    per_edge_check = (
        pd.concat([exp_counts, obs_counts], axis=1)
          .fillna(0)
          .astype(int)
          .reset_index()
    )
    per_edge_mismatch = per_edge_check[per_edge_check['expected_hours'] != per_edge_check['observed_hours']]
    print(f"Edges with hour-count mismatch: {len(per_edge_mismatch):,} / {per_edge_check.shape[0]:,}")

    if len(missing_pairs) > 0:
        print("\nSample missing pairs (up to 10):")
        print(missing_pairs.head(10).to_string(index=False))
        # Show which hours are missing for first problematic edge
        first_edge = missing_pairs['edge_uid'].iloc[0]
        e_expected = expected_pairs.loc[expected_pairs['edge_uid'] == first_edge, 'time'].sort_values()
        e_observed = observed_pairs.loc[observed_pairs['edge_uid'] == first_edge, 'time'].sort_values()
        missing_times = e_expected[~e_expected.isin(e_observed)]
        print(f"\nEdge {first_edge} missing {len(missing_times)} times. First 12:")
        print(missing_times.head(12).to_list())

    if len(extra_pairs) > 0:
        print("\nSample extra pairs (up to 10):")
        print(extra_pairs.head(10).to_string(index=False))

    if len(per_edge_mismatch) > 0:
        print("\nPer-edge mismatches (up to 10):")
        print(per_edge_mismatch.head(10).to_string(index=False))

    # Sanity check: for a random edge, list a few times
    if not df.empty:
        sample_edge = df['edge_uid'].iloc[0]
        sample_times = (df.loc[df['edge_uid'] == sample_edge, 'time']
                          .drop_duplicates()
                          .sort_values()
                          .head(10)
                          .tolist())
        print(f"Example times for edge {sample_edge}: {sample_times}")

    # 6) Save Parquet for the pipeline (columns match your config expectations)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    prepared_out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(prepared_out)
    print(f"Saved {len(gdf):,} rows to {prepared_out}")

if __name__ == "__main__":
    main()
