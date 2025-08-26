import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
from pathlib import Path
from rasterio.io import MemoryFile
import rasterio
from rasterio.merge import merge as rio_merge
from datetime import datetime, timedelta
import argparse
import yaml
import os

from glob import glob

def _list_available_times(tile_folder: Path, osmid: str, tile_id: str, date_str: str, base_name: str) -> list:
    pattern1 = str(tile_folder / f"{osmid}_{tile_id}_{base_name}_{date_str}_*.tif")
    files = glob(pattern1)
    times = []
    for fp in files:
        name = os.path.basename(fp)
        # expected like: {osmid}_{tile_id}_{base}_{YYYYMMDD}_{HHMM}[...].tif
        parts = name.split("_")
        # find the part immediately after date_str
        for i, p in enumerate(parts):
            if p == date_str and i + 1 < len(parts):
                # strip extension and trailing suffixes
                time_part = parts[i+1]
                time_part = time_part.replace(".tif", "")
                times.append(time_part)
                break
    return sorted(set(times))

def _print_folder_snapshot(base: Path, shade_type: str, tile_id: str, date_str: str, osmid: str, base_name: str):
    folder_num = tile_id.split('_')[-1] if '_' in tile_id else tile_id
    folders = [base / shade_type / folder_num, base / shade_type / tile_id]
    print(f"[debug] Shade base: {base}")
    print(f"[debug] Shade type: {shade_type}")
    for f in folders:
        print(f"[debug] Checking folder: {f} exists={f.exists()}")
        if f.exists():
            try:
                entries = list(f.iterdir())
                print(f"[debug]  files/dirs: {len(entries)} (showing up to 5)")
                for e in entries[:5]:
                    print(f"         - {e.name}")
                times = _list_available_times(f, osmid, tile_id, date_str, base_name)
                if times:
                    print(f"[debug]  available times for date {date_str}: {times[:10]}{' ...' if len(times)>10 else ''}")
                else:
                    print(f"[debug]  no files matching pattern for date {date_str}")
            except Exception as ex:
                print(f"[warn] could not list folder {f}: {ex}")

def _candidate_filenames(osmid, tile_id, date_str, time_str, base_name="Shadow"):
    # Match final_corrected_extraction pattern: ..._{date}_{time}_LST.tif (and without _LST)
    return [
        f"{osmid}_{tile_id}_{base_name}_{date_str}_{time_str}_LST.tif",
        f"{osmid}_{tile_id}_{base_name}_{date_str}_{time_str}.tif",
        # A couple of safe variants just in case
        f"{osmid}_{tile_id}_{base_name.capitalize()}_{date_str}_{time_str}_LST.tif",
        f"{osmid}_{tile_id}_{base_name.capitalize()}_{date_str}_{time_str}.tif",
    ]

def _find_single_raster(base, shade_type, tile_id, date_str, time_str, osmid, base_name="Shadow"):
    folder_num = tile_id.split('_')[-1] if '_' in tile_id else tile_id
    candidates = [base / shade_type / folder_num, base / shade_type / tile_id]
    for tile_folder in candidates:
        if not tile_folder.exists():
            continue
        for name in _candidate_filenames(osmid, tile_id, date_str, time_str, base_name):
            p = tile_folder / name
            if p.exists():
                return p
    # Not found – print a concise troubleshooting snapshot for the first folder
    # (avoid spamming inside tight loops by keeping it short)
    print(f"[miss:file] {shade_type} :: tile={tile_id} date={date_str} time={time_str} → no exact file found")
    _print_folder_snapshot(base, shade_type, tile_id, date_str, osmid, base_name)
    return None

def find_raster_path(base, shade_type, tile, timestamp, osmid, suffix):
    ts_str = timestamp.strftime('%Y%m%d_%H%M')
    path = base / shade_type / tile / f"{osmid}_{tile}_{suffix}_{ts_str}.tif"
    return path if path.exists() else None

def find_hours_before_rasters(base, shade_type, tile, timestamp, osmid, hours):
    paths = []
    for hr in hours:
        ts_past = timestamp - timedelta(hours=hr)
        ts_str = ts_past.strftime('%Y%m%d_%H%M')
        path = base / shade_type / tile / f"{osmid}_{tile}_Shadow_{ts_str}_LST.tif"
        if path.exists():
            paths.append((hr, path))
    return paths

def zonal(gdf, raster, buffer_m):
    if buffer_m > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(buffer_m)
    stats = zonal_stats(gdf, raster, stats="mean", nodata=np.nan)
    return stats[0]["mean"] if stats and stats[0]["mean"] is not None else np.nan

def zonal_from_array(gdf, array, transform, nodata_val, buffer_m):
    if buffer_m > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(buffer_m)
    stats = zonal_stats(
        gdf,
        array,
        affine=transform,
        stats="mean",
        nodata=nodata_val if nodata_val is not None else np.nan,
    )
    return stats[0]["mean"] if stats and stats[0]["mean"] is not None else np.nan

def mosaic_rasters(base, shade_type, tile_ids, date_str, time_str, osmid, base_name="Shadow"):
    """Return (array, transform, nodata, crs) mosaic for given tiles at date+time.
    Returns (None, None, None, None) if no rasters found.
    """
    sources = []
    for t in tile_ids:
        rp = _find_single_raster(base, shade_type, t, date_str, time_str, osmid, base_name)
        if rp is None:
            continue
        src = rasterio.open(rp)
        sources.append(src)
    if not sources:
        return None, None, None, None
    if len(sources) == 1:
        with sources[0] as s:
            arr = s.read(1)
            return arr, s.transform, s.nodata, s.crs
    try:
        mosaic, transform = rio_merge(sources)
        return mosaic[0], transform, sources[0].nodata, sources[0].crs
    finally:
        for s in sources:
            s.close()

def main(points_path, edges_path, output_path, config_path, osmid, buffers, include_building, sample_n, sample_seed, sample_flag):
    # Load data
    points = gpd.read_file(points_path)
    edges = gpd.read_file(edges_path)
    points["rounded_timestamp"] = pd.to_datetime(points["rounded_timestamp"])
    points["binned_date"] = pd.to_datetime(points["binned_date"]).dt.strftime("%Y%m%d")
    edges["edge_uid"] = edges["edge_uid"].astype(str)
    points["edge_uid"] = points["edge_uid"].astype(str)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base = Path(config["output_dir"]) / "step5_shade_results" / osmid
    print(f"[env] base shade path: {base} exists={base.exists()}")
    if base.exists():
        try:
            shade_types = [p.name for p in base.iterdir() if p.is_dir()]
            print(f"[env] shade types under base: {shade_types}")
            if "combined_shade" not in shade_types:
                print("[warn] 'combined_shade' folder not found under base – check config/output_dir or osmid")
        except Exception as ex:
            print(f"[warn] could not list base: {ex}")

    hours_before = config["extra_outputs"]["hours_before"]

    results = []

    # Build mapping: for each edge_uid & rounded_timestamp & binned_date → list of tiles (may be 1+)
    pts_subset = points[["edge_uid", "rounded_timestamp", "binned_date", "tile_number"]].drop_duplicates()
    tiles_by_edge_time = (
        pts_subset.groupby(["edge_uid", "rounded_timestamp", "binned_date"])['tile_number']
                  .apply(lambda s: sorted(set(s.tolist())))
                  .reset_index()
    )

    # Optional sampling for quick runs
    if sample_flag and (not sample_n or sample_n <= 0):
        sample_n = 50  # sensible default
    if sample_n and sample_n > 0:
        n_before = len(tiles_by_edge_time)
        tiles_by_edge_time = tiles_by_edge_time.sample(n=min(sample_n, n_before), random_state=sample_seed)
        print(f"Sampling enabled: {len(tiles_by_edge_time)}/{n_before} (edge_uid, time) pairs will be processed (seed={sample_seed}).")

    if len(tiles_by_edge_time) > 0:
        t0 = tiles_by_edge_time.iloc[0]
        print(f"Preview → edge: {t0['edge_uid']}, date: {t0['binned_date']}, time: {t0['rounded_timestamp'].strftime('%H%M')}, tiles: {t0['tile_number']}")

    pairs_total = len(tiles_by_edge_time)
    pairs_found = 0
    pairs_missed = 0

    for edge_uid, timestamp, binned_date, tile_list in tiles_by_edge_time.itertuples(index=False):
        # Prepare geometry row for this edge
        edge_geom = edges[edges["edge_uid"] == edge_uid]
        if edge_geom.empty:
            continue

        row = {"edge_uid": edge_uid, "timestamp": timestamp}

        time_str = timestamp.strftime('%H%M')
        arr, transform, nodata_val, raster_crs = mosaic_rasters(base, "combined_shade", tile_list, binned_date, time_str, osmid, base_name="Shadow")
        # If you also need shadow fraction later, keep this attempt but it may not exist in all runs
        arr_frac, transform_frac, nodata_frac, raster_crs_frac = mosaic_rasters(base, "combined_shade", tile_list, binned_date, time_str, osmid, base_name="shadow_fraction_on")

        if arr is None:
            print(f"[miss] No combined_shade for edge={edge_uid} date={binned_date} time={time_str} tiles={tile_list}")
            pairs_missed += 1
        else:
            pairs_found += 1

        for buffer in buffers:
            suffix = f"_buffer{int(buffer)}" if buffer else ""
            if arr is not None:
                eg = edge_geom
                if raster_crs is not None:
                    try:
                        if eg.crs is None or eg.crs != raster_crs:
                            eg = eg.to_crs(raster_crs)
                    except Exception as ex:
                        print(f"[warn] CRS reprojection failed for edge {edge_uid}: {ex}")
                row[f"combined_shade{suffix}"] = zonal_from_array(eg, arr, transform, nodata_val, buffer)
            if arr_frac is not None:
                egf = edge_geom
                if raster_crs_frac is not None:
                    try:
                        if egf.crs is None or egf.crs != raster_crs_frac:
                            egf = egf.to_crs(raster_crs_frac)
                    except Exception as ex:
                        print(f"[warn] CRS reprojection failed (frac) for edge {edge_uid}: {ex}")
                row[f"combined_shadow_fraction{suffix}"] = zonal_from_array(egf, arr_frac, transform_frac, nodata_frac, buffer)

        results.append(row)

    # Merge into edges
    df = pd.DataFrame(results)
    merged = edges.merge(df, on="edge_uid", how="left")
    merged.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved: {output_path}")

    # ===== Summary statistics and sample preview =====
    value_cols = [c for c in merged.columns if c.startswith("combined_shade") or c.startswith("combined_shadow_fraction")]
    print("\n===== RUN SUMMARY =====")
    print(f"Pairs total:  {pairs_total}")
    print(f"Pairs found:  {pairs_found}")
    print(f"Pairs missed: {pairs_missed}")
    for col in value_cols:
        ser = merged[col].dropna()
        if len(ser) == 0:
            print(f"{col}: no values")
        else:
            print(f"{col}: n={len(ser)} min={ser.min():.4f} p25={ser.quantile(0.25):.4f} median={ser.median():.4f} mean={ser.mean():.4f} p75={ser.quantile(0.75):.4f} max={ser.max():.4f}")

    # Print a compact preview of a few rows (selected columns only)
    preview_cols = ["edge_uid", "timestamp"] + value_cols
    preview = merged[preview_cols].head(12)
    try:
        print("\n===== PREVIEW (first 12 rows) =====")
        print(preview.to_string(index=False))
    except Exception:
        print("[warn] could not print preview table")

    # Also write a CSV preview next to the output
    out_path = Path(output_path)
    sample_csv = out_path.with_suffix("")
    sample_csv = sample_csv.parent / (sample_csv.name + ".sample.csv")
    try:
        preview.to_csv(sample_csv, index=False)
        print(f"Saved preview CSV: {sample_csv}")
    except Exception as ex:
        print(f"[warn] could not write preview CSV: {ex}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--osmid", required=True)
    parser.add_argument("--buffers", nargs="+", type=float, default=[0])
    parser.add_argument("--include_building", action="store_true")
    parser.add_argument("--sample_n", type=int, default=0, help="Process only N (edge_uid, time) pairs for a quick test. 0=all.")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--sample", action="store_true", help="Enable sampling mode. If --sample_n is 0, defaults to 50 pairs.")
    args = parser.parse_args()

    main(
        args.points,
        args.edges,
        args.output,
        args.config,
        args.osmid,
        args.buffers,
        args.include_building,
        args.sample_n,
        args.sample_seed,
        args.sample,
    )


# python aggregate_edge_shade_stats.py \
#   --points results/output/step6_final_result/cbdb17d4/binned_dataset_2024.geojson \
#   --edges  data/clean_data/strava/back_bay_edges_aoi.geojson \
#   --output results/output/step6_final_result/cbdb17d4/edge_stats_sample.geojson \
#   --config config.yaml \
#   --osmid cbdb17d4 \
#   --buffers 0 10 \
#   --sample

# nohup python src/aggregate_edge_shade_stats.py \
#   --points results/output/step6_final_result/cbdb17d4/binned_dataset_2024.geojson \
#   --edges  data/clean_data/strava/back_bay_edges_aoi.geojson \
#   --output results/output/step6_final_result/cbdb17d4/edge_stats_full.geojson \
#   --config config.yaml \
#   --osmid cbdb17d4 \
#   --buffers 0 10 \
#   > logs/aggregate_full.log 2>&1 &