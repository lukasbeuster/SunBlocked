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
import multiprocessing as mp
from functools import partial
import time

from glob import glob

def _candidate_filenames(osmid, tile_id, date_str, time_str, base_name="Shadow"):
    if base_name == "shadow_fraction_on":
        # Special case for shadow fraction files (no _LST suffix)
        return [
            f"{osmid}_{tile_id}_{base_name}_{date_str}_{time_str}.tif",
            f"{osmid}_{tile_id}_{base_name}_{date_str}_{time_str}_LST.tif",  # just in case
        ]
    else:
        # Standard shadow files
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
    return None

def find_hours_before_rasters(base, shade_type, tile_ids, timestamp, binned_date, osmid, hours_before):
    """Find all shade rasters for the specified hours before the timestamp across multiple tiles
    
    Use binned_date for raster lookup, only change the time component
    """
    # Keep the binned_date, only go back in time (hours)
    target_times = []
    for hr in range(int(hours_before) + 1):  # 0, 1, 2, ... hours_before
        target_time = timestamp - timedelta(hours=hr)
        target_times.append(target_time)
    
    found_rasters = []
    for target_time in target_times:
        # Use binned_date for file lookup, target_time only for hour
        time_str = target_time.strftime('%H%M')
        
        # Try to find raster for each tile at this time
        tile_rasters = []
        for tile_id in tile_ids:
            raster_path = _find_single_raster(base, shade_type, tile_id, binned_date, time_str, osmid, "Shadow")
            if raster_path:
                tile_rasters.append(raster_path)
        
        if tile_rasters:  # If we found at least one tile for this time
            found_rasters.append((target_time, tile_rasters))
    
    return found_rasters

def compute_hours_before_shade(base, shade_type, tile_list, timestamp, binned_date, osmid, hours_before, edge_geom):
    """Compute average shade fraction for N hours before the timestamp"""
    raster_times = find_hours_before_rasters(base, shade_type, tile_list, timestamp, binned_date, osmid, hours_before)
    
    if not raster_times:
        return np.nan
    
    shade_values = []
    for target_time, raster_paths in raster_times:
        # Create mosaic for this time point
        sources = []
        for rp in raster_paths:
            try:
                src = rasterio.open(rp)
                sources.append(src)
            except Exception:
                continue
        
        if not sources:
            continue
            
        try:
            if len(sources) == 1:
                with sources[0] as s:
                    arr = s.read(1)
                    transform = s.transform
                    nodata_val = s.nodata
                    raster_crs = s.crs
            else:
                mosaic, transform = rio_merge(sources)
                arr = mosaic[0]
                nodata_val = sources[0].nodata
                raster_crs = sources[0].crs
            
            # Extract value for this time point
            eg = edge_geom.copy()
            if raster_crs is not None and (eg.crs is None or eg.crs != raster_crs):
                try:
                    eg = eg.to_crs(raster_crs)
                except Exception:
                    continue
            
            value = zonal_from_array(eg, arr, transform, nodata_val, 0)  # No buffer for hours_before
            if not np.isnan(value):
                shade_values.append(value)
                
        finally:
            for s in sources:
                s.close()
    
    if shade_values:
        avg_shade = np.mean(shade_values)
        return avg_shade
    else:
        return np.nan

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

def process_edge_batch(args):
    """Process a batch of (edge_uid, timestamp, binned_date, tile_list) tuples"""
    batch, edges_dict, base_path, osmid, buffers, hours_before = args
    
    results = []
    batch_found = 0
    batch_missed = 0
    
    for edge_uid, timestamp, binned_date, tile_list in batch:
        # Get edge geometry
        if edge_uid not in edges_dict:
            continue
        edge_geom = edges_dict[edge_uid]

        row = {"edge_uid": edge_uid, "timestamp": timestamp}
        time_str = timestamp.strftime('%H%M')
        
        # 1. CURRENT SHADE (instantaneous): Shadow files
        arr, transform, nodata_val, raster_crs = mosaic_rasters(base_path, "combined_shade", tile_list, binned_date, time_str, osmid, base_name="Shadow")
        
        # 2. CURRENT SHADOW FRACTION (since dawn): shadow_fraction_on files  
        arr_frac, transform_frac, nodata_frac, raster_crs_frac = mosaic_rasters(base_path, "combined_shade", tile_list, binned_date, time_str, osmid, base_name="shadow_fraction_on")

        if arr is None:
            batch_missed += 1
            # Assume nighttime for all metrics
            for buffer in buffers:
                suffix = f"_buffer{int(buffer)}m" if buffer else ""
                row[f"current_shade{suffix}"] = 1.0
                row[f"shadow_fraction{suffix}"] = 1.0
                
                # Hours before calculations
                for hr in hours_before:
                    row[f"shade_{hr}h_before{suffix}"] = 1.0
        else:
            batch_found += 1
            # Process normally with existing raster data
            for buffer in buffers:
                suffix = f"_buffer{int(buffer)}m" if buffer else ""
                eg = edge_geom
                if raster_crs is not None:
                    try:
                        if eg.crs is None or eg.crs != raster_crs:
                            eg = eg.to_crs(raster_crs)
                    except Exception:
                        pass
                
                # 1. CURRENT SHADE (instantaneous)
                shade_val = zonal_from_array(eg, arr, transform, nodata_val, buffer)
                row[f"current_shade{suffix}"] = round(shade_val, 2) if not np.isnan(shade_val) else np.nan
                
                # 2. CURRENT SHADOW FRACTION (since dawn)
                if arr_frac is not None:
                    egf = edge_geom
                    if raster_crs_frac is not None:
                        try:
                            if egf.crs is None or egf.crs != raster_crs_frac:
                                egf = egf.to_crs(raster_crs_frac)
                        except Exception:
                            pass
                    shadow_val = zonal_from_array(egf, arr_frac, transform_frac, nodata_frac, buffer)
                    row[f"shadow_fraction{suffix}"] = round(shadow_val, 2) if not np.isnan(shadow_val) else np.nan
                else:
                    row[f"shadow_fraction{suffix}"] = np.nan
                
                # 3. HOURS BEFORE CALCULATIONS
                for hr in hours_before:
                    shade_before = compute_hours_before_shade(base_path, "combined_shade", tile_list, timestamp, binned_date, osmid, hr, edge_geom)
                    row[f"shade_{hr}h_before{suffix}"] = round(shade_before, 2) if not np.isnan(shade_before) else np.nan

        results.append(row)
    
    return results, batch_found, batch_missed

def create_batches(tiles_by_edge_time, batch_size):
    """Split data into batches for multiprocessing"""
    batches = []
    for i in range(0, len(tiles_by_edge_time), batch_size):
        batch = []
        for j in range(i, min(i + batch_size, len(tiles_by_edge_time))):
            row = tiles_by_edge_time.iloc[j]
            batch.append((row['edge_uid'], row['rounded_timestamp'], row['binned_date'], row['tile_number']))
        batches.append(batch)
    return batches

def main(points_path, edges_path, output_path, config_path, osmid, buffers, include_building, sample_n, sample_seed, sample_flag, n_processes, batch_size):
    print(f"[mp] Starting with {n_processes} processes, batch_size={batch_size}")
    
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

    # Get hours_before from config
    hours_before = config.get("extra_outputs", {}).get("hours_before", [2, 4])
    print(f"[config] hours_before = {hours_before}")

    # Create edges dictionary for faster lookup
    edges_dict = {}
    for idx, row in edges.iterrows():
        edges_dict[row['edge_uid']] = gpd.GeoDataFrame([row], crs=edges.crs)

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

    print(f"[mp] Processing {len(tiles_by_edge_time):,} (edge_uid, timestamp, binned_date) combinations")
    
    if len(tiles_by_edge_time) > 0:
        t0 = tiles_by_edge_time.iloc[0]
        print(f"Preview → edge: {t0['edge_uid']}, actual_date: {t0['rounded_timestamp'].strftime('%Y-%m-%d %H%M')}, binned_date: {t0['binned_date']}, tiles: {t0['tile_number']}")

    # Create batches for multiprocessing
    batches = create_batches(tiles_by_edge_time, batch_size)
    print(f"[mp] Created {len(batches)} batches of ~{batch_size} items each")
    
    # Prepare arguments for each batch
    batch_args = []
    for batch in batches:
        batch_args.append((batch, edges_dict, base, osmid, buffers, hours_before))

    # Process batches in parallel
    print(f"[mp] Starting multiprocessing with {n_processes} workers...")
    start_time = time.time()
    
    with mp.Pool(processes=n_processes) as pool:
        batch_results = pool.map(process_edge_batch, batch_args)
    
    end_time = time.time()
    print(f"[mp] Multiprocessing completed in {end_time - start_time:.1f} seconds")

    # Combine results from all batches
    all_results = []
    total_found = 0
    total_missed = 0
    
    for batch_result, batch_found, batch_missed in batch_results:
        all_results.extend(batch_result)
        total_found += batch_found
        total_missed += batch_missed

    pairs_total = len(tiles_by_edge_time)
    print(f"[mp] Total pairs: {pairs_total}, found: {total_found}, missed: {total_missed}")

    # Merge into edges
    df = pd.DataFrame(all_results)
    
    # Only merge the edges that were actually processed (for correct preview)
    processed_edge_uids = set(df["edge_uid"])
    edges_processed = edges[edges["edge_uid"].isin(processed_edge_uids)]
    merged = edges_processed.merge(df, on="edge_uid", how="left")
    
    # Round all shade/shadow columns to 2 decimal places for consistent file size
    shade_cols = [c for c in merged.columns if ('shade' in c or 'shadow' in c) and c not in ['edge_uid', 'timestamp']]
    for col in shade_cols:
        if merged[col].dtype in ['float64', 'float32']:
            merged[col] = merged[col].round(2)
    
    # For the final output, merge with ALL edges
    final_merged = edges.merge(df, on="edge_uid", how="left")
    for col in shade_cols:
        if final_merged[col].dtype in ['float64', 'float32']:
            final_merged[col] = final_merged[col].round(2)
    
    print(f"[mp] Saving final output with {len(final_merged):,} edges...")
    final_merged.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved: {output_path}")

    # ===== Summary statistics and sample preview =====
    value_cols = [c for c in merged.columns if ('shade' in c or 'shadow' in c) and c not in ['edge_uid', 'timestamp']]
    print("\n===== RUN SUMMARY =====")
    print(f"Pairs total:  {pairs_total}")
    print(f"Pairs found:  {total_found}")
    print(f"Pairs missed: {total_missed} (treated as nighttime)")
    
    for col in value_cols:
        ser = merged[col].dropna()
        if len(ser) == 0:
            print(f"{col}: no values")
        else:
            print(f"{col}: n={len(ser)} min={ser.min():.2f} p25={ser.quantile(0.25):.2f} median={ser.median():.2f} mean={ser.mean():.2f} p75={ser.quantile(0.75):.2f} max={ser.max():.2f}")

    # Print preview of PROCESSED data only
    preview_cols = ["edge_uid", "timestamp"] + value_cols
    preview = merged[preview_cols].head(12)
    try:
        print(f"\n===== PREVIEW (processed {len(merged)} edges) =====")
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
    parser.add_argument("--n_processes", type=int, default=16, help="Number of parallel processes to use.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of edge-time pairs per batch.")
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
        args.n_processes,
        args.batch_size,
    )

# FINAL COLUMN EXPLANATION:
# 
# 1. current_shade: Instantaneous shade at timestamp (0=no shade, 1=full shade)
#    → Uses: cbdb17d4_p_5_Shadow_20240819_1300_LST.tif
# 
# 2. shadow_fraction: Cumulative shadow fraction since dawn at timestamp  
#    → Uses: cbdb17d4_p_5_shadow_fraction_on_20240819_1300.tif
# 
# 3. shade_2h_before: Average shade 2h before timestamp
#    → Uses: Average of Shadow_20240819_1100_LST.tif, 1200_LST.tif, 1300_LST.tif
# 
# 4. shade_4h_before: Average shade 4h before timestamp  
#    → Uses: Average of Shadow_20240819_0900_LST.tif through 1300_LST.tif

### How to run:
# nohup python aggregate_edge_shade_stats_mp.py \
#   --points results/output/step6_final_result/cbdb17d4/binned_dataset_2024.geojson \
#   --edges data/clean_data/strava/back_bay_edges_aoi.geojson \
#   --output results/output/step6_final_result/cbdb17d4/edge_stats_final_4months.geojson \
#   --config config.yaml \
#   --osmid cbdb17d4 \
#   --n_processes 25 \
#   --batch_size 1500 \
#   > logs/aggregate_4month_full.log 2>&1 &
