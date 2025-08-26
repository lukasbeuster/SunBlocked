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

def _candidate_filenames(osmid, tile_id, base_name, ts_str):
    # Tries common variants we use across the pipeline
    # Examples observed: "{osmid}_{tile_id}_Shadow_{ts}_LST.tif", "{osmid}_{tile_id}_Shadow_{ts}.tif",
    #                    "{osmid}_{tile_short}_{base_name}_{ts}.tif"
    return [
        f"{osmid}_{tile_id}_{base_name}_{ts_str}_LST.tif",
        f"{osmid}_{tile_id}_{base_name}_{ts_str}.tif",
        f"{osmid}_{tile_id}_{base_name.capitalize()}_{ts_str}_LST.tif",
        f"{osmid}_{tile_id}_{base_name.capitalize()}_{ts_str}.tif",
    ]

def _find_single_raster(base, shade_type, tile_id, timestamp, osmid, base_name):
    ts_str = timestamp.strftime('%Y%m%d_%H%M')
    tile_folder = base / shade_type / tile_id
    if not tile_folder.exists():
        return None
    for name in _candidate_filenames(osmid, tile_id, base_name, ts_str):
        p = tile_folder / name
        if p.exists():
            return p
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

def mosaic_rasters(base, shade_type, tile_ids, timestamp, osmid, base_name):
    """Return a (array, transform, nodata) mosaic for the given tiles at timestamp.
    Tries each tile; if only one found, returns that array. Returns (None, None, None) if none found.
    """
    sources = []
    for t in tile_ids:
        # tile ids are expected like 'p_12'. Keep as-is; folder names follow this convention.
        rp = _find_single_raster(base, shade_type, t, timestamp, osmid, base_name)
        if rp is None:
            continue
        src = rasterio.open(rp)
        sources.append(src)
    if not sources:
        return None, None, None
    if len(sources) == 1:
        with sources[0] as s:
            arr = s.read(1)
            return arr, s.transform, s.nodata
    # Multiple sources → merge
    try:
        mosaic, transform = rio_merge(sources)
        # rio_merge yields shape (bands, rows, cols); take band 1
        return mosaic[0], transform, sources[0].nodata
    finally:
        for s in sources:
            s.close()

def main(points_path, edges_path, output_path, config_path, osmid, buffers, include_building, sample_n, sample_seed):
    # Load data
    points = gpd.read_file(points_path)
    edges = gpd.read_file(edges_path)
    points["rounded_timestamp"] = pd.to_datetime(points["rounded_timestamp"])
    edges["edge_uid"] = edges["edge_uid"].astype(str)
    points["edge_uid"] = points["edge_uid"].astype(str)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base = Path(config["output_dir"]) / "step5_shade_results" / osmid
    hours_before = config["extra_outputs"]["hours_before"]

    results = []

    # Build mapping: for each edge_uid & rounded_timestamp → list of tiles (may be 1+)
    pts_subset = points[["edge_uid", "rounded_timestamp", "tile_number"]].drop_duplicates()
    tiles_by_edge_time = (
        pts_subset.groupby(["edge_uid", "rounded_timestamp"])["tile_number"]
                  .apply(lambda s: sorted(set(s.tolist())))
                  .reset_index()
    )

    # Optional sampling for quick runs
    if sample_n and sample_n > 0:
        n_before = len(tiles_by_edge_time)
        tiles_by_edge_time = tiles_by_edge_time.sample(n=min(sample_n, n_before), random_state=sample_seed)
        print(f"Sampling enabled: {len(tiles_by_edge_time)}/{n_before} (edge_uid, time) pairs will be processed (seed={sample_seed}).")

    for edge_uid, timestamp, tile_list in tiles_by_edge_time.itertuples(index=False):
        # Prepare geometry row for this edge
        edge_geom = edges[edges["edge_uid"] == edge_uid]
        if edge_geom.empty:
            continue

        row = {"edge_uid": edge_uid, "timestamp": timestamp}

        # Mosaic combined shade across tiles
        arr, transform, nodata_val = mosaic_rasters(base, "combined_shade", tile_list, timestamp, osmid, "Shadow")
        # Optional: combined shadow fraction
        arr_frac, transform_frac, nodata_frac = mosaic_rasters(base, "combined_shade", tile_list, timestamp, osmid, "shadow_fraction_on")

        for buffer in buffers:
            suffix = f"_buffer{int(buffer)}" if buffer else ""
            if arr is not None:
                row[f"combined_shade{suffix}"] = zonal_from_array(edge_geom, arr, transform, nodata_val, buffer)
            if arr_frac is not None:
                row[f"combined_shadow_fraction{suffix}"] = zonal_from_array(edge_geom, arr_frac, transform_frac, nodata_frac, buffer)

        results.append(row)

    # Merge into edges
    df = pd.DataFrame(results)
    merged = edges.merge(df, on="edge_uid", how="left")
    merged.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved: {output_path}")

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
    )


# python aggregate_edge_shade_stats.py \
#   --points ../results/output/step6_final_result/cbdb17d4/binned_dataset_2024.geojson \
#   --edges  ../data/clean_data/strava/back_bay_edges_aoi.geojson \
#   --output ../results/output/step6_final_result/cbdb17d4/edge_stats_sample.geojson \
#   --config config.yaml \
#   --osmid cbdb17d4 \
#   --buffers 0 10 \
#   --sample_n 50 \
#   --sample_seed 42
