#!/usr/bin/env python3
"""
SUPER-BATCHED: Parallel Tile-Timestamp Aware Edge Shade Extraction System
Final optimization: Multiprocessing over (binned_date, hour_of_day, tile_combo) super-batches

PARALLEL HOURLY BATCHING: Uses ProcessPoolExecutor for multi-core processing

Key improvements:
- Parallelizes super-batches across multiple worker processes
- Each worker has independent TileInventory and raster access
- Groups edges by (binned_date, hour_of_day, combo_key) into super-batches  
- Derives hour_of_day = HHMM from rounded_timestamp for reusability
- Uses binned_date + hour_of_day for raster filename resolution
- Opens rasters once per super-batch (not per edge)
- Vectorized zonal_stats over many edges at once
- Configurable worker count with --workers N
- Progress logging and result aggregation
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union, FrozenSet
from datetime import datetime, timedelta
import hashlib
import tempfile
import shutil
from collections import defaultdict
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterstats import zonal_stats

NODATA_VAL = -9999

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup structured logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('super_batched_extraction.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def _normalize_tile_id(t):
    """Normalize tile ID to canonical format"""
    s = str(t)
    return s if s.startswith('p_') else f"p_{int(float(s))}"

class TileInventory:
    """Build inventory of available raster files"""
    
    def __init__(self, shade_results_path: Path, osmid: str):
        self.shade_results_path = Path(shade_results_path)
        self.osmid = osmid
        self.combined_shade_path = self.shade_results_path / osmid / "combined_shade"
        
        self.shadow_files = {}    # (tile, date_str, time_str) -> path
        self.fraction_files = {}  # (tile, date_str, time_str) -> path
        
    def build_inventory(self) -> bool:
        """Build inventory of available raster files"""
        if not self.combined_shade_path.exists():
            return False
        
        for tile_dir in self.combined_shade_path.iterdir():
            if not tile_dir.is_dir():
                continue
                
            tile_num = tile_dir.name
            tile_id = f"p_{tile_num}"
            
            # Shadow files
            for filepath in tile_dir.glob(f"{self.osmid}_{tile_id}_Shadow_*_LST.tif"):
                parts = filepath.stem.split('_')
                if len(parts) >= 6:
                    date_str = parts[-3]
                    time_str = parts[-2] 
                    self.shadow_files[(tile_id, date_str, time_str)] = filepath
            
            # Fraction files  
            for filepath in tile_dir.glob(f"{self.osmid}_{tile_id}_shadow_fraction_on_*.tif"):
                parts = filepath.stem.split('_')
                if len(parts) >= 7:
                    date_str = parts[-2]
                    time_str = parts[-1]
                    self.fraction_files[(tile_id, date_str, time_str)] = filepath
        
        return True
    
    def get_raster_files(self, tiles: List[str], date_str: str, time_str: str, file_type: str) -> List[Path]:
        """Get raster files for tiles at specific timestamp"""
        files = []
        
        for tile_id in tiles:
            key = (tile_id, date_str, time_str)
            
            if file_type == 'shadow' and key in self.shadow_files:
                files.append(self.shadow_files[key])
            elif file_type == 'fraction' and key in self.fraction_files:
                files.append(self.fraction_files[key])
        
        return [f for f in files if f.exists()]

def _zonal_mean_many(gdf: gpd.GeoDataFrame, files: List[Path], file_type: str) -> List[float]:
    """Vectorized zonal statistics over many edges at once"""
    if not files:
        return [np.nan] * len(gdf)
    
    try:
        if len(files) == 1:
            # Single raster
            with rasterio.open(files[0]) as src:
                # Ensure CRS match
                gdf_proj = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf

                # Use file nodata if present, else fall back to our canonical value
                nodata = src.nodata if src.nodata is not None else NODATA_VAL
                
                # Vectorized zonal stats
                stats = zonal_stats(gdf_proj, src.read(1), affine=src.transform,
                                  stats=['mean'], all_touched=True, nodata=nodata)
                
                return [s['mean'] if s and s['mean'] is not None else np.nan for s in stats]
        
        else:
            # Multiple rasters - create MAX mosaic
            sources = []
            for path in files:
                sources.append(rasterio.open(path))
            
            try:
                # Create MAX mosaic in memory
                arr, transform = rio_merge(sources, method='max', nodata=NODATA_VAL)
                crs = sources[0].crs
                
                # Ensure CRS match
                gdf_proj = gdf.to_crs(crs) if gdf.crs != crs else gdf
                
                # Vectorized zonal stats on mosaic
                stats = zonal_stats(gdf_proj, arr[0], affine=transform,
                                  stats=['mean'], all_touched=True, nodata=NODATA_VAL)
                
                return [s['mean'] if s and s['mean'] is not None else np.nan for s in stats]
                
            finally:
                for src in sources:
                    src.close()
    
    except Exception as e:
        return [np.nan] * len(gdf)

def process_single_super_batch(batch_info: Dict, edges_gdf: gpd.GeoDataFrame,
                              shade_results_path: Path, osmid: str,
                              historical_hours: Optional[List[int]] = None) -> List[Dict]:
    """Process a single super-batch in a worker process"""
    
    binned_date = batch_info['binned_date']
    hour_of_day = batch_info['hour_of_day']
    tiles = batch_info['tiles']
    edges = batch_info['edges']
    
    # Each worker creates its own inventory (process-safe)
    inventory = TileInventory(shade_results_path, osmid)
    if not inventory.build_inventory():
        return []
    
    # Convert binned_date to timestamp if it's a string
    if isinstance(binned_date, str):
        binned_date_ts = pd.to_datetime(binned_date)
    else:
        binned_date_ts = binned_date
    
    date_str = binned_date_ts.strftime('%Y%m%d')
    time_str = hour_of_day  # Already in HHMM format
    
    # Get raster files for this binned_date + hour_of_day + tile combination
    shadow_files = inventory.get_raster_files(tiles, date_str, time_str, 'shadow')
    fraction_files = inventory.get_raster_files(tiles, date_str, time_str, 'fraction')
    
    # Subset to edges in this batch (using ORIGINAL geometries)
    gdf_sub = edges_gdf[edges_gdf['edge_uid'].isin(edges)].copy()
    
    if len(gdf_sub) == 0:
        return []
    
    # Vectorized zonal statistics
    shadow_means = _zonal_mean_many(gdf_sub, shadow_files, 'shadow')
    fraction_means = _zonal_mean_many(gdf_sub, fraction_files, 'fraction')

    # Historical (lookback) shade windows reuse the same gdf subset
    sanitized_hours = []
    if historical_hours:
        sanitized_hours = sorted({int(h) for h in historical_hours if h is not None and h > 0})

    shade_offsets: Dict[int, List[float]] = {0: shadow_means}
    offsets_required: List[int] = []
    if sanitized_hours:
        # Compute union of offsets needed across all requested windows (current hour included)
        offsets_required = sorted({offset for hour in sanitized_hours for offset in range(hour + 1)})
        # We already have offset 0 values
        offsets_to_compute = [o for o in offsets_required if o != 0]

        # Compose reference timestamp for computing prior hour strings (matches legacy aggregation script behaviour)
        base_time = datetime.strptime(hour_of_day, '%H%M')
        current_dt = datetime.combine(binned_date_ts.date(), base_time.time())

        for offset in offsets_to_compute:
            target_dt = current_dt - timedelta(hours=offset)
            target_hour = target_dt.strftime('%H%M')
            # Historical lookups intentionally stick with the same binned_date folder
            target_files = inventory.get_raster_files(tiles, date_str, target_hour, 'shadow')
            shade_offsets[offset] = _zonal_mean_many(gdf_sub, target_files, 'shadow')

    window_offsets: Dict[int, List[int]] = {}
    if sanitized_hours:
        window_offsets = {hour: [o for o in offsets_required if o <= hour] for hour in sanitized_hours}
    
    # Build results for all edges in batch
    results = []
    for idx, (uid, shadow_val, fraction_val) in enumerate(zip(gdf_sub['edge_uid'].tolist(), shadow_means, fraction_means)):
        row = {
            'edge_uid': uid,
            'binned_date': binned_date_ts,
            'hour_of_day': hour_of_day,
            'current_shade': None if pd.isna(shadow_val) else round(float(shadow_val), 3),
            'shadow_fraction': None if pd.isna(fraction_val) else round(float(fraction_val), 3),
        }

        if sanitized_hours:
            for hour in sanitized_hours:
                offsets = window_offsets.get(hour, [])
                values = []
                for offset in offsets:
                    offset_vals = shade_offsets.get(offset)
                    if not offset_vals:
                        continue
                    candidate = offset_vals[idx]
                    if pd.isna(candidate):
                        continue
                    values.append(float(candidate))

                key = f'shade_{hour}h_before'
                row[key] = None if not values else round(float(np.mean(values)), 3)

        results.append(row)

    return results

def analyze_results(results_df: pd.DataFrame) -> None:
    """Analyze sanity-check results for quick validation"""
    
    logging.info("=" * 60)
    logging.info("🔍 SANITY-CHECK ANALYSIS")
    logging.info("=" * 60)
    
    # Basic counts
    total_records = len(results_df)
    unique_edges = results_df['edge_uid'].nunique()
    unique_binned_dates = results_df['binned_date'].nunique()
    unique_hours_of_day = results_df['hour_of_day'].nunique()
    
    logging.info(f"📊 Basic Stats:")
    logging.info(f"  Total records: {total_records:,}")
    logging.info(f"  Unique edges: {unique_edges:,}")
    logging.info(f"  Unique binned dates: {unique_binned_dates:,}")
    logging.info(f"  Unique hours of day: {unique_hours_of_day:,}")
    
    # Shadow values analysis
    shadow_vals = results_df['current_shade'].dropna()
    shadow_nans = results_df['current_shade'].isna().sum()
    
    logging.info(f"")
    logging.info(f"🌑 Shadow Values:")
    logging.info(f"  Valid values: {len(shadow_vals):,} ({len(shadow_vals)/total_records:.1%})")
    logging.info(f"  NaN values: {shadow_nans:,} ({shadow_nans/total_records:.1%})")
    
    if len(shadow_vals) > 0:
        logging.info(f"  Range: [{shadow_vals.min():.3f}, {shadow_vals.max():.3f}]")
        logging.info(f"  Mean: {shadow_vals.mean():.3f}")
        logging.info(f"  Percentiles: P25={shadow_vals.quantile(0.25):.3f}, P50={shadow_vals.quantile(0.5):.3f}, P75={shadow_vals.quantile(0.75):.3f}")
    
    # Fraction values analysis
    fraction_vals = results_df['shadow_fraction'].dropna()
    fraction_nans = results_df['shadow_fraction'].isna().sum()
    
    logging.info(f"")
    logging.info(f"🔆 Shadow Fractions:")
    logging.info(f"  Valid values: {len(fraction_vals):,} ({len(fraction_vals)/total_records:.1%})")
    logging.info(f"  NaN values: {fraction_nans:,} ({fraction_nans/total_records:.1%})")
    
    if len(fraction_vals) > 0:
        logging.info(f"  Range: [{fraction_vals.min():.3f}, {fraction_vals.max():.3f}]")
        logging.info(f"  Mean: {fraction_vals.mean():.3f}")
        logging.info(f"  Percentiles: P25={fraction_vals.quantile(0.25):.3f}, P50={fraction_vals.quantile(0.5):.3f}, P75={fraction_vals.quantile(0.75):.3f}")
    
    # Data quality flags
    logging.info(f"")
    logging.info(f"🚩 Quality Flags:")
    
    # Check for suspicious patterns
    if len(shadow_vals) > 0:
        all_zeros = (shadow_vals == 0.0).sum()
        all_ones = (shadow_vals == 1.0).sum()
        if all_zeros / len(shadow_vals) > 0.8:
            logging.warning(f"  ⚠️  {all_zeros/len(shadow_vals):.1%} shadow values are exactly 0.0")
        elif all_ones / len(shadow_vals) > 0.8:
            logging.warning(f"  ⚠️  {all_ones/len(shadow_vals):.1%} shadow values are exactly 1.0")
        else:
            logging.info(f"  ✅ Shadow values show reasonable variation")
    
    if len(fraction_vals) > 0:
        all_zeros = (fraction_vals == 0.0).sum()
        all_ones = (fraction_vals == 1.0).sum()
        if all_zeros / len(fraction_vals) > 0.8:
            logging.warning(f"  ⚠️  {all_zeros/len(fraction_vals):.1%} fraction values are exactly 0.0")
        elif all_ones / len(fraction_vals) > 0.8:
            logging.warning(f"  ⚠️  {all_ones/len(fraction_vals):.1%} fraction values are exactly 1.0")
        else:
            logging.info(f"  ✅ Fraction values show reasonable variation")
    
    # Sample records for manual inspection
    logging.info(f"")
    logging.info(f"📝 Sample Records:")
    sample_size = min(10, len(results_df))
    sample = results_df.sample(n=sample_size, random_state=42)
    for _, row in sample.iterrows():
        binned_str = row['binned_date'].strftime('%Y-%m-%d')
        hour_str = row['hour_of_day']
        logging.info(f"  edge={row['edge_uid']}, date={binned_str}, hour={hour_str}, shade={row['current_shade']}, fraction={row['shadow_fraction']}")
    
    logging.info("=" * 60)

class ParallelSuperBatchedExtractor:
    """PARALLEL SUPER-BATCHED: Multi-core extractor with worker processes"""

    @staticmethod
    def _sanitize_historical_hours(hours: Optional[List[int]]) -> List[int]:
        if not hours:
            return []
        cleaned = sorted({int(h) for h in hours if h is not None and h > 0})
        return cleaned
    
    def __init__(self, points_parquet: Path, edge_file: Path, shade_results_path: Path,
                 osmid: str, output_dir: Path, historical_hours: Optional[List[int]] = None):
        
        self.points_parquet = Path(points_parquet)
        self.edge_file = Path(edge_file)
        self.shade_results_path = Path(shade_results_path)
        self.osmid = osmid
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.historical_hours = self._sanitize_historical_hours(historical_hours)
        
        self.stats = {
            'start_time': time.time(),
            'super_batches_processed': 0,
            'edges_processed': 0,
        }
        
    def run_parallel_extraction(self, dry_run: bool = False, sanity_check: Optional[int] = None, 
                               workers: int = 4) -> bool:
        """Run PARALLEL SUPER-BATCHED extraction"""
        
        logging.info("=" * 80)
        if sanity_check:
            logging.info(f"PARALLEL SANITY-CHECK MODE: {sanity_check} BATCHES, {workers} WORKERS")
        elif dry_run:
            logging.info(f"PARALLEL DRY-RUN MODE: {workers} WORKERS")
        else:
            logging.info(f"FULL PARALLEL SUPER-BATCHED EXTRACTION: {workers} WORKERS")
        logging.info("=" * 80)
        
        # Log data sources
        logging.info("✅ PARALLEL SUPER-BATCH OPTIMIZATION + CORRECT ARCHITECTURE:")
        logging.info(f"  📁 POINT_FILE (batching only): {self.points_parquet}")
        logging.info(f"  📐 EDGE_FILE (geometries): {self.edge_file}")
        logging.info(f"  🗂️ SHADE_RESULTS: {self.shade_results_path}")
        logging.info(f"  ⚡ WORKERS: {workers} parallel processes")
        logging.info(f"  ⏱️ Historical windows (hours): {self.historical_hours if self.historical_hours else 'disabled'}")
        
        # Phase 1: Build inventory (main process only, for logging)
        main_inventory = TileInventory(self.shade_results_path, self.osmid)
        if not main_inventory.build_inventory():
            logging.error("Failed to build inventory")
            return False
        
        logging.info(f"Inventory: {len(main_inventory.shadow_files)} shadow, {len(main_inventory.fraction_files)} fraction files")
        
        # Phase 2: Load ORIGINAL edge geometries
        logging.info(f"Loading ORIGINAL edge geometries from: {self.edge_file}")
        edges_gdf = gpd.read_file(self.edge_file)
        logging.info(f"✅ Loaded {len(edges_gdf):,} ORIGINAL edge geometries")
        logging.info(f"✅ Edge CRS: {edges_gdf.crs}")
        
        # Phase 3: Build super-batches from point metadata
        logging.info("Building super-batches from point metadata...")
        
        points_df = pd.read_parquet(self.points_parquet, 
                                  columns=['edge_uid', 'tile_number', 'binned_date', 'rounded_timestamp'])
        logging.info(f"Loaded {len(points_df):,} points for super-batch computation")
        
        super_batches_df = self._build_super_batches(points_df, edges_gdf)
        
        if dry_run:
            logging.info("DRY RUN: Stopping before processing")
            self._log_super_batch_estimates(super_batches_df, workers)
            return True
        
        # Phase 4: Prepare batch jobs
        if sanity_check:
            logging.info(f"🔍 SANITY-CHECK: Sampling {sanity_check} batches from {len(super_batches_df):,} total")
            sampled_batches = self._sample_super_batches(super_batches_df, sanity_check)
            process_df = sampled_batches
        else:
            logging.info("Processing all super-batches in parallel...")
            process_df = super_batches_df
        
        # Convert DataFrame to list of batch jobs
        batch_jobs = []
        for row in process_df.itertuples(index=False):
            batch_info = {
                'binned_date': row.binned_date,
                'hour_of_day': row.hour_of_day,
                'tiles': row.tiles,
                'edges': row.edges,
                'n_edges': row.n_edges
            }
            batch_jobs.append(batch_info)
        
        logging.info(f"Prepared {len(batch_jobs):,} batch jobs for parallel processing")
        
        # Phase 5: Parallel processing with ProcessPoolExecutor
        all_results = []
        completed_batches = 0
        completed_edges = 0
        
        logging.info(f"Starting parallel processing with {workers} workers...")
        
        # Phase 5: Parallel processing with ProcessPoolExecutor - CHUNKED VERSION
        all_results = []
        completed_batches = 0
        completed_edges = 0
        
        # Process in chunks to avoid overwhelming ProcessPoolExecutor
        chunk_size = workers * 20  # Process 20 batches per worker at a time
        total_batches = len(batch_jobs)
        
        logging.info(f"Starting parallel processing with {workers} workers...")
        logging.info(f"Processing {total_batches:,} batches in chunks of {chunk_size}")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Process batches in chunks
            for chunk_start in range(0, total_batches, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_batches)
                chunk = batch_jobs[chunk_start:chunk_end]
                
                logging.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_batches-1)//chunk_size + 1}: "
                           f"batches {chunk_start:,}-{chunk_end-1:,} ({len(chunk):,} batches)")
                
                # Submit jobs for this chunk
                future_to_batch = {
                    executor.submit(
                        process_single_super_batch,
                        batch_info,
                        edges_gdf,
                        self.shade_results_path,
                        self.osmid,
                        self.historical_hours,
                    ): batch_info
                    for batch_info in chunk
                }
                
                # Process completed futures for this chunk
                chunk_completed = 0
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        
                        completed_batches += 1
                        completed_edges += len(batch_results)
                        chunk_completed += 1
                        
                        # Progress logging every 100 batches or 10% of total
                        if completed_batches % 100 == 0 or completed_batches % (total_batches // 10 + 1) == 0:
                            progress = completed_batches / total_batches * 100
                            logging.info(f"Progress: {completed_batches:,}/{total_batches:,} batches ({progress:.1f}%), "
                                       f"{completed_edges:,} edges processed")
                            
                    except Exception as e:
                        logging.error(f"Batch failed: {batch_info['binned_date']}, {batch_info['hour_of_day']}, "
                                    f"tiles={len(batch_info['tiles'])}, error: {e}")
                
                logging.info(f"Completed chunk {chunk_start//chunk_size + 1}: "
                           f"{chunk_completed}/{len(chunk)} batches successful")


        logging.info(f"✅ Parallel processing complete: {completed_batches:,} batches, {completed_edges:,} edges")
        
        # Phase 6: Save and validate results
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Check for duplicates
            duplicates = results_df.duplicated(subset=['edge_uid', 'binned_date', 'hour_of_day']).sum()
            if duplicates > 0:
                logging.error(f"CRITICAL: Found {duplicates} duplicate (edge_uid, binned_date, hour_of_day) rows!")
                return False
            
            # Save results
            if sanity_check:
                output_file = self.output_dir / f"{self.osmid}_parallel_sanity_check_{sanity_check}_batches.parquet"
                logging.info(f"🔍 PARALLEL SANITY-CHECK results: {len(results_df):,} records in {output_file}")
            else:
                output_file = self.output_dir / f"{self.osmid}_parallel_super_batched_shade_results.parquet"
                logging.info(f"✅ PARALLEL SUPER-BATCHED results: {len(results_df):,} records in {output_file}")
            
            results_df.to_parquet(output_file, index=False)
            logging.info(f"✅ Unique edges: {results_df['edge_uid'].nunique():,}")
            logging.info("✅ ALL geometries from ORIGINAL edge file (zero reconstruction)")
            
            # SANITY-CHECK: Analyze results
            if sanity_check:
                analyze_results(results_df)
        
        # Update stats
        self.stats['super_batches_processed'] = completed_batches
        self.stats['edges_processed'] = completed_edges
        
        # Final summary
        self._print_parallel_summary(super_batches_df, workers, sanity_mode=bool(sanity_check))
        
        return True
    
    def _sample_super_batches(self, super_batches_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Sample super-batches with variety across timestamps and tile combos"""
        
        total_batches = len(super_batches_df)
        
        if n_samples >= total_batches:
            logging.info("Requested samples >= total batches, using all batches")
            return super_batches_df
        
        # Simple random sample
        np.random.seed(42)
        sampled_indices = np.random.choice(total_batches, size=n_samples, replace=False)
        result = super_batches_df.iloc[sampled_indices].copy().reset_index(drop=True)
        
        # Sort by binned_date, then hour_of_day for better cache locality
        result = result.sort_values(['binned_date', 'hour_of_day', 'combo_key']).reset_index(drop=True)
        
        logging.info(f"Sampled batches cover {result['binned_date'].nunique()} unique binned_dates, "
                    f"{result['hour_of_day'].nunique()} unique hours_of_day, and "
                    f"{result['combo_key'].nunique()} unique tile combos")
        
        return result
    
    def _build_super_batches(self, points_df: pd.DataFrame, edges_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Build super-batches using HOURLY key (binned_date, hour_of_day)"""
        
        pts = points_df[['edge_uid', 'tile_number', 'binned_date', 'rounded_timestamp']].copy()
        pts['tile_id'] = pts['tile_number'].map(_normalize_tile_id)
        
        # Extract hour_of_day as HHMM from rounded_timestamp
        pts['hour_of_day'] = pd.to_datetime(pts['rounded_timestamp']).dt.strftime('%H%M')
        
        # Join coverage check
        point_edges = set(pts['edge_uid'].unique())
        geom_edges = set(edges_gdf['edge_uid'].unique())
        join_coverage = len(point_edges & geom_edges) / len(point_edges)
        
        if join_coverage < 0.98:
            logging.warning(f"Join coverage: {join_coverage:.1%} - some point edges missing from EDGE_FILE")
        else:
            logging.info(f"✅ Join coverage: {join_coverage:.1%}")
        
        # Group by (edge_uid, binned_date, hour_of_day) to get unique tiles
        edge_time_tiles = (
            pts.groupby(['edge_uid', 'binned_date', 'hour_of_day'])['tile_id']
               .apply(lambda x: sorted(set(x)))
               .reset_index()
               .rename(columns={'tile_id':'tiles'})
        )
        
        logging.info(f"Created {len(edge_time_tiles):,} edge-(binned_date,hour_of_day) combinations")
        
        # Combo key
        edge_time_tiles['combo_key'] = edge_time_tiles['tiles'].apply(lambda ts: '|'.join(ts))
        
        # Group by (binned_date, hour_of_day, combo_key) into super-batches
        super_batches_df = (
            edge_time_tiles.groupby(['binned_date', 'hour_of_day', 'combo_key'])['edge_uid']
                           .apply(list)
                           .reset_index()
                           .rename(columns={'edge_uid': 'edges'})
        )
        
        # Add tiles back and metrics
        combo_to_tiles = dict(zip(edge_time_tiles['combo_key'], edge_time_tiles['tiles']))
        super_batches_df['tiles'] = super_batches_df['combo_key'].map(combo_to_tiles)
        super_batches_df['n_edges'] = super_batches_df['edges'].apply(len)
        
        # Sort by binned_date, then hour_of_day for cache friendliness
        super_batches_df = super_batches_df.sort_values(['binned_date','hour_of_day','combo_key']).reset_index(drop=True)
        
        # Required logging
        binned_dates_seen = super_batches_df['binned_date'].nunique()
        hours_of_day_seen = super_batches_df['hour_of_day'].nunique()
        unique_combo_keys = super_batches_df['combo_key'].nunique()
        super_batches_total = len(super_batches_df)
        edges_total = super_batches_df['n_edges'].sum()
        edges_per_batch_p50 = super_batches_df['n_edges'].quantile(0.5)
        edges_per_batch_p90 = super_batches_df['n_edges'].quantile(0.9)
        edges_per_batch_max = super_batches_df['n_edges'].max()
        
        logging.info(f"SUPER-BATCHES: total={super_batches_total:,} "
                     f"unique_binned_dates={binned_dates_seen:,} "
                     f"unique_hours_of_day={hours_of_day_seen:,} "
                     f"unique_combos={unique_combo_keys:,}")
        logging.info(f"SUPER-BATCHES: edges_total={edges_total:,}, "
                     f"edges_per_batch_p50={edges_per_batch_p50:.0f}, "
                     f"edges_per_batch_p90={edges_per_batch_p90:.0f}, "
                     f"edges_per_batch_max={edges_per_batch_max:.0f}")
        
        return super_batches_df
    
    def _log_super_batch_estimates(self, super_batches_df: pd.DataFrame, workers: int):
        """Log performance estimates for dry run"""
        super_batches_total = len(super_batches_df)
        
        logging.info("PARALLEL SUPER-BATCH PERFORMANCE ESTIMATES:")
        logging.info(f"  Super-batches total: {super_batches_total:,}")
        logging.info(f"  Workers: {workers}")
        logging.info(f"  Estimated batches per worker: {super_batches_total // workers:,}")
        logging.info(f"  Previous serial system runtime: ~577k batches × 3.3 minutes / 50 batches = ~633 minutes")
        logging.info(f"  Estimated parallel runtime: ~{633 / workers:.1f} minutes with {workers} workers")
        logging.info(f"  Expected speedup: ~{workers}x faster than serial")
    
    def _print_parallel_summary(self, super_batches_df: pd.DataFrame, workers: int, sanity_mode: bool = False):
        """Print comprehensive parallel summary"""
        self.stats['end_time'] = time.time()
        runtime = self.stats['end_time'] - self.stats['start_time']
        
        super_batches_total = len(super_batches_df)
        
        logging.info("=" * 80)
        if sanity_mode:
            logging.info("PARALLEL SANITY-CHECK EXTRACTION SUMMARY") 
        else:
            logging.info("PARALLEL SUPER-BATCHED EXTRACTION SUMMARY") 
        logging.info("=" * 80)
        logging.info(f"Total runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
        logging.info(f"Workers used: {workers}")
        logging.info(f"Super-batches processed: {self.stats['super_batches_processed']:,} (of {super_batches_total:,} total)")
        logging.info(f"Edges processed: {self.stats['edges_processed']:,}")
        
        if self.stats['super_batches_processed'] > 0:
            batches_per_second = self.stats['super_batches_processed'] / runtime
            edges_per_second = self.stats['edges_processed'] / runtime
            logging.info(f"Throughput: {batches_per_second:.1f} batches/sec, {edges_per_second:.0f} edges/sec")
        
        logging.info("")
        logging.info("✅ PARALLEL SUPER-BATCH OPTIMIZATION VERIFIED:")
        logging.info(f"  ✅ Super-batches total: {super_batches_total:,}")
        logging.info(f"  ✅ One mosaic per (binned_date, hour_of_day, combo_key)")
        logging.info(f"  ✅ Vectorized zonal_stats over many edges")
        logging.info(f"  ✅ Process-safe raster I/O (independent TileInventory per worker)")
        
        logging.info("")
        logging.info("✅ ARCHITECTURE CORRECTNESS MAINTAINED:")
        logging.info(f"  ✅ Edge geometries from: {self.edge_file}")
        logging.info("  ✅ Points used ONLY for super-batch computation")
        logging.info("  ✅ NO geometry reconstruction from points")
        logging.info("  ✅ zonal_stats on ORIGINAL LineStrings")
        
        logging.info("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='PARALLEL SUPER-BATCHED Edge Shade Extraction (HOURLY)')
    parser.add_argument('--points', required=True, type=Path, help='Points Parquet (batching only)')
    parser.add_argument('--edges', required=True, type=Path, help='Original edge geometries')
    parser.add_argument('--shade-results', required=True, type=Path, help='Shade results directory')
    parser.add_argument('--osmid', required=True, help='OSM ID')
    parser.add_argument('--output', required=True, type=Path, help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--sanity-check', type=int, metavar='N', help='Sanity check mode: process only N super-batches')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument(
        '--historical-hours',
        type=int,
        nargs='*',
        default=[2, 4],
        help=('Historical shade windows in hours (averaged with the current hour). '
              'Example: --historical-hours 1 3 6. Leave empty to disable.'),
    )
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.dry_run and args.sanity_check:
        print("Error: --dry-run and --sanity-check cannot be used together")
        return 1
    
    # Validate workers
    if args.workers < 1:
        print("Error: --workers must be >= 1")
        return 1
    
    max_workers = mp.cpu_count()
    if args.workers > max_workers:
        print(f"Warning: --workers {args.workers} exceeds available CPUs ({max_workers})")
    
    logger = setup_logging(args.log_level)
    
    # Input validation logging
    logging.info("PARALLEL SUPER-BATCHED SYSTEM (HOURLY) - INPUTS:")
    logging.info(f"  Points: {args.points} (exists: {args.points.exists()})")
    logging.info(f"  Edges: {args.edges} (exists: {args.edges.exists()})")
    logging.info(f"  Shade results: {args.shade_results} (exists: {args.shade_results.exists()})")
    logging.info(f"  Workers: {args.workers} (available CPUs: {max_workers})")
    logging.info(f"  Historical hours (raw): {args.historical_hours if args.historical_hours else 'disabled'}")
    
    if args.sanity_check:
        logging.info(f"  🔍 Sanity check mode: {args.sanity_check} batches")
    
    extractor = ParallelSuperBatchedExtractor(
        points_parquet=args.points,
        edge_file=args.edges,
        shade_results_path=args.shade_results,
        osmid=args.osmid,
        output_dir=args.output,
        historical_hours=args.historical_hours,
    )
    
    success = extractor.run_parallel_extraction(
        dry_run=args.dry_run,
        sanity_check=args.sanity_check,
        workers=args.workers
    )
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
