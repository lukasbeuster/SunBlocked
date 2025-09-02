#!/usr/bin/env python3
"""
Final corrected extraction script with proper CRS handling.
The issue was that geometry coordinates are already in the raster's CRS (EPSG:32619),
not in EPSG:4326 as assumed.

This function is part of the workflow going from points back to edges. 
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import sys
import time
import yaml
from pathlib import Path
import rasterio
import warnings
from datetime import datetime
import pickle
import glob

warnings.filterwarnings("ignore", category=FutureWarning)

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def deserialize_geometry(geom_obj):
    """Deserialize geometry from bytes or return as-is if already a geometry"""
    if isinstance(geom_obj, bytes):
        try:
            return pickle.loads(geom_obj)
        except:
            try:
                from shapely import wkb
                return wkb.loads(geom_obj)
            except:
                return None
    elif hasattr(geom_obj, 'x') and hasattr(geom_obj, 'y'):
        return geom_obj
    else:
        return None

def extract_shade_value_corrected(row, config, osmid="cbdb17d4"):
    """Extract shade values with correct CRS handling"""
    try:
        tile_id = row['tile_number']
        tile_number = tile_id.split('_')[-1] if '_' in tile_id else tile_id
        
        # Get binned date and time
        binned_date = pd.to_datetime(row['binned_date']).strftime('%Y%m%d')
        rounded_ts = pd.to_datetime(row['rounded_timestamp'])
        time_part = rounded_ts.strftime('%H%M')
        
        # Build shade file path
        base_path = Path(config['output_dir']) / f"step5_shade_results/{osmid}"
        shade_file = f"{base_path}/building_shade/{tile_number}/{osmid}_{tile_id}_Shadow_{binned_date}_{time_part}_LST.tif"
        
        # Find file if exact doesn't exist
        if not os.path.exists(shade_file):
            pattern_dir = f"{base_path}/building_shade/{tile_number}/"
            if os.path.exists(pattern_dir):
                pattern = f"{pattern_dir}{osmid}_{tile_id}_Shadow_{binned_date}_*_LST.tif"
                matches = glob.glob(pattern)
                if matches:
                    shade_file = matches[0]
                else:
                    return None
            else:
                return None
        
        # Get coordinates
        geom = deserialize_geometry(row['geometry'])
        if geom is None:
            # Fallback to lat/lon and transform to raster CRS
            if 'longitude' in row and 'latitude' in row:
                # These are in EPSG:4326, need to transform
                import pyproj
                with rasterio.open(shade_file) as src:
                    transformer = pyproj.Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
                    x, y = transformer.transform(row['longitude'], row['latitude'])
            else:
                return None
        else:
            # Geometry is already in the correct CRS (same as raster)
            x, y = geom.x, geom.y
        
        # Extract values
        with rasterio.open(shade_file) as src:
            # No coordinate transformation needed - geometry is in same CRS as raster
            values = list(src.sample([(x, y)]))
            if values and len(values[0]) > 0:
                shade_value = values[0][0]
                
                # Skip NODATA values
                if shade_value == src.nodata or np.isnan(shade_value):
                    return None
                
                # Extract combined shade
                combined_file = shade_file.replace('/building_shade/', '/combined_shade/')
                combined_value = None
                if os.path.exists(combined_file):
                    with rasterio.open(combined_file) as csrc:
                        combined_values = list(csrc.sample([(x, y)]))
                        if combined_values and len(combined_values[0]) > 0:
                            cv = combined_values[0][0]
                            if cv != csrc.nodata and not np.isnan(cv):
                                combined_value = float(cv)
                
                return {
                    'building_shade_buffer0': float(shade_value),
                    'combined_shade_buffer0': combined_value,
                    'used_file': os.path.basename(shade_file),
                    'extraction_success': True,
                    'coordinates': (x, y)
                }
        
        return None
        
    except Exception as e:
        return {'error': str(e), 'extraction_success': False}

def process_full_extraction(df, config, osmid="cbdb17d4"):
    """Process all missing records"""
    missing_mask = df['building_shade_buffer0'].isna()
    missing_count = missing_mask.sum()
    
    if missing_count == 0:
        print("🎉 All shade values already extracted!")
        return df
    
    print(f"🔍 Processing {missing_count} missing records...")
    
    # Process in batches for progress tracking
    batch_size = 1000
    successful_extractions = 0
    
    for i in range(0, missing_count, batch_size):
        missing_indices = df.index[missing_mask][i:i+batch_size]
        batch_start = time.time()
        
        for idx in missing_indices:
            row = df.loc[idx]
            result = extract_shade_value_corrected(row, config, osmid)
            
            if result and result.get('extraction_success', False):
                # Update the dataframe
                if 'building_shade_buffer0' in result:
                    df.at[idx, 'building_shade_buffer0'] = result['building_shade_buffer0']
                if 'combined_shade_buffer0' in result and result['combined_shade_buffer0'] is not None:
                    df.at[idx, 'combined_shade_buffer0'] = result['combined_shade_buffer0']
                successful_extractions += 1
        
        batch_time = time.time() - batch_start
        current_batch_size = len(missing_indices)
        
        print(f"  Batch {i//batch_size + 1}: Processed {current_batch_size} records in {batch_time:.2f}s")
        print(f"  Success rate: {successful_extractions}/{i+current_batch_size} ({successful_extractions/(i+current_batch_size)*100:.1f}%)")
    
    print(f"✅ Final results: Successfully extracted {successful_extractions}/{missing_count} missing values")
    return df

def main():
    print("🚀 Final Corrected Shade Data Extraction")
    print("=" * 60)
    
    config = load_config()
    osmid = "cbdb17d4"
    
    # Load dataset
    temp_parquet = Path(config['output_dir']) / f"temp_extracted_results_{osmid}.parquet"
    print(f"📊 Loading dataset from {temp_parquet}")
    df = pd.read_parquet(temp_parquet)
    print(f"   Dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Show current status
    missing_count = df['building_shade_buffer0'].isna().sum()
    print(f"📈 Current status: {missing_count} records missing shade values ({missing_count/len(df)*100:.1f}%)")
    
    if missing_count == 0:
        print("🎉 Extraction is already complete!")
        return
    
    # Test on a single record first
    print("\n🧪 Testing extraction on one record...")
    test_idx = df[df['building_shade_buffer0'].isna()].index[0]
    test_row = df.loc[test_idx]
    
    print(f"Test record {test_idx}:")
    print(f"  tile_number: {test_row['tile_number']}")
    print(f"  binned_date: {test_row['binned_date']}")
    print(f"  rounded_timestamp: {test_row['rounded_timestamp']}")
    
    result = extract_shade_value_corrected(test_row, config, osmid)
    if result and result.get('extraction_success', False):
        print(f"✅ Test successful!")
        print(f"  building_shade: {result['building_shade_buffer0']}")
        print(f"  combined_shade: {result.get('combined_shade_buffer0', 'N/A')}")
        print(f"  coordinates: {result.get('coordinates', 'N/A')}")
        
        # Ask user if they want to proceed with full extraction
        print(f"\n🤔 Ready to process all {missing_count} missing records?")
        print("This will take approximately {:.1f} minutes".format(missing_count * 0.001 / 60))  # Estimate
        
        response = input("Proceed? (y/n): ").lower().strip()
        if response == 'y':
            print("\n🚀 Starting full extraction...")
            start_time = time.time()
            
            updated_df = process_full_extraction(df, config, osmid)
            
            total_time = time.time() - start_time
            print(f"✅ Extraction completed in {total_time:.2f}s")
            
            # Save results
            output_file = f"completed_extraction_results_{osmid}.parquet"
            updated_df.to_parquet(output_file)
            print(f"💾 Results saved to {output_file}")
            
            # Show final statistics
            final_missing = updated_df['building_shade_buffer0'].isna().sum()
            improvement = missing_count - final_missing
            print(f"📊 Final statistics:")
            print(f"  Records processed: {missing_count}")
            print(f"  Successfully extracted: {improvement}")
            print(f"  Still missing: {final_missing}")
            print(f"  Success rate: {improvement/missing_count*100:.1f}%")
        else:
            print("❌ Extraction cancelled.")
    else:
        print(f"❌ Test failed: {result}")
        print("Need to debug further before processing all records.")

if __name__ == "__main__":
    main()
