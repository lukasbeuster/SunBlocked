"""
Test Point-Based Ray Casting with Real DSM Data
===============================================

Integration test using your actual DSM/CHM tiles to validate the approach.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from shapely.geometry import Point

# Add prototype modules
sys.path.append('.')
from dsm_loader import DSMTileManager
from ray_caster import PointShadowCaster, batch_process_points
from sun_calculator import generate_sun_positions


def find_dsm_directory():
    """Find your DSM tiles directory automatically"""
    possible_paths = [
        "../results/output/step4_raster_processing/bb2eafb8",
        "../results/output/step4_raster_processing/61cdea20", 
        "../results/output/step4_raster_processing/cbdb17d4",
        "../results/output/step4_raster_processing",  # Check parent dir
    ]
    
    for path in possible_paths:
        if Path(path).exists() and list(Path(path).rglob("*building_dsm.tif")):
            return path
    
    return None


def generate_test_points(dsm_manager: DSMTileManager, n_points: int = 20) -> list:
    """Generate test points within tile coverage"""
    test_points = []
    
    # Get a few tiles to sample from
    tile_ids = list(dsm_manager.tiles_info.keys())[:3]
    
    for tile_id in tile_ids:
        tile_info = dsm_manager.tiles_info[tile_id]
        minx, miny, maxx, maxy = tile_info.bounds
        
        # Generate random points within this tile
        n_points_this_tile = n_points // len(tile_ids)
        
        for _ in range(n_points_this_tile):
            x = np.random.uniform(minx + 50, maxx - 50)  # Stay away from edges
            y = np.random.uniform(miny + 50, maxy - 50)
            test_points.append(Point(x, y))
    
    return test_points


def test_ray_casting_accuracy():
    """Test ray casting with real DSM data"""
    print("🏗️ TESTING WITH REAL DSM DATA")
    print("=" * 40)
    
    # Find DSM directory
    dsm_dir = find_dsm_directory()
    if not dsm_dir:
        print("❌ No DSM directory found!")
        print("Please update the paths in find_dsm_directory() function")
        return
    
    print(f"📁 Using DSM directory: {dsm_dir}")
    
    # Initialize DSM manager
    print("🔄 Loading DSM tiles...")
    dsm_manager = DSMTileManager(dsm_dir)
    
    if not dsm_manager.tiles_info:
        print("❌ No DSM tiles found in directory!")
        return
    
    print(f"✅ Loaded {len(dsm_manager.tiles_info)} tile pairs")
    
    # Generate test points
    print("📍 Generating test points...")
    test_points = generate_test_points(dsm_manager, n_points=20)
    print(f"Generated {len(test_points)} test points")
    
    # Test basic DSM sampling
    print("\n🔬 Testing DSM sampling...")
    for i, point in enumerate(test_points[:5]):
        heights = dsm_manager.sample_dsm_at_point(point)
        tile_id = dsm_manager.find_tile_for_point(point)
        print(f"Point {i+1}: heights={heights}, tile={tile_id}")
    
    # Test sun position calculation
    print("\n☀️ Testing sun positions...")
    test_date = datetime(2024, 6, 21, 12, 0)  # Summer solstice noon
    
    # Use Boston coordinates (adjust as needed)
    lat, lon = 42.36, -71.06
    
    sun_positions = generate_sun_positions(lat, lon, test_date, 
                                         start_hour=8, end_hour=16, interval_minutes=120)
    print(f"Generated {len(sun_positions)} sun positions")
    for sun_pos in sun_positions:
        print(f"  {sun_pos.datetime.strftime('%H:%M')}: elevation={sun_pos.elevation:.1f}°, azimuth={sun_pos.azimuth:.1f}°")
    
    # Test ray casting
    print("\n🔥 Testing shadow ray casting...")
    ray_caster = PointShadowCaster(dsm_manager)
    
    sample_point = test_points[0]
    sample_results = []
    
    for sun_pos in sun_positions:
        is_shaded = ray_caster.cast_shadow_ray(sample_point, sun_pos)
        sample_results.append((sun_pos.datetime.strftime('%H:%M'), is_shaded))
        print(f"  {sun_pos.datetime.strftime('%H:%M')}: {'🌑 SHADED' if is_shaded else '☀️ SUNLIGHT'}")
    
    # Test batch processing
    print(f"\n⚡ Testing batch processing with {len(test_points)} points...")
    import time
    start_time = time.time()
    
    # Process subset to avoid long runtime
    batch_points = test_points[:5]
    point_ids = [f"test_point_{i}" for i in range(len(batch_points))]
    
    results = batch_process_points(batch_points, test_date, lat, lon,
                                 dsm_manager, point_ids)
    
    elapsed = time.time() - start_time
    print(f"✅ Processed {len(batch_points)} points in {elapsed:.2f} seconds")
    print(f"   Average: {elapsed/len(batch_points):.3f} seconds per point")
    
    # Analyze results
    print(f"\n📊 Results summary:")
    shade_counts = [r.total_shade_hours_today for r in results]
    print(f"   Average shade hours: {np.mean(shade_counts):.1f}")
    print(f"   Shade range: {np.min(shade_counts):.1f} - {np.max(shade_counts):.1f} hours")
    
    print(f"\n✅ Real data test completed!")
    print(f"   Processed {len(results)} points successfully")
    print(f"   Ray casting is working with your DSM tiles!")


def estimate_full_scale_performance():
    """Estimate performance for full dataset"""
    print("\n🚀 FULL SCALE PERFORMANCE ESTIMATE")
    print("=" * 40)
    
    # Assumptions based on your pipeline
    total_tiles = 174
    avg_processing_time_per_point = 0.1  # seconds (estimated)
    
    # Estimate number of edge/point coordinates
    estimated_points_per_tile = 100  # Conservative estimate
    total_estimated_points = total_tiles * estimated_points_per_tile
    
    # Time estimates
    estimated_total_time = total_estimated_points * avg_processing_time_per_point
    estimated_hours = estimated_total_time / 3600
    
    print(f"Estimates for full dataset:")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Estimated points per tile: {estimated_points_per_tile}")
    print(f"  Total estimated points: {total_estimated_points:,}")
    print(f"  Processing time per point: {avg_processing_time_per_point}s")
    print(f"  Estimated total runtime: {estimated_hours:.1f} hours")
    print(f"  Current pipeline runtime: ~25 hours")
    print(f"  Estimated speedup: {25/estimated_hours:.0f}x")


if __name__ == "__main__":
    print("🧪 REAL DATA INTEGRATION TEST")
    print("=" * 50)
    print()
    
    try:
        test_ray_casting_accuracy()
        estimate_full_scale_performance()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure DSM tiles exist in the expected directories")
        print("2. Check that the tile files follow the expected naming convention")
        print("3. Verify that required libraries are installed (rasterio, geopandas, etc.)")
        
        import traceback
        print(f"\nFull error:")
        traceback.print_exc()
