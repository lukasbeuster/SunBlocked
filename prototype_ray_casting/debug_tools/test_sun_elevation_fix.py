#!/usr/bin/env python3
"""
Test script to verify sun elevation calculations are working properly
"""

import sys
sys.path.append('.')

from ray_caster_enhanced import ProductionShadowCaster
from dsm_loader_enhanced import DSMTileManager
from shapely.geometry import Point
from datetime import datetime
import pyproj

def test_sun_elevation_fix():
    """Test the sun elevation issue with proper coordinate handling"""
    print("🔧 TESTING SUN ELEVATION FIX")
    print("=" * 50)
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    # Initialize DSM manager
    print("🔍 Loading DSM manager...")
    dsm_manager = DSMTileManager(dsm_dir)
    caster = ProductionShadowCaster(dsm_manager)
    
    # Test coordinates (in lat/lon) that we know are within DSM bounds
    test_lat, test_lon = 42.268030, -71.073773
    
    # Convert to UTM for DSM tile lookup
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32619', always_xy=True)
    utm_x, utm_y = transformer.transform(test_lon, test_lat)
    
    print(f"📍 Test coordinates:")
    print(f"   Lat/Lon: ({test_lat}, {test_lon})")
    print(f"   UTM: ({utm_x:.1f}, {utm_y:.1f})")
    
    # Create UTM point for DSM lookup
    utm_point = Point(utm_x, utm_y)
    latlon_point = Point(test_lon, test_lat)
    
    # Test tile lookup with UTM coordinates
    tile_id = dsm_manager.find_tile_for_point(utm_point)
    print(f"   Tile found: {tile_id}")
    
    if tile_id:
        heights = dsm_manager.sample_dsm_at_point(utm_point, tile_id)
        print(f"   Heights at point: {heights}")
    
    # Test times
    test_times = [
        (datetime(2024, 6, 21, 12, 0), "Summer noon"),
        (datetime(2024, 12, 21, 12, 0), "Winter noon"), 
        (datetime(2024, 6, 21, 8, 0), "Summer morning"),
        (datetime(2024, 6, 21, 18, 0), "Summer evening"),
    ]
    
    print(f"\n🌞 Testing sun position calculations:")
    
    for test_time, label in test_times:
        print(f"\n⏰ {label} ({test_time})")
        
        # Test direct sun calculation
        from sun_calculator import generate_sun_positions
        sun_positions = generate_sun_positions(test_lat, test_lon, test_time, 
                                               start_hour=test_time.hour, 
                                               end_hour=test_time.hour+1, 
                                               interval_minutes=60)
        
        if sun_positions:
            sun_pos = sun_positions[0]
            print(f"   Direct calculation: Elevation {sun_pos.elevation:.1f}°, Azimuth {sun_pos.azimuth:.1f}°")
        
        # Test through ray caster (this is what's currently broken)
        try:
            # We need to pass the lat/lon point for sun calculations, but handle UTM conversion internally
            metrics = caster.compute_shade_metrics(
                point=latlon_point,  # Pass lat/lon point  
                current_time=test_time,
                lat=test_lat,
                lon=test_lon,
                point_id=f"test_{test_time.hour}"
            )
            
            print(f"   Ray caster result: Elevation {metrics.current_sun_elevation:.1f}°, Azimuth {metrics.current_sun_azimuth:.1f}°")
            print(f"   Shade status: {'SHADED' if metrics.current_shade_status else 'SUNLIT'}")
            
        except Exception as e:
            print(f"   Ray caster error: {e}")
    
    print(f"\n✅ Sun elevation fix test completed!")

if __name__ == "__main__":
    test_sun_elevation_fix()
