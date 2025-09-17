#!/usr/bin/env python3
"""
Debug script to investigate why we're getting 100% shade
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from ray_caster_enhanced import ProductionShadowCaster
from dsm_loader_enhanced import DSMTileManager
from sun_calculator import generate_sun_positions, sun_ray_direction
from shapely.geometry import Point
from datetime import datetime
import pyproj

def debug_single_point_analysis():
    """Debug a single point in detail"""
    print("🔍 DEBUGGING SINGLE POINT SHADE ANALYSIS")
    print("=" * 60)
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    # Initialize DSM manager and caster
    dsm_manager = DSMTileManager(dsm_dir)
    caster = ProductionShadowCaster(dsm_manager)
    
    # Test point (center of DSM coverage)
    test_lat, test_lon = 42.268030, -71.073773
    test_point = Point(test_lon, test_lat)
    
    print(f"📍 Test Point: ({test_lat:.6f}, {test_lon:.6f})")
    
    # Convert to UTM
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32619', always_xy=True)
    utm_x, utm_y = transformer.transform(test_lon, test_lat)
    utm_point = Point(utm_x, utm_y)
    
    print(f"📍 UTM Point: ({utm_x:.1f}, {utm_y:.1f})")
    
    # Check tile and height data
    tile_id = dsm_manager.find_tile_for_point(utm_point)
    print(f"🏠 Tile ID: {tile_id}")
    
    if tile_id:
        heights = dsm_manager.sample_dsm_at_point(utm_point, tile_id)
        print(f"📏 Heights: {heights}")
        
        # Test different sun angles
        test_scenarios = [
            (datetime(2024, 6, 21, 12, 0), "Summer noon - high sun"),
            (datetime(2024, 6, 21, 6, 0), "Summer dawn - low sun"),
            (datetime(2024, 12, 21, 12, 0), "Winter noon - medium sun"),
        ]
        
        for test_time, label in test_scenarios:
            print(f"\n🌞 {label}")
            print(f"   Time: {test_time}")
            
            # Get sun position
            sun_positions = generate_sun_positions(test_lat, test_lon, test_time, 
                                                 start_hour=test_time.hour, 
                                                 end_hour=test_time.hour+1, 
                                                 interval_minutes=60)
            
            if sun_positions:
                sun_pos = sun_positions[0]
                print(f"   Sun elevation: {sun_pos.elevation:.1f}°")
                print(f"   Sun azimuth: {sun_pos.azimuth:.1f}°")
                
                if sun_pos.elevation > 0:
                    # Test the ray casting logic step by step
                    print(f"   🔍 Testing ray casting...")
                    
                    # Get ray direction
                    ray_dx, ray_dy, ray_dz = sun_ray_direction(sun_pos)
                    print(f"   Ray direction: dx={ray_dx:.3f}, dy={ray_dy:.3f}, dz={ray_dz:.3f}")
                    
                    # Test ray casting manually
                    ground_height = heights['combined']
                    print(f"   Ground height: {ground_height:.2f}m")
                    
                    # Cast ray with debug info
                    is_shaded = debug_ray_casting(caster, test_point, sun_pos, tile_id, dsm_manager)
                    print(f"   🌳 Final result: {'SHADED' if is_shaded else 'SUNLIT'}")
                    
                    # Also test the full metrics
                    metrics = caster.compute_shade_metrics(test_point, test_time, test_lat, test_lon, "debug_point")
                    print(f"   📊 Full metrics: shade={metrics.current_shade_status}, sun_elev={metrics.current_sun_elevation:.1f}°")
            
    else:
        print("❌ Point outside DSM coverage!")

def debug_ray_casting(caster, latlon_point, sun_pos, tile_id, dsm_manager):
    """Debug ray casting step by step"""
    
    # Convert to UTM
    utm_point = caster._transform_to_utm(latlon_point)
    
    # Get ground height
    heights_at_point = dsm_manager.sample_dsm_at_point(utm_point, tile_id)
    ground_height = heights_at_point['combined']
    
    # Get ray direction
    ray_dx, ray_dy, ray_dz = sun_ray_direction(sun_pos)
    
    print(f"      Starting ray from ground: {ground_height:.2f}m")
    
    # Cast ray with detailed logging
    distance = caster.ray_step_size
    max_canopy_height = 0.0
    has_building_obstruction = False
    obstruction_found = False
    steps_logged = 0
    
    while distance <= min(100.0, caster.max_ray_distance) and steps_logged < 10:  # Limit for debugging
        # Calculate ray position
        ray_x = utm_point.x + (ray_dx * distance)
        ray_y = utm_point.y + (ray_dy * distance)  
        ray_z = ground_height + (ray_dz * distance)
        
        # Sample DSM at ray position
        ray_point = Point(ray_x, ray_y)
        ray_tile_id = dsm_manager.find_tile_for_point(ray_point)
        
        if ray_tile_id is None:
            print(f"      Step {steps_logged}: dist={distance:.1f}m, ray left coverage area")
            break
        
        try:
            heights = dsm_manager.sample_dsm_at_point(ray_point, ray_tile_id)
            building_height = heights['building']
            canopy_height = heights['canopy']
            max_canopy_height = max(max_canopy_height, canopy_height)
            
            if steps_logged < 5:  # Log first few steps
                print(f"      Step {steps_logged}: dist={distance:.1f}m, ray_z={ray_z:.2f}m, building={building_height:.2f}m, canopy={canopy_height:.2f}m")
            
            # Check building obstruction
            if ray_z < building_height:
                print(f"      🏢 Building obstruction at step {steps_logged}! ray_z={ray_z:.2f}m < building={building_height:.2f}m")
                has_building_obstruction = True
                obstruction_found = True
                break
                
            # Check canopy obstruction
            if ray_z < canopy_height:
                trunk_zone_height = max_canopy_height * caster.trunk_zone_threshold
                print(f"      🌳 Canopy obstruction at step {steps_logged}! ray_z={ray_z:.2f}m < canopy={canopy_height:.2f}m")
                print(f"         Trunk zone height: {trunk_zone_height:.2f}m (25% of max {max_canopy_height:.2f}m)")
                
                if ray_z >= trunk_zone_height and not has_building_obstruction:
                    print(f"         ✅ In trunk zone - allowing sunlight through")
                else:
                    print(f"         ❌ Above trunk zone or has building obstruction - shade!")
                    obstruction_found = True
                    break
                    
        except Exception as e:
            print(f"      ⚠️ Sampling error at step {steps_logged}: {e}")
            
        distance += caster.ray_step_size
        steps_logged += 1
    
    if not obstruction_found and steps_logged >= 10:
        print(f"      ✅ No obstructions found in first {steps_logged} steps")
    
    return has_building_obstruction or obstruction_found

def create_height_visualization():
    """Create visualization of DSM heights around test points"""
    print("\n📊 CREATING HEIGHT VISUALIZATION")
    print("=" * 40)
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    # Sample points in a small grid around center
    center_lat, center_lon = 42.268030, -71.073773
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32619', always_xy=True)
    
    # Create 10x10 grid around center point  
    lat_range = np.linspace(center_lat - 0.001, center_lat + 0.001, 10)
    lon_range = np.linspace(center_lon - 0.001, center_lon + 0.001, 10)
    
    building_heights = np.zeros((10, 10))
    canopy_heights = np.zeros((10, 10))
    combined_heights = np.zeros((10, 10))
    
    print("📍 Sampling height data...")
    
    for i, lat in enumerate(lat_range):
        for j, lon in enumerate(lon_range):
            utm_x, utm_y = transformer.transform(lon, lat)
            utm_point = Point(utm_x, utm_y)
            
            tile_id = dsm_manager.find_tile_for_point(utm_point)
            if tile_id:
                try:
                    heights = dsm_manager.sample_dsm_at_point(utm_point, tile_id)
                    building_heights[i, j] = heights['building']
                    canopy_heights[i, j] = heights['canopy'] 
                    combined_heights[i, j] = heights['combined']
                except:
                    building_heights[i, j] = 0
                    canopy_heights[i, j] = 0
                    combined_heights[i, j] = 0
            else:
                building_heights[i, j] = np.nan
                canopy_heights[i, j] = np.nan
                combined_heights[i, j] = np.nan
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(building_heights, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Building Heights (m)')
    axes[0].set_xlabel('Longitude →')
    axes[0].set_ylabel('Latitude →')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(canopy_heights, cmap='Greens', interpolation='nearest') 
    axes[1].set_title('Canopy Heights (m)')
    axes[1].set_xlabel('Longitude →')
    axes[1].set_ylabel('Latitude →')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(combined_heights, cmap='terrain', interpolation='nearest')
    axes[2].set_title('Combined Heights (m)')
    axes[2].set_xlabel('Longitude →')
    axes[2].set_ylabel('Latitude →')
    plt.colorbar(im3, ax=axes[2])
    
    # Mark center point
    center_i, center_j = 5, 5
    for ax in axes:
        ax.plot(center_j, center_i, 'r*', markersize=15, label='Test Point')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('dsm_heights_visualization.png', dpi=150, bbox_inches='tight')
    print("💾 Height visualization saved to: dsm_heights_visualization.png")
    
    # Print statistics
    print(f"\n📊 HEIGHT STATISTICS:")
    print(f"   Building heights: min={np.nanmin(building_heights):.2f}m, max={np.nanmax(building_heights):.2f}m, mean={np.nanmean(building_heights):.2f}m")
    print(f"   Canopy heights: min={np.nanmin(canopy_heights):.2f}m, max={np.nanmax(canopy_heights):.2f}m, mean={np.nanmean(canopy_heights):.2f}m")
    print(f"   Combined heights: min={np.nanmin(combined_heights):.2f}m, max={np.nanmax(combined_heights):.2f}m, mean={np.nanmean(combined_heights):.2f}m")
    
    return building_heights, canopy_heights, combined_heights

def main():
    """Run debug analysis"""
    print("🔧 DEBUGGING SHADE ANALYSIS")
    print("=" * 60)
    
    try:
        # Debug single point in detail
        debug_single_point_analysis()
        
        # Create height visualization
        create_height_visualization()
        
        print(f"\n✅ Debug analysis completed!")
        print(f"Check dsm_heights_visualization.png for height data visualization")
        
    except Exception as e:
        print(f"❌ Debug analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
