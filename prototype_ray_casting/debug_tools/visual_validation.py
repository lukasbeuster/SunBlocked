"""
Visual Validation of Point-Based Shadow Ray Casting
===================================================

Creates visualizations to validate ray casting accuracy and understand results.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point
import sys
import rasterio
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from dsm_loader import DSMTileManager
from ray_caster import PointShadowCaster
from sun_calculator import generate_sun_positions, sun_ray_direction


def visualize_dsm_tile_and_points(dsm_manager, tile_id, test_points):
    """Visualize a DSM tile with test points overlaid"""
    print(f"📊 Creating visualization for tile {tile_id}...")
    
    # Load tile data
    tile_data = dsm_manager.load_tile(tile_id)
    building_dsm = tile_data['building']
    canopy_dsm = tile_data['canopy']
    combined_dsm = np.maximum(building_dsm, canopy_dsm)
    
    # Get tile bounds
    tile_info = dsm_manager.tiles_info[tile_id]
    minx, miny, maxx, maxy = tile_info.bounds
    
    # Filter points to this tile
    tile_points = []
    tile_point_coords = []
    for point in test_points:
        if minx <= point.x <= maxx and miny <= point.y <= maxy:
            tile_points.append(point)
            # Convert to raster coordinates for plotting
            row, col = rasterio.transform.rowcol(tile_info.transform, point.x, point.y)
            tile_point_coords.append((col, row))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Building DSM
    im1 = axes[0].imshow(building_dsm, cmap='viridis', origin='upper')
    axes[0].set_title(f'Building DSM - Tile {tile_id}')
    plt.colorbar(im1, ax=axes[0], label='Height (m)')
    
    # Add test points
    if tile_point_coords:
        points_x, points_y = zip(*tile_point_coords)
        axes[0].scatter(points_x, points_y, c='red', s=50, marker='x', linewidths=2)
    
    # Canopy DSM
    im2 = axes[1].imshow(canopy_dsm, cmap='Greens', origin='upper')
    axes[1].set_title(f'Canopy DSM - Tile {tile_id}')
    plt.colorbar(im2, ax=axes[1], label='Height (m)')
    
    # Add test points
    if tile_point_coords:
        axes[1].scatter(points_x, points_y, c='red', s=50, marker='x', linewidths=2)
    
    # Combined DSM
    im3 = axes[2].imshow(combined_dsm, cmap='terrain', origin='upper')
    axes[2].set_title(f'Combined DSM - Tile {tile_id}')
    plt.colorbar(im3, ax=axes[2], label='Height (m)')
    
    # Add test points
    if tile_point_coords:
        axes[2].scatter(points_x, points_y, c='red', s=50, marker='x', linewidths=2)
    
    plt.tight_layout()
    plt.savefig(f'dsm_visualization_{tile_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved DSM visualization: dsm_visualization_{tile_id}.png")
    return tile_points, combined_dsm, tile_info


def visualize_ray_casting_process(dsm_manager, point, sun_positions, tile_id):
    """Visualize the ray casting process for a specific point"""
    print(f"🔥 Visualizing ray casting for point at {point.x:.1f}, {point.y:.1f}")
    
    # Load tile data
    tile_data = dsm_manager.load_tile(tile_id)
    combined_dsm = np.maximum(tile_data['building'], tile_data['canopy'])
    tile_info = dsm_manager.tiles_info[tile_id]
    
    # Create ray caster
    ray_caster = PointShadowCaster(dsm_manager)
    
    # Test ray casting for each sun position
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, sun_pos in enumerate(sun_positions[:6]):  # Show first 6 positions
        ax = axes[i]
        
        # Show DSM
        im = ax.imshow(combined_dsm, cmap='terrain', origin='upper', alpha=0.7)
        
        # Convert point to raster coordinates
        point_row, point_col = rasterio.transform.rowcol(tile_info.transform, point.x, point.y)
        
        # Cast shadow ray
        is_shaded = ray_caster.cast_shadow_ray(point, sun_pos, tile_id)
        
        # Visualize ray path
        ray_dx, ray_dy, ray_dz = sun_ray_direction(sun_pos)
        
        # Sample points along the ray
        ray_points_x = []
        ray_points_y = []
        ray_heights = []
        
        max_distance = 200.0  # meters
        step_size = 2.0
        distance = step_size
        
        while distance <= max_distance:
            ray_x = point.x + (ray_dx * distance)
            ray_y = point.y + (ray_dy * distance)
            
            # Convert to raster coordinates
            try:
                ray_row, ray_col = rasterio.transform.rowcol(tile_info.transform, ray_x, ray_y)
                if 0 <= ray_row < combined_dsm.shape[0] and 0 <= ray_col < combined_dsm.shape[1]:
                    ray_points_x.append(ray_col)
                    ray_points_y.append(ray_row)
                    ray_heights.append(combined_dsm[ray_row, ray_col])
            except:
                pass
            
            distance += step_size
        
        # Plot ray path
        if ray_points_x:
            ax.plot(ray_points_x, ray_points_y, 'yellow', linewidth=2, alpha=0.8)
        
        # Mark the test point
        color = 'red' if is_shaded else 'lime'
        ax.scatter([point_col], [point_row], c=color, s=100, marker='o', 
                  edgecolors='black', linewidths=2)
        
        # Set title
        shade_status = "SHADED" if is_shaded else "SUNLIGHT"
        ax.set_title(f'{sun_pos.datetime.strftime("%H:%M")} - {shade_status}\n'
                    f'Sun: {sun_pos.elevation:.1f}° elev, {sun_pos.azimuth:.1f}° az')
        ax.set_xlim(max(0, point_col-100), min(combined_dsm.shape[1], point_col+100))
        ax.set_ylim(max(0, point_row-100), min(combined_dsm.shape[0], point_row+100))
    
    plt.tight_layout()
    plt.savefig(f'ray_casting_visualization_{tile_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved ray casting visualization: ray_casting_visualization_{tile_id}.png")


def create_shade_timeline_chart(results, point_ids):
    """Create a timeline chart showing shade patterns"""
    print("📈 Creating shade timeline chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract timeline data
    for i, result in enumerate(results[:5]):  # Show first 5 points
        times = []
        shade_values = []
        
        for time_point, is_shaded in result.hourly_shade_status:
            hour = time_point.hour + time_point.minute / 60.0
            times.append(hour)
            shade_values.append(1 if is_shaded else 0)
        
        # Offset each point slightly for visibility
        offset_shade_values = [val + i * 0.05 for val in shade_values]
        
        ax.plot(times, offset_shade_values, 'o-', linewidth=2, markersize=8,
                label=f'{point_ids[i]} ({result.total_shade_hours_today:.1f}h shade)')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Shade Status (1=Shaded, 0=Sunlight)')
    ax.set_title('Shadow Timeline for Test Points')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(-0.1, 1.3)
    
    plt.tight_layout()
    plt.savefig('shade_timeline_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved shade timeline chart: shade_timeline_chart.png")


def create_height_profile_analysis(dsm_manager, test_points):
    """Analyze height profiles at test points"""
    print("📏 Creating height profile analysis...")
    
    heights_data = []
    for point in test_points[:10]:  # Sample first 10 points
        tile_id = dsm_manager.find_tile_for_point(point)
        if tile_id:
            heights = dsm_manager.sample_dsm_at_point(point, tile_id)
            heights_data.append(heights)
    
    if not heights_data:
        print("⚠️ No height data available for analysis")
        return
    
    # Extract data for plotting
    building_heights = [h['building'] for h in heights_data]
    canopy_heights = [h['canopy'] for h in heights_data]
    combined_heights = [h['combined'] for h in heights_data]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Height comparison
    x_pos = np.arange(len(heights_data))
    width = 0.25
    
    ax1.bar(x_pos - width, building_heights, width, label='Buildings', color='skyblue')
    ax1.bar(x_pos, canopy_heights, width, label='Canopy', color='green')
    ax1.bar(x_pos + width, combined_heights, width, label='Combined', color='orange')
    
    ax1.set_xlabel('Test Point Index')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Height Comparison at Test Points')
    ax1.legend()
    ax1.set_xticks(x_pos)
    
    # Height distribution
    ax2.hist(combined_heights, bins=10, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Height (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Combined Heights')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('height_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved height analysis: height_analysis.png")
    
    # Print statistics
    print(f"\n📊 Height Statistics:")
    print(f"   Building heights: {np.mean(building_heights):.1f}±{np.std(building_heights):.1f}m")
    print(f"   Canopy heights: {np.mean(canopy_heights):.1f}±{np.std(canopy_heights):.1f}m")
    print(f"   Combined heights: {np.mean(combined_heights):.1f}±{np.std(combined_heights):.1f}m")


def main():
    """Run comprehensive visual validation"""
    print("🎨 VISUAL VALIDATION OF RAY CASTING PROTOTYPE")
    print("=" * 55)
    print()
    
    # Initialize DSM manager
    dsm_dir = "../results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    if not dsm_manager.tiles_info:
        print("❌ No DSM tiles found!")
        return
    
    print(f"✅ Loaded {len(dsm_manager.tiles_info)} DSM tile pairs")
    
    # Generate test points (focused on first few tiles)
    test_points = []
    tile_ids = list(dsm_manager.tiles_info.keys())[:3]
    
    for tile_id in tile_ids:
        tile_info = dsm_manager.tiles_info[tile_id]
        minx, miny, maxx, maxy = tile_info.bounds
        
        # Generate a few points per tile
        for _ in range(3):
            x = np.random.uniform(minx + 100, maxx - 100)
            y = np.random.uniform(miny + 100, maxy - 100)
            test_points.append(Point(x, y))
    
    print(f"📍 Generated {len(test_points)} test points across {len(tile_ids)} tiles")
    
    # 1. Visualize DSM tiles with points
    sample_tile_id = tile_ids[0]
    tile_points, combined_dsm, tile_info = visualize_dsm_tile_and_points(
        dsm_manager, sample_tile_id, test_points)
    
    if not tile_points:
        print("⚠️ No points found in sample tile")
        return
    
    # 2. Generate sun positions for analysis
    test_date = datetime(2024, 6, 21)  # Summer solstice
    lat, lon = 42.36, -71.06  # Boston coordinates
    sun_positions = generate_sun_positions(lat, lon, test_date, 
                                         start_hour=6, end_hour=18, interval_minutes=120)
    
    print(f"☀️ Generated {len(sun_positions)} sun positions")
    
    # 3. Visualize ray casting process
    sample_point = tile_points[0]
    visualize_ray_casting_process(dsm_manager, sample_point, sun_positions, sample_tile_id)
    
    # 4. Run full analysis on test points
    from ray_caster import batch_process_points
    
    print("\n⚡ Running full shade analysis...")
    results = batch_process_points(test_points[:5], test_date, lat, lon, dsm_manager,
                                 [f"point_{i}" for i in range(5)])
    
    # 5. Create timeline visualization
    create_shade_timeline_chart(results, [f"point_{i}" for i in range(5)])
    
    # 6. Height analysis
    create_height_profile_analysis(dsm_manager, test_points)
    
    print(f"\n✅ Visual validation complete!")
    print(f"\nGenerated files:")
    print(f"  - dsm_visualization_{sample_tile_id}.png")
    print(f"  - ray_casting_visualization_{sample_tile_id}.png") 
    print(f"  - shade_timeline_chart.png")
    print(f"  - height_analysis.png")
    print(f"\nThese files show:")
    print(f"  📊 DSM data quality and test point locations")
    print(f"  🔥 Ray casting paths and shadow detection")
    print(f"  📈 Temporal shade patterns")
    print(f"  📏 Height distribution analysis")


if __name__ == "__main__":
    main()
