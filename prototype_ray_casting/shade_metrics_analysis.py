"""
Comprehensive Shade Metrics Analysis
===================================

Provides the exact shade metrics needed for thermal drift analysis:
- Current shade status (binary at timestep)
- Shade duration in last 1, 2, 4 hours
- Cumulative shade fraction since start of day
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from shapely.geometry import Point
import sys

sys.path.append('.')

from dsm_loader import DSMTileManager
from ray_caster import PointShadowCaster, batch_process_points
from sun_calculator import generate_sun_positions


def calculate_detailed_shade_metrics(point, date, lat, lon, dsm_manager, point_id="test_point"):
    """Calculate comprehensive shade metrics for a specific point"""
    
    # Generate high-resolution sun positions (every 30 minutes)
    sun_positions = generate_sun_positions(lat, lon, date, 
                                         start_hour=5, end_hour=21, 
                                         interval_minutes=30)
    
    # Create ray caster
    ray_caster = PointShadowCaster(dsm_manager)
    
    # Find tile for this point
    tile_id = dsm_manager.find_tile_for_point(point)
    if tile_id is None:
        print(f"⚠️ Point {point_id} outside tile coverage")
        return None
    
    # Calculate shade status for each time step
    shade_timeline = []
    for sun_pos in sun_positions:
        is_shaded = ray_caster.cast_shadow_ray(point, sun_pos, tile_id)
        shade_timeline.append({
            'datetime': sun_pos.datetime,
            'hour': sun_pos.datetime.hour + sun_pos.datetime.minute / 60.0,
            'is_shaded': is_shaded,
            'sun_elevation': sun_pos.elevation,
            'sun_azimuth': sun_pos.azimuth
        })
    
    return shade_timeline


def compute_temporal_shade_metrics(shade_timeline, current_time):
    """
    Compute shade metrics for different time windows
    
    Args:
        shade_timeline: List of shade status records
        current_time: Current datetime to compute metrics for
        
    Returns:
        Dictionary with all temporal shade metrics
    """
    if not shade_timeline:
        return None
    
    # Find current time index
    current_idx = None
    for i, record in enumerate(shade_timeline):
        if record['datetime'] <= current_time:
            current_idx = i
        else:
            break
    
    if current_idx is None:
        current_idx = 0
    
    current_record = shade_timeline[current_idx]
    
    # 1. Current shade status (binary)
    current_shade = current_record['is_shaded']
    
    # 2. Shade duration in last 1, 2, 4 hours
    time_windows = [1, 2, 4]  # hours
    shade_durations = {}
    
    for window_hours in time_windows:
        cutoff_time = current_time - timedelta(hours=window_hours)
        
        # Count shade intervals in the time window
        shade_count = 0
        total_intervals = 0
        
        for record in shade_timeline:
            if record['datetime'] >= cutoff_time and record['datetime'] <= current_time:
                total_intervals += 1
                if record['is_shaded']:
                    shade_count += 1
        
        # Convert to hours (assuming 30-minute intervals)
        interval_hours = 0.5
        shade_hours = shade_count * interval_hours
        total_hours = total_intervals * interval_hours
        shade_fraction = shade_count / total_intervals if total_intervals > 0 else 0
        
        shade_durations[f'last_{window_hours}h'] = {
            'shade_hours': shade_hours,
            'total_hours': total_hours,
            'shade_fraction': shade_fraction
        }
    
    # 3. Cumulative shade fraction since start of day
    day_start = current_time.replace(hour=5, minute=0, second=0, microsecond=0)
    
    total_day_intervals = 0
    shade_day_intervals = 0
    
    for record in shade_timeline:
        if record['datetime'] >= day_start and record['datetime'] <= current_time:
            total_day_intervals += 1
            if record['is_shaded']:
                shade_day_intervals += 1
    
    cumulative_shade_hours = shade_day_intervals * 0.5  # 30-min intervals
    cumulative_total_hours = total_day_intervals * 0.5
    cumulative_shade_fraction = shade_day_intervals / total_day_intervals if total_day_intervals > 0 else 0
    
    return {
        'current_time': current_time,
        'current_shade_status': current_shade,
        'current_sun_elevation': current_record['sun_elevation'],
        'current_sun_azimuth': current_record['sun_azimuth'],
        'temporal_windows': shade_durations,
        'cumulative_since_dawn': {
            'shade_hours': cumulative_shade_hours,
            'total_hours': cumulative_total_hours,
            'shade_fraction': cumulative_shade_fraction
        }
    }


def create_corrected_timeline_visualization(shade_timelines, point_ids, metrics_list):
    """Create corrected visualization with proper shade values (0 or 1 only)"""
    print("📊 Creating corrected shade timeline visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Raw shade timeline (binary 0/1 values)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (timeline, point_id, color) in enumerate(zip(shade_timelines, point_ids, colors)):
        times = [record['hour'] for record in timeline]
        shade_values = [1 if record['is_shaded'] else 0 for record in timeline]
        
        # Offset each line slightly for visibility, but keep within 0-1 range
        offset = i * 0.02  # Small offset
        offset_shade_values = [val + offset for val in shade_values]
        
        ax1.plot(times, offset_shade_values, 'o-', color=color, linewidth=2, 
                markersize=6, label=f'{point_id}', alpha=0.8)
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Shade Status')
    ax1.set_title('Raw Shade Timeline (Binary: 1=Shaded, 0=Sunlight)')
    ax1.set_ylim(-0.05, 1.15)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Cumulative shade metrics throughout day
    for i, (timeline, point_id, color, metrics) in enumerate(zip(shade_timelines, point_ids, colors, metrics_list)):
        if not metrics:
            continue
            
        # Calculate cumulative shade fraction at each time point
        cumulative_fractions = []
        for j, record in enumerate(timeline):
            current_time = record['datetime']
            temp_metrics = compute_temporal_shade_metrics(timeline, current_time)
            if temp_metrics:
                cumulative_fractions.append(temp_metrics['cumulative_since_dawn']['shade_fraction'])
            else:
                cumulative_fractions.append(0)
        
        times = [record['hour'] for record in timeline]
        ax2.plot(times, cumulative_fractions, '-', color=color, linewidth=3, 
                label=f'{point_id}', alpha=0.8)
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Cumulative Shade Fraction')
    ax2.set_title('Cumulative Shade Fraction Since Dawn')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('corrected_shade_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved corrected timeline: corrected_shade_timeline.png")


def create_metrics_summary_table(metrics_list, point_ids):
    """Create a summary table of all shade metrics"""
    print("\n📋 SHADE METRICS SUMMARY TABLE")
    print("=" * 80)
    
    # Header
    print(f"{'Point':<12} {'Current':<8} {'Last 1h':<10} {'Last 2h':<10} {'Last 4h':<10} {'Since Dawn':<12}")
    print(f"{'ID':<12} {'Shade':<8} {'Fraction':<10} {'Fraction':<10} {'Fraction':<10} {'Fraction':<12}")
    print("-" * 80)
    
    for point_id, metrics in zip(point_ids, metrics_list):
        if not metrics:
            print(f"{point_id:<12} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
            continue
        
        current_shade = "YES" if metrics['current_shade_status'] else "NO"
        last_1h = metrics['temporal_windows']['last_1h']['shade_fraction']
        last_2h = metrics['temporal_windows']['last_2h']['shade_fraction'] 
        last_4h = metrics['temporal_windows']['last_4h']['shade_fraction']
        since_dawn = metrics['cumulative_since_dawn']['shade_fraction']
        
        print(f"{point_id:<12} {current_shade:<8} {last_1h:<10.2f} {last_2h:<10.2f} {last_4h:<10.2f} {since_dawn:<12.2f}")
    
    print("-" * 80)


def demonstrate_metrics_at_different_times():
    """Demonstrate how metrics change throughout the day"""
    print("\n🕐 METRICS EVOLUTION THROUGHOUT DAY")
    print("=" * 45)
    
    # Initialize DSM manager
    dsm_dir = "../results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    # Get a test point
    sample_tile_id = list(dsm_manager.tiles_info.keys())[0]
    tile_info = dsm_manager.tiles_info[sample_tile_id]
    minx, miny, maxx, maxy = tile_info.bounds
    
    test_point = Point((minx + maxx) / 2, (miny + maxy) / 2)
    
    # Calculate shade timeline for the day
    test_date = datetime(2024, 6, 21)
    lat, lon = 42.36, -71.06
    
    shade_timeline = calculate_detailed_shade_metrics(
        test_point, test_date, lat, lon, dsm_manager, "demo_point"
    )
    
    if not shade_timeline:
        print("⚠️ Could not calculate shade timeline")
        return
    
    # Show metrics at key times throughout the day
    key_times = [
        datetime(2024, 6, 21, 8, 0),   # Morning
        datetime(2024, 6, 21, 12, 0),  # Noon
        datetime(2024, 6, 21, 16, 0),  # Afternoon
        datetime(2024, 6, 21, 20, 0),  # Evening
    ]
    
    for current_time in key_times:
        metrics = compute_temporal_shade_metrics(shade_timeline, current_time)
        
        if metrics:
            print(f"\n⏰ Metrics at {current_time.strftime('%H:%M')}:")
            print(f"   Current shade: {'YES' if metrics['current_shade_status'] else 'NO'}")
            print(f"   Sun elevation: {metrics['current_sun_elevation']:.1f}°")
            
            for window in ['last_1h', 'last_2h', 'last_4h']:
                window_data = metrics['temporal_windows'][window]
                print(f"   {window.replace('_', ' ').title()}: {window_data['shade_fraction']:.2f} "
                      f"({window_data['shade_hours']:.1f}h shade)")
            
            dawn_data = metrics['cumulative_since_dawn']
            print(f"   Since dawn: {dawn_data['shade_fraction']:.2f} "
                  f"({dawn_data['shade_hours']:.1f}h total shade)")


def main():
    """Run comprehensive shade metrics analysis"""
    print("📊 COMPREHENSIVE SHADE METRICS ANALYSIS")
    print("=" * 50)
    
    # Initialize DSM manager
    dsm_dir = "../results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    if not dsm_manager.tiles_info:
        print("❌ No DSM tiles found!")
        return
    
    print(f"✅ Loaded {len(dsm_manager.tiles_info)} DSM tile pairs")
    
    # Generate test points
    test_points = []
    point_ids = []
    tile_ids = list(dsm_manager.tiles_info.keys())[:2]
    
    for i, tile_id in enumerate(tile_ids):
        tile_info = dsm_manager.tiles_info[tile_id]
        minx, miny, maxx, maxy = tile_info.bounds
        
        for j in range(2):  # 2 points per tile
            x = np.random.uniform(minx + 200, maxx - 200)
            y = np.random.uniform(miny + 200, maxy - 200)
            test_points.append(Point(x, y))
            point_ids.append(f"point_{i}_{j}")
    
    print(f"📍 Generated {len(test_points)} test points")
    
    # Calculate detailed shade timelines
    test_date = datetime(2024, 6, 21)
    lat, lon = 42.36, -71.06
    
    shade_timelines = []
    metrics_list = []
    
    for point, point_id in zip(test_points, point_ids):
        print(f"🔄 Processing {point_id}...")
        timeline = calculate_detailed_shade_metrics(point, test_date, lat, lon, dsm_manager, point_id)
        shade_timelines.append(timeline)
        
        if timeline:
            # Calculate metrics for afternoon (16:00)
            current_time = datetime(2024, 6, 21, 16, 0)
            metrics = compute_temporal_shade_metrics(timeline, current_time)
            metrics_list.append(metrics)
        else:
            metrics_list.append(None)
    
    # Create corrected visualizations
    create_corrected_timeline_visualization(shade_timelines, point_ids, metrics_list)
    
    # Create metrics summary
    create_metrics_summary_table(metrics_list, point_ids)
    
    # Demonstrate temporal evolution
    demonstrate_metrics_at_different_times()
    
    print(f"\n✅ Shade metrics analysis complete!")
    print(f"\nKey outputs:")
    print(f"  📊 corrected_shade_timeline.png - Proper binary shade visualization")
    print(f"  📋 Metrics table - All required shade metrics")
    print(f"  🕐 Temporal evolution - How metrics change throughout day")
    
    print(f"\n🎯 REQUIRED METRICS PROVIDED:")
    print(f"  ✅ Current shade status (binary at timestep)")
    print(f"  ✅ Shade duration in last 1, 2, 4 hours")
    print(f"  ✅ Cumulative shade fraction since start of day")


if __name__ == "__main__":
    main()
