"""
Debug Shade Metrics Calculation
===============================

Debug and fix the temporal metrics calculation logic.
"""

import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.append('.')

from dsm_loader import DSMTileManager
from ray_caster import PointShadowCaster
from sun_calculator import generate_sun_positions
from shapely.geometry import Point


def debug_temporal_calculation():
    """Debug the temporal metrics calculation step by step"""
    print("🐛 DEBUGGING TEMPORAL METRICS CALCULATION")
    print("=" * 50)
    
    # Create a simple test timeline
    test_timeline = []
    base_time = datetime(2024, 6, 21, 16, 0)  # 4 PM
    
    # Create timeline going backwards (every 30 minutes)
    for i in range(10):  # 10 intervals = 5 hours back
        time_point = base_time - timedelta(minutes=30 * i)
        # All shaded for this test
        test_timeline.insert(0, {
            'datetime': time_point,
            'hour': time_point.hour + time_point.minute / 60.0,
            'is_shaded': True,
            'sun_elevation': 45.0,
            'sun_azimuth': 180.0
        })
    
    print("📅 Test Timeline:")
    for i, record in enumerate(test_timeline):
        print(f"  {i}: {record['datetime'].strftime('%H:%M')} - {'SHADED' if record['is_shaded'] else 'SUN'}")
    
    current_time = base_time  # 16:00
    print(f"\n⏰ Analyzing metrics at {current_time.strftime('%H:%M')}")
    
    # Debug 1-hour window calculation
    print(f"\n🔍 Debugging 1-hour window:")
    cutoff_1h = current_time - timedelta(hours=1)  # 15:00
    print(f"  Cutoff time (1h ago): {cutoff_1h.strftime('%H:%M')}")
    
    intervals_in_1h = []
    for record in test_timeline:
        if record['datetime'] >= cutoff_1h and record['datetime'] <= current_time:
            intervals_in_1h.append(record)
            print(f"    Include: {record['datetime'].strftime('%H:%M')} - {'SHADED' if record['is_shaded'] else 'SUN'}")
    
    shade_count_1h = sum(1 for r in intervals_in_1h if r['is_shaded'])
    total_count_1h = len(intervals_in_1h)
    
    print(f"  Intervals found: {total_count_1h}")
    print(f"  Shaded intervals: {shade_count_1h}")
    print(f"  Shade fraction: {shade_count_1h / total_count_1h if total_count_1h > 0 else 0:.2f}")
    
    # THIS IS THE BUG - I was multiplying by interval hours incorrectly
    interval_duration_hours = 0.5  # 30 minutes = 0.5 hours
    
    print(f"\n❌ INCORRECT calculation (what I was doing):")
    wrong_shade_hours = shade_count_1h * interval_duration_hours
    print(f"  Shade hours = {shade_count_1h} intervals × {interval_duration_hours}h = {wrong_shade_hours}h")
    print(f"  This is wrong because it counts intervals, not actual time!")
    
    print(f"\n✅ CORRECT calculation:")
    # The correct way: actual time span covered by shaded intervals
    if intervals_in_1h:
        first_time = intervals_in_1h[0]['datetime']
        last_time = intervals_in_1h[-1]['datetime']
        actual_time_span = (last_time - first_time).total_seconds() / 3600  # hours
        
        # For a 1-hour window with all shaded, it should be 1.0 hour
        window_hours = 1.0
        shade_fraction = shade_count_1h / total_count_1h if total_count_1h > 0 else 0
        correct_shade_hours = window_hours * shade_fraction
        
        print(f"  Window duration: {window_hours}h")
        print(f"  Shade fraction: {shade_fraction:.2f}")
        print(f"  Shade hours = {window_hours}h × {shade_fraction:.2f} = {correct_shade_hours}h")
    
    print(f"\n🎯 The bug was: I was counting intervals instead of calculating actual time coverage!")


def fixed_temporal_shade_metrics(shade_timeline, current_time):
    """
    FIXED version of temporal shade metrics calculation
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
    
    # 2. FIXED: Shade duration in last 1, 2, 4 hours
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
        
        # FIXED: Calculate shade fraction first, then multiply by window duration
        shade_fraction = shade_count / total_intervals if total_intervals > 0 else 0
        shade_hours = window_hours * shade_fraction  # This is the correct calculation!
        
        shade_durations[f'last_{window_hours}h'] = {
            'shade_hours': shade_hours,
            'total_hours': window_hours,  # This should always equal the window size
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
    
    cumulative_shade_fraction = shade_day_intervals / total_day_intervals if total_day_intervals > 0 else 0
    cumulative_total_hours = (current_time - day_start).total_seconds() / 3600
    cumulative_shade_hours = cumulative_total_hours * cumulative_shade_fraction
    
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


def test_fixed_calculation():
    """Test the fixed calculation with real data"""
    print("\n🧪 TESTING FIXED CALCULATION")
    print("=" * 35)
    
    # Initialize DSM manager
    dsm_dir = "../results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    # Get a test point
    sample_tile_id = list(dsm_manager.tiles_info.keys())[0]
    tile_info = dsm_manager.tiles_info[sample_tile_id]
    minx, miny, maxx, maxy = tile_info.bounds
    
    test_point = Point((minx + maxx) / 2, (miny + maxy) / 2)
    
    # Calculate shade timeline
    test_date = datetime(2024, 6, 21)
    lat, lon = 42.36, -71.06
    
    # Generate sun positions (every 30 minutes)
    sun_positions = generate_sun_positions(lat, lon, test_date, 
                                         start_hour=5, end_hour=21, 
                                         interval_minutes=30)
    
    # Create ray caster and get timeline
    ray_caster = PointShadowCaster(dsm_manager)
    tile_id = dsm_manager.find_tile_for_point(test_point)
    
    shade_timeline = []
    for sun_pos in sun_positions:
        is_shaded = ray_caster.cast_shadow_ray(test_point, sun_pos, tile_id)
        shade_timeline.append({
            'datetime': sun_pos.datetime,
            'hour': sun_pos.datetime.hour + sun_pos.datetime.minute / 60.0,
            'is_shaded': is_shaded,
            'sun_elevation': sun_pos.elevation,
            'sun_azimuth': sun_pos.azimuth
        })
    
    # Test at different times
    test_times = [
        datetime(2024, 6, 21, 8, 0),   # Morning
        datetime(2024, 6, 21, 12, 0),  # Noon
        datetime(2024, 6, 21, 16, 0),  # Afternoon
    ]
    
    for current_time in test_times:
        print(f"\n⏰ FIXED metrics at {current_time.strftime('%H:%M')}:")
        
        metrics = fixed_temporal_shade_metrics(shade_timeline, current_time)
        
        if metrics:
            print(f"   Current shade: {'YES' if metrics['current_shade_status'] else 'NO'}")
            
            for window in ['last_1h', 'last_2h', 'last_4h']:
                window_data = metrics['temporal_windows'][window]
                print(f"   {window.replace('_', ' ').title()}: {window_data['shade_fraction']:.2f} "
                      f"({window_data['shade_hours']:.1f}h shade out of {window_data['total_hours']:.1f}h total)")
            
            dawn_data = metrics['cumulative_since_dawn']
            print(f"   Since dawn: {dawn_data['shade_fraction']:.2f} "
                  f"({dawn_data['shade_hours']:.1f}h out of {dawn_data['total_hours']:.1f}h total)")


def main():
    """Run debugging analysis"""
    print("🔧 SHADE METRICS CALCULATION DEBUG")
    print("=" * 40)
    
    # First debug the calculation logic
    debug_temporal_calculation()
    
    # Then test with real data
    test_fixed_calculation()
    
    print(f"\n✅ Bug identified and fixed!")
    print(f"\n📝 The issue was:")
    print(f"   ❌ Wrong: shade_hours = shade_intervals × interval_duration")
    print(f"   ✅ Right: shade_hours = window_duration × shade_fraction")
    print(f"\n🎯 Now the metrics make sense:")
    print(f"   • Last 1h with 100% shade = 1.0 hours of shade")
    print(f"   • Last 2h with 50% shade = 1.0 hours of shade") 
    print(f"   • Last 4h with 25% shade = 1.0 hours of shade")


if __name__ == "__main__":
    main()
