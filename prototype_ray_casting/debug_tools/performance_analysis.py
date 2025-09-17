#!/usr/bin/env python3
"""
Critical Performance Analysis of the Shade Ray Casting Method
"""

import time
import sys
sys.path.append('.')

from ray_caster_enhanced import ProductionShadowCaster
from dsm_loader_enhanced import DSMTileManager
from shapely.geometry import Point
from datetime import datetime

def analyze_performance():
    """Analyze the performance characteristics and bottlenecks"""
    print("⚡ CRITICAL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    # Initialize system
    start_time = time.time()
    dsm_manager = DSMTileManager(dsm_dir)
    init_time = time.time() - start_time
    print(f"🔍 DSM initialization: {init_time:.2f} seconds")
    
    caster = ProductionShadowCaster(dsm_manager)
    
    # Test single point performance
    test_point = Point(-71.073773, 42.268030)
    test_time = datetime(2024, 6, 21, 12, 0)
    
    print(f"\n📊 SINGLE POINT ANALYSIS:")
    start = time.time()
    metrics = caster.compute_shade_metrics(test_point, test_time, 42.268030, -71.073773, "perf_test")
    single_point_time = time.time() - start
    print(f"   Single point processing: {single_point_time:.3f} seconds")
    
    # Analyze the computational breakdown
    print(f"\n🔍 COMPUTATIONAL BREAKDOWN ANALYSIS:")
    
    # Ray casting parameters from the system
    ray_step_size = caster.ray_step_size  # 1.0 meter
    max_ray_distance = caster.max_ray_distance  # 1000 meters 
    interval_minutes = 30  # from generate_sun_positions
    daily_hours = 16  # 5am to 21pm = 16 hours
    
    # Calculate computational complexity
    steps_per_ray = int(max_ray_distance / ray_step_size)  # 1000 steps max
    intervals_per_day = int((daily_hours * 60) / interval_minutes)  # 32 intervals
    
    print(f"   Ray step size: {ray_step_size}m")
    print(f"   Max ray distance: {max_ray_distance}m") 
    print(f"   Steps per ray: {steps_per_ray}")
    print(f"   Time intervals per day: {intervals_per_day}")
    print(f"   Total operations per point: {steps_per_ray * intervals_per_day:,}")
    
    # Estimate computational load
    operations_per_point = steps_per_ray * intervals_per_day
    
    print(f"\n📈 SCALABILITY ANALYSIS:")
    test_scenarios = [
        (10, "Small batch"),
        (100, "Medium batch"), 
        (1000, "Large batch"),
        (10000, "Production scale"),
        (100000, "City-wide scale")
    ]
    
    for n_points, scenario in test_scenarios:
        estimated_time = single_point_time * n_points
        total_operations = operations_per_point * n_points
        
        hours = estimated_time / 3600
        print(f"   {scenario:<20}: {n_points:>6,} points → {estimated_time:>8.1f}s ({hours:>5.2f}h) | {total_operations:>12,} ops")
    
    # Memory analysis
    print(f"\n💾 MEMORY USAGE ANALYSIS:")
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"   Current memory usage: {memory_mb:.1f} MB")
    print(f"   DSM tiles loaded: {len(dsm_manager.tiles_info)}")
    
    # Bottleneck identification
    print(f"\n⚠️  PERFORMANCE BOTTLENECKS IDENTIFIED:")
    print(f"   1. 🐌 Ray casting resolution: {ray_step_size}m steps (could increase to 2-5m)")
    print(f"   2. 🐌 Temporal resolution: {interval_minutes}min intervals (could increase to 60min)")
    print(f"   3. 🐌 Ray distance: {max_ray_distance}m max (could reduce to 200-500m)")
    print(f"   4. 🐌 DSM sampling: Individual raster lookups (could optimize)")
    print(f"   5. 🐌 Timeline computation: Full day analysis per point")
    
    # Optimization recommendations
    print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS:")
    
    # Scenario 1: Relaxed precision
    relaxed_steps = int(500 / 5)  # 5m steps, 500m max distance
    relaxed_intervals = int((daily_hours * 60) / 60)  # 60min intervals
    relaxed_ops = relaxed_steps * relaxed_intervals
    speedup_relaxed = operations_per_point / relaxed_ops
    
    print(f"   🎯 RELAXED PRECISION:")
    print(f"      Ray steps: 5m (vs {ray_step_size}m) | Distance: 500m (vs {max_ray_distance}m)")
    print(f"      Time intervals: 60min (vs {interval_minutes}min)")
    print(f"      Operations per point: {relaxed_ops:,} (vs {operations_per_point:,})")
    print(f"      Speedup factor: {speedup_relaxed:.1f}x faster")
    
    # Scenario 2: Optimized precision
    opt_steps = int(200 / 2)  # 2m steps, 200m max distance  
    opt_intervals = int((8 * 60) / 30)  # 8 hours, 30min intervals (business hours)
    opt_ops = opt_steps * opt_intervals
    speedup_opt = operations_per_point / opt_ops
    
    print(f"   🎯 OPTIMIZED PRECISION:")
    print(f"      Ray steps: 2m | Distance: 200m | Hours: 8h business hours")
    print(f"      Operations per point: {opt_ops:,}")
    print(f"      Speedup factor: {speedup_opt:.1f}x faster")
    
    # Real-world deployment scenarios
    print(f"\n🌍 REAL-WORLD DEPLOYMENT SCENARIOS:")
    city_points = 100000
    current_time_hours = (single_point_time * city_points) / 3600
    relaxed_time_hours = current_time_hours / speedup_relaxed
    opt_time_hours = current_time_hours / speedup_opt
    
    print(f"   City-wide analysis ({city_points:,} points):")
    print(f"      Current method: {current_time_hours:.1f} hours ({current_time_hours/24:.1f} days)")
    print(f"      Relaxed method: {relaxed_time_hours:.1f} hours ({relaxed_time_hours/24:.1f} days)")
    print(f"      Optimized method: {opt_time_hours:.1f} hours ({opt_time_hours/24:.1f} days)")
    
    return {
        'single_point_time': single_point_time,
        'operations_per_point': operations_per_point,
        'memory_mb': memory_mb,
        'speedup_relaxed': speedup_relaxed,
        'speedup_optimized': speedup_opt
    }

def main():
    """Run performance analysis"""
    try:
        results = analyze_performance()
        
        print(f"\n✅ PERFORMANCE ANALYSIS COMPLETE")
        print(f"\n📋 SUMMARY:")
        print(f"   Current performance: {results['single_point_time']:.3f}s per point")
        print(f"   Computational complexity: {results['operations_per_point']:,} operations per point")
        print(f"   Memory usage: {results['memory_mb']:.1f} MB")
        print(f"   Optimization potential: {results['speedup_relaxed']:.1f}x - {results['speedup_optimized']:.1f}x speedup")
        
        print(f"\n⚠️  RECOMMENDATION: Current method suitable for research/prototype.")
        print(f"   For production deployment, implement optimized version for {results['speedup_optimized']:.0f}x speedup!")
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
