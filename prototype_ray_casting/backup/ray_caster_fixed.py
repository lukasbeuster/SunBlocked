"""
Point-Based Shadow Ray Casting Engine - PRODUCTION VERSION
==========================================================

Core shadow ray casting implementation with FIXED temporal metrics calculation.
Provides exact shade metrics needed for thermal drift analysis.

FIXES APPLIED:
- Corrected temporal metrics calculation (window_duration × shade_fraction)
- Binary shade values only (0 or 1)
- Proper time window handling
- Accurate cumulative calculations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Point
from datetime import datetime, timedelta

from dsm_loader import DSMTileManager
from sun_calculator import SunPosition, generate_sun_positions, sun_ray_direction


@dataclass
class ShadeMetrics:
    """Comprehensive shade metrics for thermal drift analysis"""
    point_id: str
    coordinate: Point
    timestamp: datetime
    
    # Current shade status (binary)
    current_shade_status: bool
    current_sun_elevation: float
    current_sun_azimuth: float
    
    # Temporal window metrics (FIXED calculation)
    last_1h_shade_fraction: float
    last_1h_shade_hours: float
    last_2h_shade_fraction: float 
    last_2h_shade_hours: float
    last_4h_shade_fraction: float
    last_4h_shade_hours: float
    
    # Cumulative metrics since dawn
    cumulative_shade_fraction: float
    cumulative_shade_hours: float
    cumulative_total_hours: float
    
    # Full timeline for debugging/analysis
    shade_timeline: List[Tuple[datetime, bool]]


class ProductionShadowCaster:
    """Production-ready shadow ray casting engine"""
    
    def __init__(self, dsm_manager: DSMTileManager, ray_step_size: float = 2.0, 
                 max_ray_distance: float = 500.0):
        """
        Initialize shadow caster
        
        Args:
            dsm_manager: DSM tile manager for height data
            ray_step_size: Ray sampling step size in meters
            max_ray_distance: Maximum ray distance in meters
        """
        self.dsm_manager = dsm_manager
        self.ray_step_size = ray_step_size
        self.max_ray_distance = max_ray_distance
        
    def cast_shadow_ray(self, point: Point, sun_pos: SunPosition, 
                       tile_id: Optional[str] = None) -> bool:
        """Cast a shadow ray from point toward sun to check for obstructions"""
        if sun_pos.elevation <= 0:
            return True  # Sun below horizon = shadow
        
        # Find relevant tile
        if tile_id is None:
            tile_id = self.dsm_manager.find_tile_for_point(point)
            if tile_id is None:
                return False  # Outside coverage = assume no shadow
        
        # Get ground height at point
        heights_at_point = self.dsm_manager.sample_dsm_at_point(point, tile_id)
        ground_height = heights_at_point['combined']
        
        # Get ray direction (from ground toward sun)
        ray_dx, ray_dy, ray_dz = sun_ray_direction(sun_pos)
        
        # Cast ray and check for obstructions
        distance = self.ray_step_size
        while distance <= self.max_ray_distance:
            # Calculate ray position at this distance
            ray_x = point.x + (ray_dx * distance)
            ray_y = point.y + (ray_dy * distance)
            ray_z = ground_height + (ray_dz * distance)
            
            # Sample obstacle height at ray position
            ray_point = Point(ray_x, ray_y)
            obstacle_heights = self.dsm_manager.sample_dsm_at_point(ray_point)
            obstacle_height = obstacle_heights.get('combined', 0.0)
            
            # Check if ray is blocked
            if obstacle_height > ray_z:
                return True  # Ray blocked = point is in shadow
            
            distance += self.ray_step_size
        
        return False  # Ray reached max distance unobstructed
    
    def compute_shade_metrics(self, point: Point, current_time: datetime, 
                            lat: float, lon: float, 
                            point_id: str = "unknown") -> ShadeMetrics:
        """
        Compute comprehensive shade metrics for a point at a specific time
        
        Args:
            point: GPS coordinate to analyze
            current_time: Current datetime to compute metrics for
            lat: Latitude for sun calculations
            lon: Longitude for sun calculations  
            point_id: Identifier for the point
            
        Returns:
            ShadeMetrics with all computed values
        """
        # Find tile for this point
        tile_id = self.dsm_manager.find_tile_for_point(point)
        if tile_id is None:
            # Return default metrics if outside coverage
            return self._create_default_metrics(point, point_id, current_time)
        
        # Generate detailed timeline (every 30 minutes from dawn to dusk)
        date = current_time.date()
        sun_positions = generate_sun_positions(lat, lon, datetime.combine(date, datetime.min.time()), 
                                             start_hour=5, end_hour=21, 
                                             interval_minutes=30)
        
        # Calculate shade status for each time point up to current time
        shade_timeline = []
        for sun_pos in sun_positions:
            if sun_pos.datetime <= current_time:
                is_shaded = self.cast_shadow_ray(point, sun_pos, tile_id)
                shade_timeline.append((sun_pos.datetime, is_shaded))
        
        # Compute all metrics using FIXED calculation
        return self._compute_fixed_metrics(point, point_id, current_time, 
                                         shade_timeline, sun_positions)
    
    def _compute_fixed_metrics(self, point: Point, point_id: str, current_time: datetime,
                             shade_timeline: List[Tuple[datetime, bool]],
                             sun_positions: List[SunPosition]) -> ShadeMetrics:
        """Compute metrics using FIXED temporal calculation"""
        
        if not shade_timeline:
            return self._create_default_metrics(point, point_id, current_time)
        
        # Find current sun position and shade status
        current_shade = False
        current_sun_elevation = 0.0
        current_sun_azimuth = 0.0
        
        for i, (timeline_time, is_shaded) in enumerate(shade_timeline):
            if timeline_time <= current_time:
                current_shade = is_shaded
                if i < len(sun_positions):
                    current_sun_elevation = sun_positions[i].elevation
                    current_sun_azimuth = sun_positions[i].azimuth
        
        # FIXED: Compute temporal window metrics
        time_windows = [1, 2, 4]  # hours
        window_metrics = {}
        
        for window_hours in time_windows:
            cutoff_time = current_time - timedelta(hours=window_hours)
            
            # Count intervals in the time window
            shade_count = 0
            total_intervals = 0
            
            for timeline_time, is_shaded in shade_timeline:
                if timeline_time >= cutoff_time and timeline_time <= current_time:
                    total_intervals += 1
                    if is_shaded:
                        shade_count += 1
            
            # FIXED calculation: shade_hours = window_duration × shade_fraction
            if total_intervals > 0:
                shade_fraction = shade_count / total_intervals
                shade_hours = window_hours * shade_fraction  # CORRECT!
            else:
                shade_fraction = 0.0
                shade_hours = 0.0
            
            window_metrics[f'last_{window_hours}h'] = {
                'shade_fraction': shade_fraction,
                'shade_hours': shade_hours
            }
        
        # FIXED: Cumulative metrics since dawn
        dawn_time = current_time.replace(hour=5, minute=0, second=0, microsecond=0)
        
        total_day_intervals = 0
        shade_day_intervals = 0
        
        for timeline_time, is_shaded in shade_timeline:
            if timeline_time >= dawn_time and timeline_time <= current_time:
                total_day_intervals += 1
                if is_shaded:
                    shade_day_intervals += 1
        
        if total_day_intervals > 0:
            cumulative_shade_fraction = shade_day_intervals / total_day_intervals
        else:
            cumulative_shade_fraction = 0.0
        
        cumulative_total_hours = (current_time - dawn_time).total_seconds() / 3600
        cumulative_shade_hours = cumulative_total_hours * cumulative_shade_fraction
        
        return ShadeMetrics(
            point_id=point_id,
            coordinate=point,
            timestamp=current_time,
            current_shade_status=current_shade,
            current_sun_elevation=current_sun_elevation,
            current_sun_azimuth=current_sun_azimuth,
            last_1h_shade_fraction=window_metrics['last_1h']['shade_fraction'],
            last_1h_shade_hours=window_metrics['last_1h']['shade_hours'],
            last_2h_shade_fraction=window_metrics['last_2h']['shade_fraction'],
            last_2h_shade_hours=window_metrics['last_2h']['shade_hours'],
            last_4h_shade_fraction=window_metrics['last_4h']['shade_fraction'],
            last_4h_shade_hours=window_metrics['last_4h']['shade_hours'],
            cumulative_shade_fraction=cumulative_shade_fraction,
            cumulative_shade_hours=cumulative_shade_hours,
            cumulative_total_hours=cumulative_total_hours,
            shade_timeline=shade_timeline
        )
    
    def _create_default_metrics(self, point: Point, point_id: str, 
                               current_time: datetime) -> ShadeMetrics:
        """Create default metrics for points outside coverage"""
        return ShadeMetrics(
            point_id=point_id,
            coordinate=point,
            timestamp=current_time,
            current_shade_status=False,
            current_sun_elevation=0.0,
            current_sun_azimuth=0.0,
            last_1h_shade_fraction=0.0,
            last_1h_shade_hours=0.0,
            last_2h_shade_fraction=0.0,
            last_2h_shade_hours=0.0,
            last_4h_shade_fraction=0.0,
            last_4h_shade_hours=0.0,
            cumulative_shade_fraction=0.0,
            cumulative_shade_hours=0.0,
            cumulative_total_hours=0.0,
            shade_timeline=[]
        )


def batch_process_points_production(points: List[Point], current_time: datetime, 
                                   lat: float, lon: float,
                                   dsm_manager: DSMTileManager, 
                                   point_ids: Optional[List[str]] = None) -> List[ShadeMetrics]:
    """
    PRODUCTION version: Process multiple points efficiently with fixed metrics
    
    Args:
        points: List of GPS coordinates
        current_time: Current datetime to compute metrics for
        lat: Latitude for sun calculations
        lon: Longitude for sun calculations
        dsm_manager: DSM tile manager
        point_ids: Optional list of point identifiers
        
    Returns:
        List of ShadeMetrics with FIXED calculations
    """
    if point_ids is None:
        point_ids = [f"point_{i}" for i in range(len(points))]
    
    caster = ProductionShadowCaster(dsm_manager)
    results = []
    
    print(f"🔥 Processing {len(points)} points at {current_time.strftime('%Y-%m-%d %H:%M')}")
    
    for i, (point, point_id) in enumerate(zip(points, point_ids)):
        if i % 50 == 0 and i > 0:
            print(f"   Processed {i}/{len(points)} points...")
        
        metrics = caster.compute_shade_metrics(point, current_time, lat, lon, point_id)
        results.append(metrics)
    
    print(f"✅ Completed processing {len(points)} points")
    return results


def export_metrics_to_csv(metrics_list: List[ShadeMetrics], output_file: str):
    """Export shade metrics to CSV for thermal drift analysis"""
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'point_id', 'timestamp', 'longitude', 'latitude',
            'current_shade_status', 'current_sun_elevation', 'current_sun_azimuth',
            'last_1h_shade_fraction', 'last_1h_shade_hours',
            'last_2h_shade_fraction', 'last_2h_shade_hours', 
            'last_4h_shade_fraction', 'last_4h_shade_hours',
            'cumulative_shade_fraction', 'cumulative_shade_hours', 'cumulative_total_hours'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for metrics in metrics_list:
            writer.writerow({
                'point_id': metrics.point_id,
                'timestamp': metrics.timestamp.isoformat(),
                'longitude': metrics.coordinate.x,
                'latitude': metrics.coordinate.y,
                'current_shade_status': int(metrics.current_shade_status),
                'current_sun_elevation': metrics.current_sun_elevation,
                'current_sun_azimuth': metrics.current_sun_azimuth,
                'last_1h_shade_fraction': metrics.last_1h_shade_fraction,
                'last_1h_shade_hours': metrics.last_1h_shade_hours,
                'last_2h_shade_fraction': metrics.last_2h_shade_fraction,
                'last_2h_shade_hours': metrics.last_2h_shade_hours,
                'last_4h_shade_fraction': metrics.last_4h_shade_fraction,
                'last_4h_shade_hours': metrics.last_4h_shade_hours,
                'cumulative_shade_fraction': metrics.cumulative_shade_fraction,
                'cumulative_shade_hours': metrics.cumulative_shade_hours,
                'cumulative_total_hours': metrics.cumulative_total_hours
            })
    
    print(f"✅ Exported {len(metrics_list)} metrics to {output_file}")


if __name__ == "__main__":
    # Test the production version
    print("🔥 Testing PRODUCTION Shadow Ray Casting with FIXED metrics")
    
    from dsm_loader import DSMTileManager
    
    # Initialize DSM manager  
    dsm_dir = "../results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    if dsm_manager.tiles_info:
        # Test with a few points
        sample_tile_id = list(dsm_manager.tiles_info.keys())[0]
        tile_info = dsm_manager.tiles_info[sample_tile_id]
        minx, miny, maxx, maxy = tile_info.bounds
        
        test_points = [
            Point((minx + maxx) / 2, (miny + maxy) / 2),
            Point(minx + 100, miny + 100)
        ]
        
        current_time = datetime(2024, 6, 21, 16, 0)
        lat, lon = 42.36, -71.06
        
        # Process points
        results = batch_process_points_production(
            test_points, current_time, lat, lon, dsm_manager, 
            ["test_point_1", "test_point_2"]
        )
        
        # Display results
        print(f"\n📊 PRODUCTION RESULTS:")
        for metrics in results:
            print(f"\n{metrics.point_id}:")
            print(f"  Current shade: {'YES' if metrics.current_shade_status else 'NO'}")
            print(f"  Last 1h: {metrics.last_1h_shade_fraction:.2f} ({metrics.last_1h_shade_hours:.1f}h)")
            print(f"  Last 2h: {metrics.last_2h_shade_fraction:.2f} ({metrics.last_2h_shade_hours:.1f}h)")
            print(f"  Last 4h: {metrics.last_4h_shade_fraction:.2f} ({metrics.last_4h_shade_hours:.1f}h)")
            print(f"  Since dawn: {metrics.cumulative_shade_fraction:.2f} ({metrics.cumulative_shade_hours:.1f}h)")
        
        # Export to CSV
        export_metrics_to_csv(results, "production_shade_metrics.csv")
        
        print(f"\n✅ Production testing complete!")
    else:
        print("❌ No DSM tiles found for testing")
