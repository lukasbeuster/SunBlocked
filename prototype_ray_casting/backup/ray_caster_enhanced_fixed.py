#!/usr/bin/env python3
"""
Enhanced Production Shadow Ray Casting with FIXED coordinate handling

This version properly handles coordinate transformations between lat/lon and UTM.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from shapely.geometry import Point
from datetime import datetime, timedelta
import pyproj

from dsm_loader_enhanced import DSMTileManager
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
    """Production-ready shadow ray casting engine with proper coordinate handling"""
    
    def __init__(self, dsm_manager: DSMTileManager):
        self.dsm_manager = dsm_manager
        self.ray_step_size = 1.0  # meters
        self.max_ray_distance = 1000.0  # meters
        self.trunk_zone_threshold = 0.25  # 25% of tree height
        
        # Set up coordinate transformer for lat/lon to UTM conversion
        # Assuming UTM Zone 19N (EPSG:32619) based on the DSM data
        self.transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32619', always_xy=True)
        
    def _transform_to_utm(self, latlon_point: Point) -> Point:
        """Transform lat/lon point to UTM coordinates"""
        utm_x, utm_y = self.transformer.transform(latlon_point.x, latlon_point.y)
        return Point(utm_x, utm_y)
    
    def cast_shadow_ray(self, latlon_point: Point, sun_pos: SunPosition, 
                       tile_id: str = None) -> bool:
        """
        Cast a shadow ray for a given point and sun position.
        
        Enhanced with trunk zone handling:
        - If ray is below 25% of tree height and no other obstructions exist, 
          consider the point sunlit
        """
        if sun_pos.elevation <= 0:
            return True  # Sun below horizon = shadow
        
        # Convert lat/lon point to UTM for DSM operations
        utm_point = self._transform_to_utm(latlon_point)
        
        # Find relevant tile
        if tile_id is None:
            tile_id = self.dsm_manager.find_tile_for_point(utm_point)
            if tile_id is None:
                return False  # Outside coverage = assume no shadow
        
        # Get ground height at point
        heights_at_point = self.dsm_manager.sample_dsm_at_point(utm_point, tile_id)
        ground_height = heights_at_point['combined']
        
        # Get ray direction (from ground toward sun)
        ray_dx, ray_dy, ray_dz = sun_ray_direction(sun_pos)
        
        # Cast ray and check for obstructions with trunk zone handling
        distance = self.ray_step_size
        max_canopy_height = 0.0  # Track maximum canopy height encountered
        has_building_obstruction = False
        
        while distance <= self.max_ray_distance:
            # Calculate ray position at this distance (in UTM coordinates)
            ray_x = utm_point.x + (ray_dx * distance)
            ray_y = utm_point.y + (ray_dy * distance)
            ray_z = ground_height + (ray_dz * distance)
            
            # Sample DSM at ray position
            ray_point = Point(ray_x, ray_y)
            ray_tile_id = self.dsm_manager.find_tile_for_point(ray_point)
            
            if ray_tile_id is None:
                break  # Ray left coverage area
            
            try:
                heights = self.dsm_manager.sample_dsm_at_point(ray_point, ray_tile_id)
                building_height = heights['building']
                canopy_height = heights['canopy'] 
                max_canopy_height = max(max_canopy_height, canopy_height)
                
                # Check building obstruction (definitive shadow)
                if ray_z < building_height:
                    has_building_obstruction = True
                    break
                    
                # Check canopy obstruction (with trunk zone handling)
                if ray_z < canopy_height:
                    # Apply trunk zone rule: if ray is below 25% of max tree height,
                    # and there are no building obstructions, consider it sunlit
                    trunk_zone_height = max_canopy_height * self.trunk_zone_threshold
                    if ray_z >= trunk_zone_height or has_building_obstruction:
                        return True  # Shadow from canopy
                    
            except Exception:
                # Handle sampling errors gracefully
                continue
                
            distance += self.ray_step_size
        
        return has_building_obstruction  # Only shadowed if building obstruction found

    def compute_shade_metrics(self, point: Point, current_time: datetime, 
                            lat: float, lon: float, 
                            point_id: str = "unknown") -> ShadeMetrics:
        """
        Compute comprehensive shade metrics for a point at a specific time
        
        Args:
            point: GPS coordinate (lat/lon) to analyze
            current_time: Current datetime to compute metrics for
            lat: Latitude for sun calculations
            lon: Longitude for sun calculations  
            point_id: Identifier for the point
            
        Returns:
            ShadeMetrics with all computed values
        """
        # Convert to UTM for DSM operations
        utm_point = self._transform_to_utm(point)
        
        # Find tile for this point
        tile_id = self.dsm_manager.find_tile_for_point(utm_point)
        if tile_id is None:
            # Return default metrics if outside coverage, but with proper sun calculations
            return self._create_default_metrics(point, point_id, current_time, lat, lon)
        
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
                                         shade_timeline, sun_positions, lat, lon)
    
    def _compute_fixed_metrics(self, point: Point, point_id: str, current_time: datetime,
                             shade_timeline: List[Tuple[datetime, bool]],
                             sun_positions: List[SunPosition], lat: float, lon: float) -> ShadeMetrics:
        """Compute metrics using FIXED temporal calculation"""
        
        if not shade_timeline:
            return self._create_default_metrics(point, point_id, current_time, lat, lon)
        
        # Find current sun position and shade status - FIXED VERSION
        current_shade = False
        current_sun_elevation = 0.0
        current_sun_azimuth = 0.0
        
        # Get the current sun position directly
        current_sun_positions = generate_sun_positions(lat, lon, current_time, 
                                                      start_hour=current_time.hour, 
                                                      end_hour=current_time.hour+1, 
                                                      interval_minutes=60)
        
        if current_sun_positions:
            current_sun_pos = current_sun_positions[0]
            current_sun_elevation = current_sun_pos.elevation
            current_sun_azimuth = current_sun_pos.azimuth
        
        # Get current shade status from timeline
        for timeline_time, is_shaded in shade_timeline:
            if timeline_time <= current_time:
                current_shade = is_shaded
        
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
                shade_hours = window_hours * shade_fraction
            else:
                shade_fraction = 0.0
                shade_hours = 0.0
            
            window_metrics[f'last_{window_hours}h'] = {
                'shade_fraction': shade_fraction,
                'shade_hours': shade_hours
            }
        
        # Cumulative metrics since dawn
        if shade_timeline:
            dawn_time = shade_timeline[0][0]
            total_intervals = len(shade_timeline)
            total_shade_intervals = sum(1 for _, is_shaded in shade_timeline if is_shaded)
            
            cumulative_shade_fraction = total_shade_intervals / total_intervals if total_intervals > 0 else 0.0
            
            # Calculate actual hours since dawn
            cumulative_total_hours = (current_time - dawn_time).total_seconds() / 3600
            cumulative_shade_hours = cumulative_total_hours * cumulative_shade_fraction
        else:
            cumulative_shade_fraction = 0.0
            cumulative_shade_hours = 0.0
            cumulative_total_hours = 0.0
        
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
                               current_time: datetime, lat: float, lon: float) -> ShadeMetrics:
        """Create default metrics for points outside coverage with proper sun calculations"""
        
        # Calculate proper sun position even for points outside DSM coverage
        current_sun_positions = generate_sun_positions(lat, lon, current_time, 
                                                      start_hour=current_time.hour, 
                                                      end_hour=current_time.hour+1, 
                                                      interval_minutes=60)
        
        current_sun_elevation = 0.0
        current_sun_azimuth = 0.0
        
        if current_sun_positions:
            current_sun_pos = current_sun_positions[0]
            current_sun_elevation = current_sun_pos.elevation
            current_sun_azimuth = current_sun_pos.azimuth
        
        return ShadeMetrics(
            point_id=point_id,
            coordinate=point,
            timestamp=current_time,
            current_shade_status=False,  # Assume no shade if outside coverage
            current_sun_elevation=current_sun_elevation,
            current_sun_azimuth=current_sun_azimuth,
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
    
    This is the main entry point for production batch processing.
    """
    caster = ProductionShadowCaster(dsm_manager)
    
    if point_ids is None:
        point_ids = [f"point_{i}" for i in range(len(points))]
    
    results = []
    for i, point in enumerate(points):
        point_id = point_ids[i] if i < len(point_ids) else f"point_{i}"
        metrics = caster.compute_shade_metrics(point, current_time, lat, lon, point_id)
        results.append(metrics)
    
    return results


if __name__ == "__main__":
    # Test the fixed production version
    print("🔥 Testing FIXED PRODUCTION Shadow Ray Casting")
    
    from dsm_loader_enhanced import DSMTileManager
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    # Test point within DSM coverage
    test_point = Point(-71.073773, 42.268030)  # lat/lon
    test_time = datetime(2024, 6, 21, 12, 0)
    
    caster = ProductionShadowCaster(dsm_manager)
    metrics = caster.compute_shade_metrics(test_point, test_time, 42.268030, -71.073773, "test_point")
    
    print(f"✅ Test Results:")
    print(f"   Sun elevation: {metrics.current_sun_elevation:.1f}°")
    print(f"   Sun azimuth: {metrics.current_sun_azimuth:.1f}°")
    print(f"   Shade status: {'SHADED' if metrics.current_shade_status else 'SUNLIT'}")
