"""
Point-Based Shadow Ray Casting Engine
=====================================

Core shadow ray casting implementation that works with DSM/CHM tiles.
Casts rays from specific GPS coordinates to determine shade status.
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
    """Comprehensive shade metrics for a point"""
    point_id: str
    coordinate: Point
    
    # Current shade status
    is_shaded_now: bool
    current_sun_position: SunPosition
    
    # Cumulative metrics
    total_shade_hours_today: float
    shade_fraction_last_2h: float
    shade_fraction_last_4h: float
    
    # Detailed time series
    hourly_shade_status: List[Tuple[datetime, bool]]
    
    # Shadow characteristics  
    longest_continuous_shade_minutes: int
    shortest_sun_exposure_minutes: int


class PointShadowCaster:
    """Main shadow ray casting engine"""
    
    def __init__(self, dsm_manager: DSMTileManager):
        """Initialize shadow caster"""
        self.dsm_manager = dsm_manager
        self.ray_step_size = 2.0  # meters
        self.max_ray_distance = 500.0  # meters
        
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
    
    def compute_daily_shade_metrics(self, point: Point, date: datetime, 
                                  lat: float, lon: float, 
                                  point_id: str = "unknown") -> ShadeMetrics:
        """Compute comprehensive shade metrics for a point on a given date"""
        # Generate sun positions throughout the day
        sun_positions = generate_sun_positions(lat, lon, date, 
                                             start_hour=5, end_hour=19, 
                                             interval_minutes=30)
        
        # Find tile for this point
        tile_id = self.dsm_manager.find_tile_for_point(point)
        if tile_id is None:
            # Return default metrics if outside coverage
            return self._create_default_metrics(point, point_id, date)
        
        # Cast rays for each sun position
        shade_timeline = []
        for sun_pos in sun_positions:
            is_shaded = self.cast_shadow_ray(point, sun_pos, tile_id)
            shade_timeline.append((sun_pos.datetime, is_shaded))
        
        # Compute metrics
        return self._compute_metrics_from_timeline(
            point, point_id, shade_timeline, sun_positions
        )
    
    def _create_default_metrics(self, point: Point, point_id: str, 
                               date: datetime) -> ShadeMetrics:
        """Create default metrics for points outside coverage"""
        return ShadeMetrics(
            point_id=point_id,
            coordinate=point,
            is_shaded_now=False,
            current_sun_position=SunPosition(45.0, 180.0, date),
            total_shade_hours_today=0.0,
            shade_fraction_last_2h=0.0,
            shade_fraction_last_4h=0.0,
            hourly_shade_status=[],
            longest_continuous_shade_minutes=0,
            shortest_sun_exposure_minutes=0
        )
    
    def _compute_metrics_from_timeline(self, point: Point, point_id: str,
                                     shade_timeline: List[Tuple[datetime, bool]],
                                     sun_positions: List[SunPosition]) -> ShadeMetrics:
        """Compute shade metrics from timeline of shade status"""
        
        if not shade_timeline:
            return self._create_default_metrics(point, point_id, datetime.now())
        
        # Total shade hours
        total_shade_intervals = sum(1 for _, is_shaded in shade_timeline if is_shaded)
        interval_hours = 0.5  # 30 minutes
        total_shade_hours = total_shade_intervals * interval_hours
        
        # Current status (latest time)
        current_time, current_shade = shade_timeline[-1]
        current_sun_pos = sun_positions[-1] if sun_positions else SunPosition(0, 0, current_time)
        
        # Last 2 hours and 4 hours metrics
        now = current_time
        last_2h_cutoff = now - timedelta(hours=2)
        last_4h_cutoff = now - timedelta(hours=4)
        
        recent_2h = [shaded for time, shaded in shade_timeline if time >= last_2h_cutoff]
        recent_4h = [shaded for time, shaded in shade_timeline if time >= last_4h_cutoff]
        
        shade_fraction_2h = sum(recent_2h) / len(recent_2h) if recent_2h else 0.0
        shade_fraction_4h = sum(recent_4h) / len(recent_4h) if recent_4h else 0.0
        
        # Continuous shade periods
        continuous_periods = self._find_continuous_periods(shade_timeline)
        longest_shade = max((end - start for start, end, is_shaded in continuous_periods 
                           if is_shaded), default=timedelta(0))
        shortest_sun = min((end - start for start, end, is_shaded in continuous_periods 
                          if not is_shaded), default=timedelta(0))
        
        return ShadeMetrics(
            point_id=point_id,
            coordinate=point,
            is_shaded_now=current_shade,
            current_sun_position=current_sun_pos,
            total_shade_hours_today=total_shade_hours,
            shade_fraction_last_2h=shade_fraction_2h,
            shade_fraction_last_4h=shade_fraction_4h,
            hourly_shade_status=shade_timeline,
            longest_continuous_shade_minutes=int(longest_shade.total_seconds() / 60),
            shortest_sun_exposure_minutes=int(shortest_sun.total_seconds() / 60)
        )
    
    def _find_continuous_periods(self, timeline: List[Tuple[datetime, bool]]) -> List[Tuple[datetime, datetime, bool]]:
        """Find continuous periods of shade/sun"""
        if not timeline:
            return []
        
        periods = []
        current_start = timeline[0][0]
        current_state = timeline[0][1]
        
        for time, state in timeline[1:]:
            if state != current_state:
                # State changed, end current period
                periods.append((current_start, time, current_state))
                current_start = time
                current_state = state
        
        # Add final period
        periods.append((current_start, timeline[-1][0], current_state))
        
        return periods


def batch_process_points(points: List[Point], date: datetime, lat: float, lon: float,
                        dsm_manager: DSMTileManager, 
                        point_ids: Optional[List[str]] = None) -> List[ShadeMetrics]:
    """Process multiple points efficiently"""
    if point_ids is None:
        point_ids = [f"point_{i}" for i in range(len(points))]
    
    caster = PointShadowCaster(dsm_manager)
    results = []
    
    print(f"🔥 Processing {len(points)} points for date {date.date()}")
    
    for i, (point, point_id) in enumerate(zip(points, point_ids)):
        if i % 100 == 0 and i > 0:
            print(f"   Processed {i}/{len(points)} points...")
        
        metrics = caster.compute_daily_shade_metrics(point, date, lat, lon, point_id)
        results.append(metrics)
    
    print(f"✅ Completed processing {len(points)} points")
    return results


if __name__ == "__main__":
    print("🔥 Testing shadow ray casting...")
    print("Note: Use prototype_demo.py for complete test with synthetic data")
