"""
Sun Position Calculator for Shadow Ray Casting
==============================================

Calculates sun positions for shadow ray casting based on date, time, and location.
Uses the same solar calculations as your current pipeline for consistency.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, NamedTuple


class SunPosition(NamedTuple):
    """Sun position with elevation and azimuth angles"""
    elevation: float  # Degrees above horizon
    azimuth: float    # Degrees from north (clockwise)
    datetime: datetime


def julian_day(date: datetime) -> float:
    """Calculate Julian day number"""
    a = (14 - date.month) // 12
    y = date.year + 4800 - a
    m = date.month + 12 * a - 3
    return date.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def sun_position(lat: float, lon: float, dt: datetime) -> SunPosition:
    """
    Calculate sun position for given location and time
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees  
        dt: Datetime (assumed to be local solar time)
        
    Returns:
        SunPosition with elevation and azimuth angles
    """
    # Convert to radians
    lat_rad = np.radians(lat)
    
    # Calculate Julian day
    jd = julian_day(dt)
    
    # Days since J2000.0
    n = jd - 2451545.0
    
    # Mean longitude of sun
    L = (280.460 + 0.9856474 * n) % 360
    
    # Mean anomaly
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    
    # Ecliptic longitude
    lambda_sun = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    
    # Declination
    delta = np.arcsin(0.39782 * np.sin(lambda_sun))
    
    # Hour angle (approximate, assumes local solar time)
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_angle = np.radians(15 * (hour - 12))
    
    # Solar elevation
    elevation = np.arcsin(
        np.sin(delta) * np.sin(lat_rad) + 
        np.cos(delta) * np.cos(lat_rad) * np.cos(hour_angle)
    )
    
    # Solar azimuth
    azimuth = np.arctan2(
        -np.sin(hour_angle),
        np.tan(delta) * np.cos(lat_rad) - np.sin(lat_rad) * np.cos(hour_angle)
    )
    
    # Convert to degrees
    elevation_deg = np.degrees(elevation)
    azimuth_deg = (np.degrees(azimuth) + 180) % 360  # Convert to 0-360 from north
    
    return SunPosition(elevation_deg, azimuth_deg, dt)


def generate_sun_positions(lat: float, lon: float, date: datetime, 
                          start_hour: int = 6, end_hour: int = 18, 
                          interval_minutes: int = 60) -> List[SunPosition]:
    """
    Generate sun positions for a day at regular intervals
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        date: Date to calculate for
        start_hour: Start hour (default: 6 AM)
        end_hour: End hour (default: 6 PM) 
        interval_minutes: Interval in minutes (default: 60)
        
    Returns:
        List of SunPosition objects
    """
    positions = []
    
    current_time = date.replace(hour=start_hour, minute=0, second=0)
    end_time = date.replace(hour=end_hour, minute=0, second=0)
    interval = timedelta(minutes=interval_minutes)
    
    while current_time <= end_time:
        sun_pos = sun_position(lat, lon, current_time)
        
        # Only include positions where sun is above horizon
        if sun_pos.elevation > 0:
            positions.append(sun_pos)
            
        current_time += interval
    
    return positions


def sun_ray_direction(sun_pos: SunPosition) -> Tuple[float, float, float]:
    """
    Calculate 3D ray direction vector from sun position
    
    Args:
        sun_pos: Sun position with elevation and azimuth
        
    Returns:
        (dx, dy, dz) normalized direction vector
    """
    # Convert to radians
    elevation_rad = np.radians(sun_pos.elevation)
    azimuth_rad = np.radians(sun_pos.azimuth)
    
    # Calculate direction vector (pointing FROM sun TO ground)
    dx = -np.sin(azimuth_rad) * np.cos(elevation_rad)
    dy = -np.cos(azimuth_rad) * np.cos(elevation_rad) 
    dz = -np.sin(elevation_rad)
    
    return dx, dy, dz


if __name__ == "__main__":
    # Test sun calculations
    print("🌞 Testing sun position calculations...")
    
    # Example: Boston coordinates
    lat, lon = 42.3601, -71.0589
    test_date = datetime(2024, 6, 21, 12, 0)  # Summer solstice, noon
    
    sun_pos = sun_position(lat, lon, test_date)
    print(f"Sun position at {test_date}: {sun_pos.elevation:.1f}° elevation, {sun_pos.azimuth:.1f}° azimuth")
    
    # Generate daily positions
    positions = generate_sun_positions(lat, lon, test_date)
    print(f"Generated {len(positions)} sun positions for the day")
    
    # Test ray direction
    ray_dir = sun_ray_direction(sun_pos)
    print(f"Ray direction: ({ray_dir[0]:.3f}, {ray_dir[1]:.3f}, {ray_dir[2]:.3f})")
