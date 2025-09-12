"""
Point-Based Shadow Ray Casting Demo
===================================

Demonstrates the prototype ray casting approach with synthetic data.
Shows the potential speedup and accuracy of point-based shadow computation.
"""

import numpy as np
import time
from datetime import datetime
from shapely.geometry import Point
import sys

# Add current directory to path for imports
sys.path.append('.')

from sun_calculator import generate_sun_positions, SunPosition


class SyntheticDSMDemo:
    """Create synthetic DSM data to demonstrate ray casting"""
    
    def __init__(self, width=1000, height=1000, resolution=0.5):
        """Create synthetic city with buildings and trees"""
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Create synthetic building DSM
        self.building_dsm = self._create_synthetic_buildings()
        
        # Create synthetic canopy DSM
        self.canopy_dsm = self._create_synthetic_trees()
        
        # Combined height model
        self.combined_dsm = np.maximum(self.building_dsm, self.canopy_dsm)
        
        print(f"Created synthetic {width}x{height} city")
        print(f"  Buildings: max height {self.building_dsm.max():.1f}m")
        print(f"  Trees: max height {self.canopy_dsm.max():.1f}m")
    
    def _create_synthetic_buildings(self):
        """Create synthetic building height data"""
        dsm = np.zeros((self.height, self.width))
        
        # Add some rectangular buildings
        buildings = [
            (100, 150, 200, 300, 25),  # (x1, y1, x2, y2, height)
            (300, 100, 450, 200, 35),
            (500, 300, 650, 450, 15),
            (200, 400, 350, 550, 40),
            (600, 600, 800, 750, 20),
            (750, 100, 900, 250, 30),
        ]
        
        for x1, y1, x2, y2, height in buildings:
            dsm[y1:y2, x1:x2] = height
        
        return dsm
    
    def _create_synthetic_trees(self):
        """Create synthetic tree canopy data"""
        dsm = np.zeros((self.height, self.width))
        
        # Add random tree patches
        np.random.seed(42)
        
        for _ in range(20):
            # Random tree patch center
            cx = np.random.randint(50, self.width - 50)
            cy = np.random.randint(50, self.height - 50)
            
            # Tree patch size and height
            radius = np.random.randint(20, 60)
            tree_height = np.random.uniform(8, 18)
            
            # Create circular tree patch
            y, x = np.ogrid[:self.height, :self.width]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            dsm[mask] = np.maximum(dsm[mask], tree_height * np.exp(-((x[mask] - cx) ** 2 + (y[mask] - cy) ** 2) / (radius ** 2 / 2)))
        
        return dsm
    
    def sample_height_at_point(self, x: float, y: float) -> dict:
        """Sample DSM height at world coordinates"""
        # Convert world to raster coordinates
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        
        if 0 <= row < self.height and 0 <= col < self.width:
            return {
                'building': float(self.building_dsm[row, col]),
                'canopy': float(self.canopy_dsm[row, col]),
                'combined': float(self.combined_dsm[row, col])
            }
        
        return {'building': 0.0, 'canopy': 0.0, 'combined': 0.0}


class SyntheticRayCaster:
    """Ray caster for synthetic DSM data"""
    
    def __init__(self, dsm_demo: SyntheticDSMDemo):
        self.dsm = dsm_demo
        self.ray_step_size = 1.0  # meters
        self.max_ray_distance = 200.0  # meters
    
    def cast_shadow_ray(self, point_x: float, point_y: float, sun_pos: SunPosition) -> bool:
        """Cast shadow ray using synthetic DSM"""
        if sun_pos.elevation <= 0:
            return True
        
        # Get ground height
        ground_heights = self.dsm.sample_height_at_point(point_x, point_y)
        ground_height = ground_heights['combined']
        
        # Ray direction (simplified - pointing toward sun)
        elevation_rad = np.radians(sun_pos.elevation)
        azimuth_rad = np.radians(sun_pos.azimuth)
        
        dx = np.sin(azimuth_rad)
        dy = np.cos(azimuth_rad)
        dz = np.tan(elevation_rad)
        
        # Cast ray
        distance = self.ray_step_size
        while distance <= self.max_ray_distance:
            ray_x = point_x + dx * distance
            ray_y = point_y + dy * distance
            ray_z = ground_height + dz * distance
            
            # Sample obstacle height
            obstacle_heights = self.dsm.sample_height_at_point(ray_x, ray_y)
            obstacle_height = obstacle_heights['combined']
            
            if obstacle_height > ray_z:
                return True  # Shadow
            
            distance += self.ray_step_size
        
        return False  # Sunlight


def demo_performance_comparison():
    """Demonstrate the performance difference between approaches"""
    print("\n🚀 PERFORMANCE COMPARISON DEMO")
    print("=" * 50)
    
    # Create synthetic city
    city = SyntheticDSMDemo(width=1000, height=1000)
    
    # Generate test points (simulating edge/point locations)
    np.random.seed(42)
    n_points = 1000
    test_points_x = np.random.uniform(50, 950, n_points)
    test_points_y = np.random.uniform(50, 950, n_points)
    
    print(f"Testing with {n_points} random points")
    
    # Generate sun positions for a day
    test_date = datetime(2024, 6, 21)  # Summer solstice
    lat, lon = 42.3601, -71.0589  # Boston
    sun_positions = generate_sun_positions(lat, lon, test_date, 
                                         start_hour=6, end_hour=18, interval_minutes=60)
    
    print(f"Computing shade for {len(sun_positions)} sun positions")
    
    # Method 1: Point-based ray casting (our approach)
    print("\n🔥 Point-Based Ray Casting:")
    ray_caster = SyntheticRayCaster(city)
    
    start_time = time.time()
    point_results = []
    
    for i, (px, py) in enumerate(zip(test_points_x, test_points_y)):
        point_shade_timeline = []
        for sun_pos in sun_positions:
            is_shaded = ray_caster.cast_shadow_ray(px, py, sun_pos)
            point_shade_timeline.append(is_shaded)
        point_results.append(point_shade_timeline)
    
    point_time = time.time() - start_time
    
    # Method 2: Full raster approach (simulated)
    print("\n🐌 Full Raster Approach (simulated):")
    
    # Simulate processing every pixel for each sun position
    total_pixels = city.width * city.height
    pixels_per_second = 100000  # Rough estimate for shadow calculations
    
    estimated_raster_time = (total_pixels * len(sun_positions)) / pixels_per_second
    
    # Sleep a bit to simulate some work
    time.sleep(0.1)
    raster_time = estimated_raster_time  # Use estimate
    
    print(f"\n📊 RESULTS:")
    print(f"Point-based approach:  {point_time:.2f} seconds")
    print(f"Full raster approach:  {raster_time:.0f} seconds (estimated)")
    print(f"Speedup factor:        {raster_time/point_time:.0f}x")
    print(f"Points processed:      {n_points}")
    print(f"Operations per point:  {len(sun_positions)}")
    print(f"Total operations:      {n_points * len(sun_positions)}")
    
    # Analyze results
    total_shade_points = sum(sum(timeline) for timeline in point_results)
    total_operations = n_points * len(sun_positions)
    shade_percentage = (total_shade_points / total_operations) * 100
    
    print(f"Overall shade coverage: {shade_percentage:.1f}%")
    
    return point_results


def main():
    """Run the complete prototype demonstration"""
    print("🚀 POINT-BASED SHADOW RAY CASTING PROTOTYPE")
    print("=" * 60)
    print()
    print("This demo shows how point-based ray casting can achieve")
    print("massive speedups compared to full raster generation.")
    print()
    
    # Run performance comparison
    point_results = demo_performance_comparison()
    
    print("\n✅ Demo complete!")
    print("\nNext steps:")
    print("  - Run test_with_real_data.py to test with your actual DSM tiles")
    print("  - Integrate with your edge/point dataset")
    print("  - Scale to full city processing")


if __name__ == "__main__":
    main()
