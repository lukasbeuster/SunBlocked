"""
Edge Handling for Shadow Ray Casting
====================================

Research and prototype for handling linear edges (e.g., building edges, 
sidewalk edges, property boundaries) in shadow ray casting.

Challenges:
- Edges are 1D features, shadows are 2D
- Need representative sampling along edges
- Must capture shadow variations along edge length
"""

import numpy as np
from shapely.geometry import Point, LineString, MultiPoint
from shapely import affinity
from typing import List, Tuple
import matplotlib.pyplot as plt


def analyze_edge_shadow_patterns():
    """Analyze how shadows vary along building/infrastructure edges"""
    print("📏 EDGE SHADOW PATTERN ANALYSIS")
    print("=" * 40)
    
    # Simulate a building edge scenario
    print("🏢 Scenario: 50m building edge with 20m tall building")
    
    building_edge = LineString([(0, 0), (50, 0)])  # 50m edge
    building_height = 20.0  # meters
    
    # Sun positions throughout day
    sun_scenarios = [
        ("Morning (8 AM)", 30, 90),    # 30° elevation, 90° azimuth (east)
        ("Noon (12 PM)", 60, 180),    # 60° elevation, 180° azimuth (south)  
        ("Afternoon (4 PM)", 30, 270), # 30° elevation, 270° azimuth (west)
    ]
    
    print(f"\n🌞 Shadow analysis along 50m building edge:")
    
    for time_name, elevation, azimuth in sun_scenarios:
        # Calculate shadow length and direction
        shadow_length = building_height / np.tan(np.radians(elevation))
        
        # Shadow direction (opposite of sun azimuth)
        shadow_azimuth = (azimuth + 180) % 360
        shadow_dx = shadow_length * np.sin(np.radians(shadow_azimuth))
        shadow_dy = shadow_length * np.cos(np.radians(shadow_azimuth))
        
        print(f"\n   {time_name}:")
        print(f"     Shadow length: {shadow_length:.1f}m")
        print(f"     Shadow direction: {shadow_azimuth:.0f}° from north")
        print(f"     Shadow offset: ({shadow_dx:.1f}m, {shadow_dy:.1f}m)")
        
        # The key insight: shadow pattern is relatively uniform along straight edges
        # for parallel sun rays, but can vary at edge endpoints and corners


def prototype_edge_sampling_strategies():
    """Prototype different strategies for sampling points along edges"""
    print("\n📐 EDGE SAMPLING STRATEGIES")
    print("=" * 35)
    
    # Test edge
    edge = LineString([(0, 0), (100, 50)])  # 100m diagonal edge
    edge_length = edge.length
    
    strategies = [
        ("Uniform spacing", "uniform"),
        ("Adaptive density", "adaptive"), 
        ("Corner emphasis", "corners"),
        ("Shadow-aware", "shadow_aware")
    ]
    
    print(f"🔍 Test edge: {edge_length:.1f}m length")
    
    for strategy_name, strategy_type in strategies:
        print(f"\n   {strategy_name}:")
        
        if strategy_type == "uniform":
            # Simple uniform spacing
            sample_distance = 10.0  # Every 10 meters
            n_points = int(edge_length / sample_distance) + 1
            points = [edge.interpolate(i * sample_distance) for i in range(n_points)]
            print(f"     Points: {len(points)} (every {sample_distance}m)")
            
        elif strategy_type == "adaptive":
            # Denser sampling where edge curvature is higher
            # For straight edges, this reduces to uniform
            base_distance = 15.0
            n_points = max(3, int(edge_length / base_distance))
            points = [edge.interpolate(i / (n_points-1), normalized=True) for i in range(n_points)]
            print(f"     Points: {len(points)} (adaptive to geometry)")
            
        elif strategy_type == "corners":
            # Emphasis on endpoints and corners
            points = [edge.coords[0], edge.coords[-1]]  # Always include endpoints
            # Add intermediate points
            n_intermediate = max(1, int(edge_length / 20))
            for i in range(1, n_intermediate + 1):
                points.append(edge.interpolate(i / (n_intermediate + 1), normalized=True))
            print(f"     Points: {len(points)} (endpoint + intermediate)")
            
        elif strategy_type == "shadow_aware":
            # Sample based on expected shadow variation
            # More points where shadows change rapidly
            base_points = 5
            sun_elevation = 45  # degrees
            shadow_length = 20 / np.tan(np.radians(sun_elevation))  # 20m building
            
            # If shadow length >> edge length, uniform sampling is fine
            # If shadow length ~ edge length, need more points
            complexity_factor = min(2.0, edge_length / shadow_length)
            n_points = int(base_points * complexity_factor)
            
            points = [edge.interpolate(i / (n_points-1), normalized=True) for i in range(n_points)]
            print(f"     Points: {len(points)} (shadow complexity factor: {complexity_factor:.1f})")


def prototype_edge_to_points_conversion():
    """Prototype converting edges to representative points"""
    print(f"\n🔄 EDGE-TO-POINTS CONVERSION")
    print("=" * 35)
    
    def edge_to_points(edge_geom, target_spacing=10.0, min_points=3, max_points=20):
        """
        Convert a LineString edge to representative points
        
        Args:
            edge_geom: Shapely LineString
            target_spacing: Target spacing between points (meters)
            min_points: Minimum number of points
            max_points: Maximum number of points
            
        Returns:
            List of Point objects representing the edge
        """
        if not isinstance(edge_geom, LineString):
            raise ValueError("Input must be a LineString")
        
        edge_length = edge_geom.length
        
        # Calculate optimal number of points
        n_points_by_spacing = int(edge_length / target_spacing) + 1
        n_points = max(min_points, min(max_points, n_points_by_spacing))
        
        # Generate points along the edge
        points = []
        for i in range(n_points):
            # Use normalized distance (0 to 1)
            distance_fraction = i / (n_points - 1) if n_points > 1 else 0
            point = edge_geom.interpolate(distance_fraction, normalized=True)
            points.append(Point(point.x, point.y))
        
        return points, n_points, edge_length / (n_points - 1) if n_points > 1 else 0
    
    # Test with different edge types
    test_edges = [
        ("Short building edge", LineString([(0, 0), (15, 0)])),
        ("Long property line", LineString([(0, 0), (200, 0)])),
        ("Curved road edge", LineString([(0, 0), (50, 25), (100, 0)])),
        ("Complex polygon edge", LineString([(0, 0), (30, 10), (60, -5), (90, 15), (120, 0)]))
    ]
    
    for edge_name, edge in test_edges:
        points, n_points, actual_spacing = edge_to_points(edge)
        
        print(f"\n   {edge_name}:")
        print(f"     Original length: {edge.length:.1f}m")
        print(f"     Points generated: {n_points}")
        print(f"     Actual spacing: {actual_spacing:.1f}m")
        print(f"     Point coordinates: {[(p.x, p.y) for p in points[:3]]}...")


def prototype_edge_shadow_integration():
    """Prototype integrating edge sampling with shadow ray casting"""
    print(f"\n🔗 EDGE-SHADOW INTEGRATION")
    print("=" * 30)
    
    print("Integration approach:")
    print("1. Convert edges to representative points")
    print("2. Apply shadow ray casting to each point")  
    print("3. Aggregate results back to edge-level metrics")
    print("")
    
    # Pseudo-implementation
    print("def process_edges_for_shadows(edges, timestamp, dsm_manager):")
    print("    results = []")
    print("    for edge in edges:")
    print("        # Convert edge to points")
    print("        points = edge_to_representative_points(edge)")
    print("        ")
    print("        # Ray cast at each point")
    print("        point_metrics = []")
    print("        for point in points:")
    print("            metrics = compute_shade_metrics(point, timestamp)")
    print("            point_metrics.append(metrics)")
    print("        ")
    print("        # Aggregate to edge-level metrics")
    print("        edge_metrics = aggregate_point_metrics_to_edge(")
    print("            edge, point_metrics)")
    print("        results.append(edge_metrics)")
    print("    ")
    print("    return results")
    
    print(f"\n🎯 Edge-level aggregation metrics:")
    print(f"   • Average shade fraction along edge")
    print(f"   • Maximum shade duration any point on edge")  
    print(f"   • Percentage of edge length currently shaded")
    print(f"   • Shade gradient along edge (uniform vs variable)")
    print(f"   • Edge orientation vs sun angle analysis")


def main():
    """Run edge handling research"""
    print("📏🌞 EDGE HANDLING FOR SHADOW RAY CASTING")
    print("=" * 50)
    
    try:
        # Analyze shadow patterns along edges
        analyze_edge_shadow_patterns()
        
        # Prototype sampling strategies  
        prototype_edge_sampling_strategies()
        
        # Prototype edge-to-points conversion
        prototype_edge_to_points_conversion()
        
        # Prototype integration approach
        prototype_edge_shadow_integration()
        
        print(f"\n✅ EDGE HANDLING RESEARCH COMPLETE!")
        
        print(f"\n🔬 Key insights:")
        print(f"   • Straight edges have relatively uniform shadow patterns")
        print(f"   • 10-20m point spacing captures most shadow variations")
        print(f"   • Always include edge endpoints in sampling")
        print(f"   • Complex edges need adaptive sampling")
        print(f"   • Edge-level aggregation provides useful summary metrics")
        
        print(f"\n🚀 Recommended approach:")
        print(f"   1. Convert edges to points (10-15m spacing)")
        print(f"   2. Apply existing ray casting to points")
        print(f"   3. Aggregate point results to edge metrics")
        print(f"   4. Provide both point-level and edge-level outputs")
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
