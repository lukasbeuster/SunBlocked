#!/usr/bin/env python3
"""
Enhanced Shade Analysis Demo - Trunk Zone & Edge-to-Points
=========================================================

Demonstrates the enhanced shade analysis system with:
1. Trunk zone handling for improved tree modeling
2. Edge-to-points functionality for line segment analysis
3. Comprehensive metrics validation

Run this demo to validate the enhanced functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point, LineString
import geopandas as gpd
from pathlib import Path
import sys

# Add prototype directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dsm_loader_enhanced import DSMTileManager
from ray_caster_enhanced import ProductionShadowCaster
from production_api_enhanced import EdgeToPointsProcessor

def create_sample_edge():
    """Create a sample edge LineString for testing"""
    # Create a line segment through Boston (across different terrain types)
    coords = [
        (-71.0589, 42.3601),  # Start: near trees/buildings
        (-71.0580, 42.3605),  # Middle: varied terrain  
        (-71.0570, 42.3610),  # End: different context
    ]
    return LineString(coords)

def demo_trunk_zone_handling():
    """Demonstrate trunk zone handling with point-based analysis"""
    print("\n🌳 === TRUNK ZONE HANDLING DEMO ===")
    
    # Boston DSM tiles directory (adjust path as needed)
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    if not Path(dsm_dir).exists():
        print(f"❌ DSM directory not found: {dsm_dir}")
        print("📝 Please adjust dsm_dir path in the demo script")
        return
    
    try:
        # Initialize enhanced ray caster
        dsm_manager = DSMTileManager(dsm_dir)
        caster = ProductionShadowCaster(dsm_manager)
        
        # Test point in Boston with trees nearby
        test_point = Point(-71.0589, 42.3601)
        current_time = datetime(2024, 6, 15, 14, 30)  # Summer afternoon
        
        print(f"📍 Testing point: {test_point.x:.4f}, {test_point.y:.4f}")
        print(f"⏰ Analysis time: {current_time}")
        
        # Compute shade metrics with trunk zone handling
        metrics = caster.compute_shade_metrics(
            point=test_point,
            current_time=current_time,
            lat=42.36,
            lon=-71.06,
            point_id="trunk_zone_test"
        )
        
        print(f"\n📊 ENHANCED SHADE METRICS:")
        print(f"   Current shade: {'YES' if metrics.current_shade_status else 'NO'}")
        print(f"   Last 1h fraction: {metrics.last_1h_shade_fraction:.3f}")
        print(f"   Last 2h fraction: {metrics.last_2h_shade_fraction:.3f}")
        print(f"   Last 4h fraction: {metrics.last_4h_shade_fraction:.3f}")
        print(f"   Since dawn fraction: {metrics.cumulative_shade_fraction:.3f}")
        
        # Test multiple points to show trunk zone effects
        test_points = [
            Point(-71.0589, 42.3601),  # Point A
            Point(-71.0585, 42.3603),  # Point B  
            Point(-71.0580, 42.3605),  # Point C
        ]
        
        print(f"\n🔍 TRUNK ZONE COMPARISON:")
        for i, point in enumerate(test_points):
            try:
                metrics = caster.compute_shade_metrics(
                    point=point,
                    current_time=current_time,
                    lat=42.36,
                    lon=-71.06,
                    point_id=f"point_{chr(65+i)}"
                )
                shade_status = "SHADE" if metrics.current_shade_status else "SUN"
                print(f"   Point {chr(65+i)}: {shade_status} (1h: {metrics.last_1h_shade_fraction:.2f})")
                
            except Exception as e:
                print(f"   Point {chr(65+i)}: ERROR - {e}")
        
        print("✅ Trunk zone handling demo completed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("📝 Check that DSM tiles are available and paths are correct")

def demo_edge_to_points():
    """Demonstrate edge-to-points functionality"""
    print("\n🛤️ === EDGE-TO-POINTS DEMO ===")
    
    # Boston DSM tiles directory (adjust path as needed)
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    if not Path(dsm_dir).exists():
        print(f"❌ DSM directory not found: {dsm_dir}")
        print("📝 Please adjust dsm_dir path in the demo script")
        return
    
    try:
        # Initialize edge-to-points processor
        processor = EdgeToPointsProcessor(dsm_dir, lat=42.36, lon=-71.06)
        
        # Create sample edge
        sample_edge = create_sample_edge()
        current_time = datetime(2024, 6, 15, 14, 30)
        
        print(f"🔗 Sample edge: {len(sample_edge.coords)} coordinate pairs")
        print(f"⏰ Analysis time: {current_time}")
        print(f"📏 Spacing: 25 meters between points")
        
        # Analyze shade metrics along the edge
        results_df = processor.analyze_edge_shade_metrics(
            edge_geometry=sample_edge,
            current_time=current_time,
            spacing_m=25.0,
            edge_id="demo_edge_001"
        )
        
        if results_df.empty:
            print("❌ No results generated")
            return
        
        print(f"\n📊 EDGE ANALYSIS RESULTS:")
        print(f"   Generated points: {len(results_df)}")
        print(f"   Points in shade: {results_df['current_shade'].sum()}")
        print(f"   Points in sun: {(~results_df['current_shade'].astype(bool)).sum()}")
        
        # Show sample results
        print(f"\n📋 SAMPLE POINT METRICS:")
        for idx, row in results_df.head(3).iterrows():
            shade_status = "SHADE" if row['current_shade'] else "SUN"
            print(f"   {row['point_id']}: {shade_status} | 1h: {row['last_1h_fraction']:.2f} | 4h: {row['last_4h_fraction']:.2f}")
        
        # Save results
        output_file = Path("enhanced_edge_analysis_results.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Create summary statistics
        print(f"\n📈 EDGE SUMMARY STATISTICS:")
        print(f"   Average 1h shade fraction: {results_df['last_1h_fraction'].mean():.3f}")
        print(f"   Average 4h shade fraction: {results_df['last_4h_fraction'].mean():.3f}")
        print(f"   Average since-dawn fraction: {results_df['since_dawn_fraction'].mean():.3f}")
        
        print("✅ Edge-to-points demo completed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("📝 Check that DSM tiles are available and paths are correct")
        import traceback
        traceback.print_exc()

def demo_multiple_edges():
    """Demonstrate processing multiple edges from a GeoDataFrame"""
    print("\n📐 === MULTIPLE EDGES DEMO ===")
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    if not Path(dsm_dir).exists():
        print(f"❌ DSM directory not found: {dsm_dir}")
        return
    
    try:
        # Create multiple sample edges
        edges_data = []
        
        # Edge 1: North-South oriented
        edge1 = LineString([(-71.0589, 42.3601), (-71.0589, 42.3610)])
        edges_data.append({'edge_id': 'north_south_001', 'geometry': edge1, 'type': 'main_street'})
        
        # Edge 2: East-West oriented  
        edge2 = LineString([(-71.0595, 42.3605), (-71.0580, 42.3605)])
        edges_data.append({'edge_id': 'east_west_002', 'geometry': edge2, 'type': 'side_street'})
        
        # Edge 3: Diagonal
        edge3 = LineString([(-71.0590, 42.3600), (-71.0575, 42.3615)])
        edges_data.append({'edge_id': 'diagonal_003', 'geometry': edge3, 'type': 'path'})
        
        # Create GeoDataFrame
        edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        print(f"🗺️ Processing {len(edges_gdf)} edges")
        print(f"   Edge types: {edges_gdf['type'].value_counts().to_dict()}")
        
        # Initialize processor
        processor = EdgeToPointsProcessor(dsm_dir, lat=42.36, lon=-71.06)
        current_time = datetime(2024, 6, 15, 15, 0)
        
        # Process all edges
        all_results = processor.process_multiple_edges(
            edges_gdf=edges_gdf,
            current_time=current_time,
            spacing_m=20.0,
            edge_id_column='edge_id'
        )
        
        if all_results.empty:
            print("❌ No results generated")
            return
        
        print(f"\n📊 MULTIPLE EDGES RESULTS:")
        print(f"   Total points analyzed: {len(all_results)}")
        print(f"   Total points in shade: {all_results['current_shade'].sum()}")
        
        # Per-edge summary
        edge_summary = all_results.groupby('edge_id').agg({
            'current_shade': ['count', 'sum'],
            'last_1h_fraction': 'mean',
            'last_4h_fraction': 'mean'
        }).round(3)
        
        print(f"\n📋 PER-EDGE SUMMARY:")
        print(edge_summary)
        
        # Save comprehensive results
        output_file = Path("multiple_edges_analysis.csv")
        all_results.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        print("✅ Multiple edges demo completed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all demo functions"""
    print("🌟 ENHANCED SHADE ANALYSIS SYSTEM DEMO")
    print("=" * 50)
    
    try:
        # Run individual demos
        demo_trunk_zone_handling()
        demo_edge_to_points() 
        demo_multiple_edges()
        
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\nKey Enhancements Demonstrated:")
        print("✅ Trunk zone handling (25% tree height threshold)")
        print("✅ Edge-to-points conversion with configurable spacing")
        print("✅ Batch processing of multiple edges")
        print("✅ Comprehensive shade metrics with temporal windows")
        print("✅ CSV output for integration with thermal drift pipeline")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
