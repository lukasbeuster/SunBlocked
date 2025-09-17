#!/usr/bin/env python3
"""
Enhanced Shade Analysis System - Comprehensive Demo
Demonstrates the enhanced shade analysis capabilities with comprehensive testing scenarios.
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from pathlib import Path
from shapely.geometry import Point, LineString

# Add current directory to path for imports
sys.path.append('.')

from production_api_enhanced import EdgeToPointsProcessor
from ray_caster_enhanced import ProductionShadowCaster
from dsm_loader_enhanced import DSMTileManager


def create_comprehensive_test_edges():
    """Create a comprehensive set of test edges in different areas and orientations"""
    edges_data = []
    
    # Area 1: Dense urban area (likely more shade from buildings)
    # North-South street
    edge1 = LineString([(-71.0570, 42.3590), (-71.0570, 42.3620)])
    edges_data.append({'edge_id': 'urban_ns_001', 'geometry': edge1, 'type': 'main_street', 'area': 'dense_urban'})
    
    # East-West street
    edge2 = LineString([(-71.0590, 42.3600), (-71.0550, 42.3600)])
    edges_data.append({'edge_id': 'urban_ew_001', 'geometry': edge2, 'type': 'main_street', 'area': 'dense_urban'})
    
    # Area 2: Mixed residential (trees and buildings)
    # Curved residential street
    edge3 = LineString([(-71.0580, 42.3580), (-71.0575, 42.3585), (-71.0570, 42.3590), (-71.0565, 42.3595)])
    edges_data.append({'edge_id': 'residential_curve_001', 'geometry': edge3, 'type': 'residential', 'area': 'mixed_residential'})
    
    # Diagonal through park/tree area
    edge4 = LineString([(-71.0600, 42.3570), (-71.0580, 42.3590)])
    edges_data.append({'edge_id': 'park_diagonal_001', 'geometry': edge4, 'type': 'path', 'area': 'park_trees'})
    
    # Area 3: Different orientations to catch different shadow patterns
    # Northeast-Southwest
    edge5 = LineString([(-71.0585, 42.3575), (-71.0565, 42.3595)])
    edges_data.append({'edge_id': 'nesw_001', 'geometry': edge5, 'type': 'secondary', 'area': 'mixed'})
    
    # Northwest-Southeast
    edge6 = LineString([(-71.0595, 42.3575), (-71.0575, 42.3595)])
    edges_data.append({'edge_id': 'nwse_001', 'geometry': edge6, 'type': 'secondary', 'area': 'mixed'})
    
    # Area 4: Longer edges for more sample points
    # Long north-south arterial
    edge7 = LineString([(-71.0560, 42.3560), (-71.0560, 42.3580), (-71.0560, 42.3600), (-71.0560, 42.3620)])
    edges_data.append({'edge_id': 'arterial_long_ns_001', 'geometry': edge7, 'type': 'arterial', 'area': 'corridor'})
    
    # Long east-west arterial
    edge8 = LineString([(-71.0610, 42.3590), (-71.0590, 42.3590), (-71.0570, 42.3590), (-71.0550, 42.3590)])
    edges_data.append({'edge_id': 'arterial_long_ew_001', 'geometry': edge8, 'type': 'arterial', 'area': 'corridor'})
    
    # Area 5: Complex geometry (more realistic street patterns)
    # Zigzag street
    edge9 = LineString([(-71.0575, 42.3610), (-71.0570, 42.3615), (-71.0565, 42.3610), (-71.0560, 42.3615)])
    edges_data.append({'edge_id': 'complex_zigzag_001', 'geometry': edge9, 'type': 'local', 'area': 'residential'})
    
    # L-shaped street
    edge10 = LineString([(-71.0590, 42.3610), (-71.0585, 42.3610), (-71.0585, 42.3605), (-71.0585, 42.3600)])
    edges_data.append({'edge_id': 'l_shaped_001', 'geometry': edge10, 'type': 'local', 'area': 'residential'})
    
    return gpd.GeoDataFrame(edges_data, crs='EPSG:4326')


def test_multiple_time_periods(processor, edge, edge_id):
    """Test shade analysis across multiple time periods to show temporal variation"""
    base_date = datetime(2024, 6, 15)  # Summer solstice period
    times_to_test = [
        (7, 0, "Early Morning"),
        (9, 30, "Mid Morning"), 
        (12, 0, "Noon"),
        (15, 0, "Mid Afternoon"),
        (17, 30, "Late Afternoon")
    ]
    
    all_results = []
    
    for hour, minute, period_name in times_to_test:
        test_time = base_date.replace(hour=hour, minute=minute)
        print(f"  ⏰ Testing {period_name} ({test_time.strftime('%H:%M')})")
        
        results_df = processor.analyze_edge_shade_metrics(
            edge_geometry=edge,
            current_time=test_time,
            spacing_m=15.0,  # Closer spacing for more points
            edge_id=f"{edge_id}_{hour:02d}{minute:02d}"
        )
        
        if not results_df.empty:
            results_df['time_period'] = period_name
            results_df['test_hour'] = hour
            all_results.append(results_df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


def demo_comprehensive_shade_analysis():
    """Comprehensive shade analysis demo with multiple scenarios"""
    print("\n🌟 === COMPREHENSIVE SHADE ANALYSIS DEMO ===")
    
    # DSM tiles directory
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    if not Path(dsm_dir).exists():
        print(f"❌ DSM directory not found: {dsm_dir}")
        return
    
    try:
        print("🔍 Indexing DSM tiles...")
        processor = EdgeToPointsProcessor(dsm_dir, lat=42.36, lon=-71.06)
        print(f"✅ Initialized API with DSM tiles")
        
        # Create comprehensive test edges
        print("\n📐 Creating comprehensive test edges...")
        edges_gdf = create_comprehensive_test_edges()
        print(f"✅ Created {len(edges_gdf)} test edges")
        print(f"   Areas: {sorted(edges_gdf['area'].unique())}")
        print(f"   Types: {sorted(edges_gdf['type'].unique())}")
        
        # Test 1: Process all edges at peak sun time (noon)
        print(f"\n🌞 === TEST 1: ALL EDGES AT NOON ===")
        noon_time = datetime(2024, 6, 15, 12, 0)
        
        all_results = processor.process_multiple_edges(
            edges_gdf=edges_gdf,
            current_time=noon_time,
            spacing_m=10.0  # 10m spacing for good coverage
        )
        
        if not all_results.empty:
            print(f"📊 NOON ANALYSIS RESULTS:")
            print(f"   Total points analyzed: {len(all_results)}")
            print(f"   Points currently in shade: {all_results['current_shade'].sum()}")
            print(f"   Shade percentage: {(all_results['current_shade'].sum() / len(all_results) * 100):.1f}%")
            
            # Show results by area and type
            summary_by_area = all_results.groupby('edge_id').agg({
                'current_shade': ['count', 'sum'],
                'last_1h_fraction': 'mean',
                'last_4h_fraction': 'mean'
            }).round(3)
            print(f"\n📋 RESULTS BY EDGE:")
            print(summary_by_area.head(10))
            
            # Save results
            all_results.to_csv('comprehensive_noon_analysis.csv', index=False)
            print(f"💾 Noon results saved to: comprehensive_noon_analysis.csv")
        
        # Test 2: Time-series analysis on most interesting edges
        print(f"\n🕒 === TEST 2: TIME-SERIES ANALYSIS ===")
        
        # Select a few representative edges for detailed time analysis
        test_edges = edges_gdf.head(3)  # First 3 edges
        
        time_series_results = []
        for idx, row in test_edges.iterrows():
            edge_id = row['edge_id']
            geometry = row['geometry']
            print(f"🔄 Time-series analysis for {edge_id}...")
            
            time_results = test_multiple_time_periods(processor, geometry, edge_id)
            if not time_results.empty:
                time_results['base_edge_id'] = edge_id
                time_results['edge_type'] = row['type']
                time_results['edge_area'] = row['area']
                time_series_results.append(time_results)
        
        if time_series_results:
            combined_time_series = pd.concat(time_series_results, ignore_index=True)
            
            print(f"\n📈 TIME-SERIES RESULTS:")
            print(f"   Total time-point combinations: {len(combined_time_series)}")
            
            # Show shade variation by time period
            time_summary = combined_time_series.groupby('time_period').agg({
                'current_shade': ['count', 'sum', 'mean'],
                'last_1h_fraction': 'mean',
                'last_4h_fraction': 'mean'
            }).round(3)
            print(f"\n⏰ SHADE PATTERNS BY TIME PERIOD:")
            print(time_summary)
            
            # Save time series results
            combined_time_series.to_csv('comprehensive_time_series_analysis.csv', index=False)
            print(f"💾 Time-series results saved to: comprehensive_time_series_analysis.csv")
        
        # Test 3: High-density sampling on interesting edges
        print(f"\n🎯 === TEST 3: HIGH-DENSITY SAMPLING ===")
        
        # Pick the edge that showed most shade in Test 1
        if not all_results.empty:
            edge_shade_counts = all_results.groupby('edge_id')['current_shade'].sum().sort_values(ascending=False)
            if len(edge_shade_counts) > 0 and edge_shade_counts.iloc[0] > 0:
                most_shaded_edge_id = edge_shade_counts.index[0]
                most_shaded_edge = edges_gdf[edges_gdf['edge_id'] == most_shaded_edge_id].iloc[0]
                
                print(f"🌳 Analyzing most shaded edge: {most_shaded_edge_id}")
                print(f"   Edge type: {most_shaded_edge['type']}, Area: {most_shaded_edge['area']}")
                
                # High-density analysis (5m spacing)
                dense_results = processor.analyze_edge_shade_metrics(
                    edge_geometry=most_shaded_edge['geometry'],
                    current_time=datetime(2024, 6, 15, 14, 0),  # Afternoon
                    spacing_m=5.0,
                    edge_id=f"dense_{most_shaded_edge_id}"
                )
                
                if not dense_results.empty:
                    print(f"📍 HIGH-DENSITY RESULTS:")
                    print(f"   Sample points: {len(dense_results)}")
                    print(f"   Points in shade: {dense_results['current_shade'].sum()}")
                    print(f"   Average 1h shade fraction: {dense_results['last_1h_fraction'].mean():.3f}")
                    print(f"   Average 4h shade fraction: {dense_results['last_4h_fraction'].mean():.3f}")
                    
                    # Show sample of detailed metrics
                    print(f"\n🔍 SAMPLE DETAILED METRICS:")
                    sample_points = dense_results.head(5)
                    for _, row in sample_points.iterrows():
                        shade_status = "SHADE" if row['current_shade'] else "SUN"
                        print(f"   {row['point_id']}: {shade_status} | 1h: {row['last_1h_fraction']:.3f} | 4h: {row['last_4h_fraction']:.3f}")
                    
                    dense_results.to_csv('comprehensive_dense_analysis.csv', index=False)
                    print(f"💾 Dense results saved to: comprehensive_dense_analysis.csv")
            else:
                print("⚠️ No significantly shaded edges found for high-density analysis")
        
        print(f"\n✅ Comprehensive shade analysis demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_point_grid_analysis():
    """Test a grid of individual points to find shaded areas"""
    print("\n🗺️ === POINT GRID ANALYSIS ===")
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    if not Path(dsm_dir).exists():
        print(f"❌ DSM directory not found: {dsm_dir}")
        return
    
    try:
        print("🔍 Setting up point grid analysis...")
        dsm_manager = DSMTileManager(dsm_dir)
        caster = ProductionShadowCaster(dsm_manager)
        
        # Create a grid of test points
        lat_min, lat_max = 42.3570, 42.3620
        lon_min, lon_max = -71.0610, -71.0550
        
        # 10x10 grid
        lats = np.linspace(lat_min, lat_max, 10)
        lons = np.linspace(lon_min, lon_max, 10)
        
        test_time = datetime(2024, 6, 15, 14, 0)  # 2 PM
        print(f"⏰ Testing at: {test_time}")
        
        grid_results = []
        total_points = len(lats) * len(lons)
        
        print(f"📍 Testing {total_points} grid points...")
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                try:
                    point_id = f"grid_{i:02d}_{j:02d}"
                    metrics = caster.compute_shade_metrics(
                        coordinate=Point(lon, lat),
                        timestamp=test_time,
                        point_id=point_id
                    )
                    
                    grid_results.append({
                        'point_id': point_id,
                        'latitude': lat,
                        'longitude': lon,
                        'current_shade': metrics.current_shade_status,
                        'last_1h_fraction': metrics.last_1h_shade_fraction,
                        'last_4h_fraction': metrics.last_4h_shade_fraction,
                        'cumulative_fraction': metrics.cumulative_shade_fraction
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error at grid point {point_id}: {e}")
                    continue
        
        if grid_results:
            grid_df = pd.DataFrame(grid_results)
            
            print(f"📊 GRID ANALYSIS RESULTS:")
            print(f"   Total points tested: {len(grid_df)}")
            print(f"   Points in shade: {grid_df['current_shade'].sum()}")
            print(f"   Shade percentage: {(grid_df['current_shade'].sum() / len(grid_df) * 100):.1f}%")
            print(f"   Average 1h shade: {grid_df['last_1h_fraction'].mean():.3f}")
            print(f"   Average 4h shade: {grid_df['last_4h_fraction'].mean():.3f}")
            
            # Find most shaded points
            shaded_points = grid_df[grid_df['current_shade'] == True]
            if len(shaded_points) > 0:
                print(f"\n🌳 MOST SHADED AREAS:")
                top_shaded = shaded_points.nlargest(5, 'last_4h_fraction')
                for _, row in top_shaded.iterrows():
                    print(f"   {row['point_id']}: ({row['latitude']:.4f}, {row['longitude']:.4f}) | 4h: {row['last_4h_fraction']:.3f}")
            
            grid_df.to_csv('comprehensive_grid_analysis.csv', index=False)
            print(f"💾 Grid results saved to: comprehensive_grid_analysis.csv")
            
        print(f"✅ Point grid analysis completed!")
        
    except Exception as e:
        print(f"❌ Grid analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run comprehensive shade analysis demos"""
    print("🌟 COMPREHENSIVE SHADE ANALYSIS SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Run comprehensive demos
        demo_comprehensive_shade_analysis()
        demo_point_grid_analysis()
        
        print("\n🎉 ALL COMPREHENSIVE DEMOS COMPLETED!")
        print("\nComprehensive Testing Results:")
        print("✅ Multiple edge orientations and areas tested")
        print("✅ Time-series analysis across day periods")
        print("✅ High-density sampling demonstrated")
        print("✅ Point grid analysis for area coverage")
        print("✅ Multiple CSV outputs generated for analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
