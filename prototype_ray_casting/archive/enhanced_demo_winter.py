#!/usr/bin/env python3
"""
Enhanced Shade Analysis System - Winter Demo
Test with winter conditions for longer shadows and different lighting scenarios.
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


def demo_winter_shade_analysis():
    """Test shade analysis during winter months when shadows are longer"""
    print("\n❄️ === WINTER SHADE ANALYSIS DEMO ===")
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    if not Path(dsm_dir).exists():
        print(f"❌ DSM directory not found: {dsm_dir}")
        return
    
    try:
        print("🔍 Indexing DSM tiles...")
        processor = EdgeToPointsProcessor(dsm_dir, lat=42.36, lon=-71.06)
        print(f"✅ Initialized API with DSM tiles")
        
        # Winter test scenarios - December when sun is lowest
        winter_scenarios = [
            (datetime(2024, 12, 21, 8, 0), "Winter Dawn"),
            (datetime(2024, 12, 21, 12, 0), "Winter Noon"),  
            (datetime(2024, 12, 21, 16, 0), "Winter Late Afternoon"),
            (datetime(2024, 3, 20, 9, 0), "Spring Equinox Morning"),
            (datetime(2024, 3, 20, 15, 0), "Spring Equinox Afternoon")
        ]
        
        # Create test edges in potentially more shaded areas
        winter_edges = [
            # North-facing street (should get more shade in winter)
            {'edge_id': 'north_facing_001', 'geometry': LineString([(-71.0580, 42.3600), (-71.0560, 42.3600)]), 'orientation': 'north_facing'},
            # East-West street (shadows from buildings)
            {'edge_id': 'ew_building_002', 'geometry': LineString([(-71.0590, 42.3595), (-71.0570, 42.3595)]), 'orientation': 'east_west'},
            # Area near potential buildings/trees
            {'edge_id': 'urban_canyon_003', 'geometry': LineString([(-71.0575, 42.3590), (-71.0575, 42.3610)]), 'orientation': 'ns_canyon'},
        ]
        
        winter_edges_gdf = gpd.GeoDataFrame(winter_edges, crs='EPSG:4326')
        
        all_winter_results = []
        
        for test_time, scenario_name in winter_scenarios:
            print(f"\n🌨️ Testing {scenario_name} ({test_time.strftime('%Y-%m-%d %H:%M')})")
            
            results = processor.process_multiple_edges(
                edges_gdf=winter_edges_gdf,
                current_time=test_time,
                spacing_m=8.0  # Dense sampling
            )
            
            if not results.empty:
                results['scenario'] = scenario_name
                results['season'] = 'winter' if test_time.month == 12 else 'spring'
                results['test_time'] = test_time
                all_winter_results.append(results)
                
                print(f"   📊 Points analyzed: {len(results)}")
                print(f"   🌳 Points in shade: {results['current_shade'].sum()}")
                print(f"   📈 Shade percentage: {(results['current_shade'].sum() / len(results) * 100):.1f}%")
                print(f"   ⏰ Avg 1h shade fraction: {results['last_1h_fraction'].mean():.3f}")
                print(f"   🕐 Avg 4h shade fraction: {results['last_4h_fraction'].mean():.3f}")
        
        if all_winter_results:
            combined_winter = pd.concat(all_winter_results, ignore_index=True)
            
            print(f"\n❄️ WINTER ANALYSIS SUMMARY:")
            print(f"   Total test scenarios: {len(winter_scenarios)}")
            print(f"   Total points analyzed: {len(combined_winter)}")
            print(f"   Overall shade percentage: {(combined_winter['current_shade'].sum() / len(combined_winter) * 100):.1f}%")
            
            # Show results by scenario
            scenario_summary = combined_winter.groupby('scenario').agg({
                'current_shade': ['count', 'sum', 'mean'],
                'last_1h_fraction': 'mean',
                'last_4h_fraction': 'mean'
            }).round(3)
            
            print(f"\n📋 RESULTS BY WINTER SCENARIO:")
            print(scenario_summary)
            
            # Save winter results
            combined_winter.to_csv('winter_shade_analysis.csv', index=False)
            print(f"💾 Winter results saved to: winter_shade_analysis.csv")
            
        print(f"✅ Winter shade analysis completed!")
            
    except Exception as e:
        print(f"❌ Winter demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_targeted_point_analysis():
    """Test specific points that are more likely to be in shade"""
    print("\n🎯 === TARGETED POINT ANALYSIS ===")
    
    dsm_dir = "/data2/lukas/projects/thermal_drift/results/output/step4_raster_processing/bb2eafb8"
    
    if not Path(dsm_dir).exists():
        print(f"❌ DSM directory not found: {dsm_dir}")
        return
    
    try:
        print("🔍 Setting up targeted analysis...")
        dsm_manager = DSMTileManager(dsm_dir)
        caster = ProductionShadowCaster(dsm_manager)
        
        # Test points spread across the available area
        target_points = [
            (-71.0600, 42.3580, "Southwest corner"),
            (-71.0550, 42.3580, "Southeast corner"), 
            (-71.0600, 42.3620, "Northwest corner"),
            (-71.0550, 42.3620, "Northeast corner"),
            (-71.0575, 42.3600, "Center area"),
            (-71.0590, 42.3590, "Southwest quadrant"),
            (-71.0560, 42.3590, "Southeast quadrant"),
            (-71.0590, 42.3610, "Northwest quadrant"),
            (-71.0560, 42.3610, "Northeast quadrant"),
        ]
        
        # Test at different times including winter
        test_times = [
            (datetime(2024, 12, 21, 14, 0), "Winter afternoon"),
            (datetime(2024, 6, 21, 8, 0), "Summer morning"),
            (datetime(2024, 6, 21, 18, 0), "Summer evening"),
            (datetime(2024, 9, 22, 12, 0), "Fall equinox noon"),
        ]
        
        targeted_results = []
        
        for test_time, time_label in test_times:
            print(f"\n⏰ Testing {time_label} ({test_time.strftime('%m/%d %H:%M')})")
            
            for i, (lon, lat, location_label) in enumerate(target_points):
                try:
                    point_id = f"target_{i:02d}_{test_time.month:02d}{test_time.hour:02d}"
                    
                    # Fix the API call - use correct parameter name
                    metrics = caster.compute_shade_metrics(
                        point=Point(lon, lat),
                        current_time=test_time,
                        lat=lat,
                        lon=lon,
                        point_id=point_id
                    )
                    
                    targeted_results.append({
                        'point_id': point_id,
                        'latitude': lat,
                        'longitude': lon,
                        'location_label': location_label,
                        'time_label': time_label,
                        'test_time': test_time,
                        'current_shade': metrics.current_shade_status,
                        'last_1h_fraction': metrics.last_1h_shade_fraction,
                        'last_4h_fraction': metrics.last_4h_shade_fraction,
                        'cumulative_fraction': metrics.cumulative_shade_fraction,
                        'sun_elevation': metrics.current_sun_elevation,
                        'sun_azimuth': metrics.current_sun_azimuth
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ Error at {location_label}: {e}")
                    continue
        
        if targeted_results:
            targeted_df = pd.DataFrame(targeted_results)
            
            print(f"\n🎯 TARGETED ANALYSIS RESULTS:")
            print(f"   Total points tested: {len(targeted_df)}")
            print(f"   Points in shade: {targeted_df['current_shade'].sum()}")
            print(f"   Shade percentage: {(targeted_df['current_shade'].sum() / len(targeted_df) * 100):.1f}%")
            print(f"   Average sun elevation: {targeted_df['sun_elevation'].mean():.1f}°")
            
            # Show results by time period
            time_summary = targeted_df.groupby('time_label').agg({
                'current_shade': ['count', 'sum', 'mean'],
                'last_4h_fraction': 'mean',
                'sun_elevation': 'mean'
            }).round(3)
            
            print(f"\n⏰ RESULTS BY TIME PERIOD:")
            print(time_summary)
            
            # Show most interesting points (highest shade fractions)
            interesting_points = targeted_df.nlargest(10, 'last_4h_fraction')
            if len(interesting_points) > 0:
                print(f"\n🌳 TOP SHADED LOCATIONS:")
                for _, row in interesting_points.iterrows():
                    shade_status = "SHADE" if row['current_shade'] else "SUN"
                    print(f"   {row['location_label']} @ {row['time_label']}: {shade_status} | 4h: {row['last_4h_fraction']:.3f} | Sun: {row['sun_elevation']:.1f}°")
            
            targeted_df.to_csv('targeted_point_analysis.csv', index=False)
            print(f"💾 Targeted results saved to: targeted_point_analysis.csv")
            
        print(f"✅ Targeted point analysis completed!")
        
    except Exception as e:
        print(f"❌ Targeted analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run winter and targeted shade analysis demos"""
    print("❄️ WINTER & TARGETED SHADE ANALYSIS DEMO")
    print("=" * 50)
    
    try:
        # Run winter-focused demos
        demo_winter_shade_analysis()
        demo_targeted_point_analysis()
        
        print("\n🎉 ALL WINTER DEMOS COMPLETED!")
        print("\nWinter Testing Results:")
        print("✅ Low-angle sun scenarios tested (winter solstice)")
        print("✅ Multiple seasonal comparisons")
        print("✅ Targeted point analysis across area")
        print("✅ Sun angle and elevation data captured")
        print("✅ Comprehensive CSV outputs for analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
