"""
Tree vs Building Shadow Analysis
===============================

Research and prototype for handling trees differently from buildings:
- Trees: Allow walking underneath, partial light transmissivity
- Buildings: Complete obstruction, no walking underneath

This explores how the current DSM data represents these differences.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('.')
from dsm_loader import DSMTileManager
from shapely.geometry import Point


def analyze_dsm_data_characteristics():
    """Analyze how current DSM data represents trees vs buildings"""
    print("🌳 ANALYZING TREE VS BUILDING CHARACTERISTICS")
    print("=" * 50)
    
    # Load DSM data
    dsm_dir = "../results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    if not dsm_manager.tiles_info:
        print("❌ No DSM data found!")
        return
    
    # Sample a tile for analysis
    sample_tile_id = list(dsm_manager.tiles_info.keys())[0]
    tile_data = dsm_manager.load_tile(sample_tile_id)
    
    building_dsm = tile_data['building']
    canopy_dsm = tile_data['canopy']
    
    print(f"📊 DSM Data Analysis for tile {sample_tile_id}:")
    print(f"   Building DSM shape: {building_dsm.shape}")
    print(f"   Canopy DSM shape: {canopy_dsm.shape}")
    
    # Analyze height distributions
    building_valid = building_dsm[building_dsm > 0]
    canopy_valid = canopy_dsm[canopy_dsm > 0]
    
    print(f"\n🏗️ Building Heights:")
    print(f"   Valid pixels: {len(building_valid):,}")
    print(f"   Range: {building_valid.min():.1f} - {building_valid.max():.1f}m")
    print(f"   Mean: {building_valid.mean():.1f}m")
    
    print(f"\n🌳 Canopy Heights:")
    print(f"   Valid pixels: {len(canopy_valid):,}")
    print(f"   Range: {canopy_valid.min():.1f} - {canopy_valid.max():.1f}m")
    print(f"   Mean: {canopy_valid.mean():.1f}m")
    
    # Check for overlapping areas (where both building and canopy exist)
    both_exist = (building_dsm > 0) & (canopy_dsm > 0)
    overlap_count = np.sum(both_exist)
    
    print(f"\n🔍 Overlap Analysis:")
    print(f"   Pixels with both building and canopy: {overlap_count:,}")
    print(f"   Percentage of total pixels: {(overlap_count / building_dsm.size) * 100:.2f}%")
    
    if overlap_count > 0:
        building_overlap = building_dsm[both_exist]
        canopy_overlap = canopy_dsm[both_exist]
        print(f"   In overlap areas - Building avg: {building_overlap.mean():.1f}m")
        print(f"   In overlap areas - Canopy avg: {canopy_overlap.mean():.1f}m")
    
    return building_dsm, canopy_dsm, both_exist


def prototype_tree_transmissivity():
    """Prototype for handling tree light transmissivity"""
    print("\n🌿 TREE TRANSMISSIVITY PROTOTYPE")
    print("=" * 40)
    
    print("Current approach: Trees block 100% of light")
    print("Proposed improvement: Trees allow partial light transmission")
    print("")
    
    # Different tree density scenarios
    tree_scenarios = [
        ("Dense urban canopy", 0.8),      # 80% light blocked
        ("Medium tree coverage", 0.6),    # 60% light blocked  
        ("Sparse trees", 0.3),           # 30% light blocked
        ("Young/small trees", 0.2),      # 20% light blocked
    ]
    
    print("🌳 Proposed transmissivity values:")
    for scenario, transmissivity in tree_scenarios:
        blocked = 1 - transmissivity
        print(f"   {scenario:<20}: {blocked:.1%} blocked, {transmissivity:.1%} transmitted")
    
    # Implementation concept
    print(f"\n💡 Implementation concept:")
    print(f"   1. Classify canopy density from DSM height/texture")
    print(f"   2. Assign transmissivity based on density")
    print(f"   3. Modify ray casting to handle partial obstruction")
    print(f"   4. Accumulated light reduction along ray path")


def prototype_walkable_tree_areas():
    """Prototype for handling walkable areas under trees"""
    print("\n🚶 WALKABLE TREE AREAS PROTOTYPE") 
    print("=" * 40)
    
    print("Current assumption: Ground level = 0 everywhere")
    print("Proposed improvement: Model tree trunk/canopy separation")
    print("")
    
    # Tree modeling concept
    print("🌳 Tree structure modeling:")
    print("   Trunk zone (0-2m):     Usually walkable, minimal shade")
    print("   Lower canopy (2-4m):   Partial shade, usually walkable")  
    print("   Dense canopy (4m+):    Full shade, not walkable")
    print("")
    
    print("🎯 Implementation approach:")
    print("   1. Estimate trunk diameter from canopy area/height")
    print("   2. Model canopy base height (typically 2-4m)")
    print("   3. Apply transmissivity only above canopy base")
    print("   4. Allow 'walking under' for points below canopy base")
    
    # Example calculation
    canopy_height = 15.0  # meters
    canopy_base_height = 3.0  # meters (walkable underneath)
    trunk_radius = 0.5  # meters
    
    print(f"\n📏 Example tree (15m total height):")
    print(f"   Walkable trunk area: 0-{canopy_base_height}m height")
    print(f"   Shade-casting canopy: {canopy_base_height}-{canopy_height}m")
    print(f"   Effective shade height: {canopy_height - canopy_base_height}m")


def test_current_shadow_behavior():
    """Test how current implementation handles trees vs buildings"""
    print("\n🧪 TESTING CURRENT SHADOW BEHAVIOR")
    print("=" * 40)
    
    dsm_dir = "../results/output/step4_raster_processing/bb2eafb8"
    dsm_manager = DSMTileManager(dsm_dir)
    
    if not dsm_manager.tiles_info:
        print("❌ No DSM data for testing")
        return
    
    # Get a test point
    sample_tile_id = list(dsm_manager.tiles_info.keys())[0]
    tile_info = dsm_manager.tiles_info[sample_tile_id]
    minx, miny, maxx, maxy = tile_info.bounds
    
    test_point = Point((minx + maxx) / 2, (miny + maxy) / 2)
    
    # Sample heights at test point
    heights = dsm_manager.sample_dsm_at_point(test_point, sample_tile_id)
    
    print(f"🔍 Test point analysis:")
    print(f"   Coordinates: {test_point.x:.1f}, {test_point.y:.1f}")
    print(f"   Building height: {heights['building']:.1f}m")
    print(f"   Canopy height: {heights['canopy']:.1f}m")
    print(f"   Combined (used for shadowing): {heights['combined']:.1f}m")
    
    # Determine what's providing the shade
    if heights['building'] > heights['canopy']:
        shade_source = "Building (solid obstruction)"
        walkable = "No - building footprint" 
        transmissivity = "0% (complete blockage)"
    elif heights['canopy'] > heights['building']:
        shade_source = "Tree canopy (current: treated as solid)"
        walkable = "Potentially yes - under canopy"
        transmissivity = "Current: 0%, Should be: 10-80%"
    elif heights['building'] == heights['canopy'] > 0:
        shade_source = "Building + Tree (both present)"
        walkable = "No - building present"
        transmissivity = "0% (building dominates)"
    else:
        shade_source = "No obstruction"
        walkable = "Yes"
        transmissivity = "100% (no obstruction)"
    
    print(f"\n🎯 Shadow analysis:")
    print(f"   Primary shade source: {shade_source}")
    print(f"   Walkable underneath: {walkable}")
    print(f"   Light transmissivity: {transmissivity}")
    
    print(f"\n💡 Improvement opportunities:")
    if heights['canopy'] > 0:
        print(f"   ✅ Could model tree transmissivity")
        print(f"   ✅ Could allow walking under canopy base")
    if heights['building'] > 0:
        print(f"   ✅ Buildings correctly modeled as solid")


def main():
    """Run tree vs building analysis"""
    print("🌳🏢 TREE VS BUILDING SHADOW ANALYSIS")
    print("=" * 50)
    
    try:
        # Analyze current DSM data
        building_dsm, canopy_dsm, overlap = analyze_dsm_data_characteristics()
        
        # Prototype improvements
        prototype_tree_transmissivity()
        prototype_walkable_tree_areas()
        
        # Test current behavior
        test_current_shadow_behavior()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"\n🔬 Key findings:")
        print(f"   • Current method treats trees identical to buildings")
        print(f"   • DSM data provides separate building/canopy layers") 
        print(f"   • Opportunity to model tree transmissivity (10-80%)")
        print(f"   • Opportunity to model walkable areas under canopy")
        print(f"   • Buildings should remain 100% opaque")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Implement transmissivity-based ray casting")
        print(f"   2. Add canopy base height modeling")
        print(f"   3. Test against known tree/building locations")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
