#!/usr/bin/env python3
"""
Create final visualization showing the working shade analysis results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_shade_results_visualization():
    """Create visualization of shade results by scenario"""
    
    # Data from the corrected results
    scenarios = [
        ("Summer Noon", 71.2, 16.5),
        ("Summer Morning", 37.4, 44.3), 
        ("Summer Evening", 15.5, 79.7),
        ("Winter Noon", 24.3, 63.5),
        ("Winter Afternoon", 18.7, 71.3),
        ("Spring Morning", 45.0, 45.4)  # estimated elevation
    ]
    
    scenario_names = [s[0] for s in scenarios]
    sun_elevations = [s[1] for s in scenarios]
    shade_percentages = [s[2] for s in scenarios]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Shade percentage by scenario
    bars = ax1.bar(range(len(scenarios)), shade_percentages, 
                   color=['gold', 'orange', 'red', 'lightblue', 'blue', 'green'])
    ax1.set_title('Shade Percentage by Scenario\n(Fixed Ray Casting Results)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Shade Percentage (%)')
    ax1.set_xlabel('Scenario')
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, shade_percentages)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Shade vs Sun Elevation (scatter plot)
    colors = ['gold', 'orange', 'red', 'lightblue', 'blue', 'green'] 
    scatter = ax2.scatter(sun_elevations, shade_percentages, c=colors, s=120, alpha=0.8)
    
    # Add trend line
    z = np.polyfit(sun_elevations, shade_percentages, 1)
    p = np.poly1d(z)
    ax2.plot(sun_elevations, p(sun_elevations), "r--", alpha=0.8, linewidth=2)
    
    ax2.set_title('Shade vs Sun Elevation\n(Physics-Based Relationship)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sun Elevation (degrees)')
    ax2.set_ylabel('Shade Percentage (%)')
    ax2.grid(alpha=0.3)
    
    # Add scenario labels
    for i, (name, elev, shade) in enumerate(scenarios):
        ax2.annotate(name.replace(' ', '\n'), (elev, shade), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, alpha=0.8)
    
    # Add correlation info
    correlation = np.corrcoef(sun_elevations, shade_percentages)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.2f}\n(Negative as expected)', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('final_shade_analysis_results.png', dpi=150, bbox_inches='tight')
    print("💾 Final visualization saved to: final_shade_analysis_results.png")
    
    # Print summary stats
    print(f"\n📊 FINAL SHADE ANALYSIS SUMMARY:")
    print(f"   Scenarios tested: {len(scenarios)}")
    print(f"   Sun elevation range: {min(sun_elevations):.1f}° to {max(sun_elevations):.1f}°")
    print(f"   Shade range: {min(shade_percentages):.1f}% to {max(shade_percentages):.1f}%") 
    print(f"   Physics correlation: {correlation:.2f} (negative = higher sun → less shade)")

if __name__ == "__main__":
    create_shade_results_visualization()
