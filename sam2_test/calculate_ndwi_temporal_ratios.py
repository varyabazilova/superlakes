#!/usr/bin/env python3
"""
Calculate temporal ratios for NDWI images to detect lake changes
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def parse_date_from_filename(filename):
    """Extract date from filename like '2021-01-28_ndwi.tif'"""
    try:
        # Extract date part before '_ndwi.tif'
        date_str = os.path.basename(filename).split('_ndwi')[0]
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        print(f"⚠️ Could not parse date from: {filename}")
        return None

def calculate_temporal_ratios(input_dir, output_dir):
    """
    Calculate temporal ratios between consecutive NDWI images
    
    Parameters:
    - input_dir: Directory with NDWI files
    - output_dir: Directory to save ratio files
    """
    print("🔄 Temporal NDWI Ratio Calculation")
    print("=" * 50)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all NDWI files
    ndwi_pattern = os.path.join(input_dir, "*_ndwi.tif")
    ndwi_files = glob.glob(ndwi_pattern)
    
    if not ndwi_files:
        print(f"❌ No NDWI files found matching pattern: {ndwi_pattern}")
        return
    
    print(f"📁 Found {len(ndwi_files)} NDWI files")
    
    # Parse dates and sort chronologically
    file_dates = []
    for file in ndwi_files:
        date = parse_date_from_filename(file)
        if date:
            file_dates.append((date, file))
    
    # Sort by date
    file_dates.sort(key=lambda x: x[0])
    
    print(f"📅 Date range: {file_dates[0][0].strftime('%Y-%m-%d')} to {file_dates[-1][0].strftime('%Y-%m-%d')}")
    
    # Calculate ratios between consecutive dates
    ratios_calculated = []
    
    for i in range(len(file_dates) - 1):
        date1, file1 = file_dates[i]
        date2, file2 = file_dates[i + 1]
        
        print(f"\n🔄 Calculating ratio: {date2.strftime('%Y-%m-%d')} / {date1.strftime('%Y-%m-%d')}")
        
        try:
            # Load NDWI images
            with rasterio.open(file1) as src1:
                ndwi1 = src1.read(1).astype(float)
                profile = src1.profile.copy()
            
            with rasterio.open(file2) as src2:
                ndwi2 = src2.read(1).astype(float)
            
            print(f"   NDWI1 shape: {ndwi1.shape}, range: {ndwi1.min():.3f} to {ndwi1.max():.3f}")
            print(f"   NDWI2 shape: {ndwi2.shape}, range: {ndwi2.min():.3f} to {ndwi2.max():.3f}")
            
            # Handle nodata values (assume values < -1 or > 1 are invalid)
            valid_mask = (ndwi1 > -1) & (ndwi1 < 1) & (ndwi2 > -1) & (ndwi2 < 1)
            
            # Calculate ratio: ndwi2 / ndwi1
            # Add small epsilon to avoid division by zero
            epsilon = 0.001
            ratio = np.full_like(ndwi1, np.nan)
            
            # Only calculate ratio where both images have valid data
            valid_pixels = valid_mask & (np.abs(ndwi1) > epsilon)
            ratio[valid_pixels] = ndwi2[valid_pixels] / ndwi1[valid_pixels]
            
            # Clip extreme ratios (likely noise)
            ratio = np.clip(ratio, 0.1, 10.0)
            
            # Calculate some statistics
            valid_ratios = ratio[~np.isnan(ratio)]
            if len(valid_ratios) > 0:
                print(f"   Ratio stats: {valid_ratios.min():.3f} to {valid_ratios.max():.3f}, mean: {valid_ratios.mean():.3f}")
                
                # Identify significant changes
                change_pixels = np.sum((valid_ratios > 1.5) | (valid_ratios < 0.67))
                change_percent = change_pixels / len(valid_ratios) * 100
                print(f"   Significant changes (>50%): {change_pixels:,} pixels ({change_percent:.2f}%)")
            
            # Create output filename
            output_name = f"{date1.strftime('%Y-%m-%d')}_to_{date2.strftime('%Y-%m-%d')}_ratio.tif"
            output_path = os.path.join(output_dir, output_name)
            
            # Update profile for output
            profile.update({
                'dtype': 'float32',
                'nodata': np.nan
            })
            
            # Save ratio image
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(ratio.astype(np.float32), 1)
            
            print(f"   ✅ Saved: {output_name}")
            ratios_calculated.append((date1, date2, output_path, change_percent))
            
        except Exception as e:
            print(f"   ❌ Error calculating ratio: {e}")
            continue
    
    # Create summary visualization
    if ratios_calculated:
        create_ratio_summary(ratios_calculated, output_dir)
    
    print(f"\n🎉 Temporal ratio calculation complete!")
    print(f"   Created {len(ratios_calculated)} ratio images")
    print(f"   Results saved to: {output_dir}")
    
    return ratios_calculated

def create_ratio_summary(ratios_data, output_dir):
    """Create summary plot of change percentages over time"""
    
    dates = [data[1] for data in ratios_data]  # End dates
    changes = [data[3] for data in ratios_data]  # Change percentages
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, changes, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Date')
    plt.ylabel('Significant Change (%)')
    plt.title('NDWI Temporal Changes - 2021\n(Pixels with >50% change between consecutive dates)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add annotations for high change periods
    max_change_idx = np.argmax(changes)
    max_change_date = dates[max_change_idx]
    max_change_val = changes[max_change_idx]
    
    plt.annotate(f'Max change: {max_change_val:.1f}%\n{max_change_date.strftime("%Y-%m-%d")}',
                xy=(max_change_date, max_change_val),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    summary_plot = os.path.join(output_dir, "temporal_changes_summary.png")
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    print(f"   📊 Summary plot saved: {os.path.basename(summary_plot)}")
    plt.show()

def main():
    """Main execution"""
    
    # Input and output directories
    input_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/ndwi/langtang2021"
    output_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/ndwi_ratio/langtang2021"
    
    # Check input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Calculate temporal ratios
    try:
        ratios_data = calculate_temporal_ratios(input_dir, output_dir)
        
        if ratios_data:
            print(f"\n💡 Usage tips:")
            print(f"   - Values ≈ 1.0: No change")
            print(f"   - Values > 1.5: NDWI increased (water appeared/brightened)")
            print(f"   - Values < 0.67: NDWI decreased (water disappeared/darkened)")
            print(f"   - Look for spatial clusters of extreme ratios")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()