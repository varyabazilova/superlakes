#!/usr/bin/env python3
"""
Plot NDWI histogram for image clipped by shapefile
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os

def plot_ndwi_histogram_clipped(image_path, shapefile_path):
    """
    Calculate NDWI and plot histogram for image clipped by shapefile
    
    Parameters:
    - image_path: Path to satellite image (4-band: R,G,B,NIR)
    - shapefile_path: Path to shapefile for clipping
    """
    print(f"📊 NDWI Histogram Analysis")
    print("=" * 50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Shapefile: {os.path.basename(shapefile_path)}")
    
    # Load the shapefile
    print("🗂️ Loading shapefile...")
    shapefile = gpd.read_file(shapefile_path)
    print(f"   Found {len(shapefile)} polygon(s)")
    print(f"   Shapefile CRS: {shapefile.crs}")
    
    # Load and clip the image
    print("🖼️ Loading and clipping image...")
    with rasterio.open(image_path) as src:
        print(f"   Image CRS: {src.crs}")
        
        # Reproject shapefile to match image CRS if needed
        if shapefile.crs != src.crs:
            print(f"   Reprojecting shapefile from {shapefile.crs} to {src.crs}")
            shapefile = shapefile.to_crs(src.crs)
        
        # Clip image to shapefile
        clipped_data, clipped_transform = mask(src, shapefile.geometry, crop=True)
        
        print(f"   Original image shape: {src.shape}")
        print(f"   Clipped image shape: {clipped_data.shape}")
        print(f"   Number of bands: {clipped_data.shape[0]}")
        
        # Calculate NDWI from clipped data
        # Assuming bands are: Red(0), Green(1), Blue(2), NIR(3)
        green = clipped_data[1].astype(float)
        nir = clipped_data[3].astype(float)
        
        print("🧮 Calculating NDWI...")
        print("   Formula: NDWI = (Green - NIR) / (Green + NIR)")
        
        # Calculate NDWI = (Green - NIR) / (Green + NIR)
        ndwi = np.where((green + nir) != 0, (green - nir) / (green + nir), -999)
        
        # Mask out nodata/invalid values
        valid_mask = (ndwi > -2) & (ndwi < 2)
        valid_ndwi = ndwi[valid_mask]
        
        print(f"   Valid pixels: {len(valid_ndwi):,} / {ndwi.size:,}")
        print(f"   NDWI range: {valid_ndwi.min():.3f} to {valid_ndwi.max():.3f}")
        print(f"   NDWI mean: {valid_ndwi.mean():.3f}")
        print(f"   NDWI std: {valid_ndwi.std():.3f}")
    
    # Calculate water percentages for different thresholds
    water_0 = np.sum(valid_ndwi > 0.0)
    water_01 = np.sum(valid_ndwi > 0.1)
    water_02 = np.sum(valid_ndwi > 0.2)
    
    print(f"\n🌊 Water detection with different thresholds:")
    print(f"   NDWI > 0.0:  {water_0:,} pixels ({water_0/len(valid_ndwi)*100:.2f}%)")
    print(f"   NDWI > 0.1:  {water_01:,} pixels ({water_01/len(valid_ndwi)*100:.2f}%)")
    print(f"   NDWI > 0.2:  {water_02:,} pixels ({water_02/len(valid_ndwi)*100:.2f}%)")
    
    # Create histogram plots
    print("\n📈 Creating histogram plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Full histogram
    ax1.hist(valid_ndwi, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(0.0, color='red', linestyle='--', linewidth=2, label='NDWI = 0.0')
    ax1.axvline(0.1, color='orange', linestyle='--', linewidth=2, label='NDWI = 0.1')
    ax1.axvline(0.2, color='green', linestyle='--', linewidth=2, label='NDWI = 0.2')
    ax1.set_xlabel('NDWI Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('NDWI Histogram (Full Range)\nClipped to Glacier Area')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zoomed histogram (water range)
    water_range_mask = (valid_ndwi > -0.3) & (valid_ndwi < 0.5)
    water_range_ndwi = valid_ndwi[water_range_mask]
    
    ax2.hist(water_range_ndwi, bins=50, alpha=0.7, color='cyan', edgecolor='black')
    ax2.axvline(0.0, color='red', linestyle='--', linewidth=2, label='NDWI = 0.0')
    ax2.axvline(0.1, color='orange', linestyle='--', linewidth=2, label='NDWI = 0.1')
    ax2.axvline(0.2, color='green', linestyle='--', linewidth=2, label='NDWI = 0.2')
    ax2.set_xlabel('NDWI Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('NDWI Histogram (Water Range: -0.3 to 0.5)\nClipped to Glacier Area')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(image_path), "ndwi_histogram_clipped.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Histogram saved: {output_path}")
    
    plt.show()
    
    # Threshold recommendations
    print(f"\n💡 Threshold Recommendations:")
    if water_02/len(valid_ndwi)*100 > 1.0 and water_02/len(valid_ndwi)*100 < 10.0:
        print(f"   ✅ NDWI > 0.2 looks good ({water_02/len(valid_ndwi)*100:.2f}% water)")
    elif water_01/len(valid_ndwi)*100 > 1.0 and water_01/len(valid_ndwi)*100 < 15.0:
        print(f"   ✅ NDWI > 0.1 looks good ({water_01/len(valid_ndwi)*100:.2f}% water)")
    elif water_0/len(valid_ndwi)*100 > 1.0 and water_0/len(valid_ndwi)*100 < 20.0:
        print(f"   ✅ NDWI > 0.0 looks good ({water_0/len(valid_ndwi)*100:.2f}% water)")
    else:
        print(f"   ⚠️ All thresholds might need adjustment")
        print(f"   💭 Consider custom threshold between 0.0 and 0.2")
    
    return valid_ndwi

def main():
    """Main execution"""
    
    # File paths
    image_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/2021-09-04_fcc_blurred_light_blur.tif"
    shapefile_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/clip_by_glacier.shp"
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    if not os.path.exists(shapefile_path):
        print(f"❌ Shapefile not found: {shapefile_path}")
        return
    
    # Generate histogram
    try:
        ndwi_data = plot_ndwi_histogram_clipped(image_path, shapefile_path)
        print(f"\n🎉 Analysis complete!")
        print(f"   Total valid NDWI pixels analyzed: {len(ndwi_data):,}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()