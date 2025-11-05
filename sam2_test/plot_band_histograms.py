#!/usr/bin/env python3
"""
Plot histograms for each band of the clipped image
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os

def plot_band_histograms_clipped(image_path, shapefile_path):
    """
    Plot histogram for each band of image clipped by shapefile
    """
    print(f"📊 Band Histogram Analysis")
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
        
        num_bands = clipped_data.shape[0]
        band_names = ['Red', 'Green', 'Blue', 'NIR'][:num_bands]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(num_bands):
            band_data = clipped_data[i]
            
            # Remove any extreme values/nodata
            valid_mask = (band_data > 0) & (band_data < 10000)  # Reasonable range
            valid_data = band_data[valid_mask]
            
            print(f"\n📊 {band_names[i]} Band (Band {i}):")
            print(f"   Valid pixels: {len(valid_data):,}")
            print(f"   Value range: {valid_data.min()} to {valid_data.max()}")
            print(f"   Mean: {valid_data.mean():.1f}")
            
            # Plot histogram
            axes[i].hist(valid_data, bins=100, alpha=0.7, color=['red', 'green', 'blue', 'purple'][i])
            axes[i].set_title(f'{band_names[i]} Band Histogram\n(Band {i})')
            axes[i].set_xlabel('Pixel Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots if less than 4 bands
        for i in range(num_bands, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(os.path.dirname(image_path), "band_histograms_clipped.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Histogram saved: {output_path}")
        
        plt.show()

def main():
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
    
    # Generate histograms
    try:
        plot_band_histograms_clipped(image_path, shapefile_path)
        print(f"\n🎉 Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()