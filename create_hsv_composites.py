#!/usr/bin/env python3
"""
Create HSV composites from Planet 4-band imagery using false color (NIR-Red-Green).
Processes all 2025 mosaic images and saves HSV channels.
"""

import os
import glob
import numpy as np
import rasterio
import cv2
from rasterio.transform import from_bounds
import warnings
warnings.filterwarnings('ignore')

def create_false_color_hsv(input_tif, output_dir):
    """
    Create HSV composite from Planet 4-band imagery using false color (NIR-Red-Green).
    
    Parameters:
    - input_tif: Path to 4-band Planet mosaic (B-G-R-NIR)
    - output_dir: Directory to save HSV outputs
    """
    
    # Extract date from filename
    filename = os.path.basename(input_tif)
    date = filename.split('_')[0]  # Extract YYYY-MM-DD
    
    print(f"Processing {date}...")
    
    with rasterio.open(input_tif) as src:
        # Read all 4 bands (Blue, Green, Red, NIR)
        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32) 
        red = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        
        # Get metadata for output files
        meta = src.meta.copy()
        meta.update(dtype='uint8', count=1)
        
        # Create false color composite (NIR-Red-Green as RGB)
        # Stack as (NIR, Red, Green) for false color visualization
        false_color = np.stack([nir, red, green], axis=2)
        
        # Handle nodata and normalize to 0-255 for each channel
        for i in range(3):
            band = false_color[:, :, i]
            # Mask nodata values
            valid_mask = (band != src.nodata) & (band > 0)
            
            if np.any(valid_mask):
                # Normalize using 2nd and 98th percentiles for better contrast
                p2, p98 = np.percentile(band[valid_mask], [2, 98])
                band_norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                band_norm[~valid_mask] = 0  # Set invalid pixels to 0
                false_color[:, :, i] = band_norm
            else:
                false_color[:, :, i] = 0
        
        # Convert to uint8
        false_color = false_color.astype(np.uint8)
        
        # Convert RGB to HSV using OpenCV
        hsv = cv2.cvtColor(false_color, cv2.COLOR_RGB2HSV)
        
        # Extract HSV channels
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1] 
        value = hsv[:, :, 2]
        
        # Save HSV as single 3-band GeoTIFF
        hsv_path = os.path.join(output_dir, f"{date}_hsv.tif")
        meta_hsv = meta.copy()
        meta_hsv.update(count=3)
        
        with rasterio.open(hsv_path, 'w', **meta_hsv) as dst:
            dst.write(hue, 1)        # Band 1: Hue
            dst.write(saturation, 2) # Band 2: Saturation  
            dst.write(value, 3)      # Band 3: Value
        
        # Also save the false color composite for reference
        false_color_path = os.path.join(output_dir, f"{date}_false_color.tif")
        meta_rgb = meta.copy()
        meta_rgb.update(count=3)
        
        with rasterio.open(false_color_path, 'w', **meta_rgb) as dst:
            for i in range(3):
                dst.write(false_color[:, :, i], i+1)
        
        print(f"  Saved 3-band HSV and false color composite for {date}")
        
        # Print some statistics
        valid_pixels = (hue > 0) | (saturation > 0) | (value > 0)
        if np.any(valid_pixels):
            print(f"  HSV ranges - H: {hue[valid_pixels].min()}-{hue[valid_pixels].max()}, "
                  f"S: {saturation[valid_pixels].min()}-{saturation[valid_pixels].max()}, "
                  f"V: {value[valid_pixels].min()}-{value[valid_pixels].max()}")

def main():
    # Input and output directories
    input_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics/langtang2025_harmonized_mosaics"
    output_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/HSV_test"
    
    # Find all mosaic files
    mosaic_files = glob.glob(os.path.join(input_dir, "*_composite_mosaic.tif"))
    mosaic_files.sort()
    
    print(f"Found {len(mosaic_files)} mosaic files to process")
    print("Creating HSV composites from false color (NIR-Red-Green)...")
    print()
    
    # Process each mosaic
    for mosaic_file in mosaic_files:
        try:
            create_false_color_hsv(mosaic_file, output_dir)
        except Exception as e:
            print(f"Error processing {mosaic_file}: {e}")
            continue
    
    print()
    print("HSV processing complete!")
    print(f"Output files saved to: {output_dir}")
    
    # List output files
    output_files = glob.glob(os.path.join(output_dir, "*.tif"))
    print(f"Created {len(output_files)} HSV output files")

if __name__ == "__main__":
    main()