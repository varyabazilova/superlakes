#!/usr/bin/env python3
"""
Calculate NDWI for all years (2020-2024) mosaics
Output to organized year-based folders
"""

import numpy as np
import rasterio
import glob
import os
from pathlib import Path

def read_planet_image(tif_path):
    """
    Read Planet imagery with 4 bands (RGB + NIR)
    
    Parameters:
    tif_path (str): Path to the Planet TIFF file
    
    Returns:
    tuple: (image_array, profile) where image_array is (height, width, 4)
    """
    with rasterio.open(tif_path) as src:
        image = src.read()
        profile = src.profile
        image = np.transpose(image, (1, 2, 0))
        
        print(f"  Image shape: {image.shape}")
        print(f"  Data type: {image.dtype}")
        print(f"  Value range: {image.min()} - {image.max()}")
        
        return image, profile

def compute_ndwi(image):
    """
    Compute NDWI using Green and NIR bands
    NDWI = (Green - NIR) / (Green + NIR)
    
    Parameters:
    image (numpy.ndarray): Image array with shape (height, width, 4)
                          Assumes band order: R, G, B, NIR
    
    Returns:
    numpy.ndarray: NDWI values
    """
    green = image[:, :, 1].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    
    denominator = green + nir
    denominator[denominator == 0] = np.finfo(np.float32).eps
    
    ndwi = (green - nir) / denominator
    
    print(f"  NDWI range: {ndwi.min():.3f} - {ndwi.max():.3f}")
    
    return ndwi

def process_single_image(tif_path, output_dir):
    """
    Process a single image to calculate and save NDWI
    
    Parameters:
    tif_path (str): Path to input Planet TIFF
    output_dir (str): Directory to save NDWI output
    
    Returns:
    str: Path to saved NDWI file
    """
    basename = os.path.basename(tif_path)
    print(f"Processing {basename}...")
    
    try:
        # Read image
        image, profile = read_planet_image(tif_path)
        
        # Compute NDWI
        ndwi = compute_ndwi(image)
        
        # Prepare output filename
        ndwi_filename = basename.replace('_composite_mosaic.tif', '_ndwi.tif')
        ndwi_output = os.path.join(output_dir, ndwi_filename)
        
        # Save NDWI
        profile_ndwi = profile.copy()
        profile_ndwi.update(dtype='float32', count=1, nodata=np.nan)
        
        with rasterio.open(ndwi_output, 'w', **profile_ndwi) as dst:
            dst.write(ndwi.astype(np.float32), 1)
        
        print(f"  [SUCCESS] NDWI saved to: {ndwi_output}")
        return ndwi_output
        
    except Exception as e:
        print(f"  [ERROR] Error processing {basename}: {str(e)}")
        return None

def process_year(year, input_dir, output_base_dir):
    """
    Process all mosaics for a specific year
    
    Parameters:
    year (str): Year (e.g., "2020")
    input_dir (str): Directory containing mosaics for this year
    output_base_dir (str): Base directory for NDWI outputs
    
    Returns:
    int: Number of successfully processed files
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING YEAR: {year}")
    print(f"{'='*60}")
    
    # Create year-specific output directory
    output_dir = os.path.join(output_base_dir, f"langtang{year}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mosaic files for this year
    tif_files = glob.glob(os.path.join(input_dir, "*_composite_mosaic.tif"))
    tif_files.sort()
    
    print(f"Found {len(tif_files)} mosaic files for {year}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all files
    successful = 0
    for tif_path in tif_files:
        result = process_single_image(tif_path, output_dir)
        if result:
            successful += 1
        print()  # Empty line between files
    
    print(f"[SUMMARY {year}] Successfully processed {successful}/{len(tif_files)} images")
    return successful

def main():
    """
    Main function to process all years
    """
    print("NDWI CALCULATION FOR ALL YEARS (2020-2024)")
    print("="*60)
    
    # Configuration
    base_mosaic_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics"
    output_base_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/ndwi"
    
    # Year configurations
    years = {
        "2020": "langtang2020_harmonized_mosaics",
        "2021": "langtang2021_harmonized_mosaics", 
        "2022": "langtang2022_harmonized_mosaics",
        "2023": "langtang2023_harmonized_mosaics",
        "2024": "langtang2024_harmonized_mosaics"
    }
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each year
    total_successful = 0
    total_files = 0
    
    for year, mosaic_folder in years.items():
        input_dir = os.path.join(base_mosaic_dir, mosaic_folder)
        
        if os.path.exists(input_dir):
            successful = process_year(year, input_dir, output_base_dir)
            total_successful += successful
            
            # Count total files
            tif_files = glob.glob(os.path.join(input_dir, "*_composite_mosaic.tif"))
            total_files += len(tif_files)
        else:
            print(f"[WARNING] Directory not found: {input_dir}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {total_successful}/{total_files}")
    print(f"Output base directory: {output_base_dir}")
    print()
    print("NDWI files organized by year:")
    for year in years.keys():
        year_dir = os.path.join(output_base_dir, f"langtang{year}")
        if os.path.exists(year_dir):
            ndwi_count = len(glob.glob(os.path.join(year_dir, "*_ndwi.tif")))
            print(f"  langtang{year}/: {ndwi_count} NDWI files")
    
    print()
    print("Next step: Use 'map_water_from_ndwi.py' to create water masks")

if __name__ == "__main__":
    main()