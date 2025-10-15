import os
import glob
import pandas as pd
import numpy as np
import rasterio
from datetime import datetime
import re
from water_detection_cleaned import read_planet_image, compute_ndwi, create_glacier_mask, otsu_threshold_water_cleaned
import argparse

def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from Planet filename
    Supports various Planet naming conventions
    """
    # Common Planet patterns
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{8})',              # YYYYMMDD
        r'(\d{4}_\d{2}_\d{2})',  # YYYY_MM_DD
    ]
    
    basename = os.path.basename(filename)
    
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            date_str = match.group(1)
            # Normalize to YYYY-MM-DD format
            if len(date_str) == 8:  # YYYYMMDD
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            elif '_' in date_str:  # YYYY_MM_DD
                date_str = date_str.replace('_', '-')
            
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue
    
    print(f"Warning: Could not extract timestamp from {basename}")
    return None

def process_single_image_cleaned(image_path, glacier_shp, output_dir=None):
    """
    Process a single image with morphological cleaning and return statistics
    
    Parameters:
    image_path (str): Path to Planet TIFF file
    glacier_shp (str): Path to RGI glacier shapefile
    output_dir (str, optional): Directory to save outputs
    
    Returns:
    dict: Statistics dictionary
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    try:
        # Read image
        image, profile = read_planet_image(image_path)
        
        # Extract timestamp
        timestamp = extract_timestamp_from_filename(image_path)
        
        # Create glacier mask for this specific image extent
        glacier_mask = None
        if glacier_shp:
            print(f"  Creating glacier mask for image extent: {image.shape[:2]}")
            glacier_mask = create_glacier_mask(glacier_shp, profile, image.shape[:2])
        
        # Compute NDWI
        ndwi = compute_ndwi(image)
        
        # Apply Otsu thresholding with morphological cleaning
        water_mask, threshold_val, cleaning_stats = otsu_threshold_water_cleaned(ndwi, glacier_mask)
        
        # Calculate statistics
        total_pixels = ndwi.size
        glacier_pixels = np.sum(glacier_mask) if glacier_mask is not None else total_pixels
        water_pixels = np.sum(water_mask)
        
        # NDWI statistics for glacier areas
        if glacier_mask is not None:
            glacier_ndwi = ndwi[glacier_mask & np.isfinite(ndwi)]
        else:
            glacier_ndwi = ndwi[np.isfinite(ndwi)]
        
        stats = {
            'filename': os.path.basename(image_path),
            'timestamp': timestamp,
            'date': timestamp.strftime('%Y-%m-%d') if timestamp else 'Unknown',
            'total_pixels': total_pixels,
            'glacier_pixels': glacier_pixels,
            'water_pixels': water_pixels,
            'water_pct_of_glacier': 100 * water_pixels / glacier_pixels if glacier_pixels > 0 else 0,
            'water_pct_of_total': 100 * water_pixels / total_pixels,
            'otsu_threshold': threshold_val,
            'ndwi_min': glacier_ndwi.min(),
            'ndwi_max': glacier_ndwi.max(),
            'ndwi_mean': glacier_ndwi.mean(),
            'ndwi_std': glacier_ndwi.std(),
            'glacier_area_km2': glacier_pixels * (profile['transform'][0] ** 2) / 1e6,  # Assuming UTM
            'water_area_km2': water_pixels * (profile['transform'][0] ** 2) / 1e6,
            # Cleaning statistics
            'initial_water_pixels': cleaning_stats['initial_pixels'],
            'noise_removed_pixels': cleaning_stats['total_removed'],
            'cleaning_efficiency_pct': cleaning_stats['cleaning_efficiency'],
            'removed_by_opening': cleaning_stats['removed_by_opening'],
            'added_by_closing': cleaning_stats['added_by_closing'],
            'removed_by_size': cleaning_stats['removed_by_size']
        }
        
        # Save outputs if directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output filename base
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save water mask
            water_output = os.path.join(output_dir, f"{base_name}_water_mask_cleaned.tif")
            profile_out = profile.copy()
            profile_out.update(dtype='uint8', count=1)
            
            with rasterio.open(water_output, 'w', **profile_out) as dst:
                dst.write(water_mask.astype(np.uint8), 1)
            
            # Save NDWI
            ndwi_output = os.path.join(output_dir, f"{base_name}_ndwi.tif")
            profile_ndwi = profile.copy()
            profile_ndwi.update(dtype='float32', count=1, nodata=np.nan)
            
            with rasterio.open(ndwi_output, 'w', **profile_ndwi) as dst:
                dst.write(ndwi.astype(np.float32), 1)
            
            stats['water_mask_file'] = water_output
            stats['ndwi_file'] = ndwi_output
        
        print(f"  Water: {water_pixels:,} pixels ({stats['water_pct_of_glacier']:.2f}% of glacier)")
        print(f"  Cleaning: Removed {cleaning_stats['total_removed']:,} noise pixels ({cleaning_stats['cleaning_efficiency']:.1f}% reduction)")
        
        return stats
    
    except Exception as e:
        print(f"  Error processing {image_path}: {str(e)}")
        return {
            'filename': os.path.basename(image_path),
            'timestamp': extract_timestamp_from_filename(image_path),
            'error': str(e)
        }

def batch_process_images_cleaned(input_pattern, glacier_shp=None, output_dir=None, output_csv=None):
    """
    Batch process multiple Planet images with morphological cleaning
    
    Parameters:
    input_pattern (str): Glob pattern for input TIFF files
    glacier_shp (str, optional): Path to RGI glacier shapefile
    output_dir (str, optional): Directory to save individual outputs
    output_csv (str, optional): Path to save summary CSV
    """
    # Find all matching files
    image_files = glob.glob(input_pattern)
    image_files.sort()
    
    if not image_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("Using NDWI + Otsu thresholding with morphological cleaning")
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing images...")
        stats = process_single_image_cleaned(image_path, glacier_shp, output_dir)
        results.append(stats)
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        df = df.sort_values('timestamp')
    
    # Save summary CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSummary saved to: {output_csv}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY - MORPHOLOGICAL CLEANING")
    print("="*60)
    
    # Filter successful results
    successful = df[~df.get('error', pd.Series(dtype=object)).notna()]
    
    if len(successful) > 0:
        print(f"Successfully processed: {len(successful)}/{len(df)} images")
        print(f"Date range: {successful['date'].min()} to {successful['date'].max()}")
        print(f"Water % range: {successful['water_pct_of_glacier'].min():.2f}% to {successful['water_pct_of_glacier'].max():.2f}%")
        print(f"Mean water coverage: {successful['water_pct_of_glacier'].mean():.2f}% ± {successful['water_pct_of_glacier'].std():.2f}%")
        
        if 'water_area_km2' in successful.columns:
            print(f"Water area range: {successful['water_area_km2'].min():.3f} to {successful['water_area_km2'].max():.3f} km²")
        
        # Cleaning statistics
        if 'cleaning_efficiency_pct' in successful.columns:
            print(f"\nCleaning efficiency:")
            print(f"  Average noise reduction: {successful['cleaning_efficiency_pct'].mean():.1f}%")
            print(f"  Range: {successful['cleaning_efficiency_pct'].min():.1f}% to {successful['cleaning_efficiency_pct'].max():.1f}%")
            total_initial = successful['initial_water_pixels'].sum()
            total_removed = successful['noise_removed_pixels'].sum()
            print(f"  Total noise removed: {total_removed:,} pixels ({100*total_removed/total_initial:.1f}% overall)")
    
    errors = df[df.get('error', pd.Series(dtype=object)).notna()]
    if len(errors) > 0:
        print(f"\nErrors: {len(errors)} images failed")
        for _, row in errors.iterrows():
            print(f"  {row['filename']}: {row['error']}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Batch water detection with morphological cleaning')
    parser.add_argument('input_pattern', help='Glob pattern for input TIFF files (e.g., "*.tif" or "path/to/*.tif")')
    parser.add_argument('--glacier_shp', help='Path to RGI glacier shapefile')
    parser.add_argument('--output_dir', default='outputs_cleaned', 
                       help='Directory to save individual water masks and NDWI files (default: outputs_cleaned)')
    parser.add_argument('--output_csv', default='water_detection_cleaned_summary.csv', 
                       help='Path to save summary CSV (default: water_detection_cleaned_summary.csv)')
    
    args = parser.parse_args()
    
    # Run batch processing
    df = batch_process_images_cleaned(
        input_pattern=args.input_pattern,
        glacier_shp=args.glacier_shp,
        output_dir=args.output_dir,
        output_csv=args.output_csv
    )
    
    print(f"\n🎉 Processing complete!")
    print(f"Results saved to: {args.output_csv}")
    if args.output_dir:
        print(f"Individual outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()