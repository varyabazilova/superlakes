import numpy as np
import rasterio
from rasterio.features import rasterize
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
import geopandas as gpd
import glob
import os
import pandas as pd
from datetime import datetime
import argparse

def create_analysis_mask(mask_shapefile, image_profile, image_shape):
    """
    Create an analysis mask from shapefile (e.g., slope mask, glacier mask)
    
    Parameters:
    mask_shapefile (str): Path to mask shapefile
    image_profile: Rasterio profile of the image
    image_shape (tuple): Shape of the image (height, width)
    
    Returns:
    numpy.ndarray: Boolean mask where True = analysis area
    """
    mask_polygons = gpd.read_file(mask_shapefile)
    print(f"Read {len(mask_polygons)} mask polygons")
    
    if mask_polygons.crs != image_profile['crs']:
        print(f"Reprojecting mask from {mask_polygons.crs} to {image_profile['crs']}")
        mask_polygons = mask_polygons.to_crs(image_profile['crs'])
    
    transform = image_profile['transform']
    
    analysis_mask = rasterize(
        mask_polygons.geometry,
        out_shape=image_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8'
    ).astype(bool)
    
    mask_pixels = np.sum(analysis_mask)
    total_pixels = analysis_mask.size
    print(f"Analysis area: {mask_pixels} pixels ({100*mask_pixels/total_pixels:.2f}% of image)")
    
    return analysis_mask

def apply_water_threshold(ndwi, threshold=0.2, analysis_mask=None, apply_cleaning=False):
    """
    Apply threshold for water detection
    
    Parameters:
    ndwi (numpy.ndarray): NDWI values
    threshold (float): Threshold value (default: 0.2)
    analysis_mask (numpy.ndarray, optional): Boolean mask to restrict analysis to specific areas
    apply_cleaning (bool): Whether to apply morphological cleaning
    
    Returns:
    tuple: (water_mask, cleaning_stats)
    """
    print(f"Applying threshold: NDWI > {threshold}")
    
    # Create initial water mask
    water_mask = ndwi > threshold
    
    # Restrict to analysis areas if mask provided
    if analysis_mask is not None:
        water_mask = water_mask & analysis_mask
        
        # Count valid NDWI pixels in analysis area for context
        valid_ndwi = ndwi[analysis_mask & np.isfinite(ndwi)]
        above_threshold = np.sum(valid_ndwi > threshold)
        print(f"Pixels above threshold within analysis area: {above_threshold:,} / {len(valid_ndwi):,} ({100*above_threshold/len(valid_ndwi):.2f}%)")
    
    # Count initial detection
    initial_water_pixels = np.sum(water_mask)
    print(f"Initial water pixels: {initial_water_pixels:,}")
    
    cleaning_stats = {
        'initial_pixels': initial_water_pixels,
        'final_pixels': initial_water_pixels,
        'removed_by_opening': 0,
        'added_by_closing': 0,
        'removed_by_size': 0,
        'total_removed': 0,
        'cleaning_efficiency': 0
    }
    
    if apply_cleaning and initial_water_pixels > 0:
        print("Applying morphological cleaning...")
        
        # Step 1: Opening - Remove scattered noise pixels
        print("  Step 1: Opening (remove noise)")
        water_mask_opened = binary_opening(water_mask, disk(2))
        opened_water_pixels = np.sum(water_mask_opened)
        removed_by_opening = initial_water_pixels - opened_water_pixels
        print(f"    Removed {removed_by_opening:,} noisy pixels")
        
        # Step 2: Closing - Fill small gaps in water bodies
        print("  Step 2: Closing (fill gaps)")
        water_mask_closed = binary_closing(water_mask_opened, disk(3))
        closed_water_pixels = np.sum(water_mask_closed)
        added_by_closing = closed_water_pixels - opened_water_pixels
        print(f"    Added {added_by_closing:,} pixels to fill gaps")
        
        # Step 3: Remove small objects - Remove tiny remaining artifacts
        print("  Step 3: Size filtering (remove small objects)")
        min_size = 50  # Minimum 50 pixels for a water body
        water_mask_final = remove_small_objects(water_mask_closed, min_size=min_size)
        final_water_pixels = np.sum(water_mask_final)
        removed_by_size = closed_water_pixels - final_water_pixels
        print(f"    Removed {removed_by_size:,} pixels from small objects")
        
        # Update cleaning statistics
        cleaning_stats.update({
            'final_pixels': final_water_pixels,
            'removed_by_opening': removed_by_opening,
            'added_by_closing': added_by_closing,
            'removed_by_size': removed_by_size,
            'total_removed': initial_water_pixels - final_water_pixels,
            'cleaning_efficiency': 100 * (initial_water_pixels - final_water_pixels) / initial_water_pixels
        })
        
        water_mask = water_mask_final
        
        print(f"Cleaning summary:")
        print(f"  Initial: {initial_water_pixels:,} pixels")
        print(f"  Final: {final_water_pixels:,} pixels")
        print(f"  Removed: {cleaning_stats['total_removed']:,} pixels ({cleaning_stats['cleaning_efficiency']:.1f}% reduction)")
    
    return water_mask, cleaning_stats

def process_single_ndwi(ndwi_path, threshold, mask_shp, output_dir, apply_cleaning=False):
    """
    Process a single NDWI image to create water mask
    
    Parameters:
    ndwi_path (str): Path to NDWI TIFF file
    threshold (float): NDWI threshold for water detection
    mask_shp (str): Path to mask shapefile (slope mask, glacier mask, etc.)
    output_dir (str): Directory to save water mask
    apply_cleaning (bool): Whether to apply morphological cleaning
    
    Returns:
    dict: Statistics for this image
    """
    basename = os.path.basename(ndwi_path)
    print(f"Processing {basename}...")
    
    # Extract date from filename
    date_str = basename.split('_')[0]  # e.g., "2025-01-08"
    
    try:
        # Read NDWI
        with rasterio.open(ndwi_path) as src:
            ndwi = src.read(1)
            profile = src.profile
        
        print(f"NDWI range: {np.nanmin(ndwi):.3f} - {np.nanmax(ndwi):.3f}")
        
        # Create analysis mask
        analysis_mask = create_analysis_mask(mask_shp, profile, ndwi.shape)
        
        # Apply threshold
        water_mask, cleaning_stats = apply_water_threshold(
            ndwi, threshold=threshold, analysis_mask=analysis_mask, apply_cleaning=apply_cleaning
        )
        
        # Save water mask
        water_filename = basename.replace('_ndwi.tif', '_water_mask.tif')
        water_output = os.path.join(output_dir, water_filename)
        
        profile_out = profile.copy()
        profile_out.update(dtype='uint8', count=1, nodata=255)
        
        with rasterio.open(water_output, 'w', **profile_out) as dst:
            dst.write(water_mask.astype(np.uint8), 1)
        
        # Calculate statistics
        water_pixels = np.sum(water_mask)
        analysis_pixels = np.sum(analysis_mask)
        total_pixels = ndwi.size
        
        # Calculate areas (assuming 3m pixel size for Planet)
        pixel_area_km2 = (3 * 3) / 1e6  # 9 m² per pixel = 9e-6 km²
        analysis_area_km2 = analysis_pixels * pixel_area_km2
        water_area_km2 = water_pixels * pixel_area_km2
        
        stats = {
            'filename': basename.replace('_ndwi.tif', '.tif'),
            'date': date_str,
            'timestamp': pd.to_datetime(date_str),
            'total_pixels': total_pixels,
            'analysis_pixels': analysis_pixels,
            'water_pixels': water_pixels,
            'water_pct_of_analysis': 100 * water_pixels / analysis_pixels if analysis_pixels > 0 else 0,
            'water_pct_of_total': 100 * water_pixels / total_pixels,
            'threshold_used': threshold,
            'ndwi_min': float(np.nanmin(ndwi[analysis_mask])),
            'ndwi_max': float(np.nanmax(ndwi[analysis_mask])),
            'ndwi_mean': float(np.nanmean(ndwi[analysis_mask])),
            'ndwi_std': float(np.nanstd(ndwi[analysis_mask])),
            'analysis_area_km2': analysis_area_km2,
            'water_area_km2': water_area_km2,
            'initial_water_pixels': cleaning_stats['initial_pixels'],
            'noise_removed_pixels': cleaning_stats['total_removed'],
            'cleaning_efficiency_pct': cleaning_stats['cleaning_efficiency'],
            'removed_by_opening': cleaning_stats['removed_by_opening'],
            'added_by_closing': cleaning_stats['added_by_closing'],
            'removed_by_size': cleaning_stats['removed_by_size'],
            'water_mask_file': water_output,
            'ndwi_file': ndwi_path
        }
        
        print(f"  [SUCCESS] {basename}: {water_pixels:,} water pixels ({100*water_pixels/analysis_pixels:.2f}% of analysis area)")
        print(f"  Water mask saved to: {water_output}")
        return stats
        
    except Exception as e:
        print(f"  [ERROR] Error processing {basename}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Create water masks from NDWI images')
    parser.add_argument('--ndwi_dir', default='/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/images/langtang2025/ndwi',
                       help='Directory containing NDWI files')
    parser.add_argument('--threshold', type=float, default=0.2, help='NDWI threshold (default: 0.2)')
    parser.add_argument('--mask_shp', 
                       default='/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/other_data/vect/langtang_slope_lte10_mask_utm_vect_fix.shp',
                       help='Path to analysis mask shapefile (slope mask, glacier mask, etc.)')
    parser.add_argument('--output_dir', default='outputs_water_masks', help='Output directory for water masks')
    parser.add_argument('--cleaning', action='store_true', help='Apply morphological cleaning')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all NDWI files
    ndwi_files = glob.glob(os.path.join(args.ndwi_dir, "*_ndwi.tif"))
    ndwi_files.sort()
    
    if len(ndwi_files) == 0:
        print(f"[ERROR] No NDWI files found in {args.ndwi_dir}")
        print("Make sure you've run 'calculate_ndwi.py' first!")
        return
    
    print(f"Found {len(ndwi_files)} NDWI files to process")
    print(f"NDWI directory: {args.ndwi_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using threshold: {args.threshold}")
    print(f"Morphological cleaning: {'Yes' if args.cleaning else 'No'}")
    print("Processing: NDWI → Thresholding → Analysis masking → Water mask")
    print()
    
    # Process all files
    all_stats = []
    for ndwi_path in ndwi_files:
        stats = process_single_ndwi(ndwi_path, args.threshold, args.mask_shp, args.output_dir, args.cleaning)
        if stats:
            all_stats.append(stats)
        print()  # Empty line between files
    
    # Save summary CSV
    if all_stats:
        df = pd.DataFrame(all_stats)
        df = df.sort_values('timestamp')
        
        cleaning_suffix = "_cleaned" if args.cleaning else "_raw"
        csv_output = os.path.join(args.output_dir, f"langtang_water_detection{cleaning_suffix}.csv")
        df.to_csv(csv_output, index=False)
        
        print(f"SUMMARY RESULTS:")
        print(f"Processed {len(all_stats)} images successfully")
        print(f"Results saved to: {csv_output}")
        print()
        print("Water detection summary:")
        for _, row in df.iterrows():
            print(f"  {row['date']}: {row['water_pct_of_analysis']:.2f}% of analysis area ({row['water_pixels']:,} pixels)")
        
        print(f"\nTemporal pattern:")
        print(f"  Min water: {df['water_pct_of_analysis'].min():.2f}% ({df.loc[df['water_pct_of_analysis'].idxmin(), 'date']})")
        print(f"  Max water: {df['water_pct_of_analysis'].max():.2f}% ({df.loc[df['water_pct_of_analysis'].idxmax(), 'date']})")
        print(f"  Average: {df['water_pct_of_analysis'].mean():.2f}%")

if __name__ == "__main__":
    main()