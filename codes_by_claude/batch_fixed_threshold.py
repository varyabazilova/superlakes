import numpy as np
import rasterio
from rasterio.features import rasterize
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
import geopandas as gpd
import glob
import os
import pandas as pd
from datetime import datetime

def read_planet_image(tif_path):
    """Read Planet imagery with 4 bands (RGB + NIR)"""
    with rasterio.open(tif_path) as src:
        image = src.read()
        profile = src.profile
        image = np.transpose(image, (1, 2, 0))
        return image, profile

def compute_ndwi(image):
    """Compute NDWI using Green and NIR bands"""
    green = image[:, :, 1].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    
    denominator = green + nir
    denominator[denominator == 0] = np.finfo(np.float32).eps
    
    ndwi = (green - nir) / denominator
    return ndwi

def create_glacier_mask(glacier_shapefile, image_profile, image_shape):
    """Create a glacier mask from RGI shapefile"""
    glaciers = gpd.read_file(glacier_shapefile)
    
    if glaciers.crs != image_profile['crs']:
        glaciers = glaciers.to_crs(image_profile['crs'])
    
    transform = image_profile['transform']
    
    glacier_mask = rasterize(
        glaciers.geometry,
        out_shape=image_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8'
    ).astype(bool)
    
    return glacier_mask

def fixed_threshold_water(ndwi, threshold=0.2, glacier_mask=None, apply_cleaning=True):
    """Apply fixed threshold for water detection"""
    # Create initial water mask
    water_mask = ndwi > threshold
    
    # Restrict to glacier areas if mask provided
    if glacier_mask is not None:
        water_mask = water_mask & glacier_mask
    
    # Count initial detection
    initial_water_pixels = np.sum(water_mask)
    
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
        # Step 1: Opening - Remove scattered noise pixels
        water_mask_opened = binary_opening(water_mask, disk(2))
        opened_water_pixels = np.sum(water_mask_opened)
        removed_by_opening = initial_water_pixels - opened_water_pixels
        
        # Step 2: Closing - Fill small gaps in water bodies
        water_mask_closed = binary_closing(water_mask_opened, disk(3))
        closed_water_pixels = np.sum(water_mask_closed)
        added_by_closing = closed_water_pixels - opened_water_pixels
        
        # Step 3: Remove small objects - Remove tiny remaining artifacts
        min_size = 50  # Minimum 50 pixels for a water body
        water_mask_final = remove_small_objects(water_mask_closed, min_size=min_size)
        final_water_pixels = np.sum(water_mask_final)
        removed_by_size = closed_water_pixels - final_water_pixels
        
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
    
    return water_mask, cleaning_stats

def process_single_image(tif_path, threshold, glacier_shp, output_dir):
    """Process a single image and return statistics"""
    print(f"Processing {os.path.basename(tif_path)}...")
    
    # Extract date from filename
    basename = os.path.basename(tif_path)
    date_str = basename.split('_')[0]  # e.g., "2025-01-08"
    
    try:
        # Read image
        image, profile = read_planet_image(tif_path)
        
        # Create glacier mask
        glacier_mask = create_glacier_mask(glacier_shp, profile, image.shape[:2])
        
        # Compute NDWI
        ndwi = compute_ndwi(image)
        
        # Apply fixed threshold
        water_mask, cleaning_stats = fixed_threshold_water(
            ndwi, threshold=threshold, glacier_mask=glacier_mask, apply_cleaning=False
        )
        
        # Save outputs
        base_output = os.path.join(output_dir, basename.replace('.tif', '_fixed'))
        
        # Save water mask
        profile_out = profile.copy()
        profile_out.update(dtype='uint8', count=1)
        water_output = f"{base_output}_water.tif"
        
        with rasterio.open(water_output, 'w', **profile_out) as dst:
            dst.write(water_mask.astype(np.uint8), 1)
        
        # Save NDWI
        profile_ndwi = profile.copy()
        profile_ndwi.update(dtype='float32', count=1, nodata=np.nan)
        ndwi_output = f"{base_output}_ndwi.tif"
        
        with rasterio.open(ndwi_output, 'w', **profile_ndwi) as dst:
            dst.write(ndwi.astype(np.float32), 1)
        
        # Calculate statistics
        water_pixels = np.sum(water_mask)
        glacier_pixels = np.sum(glacier_mask)
        total_pixels = ndwi.size
        
        # Calculate areas (assuming 3m pixel size for Planet)
        pixel_area_km2 = (3 * 3) / 1e6  # 9 m² per pixel = 9e-6 km²
        glacier_area_km2 = glacier_pixels * pixel_area_km2
        water_area_km2 = water_pixels * pixel_area_km2
        
        stats = {
            'filename': basename,
            'date': date_str,
            'timestamp': pd.to_datetime(date_str),
            'total_pixels': total_pixels,
            'glacier_pixels': glacier_pixels,
            'water_pixels': water_pixels,
            'water_pct_of_glacier': 100 * water_pixels / glacier_pixels if glacier_pixels > 0 else 0,
            'water_pct_of_total': 100 * water_pixels / total_pixels,
            'threshold_used': threshold,
            'ndwi_min': float(np.nanmin(ndwi[glacier_mask])),
            'ndwi_max': float(np.nanmax(ndwi[glacier_mask])),
            'ndwi_mean': float(np.nanmean(ndwi[glacier_mask])),
            'ndwi_std': float(np.nanstd(ndwi[glacier_mask])),
            'glacier_area_km2': glacier_area_km2,
            'water_area_km2': water_area_km2,
            'initial_water_pixels': cleaning_stats['initial_pixels'],
            'noise_removed_pixels': cleaning_stats['total_removed'],
            'cleaning_efficiency_pct': cleaning_stats['cleaning_efficiency'],
            'removed_by_opening': cleaning_stats['removed_by_opening'],
            'added_by_closing': cleaning_stats['added_by_closing'],
            'removed_by_size': cleaning_stats['removed_by_size'],
            'water_mask_file': water_output,
            'ndwi_file': ndwi_output
        }
        
        print(f"  ✅ {basename}: {water_pixels:,} water pixels ({100*water_pixels/glacier_pixels:.2f}% of glaciers)")
        return stats
        
    except Exception as e:
        print(f"  ❌ Error processing {basename}: {str(e)}")
        return None

def main():
    # Configuration
    input_dir = "testimages25"
    output_dir = "outputs_fixed_raw"
    glacier_shp = "/Users/varyabazilova/Desktop/glacial_lakes/RGI2000-v7-3/RGI2000-v7.0-C-15_south_asia_east.shp"
    threshold = 0.2
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mosaic files
    tif_files = glob.glob(os.path.join(input_dir, "*_composite_mosaic.tif"))
    tif_files.sort()
    
    print(f"Found {len(tif_files)} mosaic files to process")
    print(f"Using fixed threshold: {threshold}")
    print("Processing includes: NDWI calculation → Fixed thresholding → Glacier masking (NO cleaning)")
    print()
    
    # Process all files
    all_stats = []
    for tif_path in tif_files:
        stats = process_single_image(tif_path, threshold, glacier_shp, output_dir)
        if stats:
            all_stats.append(stats)
    
    # Save summary CSV
    if all_stats:
        df = pd.DataFrame(all_stats)
        df = df.sort_values('timestamp')
        
        csv_output = os.path.join(output_dir, "langtang_timeseries_fixed_raw.csv")
        df.to_csv(csv_output, index=False)
        
        print(f"\n📊 SUMMARY RESULTS:")
        print(f"Processed {len(all_stats)} images successfully")
        print(f"Results saved to: {csv_output}")
        print()
        print("Water detection summary:")
        for _, row in df.iterrows():
            print(f"  {row['date']}: {row['water_pct_of_glacier']:.2f}% of glaciers ({row['water_pixels']:,} pixels)")
        
        print(f"\nTemporal pattern:")
        print(f"  Min water: {df['water_pct_of_glacier'].min():.2f}% ({df.loc[df['water_pct_of_glacier'].idxmin(), 'date']})")
        print(f"  Max water: {df['water_pct_of_glacier'].max():.2f}% ({df.loc[df['water_pct_of_glacier'].idxmax(), 'date']})")
        print(f"  Average: {df['water_pct_of_glacier'].mean():.2f}%")

if __name__ == "__main__":
    main()