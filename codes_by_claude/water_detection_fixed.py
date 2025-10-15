import numpy as np
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
import geopandas as gpd
import argparse

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
        
        print(f"Image shape: {image.shape}")
        print(f"Data type: {image.dtype}")
        print(f"Value range: {image.min()} - {image.max()}")
        
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
    
    print(f"NDWI range: {ndwi.min():.3f} - {ndwi.max():.3f}")
    
    return ndwi

def create_glacier_mask(glacier_shapefile, image_profile, image_shape):
    """
    Create a glacier mask from RGI shapefile
    
    Parameters:
    glacier_shapefile (str): Path to RGI glacier shapefile
    image_profile: Rasterio profile of the image
    image_shape (tuple): Shape of the image (height, width)
    
    Returns:
    numpy.ndarray: Boolean mask where True = glacier area
    """
    glaciers = gpd.read_file(glacier_shapefile)
    print(f"Read {len(glaciers)} glacier polygons")
    
    if glaciers.crs != image_profile['crs']:
        print(f"Reprojecting glaciers from {glaciers.crs} to {image_profile['crs']}")
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
    
    glacier_pixels = np.sum(glacier_mask)
    total_pixels = glacier_mask.size
    print(f"Glacier area: {glacier_pixels} pixels ({100*glacier_pixels/total_pixels:.2f}% of image)")
    
    return glacier_mask

def fixed_threshold_water(ndwi, threshold=0.2, glacier_mask=None, apply_cleaning=True):
    """
    Apply fixed threshold for water detection
    
    Parameters:
    ndwi (numpy.ndarray): NDWI values
    threshold (float): Fixed threshold value (default: 0.2)
    glacier_mask (numpy.ndarray, optional): Boolean mask to restrict analysis to glacier areas
    apply_cleaning (bool): Whether to apply morphological cleaning
    
    Returns:
    tuple: (water_mask, cleaning_stats)
    """
    print(f"Applying fixed threshold: NDWI > {threshold}")
    
    # Create initial water mask
    water_mask = ndwi > threshold
    
    # Restrict to glacier areas if mask provided
    if glacier_mask is not None:
        water_mask = water_mask & glacier_mask
        
        # Count valid NDWI pixels in glacier area for context
        valid_ndwi = ndwi[glacier_mask & np.isfinite(ndwi)]
        above_threshold = np.sum(valid_ndwi > threshold)
        print(f"Pixels above threshold within glaciers: {above_threshold:,} / {len(valid_ndwi):,} ({100*above_threshold/len(valid_ndwi):.2f}%)")
    
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

def visualize_fixed_threshold_results(image, ndwi, water_mask, threshold_val, cleaning_stats, glacier_mask=None):
    """
    Visualize the fixed threshold results
    """
    fig = plt.figure(figsize=(18, 12))
    
    # RGB composite
    ax1 = plt.subplot(2, 4, 1)
    rgb = image[:, :, :3]
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    ax1.imshow(rgb_norm)
    ax1.set_title('RGB Composite')
    ax1.axis('off')
    
    # NDWI
    ax2 = plt.subplot(2, 4, 2)
    im1 = ax2.imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    ax2.set_title('NDWI')
    ax2.axis('off')
    plt.colorbar(im1, ax=ax2, shrink=0.8)
    
    # NDWI with threshold line
    ax3 = plt.subplot(2, 4, 3)
    ndwi_thresh = ndwi.copy()
    ndwi_thresh[ndwi <= threshold_val] = np.nan  # Hide below threshold
    im2 = ax3.imshow(ndwi_thresh, cmap='Blues', vmin=threshold_val, vmax=1)
    ax3.set_title(f'NDWI > {threshold_val}')
    ax3.axis('off')
    plt.colorbar(im2, ax=ax3, shrink=0.8)
    
    # Water mask
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(water_mask, cmap='Blues')
    ax4.set_title(f'Water Mask\n({cleaning_stats["final_pixels"]:,} pixels)')
    ax4.axis('off')
    
    # RGB with water overlay
    ax5 = plt.subplot(2, 4, 5)
    rgb_overlay = rgb_norm.copy()
    rgb_overlay[water_mask] = [0, 0.8, 1]  # Cyan for water
    ax5.imshow(rgb_overlay)
    ax5.set_title('RGB + Water Detection')
    ax5.axis('off')
    
    # NDWI histogram
    ax6 = plt.subplot(2, 4, 6)
    if glacier_mask is not None:
        valid_ndwi = ndwi[glacier_mask & np.isfinite(ndwi)]
        ax6.hist(valid_ndwi, bins=100, alpha=0.7, color='skyblue', label='Glacier pixels')
        title_suffix = ' (Glacier areas)'
    else:
        valid_ndwi = ndwi[np.isfinite(ndwi)]
        ax6.hist(valid_ndwi, bins=100, alpha=0.7, color='skyblue', label='All pixels')
        title_suffix = ' (All areas)'
    
    ax6.axvline(threshold_val, color='red', linestyle='--', linewidth=2,
                label=f'Fixed threshold: {threshold_val}')
    ax6.set_xlabel('NDWI')
    ax6.set_ylabel('Frequency')
    ax6.set_title(f'NDWI Histogram{title_suffix}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Glacier mask
    ax7 = plt.subplot(2, 4, 7)
    if glacier_mask is not None:
        ax7.imshow(glacier_mask, cmap='Greys')
        ax7.set_title('Glacier Mask (RGI)')
    else:
        ax7.axis('off')
        ax7.text(0.5, 0.5, 'No glacier mask\nprovided', ha='center', va='center', 
                transform=ax7.transAxes, fontsize=12)
    ax7.axis('off')
    
    # Statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    water_pixels = cleaning_stats['final_pixels']
    if glacier_mask is not None:
        glacier_pixels = np.sum(glacier_mask)
        total_pixels = ndwi.size
        stats_text = f"""FIXED THRESHOLD RESULTS

THRESHOLD: {threshold_val}

DETECTION SUMMARY:
Total pixels: {total_pixels:,}
Glacier pixels: {glacier_pixels:,}
Water pixels: {water_pixels:,}

Water % of glaciers: {100*water_pixels/glacier_pixels:.2f}%
Water % of image: {100*water_pixels/total_pixels:.2f}%

CLEANING APPLIED:
Initial: {cleaning_stats['initial_pixels']:,}
Final: {cleaning_stats['final_pixels']:,}
Removed: {cleaning_stats['total_removed']:,}
Efficiency: {cleaning_stats['cleaning_efficiency']:.1f}%"""
    else:
        total_pixels = ndwi.size
        stats_text = f"""FIXED THRESHOLD RESULTS

THRESHOLD: {threshold_val}

DETECTION SUMMARY:
Total pixels: {total_pixels:,}
Water pixels: {water_pixels:,}
Water %: {100*water_pixels/total_pixels:.2f}%

CLEANING APPLIED:
Initial: {cleaning_stats['initial_pixels']:,}
Final: {cleaning_stats['final_pixels']:,}
Removed: {cleaning_stats['total_removed']:,}
Efficiency: {cleaning_stats['cleaning_efficiency']:.1f}%"""
    
    ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('water_detection_fixed_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'water_detection_fixed_results.png'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Water detection with fixed NDWI threshold')
    parser.add_argument('input_tif', help='Path to input Planet TIFF file')
    parser.add_argument('--threshold', type=float, default=0.2, help='NDWI threshold (default: 0.2)')
    parser.add_argument('--glacier_shp', help='Path to RGI glacier shapefile')
    parser.add_argument('--output', help='Path to save water mask as TIFF')
    parser.add_argument('--no_cleaning', action='store_true', help='Skip morphological cleaning')
    
    args = parser.parse_args()
    
    # Read image
    print("Reading Planet image...")
    image, profile = read_planet_image(args.input_tif)
    
    # Create glacier mask if provided
    glacier_mask = None
    if args.glacier_shp:
        print("Creating glacier mask...")
        glacier_mask = create_glacier_mask(args.glacier_shp, profile, image.shape[:2])
    
    # Compute NDWI
    print("Computing NDWI...")
    ndwi = compute_ndwi(image)
    
    # Apply fixed threshold
    print(f"Applying fixed threshold water detection (threshold = {args.threshold})...")
    water_mask, cleaning_stats = fixed_threshold_water(
        ndwi, 
        threshold=args.threshold, 
        glacier_mask=glacier_mask,
        apply_cleaning=not args.no_cleaning
    )
    
    water_pixels = np.sum(water_mask)
    if glacier_mask is not None:
        glacier_pixels = np.sum(glacier_mask)
        print(f"\nFinal Results:")
        print(f"Water pixels detected: {water_pixels:,}")
        print(f"Water percentage of glacier area: {100 * water_pixels / glacier_pixels:.2f}%")
        print(f"Water percentage of total image: {100 * water_pixels / water_mask.size:.2f}%")
    else:
        print(f"\nFinal Results:")
        print(f"Water pixels detected: {water_pixels:,}")
        print(f"Water percentage: {100 * water_pixels / water_mask.size:.2f}%")
    
    # Save outputs
    if args.output:
        # Save water mask
        profile_out = profile.copy()
        profile_out.update(dtype='uint8', count=1)
        
        with rasterio.open(args.output, 'w', **profile_out) as dst:
            dst.write(water_mask.astype(np.uint8), 1)
        print(f"Water mask saved to: {args.output}")
        
        # Save NDWI
        ndwi_output = args.output.replace('.tif', '_ndwi.tif')
        profile_ndwi = profile.copy()
        profile_ndwi.update(dtype='float32', count=1, nodata=np.nan)
        
        with rasterio.open(ndwi_output, 'w', **profile_ndwi) as dst:
            dst.write(ndwi.astype(np.float32), 1)
        print(f"NDWI saved to: {ndwi_output}")
    
    # Visualize results
    visualize_fixed_threshold_results(image, ndwi, water_mask, args.threshold, 
                                    cleaning_stats, glacier_mask)

if __name__ == "__main__":
    main()