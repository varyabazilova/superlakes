import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk
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
        # Read all bands
        image = src.read()  # Shape: (bands, height, width)
        profile = src.profile
        
        # Transpose to (height, width, bands)
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
    # Extract bands (assuming R, G, B, NIR order)
    green = image[:, :, 1].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    
    # Avoid division by zero
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
    # Read glacier polygons
    glaciers = gpd.read_file(glacier_shapefile)
    print(f"Read {len(glaciers)} glacier polygons")
    
    # Reproject to image CRS if needed
    if glaciers.crs != image_profile['crs']:
        print(f"Reprojecting glaciers from {glaciers.crs} to {image_profile['crs']}")
        glaciers = glaciers.to_crs(image_profile['crs'])
    
    # Get image bounds
    transform = image_profile['transform']
    
    # Create glacier mask by rasterizing polygons
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

def calculate_otsu_detailed(values):
    """
    Calculate Otsu threshold with detailed analysis
    
    Parameters:
    values (numpy.ndarray): 1D array of values to threshold
    
    Returns:
    tuple: (threshold, thresholds_array, between_class_variances)
    """
    # Subsample for efficiency if too many values
    if len(values) > 100000:
        idx = np.random.choice(len(values), 100000, replace=False)
        values = values[idx]
        print(f"Subsampled to {len(values)} pixels for Otsu analysis")
    
    # Create histogram
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram (probabilities)
    hist = hist.astype(float) / hist.sum()
    
    # Calculate cumulative sums
    cumsum_prob = np.cumsum(hist)
    cumsum_mean = np.cumsum(hist * bin_centers)
    
    # Calculate global mean
    global_mean = cumsum_mean[-1]
    
    # Calculate between-class variance for each possible threshold
    between_class_variances = []
    thresholds = []
    
    for i in range(1, len(hist)):
        # Weight of background class
        w1 = cumsum_prob[i-1]
        # Weight of foreground class  
        w2 = 1 - w1
        
        if w1 == 0 or w2 == 0:
            between_class_variances.append(0)
            thresholds.append(bin_centers[i])
            continue
            
        # Mean of background class
        mu1 = cumsum_mean[i-1] / w1 if w1 > 0 else 0
        # Mean of foreground class
        mu2 = (global_mean - cumsum_mean[i-1]) / w2 if w2 > 0 else 0
        
        # Between-class variance
        between_var = w1 * w2 * (mu1 - mu2) ** 2
        between_class_variances.append(between_var)
        thresholds.append(bin_centers[i])
    
    # Find optimal threshold
    max_idx = np.argmax(between_class_variances)
    optimal_threshold = thresholds[max_idx]
    
    return optimal_threshold, np.array(thresholds), np.array(between_class_variances)

def otsu_threshold_water(ndwi, glacier_mask=None):
    """
    Apply Otsu thresholding to NDWI for water detection
    
    Parameters:
    ndwi (numpy.ndarray): NDWI values
    glacier_mask (numpy.ndarray, optional): Boolean mask to restrict analysis to glacier areas
    
    Returns:
    tuple: (water_mask, threshold_value, otsu_analysis)
    """
    # Apply glacier mask if provided
    if glacier_mask is not None:
        valid_ndwi = ndwi[glacier_mask & np.isfinite(ndwi)]
        print(f"Using {len(valid_ndwi)} pixels within glacier areas for thresholding")
    else:
        valid_ndwi = ndwi[np.isfinite(ndwi)]
    
    if len(valid_ndwi) == 0:
        print("Warning: No valid NDWI values found!")
        return np.zeros_like(ndwi, dtype=bool), 0, None
    
    # Apply Otsu thresholding with detailed analysis
    threshold = threshold_otsu(valid_ndwi)
    optimal_threshold, thresholds, between_variances = calculate_otsu_detailed(valid_ndwi)
    
    print(f"Otsu threshold: {threshold:.3f}")
    print(f"Detailed Otsu threshold: {optimal_threshold:.3f}")
    
    # Create water mask
    water_mask = ndwi > threshold
    
    # Restrict to glacier areas if mask provided
    if glacier_mask is not None:
        water_mask = water_mask & glacier_mask
    
    otsu_analysis = {
        'valid_ndwi': valid_ndwi,
        'thresholds': thresholds,
        'between_variances': between_variances,
        'optimal_threshold': optimal_threshold
    }
    
    return water_mask, threshold, otsu_analysis

def refine_water_mask(water_mask, min_size=10):
    """
    Refine water mask using morphological operations
    
    Parameters:
    water_mask (numpy.ndarray): Binary water mask
    min_size (int): Minimum size of water bodies to keep
    
    Returns:
    numpy.ndarray: Refined water mask
    """
    # Close small gaps
    refined_mask = binary_closing(water_mask, disk(2))
    
    # Remove small objects
    refined_mask = remove_small_objects(refined_mask, min_size=min_size)
    
    return refined_mask

def visualize_results(image, ndwi, water_mask, threshold_val, glacier_mask=None, otsu_analysis=None):
    """
    Visualize the results including Otsu analysis
    """
    # Create figure with more subplots for Otsu analysis
    fig = plt.figure(figsize=(20, 16))
    
    # RGB composite
    ax1 = plt.subplot(3, 3, 1)
    rgb = image[:, :, :3]
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    ax1.imshow(rgb_norm)
    ax1.set_title('RGB Composite')
    ax1.axis('off')
    
    # NDWI
    ax2 = plt.subplot(3, 3, 2)
    im1 = ax2.imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    ax2.set_title('NDWI')
    ax2.axis('off')
    plt.colorbar(im1, ax=ax2)
    
    # Glacier mask
    ax3 = plt.subplot(3, 3, 3)
    if glacier_mask is not None:
        ax3.imshow(glacier_mask, cmap='Greys')
        ax3.set_title('Glacier Mask (RGI)')
        ax3.axis('off')
    else:
        ax3.axis('off')
    
    # NDWI histogram
    ax4 = plt.subplot(3, 3, 4)
    valid_ndwi = ndwi[np.isfinite(ndwi)]
    ax4.hist(valid_ndwi, bins=100, alpha=0.7, color='blue', label='All pixels')
    if glacier_mask is not None:
        glacier_ndwi = ndwi[glacier_mask & np.isfinite(ndwi)]
        ax4.hist(glacier_ndwi, bins=100, alpha=0.7, color='orange', label='Glacier pixels')
    ax4.axvline(threshold_val, color='red', linestyle='--', 
                label=f'Otsu threshold: {threshold_val:.3f}')
    ax4.set_xlabel('NDWI')
    ax4.set_ylabel('Frequency')
    ax4.set_title('NDWI Histogram')
    ax4.legend()
    
    # Otsu optimization curve
    ax5 = plt.subplot(3, 3, 5)
    if otsu_analysis is not None:
        ax5.plot(otsu_analysis['thresholds'], otsu_analysis['between_variances'], 'b-', linewidth=2)
        max_idx = np.argmax(otsu_analysis['between_variances'])
        ax5.plot(otsu_analysis['thresholds'][max_idx], otsu_analysis['between_variances'][max_idx], 
                'ro', markersize=8, label=f'Optimal: {otsu_analysis["optimal_threshold"]:.3f}')
        ax5.axvline(threshold_val, color='red', linestyle='--', alpha=0.7)
        ax5.set_xlabel('Threshold Value')
        ax5.set_ylabel('Between-Class Variance')
        ax5.set_title('Otsu Optimization Curve')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Water mask
    ax6 = plt.subplot(3, 3, 6)
    ax6.imshow(water_mask, cmap='Blues')
    ax6.set_title('Water Mask (Glacier Areas Only)')
    ax6.axis('off')
    
    # RGB with water overlay
    ax7 = plt.subplot(3, 3, 7)
    rgb_overlay = rgb_norm.copy()
    rgb_overlay[water_mask] = [0, 0, 1]  # Blue for water
    ax7.imshow(rgb_overlay)
    ax7.set_title('RGB + Water Detection')
    ax7.axis('off')
    
    # Class separation visualization
    ax8 = plt.subplot(3, 3, 8)
    if otsu_analysis is not None:
        glacier_ndwi = otsu_analysis['valid_ndwi']
        class1 = glacier_ndwi[glacier_ndwi <= threshold_val]  # Non-water
        class2 = glacier_ndwi[glacier_ndwi > threshold_val]   # Water
        
        ax8.hist(class1, bins=50, alpha=0.7, color='brown', label=f'Non-water ({len(class1)} pixels)')
        ax8.hist(class2, bins=50, alpha=0.7, color='blue', label=f'Water ({len(class2)} pixels)')
        ax8.axvline(threshold_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold_val:.3f}')
        ax8.set_xlabel('NDWI')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Class Separation')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    stats_text = f"""
    OTSU ANALYSIS STATISTICS
    
    Total pixels: {ndwi.size:,}
    """
    if glacier_mask is not None:
        glacier_pixels = np.sum(glacier_mask)
        water_pixels = np.sum(water_mask)
        stats_text += f"""Glacier pixels: {glacier_pixels:,} ({100*glacier_pixels/ndwi.size:.1f}%)
    Water pixels: {water_pixels:,}
    Water % of glaciers: {100*water_pixels/glacier_pixels:.1f}%
    Water % of image: {100*water_pixels/ndwi.size:.1f}%"""
    
    if otsu_analysis is not None:
        stats_text += f"""
    
    NDWI range: {otsu_analysis['valid_ndwi'].min():.3f} to {otsu_analysis['valid_ndwi'].max():.3f}
    Optimal threshold: {otsu_analysis['optimal_threshold']:.3f}
    Max between-class variance: {otsu_analysis['between_variances'].max():.6f}"""
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('water_detection_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'water_detection_results.png'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Water detection on debris-covered ice')
    parser.add_argument('input_tif', help='Path to input Planet TIFF file')
    parser.add_argument('--glacier_shp', help='Path to RGI glacier shapefile to mask analysis')
    parser.add_argument('--min_size', type=int, default=10, 
                       help='Minimum size of water bodies to keep (default: 10)')
    parser.add_argument('--output', help='Path to save water mask as TIFF')
    
    args = parser.parse_args()
    
    # Read image
    print("Reading Planet image...")
    image, profile = read_planet_image(args.input_tif)
    
    # Create glacier mask if shapefile provided
    glacier_mask = None
    if args.glacier_shp:
        print("Creating glacier mask...")
        glacier_mask = create_glacier_mask(args.glacier_shp, profile, image.shape[:2])
    
    # Compute NDWI
    print("Computing NDWI...")
    ndwi = compute_ndwi(image)
    
    # Apply Otsu thresholding
    print("Applying Otsu thresholding...")
    water_mask, threshold_val, otsu_analysis = otsu_threshold_water(ndwi, glacier_mask)
    
    # Refine mask
    print("Refining water mask...")
    refined_mask = refine_water_mask(water_mask, min_size=args.min_size)
    
    water_pixels = np.sum(refined_mask)
    if glacier_mask is not None:
        glacier_pixels = np.sum(glacier_mask)
        print(f"Water pixels detected: {water_pixels}")
        print(f"Water percentage of glacier area: {100 * water_pixels / glacier_pixels:.2f}%")
        print(f"Water percentage of total image: {100 * water_pixels / refined_mask.size:.2f}%")
    else:
        print(f"Water pixels detected: {water_pixels}")
        print(f"Water percentage: {100 * water_pixels / refined_mask.size:.2f}%")
    
    # Save outputs
    if args.output:
        # Save water mask
        profile_out = profile.copy()
        profile_out.update(dtype='uint8', count=1)
        
        with rasterio.open(args.output, 'w', **profile_out) as dst:
            dst.write(refined_mask.astype(np.uint8), 1)
        print(f"Water mask saved to: {args.output}")
        
        # Save NDWI as well
        ndwi_output = args.output.replace('.tif', '_ndwi.tif')
        profile_ndwi = profile.copy()
        profile_ndwi.update(dtype='float32', count=1, nodata=np.nan)
        
        with rasterio.open(ndwi_output, 'w', **profile_ndwi) as dst:
            dst.write(ndwi.astype(np.float32), 1)
        print(f"NDWI saved to: {ndwi_output}")
    
    # Visualize results
    visualize_results(image, ndwi, refined_mask, threshold_val, glacier_mask, otsu_analysis)

if __name__ == "__main__":
    main()