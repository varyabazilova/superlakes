import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import argparse

def read_planet_image(tif_path):
    """
    Read Planet imagery with 4 bands (RGB + NIR)
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

def kmeans_water_detection(image, ndwi, glacier_mask=None, n_clusters=3):
    """
    Use K-means clustering to detect multiple types of water
    
    Parameters:
    image (numpy.ndarray): RGB+NIR image
    ndwi (numpy.ndarray): NDWI values
    glacier_mask (numpy.ndarray): Glacier mask
    n_clusters (int): Number of clusters (default: 3 for ice/debris, dark water, bright water)
    
    Returns:
    tuple: (water_mask, cluster_labels, cluster_info)
    """
    print(f"Running K-means clustering with {n_clusters} clusters...")
    
    # Apply glacier mask if provided
    if glacier_mask is not None:
        valid_mask = glacier_mask & np.isfinite(ndwi)
        print(f"Using {np.sum(valid_mask)} pixels within glacier areas")
    else:
        valid_mask = np.isfinite(ndwi)
    
    # Extract features for clustering
    valid_pixels = np.where(valid_mask)
    
    # Features: R, G, B, NIR, NDWI, Blue/NIR ratio
    red = image[:, :, 0][valid_pixels].astype(np.float32)
    green = image[:, :, 1][valid_pixels].astype(np.float32)
    blue = image[:, :, 2][valid_pixels].astype(np.float32)
    nir = image[:, :, 3][valid_pixels].astype(np.float32)
    ndwi_vals = ndwi[valid_pixels]
    
    # Additional water-sensitive features
    blue_nir_ratio = blue / (nir + 1e-6)  # Bright water often has high blue/NIR
    brightness = (red + green + blue) / 3  # Overall brightness
    
    # Stack features
    features = np.column_stack([
        red, green, blue, nir, 
        ndwi_vals, blue_nir_ratio, brightness
    ])
    
    # Subsample for efficiency if too many pixels
    if len(features) > 500000:
        idx = np.random.choice(len(features), 500000, replace=False)
        features_sample = features[idx]
        print(f"Subsampled to {len(features_sample)} pixels for clustering")
    else:
        features_sample = features
        idx = np.arange(len(features))
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_sample)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels_sample = kmeans.fit_predict(features_scaled)
    
    # Map back to full dataset
    cluster_labels_full = np.zeros(len(features), dtype=int)
    cluster_labels_full[idx] = cluster_labels_sample
    
    # Analyze clusters to identify water
    cluster_info = {}
    water_clusters = []
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels_full == i
        if np.sum(cluster_mask) == 0:
            continue
            
        cluster_ndwi = ndwi_vals[cluster_mask]
        cluster_blue = blue[cluster_mask]
        cluster_brightness = brightness[cluster_mask]
        cluster_blue_nir = blue_nir_ratio[cluster_mask]
        
        cluster_info[i] = {
            'n_pixels': np.sum(cluster_mask),
            'ndwi_mean': np.mean(cluster_ndwi),
            'ndwi_std': np.std(cluster_ndwi),
            'blue_mean': np.mean(cluster_blue),
            'brightness_mean': np.mean(cluster_brightness),
            'blue_nir_mean': np.mean(cluster_blue_nir)
        }
        
        # Water detection criteria - balanced approach:
        # 1. Dark water: High NDWI 
        # 2. Bright water: High blue/NIR ratio with reasonable NDWI
        
        min_pixels = len(features) * 0.005  # At least 0.5% of pixels
        has_enough_pixels = cluster_info[i]['n_pixels'] > min_pixels
        
        is_dark_water = (cluster_info[i]['ndwi_mean'] > 0.05 and 
                        has_enough_pixels)
        
        # Alternative bright water: moderate NDWI + high blue/NIR ratio
        is_bright_water = (cluster_info[i]['blue_nir_mean'] > 1.08 and 
                          cluster_info[i]['ndwi_mean'] > 0.03 and
                          cluster_info[i]['ndwi_mean'] < 0.065 and  # Different range than dark water
                          has_enough_pixels)
        
        # Debug output
        print(f"  Cluster {i}: {cluster_info[i]['n_pixels']:,} pixels")
        print(f"    NDWI: {cluster_info[i]['ndwi_mean']:.3f}, Blue/NIR: {cluster_info[i]['blue_nir_mean']:.3f}")
        print(f"    Brightness: {cluster_info[i]['brightness_mean']:.1f}, Min pixels: {min_pixels:.0f}")
        print(f"    Dark water: {is_dark_water}, Bright water: {is_bright_water}")
        
        if is_dark_water or is_bright_water:
            water_clusters.append(i)
            water_type = "dark" if is_dark_water else "bright"
            print(f"  Cluster {i}: {water_type} water ({cluster_info[i]['n_pixels']:,} pixels)")
            print(f"    NDWI: {cluster_info[i]['ndwi_mean']:.3f} ± {cluster_info[i]['ndwi_std']:.3f}")
            print(f"    Blue/NIR: {cluster_info[i]['blue_nir_mean']:.3f}")
    
    # Create water mask
    water_mask_flat = np.zeros(len(features), dtype=bool)
    for cluster_id in water_clusters:
        water_mask_flat |= (cluster_labels_full == cluster_id)
    
    # Map back to 2D
    water_mask = np.zeros_like(ndwi, dtype=bool)
    water_mask[valid_pixels] = water_mask_flat
    
    # Restrict to glacier areas if mask provided
    if glacier_mask is not None:
        water_mask = water_mask & glacier_mask
    
    # Also create cluster map for visualization
    cluster_map = np.full_like(ndwi, -1, dtype=int)
    cluster_map[valid_pixels] = cluster_labels_full
    
    water_pixels = np.sum(water_mask)
    total_valid = np.sum(valid_mask)
    print(f"Water clusters identified: {water_clusters}")
    print(f"Total water pixels: {water_pixels:,} ({100*water_pixels/total_valid:.2f}% of valid area)")
    
    return water_mask, cluster_map, cluster_info, water_clusters

def refine_water_mask(water_mask, min_size=10):
    """
    Refine water mask using morphological operations
    """
    refined_mask = binary_closing(water_mask, disk(2))
    refined_mask = remove_small_objects(refined_mask, min_size=min_size)
    return refined_mask

def visualize_kmeans_results(image, ndwi, water_mask, cluster_map, cluster_info, water_clusters, glacier_mask=None):
    """
    Visualize K-means clustering results
    """
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
    
    # Cluster map
    ax3 = plt.subplot(3, 3, 3)
    cluster_display = np.ma.masked_where(cluster_map == -1, cluster_map)
    im2 = ax3.imshow(cluster_display, cmap='tab10')
    ax3.set_title('K-means Clusters')
    ax3.axis('off')
    plt.colorbar(im2, ax=ax3)
    
    # Water mask
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(water_mask, cmap='Blues')
    ax4.set_title('Water Mask (K-means)')
    ax4.axis('off')
    
    # RGB with water overlay
    ax5 = plt.subplot(3, 3, 5)
    rgb_overlay = rgb_norm.copy()
    rgb_overlay[water_mask] = [0, 0, 1]  # Blue for water
    ax5.imshow(rgb_overlay)
    ax5.set_title('RGB + Water Detection')
    ax5.axis('off')
    
    # Cluster statistics
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('off')
    
    stats_text = "CLUSTER ANALYSIS\n\n"
    for i, info in cluster_info.items():
        cluster_type = "WATER" if i in water_clusters else "NON-WATER"
        stats_text += f"Cluster {i} ({cluster_type}):\n"
        stats_text += f"  Pixels: {info['n_pixels']:,}\n"
        stats_text += f"  NDWI: {info['ndwi_mean']:.3f}±{info['ndwi_std']:.3f}\n"
        stats_text += f"  Blue/NIR: {info['blue_nir_mean']:.3f}\n"
        stats_text += f"  Brightness: {info['brightness_mean']:.1f}\n\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Glacier mask
    if glacier_mask is not None:
        ax7 = plt.subplot(3, 3, 7)
        ax7.imshow(glacier_mask, cmap='Greys')
        ax7.set_title('Glacier Mask (RGI)')
        ax7.axis('off')
    
    # NDWI histogram by cluster
    ax8 = plt.subplot(3, 3, 8)
    colors = plt.cm.tab10(np.arange(len(cluster_info)))
    
    for i, info in cluster_info.items():
        if glacier_mask is not None:
            cluster_pixels = (cluster_map == i) & glacier_mask & np.isfinite(ndwi)
        else:
            cluster_pixels = (cluster_map == i) & np.isfinite(ndwi)
            
        if np.sum(cluster_pixels) > 0:
            cluster_ndwi = ndwi[cluster_pixels]
            label = f"Cluster {i}" + (" (WATER)" if i in water_clusters else "")
            ax8.hist(cluster_ndwi, bins=50, alpha=0.7, color=colors[i], 
                    label=label, density=True)
    
    ax8.set_xlabel('NDWI')
    ax8.set_ylabel('Density')
    ax8.set_title('NDWI Distribution by Cluster')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Overall statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    water_pixels = np.sum(water_mask)
    if glacier_mask is not None:
        glacier_pixels = np.sum(glacier_mask)
        total_pixels = ndwi.size
        overall_stats = f"""DETECTION SUMMARY

Total pixels: {total_pixels:,}
Glacier pixels: {glacier_pixels:,}
Water pixels: {water_pixels:,}

Water % of glaciers: {100*water_pixels/glacier_pixels:.1f}%
Water % of image: {100*water_pixels/total_pixels:.1f}%

Water clusters: {water_clusters}
Method: K-means clustering"""
    else:
        total_pixels = ndwi.size
        overall_stats = f"""DETECTION SUMMARY

Total pixels: {total_pixels:,}
Water pixels: {water_pixels:,}

Water %: {100*water_pixels/total_pixels:.1f}%

Water clusters: {water_clusters}
Method: K-means clustering"""
    
    ax9.text(0.1, 0.9, overall_stats, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('water_detection_kmeans_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'water_detection_kmeans_results.png'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Water detection using K-means clustering')
    parser.add_argument('input_tif', help='Path to input Planet TIFF file')
    parser.add_argument('--glacier_shp', help='Path to RGI glacier shapefile')
    parser.add_argument('--n_clusters', type=int, default=3, 
                       help='Number of clusters for K-means (default: 3)')
    parser.add_argument('--min_size', type=int, default=10, 
                       help='Minimum size of water bodies to keep (default: 10)')
    parser.add_argument('--output', help='Path to save water mask as TIFF')
    
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
    
    # Apply K-means clustering
    print("Applying K-means clustering...")
    water_mask, cluster_map, cluster_info, water_clusters = kmeans_water_detection(
        image, ndwi, glacier_mask, n_clusters=args.n_clusters
    )
    
    # Refine mask
    print("Refining water mask...")
    refined_mask = refine_water_mask(water_mask, min_size=args.min_size)
    
    water_pixels = np.sum(refined_mask)
    if glacier_mask is not None:
        glacier_pixels = np.sum(glacier_mask)
        print(f"Final water pixels: {water_pixels:,}")
        print(f"Water percentage of glacier area: {100 * water_pixels / glacier_pixels:.2f}%")
        print(f"Water percentage of total image: {100 * water_pixels / refined_mask.size:.2f}%")
    else:
        print(f"Final water pixels: {water_pixels:,}")
        print(f"Water percentage: {100 * water_pixels / refined_mask.size:.2f}%")
    
    # Save output
    if args.output:
        profile_out = profile.copy()
        profile_out.update(dtype='uint8', count=1)
        
        with rasterio.open(args.output, 'w', **profile_out) as dst:
            dst.write(refined_mask.astype(np.uint8), 1)
        print(f"Water mask saved to: {args.output}")
    
    # Visualize results
    visualize_kmeans_results(image, ndwi, refined_mask, cluster_map, 
                           cluster_info, water_clusters, glacier_mask)

if __name__ == "__main__":
    main()