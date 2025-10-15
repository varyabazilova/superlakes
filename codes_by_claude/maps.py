import numpy as np
import rasterio
import matplotlib.pyplot as plt

def load_band(image_path, band=0):
    """
    Load a single band from an image
    
    Parameters:
    image_path (str): Path to image
    band (int): Band number (0-indexed, default=0)
    
    Returns:
    numpy.ndarray: Single band array
    """
    with rasterio.open(image_path) as src:
        return src.read(band + 1)  # rasterio is 1-indexed

def normalize_band(band, method='percentile'):
    """
    Normalize band for display
    
    Parameters:
    band (numpy.ndarray): Input band
    method (str): 'percentile', 'minmax', or 'none'
    
    Returns:
    numpy.ndarray: Normalized band (0-1)
    """
    band = band.astype(np.float32)
    
    if method == 'percentile':
        p2, p98 = np.percentile(band, [2, 98])
        return np.clip((band - p2) / (p98 - p2), 0, 1)
    elif method == 'minmax':
        return (band - band.min()) / (band.max() - band.min())
    else:
        return band

def save_rgb_composite(red_img, green_img, blue_img, 
                      red_band=0, green_band=0, blue_band=0,
                      normalization='percentile', save_path=None):
    """
    Save RGB composite as GeoTIFF for QGIS
    
    Parameters:
    red_img, green_img, blue_img (str): Paths to images
    red_band, green_band, blue_band (int): Band numbers to use
    normalization (str): Normalization method
    save_path (str): Path to save composite GeoTIFF
    """
    
    # Load bands
    red = load_band(red_img, red_band)
    green = load_band(green_img, green_band) 
    blue = load_band(blue_img, blue_band)
    
    # Normalize and convert to uint8
    red_norm = (normalize_band(red, normalization) * 255).astype(np.uint8)
    green_norm = (normalize_band(green, normalization) * 255).astype(np.uint8)
    blue_norm = (normalize_band(blue, normalization) * 255).astype(np.uint8)
    
    # Get spatial reference from first image
    with rasterio.open(red_img) as src:
        transform = src.transform
        crs = src.crs
        height, width = red_norm.shape
    
    # Save RGB composite as GeoTIFF
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with rasterio.open(
            save_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(red_norm, 1)    # Red band
            dst.write(green_norm, 2)  # Green band  
            dst.write(blue_norm, 3)   # Blue band
        
        print(f"✅ Saved RGB composite GeoTIFF to: {save_path}")
        print(f"  Red: {red_img} (band {red_band})")
        print(f"  Green: {green_img} (band {green_band})")
        print(f"  Blue: {blue_img} (band {blue_band})")
        print(f"  Normalization: {normalization}")
        print(f"  Ready for QGIS!")
    
    return red_norm, green_norm, blue_norm

def plot_rgb(red_img, green_img, blue_img, 
             red_band=0, green_band=0, blue_band=0,
             normalization='percentile', figsize=(15, 5), save_path=None):
    """
    Simple RGB plotting function
    
    Parameters:
    red_img, green_img, blue_img (str): Paths to images
    red_band, green_band, blue_band (int): Band numbers to use
    normalization (str): Normalization method
    figsize (tuple): Figure size
    save_path (str): Path to save composite image (optional)
    """
    
    # Load bands
    red = load_band(red_img, red_band)
    green = load_band(green_img, green_band) 
    blue = load_band(blue_img, blue_band)
    
    # Normalize
    red_norm = normalize_band(red, normalization)
    green_norm = normalize_band(green, normalization)
    blue_norm = normalize_band(blue, normalization)
    
    # Stack RGB
    rgb = np.stack([red_norm, green_norm, blue_norm], axis=2)
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Individual bands
    axes[0].imshow(red_norm, cmap='Reds')
    axes[0].set_title(f'Red\n{red_img.split("/")[-1]}\nBand {red_band}')
    axes[0].axis('off')
    
    axes[1].imshow(green_norm, cmap='Greens')
    axes[1].set_title(f'Green\n{green_img.split("/")[-1]}\nBand {green_band}')
    axes[1].axis('off')
    
    axes[2].imshow(blue_norm, cmap='Blues')
    axes[2].set_title(f'Blue\n{blue_img.split("/")[-1]}\nBand {blue_band}')
    axes[2].axis('off')
    
    # RGB composite
    axes[3].imshow(rgb)
    axes[3].set_title('RGB Composite')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved composite to: {save_path}")
    
    plt.show()
    
    print(f"RGB Composite created:")
    print(f"  Red: {red_img} (band {red_band})")
    print(f"  Green: {green_img} (band {green_band})")
    print(f"  Blue: {blue_img} (band {blue_band})")
    print(f"  Normalization: {normalization}")

# Example usage:
if __name__ == "__main__":
    
    # Example 1: NDWI temporal composite for QGIS
    red_image = "outputs_cleaned/2025-01-08_composite_mosaic_ndwi.tif"
    green_image = "outputs_cleaned/2025-05-25_composite_mosaic_ndwi.tif"
    blue_image = "outputs_cleaned/2025-08-21_composite_mosaic_ndwi.tif"
    
    save_rgb_composite(red_image, green_image, blue_image, 
                      red_band=0, green_band=0, blue_band=0,
                      save_path="analysis/testoutput/ndwi_temporal_composite.tif")
    
    # Example 2: Planet true color for QGIS (R-G-B)
    save_rgb_composite("testimages25/2025-08-21_composite_mosaic.tif",
                      "testimages25/2025-08-21_composite_mosaic.tif",
                      "testimages25/2025-08-21_composite_mosaic.tif",
                      red_band=0, green_band=1, blue_band=2,
                      save_path="analysis/testoutput/planet_true_color.tif")
    
    # Example 3: Planet false color for QGIS (NIR-R-G)
    save_rgb_composite("testimages25/2025-08-21_composite_mosaic.tif",
                      "testimages25/2025-08-21_composite_mosaic.tif",
                      "testimages25/2025-08-21_composite_mosaic.tif",
                      red_band=3, green_band=0, blue_band=1,
                      save_path="analysis/testoutput/planet_false_color.tif")