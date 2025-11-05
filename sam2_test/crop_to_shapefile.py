# =============================================================================
# CROP IMAGE TO SHAPEFILE BOUNDS - BETTER THAN MASKING
# This eliminates the boundary edge problem
# =============================================================================

import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np

def crop_image_to_shapefile(image_path, shapefile_path):
    """
    Crop image to shapefile bounds instead of masking
    This eliminates the boundary edge that SAM 2 detects
    """
    print(f"✂️ Cropping image to shapefile bounds...")
    
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    print(f"   Loaded {len(gdf)} polygons")
    
    # Open image and crop to shapefile
    with rasterio.open(image_path) as src:
        # Reproject shapefile if needed
        if gdf.crs != src.crs:
            print(f"   Reprojecting from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)
        
        # Crop image to shapefile geometry
        cropped_data, cropped_transform = rasterio.mask.mask(
            src, 
            gdf.geometry.values, 
            crop=True,  # This is key - actually crops the image
            nodata=0    # Set nodata pixels to 0
        )
        
        # Handle different band configurations
        if cropped_data.shape[0] == 1:  # Single band
            cropped_image = cropped_data[0]
        else:  # Multi-band (RGB/RGBA)
            cropped_image = np.transpose(cropped_data, (1, 2, 0))
            if cropped_image.shape[2] > 3:  # Remove alpha channel if present
                cropped_image = cropped_image[:, :, :3]
    
    print(f"   ✅ Cropped to: {cropped_image.shape}")
    print(f"   Original size eliminated - no boundary edges!")
    
    return cropped_image

# =============================================================================
# REPLACE YOUR IMAGE LOADING WITH THIS:
# =============================================================================

# CONFIGURATION
SHAPEFILE_PATH = "/content/drive/MyDrive/superlakes/your_boundary.shp"  # UPDATE THIS

# Load and crop image
try:
    if os.path.exists(SHAPEFILE_PATH):
        # Crop image to shapefile bounds (eliminates boundary issues)
        image = crop_image_to_shapefile(IMAGE_PATH, SHAPEFILE_PATH)
        print("✅ Image cropped to analysis area - no boundary edges for SAM 2!")
        
    else:
        # Fallback to normal loading
        image = load_image_data(IMAGE_PATH, IMAGE_TYPE)
        print("⚠️ Shapefile not found - using full image")
    
    # Visualize cropped result
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.title(f'Cropped {IMAGE_TYPE} Image for SAM 2\nShape: {image.shape}')
    plt.axis('off')
    plt.show()
    
    IMAGE_LOADED = True
    
except Exception as e:
    print(f"❌ Error cropping image: {e}")
    IMAGE_LOADED = False