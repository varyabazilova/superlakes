# =============================================================================
# IMAGE MASKING WITH SHAPEFILE - ADD TO YOUR SAM 2 NOTEBOOK
# Insert this RIGHT AFTER loading your image, BEFORE running SAM 2
# =============================================================================

import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask

# SHAPEFILE MASKING CONFIGURATION
SHAPEFILE_PATH = "/content/drive/MyDrive/superlakes/your_boundary.shp"  # UPDATE THIS PATH
MASK_NODATA_VALUE = 0  # Value to use for areas outside shapefile (black pixels)

def mask_image_with_shapefile(image_array, image_path, shapefile_path):
    """
    Mask satellite image to only include areas within shapefile boundary
    """
    print(f"🗺️ Applying shapefile mask from: {shapefile_path}")
    
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    print(f"   Loaded {len(gdf)} polygons from shapefile")
    
    # Get image metadata for proper transformation
    with rasterio.open(image_path) as src:
        # Reproject shapefile to match image CRS if needed
        if gdf.crs != src.crs:
            print(f"   Reprojecting from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)
        
        # Create mask from geometries
        geometries = gdf.geometry.values
        mask = geometry_mask(
            geometries,
            transform=src.transform,
            invert=True,  # True means inside polygon = True
            out_shape=(src.height, src.width)
        )
    
    print(f"   Analysis area: {mask.sum():,} pixels ({mask.mean()*100:.1f}% of image)")
    
    # Apply mask to image
    if len(image_array.shape) == 3:  # RGB image
        masked_image = image_array.copy()
        # Set areas outside shapefile to black/nodata
        masked_image[~mask] = [MASK_NODATA_VALUE, MASK_NODATA_VALUE, MASK_NODATA_VALUE]
    else:  # Single band image
        masked_image = image_array.copy()
        masked_image[~mask] = MASK_NODATA_VALUE
    
    print(f"   ✅ Image masked successfully")
    
    return masked_image, mask

# =============================================================================
# REPLACE THE IMAGE LOADING SECTION WITH THIS:
# =============================================================================

# Load satellite image (UNCHANGED)
try:
    image = load_image_data(IMAGE_PATH, IMAGE_TYPE)
    print(f"✅ Image loaded successfully!")
    print(f"   Shape: {image.shape}")
    
    # NEW: Apply shapefile masking
    if os.path.exists(SHAPEFILE_PATH):
        masked_image, analysis_mask = mask_image_with_shapefile(image, IMAGE_PATH, SHAPEFILE_PATH)
        
        # Update image to use masked version
        image = masked_image
        
        # Visualize the masking effect
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Original image
        axes[0].imshow(load_image_data(IMAGE_PATH, IMAGE_TYPE))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Analysis mask
        axes[1].imshow(analysis_mask, cmap='Blues')
        axes[1].set_title(f'Analysis Area\n{analysis_mask.sum():,} pixels')
        axes[1].axis('off')
        
        # Masked image
        axes[2].imshow(image)
        axes[2].set_title('Masked Image for SAM 2')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Shapefile masking applied!")
        print(f"   Analysis area: {analysis_mask.sum():,} pixels")
        print(f"   Masked out: {(~analysis_mask).sum():,} pixels")
        
    else:
        print(f"⚠️ Shapefile not found: {SHAPEFILE_PATH}")
        print("   Proceeding without spatial masking")
        analysis_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
    
    IMAGE_LOADED = True
    
except Exception as e:
    print(f"❌ Error loading/masking image: {e}")
    IMAGE_LOADED = False
    image = None

# =============================================================================
# THEN PROCEED WITH SAM 2 AS NORMAL - IT WILL ONLY PROCESS THE MASKED AREAS
# =============================================================================