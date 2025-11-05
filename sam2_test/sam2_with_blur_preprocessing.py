# =============================================================================
# SAM 2 WITH BLUR PREPROCESSING
# Blurs areas outside shapefile so SAM 2 focuses only on areas of interest
# =============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from PIL import Image

def load_image_data(image_path, image_type="RGB"):
    """Load and prepare image data based on type"""
    print(f"📂 Loading {image_type} image from: {image_path}")
    
    if image_type == "FCC" or image_type == "RGB":
        # False Color Composite or RGB
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        
    elif image_type == "NDWI":
        # NDWI converted to 3-channel
        with rasterio.open(image_path) as src:
            ndwi = src.read(1)
        
        # Normalize NDWI from [-1, 1] to [0, 255]
        ndwi_normalized = ((ndwi + 1) / 2 * 255).astype(np.uint8)
        
        # Create 3-channel array
        image = np.stack([ndwi_normalized, ndwi_normalized, ndwi_normalized], axis=-1)
    
    else:
        raise ValueError(f"Unknown image type: {image_type}")
    
    return image

def blur_outside_shapefile(image_array, image_path, shapefile_path, blur_strength=15):
    """
    Blur areas outside shapefile so SAM 2 ignores them
    
    Parameters:
    - image_array: Input image as numpy array
    - image_path: Path to original image (for geospatial info)
    - shapefile_path: Path to boundary shapefile
    - blur_strength: Higher values = more blur (5=light, 15=medium, 30=heavy)
    
    Returns:
    - result_image: Image with blurred outside areas
    - inside_mask: Boolean mask of inside areas
    """
    print(f"🌫️ Blurring areas outside shapefile...")
    print(f"   Blur strength: {blur_strength} (5=light, 15=medium, 30=heavy)")
    
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    print(f"   Loaded {len(gdf)} polygons from shapefile")
    
    # Get image geospatial info and create mask
    with rasterio.open(image_path) as src:
        # Reproject shapefile if needed
        if gdf.crs != src.crs:
            print(f"   Reprojecting from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)
        
        # Create mask (True = inside shapefile)
        inside_mask = geometry_mask(
            gdf.geometry.values,
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width)
        )
    
    print(f"   Analysis area: {inside_mask.sum():,} pixels ({inside_mask.mean()*100:.1f}%)")
    
    # Create blurred version of entire image
    blurred_image = image_array.copy()
    
    # Apply Gaussian blur to each channel
    kernel_size = blur_strength * 2 + 1  # Must be odd
    sigma = blur_strength / 3
    
    print(f"   Applying Gaussian blur (kernel={kernel_size}x{kernel_size}, sigma={sigma:.1f})...")
    
    for i in range(image_array.shape[2]):  # For each RGB channel
        blurred_image[:,:,i] = cv2.GaussianBlur(
            image_array[:,:,i], 
            (kernel_size, kernel_size),
            sigma
        )
    
    # Combine: original inside shapefile, blurred outside
    result_image = image_array.copy()
    result_image[~inside_mask] = blurred_image[~inside_mask]
    
    outside_pixels = (~inside_mask).sum()
    print(f"   ✅ Blurred {outside_pixels:,} pixels outside shapefile")
    print(f"   Sharp areas: {inside_mask.sum():,} pixels (for SAM 2 analysis)")
    
    return result_image, inside_mask

def preprocess_image_for_sam2(image_path, shapefile_path, image_type="RGB", 
                            blur_strength=15, show_visualization=True):
    """
    Load and preprocess image for SAM 2:
    1. Load image
    2. Blur areas outside shapefile
    3. Return SAM 2-ready image
    
    Parameters:
    - image_path: Path to satellite image
    - shapefile_path: Path to analysis boundary shapefile
    - image_type: "RGB", "FCC", or "NDWI"
    - blur_strength: Blur intensity (5-30)
    - show_visualization: Whether to show before/after comparison
    
    Returns:
    - processed_image: Image ready for SAM 2
    - inside_mask: Analysis area mask
    """
    print("🎯 SAM 2 Image Preprocessing Pipeline")
    print("=" * 50)
    
    # Load original image
    original_image = load_image_data(image_path, image_type)
    print(f"   Original image shape: {original_image.shape}")
    print(f"   Value range: {original_image.min()} - {original_image.max()}")
    
    # Apply blur preprocessing if shapefile exists
    if os.path.exists(shapefile_path):
        processed_image, inside_mask = blur_outside_shapefile(
            original_image, image_path, shapefile_path, blur_strength
        )
        
        if show_visualization:
            print("🖼️ Creating before/after visualization...")
            
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title(f'Original {image_type} Image')
            axes[0].axis('off')
            
            # Analysis area mask
            axes[1].imshow(inside_mask, cmap='Blues')
            axes[1].set_title(f'Analysis Area\n{inside_mask.sum():,} pixels')
            axes[1].axis('off')
            
            # Preprocessed image
            axes[2].imshow(processed_image)
            axes[2].set_title(f'Preprocessed for SAM 2\n(Blur strength: {blur_strength})')
            axes[2].axis('off')
            
            plt.suptitle('SAM 2 Preprocessing: Blur Outside Analysis Area', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        print("✅ Image preprocessing complete!")
        
    else:
        print(f"⚠️ Shapefile not found: {shapefile_path}")
        print("   Proceeding without blur preprocessing")
        processed_image = original_image
        inside_mask = np.ones((original_image.shape[0], original_image.shape[1]), dtype=bool)
    
    return processed_image, inside_mask

def save_preprocessed_image(processed_image, output_path, original_path):
    """Save preprocessed image as GeoTIFF with original geospatial info"""
    print(f"💾 Saving preprocessed image...")
    
    # Get original geospatial profile
    with rasterio.open(original_path) as src:
        profile = src.profile.copy()
        profile.update({
            'count': 3,  # RGB channels
            'dtype': 'uint8'
        })
    
    # Save preprocessed image
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i in range(3):
            dst.write(processed_image[:,:,i], i+1)
    
    print(f"✅ Preprocessed image saved: {output_path}")

# =============================================================================
# DIFFERENT BLUR STRENGTH PRESETS
# =============================================================================

def get_blur_preset(preset_name):
    """Get predefined blur strength settings"""
    presets = {
        "light": 5,      # Subtle blur - some features still visible outside
        "medium": 15,    # Moderate blur - most features suppressed
        "heavy": 30,     # Strong blur - almost completely smooth outside
        "extreme": 50    # Very strong blur - completely smooth outside
    }
    
    if preset_name in presets:
        return presets[preset_name]
    else:
        print(f"⚠️ Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
        return 15  # Default to medium

# =============================================================================
# BATCH PROCESSING FUNCTION
# =============================================================================

def batch_preprocess_images(image_dir, shapefile_path, output_dir, 
                           image_type="RGB", blur_strength=15):
    """
    Preprocess multiple images for SAM 2
    
    Parameters:
    - image_dir: Directory containing images
    - shapefile_path: Analysis boundary shapefile
    - output_dir: Where to save preprocessed images
    - image_type: Image type for loading
    - blur_strength: Blur intensity
    """
    print(f"🔄 Batch preprocessing images...")
    print(f"   Input directory: {image_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Blur strength: {blur_strength}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find image files
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    print(f"   Found {len(image_files)} images to process")
    
    # Process each image
    for i, filename in enumerate(image_files, 1):
        print(f"\n📷 Processing {i}/{len(image_files)}: {filename}")
        
        input_path = os.path.join(image_dir, filename)
        output_filename = filename.replace('.tif', '_preprocessed.tif').replace('.png', '_preprocessed.tif')
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Preprocess image
            processed_image, _ = preprocess_image_for_sam2(
                input_path, shapefile_path, image_type, blur_strength, 
                show_visualization=False  # Don't show plots for batch processing
            )
            
            # Save result
            save_preprocessed_image(processed_image, output_path, input_path)
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    print(f"\n✅ Batch preprocessing complete!")
    print(f"   Processed images saved to: {output_dir}")

# =============================================================================
# MAIN EXECUTION EXAMPLES
# =============================================================================

def main():
    """Example usage of SAM 2 preprocessing"""
    
    # CONFIGURATION - UPDATE THESE PATHS
    IMAGE_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/2021-09-04_fcc_testclip2.tif"
    SHAPEFILE_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/clip_by_glacier.shp"
    OUTPUT_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/2021-09-04_fcc_blurred.tif"
    
    IMAGE_TYPE = "FCC"  # "RGB", "FCC", or "NDWI"
    
    print("🎯 SAM 2 Preprocessing Example")
    print("=" * 60)
    
    # Example 1: Single image preprocessing with different blur strengths
    print("\n1️⃣ Testing different blur strengths:")
    
    blur_levels = ["light", "medium", "heavy"]
    
    for level in blur_levels:
        blur_strength = get_blur_preset(level)
        print(f"\n--- Testing {level} blur (strength={blur_strength}) ---")
        
        processed_image, inside_mask = preprocess_image_for_sam2(
            IMAGE_PATH, SHAPEFILE_PATH, IMAGE_TYPE, 
            blur_strength=blur_strength, show_visualization=True
        )
        
        # Save this version
        output_file = OUTPUT_PATH.replace('.tif', f'_{level}_blur.tif')
        save_preprocessed_image(processed_image, output_file, IMAGE_PATH)
    
    print(f"\n🎉 Preprocessing examples complete!")
    print(f"📁 Check output files for different blur levels")
    print(f"\n💡 Next steps:")
    print(f"   1. Choose your preferred blur level")
    print(f"   2. Run SAM 2 on the preprocessed image")
    print(f"   3. Compare results with/without preprocessing")

if __name__ == "__main__":
    main()