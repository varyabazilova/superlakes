# =============================================================================
# WATER MAPPING: SAM 2 REGIONS + NDWI THRESHOLD
# Uses SAM 2 annotations as candidate regions, then applies NDWI thresholding
# Final water = (SAM 2 regions) AND (NDWI > threshold)
# =============================================================================

import rasterio
import numpy as np
import matplotlib.pyplot as plt

def map_water_with_sam2_regions(annotations_path, ndwi_path, ndwi_threshold=0.1):
    """
    Map water using: SAM 2 regions AND NDWI threshold
    
    Parameters:
    - annotations_path: Path to SAM 2 annotations GeoTIFF
    - ndwi_path: Path to NDWI GeoTIFF
    - ndwi_threshold: NDWI threshold for water (default 0.1)
    
    Returns:
    - water_mask: Binary array with final water detection
    """
    print(f"🌊 Water Mapping: SAM 2 + NDWI")
    print(f"   NDWI threshold: {ndwi_threshold}")
    print("=" * 50)
    
    # Load SAM 2 annotations
    print(f"📂 Loading SAM 2 annotations...")
    with rasterio.open(annotations_path) as src:
        annotations = src.read(1)
        profile = src.profile
    
    unique_segments = len(np.unique(annotations)) - 1  # Exclude background (0)
    print(f"   Found {unique_segments} SAM 2 segments")
    print(f"   Image size: {annotations.shape}")
    
    # Load NDWI
    print(f"📂 Loading NDWI...")
    with rasterio.open(ndwi_path) as src:
        ndwi = src.read(1)
    
    print(f"   NDWI range: {ndwi.min():.3f} to {ndwi.max():.3f}")
    print(f"   NDWI size: {ndwi.shape}")
    
    # Handle dimension mismatch by cropping NDWI
    if annotations.shape != ndwi.shape:
        print(f"🔧 Cropping NDWI to match annotations...")
        ann_h, ann_w = annotations.shape
        ndwi_h, ndwi_w = ndwi.shape
        start_h = (ndwi_h - ann_h) // 2
        start_w = (ndwi_w - ann_w) // 2
        ndwi = ndwi[start_h:start_h+ann_h, start_w:start_w+ann_w]
        print(f"   ✅ NDWI cropped to: {ndwi.shape}")
    
    # Create water detection masks
    print(f"🔍 Creating water masks...")
    
    # Step 1: SAM 2 regions (any annotation > 0)
    sam2_regions = annotations > 0
    sam2_pixels = sam2_regions.sum()
    
    # Step 2: High NDWI regions
    high_ndwi = ndwi > ndwi_threshold
    high_ndwi_pixels = high_ndwi.sum()
    
    # Step 3: Final water = intersection of both
    water_mask = sam2_regions & high_ndwi
    water_pixels = water_mask.sum()
    
    # Calculate statistics
    total_pixels = annotations.size
    print(f"\n📊 Detection Statistics:")
    print(f"   Total image pixels: {total_pixels:,}")
    print(f"   SAM 2 regions: {sam2_pixels:,} pixels ({sam2_pixels/total_pixels*100:.1f}%)")
    print(f"   High NDWI (>{ndwi_threshold}): {high_ndwi_pixels:,} pixels ({high_ndwi_pixels/total_pixels*100:.1f}%)")
    print(f"   Final water: {water_pixels:,} pixels ({water_pixels/total_pixels*100:.1f}%)")
    
    if sam2_pixels > 0:
        print(f"   Water within SAM 2 regions: {water_pixels/sam2_pixels*100:.1f}%")
    
    # Create comprehensive visualization
    print(f"🖼️ Creating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Row 1: Input data
    # 1. SAM 2 annotations (colored by segment)
    axes[0,0].imshow(annotations, cmap='tab20')
    axes[0,0].set_title(f'SAM 2 Annotations\n{unique_segments} segments')
    axes[0,0].axis('off')
    
    # 2. NDWI values
    ndwi_plot = axes[0,1].imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[0,1].set_title(f'NDWI Values\n(Blue=water, Red=land)')
    axes[0,1].axis('off')
    plt.colorbar(ndwi_plot, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # 3. SAM 2 regions (binary)
    axes[0,2].imshow(sam2_regions, cmap='Reds')
    axes[0,2].set_title(f'SAM 2 Regions (Binary)\n{sam2_pixels:,} pixels')
    axes[0,2].axis('off')
    
    # Row 2: Processing steps
    # 4. High NDWI regions
    axes[1,0].imshow(high_ndwi, cmap='Blues')
    axes[1,0].set_title(f'High NDWI (>{ndwi_threshold})\n{high_ndwi_pixels:,} pixels')
    axes[1,0].axis('off')
    
    # 5. Final water mask
    axes[1,1].imshow(water_mask, cmap='Greens')
    axes[1,1].set_title(f'Final Water Map\n{water_pixels:,} pixels')
    axes[1,1].axis('off')
    
    # 6. Overlay on NDWI
    axes[1,2].imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[1,2].imshow(water_mask, alpha=0.7, cmap='Greens')
    axes[1,2].set_title(f'Water Overlay on NDWI\n{water_pixels:,} water pixels')
    axes[1,2].axis('off')
    
    plt.suptitle(f'Water Mapping: SAM 2 + NDWI (threshold={ndwi_threshold})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Save result
    output_path = annotations_path.replace('.tif', '_water.tif')
    print(f"💾 Saving water mask...")
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(water_mask.astype('uint8'), 1)
    
    print(f"✅ Water mask saved: {output_path}")
    
    return water_mask

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # FILE PATHS - UPDATE THESE
    ANNOTATIONS_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/20211-09-04_annotations.tif"
    NDWI_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/2021-09-04_ndwi_clipped.tif"
    
    # WATER DETECTION PARAMETERS
    NDWI_THRESHOLD = 0.1  # Adjust this as needed (0.0, 0.1, 0.2, etc.)
    
    # Run water mapping
    try:
        water_mask = map_water_with_sam2_regions(
            ANNOTATIONS_PATH, 
            NDWI_PATH, 
            ndwi_threshold=NDWI_THRESHOLD
        )
        
        print(f"\n🎉 Water mapping complete!")
        print(f"   Method: SAM 2 regions + NDWI > {NDWI_THRESHOLD}")
        print(f"   Water pixels detected: {water_mask.sum():,}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()