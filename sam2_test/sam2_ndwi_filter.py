# =============================================================================
# FILTER SAM 2 ANNOTATIONS USING NDWI VALUES
# Removes masks that don't have high NDWI (water signatures)
# =============================================================================

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score

def load_annotations(annotations_path):
    """
    Load SAM 2 annotations from GeoTIFF
    Assumes each unique value is a different mask/segment
    """
    print(f"📂 Loading annotations from: {annotations_path}")
    
    with rasterio.open(annotations_path) as src:
        annotations = src.read(1)  # Read first band
        profile = src.profile
    
    # Get unique segment IDs (excluding 0 = background)
    unique_ids = np.unique(annotations)
    unique_ids = unique_ids[unique_ids != 0]  # Remove background
    
    print(f"   Found {len(unique_ids)} annotation segments")
    print(f"   Segment IDs range: {unique_ids.min()} - {unique_ids.max()}")
    
    return annotations, unique_ids, profile

def load_ndwi(ndwi_path):
    """Load NDWI raster"""
    print(f"📂 Loading NDWI from: {ndwi_path}")
    
    with rasterio.open(ndwi_path) as src:
        ndwi = src.read(1)
        profile = src.profile
    
    print(f"   NDWI range: {ndwi.min():.3f} - {ndwi.max():.3f}")
    
    return ndwi, profile

def filter_annotations_by_ndwi(annotations, unique_ids, ndwi, 
                               ndwi_threshold=0.1, 
                               min_high_ndwi_ratio=0.5):
    """
    Filter annotation segments based on NDWI values
    
    Parameters:
    - annotations: 2D array with segment IDs
    - unique_ids: list of segment IDs to check
    - ndwi: 2D NDWI array
    - ndwi_threshold: minimum NDWI value considered "water-like"
    - min_high_ndwi_ratio: minimum fraction of segment that must be above threshold
    
    Returns:
    - kept_ids: segment IDs that passed the filter
    - removed_ids: segment IDs that were filtered out
    - filtered_annotations: new annotation array with only kept segments
    """
    print(f"🔍 Filtering segments using NDWI threshold: {ndwi_threshold}")
    print(f"   Minimum high-NDWI ratio: {min_high_ndwi_ratio}")
    
    kept_ids = []
    removed_ids = []
    
    for segment_id in unique_ids:
        # Get pixels belonging to this segment
        segment_mask = (annotations == segment_id)
        segment_ndwi = ndwi[segment_mask]
        
        # Calculate statistics for this segment
        total_pixels = segment_mask.sum()
        high_ndwi_pixels = (segment_ndwi >= ndwi_threshold).sum()
        high_ndwi_ratio = high_ndwi_pixels / total_pixels if total_pixels > 0 else 0
        mean_ndwi = segment_ndwi.mean()
        
        # Decide whether to keep this segment
        if high_ndwi_ratio >= min_high_ndwi_ratio:
            kept_ids.append(segment_id)
            status = "✅ KEEP"
        else:
            removed_ids.append(segment_id)
            status = "❌ REMOVE"
        
        print(f"   Segment {segment_id:3d}: {total_pixels:4d} pixels, "
              f"mean NDWI: {mean_ndwi:+.3f}, "
              f"high-NDWI: {high_ndwi_ratio:.1%} {status}")
    
    # Create filtered annotation array
    filtered_annotations = np.zeros_like(annotations)
    for segment_id in kept_ids:
        filtered_annotations[annotations == segment_id] = segment_id
    
    print(f"\n📊 Filtering results:")
    print(f"   ✅ Kept: {len(kept_ids)} segments")
    print(f"   ❌ Removed: {len(removed_ids)} segments")
    print(f"   📈 Retention rate: {len(kept_ids)/len(unique_ids)*100:.1f}%")
    
    return kept_ids, removed_ids, filtered_annotations

def visualize_filtering_results(annotations, filtered_annotations, ndwi, 
                                ground_truth=None):
    """
    Create comprehensive visualization of filtering results
    """
    print("🖼️ Creating visualization...")
    
    if ground_truth is not None:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
    
    # 1. Original annotations
    axes[0].imshow(annotations, cmap='tab20')
    axes[0].set_title(f'Original SAM 2 Annotations\n{len(np.unique(annotations))-1} segments')
    axes[0].axis('off')
    
    # 2. NDWI
    ndwi_plot = axes[1].imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[1].set_title('NDWI Values\n(Blue = High Water, Red = Low Water)')
    axes[1].axis('off')
    plt.colorbar(ndwi_plot, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Filtered annotations
    axes[2].imshow(filtered_annotations, cmap='tab20')
    axes[2].set_title(f'Filtered Annotations\n{len(np.unique(filtered_annotations))-1} segments')
    axes[2].axis('off')
    
    # 4. NDWI with filtered overlay
    axes[3].imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[3].imshow(filtered_annotations > 0, alpha=0.5, cmap='Greens')
    axes[3].set_title('NDWI + Filtered Segments\n(Green = Kept segments)')
    axes[3].axis('off')
    
    # 5. Ground truth comparison (if available)
    if ground_truth is not None:
        axes[4].imshow(ground_truth, cmap='Blues', alpha=0.7)
        axes[4].set_title('Ground Truth Lakes')
        axes[4].axis('off')
        
        # 6. Filtered vs Ground Truth
        axes[5].imshow(ground_truth, alpha=0.5, cmap='Blues')
        axes[5].imshow(filtered_annotations > 0, alpha=0.5, cmap='Reds')
        axes[5].set_title('Filtered (Red) vs Ground Truth (Blue)')
        axes[5].axis('off')
        
        # Calculate metrics
        gt_binary = ground_truth.astype(bool)
        filtered_binary = (filtered_annotations > 0)
        
        if gt_binary.sum() > 0 and filtered_binary.sum() > 0:
            iou = jaccard_score(gt_binary.flatten(), filtered_binary.flatten())
            precision = precision_score(gt_binary.flatten(), filtered_binary.flatten())
            recall = recall_score(gt_binary.flatten(), filtered_binary.flatten())
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            fig.suptitle(f'NDWI-Filtered SAM 2 Results\n'
                        f'IoU: {iou:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}',
                        fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def save_filtered_annotations(filtered_annotations, output_path, profile):
    """Save filtered annotations as GeoTIFF"""
    print(f"💾 Saving filtered annotations to: {output_path}")
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(filtered_annotations, 1)
    
    print("✅ Filtered annotations saved!")

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    # FILE PATHS - UPDATE THESE
    ANNOTATIONS_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/20211-09-04_annotations.tif"
    NDWI_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/ndwi/langtang2021/2021-09-04_ndwi.tif"
    GROUND_TRUTH_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3_tryout/test_data/lake_mask_testclip.tif"  # Optional
    OUTPUT_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/20211-09-04_annotations_filtered.tif"
    
    # FILTERING PARAMETERS
    NDWI_THRESHOLD = 0.1      # Minimum NDWI for "water-like"
    MIN_HIGH_NDWI_RATIO = 0.6 # 60% of segment must be above threshold
    
    print("🌊 SAM 2 + NDWI Filtering Pipeline")
    print("=" * 50)
    
    # Load data
    annotations, unique_ids, ann_profile = load_annotations(ANNOTATIONS_PATH)
    ndwi, ndwi_profile = load_ndwi(NDWI_PATH)
    
    # Handle dimension mismatch
    if annotations.shape != ndwi.shape:
        print(f"⚠️ Dimension mismatch detected!")
        print(f"   Annotations: {annotations.shape}")
        print(f"   NDWI: {ndwi.shape}")
        
        # Crop NDWI to match annotations size (assuming annotations are a subset)
        ann_h, ann_w = annotations.shape
        ndwi_h, ndwi_w = ndwi.shape
        
        if ann_h <= ndwi_h and ann_w <= ndwi_w:
            print("🔧 Cropping NDWI to match annotations size...")
            # Crop from center
            start_h = (ndwi_h - ann_h) // 2
            start_w = (ndwi_w - ann_w) // 2
            ndwi = ndwi[start_h:start_h+ann_h, start_w:start_w+ann_w]
            print(f"   ✅ NDWI cropped to: {ndwi.shape}")
        else:
            print("❌ Cannot handle this dimension mismatch - annotations larger than NDWI")
            return
    
    # Load ground truth if available
    ground_truth = None
    try:
        with rasterio.open(GROUND_TRUTH_PATH) as src:
            ground_truth = src.read(1).astype(bool)
        print(f"✅ Ground truth loaded: {ground_truth.sum():,} lake pixels")
    except:
        print("⚠️ Ground truth not available - proceeding without validation")
    
    # Filter annotations
    kept_ids, removed_ids, filtered_annotations = filter_annotations_by_ndwi(
        annotations, unique_ids, ndwi, 
        ndwi_threshold=NDWI_THRESHOLD,
        min_high_ndwi_ratio=MIN_HIGH_NDWI_RATIO
    )
    
    # Visualize results
    visualize_filtering_results(annotations, filtered_annotations, ndwi, ground_truth)
    
    # Save filtered result
    save_filtered_annotations(filtered_annotations, OUTPUT_PATH, ann_profile)
    
    print("\n🎉 Processing complete!")
    print(f"   Filtered annotations saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()