# =============================================================================
# SAMGEO FEW-SHOT LEARNING FOR LAKE DETECTION
# Use 1-2 manual annotations to teach SAM, then apply to many images
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from samgeo import SamGeo
import cv2
from sklearn.cluster import KMeans
import json

def extract_lake_characteristics_from_manual_mask(mask_path, image_path=None):
    """
    Analyze manual lake mask to extract characteristics that can guide SAM
    
    Returns:
    - positive_points: Representative lake center points
    - lake_stats: Statistics about lake sizes, shapes, etc.
    """
    print(f"🔍 Analyzing manual lake mask: {mask_path}")
    
    # Load manual mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(bool)
    
    # Find connected components (individual lakes)
    labeled_mask = cv2.connectedComponents(mask.astype(np.uint8))[1]
    unique_labels = np.unique(labeled_mask)[1:]  # Exclude background
    
    positive_points = []
    lake_sizes = []
    
    print(f"   Found {len(unique_labels)} individual lakes")
    
    for label in unique_labels:
        # Get lake component
        component_mask = (labeled_mask == label)
        component_coords = np.where(component_mask)
        
        # Calculate lake properties
        lake_size = len(component_coords[0])
        center_y = int(np.mean(component_coords[0]))
        center_x = int(np.mean(component_coords[1]))
        
        # Store lake center as positive point
        positive_points.append((center_x, center_y))
        lake_sizes.append(lake_size)
    
    # Calculate lake statistics
    lake_stats = {
        'num_lakes': len(unique_labels),
        'avg_size': np.mean(lake_sizes) if lake_sizes else 0,
        'min_size': np.min(lake_sizes) if lake_sizes else 0,
        'max_size': np.max(lake_sizes) if lake_sizes else 0,
        'total_lake_pixels': mask.sum(),
        'lake_coverage_percent': (mask.sum() / mask.size) * 100
    }
    
    print(f"   Lake statistics:")
    print(f"     - Number of lakes: {lake_stats['num_lakes']}")
    print(f"     - Average size: {lake_stats['avg_size']:.0f} pixels")
    print(f"     - Size range: {lake_stats['min_size']} - {lake_stats['max_size']} pixels")
    print(f"     - Total coverage: {lake_stats['lake_coverage_percent']:.2f}%")
    
    return positive_points, lake_stats

def create_optimized_sam_config(lake_stats):
    """
    Create SAM configuration optimized for the lake characteristics found
    """
    print("⚙️ Creating optimized SAM configuration based on lake characteristics...")
    
    # Base configuration
    sam_kwargs = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.76,
        "stability_score_thresh": 0.62,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 30,
    }
    
    # Adjust based on lake characteristics
    avg_size = lake_stats['avg_size']
    coverage = lake_stats['lake_coverage_percent']
    
    # If lakes are small, increase sampling density
    if avg_size < 100:
        sam_kwargs["points_per_side"] = 64
        sam_kwargs["min_mask_region_area"] = max(10, avg_size // 4)
        print("   → Optimized for small lakes: increased sampling, lowered min area")
    
    # If lakes are sparse, be more aggressive
    if coverage < 2.0:  # Less than 2% coverage
        sam_kwargs["pred_iou_thresh"] = 0.6
        sam_kwargs["stability_score_thresh"] = 0.5
        print("   → Optimized for sparse lakes: lowered quality thresholds")
    
    # If many small lakes, use multi-scale approach
    if lake_stats['num_lakes'] > 10 and avg_size < 200:
        sam_kwargs["crop_n_layers"] = 2
        sam_kwargs["crop_n_points_downscale_factor"] = 1
        print("   → Optimized for many small lakes: multi-scale processing")
    
    print(f"   Final SAM configuration: {sam_kwargs}")
    return sam_kwargs

def apply_sam_to_image_with_guidance(sam, image_path, positive_points, 
                                   output_path=None, use_prompts=True):
    """
    Apply SAM to a single image using learned characteristics
    
    Parameters:
    - sam: Initialized SamGeo object
    - image_path: Path to image to process
    - positive_points: Lake center points from training
    - output_path: Where to save results
    - use_prompts: Whether to use point prompts or just automatic generation
    """
    print(f"🎯 Processing: {os.path.basename(image_path)}")
    
    if use_prompts and positive_points:
        # Method 1: Use positive points as guidance
        print(f"   Using {len(positive_points)} guidance points from training")
        
        sam.set_image(image_path)
        
        # Use positive points (adapted to this image size if needed)
        point_labels = [1] * len(positive_points)  # All positive
        
        masks = sam.predict(
            point_coords=positive_points,
            point_labels=point_labels,
            multimask_output=True
        )
        
        if output_path:
            sam.save_prediction(output_path)
    
    else:
        # Method 2: Use optimized automatic generation
        print("   Using optimized automatic generation")
        
        sam.generate(
            source=image_path,
            output=output_path,
            foreground=True,
            unique=True
        )
    
    return sam

def batch_process_with_few_shot_learning(training_images, training_masks, 
                                       target_image_dir, output_dir,
                                       model_type="vit_l"):
    """
    Main few-shot learning workflow:
    1. Learn from 1-2 training examples
    2. Apply to many target images
    
    Parameters:
    - training_images: List of 1-2 image paths with manual annotations
    - training_masks: List of corresponding mask paths
    - target_image_dir: Directory with many images to process
    - output_dir: Where to save all results
    """
    print("🎓 Few-Shot Learning for Lake Detection")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Learn from training examples
    print("\n📚 LEARNING PHASE - Analyzing training examples...")
    
    all_positive_points = []
    all_lake_stats = []
    
    for i, (img_path, mask_path) in enumerate(zip(training_images, training_masks)):
        print(f"\n--- Training Example {i+1} ---")
        
        # Extract characteristics from this example
        positive_points, lake_stats = extract_lake_characteristics_from_manual_mask(
            mask_path, img_path
        )
        
        all_positive_points.extend(positive_points)
        all_lake_stats.append(lake_stats)
    
    # Combine statistics from all training examples
    combined_stats = {
        'num_lakes': sum(stats['num_lakes'] for stats in all_lake_stats),
        'avg_size': np.mean([stats['avg_size'] for stats in all_lake_stats]),
        'min_size': min(stats['min_size'] for stats in all_lake_stats),
        'max_size': max(stats['max_size'] for stats in all_lake_stats),
        'avg_coverage': np.mean([stats['lake_coverage_percent'] for stats in all_lake_stats])
    }
    
    print(f"\n📊 Combined learning from {len(training_images)} examples:")
    print(f"   Total lakes analyzed: {combined_stats['num_lakes']}")
    print(f"   Average lake size: {combined_stats['avg_size']:.0f} pixels")
    print(f"   Average coverage: {combined_stats['avg_coverage']:.2f}%")
    print(f"   Extracted {len(all_positive_points)} positive example points")
    
    # Step 2: Create optimized SAM configuration
    optimized_sam_kwargs = create_optimized_sam_config(combined_stats)
    
    # Save the learned configuration
    config_path = os.path.join(output_dir, "learned_configuration.json")
    with open(config_path, 'w') as f:
        json.dump({
            'sam_kwargs': optimized_sam_kwargs,
            'lake_stats': combined_stats,
            'num_training_examples': len(training_images),
            'positive_points': all_positive_points
        }, f, indent=2)
    
    print(f"✅ Saved learned configuration to: {config_path}")
    
    # Step 3: Initialize SAM with learned configuration
    print(f"\n🤖 Initializing SAM with learned configuration...")
    sam = SamGeo(
        model_type=model_type,
        sam_kwargs=optimized_sam_kwargs
    )
    
    # Step 4: Apply to target images
    print(f"\n🎯 APPLICATION PHASE - Processing target images...")
    
    # Find target images
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    target_images = []
    
    for ext in image_extensions:
        target_images.extend([
            os.path.join(target_image_dir, f) 
            for f in os.listdir(target_image_dir) 
            if f.lower().endswith(ext)
        ])
    
    print(f"   Found {len(target_images)} target images to process")
    
    # Process each target image
    results_summary = []
    
    for i, image_path in enumerate(target_images, 1):
        print(f"\n--- Processing {i}/{len(target_images)} ---")
        
        # Create output path
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{image_name}_sam_lakes.tif")
        
        try:
            # Apply SAM with learned characteristics
            sam = apply_sam_to_image_with_guidance(
                sam, image_path, all_positive_points[:10],  # Use first 10 guidance points
                output_path=output_path,
                use_prompts=True  # Use guidance from training
            )
            
            print(f"   ✅ Saved: {output_path}")
            results_summary.append({
                'image': image_name,
                'status': 'success',
                'output': output_path
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results_summary.append({
                'image': image_name,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results summary
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n🎉 Few-shot learning complete!")
    print(f"   Processed: {len([r for r in results_summary if r['status'] == 'success'])}/{len(target_images)} images")
    print(f"   Results saved to: {output_dir}")
    print(f"   Configuration saved: {config_path}")
    print(f"   Summary saved: {summary_path}")
    
    return sam, combined_stats

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Example few-shot learning workflow"""
    
    print("🎓 SamGeo Few-Shot Learning for Lake Detection")
    print("=" * 60)
    
    # CONFIGURATION - UPDATE THESE PATHS
    
    # Training data (1-2 examples with manual annotations)
    TRAINING_IMAGES = [
        "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/2021-09-04_fcc_blurred_medium_blur.tif"
    ]
    
    TRAINING_MASKS = [
        "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/lake_mask_testclip.tif"
    ]
    
    # Target images (many images to process)
    TARGET_IMAGE_DIR = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data"  # Directory with many images
    
    # Output directory
    OUTPUT_DIR = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/few_shot_results"
    
    # Model type
    MODEL_TYPE = "vit_l"  # or "vit_h" for best quality, "vit_b" for speed
    
    # Validate inputs
    print("🔍 Validating inputs...")
    
    for img, mask in zip(TRAINING_IMAGES, TRAINING_MASKS):
        if not os.path.exists(img):
            print(f"❌ Training image not found: {img}")
            return
        if not os.path.exists(mask):
            print(f"❌ Training mask not found: {mask}")
            return
    
    if not os.path.exists(TARGET_IMAGE_DIR):
        print(f"❌ Target image directory not found: {TARGET_IMAGE_DIR}")
        return
    
    print("✅ All inputs validated")
    
    # Run few-shot learning workflow
    sam, learned_stats = batch_process_with_few_shot_learning(
        training_images=TRAINING_IMAGES,
        training_masks=TRAINING_MASKS,
        target_image_dir=TARGET_IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        model_type=MODEL_TYPE
    )
    
    print("\n💡 Next steps:")
    print("   1. Check the results in the output directory")
    print("   2. Compare detected lakes with your expectations")
    print("   3. If needed, add more training examples and re-run")
    print("   4. Fine-tune the learned configuration")

if __name__ == "__main__":
    main()