# =============================================================================
# SAMGEO GUIDED LAKE DETECTION
# Teaching SAM what lakes look like using manual annotations and prompts
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from samgeo import SamGeo
from PIL import Image
import cv2
from shapely.geometry import Point
from sklearn.cluster import KMeans

def load_image_for_samgeo(image_path):
    """Load image in format compatible with SamGeo"""
    print(f"📂 Loading image: {image_path}")
    
    with rasterio.open(image_path) as src:
        # Read all bands
        image_data = src.read()
        profile = src.profile
        
        # Convert to H×W×C format if needed
        if image_data.shape[0] <= 4:  # Band-first format
            image_data = np.transpose(image_data, (1, 2, 0))
        
        # Take only RGB channels if more than 3
        if image_data.shape[2] > 3:
            image_data = image_data[:, :, :3]
    
    print(f"   Image shape: {image_data.shape}")
    print(f"   Value range: {image_data.min()} - {image_data.max()}")
    
    return image_data, profile

def load_manual_annotations(annotation_path):
    """Load manual lake annotations (shapefile or raster)"""
    print(f"📂 Loading manual annotations: {annotation_path}")
    
    if annotation_path.endswith('.shp'):
        # Load shapefile annotations
        gdf = gpd.read_file(annotation_path)
        print(f"   Found {len(gdf)} manual lake polygons")
        return gdf, 'vector'
        
    else:
        # Load raster annotations
        with rasterio.open(annotation_path) as src:
            mask = src.read(1).astype(bool)
            profile = src.profile
        
        lake_pixels = mask.sum()
        print(f"   Found {lake_pixels:,} lake pixels")
        return mask, 'raster'

def extract_positive_points_from_mask(mask, num_points=20, method='centers'):
    """
    Extract positive example points from manual lake mask
    
    Parameters:
    - mask: Binary mask of lakes
    - num_points: Number of points to extract
    - method: 'centers', 'random', or 'clustered'
    """
    print(f"🎯 Extracting {num_points} positive points using '{method}' method...")
    
    # Get lake pixel coordinates
    lake_coords = np.where(mask)
    y_coords, x_coords = lake_coords[0], lake_coords[1]
    
    if len(x_coords) == 0:
        print("⚠️ No lake pixels found in mask!")
        return []
    
    if method == 'random':
        # Random sampling of lake pixels
        indices = np.random.choice(len(x_coords), min(num_points, len(x_coords)), replace=False)
        points = [(x_coords[i], y_coords[i]) for i in indices]
        
    elif method == 'clustered':
        # Use K-means to find representative lake centers
        lake_points = np.column_stack([x_coords, y_coords])
        n_clusters = min(num_points, len(lake_points))
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_centers = kmeans.fit(lake_points).cluster_centers_
            points = [(int(center[0]), int(center[1])) for center in cluster_centers]
        else:
            # If only one cluster, use center of mass
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            points = [(center_x, center_y)]
            
    else:  # 'centers' method
        # Find connected components and use their centers
        labeled_mask = cv2.connectedComponents(mask.astype(np.uint8))[1]
        unique_labels = np.unique(labeled_mask)[1:]  # Exclude background
        
        points = []
        for label in unique_labels[:num_points]:
            component_coords = np.where(labeled_mask == label)
            center_y = int(np.mean(component_coords[0]))
            center_x = int(np.mean(component_coords[1]))
            points.append((center_x, center_y))
    
    print(f"   ✅ Extracted {len(points)} positive points")
    return points

def extract_negative_points_outside_boundary(image_shape, boundary_mask, num_points=10):
    """Extract negative points from areas outside the analysis boundary"""
    print(f"🚫 Extracting {num_points} negative points outside boundary...")
    
    h, w = image_shape[:2]
    outside_mask = ~boundary_mask
    
    # Get coordinates outside boundary
    outside_coords = np.where(outside_mask)
    y_coords, x_coords = outside_coords[0], outside_coords[1]
    
    if len(x_coords) == 0:
        print("⚠️ No area outside boundary found!")
        return []
    
    # Random sampling of outside points
    indices = np.random.choice(len(x_coords), min(num_points, len(x_coords)), replace=False)
    points = [(x_coords[i], y_coords[i]) for i in indices]
    
    print(f"   ✅ Extracted {len(points)} negative points")
    return points

def initialize_samgeo(model_type="vit_l", sam_kwargs=None):
    """Initialize SamGeo with specified parameters"""
    print(f"🤖 Initializing SamGeo with model: {model_type}")
    
    if sam_kwargs is None:
        # Default aggressive parameters for lake detection
        sam_kwargs = {
            "points_per_side": 64,
            "pred_iou_thresh": 0.5,
            "stability_score_thresh": 0.4,
            "crop_n_layers": 2,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 10,
        }
    
    print(f"   SAM parameters: {sam_kwargs}")
    
    sam = SamGeo(
        model_type=model_type,
        sam_kwargs=sam_kwargs,
    )
    
    print("✅ SamGeo initialized successfully")
    return sam

def guided_lake_detection(sam, image_path, positive_points=None, negative_points=None,
                         manual_mask=None, output_path=None):
    """
    Run guided lake detection using positive/negative prompts
    
    Parameters:
    - sam: Initialized SamGeo object
    - image_path: Path to input image
    - positive_points: List of (x, y) coordinates on lakes
    - negative_points: List of (x, y) coordinates NOT on lakes
    - manual_mask: Optional manual mask to use as prompt
    - output_path: Where to save results
    """
    print("🎯 Running guided lake detection...")
    
    if positive_points:
        print(f"   Using {len(positive_points)} positive points")
    if negative_points:
        print(f"   Using {len(negative_points)} negative points")
    if manual_mask is not None:
        print(f"   Using manual mask as prompt")
    
    # Method 1: Use automatic generation first
    if positive_points is None and negative_points is None and manual_mask is None:
        print("   Running automatic segmentation...")
        sam.generate(
            source=image_path,
            output=output_path,
            foreground=True,
            unique=True
        )
    
    # Method 2: Use point prompts
    elif positive_points or negative_points:
        print("   Running point-guided segmentation...")
        
        # Combine points and labels
        all_points = []
        all_labels = []
        
        if positive_points:
            all_points.extend(positive_points)
            all_labels.extend([1] * len(positive_points))  # 1 = positive
            
        if negative_points:
            all_points.extend(negative_points)
            all_labels.extend([0] * len(negative_points))  # 0 = negative
        
        # Set the image source
        sam.set_image(image_path)
        
        # Run prediction with prompts
        masks = sam.predict(
            point_coords=all_points,
            point_labels=all_labels,
            multimask_output=True
        )
        
        if output_path:
            sam.save_prediction(output_path)
    
    # Method 3: Use mask prompt
    elif manual_mask is not None:
        print("   Running mask-guided segmentation...")
        sam.set_image(image_path)
        
        masks = sam.predict(
            mask_input=manual_mask,
            multimask_output=True
        )
        
        if output_path:
            sam.save_prediction(output_path)
    
    print("✅ Guided detection complete!")
    return sam

def visualize_prompts_and_results(image, positive_points=None, negative_points=None, 
                                manual_mask=None, sam_results=None):
    """Visualize the prompts and SAM results"""
    print("🖼️ Creating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: Original image with prompts
    axes[0].imshow(image)
    
    if positive_points:
        pos_x, pos_y = zip(*positive_points)
        axes[0].scatter(pos_x, pos_y, c='green', s=50, marker='+', linewidth=3, label='Positive points')
    
    if negative_points:
        neg_x, neg_y = zip(*negative_points)
        axes[0].scatter(neg_x, neg_y, c='red', s=50, marker='x', linewidth=3, label='Negative points')
    
    axes[0].set_title('Image with Prompts')
    axes[0].axis('off')
    if positive_points or negative_points:
        axes[0].legend()
    
    # Panel 2: Manual annotations (if available)
    if manual_mask is not None:
        axes[1].imshow(image)
        axes[1].imshow(manual_mask, alpha=0.5, cmap='Blues')
        axes[1].set_title('Manual Lake Annotations')
    else:
        axes[1].imshow(image)
        axes[1].set_title('Input Image')
    axes[1].axis('off')
    
    # Panel 3: SAM results (placeholder - would need actual results)
    axes[2].imshow(image)
    axes[2].set_title('SAM Detection Results\n(Run detection to see results)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def iterative_refinement_workflow(sam, image_path, manual_annotations_path, 
                                boundary_shapefile=None, output_dir=None):
    """
    Complete iterative workflow for teaching SAM about lakes
    
    Workflow:
    1. Load image and manual annotations
    2. Extract positive points from manual lakes
    3. Extract negative points from outside boundary
    4. Run guided detection
    5. Visualize and save results
    """
    print("🔄 Starting iterative lake detection workflow")
    print("=" * 60)
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Load image
    image, image_profile = load_image_for_samgeo(image_path)
    
    # Step 2: Load manual annotations
    manual_data, annotation_type = load_manual_annotations(manual_annotations_path)
    
    # Step 3: Extract positive points
    if annotation_type == 'raster':
        positive_points = extract_positive_points_from_mask(
            manual_data, num_points=20, method='centers'
        )
        manual_mask = manual_data
    else:
        # For vector data, convert to raster first
        print("🔄 Converting vector annotations to raster...")
        manual_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        # This would need implementation based on your specific vector format
        positive_points = []  # Would extract from polygons
    
    # Step 4: Extract negative points (if boundary provided)
    negative_points = []
    if boundary_shapefile and os.path.exists(boundary_shapefile):
        print("🔄 Loading boundary for negative points...")
        # Load boundary and create mask
        # This would need implementation based on your boundary format
        pass
    
    # Step 5: Visualize prompts
    visualize_prompts_and_results(
        image, positive_points, negative_points, manual_mask
    )
    
    # Step 6: Run guided detection
    output_path = None
    if output_dir:
        output_path = os.path.join(output_dir, "guided_lake_detection.tif")
    
    sam = guided_lake_detection(
        sam, image_path, 
        positive_points=positive_points,
        negative_points=negative_points,
        output_path=output_path
    )
    
    print("🎉 Iterative workflow complete!")
    print(f"   Check results in: {output_dir}")
    
    return sam

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Example workflow for guided lake detection"""
    
    # CONFIGURATION - UPDATE THESE PATHS
    IMAGE_PATH = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/2021-09-04_fcc_blurred_medium_blur.tif"
    MANUAL_ANNOTATIONS = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/lake_mask_testclip.tif"
    BOUNDARY_SHAPEFILE = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/data/clip_by_glacier.shp"
    OUTPUT_DIR = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/sam2_test/guided_results"
    
    print("🎯 SamGeo Guided Lake Detection")
    print("=" * 50)
    
    # Initialize SamGeo
    sam_kwargs = {
        "points_per_side": 64,        # Dense sampling
        "pred_iou_thresh": 0.5,       # Accept lower quality for more detections
        "stability_score_thresh": 0.4, # Accept less stable masks
        "crop_n_layers": 2,           # Multi-scale processing
        "crop_n_points_downscale_factor": 1,  # Don't reduce points in crops
        "min_mask_region_area": 10,   # Keep small features
    }
    
    sam = initialize_samgeo(model_type="vit_l", sam_kwargs=sam_kwargs)
    
    # Run complete workflow
    sam = iterative_refinement_workflow(
        sam=sam,
        image_path=IMAGE_PATH,
        manual_annotations_path=MANUAL_ANNOTATIONS,
        boundary_shapefile=BOUNDARY_SHAPEFILE,
        output_dir=OUTPUT_DIR
    )
    
    print("\n💡 Next steps:")
    print("   1. Check the guided detection results")
    print("   2. Compare with automatic detection")
    print("   3. Adjust parameters if needed")
    print("   4. Apply to more images in your dataset")

if __name__ == "__main__":
    main()