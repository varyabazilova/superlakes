#!/usr/bin/env python3
"""
Create binary raster mask from lake polygons, matching original image extent and resolution.
"""

import rasterio
import geopandas as gpd
from rasterio.features import rasterize
import numpy as np

def create_lake_mask(image_path, shapefile_path, output_path):
    """
    Create binary raster mask from lake polygons.
    
    Parameters:
    - image_path: Path to original Planet mosaic (for extent/resolution reference)
    - shapefile_path: Path to shapefile with lake polygons
    - output_path: Path to save binary mask
    """
    
    print(f"Reading reference image: {image_path}")
    with rasterio.open(image_path) as src:
        # Get image properties
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        bounds = src.bounds
        
        print(f"Image dimensions: {width} x {height}")
        print(f"CRS: {crs}")
        print(f"Bounds: {bounds}")
    
    print(f"Reading lake polygons: {shapefile_path}")
    lakes = gpd.read_file(shapefile_path)
    
    # Reproject to match image CRS if needed
    if lakes.crs != crs:
        print(f"Reprojecting from {lakes.crs} to {crs}")
        lakes = lakes.to_crs(crs)
    
    print(f"Found {len(lakes)} lake polygons")
    
    # Create binary mask
    print("Creating binary mask...")
    
    # Rasterize polygons
    mask = rasterize(
        shapes=lakes.geometry,
        out_shape=(height, width),
        transform=transform,
        fill=0,      # Background value (not water)
        default_value=1,  # Polygon value (water)
        dtype='uint8'
    )
    
    print(f"Mask statistics:")
    print(f"  Water pixels: {np.sum(mask == 1):,} ({np.sum(mask == 1)/mask.size*100:.2f}%)")
    print(f"  Non-water pixels: {np.sum(mask == 0):,} ({np.sum(mask == 0)/mask.size*100:.2f}%)")
    
    # Save as GeoTIFF
    print(f"Saving mask to: {output_path}")
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask, 1)
    
    print("✅ Binary lake mask created successfully!")
    
    return mask

def main():
    # Example usage - you'll need to update these paths
    #image_path = "path/to/your/original_image.tif"
    image_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3/prapare_data/2021-09-04_composite_mosaic.tif"
    # shapefile_path = "path/to/your/lake_polygons.shp"
    shapefile_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3/prapare_data/2021-09-04_manual_lakes.shp"
    # output_path = "path/to/output/lake_mask.tif"
    output_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3/prapare_data/lake_mask.tif"

    print("Lake Mask Creation Tool")
    print("=" * 40)
    
    # Create the mask
    mask = create_lake_mask(image_path, shapefile_path, output_path)
    
    print("\nMask creation complete!")

if __name__ == "__main__":
    main()