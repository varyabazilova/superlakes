#!/usr/bin/env python3
"""
Create binary raster mask from lake polygons using only rasterio (no geopandas).
"""

import rasterio
from rasterio.features import rasterize
import fiona
import numpy as np

def create_lake_mask(image_path, shapefile_path, output_path):
    """
    Create binary raster mask from lake polygons.
    """
    
    print(f"Reading reference image: {image_path}")
    with rasterio.open(image_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        bounds = src.bounds
        
        print(f"Image dimensions: {width} x {height}")
        print(f"CRS: {crs}")
    
    print(f"Reading lake polygons: {shapefile_path}")
    
    # Read shapefile with fiona (lighter than geopandas)
    shapes = []
    with fiona.open(shapefile_path) as src_shp:
        print(f"Shapefile CRS: {src_shp.crs}")
        for feature in src_shp:
            shapes.append(feature['geometry'])
    
    print(f"Found {len(shapes)} lake polygons")
    
    # Create binary mask
    print("Creating binary mask...")
    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,           # Background (not water)
        default_value=1,  # Polygons (water)
        dtype='uint8'
    )
    
    print(f"Mask statistics:")
    print(f"  Water pixels: {np.sum(mask == 1):,} ({np.sum(mask == 1)/mask.size*100:.2f}%)")
    print(f"  Non-water pixels: {np.sum(mask == 0):,}")
    
    # Save mask
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

# Run the function
image_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3/prapare_data/2021-09-04_composite_mosaic.tif"
shapefile_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3/prapare_data/2021-09-04_manual_lakes.shp"
output_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3/prapare_data/lake_mask.tif"

mask = create_lake_mask(image_path, shapefile_path, output_path)