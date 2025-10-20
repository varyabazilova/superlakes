import rasterio
import numpy as np
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
import os
import glob
from pathlib import Path

def raster_to_vector(water_mask_path, output_dir):
    """
    Convert binary water mask to vector polygons
    
    Parameters:
    water_mask_path (str): Path to binary water mask TIFF
    output_dir (str): Directory to save vector polygons
    
    Returns:
    dict: Statistics about the conversion
    """
    
    filename = os.path.basename(water_mask_path)
    print(f"Processing {filename}...")
    
    with rasterio.open(water_mask_path) as src:
        # Read the binary mask
        mask = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs
        
        # Convert raster to vector polygons (only value 1 = water)
        water_shapes = []
        for geom, value in features.shapes(mask, mask=(mask == 1), transform=transform):
            if value == 1:  # Only water pixels
                water_shapes.append(shape(geom))
    
    if not water_shapes:
        print(f"  No water features found in {filename}")
        return {
            'filename': filename,
            'feature_count': 0,
            'total_area_km2': 0.0
        }
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=water_shapes, crs=crs)
    
    # Calculate area for each polygon
    if gdf.crs and not gdf.crs.is_geographic:
        # Already in projected coordinates
        gdf['area_m2'] = gdf.geometry.area
    else:
        # Project to calculate area
        gdf_proj = gdf.to_crs('EPSG:32645')  # UTM Zone 45N for Nepal
        gdf['area_m2'] = gdf_proj.geometry.area
    
    # Convert to km²
    gdf['area_km2'] = gdf['area_m2'] / 1e6
    
    # Add feature ID
    gdf['feature_id'] = range(1, len(gdf) + 1)
    
    # Save vector file
    output_filename = filename.replace('_water_mask.tif', '_water_polygons.gpkg')
    output_path = os.path.join(output_dir, output_filename)
    gdf.to_file(output_path, driver='GPKG')
    
    total_area_km2 = gdf['area_km2'].sum()
    feature_count = len(gdf)
    
    print(f"  Saved {feature_count} features to {output_filename}")
    print(f"  Total area: {total_area_km2:.6f} km²")
    
    return {
        'filename': filename,
        'feature_count': feature_count,
        'total_area_km2': total_area_km2,
        'vector_file': output_path
    }

def process_langtang_2021():
    """
    Process all 2021 water masks and create vector polygons
    """
    
    # Input and output directories
    water_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/water/langtang2021"
    output_dir = os.path.join(water_dir, "vect")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Find all water mask files
    mask_files = glob.glob(os.path.join(water_dir, "*_water_mask.tif"))
    mask_files.sort()
    
    if not mask_files:
        print(f"No water mask files found in {water_dir}")
        return
    
    print(f"Found {len(mask_files)} water mask files")
    print("=" * 50)
    
    # Process each file
    all_stats = []
    for mask_file in mask_files:
        stats = raster_to_vector(mask_file, output_dir)
        all_stats.append(stats)
        print()
    
    # Summary statistics
    print("=" * 50)
    print("SUMMARY:")
    total_features = sum([s['feature_count'] for s in all_stats])
    total_area = sum([s['total_area_km2'] for s in all_stats])
    files_with_water = len([s for s in all_stats if s['feature_count'] > 0])
    
    print(f"Total files processed: {len(all_stats)}")
    print(f"Files with water features: {files_with_water}")
    print(f"Total water features across all dates: {total_features}")
    print(f"Total water area across all dates: {total_area:.6f} km²")
    print(f"Average water area per date: {total_area/len(all_stats):.6f} km²")
    
    # Show individual results
    print("\nDetailed results:")
    for stats in all_stats:
        date = stats['filename'].split('_')[0]  # Extract date
        print(f"  {date}: {stats['feature_count']} features, {stats['total_area_km2']:.6f} km²")

if __name__ == "__main__":
    process_langtang_2021()