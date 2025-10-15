import os
import glob
import re
from collections import defaultdict
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import argparse

def extract_date_from_filename(filename):
    """
    Extract date from Planet filename
    Returns date string in YYYY-MM-DD format
    """
    basename = os.path.basename(filename)
    
    # Look for YYYY-MM-DD pattern
    match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
    if match:
        return match.group(1)
    
    return None

def group_files_by_date(input_pattern):
    """
    Group Planet composite files by date
    
    Parameters:
    input_pattern (str): Glob pattern for input files
    
    Returns:
    dict: Dictionary with dates as keys and list of file paths as values
    """
    files = glob.glob(input_pattern)
    
    # Group files by date
    date_groups = defaultdict(list)
    
    for file_path in files:
        date_str = extract_date_from_filename(file_path)
        if date_str:
            date_groups[date_str].append(file_path)
        else:
            print(f"Warning: Could not extract date from {file_path}")
    
    return dict(date_groups)

def create_mosaic_for_date(file_list, output_path, nodata_value=0):
    """
    Create a mosaic from multiple Planet strips for a single date
    
    Parameters:
    file_list (list): List of file paths to mosaic
    output_path (str): Output path for the mosaic
    nodata_value (float): NoData value to use
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        print(f"  Creating mosaic from {len(file_list)} strips:")
        for f in file_list:
            print(f"    - {os.path.basename(f)}")
        
        # Open all source files
        src_files_to_mosaic = []
        for fp in file_list:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)
        
        # Merge the files
        mosaic, out_trans = merge(
            src_files_to_mosaic,
            nodata=nodata_value,
            method='first'  # Use first valid pixel (can also use 'last', 'min', 'max', 'mean')
        )
        
        # Get metadata from the first file
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        # Update metadata for the mosaic
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "nodata": nodata_value,
            "compress": "lzw"  # Add compression to save space
        })
        
        # Write the mosaic
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Close source files
        for src in src_files_to_mosaic:
            src.close()
        
        print(f"  ✓ Mosaic saved: {output_path}")
        print(f"    Size: {mosaic.shape[1]} x {mosaic.shape[2]} pixels")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error creating mosaic: {str(e)}")
        return False

def create_all_mosaics(input_pattern, output_dir):
    """
    Create date-based mosaics for all Planet imagery
    
    Parameters:
    input_pattern (str): Glob pattern for input composite files
    output_dir (str): Output directory for mosaics
    
    Returns:
    dict: Dictionary mapping dates to output file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group files by date
    print("Grouping files by date...")
    date_groups = group_files_by_date(input_pattern)
    
    print(f"Found {len(date_groups)} unique dates:")
    for date, files in date_groups.items():
        print(f"  {date}: {len(files)} strips")
    
    # Create mosaics
    mosaic_files = {}
    successful = 0
    
    for date, file_list in date_groups.items():
        print(f"\nProcessing date: {date}")
        
        if len(file_list) == 1:
            # Single file - just copy it
            source_file = file_list[0]
            output_file = os.path.join(output_dir, f"{date}_composite_mosaic.tif")
            
            print(f"  Single strip - copying to: {output_file}")
            
            # Copy the file
            with rasterio.open(source_file) as src:
                profile = src.profile.copy()
                profile.update(compress='lzw')  # Add compression
                
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(src.read())
            
            mosaic_files[date] = output_file
            successful += 1
            print(f"  ✓ File copied successfully")
            
        else:
            # Multiple files - create mosaic
            output_file = os.path.join(output_dir, f"{date}_composite_mosaic.tif")
            
            if create_mosaic_for_date(file_list, output_file):
                mosaic_files[date] = output_file
                successful += 1
    
    print(f"\n{'='*50}")
    print(f"MOSAIC CREATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total dates processed: {len(date_groups)}")
    print(f"Successful mosaics: {successful}")
    print(f"Output directory: {output_dir}")
    
    if successful > 0:
        print(f"\nCreated mosaics:")
        for date, filepath in sorted(mosaic_files.items()):
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {date}: {os.path.basename(filepath)} ({file_size_mb:.1f} MB)")
    
    return mosaic_files

def main():
    parser = argparse.ArgumentParser(description='Create date-based mosaics from Planet imagery strips')
    parser.add_argument('input_pattern', 
                       help='Glob pattern for input composite TIFF files (e.g., "path/to/*_composite.tif")')
    parser.add_argument('--output_dir', default='testimages25',
                       help='Output directory for mosaics (default: testimages25)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PLANET IMAGERY MOSAIC CREATOR")
    print("="*60)
    print(f"Input pattern: {args.input_pattern}")
    print(f"Output directory: {args.output_dir}")
    
    # Create mosaics
    mosaic_files = create_all_mosaics(args.input_pattern, args.output_dir)
    
    print(f"\n🎉 Processing complete!")
    print(f"Created {len(mosaic_files)} date-based mosaic files")
    
    # Suggest next steps
    print(f"\nNext steps:")
    print(f"1. Run water detection on mosaics:")
    print(f"   python batch_water_detection.py \"{args.output_dir}/*_mosaic.tif\" --glacier_shp glacier.shp")
    print(f"2. Check mosaic quality in QGIS or similar")

if __name__ == "__main__":
    main()