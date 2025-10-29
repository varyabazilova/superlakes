import numpy as np
import rasterio
import glob
import os

def read_planet_image(tif_path):
    """
    Read Planet imagery with 4 bands (RGB + NIR)
    
    Parameters:
    tif_path (str): Path to the Planet TIFF file
    
    Returns:
    tuple: (image_array, profile) where image_array is (height, width, 4)
    """
    with rasterio.open(tif_path) as src:
        image = src.read()
        profile = src.profile
        image = np.transpose(image, (1, 2, 0))
        
        print(f"Image shape: {image.shape}")
        print(f"Data type: {image.dtype}")
        print(f"Value range: {image.min()} - {image.max()}")
        
        return image, profile

def compute_ndwi(image):
    """
    Compute NDWI using Green and NIR bands
    NDWI = (Green - NIR) / (Green + NIR)
    
    Parameters:
    image (numpy.ndarray): Image array with shape (height, width, 4)
                          Assumes band order: R, G, B, NIR
    
    Returns:
    numpy.ndarray: NDWI values
    """
    green = image[:, :, 1].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    
    denominator = green + nir
    denominator[denominator == 0] = np.finfo(np.float32).eps
    
    ndwi = (green - nir) / denominator
    
    print(f"NDWI range: {ndwi.min():.3f} - {ndwi.max():.3f}")
    
    return ndwi

def process_single_image(tif_path, output_dir):
    """
    Process a single image to calculate and save NDWI
    
    Parameters:
    tif_path (str): Path to input Planet TIFF
    output_dir (str): Directory to save NDWI output
    
    Returns:
    str: Path to saved NDWI file
    """
    basename = os.path.basename(tif_path)
    print(f"Processing {basename}...")
    
    try:
        # Read image
        image, profile = read_planet_image(tif_path)
        
        # Compute NDWI
        ndwi = compute_ndwi(image)
        
        # Prepare output filename
        ndwi_filename = basename.replace('.tif', '_ndwi.tif')
        ndwi_output = os.path.join(output_dir, ndwi_filename)
        
        # Save NDWI
        profile_ndwi = profile.copy()
        profile_ndwi.update(dtype='float32', count=1, nodata=np.nan)
        
        with rasterio.open(ndwi_output, 'w', **profile_ndwi) as dst:
            dst.write(ndwi.astype(np.float32), 1)
        
        print(f"  ✅ NDWI saved to: {ndwi_output}")
        return ndwi_output
        
    except Exception as e:
        print(f"  ❌ Error processing {basename}: {str(e)}")
        return None

def main():
    # Configuration
    input_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics/langtang2025_harmonized_mosaics"
    output_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/ndwi/langtang2025"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mosaic files
    tif_files = glob.glob(os.path.join(input_dir, "*_composite_mosaic.tif"))
    tif_files.sort()
    
    print(f"Found {len(tif_files)} mosaic files to process")
    print(f"Output directory: {output_dir}")
    print("Processing: Planet imagery → NDWI calculation → Save as GeoTIFF")
    print()
    
    # Process all files
    successful = 0
    for tif_path in tif_files:
        result = process_single_image(tif_path, output_dir)
        if result:
            successful += 1
        print()  # Empty line between files
    
    print(f"📊 SUMMARY:")
    print(f"Successfully processed {successful}/{len(tif_files)} images")
    print(f"NDWI files saved to: {output_dir}")
    print()
    print("Next step: Use 'map_water_from_ndwi.py' to create water masks")

if __name__ == "__main__":
    main()