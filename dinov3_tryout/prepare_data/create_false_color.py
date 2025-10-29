#!/usr/bin/env python3
"""
Create false color composite (NIR-Red-Green) from Planet 4-band imagery.
"""

import rasterio
import numpy as np

def create_false_color_composite(input_path, output_path):
    """
    Create false color composite from Planet 4-band imagery.
    
    Planet band order: Blue(1), Green(2), Red(3), NIR(4)
    False color output: NIR(R), Red(G), Green(B)
    
    Parameters:
    - input_path: Path to 4-band Planet image
    - output_path: Path to save 3-band false color composite
    """
    
    print(f"Reading 4-band Planet image: {input_path}")
    
    with rasterio.open(input_path) as src:
        # Read bands: Blue(1), Green(2), Red(3), NIR(4)
        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        
        print(f"Original image shape: {src.shape}")
        print(f"Number of bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        
        # Get metadata for output
        profile = src.profile.copy()
        
        # Handle nodata values and normalize each band
        nodata = src.nodata if src.nodata is not None else 0
        
        # Create false color composite: NIR-Red-Green
        false_color_bands = [nir, red, green]
        false_color_names = ['NIR', 'Red', 'Green']
        
        print("\nProcessing false color composite...")
        
        processed_bands = []
        for i, (band, name) in enumerate(zip(false_color_bands, false_color_names)):
            print(f"Processing {name} band...")
            
            # Mask nodata values
            valid_mask = (band != nodata) & (band > 0)
            
            if np.any(valid_mask):
                # Get statistics for valid pixels
                valid_pixels = band[valid_mask]
                min_val = np.min(valid_pixels)
                max_val = np.max(valid_pixels)
                mean_val = np.mean(valid_pixels)
                
                print(f"  {name} - Min: {min_val:.1f}, Max: {max_val:.1f}, Mean: {mean_val:.1f}")
                
                # Normalize using percentiles for better contrast
                p2, p98 = np.percentile(valid_pixels, [2, 98])
                print(f"  {name} - 2nd percentile: {p2:.1f}, 98th percentile: {p98:.1f}")
                
                # Normalize to 0-255 range
                if p98 > p2:  # Avoid division by zero
                    band_norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    band_norm = np.zeros_like(band)
                
                # Set invalid pixels to 0
                band_norm[~valid_mask] = 0
                
            else:
                print(f"  {name} - No valid pixels found")
                band_norm = np.zeros_like(band)
            
            processed_bands.append(band_norm.astype(np.uint8))
        
        # Update profile for 3-band output
        profile.update({
            'count': 3,
            'dtype': 'uint8',
            'compress': 'lzw'
        })
        
        print(f"\nSaving false color composite: {output_path}")
        
        # Save false color composite
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, band in enumerate(processed_bands, 1):
                dst.write(band, i)
        
        print("✅ False color composite created successfully!")
        print(f"Output bands: NIR(R), Red(G), Green(B)")
        
        # Print some final statistics
        total_pixels = processed_bands[0].size
        valid_pixels = np.sum((processed_bands[0] > 0) | (processed_bands[1] > 0) | (processed_bands[2] > 0))
        print(f"Total pixels: {total_pixels:,}")
        print(f"Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
        
        return output_path

def main():
    # File paths
    input_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3/prapare_data/2021-09-04_composite_mosaic.tif"
    output_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/dinov3/prapare_data/2021-09-04_false_color.tif"
    
    print("False Color Composite Creator")
    print("=" * 40)
    print("Converting Planet 4-band imagery to false color (NIR-Red-Green)")
    print()
    
    # Create false color composite
    result = create_false_color_composite(input_path, output_path)
    
    print(f"\nFalse color composite saved to: {result}")

if __name__ == "__main__":
    main()