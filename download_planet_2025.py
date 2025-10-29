#!/usr/bin/env python3
"""
Planet Scene Download Script for 2025 Langtang Data
Download individual Planet scenes using Planet CLI
"""

import subprocess
import os
import json
from datetime import datetime

def download_planet_scenes(scene_ids, output_dir, asset_type="analytic_sr_udm2"):
    """
    Download Planet scenes using Planet CLI
    
    Parameters:
    scene_ids (list): List of Planet scene IDs
    output_dir (str): Directory to save downloaded files
    asset_type (str): Planet asset type to download
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write scene IDs to temporary file
    ids_file = os.path.join(output_dir, "scene_ids.txt")
    with open(ids_file, 'w') as f:
        for scene_id in scene_ids:
            f.write(f"{scene_id}\n")
    
    print(f"Created scene ID file: {ids_file}")
    print(f"Total scenes to download: {len(scene_ids)}")
    print(f"Output directory: {output_dir}")
    print(f"Asset type: {asset_type}")
    
    # Planet CLI download command
    cmd = [
        "planet", "data", "download",
        "--item-type", "PSScene",
        "--asset-type", asset_type,
        "--dest", output_dir,
        "--id-list", ids_file
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("\nStarting download...")
    
    try:
        # Execute the download
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Download completed successfully!")
        print(result.stdout)
        
        # Clean up the temporary IDs file
        os.remove(ids_file)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Download failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        return False
    
    except FileNotFoundError:
        print("Error: Planet CLI not found. Please install it first:")
        print("pip install planet")
        return False

def main():
    """
    Main function - Add your 2025 scene IDs here
    """
    
    # 2025 Planet scene IDs from ids_2025.rtf
    scene_ids_2025 = [
        "20250422_051421_90_251b",
        "20250422_051424_15_251b",
        "20250422_052440_81_253c",
        "20250422_052443_18_253c",
        "20250425_051400_02_2409",
        "20250425_051402_28_2409",
        "20250426_051700_80_2511",
        "20250426_051703_05_2511",
        "20250509_052427_49_2546",
        "20250509_052429_61_2546",
        "20250509_052431_74_2546",
        "20250509_052450_02_2541",
        "20250509_052452_41_2541",
        "20250509_052531_29_2532",
        "20250509_052533_67_2532",
        "20250609_052152_07_24f2",
        "20250609_052154_21_24f2",
        "20250610_051513_22_2521",
        "20250610_051515_44_2521",
        "20250610_052051_35_24fd",
        "20250610_052053_48_24fd",
        "20250610_052629_26_250f",
        "20250610_052631_59_250f",
        "20250614_052026_17_24ee",
        "20250614_052028_34_24ee",
        "20250614_052030_50_24ee",
        "20250628_045117_01_2417",
        "20250628_045118_85_2417",
        "20250628_051737_12_2507",
        "20250628_051739_32_2507",
        "20250706_052311_59_2522",
        "20250706_052313_71_2522",
        "20250706_052315_83_2522",
        "20250905_051649_13_2515",
        "20250905_051651_35_2515",
        "20250905_051653_57_2515"
    ]
    
    if not scene_ids_2025:
        print("ERROR: No scene IDs provided!")
        print("Please edit this script and add your 2025 Planet scene IDs to the 'scene_ids_2025' list.")
        print("Scene IDs should be in format: 'YYYYMMDD_HHMMSS_XX_XXXX'")
        return
    
    # Output directory for 2025 data
    output_directory = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/downloads/langtang2025_harmonized"
    
    # Download the scenes
    success = download_planet_scenes(
        scene_ids=scene_ids_2025,
        output_dir=output_directory,
        asset_type="analytic_sr_udm2"  # or "analytic_sr_harmonized" if you prefer harmonized
    )
    
    if success:
        print(f"\n✅ Download completed successfully!")
        print(f"Files saved to: {output_directory}")
        print(f"\nNext steps:")
        print(f"1. Check downloaded files in {output_directory}")
        print(f"2. Run mosaic creation: python image_processing/create_date_mosaics.py")
        print(f"3. Calculate NDWI: python image_processing/calculate_ndwi.py")
        print(f"4. Detect water: python image_processing/map_water_from_ndwi.py")
    else:
        print(f"\n❌ Download failed. Please check the error messages above.")

if __name__ == "__main__":
    main()