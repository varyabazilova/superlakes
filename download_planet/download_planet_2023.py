#!/usr/bin/env python3
"""
Planet imagery download script for Langtang 2023
Downloads harmonized, clipped Planet imagery using the Orders API
"""

import subprocess
import sys
import json
import time
from pathlib import Path

# Configuration
AOI_FILE = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/roi/roi_langtang_clean.geojson"
OUTPUT_DIR = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/downloads/langtang2023_harmonized"

# 2023 Image IDs (extracted from RTF file)
IMAGE_IDS = [
    "20230927_040639_20_24c3", "20230927_040637_08_24c3", "20230926_044249_58_2475", 
    "20230926_044247_44_2475", "20230917_044203_07_2251", "20230917_044201_24_2251", 
    "20230917_044159_42_2251", "20230916_040720_71_24b4", "20230916_040718_60_24b4", 
    "20230914_040455_82_241d", "20230914_040453_73_241d", "20230902_045756_37_2414", 
    "20230902_045754_53_2414", "20230902_045752_70_2414", "20230902_044301_06_241c", 
    "20230902_044256_83_241c", "20230902_044125_22_24a4", "20230902_044123_06_24a4", 
    "20230902_040236_00_2431", "20230831_044053_48_2416", "20230831_044051_37_2416", 
    "20230830_044213_66_2477", "20230830_044211_49_2477", "20230830_044209_32_2477", 
    "20230830_040456_36_24b0", "20230830_040454_10_24b0", "20230827_044354_37_227a", 
    "20230827_044352_25_227a", "20230827_044350_13_227a", "20230621_040512_16_24c3", 
    "20230621_040509_80_24c3", "20230621_040319_91_24bb", "20230621_040317_55_24bb", 
    "20230611_044057_90_248f", "20230611_044055_73_248f", "20230611_044012_76_241c", 
    "20230611_044010_62_241c", "20230611_043929_91_249e", "20230611_043927_74_249e", 
    "20230609_040538_39_24b4", "20230609_040536_05_24b4", "20230607_040159_90_241f", 
    "20230607_040157_80_241f", "20230606_040658_67_24a1", "20230606_040656_33_24a1", 
    "20230606_040654_00_24a1", "20230606_040156_06_2427", "20230606_040153_98_2427", 
    "20230605_040328_15_24b6", "20230605_040325_81_24b6", "20230605_035925_56_2442", 
    "20230605_035923_47_2442", "20230603_043920_83_2490", "20230603_043918_64_2490", 
    "20230603_035623_36_2459", "20230603_035621_19_2459", "20230602_043916_57_2446", 
    "20230602_043914_38_2446", "20230602_035758_04_242b", "20230602_035755_94_242b", 
    "20230602_035621_59_2449", "20230602_035619_51_2449", "20230601_044020_94_2479", 
    "20230601_040550_48_24c4", "20230601_040548_19_24c4", "20230601_040308_93_24b3", 
    "20230601_040306_60_24b3", "20230530_043946_73_247d", "20230530_043944_72_247d", 
    "20230530_040532_74_24ce", "20230530_040530_61_24ce", "20230530_040528_28_24ce", 
    "20230225_045050_23_2413", "20230225_045048_13_2413", "20230225_042244_46_2276", 
    "20230225_042242_40_2276", "20230225_042240_35_2276"
]

def check_planet_auth():
    """Check if Planet CLI is authenticated"""
    try:
        # Try multiple auth check commands depending on Planet CLI version
        auth_commands = [
            ['planet', 'auth', 'info'],
            ['planet', 'auth', 'status'],
            ['planet', 'auth', 'show']
        ]
        
        for cmd in auth_commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("[SUCCESS] Planet CLI authenticated")
                return True
        
        # If none worked, try a simple planet command to test auth
        result = subprocess.run(['planet', 'orders', 'list'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[SUCCESS] Planet CLI authenticated")
            return True
        else:
            print("[ERROR] Planet CLI not authenticated")
            print("Run: planet auth login")
            return False
            
    except FileNotFoundError:
        print("[ERROR] Planet CLI not found. Install with: pip install planet")
        return False

def create_order():
    """Create Planet order with clipping"""
    print(f"Creating order for {len(IMAGE_IDS)} images...")
    
    # Create order request
    order_request = {
        "name": "Langtang_2023_Harmonized_Clipped",
        "products": [
            {
                "item_ids": IMAGE_IDS,
                "item_type": "PSScene",
                "product_bundle": "analytic_sr_udm2"
            }
        ],
        "tools": [
            {
                "clip": {
                    "aoi": json.loads(open(AOI_FILE).read())["features"][0]["geometry"]
                }
            },
            {
                "harmonize": {
                    "target_sensor": "Sentinel-2"
                }
            }
        ]
    }
    
    # Save order request to file
    order_file = "order_2023.json"
    with open(order_file, 'w') as f:
        json.dump(order_request, f, indent=2)
    
    # Submit order
    result = subprocess.run(['planet', 'orders', 'create', order_file], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract order ID from JSON response
        try:
            order_response = json.loads(result.stdout)
            order_id = order_response["id"]
            print(f"[SUCCESS] Order created: {order_id}")
            return order_id
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] Could not parse order response: {e}")
            print(f"Raw response: {result.stdout}")
            return None
    else:
        print(f"[ERROR] Order creation failed: {result.stderr}")
        return None

def wait_for_order(order_id):
    """Wait for order to be ready"""
    print(f"Waiting for order {order_id} to be ready...")
    
    while True:
        result = subprocess.run(['planet', 'orders', 'get', order_id], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            order_info = json.loads(result.stdout)
            state = order_info.get('state', 'unknown')
            
            print(f"Order status: {state}")
            
            if state == 'success':
                print("[SUCCESS] Order ready for download!")
                return True
            elif state in ['failed', 'cancelled']:
                print(f"[ERROR] Order {state}")
                return False
            else:
                print("Still processing... waiting 60 seconds")
                time.sleep(60)
        else:
            print(f"[ERROR] Error checking order: {result.stderr}")
            return False

def download_order(order_id):
    """Download the completed order"""
    print(f"Downloading order {order_id}...")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(['planet', 'orders', 'download', order_id, 
                           '--directory', OUTPUT_DIR], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"[SUCCESS] Download completed to: {OUTPUT_DIR}")
        return True
    else:
        print(f"[ERROR] Download failed: {result.stderr}")
        return False

def main():
    """Main download workflow"""
    print("Planet 2023 Langtang Download Script")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"AOI file: {AOI_FILE}")
    print(f"Images to download: {len(IMAGE_IDS)}")
    print()
    
    # Check authentication
    if not check_planet_auth():
        sys.exit(1)
    
    # Create order
    order_id = create_order()
    if not order_id:
        sys.exit(1)
    
    # Wait for order
    if not wait_for_order(order_id):
        sys.exit(1)
    
    # Download
    if not download_order(order_id):
        sys.exit(1)
    
    print("All done! Check your images in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()