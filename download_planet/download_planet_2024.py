#!/usr/bin/env python3
"""
Planet imagery download script for Langtang 2024
Downloads harmonized, clipped Planet imagery using the Orders API
"""

import subprocess
import sys
import json
import time
from pathlib import Path

# Configuration
AOI_FILE = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/roi/roi_langtang_clean.geojson"
OUTPUT_DIR = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/downloads/langtang2024_harmonized"

# 2024 Image IDs (extracted from RTF file)
IMAGE_IDS = [
    "20240919_050907_42_24f0", "20240919_050905_25_24f0", "20240919_050903_08_24f0", 
    "20240919_050742_51_24f6", "20240919_050740_21_24f6", "20240911_041759_66_24c7", 
    "20240911_041757_53_24c7", "20240816_050557_96_24b7", "20240816_050555_84_24b7", 
    "20240816_050553_71_24b7", "20240815_051653_50_2461", "20240815_051651_67_2461", 
    "20240815_043131_15_2423", "20240815_043129_33_2423", "20240815_043127_51_2423", 
    "20240729_041542_31_24c2", "20240729_041540_19_24c2", "20240729_041538_07_24c2", 
    "20240615_042252_49_2423", "20240615_042250_70_2423", "20240615_042248_91_2423", 
    "20240612_041350_91_24cf", "20240612_041348_83_24cf", "20240611_050914_30_24d6", 
    "20240611_050912_15_24d6", "20240611_050326_72_24e5", "20240611_050324_46_24e5", 
    "20240610_042559_33_2464", "20240610_042557_53_2464", "20240610_042555_73_2464", 
    "20240605_041437_80_24b4", "20240605_041435_70_24b4", "20240605_041347_64_24b0", 
    "20240605_041345_55_24b0", "20240605_041343_46_24b0", "20240605_041312_67_24ce", 
    "20240605_041310_56_24ce", "20240518_050501_14_24e8", "20240518_050458_84_24e8", 
    "20240511_050447_96_2473", "20240511_050446_09_2473", "20240511_050446_04_24cd", 
    "20240511_050443_74_24cd", "20240511_041430_90_24cc", "20240511_041428_74_24cc", 
    "20240503_050617_21_24e1", "20240503_050614_91_24e1", "20240430_041152_39_24c5", 
    "20240430_041150_21_24c5", "20240430_041115_05_24bc", "20240430_041112_86_24bc", 
    "20240429_042006_96_2464", "20240429_042005_09_2464", "20240428_050357_06_24e6", 
    "20240428_050354_76_24e6", "20240428_041242_33_24c0", "20240428_041240_15_24c0", 
    "20240428_041237_97_24c0", "20240426_050530_54_24eb", "20240426_050528_25_24eb", 
    "20240426_050525_96_24eb", "20240426_041230_49_24c9", "20240426_041228_32_24c9", 
    "20240217_045530_20_2461", "20240217_045528_22_2461", "20240217_045445_64_2473", 
    "20240217_041316_27_2431", "20240217_041314_31_2431", "20240217_041312_36_2431", 
    "20240212_045709_54_2438", "20240212_045707_57_2438", "20240212_045705_60_2438", 
    "20240212_040746_92_24cc", "20240212_040744_75_24cc", "20240206_045446_41_248f", 
    "20240206_045444_44_248f", "20240206_045442_46_248f", "20240206_040945_52_24b4", 
    "20240206_040943_33_24b4", "20240203_045229_36_2488", "20240203_045227_37_2488", 
    "20240203_044950_19_241c", "20240203_044948_20_241c"
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
        "name": "Langtang_2024_Harmonized_Clipped",
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
    order_file = "order_2024.json"
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
    print("Planet 2024 Langtang Download Script")
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